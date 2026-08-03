//! S14 Starfold：B=4 精确专家 micro-union 的流式调度与两级页缓存元数据。
//!
//! 这里只定义新 S14 执行器的物理层原语，不依赖旧版运行时、普通模型或
//! 具体 Vulkan wrapper：
//!
//! - 固定 B=4，每 lane 保留精确 top-6 专家 ID、顺序和 `f32` 权重；
//! - 只对四个 lane 里重复的物理专家去重，不裁剪、不重排 lane、不改写权重；
//! - 专家的六个物理 segment 按可配 microtile（建议 1 MiB）切开并串流执行；
//! - VRAM 永远只需两个 microtile 窗口，一个计算时另一个可上传；
//! - RAM 热页缓存的容量为 1..=15000 页，单页超限会在入缓存前被拒绝；
//! - demand 队列绝对优先于 speculative 队列，L+2..L+8 预取可按票据取消。
//!
//! 本模块不拥有 GPU buffer 或 I/O 线程。运行时用 [`StarfoldVramDoubleWindow`]
//! 给实际 `GpuBuffer` 分配 offset，并在每次 SSD/RAM 读取和 Vulkan 提交前检查
//! [`StarfoldCancellationToken::is_cancelled`] 即可接入。

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap, HashMap},
    error::Error,
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering as AtomicOrdering},
        Arc,
    },
};

pub const STARFOLD_B4_LANES: usize = 4;
pub const STARFOLD_TOP_K: usize = 6;
pub const STARFOLD_MAX_UNION_EXPERTS: usize = STARFOLD_B4_LANES * STARFOLD_TOP_K;
pub const STARFOLD_ONE_MIB: u32 = 1024 * 1024;
pub const STARFOLD_MIN_RAM_PAGES: usize = 1;
pub const STARFOLD_MAX_RAM_PAGES: usize = 15_000;
pub const STARFOLD_MIN_PREFETCH_DISTANCE: u16 = 2;
pub const STARFOLD_MAX_PREFETCH_DISTANCE: u16 = 8;

pub type StarfoldResult<T> = Result<T, StarfoldError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarfoldError {
    message: String,
}

impl StarfoldError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for StarfoldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for StarfoldError {}

/// 专家页的六个规范物理 segment。此顺序就是 micro-union 的精确串流顺序。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum StarfoldTensorSegment {
    W1Weight = 0,
    W1Scale = 1,
    W2Weight = 2,
    W2Scale = 3,
    W3Weight = 4,
    W3Scale = 5,
}

impl StarfoldTensorSegment {
    pub const ALL: [Self; 6] = [
        Self::W1Weight,
        Self::W1Scale,
        Self::W2Weight,
        Self::W2Scale,
        Self::W3Weight,
        Self::W3Scale,
    ];

    pub const fn ordinal(self) -> usize {
        self as usize
    }
}

/// RAM/SSD/VRAM 三层共用的稳定 microtile 身份。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StarfoldPageKey {
    pub layer: u16,
    pub expert_id: u16,
    pub segment: StarfoldTensorSegment,
    /// segment 内的 microtile 序号，从 0 开始。
    pub tile_index: u32,
}

/// 一个真实 route 项。`weight` 在计划内按位原样保存。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarfoldRouteEntry {
    pub expert_id: u16,
    pub weight: f32,
}

/// 连续四个位置的权威 route；只允许精确 B=4、top-6。
#[derive(Clone, Debug, PartialEq)]
pub struct StarfoldB4RouteBlock {
    layer: u16,
    base_position: u64,
    lanes: [[StarfoldRouteEntry; STARFOLD_TOP_K]; STARFOLD_B4_LANES],
}

impl StarfoldB4RouteBlock {
    pub fn new(
        layer: u16,
        base_position: u64,
        lanes: [[StarfoldRouteEntry; STARFOLD_TOP_K]; STARFOLD_B4_LANES],
    ) -> StarfoldResult<Self> {
        base_position
            .checked_add((STARFOLD_B4_LANES - 1) as u64)
            .ok_or_else(|| StarfoldError::new("Starfold B=4 position overflow"))?;
        for (lane_index, lane) in lanes.iter().enumerate() {
            for (rank, route) in lane.iter().enumerate() {
                if !route.weight.is_finite() {
                    return Err(StarfoldError::new(format!(
                        "Starfold route 权重非有限数: lane={lane_index} rank={rank}"
                    )));
                }
                if lane[..rank]
                    .iter()
                    .any(|earlier| earlier.expert_id == route.expert_id)
                {
                    return Err(StarfoldError::new(format!(
                        "Starfold 同一 lane 的 top-6 专家重复: lane={lane_index} expert={}",
                        route.expert_id
                    )));
                }
            }
        }
        Ok(Self {
            layer,
            base_position,
            lanes,
        })
    }

    pub const fn layer(&self) -> u16 {
        self.layer
    }

    pub const fn base_position(&self) -> u64 {
        self.base_position
    }

    pub fn lanes(&self) -> &[[StarfoldRouteEntry; STARFOLD_TOP_K]; STARFOLD_B4_LANES] {
        &self.lanes
    }

    pub fn lane_position(&self, lane: usize) -> Option<u64> {
        (lane < STARFOLD_B4_LANES).then(|| self.base_position + lane as u64)
    }
}

/// 一个专家的物理 segment 尺寸与microtile大小。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarfoldMicrotileLayout {
    segment_bytes: [u64; 6],
    microtile_bytes: u32,
    expert_bytes: u64,
}

impl StarfoldMicrotileLayout {
    pub fn new(segment_bytes: [u64; 6], microtile_bytes: u32) -> StarfoldResult<Self> {
        if microtile_bytes == 0 {
            return Err(StarfoldError::new("Starfold microtile_bytes 不能为 0"));
        }
        if segment_bytes.contains(&0) {
            return Err(StarfoldError::new(
                "Starfold 六个专家 segment 的字节数都必须大于 0",
            ));
        }
        let expert_bytes = segment_bytes.iter().try_fold(0u64, |sum, bytes| {
            sum.checked_add(*bytes)
                .ok_or_else(|| StarfoldError::new("Starfold expert bytes overflow"))
        })?;
        for bytes in segment_bytes {
            let tiles = bytes.div_ceil(u64::from(microtile_bytes));
            if tiles > u64::from(u32::MAX) {
                return Err(StarfoldError::new(
                    "Starfold 单个 segment 的 microtile 数超过 u32",
                ));
            }
        }
        Ok(Self {
            segment_bytes,
            microtile_bytes,
            expert_bytes,
        })
    }

    pub fn one_mib(segment_bytes: [u64; 6]) -> StarfoldResult<Self> {
        Self::new(segment_bytes, STARFOLD_ONE_MIB)
    }

    pub const fn microtile_bytes(&self) -> u32 {
        self.microtile_bytes
    }

    pub const fn expert_bytes(&self) -> u64 {
        self.expert_bytes
    }

    pub const fn segment_bytes(&self, segment: StarfoldTensorSegment) -> u64 {
        self.segment_bytes[segment.ordinal()]
    }

    pub fn microtiles_per_expert(&self) -> u64 {
        self.segment_bytes
            .iter()
            .map(|bytes| bytes.div_ceil(u64::from(self.microtile_bytes)))
            .sum()
    }
}

/// 一个 microtile 在源 segment 和概念上连续 union 中的精确位置。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarfoldMicrotileSpan {
    pub key: StarfoldPageKey,
    pub source_segment_offset: u64,
    pub union_stream_offset: u64,
    pub byte_len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarfoldUnionExpert {
    pub expert_id: u16,
    /// bit 0..3 表示哪些 lane 使用该专家。
    pub lane_mask: u8,
    pub first_microtile: u32,
    pub microtile_count: u32,
}

/// lane 内的 route 位置到去重 union 专家的无损绑定。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarfoldLaneBinding {
    pub expert_id: u16,
    pub union_expert_index: u8,
    pub route_rank: u8,
    pub weight: f32,
}

/// B=4 Exact Streamed Expert Micro-Union 的不可变计划。
#[derive(Clone, Debug, PartialEq)]
pub struct StarfoldB4MicroUnionPlan {
    pub layer: u16,
    pub base_position: u64,
    pub streamed_bytes: u64,
    pub layout: StarfoldMicrotileLayout,
    pub experts: Vec<StarfoldUnionExpert>,
    pub microtiles: Vec<StarfoldMicrotileSpan>,
    pub lane_bindings: [[StarfoldLaneBinding; STARFOLD_TOP_K]; STARFOLD_B4_LANES],
}

impl StarfoldB4MicroUnionPlan {
    pub fn build(
        routes: &StarfoldB4RouteBlock,
        layout: StarfoldMicrotileLayout,
    ) -> StarfoldResult<Self> {
        let mut membership = BTreeMap::<u16, u8>::new();
        for (lane_index, lane) in routes.lanes.iter().enumerate() {
            let lane_bit = 1u8 << lane_index;
            for route in lane {
                *membership.entry(route.expert_id).or_default() |= lane_bit;
            }
        }
        if membership.is_empty() || membership.len() > STARFOLD_MAX_UNION_EXPERTS {
            return Err(StarfoldError::new(
                "Starfold B=4 union 的唯一专家数超出 1..=24",
            ));
        }

        let estimated_tiles = (layout.microtiles_per_expert())
            .checked_mul(membership.len() as u64)
            .ok_or_else(|| StarfoldError::new("Starfold microtile count overflow"))?;
        if estimated_tiles > u64::from(u32::MAX) {
            return Err(StarfoldError::new(
                "Starfold union microtile count 超出 u32 可索引上限",
            ));
        }
        let capacity = usize::try_from(estimated_tiles)
            .map_err(|_| StarfoldError::new("Starfold microtile count 超出 usize"))?;
        let mut microtiles = Vec::with_capacity(capacity);
        let mut experts = Vec::with_capacity(membership.len());
        let mut expert_to_union = HashMap::<u16, u8>::with_capacity(membership.len());
        let mut union_stream_offset = 0u64;

        for (union_index, (&expert_id, &lane_mask)) in membership.iter().enumerate() {
            let first_microtile = u32::try_from(microtiles.len())
                .map_err(|_| StarfoldError::new("Starfold union microtile index 超出 u32"))?;
            let expert_start = union_stream_offset;
            for segment in StarfoldTensorSegment::ALL {
                let segment_bytes = layout.segment_bytes(segment);
                let tile_count = segment_bytes.div_ceil(u64::from(layout.microtile_bytes));
                for tile_index_u64 in 0..tile_count {
                    let source_segment_offset = tile_index_u64
                        .checked_mul(u64::from(layout.microtile_bytes))
                        .ok_or_else(|| StarfoldError::new("Starfold segment offset overflow"))?;
                    let remaining = segment_bytes - source_segment_offset;
                    let byte_len = remaining.min(u64::from(layout.microtile_bytes)) as u32;
                    let tile_index = u32::try_from(tile_index_u64).map_err(|_| {
                        StarfoldError::new("Starfold segment microtile index 超出 u32")
                    })?;
                    microtiles.push(StarfoldMicrotileSpan {
                        key: StarfoldPageKey {
                            layer: routes.layer,
                            expert_id,
                            segment,
                            tile_index,
                        },
                        source_segment_offset,
                        union_stream_offset,
                        byte_len,
                    });
                    union_stream_offset = union_stream_offset
                        .checked_add(u64::from(byte_len))
                        .ok_or_else(|| StarfoldError::new("Starfold union bytes overflow"))?;
                }
            }
            if union_stream_offset - expert_start != layout.expert_bytes {
                return Err(StarfoldError::new(
                    "Starfold microtile 切分后的专家字节数漂移",
                ));
            }
            let microtile_count = u32::try_from(microtiles.len())
                .map_err(|_| StarfoldError::new("Starfold union microtile count 超出 u32"))?
                - first_microtile;
            experts.push(StarfoldUnionExpert {
                expert_id,
                lane_mask,
                first_microtile,
                microtile_count,
            });
            expert_to_union.insert(expert_id, union_index as u8);
        }

        let lane_bindings = std::array::from_fn(|lane_index| {
            std::array::from_fn(|rank| {
                let route = routes.lanes[lane_index][rank];
                StarfoldLaneBinding {
                    expert_id: route.expert_id,
                    union_expert_index: expert_to_union[&route.expert_id],
                    route_rank: rank as u8,
                    weight: route.weight,
                }
            })
        });
        let expected_bytes = layout
            .expert_bytes
            .checked_mul(experts.len() as u64)
            .ok_or_else(|| StarfoldError::new("Starfold streamed bytes overflow"))?;
        if expected_bytes != union_stream_offset {
            return Err(StarfoldError::new(
                "Starfold union 规范字节数与 microtile 流不一致",
            ));
        }
        Ok(Self {
            layer: routes.layer,
            base_position: routes.base_position,
            streamed_bytes: union_stream_offset,
            layout,
            experts,
            microtiles,
            lane_bindings,
        })
    }

    pub fn unique_expert_count(&self) -> usize {
        self.experts.len()
    }

    pub fn microtile_count(&self) -> usize {
        self.microtiles.len()
    }

    pub fn microtiles_for_expert(
        &self,
        union_expert_index: usize,
    ) -> Option<&[StarfoldMicrotileSpan]> {
        let expert = self.experts.get(union_expert_index)?;
        let start = expert.first_microtile as usize;
        let end = start + expert.microtile_count as usize;
        self.microtiles.get(start..end)
    }
}

// ---- VRAM 双 microtile 窗口 ---------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StarfoldWindowId {
    A,
    B,
}

impl StarfoldWindowId {
    pub const fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarfoldWindowPhase {
    Empty,
    Uploading,
    Ready,
    Computing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarfoldWindowSnapshot {
    pub window: StarfoldWindowId,
    pub phase: StarfoldWindowPhase,
    pub generation: u64,
    pub byte_offset: u64,
    pub capacity_bytes: u32,
    pub used_bytes: u32,
    pub page: Option<StarfoldPageKey>,
    pub transfer_timeline_value: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarfoldUploadReservation {
    pub window: StarfoldWindowId,
    pub generation: u64,
    pub byte_offset: u64,
    pub page: StarfoldPageKey,
    pub byte_len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarfoldReadyBinding {
    pub window: StarfoldWindowId,
    pub generation: u64,
    pub byte_offset: u64,
    pub page: StarfoldPageKey,
    pub byte_len: u32,
    /// compute submit 必须等待这个 transfer timeline value。
    pub transfer_timeline_value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarfoldComputeLease {
    pub window: StarfoldWindowId,
    pub generation: u64,
    pub byte_offset: u64,
    pub page: StarfoldPageKey,
    pub byte_len: u32,
    pub wait_transfer_timeline_value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WindowState {
    Empty,
    Uploading {
        page: StarfoldPageKey,
        byte_len: u32,
    },
    Ready {
        page: StarfoldPageKey,
        byte_len: u32,
        transfer_value: u64,
    },
    Computing {
        page: StarfoldPageKey,
        byte_len: u32,
        transfer_value: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WindowSlot {
    generation: u64,
    state: WindowState,
}

/// 只占 `2 * window_bytes` 的 VRAM 双窗口生命周期所有者。
#[derive(Debug)]
pub struct StarfoldVramDoubleWindow {
    window_bytes: u32,
    slots: [WindowSlot; 2],
    next_hint: usize,
}

impl StarfoldVramDoubleWindow {
    pub fn new(window_bytes: u32) -> StarfoldResult<Self> {
        if window_bytes == 0 {
            return Err(StarfoldError::new("Starfold VRAM window bytes 不能为 0"));
        }
        Ok(Self {
            window_bytes,
            slots: [
                WindowSlot {
                    generation: 0,
                    state: WindowState::Empty,
                },
                WindowSlot {
                    generation: 0,
                    state: WindowState::Empty,
                },
            ],
            next_hint: 0,
        })
    }

    pub const fn window_bytes(&self) -> u32 {
        self.window_bytes
    }

    pub const fn required_vram_bytes(&self) -> u64 {
        self.window_bytes as u64 * 2
    }

    pub fn reserve_upload(
        &mut self,
        page: StarfoldPageKey,
        byte_len: u32,
    ) -> StarfoldResult<StarfoldUploadReservation> {
        if byte_len == 0 || byte_len > self.window_bytes {
            return Err(StarfoldError::new(format!(
                "Starfold upload bytes 超出窗口: requested={byte_len} capacity={}",
                self.window_bytes
            )));
        }
        let index = [self.next_hint, self.next_hint ^ 1]
            .into_iter()
            .find(|index| self.slots[*index].state == WindowState::Empty)
            .ok_or_else(|| StarfoldError::new("Starfold VRAM 双窗口当前都不可复用"))?;
        let slot = &mut self.slots[index];
        slot.generation = slot
            .generation
            .checked_add(1)
            .ok_or_else(|| StarfoldError::new("Starfold VRAM window generation overflow"))?;
        slot.state = WindowState::Uploading { page, byte_len };
        self.next_hint = index ^ 1;
        let window = if index == 0 {
            StarfoldWindowId::A
        } else {
            StarfoldWindowId::B
        };
        Ok(StarfoldUploadReservation {
            window,
            generation: slot.generation,
            byte_offset: index as u64 * u64::from(self.window_bytes),
            page,
            byte_len,
        })
    }

    pub fn publish_upload(
        &mut self,
        reservation: StarfoldUploadReservation,
        transfer_timeline_value: u64,
    ) -> StarfoldResult<StarfoldReadyBinding> {
        let slot = &mut self.slots[reservation.window.index()];
        if slot.generation != reservation.generation
            || slot.state
                != (WindowState::Uploading {
                    page: reservation.page,
                    byte_len: reservation.byte_len,
                })
        {
            return Err(StarfoldError::new(
                "Starfold 过期/已取消 upload 不能发布到 VRAM 窗口",
            ));
        }
        slot.state = WindowState::Ready {
            page: reservation.page,
            byte_len: reservation.byte_len,
            transfer_value: transfer_timeline_value,
        };
        Ok(StarfoldReadyBinding {
            window: reservation.window,
            generation: reservation.generation,
            byte_offset: reservation.byte_offset,
            page: reservation.page,
            byte_len: reservation.byte_len,
            transfer_timeline_value,
        })
    }

    /// 取消未发布的上传；generation 校验会阻止迟到回调污染新页。
    pub fn cancel_upload(&mut self, reservation: StarfoldUploadReservation) -> StarfoldResult<()> {
        let slot = &mut self.slots[reservation.window.index()];
        if slot.generation != reservation.generation
            || slot.state
                != (WindowState::Uploading {
                    page: reservation.page,
                    byte_len: reservation.byte_len,
                })
        {
            return Err(StarfoldError::new(
                "Starfold 取消 upload 时窗口身份/generation 漂移",
            ));
        }
        slot.state = WindowState::Empty;
        Ok(())
    }

    pub fn begin_compute(
        &mut self,
        binding: StarfoldReadyBinding,
    ) -> StarfoldResult<StarfoldComputeLease> {
        let slot = &mut self.slots[binding.window.index()];
        let expected = WindowState::Ready {
            page: binding.page,
            byte_len: binding.byte_len,
            transfer_value: binding.transfer_timeline_value,
        };
        if slot.generation != binding.generation || slot.state != expected {
            return Err(StarfoldError::new(
                "Starfold compute 不能绑定过期或未 ready 的窗口",
            ));
        }
        slot.state = WindowState::Computing {
            page: binding.page,
            byte_len: binding.byte_len,
            transfer_value: binding.transfer_timeline_value,
        };
        Ok(StarfoldComputeLease {
            window: binding.window,
            generation: binding.generation,
            byte_offset: binding.byte_offset,
            page: binding.page,
            byte_len: binding.byte_len,
            wait_transfer_timeline_value: binding.transfer_timeline_value,
        })
    }

    /// 只能在覆盖该 lease 的 compute fence/timeline 完成后调用。
    pub fn finish_compute(&mut self, lease: StarfoldComputeLease) -> StarfoldResult<()> {
        let slot = &mut self.slots[lease.window.index()];
        let expected = WindowState::Computing {
            page: lease.page,
            byte_len: lease.byte_len,
            transfer_value: lease.wait_transfer_timeline_value,
        };
        if slot.generation != lease.generation || slot.state != expected {
            return Err(StarfoldError::new(
                "Starfold finish_compute 的窗口身份/generation 漂移",
            ));
        }
        slot.state = WindowState::Empty;
        Ok(())
    }

    pub fn discard_ready(&mut self, binding: StarfoldReadyBinding) -> StarfoldResult<()> {
        let slot = &mut self.slots[binding.window.index()];
        let expected = WindowState::Ready {
            page: binding.page,
            byte_len: binding.byte_len,
            transfer_value: binding.transfer_timeline_value,
        };
        if slot.generation != binding.generation || slot.state != expected {
            return Err(StarfoldError::new(
                "Starfold discard_ready 的窗口身份/generation 漂移",
            ));
        }
        slot.state = WindowState::Empty;
        Ok(())
    }

    pub fn snapshot(&self, window: StarfoldWindowId) -> StarfoldWindowSnapshot {
        let index = window.index();
        let slot = self.slots[index];
        let (phase, used_bytes, page, transfer_timeline_value) = match slot.state {
            WindowState::Empty => (StarfoldWindowPhase::Empty, 0, None, None),
            WindowState::Uploading { page, byte_len } => {
                (StarfoldWindowPhase::Uploading, byte_len, Some(page), None)
            }
            WindowState::Ready {
                page,
                byte_len,
                transfer_value,
            } => (
                StarfoldWindowPhase::Ready,
                byte_len,
                Some(page),
                Some(transfer_value),
            ),
            WindowState::Computing {
                page,
                byte_len,
                transfer_value,
            } => (
                StarfoldWindowPhase::Computing,
                byte_len,
                Some(page),
                Some(transfer_value),
            ),
        };
        StarfoldWindowSnapshot {
            window,
            phase,
            generation: slot.generation,
            byte_offset: index as u64 * u64::from(self.window_bytes),
            capacity_bytes: self.window_bytes,
            used_bytes,
            page,
            transfer_timeline_value,
        }
    }

    pub fn free_window_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.state == WindowState::Empty)
            .count()
    }
}

// ---- RAM 热页缓存 -----------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarfoldRamCacheConfig {
    pub capacity_pages: usize,
    pub max_page_bytes: u32,
}

impl StarfoldRamCacheConfig {
    pub fn new(capacity_pages: usize, max_page_bytes: u32) -> StarfoldResult<Self> {
        if !(STARFOLD_MIN_RAM_PAGES..=STARFOLD_MAX_RAM_PAGES).contains(&capacity_pages) {
            return Err(StarfoldError::new(format!(
                "Starfold RAM cache pages 必须为 1..=15000，实际为 {capacity_pages}"
            )));
        }
        if max_page_bytes == 0 {
            return Err(StarfoldError::new(
                "Starfold RAM cache max_page_bytes 不能为 0",
            ));
        }
        Ok(Self {
            capacity_pages,
            max_page_bytes,
        })
    }

    pub fn one_mib(capacity_pages: usize) -> StarfoldResult<Self> {
        Self::new(capacity_pages, STARFOLD_ONE_MIB)
    }

    /// 不含 HashMap/entry 元数据的 payload 硬上限。
    pub fn payload_upper_bound_bytes(&self) -> u64 {
        self.capacity_pages as u64 * u64::from(self.max_page_bytes)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StarfoldRamCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub replacements: u64,
    pub evictions: u64,
    pub resident_pages: usize,
    pub resident_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarfoldCacheInsert {
    pub replaced: bool,
    pub evicted: Option<StarfoldPageKey>,
}

#[derive(Debug)]
struct CachedPage {
    bytes: Box<[u8]>,
    last_touch: u64,
}

/// 单 owner 的有界 LRU 热页缓存。跨 I/O 线程共享时由接入层包装 Mutex。
#[derive(Debug)]
pub struct StarfoldRamPageCache {
    config: StarfoldRamCacheConfig,
    entries: HashMap<StarfoldPageKey, CachedPage>,
    clock: u64,
    stats: StarfoldRamCacheStats,
}

impl StarfoldRamPageCache {
    pub fn new(config: StarfoldRamCacheConfig) -> Self {
        Self {
            entries: HashMap::with_capacity(config.capacity_pages),
            config,
            clock: 0,
            stats: StarfoldRamCacheStats::default(),
        }
    }

    pub const fn config(&self) -> StarfoldRamCacheConfig {
        self.config
    }

    pub fn get(&mut self, key: StarfoldPageKey) -> Option<&[u8]> {
        self.clock = self.clock.saturating_add(1);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_touch = self.clock;
            self.stats.hits = self.stats.hits.saturating_add(1);
            Some(&entry.bytes)
        } else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            None
        }
    }

    pub fn contains(&self, key: StarfoldPageKey) -> bool {
        self.entries.contains_key(&key)
    }

    pub fn insert(
        &mut self,
        key: StarfoldPageKey,
        bytes: Box<[u8]>,
    ) -> StarfoldResult<StarfoldCacheInsert> {
        if bytes.is_empty() || bytes.len() > self.config.max_page_bytes as usize {
            return Err(StarfoldError::new(format!(
                "Starfold RAM page bytes 超出 1..={}: actual={}",
                self.config.max_page_bytes,
                bytes.len()
            )));
        }
        self.clock = self.clock.saturating_add(1);
        let replaced = self.entries.remove(&key);
        if let Some(old) = &replaced {
            self.stats.resident_bytes = self
                .stats
                .resident_bytes
                .checked_sub(old.bytes.len() as u64)
                .ok_or_else(|| StarfoldError::new("Starfold RAM resident bytes underflow"))?;
            self.stats.replacements = self.stats.replacements.saturating_add(1);
        }

        let evicted = if replaced.is_none() && self.entries.len() == self.config.capacity_pages {
            let victim = self
                .entries
                .iter()
                .min_by(|(left_key, left), (right_key, right)| {
                    left.last_touch
                        .cmp(&right.last_touch)
                        .then_with(|| left_key.cmp(right_key))
                })
                .map(|(victim, _)| *victim)
                .ok_or_else(|| StarfoldError::new("Starfold RAM LRU victim 缺失"))?;
            let old = self
                .entries
                .remove(&victim)
                .ok_or_else(|| StarfoldError::new("Starfold RAM LRU victim 消失"))?;
            self.stats.resident_bytes = self
                .stats
                .resident_bytes
                .checked_sub(old.bytes.len() as u64)
                .ok_or_else(|| StarfoldError::new("Starfold RAM resident bytes underflow"))?;
            self.stats.evictions = self.stats.evictions.saturating_add(1);
            Some(victim)
        } else {
            None
        };

        self.stats.resident_bytes = self
            .stats
            .resident_bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| StarfoldError::new("Starfold RAM resident bytes overflow"))?;
        self.entries.insert(
            key,
            CachedPage {
                bytes,
                last_touch: self.clock,
            },
        );
        self.stats.inserts = self.stats.inserts.saturating_add(1);
        self.stats.resident_pages = self.entries.len();
        debug_assert!(self.stats.resident_bytes <= self.config.payload_upper_bound_bytes());
        Ok(StarfoldCacheInsert {
            replaced: replaced.is_some(),
            evicted,
        })
    }

    pub fn remove(&mut self, key: StarfoldPageKey) -> bool {
        let Some(old) = self.entries.remove(&key) else {
            return false;
        };
        self.stats.resident_bytes -= old.bytes.len() as u64;
        self.stats.resident_pages = self.entries.len();
        true
    }

    pub fn invalidate_layer(&mut self, layer: u16) -> usize {
        let victims = self
            .entries
            .keys()
            .copied()
            .filter(|key| key.layer == layer)
            .collect::<Vec<_>>();
        for key in &victims {
            self.remove(*key);
        }
        victims.len()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats.resident_pages = 0;
        self.stats.resident_bytes = 0;
    }

    pub fn stats(&self) -> StarfoldRamCacheStats {
        let mut stats = self.stats;
        stats.resident_pages = self.entries.len();
        stats
    }
}

// ---- demand/speculative 两级队列与可取消预取 -------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarfoldFetchClass {
    Demand,
    Speculative,
}

/// 工作线程可保留该 token；预取取消后，已出队的请求也会立即可见。
#[derive(Clone, Debug)]
pub struct StarfoldCancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl StarfoldCancellationToken {
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(AtomicOrdering::Acquire)
    }
}

#[derive(Clone, Debug)]
pub struct StarfoldPrefetchTicket {
    id: u64,
    pub epoch: u64,
    pub source_layer: u16,
    cancellation: StarfoldCancellationToken,
}

impl StarfoldPrefetchTicket {
    pub const fn id(&self) -> u64 {
        self.id
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// 只设置原子标记。需要同时从队列删除时使用
    /// [`StarfoldFetchQueue::cancel_prefetch`]。
    pub fn cancel(&self) {
        self.cancellation
            .cancelled
            .store(true, AtomicOrdering::Release);
    }
}

#[derive(Clone, Debug)]
pub struct StarfoldFetchRequest {
    pub request_id: u64,
    pub class: StarfoldFetchClass,
    pub page: StarfoldPageKey,
    pub priority: i32,
    pub epoch: u64,
    pub prefetch_ticket_id: Option<u64>,
    pub prefetch_distance: Option<u16>,
    cancellation: Option<StarfoldCancellationToken>,
}

impl StarfoldFetchRequest {
    pub fn cancellation_token(&self) -> Option<&StarfoldCancellationToken> {
        self.cancellation.as_ref()
    }

    pub fn should_abort(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(StarfoldCancellationToken::is_cancelled)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarfoldEnqueueOutcome {
    pub request_id: u64,
    pub inserted: bool,
    /// 同页的 speculative 请求被 demand 覆盖。
    pub promoted_to_demand: bool,
}

#[derive(Clone, Debug)]
struct QueueNode(StarfoldFetchRequest);

impl PartialEq for QueueNode {
    fn eq(&self, other: &Self) -> bool {
        self.0.request_id == other.0.request_id
    }
}

impl Eq for QueueNode {}

impl PartialOrd for QueueNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .priority
            .cmp(&other.0.priority)
            // 同优先级下，更近的预取层先出队。
            .then_with(|| {
                other
                    .0
                    .prefetch_distance
                    .unwrap_or(0)
                    .cmp(&self.0.prefetch_distance.unwrap_or(0))
            })
            // BinaryHeap 的“大”先出；因此更早的 request_id 视为更大。
            .then_with(|| other.0.request_id.cmp(&self.0.request_id))
    }
}

#[derive(Clone, Copy, Debug)]
struct PendingRequest {
    request_id: u64,
    class: StarfoldFetchClass,
    priority: i32,
    ticket_id: Option<u64>,
}

/// demand 绝对优先的两级队列。同一物理页在任意时刻只有一个有效请求。
#[derive(Debug, Default)]
pub struct StarfoldFetchQueue {
    demand: BinaryHeap<QueueNode>,
    speculative: BinaryHeap<QueueNode>,
    pending_by_page: HashMap<StarfoldPageKey, PendingRequest>,
    next_request_id: u64,
    next_ticket_id: u64,
}

impl StarfoldFetchQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin_prefetch(
        &mut self,
        source_layer: u16,
        epoch: u64,
    ) -> StarfoldResult<StarfoldPrefetchTicket> {
        self.next_ticket_id = self
            .next_ticket_id
            .checked_add(1)
            .ok_or_else(|| StarfoldError::new("Starfold prefetch ticket id overflow"))?;
        Ok(StarfoldPrefetchTicket {
            id: self.next_ticket_id,
            epoch,
            source_layer,
            cancellation: StarfoldCancellationToken {
                cancelled: Arc::new(AtomicBool::new(false)),
            },
        })
    }

    pub fn enqueue_demand(
        &mut self,
        page: StarfoldPageKey,
        priority: i32,
        epoch: u64,
    ) -> StarfoldResult<StarfoldEnqueueOutcome> {
        let existing = self.pending_by_page.get(&page).copied();
        if let Some(existing) = existing {
            if existing.class == StarfoldFetchClass::Demand && existing.priority >= priority {
                return Ok(StarfoldEnqueueOutcome {
                    request_id: existing.request_id,
                    inserted: false,
                    promoted_to_demand: false,
                });
            }
        }
        let promoted = existing.is_some_and(|entry| entry.class == StarfoldFetchClass::Speculative);
        let request = StarfoldFetchRequest {
            request_id: self.allocate_request_id()?,
            class: StarfoldFetchClass::Demand,
            page,
            priority,
            epoch,
            prefetch_ticket_id: None,
            prefetch_distance: None,
            cancellation: None,
        };
        self.pending_by_page.insert(
            page,
            PendingRequest {
                request_id: request.request_id,
                class: request.class,
                priority,
                ticket_id: None,
            },
        );
        self.demand.push(QueueNode(request.clone()));
        self.compact_if_needed();
        Ok(StarfoldEnqueueOutcome {
            request_id: request.request_id,
            inserted: true,
            promoted_to_demand: promoted,
        })
    }

    /// 只接受该 ticket 源层的 L+2..L+8 页。
    pub fn enqueue_speculative(
        &mut self,
        ticket: &StarfoldPrefetchTicket,
        page: StarfoldPageKey,
        priority: i32,
    ) -> StarfoldResult<StarfoldEnqueueOutcome> {
        if ticket.is_cancelled() {
            return Err(StarfoldError::new(
                "Starfold 不能向已取消的 prefetch ticket 追加页",
            ));
        }
        let distance = page
            .layer
            .checked_sub(ticket.source_layer)
            .ok_or_else(|| StarfoldError::new("Starfold speculative page 位于预取源层之前"))?;
        if !(STARFOLD_MIN_PREFETCH_DISTANCE..=STARFOLD_MAX_PREFETCH_DISTANCE).contains(&distance) {
            return Err(StarfoldError::new(format!(
                "Starfold speculative 预取只允许 L+2..L+8，实际距离为 {distance}"
            )));
        }
        if let Some(existing) = self.pending_by_page.get(&page).copied() {
            if existing.class == StarfoldFetchClass::Demand
                || (existing.priority >= priority && existing.ticket_id == Some(ticket.id))
            {
                return Ok(StarfoldEnqueueOutcome {
                    request_id: existing.request_id,
                    inserted: false,
                    promoted_to_demand: false,
                });
            }
        }
        let request = StarfoldFetchRequest {
            request_id: self.allocate_request_id()?,
            class: StarfoldFetchClass::Speculative,
            page,
            priority,
            epoch: ticket.epoch,
            prefetch_ticket_id: Some(ticket.id),
            prefetch_distance: Some(distance),
            cancellation: Some(ticket.cancellation.clone()),
        };
        self.pending_by_page.insert(
            page,
            PendingRequest {
                request_id: request.request_id,
                class: request.class,
                priority,
                ticket_id: Some(ticket.id),
            },
        );
        self.speculative.push(QueueNode(request.clone()));
        self.compact_if_needed();
        Ok(StarfoldEnqueueOutcome {
            request_id: request.request_id,
            inserted: true,
            promoted_to_demand: false,
        })
    }

    /// 原子取消已出队工作，并立即清理未出队的同 ticket 页。
    pub fn cancel_prefetch(&mut self, ticket: &StarfoldPrefetchTicket) -> usize {
        ticket.cancel();
        let before = self.pending_by_page.len();
        self.pending_by_page
            .retain(|_, pending| pending.ticket_id != Some(ticket.id));
        self.speculative = self
            .speculative
            .drain()
            .filter(|node| node.0.prefetch_ticket_id != Some(ticket.id))
            .collect();
        before - self.pending_by_page.len()
    }

    pub fn pop_next(&mut self) -> Option<StarfoldFetchRequest> {
        if let Some(request) = pop_live(&mut self.demand, &mut self.pending_by_page) {
            return Some(request);
        }
        pop_live(&mut self.speculative, &mut self.pending_by_page)
    }

    pub fn pending_len(&self) -> usize {
        self.pending_by_page.len()
    }

    pub fn pending_demand_len(&self) -> usize {
        self.pending_by_page
            .values()
            .filter(|pending| pending.class == StarfoldFetchClass::Demand)
            .count()
    }

    pub fn pending_speculative_len(&self) -> usize {
        self.pending_by_page
            .values()
            .filter(|pending| pending.class == StarfoldFetchClass::Speculative)
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.pending_by_page.is_empty()
    }

    fn allocate_request_id(&mut self) -> StarfoldResult<u64> {
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or_else(|| StarfoldError::new("Starfold fetch request id overflow"))?;
        Ok(self.next_request_id)
    }

    fn compact_if_needed(&mut self) {
        let live_demand = self.pending_demand_len();
        if self.demand.len() > live_demand.saturating_mul(2).saturating_add(64) {
            self.demand = rebuild_live_heap(
                self.demand.drain(),
                &self.pending_by_page,
                StarfoldFetchClass::Demand,
            );
        }
        let live_speculative = self.pending_speculative_len();
        if self.speculative.len() > live_speculative.saturating_mul(2).saturating_add(64) {
            self.speculative = rebuild_live_heap(
                self.speculative.drain(),
                &self.pending_by_page,
                StarfoldFetchClass::Speculative,
            );
        }
    }
}

fn pop_live(
    heap: &mut BinaryHeap<QueueNode>,
    pending_by_page: &mut HashMap<StarfoldPageKey, PendingRequest>,
) -> Option<StarfoldFetchRequest> {
    while let Some(node) = heap.pop() {
        let request = node.0;
        let is_live = pending_by_page
            .get(&request.page)
            .is_some_and(|pending| pending.request_id == request.request_id);
        if !is_live {
            continue;
        }
        pending_by_page.remove(&request.page);
        if request.should_abort() {
            continue;
        }
        return Some(request);
    }
    None
}

fn rebuild_live_heap(
    nodes: impl Iterator<Item = QueueNode>,
    pending_by_page: &HashMap<StarfoldPageKey, PendingRequest>,
    class: StarfoldFetchClass,
) -> BinaryHeap<QueueNode> {
    nodes
        .filter(|node| {
            node.0.class == class
                && pending_by_page
                    .get(&node.0.page)
                    .is_some_and(|pending| pending.request_id == node.0.request_id)
        })
        .collect()
}
