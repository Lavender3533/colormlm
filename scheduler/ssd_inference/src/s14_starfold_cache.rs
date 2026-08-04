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

use crate::{
    s14_input_asset_plan::{S14PlannedRangeAsset, S14RangeIdentity},
    s14_position0_mapped_assets::{VerifiedMappedAsset, VerifiedMappedAssetStore},
    s14_range_pack_store::{process_s14_range_pack_store, S14PackedRangeSource},
};
use polaris_s14_runner::Position0Asset;
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap, HashMap},
    error::Error,
    fmt,
    fs::{File, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering},
        Arc, Condvar, Mutex, OnceLock,
    },
    time::SystemTime,
};

pub const STARFOLD_B4_LANES: usize = 4;
pub const STARFOLD_TOP_K: usize = 6;
pub const STARFOLD_MAX_UNION_EXPERTS: usize = STARFOLD_B4_LANES * STARFOLD_TOP_K;
pub const STARFOLD_ONE_MIB: u32 = 1024 * 1024;
pub const STARFOLD_MIN_RAM_PAGES: usize = 1;
pub const STARFOLD_MAX_RAM_PAGES: usize = 15_000;
pub const STARFOLD_MIN_PREFETCH_DISTANCE: u16 = 2;
pub const STARFOLD_MAX_PREFETCH_DISTANCE: u16 = 8;
pub const STARFOLD_DEFAULT_VERIFIED_LEASES: usize = 4_096;
pub const STARFOLD_MAX_VERIFIED_LEASES: usize = STARFOLD_MAX_RAM_PAGES;
pub const STARFOLD_VERIFIED_LEASE_CACHE_CONTRACT_VERSION: u32 = 1;
const STARFOLD_MAX_PROOF_BYTES: u64 = 16 * 1024 * 1024;
static NEXT_VERIFIED_LEASE_CACHE_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

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

// ---- 进程级 verified mmap/proof/SHA 热 lease --------------------------------------

/// 一份可跨 K4 block、跨同进程请求复用的完整 Range 身份。
///
/// `payload_sha256` 与 `proof_sha256` 都来自已解析的 Range proof；文件长度和修改时间
/// 另由热缓存记录。调用方只能从 [`StarfoldVerifiedLeaseCache::acquire_planned`] 得到
/// 此身份，不能把普通 mmap 冒充为已验证 lease。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarfoldVerifiedLeaseIdentity {
    pub tensor: String,
    pub cache_key: String,
    pub range_key: String,
    pub payload_path: PathBuf,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub proof_path: PathBuf,
    pub proof_sha256: String,
    pub hash_authority: String,
    pub expert_id: Option<u16>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StarfoldVerifiedLeaseKey {
    tensor: String,
    parent_tensor: Option<String>,
    kind: String,
    dtype: String,
    shape: Vec<u64>,
    cache_key: String,
    range_key: String,
    payload_path: PathBuf,
    payload_bytes: u64,
    proof_path: PathBuf,
    expert_id: Option<u16>,
    identity_repo: String,
    identity_revision: String,
    identity_source_file: String,
    identity_source_file_bytes: u64,
    identity_start: u64,
    identity_end: u64,
    identity_header_tensor_table_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StarfoldFileStamp {
    bytes: u64,
    modified: SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

/// verified payload 的物理持有方式。loose 路径继续持有 proof 文件和 payload mmap；
/// packed 路径持有进程级只读 pack mmap 与已经通过首次 SHA 的精确 slice。
#[derive(Debug)]
enum StarfoldVerifiedPayloadLease {
    Loose {
        mapped: Arc<VerifiedMappedAsset>,
        #[allow(dead_code)]
        proof_file: File,
        payload_stamp: StarfoldFileStamp,
        proof_stamp: StarfoldFileStamp,
    },
    Packed {
        source: S14PackedRangeSource,
    },
}

/// proof/pack 文件句柄和 verified payload mmap 必须与 microtile 一起存活。
///
/// Windows 上句柄拒绝写入/删除共享；Unix 上即使路径被原子替换，当前 lease 仍指向
/// 已经散列验证的旧 inode。pack index 是进程级不可变快照，更新后由新进程接管。
#[derive(Debug)]
pub struct StarfoldVerifiedMappedLease {
    identity: StarfoldVerifiedLeaseIdentity,
    planned_identity: S14RangeIdentity,
    asset: Position0Asset,
    payload: StarfoldVerifiedPayloadLease,
    last_validated_epoch: AtomicU64,
}

impl StarfoldVerifiedMappedLease {
    pub fn identity(&self) -> &StarfoldVerifiedLeaseIdentity {
        &self.identity
    }

    pub fn asset(&self) -> &Position0Asset {
        &self.asset
    }

    fn payload_bytes(&self) -> StarfoldResult<&[u8]> {
        match &self.payload {
            StarfoldVerifiedPayloadLease::Loose { mapped, .. } => Ok(mapped.bytes()),
            StarfoldVerifiedPayloadLease::Packed { source } => {
                source.payload_bytes().map_err(|error| {
                    StarfoldError::new(format!("读取 S14 packed Range slice: {error:#}"))
                })
            }
        }
    }

    /// 逐字段绑定 planned Range 与已发布 lease。packed lease 的物理路径是 pack/index，
    /// 因此不能再次要求 loose `.bin/.json` 路径 canonicalize；逻辑身份仍完整绑定。
    pub fn validate_planned(
        &self,
        planned: &S14PlannedRangeAsset,
        expert_id: Option<u16>,
    ) -> StarfoldResult<()> {
        if self.asset.tensor != planned.tensor
            || self.asset.kind != planned.kind
            || self.asset.expert_id != expert_id
            || self.asset.dtype != planned.dtype
            || self.asset.shape != planned.shape
            || self.asset.bytes != planned.bytes
            || self.asset.range_key != planned.range_key
            || self.asset.cache_key != planned.cache_key
            || self.planned_identity != planned.identity
            || self.asset.path != self.identity.payload_path
            || self.asset.proof_path != self.identity.proof_path
            || !self.asset.payload_rehashed_by_builder
            || self.identity.tensor != planned.tensor
            || self.identity.cache_key != planned.cache_key
            || self.identity.range_key != planned.range_key
            || self.identity.payload_bytes != planned.bytes
            || self.identity.payload_sha256 != self.asset.sha256
            || self.identity.proof_sha256 != self.asset.proof_sha256
            || self.identity.hash_authority != self.asset.hash_authority
            || self.identity.expert_id != expert_id
        {
            return Err(StarfoldError::new(format!(
                "S14 StarFold verified lease 与 planned identity 漂移: {}",
                planned.tensor
            )));
        }
        validate_lower_sha256(&self.asset.sha256, "payload")?;
        validate_lower_sha256(&self.asset.proof_sha256, "proof")?;
        Ok(())
    }

    /// microtile 的唯一 host 读取入口。取得本类型本身就证明完整 proof 与 payload SHA
    /// 已经闭合；这里再绑定 offset/length，禁止使用方绕过 planned Range 边界。
    pub fn microtile(&self, offset: u64, bytes: u32) -> StarfoldResult<&[u8]> {
        if bytes == 0 {
            return Err(StarfoldError::new(
                "Starfold verified microtile bytes 不能为 0",
            ));
        }
        let end = offset
            .checked_add(u64::from(bytes))
            .ok_or_else(|| StarfoldError::new("Starfold verified microtile end overflow"))?;
        if end > self.identity.payload_bytes {
            return Err(StarfoldError::new(
                "Starfold verified microtile 越出 proof-bound payload",
            ));
        }
        let start = usize::try_from(offset)
            .map_err(|_| StarfoldError::new("Starfold verified microtile offset 超出 usize"))?;
        let end = usize::try_from(end)
            .map_err(|_| StarfoldError::new("Starfold verified microtile end 超出 usize"))?;
        self.payload_bytes()?
            .get(start..end)
            .ok_or_else(|| StarfoldError::new("Starfold verified mmap slice 越界"))
    }

    fn files_are_current_in_epoch(&self, epoch: u64) -> StarfoldResult<(bool, bool)> {
        if self.last_validated_epoch.load(AtomicOrdering::Acquire) == epoch {
            return Ok((true, true));
        }
        let current = match &self.payload {
            StarfoldVerifiedPayloadLease::Loose {
                payload_stamp,
                proof_stamp,
                ..
            } => {
                file_stamp(&self.identity.payload_path)? == *payload_stamp
                    && file_stamp(&self.identity.proof_path)? == *proof_stamp
            }
            StarfoldVerifiedPayloadLease::Packed { .. } => true,
        };
        if current {
            self.last_validated_epoch
                .store(epoch, AtomicOrdering::Release);
        }
        Ok((current, false))
    }
}

/// 一次 request 级不可伪造的文件身份复核纪元。相同 lease 在同一纪元首次命中时
/// 复核 payload/proof stamp，随后 microtile 只复验完整 planned identity，不再重复 stat。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StarfoldVerifiedLeaseValidationEpoch {
    cache_instance_id: u64,
    epoch: u64,
}

impl StarfoldVerifiedLeaseValidationEpoch {
    pub const fn id(self) -> u64 {
        self.epoch
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StarfoldVerifiedLeaseCacheStats {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub waits: u64,
    pub invalidations: u64,
    pub evictions: u64,
    pub resident_entries: usize,
    pub resident_logical_bytes: u64,
    /// cache hit 中真正重新读取 payload/proof 文件 stamp 的次数。
    pub hit_file_stamp_validations: u64,
    /// 同一 request 纪元内复用已认证 stamp、避免两次 metadata syscall 的命中次数。
    pub validation_epoch_reuses: u64,
    /// 命中热 lease 后避免再次完整散列的 payload 逻辑字节。
    pub sha256_bytes_avoided: u64,
}

#[derive(Debug)]
enum VerifiedLeaseSlotState {
    Vacant,
    Loading,
    Ready(Arc<StarfoldVerifiedMappedLease>),
}

#[derive(Debug)]
struct VerifiedLeaseSlot {
    state: Mutex<VerifiedLeaseSlotState>,
    changed: Condvar,
}

impl VerifiedLeaseSlot {
    fn new() -> Self {
        Self {
            state: Mutex::new(VerifiedLeaseSlotState::Vacant),
            changed: Condvar::new(),
        }
    }
}

#[derive(Debug)]
struct VerifiedLeaseRegistryEntry {
    slot: Arc<VerifiedLeaseSlot>,
    last_touch: u64,
    logical_bytes: u64,
}

#[derive(Debug, Default)]
struct VerifiedLeaseRegistry {
    entries: HashMap<StarfoldVerifiedLeaseKey, VerifiedLeaseRegistryEntry>,
    clock: u64,
    stats: StarfoldVerifiedLeaseCacheStats,
}

/// 有界、并发安全的 verified mmap 热 lease 缓存。
///
/// 不同资产可并发执行首次 SHA；同一资产只有一个 loader，其余线程等待同一个 slot。
/// 缓存只保存完整 Range mmap，microtile 仍由 lease 的 [`StarfoldVerifiedMappedLease::microtile`]
/// 做 proof-bound 切片，不会把未验证 RAM 页发布给 Vulkan 上传路径。
#[derive(Debug)]
pub struct StarfoldVerifiedLeaseCache {
    instance_id: u64,
    next_validation_epoch: AtomicU64,
    capacity_entries: usize,
    registry: Mutex<VerifiedLeaseRegistry>,
}

impl StarfoldVerifiedLeaseCache {
    pub fn new(capacity_entries: usize) -> StarfoldResult<Self> {
        if !(1..=STARFOLD_MAX_VERIFIED_LEASES).contains(&capacity_entries) {
            return Err(StarfoldError::new(format!(
                "Starfold verified lease capacity 必须位于 1..={STARFOLD_MAX_VERIFIED_LEASES}: actual={capacity_entries}"
            )));
        }
        let instance_id = NEXT_VERIFIED_LEASE_CACHE_INSTANCE_ID
            .fetch_update(
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
                |current| (current < u64::MAX).then_some(current + 1),
            )
            .map_err(|_| {
                StarfoldError::new("Starfold verified lease cache instance id exhausted")
            })?;
        Ok(Self {
            instance_id,
            next_validation_epoch: AtomicU64::new(1),
            capacity_entries,
            registry: Mutex::new(VerifiedLeaseRegistry::default()),
        })
    }

    pub const fn capacity_entries(&self) -> usize {
        self.capacity_entries
    }

    pub(crate) fn begin_validation_epoch(
        &self,
    ) -> StarfoldResult<StarfoldVerifiedLeaseValidationEpoch> {
        let epoch = self
            .next_validation_epoch
            .fetch_update(
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
                |current| (current < u64::MAX).then_some(current + 1),
            )
            .map_err(|_| {
                StarfoldError::new("Starfold verified lease validation epoch exhausted")
            })?;
        Ok(StarfoldVerifiedLeaseValidationEpoch {
            cache_instance_id: self.instance_id,
            epoch,
        })
    }

    /// 从 planned Range 获取进程级热 lease。cache hit 不再读取 proof、不重新 mmap，
    /// 也不重新散列 payload；但每次仍复核 payload/proof 的长度、mtime（Unix 额外复核
    /// dev/inode）。任何漂移都会先失效旧 entry，再走完整 proof/SHA loader。
    pub fn acquire_planned(
        &self,
        allowed_root: &Path,
        planned: &S14PlannedRangeAsset,
        expert_id: Option<u16>,
    ) -> StarfoldResult<Arc<StarfoldVerifiedMappedLease>> {
        let epoch = self.begin_validation_epoch()?;
        let canonical_root = canonical_directory(allowed_root)?;
        self.acquire_planned_in_epoch(&canonical_root, planned, expert_id, epoch)
    }

    /// request 级热路径。canonical_root 必须由 runtime 初始化时 canonicalize 一次；
    /// 同一个 epoch 只能由当前 cache 签发。它把文件 stamp 复核从“每个 microtile 两次
    /// stat”收敛为“每个完整 Range、每个 request 最多一次”。
    pub(crate) fn acquire_planned_in_epoch(
        &self,
        canonical_root: &Path,
        planned: &S14PlannedRangeAsset,
        expert_id: Option<u16>,
        epoch: StarfoldVerifiedLeaseValidationEpoch,
    ) -> StarfoldResult<Arc<StarfoldVerifiedMappedLease>> {
        if epoch.cache_instance_id != self.instance_id || epoch.epoch == 0 {
            return Err(StarfoldError::new(
                "Starfold verified lease validation epoch owner 漂移",
            ));
        }
        if !canonical_root.is_absolute() || !canonical_root.is_dir() {
            return Err(StarfoldError::new(
                "Starfold verified lease canonical root 合同非法",
            ));
        }
        let packed_source = process_s14_range_pack_store(canonical_root)
            .map_err(|error| StarfoldError::new(format!("取得 S14 Range pack store: {error:#}")))?
            .map(|store| store.lookup(planned))
            .transpose()
            .map_err(|error| StarfoldError::new(format!("查找 S14 Range pack entry: {error:#}")))?
            .flatten();
        let key = normalized_lease_key(canonical_root, planned, expert_id, packed_source.as_ref())?;
        let slot = self.slot_for(&key)?;

        loop {
            let mut state = slot
                .state
                .lock()
                .map_err(|_| StarfoldError::new("Starfold verified lease slot poisoned"))?;
            match &*state {
                VerifiedLeaseSlotState::Ready(lease) => {
                    let lease = Arc::clone(lease);
                    drop(state);
                    lease
                        .validate_planned(planned, expert_id)
                        .map_err(|error| {
                            StarfoldError::new(format!(
                                "Starfold verified lease cache hit planned identity 漂移: {error}"
                            ))
                        })?;
                    let (files_current, epoch_reused) =
                        lease.files_are_current_in_epoch(epoch.epoch)?;
                    if files_current {
                        self.record_hit(&key, &slot, lease.identity.payload_bytes, epoch_reused)?;
                        return Ok(lease);
                    }
                    let mut state = slot.state.lock().map_err(|_| {
                        StarfoldError::new("Starfold verified lease invalidation slot poisoned")
                    })?;
                    if matches!(&*state, VerifiedLeaseSlotState::Ready(current) if Arc::ptr_eq(current, &lease))
                    {
                        *state = VerifiedLeaseSlotState::Vacant;
                        slot.changed.notify_all();
                        drop(state);
                        self.record_invalidation(&key, &slot)?;
                    }
                }
                VerifiedLeaseSlotState::Loading => {
                    state = slot.changed.wait(state).map_err(|_| {
                        StarfoldError::new("Starfold verified lease wait slot poisoned")
                    })?;
                    drop(state);
                    // 不得在持有 slot mutex 时再取 registry mutex；slot_for 的淘汰
                    // 路径按 registry -> slot 的顺序取锁，反向取锁会形成 ABBA 死锁。
                    self.record_wait()?;
                }
                VerifiedLeaseSlotState::Vacant => {
                    *state = VerifiedLeaseSlotState::Loading;
                    drop(state);
                    self.record_miss()?;
                    let loaded = load_verified_lease(
                        canonical_root,
                        planned,
                        expert_id,
                        &key,
                        packed_source.clone(),
                        epoch.epoch,
                    );
                    let mut state = slot.state.lock().map_err(|_| {
                        StarfoldError::new("Starfold verified lease publish slot poisoned")
                    })?;
                    match loaded {
                        Ok(lease) => {
                            let lease = Arc::new(lease);
                            *state = VerifiedLeaseSlotState::Ready(Arc::clone(&lease));
                            slot.changed.notify_all();
                            drop(state);
                            self.record_publish(&key, &slot, lease.identity.payload_bytes)?;
                            return Ok(lease);
                        }
                        Err(error) => {
                            *state = VerifiedLeaseSlotState::Vacant;
                            slot.changed.notify_all();
                            return Err(error);
                        }
                    }
                }
            }
        }
    }

    /// 管理面全失效入口。调用方应先停止接收新的 acquire；已经取得的 `Arc` lease
    /// 仍可安全完成当前 microtile，清空 registry 不会使其底层句柄提前失效。
    pub fn invalidate_all(&self) -> StarfoldResult<usize> {
        let mut registry = self
            .registry
            .lock()
            .map_err(|_| StarfoldError::new("Starfold verified lease registry poisoned"))?;
        let removed = registry.entries.len();
        registry.entries.clear();
        registry.stats.invalidations = registry.stats.invalidations.saturating_add(removed as u64);
        registry.stats.resident_entries = 0;
        registry.stats.resident_logical_bytes = 0;
        Ok(removed)
    }

    pub fn stats(&self) -> StarfoldResult<StarfoldVerifiedLeaseCacheStats> {
        let registry = self
            .registry
            .lock()
            .map_err(|_| StarfoldError::new("Starfold verified lease registry poisoned"))?;
        let mut stats = registry.stats;
        stats.resident_entries = registry.entries.len();
        Ok(stats)
    }

    fn slot_for(&self, key: &StarfoldVerifiedLeaseKey) -> StarfoldResult<Arc<VerifiedLeaseSlot>> {
        let mut registry = self
            .registry
            .lock()
            .map_err(|_| StarfoldError::new("Starfold verified lease registry poisoned"))?;
        registry.stats.requests = registry.stats.requests.saturating_add(1);
        registry.clock = registry.clock.saturating_add(1);
        let touch = registry.clock;
        if let Some(entry) = registry.entries.get_mut(key) {
            entry.last_touch = touch;
            return Ok(Arc::clone(&entry.slot));
        }
        if registry.entries.len() >= self.capacity_entries {
            let victim = registry
                .entries
                .iter()
                // 正在执行完整 proof/SHA 的 slot 不能被逐出；否则同一 key 会创建
                // 第二个 loader，既浪费 I/O，也破坏 single-flight 契约。
                .filter(|(_, entry)| {
                    entry
                        .slot
                        .state
                        .lock()
                        .map(|state| !matches!(&*state, VerifiedLeaseSlotState::Loading))
                        .unwrap_or(false)
                })
                .min_by(|(left_key, left), (right_key, right)| {
                    left.last_touch
                        .cmp(&right.last_touch)
                        .then_with(|| left_key.cache_key.cmp(&right_key.cache_key))
                })
                .map(|(key, _)| key.clone())
                .ok_or_else(|| {
                    StarfoldError::new(
                        "Starfold verified lease cache 已满且所有 slot 都在验证；拒绝重复 loader",
                    )
                })?;
            let evicted = registry
                .entries
                .remove(&victim)
                .ok_or_else(|| StarfoldError::new("Starfold verified lease LRU victim 消失"))?;
            registry.stats.evictions = registry.stats.evictions.saturating_add(1);
            registry.stats.resident_logical_bytes = registry
                .stats
                .resident_logical_bytes
                .saturating_sub(evicted.logical_bytes);
        }
        let slot = Arc::new(VerifiedLeaseSlot::new());
        registry.entries.insert(
            key.clone(),
            VerifiedLeaseRegistryEntry {
                slot: Arc::clone(&slot),
                last_touch: touch,
                logical_bytes: 0,
            },
        );
        registry.stats.resident_entries = registry.entries.len();
        Ok(slot)
    }

    fn record_hit(
        &self,
        key: &StarfoldVerifiedLeaseKey,
        slot: &Arc<VerifiedLeaseSlot>,
        bytes: u64,
        epoch_reused: bool,
    ) -> StarfoldResult<()> {
        let mut registry = self.registry_lock()?;
        registry.stats.hits = registry.stats.hits.saturating_add(1);
        registry.stats.sha256_bytes_avoided =
            registry.stats.sha256_bytes_avoided.saturating_add(bytes);
        if epoch_reused {
            registry.stats.validation_epoch_reuses =
                registry.stats.validation_epoch_reuses.saturating_add(1);
        } else {
            registry.stats.hit_file_stamp_validations =
                registry.stats.hit_file_stamp_validations.saturating_add(1);
        }
        touch_matching_entry(&mut registry, key, slot);
        Ok(())
    }

    fn record_wait(&self) -> StarfoldResult<()> {
        let mut registry = self.registry_lock()?;
        registry.stats.waits = registry.stats.waits.saturating_add(1);
        Ok(())
    }

    fn record_miss(&self) -> StarfoldResult<()> {
        let mut registry = self.registry_lock()?;
        registry.stats.misses = registry.stats.misses.saturating_add(1);
        Ok(())
    }

    fn record_invalidation(
        &self,
        key: &StarfoldVerifiedLeaseKey,
        slot: &Arc<VerifiedLeaseSlot>,
    ) -> StarfoldResult<()> {
        let mut registry = self.registry_lock()?;
        registry.stats.invalidations = registry.stats.invalidations.saturating_add(1);
        let old_bytes = registry
            .entries
            .get_mut(key)
            .filter(|entry| Arc::ptr_eq(&entry.slot, slot))
            .map(|entry| {
                let old_bytes = entry.logical_bytes;
                entry.logical_bytes = 0;
                old_bytes
            });
        if let Some(old_bytes) = old_bytes {
            registry.stats.resident_logical_bytes = registry
                .stats
                .resident_logical_bytes
                .saturating_sub(old_bytes);
        }
        Ok(())
    }

    fn record_publish(
        &self,
        key: &StarfoldVerifiedLeaseKey,
        slot: &Arc<VerifiedLeaseSlot>,
        bytes: u64,
    ) -> StarfoldResult<()> {
        let mut registry = self.registry_lock()?;
        let mut old_bytes = None;
        if let Some(entry) = registry
            .entries
            .get_mut(key)
            .filter(|entry| Arc::ptr_eq(&entry.slot, slot))
        {
            old_bytes = Some(entry.logical_bytes);
            entry.logical_bytes = bytes;
        }
        if let Some(old_bytes) = old_bytes {
            registry.stats.resident_logical_bytes = registry
                .stats
                .resident_logical_bytes
                .saturating_sub(old_bytes)
                .saturating_add(bytes);
        }
        Ok(())
    }

    fn registry_lock(&self) -> StarfoldResult<std::sync::MutexGuard<'_, VerifiedLeaseRegistry>> {
        self.registry
            .lock()
            .map_err(|_| StarfoldError::new("Starfold verified lease registry poisoned"))
    }
}

fn touch_matching_entry(
    registry: &mut VerifiedLeaseRegistry,
    key: &StarfoldVerifiedLeaseKey,
    slot: &Arc<VerifiedLeaseSlot>,
) {
    registry.clock = registry.clock.saturating_add(1);
    let touch = registry.clock;
    if let Some(entry) = registry
        .entries
        .get_mut(key)
        .filter(|entry| Arc::ptr_eq(&entry.slot, slot))
    {
        entry.last_touch = touch;
    }
}

fn canonical_directory(path: &Path) -> StarfoldResult<PathBuf> {
    let canonical = path.canonicalize().map_err(|error| {
        StarfoldError::new(format!(
            "resolve Starfold verified lease root {}: {error}",
            path.display()
        ))
    })?;
    if !canonical.is_dir() {
        return Err(StarfoldError::new(format!(
            "Starfold verified lease root 不是目录: {}",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn normalized_lease_key(
    canonical_root: &Path,
    planned: &S14PlannedRangeAsset,
    expert_id: Option<u16>,
    packed_source: Option<&S14PackedRangeSource>,
) -> StarfoldResult<StarfoldVerifiedLeaseKey> {
    if planned.tensor.is_empty()
        || planned.cache_key.is_empty()
        || planned.range_key.is_empty()
        || planned.bytes == 0
    {
        return Err(StarfoldError::new(
            "Starfold verified lease planned identity 为空",
        ));
    }
    let (payload_path, proof_path) = if let Some(source) = packed_source {
        (
            source.pack_path().to_path_buf(),
            source.index_path().to_path_buf(),
        )
    } else {
        (
            canonical_child(canonical_root, &planned.payload_path, "payload")?,
            canonical_child(canonical_root, &planned.proof_path, "proof")?,
        )
    };
    Ok(StarfoldVerifiedLeaseKey {
        tensor: planned.tensor.clone(),
        parent_tensor: planned.parent_tensor.clone(),
        kind: planned.kind.clone(),
        dtype: planned.dtype.clone(),
        shape: planned.shape.clone(),
        cache_key: planned.cache_key.clone(),
        range_key: planned.range_key.clone(),
        payload_path,
        payload_bytes: planned.bytes,
        proof_path,
        expert_id,
        identity_repo: planned.identity.repo.clone(),
        identity_revision: planned.identity.revision.clone(),
        identity_source_file: planned.identity.source_file.clone(),
        identity_source_file_bytes: planned.identity.source_file_bytes,
        identity_start: planned.identity.start,
        identity_end: planned.identity.end,
        identity_header_tensor_table_sha256: planned.identity.header_tensor_table_sha256.clone(),
    })
}

fn canonical_child(root: &Path, path: &Path, label: &str) -> StarfoldResult<PathBuf> {
    let canonical = path.canonicalize().map_err(|error| {
        StarfoldError::new(format!(
            "resolve Starfold verified {label} {}: {error}",
            path.display()
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(StarfoldError::new(format!(
            "Starfold verified {label} 越出允许根目录: root={} path={}",
            root.display(),
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn load_verified_lease(
    canonical_root: &Path,
    planned: &S14PlannedRangeAsset,
    expert_id: Option<u16>,
    key: &StarfoldVerifiedLeaseKey,
    packed_source: Option<S14PackedRangeSource>,
    validation_epoch: u64,
) -> StarfoldResult<StarfoldVerifiedMappedLease> {
    if let Some(source) = packed_source {
        source.verify_payload().map_err(|error| {
            StarfoldError::new(format!("校验 S14 packed Range payload SHA: {error:#}"))
        })?;
        let source_identity = serde_json::to_value(&planned.identity).map_err(|error| {
            StarfoldError::new(format!("编码 S14 packed Range source identity: {error}"))
        })?;
        let asset = Position0Asset {
            tensor: planned.tensor.clone(),
            kind: planned.kind.clone(),
            expert_id,
            dtype: planned.dtype.clone(),
            shape: planned.shape.clone(),
            bytes: planned.bytes,
            range_key: planned.range_key.clone(),
            cache_key: planned.cache_key.clone(),
            path: canonical_root.join(format!("{}.bin", planned.cache_key)),
            sha256: source.payload_sha256().to_owned(),
            proof_path: canonical_root.join(format!("{}.json", planned.cache_key)),
            proof_sha256: source.proof_sha256().to_owned(),
            hash_authority: source.hash_authority().to_owned(),
            payload_rehashed_by_builder: true,
            source: source_identity,
        };
        validate_lower_sha256(&asset.sha256, "payload")?;
        validate_lower_sha256(&asset.proof_sha256, "proof")?;
        let identity = StarfoldVerifiedLeaseIdentity {
            tensor: asset.tensor.clone(),
            cache_key: asset.cache_key.clone(),
            range_key: asset.range_key.clone(),
            payload_path: asset.path.clone(),
            payload_bytes: asset.bytes,
            payload_sha256: asset.sha256.clone(),
            proof_path: asset.proof_path.clone(),
            proof_sha256: asset.proof_sha256.clone(),
            hash_authority: asset.hash_authority.clone(),
            expert_id,
        };
        let lease = StarfoldVerifiedMappedLease {
            identity,
            planned_identity: planned.identity.clone(),
            asset,
            payload: StarfoldVerifiedPayloadLease::Packed { source },
            last_validated_epoch: AtomicU64::new(validation_epoch),
        };
        lease.validate_planned(planned, expert_id)?;
        return Ok(lease);
    }

    let payload_before = file_stamp(&key.payload_path)?;
    let proof_before = file_stamp(&key.proof_path)?;
    if payload_before.bytes != key.payload_bytes {
        return Err(StarfoldError::new(format!(
            "Starfold payload bytes 漂移: expected={} actual={}",
            key.payload_bytes, payload_before.bytes
        )));
    }
    if proof_before.bytes == 0 || proof_before.bytes > STARFOLD_MAX_PROOF_BYTES {
        return Err(StarfoldError::new(format!(
            "Starfold proof bytes 超出 1..={STARFOLD_MAX_PROOF_BYTES}: actual={}",
            proof_before.bytes
        )));
    }

    let mut asset = planned
        .resolve_cached_position0_asset(canonical_root, expert_id)
        .map_err(|error| StarfoldError::new(format!("解析 Starfold Range proof: {error:#}")))?;
    planned
        .validate_resolved_position0_asset(&asset, expert_id)
        .map_err(|error| StarfoldError::new(format!("绑定 Starfold Range proof: {error:#}")))?;
    validate_lower_sha256(&asset.sha256, "payload")?;
    validate_lower_sha256(&asset.proof_sha256, "proof")?;

    let (mut proof_file, proof_open_stamp) = open_immutable_with_stamp(&key.proof_path, "proof")?;
    let proof_capacity = usize::try_from(proof_open_stamp.bytes)
        .map_err(|_| StarfoldError::new("Starfold proof bytes 超出 usize"))?;
    let mut proof_bytes = Vec::with_capacity(proof_capacity);
    proof_file.read_to_end(&mut proof_bytes).map_err(|error| {
        StarfoldError::new(format!(
            "读取 Starfold proof {}: {error}",
            key.proof_path.display()
        ))
    })?;
    if proof_bytes.len() as u64 != proof_open_stamp.bytes {
        return Err(StarfoldError::new("Starfold proof 读取长度漂移"));
    }
    let proof_sha256 = sha256_hex(&proof_bytes);
    if proof_sha256 != asset.proof_sha256 {
        return Err(StarfoldError::new(format!(
            "Starfold proof SHA-256 漂移: expected={} actual={proof_sha256}",
            asset.proof_sha256
        )));
    }

    let mut store = VerifiedMappedAssetStore::new(canonical_root).map_err(|error| {
        StarfoldError::new(format!("初始化 Starfold verified mmap store: {error:#}"))
    })?;
    let mapped = store
        .map_verified_batch(std::slice::from_ref(&asset))
        .map_err(|error| StarfoldError::new(format!("校验 Starfold payload SHA: {error:#}")))?
        .pop()
        .ok_or_else(|| StarfoldError::new("Starfold verified mmap lease 缺失"))?;

    let payload_after = file_stamp(&key.payload_path)?;
    let proof_after = file_stamp(&key.proof_path)?;
    if payload_before != payload_after
        || proof_before != proof_open_stamp
        || proof_open_stamp != proof_after
    {
        return Err(StarfoldError::new(
            "Starfold payload/proof 在验证期间发生身份漂移",
        ));
    }
    if mapped.path() != key.payload_path
        || mapped.tensor() != key.tensor
        || mapped.bytes().len() as u64 != key.payload_bytes
        || mapped.expected_sha256() != asset.sha256
        || asset.cache_key != key.cache_key
        || asset.range_key != key.range_key
        || asset.proof_path != key.proof_path
        || asset.expert_id != key.expert_id
    {
        return Err(StarfoldError::new("Starfold proof/SHA/mmap 强身份漂移"));
    }

    asset.payload_rehashed_by_builder = true;
    let identity = StarfoldVerifiedLeaseIdentity {
        tensor: asset.tensor.clone(),
        cache_key: asset.cache_key.clone(),
        range_key: asset.range_key.clone(),
        payload_path: key.payload_path.clone(),
        payload_bytes: asset.bytes,
        payload_sha256: asset.sha256.clone(),
        proof_path: key.proof_path.clone(),
        proof_sha256: asset.proof_sha256.clone(),
        hash_authority: asset.hash_authority.clone(),
        expert_id,
    };
    Ok(StarfoldVerifiedMappedLease {
        identity,
        planned_identity: planned.identity.clone(),
        asset,
        payload: StarfoldVerifiedPayloadLease::Loose {
            mapped,
            proof_file,
            payload_stamp: payload_after,
            proof_stamp: proof_after,
        },
        last_validated_epoch: AtomicU64::new(validation_epoch),
    })
}

fn file_stamp(path: &Path) -> StarfoldResult<StarfoldFileStamp> {
    let metadata = path.metadata().map_err(|error| {
        StarfoldError::new(format!(
            "stat Starfold verified file {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(StarfoldError::new(format!(
            "Starfold verified path 不是文件: {}",
            path.display()
        )));
    }
    let modified = metadata.modified().map_err(|error| {
        StarfoldError::new(format!(
            "读取 Starfold verified file mtime {}: {error}",
            path.display()
        ))
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(StarfoldFileStamp {
            bytes: metadata.len(),
            modified,
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        Ok(StarfoldFileStamp {
            bytes: metadata.len(),
            modified,
        })
    }
}

fn open_immutable_with_stamp(
    path: &Path,
    label: &str,
) -> StarfoldResult<(File, StarfoldFileStamp)> {
    let file = open_starfold_immutable(path).map_err(|error| {
        StarfoldError::new(format!(
            "打开 Starfold verified {label} {}: {error}",
            path.display()
        ))
    })?;
    let metadata = file.metadata().map_err(|error| {
        StarfoldError::new(format!(
            "stat Starfold verified {label} handle {}: {error}",
            path.display()
        ))
    })?;
    let stamp = file_stamp_from_metadata(path, &metadata)?;
    Ok((file, stamp))
}

fn file_stamp_from_metadata(
    path: &Path,
    metadata: &std::fs::Metadata,
) -> StarfoldResult<StarfoldFileStamp> {
    if !metadata.is_file() {
        return Err(StarfoldError::new(format!(
            "Starfold verified handle 不是文件: {}",
            path.display()
        )));
    }
    let modified = metadata.modified().map_err(|error| {
        StarfoldError::new(format!(
            "读取 Starfold verified handle mtime {}: {error}",
            path.display()
        ))
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(StarfoldFileStamp {
            bytes: metadata.len(),
            modified,
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        Ok(StarfoldFileStamp {
            bytes: metadata.len(),
            modified,
        })
    }
}

#[cfg(windows)]
fn open_starfold_immutable(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .share_mode(0x0000_0001)
        .open(path)
}

#[cfg(not(windows))]
fn open_starfold_immutable(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().read(true).open(path)
}

fn validate_lower_sha256(value: &str, label: &str) -> StarfoldResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(StarfoldError::new(format!(
            "Starfold {label} SHA-256 必须是 64 位小写十六进制"
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

static PROCESS_VERIFIED_LEASE_CACHE: OnceLock<StarfoldVerifiedLeaseCache> = OnceLock::new();

/// 默认进程级 owner。production root 应借用它而不是为每个请求创建 mmap store。
pub fn process_starfold_verified_lease_cache() -> &'static StarfoldVerifiedLeaseCache {
    PROCESS_VERIFIED_LEASE_CACHE.get_or_init(|| {
        StarfoldVerifiedLeaseCache::new(STARFOLD_DEFAULT_VERIFIED_LEASES)
            .expect("固定 Starfold verified lease capacity 必须合法")
    })
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
