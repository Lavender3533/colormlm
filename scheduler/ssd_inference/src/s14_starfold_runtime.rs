//! Polaris S14 StarFold 的 production host 接线。
//!
//! 本层只接权威 GPU top-6 路由、catalog proof identity、payload SHA lease 与恒定双
//! resident 窗口；单专家 microtile 与多专家 constellation 共用同一组
//! A/B 物理 owner。它不发布 token，也不替代既有最长可靠前缀事务提交。

use crate::{
    s14_dynamic_page_cache_readiness::{
        fetch_dynamic_page_plans_batched_only, fetch_planned_range_assets_batched_only,
        DynamicPageFetchMode, DynamicPagePlannedFetchAsset,
    },
    s14_dynamic_routed_page_plan::{
        DynamicRoutedPagePlan, FullDepthExpertCatalog, OnlineTop6, RoutedProjection,
        RoutedRangePart,
    },
    s14_input_asset_plan::S14PlannedRangeAsset,
    s14_starfold_cache::{
        process_starfold_verified_lease_cache, StarfoldB4MicroUnionPlan, StarfoldB4RouteBlock,
        StarfoldMicrotileLayout, StarfoldMicrotileSpan, StarfoldPageKey, StarfoldRouteEntry,
        StarfoldTensorSegment, StarfoldVerifiedLeaseCacheStats,
        StarfoldVerifiedLeaseValidationEpoch, StarfoldVerifiedMappedLease, STARFOLD_B4_LANES,
        STARFOLD_ONE_MIB, STARFOLD_TOP_K,
    },
    s14_starfold_expert_schedule::S14StarfoldExpertProjection,
    s14_starfold_packed_l2::{
        S14StarfoldPackedL2Cache, S14StarfoldPackedL2Config, S14StarfoldPackedL2Key,
        S14StarfoldPackedL2Stats,
    },
    s14_starfold_routed_executor::constellation_packet::{
        S14StarfoldConstellationPacket, S14StarfoldConstellationReadyPacket,
        S14StarfoldConstellationRuntimeHook, S14StarfoldResidentWindowKey,
    },
    s14_starfold_transfer_executor::S14StarfoldTransferExecutor,
    s14_starfold_vulkan_windows::{
        S14StarfoldBufferSpec, S14StarfoldComputePairRecording, S14StarfoldComputePairTicket,
        S14StarfoldComputeReceipt as VulkanComputeReceipt, S14StarfoldComputeRecording,
        S14StarfoldComputeTicket as VulkanComputeTicket, S14StarfoldQueueBinding,
        S14StarfoldReadyBinding as VulkanReadyBinding, S14StarfoldScratchBufferSpec,
        S14StarfoldTimelineBinding, S14StarfoldUploadRecording,
        S14StarfoldUploadTicket as VulkanUploadTicket, S14StarfoldVulkanConfig,
        S14StarfoldVulkanWindows,
    },
    s14_vulkan_arena::{S14ExternalDeviceMemory, S14VulkanArena, S14VulkanArenaStats},
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    GraphProfile, Position0Asset, RouteDecision, EXPERTS_PER_TOKEN, FULL_DEPTH_LAYERS,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

pub const S14_STARFOLD_WINDOW_COUNT: usize = 2;
pub const S14_STARFOLD_DEFAULT_MICROTILE_BYTES: u32 = STARFOLD_ONE_MIB;

/// K=4 使用一个 B4 route block，K=8 顺序使用两个；两者共用同一对物理窗口。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldDoubleWindowContract {
    pub window_count: usize,
    pub microtile_bytes: u32,
    pub required_vram_bytes: u64,
}

impl S14StarfoldDoubleWindowContract {
    pub fn new(microtile_bytes: u32) -> Result<Self> {
        if microtile_bytes == 0 {
            bail!("S14 StarFold microtile bytes 不能为0");
        }
        let required_vram_bytes = u64::from(microtile_bytes)
            .checked_mul(S14_STARFOLD_WINDOW_COUNT as u64)
            .context("S14 StarFold 双 microtile 窗口字节数溢出")?;
        Ok(Self {
            window_count: S14_STARFOLD_WINDOW_COUNT,
            microtile_bytes,
            required_vram_bytes,
        })
    }

    pub fn one_mib() -> Self {
        Self {
            window_count: S14_STARFOLD_WINDOW_COUNT,
            microtile_bytes: S14_STARFOLD_DEFAULT_MICROTILE_BYTES,
            required_vram_bytes: u64::from(S14_STARFOLD_DEFAULT_MICROTILE_BYTES)
                * S14_STARFOLD_WINDOW_COUNT as u64,
        }
    }
}

/// 物理 owner 的无资源构造器。它只固定双窗口容量合同，不复制 `VkDeviceMemory` 或
/// `VkBuffer`；owner 把自己创建的 arena 与未绑定 buffer specs 借给 `allocate`，得到
/// 唯一的 Vulkan 窗口状态机。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldVulkanWindowsFactory {
    contract: S14StarfoldDoubleWindowContract,
}

impl S14StarfoldVulkanWindowsFactory {
    pub fn new(contract: S14StarfoldDoubleWindowContract) -> Self {
        Self { contract }
    }

    pub fn contract(self) -> S14StarfoldDoubleWindowContract {
        self.contract
    }

    pub fn allocate(
        self,
        arena: &mut S14VulkanArena,
        config: S14StarfoldVulkanConfig,
        window_buffers: [S14StarfoldBufferSpec; S14_STARFOLD_WINDOW_COUNT],
        scratch_buffers: &[S14StarfoldScratchBufferSpec],
    ) -> Result<S14StarfoldVulkanWindows<S14StarfoldResidentWindowKey>> {
        if config.window_bytes != u64::from(self.contract.microtile_bytes) {
            bail!(
                "S14 StarFold Vulkan window bytes 与 runtime 合同漂移: config={} contract={}",
                config.window_bytes,
                self.contract.microtile_bytes
            );
        }
        S14StarfoldVulkanWindows::allocate(arena, config, window_buffers, scratch_buffers)
            .map_err(anyhow::Error::new)
            .context("从外部 Vulkan arena 构造 S14 StarFold 双窗口")
    }
}

/// 最小 production 物理 owner：两只 EXCLUSIVE storage/transfer-dst buffer 共用一块
/// device-local memory，并各自使用独立的 transfer/compute timeline semaphore。
/// scratch 当前为空；后续只能按真实 kernel 生命周期追加，不能恢复完整 union bank。
pub struct S14StarfoldVulkanResourceOwner {
    ctx: Arc<VulkanContext>,
    windows: Option<S14StarfoldVulkanWindows<S14StarfoldResidentWindowKey>>,
    arena: Option<S14VulkanArena>,
    window_buffers: [vk::Buffer; S14_STARFOLD_WINDOW_COUNT],
    memory: vk::DeviceMemory,
    allocation_bytes: u64,
    transfer_timeline: vk::Semaphore,
    compute_timeline: vk::Semaphore,
    destroyed: bool,
}

impl S14StarfoldVulkanResourceOwner {
    pub fn new(ctx: Arc<VulkanContext>, window_bytes: u32) -> Result<Self> {
        let contract = S14StarfoldDoubleWindowContract::new(window_bytes)?;
        let factory = S14StarfoldVulkanWindowsFactory::new(contract);
        let mut pending = PendingStarfoldVulkanResources::new(Arc::clone(&ctx));
        let buffer_info = vk::BufferCreateInfo::default()
            .size(u64::from(window_bytes))
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        for index in 0..S14_STARFOLD_WINDOW_COUNT {
            pending.window_buffers[index] = unsafe { ctx.device.create_buffer(&buffer_info, None) }
                .with_context(|| format!("创建 S14 StarFold Vulkan window {index} buffer"))?;
        }
        let requirements = pending
            .window_buffers
            .map(|buffer| unsafe { ctx.device.get_buffer_memory_requirements(buffer) });
        let common_memory_type_bits = requirements.iter().fold(u32::MAX, |bits, requirement| {
            bits & requirement.memory_type_bits
        });
        let memory_type_index = ctx
            .find_memory_type(
                common_memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            )
            .context("S14 StarFold 双窗口没有共同的纯 device-local memory type")?;
        let second_offset = align_up_device_size(requirements[0].size, requirements[1].alignment)
            .context("计算 S14 StarFold window B memory offset")?;
        let allocation_bytes = second_offset
            .checked_add(requirements[1].size)
            .context("S14 StarFold 双窗口 device memory bytes 溢出")?;
        pending.memory = unsafe {
            ctx.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(allocation_bytes)
                    .memory_type_index(memory_type_index),
                None,
            )
        }
        .context("分配 S14 StarFold 双窗口 device memory")?;

        pending.transfer_timeline = create_timeline_semaphore(&ctx.device)
            .context("创建 S14 StarFold transfer timeline semaphore")?;
        pending.compute_timeline = create_timeline_semaphore(&ctx.device)
            .context("创建 S14 StarFold compute timeline semaphore")?;

        let external = S14ExternalDeviceMemory::bind(pending.memory, 0, allocation_bytes)
            .map_err(anyhow::Error::new)
            .context("绑定 S14 StarFold 外部 device memory")?;
        let mut arena = S14VulkanArena::new(external, 1)
            .map_err(anyhow::Error::new)
            .context("创建 S14 StarFold Vulkan arena")?;
        let config = S14StarfoldVulkanConfig {
            window_bytes: u64::from(window_bytes),
            memory_type_index,
            queues: S14StarfoldQueueBinding {
                transfer_queue: ctx.q_transfer,
                transfer_family: ctx.qf_transfer,
                compute_queue: ctx.q_graphics,
                compute_family: ctx.qf_graphics,
            },
            timelines: S14StarfoldTimelineBinding {
                transfer: pending.transfer_timeline,
                compute: pending.compute_timeline,
                generation: 1,
                initial_transfer_value: 0,
                initial_compute_value: 0,
            },
        };
        let specs = std::array::from_fn(|index| S14StarfoldBufferSpec {
            buffer: pending.window_buffers[index],
            buffer_bytes: u64::from(window_bytes),
            memory_requirements: requirements[index],
        });
        let mut windows = factory.allocate(&mut arena, config, specs, &[])?;
        unsafe { windows.bind_buffers(&ctx.device) }
            .map_err(anyhow::Error::new)
            .context("绑定 S14 StarFold 双窗口 buffers 到统一 device memory")?;

        pending.armed = false;
        Ok(Self {
            ctx,
            windows: Some(windows),
            arena: Some(arena),
            window_buffers: pending.window_buffers,
            memory: pending.memory,
            allocation_bytes,
            transfer_timeline: pending.transfer_timeline,
            compute_timeline: pending.compute_timeline,
            destroyed: false,
        })
    }

    pub fn allocation_bytes(&self) -> u64 {
        self.allocation_bytes
    }

    pub fn arena_stats(&self) -> S14VulkanArenaStats {
        self.arena
            .as_ref()
            .map(S14VulkanArena::stats)
            .unwrap_or_default()
    }

    pub fn windows(&self) -> &S14StarfoldVulkanWindows<S14StarfoldResidentWindowKey> {
        self.windows
            .as_ref()
            .expect("S14 StarFold Vulkan owner 已销毁")
    }

    pub fn windows_mut(&mut self) -> &mut S14StarfoldVulkanWindows<S14StarfoldResidentWindowKey> {
        self.windows
            .as_mut()
            .expect("S14 StarFold Vulkan owner 已销毁")
    }

    unsafe fn submit_upload(
        &mut self,
        ticket: VulkanUploadTicket<S14StarfoldResidentWindowKey>,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> Result<VulkanReadyBinding<S14StarfoldResidentWindowKey>> {
        let device = self.ctx.device.clone();
        unsafe {
            self.windows_mut()
                .submit_upload(&device, ticket, command_buffer, fence)
        }
        .map_err(anyhow::Error::new)
        .context("提交 S14 StarFold Vulkan upload")
    }

    unsafe fn submit_compute(
        &mut self,
        ticket: VulkanComputeTicket<S14StarfoldResidentWindowKey>,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> Result<VulkanComputeReceipt<S14StarfoldResidentWindowKey>> {
        let device = self.ctx.device.clone();
        unsafe {
            self.windows_mut()
                .submit_compute(&device, ticket, command_buffer, fence)
        }
        .map_err(anyhow::Error::new)
        .context("提交 S14 StarFold Vulkan compute")
    }

    unsafe fn submit_compute_pair(
        &mut self,
        ticket: S14StarfoldComputePairTicket<S14StarfoldResidentWindowKey>,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> Result<[VulkanComputeReceipt<S14StarfoldResidentWindowKey>; 2]> {
        let device = self.ctx.device.clone();
        unsafe {
            self.windows_mut()
                .submit_compute_pair(&device, ticket, command_buffer, fence)
        }
        .map_err(anyhow::Error::new)
        .context("提交 S14 StarFold Vulkan weight/scale pair compute")
    }

    pub fn destroy(mut self) -> Result<()> {
        self.destroy_inner()
    }

    fn destroy_inner(&mut self) -> Result<()> {
        if self.destroyed {
            return Ok(());
        }
        let wait_error = unsafe { self.ctx.device.device_wait_idle() }
            .err()
            .map(|error| anyhow!(error).context("等待 S14 StarFold Vulkan owner idle"));
        let release_error = if wait_error.is_none() {
            match (self.windows.take(), self.arena.as_mut()) {
                (Some(windows), Some(arena)) => {
                    let completed = windows.submitted_timelines();
                    windows
                        .release(arena, completed)
                        .map_err(anyhow::Error::new)
                        .context("释放 S14 StarFold Vulkan windows arena slices")
                        .err()
                }
                _ => None,
            }
        } else {
            self.windows.take();
            None
        };
        self.arena.take();
        unsafe {
            for buffer in self.window_buffers {
                if buffer != vk::Buffer::null() {
                    self.ctx.device.destroy_buffer(buffer, None);
                }
            }
            if self.transfer_timeline != vk::Semaphore::null() {
                self.ctx
                    .device
                    .destroy_semaphore(self.transfer_timeline, None);
            }
            if self.compute_timeline != vk::Semaphore::null() {
                self.ctx
                    .device
                    .destroy_semaphore(self.compute_timeline, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                self.ctx.device.free_memory(self.memory, None);
            }
        }
        self.window_buffers = [vk::Buffer::null(); S14_STARFOLD_WINDOW_COUNT];
        self.memory = vk::DeviceMemory::null();
        self.transfer_timeline = vk::Semaphore::null();
        self.compute_timeline = vk::Semaphore::null();
        self.destroyed = true;
        if let Some(error) = wait_error {
            return Err(error);
        }
        if let Some(error) = release_error {
            return Err(error);
        }
        Ok(())
    }
}

impl Drop for S14StarfoldVulkanResourceOwner {
    fn drop(&mut self) {
        let _ = self.destroy_inner();
    }
}

struct PendingStarfoldVulkanResources {
    ctx: Arc<VulkanContext>,
    window_buffers: [vk::Buffer; S14_STARFOLD_WINDOW_COUNT],
    memory: vk::DeviceMemory,
    transfer_timeline: vk::Semaphore,
    compute_timeline: vk::Semaphore,
    armed: bool,
}

impl PendingStarfoldVulkanResources {
    fn new(ctx: Arc<VulkanContext>) -> Self {
        Self {
            ctx,
            window_buffers: [vk::Buffer::null(); S14_STARFOLD_WINDOW_COUNT],
            memory: vk::DeviceMemory::null(),
            transfer_timeline: vk::Semaphore::null(),
            compute_timeline: vk::Semaphore::null(),
            armed: true,
        }
    }
}

impl Drop for PendingStarfoldVulkanResources {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        unsafe {
            for buffer in self.window_buffers {
                if buffer != vk::Buffer::null() {
                    self.ctx.device.destroy_buffer(buffer, None);
                }
            }
            if self.transfer_timeline != vk::Semaphore::null() {
                self.ctx
                    .device
                    .destroy_semaphore(self.transfer_timeline, None);
            }
            if self.compute_timeline != vk::Semaphore::null() {
                self.ctx
                    .device
                    .destroy_semaphore(self.compute_timeline, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                self.ctx.device.free_memory(self.memory, None);
            }
        }
    }
}

fn create_timeline_semaphore(device: &ash::Device) -> Result<vk::Semaphore> {
    let mut timeline = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    unsafe {
        device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
            None,
        )
    }
    .map_err(anyhow::Error::new)
}

fn align_up_device_size(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("S14 StarFold Vulkan memory alignment 非法: {alignment}");
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("S14 StarFold Vulkan memory alignment 溢出")
}

/// StarFold 只生产 future；权威 state 的发布继续使用既有最长可靠前缀两阶段 commit。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldCommitBoundary {
    ExistingLongestReliablePrefix,
}

/// 一个 microtile 的完整 Range proof 来源。`planned` 绑定 repo/revision、源文件、
/// header SHA、Range offset、proof 路径和 payload cache key。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldMicrotileSource {
    pub span: StarfoldMicrotileSpan,
    pub planned: S14PlannedRangeAsset,
}

/// 一层连续四个位置的无损 StarFold 计划。
#[derive(Clone, Debug)]
pub struct S14StarfoldB4LayerPlan {
    pub authoritative_routes: StarfoldB4RouteBlock,
    pub micro_union: StarfoldB4MicroUnionPlan,
    route_plans: Vec<DynamicRoutedPagePlan>,
    microtile_sources: Vec<S14StarfoldMicrotileSource>,
}

impl S14StarfoldB4LayerPlan {
    pub fn route_plans(&self) -> &[DynamicRoutedPagePlan] {
        &self.route_plans
    }

    pub fn microtile_sources(&self) -> &[S14StarfoldMicrotileSource] {
        &self.microtile_sources
    }

    pub fn source_for_microtile(&self, index: usize) -> Option<&S14StarfoldMicrotileSource> {
        self.microtile_sources.get(index)
    }
}

/// K=4/8 的层计划。K 只改变 B4 事务块数量，不改变常驻 VRAM 字节数。
#[derive(Clone, Debug)]
pub struct S14StarfoldKBlockLayerPlan {
    pub layer: u16,
    pub base_position: u64,
    pub block_size: usize,
    pub streamed_bytes: u64,
    pub windows: S14StarfoldDoubleWindowContract,
    pub commit_boundary: S14StarfoldCommitBoundary,
    pub b4_blocks: Vec<S14StarfoldB4LayerPlan>,
}

/// 把既有权威 `RouteDecision` 精确转换成 B=4/top-6 StarFold route。
pub fn build_starfold_b4_route_block(
    base_position: u64,
    routes: &[RouteDecision],
) -> Result<StarfoldB4RouteBlock> {
    if routes.len() != STARFOLD_B4_LANES {
        bail!("S14 StarFold route 要求精确 B=4，实际 B={}", routes.len());
    }
    let first = routes.first().context("S14 StarFold B4 route 为空")?;
    let layer = first.layer;
    if !FULL_DEPTH_LAYERS.contains(&layer) {
        bail!("S14 StarFold route layer {layer} 不在 FullDepth43");
    }

    let mut lane_rows = Vec::with_capacity(STARFOLD_B4_LANES);
    for (lane, route) in routes.iter().enumerate() {
        route
            .validate_for(GraphProfile::FullDepth43NativeTop6)
            .map_err(anyhow::Error::new)
            .with_context(|| format!("校验 S14 StarFold lane {lane} 权威 top-6 route"))?;
        if route.layer != layer {
            bail!("S14 StarFold B4 route layer 漂移: lane={lane}");
        }
        let entries = route
            .expert_ids
            .iter()
            .copied()
            .zip(route.weights.iter().copied())
            .map(|(expert_id, weight)| StarfoldRouteEntry { expert_id, weight })
            .collect::<Vec<_>>();
        let row: [StarfoldRouteEntry; STARFOLD_TOP_K] =
            entries.try_into().map_err(|entries: Vec<_>| {
                anyhow!(
                    "S14 StarFold lane {lane} 不是精确 top-{STARFOLD_TOP_K}: {}",
                    entries.len()
                )
            })?;
        lane_rows.push(row);
    }
    let lanes: [[StarfoldRouteEntry; STARFOLD_TOP_K]; STARFOLD_B4_LANES] = lane_rows
        .try_into()
        .map_err(|rows: Vec<_>| anyhow!("S14 StarFold B4 lane 数漂移: {}", rows.len()))?;
    StarfoldB4RouteBlock::new(u16::from(layer), base_position, lanes)
        .map_err(anyhow::Error::new)
        .context("构造 S14 StarFold B4 route block")
}

/// 从权威 B4 route、完整 catalog identity 构建 micro-union。这里只规划，不打开 payload。
pub fn build_starfold_b4_layer_plan(
    catalog: &FullDepthExpertCatalog,
    cache_root: &Path,
    base_position: u64,
    routes: &[RouteDecision],
    microtile_bytes: u32,
) -> Result<S14StarfoldB4LayerPlan> {
    let authoritative_routes = build_starfold_b4_route_block(base_position, routes)?;
    let route_plans = build_catalog_route_plans(catalog, &authoritative_routes)?;
    let mut sources = BTreeMap::<(u16, StarfoldTensorSegment), S14PlannedRangeAsset>::new();
    let mut segment_bytes = [None; 6];

    for route_plan in &route_plans {
        for physical in route_plan.physical_ranges()? {
            let segment = starfold_segment(physical.projection, physical.part);
            let slot = &mut segment_bytes[segment.ordinal()];
            match *slot {
                Some(expected) if expected != physical.range.bytes => {
                    bail!(
                        "S14 StarFold {:?} segment bytes 在专家间漂移: expected={expected} actual={}",
                        segment,
                        physical.range.bytes
                    );
                }
                None => *slot = Some(physical.range.bytes),
                _ => {}
            }
            let planned = physical
                .planned_asset(cache_root)
                .context("绑定 S14 StarFold microtile Range proof identity")?;
            let key = (physical.expert_id, segment);
            if let Some(existing) = sources.get(&key) {
                if existing != &planned {
                    bail!("S14 StarFold 同一专家 segment 的 proof identity 漂移");
                }
            } else {
                sources.insert(key, planned);
            }
        }
    }

    let segment_bytes: [u64; 6] = segment_bytes
        .into_iter()
        .enumerate()
        .map(|(index, bytes)| {
            bytes.with_context(|| format!("S14 StarFold 缺少 segment {index} bytes"))
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|bytes: Vec<_>| anyhow!("S14 StarFold segment count 漂移: {}", bytes.len()))?;
    let layout = StarfoldMicrotileLayout::new(segment_bytes, microtile_bytes)
        .map_err(anyhow::Error::new)
        .context("构造 S14 StarFold microtile layout")?;
    let micro_union = StarfoldB4MicroUnionPlan::build(&authoritative_routes, layout)
        .map_err(anyhow::Error::new)
        .context("构造 S14 StarFold B4 micro-union")?;
    let microtile_sources = micro_union
        .microtiles
        .iter()
        .map(|span| {
            let planned = sources
                .get(&(span.key.expert_id, span.key.segment))
                .with_context(|| {
                    format!(
                        "S14 StarFold 缺少 expert={} segment={:?} proof source",
                        span.key.expert_id, span.key.segment
                    )
                })?
                .clone();
            let end = span
                .source_segment_offset
                .checked_add(u64::from(span.byte_len))
                .context("S14 StarFold microtile source offset overflow")?;
            if end > planned.bytes || span.byte_len > microtile_bytes {
                bail!("S14 StarFold microtile 越出 proof-bound segment");
            }
            Ok(S14StarfoldMicrotileSource {
                span: *span,
                planned,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if microtile_sources.len() != micro_union.microtile_count() {
        bail!("S14 StarFold microtile/source 数量漂移");
    }
    Ok(S14StarfoldB4LayerPlan {
        authoritative_routes,
        micro_union,
        route_plans,
        microtile_sources,
    })
}

/// K8 被拆成两个连续 B4 block；两者复用同一恒定双窗口。
pub fn build_starfold_k4_k8_layer_plan(
    catalog: &FullDepthExpertCatalog,
    cache_root: &Path,
    base_position: u64,
    routes: &[RouteDecision],
    microtile_bytes: u32,
) -> Result<S14StarfoldKBlockLayerPlan> {
    if !matches!(routes.len(), 4 | 8) {
        bail!("S14 StarFold production block 只接受 K=4/8");
    }
    let layer = routes[0].layer;
    if routes.iter().any(|route| route.layer != layer) {
        bail!("S14 StarFold K-block route layer 漂移");
    }
    let windows = S14StarfoldDoubleWindowContract::new(microtile_bytes)?;
    let mut b4_blocks = Vec::with_capacity(routes.len() / STARFOLD_B4_LANES);
    let mut streamed_bytes = 0u64;
    for (block_index, lanes) in routes.chunks_exact(STARFOLD_B4_LANES).enumerate() {
        let block_position = base_position
            .checked_add((block_index * STARFOLD_B4_LANES) as u64)
            .context("S14 StarFold K-block position overflow")?;
        let block = build_starfold_b4_layer_plan(
            catalog,
            cache_root,
            block_position,
            lanes,
            microtile_bytes,
        )?;
        streamed_bytes = streamed_bytes
            .checked_add(block.micro_union.streamed_bytes)
            .context("S14 StarFold K-block streamed bytes overflow")?;
        b4_blocks.push(block);
    }
    Ok(S14StarfoldKBlockLayerPlan {
        layer: u16::from(layer),
        base_position,
        block_size: routes.len(),
        streamed_bytes,
        windows,
        commit_boundary: S14StarfoldCommitBoundary::ExistingLongestReliablePrefix,
        b4_blocks,
    })
}

fn build_catalog_route_plans(
    catalog: &FullDepthExpertCatalog,
    routes: &StarfoldB4RouteBlock,
) -> Result<Vec<DynamicRoutedPagePlan>> {
    routes
        .lanes()
        .iter()
        .enumerate()
        .map(|(lane, entries)| {
            let layer = u8::try_from(routes.layer()).context("S14 StarFold layer 超出 u8")?;
            let expert_ids = entries.map(|entry| entry.expert_id);
            let route_weights = entries.map(|entry| entry.weight);
            if EXPERTS_PER_TOKEN != STARFOLD_TOP_K {
                bail!("S14 StarFold top-k 与 runner ABI 漂移");
            }
            catalog
                .plan(OnlineTop6 {
                    layer,
                    position: routes
                        .lane_position(lane)
                        .context("S14 StarFold lane position 缺失")?,
                    expert_ids,
                    route_weights,
                })
                .with_context(|| format!("构造 S14 StarFold lane {lane} catalog route plan"))
        })
        .collect()
}

const fn starfold_segment(
    projection: RoutedProjection,
    part: RoutedRangePart,
) -> StarfoldTensorSegment {
    match (projection, part) {
        (RoutedProjection::W1, RoutedRangePart::Weight) => StarfoldTensorSegment::W1Weight,
        (RoutedProjection::W1, RoutedRangePart::Scale) => StarfoldTensorSegment::W1Scale,
        (RoutedProjection::W2, RoutedRangePart::Weight) => StarfoldTensorSegment::W2Weight,
        (RoutedProjection::W2, RoutedRangePart::Scale) => StarfoldTensorSegment::W2Scale,
        (RoutedProjection::W3, RoutedRangePart::Weight) => StarfoldTensorSegment::W3Weight,
        (RoutedProjection::W3, RoutedRangePart::Scale) => StarfoldTensorSegment::W3Scale,
    }
}

fn validate_mxfp4_proof_pair(
    weight: &S14StarfoldMicrotileSource,
    scale: &S14StarfoldMicrotileSource,
) -> Result<()> {
    let expected_scale = match weight.span.key.segment {
        StarfoldTensorSegment::W1Weight => StarfoldTensorSegment::W1Scale,
        StarfoldTensorSegment::W2Weight => StarfoldTensorSegment::W2Scale,
        StarfoldTensorSegment::W3Weight => StarfoldTensorSegment::W3Scale,
        segment => bail!("S14 StarFold MXFP4 weight proof segment 非 weight: {segment:?}"),
    };
    if scale.span.key.segment != expected_scale
        || scale.span.key.layer != weight.span.key.layer
        || scale.span.key.expert_id != weight.span.key.expert_id
    {
        bail!(
            "S14 StarFold MXFP4 weight/scale proof 不属于同一 layer/expert/projection: weight={:?}, scale={:?}",
            weight.span.key,
            scale.span.key
        );
    }
    Ok(())
}

/// 单一 Range proof 的零拷贝 lease。`bytes()` 始终直接借用 immutable mmap。
#[derive(Debug)]
pub struct S14StarfoldSingleProofLease {
    source: S14StarfoldMicrotileSource,
    asset: Position0Asset,
    hot_lease: Arc<StarfoldVerifiedMappedLease>,
}

/// Packed MXFP4 的稳定 payload 布局：`[weight bytes][scale bytes]`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldPackedMxfp4Layout {
    pub weight_offset: u64,
    pub weight_bytes: u64,
    pub scale_offset: u64,
    pub scale_bytes: u64,
    pub total_bytes: u64,
}

/// 已通过原始 Range proof/SHA 门的不可变身份快照。
///
/// Packed payload 已经拥有独立字节副本，因此后续只需要保留能够重建 packet identity 的
/// 权威元数据，不应继续钉住原始 mmap lease。这样 RAM L2 淘汰和 L3 文件回收不会被每个
/// packed entry 隐式阻塞；数值字节和 source proof 身份仍完整绑定。
#[derive(Clone, Debug)]
pub struct S14StarfoldPackedMxfp4SourceIdentity {
    source: S14StarfoldMicrotileSource,
    asset: Position0Asset,
}

impl S14StarfoldPackedMxfp4SourceIdentity {
    pub fn source(&self) -> &S14StarfoldMicrotileSource {
        &self.source
    }

    pub fn asset(&self) -> &Position0Asset {
        &self.asset
    }
}

/// 复合 MXFP4 payload lease。packed bytes 只构造一次；weight/scale 保存已经通过验证的
/// 身份快照，而不是继续强持有原始 mmap proof。
#[derive(Debug)]
pub struct S14StarfoldPackedMxfp4ProofLease {
    bytes: Arc<[u8]>,
    weight: S14StarfoldPackedMxfp4SourceIdentity,
    scale: S14StarfoldPackedMxfp4SourceIdentity,
    layout: S14StarfoldPackedMxfp4Layout,
}

/// 已通过 cache proof、payload SHA-256 的上传 payload。
///
/// `Single` 不复制 mmap；`PackedMxfp4` 固定拼接 weight/scale，并保存两份已验证身份快照。
#[derive(Debug)]
pub enum S14StarfoldVerifiedMicrotile {
    Single(S14StarfoldSingleProofLease),
    PackedMxfp4(S14StarfoldPackedMxfp4ProofLease),
}

impl S14StarfoldVerifiedMicrotile {
    /// payload 的权威窗口 key。Packed MXFP4 使用 weight proof key。
    pub fn key(&self) -> StarfoldPageKey {
        self.source().span.key
    }

    /// Single 返回自身 source；Packed MXFP4 返回主 weight source。
    pub fn source(&self) -> &S14StarfoldMicrotileSource {
        match self {
            Self::Single(single) => &single.source,
            Self::PackedMxfp4(packed) => packed.weight.source(),
        }
    }

    /// Single 返回自身 asset；Packed MXFP4 返回主 weight asset。
    pub fn asset(&self) -> &Position0Asset {
        match self {
            Self::Single(single) => &single.asset,
            Self::PackedMxfp4(packed) => packed.weight.asset(),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::Single(single) => single.bytes(),
            Self::PackedMxfp4(packed) => &packed.bytes,
        }
    }

    pub fn byte_len(&self) -> u64 {
        match self {
            Self::Single(single) => u64::from(single.source.span.byte_len),
            Self::PackedMxfp4(packed) => packed.layout.total_bytes,
        }
    }

    pub fn as_single(&self) -> Option<&S14StarfoldSingleProofLease> {
        match self {
            Self::Single(single) => Some(single),
            Self::PackedMxfp4(_) => None,
        }
    }

    pub fn packed_mxfp4(&self) -> Option<&S14StarfoldPackedMxfp4ProofLease> {
        match self {
            Self::Single(_) => None,
            Self::PackedMxfp4(packed) => Some(packed),
        }
    }
}

impl S14StarfoldSingleProofLease {
    pub fn source(&self) -> &S14StarfoldMicrotileSource {
        &self.source
    }

    pub fn asset(&self) -> &Position0Asset {
        &self.asset
    }

    pub fn bytes(&self) -> &[u8] {
        self.hot_lease
            .microtile(
                self.source.span.source_segment_offset,
                self.source.span.byte_len,
            )
            .expect("S14 StarFold proof-bound microtile 已在构造时验证")
    }
}

impl S14StarfoldPackedMxfp4ProofLease {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn layout(&self) -> S14StarfoldPackedMxfp4Layout {
        self.layout
    }

    pub fn weight_identity(&self) -> &S14StarfoldPackedMxfp4SourceIdentity {
        &self.weight
    }

    pub fn scale_identity(&self) -> &S14StarfoldPackedMxfp4SourceIdentity {
        &self.scale
    }
}

#[derive(Debug)]
pub struct S14StarfoldUploadTicket {
    ticket: VulkanUploadTicket<S14StarfoldResidentWindowKey>,
    recording: S14StarfoldUploadRecording,
    verified: Arc<S14StarfoldVerifiedMicrotile>,
}

impl S14StarfoldUploadTicket {
    pub fn ticket(&self) -> VulkanUploadTicket<S14StarfoldResidentWindowKey> {
        self.ticket
    }

    pub fn recording(&self) -> S14StarfoldUploadRecording {
        self.recording
    }

    pub fn bytes(&self) -> &[u8] {
        self.verified.bytes()
    }

    pub fn proof(&self) -> &Arc<S14StarfoldVerifiedMicrotile> {
        &self.verified
    }
}

#[derive(Clone, Debug)]
pub struct S14StarfoldReadyMicrotile {
    binding: VulkanReadyBinding<S14StarfoldResidentWindowKey>,
    verified: Arc<S14StarfoldVerifiedMicrotile>,
}

impl S14StarfoldReadyMicrotile {
    pub fn binding(&self) -> VulkanReadyBinding<S14StarfoldResidentWindowKey> {
        self.binding
    }

    pub fn source(&self) -> &S14StarfoldMicrotileSource {
        self.verified.source()
    }

    pub fn proof(&self) -> &Arc<S14StarfoldVerifiedMicrotile> {
        &self.verified
    }
}

#[derive(Debug)]
pub struct S14StarfoldComputeMicrotile {
    ticket: VulkanComputeTicket<S14StarfoldResidentWindowKey>,
    recording: S14StarfoldComputeRecording,
    verified: Arc<S14StarfoldVerifiedMicrotile>,
}

impl S14StarfoldComputeMicrotile {
    pub fn ticket(&self) -> VulkanComputeTicket<S14StarfoldResidentWindowKey> {
        self.ticket
    }

    pub fn recording(&self) -> S14StarfoldComputeRecording {
        self.recording
    }

    pub fn source(&self) -> &S14StarfoldMicrotileSource {
        self.verified.source()
    }

    pub fn proof(&self) -> &Arc<S14StarfoldVerifiedMicrotile> {
        &self.verified
    }
}

/// compute 已提交后的 verified payload/identity lease；调用方至少保留到
/// `receipt.completion` 完成。
#[derive(Debug)]
pub struct S14StarfoldComputeSubmissionReceipt {
    receipt: VulkanComputeReceipt<S14StarfoldResidentWindowKey>,
    verified: Arc<S14StarfoldVerifiedMicrotile>,
}

#[derive(Debug)]
pub struct S14StarfoldComputePairMicrotile {
    ticket: S14StarfoldComputePairTicket<S14StarfoldResidentWindowKey>,
    recording: S14StarfoldComputePairRecording,
    verified: [Arc<S14StarfoldVerifiedMicrotile>; 2],
}

impl S14StarfoldComputePairMicrotile {
    pub fn ticket(&self) -> S14StarfoldComputePairTicket<S14StarfoldResidentWindowKey> {
        self.ticket
    }

    pub fn recording(&self) -> S14StarfoldComputePairRecording {
        self.recording
    }

    pub fn sources(&self) -> [&S14StarfoldMicrotileSource; 2] {
        [self.verified[0].source(), self.verified[1].source()]
    }

    pub fn proofs(&self) -> [&Arc<S14StarfoldVerifiedMicrotile>; 2] {
        [&self.verified[0], &self.verified[1]]
    }
}

#[derive(Debug)]
pub struct S14StarfoldComputePairSubmissionReceipt {
    receipts: [VulkanComputeReceipt<S14StarfoldResidentWindowKey>; 2],
    verified: [Arc<S14StarfoldVerifiedMicrotile>; 2],
}

impl S14StarfoldComputePairSubmissionReceipt {
    pub fn receipts(&self) -> [VulkanComputeReceipt<S14StarfoldResidentWindowKey>; 2] {
        self.receipts
    }

    pub fn sources(&self) -> [&S14StarfoldMicrotileSource; 2] {
        [self.verified[0].source(), self.verified[1].source()]
    }

    pub fn proofs(&self) -> [&Arc<S14StarfoldVerifiedMicrotile>; 2] {
        [&self.verified[0], &self.verified[1]]
    }
}

impl S14StarfoldComputeSubmissionReceipt {
    pub fn receipt(&self) -> VulkanComputeReceipt<S14StarfoldResidentWindowKey> {
        self.receipt
    }

    pub fn source(&self) -> &S14StarfoldMicrotileSource {
        self.verified.source()
    }

    pub fn proof(&self) -> &Arc<S14StarfoldVerifiedMicrotile> {
        &self.verified
    }
}

/// 生产 owner：借用进程级 proof/SHA mmap 热 lease 缓存，并独占唯一 Vulkan 双窗口
/// 物理 executor。下面的票据接口让同一 proof lease 穿过 upload→ready→compute；
/// 这里不发布 token。
pub struct S14StarfoldRuntime {
    cache_root: PathBuf,
    page_fetch_mode: DynamicPageFetchMode,
    validation_epoch: StarfoldVerifiedLeaseValidationEpoch,
    contract: S14StarfoldDoubleWindowContract,
    resource_owner: Option<S14StarfoldVulkanResourceOwner>,
    transfer_executor: Option<S14StarfoldTransferExecutor>,
    packed_l2: S14StarfoldPackedL2Cache,
}

impl S14StarfoldRuntime {
    pub fn new(ctx: Arc<VulkanContext>, cache_root: &Path, microtile_bytes: u32) -> Result<Self> {
        Self::new_with_fetch_mode(
            ctx,
            cache_root,
            microtile_bytes,
            DynamicPageFetchMode::LocalOnly,
        )
    }

    pub fn new_with_fetch_mode(
        ctx: Arc<VulkanContext>,
        cache_root: &Path,
        microtile_bytes: u32,
        page_fetch_mode: DynamicPageFetchMode,
    ) -> Result<Self> {
        let contract = S14StarfoldDoubleWindowContract::new(microtile_bytes)?;
        let resource_owner = S14StarfoldVulkanResourceOwner::new(ctx, microtile_bytes)
            .context("初始化 S14 StarFold Vulkan 双窗口物理 owner")?;
        let transfer_executor = S14StarfoldTransferExecutor::new(
            Arc::clone(&resource_owner.ctx),
            u64::from(microtile_bytes),
        )
        .context("初始化 S14 StarFold 常驻双 staging transfer executor")?;
        let validation_epoch = process_starfold_verified_lease_cache()
            .begin_validation_epoch()
            .map_err(anyhow::Error::new)
            .context("签发 S14 StarFold 初始 verified lease validation epoch")?;
        let packed_l2 = S14StarfoldPackedL2Cache::new(S14StarfoldPackedL2Config::from_env()?)
            .context("初始化 S14 StarFold packed MXFP4 RAM L2")?;
        Ok(Self {
            cache_root: cache_root.to_path_buf(),
            page_fetch_mode,
            validation_epoch,
            contract,
            resource_owner: Some(resource_owner),
            transfer_executor: Some(transfer_executor),
            packed_l2,
        })
    }

    pub fn one_mib(ctx: Arc<VulkanContext>, cache_root: &Path) -> Result<Self> {
        Self::new(ctx, cache_root, S14_STARFOLD_DEFAULT_MICROTILE_BYTES)
    }

    pub fn contract(&self) -> S14StarfoldDoubleWindowContract {
        self.contract
    }

    pub(crate) fn context(&self) -> Option<&Arc<VulkanContext>> {
        self.resource_owner.as_ref().map(|owner| &owner.ctx)
    }

    /// 将同一 B4 层的四条权威路由按 `range_key` 去重后交给现有常驻
    /// Range transport。这里只补 cache，不创建 mmap/model lease；随后每个
    /// microtile 仍由 `verify_microtile` 独立执行 proof/SHA 门。
    pub fn fetch_b4_layer_ranges(&self, plan: &S14StarfoldB4LayerPlan) -> Result<()> {
        fetch_dynamic_page_plans_batched_only(
            plan.route_plans(),
            &self.cache_root,
            self.page_fetch_mode,
        )
        .map(|_| ())
        .map_err(anyhow::Error::new)
        .context("补齐 S14 StarFold B4 exact-route Range cache")
    }

    /// Constellation-only cold-page gate.  Unlike `fetch_b4_layer_ranges`, this
    /// selects exactly the weight/scale Range pair for each expert carried by
    /// the next immutable packet.  The following `verify_microtile` calls still
    /// own proof/SHA/mmap publication; this method only makes those files ready.
    pub fn fetch_b4_packet_ranges(
        &self,
        plan: &S14StarfoldB4LayerPlan,
        projection: S14StarfoldExpertProjection,
        expert_ids: &[u16],
    ) -> Result<()> {
        if expert_ids.is_empty() || expert_ids.len() > STARFOLD_B4_LANES * STARFOLD_TOP_K {
            bail!(
                "S14 StarFold packet expert 数量非法: actual={}, max={}",
                expert_ids.len(),
                STARFOLD_B4_LANES * STARFOLD_TOP_K
            );
        }
        let layer = u8::try_from(plan.authoritative_routes.layer())
            .context("S14 StarFold packet layer 超出 u8")?;
        let position = plan.authoritative_routes.base_position();
        let mut seen = BTreeMap::<u16, ()>::new();
        let mut assets = Vec::with_capacity(expert_ids.len() * 2);
        for &expert_id in expert_ids {
            if seen.insert(expert_id, ()).is_some() {
                bail!("S14 StarFold packet expert identity 重复: {expert_id}");
            }
            for segment in [projection.weight_segment(), projection.scale_segment()] {
                let source = plan
                    .microtile_sources()
                    .iter()
                    .find(|source| {
                        source.span.key.expert_id == expert_id
                            && source.span.key.segment == segment
                    })
                    .with_context(|| {
                        format!(
                            "S14 StarFold packet 缺少 expert={expert_id} segment={segment:?} Range source"
                        )
                    })?;
                assets.push(DynamicPagePlannedFetchAsset::new(
                    expert_id,
                    source.planned.clone(),
                ));
            }
        }
        fetch_planned_range_assets_batched_only(
            layer,
            position,
            &assets,
            &self.cache_root,
            self.page_fetch_mode,
        )
        .map(|_| ())
        .map_err(anyhow::Error::new)
        .context("补齐 S14 StarFold constellation packet Range cache")
    }

    /// StarFold compute owner 的唯一窗口入口。窗口仍由 runtime/resource owner 独占，
    /// 外部执行器只能在一次调用期间借用，不能复制或替换物理 owner。
    pub(crate) fn vulkan_windows_mut(
        &mut self,
    ) -> Result<&mut S14StarfoldVulkanWindows<S14StarfoldResidentWindowKey>> {
        Ok(self
            .resource_owner
            .as_mut()
            .context("S14 StarFold Vulkan owner 已销毁")?
            .windows_mut())
    }

    pub fn verified_lease_cache_stats(&self) -> Result<StarfoldVerifiedLeaseCacheStats> {
        process_starfold_verified_lease_cache()
            .stats()
            .map_err(anyhow::Error::new)
            .context("读取 S14 StarFold 进程级 verified lease cache stats")
    }

    pub fn packed_l2_cache_stats(&self) -> S14StarfoldPackedL2Stats {
        self.packed_l2.stats()
    }

    /// 每个 prompt request 只签发一次；同一请求的所有 K4 block 共用该纪元。
    pub fn begin_verified_lease_request_epoch(&mut self) -> Result<u64> {
        let epoch = process_starfold_verified_lease_cache()
            .begin_validation_epoch()
            .map_err(anyhow::Error::new)
            .context("签发 S14 StarFold request verified lease validation epoch")?;
        self.validation_epoch = epoch;
        Ok(epoch.id())
    }

    pub fn physical_allocation_bytes(&self) -> u64 {
        self.resource_owner
            .as_ref()
            .map(S14StarfoldVulkanResourceOwner::allocation_bytes)
            .unwrap_or(0)
    }

    pub fn physical_arena_stats(&self) -> S14VulkanArenaStats {
        self.resource_owner
            .as_ref()
            .map(S14StarfoldVulkanResourceOwner::arena_stats)
            .unwrap_or_default()
    }

    pub fn plan_k4_k8_layer(
        &self,
        catalog: &FullDepthExpertCatalog,
        base_position: u64,
        routes: &[RouteDecision],
    ) -> Result<S14StarfoldKBlockLayerPlan> {
        build_starfold_k4_k8_layer_plan(
            catalog,
            &self.cache_root,
            base_position,
            routes,
            self.contract.microtile_bytes,
        )
    }

    pub fn verify_microtile(
        &mut self,
        source: &S14StarfoldMicrotileSource,
    ) -> Result<Arc<S14StarfoldVerifiedMicrotile>> {
        if source.span.byte_len == 0 || source.span.byte_len > self.contract.microtile_bytes {
            bail!("S14 StarFold microtile bytes 超出双窗口合同");
        }
        let end = source
            .span
            .source_segment_offset
            .checked_add(u64::from(source.span.byte_len))
            .context("S14 StarFold verified microtile offset overflow")?;
        if end > source.planned.bytes {
            bail!("S14 StarFold verified microtile 越出 planned Range");
        }
        let hot_lease = process_starfold_verified_lease_cache()
            .acquire_planned_in_epoch(
                &self.cache_root,
                &source.planned,
                Some(source.span.key.expert_id),
                self.validation_epoch,
            )
            .map_err(anyhow::Error::new)
            .context("取得 S14 StarFold 进程级 proof/SHA/mmap 热 lease")?;
        let asset = hot_lease.asset().clone();
        source
            .planned
            .validate_resolved_position0_asset(&asset, Some(source.span.key.expert_id))?;
        let microtile = hot_lease
            .microtile(source.span.source_segment_offset, source.span.byte_len)
            .map_err(anyhow::Error::new)
            .context("切出 S14 StarFold proof-bound microtile")?;
        if hot_lease.identity().tensor != source.planned.tensor
            || hot_lease.identity().payload_sha256 != asset.sha256
            || hot_lease.identity().payload_bytes != source.planned.bytes
            || microtile.len() != source.span.byte_len as usize
        {
            bail!("S14 StarFold proof/SHA/mmap identity 漂移");
        }
        Ok(Arc::new(S14StarfoldVerifiedMicrotile::Single(
            S14StarfoldSingleProofLease {
                source: source.clone(),
                asset,
                hot_lease,
            },
        )))
    }

    /// 把同一 layer/expert/projection 的 MXFP4 weight 与 scale proof 固定拼成一个 payload。
    /// 两个输入都必须是 `Single`，从而禁止 packed-on-packed 造成 proof 身份递归或重复打包。
    pub fn pack_verified_mxfp4(
        &mut self,
        weight: Arc<S14StarfoldVerifiedMicrotile>,
        scale: Arc<S14StarfoldVerifiedMicrotile>,
    ) -> Result<Arc<S14StarfoldVerifiedMicrotile>> {
        let weight_single = weight
            .as_single()
            .context("S14 StarFold MXFP4 weight proof 必须是 Single")?;
        let scale_single = scale
            .as_single()
            .context("S14 StarFold MXFP4 scale proof 必须是 Single")?;
        validate_mxfp4_proof_pair(weight_single.source(), scale_single.source())?;
        let packed_l2_key = S14StarfoldPackedL2Key::from_sources(
            weight_single.source(),
            scale_single.source(),
            self.contract.microtile_bytes,
        )?;
        if let Some(cached) = self.packed_l2.lookup(&packed_l2_key) {
            return Ok(cached);
        }

        let weight_bytes = u64::try_from(weight_single.bytes().len())
            .context("S14 StarFold MXFP4 weight bytes 超出 u64")?;
        let scale_bytes = u64::try_from(scale_single.bytes().len())
            .context("S14 StarFold MXFP4 scale bytes 超出 u64")?;
        let total_bytes = weight_bytes
            .checked_add(scale_bytes)
            .context("S14 StarFold packed MXFP4 bytes 溢出")?;
        if total_bytes == 0 || total_bytes > u64::from(self.contract.microtile_bytes) {
            bail!(
                "S14 StarFold packed MXFP4 超出窗口: weight={weight_bytes}, scale={scale_bytes}, window={}",
                self.contract.microtile_bytes
            );
        }
        let packed_capacity =
            usize::try_from(total_bytes).context("S14 StarFold packed MXFP4 bytes 超出 usize")?;
        let mut packed = Vec::with_capacity(packed_capacity);
        packed.extend_from_slice(weight_single.bytes());
        packed.extend_from_slice(scale_single.bytes());
        if packed.len() != packed_capacity {
            bail!("S14 StarFold packed MXFP4 payload 长度漂移");
        }
        let bytes: Arc<[u8]> = Arc::from(packed.into_boxed_slice());
        let weight_identity = S14StarfoldPackedMxfp4SourceIdentity {
            source: weight_single.source().clone(),
            asset: weight_single.asset().clone(),
        };
        let scale_identity = S14StarfoldPackedMxfp4SourceIdentity {
            source: scale_single.source().clone(),
            asset: scale_single.asset().clone(),
        };
        let packed = Arc::new(S14StarfoldVerifiedMicrotile::PackedMxfp4(
            S14StarfoldPackedMxfp4ProofLease {
                bytes,
                weight: weight_identity,
                scale: scale_identity,
                layout: S14StarfoldPackedMxfp4Layout {
                    weight_offset: 0,
                    weight_bytes,
                    scale_offset: weight_bytes,
                    scale_bytes,
                    total_bytes,
                },
            },
        ));
        self.packed_l2.admit(packed_l2_key, packed)
    }

    pub fn reserve_verified_upload(
        &mut self,
        verified: Arc<S14StarfoldVerifiedMicrotile>,
        consumer_count: u32,
    ) -> Result<S14StarfoldUploadTicket> {
        let byte_len = verified.byte_len();
        if byte_len == 0 || byte_len > u64::from(self.contract.microtile_bytes) {
            bail!("S14 StarFold proof payload bytes 超出双窗口合同");
        }
        let (ticket, recording) = self
            .resource_owner
            .as_mut()
            .context("S14 StarFold Vulkan owner 已销毁")?
            .windows_mut()
            .reserve_upload(
                S14StarfoldResidentWindowKey::Microtile(verified.key()),
                byte_len,
                consumer_count,
            )
            .map_err(anyhow::Error::new)
            .context("预留 S14 StarFold microtile upload window")?;
        Ok(S14StarfoldUploadTicket {
            ticket,
            recording,
            verified,
        })
    }

    pub fn cancel_upload(&mut self, ticket: S14StarfoldUploadTicket) -> Result<()> {
        self.resource_owner
            .as_mut()
            .context("S14 StarFold Vulkan owner 已销毁")?
            .windows_mut()
            .cancel_upload(ticket.ticket)
            .map_err(anyhow::Error::new)
            .context("取消 S14 StarFold microtile upload")
    }

    /// command buffer 必须按 `ticket.recording()` 录制覆盖 `ticket.bytes()` 的 copy/barrier。
    /// 本调用异步提交 transfer，不做 host wait。
    ///
    /// # Safety
    ///
    /// `command_buffer` 必须已结束录制，且所有 Vulkan 资源必须来自同一 device。
    pub unsafe fn submit_upload(
        &mut self,
        ticket: S14StarfoldUploadTicket,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> Result<S14StarfoldReadyMicrotile> {
        let binding = unsafe {
            self.resource_owner
                .as_mut()
                .context("S14 StarFold Vulkan owner 已销毁")?
                .submit_upload(ticket.ticket, command_buffer, fence)
        }
        .context("提交 S14 StarFold microtile Vulkan upload")?;
        Ok(S14StarfoldReadyMicrotile {
            binding,
            verified: ticket.verified,
        })
    }

    /// 完整生产上传：verified mmap → 常驻 staging → Vulkan window → transfer timeline。
    /// 录制或 queue submit 失败时同时回滚 transfer slot 与窗口 reservation。
    pub fn upload_verified_microtile(
        &mut self,
        ticket: S14StarfoldUploadTicket,
    ) -> Result<S14StarfoldReadyMicrotile> {
        let recorded = self
            .transfer_executor
            .as_mut()
            .context("S14 StarFold transfer executor 已销毁")?
            .record_verified_upload(&ticket)
            .context("录制 S14 StarFold verified microtile upload")?;
        let submit = unsafe {
            self.resource_owner
                .as_mut()
                .context("S14 StarFold Vulkan owner 已销毁")?
                .submit_upload(ticket.ticket, recorded.command_buffer(), recorded.fence())
        };
        match submit {
            Ok(binding) => Ok(S14StarfoldReadyMicrotile {
                binding,
                verified: ticket.verified,
            }),
            Err(error) => {
                let abandon = unsafe {
                    self.transfer_executor
                        .as_mut()
                        .context("S14 StarFold transfer executor 已销毁")?
                        .abandon_unsubmitted(recorded)
                };
                let cancel = self
                    .resource_owner
                    .as_mut()
                    .context("S14 StarFold Vulkan owner 已销毁")?
                    .windows_mut()
                    .cancel_upload(ticket.ticket)
                    .map_err(anyhow::Error::new);
                match (abandon, cancel) {
                    (Ok(()), Ok(())) => Err(error),
                    (abandon, cancel) => Err(anyhow!(
                        "{error:#}; upload rollback: transfer={abandon:?} window={cancel:?}"
                    )),
                }
            }
        }
    }

    pub fn begin_transfer_block_epoch(&mut self, block_epoch: u64) -> Result<()> {
        self.transfer_executor
            .as_mut()
            .context("S14 StarFold transfer executor 已销毁")?
            .begin_block_epoch(block_epoch)
            .context("开启 S14 StarFold transfer block epoch")
    }

    /// 显式 epoch 热路径：verified lease → Prepared staging → Armed → queue submit。
    /// 任一 queue-submit 失败都归还 transfer slot 与 window reservation。
    pub fn upload_verified_microtile_in_epoch(
        &mut self,
        block_epoch: u64,
        ticket: S14StarfoldUploadTicket,
    ) -> Result<S14StarfoldReadyMicrotile> {
        let prepared = match self
            .transfer_executor
            .as_mut()
            .context("S14 StarFold transfer executor 已销毁")?
            .prepare_verified_upload(block_epoch, &ticket)
        {
            Ok(prepared) => prepared,
            Err(error) => {
                let cancel = self
                    .resource_owner
                    .as_mut()
                    .context("S14 StarFold Vulkan owner 已销毁")?
                    .windows_mut()
                    .cancel_upload(ticket.ticket)
                    .map_err(anyhow::Error::new);
                return Err(anyhow!(
                    "准备 S14 StarFold epoch microtile upload: {error:#}; window cancel={cancel:?}"
                ));
            }
        };
        let committed = match self
            .transfer_executor
            .as_mut()
            .context("S14 StarFold transfer executor 已销毁")?
            .commit_prepared_upload(&ticket, prepared)
        {
            Ok(committed) => committed,
            Err(error) => {
                let cancel = self
                    .resource_owner
                    .as_mut()
                    .context("S14 StarFold Vulkan owner 已销毁")?
                    .windows_mut()
                    .cancel_upload(ticket.ticket)
                    .map_err(anyhow::Error::new);
                return Err(anyhow!(
                    "提交 S14 StarFold Prepared upload owner: {error:#}; window cancel={cancel:?}"
                ));
            }
        };
        let recorded = committed.into_recorded();
        let submit = unsafe {
            self.resource_owner
                .as_mut()
                .context("S14 StarFold Vulkan owner 已销毁")?
                .submit_upload(ticket.ticket, recorded.command_buffer(), recorded.fence())
        };
        match submit {
            Ok(binding) => Ok(S14StarfoldReadyMicrotile {
                binding,
                verified: ticket.verified,
            }),
            Err(error) => {
                let abandon = unsafe {
                    self.transfer_executor
                        .as_mut()
                        .context("S14 StarFold transfer executor 已销毁")?
                        .abandon_unsubmitted(recorded)
                };
                let cancel = self
                    .resource_owner
                    .as_mut()
                    .context("S14 StarFold Vulkan owner 已销毁")?
                    .windows_mut()
                    .cancel_upload(ticket.ticket)
                    .map_err(anyhow::Error::new);
                match (abandon, cancel) {
                    (Ok(()), Ok(())) => Err(error),
                    (abandon, cancel) => Err(anyhow!(
                        "{error:#}; epoch upload rollback: transfer={abandon:?} window={cancel:?}"
                    )),
                }
            }
        }
    }

    pub fn drain_transfer_block_epoch(&mut self, block_epoch: u64) -> Result<()> {
        self.transfer_executor
            .as_mut()
            .context("S14 StarFold transfer executor 已销毁")?
            .drain_block_epoch(block_epoch)
            .context("drain S14 StarFold transfer block epoch")
    }

    pub fn reserve_compute(
        &mut self,
        ready: &S14StarfoldReadyMicrotile,
        consumer_id: u64,
    ) -> Result<S14StarfoldComputeMicrotile> {
        let (ticket, recording) = self
            .resource_owner
            .as_mut()
            .context("S14 StarFold Vulkan owner 已销毁")?
            .windows_mut()
            .reserve_compute(ready.binding, consumer_id)
            .map_err(anyhow::Error::new)
            .context("绑定 S14 StarFold compute microtile")?;
        Ok(S14StarfoldComputeMicrotile {
            ticket,
            recording,
            verified: Arc::clone(&ready.verified),
        })
    }

    /// 为同一条 MXFP4 command 原子绑定 weight 与 scale 两个驻留窗口。
    pub fn reserve_compute_pair(
        &mut self,
        ready: [&S14StarfoldReadyMicrotile; 2],
        consumer_id: u64,
    ) -> Result<S14StarfoldComputePairMicrotile> {
        let bindings = [ready[0].binding, ready[1].binding];
        let (ticket, recording) = self
            .resource_owner
            .as_mut()
            .context("S14 StarFold Vulkan owner 已销毁")?
            .windows_mut()
            .reserve_compute_pair(bindings, consumer_id)
            .map_err(anyhow::Error::new)
            .context("绑定 S14 StarFold weight/scale pair compute")?;
        Ok(S14StarfoldComputePairMicrotile {
            ticket,
            recording,
            verified: [
                Arc::clone(&ready[0].verified),
                Arc::clone(&ready[1].verified),
            ],
        })
    }

    pub fn cancel_compute_pair(&mut self, compute: S14StarfoldComputePairMicrotile) -> Result<()> {
        self.resource_owner
            .as_mut()
            .context("S14 StarFold Vulkan owner 已销毁")?
            .windows_mut()
            .cancel_compute_pair(compute.ticket)
            .map_err(anyhow::Error::new)
            .context("取消 S14 StarFold weight/scale pair compute")
    }

    pub fn cancel_compute(&mut self, compute: S14StarfoldComputeMicrotile) -> Result<()> {
        self.resource_owner
            .as_mut()
            .context("S14 StarFold Vulkan owner 已销毁")?
            .windows_mut()
            .cancel_compute(compute.ticket)
            .map_err(anyhow::Error::new)
            .context("取消 S14 StarFold compute microtile")
    }

    /// command buffer 必须按 `compute.recording()` 录制 dispatch/barrier。本方法只提交
    /// 一个驻留 consumer，不发布 token；最长可靠前缀仍由既有事务 owner 提交。
    ///
    /// # Safety
    ///
    /// `command_buffer` 必须已结束录制，且所有 Vulkan 资源必须来自同一 device。
    pub unsafe fn submit_compute(
        &mut self,
        compute: S14StarfoldComputeMicrotile,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> Result<S14StarfoldComputeSubmissionReceipt> {
        let receipt = unsafe {
            self.resource_owner
                .as_mut()
                .context("S14 StarFold Vulkan owner 已销毁")?
                .submit_compute(compute.ticket, command_buffer, fence)
        }
        .context("提交 S14 StarFold microtile Vulkan compute")?;
        Ok(S14StarfoldComputeSubmissionReceipt {
            receipt,
            verified: compute.verified,
        })
    }

    /// 提交一条同时消费 weight/scale 两窗口的 compute command；只 signal 一次
    /// compute timeline，verified payload/identity lease 保留至回执释放。
    ///
    /// # Safety
    /// command_buffer 必须按 `compute.recording()` 录制两个窗口的 acquire/release。
    pub unsafe fn submit_compute_pair(
        &mut self,
        compute: S14StarfoldComputePairMicrotile,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> Result<S14StarfoldComputePairSubmissionReceipt> {
        let receipts = unsafe {
            self.resource_owner
                .as_mut()
                .context("S14 StarFold Vulkan owner 已销毁")?
                .submit_compute_pair(compute.ticket, command_buffer, fence)
        }
        .context("提交 S14 StarFold weight/scale pair compute")?;
        Ok(S14StarfoldComputePairSubmissionReceipt {
            receipts,
            verified: compute.verified,
        })
    }

    pub fn destroy(mut self) -> Result<()> {
        if let Some(owner) = self.resource_owner.take() {
            owner.destroy()?;
        }
        if let Some(mut transfer) = self.transfer_executor.take() {
            transfer.try_destroy()?;
        }
        Ok(())
    }
}

impl S14StarfoldConstellationRuntimeHook for S14StarfoldRuntime {
    fn begin_constellation_epoch(&mut self, epoch: u64) -> Result<()> {
        self.begin_transfer_block_epoch(epoch)
    }

    fn upload_constellation_packet_in_epoch(
        &mut self,
        epoch: u64,
        packet: Arc<S14StarfoldConstellationPacket>,
    ) -> Result<S14StarfoldConstellationReadyPacket> {
        packet.validate()?;
        if packet.payload_bytes == 0
            || packet.payload_bytes > u64::from(self.contract.microtile_bytes)
        {
            bail!("S14 StarFold constellation packet bytes 超出 resident window 合同");
        }
        let (ticket, recording) = self
            .resource_owner
            .as_mut()
            .context("S14 StarFold Vulkan owner 已销毁")?
            .windows_mut()
            .reserve_upload(
                S14StarfoldResidentWindowKey::Constellation(packet.key()),
                packet.payload_bytes,
                1,
            )
            .map_err(anyhow::Error::new)
            .context("预留 S14 StarFold constellation upload window")?;
        if ticket.key() != S14StarfoldResidentWindowKey::Constellation(packet.key())
            || ticket.byte_len() != packet.payload_bytes
            || recording.byte_len != packet.payload_bytes
        {
            let cancel = self
                .resource_owner
                .as_mut()
                .context("S14 StarFold Vulkan owner 已销毁")?
                .windows_mut()
                .cancel_upload(ticket)
                .map_err(anyhow::Error::new);
            return Err(anyhow!(
                "S14 StarFold constellation reserve key/bytes 回执漂移; window cancel={cancel:?}"
            ));
        }

        let prepared = match self
            .transfer_executor
            .as_mut()
            .context("S14 StarFold transfer executor 已销毁")?
            .prepare_constellation_upload(epoch, ticket, recording, &packet)
        {
            Ok(prepared) => prepared,
            Err(error) => {
                let cancel = self
                    .resource_owner
                    .as_mut()
                    .context("S14 StarFold Vulkan owner 已销毁")?
                    .windows_mut()
                    .cancel_upload(ticket)
                    .map_err(anyhow::Error::new);
                return Err(anyhow!(
                    "准备 S14 StarFold constellation upload: {error:#}; window cancel={cancel:?}"
                ));
            }
        };
        let committed = match self
            .transfer_executor
            .as_mut()
            .context("S14 StarFold transfer executor 已销毁")?
            .commit_prepared_constellation_upload(ticket, recording, &packet, prepared)
        {
            Ok(committed) => committed,
            Err(error) => {
                let cancel = self
                    .resource_owner
                    .as_mut()
                    .context("S14 StarFold Vulkan owner 已销毁")?
                    .windows_mut()
                    .cancel_upload(ticket)
                    .map_err(anyhow::Error::new);
                return Err(anyhow!(
                    "提交 S14 StarFold constellation Prepared owner: {error:#}; window cancel={cancel:?}"
                ));
            }
        };
        let (recorded, packet) = committed.into_parts();
        let submit = unsafe {
            self.resource_owner
                .as_mut()
                .context("S14 StarFold Vulkan owner 已销毁")?
                .submit_upload(ticket, recorded.command_buffer(), recorded.fence())
        };
        match submit {
            Ok(binding) => Ok(S14StarfoldConstellationReadyPacket::from_verified_parts(
                binding, packet,
            )),
            Err(error) => {
                let abandon = unsafe {
                    self.transfer_executor
                        .as_mut()
                        .context("S14 StarFold transfer executor 已销毁")?
                        .abandon_unsubmitted(recorded)
                };
                let cancel = self
                    .resource_owner
                    .as_mut()
                    .context("S14 StarFold Vulkan owner 已销毁")?
                    .windows_mut()
                    .cancel_upload(ticket)
                    .map_err(anyhow::Error::new);
                match (abandon, cancel) {
                    (Ok(()), Ok(())) => Err(error),
                    (abandon, cancel) => Err(anyhow!(
                        "{error:#}; constellation upload rollback: transfer={abandon:?} window={cancel:?}"
                    )),
                }
            }
        }
    }

    fn constellation_windows_mut(
        &mut self,
    ) -> Result<&mut S14StarfoldVulkanWindows<S14StarfoldResidentWindowKey>> {
        self.vulkan_windows_mut()
    }

    fn drain_constellation_epoch(&mut self, epoch: u64) -> Result<()> {
        self.drain_transfer_block_epoch(epoch)
    }
}
