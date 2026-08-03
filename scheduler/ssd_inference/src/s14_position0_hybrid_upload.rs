//! FullDepth43 position0 Hybrid 权重上传链。
//!
//! 静态 embedding/attention/router/shared/final-small 参数只在启动阶段逐资产上传一次；
//! routed expert 只占两个逐层交替 bank；BF16 vocab head 只占两个 4096-row chunk。
//! 所有源字节必须先取得 `VerifiedMappedAssetStore` lease，本模块不提供直接文件读取路径，
//! 也不会建立与 11.24 GB 逻辑资产等大的 host staging。

use crate::{
    s14_dynamic_routed_packing::S14DynamicRoutedUploadPlan,
    s14_dynamic_routed_page_plan::{DynamicRoutedPagePlan, DYNAMIC_ROUTED_RANGE_COUNT},
    s14_position0_hybrid_weight_arena::{
        S14Position0HybridArenaLayout, S14Position0HybridWeightArena,
    },
    s14_position0_mapped_assets::{VerifiedMappedAsset, VerifiedMappedAssetStore},
    s14_position0_paged_weight_arena::{
        S14Position0PagedWeightArena, S14Position0StaticAssetBinding,
    },
    s14_position0_weight_plan::{
        S14Position0AssetPlacement, S14Position0HybridWeightPlan, S14_POSITION0_ROLLING_BANKS,
    },
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{Position0Asset, Position0WholeTokenManifest, FULL_DEPTH_LAYERS};
use std::sync::Arc;

const MIN_RUNTIME_VRAM_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

/// 目标 buffer 必须支持 transfer/graphics queue family 并发访问，或者两者属于同一 family。
/// Hybrid arena 可独立实现该接口，上传器不依赖或修改现有 full-upload arena。
pub trait S14Position0HybridUploadTarget {
    fn static_asset_destination(
        &self,
        placement: &S14Position0AssetPlacement,
    ) -> Result<S14Position0UploadDestination<'_>>;
    fn routed_bank(&self, bank: usize) -> Result<&GpuBuffer>;
    fn head_chunk_bank(&self, bank: usize) -> Result<&GpuBuffer>;

    /// verified store 已完成 proof/SHA/mmap 且本次同步 transfer 已 wait 后才会调用。
    /// 非 paged target 无需维护换页 identity。
    fn publish_static_layer_ready(
        &self,
        _layer: u8,
        _resident_hit: bool,
        _bank: Option<usize>,
        _assets: usize,
        _uploaded_bytes: u64,
    ) -> Result<()> {
        Ok(())
    }
}

pub struct S14Position0UploadDestination<'a> {
    pub buffer: &'a GpuBuffer,
    pub offset: u64,
    pub resident_once: bool,
    pub layer: Option<u8>,
    pub bank: Option<usize>,
}

impl S14Position0HybridUploadTarget for S14Position0HybridWeightArena {
    fn static_asset_destination(
        &self,
        placement: &S14Position0AssetPlacement,
    ) -> Result<S14Position0UploadDestination<'_>> {
        if let Some(layer) = parse_layer_tensor(&placement.tensor) {
            let layout = self
                .layout()
                .static_layers
                .iter()
                .find(|entry| entry.layer == layer)
                .ok_or_else(|| anyhow!("hybrid static target missing L{layer}"))?;
            let physical = layout
                .assets
                .iter()
                .find(|entry| {
                    entry.tensor == placement.tensor
                        && entry.source_offset == placement.offset
                        && entry.bytes == placement.bytes
                })
                .ok_or_else(|| anyhow!("hybrid static physical placement drift"))?;
            Ok(S14Position0UploadDestination {
                buffer: self.static_layer(layer)?,
                offset: physical.local_offset,
                resident_once: true,
                layer: Some(layer),
                bank: None,
            })
        } else {
            let physical = self
                .layout()
                .resident_small
                .assets
                .iter()
                .find(|entry| {
                    entry.tensor == placement.tensor
                        && entry.source_offset == placement.offset
                        && entry.bytes == placement.bytes
                })
                .ok_or_else(|| anyhow!("hybrid resident-small physical placement drift"))?;
            Ok(S14Position0UploadDestination {
                buffer: self.resident_small(),
                offset: physical.local_offset,
                resident_once: true,
                layer: None,
                bank: None,
            })
        }
    }

    fn routed_bank(&self, bank: usize) -> Result<&GpuBuffer> {
        self.routed(bank)
    }

    fn head_chunk_bank(&self, bank: usize) -> Result<&GpuBuffer> {
        self.head_chunk(bank)
    }
}

impl S14Position0HybridUploadTarget for S14Position0PagedWeightArena {
    fn static_asset_destination(
        &self,
        placement: &S14Position0AssetPlacement,
    ) -> Result<S14Position0UploadDestination<'_>> {
        let S14Position0StaticAssetBinding {
            buffer,
            destination_offset,
            resident_once,
            layer,
            bank,
        } = self.static_asset(&placement.tensor)?;
        Ok(S14Position0UploadDestination {
            buffer,
            offset: destination_offset,
            resident_once,
            layer,
            bank,
        })
    }

    fn routed_bank(&self, bank: usize) -> Result<&GpuBuffer> {
        self.routed(bank)
    }

    fn head_chunk_bank(&self, bank: usize) -> Result<&GpuBuffer> {
        self.head_chunk(bank)
    }

    fn publish_static_layer_ready(
        &self,
        layer: u8,
        resident_hit: bool,
        bank: Option<usize>,
        assets: usize,
        uploaded_bytes: u64,
    ) -> Result<()> {
        S14Position0PagedWeightArena::publish_static_layer_ready(
            self,
            layer,
            resident_hit,
            bank,
            assets,
            uploaded_bytes,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0HybridStagingPlan {
    pub static_scratch_bytes: u64,
    pub routed_bank_bytes: u64,
    pub head_chunk_bytes: u64,
    pub allocated_staging_bytes: u64,
}

impl S14Position0HybridStagingPlan {
    pub fn build(plan: &S14Position0HybridWeightPlan) -> Result<Self> {
        let static_scratch_bytes =
            S14Position0HybridArenaLayout::build(plan)?.largest_requested_allocation_bytes();
        if static_scratch_bytes == 0 || plan.routed_bank_bytes == 0 || plan.head_chunk_bytes == 0 {
            bail!("position0 hybrid staging size 不能为零");
        }
        let static_staging_bytes = static_scratch_bytes
            .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
            .ok_or_else(|| anyhow!("hybrid static staging bytes overflow"))?;
        let routed_staging_bytes = plan
            .routed_bank_bytes
            .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
            .ok_or_else(|| anyhow!("hybrid routed staging bytes overflow"))?;
        let head_staging_bytes = plan
            .head_chunk_bytes
            .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
            .ok_or_else(|| anyhow!("hybrid head staging bytes overflow"))?;
        let allocated_staging_bytes = static_staging_bytes
            .checked_add(routed_staging_bytes)
            .and_then(|bytes| bytes.checked_add(head_staging_bytes))
            .ok_or_else(|| anyhow!("hybrid staging bytes overflow"))?;
        if allocated_staging_bytes >= plan.logical_payload_bytes {
            bail!(
                "hybrid staging 不得退化成全量 staging: staging={} logical={}",
                allocated_staging_bytes,
                plan.logical_payload_bytes
            );
        }
        Ok(Self {
            static_scratch_bytes,
            routed_bank_bytes: plan.routed_bank_bytes,
            head_chunk_bytes: plan.head_chunk_bytes,
            allocated_staging_bytes,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0StaticUploadReceipt {
    pub assets_uploaded_this_call: usize,
    pub assets_deferred_to_streaming: usize,
    pub bytes_uploaded_this_call: u64,
    pub total_assets_uploaded: usize,
    pub complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0RoutedUploadReceipt {
    pub layer: u8,
    pub bank: usize,
    pub assets: usize,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0StaticLayerUploadReceipt {
    pub layer: u8,
    pub resident_hit: bool,
    pub bank: Option<usize>,
    pub assets: usize,
    pub bytes: u64,
}

/// 已在外部 transfer command 中冻结、等待 timeline 安全复用 staging 后填充的
/// 完整层权重副本。static resident hit 不产生静态 copy，但 routed 始终随层上传。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0LayerCopyReceipt {
    pub layer: u8,
    pub bank: usize,
    pub static_resident_hit: bool,
    pub static_assets: usize,
    pub static_bytes: u64,
    pub routed_assets: usize,
    pub routed_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0HeadChunkReceipt {
    pub chunk: u64,
    pub bank: usize,
    pub first_row: u64,
    pub rows: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct S14Position0HybridUploadStats {
    pub static_assets: u64,
    pub static_bytes: u64,
    pub streamed_static_layers: u64,
    pub streamed_static_bytes: u64,
    pub routed_layers: u64,
    pub routed_bytes: u64,
    pub head_chunks: u64,
    pub head_bytes: u64,
    pub transfer_submits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HybridUploadProgress {
    static_complete: bool,
    next_runtime_static_layer: usize,
    next_routed_layer: usize,
    pending_layer: Option<S14Position0LayerCopyReceipt>,
    next_head_chunk: u64,
    pending_head_chunk: Option<S14Position0HeadChunkReceipt>,
}

impl HybridUploadProgress {
    fn new() -> Self {
        Self {
            static_complete: false,
            next_runtime_static_layer: 0,
            next_routed_layer: 0,
            pending_layer: None,
            next_head_chunk: 0,
            pending_head_chunk: None,
        }
    }

    fn static_complete(&self, plan: &S14Position0HybridWeightPlan) -> bool {
        let _ = plan;
        self.static_complete
    }

    fn all_complete(&self, plan: &S14Position0HybridWeightPlan) -> bool {
        self.static_complete(plan)
            && self.next_runtime_static_layer == plan.routed_layers.len()
            && self.next_routed_layer == plan.routed_layers.len()
            && self.pending_layer.is_none()
            && self.next_head_chunk == plan.head_chunk_count
            && self.pending_head_chunk.is_none()
    }

    fn at_token_start(&self) -> bool {
        self.next_runtime_static_layer == 0
            && self.next_routed_layer == 0
            && self.pending_layer.is_none()
            && self.next_head_chunk == 0
            && self.pending_head_chunk.is_none()
    }

    fn prepare_persistent_token(
        &mut self,
        plan: &S14Position0HybridWeightPlan,
        first_token: bool,
    ) -> Result<()> {
        if first_token || self.at_token_start() {
            if !self.at_token_start() {
                bail!("persistent uploader token起始游标不是初始化态");
            }
        } else if self.all_complete(plan) {
            self.reset_token();
        } else {
            bail!("persistent uploader 上一token未完整闭合且未drain回滚，禁止复用");
        }
        Ok(())
    }

    fn prepare_next_causal_block_head(
        &mut self,
        plan: &S14Position0HybridWeightPlan,
    ) -> Result<()> {
        if !self.static_complete
            || self.next_runtime_static_layer != plan.routed_layers.len()
            || self.next_routed_layer != 0
            || self.pending_layer.is_some()
            || self.next_head_chunk != plan.head_chunk_count
            || self.pending_head_chunk.is_some()
        {
            bail!("causal-block static/head stream上一块未完整闭合，禁止复用");
        }
        self.next_runtime_static_layer = 0;
        self.next_head_chunk = 0;
        Ok(())
    }

    fn reset_token(&mut self) {
        debug_assert!(self.static_complete);
        self.next_runtime_static_layer = 0;
        self.next_routed_layer = 0;
        self.pending_layer = None;
        self.next_head_chunk = 0;
        self.pending_head_chunk = None;
    }

    fn validate_next_head_chunk(&self, receipt: S14Position0HeadChunkReceipt) -> Result<()> {
        if self.pending_layer.is_some() || self.pending_head_chunk.is_some() {
            bail!("position0 async uploader 已有 pending layer/head");
        }
        if receipt.chunk != self.next_head_chunk
            || receipt.bank != receipt.chunk as usize % S14_POSITION0_ROLLING_BANKS
        {
            bail!(
                "position0 async head chunk 顺序/bank 漂移: expected={} actual={} bank={}",
                self.next_head_chunk,
                receipt.chunk,
                receipt.bank
            );
        }
        Ok(())
    }

    fn record_head_copy(&mut self, receipt: S14Position0HeadChunkReceipt) -> Result<()> {
        self.validate_next_head_chunk(receipt)?;
        self.pending_head_chunk = Some(receipt);
        Ok(())
    }

    fn finish_recorded_head_stage(
        &mut self,
        receipt: S14Position0HeadChunkReceipt,
        timeline_bank: usize,
    ) -> Result<()> {
        if self.pending_head_chunk != Some(receipt)
            || timeline_bank != receipt.bank
            || receipt.chunk != self.next_head_chunk
        {
            bail!(
                "position0 async head staged receipt 漂移: expected={:?} actual={receipt:?} timeline_bank={timeline_bank}",
                self.pending_head_chunk
            );
        }
        self.pending_head_chunk = None;
        self.next_head_chunk = self
            .next_head_chunk
            .checked_add(1)
            .ok_or_else(|| anyhow!("position0 async head chunk cursor overflow"))?;
        Ok(())
    }

    fn finish_synchronous_head(&mut self, receipt: S14Position0HeadChunkReceipt) -> Result<()> {
        self.validate_next_head_chunk(receipt)?;
        self.next_head_chunk = self
            .next_head_chunk
            .checked_add(1)
            .ok_or_else(|| anyhow!("position0 synchronous head chunk cursor overflow"))?;
        Ok(())
    }
}

/// 有限 staging 的同步 transfer 实现。每次调用返回前 fence 已完成，所以 scratch
/// 可以安全复用；后续可在不改变 payload/布局合同的前提下替换为 timeline 异步提交。
pub struct S14Position0HybridUploader {
    static_staging: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
    routed_staging: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
    head_staging: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
    staging_plan: S14Position0HybridStagingPlan,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    progress: HybridUploadProgress,
    stats: S14Position0HybridUploadStats,
}

impl S14Position0HybridUploader {
    pub fn new(ctx: &VulkanContext, plan: &S14Position0HybridWeightPlan) -> Result<Self> {
        let staging_plan = S14Position0HybridStagingPlan::build(plan)?;
        let required_vram = plan
            .device_weight_bytes
            .checked_add(MIN_RUNTIME_VRAM_RESERVE_BYTES)
            .ok_or_else(|| anyhow!("hybrid VRAM requirement overflow"))?;
        if required_vram > ctx.vram_size() {
            bail!(
                "hybrid device weights exceed VRAM budget: weights={} reserve={} vram={}",
                plan.device_weight_bytes,
                MIN_RUNTIME_VRAM_RESERVE_BYTES,
                ctx.vram_size()
            );
        }

        let mut staging = Vec::with_capacity(6);
        for bytes in [
            staging_plan.static_scratch_bytes,
            staging_plan.static_scratch_bytes,
            staging_plan.routed_bank_bytes,
            staging_plan.routed_bank_bytes,
            staging_plan.head_chunk_bytes,
            staging_plan.head_chunk_bytes,
        ] {
            match GpuBuffer::new_staging(ctx, bytes) {
                Ok(buffer) => staging.push(buffer),
                Err(error) => {
                    for buffer in staging {
                        buffer.destroy(ctx);
                    }
                    return Err(error).context("allocate position0 hybrid staging");
                }
            }
        }
        let mut iter = staging.into_iter();
        let static_staging = [
            iter.next().expect("static staging0"),
            iter.next().expect("static staging1"),
        ];
        let routed_staging = [
            iter.next().expect("routed staging0"),
            iter.next().expect("routed staging1"),
        ];
        let head_staging = [
            iter.next().expect("head staging0"),
            iter.next().expect("head staging1"),
        ];

        let command_pool = match unsafe {
            ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.qf_transfer)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        } {
            Ok(pool) => pool,
            Err(error) => {
                destroy_staging(ctx, static_staging, routed_staging, head_staging);
                return Err(error).context("create hybrid transfer command pool");
            }
        };
        let command = match unsafe {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        } {
            Ok(commands) => commands[0],
            Err(error) => {
                unsafe { ctx.device.destroy_command_pool(command_pool, None) };
                destroy_staging(ctx, static_staging, routed_staging, head_staging);
                return Err(error).context("allocate hybrid transfer command");
            }
        };
        let fence = match unsafe {
            ctx.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(error) => {
                unsafe { ctx.device.destroy_command_pool(command_pool, None) };
                destroy_staging(ctx, static_staging, routed_staging, head_staging);
                return Err(error).context("create hybrid transfer fence");
            }
        };
        Ok(Self {
            static_staging,
            routed_staging,
            head_staging,
            staging_plan,
            command_pool,
            command,
            fence,
            progress: HybridUploadProgress::new(),
            stats: S14Position0HybridUploadStats::default(),
        })
    }

    pub fn staging_plan(&self) -> S14Position0HybridStagingPlan {
        self.staging_plan
    }

    pub fn stats(&self) -> S14Position0HybridUploadStats {
        self.stats
    }

    pub fn is_complete(&self, plan: &S14Position0HybridWeightPlan) -> bool {
        self.progress.all_complete(plan)
    }

    /// FullDepth43 已完成 static/routed 游标、尚未开始 head 扫描时才成立。
    /// production terminal 用它阻止提前扫描或复用半完成 token 的 uploader。
    pub fn ready_for_paged_head_stream(&self, plan: &S14Position0HybridWeightPlan) -> bool {
        self.progress.static_complete(plan)
            && self.progress.next_runtime_static_layer == plan.routed_layers.len()
            && self.progress.next_routed_layer == plan.routed_layers.len()
            && self.progress.pending_layer.is_none()
            && self.progress.next_head_chunk == 0
            && self.progress.pending_head_chunk.is_none()
    }

    /// Causal-block 的43层 static/routed 由 paged HC/QKV + union materializer 记录，
    /// 不推进旧单 token uploader 的 layer 游标。切换后的 uploader 只作为
    /// verified head staging owner，因此只验收 resident-small 与全新 head 流状态。
    pub fn ready_for_causal_block_head_stream(&self) -> bool {
        self.progress.static_complete
            && self.progress.pending_layer.is_none()
            && self.progress.next_head_chunk == 0
            && self.progress.pending_head_chunk.is_none()
    }

    /// 在一个完整 causal block 的43层 static 与32个head chunk闭合后重开下一块。
    /// causal-block provider 会推进 static 层游标，在线 routed 页则由 union
    /// materializer 直接提供、不会推进旧 uploader 的 routed 游标，因此这里要求
    /// `static=43/routed=0/head=32` 后只重置 static/head 控制游标，不重新分配资源。
    pub fn begin_next_causal_block_head_stream(
        &mut self,
        plan: &S14Position0HybridWeightPlan,
    ) -> Result<()> {
        self.progress.prepare_next_causal_block_head(plan)
    }

    /// resident-small 与可选 resident static 页已经由同源 runtime 完成 verified
    /// proof/SHA/mmap/upload。跨到 causal-block 后只能复用，不能再次执行 one-shot 上传。
    pub fn resident_static_uploaded(&self) -> bool {
        self.progress.static_complete
    }

    /// 为跨 token 常驻 uploader 打开下一次事务。首个 token 只校验初始化态；
    /// 后续 token 必须确认上一 token 已完整走过43层和32个head chunk，随后只
    /// 重置游标/pending，不重新分配 staging、command pool、command buffer或fence。
    pub fn begin_persistent_token(
        &mut self,
        plan: &S14Position0HybridWeightPlan,
        first_token: bool,
    ) -> Result<()> {
        self.require_static_complete(plan)?;
        self.progress.prepare_persistent_token(plan, first_token)
    }

    /// 只允许在whole-token timeline已完成或drain后调用。失败token不会保留
    /// 半层/半head游标；静态常驻权重和所有已分配Vulkan资源继续有效。
    pub fn abort_persistent_token_after_drain(&mut self) {
        self.reset_token_progress();
    }

    fn reset_token_progress(&mut self) {
        self.progress.reset_token();
    }

    /// 按目标物理 buffer 分组：每个常驻层只做一次并行 SHA batch 与一次 transfer
    /// submit；分页层仅分类为 deferred，不在启动阶段错误上传。
    pub fn upload_static_once(
        &mut self,
        ctx: &VulkanContext,
        manifest: &Position0WholeTokenManifest,
        plan: &S14Position0HybridWeightPlan,
        store: &mut VerifiedMappedAssetStore,
        target: &dyn S14Position0HybridUploadTarget,
    ) -> Result<S14Position0StaticUploadReceipt> {
        plan.validate(manifest)?;
        validate_target_sizes(plan, target)?;
        if self.progress.static_complete(plan) {
            bail!("position0 hybrid static weights 已上传，禁止重复上传");
        }
        let mut bytes_this_call = 0u64;
        let mut uploaded_this_call = 0usize;
        let mut deferred_this_call = 0usize;
        let mut groups = Vec::<(vk::Buffer, Vec<(&S14Position0AssetPlacement, u64)>)>::new();
        for placement in &plan.resident.assets {
            let destination = target.static_asset_destination(placement)?;
            if !destination.resident_once {
                deferred_this_call += 1;
                continue;
            }
            let handle = destination.buffer.handle();
            if let Some((_, entries)) = groups.iter_mut().find(|(buffer, _)| *buffer == handle) {
                entries.push((placement, destination.offset));
            } else {
                groups.push((handle, vec![(placement, destination.offset)]));
            }
        }

        for (destination, entries) in groups {
            let assets = entries
                .iter()
                .map(|(placement, _)| resolve_manifest_asset(manifest, placement).cloned())
                .collect::<Result<Vec<_>>>()?;
            let mapped = store.map_verified_batch(&assets)?;
            let mut copies = Vec::with_capacity(entries.len());
            for ((placement, offset), payload) in entries.iter().zip(&mapped) {
                validate_mapped_identity(payload, placement)?;
                let end = checked_end(*offset, placement.bytes, "resident placement")?;
                if end > self.staging_plan.static_scratch_bytes {
                    bail!(
                        "resident group exceeds static scratch: {}",
                        placement.tensor
                    );
                }
                unsafe {
                    self.static_staging[0].write_at(usize::try_from(*offset)?, payload.bytes())
                };
                copies.push(
                    vk::BufferCopy::default()
                        .src_offset(*offset)
                        .dst_offset(*offset)
                        .size(placement.bytes),
                );
                uploaded_this_call += 1;
                bytes_this_call = checked_add(bytes_this_call, placement.bytes, "static bytes")?;
            }
            self.copy_and_wait(ctx, self.static_staging[0].handle(), destination, &copies)?;
        }
        self.progress.static_complete = true;
        self.stats.static_assets = checked_add(
            self.stats.static_assets,
            uploaded_this_call as u64,
            "static assets",
        )?;
        self.stats.static_bytes =
            checked_add(self.stats.static_bytes, bytes_this_call, "static bytes")?;
        Ok(S14Position0StaticUploadReceipt {
            assets_uploaded_this_call: uploaded_this_call,
            assets_deferred_to_streaming: deferred_this_call,
            bytes_uploaded_this_call: bytes_this_call,
            total_assets_uploaded: uploaded_this_call,
            complete: uploaded_this_call + deferred_this_call == plan.resident.assets.len(),
        })
    }

    /// 为当前层准备 non-expert/router/shared 静态权重。常驻层返回零拷贝命中；
    /// 分页层把该层全部资产合并为一次 transfer submit。调用顺序必须为 L0→L42。
    pub fn prepare_next_static_layer(
        &mut self,
        ctx: &VulkanContext,
        manifest: &Position0WholeTokenManifest,
        plan: &S14Position0HybridWeightPlan,
        store: &mut VerifiedMappedAssetStore,
        target: &dyn S14Position0HybridUploadTarget,
    ) -> Result<S14Position0StaticLayerUploadReceipt> {
        plan.validate(manifest)?;
        validate_target_sizes(plan, target)?;
        self.require_static_complete(plan)?;
        if self.progress.pending_layer.is_some() {
            bail!("position0 async layer 尚未 stage，禁止同步 static upload");
        }
        let index = self.progress.next_runtime_static_layer;
        let &layer = FULL_DEPTH_LAYERS
            .get(index)
            .ok_or_else(|| anyhow!("position0 static layers 已全部准备"))?;
        let prefix = format!("layers.{layer}.");
        let placements = plan
            .resident
            .assets
            .iter()
            .filter(|placement| placement.tensor.starts_with(&prefix))
            .collect::<Vec<_>>();
        if placements.is_empty() {
            bail!("position0 static plan missing L{layer}");
        }

        let first_destination = target.static_asset_destination(placements[0])?;
        let resident_hit = first_destination.resident_once;
        let bank = first_destination.bank;
        let destination_buffer = first_destination.buffer;
        let mut copies = Vec::with_capacity(placements.len());
        let mut bytes = 0u64;
        for placement in &placements {
            let destination = target.static_asset_destination(placement)?;
            if destination.resident_once != resident_hit
                || destination.layer != Some(layer)
                || destination.bank != bank
                || destination.buffer.handle() != destination_buffer.handle()
            {
                bail!("position0 static L{layer} physical destination drift");
            }
            let end = checked_end(
                destination.offset,
                placement.bytes,
                "static layer placement",
            )?;
            if end > self.staging_plan.static_scratch_bytes {
                bail!("position0 static L{layer} exceeds scratch capacity");
            }
            if !resident_hit {
                let asset = resolve_manifest_asset(manifest, placement)?;
                let mapped = map_verified_placement(store, asset, placement)?;
                unsafe {
                    self.static_staging[0]
                        .write_at(usize::try_from(destination.offset)?, mapped.bytes())
                };
                copies.push(
                    vk::BufferCopy::default()
                        .src_offset(destination.offset)
                        .dst_offset(destination.offset)
                        .size(placement.bytes),
                );
                bytes = checked_add(bytes, placement.bytes, "streamed static bytes")?;
            }
        }
        if resident_hit {
            if bank.is_some() || !copies.is_empty() || bytes != 0 {
                bail!("position0 resident static receipt drift");
            }
        } else {
            if bank.is_none() || copies.is_empty() {
                bail!("position0 streamed static receipt drift");
            }
            self.copy_and_wait(
                ctx,
                self.static_staging[0].handle(),
                destination_buffer.handle(),
                &copies,
            )?;
            self.stats.streamed_static_layers = checked_add(
                self.stats.streamed_static_layers,
                1,
                "streamed static layers",
            )?;
            self.stats.streamed_static_bytes = checked_add(
                self.stats.streamed_static_bytes,
                bytes,
                "streamed static bytes",
            )?;
        }
        target.publish_static_layer_ready(layer, resident_hit, bank, placements.len(), bytes)?;
        self.progress.next_runtime_static_layer += 1;
        Ok(S14Position0StaticLayerUploadReceipt {
            layer,
            resident_hit,
            bank,
            assets: placements.len(),
            bytes,
        })
    }

    pub fn upload_next_routed_layer(
        &mut self,
        ctx: &VulkanContext,
        manifest: &Position0WholeTokenManifest,
        plan: &S14Position0HybridWeightPlan,
        store: &mut VerifiedMappedAssetStore,
        target: &dyn S14Position0HybridUploadTarget,
    ) -> Result<S14Position0RoutedUploadReceipt> {
        plan.validate(manifest)?;
        validate_target_sizes(plan, target)?;
        self.require_static_complete(plan)?;
        if self.progress.pending_layer.is_some() {
            bail!("position0 async layer 尚未 stage，禁止同步 routed upload");
        }
        let index = self.progress.next_routed_layer;
        let layer_plan = plan
            .routed_layers
            .get(index)
            .ok_or_else(|| anyhow!("position0 routed layers 已全部上传"))?;
        if layer_plan.layer != FULL_DEPTH_LAYERS[index] {
            bail!("position0 routed upload layer order drift");
        }
        let manifest_layer = &manifest.layers[index];
        let mapped = store.map_verified_batch(&manifest_layer.assets.routed)?;
        stage_packed_verified_assets(
            &self.routed_staging[layer_plan.bank],
            self.staging_plan.routed_bank_bytes,
            &layer_plan.assets,
            &mapped,
        )?;
        let copies = placement_copies(&layer_plan.assets, self.staging_plan.routed_bank_bytes)?;
        self.copy_and_wait(
            ctx,
            self.routed_staging[layer_plan.bank].handle(),
            target.routed_bank(layer_plan.bank)?.handle(),
            &copies,
        )?;
        self.progress.next_routed_layer += 1;
        self.stats.routed_layers = checked_add(self.stats.routed_layers, 1, "routed layers")?;
        self.stats.routed_bytes = checked_add(
            self.stats.routed_bytes,
            layer_plan.logical_bytes,
            "routed bytes",
        )?;
        Ok(S14Position0RoutedUploadReceipt {
            layer: layer_plan.layer,
            bank: layer_plan.bank,
            assets: layer_plan.assets.len(),
            bytes: layer_plan.logical_bytes,
        })
    }

    /// Router probe 后的 production bridge：把 proof-checked online top-6
    /// payload 按既有slot metadata布局写入当前 rolling bank。这里绝不从
    /// position0 manifest 选择专家身份；manifest 只提供每个slot的目标 offset、
    /// dtype、shape 与字节合同。
    pub fn upload_next_dynamic_routed_layer(
        &mut self,
        ctx: &VulkanContext,
        manifest: &Position0WholeTokenManifest,
        plan: &S14Position0HybridWeightPlan,
        target: &dyn S14Position0HybridUploadTarget,
        dynamic_plan: &DynamicRoutedPagePlan,
        dynamic_upload: &S14DynamicRoutedUploadPlan,
    ) -> Result<S14Position0RoutedUploadReceipt> {
        plan.validate(manifest)?;
        validate_target_sizes(plan, target)?;
        self.require_static_complete(plan)?;
        if self.progress.pending_layer.is_some() || self.progress.pending_head_chunk.is_some() {
            bail!("dynamic routed upload 禁止跨过 pending layer/head");
        }
        let index = self.progress.next_routed_layer;
        let layer_plan = plan
            .routed_layers
            .get(index)
            .ok_or_else(|| anyhow!("dynamic routed layers 已全部上传"))?;
        if dynamic_plan.layer != layer_plan.layer
            || self.progress.next_runtime_static_layer != index + 1
            || layer_plan.layer != FULL_DEPTH_LAYERS[index]
        {
            bail!("dynamic routed upload layer/static cursor 漂移");
        }
        if dynamic_upload.layout.layer != dynamic_plan.layer
            || dynamic_upload.layout.position != dynamic_plan.position
            || dynamic_upload.layout.slots.map(|slot| slot.expert_id) != dynamic_plan.expert_ids
            || dynamic_upload
                .layout
                .slots
                .map(|slot| slot.route_weight_bits)
                != dynamic_plan.route_weights.map(f32::to_bits)
            || dynamic_upload.layout.arena_logical_bytes != layer_plan.logical_bytes
        {
            bail!("dynamic routed canonical upload plan identity/layout 漂移");
        }
        stage_dynamic_upload_plan(
            &self.routed_staging[layer_plan.bank],
            self.staging_plan.routed_bank_bytes,
            dynamic_upload,
        )?;
        let copies = [vk::BufferCopy::default().size(dynamic_upload.layout.arena_logical_bytes)];
        self.copy_and_wait(
            ctx,
            self.routed_staging[layer_plan.bank].handle(),
            target.routed_bank(layer_plan.bank)?.handle(),
            &copies,
        )?;
        self.progress.next_routed_layer += 1;
        self.stats.routed_layers =
            checked_add(self.stats.routed_layers, 1, "dynamic routed layers")?;
        self.stats.routed_bytes = checked_add(
            self.stats.routed_bytes,
            layer_plan.logical_bytes,
            "dynamic routed bytes",
        )?;
        Ok(S14Position0RoutedUploadReceipt {
            layer: layer_plan.layer,
            bank: layer_plan.bank,
            assets: layer_plan.assets.len(),
            bytes: layer_plan.logical_bytes,
        })
    }

    /// 把 online top-6 routed arena 写入当前 host staging，并把唯一一次
    /// staging→device copy 录入调用方提供的 transfer command。这里不 submit、
    /// 不 wait；调用方必须把 command 接入 whole-token timeline，并让 matching
    /// continuation compute 等待该 transfer ticket。
    ///
    /// Router probe 已经完成时，graphics queue 也已经越过上一层 continuation，
    /// 因而当前双 bank 的 host staging/device 页都不再被 L(n-2) 使用。这个边界
    /// 允许删除旧 `copy_and_wait` 的逐层 fence，同时保持相同 canonical packing。
    ///
    /// # Safety
    ///
    /// `command` 必须尚未 begin；staging、target routed bank 与 command 引用的
    /// payload 必须存活到外部 timeline 的 transfer/compute 完成或 drain。
    pub unsafe fn record_next_dynamic_routed_layer(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        manifest: &Position0WholeTokenManifest,
        plan: &S14Position0HybridWeightPlan,
        target: &dyn S14Position0HybridUploadTarget,
        dynamic_plan: &DynamicRoutedPagePlan,
        dynamic_upload: &S14DynamicRoutedUploadPlan,
    ) -> Result<S14Position0RoutedUploadReceipt> {
        plan.validate(manifest)?;
        validate_target_sizes(plan, target)?;
        self.require_static_complete(plan)?;
        if self.progress.pending_layer.is_some() || self.progress.pending_head_chunk.is_some() {
            bail!("dynamic routed async record 禁止跨过 pending layer/head");
        }
        let index = self.progress.next_routed_layer;
        let layer_plan = plan
            .routed_layers
            .get(index)
            .ok_or_else(|| anyhow!("dynamic routed layers 已全部录制"))?;
        if dynamic_plan.layer != layer_plan.layer
            || self.progress.next_runtime_static_layer != index + 1
            || layer_plan.layer != FULL_DEPTH_LAYERS[index]
        {
            bail!("dynamic routed async record layer/static cursor 漂移");
        }
        if dynamic_upload.layout.layer != dynamic_plan.layer
            || dynamic_upload.layout.position != dynamic_plan.position
            || dynamic_upload.layout.slots.map(|slot| slot.expert_id) != dynamic_plan.expert_ids
            || dynamic_upload
                .layout
                .slots
                .map(|slot| slot.route_weight_bits)
                != dynamic_plan.route_weights.map(f32::to_bits)
            || dynamic_upload.layout.arena_logical_bytes != layer_plan.logical_bytes
        {
            bail!("dynamic routed async canonical upload plan identity/layout 漂移");
        }

        let bank = layer_plan.bank;
        stage_dynamic_upload_plan(
            &self.routed_staging[bank],
            self.staging_plan.routed_bank_bytes,
            dynamic_upload,
        )?;
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        let copy = [vk::BufferCopy::default().size(dynamic_upload.layout.arena_logical_bytes)];
        ctx.device.cmd_copy_buffer(
            command,
            self.routed_staging[bank].handle(),
            target.routed_bank(bank)?.handle(),
            &copy,
        );
        ctx.device.end_command_buffer(command)?;

        self.progress.next_routed_layer += 1;
        self.stats.routed_layers =
            checked_add(self.stats.routed_layers, 1, "dynamic routed async layers")?;
        self.stats.routed_bytes = checked_add(
            self.stats.routed_bytes,
            layer_plan.logical_bytes,
            "dynamic routed async bytes",
        )?;
        Ok(S14Position0RoutedUploadReceipt {
            layer: layer_plan.layer,
            bank,
            assets: layer_plan.assets.len(),
            bytes: layer_plan.logical_bytes,
        })
    }

    /// 在调用方已 begin 的 transfer command 中录制下一层 static/routed copy。
    /// 本方法不读取 payload、不提交、不等待；只有 `stage_recorded_layer` 成功后才推进层游标。
    ///
    /// # Safety
    /// `command` 必须处于 recording，双 bank staging/device 页必须存活到外部 timeline 完成。
    pub unsafe fn record_next_layer_copies(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        manifest: &Position0WholeTokenManifest,
        plan: &S14Position0HybridWeightPlan,
        target: &dyn S14Position0HybridUploadTarget,
    ) -> Result<S14Position0LayerCopyReceipt> {
        plan.validate(manifest)?;
        validate_target_sizes(plan, target)?;
        self.require_static_complete(plan)?;
        if self.progress.pending_layer.is_some()
            || self.progress.pending_head_chunk.is_some()
            || self.progress.next_runtime_static_layer != self.progress.next_routed_layer
        {
            bail!("position0 async layer record phase/cursor 漂移");
        }
        let index = self.progress.next_routed_layer;
        let layer_plan = plan
            .routed_layers
            .get(index)
            .ok_or_else(|| anyhow!("position0 async routed layers 已全部录制"))?;
        let layer = *FULL_DEPTH_LAYERS
            .get(index)
            .ok_or_else(|| anyhow!("position0 async layer index 越界"))?;
        if layer_plan.layer != layer || layer_plan.bank != index % S14_POSITION0_ROLLING_BANKS {
            bail!("position0 async L{layer} routed plan/bank 漂移");
        }
        let bank = layer_plan.bank;
        let prefix = format!("layers.{layer}.");
        let static_placements = plan
            .resident
            .assets
            .iter()
            .filter(|placement| placement.tensor.starts_with(&prefix))
            .collect::<Vec<_>>();
        let first = static_placements
            .first()
            .ok_or_else(|| anyhow!("position0 async static plan 缺少 L{layer}"))?;
        let first_destination = target.static_asset_destination(first)?;
        let static_resident_hit = first_destination.resident_once;
        let static_buffer = first_destination.buffer.handle();
        let mut static_copies = Vec::with_capacity(static_placements.len());
        let mut static_bytes = 0u64;
        for placement in &static_placements {
            let destination = target.static_asset_destination(placement)?;
            if destination.resident_once != static_resident_hit
                || destination.layer != Some(layer)
                || destination.buffer.handle() != static_buffer
                || (!static_resident_hit && destination.bank != Some(bank))
            {
                bail!("position0 async L{layer} static destination 漂移");
            }
            if !static_resident_hit {
                let end = checked_end(destination.offset, placement.bytes, "async static layer")?;
                if end > self.staging_plan.static_scratch_bytes {
                    bail!("position0 async L{layer} static staging 越界");
                }
                static_copies.push(
                    vk::BufferCopy::default()
                        .src_offset(destination.offset)
                        .dst_offset(destination.offset)
                        .size(placement.bytes),
                );
                static_bytes = checked_add(static_bytes, placement.bytes, "async static bytes")?;
            }
        }
        if static_resident_hit {
            if first_destination.bank.is_some() || !static_copies.is_empty() || static_bytes != 0 {
                bail!("position0 async L{layer} resident static receipt 漂移");
            }
        } else {
            if static_copies.is_empty() {
                bail!("position0 async L{layer} streamed static copies 为空");
            }
            ctx.device.cmd_copy_buffer(
                command,
                self.static_staging[bank].handle(),
                static_buffer,
                &static_copies,
            );
        }

        let routed_copies =
            placement_copies(&layer_plan.assets, self.staging_plan.routed_bank_bytes)?;
        ctx.device.cmd_copy_buffer(
            command,
            self.routed_staging[bank].handle(),
            target.routed_bank(bank)?.handle(),
            &routed_copies,
        );
        let receipt = S14Position0LayerCopyReceipt {
            layer,
            bank,
            static_resident_hit,
            static_assets: static_placements.len(),
            static_bytes,
            routed_assets: layer_plan.assets.len(),
            routed_bytes: layer_plan.logical_bytes,
        };
        self.progress.pending_layer = Some(receipt);
        Ok(receipt)
    }

    /// 在 timeline 已确认双 bank staging 可复用后填充真实 verified payload。
    /// 不录 command、不 submit、不 wait；成功后以原子式状态推进 static/routed 两个游标。
    pub fn stage_recorded_layer(
        &mut self,
        manifest: &Position0WholeTokenManifest,
        plan: &S14Position0HybridWeightPlan,
        store: &mut VerifiedMappedAssetStore,
        target: &dyn S14Position0HybridUploadTarget,
        receipt: S14Position0LayerCopyReceipt,
        timeline_bank: usize,
    ) -> Result<S14Position0LayerCopyReceipt> {
        plan.validate(manifest)?;
        self.require_static_complete(plan)?;
        let index = self.progress.next_routed_layer;
        if self.progress.pending_layer != Some(receipt)
            || self.progress.next_runtime_static_layer != index
            || receipt.layer as usize != index
            || receipt.bank != timeline_bank
        {
            bail!("position0 async layer stage receipt/cursor/bank 漂移");
        }
        let layer_plan = &plan.routed_layers[index];
        let manifest_layer = &manifest.layers[index];
        let prefix = format!("layers.{}.", receipt.layer);
        let static_placements = plan
            .resident
            .assets
            .iter()
            .filter(|placement| placement.tensor.starts_with(&prefix))
            .collect::<Vec<_>>();
        if static_placements.len() != receipt.static_assets
            || layer_plan.assets.len() != receipt.routed_assets
            || layer_plan.logical_bytes != receipt.routed_bytes
        {
            bail!(
                "position0 async L{} staged asset contract 漂移",
                receipt.layer
            );
        }
        if !receipt.static_resident_hit {
            let assets = static_placements
                .iter()
                .map(|placement| resolve_manifest_asset(manifest, placement).cloned())
                .collect::<Result<Vec<_>>>()?;
            let mapped = store.map_verified_batch(&assets)?;
            let mut bytes = 0u64;
            for (placement, payload) in static_placements.iter().zip(&mapped) {
                validate_mapped_identity(payload, placement)?;
                let destination = target.static_asset_destination(placement)?;
                if destination.bank != Some(receipt.bank)
                    || destination.resident_once
                    || destination.layer != Some(receipt.layer)
                {
                    bail!(
                        "position0 async L{} staged static destination 漂移",
                        receipt.layer
                    );
                }
                unsafe {
                    self.static_staging[receipt.bank]
                        .write_at(usize::try_from(destination.offset)?, payload.bytes())
                };
                bytes = checked_add(bytes, placement.bytes, "async staged static bytes")?;
            }
            if bytes != receipt.static_bytes {
                bail!(
                    "position0 async L{} staged static bytes 漂移",
                    receipt.layer
                );
            }
        } else if receipt.static_bytes != 0 {
            bail!("position0 async resident static bytes 必须为零");
        }

        let mapped = store.map_verified_batch(&manifest_layer.assets.routed)?;
        stage_packed_verified_assets(
            &self.routed_staging[receipt.bank],
            self.staging_plan.routed_bank_bytes,
            &layer_plan.assets,
            &mapped,
        )?;

        let next_static = self
            .progress
            .next_runtime_static_layer
            .checked_add(1)
            .ok_or_else(|| anyhow!("position0 async static cursor overflow"))?;
        let next_routed = self
            .progress
            .next_routed_layer
            .checked_add(1)
            .ok_or_else(|| anyhow!("position0 async routed cursor overflow"))?;
        let next_streamed_layers = if receipt.static_resident_hit {
            self.stats.streamed_static_layers
        } else {
            checked_add(
                self.stats.streamed_static_layers,
                1,
                "async streamed static layers",
            )?
        };
        let next_streamed_bytes = checked_add(
            self.stats.streamed_static_bytes,
            receipt.static_bytes,
            "async streamed static bytes",
        )?;
        let next_routed_layers = checked_add(self.stats.routed_layers, 1, "async routed layers")?;
        let next_routed_bytes = checked_add(
            self.stats.routed_bytes,
            receipt.routed_bytes,
            "async routed bytes",
        )?;

        self.progress.pending_layer = None;
        self.progress.next_runtime_static_layer = next_static;
        self.progress.next_routed_layer = next_routed;
        self.stats.streamed_static_layers = next_streamed_layers;
        self.stats.streamed_static_bytes = next_streamed_bytes;
        self.stats.routed_layers = next_routed_layers;
        self.stats.routed_bytes = next_routed_bytes;
        Ok(receipt)
    }

    /// 在调用方已 begin 的 transfer command 中录制下一个 head chunk 的
    /// staging→device copy。本方法不读 payload、不 submit、不 wait；它只冻结
    /// chunk 顺序/bank/bytes 并把一个 pending receipt 绑定到 uploader。
    ///
    /// 调用方必须在 timeline 确认该 staging bank 可复用后，使用
    /// [`Self::stage_recorded_head_chunk`] 填充这个 pending chunk，然后才能 submit
    /// 已录制的 command。
    ///
    /// # Safety
    ///
    /// `command` 必须处于 recording 状态，且 staging/device bank 必须存活到
    /// 外部 timeline 完成。
    pub unsafe fn record_next_head_chunk_copy(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        manifest: &Position0WholeTokenManifest,
        plan: &S14Position0HybridWeightPlan,
        target: &dyn S14Position0HybridUploadTarget,
    ) -> Result<S14Position0HeadChunkReceipt> {
        plan.validate(manifest)?;
        validate_target_sizes(plan, target)?;
        self.require_static_complete(plan)?;
        let asset = resolve_manifest_asset(manifest, &plan.head_weight)?;
        let receipt = head_chunk_receipt(plan, asset.shape[0], self.progress.next_head_chunk)?;
        self.progress.validate_next_head_chunk(receipt)?;
        let destination = target.head_chunk_bank(receipt.bank)?;
        let copy = vk::BufferCopy::default().size(receipt.bytes);
        ctx.device.cmd_copy_buffer(
            command,
            self.head_staging[receipt.bank].handle(),
            destination.handle(),
            &[copy],
        );
        self.progress.record_head_copy(receipt)?;
        Ok(receipt)
    }

    /// 在外部 timeline 的 bank-reuse 检查之后，用 verified mmap 的真实
    /// head Range 填充已录制 copy 引用的 staging bank。本方法不录制
    /// Vulkan command、不 submit、不 wait。成功后才推进 chunk 顺序与字节统计。
    pub fn stage_recorded_head_chunk(
        &mut self,
        manifest: &Position0WholeTokenManifest,
        plan: &S14Position0HybridWeightPlan,
        store: &mut VerifiedMappedAssetStore,
        receipt: S14Position0HeadChunkReceipt,
        timeline_bank: usize,
    ) -> Result<S14Position0HeadChunkReceipt> {
        plan.validate(manifest)?;
        self.require_static_complete(plan)?;
        let asset = resolve_manifest_asset(manifest, &plan.head_weight)?;
        let expected = head_chunk_receipt(plan, asset.shape[0], self.progress.next_head_chunk)?;
        if receipt != expected
            || self.progress.pending_head_chunk != Some(receipt)
            || timeline_bank != receipt.bank
        {
            bail!(
                "position0 async head stage 顺序/bank/receipt 漂移: expected={expected:?} pending={:?} actual={receipt:?} timeline_bank={timeline_bank}",
                self.progress.pending_head_chunk
            );
        }

        let mapped = map_verified_placement(store, asset, &plan.head_weight)?;
        let source_offset = receipt
            .first_row
            .checked_mul(plan.head_row_bytes)
            .ok_or_else(|| anyhow!("head source offset overflow"))?;
        let source_end = source_offset
            .checked_add(receipt.bytes)
            .ok_or_else(|| anyhow!("head source range overflow"))?;
        let source = mapped
            .bytes()
            .get(usize::try_from(source_offset)?..usize::try_from(source_end)?)
            .ok_or_else(|| anyhow!("head mapped range 越界"))?;
        if source.len() as u64 != receipt.bytes
            || receipt.bytes > self.staging_plan.head_chunk_bytes
        {
            bail!("position0 async head staged bytes 漂移");
        }
        let next_chunks = checked_add(self.stats.head_chunks, 1, "head chunks")?;
        let next_bytes = checked_add(self.stats.head_bytes, receipt.bytes, "head bytes")?;

        unsafe { self.head_staging[receipt.bank].write_at(0, source) };
        self.progress
            .finish_recorded_head_stage(receipt, timeline_bank)?;
        self.stats.head_chunks = next_chunks;
        self.stats.head_bytes = next_bytes;
        Ok(receipt)
    }

    pub fn upload_next_head_chunk(
        &mut self,
        ctx: &VulkanContext,
        manifest: &Position0WholeTokenManifest,
        plan: &S14Position0HybridWeightPlan,
        store: &mut VerifiedMappedAssetStore,
        target: &dyn S14Position0HybridUploadTarget,
    ) -> Result<S14Position0HeadChunkReceipt> {
        plan.validate(manifest)?;
        validate_target_sizes(plan, target)?;
        self.require_static_complete(plan)?;
        let asset = resolve_manifest_asset(manifest, &plan.head_weight)?;
        let receipt = head_chunk_receipt(plan, asset.shape[0], self.progress.next_head_chunk)?;
        self.progress.validate_next_head_chunk(receipt)?;
        let mapped = map_verified_placement(store, asset, &plan.head_weight)?;
        let source_offset = receipt
            .first_row
            .checked_mul(plan.head_row_bytes)
            .ok_or_else(|| anyhow!("head source offset overflow"))?;
        let source_end = source_offset
            .checked_add(receipt.bytes)
            .ok_or_else(|| anyhow!("head source range overflow"))?;
        let source = mapped
            .bytes()
            .get(usize::try_from(source_offset)?..usize::try_from(source_end)?)
            .ok_or_else(|| anyhow!("head mapped range 越界"))?;
        unsafe { self.head_staging[receipt.bank].write_at(0, source) };
        let copy = vk::BufferCopy::default().size(receipt.bytes);
        self.copy_and_wait(
            ctx,
            self.head_staging[receipt.bank].handle(),
            target.head_chunk_bank(receipt.bank)?.handle(),
            &[copy],
        )?;
        self.progress.finish_synchronous_head(receipt)?;
        self.stats.head_chunks = checked_add(self.stats.head_chunks, 1, "head chunks")?;
        self.stats.head_bytes = checked_add(self.stats.head_bytes, receipt.bytes, "head bytes")?;
        Ok(receipt)
    }

    fn require_static_complete(&self, plan: &S14Position0HybridWeightPlan) -> Result<()> {
        if !self.progress.static_complete(plan) {
            bail!("position0 hybrid static weights 必须先完成启动上传");
        }
        Ok(())
    }

    fn copy_and_wait(
        &mut self,
        ctx: &VulkanContext,
        source: vk::Buffer,
        destination: vk::Buffer,
        copies: &[vk::BufferCopy],
    ) -> Result<()> {
        if copies.is_empty() {
            bail!("hybrid upload copy list 不能为空");
        }
        unsafe {
            ctx.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
            ctx.device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            ctx.device
                .cmd_copy_buffer(self.command, source, destination, copies);
            ctx.device.end_command_buffer(self.command)?;
            ctx.device.reset_fences(&[self.fence])?;
            let commands = [self.command];
            ctx.device.queue_submit(
                ctx.q_transfer,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                self.fence,
            )?;
            ctx.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        self.stats.transfer_submits =
            checked_add(self.stats.transfer_submits, 1, "transfer submits")?;
        Ok(())
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_fence(self.fence, None);
            ctx.device.destroy_command_pool(self.command_pool, None);
        }
        destroy_staging(
            ctx,
            self.static_staging,
            self.routed_staging,
            self.head_staging,
        );
    }
}

fn validate_target_sizes(
    plan: &S14Position0HybridWeightPlan,
    target: &dyn S14Position0HybridUploadTarget,
) -> Result<()> {
    for placement in &plan.resident.assets {
        let destination = target.static_asset_destination(placement)?;
        let end = checked_end(destination.offset, placement.bytes, "static target")?;
        if end > destination.buffer.size() {
            bail!("hybrid static target too small: {}", placement.tensor);
        }
        if let Some(layer) = parse_layer_tensor(&placement.tensor) {
            if destination.layer != Some(layer) {
                bail!("hybrid static target layer drift: {}", placement.tensor);
            }
        } else if destination.layer.is_some() || !destination.resident_once {
            bail!("hybrid resident-small target drift: {}", placement.tensor);
        }
    }
    for bank in 0..S14_POSITION0_ROLLING_BANKS {
        if target.routed_bank(bank)?.size() < plan.routed_bank_bytes
            || target.head_chunk_bank(bank)?.size() < plan.head_chunk_bytes
        {
            bail!("hybrid bank {bank} target too small");
        }
    }
    Ok(())
}

fn resolve_manifest_asset<'a>(
    manifest: &'a Position0WholeTokenManifest,
    placement: &S14Position0AssetPlacement,
) -> Result<&'a Position0Asset> {
    let mut matches = manifest
        .all_assets()
        .filter(|asset| asset.tensor == placement.tensor);
    let asset = matches.next().ok_or_else(|| {
        anyhow!(
            "hybrid placement tensor 不在 manifest: {}",
            placement.tensor
        )
    })?;
    if matches.next().is_some() {
        bail!("hybrid manifest tensor 名不唯一: {}", placement.tensor);
    }
    validate_placement_identity(asset, placement)?;
    Ok(asset)
}

fn map_verified_placement(
    store: &mut VerifiedMappedAssetStore,
    asset: &Position0Asset,
    placement: &S14Position0AssetPlacement,
) -> Result<Arc<VerifiedMappedAsset>> {
    validate_placement_identity(asset, placement)?;
    let mut mapped = store.map_verified_batch(std::slice::from_ref(asset))?;
    let mapped = mapped
        .pop()
        .ok_or_else(|| anyhow!("VerifiedMappedAssetStore 未返回 lease"))?;
    validate_mapped_identity(&mapped, placement)?;
    Ok(mapped)
}

fn stage_dynamic_upload_plan(
    staging: &GpuBuffer,
    capacity: u64,
    dynamic_upload: &S14DynamicRoutedUploadPlan,
) -> Result<()> {
    let arena_bytes = dynamic_upload.layout.arena_logical_bytes;
    if staging.size() < capacity
        || arena_bytes > capacity
        || dynamic_upload.layout.placements.len() != DYNAMIC_ROUTED_RANGE_COUNT
        || dynamic_upload.mapped_assets().len() != DYNAMIC_ROUTED_RANGE_COUNT
        || staging.mapped().is_null()
    {
        bail!("dynamic routed canonical staging ledger/capacity 漂移");
    }
    let target =
        unsafe { std::slice::from_raw_parts_mut(staging.mapped(), usize::try_from(arena_bytes)?) };
    dynamic_upload.stage_into(target)
}

fn stage_packed_verified_assets(
    staging: &GpuBuffer,
    capacity: u64,
    placements: &[S14Position0AssetPlacement],
    mapped: &[Arc<VerifiedMappedAsset>],
) -> Result<()> {
    if placements.is_empty() || placements.len() != mapped.len() || staging.size() < capacity {
        bail!("hybrid routed placement/mapped/staging ledger drift");
    }
    for (placement, payload) in placements.iter().zip(mapped) {
        validate_mapped_identity(payload, placement)?;
        let end = checked_end(placement.offset, placement.bytes, "routed placement")?;
        if end > capacity {
            bail!("hybrid routed placement exceeds bank");
        }
        unsafe { staging.write_at(usize::try_from(placement.offset)?, payload.bytes()) };
    }
    Ok(())
}

fn placement_copies(
    placements: &[S14Position0AssetPlacement],
    capacity: u64,
) -> Result<Vec<vk::BufferCopy>> {
    let mut copies = Vec::with_capacity(placements.len());
    for placement in placements {
        let end = checked_end(placement.offset, placement.bytes, "hybrid placement")?;
        if placement.offset % 4 != 0 || placement.bytes % 4 != 0 || end > capacity {
            bail!("hybrid placement 不是合法 Vulkan copy range");
        }
        copies.push(
            vk::BufferCopy::default()
                .src_offset(placement.offset)
                .dst_offset(placement.offset)
                .size(placement.bytes),
        );
    }
    Ok(copies)
}

fn validate_placement_identity(
    asset: &Position0Asset,
    placement: &S14Position0AssetPlacement,
) -> Result<()> {
    if asset.tensor != placement.tensor
        || asset.kind != placement.kind
        || asset.expert_id != placement.expert_id
        || asset.path != placement.path
        || asset.sha256 != placement.sha256
        || asset.bytes != placement.bytes
    {
        bail!(
            "hybrid asset placement identity drift: {}",
            placement.tensor
        );
    }
    Ok(())
}

fn validate_mapped_identity(
    mapped: &VerifiedMappedAsset,
    placement: &S14Position0AssetPlacement,
) -> Result<()> {
    let canonical = placement
        .path
        .canonicalize()
        .with_context(|| format!("resolve hybrid payload {}", placement.path.display()))?;
    if mapped.tensor() != placement.tensor
        || mapped.path() != canonical
        || mapped.expected_sha256() != placement.sha256
        || mapped.bytes().len() as u64 != placement.bytes
    {
        bail!(
            "hybrid verified mapped identity drift: {}",
            placement.tensor
        );
    }
    Ok(())
}

fn head_chunk_receipt(
    plan: &S14Position0HybridWeightPlan,
    total_rows: u64,
    chunk: u64,
) -> Result<S14Position0HeadChunkReceipt> {
    if chunk >= plan.head_chunk_count {
        bail!("position0 head chunks 已全部上传");
    }
    let first_row = chunk
        .checked_mul(plan.head_chunk_rows)
        .ok_or_else(|| anyhow!("head row offset overflow"))?;
    let remaining_rows = total_rows
        .checked_sub(first_row)
        .ok_or_else(|| anyhow!("head chunk first row 超出词表"))?;
    let rows = remaining_rows.min(plan.head_chunk_rows);
    if rows == 0 {
        bail!("head chunk rows 不能为零");
    }
    let bytes = rows
        .checked_mul(plan.head_row_bytes)
        .ok_or_else(|| anyhow!("head chunk bytes overflow"))?;
    if bytes == 0 || bytes > plan.head_chunk_bytes {
        bail!("head chunk bytes 超出物理 bank");
    }
    Ok(S14Position0HeadChunkReceipt {
        chunk,
        bank: chunk as usize % S14_POSITION0_ROLLING_BANKS,
        first_row,
        rows,
        bytes,
    })
}

fn checked_end(offset: u64, bytes: u64, label: &str) -> Result<u64> {
    if bytes == 0 {
        bail!("{label} bytes 不能为零");
    }
    offset
        .checked_add(bytes)
        .ok_or_else(|| anyhow!("{label} range overflow"))
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| anyhow!("{label} counter overflow"))
}

fn parse_layer_tensor(tensor: &str) -> Option<u8> {
    let rest = tensor.strip_prefix("layers.")?;
    let (layer, _) = rest.split_once('.')?;
    layer.parse().ok()
}

fn destroy_staging(
    ctx: &VulkanContext,
    static_staging: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
    routed_staging: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
    head_staging: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
) {
    for buffer in static_staging
        .into_iter()
        .chain(routed_staging)
        .chain(head_staging)
    {
        buffer.destroy(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn manifest() -> Position0WholeTokenManifest {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
        );
        Position0WholeTokenManifest::load(&path).unwrap()
    }

    #[test]
    fn hybrid_staging_is_bounded_and_never_full_model_sized() {
        let manifest = manifest();
        let plan = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        let staging = S14Position0HybridStagingPlan::build(&plan).unwrap();
        assert_eq!(staging.routed_bank_bytes, 80_216_064);
        assert_eq!(staging.head_chunk_bytes, 33_554_432);
        // static scratch 也必须双 bank，否则 L(n+1) 的主机 staging 会覆盖仍被
        // L(n) transfer 引用的字节。总量仍严格小于 1GiB、不到逻辑权重的十分之一。
        assert!(staging.allocated_staging_bytes < 1024 * 1024 * 1024);
        assert!(staging.allocated_staging_bytes * 10 < plan.logical_payload_bytes);
    }

    #[test]
    fn schedule_is_static_once_routed_layer_double_bank_and_head_double_chunk() {
        let manifest = manifest();
        let plan = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        let mut progress = HybridUploadProgress::new();
        assert!(!progress.static_complete(&plan));
        progress.static_complete = true;
        assert!(progress.static_complete(&plan));

        for (index, layer) in plan.routed_layers.iter().enumerate() {
            assert_eq!(layer.layer, index as u8);
            assert_eq!(layer.bank, index % 2);
            assert!(layer.assets.iter().all(|asset| asset.expert_id.is_some()));
            progress.next_runtime_static_layer += 1;
            progress.next_routed_layer += 1;
        }
        for chunk in 0..plan.head_chunk_count {
            assert_eq!(chunk as usize % 2, chunk as usize & 1);
            progress.next_head_chunk += 1;
        }
        assert!(progress.all_complete(&plan));
        assert_eq!(plan.head_chunk_count, 32);
        let final_rows = manifest
            .final_section
            .assets
            .iter()
            .find(|asset| asset.tensor == "head.weight")
            .unwrap()
            .shape[0]
            - 31 * plan.head_chunk_rows;
        assert_eq!(final_rows, 2304);
    }

    #[test]
    fn drained_partial_token_can_begin_again_without_reallocation() {
        let manifest = manifest();
        let plan = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        let mut progress = HybridUploadProgress::new();
        progress.static_complete = true;
        progress.prepare_persistent_token(&plan, true).unwrap();

        progress.next_runtime_static_layer = 7;
        progress.next_routed_layer = 6;
        progress.reset_token();

        assert!(progress.at_token_start());
        assert!(!progress.all_complete(&plan));
        progress.prepare_persistent_token(&plan, false).unwrap();
        assert!(progress.at_token_start());
    }

    #[test]
    fn completed_causal_block_reopens_static_and_head_streams() {
        let manifest = manifest();
        let plan = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        let mut progress = HybridUploadProgress::new();
        progress.static_complete = true;
        progress.next_runtime_static_layer = plan.routed_layers.len();
        progress.next_routed_layer = 0;
        progress.next_head_chunk = plan.head_chunk_count;

        progress.prepare_next_causal_block_head(&plan).unwrap();
        assert_eq!(progress.next_head_chunk, 0);
        assert_eq!(progress.next_runtime_static_layer, 0);
        assert_eq!(progress.next_routed_layer, 0);
        assert!(progress.static_complete);

        progress.next_head_chunk = plan.head_chunk_count - 1;
        assert!(progress.prepare_next_causal_block_head(&plan).is_err());
    }

    #[test]
    fn every_hybrid_placement_resolves_to_one_manifest_asset() {
        let manifest = manifest();
        let plan = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        for placement in &plan.resident.assets {
            resolve_manifest_asset(&manifest, placement).unwrap();
        }
        for layer in &plan.routed_layers {
            for placement in &layer.assets {
                resolve_manifest_asset(&manifest, placement).unwrap();
            }
        }
        resolve_manifest_asset(&manifest, &plan.head_weight).unwrap();
    }

    #[test]
    fn verified_store_is_the_only_payload_gateway_and_identity_drift_fails() {
        let fixture = FixtureDir::new();
        let bytes = b"verified-hybrid-payload";
        let asset = fixture.asset("payload.bin", "fixture.tensor", bytes);
        let placement = S14Position0AssetPlacement {
            tensor: asset.tensor.clone(),
            kind: asset.kind.clone(),
            expert_id: asset.expert_id,
            path: asset.path.clone(),
            sha256: asset.sha256.clone(),
            offset: 0,
            bytes: asset.bytes,
        };
        let mut store = VerifiedMappedAssetStore::new(&fixture.root).unwrap();
        let mapped = map_verified_placement(&mut store, &asset, &placement).unwrap();
        assert_eq!(mapped.bytes(), bytes);
        assert_eq!(store.stats().sha256_bytes, bytes.len() as u64);
        let again = map_verified_placement(&mut store, &asset, &placement).unwrap();
        assert!(Arc::ptr_eq(&mapped, &again));
        assert_eq!(store.stats().hits, 1);

        let mut drift = placement;
        drift.bytes += 1;
        assert!(map_verified_placement(&mut store, &asset, &drift).is_err());
    }

    struct FixtureDir {
        root: PathBuf,
    }

    impl FixtureDir {
        fn new() -> Self {
            let id = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir()
                .join(format!("polaris-hybrid-upload-{}-{id}", std::process::id()));
            fs::create_dir(&root).unwrap();
            Self { root }
        }

        fn asset(&self, name: &str, tensor: &str, bytes: &[u8]) -> Position0Asset {
            let path = self.root.join(name);
            fs::write(&path, bytes).unwrap();
            let sha256 = Sha256::digest(bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            Position0Asset {
                tensor: tensor.into(),
                kind: "fixture".into(),
                expert_id: None,
                dtype: "I8".into(),
                shape: vec![bytes.len() as u64],
                bytes: bytes.len() as u64,
                range_key: format!("fixture:{name}"),
                cache_key: sha256.clone(),
                path,
                sha256,
                proof_path: self.root.join(format!("{name}.json")),
                proof_sha256: "a".repeat(64),
                hash_authority: "fixture".into(),
                payload_rehashed_by_builder: true,
                source: Value::Null,
            }
        }
    }

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

#[cfg(test)]
#[path = "../tests/support/s14_position0_hybrid_upload_async_head.rs"]
mod async_head_tests;
