//! K=4/8 production terminal prelude 与资源 owner。
//!
//! 该模块只接受已经由 FullDepth43 producer 强拥有的真实资源：L42 后 `[K,4,4096]`
//! BF16 HC、final HC/norm 权重、单一 K-prefix device checkpoint arena 与32个 BF16
//! head chunk。它在一个 command 中执行 K 路 final HC、一次 K-row RMSNorm 与一次
//! BF16→F32，随后把 sticky status 复制到同 owner 的 readback，并用同 owner 的 timeline
//! 原子发布。它不接受预测 token；host snapshot finalizer 仍只能在 terminal adapter 完成
//! GPU batched head 回读后被消费。

use crate::{
    compute::StorageBufferSlice,
    s14_bf16_rmsnorm::{S14Bf16RmsNormDispatch, S14Bf16RmsNormPipeline, S14Bf16RmsNormShape},
    s14_bf16_to_f32::{S14Bf16ToF32Dispatch, S14Bf16ToF32Pipeline, S14Bf16ToF32Shape},
    s14_causal_block_layer::{
        S14CausalBlockHiddenBinding, S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE,
        S14_CAUSAL_BLOCK_STREAM_WIDTH,
    },
    s14_causal_block_prefix_arena::S14CausalBlockPrefixCheckpointArena,
    s14_causal_block_production_bundle::{
        S14CausalBlockContextBound, S14CausalBlockProductionTerminalPublisher,
    },
    s14_causal_block_production_evidence::S14CausalBlockProductionEvidenceSnapshot,
    s14_causal_block_terminal_adapter::{
        S14CausalBlockHostCandidateFinalizer,
        S14CausalBlockTerminalProductionPublisher as S14RawTerminalPublisher,
        S14CausalBlockTerminalProductionSource, S14CausalBlockTerminalResource,
        S14CausalBlockTerminalResourceOwner,
    },
    s14_final_hc_head::{
        S14FinalHcHeadBindings, S14FinalHcHeadBufferSlice, S14FinalHcHeadDispatch,
        S14FinalHcHeadPipeline, S14FinalHcHeadShape,
    },
    s14_head_chunk_argmax::{S14HeadChunkArgmaxShape, S14_HEAD_CHUNK_COUNT},
    s14_position0_hybrid_upload::{
        S14Position0CausalBlockUploadLease, S14Position0HeadChunkReceipt,
        S14Position0HybridUploader,
    },
    s14_position0_mapped_assets::VerifiedMappedAssetStore,
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{Position0WholeTokenManifest, RouteDecision, FULL_DEPTH_LAYERS};
use std::{
    fmt,
    sync::{Arc, Mutex},
};

const TERMINAL_ALIGNMENT: u64 = 256;
const FINAL_RMS_EPSILON: f32 = 1.0e-6;
const STATUS_BYTES: u64 = 4;
const PRODUCER_TIMELINE_VALUE: u64 = 1;
const PRODUCER_WAIT_TIMEOUT_NS: u64 = 60_000_000_000;

#[derive(Clone)]
pub struct S14CausalBlockOwnedBufferSlice {
    pub buffer: Arc<GpuBuffer>,
    pub offset: u64,
}

impl fmt::Debug for S14CausalBlockOwnedBufferSlice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockOwnedBufferSlice")
            .field("buffer", &self.buffer.handle())
            .field("offset", &self.offset)
            .field("capacity", &self.buffer.size())
            .finish()
    }
}

impl S14CausalBlockOwnedBufferSlice {
    pub fn new(buffer: Arc<GpuBuffer>, offset: u64) -> Self {
        Self { buffer, offset }
    }
}

pub struct S14CausalBlockTerminalHeadUploadState {
    pub uploader: S14Position0HybridUploader,
    pub store: VerifiedMappedAssetStore,
}

/// Provider one-shot 交给 terminal 的同源 uploader capability。wrapper 不实现 Clone，
/// 防止同一 block lease 被两个 terminal owner 消费。
pub struct S14CausalBlockTerminalHeadLeaseOwner {
    state: Arc<Mutex<S14CausalBlockTerminalHeadUploadState>>,
    lease: S14Position0CausalBlockUploadLease,
}

impl S14CausalBlockTerminalHeadLeaseOwner {
    pub fn new(
        state: Arc<Mutex<S14CausalBlockTerminalHeadUploadState>>,
        lease: S14Position0CausalBlockUploadLease,
    ) -> Self {
        Self { state, lease }
    }

    pub const fn lease(&self) -> S14Position0CausalBlockUploadLease {
        self.lease
    }

    fn into_parts(
        self,
    ) -> (
        Arc<Mutex<S14CausalBlockTerminalHeadUploadState>>,
        S14Position0CausalBlockUploadLease,
    ) {
        (self.state, self.lease)
    }
}

impl fmt::Debug for S14CausalBlockTerminalHeadLeaseOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockTerminalHeadLeaseOwner")
            .field("state", &Arc::as_ptr(&self.state))
            .field("lease", &self.lease)
            .finish()
    }
}

impl fmt::Debug for S14CausalBlockTerminalHeadUploadState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockTerminalHeadUploadState")
            .field("upload_stats", &self.uploader.stats())
            .finish_non_exhaustive()
    }
}

/// 外部 FullDepth43 producer 必须一次性提供的强 owner 输入。checkpoint slices 必须
/// 全部属于同一 arena；32个逻辑 head chunk 只由同一个 paged arena 的两个 bank 承载。
/// uploader/store 必须是执行本 token 43层上传的同一份 persistent verified owner。
pub struct S14CausalBlockTerminalResourceOwnerInputs {
    pub context: Arc<VulkanContext>,
    pub block_size: usize,
    pub final_hidden: S14CausalBlockOwnedBufferSlice,
    /// FullDepth43 producer 开始前绑定的单一 K-prefix arena。checkpoint slices
    /// 只能在43层累积写集全部 seal 后导出，避免 terminal 安装生命周期环。
    pub prefix_checkpoint_arena: Arc<S14CausalBlockPrefixCheckpointArena>,
    pub paged_arena: Arc<S14Position0PagedWeightArena>,
    pub head_manifest: Arc<Position0WholeTokenManifest>,
    pub head_weight_plan: Arc<S14Position0HybridWeightPlan>,
    pub head_upload: S14CausalBlockTerminalHeadLeaseOwner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockTerminalArenaLayout {
    pub final_hidden_bf16_offset: u64,
    pub normalized_bf16_offset: u64,
    pub normalized_f32_offset: u64,
    pub inverse_rms_offset: u64,
    pub hc_aux_offset: u64,
    pub status_offset: u64,
    pub arena_bytes: u64,
}

impl S14CausalBlockTerminalArenaLayout {
    pub fn build(block_size: usize, alignment: u64) -> Result<Self> {
        if !matches!(block_size, 4 | 8) {
            bail!("production terminal arena 只接受 K=4/8");
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            bail!("production terminal arena alignment 必须是非零二次幂");
        }
        let lane_hidden_bf16_bytes = (S14_CAUSAL_BLOCK_STREAM_WIDTH as u64)
            .checked_mul(2)
            .context("terminal lane hidden bytes overflow")?;
        let lane_normalized_f32_bytes = (S14_CAUSAL_BLOCK_STREAM_WIDTH as u64)
            .checked_mul(4)
            .context("terminal lane normalized bytes overflow")?;
        let mut cursor = 0u64;
        let mut take = |bytes: u64| -> Result<u64> {
            cursor = align_up(cursor, alignment)?;
            let offset = cursor;
            cursor = cursor
                .checked_add(bytes)
                .context("production terminal arena size overflow")?;
            Ok(offset)
        };
        let k = block_size as u64;
        let final_hidden_bf16_offset = take(k * lane_hidden_bf16_bytes)?;
        let normalized_bf16_offset = take(k * lane_hidden_bf16_bytes)?;
        let normalized_f32_offset = take(k * lane_normalized_f32_bytes)?;
        let inverse_rms_offset = take(k * 4)?;
        let hc_aux_offset = take(k * 8 * 4)?;
        let status_offset = take(STATUS_BYTES)?;
        let arena_bytes = align_up(cursor, alignment)?;
        Ok(Self {
            final_hidden_bf16_offset,
            normalized_bf16_offset,
            normalized_f32_offset,
            inverse_rms_offset,
            hc_aux_offset,
            status_offset,
            arena_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OwnerPhase {
    Ready,
    Recording,
    Submitted,
    Validated,
    DrainedAfterReject,
    Poisoned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockTerminalPublishReceipt {
    pub base_position: u32,
    pub block_size: usize,
    pub completed_layers: usize,
    pub producer_timeline_value: u64,
    pub normalized_head_rows_offset: u64,
    pub checkpoint_count: usize,
    pub head_chunk_count: usize,
    pub predicted_tokens_prebuilt: bool,
}

struct TerminalPipelines {
    final_hc: S14FinalHcHeadPipeline,
    final_hc_dispatches: Vec<S14FinalHcHeadDispatch>,
    rmsnorm: S14Bf16RmsNormPipeline,
    rmsnorm_dispatch: S14Bf16RmsNormDispatch,
    to_f32: S14Bf16ToF32Pipeline,
    to_f32_dispatch: S14Bf16ToF32Dispatch,
}

impl TerminalPipelines {
    fn destroy(self, ctx: &VulkanContext) {
        self.to_f32_dispatch.binder.destroy(ctx);
        self.to_f32.destroy(ctx);
        self.rmsnorm_dispatch.binder.destroy(ctx);
        self.rmsnorm.destroy(ctx);
        for dispatch in self.final_hc_dispatches {
            dispatch.binder.destroy(ctx);
        }
        self.final_hc.destroy(ctx);
    }
}

/// 一个 block 的 terminal producer 与全部被 adapter 借出的资源 owner。该对象是 one-shot；
/// 发布失败会先等待自己的 producer timeline，再进入 drained 状态，禁止复用陈旧输出。
pub struct S14CausalBlockProductionTerminalResourceOwner {
    context: Arc<VulkanContext>,
    block_size: usize,
    final_hidden: S14CausalBlockOwnedBufferSlice,
    prefix_checkpoint_arena: Arc<S14CausalBlockPrefixCheckpointArena>,
    paged_arena: Arc<S14Position0PagedWeightArena>,
    head_manifest: Arc<Position0WholeTokenManifest>,
    head_weight_plan: Arc<S14Position0HybridWeightPlan>,
    head_upload: Arc<Mutex<S14CausalBlockTerminalHeadUploadState>>,
    head_upload_lease: S14Position0CausalBlockUploadLease,
    layout: S14CausalBlockTerminalArenaLayout,
    arena: Option<GpuBuffer>,
    status_readback: Option<GpuBuffer>,
    pipelines: Option<TerminalPipelines>,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    producer_timeline: vk::Semaphore,
    phase: Mutex<OwnerPhase>,
}

impl fmt::Debug for S14CausalBlockProductionTerminalResourceOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockProductionTerminalResourceOwner")
            .field("context", &Arc::as_ptr(&self.context))
            .field("block_size", &self.block_size)
            .field("final_hidden", &self.final_hidden)
            .field(
                "checkpoint_state_bytes",
                &self.prefix_checkpoint_arena.layout().checkpoint_state_bytes,
            )
            .field(
                "checkpoint_count",
                &self.prefix_checkpoint_arena.layout().block_size,
            )
            .field("head_chunk_count", &(S14_HEAD_CHUNK_COUNT as usize))
            .field("head_upload_lease", &self.head_upload_lease)
            .field("paged_arena", &Arc::as_ptr(&self.paged_arena))
            .field("layout", &self.layout)
            .field("producer_timeline", &self.producer_timeline)
            .field("phase", &self.phase.lock().ok().map(|phase| *phase))
            .finish()
    }
}

impl S14CausalBlockProductionTerminalResourceOwner {
    pub fn new(inputs: S14CausalBlockTerminalResourceOwnerInputs) -> Result<Arc<Self>> {
        let S14CausalBlockTerminalResourceOwnerInputs {
            context,
            block_size,
            final_hidden,
            prefix_checkpoint_arena,
            paged_arena,
            head_manifest,
            head_weight_plan,
            head_upload,
        } = inputs;
        let (head_upload, head_upload_lease) = head_upload.into_parts();
        if !context.timeline_semaphore || !matches!(block_size, 4 | 8) {
            bail!("production terminal owner 要求 timeline semaphore 与 K=4/8");
        }
        if !Arc::ptr_eq(&context, prefix_checkpoint_arena.context()) {
            bail!("production terminal owner 与 prefix checkpoint arena VulkanContext 漂移");
        }
        {
            let state = head_upload
                .lock()
                .map_err(|_| anyhow::anyhow!("production terminal head uploader poisoned"))?;
            if !state.uploader.resident_static_uploaded() {
                bail!("production terminal HC/norm 绑定前 resident-small 尚未完成 verified upload");
            }
            state
                .uploader
                .validate_causal_block_terminal_head_stream_for_lease(
                    &head_weight_plan,
                    &head_upload_lease,
                )
                .context("production terminal owner 绑定 uploader lease")?;
        }
        let terminal_static = resolve_terminal_static_slices(paged_arena.as_ref())?;
        validate_external_resources(
            block_size,
            &final_hidden,
            terminal_static,
            &prefix_checkpoint_arena,
            &paged_arena,
            &head_manifest,
            &head_weight_plan,
        )?;
        let device_alignment = unsafe {
            context
                .instance
                .get_physical_device_properties(context.physical)
                .limits
                .min_storage_buffer_offset_alignment
        }
        .max(TERMINAL_ALIGNMENT);
        let layout = S14CausalBlockTerminalArenaLayout::build(block_size, device_alignment)?;
        let arena = GpuBuffer::new_vram(
            &context,
            layout.arena_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .context("allocate production terminal arena")?;
        let status_readback = match GpuBuffer::new(
            &context,
            STATUS_BYTES,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                arena.destroy(&context);
                return Err(error.context("allocate production terminal status readback"));
            }
        };
        let (command_pool, command) = match allocate_command(&context) {
            Ok(value) => value,
            Err(error) => {
                status_readback.destroy(&context);
                arena.destroy(&context);
                return Err(error);
            }
        };
        let producer_timeline = match create_timeline(&context) {
            Ok(timeline) => timeline,
            Err(error) => {
                unsafe { context.device.destroy_command_pool(command_pool, None) };
                status_readback.destroy(&context);
                arena.destroy(&context);
                return Err(error);
            }
        };
        let pipelines = match build_pipelines(
            &context,
            block_size,
            &final_hidden,
            terminal_static,
            &arena,
            layout,
        ) {
            Ok(pipelines) => pipelines,
            Err(error) => {
                unsafe {
                    context.device.destroy_semaphore(producer_timeline, None);
                    context.device.destroy_command_pool(command_pool, None);
                }
                status_readback.destroy(&context);
                arena.destroy(&context);
                return Err(error);
            }
        };
        Ok(Arc::new(Self {
            context,
            block_size,
            final_hidden,
            prefix_checkpoint_arena,
            paged_arena,
            head_manifest,
            head_weight_plan,
            head_upload,
            head_upload_lease,
            layout,
            arena: Some(arena),
            status_readback: Some(status_readback),
            pipelines: Some(pipelines),
            command_pool,
            command,
            producer_timeline,
            phase: Mutex::new(OwnerPhase::Ready),
        }))
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn checkpoint_state_bytes(&self) -> u64 {
        self.prefix_checkpoint_arena.layout().checkpoint_state_bytes
    }

    pub fn paged_weight_arena(&self) -> &Arc<S14Position0PagedWeightArena> {
        &self.paged_arena
    }

    /// 只读导出由同一 prefix arena 在 GPU 完成后累计的 production 强回执。
    /// 不暴露 ledger 写入口，也不延长任何临时 descriptor/command 生命周期。
    pub fn production_evidence_snapshot(&self) -> Result<S14CausalBlockProductionEvidenceSnapshot> {
        self.prefix_checkpoint_arena.production_evidence_snapshot()
    }

    pub fn layout(&self) -> S14CausalBlockTerminalArenaLayout {
        self.layout
    }

    /// 只接受 sealed FullDepth43 的真实 final hidden/routes/checkpoints。先提交 terminal
    /// HC/norm producer timeline，再把同一个 Arc owner 移入一次性 source。接口中没有预测
    /// token 参数；预测只能由 adapter 的 GPU batched head 产生。
    pub(crate) fn record_and_publish(
        self: &Arc<Self>,
        publisher: &S14CausalBlockProductionTerminalPublisher,
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: Vec<Vec<RouteDecision>>,
        host_candidates: Box<dyn S14CausalBlockHostCandidateFinalizer>,
    ) -> Result<S14CausalBlockTerminalPublishReceipt, String> {
        let (source, receipt) = self.prepare_terminal_publication(
            base_position,
            final_hidden,
            routes_by_position,
            host_candidates,
        )?;
        if let Err(error) = publisher.publish(S14CausalBlockContextBound::new(
            Arc::clone(&self.context),
            source,
        )) {
            return Err(self.reject_after_publish_failure(error));
        }
        Ok(receipt)
    }

    /// StarFold 直接 terminal endpoint 的原生发布入口。该路径复用同一份 identity、
    /// checkpoint seal、HC/norm prelude 与失败排空规则，但不进入旧 production bundle
    /// 的 phase 包装。预测 token 仍只能由 raw channel 的 GPU batched head 产生。
    pub(crate) fn record_and_publish_starfold(
        self: &Arc<Self>,
        publisher: &S14RawTerminalPublisher,
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: Vec<Vec<RouteDecision>>,
        host_candidates: Box<dyn S14CausalBlockHostCandidateFinalizer>,
    ) -> Result<S14CausalBlockTerminalPublishReceipt, String> {
        let (source, receipt) = self.prepare_terminal_publication(
            base_position,
            final_hidden,
            routes_by_position,
            host_candidates,
        )?;
        if let Err(error) = publisher.publish(source) {
            return Err(self.reject_after_publish_failure(error));
        }
        Ok(receipt)
    }

    fn prepare_terminal_publication(
        self: &Arc<Self>,
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: Vec<Vec<RouteDecision>>,
        host_candidates: Box<dyn S14CausalBlockHostCandidateFinalizer>,
    ) -> Result<
        (
            S14CausalBlockTerminalProductionSource,
            S14CausalBlockTerminalPublishReceipt,
        ),
        String,
    > {
        self.validate_publication_identity(base_position, final_hidden, &routes_by_position)
            .map_err(|error| error.to_string())?;
        let checkpoints = self
            .prefix_checkpoint_arena
            .seal_and_export_terminal_checkpoints()
            .map_err(|error| format!("post-seal K-prefix checkpoint 导出失败: {error:#}"))?;
        validate_checkpoint_slices(self.block_size, self.checkpoint_state_bytes(), &checkpoints)
            .map_err(|error| format!("post-seal K-prefix checkpoint 验收失败: {error:#}"))?;
        if let Err(error) = self.record_and_submit_prelude() {
            self.set_phase(OwnerPhase::Poisoned);
            return Err(format!(
                "record production terminal HC/norm 失败: {error:#}"
            ));
        }
        let resources: Arc<dyn S14CausalBlockTerminalResourceOwner> = self.clone();
        let source = S14CausalBlockTerminalProductionSource {
            completed_layers: FULL_DEPTH_LAYERS.len(),
            base_position,
            final_hidden,
            normalized_head_rows_offset: self.layout.normalized_f32_offset,
            checkpoint_offsets: checkpoints.iter().map(|slice| slice.offset).collect(),
            head_chunk_count: S14_HEAD_CHUNK_COUNT as usize,
            producer_timeline_value: PRODUCER_TIMELINE_VALUE,
            routes_by_position,
            host_candidates,
            resources,
        };
        let receipt = S14CausalBlockTerminalPublishReceipt {
            base_position,
            block_size: self.block_size,
            completed_layers: FULL_DEPTH_LAYERS.len(),
            producer_timeline_value: PRODUCER_TIMELINE_VALUE,
            normalized_head_rows_offset: self.layout.normalized_f32_offset,
            checkpoint_count: checkpoints.len(),
            head_chunk_count: S14_HEAD_CHUNK_COUNT as usize,
            predicted_tokens_prebuilt: false,
        };
        Ok((source, receipt))
    }

    fn reject_after_publish_failure(&self, error: String) -> String {
        let drain = self.wait_for_producer();
        self.set_phase(if drain.is_ok() {
            OwnerPhase::DrainedAfterReject
        } else {
            OwnerPhase::Poisoned
        });
        match drain {
            Ok(()) => error,
            Err(drain_error) => format!(
                "terminal source publish 失败: {error}; producer drain 失败: {drain_error:#}"
            ),
        }
    }

    fn validate_publication_identity(
        &self,
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<()> {
        base_position
            .checked_add(self.block_size as u32)
            .context("production terminal position overflow")?;
        let expected_hidden_bytes = hidden_bytes(self.block_size)?;
        if base_position != self.prefix_checkpoint_arena.base_position()
            || final_hidden.buffer != self.final_hidden.buffer.handle()
            || final_hidden.offset != self.final_hidden.offset
            || final_hidden.bytes != expected_hidden_bytes
            || final_hidden.block_size != self.block_size
            || routes_by_position.len() != self.block_size
            || routes_by_position
                .iter()
                .any(|routes| routes.len() != FULL_DEPTH_LAYERS.len())
            || host_candidates_identity_would_be_invalid(
                self.block_size,
                base_position,
                routes_by_position,
            )
        {
            bail!("production terminal final hidden/routes K/FullDepth43 identity 漂移");
        }
        Ok(())
    }

    fn record_and_submit_prelude(&self) -> Result<()> {
        {
            let mut phase = self.lock_phase()?;
            if *phase != OwnerPhase::Ready {
                bail!("production terminal owner 是 one-shot，禁止重复 record/publish");
            }
            *phase = OwnerPhase::Recording;
        }
        let arena = self
            .arena
            .as_ref()
            .context("production terminal arena 已销毁")?;
        let status_readback = self
            .status_readback
            .as_ref()
            .context("production terminal status readback 已销毁")?;
        let pipelines = self
            .pipelines
            .as_ref()
            .context("production terminal pipelines 已销毁")?;
        unsafe {
            self.context
                .device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
            self.context.device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.context.device.cmd_fill_buffer(
                self.command,
                arena.handle(),
                self.layout.status_offset,
                STATUS_BYTES,
                0,
            );
            transfer_to_compute_barrier(&self.context, self.command);
            for dispatch in &pipelines.final_hc_dispatches {
                pipelines
                    .final_hc
                    .cmd(&self.context, self.command, dispatch);
                compute_to_compute_barrier(&self.context, self.command);
            }
            pipelines
                .rmsnorm
                .cmd(&self.context, self.command, &pipelines.rmsnorm_dispatch);
            compute_to_compute_barrier(&self.context, self.command);
            pipelines
                .to_f32
                .cmd(&self.context, self.command, &pipelines.to_f32_dispatch);
            compute_to_transfer_barrier(
                &self.context,
                self.command,
                arena.handle(),
                self.layout.status_offset,
                STATUS_BYTES,
            );
            self.context.device.cmd_copy_buffer(
                self.command,
                arena.handle(),
                status_readback.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(self.layout.status_offset)
                    .size(STATUS_BYTES)],
            );
            self.context.device.end_command_buffer(self.command)?;
            let commands = [self.command];
            let signals = [self.producer_timeline];
            let signal_values = [PRODUCER_TIMELINE_VALUE];
            let mut timeline_info =
                vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
            let submit = vk::SubmitInfo::default()
                .push_next(&mut timeline_info)
                .command_buffers(&commands)
                .signal_semaphores(&signals);
            self.context.device.queue_submit(
                self.context.q_graphics,
                &[submit],
                vk::Fence::null(),
            )?;
        }
        self.set_phase(OwnerPhase::Submitted);
        Ok(())
    }

    fn wait_for_producer(&self) -> Result<()> {
        let semaphores = [self.producer_timeline];
        let values = [PRODUCER_TIMELINE_VALUE];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe {
            self.context
                .device
                .wait_semaphores(&info, PRODUCER_WAIT_TIMEOUT_NS)?;
        }
        Ok(())
    }

    fn read_status_after_timeline(&self, expected_value: u64) -> Result<()> {
        if expected_value != PRODUCER_TIMELINE_VALUE {
            bail!("production terminal producer timeline value 漂移");
        }
        let observed = unsafe {
            self.context
                .device
                .get_semaphore_counter_value(self.producer_timeline)?
        };
        if observed < expected_value {
            bail!("production terminal producer timeline 尚未完成");
        }
        let readback = self
            .status_readback
            .as_ref()
            .context("production terminal status readback 已销毁")?;
        if readback.mapped().is_null() {
            bail!("production terminal status readback 未映射");
        }
        let code = unsafe { readback.mapped().cast::<u32>().read_unaligned() };
        if code != 0 {
            bail!("production terminal HC/norm sticky status=0x{code:08x}");
        }
        let mut phase = self.lock_phase()?;
        if *phase != OwnerPhase::Submitted {
            bail!("production terminal status 验收 phase 漂移");
        }
        *phase = OwnerPhase::Validated;
        Ok(())
    }

    fn lock_phase(&self) -> Result<std::sync::MutexGuard<'_, OwnerPhase>> {
        self.phase
            .lock()
            .map_err(|_| anyhow::anyhow!("production terminal owner lifecycle poisoned"))
    }

    fn set_phase(&self, next: OwnerPhase) {
        if let Ok(mut phase) = self.phase.lock() {
            *phase = next;
        }
    }
}

impl S14CausalBlockTerminalResourceOwner for S14CausalBlockProductionTerminalResourceOwner {
    fn buffer(&self, resource: S14CausalBlockTerminalResource) -> Option<&GpuBuffer> {
        match resource {
            S14CausalBlockTerminalResource::FinalHidden => Some(self.final_hidden.buffer.as_ref()),
            S14CausalBlockTerminalResource::NormalizedHeadRows => self.arena.as_ref(),
            S14CausalBlockTerminalResource::CandidateCheckpoint(lane) => {
                (lane < self.block_size).then(|| self.prefix_checkpoint_arena.buffer().as_ref())
            }
            S14CausalBlockTerminalResource::HeadBank(bank) => {
                self.paged_arena.head_chunk(bank).ok()
            }
        }
    }

    fn producer_timeline(&self) -> vk::Semaphore {
        self.producer_timeline
    }

    fn paged_weight_arena(&self) -> Option<&Arc<S14Position0PagedWeightArena>> {
        Some(&self.paged_arena)
    }

    unsafe fn record_next_head_chunk_copy(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        chunk: u32,
    ) -> Result<S14Position0HeadChunkReceipt, String> {
        if !std::ptr::eq(ctx, self.context.as_ref()) {
            return Err("production terminal head copy VulkanContext 漂移".into());
        }
        let mut state = self
            .head_upload
            .lock()
            .map_err(|_| "production terminal head uploader poisoned".to_owned())?;
        if chunk == 0 {
            state
                .uploader
                .begin_causal_block_terminal_head_stream(
                    &self.head_weight_plan,
                    &self.head_upload_lease,
                )
                .map_err(|error| {
                    format!(
                        "production terminal head uploader 无法进入 lease-bound head 流: {error:#}"
                    )
                })?;
        }
        let receipt = state
            .uploader
            .record_next_head_chunk_copy_for_causal_block(
                &self.head_upload_lease,
                ctx,
                command,
                &self.head_manifest,
                &self.head_weight_plan,
                self.paged_arena.as_ref(),
            )
            .map_err(|error| error.to_string())?;
        if receipt.chunk != u64::from(chunk) || receipt.bank != chunk as usize % 2 {
            return Err("production terminal head copy receipt chunk/bank 漂移".into());
        }
        Ok(receipt)
    }

    fn stage_recorded_head_chunk(
        &self,
        receipt: S14Position0HeadChunkReceipt,
        timeline_bank: usize,
    ) -> Result<S14Position0HeadChunkReceipt, String> {
        let mut state = self
            .head_upload
            .lock()
            .map_err(|_| "production terminal head uploader poisoned".to_owned())?;
        let S14CausalBlockTerminalHeadUploadState { uploader, store } = &mut *state;
        uploader
            .stage_recorded_head_chunk_for_causal_block(
                &self.head_upload_lease,
                &self.head_manifest,
                &self.head_weight_plan,
                store,
                receipt,
                timeline_bank,
            )
            .map_err(|error| error.to_string())
    }

    fn abort_head_stream_after_drain(&self) {
        if let Ok(mut state) = self.head_upload.lock() {
            let _ = state.uploader.abort_causal_block_lease_after_drain(
                &self.head_weight_plan,
                &self.head_upload_lease,
            );
        }
    }

    fn validate_after_producer_timeline(&self, expected_value: u64) -> Result<(), String> {
        self.read_status_after_timeline(expected_value)
            .map_err(|error| error.to_string())?;
        let state = self
            .head_upload
            .lock()
            .map_err(|_| "production terminal head uploader poisoned".to_owned())?;
        state
            .uploader
            .validate_causal_block_terminal_complete(
                &self.head_weight_plan,
                &self.head_upload_lease,
            )
            .map_err(|error| format!("production terminal lease 完成态未闭合: {error:#}"))
    }
}

impl Drop for S14CausalBlockProductionTerminalResourceOwner {
    fn drop(&mut self) {
        let needs_wait = self
            .phase
            .get_mut()
            .is_ok_and(|phase| *phase == OwnerPhase::Submitted);
        if needs_wait {
            let _ = self.wait_for_producer();
        }
        if let Some(pipelines) = self.pipelines.take() {
            pipelines.destroy(&self.context);
        }
        if let Some(readback) = self.status_readback.take() {
            readback.destroy(&self.context);
        }
        if let Some(arena) = self.arena.take() {
            arena.destroy(&self.context);
        }
        unsafe {
            if self.producer_timeline != vk::Semaphore::null() {
                self.context
                    .device
                    .destroy_semaphore(self.producer_timeline, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.context
                    .device
                    .destroy_command_pool(self.command_pool, None);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_pipelines(
    ctx: &VulkanContext,
    block_size: usize,
    final_hidden: &S14CausalBlockOwnedBufferSlice,
    terminal_static: S14CausalBlockTerminalStaticSlices<'_>,
    arena: &GpuBuffer,
    layout: S14CausalBlockTerminalArenaLayout,
) -> Result<TerminalPipelines> {
    let final_hc = S14FinalHcHeadPipeline::new(ctx)?;
    let shape = S14FinalHcHeadShape::production();
    let hidden_lane_bytes = hidden_bytes(1)?;
    let output_lane_bytes = shape.output_bf16_bytes();
    let aux_lane_bytes = shape.aux_f32_bytes();
    let mut final_hc_dispatches = Vec::with_capacity(block_size);
    for lane in 0..block_size {
        let dispatch = final_hc.bind_with_offsets(
            ctx,
            shape,
            S14FinalHcHeadBindings {
                hidden: S14FinalHcHeadBufferSlice::new(
                    final_hidden.buffer.as_ref(),
                    final_hidden.offset + hidden_lane_bytes * lane as u64,
                ),
                hc_head_fn: storage_hc_slice(terminal_static.hc_head_fn),
                hc_head_scale: storage_hc_slice(terminal_static.hc_head_scale),
                hc_head_base: storage_hc_slice(terminal_static.hc_head_base),
                output: S14FinalHcHeadBufferSlice::new(
                    arena,
                    layout.final_hidden_bf16_offset + output_lane_bytes * lane as u64,
                ),
                aux: S14FinalHcHeadBufferSlice::new(
                    arena,
                    layout.hc_aux_offset + aux_lane_bytes * lane as u64,
                ),
                status: S14FinalHcHeadBufferSlice::new(arena, layout.status_offset),
            },
        );
        match dispatch {
            Ok(dispatch) => final_hc_dispatches.push(dispatch),
            Err(error) => {
                for dispatch in final_hc_dispatches {
                    dispatch.binder.destroy(ctx);
                }
                final_hc.destroy(ctx);
                return Err(error.context("bind production terminal K-row final HC"));
            }
        }
    }

    let rmsnorm = match S14Bf16RmsNormPipeline::new(ctx) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            for dispatch in final_hc_dispatches {
                dispatch.binder.destroy(ctx);
            }
            final_hc.destroy(ctx);
            return Err(error.context("create production terminal RMSNorm pipeline"));
        }
    };
    let rms_shape =
        S14Bf16RmsNormShape::new(block_size as u32, S14_CAUSAL_BLOCK_STREAM_WIDTH as u32)?;
    let rmsnorm_dispatch = match rmsnorm.bind_slices(
        ctx,
        rms_shape,
        FINAL_RMS_EPSILON,
        StorageBufferSlice {
            buffer: arena,
            offset: layout.final_hidden_bf16_offset,
        },
        terminal_static.norm_weight,
        StorageBufferSlice {
            buffer: arena,
            offset: layout.inverse_rms_offset,
        },
        StorageBufferSlice {
            buffer: arena,
            offset: layout.normalized_bf16_offset,
        },
        StorageBufferSlice {
            buffer: arena,
            offset: layout.status_offset,
        },
    ) {
        Ok(dispatch) => dispatch,
        Err(error) => {
            rmsnorm.destroy(ctx);
            for dispatch in final_hc_dispatches {
                dispatch.binder.destroy(ctx);
            }
            final_hc.destroy(ctx);
            return Err(error.context("bind production terminal K-row RMSNorm"));
        }
    };
    let to_f32 = match S14Bf16ToF32Pipeline::new(ctx) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            rmsnorm_dispatch.binder.destroy(ctx);
            rmsnorm.destroy(ctx);
            for dispatch in final_hc_dispatches {
                dispatch.binder.destroy(ctx);
            }
            final_hc.destroy(ctx);
            return Err(error.context("create production terminal BF16-to-F32 pipeline"));
        }
    };
    let scalars = u32::try_from(
        block_size
            .checked_mul(S14_CAUSAL_BLOCK_STREAM_WIDTH)
            .context("terminal normalized scalar count overflow")?,
    )?;
    let to_f32_dispatch = match to_f32.bind_slices(
        ctx,
        S14Bf16ToF32Shape::new(scalars)?,
        StorageBufferSlice {
            buffer: arena,
            offset: layout.normalized_bf16_offset,
        },
        StorageBufferSlice {
            buffer: arena,
            offset: layout.normalized_f32_offset,
        },
        StorageBufferSlice {
            buffer: arena,
            offset: layout.status_offset,
        },
    ) {
        Ok(dispatch) => dispatch,
        Err(error) => {
            to_f32.destroy(ctx);
            rmsnorm_dispatch.binder.destroy(ctx);
            rmsnorm.destroy(ctx);
            for dispatch in final_hc_dispatches {
                dispatch.binder.destroy(ctx);
            }
            final_hc.destroy(ctx);
            return Err(error.context("bind production terminal BF16-to-F32"));
        }
    };
    Ok(TerminalPipelines {
        final_hc,
        final_hc_dispatches,
        rmsnorm,
        rmsnorm_dispatch,
        to_f32,
        to_f32_dispatch,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_external_resources(
    block_size: usize,
    final_hidden: &S14CausalBlockOwnedBufferSlice,
    terminal_static: S14CausalBlockTerminalStaticSlices<'_>,
    prefix_checkpoint_arena: &Arc<S14CausalBlockPrefixCheckpointArena>,
    paged_arena: &S14Position0PagedWeightArena,
    head_manifest: &Position0WholeTokenManifest,
    head_weight_plan: &S14Position0HybridWeightPlan,
) -> Result<()> {
    head_weight_plan.validate(head_manifest)?;
    let checkpoint_layout = prefix_checkpoint_arena.layout();
    if checkpoint_layout.checkpoint_state_bytes == 0
        || checkpoint_layout.block_size != block_size
        || prefix_checkpoint_arena.buffer().handle() == vk::Buffer::null()
    {
        bail!("production terminal prefix checkpoint arena K/state/handle 非法");
    }
    let final_shape = S14FinalHcHeadShape::production();
    validate_slice(final_hidden, hidden_bytes(block_size)?, "final hidden")?;
    for (slice, bytes, label) in [
        (
            terminal_static.hc_head_fn,
            final_shape.hc_head_fn_f32_bytes(),
            "hc_head_fn",
        ),
        (
            terminal_static.hc_head_scale,
            final_shape.hc_head_scale_f32_bytes(),
            "hc_head_scale",
        ),
        (
            terminal_static.hc_head_base,
            final_shape.hc_head_base_f32_bytes(),
            "hc_head_base",
        ),
        (
            terminal_static.norm_weight,
            S14_CAUSAL_BLOCK_STREAM_WIDTH as u64 * 2,
            "norm.weight",
        ),
    ] {
        validate_storage_slice(slice, bytes, label)?;
    }
    let head_shape = S14HeadChunkArgmaxShape::production_batched(block_size as u32)?;
    if head_weight_plan.head_chunk_count != u64::from(S14_HEAD_CHUNK_COUNT)
        || head_weight_plan.head_chunk_bytes != head_shape.max_chunk_weight_bytes()?
        || paged_arena.plan().physical.head_chunk_bytes != head_weight_plan.head_chunk_bytes
    {
        bail!("production terminal paged head plan/chunk count/capacity 漂移");
    }
    for bank in 0..2 {
        let buffer = paged_arena.head_chunk(bank)?;
        if buffer.handle() == vk::Buffer::null()
            || buffer.size() != head_weight_plan.head_chunk_bytes
        {
            bail!("production terminal paged head bank {bank} 不完整");
        }
    }
    if paged_arena.head_chunk(0)?.handle() == paged_arena.head_chunk(1)?.handle() {
        bail!("production terminal paged head 双 bank 发生别名");
    }
    Ok(())
}

fn validate_checkpoint_slices(
    block_size: usize,
    checkpoint_state_bytes: u64,
    checkpoints: &[S14CausalBlockOwnedBufferSlice],
) -> Result<()> {
    if checkpoint_state_bytes == 0 || checkpoints.len() != block_size {
        bail!("production terminal checkpoint 数量或 state bytes 非法");
    }
    let checkpoint_handle = checkpoints[0].buffer.handle();
    let mut checkpoint_ranges = Vec::with_capacity(block_size);
    for checkpoint in checkpoints {
        if checkpoint.buffer.handle() != checkpoint_handle {
            bail!("production terminal K份 checkpoint 必须属于同一 device arena");
        }
        validate_slice(checkpoint, checkpoint_state_bytes, "candidate checkpoint")?;
        checkpoint_ranges.push((
            checkpoint.offset,
            checkpoint.offset + checkpoint_state_bytes,
        ));
    }
    validate_non_overlapping(&checkpoint_ranges, "candidate checkpoint")
}

fn validate_slice(slice: &S14CausalBlockOwnedBufferSlice, bytes: u64, label: &str) -> Result<()> {
    if slice.buffer.handle() == vk::Buffer::null()
        || bytes == 0
        || slice.offset % 4 != 0
        || slice
            .offset
            .checked_add(bytes)
            .is_none_or(|end| end > slice.buffer.size())
    {
        bail!("production terminal {label} owner range 越界/未对齐");
    }
    Ok(())
}

fn validate_non_overlapping(ranges: &[(u64, u64)], label: &str) -> Result<()> {
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("production terminal {label} ranges 重叠");
            }
        }
    }
    Ok(())
}

fn host_candidates_identity_would_be_invalid(
    block_size: usize,
    _base_position: u32,
    routes_by_position: &[Vec<RouteDecision>],
) -> bool {
    !matches!(block_size, 4 | 8) || routes_by_position.len() != block_size
}

#[derive(Clone, Copy)]
struct S14CausalBlockTerminalStaticSlices<'a> {
    hc_head_fn: StorageBufferSlice<'a>,
    hc_head_scale: StorageBufferSlice<'a>,
    hc_head_base: StorageBufferSlice<'a>,
    norm_weight: StorageBufferSlice<'a>,
}

fn resolve_terminal_static_slices(
    paged_arena: &S14Position0PagedWeightArena,
) -> Result<S14CausalBlockTerminalStaticSlices<'_>> {
    fn resident<'a>(
        paged_arena: &'a S14Position0PagedWeightArena,
        tensor: &str,
    ) -> Result<StorageBufferSlice<'a>> {
        let binding = paged_arena.static_asset(tensor)?;
        if !binding.resident_once || binding.layer.is_some() || binding.bank.is_some() {
            bail!("production terminal {tensor} 必须来自 paged resident-small arena");
        }
        Ok(StorageBufferSlice {
            buffer: binding.buffer,
            offset: binding.destination_offset,
        })
    }
    Ok(S14CausalBlockTerminalStaticSlices {
        hc_head_fn: resident(paged_arena, "hc_head_fn")?,
        hc_head_scale: resident(paged_arena, "hc_head_scale")?,
        hc_head_base: resident(paged_arena, "hc_head_base")?,
        norm_weight: resident(paged_arena, "norm.weight")?,
    })
}

fn storage_hc_slice(slice: StorageBufferSlice<'_>) -> S14FinalHcHeadBufferSlice<'_> {
    S14FinalHcHeadBufferSlice::new(slice.buffer, slice.offset)
}

fn validate_storage_slice(slice: StorageBufferSlice<'_>, bytes: u64, label: &str) -> Result<()> {
    if slice.buffer.handle() == vk::Buffer::null()
        || bytes == 0
        || slice.offset % 4 != 0
        || slice
            .offset
            .checked_add(bytes)
            .is_none_or(|end| end > slice.buffer.size())
    {
        bail!("production terminal {label} paged slice 越界/未对齐");
    }
    Ok(())
}

fn hidden_bytes(block_size: usize) -> Result<u64> {
    (block_size as u64)
        .checked_mul(S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64)
        .and_then(|elements| elements.checked_mul(2))
        .context("production terminal hidden bytes overflow")
}

fn allocate_command(ctx: &VulkanContext) -> Result<(vk::CommandPool, vk::CommandBuffer)> {
    let pool = unsafe {
        ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?
    };
    match unsafe {
        ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    } {
        Ok(commands) => Ok((pool, commands[0])),
        Err(error) => {
            unsafe { ctx.device.destroy_command_pool(pool, None) };
            Err(error.into())
        }
    }
}

fn create_timeline(ctx: &VulkanContext) -> Result<vk::Semaphore> {
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    Ok(unsafe {
        ctx.device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut type_info),
            None,
        )?
    })
}

unsafe fn transfer_to_compute_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

unsafe fn compute_to_compute_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

unsafe fn compute_to_transfer_barrier(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    bytes: u64,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .buffer(buffer)
        .offset(offset)
        .size(bytes);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[barrier],
        &[],
    );
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|expanded| expanded & !(alignment - 1))
        .context("production terminal alignment overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_arena_has_exact_k4_k8_row_capacity() {
        let k4 = S14CausalBlockTerminalArenaLayout::build(4, 256).unwrap();
        let k8 = S14CausalBlockTerminalArenaLayout::build(8, 256).unwrap();
        assert_eq!(k4.final_hidden_bf16_offset, 0);
        assert_eq!(k4.normalized_bf16_offset, 32_768);
        assert_eq!(k4.normalized_f32_offset, 65_536);
        assert!(k4.inverse_rms_offset >= 131_072);
        assert!(k4.arena_bytes < k8.arena_bytes);
        assert_eq!(k8.normalized_f32_offset, 131_072);
        assert!(k8.arena_bytes >= 262_144);
    }

    #[test]
    fn terminal_arena_rejects_non_production_k_and_alignment() {
        for block_size in [0, 1, 6, 9] {
            assert!(S14CausalBlockTerminalArenaLayout::build(block_size, 256).is_err());
        }
        for alignment in [0, 3, 192] {
            assert!(S14CausalBlockTerminalArenaLayout::build(4, alignment).is_err());
        }
    }

    #[test]
    fn publication_receipt_cannot_claim_prebuilt_predictions() {
        let receipt = S14CausalBlockTerminalPublishReceipt {
            base_position: 7,
            block_size: 4,
            completed_layers: FULL_DEPTH_LAYERS.len(),
            producer_timeline_value: 1,
            normalized_head_rows_offset: 65_536,
            checkpoint_count: 4,
            head_chunk_count: 32,
            predicted_tokens_prebuilt: false,
        };
        assert!(!receipt.predicted_tokens_prebuilt);
        assert_eq!(receipt.head_chunk_count, S14_HEAD_CHUNK_COUNT as usize);
    }
}
