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
    s14_bf16_rmsnorm::{
        S14Bf16RmsNormDispatch, S14Bf16RmsNormPipeline, S14Bf16RmsNormShape,
    },
    s14_bf16_to_f32::{
        S14Bf16ToF32Dispatch, S14Bf16ToF32Pipeline, S14Bf16ToF32Shape,
    },
    s14_causal_block_layer::{
        S14CausalBlockHiddenBinding, S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE,
        S14_CAUSAL_BLOCK_STREAM_WIDTH,
    },
    s14_causal_block_production_bundle::{
        S14CausalBlockContextBound, S14CausalBlockProductionTerminalPublisher,
    },
    s14_causal_block_terminal_adapter::{
        S14CausalBlockHostCandidateFinalizer, S14CausalBlockTerminalProductionSource,
        S14CausalBlockTerminalResource, S14CausalBlockTerminalResourceOwner,
    },
    s14_final_hc_head::{
        S14FinalHcHeadBindings, S14FinalHcHeadBufferSlice, S14FinalHcHeadDispatch,
        S14FinalHcHeadPipeline, S14FinalHcHeadShape,
    },
    s14_head_chunk_argmax::{S14HeadChunkArgmaxShape, S14_HEAD_CHUNK_COUNT},
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{RouteDecision, FULL_DEPTH_LAYERS};
use std::{fmt, sync::{Arc, Mutex}};

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

    fn storage_slice(&self) -> StorageBufferSlice<'_> {
        StorageBufferSlice {
            buffer: self.buffer.as_ref(),
            offset: self.offset,
        }
    }
}

/// 外部 FullDepth43 producer 必须一次性提供的强 owner 输入。checkpoint slices 必须
/// 全部属于同一 arena；head chunks 可属于一个完整 head arena 的不同 offset，也可由32个
/// 独立 owner 持有。所有 Arc 只保活真实 allocation，不从裸 handle 重建资源。
pub struct S14CausalBlockTerminalResourceOwnerInputs {
    pub context: Arc<VulkanContext>,
    pub block_size: usize,
    pub final_hidden: S14CausalBlockOwnedBufferSlice,
    pub final_hc_head_fn: S14CausalBlockOwnedBufferSlice,
    pub final_hc_head_scale: S14CausalBlockOwnedBufferSlice,
    pub final_hc_head_base: S14CausalBlockOwnedBufferSlice,
    pub final_norm_weight: S14CausalBlockOwnedBufferSlice,
    pub checkpoint_state_bytes: u64,
    pub checkpoints: Vec<S14CausalBlockOwnedBufferSlice>,
    pub head_chunks: Vec<S14CausalBlockOwnedBufferSlice>,
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
    checkpoint_state_bytes: u64,
    checkpoints: Vec<S14CausalBlockOwnedBufferSlice>,
    head_chunks: Vec<S14CausalBlockOwnedBufferSlice>,
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
            .field("checkpoint_state_bytes", &self.checkpoint_state_bytes)
            .field("checkpoint_count", &self.checkpoints.len())
            .field("head_chunk_count", &self.head_chunks.len())
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
            final_hc_head_fn,
            final_hc_head_scale,
            final_hc_head_base,
            final_norm_weight,
            checkpoint_state_bytes,
            checkpoints,
            head_chunks,
        } = inputs;
        if !context.timeline_semaphore || !matches!(block_size, 4 | 8) {
            bail!("production terminal owner 要求 timeline semaphore 与 K=4/8");
        }
        validate_external_resources(
            block_size,
            &final_hidden,
            &final_hc_head_fn,
            &final_hc_head_scale,
            &final_hc_head_base,
            &final_norm_weight,
            checkpoint_state_bytes,
            &checkpoints,
            &head_chunks,
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
            &final_hc_head_fn,
            &final_hc_head_scale,
            &final_hc_head_base,
            &final_norm_weight,
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
            checkpoint_state_bytes,
            checkpoints,
            head_chunks,
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
        self.checkpoint_state_bytes
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
        self.validate_publication_identity(base_position, final_hidden, &routes_by_position)
            .map_err(|error| error.to_string())?;
        if let Err(error) = self.record_and_submit_prelude() {
            self.set_phase(OwnerPhase::Poisoned);
            return Err(format!("record production terminal HC/norm 失败: {error:#}"));
        }
        let resources: Arc<dyn S14CausalBlockTerminalResourceOwner> = self.clone();
        let source = S14CausalBlockTerminalProductionSource {
            completed_layers: FULL_DEPTH_LAYERS.len(),
            base_position,
            final_hidden,
            normalized_head_rows_offset: self.layout.normalized_f32_offset,
            checkpoint_offsets: self.checkpoints.iter().map(|slice| slice.offset).collect(),
            head_chunk_offsets: self.head_chunks.iter().map(|slice| slice.offset).collect(),
            producer_timeline_value: PRODUCER_TIMELINE_VALUE,
            routes_by_position,
            host_candidates,
            resources,
        };
        if let Err(error) = publisher.publish(S14CausalBlockContextBound::new(
            Arc::clone(&self.context),
            source,
        )) {
            let drain = self.wait_for_producer();
            self.set_phase(if drain.is_ok() {
                OwnerPhase::DrainedAfterReject
            } else {
                OwnerPhase::Poisoned
            });
            return match drain {
                Ok(()) => Err(error),
                Err(drain_error) => Err(format!(
                    "terminal source publish 失败: {error}; producer drain 失败: {drain_error:#}"
                )),
            };
        }
        Ok(S14CausalBlockTerminalPublishReceipt {
            base_position,
            block_size: self.block_size,
            completed_layers: FULL_DEPTH_LAYERS.len(),
            producer_timeline_value: PRODUCER_TIMELINE_VALUE,
            normalized_head_rows_offset: self.layout.normalized_f32_offset,
            checkpoint_count: self.checkpoints.len(),
            head_chunk_count: self.head_chunks.len(),
            predicted_tokens_prebuilt: false,
        })
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
        if final_hidden.buffer != self.final_hidden.buffer.handle()
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
        let arena = self.arena.as_ref().context("production terminal arena 已销毁")?;
        let status_readback = self
            .status_readback
            .as_ref()
            .context("production terminal status readback 已销毁")?;
        let pipelines = self
            .pipelines
            .as_ref()
            .context("production terminal pipelines 已销毁")?;
        unsafe {
            self.context.device.reset_command_pool(
                self.command_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;
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
                pipelines.final_hc.cmd(&self.context, self.command, dispatch);
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
            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .signal_semaphore_values(&signal_values);
            let submit = vk::SubmitInfo::default()
                .push_next(&mut timeline_info)
                .command_buffers(&commands)
                .signal_semaphores(&signals);
            self.context
                .device
                .queue_submit(self.context.q_graphics, &[submit], vk::Fence::null())?;
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
            S14CausalBlockTerminalResource::CandidateCheckpoint(lane) => self
                .checkpoints
                .get(lane)
                .map(|slice| slice.buffer.as_ref()),
            S14CausalBlockTerminalResource::HeadChunk(chunk) => self
                .head_chunks
                .get(chunk)
                .map(|slice| slice.buffer.as_ref()),
        }
    }

    fn producer_timeline(&self) -> vk::Semaphore {
        self.producer_timeline
    }

    fn validate_after_producer_timeline(&self, expected_value: u64) -> Result<(), String> {
        self.read_status_after_timeline(expected_value)
            .map_err(|error| error.to_string())
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
    hc_head_fn: &S14CausalBlockOwnedBufferSlice,
    hc_head_scale: &S14CausalBlockOwnedBufferSlice,
    hc_head_base: &S14CausalBlockOwnedBufferSlice,
    norm_weight: &S14CausalBlockOwnedBufferSlice,
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
                hc_head_fn: owned_hc_slice(hc_head_fn),
                hc_head_scale: owned_hc_slice(hc_head_scale),
                hc_head_base: owned_hc_slice(hc_head_base),
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
    let rms_shape = S14Bf16RmsNormShape::new(
        block_size as u32,
        S14_CAUSAL_BLOCK_STREAM_WIDTH as u32,
    )?;
    let rmsnorm_dispatch = match rmsnorm.bind_slices(
        ctx,
        rms_shape,
        FINAL_RMS_EPSILON,
        StorageBufferSlice { buffer: arena, offset: layout.final_hidden_bf16_offset },
        norm_weight.storage_slice(),
        StorageBufferSlice { buffer: arena, offset: layout.inverse_rms_offset },
        StorageBufferSlice { buffer: arena, offset: layout.normalized_bf16_offset },
        StorageBufferSlice { buffer: arena, offset: layout.status_offset },
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
        StorageBufferSlice { buffer: arena, offset: layout.normalized_bf16_offset },
        StorageBufferSlice { buffer: arena, offset: layout.normalized_f32_offset },
        StorageBufferSlice { buffer: arena, offset: layout.status_offset },
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
    hc_head_fn: &S14CausalBlockOwnedBufferSlice,
    hc_head_scale: &S14CausalBlockOwnedBufferSlice,
    hc_head_base: &S14CausalBlockOwnedBufferSlice,
    norm_weight: &S14CausalBlockOwnedBufferSlice,
    checkpoint_state_bytes: u64,
    checkpoints: &[S14CausalBlockOwnedBufferSlice],
    head_chunks: &[S14CausalBlockOwnedBufferSlice],
) -> Result<()> {
    if checkpoint_state_bytes == 0
        || checkpoints.len() != block_size
        || head_chunks.len() != S14_HEAD_CHUNK_COUNT as usize
    {
        bail!("production terminal checkpoint/head chunk 数量或 state bytes 非法");
    }
    let final_shape = S14FinalHcHeadShape::production();
    for (slice, bytes, label) in [
        (final_hidden, hidden_bytes(block_size)?, "final hidden"),
        (hc_head_fn, final_shape.hc_head_fn_f32_bytes(), "hc_head_fn"),
        (hc_head_scale, final_shape.hc_head_scale_f32_bytes(), "hc_head_scale"),
        (hc_head_base, final_shape.hc_head_base_f32_bytes(), "hc_head_base"),
        (norm_weight, S14_CAUSAL_BLOCK_STREAM_WIDTH as u64 * 2, "norm.weight"),
    ] {
        validate_slice(slice, bytes, label)?;
    }
    let checkpoint_handle = checkpoints[0].buffer.handle();
    let mut checkpoint_ranges = Vec::with_capacity(block_size);
    for checkpoint in checkpoints {
        if checkpoint.buffer.handle() != checkpoint_handle {
            bail!("production terminal K份 checkpoint 必须属于同一 device arena");
        }
        validate_slice(checkpoint, checkpoint_state_bytes, "candidate checkpoint")?;
        checkpoint_ranges.push((checkpoint.offset, checkpoint.offset + checkpoint_state_bytes));
    }
    validate_non_overlapping(&checkpoint_ranges, "candidate checkpoint")?;
    let head_shape = S14HeadChunkArgmaxShape::production_batched(block_size as u32)?;
    for (chunk, slice) in head_chunks.iter().enumerate() {
        validate_slice(
            slice,
            head_shape.chunk(chunk as u32)?.weight_bytes(head_shape)?,
            "head chunk",
        )?;
    }
    Ok(())
}

fn validate_slice(
    slice: &S14CausalBlockOwnedBufferSlice,
    bytes: u64,
    label: &str,
) -> Result<()> {
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

fn owned_hc_slice(slice: &S14CausalBlockOwnedBufferSlice) -> S14FinalHcHeadBufferSlice<'_> {
    S14FinalHcHeadBufferSlice::new(slice.buffer.as_ref(), slice.offset)
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
