//! FullDepth43 position0 的 L42 后终端生产编排。
//!
//! 顺序固定为：final HC → final RMSNorm → BF16/F32 边界 → 双 bank 32 个
//! BF16 head chunk → 跨 chunk GPU argmax → terminal readback → 唯一 token host
//! wait → DecoderState 克隆提交。这里不执行层后端，也不读取 capture/fixture；输入
//! hidden、权重页和 candidate state 必须由真实上游提供。

use crate::{
    compute::StorageBufferSlice,
    s14_bf16_rmsnorm::{S14Bf16RmsNormDispatch, S14Bf16RmsNormPipeline, S14Bf16RmsNormShape},
    s14_bf16_to_f32::{S14Bf16ToF32Dispatch, S14Bf16ToF32Pipeline, S14Bf16ToF32Shape},
    s14_final_hc_head::{
        S14FinalHcHeadBindings, S14FinalHcHeadBufferSlice, S14FinalHcHeadDispatch,
        S14FinalHcHeadPipeline, S14FinalHcHeadShape,
    },
    s14_head_chunk_argmax::{
        decode_head_argmax, S14HeadArgmaxResult, S14HeadChunkArgmaxDispatch,
        S14HeadChunkArgmaxPipeline, S14HeadChunkArgmaxRecorder, S14HeadChunkArgmaxRecordingReceipt,
        S14HeadChunkArgmaxShape, S14HeadChunkWorkspace, S14_HEAD_ARGMAX_BYTES,
        S14_HEAD_ARGMAX_WORDS, S14_HEAD_CHUNK_COUNT,
    },
    s14_position0_paged_layer_timeline::{
        S14Position0PagedCandidateReceipt, S14Position0PagedDrainReceipt,
        S14Position0PagedLayerTimeline,
    },
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_position0_state_writeback::S14Position0StateReadback,
    s14_position0_workspace::S14Position0WorkspaceSlot,
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{DecoderStateV1, TokenRecord, WholeTokenCandidate, FULL_DEPTH_LAYERS};
use sha2::{Digest, Sha256};

pub const S14_POSITION0_FINAL_HIDDEN: u32 = 4096;
pub const S14_POSITION0_FINAL_HC_STREAMS: u32 = 4;
pub const S14_POSITION0_FINAL_HC_ELEMENTS: usize =
    S14_POSITION0_FINAL_HC_STREAMS as usize * S14_POSITION0_FINAL_HIDDEN as usize;
pub const S14_POSITION0_FINAL_NORMALIZED_ELEMENTS: usize = S14_POSITION0_FINAL_HIDDEN as usize;
pub const S14_POSITION0_FINAL_RMS_EPSILON: f32 = 1.0e-6;

const FINAL_HC_FN_BYTES: u64 = 4 * 16_384 * 4;
const FINAL_HC_SCALE_BYTES: u64 = 4;
const FINAL_HC_BASE_BYTES: u64 = 4 * 4;
const FINAL_NORM_WEIGHT_BYTES: u64 = 4096 * 2;
const FINAL_HC_STREAM_BYTES: u64 = S14_POSITION0_FINAL_HC_ELEMENTS as u64 * 2;
const FINAL_NORMALIZED_F32_BYTES: u64 = S14_POSITION0_FINAL_NORMALIZED_ELEMENTS as u64 * 4;
const TERMINAL_STATUS_BYTES: u64 = 4;
const TERMINAL_KNOWN_STATUS_BITS: u32 = 0x0f;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalPhase {
    Bound,
    PreludeRecorded,
    PreludeSubmitted,
    HeadComputeRecorded { chunk: u32 },
    HeadSubmitting { chunk: u32 },
    HeadComplete,
    TerminalRecorded,
    TerminalSubmitted,
    Completed,
    FailedAfterWait,
    Poisoned,
    Drained,
}

#[derive(Clone, Copy)]
struct TerminalWorkspace<'a> {
    final_hidden_streams: StorageBufferSlice<'a>,
    final_hidden_bf16: StorageBufferSlice<'a>,
    final_normalized_bf16: StorageBufferSlice<'a>,
    final_normalized_f32: StorageBufferSlice<'a>,
    inverse_rms: StorageBufferSlice<'a>,
    hc_aux: StorageBufferSlice<'a>,
    status: StorageBufferSlice<'a>,
    head_logits: StorageBufferSlice<'a>,
    head_argmax: StorageBufferSlice<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TerminalReadbackLayout {
    hc_streams: u64,
    normalized_f32: u64,
    status: u64,
    argmax: u64,
    bytes: u64,
}

impl TerminalReadbackLayout {
    const fn production() -> Self {
        let hc_streams = 0;
        let normalized_f32 = hc_streams + FINAL_HC_STREAM_BYTES;
        let status = normalized_f32 + FINAL_NORMALIZED_F32_BYTES;
        let argmax = status + TERMINAL_STATUS_BYTES;
        let bytes = argmax + S14_HEAD_ARGMAX_BYTES;
        Self {
            hc_streams,
            normalized_f32,
            status,
            argmax,
            bytes,
        }
    }
}

struct S14Position0TerminalReadback {
    buffer: GpuBuffer,
    layout: TerminalReadbackLayout,
}

pub struct S14Position0TerminalChain<'a> {
    final_hc_pipeline: S14FinalHcHeadPipeline,
    final_hc_dispatch: S14FinalHcHeadDispatch,
    rmsnorm_pipeline: S14Bf16RmsNormPipeline,
    rmsnorm_dispatch: S14Bf16RmsNormDispatch,
    to_f32_pipeline: S14Bf16ToF32Pipeline,
    to_f32_dispatch: S14Bf16ToF32Dispatch,
    head_pipeline: S14HeadChunkArgmaxPipeline,
    head_dispatches: Vec<S14HeadChunkArgmaxDispatch<'a>>,
    head_recorder: S14HeadChunkArgmaxRecorder,
    head_recording_receipt: Option<S14HeadChunkArgmaxRecordingReceipt>,
    workspace: TerminalWorkspace<'a>,
    readback: S14Position0TerminalReadback,
    phase: TerminalPhase,
    next_head_chunk: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct S14Position0TerminalCompletion {
    pub base_epoch: u64,
    pub candidate_bank: usize,
    pub timeline: S14Position0PagedCandidateReceipt,
    pub predicted_token_id: u32,
    pub max_logit: f32,
    pub hc_streams_bf16: Vec<u16>,
    pub normalized_f32: Vec<f32>,
    pub normalized_f32_le_sha256: String,
}

/// 首枚固定 BOS token 使用的同步终端回执。它复用完全相同的 final
/// HC/RMSNorm/head/argmax 数学，但每个 head chunk 都显式等待；因此只能用于
/// 闭合首 token，不能冒充统一 timeline 的性能或唯一-wait 证据。
#[derive(Debug, Clone, PartialEq)]
pub struct S14Position0SynchronousTerminalCompletion {
    pub base_epoch: u64,
    pub candidate_bank: usize,
    pub predicted_token_id: u32,
    pub max_logit: f32,
    pub hc_streams_bf16: Vec<u16>,
    pub normalized_f32: Vec<f32>,
    pub normalized_f32_le_sha256: String,
    pub compute_host_waits: u32,
}

impl<'a> S14Position0TerminalChain<'a> {
    /// 只建立真实 buffer/offset descriptor，不提交 GPU 工作。`final_hidden_streams`
    /// 必须指向 L42 已写好的 candidate 四流 BF16 状态，且属于同一 position0 workspace。
    pub fn new(
        ctx: &VulkanContext,
        arena: &'a S14Position0PagedWeightArena,
        final_hidden_streams: StorageBufferSlice<'a>,
    ) -> Result<Self> {
        let workspace_buffer = arena.workspace();
        if final_hidden_streams.buffer.handle() != workspace_buffer.handle() {
            bail!("position0 terminal L42 hidden 必须属于同一 whole-token workspace");
        }
        let layout = arena.workspace_layout();
        let workspace = TerminalWorkspace {
            final_hidden_streams,
            final_hidden_bf16: slot_slice(
                workspace_buffer,
                layout,
                S14Position0WorkspaceSlot::FinalHiddenBf16,
            ),
            final_normalized_bf16: slot_slice(
                workspace_buffer,
                layout,
                S14Position0WorkspaceSlot::FinalNormalizedBf16,
            ),
            final_normalized_f32: slot_slice(
                workspace_buffer,
                layout,
                S14Position0WorkspaceSlot::FinalNormalizedF32,
            ),
            inverse_rms: slot_slice(
                workspace_buffer,
                layout,
                S14Position0WorkspaceSlot::HcInverseRms,
            ),
            hc_aux: slot_slice(workspace_buffer, layout, S14Position0WorkspaceSlot::HcAux),
            // L42 已封口后 router id 临时区不再被层图读取，可作为三段 final 原语的 sticky status。
            status: slot_slice(
                workspace_buffer,
                layout,
                S14Position0WorkspaceSlot::RouterIdsU32,
            ),
            head_logits: slot_slice(
                workspace_buffer,
                layout,
                S14Position0WorkspaceSlot::HeadChunkLogitsF32,
            ),
            head_argmax: slot_slice(
                workspace_buffer,
                layout,
                S14Position0WorkspaceSlot::HeadArgmax,
            ),
        };
        validate_terminal_workspace(workspace)?;

        let hc_head_fn = final_weight_slice(arena, "hc_head_fn", FINAL_HC_FN_BYTES)?;
        let hc_head_scale = final_weight_slice(arena, "hc_head_scale", FINAL_HC_SCALE_BYTES)?;
        let hc_head_base = final_weight_slice(arena, "hc_head_base", FINAL_HC_BASE_BYTES)?;
        let norm_weight = final_weight_slice(arena, "norm.weight", FINAL_NORM_WEIGHT_BYTES)?;

        let final_hc_pipeline = S14FinalHcHeadPipeline::new(ctx)?;
        let final_hc_dispatch = match final_hc_pipeline.bind_with_offsets(
            ctx,
            S14FinalHcHeadShape::production(),
            S14FinalHcHeadBindings {
                hidden: hc_slice(workspace.final_hidden_streams),
                hc_head_fn: hc_slice(hc_head_fn),
                hc_head_scale: hc_slice(hc_head_scale),
                hc_head_base: hc_slice(hc_head_base),
                output: hc_slice(workspace.final_hidden_bf16),
                aux: hc_slice(workspace.hc_aux),
                status: hc_slice(workspace.status),
            },
        ) {
            Ok(dispatch) => dispatch,
            Err(error) => {
                final_hc_pipeline.destroy(ctx);
                return Err(error).context("bind position0 final HC");
            }
        };

        let rmsnorm_pipeline = match S14Bf16RmsNormPipeline::new(ctx) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                final_hc_dispatch.binder.destroy(ctx);
                final_hc_pipeline.destroy(ctx);
                return Err(error).context("create position0 final RMSNorm pipeline");
            }
        };
        let rmsnorm_shape = S14Bf16RmsNormShape::new(1, S14_POSITION0_FINAL_HIDDEN)?;
        let rmsnorm_dispatch = match rmsnorm_pipeline.bind_slices(
            ctx,
            rmsnorm_shape,
            S14_POSITION0_FINAL_RMS_EPSILON,
            workspace.final_hidden_bf16,
            norm_weight,
            workspace.inverse_rms,
            workspace.final_normalized_bf16,
            workspace.status,
        ) {
            Ok(dispatch) => dispatch,
            Err(error) => {
                rmsnorm_pipeline.destroy(ctx);
                final_hc_dispatch.binder.destroy(ctx);
                final_hc_pipeline.destroy(ctx);
                return Err(error).context("bind position0 final RMSNorm");
            }
        };

        let to_f32_pipeline = match S14Bf16ToF32Pipeline::new(ctx) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                rmsnorm_dispatch.binder.destroy(ctx);
                rmsnorm_pipeline.destroy(ctx);
                final_hc_dispatch.binder.destroy(ctx);
                final_hc_pipeline.destroy(ctx);
                return Err(error).context("create final BF16-to-F32 pipeline");
            }
        };
        let to_f32_shape = S14Bf16ToF32Shape::new(S14_POSITION0_FINAL_HIDDEN)?;
        let to_f32_dispatch = match to_f32_pipeline.bind_slices(
            ctx,
            to_f32_shape,
            workspace.final_normalized_bf16,
            workspace.final_normalized_f32,
            workspace.status,
        ) {
            Ok(dispatch) => dispatch,
            Err(error) => {
                to_f32_pipeline.destroy(ctx);
                rmsnorm_dispatch.binder.destroy(ctx);
                rmsnorm_pipeline.destroy(ctx);
                final_hc_dispatch.binder.destroy(ctx);
                final_hc_pipeline.destroy(ctx);
                return Err(error).context("bind final BF16-to-F32");
            }
        };

        let head_pipeline = match S14HeadChunkArgmaxPipeline::new(ctx) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                to_f32_dispatch.binder.destroy(ctx);
                to_f32_pipeline.destroy(ctx);
                rmsnorm_dispatch.binder.destroy(ctx);
                rmsnorm_pipeline.destroy(ctx);
                final_hc_dispatch.binder.destroy(ctx);
                final_hc_pipeline.destroy(ctx);
                return Err(error).context("create final head pipeline");
            }
        };
        let head_shape = S14HeadChunkArgmaxShape::production();
        if arena.plan().physical.head_chunk_bytes != head_shape.max_chunk_weight_bytes()? {
            destroy_prelude(
                ctx,
                final_hc_pipeline,
                final_hc_dispatch,
                rmsnorm_pipeline,
                rmsnorm_dispatch,
                to_f32_pipeline,
                to_f32_dispatch,
                head_pipeline,
            );
            bail!("position0 physical head bank bytes 与 production chunk 漂移");
        }
        let head_workspace = S14HeadChunkWorkspace {
            normalized_input: workspace.final_normalized_f32,
            chunk_logits: workspace.head_logits,
            accumulator: workspace.head_argmax,
        };
        let mut head_dispatches: Vec<S14HeadChunkArgmaxDispatch<'a>> =
            Vec::with_capacity(S14_HEAD_CHUNK_COUNT as usize);
        for chunk in 0..head_shape.chunk_count() {
            let bank = chunk as usize % 2;
            let dispatch = match head_pipeline.bind_chunk(
                ctx,
                head_shape,
                chunk,
                StorageBufferSlice {
                    buffer: arena.head_chunk(bank)?,
                    offset: 0,
                },
                head_workspace,
            ) {
                Ok(dispatch) => dispatch,
                Err(error) => {
                    for dispatch in head_dispatches {
                        dispatch.destroy(ctx);
                    }
                    destroy_prelude(
                        ctx,
                        final_hc_pipeline,
                        final_hc_dispatch,
                        rmsnorm_pipeline,
                        rmsnorm_dispatch,
                        to_f32_pipeline,
                        to_f32_dispatch,
                        head_pipeline,
                    );
                    return Err(error).context(format!("bind final head chunk {chunk}"));
                }
            };
            head_dispatches.push(dispatch);
        }
        let head_recorder = S14HeadChunkArgmaxRecorder::new(head_shape)?;
        let readback = match S14Position0TerminalReadback::new(ctx) {
            Ok(readback) => readback,
            Err(error) => {
                for dispatch in head_dispatches {
                    dispatch.destroy(ctx);
                }
                destroy_prelude(
                    ctx,
                    final_hc_pipeline,
                    final_hc_dispatch,
                    rmsnorm_pipeline,
                    rmsnorm_dispatch,
                    to_f32_pipeline,
                    to_f32_dispatch,
                    head_pipeline,
                );
                return Err(error).context("allocate terminal readback");
            }
        };

        Ok(Self {
            final_hc_pipeline,
            final_hc_dispatch,
            rmsnorm_pipeline,
            rmsnorm_dispatch,
            to_f32_pipeline,
            to_f32_dispatch,
            head_pipeline,
            head_dispatches,
            head_recorder,
            head_recording_receipt: None,
            workspace,
            readback,
            phase: TerminalPhase::Bound,
            next_head_chunk: 0,
        })
    }

    /// 同步闭合固定 position0 参考轨迹。`stage_head` 必须按 chunk 顺序把真实、
    /// 已校验的 head payload 写入 `arena.head_chunk(chunk % 2)`，返回前 transfer
    /// 已完成。最终 command 同时复制 HC 到 inactive device state，并抓取43层
    /// compact state，因而调用返回后可直接 snapshot/原子提交。
    pub fn execute_synchronous_reference<F>(
        &mut self,
        ctx: &VulkanContext,
        base_epoch: u64,
        candidate_bank: usize,
        candidate_hc: StorageBufferSlice<'a>,
        state_readback: &mut S14Position0StateReadback,
        mut stage_head: F,
    ) -> Result<S14Position0SynchronousTerminalCompletion>
    where
        F: FnMut(u32) -> Result<()>,
    {
        if self.phase != TerminalPhase::Bound || candidate_bank > 1 {
            return self.poison("position0 synchronous terminal 初始 phase/bank 漂移");
        }
        if candidate_hc.offset + FINAL_HC_STREAM_BYTES > candidate_hc.buffer.size() {
            return self.poison("position0 synchronous terminal HC candidate 越界");
        }

        let pool = unsafe {
            ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let command = match unsafe {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        } {
            Ok(commands) => commands[0],
            Err(error) => {
                unsafe { ctx.device.destroy_command_pool(pool, None) };
                return Err(error.into());
            }
        };
        let fence = match unsafe {
            ctx.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(error) => {
                unsafe { ctx.device.destroy_command_pool(pool, None) };
                return Err(error.into());
            }
        };

        let result = (|| -> Result<S14Position0SynchronousTerminalCompletion> {
            let mut compute_host_waits = 0u32;

            unsafe {
                begin_synchronous_terminal_command(ctx, pool, command)?;
                self.record_prelude(ctx, command)?;
                end_submit_wait_synchronous_terminal(ctx, command, fence)?;
            }
            compute_host_waits += 1;
            self.phase = TerminalPhase::PreludeSubmitted;

            for chunk in 0..S14_HEAD_CHUNK_COUNT {
                stage_head(chunk)
                    .with_context(|| format!("stage synchronous head chunk {chunk}"))?;
                unsafe {
                    begin_synchronous_terminal_command(ctx, pool, command)?;
                    self.record_head_chunk(ctx, chunk, command)?;
                    end_submit_wait_synchronous_terminal(ctx, command, fence)?;
                }
                compute_host_waits += 1;
                self.next_head_chunk += 1;
                self.phase = TerminalPhase::PreludeSubmitted;
            }

            unsafe {
                begin_synchronous_terminal_command(ctx, pool, command)?;
                self.record_terminal_readback(ctx, command)?;
                self.record_candidate_hc_copy(ctx, command, candidate_hc)?;
                state_readback.record(ctx, command, candidate_hc.buffer)?;
                end_submit_wait_synchronous_terminal(ctx, command, fence)?;
            }
            compute_host_waits += 1;

            let head_receipt = self
                .head_recording_receipt
                .as_ref()
                .ok_or_else(|| anyhow!("synchronous terminal 缺少 head recording receipt"))?;
            let snapshot = self.readback.snapshot(head_receipt)?;
            self.phase = TerminalPhase::Completed;
            Ok(S14Position0SynchronousTerminalCompletion {
                base_epoch,
                candidate_bank,
                predicted_token_id: snapshot.argmax.token_id,
                max_logit: snapshot.argmax.logit,
                hc_streams_bf16: snapshot.hc_streams_bf16,
                normalized_f32_le_sha256: sha256_f32(&snapshot.normalized_f32),
                normalized_f32: snapshot.normalized_f32,
                compute_host_waits,
            })
        })();

        if result.is_err() {
            self.phase = TerminalPhase::Poisoned;
            unsafe {
                let _ = ctx.device.device_wait_idle();
            }
        }
        unsafe {
            ctx.device.destroy_fence(fence, None);
            ctx.device.destroy_command_pool(pool, None);
        }
        result
    }

    /// # Safety
    ///
    /// `command` 必须处于 recording，且 L42 与 head compute 已完成。
    unsafe fn record_candidate_hc_copy(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        candidate_hc: StorageBufferSlice<'_>,
    ) -> Result<()> {
        let barriers = [
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(self.workspace.final_hidden_streams.buffer.handle())
                .offset(self.workspace.final_hidden_streams.offset)
                .size(FINAL_HC_STREAM_BYTES),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .buffer(candidate_hc.buffer.handle())
                .offset(candidate_hc.offset)
                .size(FINAL_HC_STREAM_BYTES),
        ];
        ctx.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &barriers,
            &[],
        );
        ctx.device.cmd_copy_buffer(
            command,
            self.workspace.final_hidden_streams.buffer.handle(),
            candidate_hc.buffer.handle(),
            &[vk::BufferCopy::default()
                .src_offset(self.workspace.final_hidden_streams.offset)
                .dst_offset(candidate_hc.offset)
                .size(FINAL_HC_STREAM_BYTES)],
        );
        Ok(())
    }

    /// 录制 final HC、RMSNorm 和 BF16→F32。只清零一次共享 sticky status。
    ///
    /// # Safety
    /// `command` 必须处于 recording 状态，全部资源活到 timeline 完成。
    pub unsafe fn record_prelude(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
    ) -> Result<()> {
        if self.phase != TerminalPhase::Bound {
            return self.poison("position0 terminal prelude phase 漂移");
        }
        ctx.device.cmd_fill_buffer(
            command,
            self.workspace.status.buffer.handle(),
            self.workspace.status.offset,
            TERMINAL_STATUS_BYTES,
            0,
        );
        transfer_to_compute_barrier(ctx, command);
        self.final_hc_pipeline
            .cmd(ctx, command, &self.final_hc_dispatch);
        compute_to_compute_barrier(ctx, command);
        self.rmsnorm_pipeline
            .cmd(ctx, command, &self.rmsnorm_dispatch);
        compute_to_compute_barrier(ctx, command);
        self.to_f32_pipeline
            .cmd(ctx, command, &self.to_f32_dispatch);
        compute_to_compute_barrier(ctx, command);
        self.phase = TerminalPhase::PreludeRecorded;
        Ok(())
    }

    /// L42 后封口 43 层，但不 host wait；prelude 继续进入同一 compute timeline。
    ///
    /// # Safety
    /// `command` 必须已结束录制。
    pub unsafe fn seal_and_submit_prelude(
        &mut self,
        ctx: &VulkanContext,
        timeline: &mut S14Position0PagedLayerTimeline,
        command: vk::CommandBuffer,
    ) -> Result<u64> {
        if self.phase != TerminalPhase::PreludeRecorded {
            return self.poison("position0 terminal prelude submit phase 漂移");
        }
        if let Err(error) = timeline.seal_layers() {
            self.phase = TerminalPhase::Poisoned;
            return Err(error).context("seal 43 layers before terminal");
        }
        self.submit_recorded_prelude(ctx, timeline, command)
    }

    /// bridge 已经完成 43 层封口后，把预先录好的 final HC/RMSNorm prelude
    /// 继续提交到同一个 compute timeline。这里不重复 `seal_layers`，也不产生 host wait。
    ///
    /// # Safety
    /// `command` 必须已结束录制，timeline 必须处于 `TailOpen`。
    pub unsafe fn submit_recorded_prelude(
        &mut self,
        ctx: &VulkanContext,
        timeline: &mut S14Position0PagedLayerTimeline,
        command: vk::CommandBuffer,
    ) -> Result<u64> {
        if self.phase != TerminalPhase::PreludeRecorded {
            return self.poison("position0 terminal presealed prelude phase 漂移");
        }
        match timeline.submit_tail_compute_only(ctx, command) {
            Ok(value) => {
                self.phase = TerminalPhase::PreludeSubmitted;
                Ok(value)
            }
            Err(error) => {
                self.phase = TerminalPhase::Poisoned;
                Err(error).context("submit final HC/RMSNorm prelude")
            }
        }
    }

    /// 将当前 chunk 的 head 与累计 argmax 录入 compute command。chunk0 同时清零
    /// head accumulator，之后禁止再次 reset。
    ///
    /// # Safety
    /// `command` 必须处于 recording 状态。
    pub unsafe fn record_head_chunk(
        &mut self,
        ctx: &VulkanContext,
        chunk: u32,
        command: vk::CommandBuffer,
    ) -> Result<()> {
        if self.phase != TerminalPhase::PreludeSubmitted || chunk != self.next_head_chunk {
            return self.poison("position0 terminal head record 顺序/phase 漂移");
        }
        let Some(dispatch) = self.head_dispatches.get(chunk as usize) else {
            return self.poison("position0 terminal head chunk 越界");
        };
        if chunk == 0 {
            if let Err(error) = self.head_recorder.cmd_reset(ctx, command, dispatch) {
                self.phase = TerminalPhase::Poisoned;
                return Err(error).context("reset terminal head accumulator");
            }
        }
        if let Err(error) =
            self.head_recorder
                .cmd_chunk(ctx, command, &self.head_pipeline, dispatch)
        {
            self.phase = TerminalPhase::Poisoned;
            return Err(error).context(format!("record terminal head chunk {chunk}"));
        }
        self.phase = TerminalPhase::HeadComputeRecorded { chunk };
        Ok(())
    }

    /// 提交已录制的双 bank head transfer/compute。stage 只能准备本 chunk 的真实
    /// verified payload；该模块不生成或回显 fixture。
    ///
    /// # Safety
    /// 两个 command 必须已结束录制，staging/device 页活到 timeline 完成。
    pub unsafe fn submit_recorded_head<F>(
        &mut self,
        ctx: &VulkanContext,
        timeline: &mut S14Position0PagedLayerTimeline,
        chunk: u32,
        transfer_command: vk::CommandBuffer,
        compute_command: vk::CommandBuffer,
        stage: F,
    ) -> Result<u64>
    where
        F: FnOnce(usize) -> Result<()>,
    {
        if self.phase != (TerminalPhase::HeadComputeRecorded { chunk })
            || chunk != self.next_head_chunk
        {
            return self.poison("position0 terminal head submit 顺序/phase 漂移");
        }
        self.phase = TerminalPhase::HeadSubmitting { chunk };
        match timeline.stage_and_submit_head(
            ctx,
            u64::from(chunk),
            transfer_command,
            compute_command,
            stage,
        ) {
            Ok(ticket) => {
                self.next_head_chunk += 1;
                self.phase = TerminalPhase::PreludeSubmitted;
                Ok(ticket.compute_value)
            }
            Err(error) => {
                self.phase = TerminalPhase::Poisoned;
                Err(error).context(format!("submit terminal head chunk {chunk}"))
            }
        }
    }

    /// 32 块全部入队后录制 terminal readback。GPU argmax 已在每块内累计；这里将
    /// argmax、sticky status、final HC streams 与 normalized F32 一起复制到一个
    /// host-visible buffer，保证最终 timeline ticket 覆盖全部可发布数据。
    ///
    /// # Safety
    /// `command` 必须处于 recording 状态。
    pub unsafe fn record_terminal_readback(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
    ) -> Result<()> {
        if self.phase != TerminalPhase::PreludeSubmitted
            || self.next_head_chunk != S14_HEAD_CHUNK_COUNT
        {
            return self.poison("position0 terminal readback 前 head chunks 不完整");
        }
        let receipt = match self.head_recorder.finish_recording() {
            Ok(receipt) => receipt,
            Err(error) => {
                self.phase = TerminalPhase::Poisoned;
                return Err(error).context("finish terminal head recording");
            }
        };
        if let Err(error) = self.readback.cmd_copy(ctx, command, self.workspace) {
            self.phase = TerminalPhase::Poisoned;
            return Err(error).context("record terminal candidate readback");
        }
        self.head_recording_receipt = Some(receipt);
        self.phase = TerminalPhase::TerminalRecorded;
        Ok(())
    }

    /// 把成功候选发布所需的三类数据收进同一个最终 command：terminal 数值回读、
    /// terminal HC 写入 inactive device state，以及 43 层 compact state 回读。
    /// 最终 timeline wait 成功前，三者都不可被 host/device commit 看见。
    ///
    /// # Safety
    /// `command` 必须处于 recording，`candidate_hc` 与 state readback 资源必须活到
    /// `finish_candidate` 或错误路径 drain 完成。
    pub unsafe fn record_terminal_commit_readback(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        candidate_hc: StorageBufferSlice<'_>,
        state_readback: &mut S14Position0StateReadback,
    ) -> Result<()> {
        self.record_terminal_readback(ctx, command)?;
        if let Err(error) = self.record_candidate_hc_copy(ctx, command, candidate_hc) {
            self.phase = TerminalPhase::Poisoned;
            return Err(error).context("record terminal candidate HC copy");
        }
        if let Err(error) = state_readback.record(ctx, command, candidate_hc.buffer) {
            self.phase = TerminalPhase::Poisoned;
            return Err(error).context("record terminal compact state readback");
        }
        Ok(())
    }

    /// 将 terminal readback 作为本 candidate 最后一个 compute timeline ticket。
    ///
    /// # Safety
    /// `command` 必须已结束录制。
    pub unsafe fn submit_terminal(
        &mut self,
        ctx: &VulkanContext,
        timeline: &mut S14Position0PagedLayerTimeline,
        command: vk::CommandBuffer,
    ) -> Result<u64> {
        if self.phase != TerminalPhase::TerminalRecorded {
            return self.poison("position0 terminal final submit phase 漂移");
        }
        match timeline.submit_final_compute(ctx, command) {
            Ok(value) => {
                self.phase = TerminalPhase::TerminalSubmitted;
                Ok(value)
            }
            Err(error) => {
                self.phase = TerminalPhase::Poisoned;
                Err(error).context("submit terminal argmax/readback")
            }
        }
    }

    /// 成功路径唯一 token host wait。wait 后任一状态或 readback 校验失败都禁止
    /// 签发 completion，且不得再次等待。
    pub fn finish_candidate(
        &mut self,
        ctx: &VulkanContext,
        timeline: &mut S14Position0PagedLayerTimeline,
        base_epoch: u64,
        candidate_bank: usize,
    ) -> Result<S14Position0TerminalCompletion> {
        if self.phase != TerminalPhase::TerminalSubmitted || candidate_bank > 1 {
            return self.poison("position0 terminal finish phase/bank 漂移");
        }
        let timeline_receipt = match timeline.finish_candidate(ctx) {
            Ok(receipt) => receipt,
            Err(error) => {
                // 前置合同或 Vulkan wait 本身失败时 timeline 仍可能只处于 Poisoned，
                // 资源尚未收敛，必须允许上层继续走一次 drain。只有 timeline 已明确
                // Drained/Finished 才能禁止第二次 host wait。
                self.phase = match timeline.stats().state {
                    crate::s14_position0_paged_layer_timeline::S14Position0PagedLayerTimelineState::Drained
                    | crate::s14_position0_paged_layer_timeline::S14Position0PagedLayerTimelineState::Finished => {
                        TerminalPhase::FailedAfterWait
                    }
                    _ => TerminalPhase::Poisoned,
                };
                return Err(error).context("wait terminal candidate");
            }
        };
        if let Err(error) = validate_terminal_timeline(&timeline_receipt) {
            self.phase = TerminalPhase::FailedAfterWait;
            return Err(error);
        }
        let Some(head_receipt) = self.head_recording_receipt.as_ref() else {
            self.phase = TerminalPhase::FailedAfterWait;
            bail!("position0 terminal 缺少 head recording receipt");
        };
        let snapshot = match self.readback.snapshot(head_receipt) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.phase = TerminalPhase::FailedAfterWait;
                return Err(error).context("validate terminal readback");
            }
        };
        self.phase = TerminalPhase::Completed;
        Ok(S14Position0TerminalCompletion {
            base_epoch,
            candidate_bank,
            timeline: timeline_receipt,
            predicted_token_id: snapshot.argmax.token_id,
            max_logit: snapshot.argmax.logit,
            hc_streams_bf16: snapshot.hc_streams_bf16,
            normalized_f32_le_sha256: sha256_f32(&snapshot.normalized_f32),
            normalized_f32: snapshot.normalized_f32,
        })
    }

    /// 成功 wait 之前的错误路径只调用一次联合 drain；orphan transfer 也会一起收敛。
    pub fn drain_all(
        &mut self,
        ctx: &VulkanContext,
        timeline: &mut S14Position0PagedLayerTimeline,
    ) -> Result<S14Position0PagedDrainReceipt> {
        if matches!(
            self.phase,
            TerminalPhase::Completed | TerminalPhase::FailedAfterWait | TerminalPhase::Drained
        ) {
            bail!("position0 terminal 已 wait/drained，禁止第二次 host wait");
        }
        self.phase = TerminalPhase::Poisoned;
        let receipt = timeline.drain_all(ctx)?;
        if receipt.host_wait_calls > 1 {
            bail!("position0 terminal drain host wait count 漂移");
        }
        self.phase = TerminalPhase::Drained;
        Ok(receipt)
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.readback.destroy(ctx);
        for dispatch in self.head_dispatches {
            dispatch.destroy(ctx);
        }
        self.head_pipeline.destroy(ctx);
        self.to_f32_dispatch.binder.destroy(ctx);
        self.to_f32_pipeline.destroy(ctx);
        self.rmsnorm_dispatch.binder.destroy(ctx);
        self.rmsnorm_pipeline.destroy(ctx);
        self.final_hc_dispatch.binder.destroy(ctx);
        self.final_hc_pipeline.destroy(ctx);
    }

    fn poison<T>(&mut self, message: &'static str) -> Result<T> {
        self.phase = TerminalPhase::Poisoned;
        bail!(message)
    }
}

/// 唯一 wait 后的 host DecoderState 提交门。先在克隆上完成全部可能失败的
/// state staging/commit/validation，最后一步才替换真实 state；失败时原状态逐字不变。
pub fn commit_terminal_candidate(
    state: &mut DecoderStateV1,
    mut candidate: WholeTokenCandidate,
    completion: &S14Position0TerminalCompletion,
) -> Result<TokenRecord> {
    state
        .validate()
        .context("validate committed DecoderState")?;
    validate_terminal_completion(completion)?;
    if state.commit_epoch != completion.base_epoch
        || candidate.inactive_fixed_bank() as usize != completion.candidate_bank
        || completion.candidate_bank == state.active_fixed_bank as usize
    {
        bail!("terminal completion epoch/candidate bank 与 DecoderState 漂移");
    }
    candidate
        .stage_position0_hc_state(&completion.hc_streams_bf16)
        .context("stage terminal HC state")?;
    candidate
        .complete_final(completion.predicted_token_id)
        .context("complete terminal token")?;

    let expected_epoch = completion
        .base_epoch
        .checked_add(1)
        .ok_or_else(|| anyhow!("terminal commit epoch overflow"))?;
    let mut next = state.clone();
    let token = candidate
        .commit(&mut next)
        .context("commit terminal candidate into cloned DecoderState")?;
    next.validate().context("validate next DecoderState")?;
    if next.commit_epoch != expected_epoch
        || next.active_fixed_bank as usize != completion.candidate_bank
        || next.input_token_id != completion.predicted_token_id
        || token.predicted_token_id != completion.predicted_token_id
    {
        bail!("terminal cloned DecoderState commit ledger 漂移");
    }
    *state = next;
    Ok(token)
}

pub fn validate_terminal_timeline(receipt: &S14Position0PagedCandidateReceipt) -> Result<()> {
    let expected_layer_final = receipt
        .prologue_compute_value
        .checked_add(FULL_DEPTH_LAYERS.len() as u64)
        .ok_or_else(|| anyhow!("position0 terminal layer timeline overflow"))?;
    let expected_final = expected_layer_final
        .checked_add(u64::from(S14_HEAD_CHUNK_COUNT))
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| anyhow!("position0 terminal final timeline overflow"))?;
    let expected_transfer = FULL_DEPTH_LAYERS.len() as u64 + u64::from(S14_HEAD_CHUNK_COUNT);
    if receipt.layers != FULL_DEPTH_LAYERS.len()
        || receipt.prologue_compute_value == 0
        || receipt.layer_final_compute_value != expected_layer_final
        || receipt.head_chunks != u64::from(S14_HEAD_CHUNK_COUNT)
        || receipt.tail_compute_segments != 2
        || receipt.token_host_waits != 1
        || receipt.final_compute_value != expected_final
        || receipt.completed_compute_value != receipt.final_compute_value
        || receipt.completed_transfer_value != expected_transfer
    {
        bail!("position0 terminal timeline receipt 不完整");
    }
    Ok(())
}

pub fn validate_terminal_completion(completion: &S14Position0TerminalCompletion) -> Result<()> {
    validate_terminal_timeline(&completion.timeline)?;
    if completion.candidate_bank > 1
        || completion.predicted_token_id >= 129_280
        || !completion.max_logit.is_finite()
        || completion.hc_streams_bf16.len() != S14_POSITION0_FINAL_HC_ELEMENTS
        || completion.normalized_f32.len() != S14_POSITION0_FINAL_NORMALIZED_ELEMENTS
        || completion
            .hc_streams_bf16
            .iter()
            .any(|bits| !f32::from_bits((*bits as u32) << 16).is_finite())
        || completion
            .normalized_f32
            .iter()
            .any(|value| !value.is_finite())
        || completion.normalized_f32_le_sha256 != sha256_f32(&completion.normalized_f32)
    {
        bail!("position0 terminal completion 数值/shape/hash 漂移");
    }
    Ok(())
}

struct TerminalSnapshot {
    argmax: S14HeadArgmaxResult,
    hc_streams_bf16: Vec<u16>,
    normalized_f32: Vec<f32>,
}

impl S14Position0TerminalReadback {
    fn new(ctx: &VulkanContext) -> Result<Self> {
        let layout = TerminalReadbackLayout::production();
        let buffer = GpuBuffer::new(
            ctx,
            layout.bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;
        Ok(Self { buffer, layout })
    }

    unsafe fn cmd_copy(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        workspace: TerminalWorkspace<'_>,
    ) -> Result<()> {
        let source = workspace.final_hidden_streams.buffer.handle();
        if workspace.final_normalized_f32.buffer.handle() != source
            || workspace.status.buffer.handle() != source
            || workspace.head_argmax.buffer.handle() != source
        {
            bail!("terminal readback sources 必须属于同一 workspace buffer");
        }
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        ctx.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        let copies = [
            vk::BufferCopy::default()
                .src_offset(workspace.final_hidden_streams.offset)
                .dst_offset(self.layout.hc_streams)
                .size(FINAL_HC_STREAM_BYTES),
            vk::BufferCopy::default()
                .src_offset(workspace.final_normalized_f32.offset)
                .dst_offset(self.layout.normalized_f32)
                .size(FINAL_NORMALIZED_F32_BYTES),
            vk::BufferCopy::default()
                .src_offset(workspace.status.offset)
                .dst_offset(self.layout.status)
                .size(TERMINAL_STATUS_BYTES),
            vk::BufferCopy::default()
                .src_offset(workspace.head_argmax.offset)
                .dst_offset(self.layout.argmax)
                .size(S14_HEAD_ARGMAX_BYTES),
        ];
        ctx.device
            .cmd_copy_buffer(command, source, self.buffer.handle(), &copies);
        let host_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        ctx.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[host_barrier],
            &[],
            &[],
        );
        Ok(())
    }

    fn snapshot(
        &self,
        head_receipt: &S14HeadChunkArgmaxRecordingReceipt,
    ) -> Result<TerminalSnapshot> {
        let status = self.read_one::<u32>(self.layout.status);
        validate_terminal_status(status)?;
        let argmax_words = self.read_vec::<u32>(self.layout.argmax, S14_HEAD_ARGMAX_WORDS);
        let argmax = decode_head_argmax(
            head_receipt,
            argmax_words
                .try_into()
                .map_err(|_| anyhow!("terminal argmax readback word count drift"))?,
        )?;
        let hc_streams_bf16 =
            self.read_vec::<u16>(self.layout.hc_streams, S14_POSITION0_FINAL_HC_ELEMENTS);
        if hc_streams_bf16
            .iter()
            .any(|bits| !f32::from_bits((*bits as u32) << 16).is_finite())
        {
            bail!("terminal HC readback contains NaN/Inf");
        }
        let normalized_f32 = self.read_vec::<f32>(
            self.layout.normalized_f32,
            S14_POSITION0_FINAL_NORMALIZED_ELEMENTS,
        );
        if normalized_f32.iter().any(|value| !value.is_finite()) {
            bail!("terminal normalized readback contains NaN/Inf");
        }
        Ok(TerminalSnapshot {
            argmax,
            hc_streams_bf16,
            normalized_f32,
        })
    }

    fn read_one<T: Copy>(&self, offset: u64) -> T {
        unsafe { *((self.buffer.mapped() as *const u8).add(offset as usize) as *const T) }
    }

    fn read_vec<T: Copy>(&self, offset: u64, len: usize) -> Vec<T> {
        unsafe {
            std::slice::from_raw_parts(
                (self.buffer.mapped() as *const u8).add(offset as usize) as *const T,
                len,
            )
            .to_vec()
        }
    }

    fn destroy(self, ctx: &VulkanContext) {
        self.buffer.destroy(ctx);
    }
}

fn slot_slice<'a>(
    buffer: &'a GpuBuffer,
    layout: &crate::s14_position0_workspace::S14Position0WorkspaceLayout,
    slot: S14Position0WorkspaceSlot,
) -> StorageBufferSlice<'a> {
    StorageBufferSlice {
        buffer,
        offset: layout.region(slot).offset,
    }
}

fn final_weight_slice<'a>(
    arena: &'a S14Position0PagedWeightArena,
    tensor: &str,
    expected_bytes: u64,
) -> Result<StorageBufferSlice<'a>> {
    let physical = arena
        .plan()
        .physical
        .resident_small
        .assets
        .iter()
        .find(|asset| asset.tensor == tensor)
        .ok_or_else(|| anyhow!("position0 final tensor 不在 resident-small: {tensor}"))?;
    let binding = arena.static_asset(tensor)?;
    if physical.bytes != expected_bytes
        || physical.local_offset != binding.destination_offset
        || !binding.resident_once
        || binding.layer.is_some()
        || binding.bank.is_some()
    {
        bail!("position0 final tensor physical contract 漂移: {tensor}");
    }
    Ok(StorageBufferSlice {
        buffer: binding.buffer,
        offset: binding.destination_offset,
    })
}

fn hc_slice(slice: StorageBufferSlice<'_>) -> S14FinalHcHeadBufferSlice<'_> {
    S14FinalHcHeadBufferSlice::new(slice.buffer, slice.offset)
}

fn validate_terminal_workspace(workspace: TerminalWorkspace<'_>) -> Result<()> {
    let buffer = workspace.final_hidden_streams.buffer.handle();
    for (slice, bytes, name) in [
        (
            workspace.final_hidden_streams,
            FINAL_HC_STREAM_BYTES,
            "L42 hidden streams",
        ),
        (workspace.final_hidden_bf16, 8192, "final HC output"),
        (workspace.final_normalized_bf16, 8192, "final RMSNorm BF16"),
        (
            workspace.final_normalized_f32,
            FINAL_NORMALIZED_F32_BYTES,
            "final normalized F32",
        ),
        (workspace.inverse_rms, 4, "final inverse RMS"),
        (workspace.hc_aux, 32, "final HC aux"),
        (workspace.status, TERMINAL_STATUS_BYTES, "terminal status"),
        (workspace.head_logits, 4096 * 4, "head chunk logits"),
        (workspace.head_argmax, S14_HEAD_ARGMAX_BYTES, "head argmax"),
    ] {
        if slice.buffer.handle() != buffer {
            bail!("{name} 不属于同一 whole-token workspace");
        }
        let end = slice
            .offset
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("{name} range overflow"))?;
        if end > slice.buffer.size() {
            bail!("{name} 越出 whole-token workspace");
        }
    }
    Ok(())
}

fn validate_terminal_status(status: u32) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    if status & !TERMINAL_KNOWN_STATUS_BITS != 0 {
        bail!("terminal primitive returned unknown status bits 0x{status:08x}");
    }
    bail!("terminal HC/RMSNorm/convert rejected candidate, status=0x{status:08x}")
}

fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

unsafe fn begin_synchronous_terminal_command(
    ctx: &VulkanContext,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
) -> Result<()> {
    ctx.device
        .reset_command_pool(pool, vk::CommandPoolResetFlags::RELEASE_RESOURCES)?;
    ctx.device.begin_command_buffer(
        command,
        &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
    )?;
    Ok(())
}

unsafe fn end_submit_wait_synchronous_terminal(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    fence: vk::Fence,
) -> Result<()> {
    ctx.device.end_command_buffer(command)?;
    ctx.device.reset_fences(&[fence])?;
    let commands = [command];
    ctx.device.queue_submit(
        ctx.q_graphics,
        &[vk::SubmitInfo::default().command_buffers(&commands)],
        fence,
    )?;
    ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn destroy_prelude(
    ctx: &VulkanContext,
    final_hc_pipeline: S14FinalHcHeadPipeline,
    final_hc_dispatch: S14FinalHcHeadDispatch,
    rmsnorm_pipeline: S14Bf16RmsNormPipeline,
    rmsnorm_dispatch: S14Bf16RmsNormDispatch,
    to_f32_pipeline: S14Bf16ToF32Pipeline,
    to_f32_dispatch: S14Bf16ToF32Dispatch,
    head_pipeline: S14HeadChunkArgmaxPipeline,
) {
    head_pipeline.destroy(ctx);
    to_f32_dispatch.binder.destroy(ctx);
    to_f32_pipeline.destroy(ctx);
    rmsnorm_dispatch.binder.destroy(ctx);
    rmsnorm_pipeline.destroy(ctx);
    final_hc_dispatch.binder.destroy(ctx);
    final_hc_pipeline.destroy(ctx);
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

#[cfg(test)]
mod tests {
    use super::*;
    use polaris_s14_runner::Position0CompressorInput;

    fn timeline_receipt() -> S14Position0PagedCandidateReceipt {
        S14Position0PagedCandidateReceipt {
            prologue_compute_value: 1,
            layers: 43,
            layer_final_compute_value: 44,
            final_compute_value: 78,
            head_chunks: 32,
            tail_compute_segments: 2,
            producer_transfer_waits: 30,
            device_bank_reuse_waits: 30,
            token_host_waits: 1,
            completed_transfer_value: 75,
            completed_compute_value: 78,
        }
    }

    fn stage_layer(candidate: &mut WholeTokenCandidate, layer: u8) {
        let kv = vec![0x3f80 + layer as u16; 512];
        match candidate.staged_native_mut().kv[layer as usize].compress_ratio {
            0 => candidate
                .stage_position0_layer_state(layer, &kv, Position0CompressorInput::None)
                .unwrap(),
            4 => candidate
                .stage_position0_layer_state(
                    layer,
                    &kv,
                    Position0CompressorInput::Ratio4 {
                        main_kv: &vec![1.0; 1024],
                        main_score: &vec![2.0; 1024],
                        indexer_kv: &vec![3.0; 256],
                        indexer_score: &vec![4.0; 256],
                    },
                )
                .unwrap(),
            128 => candidate
                .stage_position0_layer_state(
                    layer,
                    &kv,
                    Position0CompressorInput::Ratio128 {
                        main_kv: &vec![1.0; 512],
                        main_score: &vec![2.0; 512],
                    },
                )
                .unwrap(),
            _ => unreachable!(),
        }
        candidate.complete_layer(layer).unwrap();
    }

    fn ready_candidate(state: &DecoderStateV1) -> WholeTokenCandidate {
        let mut candidate = state.begin_token(0, 0, 0).unwrap();
        for &layer in &FULL_DEPTH_LAYERS {
            stage_layer(&mut candidate, layer);
        }
        candidate
    }

    fn completion(token: u32) -> S14Position0TerminalCompletion {
        let normalized_f32 = vec![0.25; S14_POSITION0_FINAL_NORMALIZED_ELEMENTS];
        S14Position0TerminalCompletion {
            base_epoch: 0,
            candidate_bank: 1,
            timeline: timeline_receipt(),
            predicted_token_id: token,
            max_logit: 9.5,
            hc_streams_bf16: vec![0x3f80; S14_POSITION0_FINAL_HC_ELEMENTS],
            normalized_f32_le_sha256: sha256_f32(&normalized_f32),
            normalized_f32,
        }
    }

    #[test]
    fn production_readback_and_timeline_contract_are_exact() {
        let layout = TerminalReadbackLayout::production();
        assert_eq!(layout.hc_streams, 0);
        assert_eq!(layout.normalized_f32, 32_768);
        assert_eq!(layout.status, 49_152);
        assert_eq!(layout.argmax, 49_156);
        assert_eq!(layout.bytes, 49_172);
        validate_terminal_timeline(&timeline_receipt()).unwrap();

        let mut bad = timeline_receipt();
        bad.head_chunks = 31;
        assert!(validate_terminal_timeline(&bad).is_err());
        let mut bad = timeline_receipt();
        bad.token_host_waits = 2;
        assert!(validate_terminal_timeline(&bad).is_err());
        let mut bad = timeline_receipt();
        bad.tail_compute_segments = 1;
        assert!(validate_terminal_timeline(&bad).is_err());
    }

    #[test]
    fn terminal_completion_commits_decoder_state_once_from_gpu_token() {
        let mut state = DecoderStateV1::new(16, 0).unwrap();
        let candidate = ready_candidate(&state);
        let result = commit_terminal_candidate(&mut state, candidate, &completion(17)).unwrap();
        assert_eq!(result.predicted_token_id, 17);
        assert_eq!(state.position, 1);
        assert_eq!(state.commit_epoch, 1);
        assert_eq!(state.active_fixed_bank, 1);
        assert_eq!(state.input_token_id, 17);
        assert_eq!(state.committed_tokens.len(), 1);
        state.validate().unwrap();
    }

    #[test]
    fn terminal_failures_leave_decoder_state_byte_equivalent() {
        let state = DecoderStateV1::new(16, 0).unwrap();

        let mut actual = state.clone();
        let candidate = ready_candidate(&actual);
        let mut bad = completion(23);
        bad.timeline.token_host_waits = 0;
        assert!(commit_terminal_candidate(&mut actual, candidate, &bad).is_err());
        assert_eq!(actual, state);

        let mut actual = state.clone();
        let candidate = ready_candidate(&actual);
        let mut bad = completion(23);
        bad.normalized_f32[3] = f32::NAN;
        assert!(commit_terminal_candidate(&mut actual, candidate, &bad).is_err());
        assert_eq!(actual, state);

        let mut actual = state.clone();
        let candidate = ready_candidate(&actual);
        let mut bad = completion(23);
        bad.candidate_bank = 0;
        assert!(commit_terminal_candidate(&mut actual, candidate, &bad).is_err());
        assert_eq!(actual, state);
    }

    #[test]
    fn terminal_status_rejects_every_failure_bit_and_unknown_bits() {
        validate_terminal_status(0).unwrap();
        for code in [1, 2, 4, 8, 15, 16] {
            assert!(validate_terminal_status(code).is_err());
        }
    }
}
