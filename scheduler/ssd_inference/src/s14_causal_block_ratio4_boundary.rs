//! K=4、base_position=1 的 ratio4 跨边界 production Vulkan 录制器。
//!
//! 不可交换顺序固定在一个 command buffer 内：position3 compressor remainder、
//! main/indexer finalize/writeback、rollover、position4 真实 remainder/index-query/indexer，
//! 最后一次 `(64,4,1)` boundary-aware attention。没有逐 token forward 或 CPU fallback 入口。

use crate::{
    compute::{ComputePipeline, DescriptorBinder, StorageBufferSlice},
    s14_sparse_attention::{
        S14Ratio4IndexQueryPipeline, S14Ratio4IndexQueryShape, S14SparseIndexerPipeline,
    },
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Result};
use ash::vk;
use std::{fmt, sync::Arc};

pub const S14_CAUSAL_BLOCK_RATIO4_BOUNDARY_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_causal_block_ratio4_boundary_attention.spv"
));

const BLOCK_SIZE: u32 = 4;
const BASE_POSITION: u32 = 1;
const BOUNDARY_POSITION: u32 = 3;
const BOUNDARY_LANE: u32 = 2;
const HEADS: u32 = 64;
const HEAD_DIM: u32 = 512;
const QUERY_ROW_BYTES: u64 = HEADS as u64 * HEAD_DIM as u64 * 2;
const KV_ROW_BYTES: u64 = HEAD_DIM as u64 * 2;
const SINK_BYTES: u64 = HEADS as u64 * 4;
const ROPE_ROW_BYTES: u64 = 32 * 2 * 4;
const STATUS_BYTES: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRatio4BoundaryShape;

impl S14CausalBlockRatio4BoundaryShape {
    pub fn new(base_position: u32, block_size: u32, compress_ratio: u16) -> Result<Self> {
        if base_position != BASE_POSITION || block_size != BLOCK_SIZE || compress_ratio != 4 {
            bail!(
                "ratio4 causal-block boundary 只闭合 base_position=1/K=4/ratio4，actual base={base_position} K={block_size} ratio={compress_ratio}"
            );
        }
        Ok(Self)
    }

    pub const fn positions(self) -> [u32; 4] {
        [1, 2, 3, 4]
    }

    pub const fn boundary_lane(self) -> u32 {
        BOUNDARY_LANE
    }

    pub const fn compressed_consumer_lanes(self) -> [u32; 2] {
        [2, 3]
    }
}

#[derive(Clone, Copy)]
pub struct S14CausalBlockRatio4AttentionBindings<'a> {
    pub query_bf16: StorageBufferSlice<'a>,
    pub committed_window_kv_bf16: StorageBufferSlice<'a>,
    pub current_block_kv_bf16: StorageBufferSlice<'a>,
    pub first_compressed_kv_bf16: StorageBufferSlice<'a>,
    pub position4_compressed_index_u32: StorageBufferSlice<'a>,
    pub sink_f32: StorageBufferSlice<'a>,
    pub rope_f32: StorageBufferSlice<'a>,
    pub output_bf16: StorageBufferSlice<'a>,
    pub sticky_status_u32: StorageBufferSlice<'a>,
}

#[derive(Clone, Copy)]
pub struct S14CausalBlockRatio4Position4IndexerBindings<'a> {
    pub raw_index_query_bf16: StorageBufferSlice<'a>,
    pub rope_f32: StorageBufferSlice<'a>,
    pub processed_index_query_bf16: StorageBufferSlice<'a>,
    pub index_cache_bf16: StorageBufferSlice<'a>,
    pub head_weights_bf16: StorageBufferSlice<'a>,
    pub index_scores_f32: StorageBufferSlice<'a>,
    pub compressed_indices_u32: StorageBufferSlice<'a>,
    pub sticky_status_u32: StorageBufferSlice<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRatio4CandidateStateBinding {
    pub layer: u8,
    pub base_position: u32,
    pub block_size: u32,
    pub candidate_base_offset: u64,
    pub candidate_logical_bytes: u64,
    pub first_compressed_kv_offset: u64,
    pub first_indexer_row_offset: u64,
}

impl S14CausalBlockRatio4CandidateStateBinding {
    pub fn validate(self, owner: &GpuBuffer) -> Result<()> {
        let candidate_end = self
            .candidate_base_offset
            .checked_add(self.candidate_logical_bytes)
            .ok_or_else(|| anyhow::anyhow!("ratio4 candidate state range overflow"))?;
        let main_end = self
            .first_compressed_kv_offset
            .checked_add(KV_ROW_BYTES)
            .ok_or_else(|| anyhow::anyhow!("ratio4 compressed main row overflow"))?;
        let indexer_end = self
            .first_indexer_row_offset
            .checked_add(128 * 2)
            .ok_or_else(|| anyhow::anyhow!("ratio4 compressed indexer row overflow"))?;
        if self.base_position != BASE_POSITION
            || self.block_size != BLOCK_SIZE
            || self.candidate_base_offset % 4 != 0
            || self.candidate_logical_bytes == 0
            || candidate_end > owner.size()
            || self.first_compressed_kv_offset < self.candidate_base_offset
            || main_end > candidate_end
            || self.first_indexer_row_offset < self.candidate_base_offset
            || indexer_end > candidate_end
        {
            bail!("ratio4 candidate state strong-owner binding 漂移");
        }
        Ok(())
    }
}

/// HC/QKV recorder 暴露给真实 position3/4 state owner 的最窄 workspace ABI。
/// state owner 仍必须自己强持有正式 recipe、candidate state、compressed RoPE 与
/// finalize/remainder pipelines；这里没有 fixture 或 CPU fallback。
#[derive(Clone, Copy)]
pub struct S14CausalBlockRatio4StateWorkspaceBindings<'a> {
    pub static_weights: StorageBufferSlice<'a>,
    pub static_logical_bytes: u64,
    pub hc_branch_bf16: StorageBufferSlice<'a>,
    pub hc_branch_f32: StorageBufferSlice<'a>,
    pub position4_query_low_f32: StorageBufferSlice<'a>,
    pub raw_index_query_bf16: StorageBufferSlice<'a>,
    pub position4_head_weights_bf16: StorageBufferSlice<'a>,
    pub index_weights_proj_weight_offset: u64,
    pub index_query_weight_offset: u64,
    pub index_query_scale_offset: u64,
    pub sticky_status_u32: StorageBufferSlice<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRatio4BoundaryFinalizeReceipt {
    pub base_position: u32,
    pub block_size: u32,
    pub boundary_position: u32,
    pub pre_boundary_remainder_record_calls: u32,
    pub main_finalize_writeback_calls: u32,
    pub indexer_finalize_writeback_calls: u32,
    pub compressed_main_rows_written: u32,
    pub compressed_indexer_rows_written: u32,
    pub serial_token_forward_calls: u32,
    pub cpu_fallback_calls: u32,
}

impl S14CausalBlockRatio4BoundaryFinalizeReceipt {
    fn validate(self) -> Result<()> {
        if self.base_position != BASE_POSITION
            || self.block_size != BLOCK_SIZE
            || self.boundary_position != BOUNDARY_POSITION
            || self.pre_boundary_remainder_record_calls != 3
            || self.main_finalize_writeback_calls != 1
            || self.indexer_finalize_writeback_calls != 1
            || self.compressed_main_rows_written != 1
            || self.compressed_indexer_rows_written != 1
            || self.serial_token_forward_calls != 0
            || self.cpu_fallback_calls != 0
        {
            bail!("ratio4 boundary finalize 回执不能证明 positions1..3 remainder/main/indexer device 写回");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRatio4RolloverReceipt {
    pub base_position: u32,
    pub block_size: u32,
    pub boundary_position: u32,
    pub rollover_record_calls: u32,
    pub serial_token_forward_calls: u32,
    pub cpu_fallback_calls: u32,
}

impl S14CausalBlockRatio4RolloverReceipt {
    fn validate(self) -> Result<()> {
        if self.base_position != BASE_POSITION
            || self.block_size != BLOCK_SIZE
            || self.boundary_position != BOUNDARY_POSITION
            || self.rollover_record_calls != 1
            || self.serial_token_forward_calls != 0
            || self.cpu_fallback_calls != 0
        {
            bail!("ratio4 boundary rollover 回执不能证明一次 post-attention device rollover");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRatio4Position4PreludeReceipt {
    pub base_position: u32,
    pub block_size: u32,
    pub position: u32,
    pub remainder_record_calls: u32,
    pub index_head_weight_projection_calls: u32,
    pub serial_token_forward_calls: u32,
    pub cpu_fallback_calls: u32,
}

impl S14CausalBlockRatio4Position4PreludeReceipt {
    fn validate(self) -> Result<()> {
        if self.base_position != BASE_POSITION
            || self.block_size != BLOCK_SIZE
            || self.position != 4
            || self.remainder_record_calls != 1
            || self.index_head_weight_projection_calls != 1
            || self.serial_token_forward_calls != 0
            || self.cpu_fallback_calls != 0
        {
            bail!("ratio4 position4 prelude 回执不能证明 post-rollover remainder/index-head device 录制");
        }
        Ok(())
    }
}

/// 真实 state owner 的窄接口。实现必须强拥有 candidate state、正式 state recipe、
/// compressor/finalize pipelines 与 descriptor，直到同一 command 完成。
pub trait S14CausalBlockRatio4BoundaryStateRecorder: fmt::Debug {
    fn candidate_state_owner(&self) -> &Arc<GpuBuffer>;

    fn candidate_state_binding(&self) -> S14CausalBlockRatio4CandidateStateBinding;

    /// # Safety
    /// `command` 必须处于 recording；实现不得提交、等待或执行 CPU fallback。
    unsafe fn record_remainder_and_finalize(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        workspace: S14CausalBlockRatio4StateWorkspaceBindings<'_>,
    ) -> Result<S14CausalBlockRatio4BoundaryFinalizeReceipt>;

    /// # Safety
    /// 只会在 main/indexer compressed row 已写回后调用；实现只能 rollover remainder，
    /// 不得覆盖稍后由 lane2 消费的 compressed block0/QKV。
    unsafe fn record_rollover_after_finalize(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        workspace: S14CausalBlockRatio4StateWorkspaceBindings<'_>,
    ) -> Result<S14CausalBlockRatio4RolloverReceipt>;

    /// # Safety
    /// 只会在 rollover 发布后调用；实现必须写 position4 remainder row4，并在 HC branch
    /// F32 被 QDQ 覆盖前投影真实动态 index-head weight。
    unsafe fn record_position4_remainder_and_index_head(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        workspace: S14CausalBlockRatio4StateWorkspaceBindings<'_>,
    ) -> Result<S14CausalBlockRatio4Position4PreludeReceipt>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRatio4BoundaryRecordingReceipt {
    pub positions: [u32; 4],
    pub boundary_lane: u32,
    pub compressed_consumer_lanes: [u32; 2],
    pub pre_boundary_remainder_record_calls: u32,
    pub main_finalize_writeback_calls: u32,
    pub indexer_finalize_writeback_calls: u32,
    pub rollover_record_calls: u32,
    pub position4_remainder_record_calls: u32,
    pub position4_index_head_weight_projection_calls: u32,
    pub position4_index_query_dispatch_calls: u32,
    pub position4_indexer_dispatch_calls: u32,
    pub attention_dispatch_calls: u32,
    pub attention_rows: u32,
    /// position4 由 post-rollover attention dispatch 的1行消费 indexer 输出。
    pub position4_sparse_attention_rows: u32,
    pub serial_token_forward_calls: u32,
    pub cpu_fallback_calls: u32,
}

impl S14CausalBlockRatio4BoundaryRecordingReceipt {
    pub fn validate(self) -> Result<()> {
        if self.positions != [1, 2, 3, 4]
            || self.boundary_lane != BOUNDARY_LANE
            || self.compressed_consumer_lanes != [2, 3]
            || self.pre_boundary_remainder_record_calls != 3
            || self.main_finalize_writeback_calls != 1
            || self.indexer_finalize_writeback_calls != 1
            || self.rollover_record_calls != 1
            || self.position4_remainder_record_calls != 1
            || self.position4_index_head_weight_projection_calls != 1
            || self.position4_index_query_dispatch_calls != 1
            || self.position4_indexer_dispatch_calls != 1
            || self.attention_dispatch_calls != 1
            || self.attention_rows != BLOCK_SIZE
            || self.position4_sparse_attention_rows != 1
            || self.serial_token_forward_calls != 0
            || self.cpu_fallback_calls != 0
        {
            bail!("ratio4 boundary recording 回执不能证明 finalize→rollover→position4 indexer→单次K=4 boundary attention 顺序");
        }
        Ok(())
    }
}

pub struct S14CausalBlockRatio4BoundaryRecording {
    binders: Vec<DescriptorBinder>,
    pub receipt: S14CausalBlockRatio4BoundaryRecordingReceipt,
}

impl S14CausalBlockRatio4BoundaryRecording {
    pub fn destroy(self, ctx: &VulkanContext) {
        for binder in self.binders {
            binder.destroy(ctx);
        }
    }
}

pub struct S14CausalBlockRatio4BoundaryRecorder {
    attention: ComputePipeline,
    position4_index_query: S14Ratio4IndexQueryPipeline,
    position4_indexer: S14SparseIndexerPipeline,
}

impl S14CausalBlockRatio4BoundaryRecorder {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let attention = ComputePipeline::new(ctx, S14_CAUSAL_BLOCK_RATIO4_BOUNDARY_SPV, 9, 24)?;
        let position4_index_query = match S14Ratio4IndexQueryPipeline::new(ctx) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                attention.destroy(ctx);
                return Err(
                    error.context("创建 causal-block position4 ratio4 index-query pipeline")
                );
            }
        };
        let position4_indexer = match S14SparseIndexerPipeline::new(ctx) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                position4_index_query.destroy(ctx);
                attention.destroy(ctx);
                return Err(error.context("创建 causal-block position4 ratio4 indexer pipeline"));
            }
        };
        Ok(Self {
            attention,
            position4_index_query,
            position4_indexer,
        })
    }

    /// 在同一 production command 内强制正式顺序。rollover 只修改 remainder，不覆盖刚写回
    /// 的 compressed block0/QKV；position4 随后录制 remainder/index-query/indexer，最后一次
    /// K-row attention 中 lane2 直读 block0、lane3 消费真实 sparse index。
    ///
    /// # Safety
    /// 全部资源必须活到 command 完成；sticky status 必须已清零。
    pub unsafe fn record_base1_k4(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        state: &dyn S14CausalBlockRatio4BoundaryStateRecorder,
        state_workspace: S14CausalBlockRatio4StateWorkspaceBindings<'_>,
        attention: S14CausalBlockRatio4AttentionBindings<'_>,
        position4: S14CausalBlockRatio4Position4IndexerBindings<'_>,
    ) -> Result<S14CausalBlockRatio4BoundaryRecording> {
        if command == vk::CommandBuffer::null() {
            bail!("ratio4 causal-block command 不能为空");
        }
        state
            .candidate_state_binding()
            .validate(state.candidate_state_owner())?;
        let finalize = state.record_remainder_and_finalize(ctx, command, state_workspace)?;
        finalize.validate()?;
        compute_to_compute_barrier(ctx, command);

        let mut binders = Vec::with_capacity(3);
        let rollover = match state.record_rollover_after_finalize(ctx, command, state_workspace) {
            Ok(receipt) => receipt,
            Err(error) => {
                destroy_binders(ctx, &mut binders);
                return Err(error.context("录制 ratio4 post-position3-attention rollover"));
            }
        };
        if let Err(error) = rollover.validate() {
            destroy_binders(ctx, &mut binders);
            return Err(error);
        }
        compute_to_compute_barrier(ctx, command);

        let position4_prelude = match state.record_position4_remainder_and_index_head(
            ctx,
            command,
            state_workspace,
        ) {
            Ok(receipt) => receipt,
            Err(error) => {
                destroy_binders(ctx, &mut binders);
                return Err(error.context("录制 ratio4 position4 remainder/index-head"));
            }
        };
        if let Err(error) = position4_prelude.validate() {
            destroy_binders(ctx, &mut binders);
            return Err(error);
        }
        compute_to_compute_barrier(ctx, command);

        let index_query_shape = match S14Ratio4IndexQueryShape::new(4, 4) {
            Ok(shape) => shape,
            Err(error) => {
                destroy_binders(ctx, &mut binders);
                return Err(error);
            }
        };
        let index_query = match self.position4_index_query.bind_slices(
            ctx,
            position4.raw_index_query_bf16,
            position4.rope_f32,
            position4.processed_index_query_bf16,
            position4.sticky_status_u32,
            index_query_shape,
        ) {
            Ok(dispatch) => dispatch,
            Err(error) => {
                destroy_binders(ctx, &mut binders);
                return Err(error.context("绑定 causal-block position4 ratio4 index-query"));
            }
        };
        self.position4_index_query.cmd(ctx, command, &index_query);
        binders.push(index_query.binder);
        compute_to_compute_barrier(ctx, command);

        let indexer = match self.position4_indexer.bind_slices(
            ctx,
            position4.processed_index_query_bf16,
            position4.index_cache_bf16,
            position4.head_weights_bf16,
            position4.index_scores_f32,
            position4.compressed_indices_u32,
            position4.sticky_status_u32,
            1,
        ) {
            Ok(dispatch) => dispatch,
            Err(error) => {
                destroy_binders(ctx, &mut binders);
                return Err(error.context("绑定 causal-block position4 ratio4 indexer"));
            }
        };
        self.position4_indexer.cmd(ctx, command, &indexer);
        binders.push(indexer.binder);
        compute_to_compute_barrier(ctx, command);

        let attention_binder = match bind_attention(ctx, &self.attention, attention) {
            Ok(binder) => binder,
            Err(error) => {
                destroy_binders(ctx, &mut binders);
                return Err(error.context("绑定 causal-block ratio4 K=4 boundary attention"));
            }
        };
        record_attention(ctx, command, &self.attention, &attention_binder);
        binders.push(attention_binder);
        let receipt = S14CausalBlockRatio4BoundaryRecordingReceipt {
            positions: [1, 2, 3, 4],
            boundary_lane: BOUNDARY_LANE,
            compressed_consumer_lanes: [2, 3],
            pre_boundary_remainder_record_calls: finalize.pre_boundary_remainder_record_calls,
            main_finalize_writeback_calls: finalize.main_finalize_writeback_calls,
            indexer_finalize_writeback_calls: finalize.indexer_finalize_writeback_calls,
            rollover_record_calls: rollover.rollover_record_calls,
            position4_remainder_record_calls: position4_prelude.remainder_record_calls,
            position4_index_head_weight_projection_calls: position4_prelude
                .index_head_weight_projection_calls,
            position4_index_query_dispatch_calls: 1,
            position4_indexer_dispatch_calls: 1,
            attention_dispatch_calls: 1,
            attention_rows: BLOCK_SIZE,
            position4_sparse_attention_rows: 1,
            serial_token_forward_calls: 0,
            cpu_fallback_calls: 0,
        };
        if let Err(error) = receipt.validate() {
            destroy_binders(ctx, &mut binders);
            return Err(error);
        }
        Ok(S14CausalBlockRatio4BoundaryRecording { binders, receipt })
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.position4_indexer.destroy(ctx);
        self.position4_index_query.destroy(ctx);
        self.attention.destroy(ctx);
    }
}

fn bind_attention(
    ctx: &VulkanContext,
    pipeline: &ComputePipeline,
    bindings: S14CausalBlockRatio4AttentionBindings<'_>,
) -> Result<DescriptorBinder> {
    DescriptorBinder::new_with_offsets(
        ctx,
        pipeline,
        &[
            range(bindings.query_bf16, BLOCK_SIZE as u64 * QUERY_ROW_BYTES)?,
            range(bindings.committed_window_kv_bf16, KV_ROW_BYTES)?,
            range(
                bindings.current_block_kv_bf16,
                BLOCK_SIZE as u64 * KV_ROW_BYTES,
            )?,
            range(bindings.first_compressed_kv_bf16, KV_ROW_BYTES)?,
            range(bindings.position4_compressed_index_u32, 4)?,
            range(bindings.sink_f32, SINK_BYTES)?,
            range(bindings.rope_f32, BLOCK_SIZE as u64 * ROPE_ROW_BYTES)?,
            range(bindings.output_bf16, BLOCK_SIZE as u64 * QUERY_ROW_BYTES)?,
            range(bindings.sticky_status_u32, STATUS_BYTES)?,
        ],
    )
}

fn range(slice: StorageBufferSlice<'_>, bytes: u64) -> Result<(&crate::GpuBuffer, u64, u64)> {
    if slice
        .offset
        .checked_add(bytes)
        .is_none_or(|end| bytes == 0 || end > slice.buffer.size())
    {
        bail!("ratio4 causal-block attention descriptor range 越界");
    }
    Ok((slice.buffer, slice.offset, bytes))
}

unsafe fn record_attention(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    binder: &DescriptorBinder,
) {
    ctx.device
        .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
    ctx.device.cmd_bind_descriptor_sets(
        command,
        vk::PipelineBindPoint::COMPUTE,
        pipeline.layout,
        0,
        &[binder.set],
        &[],
    );
    let mut push = [0u8; 16];
    for (index, value) in [HEADS, HEAD_DIM, BASE_POSITION, BLOCK_SIZE]
        .into_iter()
        .enumerate()
    {
        push[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    ctx.device.cmd_push_constants(
        command,
        pipeline.layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        &push,
    );
    ctx.device.cmd_dispatch(command, HEADS, BLOCK_SIZE, 1);
}

fn destroy_binders(ctx: &VulkanContext, binders: &mut Vec<DescriptorBinder>) {
    for binder in binders.drain(..) {
        binder.destroy(ctx);
    }
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

    #[test]
    fn base1_k4_plan_crosses_position3_then_runs_position4_sparse() {
        let shape = S14CausalBlockRatio4BoundaryShape::new(1, 4, 4).unwrap();
        assert_eq!(shape.positions(), [1, 2, 3, 4]);
        assert_eq!(shape.boundary_lane(), 2);
        assert_eq!(shape.compressed_consumer_lanes(), [2, 3]);
        let shader = include_str!("../shaders/s14_causal_block_ratio4_boundary_attention.comp");
        assert!(shader.contains("binding = 3"));
        assert!(shader.contains("binding = 4"));
        assert!(shader.contains("block_lane == 2u ? 0u : compressed_index.values[0]"));
        assert!(shader.contains("block_lane >= 2u ? 1u : 0u"));
        S14CausalBlockRatio4BoundaryRecordingReceipt {
            positions: shape.positions(),
            boundary_lane: shape.boundary_lane(),
            compressed_consumer_lanes: shape.compressed_consumer_lanes(),
            pre_boundary_remainder_record_calls: 3,
            main_finalize_writeback_calls: 1,
            indexer_finalize_writeback_calls: 1,
            rollover_record_calls: 1,
            position4_remainder_record_calls: 1,
            position4_index_head_weight_projection_calls: 1,
            position4_index_query_dispatch_calls: 1,
            position4_indexer_dispatch_calls: 1,
            attention_dispatch_calls: 1,
            attention_rows: 4,
            position4_sparse_attention_rows: 1,
            serial_token_forward_calls: 0,
            cpu_fallback_calls: 0,
        }
        .validate()
        .unwrap();
        for invalid in [(0, 4, 4), (1, 8, 4), (1, 4, 0)] {
            assert!(
                S14CausalBlockRatio4BoundaryShape::new(invalid.0, invalid.1, invalid.2).is_err()
            );
        }
    }
}
