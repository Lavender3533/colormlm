//! Ratio4 compressor block-boundary Vulkan primitives.
//!
//! The fixed-revision reference closes its first overlap block at position 3:
//! gate-softmax pooling -> BF16 -> RMSNorm -> compressed RoPE, followed by
//! group64 E4M3 QDQ for main KV or normalized Hadamard + group32 E2M1 QDQ for
//! indexer KV.  This module owns only those numerical primitives.  Candidate
//! state mirroring, cache append and attention consumption remain orchestration
//! responsibilities and therefore stay fail-closed elsewhere.

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::s14_bf16_rmsnorm::{S14Bf16RmsNormPipeline, S14Bf16RmsNormShape};
use crate::s14_position1_attention::position_rope_cos_sin;
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_RATIO4: u32 = 4;
pub const S14_RATIO4_STATE_ROWS: u32 = 8;
pub const S14_RATIO4_MAIN_WIDTH: u32 = 512;
pub const S14_RATIO4_INDEXER_WIDTH: u32 = 128;
pub const S14_RATIO4_MAIN_PROJECTED: u32 = 1024;
pub const S14_RATIO4_INDEXER_PROJECTED: u32 = 256;
pub const S14_RATIO4_RMS_EPSILON: f32 = 1.0e-6;
pub const S14_RATIO4_ROPE_SCALARS: u32 = 64;

pub const S14_RATIO4_FINALIZE_STATUS_NON_FINITE_INPUT: u32 = 1;
pub const S14_RATIO4_FINALIZE_STATUS_INVALID_INTERMEDIATE: u32 = 2;
pub const S14_RATIO4_FINALIZE_STATUS_NON_FINITE_OUTPUT: u32 = 4;
pub const S14_RATIO4_FINALIZE_STATUS_NON_FINITE_BF16: u32 = 8;

pub const S14_RATIO4_COMPRESSOR_POOL_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_ratio4_compressor_pool.spv"));
pub const S14_RATIO4_COMPRESSED_ROPE_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_ratio4_compressed_rope.spv"));
pub const S14_RATIO4_MAIN_QDQ_BF16_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_ratio4_main_qdq_bf16.spv"));
pub const S14_RATIO4_INDEXER_HADAMARD_QDQ_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_ratio4_indexer_hadamard_qdq.spv"
));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Ratio4CompressorKind {
    Main,
    Indexer,
}

impl S14Ratio4CompressorKind {
    pub const fn width(self) -> u32 {
        match self {
            Self::Main => S14_RATIO4_MAIN_WIDTH,
            Self::Indexer => S14_RATIO4_INDEXER_WIDTH,
        }
    }

    pub const fn projected(self) -> u32 {
        match self {
            Self::Main => S14_RATIO4_MAIN_PROJECTED,
            Self::Indexer => S14_RATIO4_INDEXER_PROJECTED,
        }
    }

    pub const fn state_f32_bytes(self) -> u64 {
        S14_RATIO4_STATE_ROWS as u64 * self.projected() as u64 * 4
    }

    pub const fn row_bf16_bytes(self) -> u64 {
        self.width() as u64 * 2
    }

    pub fn rmsnorm_shape(self) -> Result<S14Bf16RmsNormShape> {
        S14Bf16RmsNormShape::new(1, self.width())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Ratio4CompressorBoundary {
    pub position: u32,
    pub compressed_position: u32,
    pub cache_index: u32,
}

impl S14Ratio4CompressorBoundary {
    pub fn new(position: u32) -> Result<Self> {
        let next = position
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("ratio4 compressor position overflow"))?;
        if next < S14_RATIO4 || next % S14_RATIO4 != 0 {
            bail!("ratio4 compressor finalize requires position+1 divisible by 4 and position>=3");
        }
        Ok(Self {
            position,
            compressed_position: next - S14_RATIO4,
            cache_index: position / S14_RATIO4,
        })
    }

    /// Fixed-revision compressed RoPE table for the newly finalized block.
    pub fn rope_cos_sin(self) -> Result<[f32; 64]> {
        position_rope_cos_sin(self.compressed_position, S14_RATIO4)
    }
}

pub struct S14Ratio4CompressorPoolPipeline {
    pipeline: ComputePipeline,
}

pub struct S14Ratio4CompressorPoolDispatch {
    pub binder: DescriptorBinder,
    pub kind: S14Ratio4CompressorKind,
}

impl S14Ratio4CompressorPoolPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_RATIO4_COMPRESSOR_POOL_SPV, 4, 4)?,
        })
    }

    pub fn bind(
        &self,
        ctx: &VulkanContext,
        kind: S14Ratio4CompressorKind,
        kv_state: &GpuBuffer,
        score_state: &GpuBuffer,
        output: &GpuBuffer,
        status: &GpuBuffer,
    ) -> Result<S14Ratio4CompressorPoolDispatch> {
        self.bind_slices(
            ctx,
            kind,
            StorageBufferSlice::whole(kv_state),
            StorageBufferSlice::whole(score_state),
            StorageBufferSlice::whole(output),
            StorageBufferSlice::whole(status),
        )
    }

    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        kind: S14Ratio4CompressorKind,
        kv_state: StorageBufferSlice<'_>,
        score_state: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14Ratio4CompressorPoolDispatch> {
        let state_bytes = kind.state_f32_bytes();
        let output_bytes = kind.row_bf16_bytes();
        validate_distinct_slices(
            "ratio4 compressor pool",
            &[
                (kv_state, state_bytes),
                (score_state, state_bytes),
                (output, output_bytes),
                (status, 4),
            ],
        )?;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (kv_state.buffer, kv_state.offset, state_bytes),
                (score_state.buffer, score_state.offset, state_bytes),
                (output.buffer, output.offset, output_bytes),
                (status.buffer, status.offset, 4),
            ],
        )?;
        Ok(S14Ratio4CompressorPoolDispatch { binder, kind })
    }

    /// # Safety
    /// Resources must outlive command completion.  The caller owns barriers
    /// between pool, RMSNorm, RoPE and QDQ stages.
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Ratio4CompressorPoolDispatch,
    ) {
        bind_pipeline(ctx, command, &self.pipeline, dispatch.binder.set);
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &dispatch.kind.width().to_le_bytes(),
        );
        let pairs = dispatch.kind.width() / 2;
        ctx.device.cmd_dispatch(command, pairs.div_ceil(64), 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub struct S14Ratio4CompressedRopePipeline {
    pipeline: ComputePipeline,
}

pub struct S14Ratio4CompressedRopeDispatch {
    pub binder: DescriptorBinder,
    pub kind: S14Ratio4CompressorKind,
}

impl S14Ratio4CompressedRopePipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_RATIO4_COMPRESSED_ROPE_SPV, 4, 4)?,
        })
    }

    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        kind: S14Ratio4CompressorKind,
        input: StorageBufferSlice<'_>,
        rope: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14Ratio4CompressedRopeDispatch> {
        let row_bytes = kind.row_bf16_bytes();
        let rope_bytes = S14_RATIO4_ROPE_SCALARS as u64 * 4;
        validate_distinct_slices(
            "ratio4 compressed RoPE",
            &[
                (input, row_bytes),
                (rope, rope_bytes),
                (output, row_bytes),
                (status, 4),
            ],
        )?;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (input.buffer, input.offset, row_bytes),
                (rope.buffer, rope.offset, rope_bytes),
                (output.buffer, output.offset, row_bytes),
                (status.buffer, status.offset, 4),
            ],
        )?;
        Ok(S14Ratio4CompressedRopeDispatch { binder, kind })
    }

    /// # Safety
    /// Resources must outlive command completion; the caller inserts barriers.
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Ratio4CompressedRopeDispatch,
    ) {
        bind_pipeline(ctx, command, &self.pipeline, dispatch.binder.set);
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &dispatch.kind.width().to_le_bytes(),
        );
        ctx.device
            .cmd_dispatch(command, (dispatch.kind.width() / 2).div_ceil(64), 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub struct S14Ratio4MainQdqBf16Pipeline {
    pipeline: ComputePipeline,
}

pub struct S14Ratio4MainQdqBf16Dispatch {
    pub binder: DescriptorBinder,
}

impl S14Ratio4MainQdqBf16Pipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_RATIO4_MAIN_QDQ_BF16_SPV, 3, 0)?,
        })
    }

    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        input: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14Ratio4MainQdqBf16Dispatch> {
        let bytes = S14Ratio4CompressorKind::Main.row_bf16_bytes();
        validate_distinct_slices(
            "ratio4 main QDQ BF16",
            &[(input, bytes), (output, bytes), (status, 4)],
        )?;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (input.buffer, input.offset, bytes),
                (output.buffer, output.offset, bytes),
                (status.buffer, status.offset, 4),
            ],
        )?;
        Ok(S14Ratio4MainQdqBf16Dispatch { binder })
    }

    /// # Safety
    /// Resources must outlive command completion; the caller inserts barriers.
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Ratio4MainQdqBf16Dispatch,
    ) {
        bind_pipeline(ctx, command, &self.pipeline, dispatch.binder.set);
        ctx.device.cmd_dispatch(command, 8, 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub struct S14Ratio4IndexerHadamardQdqPipeline {
    pipeline: ComputePipeline,
}

pub struct S14Ratio4IndexerHadamardQdqDispatch {
    pub binder: DescriptorBinder,
}

impl S14Ratio4IndexerHadamardQdqPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_RATIO4_INDEXER_HADAMARD_QDQ_SPV, 3, 0)?,
        })
    }

    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        input: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14Ratio4IndexerHadamardQdqDispatch> {
        let bytes = S14Ratio4CompressorKind::Indexer.row_bf16_bytes();
        validate_distinct_slices(
            "ratio4 indexer Hadamard QDQ",
            &[(input, bytes), (output, bytes), (status, 4)],
        )?;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (input.buffer, input.offset, bytes),
                (output.buffer, output.offset, bytes),
                (status.buffer, status.offset, 4),
            ],
        )?;
        Ok(S14Ratio4IndexerHadamardQdqDispatch { binder })
    }

    /// # Safety
    /// Resources must outlive command completion; the caller inserts barriers.
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Ratio4IndexerHadamardQdqDispatch,
    ) {
        bind_pipeline(ctx, command, &self.pipeline, dispatch.binder.set);
        ctx.device.cmd_dispatch(command, 1, 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

/// The complete primitive set needed by the position3 ratio4 finalize path.
pub struct S14Ratio4CompressorFinalizePipelines {
    pub pool: S14Ratio4CompressorPoolPipeline,
    pub rmsnorm: S14Bf16RmsNormPipeline,
    pub rope: S14Ratio4CompressedRopePipeline,
    pub main_qdq_bf16: S14Ratio4MainQdqBf16Pipeline,
    pub indexer_hadamard_qdq: S14Ratio4IndexerHadamardQdqPipeline,
}

impl S14Ratio4CompressorFinalizePipelines {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let pool = S14Ratio4CompressorPoolPipeline::new(ctx)?;
        let rmsnorm = match S14Bf16RmsNormPipeline::new(ctx) {
            Ok(value) => value,
            Err(error) => {
                pool.destroy(ctx);
                return Err(error);
            }
        };
        let rope = match S14Ratio4CompressedRopePipeline::new(ctx) {
            Ok(value) => value,
            Err(error) => {
                rmsnorm.destroy(ctx);
                pool.destroy(ctx);
                return Err(error);
            }
        };
        let main_qdq_bf16 = match S14Ratio4MainQdqBf16Pipeline::new(ctx) {
            Ok(value) => value,
            Err(error) => {
                rope.destroy(ctx);
                rmsnorm.destroy(ctx);
                pool.destroy(ctx);
                return Err(error);
            }
        };
        let indexer_hadamard_qdq = match S14Ratio4IndexerHadamardQdqPipeline::new(ctx) {
            Ok(value) => value,
            Err(error) => {
                main_qdq_bf16.destroy(ctx);
                rope.destroy(ctx);
                rmsnorm.destroy(ctx);
                pool.destroy(ctx);
                return Err(error);
            }
        };
        Ok(Self {
            pool,
            rmsnorm,
            rope,
            main_qdq_bf16,
            indexer_hadamard_qdq,
        })
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.indexer_hadamard_qdq.destroy(ctx);
        self.main_qdq_bf16.destroy(ctx);
        self.rope.destroy(ctx);
        self.rmsnorm.destroy(ctx);
        self.pool.destroy(ctx);
    }
}

pub fn validate_ratio4_compressor_finalize_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let known = S14_RATIO4_FINALIZE_STATUS_NON_FINITE_INPUT
        | S14_RATIO4_FINALIZE_STATUS_INVALID_INTERMEDIATE
        | S14_RATIO4_FINALIZE_STATUS_NON_FINITE_OUTPUT
        | S14_RATIO4_FINALIZE_STATUS_NON_FINITE_BF16;
    if code & !known != 0 {
        bail!("ratio4 compressor finalize returned unknown status bits 0x{code:08x}");
    }
    bail!("ratio4 compressor finalize rejected candidate, status=0x{code:08x}")
}

fn validate_distinct_slices(label: &str, slices: &[(StorageBufferSlice<'_>, u64)]) -> Result<()> {
    for left in 0..slices.len() {
        for right in left + 1..slices.len() {
            if storage_buffer_slices_overlap(
                slices[left].0,
                slices[left].1,
                slices[right].0,
                slices[right].1,
            )? {
                bail!("{label} resources must not alias");
            }
        }
    }
    Ok(())
}

unsafe fn bind_pipeline(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
) {
    ctx.device
        .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
    ctx.device.cmd_bind_descriptor_sets(
        command,
        vk::PipelineBindPoint::COMPUTE,
        pipeline.layout,
        0,
        &[set],
        &[],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use polaris_s14_runner::bf16_round_trip;

    #[test]
    fn position3_boundary_binds_first_compressed_cache_row_and_identity_rope() {
        let boundary = S14Ratio4CompressorBoundary::new(3).unwrap();
        assert_eq!(boundary.compressed_position, 0);
        assert_eq!(boundary.cache_index, 0);
        let rope = boundary.rope_cos_sin().unwrap();
        for pair in rope.chunks_exact(2) {
            assert_eq!(pair, [1.0, 0.0]);
        }
        let second = S14Ratio4CompressorBoundary::new(7).unwrap();
        assert_eq!(second.compressed_position, 4);
        assert_eq!(second.cache_index, 1);
        for invalid in [0, 1, 2, 4, 5, u32::MAX] {
            assert!(S14Ratio4CompressorBoundary::new(invalid).is_err());
        }
    }

    #[test]
    fn ratio4_shapes_match_executor_overlap_states_and_outputs() {
        let main = S14Ratio4CompressorKind::Main;
        assert_eq!(main.width(), 512);
        assert_eq!(main.projected(), 1024);
        assert_eq!(main.state_f32_bytes(), 32_768);
        assert_eq!(main.row_bf16_bytes(), 1_024);
        assert_eq!(main.rmsnorm_shape().unwrap().hidden, 512);

        let indexer = S14Ratio4CompressorKind::Indexer;
        assert_eq!(indexer.width(), 128);
        assert_eq!(indexer.projected(), 256);
        assert_eq!(indexer.state_f32_bytes(), 8_192);
        assert_eq!(indexer.row_bf16_bytes(), 256);
        assert_eq!(indexer.rmsnorm_shape().unwrap().hidden, 128);
    }

    #[test]
    fn first_overlap_pool_ignores_negative_infinity_prefix_and_rne_rounds() {
        let scores = [
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let values = [99.0, 99.0, 99.0, 99.0, 1.0, 2.0, 3.0, 4.0];
        let pooled = reference_pool_dimension(&values, &scores);
        assert_eq!(pooled, 2.5);
        assert_eq!(bf16_round_trip(pooled).unwrap(), 2.5);
        assert_eq!(
            reference_pool_dimension(&[1.0; 8], &[f32::NEG_INFINITY; 8]),
            0.0
        );
    }

    #[test]
    fn normalized_hadamard_then_e2m1_qdq_has_official_block32_boundary() {
        let mut basis = [0.0f32; 128];
        basis[0] = 8.0;
        let output = reference_indexer_finalize(basis);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(output[0], bf16_round_trip(output[0]).unwrap());
    }

    #[test]
    fn aggregate_status_is_sticky_fail_closed() {
        validate_ratio4_compressor_finalize_status(0).unwrap();
        for code in [1, 2, 4, 8, 15, 16] {
            assert!(validate_ratio4_compressor_finalize_status(code).is_err());
        }
    }

    fn reference_pool_dimension(values: &[f32; 8], scores: &[f32; 8]) -> f32 {
        let maximum = scores
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        if !maximum.is_finite() {
            return 0.0;
        }
        let denominator: f32 = scores
            .iter()
            .map(|score| {
                if score.is_finite() {
                    (*score - maximum).exp()
                } else {
                    0.0
                }
            })
            .sum();
        values
            .iter()
            .zip(scores)
            .map(|(value, score)| {
                if score.is_finite() {
                    *value * ((*score - maximum).exp() / denominator)
                } else {
                    0.0
                }
            })
            .sum()
    }

    fn reference_indexer_finalize(mut values: [f32; 128]) -> [f32; 128] {
        let mut step = 1usize;
        while step < values.len() {
            for base in (0..values.len()).step_by(step * 2) {
                for offset in 0..step {
                    let left = values[base + offset];
                    let right = values[base + step + offset];
                    values[base + offset] = left + right;
                    values[base + step + offset] = left - right;
                }
            }
            step *= 2;
        }
        for value in &mut values {
            *value = bf16_round_trip(*value * (128.0f32).sqrt().recip()).unwrap();
        }
        const LEVELS: [f32; 15] = [
            -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        ];
        for block in values.chunks_exact_mut(32) {
            let amax = block.iter().copied().map(f32::abs).fold(0.0, f32::max);
            let amax = amax.max(6.0 * 2.0f32.powi(-126));
            let scale = 2.0f32.powf((amax / 6.0).log2().ceil());
            for value in block {
                let normalized = (*value / scale).clamp(-6.0, 6.0);
                let quantized = LEVELS
                    .iter()
                    .copied()
                    .min_by(|left, right| {
                        (normalized - *left)
                            .abs()
                            .total_cmp(&(normalized - *right).abs())
                    })
                    .unwrap();
                *value = bf16_round_trip(quantized * scale).unwrap();
            }
        }
        values
    }
}
