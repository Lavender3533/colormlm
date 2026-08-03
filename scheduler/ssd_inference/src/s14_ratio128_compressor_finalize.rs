//! Ratio128 compressor block-boundary Vulkan primitive.
//!
//! Ratio128 is non-overlap: position 127,255,... pools F32[128,512] KV and
//! scores into one BF16[512] row. RMSNorm, compressed RoPE and group64 main
//! QDQ reuse the already closed generic 512-wide primitives from the ratio4
//! chain; this module owns the distinct 128-row pooling contract.

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::s14_bf16_rmsnorm::{S14Bf16RmsNormPipeline, S14Bf16RmsNormShape};
use crate::s14_position1_attention::position_rope_cos_sin;
use crate::s14_ratio4_compressor_finalize::{
    S14Ratio4CompressedRopePipeline, S14Ratio4MainQdqBf16Pipeline,
};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_RATIO128: u32 = 128;
pub const S14_RATIO128_STATE_ROWS: u32 = 128;
pub const S14_RATIO128_WIDTH: u32 = 512;
pub const S14_RATIO128_STATE_F32_BYTES: u64 =
    S14_RATIO128_STATE_ROWS as u64 * S14_RATIO128_WIDTH as u64 * 4;
pub const S14_RATIO128_ROW_BF16_BYTES: u64 = S14_RATIO128_WIDTH as u64 * 2;
pub const S14_RATIO128_RMS_EPSILON: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Ratio128ProductionFinalizeStage {
    Pool,
    RmsNorm,
    CompressedRope,
    MainQdq,
    InactiveStateWrite,
}

/// Production recorder 必须保持的数值和发布顺序。QDQ 直接以 inactive
/// candidate cache 行为输出；最后一项表示该行进入 device dirty ledger，绝不表示提交。
pub const S14_RATIO128_PRODUCTION_FINALIZE_STAGES: [S14Ratio128ProductionFinalizeStage; 5] = [
    S14Ratio128ProductionFinalizeStage::Pool,
    S14Ratio128ProductionFinalizeStage::RmsNorm,
    S14Ratio128ProductionFinalizeStage::CompressedRope,
    S14Ratio128ProductionFinalizeStage::MainQdq,
    S14Ratio128ProductionFinalizeStage::InactiveStateWrite,
];

pub const S14_RATIO128_COMPRESSOR_POOL_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_ratio128_compressor_pool.spv"
));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Ratio128CompressorBoundary {
    pub position: u32,
    pub compressed_position: u32,
    pub cache_index: u32,
}

impl S14Ratio128CompressorBoundary {
    pub fn new(position: u32) -> Result<Self> {
        let next = position
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("ratio128 compressor position overflow"))?;
        if next < S14_RATIO128 || next % S14_RATIO128 != 0 {
            bail!(
                "ratio128 compressor finalize requires position+1 divisible by 128 and position>=127"
            );
        }
        Ok(Self {
            position,
            compressed_position: next - S14_RATIO128,
            cache_index: position / S14_RATIO128,
        })
    }

    pub fn rope_cos_sin(self) -> Result<[f32; 64]> {
        position_rope_cos_sin(self.compressed_position, S14_RATIO128)
    }

    pub fn rmsnorm_shape(self) -> Result<S14Bf16RmsNormShape> {
        S14Bf16RmsNormShape::new(1, S14_RATIO128_WIDTH)
    }
}

pub struct S14Ratio128CompressorPoolPipeline {
    pipeline: ComputePipeline,
}

pub struct S14Ratio128CompressorPoolDispatch {
    pub binder: DescriptorBinder,
}

impl S14Ratio128CompressorPoolPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_RATIO128_COMPRESSOR_POOL_SPV, 4, 0)?,
        })
    }

    pub fn bind(
        &self,
        ctx: &VulkanContext,
        kv_state: &GpuBuffer,
        score_state: &GpuBuffer,
        output: &GpuBuffer,
        status: &GpuBuffer,
    ) -> Result<S14Ratio128CompressorPoolDispatch> {
        self.bind_slices(
            ctx,
            StorageBufferSlice::whole(kv_state),
            StorageBufferSlice::whole(score_state),
            StorageBufferSlice::whole(output),
            StorageBufferSlice::whole(status),
        )
    }

    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        kv_state: StorageBufferSlice<'_>,
        score_state: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14Ratio128CompressorPoolDispatch> {
        let resources = [
            (kv_state, S14_RATIO128_STATE_F32_BYTES),
            (score_state, S14_RATIO128_STATE_F32_BYTES),
            (output, S14_RATIO128_ROW_BF16_BYTES),
            (status, 4),
        ];
        for left in 0..resources.len() {
            for right in left + 1..resources.len() {
                if storage_buffer_slices_overlap(
                    resources[left].0,
                    resources[left].1,
                    resources[right].0,
                    resources[right].1,
                )? {
                    bail!("ratio128 compressor pool resources must not alias");
                }
            }
        }
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (
                    kv_state.buffer,
                    kv_state.offset,
                    S14_RATIO128_STATE_F32_BYTES,
                ),
                (
                    score_state.buffer,
                    score_state.offset,
                    S14_RATIO128_STATE_F32_BYTES,
                ),
                (output.buffer, output.offset, S14_RATIO128_ROW_BF16_BYTES),
                (status.buffer, status.offset, 4),
            ],
        )?;
        Ok(S14Ratio128CompressorPoolDispatch { binder })
    }

    /// # Safety
    /// Bound resources must outlive command completion; the caller owns the
    /// barrier before RMSNorm and rejects a non-zero sticky status.
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Ratio128CompressorPoolDispatch,
    ) {
        ctx.device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.pipeline,
        );
        ctx.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.layout,
            0,
            &[dispatch.binder.set],
            &[],
        );
        ctx.device.cmd_dispatch(command, 4, 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

/// position127/255/... production finalize 所需的完整 pipeline owner。
/// Rope 与 main group64 QDQ 和 ratio4 main 路径数值合同相同，因此复用已闭合原语；
/// 只有 128-row non-overlap pooling 是 ratio128 独有实现。
pub struct S14Ratio128CompressorFinalizePipelines {
    pub pool: S14Ratio128CompressorPoolPipeline,
    pub rmsnorm: S14Bf16RmsNormPipeline,
    pub rope: S14Ratio4CompressedRopePipeline,
    pub main_qdq_bf16: S14Ratio4MainQdqBf16Pipeline,
}

impl S14Ratio128CompressorFinalizePipelines {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let pool = S14Ratio128CompressorPoolPipeline::new(ctx)?;
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
        Ok(Self {
            pool,
            rmsnorm,
            rope,
            main_qdq_bf16,
        })
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.main_qdq_bf16.destroy(ctx);
        self.rope.destroy(ctx);
        self.rmsnorm.destroy(ctx);
        self.pool.destroy(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio128_boundaries_and_shapes_match_native_state() {
        let first = S14Ratio128CompressorBoundary::new(127).unwrap();
        assert_eq!(first.compressed_position, 0);
        assert_eq!(first.cache_index, 0);
        assert!(first
            .rope_cos_sin()
            .unwrap()
            .chunks_exact(2)
            .all(|pair| pair == [1.0, 0.0]));
        let second = S14Ratio128CompressorBoundary::new(255).unwrap();
        assert_eq!(second.compressed_position, 128);
        assert_eq!(second.cache_index, 1);
        assert_eq!(S14_RATIO128_STATE_F32_BYTES, 262_144);
        assert_eq!(S14_RATIO128_ROW_BF16_BYTES, 1_024);
        for invalid in [0, 3, 126, 128, 254, u32::MAX] {
            assert!(S14Ratio128CompressorBoundary::new(invalid).is_err());
        }
    }
}
