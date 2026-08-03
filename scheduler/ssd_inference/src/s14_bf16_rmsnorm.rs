//! FullDepth43 通用 BF16 RMSNorm Vulkan 数值边界。
//!
//! 支持生产路径中的 `1×4096` 主隐藏态及 `64×512` Q-head。输入和权重为
//! BF16，平方和、归一化及缩放为 F32，输出再执行 RNE BF16 边界。状态在一个
//! WholeTokenCandidate 内 sticky，调用方只可在候选开始前清零并在提交前校验。

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_BF16_RMSNORM_MAX_ROWS: u32 = 64;
pub const S14_BF16_RMSNORM_MAX_HIDDEN: u32 = 4096;
pub const S14_BF16_RMSNORM_STATUS_NON_FINITE_INPUT: u32 = 1;
pub const S14_BF16_RMSNORM_STATUS_NON_FINITE_NORM: u32 = 2;
pub const S14_BF16_RMSNORM_STATUS_NON_FINITE_OUTPUT: u32 = 4;
pub const S14_BF16_RMSNORM_STATUS_NON_FINITE_BF16: u32 = 8;

pub const S14_BF16_RMSNORM_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_bf16_rmsnorm.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Bf16RmsNormShape {
    pub rows: u32,
    pub hidden: u32,
}

impl S14Bf16RmsNormShape {
    pub fn new(rows: u32, hidden: u32) -> Result<Self> {
        if rows == 0 || rows > S14_BF16_RMSNORM_MAX_ROWS {
            bail!("S14 BF16 RMSNorm rows must be in 1..={S14_BF16_RMSNORM_MAX_ROWS}");
        }
        if hidden == 0 || hidden > S14_BF16_RMSNORM_MAX_HIDDEN || hidden % 2 != 0 {
            bail!("S14 BF16 RMSNorm hidden must be even and in 1..={S14_BF16_RMSNORM_MAX_HIDDEN}");
        }
        let shape = Self { rows, hidden };
        shape.input_bf16_bytes()?;
        shape.weight_bf16_bytes()?;
        shape.output_bf16_bytes()?;
        Ok(shape)
    }

    pub fn scalar_count(self) -> Result<u64> {
        (self.rows as u64)
            .checked_mul(self.hidden as u64)
            .ok_or_else(|| anyhow::anyhow!("S14 BF16 RMSNorm shape overflow"))
    }

    pub fn input_bf16_bytes(self) -> Result<u64> {
        checked_bytes(self.scalar_count()?, 2, "S14 BF16 RMSNorm input")
    }

    pub fn weight_bf16_bytes(self) -> Result<u64> {
        checked_bytes(self.hidden as u64, 2, "S14 BF16 RMSNorm weight")
    }

    pub fn output_bf16_bytes(self) -> Result<u64> {
        checked_bytes(self.scalar_count()?, 2, "S14 BF16 RMSNorm output")
    }

    pub fn inverse_rms_f32_bytes(self) -> Result<u64> {
        checked_bytes(self.rows as u64, 4, "S14 BF16 RMSNorm inverse RMS")
    }

    pub fn status_bytes(self) -> u64 {
        4
    }
}

pub struct S14Bf16RmsNormPipeline {
    pipeline: ComputePipeline,
}

pub struct S14Bf16RmsNormDispatch {
    pub binder: DescriptorBinder,
    pub shape: S14Bf16RmsNormShape,
    pub epsilon: f32,
}

impl S14Bf16RmsNormPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_BF16_RMSNORM_SPV, 5, 16)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        &self,
        ctx: &VulkanContext,
        shape: S14Bf16RmsNormShape,
        epsilon: f32,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        inverse_rms: &GpuBuffer,
        output: &GpuBuffer,
        status: &GpuBuffer,
    ) -> Result<S14Bf16RmsNormDispatch> {
        self.bind_slices(
            ctx,
            shape,
            epsilon,
            StorageBufferSlice::whole(input),
            StorageBufferSlice::whole(weight),
            StorageBufferSlice::whole(inverse_rms),
            StorageBufferSlice::whole(output),
            StorageBufferSlice::whole(status),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        shape: S14Bf16RmsNormShape,
        epsilon: f32,
        input: StorageBufferSlice<'_>,
        weight: StorageBufferSlice<'_>,
        inverse_rms: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14Bf16RmsNormDispatch> {
        let shape = S14Bf16RmsNormShape::new(shape.rows, shape.hidden)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            bail!("S14 BF16 RMSNorm epsilon must be finite and positive");
        }
        let input_bytes = shape.input_bf16_bytes()?;
        let weight_bytes = shape.weight_bf16_bytes()?;
        let inverse_bytes = shape.inverse_rms_f32_bytes()?;
        let output_bytes = shape.output_bf16_bytes()?;
        let status_bytes = shape.status_bytes();
        for (candidate, bytes) in [
            (input, input_bytes),
            (weight, weight_bytes),
            (inverse_rms, inverse_bytes),
            (status, status_bytes),
        ] {
            if storage_buffer_slices_overlap(output, output_bytes, candidate, bytes)? {
                bail!(
                    "S14 BF16 RMSNorm output must not alias input, weight, inverse RMS or status"
                );
            }
        }
        for (candidate, bytes) in [(input, input_bytes), (weight, weight_bytes)] {
            if storage_buffer_slices_overlap(status, status_bytes, candidate, bytes)? {
                bail!("S14 BF16 RMSNorm status must not alias input or weight");
            }
        }
        for (candidate, bytes) in [
            (input, input_bytes),
            (weight, weight_bytes),
            (status, status_bytes),
        ] {
            if storage_buffer_slices_overlap(inverse_rms, inverse_bytes, candidate, bytes)? {
                bail!("S14 BF16 RMSNorm inverse RMS must not alias input, weight, or status");
            }
        }
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (input.buffer, input.offset, input_bytes),
                (weight.buffer, weight.offset, weight_bytes),
                (inverse_rms.buffer, inverse_rms.offset, inverse_bytes),
                (output.buffer, output.offset, output_bytes),
                (status.buffer, status.offset, status_bytes),
            ],
        )?;
        Ok(S14Bf16RmsNormDispatch {
            binder,
            shape,
            epsilon,
        })
    }

    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Bf16RmsNormDispatch,
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
        let mut push = [0u8; 16];
        push[..4].copy_from_slice(&dispatch.shape.rows.to_le_bytes());
        push[4..8].copy_from_slice(&dispatch.shape.hidden.to_le_bytes());
        push[8..12].copy_from_slice(&dispatch.epsilon.to_le_bytes());
        push[12..].copy_from_slice(&0u32.to_le_bytes());
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push,
        );
        ctx.device.cmd_dispatch(command, dispatch.shape.rows, 1, 1);

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
        push[12..].copy_from_slice(&1u32.to_le_bytes());
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push,
        );
        let pairs = dispatch.shape.rows * dispatch.shape.hidden / 2;
        ctx.device.cmd_dispatch(command, pairs.div_ceil(256), 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub fn validate_bf16_rmsnorm_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let known = S14_BF16_RMSNORM_STATUS_NON_FINITE_INPUT
        | S14_BF16_RMSNORM_STATUS_NON_FINITE_NORM
        | S14_BF16_RMSNORM_STATUS_NON_FINITE_OUTPUT
        | S14_BF16_RMSNORM_STATUS_NON_FINITE_BF16;
    if code & !known != 0 {
        bail!("S14 BF16 RMSNorm returned unknown status bits 0x{code:08x}");
    }
    bail!("S14 BF16 RMSNorm rejected non-finite state, status=0x{code:08x}")
}

fn checked_bytes(elements: u64, element_bytes: u64, name: &str) -> Result<u64> {
    elements
        .checked_mul(element_bytes)
        .ok_or_else(|| anyhow::anyhow!("{name} byte size overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_shapes_cover_hidden_and_q_heads() {
        let hidden = S14Bf16RmsNormShape::new(1, 4096).unwrap();
        assert_eq!(hidden.input_bf16_bytes().unwrap(), 8192);
        assert_eq!(hidden.output_bf16_bytes().unwrap(), 8192);
        assert_eq!(hidden.inverse_rms_f32_bytes().unwrap(), 4);
        let q_heads = S14Bf16RmsNormShape::new(64, 512).unwrap();
        assert_eq!(q_heads.input_bf16_bytes().unwrap(), 65_536);
        assert_eq!(q_heads.output_bf16_bytes().unwrap(), 65_536);
        assert_eq!(q_heads.inverse_rms_f32_bytes().unwrap(), 256);
    }

    #[test]
    fn rmsnorm_shape_and_status_fail_closed() {
        for (rows, hidden) in [(0, 512), (65, 512), (1, 0), (1, 3), (1, 4098)] {
            assert!(S14Bf16RmsNormShape::new(rows, hidden).is_err());
        }
        validate_bf16_rmsnorm_status(0).unwrap();
        for code in [1, 2, 4, 8, 15, 16] {
            assert!(validate_bf16_rmsnorm_status(code).is_err());
        }
    }
}
