//! DeepSeek-V4 Q-head 的 PyTorch-BF16 归一化 Vulkan 原语。
//!
//! 这是专用的 `64×512` 数值合同，不能用通用 F32 RMSNorm 代替。参考路径在
//! square、mean、epsilon add、rsqrt 与乘法之后都保留 BF16 RNE 边界。

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_Q_HEAD_ROWS: u32 = 64;
pub const S14_Q_HEAD_HIDDEN: u32 = 512;
pub const S14_Q_HEAD_STATUS_NON_FINITE_INPUT: u32 = 1;
pub const S14_Q_HEAD_STATUS_NON_FINITE_NORM: u32 = 2;
pub const S14_Q_HEAD_STATUS_NON_FINITE_OUTPUT: u32 = 4;
pub const S14_Q_HEAD_STATUS_NON_FINITE_BF16: u32 = 8;

pub const S14_BF16_Q_HEAD_NORMALIZE_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_bf16_q_head_normalize.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Bf16QHeadNormalizeShape;

impl S14Bf16QHeadNormalizeShape {
    pub const fn new() -> Self {
        Self
    }

    pub const fn rows(self) -> u32 {
        S14_Q_HEAD_ROWS
    }

    pub const fn hidden(self) -> u32 {
        S14_Q_HEAD_HIDDEN
    }

    pub const fn scalar_count(self) -> u64 {
        S14_Q_HEAD_ROWS as u64 * S14_Q_HEAD_HIDDEN as u64
    }

    pub const fn input_bf16_bytes(self) -> u64 {
        self.scalar_count() * 2
    }

    pub const fn inverse_rms_f32_bytes(self) -> u64 {
        S14_Q_HEAD_ROWS as u64 * 4
    }

    pub const fn output_bf16_bytes(self) -> u64 {
        self.input_bf16_bytes()
    }

    pub const fn status_bytes(self) -> u64 {
        4
    }
}

impl Default for S14Bf16QHeadNormalizeShape {
    fn default() -> Self {
        Self::new()
    }
}

pub struct S14Bf16QHeadNormalizePipeline {
    pipeline: ComputePipeline,
}

pub struct S14Bf16QHeadNormalizeDispatch {
    pub binder: DescriptorBinder,
    pub shape: S14Bf16QHeadNormalizeShape,
    pub epsilon: f32,
}

impl S14Bf16QHeadNormalizePipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_BF16_Q_HEAD_NORMALIZE_SPV, 4, 16)?,
        })
    }

    pub fn bind(
        &self,
        ctx: &VulkanContext,
        epsilon: f32,
        input: &GpuBuffer,
        inverse_rms: &GpuBuffer,
        output: &GpuBuffer,
        status: &GpuBuffer,
    ) -> Result<S14Bf16QHeadNormalizeDispatch> {
        self.bind_slices(
            ctx,
            epsilon,
            StorageBufferSlice::whole(input),
            StorageBufferSlice::whole(inverse_rms),
            StorageBufferSlice::whole(output),
            StorageBufferSlice::whole(status),
        )
    }

    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        epsilon: f32,
        input: StorageBufferSlice<'_>,
        inverse_rms: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14Bf16QHeadNormalizeDispatch> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            bail!("S14 BF16 Q-head normalize epsilon must be finite and positive");
        }
        let shape = S14Bf16QHeadNormalizeShape::new();
        let input_bytes = shape.input_bf16_bytes();
        let inverse_bytes = shape.inverse_rms_f32_bytes();
        let output_bytes = shape.output_bf16_bytes();
        let status_bytes = shape.status_bytes();
        for (candidate, bytes) in [
            (input, input_bytes),
            (inverse_rms, inverse_bytes),
            (status, status_bytes),
        ] {
            if storage_buffer_slices_overlap(output, output_bytes, candidate, bytes)? {
                bail!(
                    "S14 BF16 Q-head normalize output must not alias input, inverse RMS or status"
                );
            }
        }
        for (candidate, bytes) in [(input, input_bytes), (status, status_bytes)] {
            if storage_buffer_slices_overlap(inverse_rms, inverse_bytes, candidate, bytes)? {
                bail!("S14 BF16 Q-head normalize inverse RMS must not alias input or status");
            }
        }
        if storage_buffer_slices_overlap(status, status_bytes, input, input_bytes)? {
            bail!("S14 BF16 Q-head normalize status must not alias input");
        }
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (input.buffer, input.offset, input_bytes),
                (inverse_rms.buffer, inverse_rms.offset, inverse_bytes),
                (output.buffer, output.offset, output_bytes),
                (status.buffer, status.offset, status_bytes),
            ],
        )?;
        Ok(S14Bf16QHeadNormalizeDispatch {
            binder,
            shape,
            epsilon,
        })
    }

    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Bf16QHeadNormalizeDispatch,
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
        push[..4].copy_from_slice(&dispatch.shape.rows().to_le_bytes());
        push[4..8].copy_from_slice(&dispatch.shape.hidden().to_le_bytes());
        push[8..12].copy_from_slice(&dispatch.epsilon.to_le_bytes());
        push[12..].copy_from_slice(&0u32.to_le_bytes());
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push,
        );
        ctx.device
            .cmd_dispatch(command, dispatch.shape.rows(), 1, 1);

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
        let pairs = dispatch.shape.scalar_count() as u32 / 2;
        ctx.device.cmd_dispatch(command, pairs.div_ceil(256), 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub fn validate_bf16_q_head_normalize_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let known = S14_Q_HEAD_STATUS_NON_FINITE_INPUT
        | S14_Q_HEAD_STATUS_NON_FINITE_NORM
        | S14_Q_HEAD_STATUS_NON_FINITE_OUTPUT
        | S14_Q_HEAD_STATUS_NON_FINITE_BF16;
    if code & !known != 0 {
        bail!("S14 BF16 Q-head normalize returned unknown status bits 0x{code:08x}");
    }
    bail!("S14 BF16 Q-head normalize rejected non-finite state, status=0x{code:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_head_shape_is_fixed_to_production_contract() {
        let shape = S14Bf16QHeadNormalizeShape::new();
        assert_eq!(shape.rows(), 64);
        assert_eq!(shape.hidden(), 512);
        assert_eq!(shape.scalar_count(), 32_768);
        assert_eq!(shape.input_bf16_bytes(), 65_536);
        assert_eq!(shape.inverse_rms_f32_bytes(), 256);
        assert_eq!(shape.output_bf16_bytes(), 65_536);
        assert_eq!(shape.status_bytes(), 4);
    }

    #[test]
    fn q_head_status_is_fail_closed() {
        validate_bf16_q_head_normalize_status(0).unwrap();
        for code in [1, 2, 4, 8, 15, 16] {
            assert!(validate_bf16_q_head_normalize_status(code).is_err());
        }
    }
}
