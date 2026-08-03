//! FullDepth43 投影输出的 F32→BF16 RNE Vulkan 数值边界。
//!
//! FP8/BF16 matvec 原语输出 F32；官方图在 q、kv、attention output 和
//! expert projection 等位置显式转换为 BF16。该核提供可复用且 fail-closed 的
//! 物理边界，禁止 concrete backend 静默把 F32 直接传入下一个算子。

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_F32_TO_BF16_MAX_SCALARS: u32 = 8 * 32_768;
pub const S14_F32_TO_BF16_STATUS_NON_FINITE_INPUT: u32 = 1;
pub const S14_F32_TO_BF16_STATUS_NON_FINITE_BF16: u32 = 2;
pub const S14_F32_TO_BF16_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_f32_to_bf16.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14F32ToBf16Shape {
    pub scalars: u32,
}

impl S14F32ToBf16Shape {
    pub fn new(scalars: u32) -> Result<Self> {
        if scalars == 0 || scalars > S14_F32_TO_BF16_MAX_SCALARS || scalars % 2 != 0 {
            bail!(
                "S14 F32->BF16 scalar count must be even and in 1..={S14_F32_TO_BF16_MAX_SCALARS}"
            );
        }
        Ok(Self { scalars })
    }

    pub fn input_f32_bytes(self) -> u64 {
        self.scalars as u64 * 4
    }

    pub fn output_bf16_bytes(self) -> u64 {
        self.scalars as u64 * 2
    }

    pub fn status_bytes(self) -> u64 {
        4
    }
}

pub struct S14F32ToBf16Pipeline {
    pipeline: ComputePipeline,
}

pub struct S14F32ToBf16Dispatch {
    pub binder: DescriptorBinder,
    pub shape: S14F32ToBf16Shape,
}

impl S14F32ToBf16Pipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_F32_TO_BF16_SPV, 3, 4)?,
        })
    }

    pub fn bind(
        &self,
        ctx: &VulkanContext,
        shape: S14F32ToBf16Shape,
        input: &GpuBuffer,
        output: &GpuBuffer,
        status: &GpuBuffer,
    ) -> Result<S14F32ToBf16Dispatch> {
        self.bind_slices(
            ctx,
            shape,
            StorageBufferSlice::whole(input),
            StorageBufferSlice::whole(output),
            StorageBufferSlice::whole(status),
        )
    }

    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        shape: S14F32ToBf16Shape,
        input: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14F32ToBf16Dispatch> {
        let shape = S14F32ToBf16Shape::new(shape.scalars)?;
        let input_bytes = shape.input_f32_bytes();
        let output_bytes = shape.output_bf16_bytes();
        let status_bytes = shape.status_bytes();
        if storage_buffer_slices_overlap(input, input_bytes, output, output_bytes)?
            || storage_buffer_slices_overlap(input, input_bytes, status, status_bytes)?
            || storage_buffer_slices_overlap(output, output_bytes, status, status_bytes)?
        {
            bail!("S14 F32->BF16 input, output and status buffers must not alias");
        }
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (input.buffer, input.offset, input_bytes),
                (output.buffer, output.offset, output_bytes),
                (status.buffer, status.offset, status_bytes),
            ],
        )?;
        Ok(S14F32ToBf16Dispatch { binder, shape })
    }

    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14F32ToBf16Dispatch,
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
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &dispatch.shape.scalars.to_le_bytes(),
        );
        ctx.device.cmd_dispatch(command, 1, 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub fn validate_f32_to_bf16_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let known = S14_F32_TO_BF16_STATUS_NON_FINITE_INPUT | S14_F32_TO_BF16_STATUS_NON_FINITE_BF16;
    if code & !known != 0 {
        bail!("S14 F32->BF16 returned unknown status bits 0x{code:08x}");
    }
    bail!("S14 F32->BF16 rejected candidate, status=0x{code:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_cover_all_position0_projection_boundaries() {
        for scalars in [512, 1024, 2048, 4096, 8192, 32768, 131072, 262144] {
            let shape = S14F32ToBf16Shape::new(scalars).unwrap();
            assert_eq!(shape.input_f32_bytes(), scalars as u64 * 4);
            assert_eq!(shape.output_bf16_bytes(), scalars as u64 * 2);
        }
    }

    #[test]
    fn shape_and_status_fail_closed() {
        for scalars in [0, 3, 262146] {
            assert!(S14F32ToBf16Shape::new(scalars).is_err());
        }
        validate_f32_to_bf16_status(0).unwrap();
        for code in [1, 2, 3, 4] {
            assert!(validate_f32_to_bf16_status(code).is_err());
        }
    }
}
