//! FullDepth43 plain BF16→F32 Vulkan 展开边界。
//!
//! position0 attention 输出在送入 grouped `wo_a` 前只做 `.float()`，不能
//! 误用会改值的 E4M3 activation QDQ。该核逐位展开 BF16，并继承 whole-token
//! sticky status 的 fail-closed 语义。

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_BF16_TO_F32_MAX_SCALARS: u32 = 8 * 32_768;
pub const S14_BF16_TO_F32_STATUS_NON_FINITE_INPUT: u32 = 1;
pub const S14_BF16_TO_F32_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_bf16_to_f32.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Bf16ToF32Shape {
    pub scalars: u32,
}

impl S14Bf16ToF32Shape {
    pub fn new(scalars: u32) -> Result<Self> {
        if scalars == 0 || scalars > S14_BF16_TO_F32_MAX_SCALARS || scalars % 2 != 0 {
            bail!(
                "S14 BF16->F32 scalar count must be even and in 1..={S14_BF16_TO_F32_MAX_SCALARS}"
            );
        }
        Ok(Self { scalars })
    }

    pub fn input_bf16_bytes(self) -> u64 {
        self.scalars as u64 * 2
    }

    pub fn output_f32_bytes(self) -> u64 {
        self.scalars as u64 * 4
    }

    pub fn status_bytes(self) -> u64 {
        4
    }
}

pub struct S14Bf16ToF32Pipeline {
    pipeline: ComputePipeline,
}

pub struct S14Bf16ToF32Dispatch {
    pub binder: DescriptorBinder,
    pub shape: S14Bf16ToF32Shape,
}

impl S14Bf16ToF32Pipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_BF16_TO_F32_SPV, 3, 4)?,
        })
    }

    pub fn bind(
        &self,
        ctx: &VulkanContext,
        shape: S14Bf16ToF32Shape,
        input: &GpuBuffer,
        output: &GpuBuffer,
        status: &GpuBuffer,
    ) -> Result<S14Bf16ToF32Dispatch> {
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
        shape: S14Bf16ToF32Shape,
        input: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14Bf16ToF32Dispatch> {
        let shape = S14Bf16ToF32Shape::new(shape.scalars)?;
        let input_bytes = shape.input_bf16_bytes();
        let output_bytes = shape.output_f32_bytes();
        let status_bytes = shape.status_bytes();
        if storage_buffer_slices_overlap(input, input_bytes, output, output_bytes)?
            || storage_buffer_slices_overlap(input, input_bytes, status, status_bytes)?
            || storage_buffer_slices_overlap(output, output_bytes, status, status_bytes)?
        {
            bail!("S14 BF16->F32 input, output and status buffers must not alias");
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
        Ok(S14Bf16ToF32Dispatch { binder, shape })
    }

    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Bf16ToF32Dispatch,
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

pub fn validate_bf16_to_f32_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    if code & !S14_BF16_TO_F32_STATUS_NON_FINITE_INPUT != 0 {
        bail!("S14 BF16->F32 returned unknown status bits 0x{code:08x}");
    }
    bail!("S14 BF16->F32 rejected non-finite input, status=0x{code:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_attention_shapes_have_exact_bytes() {
        for scalars in [32_768, 4 * 32_768, 8 * 32_768] {
            let shape = S14Bf16ToF32Shape::new(scalars).unwrap();
            assert_eq!(shape.input_bf16_bytes(), scalars as u64 * 2);
            assert_eq!(shape.output_f32_bytes(), scalars as u64 * 4);
        }
    }

    #[test]
    fn shape_and_status_fail_closed() {
        for scalars in [0, 3, 262_146] {
            assert!(S14Bf16ToF32Shape::new(scalars).is_err());
        }
        validate_bf16_to_f32_status(0).unwrap();
        for code in [1, 2, 3] {
            assert!(validate_bf16_to_f32_status(code).is_err());
        }
    }
}
