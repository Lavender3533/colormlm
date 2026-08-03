//! FullDepth43 BF16 activation 的通用 E4M3FN quantize/dequantize 边界。

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_E4M3_QDQ_MAX_ROWS: u32 = 64;
pub const S14_E4M3_QDQ_MAX_HIDDEN: u32 = 8192;
pub const S14_E4M3_QDQ_STATUS_NON_FINITE_INPUT: u32 = 1;
pub const S14_E4M3_QDQ_STATUS_INVALID_SCALE: u32 = 2;
pub const S14_E4M3_QDQ_STATUS_NON_FINITE_OUTPUT: u32 = 4;

pub const S14_E4M3_QDQ_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/s14_e4m3_qdq.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14E4m3QdqShape {
    pub rows: u32,
    pub hidden: u32,
    pub group_size: u32,
}

impl S14E4m3QdqShape {
    pub fn new(rows: u32, hidden: u32, group_size: u32) -> Result<Self> {
        if rows == 0 || rows > S14_E4M3_QDQ_MAX_ROWS {
            bail!("S14 E4M3 QDQ rows must be in 1..={S14_E4M3_QDQ_MAX_ROWS}");
        }
        if hidden == 0 || hidden > S14_E4M3_QDQ_MAX_HIDDEN || hidden % 2 != 0 {
            bail!("S14 E4M3 QDQ hidden must be even and in 1..={S14_E4M3_QDQ_MAX_HIDDEN}");
        }
        if !matches!(group_size, 64 | 128) || hidden % group_size != 0 {
            bail!("S14 E4M3 QDQ group must be 64 or 128 and divide hidden exactly");
        }
        let shape = Self {
            rows,
            hidden,
            group_size,
        };
        shape.input_bf16_bytes()?;
        shape.scale_f32_bytes()?;
        shape.output_f32_bytes()?;
        Ok(shape)
    }

    pub fn scalar_count(self) -> Result<u64> {
        (self.rows as u64)
            .checked_mul(self.hidden as u64)
            .ok_or_else(|| anyhow::anyhow!("S14 E4M3 QDQ shape overflow"))
    }

    pub fn group_count(self) -> Result<u64> {
        Ok(self.scalar_count()? / self.group_size as u64)
    }

    pub fn input_bf16_bytes(self) -> Result<u64> {
        checked_bytes(self.scalar_count()?, 2, "S14 E4M3 QDQ input")
    }

    pub fn scale_f32_bytes(self) -> Result<u64> {
        checked_bytes(self.group_count()?, 4, "S14 E4M3 QDQ scales")
    }

    pub fn output_f32_bytes(self) -> Result<u64> {
        checked_bytes(self.scalar_count()?, 4, "S14 E4M3 QDQ output")
    }

    pub fn status_bytes(self) -> u64 {
        4
    }
}

pub struct S14E4m3QdqPipeline {
    pipeline: ComputePipeline,
}

pub struct S14E4m3QdqDispatch {
    pub binder: DescriptorBinder,
    pub shape: S14E4m3QdqShape,
}

impl S14E4m3QdqPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_E4M3_QDQ_SPV, 4, 16)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        &self,
        ctx: &VulkanContext,
        shape: S14E4m3QdqShape,
        input: &GpuBuffer,
        scales: &GpuBuffer,
        output: &GpuBuffer,
        status: &GpuBuffer,
    ) -> Result<S14E4m3QdqDispatch> {
        self.bind_slices(
            ctx,
            shape,
            StorageBufferSlice::whole(input),
            StorageBufferSlice::whole(scales),
            StorageBufferSlice::whole(output),
            StorageBufferSlice::whole(status),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        shape: S14E4m3QdqShape,
        input: StorageBufferSlice<'_>,
        scales: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14E4m3QdqDispatch> {
        let shape = S14E4m3QdqShape::new(shape.rows, shape.hidden, shape.group_size)?;
        let input_bytes = shape.input_bf16_bytes()?;
        let scale_bytes = shape.scale_f32_bytes()?;
        let output_bytes = shape.output_f32_bytes()?;
        let status_bytes = shape.status_bytes();
        for (candidate, bytes) in [
            (input, input_bytes),
            (scales, scale_bytes),
            (status, status_bytes),
        ] {
            if storage_buffer_slices_overlap(output, output_bytes, candidate, bytes)? {
                bail!("S14 E4M3 QDQ output must not alias input, scales, or status");
            }
        }
        if storage_buffer_slices_overlap(scales, scale_bytes, input, input_bytes)?
            || storage_buffer_slices_overlap(scales, scale_bytes, status, status_bytes)?
            || storage_buffer_slices_overlap(status, status_bytes, input, input_bytes)?
        {
            bail!("S14 E4M3 QDQ scratch/status must not alias input or each other");
        }
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (input.buffer, input.offset, input_bytes),
                (scales.buffer, scales.offset, scale_bytes),
                (output.buffer, output.offset, output_bytes),
                (status.buffer, status.offset, status_bytes),
            ],
        )?;
        Ok(S14E4m3QdqDispatch { binder, shape })
    }

    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14E4m3QdqDispatch,
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
        push[8..12].copy_from_slice(&dispatch.shape.group_size.to_le_bytes());
        push[12..].copy_from_slice(&0u32.to_le_bytes());
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push,
        );
        ctx.device
            .cmd_dispatch(command, dispatch.shape.group_count().unwrap() as u32, 1, 1);
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
        let scalars = dispatch.shape.rows * dispatch.shape.hidden;
        ctx.device
            .cmd_dispatch(command, scalars.div_ceil(128), 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub fn validate_e4m3_qdq_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let known = S14_E4M3_QDQ_STATUS_NON_FINITE_INPUT
        | S14_E4M3_QDQ_STATUS_INVALID_SCALE
        | S14_E4M3_QDQ_STATUS_NON_FINITE_OUTPUT;
    if code & !known != 0 {
        bail!("S14 E4M3 QDQ returned unknown status bits 0x{code:08x}");
    }
    bail!("S14 E4M3 QDQ rejected invalid state, status=0x{code:08x}")
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
    fn qdq_shapes_cover_group64_and_group128() {
        let kv = S14E4m3QdqShape::new(1, 448, 64).unwrap();
        assert_eq!(kv.group_count().unwrap(), 7);
        let projection = S14E4m3QdqShape::new(1, 8192, 128).unwrap();
        assert_eq!(projection.group_count().unwrap(), 64);
    }

    #[test]
    fn qdq_shape_and_status_fail_closed() {
        for shape in [
            (0, 448, 64),
            (65, 448, 64),
            (1, 447, 64),
            (1, 448, 32),
            (1, 8256, 128),
        ] {
            assert!(S14E4m3QdqShape::new(shape.0, shape.1, shape.2).is_err());
        }
        validate_e4m3_qdq_status(0).unwrap();
        for code in [1, 2, 4, 7, 8] {
            assert!(validate_e4m3_qdq_status(code).is_err());
        }
    }
}
