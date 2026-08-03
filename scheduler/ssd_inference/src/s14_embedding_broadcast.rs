//! FullDepth43 verified BF16 embedding row 到四流 mHC 初态的 Vulkan 边界。
//!
//! 真实 Range 资产只含当前 token 的 BF16 `[4096]` 行，而 L0 输入必须是
//! BF16 `[4,4096]`。这里不重新查整张词表，也不允许 host capture 代替输出；
//! shader 验证完整输入后才把同一行精确复制到四个流。

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_EMBEDDING_STREAMS: u32 = 4;
pub const S14_EMBEDDING_MAX_HIDDEN: u32 = 4096;
pub const S14_EMBEDDING_STATUS_NON_FINITE_INPUT: u32 = 1;
pub const S14_EMBEDDING_BROADCAST_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_embedding_broadcast.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14EmbeddingBroadcastShape {
    pub hidden: u32,
}

impl S14EmbeddingBroadcastShape {
    pub fn new(hidden: u32) -> Result<Self> {
        if hidden == 0 || hidden > S14_EMBEDDING_MAX_HIDDEN || hidden % 2 != 0 {
            bail!("S14 embedding hidden must be even and in 1..={S14_EMBEDDING_MAX_HIDDEN}");
        }
        Ok(Self { hidden })
    }

    pub fn row_bf16_bytes(self) -> u64 {
        self.hidden as u64 * 2
    }

    pub fn output_bf16_bytes(self) -> u64 {
        self.row_bf16_bytes() * S14_EMBEDDING_STREAMS as u64
    }

    pub fn status_bytes(self) -> u64 {
        4
    }
}

pub struct S14EmbeddingBroadcastPipeline {
    pipeline: ComputePipeline,
}

pub struct S14EmbeddingBroadcastDispatch {
    pub binder: DescriptorBinder,
    pub shape: S14EmbeddingBroadcastShape,
}

impl S14EmbeddingBroadcastPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_EMBEDDING_BROADCAST_SPV, 3, 4)?,
        })
    }

    pub fn bind(
        &self,
        ctx: &VulkanContext,
        shape: S14EmbeddingBroadcastShape,
        embedding_row: &GpuBuffer,
        output: &GpuBuffer,
        status: &GpuBuffer,
    ) -> Result<S14EmbeddingBroadcastDispatch> {
        self.bind_slices(
            ctx,
            shape,
            StorageBufferSlice::whole(embedding_row),
            StorageBufferSlice::whole(output),
            StorageBufferSlice::whole(status),
        )
    }

    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        shape: S14EmbeddingBroadcastShape,
        embedding_row: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14EmbeddingBroadcastDispatch> {
        let shape = S14EmbeddingBroadcastShape::new(shape.hidden)?;
        let row_bytes = shape.row_bf16_bytes();
        let output_bytes = shape.output_bf16_bytes();
        let status_bytes = shape.status_bytes();
        if storage_buffer_slices_overlap(output, output_bytes, embedding_row, row_bytes)?
            || storage_buffer_slices_overlap(output, output_bytes, status, status_bytes)?
            || storage_buffer_slices_overlap(status, status_bytes, embedding_row, row_bytes)?
        {
            bail!("S14 embedding input, output and status buffers must not alias");
        }
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (embedding_row.buffer, embedding_row.offset, row_bytes),
                (output.buffer, output.offset, output_bytes),
                (status.buffer, status.offset, status_bytes),
            ],
        )?;
        Ok(S14EmbeddingBroadcastDispatch { binder, shape })
    }

    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14EmbeddingBroadcastDispatch,
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
            &dispatch.shape.hidden.to_le_bytes(),
        );
        ctx.device.cmd_dispatch(command, 1, 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub fn validate_embedding_broadcast_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    if code & !S14_EMBEDDING_STATUS_NON_FINITE_INPUT != 0 {
        bail!("S14 embedding broadcast returned unknown status bits 0x{code:08x}");
    }
    bail!("S14 embedding broadcast rejected non-finite row, status=0x{code:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_shape_has_exact_byte_contract() {
        let shape = S14EmbeddingBroadcastShape::new(4096).unwrap();
        assert_eq!(shape.row_bf16_bytes(), 8192);
        assert_eq!(shape.output_bf16_bytes(), 32768);
        assert_eq!(shape.status_bytes(), 4);
    }

    #[test]
    fn shape_and_status_fail_closed() {
        for hidden in [0, 3, 4098] {
            assert!(S14EmbeddingBroadcastShape::new(hidden).is_err());
        }
        validate_embedding_broadcast_status(0).unwrap();
        for code in [1, 2, 3] {
            assert!(validate_embedding_broadcast_status(code).is_err());
        }
    }
}
