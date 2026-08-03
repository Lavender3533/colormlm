//! FullDepth43 的 position=0 共享-KV attention 特化路径。
//!
//! 它只覆盖没有历史 KV/压缩块、top-k 仅含当前 token 的首位置；后续
//! position 必须进入正式 sparse attention，禁止复用本特化核。

use crate::compute::{ComputePipeline, DescriptorBinder, StorageBufferSlice};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_POSITION0_HEADS: u32 = 64;
pub const S14_POSITION0_HEAD_DIM: u32 = 512;
pub const S14_POSITION0_ATTENTION_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_position0_attention.spv"));

pub struct S14Position0AttentionPipeline {
    pipeline: ComputePipeline,
}

pub struct S14Position0AttentionDispatch {
    pub binder: DescriptorBinder,
}

impl S14Position0AttentionPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_POSITION0_ATTENTION_SPV, 4, 8)?,
        })
    }

    pub fn bind(
        &self,
        ctx: &VulkanContext,
        query: &GpuBuffer,
        key_value: &GpuBuffer,
        sink: &GpuBuffer,
        output: &GpuBuffer,
    ) -> Result<S14Position0AttentionDispatch> {
        self.bind_slices(
            ctx,
            StorageBufferSlice::whole(query),
            StorageBufferSlice::whole(key_value),
            StorageBufferSlice::whole(sink),
            StorageBufferSlice::whole(output),
        )
    }

    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        query: StorageBufferSlice<'_>,
        key_value: StorageBufferSlice<'_>,
        sink: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
    ) -> Result<S14Position0AttentionDispatch> {
        const QUERY_BYTES: u64 = S14_POSITION0_HEADS as u64 * S14_POSITION0_HEAD_DIM as u64 * 2;
        const KV_BYTES: u64 = S14_POSITION0_HEAD_DIM as u64 * 2;
        const SINK_BYTES: u64 = S14_POSITION0_HEADS as u64 * 4;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (query.buffer, query.offset, QUERY_BYTES),
                (key_value.buffer, key_value.offset, KV_BYTES),
                (sink.buffer, sink.offset, SINK_BYTES),
                (output.buffer, output.offset, QUERY_BYTES),
            ],
        )?;
        Ok(S14Position0AttentionDispatch { binder })
    }

    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Position0AttentionDispatch,
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
        let mut push = [0u8; 8];
        push[..4].copy_from_slice(&S14_POSITION0_HEADS.to_le_bytes());
        push[4..].copy_from_slice(&S14_POSITION0_HEAD_DIM.to_le_bytes());
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push,
        );
        ctx.device.cmd_dispatch(command, S14_POSITION0_HEADS, 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}
