//! ratio4 首块边界的 production attention。
//!
//! position3 时 compressed cache 的基数严格为1，官方 indexer `topk(1)` 的结果恒为
//! block0；因此本核可以直接消费刚完成的 main compressed KV，不改变选择语义。
//! position4+ 的 compressed cache 选择必须重新进入真实 indexer，禁止复用此特例。

use crate::compute::{ComputePipeline, DescriptorBinder, StorageBufferSlice};
use crate::s14_position0_attention::{S14_POSITION0_HEADS, S14_POSITION0_HEAD_DIM};
use crate::VulkanContext;
use anyhow::{bail, Result};
use ash::vk;

pub const S14_POSITION3_ATTENTION_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_position3_attention.spv"));

pub struct S14Position3AttentionPipeline {
    pipeline: ComputePipeline,
}

pub struct S14Position3AttentionDispatch {
    pub binder: DescriptorBinder,
}

impl S14Position3AttentionPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_POSITION3_ATTENTION_SPV, 7, 12)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        query: StorageBufferSlice<'_>,
        previous_kv: StorageBufferSlice<'_>,
        current_kv: StorageBufferSlice<'_>,
        compressed_kv: StorageBufferSlice<'_>,
        sink: StorageBufferSlice<'_>,
        rope_cos_sin: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        position: u32,
        compress_ratio: u16,
        previous_count: u32,
        compressed_count: u32,
    ) -> Result<S14Position3AttentionDispatch> {
        const QUERY_BYTES: u64 = S14_POSITION0_HEADS as u64 * S14_POSITION0_HEAD_DIM as u64 * 2;
        const KV_ROW_BYTES: u64 = S14_POSITION0_HEAD_DIM as u64 * 2;
        const SINK_BYTES: u64 = S14_POSITION0_HEADS as u64 * 4;
        const ROPE_BYTES: u64 = 32 * 2 * 4;
        if position != 3 || compress_ratio != 4 || previous_count != 3 || compressed_count != 1 {
            bail!(
                "ratio4 首块 attention 只允许 position3/previous3/compressed1，actual position={position} ratio={compress_ratio} previous={previous_count} compressed={compressed_count}"
            );
        }
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (query.buffer, query.offset, QUERY_BYTES),
                (
                    previous_kv.buffer,
                    previous_kv.offset,
                    KV_ROW_BYTES * u64::from(previous_count),
                ),
                (current_kv.buffer, current_kv.offset, KV_ROW_BYTES),
                (compressed_kv.buffer, compressed_kv.offset, KV_ROW_BYTES),
                (sink.buffer, sink.offset, SINK_BYTES),
                (rope_cos_sin.buffer, rope_cos_sin.offset, ROPE_BYTES),
                (output.buffer, output.offset, QUERY_BYTES),
            ],
        )?;
        Ok(S14Position3AttentionDispatch { binder })
    }

    /// # Safety
    /// 所有 descriptor 资源必须活到 `command` 完成；调用前后由上层插入 compute barrier。
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Position3AttentionDispatch,
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
        let mut push = [0u8; 12];
        push[..4].copy_from_slice(&S14_POSITION0_HEADS.to_le_bytes());
        push[4..8].copy_from_slice(&S14_POSITION0_HEAD_DIM.to_le_bytes());
        push[8..].copy_from_slice(&3u32.to_le_bytes());
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

/// 首个 ratio4 compressed cache 的 indexer top-k 恒等性。
/// 该函数只签发 position3 的基数1特例，防止上层把它误推广到 position4+。
pub fn ratio4_first_block_index(position: u32, compressed_count: u32) -> Result<u32> {
    if position != 3 || compressed_count != 1 {
        bail!("ratio4 first-block indexer shortcut 只允许 position3/compressed_count1");
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_ratio4_block_is_the_only_indexer_candidate_and_later_positions_fail_closed() {
        assert_eq!(ratio4_first_block_index(3, 1).unwrap(), 0);
        assert!(ratio4_first_block_index(2, 0).is_err());
        assert!(ratio4_first_block_index(4, 1).is_err());
        assert!(ratio4_first_block_index(7, 2).is_err());
    }

    #[test]
    fn position3_softmax_uses_the_frozen_rte_path() {
        let shader = include_str!("../shaders/s14_position3_attention.comp");
        let softmax_start = shader
            .find("if (lane == 0u) {")
            .expect("position3 softmax block must exist");
        let softmax = &shader[softmax_start..];
        assert!(softmax.contains("exp_rte_softmax"));
        assert!(softmax.contains("reciprocal_rte_positive_normal"));
        assert!(!softmax.contains("exp("));
        assert!(!softmax.contains("/ denominator"));
    }
}
