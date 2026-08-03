//! ratio4 分页 indexer 的 global top-512 Vulkan merge。
//!
//! 每页先复用既有 exact sparse-indexer 得到稳定降序的局部 `(score,index)`；
//! 本原语用双 bank 把该页合入 running global top-512。它不读取 main KV，只有
//! 全部页完成后，调用方才能按全局 index 聚合命中的 main-KV 行。

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::s14_ratio4_history_paging::{S14Ratio4HistoryLayout, S14_RATIO4_INDEXER_ROW_BYTES};
use crate::s14_sparse_attention::{
    S14SparseIndexerPipeline, S14_INDEX_TOP_K, S14_SPARSE_STATUS_INDEX_SCORE_NON_FINITE,
    S14_SPARSE_STATUS_INVALID_COMPRESSED_INDEX,
};
use crate::VulkanContext;
use anyhow::{anyhow, bail, Result};
use ash::vk;

const GLOBAL_TOPK_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_ratio4_global_topk_merge.spv"
));
const WORD_BYTES: u64 = 4;
const STATUS_BYTES: u64 = 4;
const SHAPE_STATUS: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Ratio4GlobalTopKShape {
    pub global_count: u32,
    pub page_count: u32,
    pub page_base: u32,
    pub logical_count: u32,
    pub output_count: u32,
}

impl S14Ratio4GlobalTopKShape {
    pub fn new(page_base: u32, page_count: u32, logical_count: u32) -> Result<Self> {
        if page_count == 0
            || page_count > S14_INDEX_TOP_K
            || page_base % S14_INDEX_TOP_K != 0
            || page_base
                .checked_add(page_count)
                .is_none_or(|end| end != logical_count)
        {
            bail!("ratio4 global top-k page identity/shape drift");
        }
        let global_count = page_base.min(S14_INDEX_TOP_K);
        let output_count = logical_count.min(S14_INDEX_TOP_K);
        Ok(Self {
            global_count,
            page_count,
            page_base,
            logical_count,
            output_count,
        })
    }

    fn global_input_bytes(self) -> u64 {
        u64::from(self.global_count.max(1)) * WORD_BYTES
    }

    fn page_bytes(self) -> u64 {
        u64::from(self.page_count) * WORD_BYTES
    }

    fn output_bytes(self) -> u64 {
        u64::from(self.output_count) * WORD_BYTES
    }

    fn push_words(self) -> [u32; 5] {
        [
            self.global_count,
            self.page_count,
            self.page_base,
            self.logical_count,
            self.output_count,
        ]
    }
}

pub struct S14Ratio4GlobalTopKPipeline {
    pipeline: ComputePipeline,
}

pub struct S14Ratio4GlobalTopKDispatch {
    binder: DescriptorBinder,
    shape: S14Ratio4GlobalTopKShape,
}

/// production 分页 indexer/global-merge 的全部 device slice。两个 global bank
/// 必须物理分离；页间只切换 bank，不产生 host wait/readback。
#[derive(Clone, Copy)]
pub struct S14Ratio4PagedGlobalTopKBindings<'a> {
    pub processed_index_query: StorageBufferSlice<'a>,
    /// 逻辑 block0 开始的连续、search-resident indexer history。
    pub indexer_history: StorageBufferSlice<'a>,
    pub head_weights: StorageBufferSlice<'a>,
    pub page_scores: StorageBufferSlice<'a>,
    pub page_indices: StorageBufferSlice<'a>,
    pub global_score_banks: [StorageBufferSlice<'a>; 2],
    pub global_index_banks: [StorageBufferSlice<'a>; 2],
    pub status: StorageBufferSlice<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Ratio4PagedGlobalTopKReceipt {
    pub logical_count: u32,
    pub selected_count: u32,
    pub scanned_pages: u32,
    pub final_bank: usize,
}

/// descriptor owner 必须活到包含整条页链的 command 完成。receipt 只声明最终
/// global bank，不发布 host logical length；最终 token commit 仍是唯一发布点。
pub struct S14Ratio4PagedGlobalTopKRecording {
    binders: Vec<DescriptorBinder>,
    receipt: S14Ratio4PagedGlobalTopKReceipt,
}

impl S14Ratio4PagedGlobalTopKRecording {
    pub fn receipt(&self) -> S14Ratio4PagedGlobalTopKReceipt {
        self.receipt
    }

    pub fn final_scores<'a>(
        &self,
        bindings: &S14Ratio4PagedGlobalTopKBindings<'a>,
    ) -> StorageBufferSlice<'a> {
        bindings.global_score_banks[self.receipt.final_bank]
    }

    pub fn final_indices<'a>(
        &self,
        bindings: &S14Ratio4PagedGlobalTopKBindings<'a>,
    ) -> StorageBufferSlice<'a> {
        bindings.global_index_banks[self.receipt.final_bank]
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        for binder in self.binders {
            binder.destroy(ctx);
        }
    }

    /// production layer recorder 把 descriptor 并入其 token-scoped owner；receipt 与
    /// binders 必须一起转移，避免 command 完成前提前销毁。
    pub fn into_parts(self) -> (S14Ratio4PagedGlobalTopKReceipt, Vec<DescriptorBinder>) {
        (self.receipt, self.binders)
    }
}

impl S14Ratio4GlobalTopKDispatch {
    pub fn shape(&self) -> S14Ratio4GlobalTopKShape {
        self.shape
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.binder.destroy(ctx);
    }
}

impl S14Ratio4GlobalTopKPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, GLOBAL_TOPK_SPV, 7, 20)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        global_scores: StorageBufferSlice<'_>,
        global_indices: StorageBufferSlice<'_>,
        page_scores: StorageBufferSlice<'_>,
        page_indices: StorageBufferSlice<'_>,
        output_scores: StorageBufferSlice<'_>,
        output_indices: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
        shape: S14Ratio4GlobalTopKShape,
    ) -> Result<S14Ratio4GlobalTopKDispatch> {
        let shape =
            S14Ratio4GlobalTopKShape::new(shape.page_base, shape.page_count, shape.logical_count)?;
        let bindings = [
            (global_scores, shape.global_input_bytes(), "global scores"),
            (global_indices, shape.global_input_bytes(), "global indices"),
            (page_scores, shape.page_bytes(), "page scores"),
            (page_indices, shape.page_bytes(), "page indices"),
            (output_scores, shape.output_bytes(), "output scores"),
            (output_indices, shape.output_bytes(), "output indices"),
            (status, STATUS_BYTES, "status"),
        ];
        for (slice, bytes, label) in bindings {
            let end = slice
                .offset
                .checked_add(bytes)
                .ok_or_else(|| anyhow!("ratio4 top-k {label} range overflow"))?;
            if end > slice.buffer.size() {
                bail!("ratio4 top-k {label} out of bounds");
            }
        }
        for output in [bindings[4], bindings[5]] {
            for input in [
                bindings[0],
                bindings[1],
                bindings[2],
                bindings[3],
                bindings[6],
            ] {
                if storage_buffer_slices_overlap(output.0, output.1, input.0, input.1)? {
                    bail!("ratio4 top-k output/input slices overlap");
                }
            }
        }
        if storage_buffer_slices_overlap(
            bindings[4].0,
            bindings[4].1,
            bindings[5].0,
            bindings[5].1,
        )? {
            bail!("ratio4 top-k output slices overlap");
        }
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &bindings
                .iter()
                .map(|(slice, bytes, _)| (slice.buffer, slice.offset, *bytes))
                .collect::<Vec<_>>(),
        )?;
        Ok(S14Ratio4GlobalTopKDispatch { binder, shape })
    }

    /// # Safety
    /// descriptor 与所有 slice 必须活到 command 完成；status 必须在页链开始前清零。
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Ratio4GlobalTopKDispatch,
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
        let words = dispatch.shape.push_words();
        let push = std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), 20);
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push,
        );
        ctx.device.cmd_dispatch(command, 1, 1, 1);
    }

    /// 把全部 ratio4 indexer 页扫描和 global top-512 merge 记录到调用方提供的
    /// 同一个持久 command buffer。页循环只发生在 command recording 阶段；执行期
    /// 没有 host wait/readback。每页复用同一 page scratch，global 结果在双 bank
    /// 之间切换，最终 bank 由 receipt 明确返回。
    ///
    /// 本接口只闭合 indexer/global-selection；调用方仍必须在同一 command 内按
    /// final global indices 聚合 pageable main KV，之后才能签发 position2051。
    ///
    /// # Safety
    /// command 必须处于 recording；全部 buffer 与返回的 descriptor owner 必须活到
    /// command 完成；status 必须在整条链开始前清零。
    pub unsafe fn record_paged_indexer_global_topk(
        &self,
        indexer: &S14SparseIndexerPipeline,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        history: &S14Ratio4HistoryLayout,
        bindings: S14Ratio4PagedGlobalTopKBindings<'_>,
    ) -> Result<S14Ratio4PagedGlobalTopKRecording> {
        if history.logical_len == 0 || history.pages.is_empty() {
            bail!("ratio4 paged global top-k 要求非空 history");
        }
        let history_bytes = u64::from(history.logical_len)
            .checked_mul(S14_RATIO4_INDEXER_ROW_BYTES)
            .ok_or_else(|| anyhow!("ratio4 indexer history bytes overflow"))?;
        let history_end = bindings
            .indexer_history
            .offset
            .checked_add(history_bytes)
            .ok_or_else(|| anyhow!("ratio4 indexer history range overflow"))?;
        if history_end > bindings.indexer_history.buffer.size() {
            bail!("ratio4 indexer history backing buffer 不足");
        }
        let mut expected_start = 0u32;
        for (page_index, page) in history.pages.iter().enumerate() {
            if page.page_index != u32::try_from(page_index)?
                || page.logical_rows.start != expected_start
                || page.logical_rows.is_empty()
                || page.logical_len() > S14_INDEX_TOP_K
            {
                bail!("ratio4 indexer history page identity/order 漂移");
            }
            expected_start = page.logical_rows.end;
        }
        if expected_start != history.logical_len {
            bail!("ratio4 indexer history page coverage 不等于 logical length");
        }

        let mut binders = Vec::with_capacity(history.pages.len() * 2);
        let result = (|| -> Result<S14Ratio4PagedGlobalTopKReceipt> {
            let mut final_bank = 1usize;
            for page in &history.pages {
                let page_count = page.logical_len();
                let page_offset = bindings
                    .indexer_history
                    .offset
                    .checked_add(u64::from(page.logical_rows.start) * S14_RATIO4_INDEXER_ROW_BYTES)
                    .ok_or_else(|| anyhow!("ratio4 indexer page offset overflow"))?;
                let index_dispatch = indexer.bind_slices(
                    ctx,
                    bindings.processed_index_query,
                    StorageBufferSlice {
                        buffer: bindings.indexer_history.buffer,
                        offset: page_offset,
                    },
                    bindings.head_weights,
                    bindings.page_scores,
                    bindings.page_indices,
                    bindings.status,
                    page_count,
                )?;
                indexer.cmd(ctx, command, &index_dispatch);
                binders.push(index_dispatch.binder);
                shader_read_write_barrier(ctx, command);

                let output_bank = final_bank ^ 1;
                let shape = S14Ratio4GlobalTopKShape::new(
                    page.logical_rows.start,
                    page_count,
                    page.logical_rows.end,
                )?;
                let merge_dispatch = self.bind_slices(
                    ctx,
                    bindings.global_score_banks[final_bank],
                    bindings.global_index_banks[final_bank],
                    bindings.page_scores,
                    bindings.page_indices,
                    bindings.global_score_banks[output_bank],
                    bindings.global_index_banks[output_bank],
                    bindings.status,
                    shape,
                )?;
                self.cmd(ctx, command, &merge_dispatch);
                binders.push(merge_dispatch.binder);
                shader_read_write_barrier(ctx, command);
                final_bank = output_bank;
            }
            Ok(S14Ratio4PagedGlobalTopKReceipt {
                logical_count: history.logical_len,
                selected_count: history.logical_len.min(S14_INDEX_TOP_K),
                scanned_pages: u32::try_from(history.pages.len())?,
                final_bank,
            })
        })();
        match result {
            Ok(receipt) => Ok(S14Ratio4PagedGlobalTopKRecording { binders, receipt }),
            Err(error) => {
                for binder in binders.drain(..) {
                    binder.destroy(ctx);
                }
                Err(error)
            }
        }
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

unsafe fn shader_read_write_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
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
}

pub fn validate_global_topk_status(status: u32) -> Result<()> {
    let known = S14_SPARSE_STATUS_INDEX_SCORE_NON_FINITE
        | S14_SPARSE_STATUS_INVALID_COMPRESSED_INDEX
        | SHAPE_STATUS;
    if status == 0 {
        return Ok(());
    }
    if status & !known != 0 {
        bail!("ratio4 global top-k unknown status 0x{status:08x}");
    }
    bail!("ratio4 global top-k rejected page chain status=0x{status:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_chain_keeps_global_top512_separate_from_history_length() {
        let first = S14Ratio4GlobalTopKShape::new(0, 512, 512).unwrap();
        assert_eq!((first.global_count, first.output_count), (0, 512));
        let second = S14Ratio4GlobalTopKShape::new(512, 1, 513).unwrap();
        assert_eq!((second.global_count, second.output_count), (512, 512));
        let last = S14Ratio4GlobalTopKShape::new(49_664, 336, 50_000).unwrap();
        assert_eq!((last.global_count, last.output_count), (512, 512));
        assert!(S14Ratio4GlobalTopKShape::new(511, 1, 512).is_err());
        assert!(S14Ratio4GlobalTopKShape::new(512, 0, 512).is_err());
        assert!(S14Ratio4GlobalTopKShape::new(512, 2, 513).is_err());
        validate_global_topk_status(0).unwrap();
        for status in [8, 16, 32, 56, 64] {
            assert!(validate_global_topk_status(status).is_err());
        }
    }
}
