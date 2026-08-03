//! ratio4 global top-512 索引对分页 main-KV 的 production gather 原语。
//!
//! 页数据可按实际命中页物化到任意连续 arena 位置；稠密页表把逻辑
//! `page` 映射到 arena word offset 和该页真实行数。GPU 保留 global top-k 排名，
//! 输出 `[selected_count, 512]` BF16 连续 attention workspace。sticky status 非零时
//! 调用方必须拒绝 attention/commit。

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::VulkanContext;
use anyhow::{anyhow, bail, Result};
use ash::vk;
use std::collections::HashSet;

const GATHER_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_ratio4_main_page_gather.spv"));

pub const S14_RATIO4_GATHER_PAGE_ROWS: u32 = 512;
pub const S14_RATIO4_GATHER_ROW_WIDTH: u32 = 512;
pub const S14_RATIO4_GATHER_ROW_WORDS: u32 = S14_RATIO4_GATHER_ROW_WIDTH / 2;
pub const S14_RATIO4_GATHER_MAX_SELECTED: u32 = 512;
pub const S14_RATIO4_GATHER_MISSING_PAGE: u32 = u32::MAX;

pub const S14_RATIO4_GATHER_STATUS_SHAPE: u32 = 1;
pub const S14_RATIO4_GATHER_STATUS_INDEX_OUT_OF_RANGE: u32 = 2;
pub const S14_RATIO4_GATHER_STATUS_DUPLICATE_INDEX: u32 = 4;
pub const S14_RATIO4_GATHER_STATUS_MISSING_PAGE: u32 = 8;
pub const S14_RATIO4_GATHER_STATUS_ROW_OUT_OF_PAGE: u32 = 16;
pub const S14_RATIO4_GATHER_STATUS_SOURCE_OUT_OF_RANGE: u32 = 32;
pub const S14_RATIO4_GATHER_STATUS_BF16_NAN: u32 = 64;

const WORD_BYTES: u64 = 4;
const PAGE_ENTRY_WORDS: u64 = 2;
const STATUS_BYTES: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Ratio4MainGatherShape {
    pub logical_count: u32,
    pub selected_count: u32,
    pub page_count: u32,
    pub source_word_count: u32,
}

impl S14Ratio4MainGatherShape {
    pub fn new(logical_count: u32, selected_count: u32, source_word_count: u32) -> Result<Self> {
        let expected_selected = logical_count.min(S14_RATIO4_GATHER_MAX_SELECTED);
        if logical_count == 0
            || selected_count == 0
            || selected_count != expected_selected
            || source_word_count == 0
        {
            bail!("ratio4 main gather shape drift");
        }
        let page_count = logical_count.div_ceil(S14_RATIO4_GATHER_PAGE_ROWS);
        Ok(Self {
            logical_count,
            selected_count,
            page_count,
            source_word_count,
        })
    }

    fn selected_bytes(self) -> u64 {
        u64::from(self.selected_count) * WORD_BYTES
    }

    fn page_table_bytes(self) -> u64 {
        u64::from(self.page_count) * PAGE_ENTRY_WORDS * WORD_BYTES
    }

    fn source_bytes(self) -> u64 {
        u64::from(self.source_word_count) * WORD_BYTES
    }

    pub fn packed_main_bytes(self) -> u64 {
        u64::from(self.selected_count) * u64::from(S14_RATIO4_GATHER_ROW_WORDS) * WORD_BYTES
    }

    fn push_words(self) -> [u32; 4] {
        [
            self.logical_count,
            self.selected_count,
            self.page_count,
            self.source_word_count,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Ratio4MaterializedMainPage {
    pub page_index: u32,
    /// 相对 main-pages descriptor 起点的 u32 word offset。
    pub source_word_offset: u32,
    pub row_count: u32,
}

/// 生成GPU直接消费的稠密 `[word_offset,row_count]` 页表。未物化页保留
/// `MISSING_PAGE`；如果global top-k实际命中该页，shader会在读数据前sticky reject。
pub fn build_ratio4_main_page_table(
    shape: S14Ratio4MainGatherShape,
    pages: &[S14Ratio4MaterializedMainPage],
) -> Result<Vec<u32>> {
    let mut table = vec![0u32; shape.page_count as usize * 2];
    for entry in table.chunks_exact_mut(2) {
        entry[0] = S14_RATIO4_GATHER_MISSING_PAGE;
        entry[1] = 0;
    }
    let mut seen = HashSet::with_capacity(pages.len());
    for page in pages {
        if page.page_index >= shape.page_count || !seen.insert(page.page_index) {
            bail!("ratio4 main gather duplicate/out-of-range page identity");
        }
        let logical_start = page
            .page_index
            .checked_mul(S14_RATIO4_GATHER_PAGE_ROWS)
            .ok_or_else(|| anyhow!("ratio4 main page logical start overflow"))?;
        let expected_rows = (shape.logical_count - logical_start).min(S14_RATIO4_GATHER_PAGE_ROWS);
        if page.row_count != expected_rows || page.row_count == 0 {
            bail!("ratio4 main gather page row count drift");
        }
        if page.source_word_offset % S14_RATIO4_GATHER_ROW_WORDS != 0 {
            bail!("ratio4 main gather page offset is not row aligned");
        }
        let page_words = page
            .row_count
            .checked_mul(S14_RATIO4_GATHER_ROW_WORDS)
            .ok_or_else(|| anyhow!("ratio4 main gather page words overflow"))?;
        if page
            .source_word_offset
            .checked_add(page_words)
            .is_none_or(|end| end > shape.source_word_count)
        {
            bail!("ratio4 main gather materialized page exceeds source arena");
        }
        let slot = page.page_index as usize * 2;
        table[slot] = page.source_word_offset;
        table[slot + 1] = page.row_count;
    }
    Ok(table)
}

pub struct S14Ratio4MainPageGatherPipeline {
    pipeline: ComputePipeline,
}

pub struct S14Ratio4MainPageGatherDispatch {
    pub binder: DescriptorBinder,
    shape: S14Ratio4MainGatherShape,
}

impl S14Ratio4MainPageGatherDispatch {
    pub fn shape(&self) -> S14Ratio4MainGatherShape {
        self.shape
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.binder.destroy(ctx);
    }
}

impl S14Ratio4MainPageGatherPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, GATHER_SPV, 5, 16)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        selected_indices: StorageBufferSlice<'_>,
        page_table: StorageBufferSlice<'_>,
        main_pages: StorageBufferSlice<'_>,
        packed_main: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
        shape: S14Ratio4MainGatherShape,
    ) -> Result<S14Ratio4MainPageGatherDispatch> {
        let shape = S14Ratio4MainGatherShape::new(
            shape.logical_count,
            shape.selected_count,
            shape.source_word_count,
        )?;
        let bindings = [
            (selected_indices, shape.selected_bytes(), "selected indices"),
            (page_table, shape.page_table_bytes(), "page table"),
            (main_pages, shape.source_bytes(), "main pages"),
            (packed_main, shape.packed_main_bytes(), "packed main"),
            (status, STATUS_BYTES, "status"),
        ];
        for (slice, bytes, label) in bindings {
            let end = slice
                .offset
                .checked_add(bytes)
                .ok_or_else(|| anyhow!("ratio4 main gather {label} range overflow"))?;
            if end > slice.buffer.size() {
                bail!("ratio4 main gather {label} out of bounds");
            }
        }
        for output in [bindings[3], bindings[4]] {
            for input in [bindings[0], bindings[1], bindings[2]] {
                if storage_buffer_slices_overlap(output.0, output.1, input.0, input.1)? {
                    bail!("ratio4 main gather output/input slices overlap");
                }
            }
        }
        if storage_buffer_slices_overlap(
            bindings[3].0,
            bindings[3].1,
            bindings[4].0,
            bindings[4].1,
        )? {
            bail!("ratio4 main gather packed/status slices overlap");
        }
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &bindings
                .iter()
                .map(|(slice, bytes, _)| (slice.buffer, slice.offset, *bytes))
                .collect::<Vec<_>>(),
        )?;
        Ok(S14Ratio4MainPageGatherDispatch { binder, shape })
    }

    /// # Safety
    /// descriptor、slice和页arena必须存活到command完成；status必须由调用方在
    /// 一次global-topk/gather事务开始前清零，且在attention前验收。
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Ratio4MainPageGatherDispatch,
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
        let push = std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), 16);
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push,
        );
        ctx.device
            .cmd_dispatch(command, dispatch.shape.selected_count, 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub fn validate_ratio4_main_gather_status(status: u32) -> Result<()> {
    let known = S14_RATIO4_GATHER_STATUS_SHAPE
        | S14_RATIO4_GATHER_STATUS_INDEX_OUT_OF_RANGE
        | S14_RATIO4_GATHER_STATUS_DUPLICATE_INDEX
        | S14_RATIO4_GATHER_STATUS_MISSING_PAGE
        | S14_RATIO4_GATHER_STATUS_ROW_OUT_OF_PAGE
        | S14_RATIO4_GATHER_STATUS_SOURCE_OUT_OF_RANGE
        | S14_RATIO4_GATHER_STATUS_BF16_NAN;
    if status == 0 {
        return Ok(());
    }
    if status & !known != 0 {
        bail!("ratio4 main gather unknown status 0x{status:08x}");
    }
    bail!("ratio4 main gather rejected data status=0x{status:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_table_maps_partial_second_page_without_materializing_unneeded_pages() {
        let shape = S14Ratio4MainGatherShape::new(513, 512, 512 * 256 + 256).unwrap();
        let table = build_ratio4_main_page_table(
            shape,
            &[
                S14Ratio4MaterializedMainPage {
                    page_index: 0,
                    source_word_offset: 0,
                    row_count: 512,
                },
                S14Ratio4MaterializedMainPage {
                    page_index: 1,
                    source_word_offset: 512 * 256,
                    row_count: 1,
                },
            ],
        )
        .unwrap();
        assert_eq!(table, vec![0, 512, 512 * 256, 1]);
        assert!(build_ratio4_main_page_table(
            shape,
            &[
                S14Ratio4MaterializedMainPage {
                    page_index: 1,
                    source_word_offset: 512 * 256,
                    row_count: 1,
                },
                S14Ratio4MaterializedMainPage {
                    page_index: 1,
                    source_word_offset: 512 * 256,
                    row_count: 1,
                },
            ]
        )
        .is_err());
    }
}
