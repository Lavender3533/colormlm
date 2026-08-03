//! ratio4 compressed history 的分页与原子 logical-length 合同。
//!
//! `INDEX_TOP_K=512` 是一次 attention 最多消费的块数，不是历史容量上限。
//! position2051 会在同一 token 生成逻辑 block512：committed 长度从512增长到513，
//! main/indexer 历史都必须落到第1页（0-based）的第0行，不能覆盖第0页。

use anyhow::{anyhow, bail, Context, Result};
use polaris_s14_runner::{BufferSlice, DType, GraphProfile, NativeState};
use std::collections::{BTreeMap, HashSet};
use std::ops::Range;

pub const S14_RATIO4_HISTORY_PAGE_ROWS: u32 = 512;
pub const S14_RATIO4_MAIN_ROW_BYTES: u64 = 512 * 2;
pub const S14_RATIO4_INDEXER_ROW_BYTES: u64 = 128 * 2;
pub const S14_RATIO4_ATTENTION_TOP_K: u32 = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Ratio4HistoryPage {
    pub page_index: u32,
    pub logical_rows: Range<u32>,
    pub main_state_range: Range<u64>,
    pub indexer_state_range: Range<u64>,
}

impl S14Ratio4HistoryPage {
    pub fn logical_len(&self) -> u32 {
        self.logical_rows.end - self.logical_rows.start
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Ratio4HistoryLayout {
    pub layer: u8,
    /// 已原子发布或候选准备发布的全局 compressed block 数。
    pub logical_len: u32,
    pub capacity_rows: u32,
    pub pages: Vec<S14Ratio4HistoryPage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Ratio4HistoryTarget {
    pub logical_block: u32,
    pub page_index: u32,
    pub row_in_page: u32,
    pub main_state_range: Range<u64>,
    pub indexer_state_range: Range<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Ratio4HistoryPublishPlan {
    pub position: u32,
    pub committed_logical_len: u32,
    pub candidate_logical_len: u32,
    /// 只有 position%4==3 时存在；最终 commit 必须把该行与 candidate logical length
    /// 一起发布，失败时二者都不得可见。
    pub appended_target: Option<S14Ratio4HistoryTarget>,
}

impl S14Ratio4HistoryLayout {
    pub fn build(state: &NativeState, layer: u8, logical_len: u32) -> Result<Self> {
        state
            .validate_for(GraphProfile::FullDepth43NativeTop6)
            .context("validate ratio4 history native state")?;
        let kv = unique(
            state
                .kv
                .iter()
                .filter(|entry| entry.layer == layer && entry.compress_ratio == 4),
            &format!("L{layer} ratio4 main history"),
        )?;
        let indexer = unique(
            state.indexers.iter().filter(|entry| entry.layer == layer),
            &format!("L{layer} ratio4 indexer history"),
        )?;
        validate_slice(&kv.cache, DType::Bf16, 512, "main")?;
        validate_slice(&indexer.kv_cache, DType::Bf16, 128, "indexer")?;
        let main_rows = kv.cache.shape[1];
        let capacity_rows = main_rows
            .checked_sub(128)
            .ok_or_else(|| anyhow!("L{layer} ratio4 main history 缺少128行 window 前缀"))?;
        if indexer.kv_cache.shape[1] != capacity_rows {
            bail!(
                "L{layer} ratio4 main/indexer history capacity 漂移: main={capacity_rows} indexer={}",
                indexer.kv_cache.shape[1]
            );
        }
        if logical_len > capacity_rows {
            bail!("L{layer} ratio4 history logical length {logical_len} 超出容量 {capacity_rows}");
        }

        let mut pages =
            Vec::with_capacity(logical_len.div_ceil(S14_RATIO4_HISTORY_PAGE_ROWS) as usize);
        let mut logical_start = 0u32;
        while logical_start < logical_len {
            let logical_end = logical_start
                .checked_add(S14_RATIO4_HISTORY_PAGE_ROWS)
                .unwrap_or(u32::MAX)
                .min(logical_len);
            let main_start = kv
                .cache
                .offset
                .checked_add(u64::from(128 + logical_start) * S14_RATIO4_MAIN_ROW_BYTES)
                .ok_or_else(|| anyhow!("L{layer} ratio4 main page offset overflow"))?;
            let main_end = main_start
                .checked_add(u64::from(logical_end - logical_start) * S14_RATIO4_MAIN_ROW_BYTES)
                .ok_or_else(|| anyhow!("L{layer} ratio4 main page range overflow"))?;
            let indexer_start = indexer
                .kv_cache
                .offset
                .checked_add(u64::from(logical_start) * S14_RATIO4_INDEXER_ROW_BYTES)
                .ok_or_else(|| anyhow!("L{layer} ratio4 indexer page offset overflow"))?;
            let indexer_end = indexer_start
                .checked_add(u64::from(logical_end - logical_start) * S14_RATIO4_INDEXER_ROW_BYTES)
                .ok_or_else(|| anyhow!("L{layer} ratio4 indexer page range overflow"))?;
            checked_subrange(&kv.cache, &(main_start..main_end), "main page")?;
            checked_subrange(
                &indexer.kv_cache,
                &(indexer_start..indexer_end),
                "indexer page",
            )?;
            pages.push(S14Ratio4HistoryPage {
                page_index: logical_start / S14_RATIO4_HISTORY_PAGE_ROWS,
                logical_rows: logical_start..logical_end,
                main_state_range: main_start..main_end,
                indexer_state_range: indexer_start..indexer_end,
            });
            logical_start = logical_end;
        }
        Ok(Self {
            layer,
            logical_len,
            capacity_rows,
            pages,
        })
    }

    pub fn target(&self, logical_block: u32) -> Result<S14Ratio4HistoryTarget> {
        if logical_block >= self.logical_len {
            bail!(
                "L{} ratio4 logical block {} 超出已发布长度 {}",
                self.layer,
                logical_block,
                self.logical_len
            );
        }
        let page_index = logical_block / S14_RATIO4_HISTORY_PAGE_ROWS;
        let row_in_page = logical_block % S14_RATIO4_HISTORY_PAGE_ROWS;
        let page = self
            .pages
            .get(page_index as usize)
            .ok_or_else(|| anyhow!("L{} ratio4 history page{page_index} 缺失", self.layer))?;
        if !page.logical_rows.contains(&logical_block) {
            bail!("L{} ratio4 history page logical range 漂移", self.layer);
        }
        let page_row = logical_block - page.logical_rows.start;
        let main_start = page
            .main_state_range
            .start
            .checked_add(u64::from(page_row) * S14_RATIO4_MAIN_ROW_BYTES)
            .ok_or_else(|| anyhow!("ratio4 main target overflow"))?;
        let indexer_start = page
            .indexer_state_range
            .start
            .checked_add(u64::from(page_row) * S14_RATIO4_INDEXER_ROW_BYTES)
            .ok_or_else(|| anyhow!("ratio4 indexer target overflow"))?;
        Ok(S14Ratio4HistoryTarget {
            logical_block,
            page_index,
            row_in_page,
            main_state_range: main_start..main_start + S14_RATIO4_MAIN_ROW_BYTES,
            indexer_state_range: indexer_start..indexer_start + S14_RATIO4_INDEXER_ROW_BYTES,
        })
    }

    pub fn selected_top_k(&self) -> u32 {
        self.logical_len.min(S14_RATIO4_ATTENTION_TOP_K)
    }
}

impl S14Ratio4HistoryPublishPlan {
    pub fn build(state: &NativeState, layer: u8, position: u32) -> Result<Self> {
        let next = position
            .checked_add(1)
            .ok_or_else(|| anyhow!("ratio4 history publish position overflow"))?;
        let committed_logical_len = position / 4;
        let candidate_logical_len = next / 4;
        let layout = S14Ratio4HistoryLayout::build(state, layer, candidate_logical_len)?;
        let appended_target = if next % 4 == 0 {
            if candidate_logical_len != committed_logical_len + 1 {
                bail!("ratio4 boundary logical length 不是单调+1");
            }
            Some(layout.target(candidate_logical_len - 1)?)
        } else {
            if candidate_logical_len != committed_logical_len {
                bail!("ratio4 非边界 logical length 漂移");
            }
            None
        };
        Ok(Self {
            position,
            committed_logical_len,
            candidate_logical_len,
            appended_target,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Ratio4SelectedRowCopy {
    pub logical_block: u32,
    pub source_main_range: Range<u64>,
    pub packed_slot: u32,
    pub packed_target_range: Range<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Ratio4SelectedPageBinding {
    pub page_index: u32,
    pub page_main_state_range: Range<u64>,
    pub row_copies: Vec<S14Ratio4SelectedRowCopy>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Ratio4SelectedMainPlan {
    pub logical_len: u32,
    pub selected_count: u32,
    pub packed_main_bytes: u64,
    pub page_bindings: Vec<S14Ratio4SelectedPageBinding>,
}

impl S14Ratio4SelectedMainPlan {
    /// 把全局 indexer top-k 映射成分页 main 读与连续 attention 工作集。
    /// `packed_slot` 严格保留 indexer 排名；attention 随后只消费局部0..selected_count。
    pub fn build(layout: &S14Ratio4HistoryLayout, selected: &[u32]) -> Result<Self> {
        if selected.len() != layout.selected_top_k() as usize {
            bail!(
                "ratio4 selected main 数量 {}/{} 与 logical length {} 漂移",
                selected.len(),
                layout.selected_top_k(),
                layout.logical_len
            );
        }
        let mut seen = HashSet::with_capacity(selected.len());
        let mut grouped: BTreeMap<u32, Vec<S14Ratio4SelectedRowCopy>> = BTreeMap::new();
        for (packed_slot, &logical_block) in selected.iter().enumerate() {
            if !seen.insert(logical_block) {
                bail!("ratio4 selected logical block {logical_block} 重复");
            }
            let target = layout.target(logical_block)?;
            let packed_slot = u32::try_from(packed_slot)?;
            let packed_start = u64::from(packed_slot) * S14_RATIO4_MAIN_ROW_BYTES;
            grouped
                .entry(target.page_index)
                .or_default()
                .push(S14Ratio4SelectedRowCopy {
                    logical_block,
                    source_main_range: target.main_state_range,
                    packed_slot,
                    packed_target_range: packed_start..packed_start + S14_RATIO4_MAIN_ROW_BYTES,
                });
        }
        let page_bindings = grouped
            .into_iter()
            .map(|(page_index, row_copies)| {
                let page = layout
                    .pages
                    .get(page_index as usize)
                    .ok_or_else(|| anyhow!("ratio4 selected page{page_index} 缺失"))?;
                Ok(S14Ratio4SelectedPageBinding {
                    page_index,
                    page_main_state_range: page.main_state_range.clone(),
                    row_copies,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            logical_len: layout.logical_len,
            selected_count: u32::try_from(selected.len())?,
            packed_main_bytes: u64::try_from(selected.len())? * S14_RATIO4_MAIN_ROW_BYTES,
            page_bindings,
        })
    }
}

fn validate_slice(slice: &BufferSlice, dtype: DType, width: u32, label: &str) -> Result<()> {
    if slice.dtype != dtype
        || slice.shape.len() != 3
        || slice.shape[0] != 1
        || slice.shape[2] != width
        || slice.bytes != u64::from(slice.shape[1]) * u64::from(width) * 2
    {
        bail!("ratio4 {label} history slice shape/dtype 漂移");
    }
    Ok(())
}

fn checked_subrange(slice: &BufferSlice, range: &Range<u64>, label: &str) -> Result<()> {
    let end = slice
        .offset
        .checked_add(slice.bytes)
        .ok_or_else(|| anyhow!("ratio4 {label} slice end overflow"))?;
    if range.start < slice.offset || range.end > end || range.start >= range.end {
        bail!("ratio4 {label} 超出 backing slice");
    }
    Ok(())
}

fn unique<'a, T>(mut values: impl Iterator<Item = &'a T>, label: &str) -> Result<&'a T> {
    let value = values.next().ok_or_else(|| anyhow!("{label} 缺失"))?;
    if values.next().is_some() {
        bail!("{label} 重复");
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(position: u32) -> NativeState {
        let mut state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        state.position = position;
        state
    }

    #[test]
    fn position2051_appends_block512_to_second_page_without_overwrite() {
        let state = state(2051);
        let publish = S14Ratio4HistoryPublishPlan::build(&state, 2, 2051).unwrap();
        assert_eq!(publish.committed_logical_len, 512);
        assert_eq!(publish.candidate_logical_len, 513);
        let target = publish.appended_target.unwrap();
        assert_eq!(
            (target.logical_block, target.page_index, target.row_in_page),
            (512, 1, 0)
        );

        let layout = S14Ratio4HistoryLayout::build(&state, 2, 513).unwrap();
        assert_eq!(layout.pages.len(), 2);
        assert_eq!(layout.pages[0].logical_rows, 0..512);
        assert_eq!(layout.pages[1].logical_rows, 512..513);
        assert_eq!(layout.pages[0].logical_len(), 512);
        assert_eq!(layout.pages[1].logical_len(), 1);
        assert_eq!(
            target.main_state_range.start,
            layout.pages[1].main_state_range.start
        );
        assert_eq!(
            target.indexer_state_range.start,
            layout.pages[1].indexer_state_range.start
        );
        assert_eq!(layout.selected_top_k(), 512);
    }

    #[test]
    fn selected_main_plan_preserves_global_ranking_across_pages() {
        let state = state(2051);
        let layout = S14Ratio4HistoryLayout::build(&state, 2, 513).unwrap();
        let mut selected: Vec<u32> = (0..511).collect();
        selected.insert(0, 512);
        let plan = S14Ratio4SelectedMainPlan::build(&layout, &selected).unwrap();
        assert_eq!(plan.selected_count, 512);
        assert_eq!(plan.packed_main_bytes, 512 * S14_RATIO4_MAIN_ROW_BYTES);
        assert_eq!(plan.page_bindings.len(), 2);
        let second = plan
            .page_bindings
            .iter()
            .find(|binding| binding.page_index == 1)
            .unwrap();
        assert_eq!(second.row_copies.len(), 1);
        assert_eq!(second.row_copies[0].logical_block, 512);
        assert_eq!(second.row_copies[0].packed_slot, 0);
        assert_eq!(second.row_copies[0].packed_target_range, 0..1024);
        assert!(S14Ratio4SelectedMainPlan::build(
            &layout,
            &(0..512).chain([0]).collect::<Vec<_>>()
        )
        .is_err());
    }

    #[test]
    fn non_boundary_keeps_logical_length_and_capacity_is_fail_closed() {
        let state = state(2052);
        let publish = S14Ratio4HistoryPublishPlan::build(&state, 2, 2052).unwrap();
        assert_eq!(publish.committed_logical_len, 513);
        assert_eq!(publish.candidate_logical_len, 513);
        assert!(publish.appended_target.is_none());
        assert!(S14Ratio4HistoryLayout::build(&state, 2, 1025).is_err());
    }
}
