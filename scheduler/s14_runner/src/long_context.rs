//! FullDepth43 长上下文的精确状态分层计划。
//!
//! 官方 ratio4/ratio128 状态使 200K 上下文在容量上可行，但完整
//! arena A/B 复制会无谓占用数 GiB。本模块从同一 `NativeState` ABI
//! 计算 hot window/remainder、indexer search、coarse history 与可分页
//! ratio4 main history，并固定 candidate 只持有 dirty write-set 的合同。

use crate::{GraphProfile, NativeState, StateLayoutError};
use serde::{Deserialize, Serialize};

pub const POLARIS_TARGET_CONTEXT_TOKENS: u32 = 200_000;
const WINDOW_TOKENS: u64 = 128;
const KV_ROW_BYTES: u64 = 512 * 2;
const INDEX_ROW_BYTES: u64 = 128 * 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LongContextMemoryPlan {
    pub format: String,
    pub profile: GraphProfile,
    pub max_seq_len: u32,
    pub flat_arena_bytes: u64,
    pub forbidden_flat_double_buffer_bytes: u64,
    pub hot_hc_window_remainder_bytes: u64,
    pub indexer_search_resident_bytes: u64,
    pub ratio128_coarse_history_resident_bytes: u64,
    pub ratio4_main_history_pageable_bytes: u64,
    pub alignment_bytes: u64,
    pub ordinary_token_dirty_bytes: u64,
    pub ratio4_boundary_extra_dirty_bytes: u64,
    pub ratio128_boundary_extra_dirty_bytes: u64,
    pub worst_boundary_dirty_bytes: u64,
    pub candidate_strategy: String,
    pub history_strategy: String,
    pub measurement_status: String,
}

impl LongContextMemoryPlan {
    pub fn build(max_seq_len: u32) -> Result<Self, StateLayoutError> {
        let state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, max_seq_len)?;

        let mut hot = state.hc.streams.bytes;
        let mut indexer_search = 0u64;
        let mut ratio128_history = 0u64;
        let mut ratio4_history = 0u64;
        let mut ordinary_dirty = state.hc.streams.bytes;
        let mut ratio4_boundary_extra = 0u64;
        let mut ratio128_boundary_extra = 0u64;

        for kv in &state.kv {
            let window_bytes = WINDOW_TOKENS * KV_ROW_BYTES;
            hot += window_bytes;
            ordinary_dirty += KV_ROW_BYTES;
            let history_bytes = kv.cache.bytes.saturating_sub(window_bytes);
            match kv.compress_ratio {
                0 => debug_assert_eq!(history_bytes, 0),
                4 => ratio4_history += history_bytes,
                128 => {
                    ratio128_history += history_bytes;
                    ratio128_boundary_extra += KV_ROW_BYTES;
                }
                _ => unreachable!("NativeState 只允许 0/4/128 compression ratio"),
            }
        }

        for compressor in &state.compressors {
            hot += compressor.kv_state.bytes + compressor.score_state.bytes;
            ordinary_dirty += row_bytes(&compressor.kv_state) + row_bytes(&compressor.score_state);
            if compressor.compress_ratio == 4 {
                // 边界时当前活动行还要发布到下一组 prefix。
                ratio4_boundary_extra +=
                    row_bytes(&compressor.kv_state) + row_bytes(&compressor.score_state);
            }
        }

        for indexer in &state.indexers {
            indexer_search += indexer.kv_cache.bytes;
            hot += indexer.compressor_kv_state.bytes + indexer.compressor_score_state.bytes;
            ordinary_dirty += row_bytes(&indexer.compressor_kv_state)
                + row_bytes(&indexer.compressor_score_state);
            ratio4_boundary_extra += row_bytes(&indexer.compressor_kv_state)
                + row_bytes(&indexer.compressor_score_state)
                + KV_ROW_BYTES
                + INDEX_ROW_BYTES;
        }

        let categorized = hot + indexer_search + ratio128_history + ratio4_history;
        let alignment_bytes = state.arena_bytes.saturating_sub(categorized);
        let worst_boundary_dirty_bytes =
            ordinary_dirty + ratio4_boundary_extra + ratio128_boundary_extra;

        Ok(Self {
            format: "polaris-s14-long-context-memory-plan-v1".into(),
            profile: GraphProfile::FullDepth43NativeTop6,
            max_seq_len,
            flat_arena_bytes: state.arena_bytes,
            forbidden_flat_double_buffer_bytes: state
                .arena_bytes
                .checked_mul(2)
                .ok_or(StateLayoutError::Overflow)?,
            hot_hc_window_remainder_bytes: hot,
            indexer_search_resident_bytes: indexer_search,
            ratio128_coarse_history_resident_bytes: ratio128_history,
            ratio4_main_history_pageable_bytes: ratio4_history,
            alignment_bytes,
            ordinary_token_dirty_bytes: ordinary_dirty,
            ratio4_boundary_extra_dirty_bytes: ratio4_boundary_extra,
            ratio128_boundary_extra_dirty_bytes: ratio128_boundary_extra,
            worst_boundary_dirty_bytes,
            candidate_strategy: "copy_on_write_dirty_pages_then_atomic_length_publish".into(),
            history_strategy:
                "ratio4_main_host_or_ssd_pages; indexer_and_ratio128_coarse_gpu_search_resident"
                    .into(),
            measurement_status: "exact_abi_capacity_plan_not_runtime_measurement".into(),
        })
    }

    pub fn target_200k() -> Result<Self, StateLayoutError> {
        Self::build(POLARIS_TARGET_CONTEXT_TOKENS)
    }
}

fn row_bytes(slice: &crate::BufferSlice) -> u64 {
    debug_assert_eq!(slice.shape.len(), 3);
    slice.bytes / u64::from(slice.shape[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_200k_is_tiered_and_candidate_is_dirty_only() {
        let plan = LongContextMemoryPlan::target_200k().unwrap();
        assert_eq!(plan.max_seq_len, 200_000);
        assert_eq!(
            plan.forbidden_flat_double_buffer_bytes,
            plan.flat_arena_bytes * 2
        );
        assert!(plan.ratio4_main_history_pageable_bytes > 1_000_000_000);
        assert!(plan.indexer_search_resident_bytes > 200_000_000);
        assert!(plan.worst_boundary_dirty_bytes < 1_000_000);
        assert!(plan.worst_boundary_dirty_bytes * 1_000 < plan.flat_arena_bytes);
        assert_eq!(
            plan.flat_arena_bytes,
            plan.hot_hc_window_remainder_bytes
                + plan.indexer_search_resident_bytes
                + plan.ratio128_coarse_history_resident_bytes
                + plan.ratio4_main_history_pageable_bytes
                + plan.alignment_bytes
        );
    }
}
