//! FullDepth43 K=4/8 whole-token speculative block 的状态所有权合同。
//!
//! 与只保存 checkpoint ID 的 wire 合同不同，本模块持有每个 future position
//! 的完整 `DecoderStateV1`（含真实 `NativeStateArena` 字节）。权威状态在块完成、
//! 路由联合计划闭合并算出最长一致前缀前保持只读；失败或 drop 即天然 rollback。
//!
//! 本模块不把 K 次串行 step 伪装成 batch：生产构造入口硬要求后端报告一次
//! `batched_causal` whole-token forward。GPU 的 block-major layer graph 仍需由
//! `ssd_inference` 接入该合同。

use crate::{
    build_full_depth_causal_batch_plan, DecoderStateV1, FullDepthCausalBatchPlan, RouteDecision,
    VOCAB_SIZE,
};
use std::fmt;

pub const SPECULATIVE_WHOLE_TOKEN_BLOCK_SIZES: [usize; 2] = [4, 8];
pub const BATCHED_CAUSAL_WHOLE_TOKEN_MODE: &str = "batched_causal_whole_token";

/// 一次 future position 的 target 预测、43 层在线 top-6 与完整递归状态。
#[derive(Debug, Clone, PartialEq)]
pub struct BatchedWholeTokenPosition {
    pub predicted_token_id: u32,
    pub routes: Vec<RouteDecision>,
    pub checkpoint: DecoderStateV1,
}

/// 生产后端的一次 block 输出。`forward_calls` 必须精确为 1。
#[derive(Debug, Clone, PartialEq)]
pub struct BatchedWholeTokenOutput {
    pub mode: String,
    pub forward_calls: u32,
    pub positions: Vec<BatchedWholeTokenPosition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongestPrefixDecision {
    pub accepted_prefix: Vec<u32>,
    pub fallback_token_id: Option<u32>,
    pub rejected_draft_suffix: Vec<u32>,
    pub committed_token_ids: Vec<u32>,
    pub mismatch_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WholeTokenBlockCommit {
    pub decision: LongestPrefixDecision,
    pub base_position: u32,
    pub committed_position: u32,
    pub committed_epoch: u64,
    pub checkpoint_index: usize,
    pub page_plan: FullDepthCausalBatchPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WholeTokenBlockRollback {
    pub position: u32,
    pub commit_epoch: u64,
}

/// 一个尚未发布的 future block。它独占 K 个完整 checkpoint；权威 state
/// 直到 `commit_longest_prefix` 的最后一次赋值才发生变化。
#[derive(Debug, Clone, PartialEq)]
pub struct WholeTokenFutureBlock {
    base_state: DecoderStateV1,
    draft_token_ids: Vec<u32>,
    output: BatchedWholeTokenOutput,
    page_plan: FullDepthCausalBatchPlan,
}

impl WholeTokenFutureBlock {
    pub fn from_batched_output(
        authoritative: &DecoderStateV1,
        draft_token_ids: Vec<u32>,
        output: BatchedWholeTokenOutput,
    ) -> Result<Self, SpeculativeWholeTokenError> {
        authoritative
            .validate()
            .map_err(|error| block_error(format!("base DecoderState 非法: {error}")))?;
        let block_size = draft_token_ids.len();
        validate_block_size(block_size)?;
        validate_token_ids(&draft_token_ids)?;
        if output.mode != BATCHED_CAUSAL_WHOLE_TOKEN_MODE || output.forward_calls != 1 {
            return Err(block_error(
                "生产 future block 必须来自一次 batched_causal whole-token forward",
            ));
        }
        if output.positions.len() != block_size {
            return Err(block_error("future checkpoint 数必须与 K 一一对齐"));
        }
        let end = authoritative
            .position
            .checked_add(block_size as u32)
            .ok_or_else(|| block_error("future block position 溢出"))?;
        if end > authoritative.native.max_seq_len {
            return Err(block_error("future block 越出 max_seq_len"));
        }

        validate_checkpoint_chain(authoritative, &draft_token_ids, &output.positions)?;
        let routes_by_position = output
            .positions
            .iter()
            .map(|position| position.routes.clone())
            .collect::<Vec<_>>();
        let page_plan = build_full_depth_causal_batch_plan(&routes_by_position)
            .map_err(|error| block_error(format!("union top-6 page plan 非法: {error}")))?;
        Ok(Self {
            base_state: authoritative.clone(),
            draft_token_ids,
            output,
            page_plan,
        })
    }

    pub fn block_size(&self) -> usize {
        self.draft_token_ids.len()
    }

    pub fn base_position(&self) -> u32 {
        self.base_state.position
    }

    pub fn predicted_token_ids(&self) -> Vec<u32> {
        self.output
            .positions
            .iter()
            .map(|position| position.predicted_token_id)
            .collect()
    }

    pub fn page_plan(&self) -> &FullDepthCausalBatchPlan {
        &self.page_plan
    }

    pub fn decision(&self) -> LongestPrefixDecision {
        decide_longest_prefix(&self.draft_token_ids, &self.predicted_token_ids())
            .expect("constructor 已验证等长非空 token block")
    }

    /// 返回最长前缀最终会发布的同一份 host checkpoint，但不修改 authoritative state。
    /// device prefix prepare 必须使用该 checkpoint 的 position/epoch/bank 做同源校验。
    pub fn selected_checkpoint(
        &self,
    ) -> Result<(usize, DecoderStateV1), SpeculativeWholeTokenError> {
        let decision = self.decision();
        let checkpoint_index = decision
            .mismatch_index
            .unwrap_or_else(|| self.output.positions.len() - 1);
        let mut committed = self.output.positions[checkpoint_index].checkpoint.clone();
        if let Some(fallback) = decision.fallback_token_id {
            committed.input_token_id = fallback;
        }
        committed
            .validate()
            .map_err(|error| block_error(format!("待提交 checkpoint 非法: {error}")))?;
        let added_records = &committed.committed_tokens[self.base_state.committed_tokens.len()..];
        let added_tokens = added_records
            .iter()
            .map(|record| record.predicted_token_id)
            .collect::<Vec<_>>();
        if added_tokens != decision.committed_token_ids {
            return Err(block_error("checkpoint ledger 与最长一致前缀决策不闭合"));
        }
        Ok((checkpoint_index, committed))
    }

    /// 发布 mismatch fallback 所在 checkpoint，或全匹配时的最后 checkpoint。
    /// mismatch checkpoint 在验证阶段把 draft token teacher-force 为下一输入；
    /// 提交时必须恢复为 target fallback，后续被拒绝 checkpoint 完全不发布。
    pub fn commit_longest_prefix(
        self,
        authoritative: &mut DecoderStateV1,
    ) -> Result<WholeTokenBlockCommit, SpeculativeWholeTokenError> {
        if authoritative != &self.base_state {
            return Err(block_error(
                "权威 DecoderState 已变化，拒绝 stale future block",
            ));
        }
        let decision = self.decision();
        let (checkpoint_index, committed) = self.selected_checkpoint()?;

        let receipt = WholeTokenBlockCommit {
            decision,
            base_position: self.base_state.position,
            committed_position: committed.position,
            committed_epoch: committed.commit_epoch,
            checkpoint_index,
            page_plan: self.page_plan,
        };
        *authoritative = committed;
        debug_assert!(authoritative.validate().is_ok());
        Ok(receipt)
    }

    /// 显式 rollback 门。由于权威状态从未被 candidate 修改，成功即证明
    /// position/epoch/native arena 仍与轮次前完整快照逐字节相等。
    pub fn rollback(
        self,
        authoritative: &DecoderStateV1,
    ) -> Result<WholeTokenBlockRollback, SpeculativeWholeTokenError> {
        if authoritative != &self.base_state {
            return Err(block_error("rollback 时权威 DecoderState 已变化"));
        }
        Ok(WholeTokenBlockRollback {
            position: authoritative.position,
            commit_epoch: authoritative.commit_epoch,
        })
    }
}

pub fn decide_longest_prefix(
    draft_token_ids: &[u32],
    predicted_token_ids: &[u32],
) -> Result<LongestPrefixDecision, SpeculativeWholeTokenError> {
    if draft_token_ids.is_empty() || draft_token_ids.len() != predicted_token_ids.len() {
        return Err(block_error("draft/target token block 必须非空且等长"));
    }
    validate_token_ids(draft_token_ids)?;
    validate_token_ids(predicted_token_ids)?;
    let mismatch_index = draft_token_ids
        .iter()
        .zip(predicted_token_ids)
        .position(|(draft, target)| draft != target);
    Ok(match mismatch_index {
        Some(index) => {
            let accepted_prefix = draft_token_ids[..index].to_vec();
            let fallback = predicted_token_ids[index];
            let mut committed_token_ids = accepted_prefix.clone();
            committed_token_ids.push(fallback);
            LongestPrefixDecision {
                accepted_prefix,
                fallback_token_id: Some(fallback),
                rejected_draft_suffix: draft_token_ids[index..].to_vec(),
                committed_token_ids,
                mismatch_index: Some(index),
            }
        }
        None => LongestPrefixDecision {
            accepted_prefix: draft_token_ids.to_vec(),
            fallback_token_id: None,
            rejected_draft_suffix: Vec::new(),
            committed_token_ids: draft_token_ids.to_vec(),
            mismatch_index: None,
        },
    })
}

fn validate_checkpoint_chain(
    base: &DecoderStateV1,
    draft_token_ids: &[u32],
    positions: &[BatchedWholeTokenPosition],
) -> Result<(), SpeculativeWholeTokenError> {
    for (offset, position) in positions.iter().enumerate() {
        if position.predicted_token_id >= VOCAB_SIZE {
            return Err(block_error(format!(
                "future position {offset} predicted token 越界"
            )));
        }
        let checkpoint = &position.checkpoint;
        checkpoint
            .validate()
            .map_err(|error| block_error(format!("future checkpoint {offset} 非法: {error}")))?;
        let expected_position = base
            .position
            .checked_add(offset as u32 + 1)
            .ok_or_else(|| block_error("checkpoint position 溢出"))?;
        let expected_epoch = base
            .commit_epoch
            .checked_add(offset as u64 + 1)
            .ok_or_else(|| block_error("checkpoint epoch 溢出"))?;
        let expected_bank = base.active_fixed_bank ^ ((offset as u8 + 1) & 1);
        if checkpoint.position != expected_position
            || checkpoint.commit_epoch != expected_epoch
            || checkpoint.active_fixed_bank != expected_bank
            || checkpoint.input_token_id != draft_token_ids[offset]
            || checkpoint.native.max_seq_len != base.native.max_seq_len
        {
            return Err(block_error(format!(
                "future checkpoint {offset} position/epoch/bank/teacher-force 漂移"
            )));
        }
        if checkpoint.committed_tokens[..base.committed_tokens.len()] != base.committed_tokens[..] {
            return Err(block_error(format!(
                "future checkpoint {offset} 改写已提交 ledger"
            )));
        }
        let expected_len = base.committed_tokens.len() + offset + 1;
        if checkpoint.committed_tokens.len() != expected_len {
            return Err(block_error(format!(
                "future checkpoint {offset} ledger 长度漂移"
            )));
        }
        if offset > 0
            && checkpoint.committed_tokens[..expected_len - 1]
                != positions[offset - 1].checkpoint.committed_tokens[..]
        {
            return Err(block_error(format!(
                "future checkpoint {offset} 不是前一 checkpoint 的连续子代"
            )));
        }
        let record = checkpoint
            .committed_tokens
            .last()
            .expect("checkpoint position 已验证为正增长");
        let expected_input = if offset == 0 {
            base.input_token_id
        } else {
            draft_token_ids[offset - 1]
        };
        if record.position != base.position + offset as u32
            || record.input_token_id != expected_input
            || record.predicted_token_id != position.predicted_token_id
        {
            return Err(block_error(format!(
                "future checkpoint {offset} token ledger 与 target prediction 不闭合"
            )));
        }
    }
    Ok(())
}

fn validate_block_size(block_size: usize) -> Result<(), SpeculativeWholeTokenError> {
    if !SPECULATIVE_WHOLE_TOKEN_BLOCK_SIZES.contains(&block_size) {
        return Err(block_error("whole-token future block 只允许 K=4/8"));
    }
    Ok(())
}

fn validate_token_ids(token_ids: &[u32]) -> Result<(), SpeculativeWholeTokenError> {
    if token_ids.iter().any(|&token_id| token_id >= VOCAB_SIZE) {
        return Err(block_error("token ID 越出冻结 vocab"));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeWholeTokenError(String);

impl fmt::Display for SpeculativeWholeTokenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SpeculativeWholeTokenError {}

fn block_error(message: impl Into<String>) -> SpeculativeWholeTokenError {
    SpeculativeWholeTokenError(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        router_kind_for_layer, Position0CompressorInput, TokenRecord, WholeTokenCandidate,
        EXPERT_PAGE_BYTES, FULL_DEPTH_LAYERS,
    };

    fn route(layer: u8) -> RouteDecision {
        RouteDecision {
            layer,
            kind: router_kind_for_layer(layer).unwrap(),
            expert_ids: vec![1, 2, 3, 4, 5, 6],
            weights: vec![0.25; 6],
        }
    }

    fn routes() -> Vec<RouteDecision> {
        FULL_DEPTH_LAYERS
            .iter()
            .map(|&layer| route(layer))
            .collect()
    }

    fn stage_layer(candidate: &mut WholeTokenCandidate, layer: u8, position: u32) {
        let kv = vec![0x3f80 + layer as u16 + position as u16; 512];
        let bias = position as f32 * 100.0;
        let ratio = candidate.staged_native_mut().kv[layer as usize].compress_ratio;
        match ratio {
            0 => candidate
                .stage_layer_state(layer, &kv, Position0CompressorInput::None)
                .unwrap(),
            4 if position % 4 == 3 => candidate
                .stage_layer_state(
                    layer,
                    &kv,
                    Position0CompressorInput::Ratio4Boundary {
                        main_kv: &vec![layer as f32 + 1.0 + bias; 1024],
                        main_score: &vec![layer as f32 + 2.0 + bias; 1024],
                        indexer_kv: &vec![layer as f32 + 3.0 + bias; 256],
                        indexer_score: &vec![layer as f32 + 4.0 + bias; 256],
                        main_compressed_kv_bf16: &vec![
                            0x4100 + layer as u16 + position as u16;
                            512
                        ],
                        indexer_compressed_kv_bf16: &vec![
                            0x4200 + layer as u16 + position as u16;
                            128
                        ],
                    },
                )
                .unwrap(),
            4 => candidate
                .stage_layer_state(
                    layer,
                    &kv,
                    Position0CompressorInput::Ratio4 {
                        main_kv: &vec![layer as f32 + 1.0 + bias; 1024],
                        main_score: &vec![layer as f32 + 2.0 + bias; 1024],
                        indexer_kv: &vec![layer as f32 + 3.0 + bias; 256],
                        indexer_score: &vec![layer as f32 + 4.0 + bias; 256],
                    },
                )
                .unwrap(),
            128 if position % 128 == 127 => candidate
                .stage_layer_state(
                    layer,
                    &kv,
                    Position0CompressorInput::Ratio128Boundary {
                        main_kv: &vec![layer as f32 + 1.0 + bias; 512],
                        main_score: &vec![layer as f32 + 2.0 + bias; 512],
                        main_compressed_kv_bf16: &vec![
                            0x4300 + layer as u16 + position as u16;
                            512
                        ],
                    },
                )
                .unwrap(),
            128 => candidate
                .stage_layer_state(
                    layer,
                    &kv,
                    Position0CompressorInput::Ratio128 {
                        main_kv: &vec![layer as f32 + 1.0 + bias; 512],
                        main_score: &vec![layer as f32 + 2.0 + bias; 512],
                    },
                )
                .unwrap(),
            _ => unreachable!(),
        }
        candidate.complete_layer(layer).unwrap();
    }

    fn checkpoint_chain(
        base: &DecoderStateV1,
        draft: &[u32],
        predictions: &[u32],
    ) -> Vec<BatchedWholeTokenPosition> {
        let mut private = base.clone();
        draft
            .iter()
            .zip(predictions)
            .map(|(&teacher_force, &prediction)| {
                let position = private.position;
                let mut candidate = private
                    .begin_token(private.commit_epoch, position, private.input_token_id)
                    .unwrap();
                for &layer in &FULL_DEPTH_LAYERS {
                    stage_layer(&mut candidate, layer, position);
                }
                candidate.stage_hc_state(&vec![0x3f00; 4 * 4096]).unwrap();
                candidate.complete_final(prediction).unwrap();
                candidate
                    .commit_with_next_input(&mut private, Some(teacher_force))
                    .unwrap();
                BatchedWholeTokenPosition {
                    predicted_token_id: prediction,
                    routes: routes(),
                    checkpoint: private.clone(),
                }
            })
            .collect()
    }

    fn output(positions: Vec<BatchedWholeTokenPosition>) -> BatchedWholeTokenOutput {
        BatchedWholeTokenOutput {
            mode: BATCHED_CAUSAL_WHOLE_TOKEN_MODE.into(),
            forward_calls: 1,
            positions,
        }
    }

    #[test]
    fn k4_commits_real_checkpoint_through_mismatch_fallback_atomically() {
        let mut authoritative = DecoderStateV1::new(32, 7).unwrap();
        let base = authoritative.clone();
        let draft = vec![11, 12, 13, 14];
        let positions = checkpoint_chain(&base, &draft, &[11, 12, 99, 14]);
        let future =
            WholeTokenFutureBlock::from_batched_output(&authoritative, draft, output(positions))
                .unwrap();
        assert_eq!(authoritative, base);
        assert_eq!(future.page_plan().avoided_expert_loads, 43 * 18);
        assert_eq!(
            future.page_plan().union_expert_bytes,
            43 * 6 * EXPERT_PAGE_BYTES
        );

        let receipt = future.commit_longest_prefix(&mut authoritative).unwrap();
        assert_eq!(receipt.decision.accepted_prefix, [11, 12]);
        assert_eq!(receipt.decision.fallback_token_id, Some(99));
        assert_eq!(receipt.decision.rejected_draft_suffix, [13, 14]);
        assert_eq!(receipt.decision.committed_token_ids, [11, 12, 99]);
        assert_eq!(receipt.checkpoint_index, 2);
        assert_eq!(authoritative.position, 3);
        assert_eq!(authoritative.commit_epoch, 3);
        assert_eq!(authoritative.input_token_id, 99);
        assert_eq!(
            authoritative
                .committed_tokens
                .iter()
                .map(|record| record.predicted_token_id)
                .collect::<Vec<_>>(),
            [11, 12, 99]
        );
        authoritative.validate().unwrap();
    }

    #[test]
    fn k8_full_match_commits_all_and_union_plan_reuses_pages_across_positions() {
        let mut authoritative = DecoderStateV1::new(32, 7).unwrap();
        let draft = vec![11, 12, 13, 14, 15, 16, 17, 18];
        let positions = checkpoint_chain(&authoritative, &draft, &draft);
        let future = WholeTokenFutureBlock::from_batched_output(
            &authoritative,
            draft.clone(),
            output(positions),
        )
        .unwrap();
        assert_eq!(future.page_plan().avoided_expert_loads, 43 * 42);
        assert!((future.page_plan().expert_byte_reduction_ratio - 0.875).abs() < 1e-12);
        let receipt = future.commit_longest_prefix(&mut authoritative).unwrap();
        assert_eq!(receipt.decision.accepted_prefix, draft);
        assert_eq!(receipt.decision.fallback_token_id, None);
        assert_eq!(authoritative.position, 8);
        assert_eq!(authoritative.commit_epoch, 8);
        authoritative.validate().unwrap();
    }

    #[test]
    fn k4_block_can_cross_the_first_ratio128_boundary() {
        let mut authoritative = DecoderStateV1::new(256, 7).unwrap();
        authoritative.position = 124;
        authoritative.native.position = 124;
        authoritative.commit_epoch = 124;
        authoritative.committed_tokens = (0..124)
            .map(|position| TokenRecord {
                position,
                input_token_id: 7,
                predicted_token_id: 7,
            })
            .collect();
        authoritative.validate().unwrap();

        let draft = vec![11, 12, 13, 14];
        let positions = checkpoint_chain(&authoritative, &draft, &draft);
        let future = WholeTokenFutureBlock::from_batched_output(
            &authoritative,
            draft.clone(),
            output(positions),
        )
        .unwrap();
        let receipt = future.commit_longest_prefix(&mut authoritative).unwrap();
        assert_eq!(receipt.decision.accepted_prefix, draft);
        assert_eq!(authoritative.position, 128);
        assert_eq!(authoritative.commit_epoch, 128);
        authoritative.validate().unwrap();
    }

    #[test]
    fn rollback_is_full_state_equality_and_serial_forward_is_rejected() {
        let authoritative = DecoderStateV1::new(32, 7).unwrap();
        let draft = vec![11, 12, 13, 14];
        let positions = checkpoint_chain(&authoritative, &draft, &draft);
        let mut serial = output(positions.clone());
        serial.forward_calls = 4;
        assert!(
            WholeTokenFutureBlock::from_batched_output(&authoritative, draft.clone(), serial)
                .is_err()
        );

        let future =
            WholeTokenFutureBlock::from_batched_output(&authoritative, draft, output(positions))
                .unwrap();
        let receipt = future.rollback(&authoritative).unwrap();
        assert_eq!(receipt.position, 0);
        assert_eq!(receipt.commit_epoch, 0);
    }

    #[test]
    fn rejects_discontinuous_or_rewritten_checkpoint_chain() {
        let authoritative = DecoderStateV1::new(32, 7).unwrap();
        let draft = vec![11, 12, 13, 14];
        let mut positions = checkpoint_chain(&authoritative, &draft, &draft);
        positions[2].checkpoint.input_token_id = 55;
        assert!(WholeTokenFutureBlock::from_batched_output(
            &authoritative,
            draft,
            output(positions)
        )
        .is_err());
    }
}
