//! GPU head 回读后完成 K=4/8 host checkpoint ledger 的 production finalizer。
//!
//! K-lane 数值图在 head token 未知时已经生成每个前缀的完整 native state snapshot；本模块只在
//! terminal GPU argmax 返回后补齐 token ledger/epoch/bank/teacher-force identity。它不生成或修改
//! native state 字节，也不接受缺少完整 arena owner 的占位 checkpoint。

use crate::{
    s14_causal_block_terminal_adapter::S14CausalBlockHostCandidateFinalizer,
    s14_head_chunk_argmax::S14HeadArgmaxResult,
};
use anyhow::{bail, Context, Result};
use polaris_s14_runner::{
    BatchedWholeTokenOutput, BatchedWholeTokenPosition, DecoderStateV1, GraphProfile,
    NativeState, NativeStateArena, RouteDecision, TokenRecord,
    BATCHED_CAUSAL_WHOLE_TOKEN_MODE, VOCAB_SIZE,
};

#[derive(Debug, Clone)]
pub struct S14CausalBlockPreparedHostCheckpoint {
    native: NativeState,
    native_arena: NativeStateArena,
}

impl S14CausalBlockPreparedHostCheckpoint {
    pub fn new(native: NativeState, native_arena: NativeStateArena) -> Result<Self> {
        native.validate_for(GraphProfile::FullDepth43NativeTop6)?;
        native_arena.validate(&native)?;
        Ok(Self {
            native,
            native_arena,
        })
    }

    pub fn position(&self) -> u32 {
        self.native.position
    }

    pub fn arena_bytes(&self) -> usize {
        self.native_arena.len()
    }
}

/// K份 device checkpoint 的同源 host readback。`draft_token_ids[lane]` 是该 checkpoint 发布后
/// 的 teacher-force next input，和 `WholeTokenFutureBlock` 的最长前缀合同一致。
#[derive(Debug)]
pub struct S14CausalBlockPreparedHostCandidateBatch {
    authoritative: DecoderStateV1,
    draft_token_ids: Vec<u32>,
    checkpoints: Vec<S14CausalBlockPreparedHostCheckpoint>,
}

impl S14CausalBlockPreparedHostCandidateBatch {
    pub fn new(
        authoritative: DecoderStateV1,
        draft_token_ids: Vec<u32>,
        checkpoints: Vec<S14CausalBlockPreparedHostCheckpoint>,
    ) -> Result<Self> {
        authoritative.validate()?;
        let block_size = draft_token_ids.len();
        if !matches!(block_size, 4 | 8)
            || checkpoints.len() != block_size
            || draft_token_ids.iter().any(|&token| token >= VOCAB_SIZE)
        {
            bail!("prepared host candidate 只接受等长 K=4/8 有效 draft/checkpoint");
        }
        let end_position = authoritative
            .position
            .checked_add(block_size as u32)
            .context("prepared host candidate position overflow")?;
        if end_position > authoritative.native.max_seq_len {
            bail!("prepared host candidate 越出 max_seq_len");
        }
        for (lane, checkpoint) in checkpoints.iter().enumerate() {
            let expected_position = authoritative
                .position
                .checked_add(lane as u32 + 1)
                .context("prepared checkpoint position overflow")?;
            if checkpoint.native.position != expected_position
                || checkpoint.native.max_seq_len != authoritative.native.max_seq_len
                || checkpoint.native.profile != GraphProfile::FullDepth43NativeTop6
                || checkpoint.native.poisoned
                || checkpoint.native_arena.arena_id() != authoritative.native_arena.arena_id()
                || checkpoint.native_arena.len() != authoritative.native_arena.len()
            {
                bail!("prepared checkpoint {lane} position/layout/arena identity 漂移");
            }
            checkpoint.native_arena.validate(&checkpoint.native)?;
        }
        Ok(Self {
            authoritative,
            draft_token_ids,
            checkpoints,
        })
    }
}

impl S14CausalBlockHostCandidateFinalizer for S14CausalBlockPreparedHostCandidateBatch {
    fn block_size(&self) -> usize {
        self.checkpoints.len()
    }

    fn base_position(&self) -> u32 {
        self.authoritative.position
    }

    fn complete_after_gpu_head(
        self: Box<Self>,
        head_results: &[S14HeadArgmaxResult],
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<BatchedWholeTokenOutput> {
        let Self {
            authoritative,
            draft_token_ids,
            checkpoints,
        } = *self;
        let block_size = checkpoints.len();
        if head_results.len() != block_size || routes_by_position.len() != block_size {
            bail!("prepared host candidate 的 GPU head/routes K 漂移");
        }

        let mut ledger = authoritative.committed_tokens.clone();
        let mut positions = Vec::with_capacity(block_size);
        for (lane, ((prepared, head), routes)) in checkpoints
            .into_iter()
            .zip(head_results)
            .zip(routes_by_position)
            .enumerate()
        {
            if head.token_id >= VOCAB_SIZE {
                bail!("prepared host candidate GPU head token 越界");
            }
            let token_position = authoritative
                .position
                .checked_add(lane as u32)
                .context("prepared host token position overflow")?;
            let input_token_id = if lane == 0 {
                authoritative.input_token_id
            } else {
                draft_token_ids[lane - 1]
            };
            ledger.push(TokenRecord {
                position: token_position,
                input_token_id,
                predicted_token_id: head.token_id,
            });
            let checkpoint = DecoderStateV1 {
                abi_version: authoritative.abi_version,
                commit_epoch: authoritative
                    .commit_epoch
                    .checked_add(lane as u64 + 1)
                    .context("prepared host checkpoint epoch overflow")?,
                position: token_position
                    .checked_add(1)
                    .context("prepared host checkpoint position overflow")?,
                input_token_id: draft_token_ids[lane],
                active_fixed_bank: authoritative.active_fixed_bank ^ ((lane as u8 + 1) & 1),
                committed_tokens: ledger.clone(),
                native: prepared.native,
                native_arena: prepared.native_arena,
            };
            checkpoint.validate()?;
            positions.push(BatchedWholeTokenPosition {
                predicted_token_id: head.token_id,
                routes: routes.clone(),
                checkpoint,
            });
        }

        Ok(BatchedWholeTokenOutput {
            mode: BATCHED_CAUSAL_WHOLE_TOKEN_MODE.to_owned(),
            forward_calls: 1,
            positions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polaris_s14_runner::{RouterKind, WholeTokenFutureBlock, FULL_DEPTH_LAYERS};

    fn routes(seed: u16) -> Vec<RouteDecision> {
        FULL_DEPTH_LAYERS
            .iter()
            .map(|&layer| RouteDecision {
                layer,
                kind: if layer <= 2 {
                    RouterKind::Hash
                } else {
                    RouterKind::Score
                },
                expert_ids: vec![
                    seed,
                    seed + 1,
                    seed + 2,
                    seed + 3,
                    seed + 4,
                    seed + 5,
                ],
                weights: vec![0.25; 6],
            })
            .collect()
    }

    #[test]
    fn gpu_head_completes_valid_teacher_forced_k4_checkpoint_chain() {
        let authoritative = DecoderStateV1::new(32, 0).unwrap();
        let draft = vec![5, 223, 939, 21];
        let checkpoints = (1..=4)
            .map(|position| {
                let mut native = authoritative.native.clone();
                native.position = position;
                S14CausalBlockPreparedHostCheckpoint::new(
                    native,
                    authoritative.native_arena.clone(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let routes_by_position = (0..4).map(routes).collect::<Vec<_>>();
        let heads = [5, 223, 17, 19]
            .into_iter()
            .map(|token_id| S14HeadArgmaxResult {
                token_id,
                logit: 1.0,
            })
            .collect::<Vec<_>>();
        let batch = S14CausalBlockPreparedHostCandidateBatch::new(
            authoritative.clone(),
            draft.clone(),
            checkpoints,
        )
        .unwrap();
        let output = Box::new(batch)
            .complete_after_gpu_head(&heads, &routes_by_position)
            .unwrap();
        let future = WholeTokenFutureBlock::from_batched_output(
            &authoritative,
            draft,
            output,
        )
        .unwrap();
        let decision = future.decision();
        assert_eq!(decision.accepted_prefix, vec![5, 223]);
        assert_eq!(decision.fallback_token_id, Some(17));
        assert_eq!(future.selected_checkpoint().unwrap().0, 2);
    }
}
