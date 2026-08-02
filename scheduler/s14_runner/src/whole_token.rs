//! FullDepth43 单 token 的原子 DecoderState 所有权合同。
//!
//! 算子只能写未提交 candidate；43 层、final token 和全部合同都闭合后，
//! 才允许一次交换 position/epoch/bank。丢弃 candidate 即回滚。

use crate::{GraphProfile, NativeState, StateLayoutError, FULL_DEPTH_LAYERS, VOCAB_SIZE};
use serde::{Deserialize, Serialize};
use std::fmt;

pub const DECODER_STATE_ABI_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenRecord {
    pub position: u32,
    pub input_token_id: u32,
    pub predicted_token_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderStateV1 {
    pub abi_version: u32,
    pub commit_epoch: u64,
    /// 下一个待执行 token 的位置。
    pub position: u32,
    /// 当前待输入 token；成功提交后替换为本轮预测 token。
    pub input_token_id: u32,
    /// 17 MiB 固定递归状态的已提交 A/B bank。
    pub active_fixed_bank: u8,
    pub committed_tokens: Vec<TokenRecord>,
    pub native: NativeState,
}

impl DecoderStateV1 {
    pub fn new(max_seq_len: u32, input_token_id: u32) -> Result<Self, WholeTokenError> {
        validate_token(input_token_id)?;
        let native =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, max_seq_len)?;
        let state = Self {
            abi_version: DECODER_STATE_ABI_VERSION,
            commit_epoch: 0,
            position: 0,
            input_token_id,
            active_fixed_bank: 0,
            committed_tokens: Vec::new(),
            native,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<(), WholeTokenError> {
        if self.abi_version != DECODER_STATE_ABI_VERSION {
            return Err(WholeTokenError::Contract("DecoderState ABI version 漂移"));
        }
        validate_token(self.input_token_id)?;
        if self.active_fixed_bank > 1
            || self.native.profile != GraphProfile::FullDepth43NativeTop6
            || self.native.position != self.position
            || self.committed_tokens.len() != self.position as usize
        {
            return Err(WholeTokenError::Contract(
                "DecoderState position/bank/profile/token ledger 不闭合",
            ));
        }
        for (position, record) in self.committed_tokens.iter().enumerate() {
            if record.position != position as u32
                || record.input_token_id >= VOCAB_SIZE
                || record.predicted_token_id >= VOCAB_SIZE
            {
                return Err(WholeTokenError::Contract("DecoderState token ledger 漂移"));
            }
        }
        self.native
            .validate_for(GraphProfile::FullDepth43NativeTop6)?;
        Ok(())
    }

    pub fn begin_token(
        &self,
        expected_epoch: u64,
        expected_position: u32,
        input_token_id: u32,
    ) -> Result<WholeTokenCandidate, WholeTokenError> {
        self.validate()?;
        if self.commit_epoch != expected_epoch
            || self.position != expected_position
            || self.input_token_id != input_token_id
        {
            return Err(WholeTokenError::StaleRequest);
        }
        if self.position >= self.native.max_seq_len {
            return Err(WholeTokenError::SequenceExhausted);
        }
        Ok(WholeTokenCandidate {
            base_epoch: self.commit_epoch,
            base_position: self.position,
            input_token_id,
            inactive_fixed_bank: 1 - self.active_fixed_bank,
            completed_layers: Vec::with_capacity(FULL_DEPTH_LAYERS.len()),
            predicted_token_id: None,
            staged_native: self.native.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct WholeTokenCandidate {
    base_epoch: u64,
    base_position: u32,
    input_token_id: u32,
    inactive_fixed_bank: u8,
    completed_layers: Vec<u8>,
    predicted_token_id: Option<u32>,
    staged_native: NativeState,
}

impl WholeTokenCandidate {
    pub fn inactive_fixed_bank(&self) -> u8 {
        self.inactive_fixed_bank
    }

    pub fn staged_native_mut(&mut self) -> &mut NativeState {
        &mut self.staged_native
    }

    /// 只能按 L0→L42 顺序在该层全部数值合同通过后记录完成。
    pub fn complete_layer(&mut self, layer: u8) -> Result<(), WholeTokenError> {
        let expected = FULL_DEPTH_LAYERS
            .get(self.completed_layers.len())
            .copied()
            .ok_or(WholeTokenError::DuplicateOrExtraLayer)?;
        if layer != expected {
            return Err(WholeTokenError::LayerOrder {
                expected,
                actual: layer,
            });
        }
        self.completed_layers.push(layer);
        Ok(())
    }

    pub fn complete_final(&mut self, predicted_token_id: u32) -> Result<(), WholeTokenError> {
        if self.completed_layers.as_slice() != FULL_DEPTH_LAYERS {
            return Err(WholeTokenError::IncompleteLayers);
        }
        validate_token(predicted_token_id)?;
        if self
            .predicted_token_id
            .replace(predicted_token_id)
            .is_some()
        {
            return Err(WholeTokenError::DuplicateFinal);
        }
        Ok(())
    }

    /// 唯一提交点。调用前外部增长缓存只能写 committed length 之后的脏尾；
    /// 成功后由 worker 同时发布其 logical length。
    pub fn commit(self, state: &mut DecoderStateV1) -> Result<TokenRecord, WholeTokenError> {
        state.validate()?;
        if state.commit_epoch != self.base_epoch
            || state.position != self.base_position
            || state.input_token_id != self.input_token_id
            || state.active_fixed_bank == self.inactive_fixed_bank
        {
            return Err(WholeTokenError::StaleRequest);
        }
        if self.completed_layers.as_slice() != FULL_DEPTH_LAYERS {
            return Err(WholeTokenError::IncompleteLayers);
        }
        let predicted_token_id = self
            .predicted_token_id
            .ok_or(WholeTokenError::MissingFinal)?;
        if self.staged_native.position != self.base_position {
            return Err(WholeTokenError::Contract(
                "candidate 不得在唯一提交点前推进 native position",
            ));
        }
        let next_position = self
            .base_position
            .checked_add(1)
            .ok_or(WholeTokenError::SequenceExhausted)?;
        let next_epoch = self
            .base_epoch
            .checked_add(1)
            .ok_or(WholeTokenError::EpochOverflow)?;
        let record = TokenRecord {
            position: self.base_position,
            input_token_id: self.input_token_id,
            predicted_token_id,
        };
        let mut staged_native = self.staged_native;
        staged_native.position = next_position;
        state.native = staged_native;
        state.position = next_position;
        state.commit_epoch = next_epoch;
        state.active_fixed_bank = self.inactive_fixed_bank;
        state.input_token_id = predicted_token_id;
        state.committed_tokens.push(record.clone());
        state.validate()?;
        Ok(record)
    }
}

fn validate_token(token_id: u32) -> Result<(), WholeTokenError> {
    if token_id >= VOCAB_SIZE {
        return Err(WholeTokenError::InvalidToken(token_id));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WholeTokenError {
    StateLayout(StateLayoutError),
    InvalidToken(u32),
    StaleRequest,
    SequenceExhausted,
    EpochOverflow,
    LayerOrder { expected: u8, actual: u8 },
    DuplicateOrExtraLayer,
    IncompleteLayers,
    MissingFinal,
    DuplicateFinal,
    Contract(&'static str),
}

impl fmt::Display for WholeTokenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for WholeTokenError {}

impl From<StateLayoutError> for WholeTokenError {
    fn from(value: StateLayoutError) -> Self {
        Self::StateLayout(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_layers(candidate: &mut WholeTokenCandidate) {
        for &layer in &FULL_DEPTH_LAYERS {
            candidate.complete_layer(layer).unwrap();
        }
    }

    #[test]
    fn successful_token_has_one_atomic_commit_point() {
        let mut state = DecoderStateV1::new(16, 0).unwrap();
        let mut candidate = state.begin_token(0, 0, 0).unwrap();
        assert_eq!(candidate.inactive_fixed_bank(), 1);
        complete_layers(&mut candidate);
        candidate.complete_final(5).unwrap();
        let record = candidate.commit(&mut state).unwrap();
        assert_eq!(record.position, 0);
        assert_eq!(record.input_token_id, 0);
        assert_eq!(record.predicted_token_id, 5);
        assert_eq!(state.position, 1);
        assert_eq!(state.commit_epoch, 1);
        assert_eq!(state.active_fixed_bank, 1);
        assert_eq!(state.input_token_id, 5);
        assert_eq!(state.native.position, 1);
        state.validate().unwrap();
    }

    #[test]
    fn dropping_candidates_at_l0_l42_or_final_preserves_committed_state() {
        let state = DecoderStateV1::new(16, 0).unwrap();
        let snapshot = state.clone();

        let mut after_l0 = state.begin_token(0, 0, 0).unwrap();
        after_l0.complete_layer(0).unwrap();
        drop(after_l0);
        assert_eq!(state, snapshot);

        let mut after_l42 = state.begin_token(0, 0, 0).unwrap();
        complete_layers(&mut after_l42);
        drop(after_l42);
        assert_eq!(state, snapshot);

        let mut after_final = state.begin_token(0, 0, 0).unwrap();
        complete_layers(&mut after_final);
        after_final.complete_final(5).unwrap();
        drop(after_final);
        assert_eq!(state, snapshot);
    }

    #[test]
    fn rejects_stale_wrong_order_and_incomplete_candidates() {
        let mut state = DecoderStateV1::new(16, 0).unwrap();
        assert_eq!(
            state.begin_token(1, 0, 0).unwrap_err(),
            WholeTokenError::StaleRequest
        );
        let mut candidate = state.begin_token(0, 0, 0).unwrap();
        assert_eq!(
            candidate.complete_layer(1).unwrap_err(),
            WholeTokenError::LayerOrder {
                expected: 0,
                actual: 1
            }
        );
        assert_eq!(
            candidate.complete_final(5).unwrap_err(),
            WholeTokenError::IncompleteLayers
        );
        assert_eq!(
            candidate.commit(&mut state).unwrap_err(),
            WholeTokenError::IncompleteLayers
        );
    }
}
