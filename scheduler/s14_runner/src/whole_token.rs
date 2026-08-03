//! FullDepth43 单 token 的原子 DecoderState 所有权合同。
//!
//! 算子只能写未提交 candidate；43 层、final token 和全部合同都闭合后，
//! 才允许一次交换 position/epoch/bank。丢弃 candidate 即回滚。

use crate::{
    GraphProfile, NativeState, NativeStateArena, Position0CompressorInput, Position0StateError,
    StateLayoutError, TokenStateTxn, FULL_DEPTH_LAYERS, VOCAB_SIZE,
};
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
    /// `NativeState` BufferSlice 的真实已提交字节 owner。
    pub native_arena: NativeStateArena,
}

impl DecoderStateV1 {
    pub fn new(max_seq_len: u32, input_token_id: u32) -> Result<Self, WholeTokenError> {
        validate_token(input_token_id)?;
        let native =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, max_seq_len)?;
        let native_arena = NativeStateArena::initialized(&native)?;
        let state = Self {
            abi_version: DECODER_STATE_ABI_VERSION,
            commit_epoch: 0,
            position: 0,
            input_token_id,
            active_fixed_bank: 0,
            committed_tokens: Vec::new(),
            native,
            native_arena,
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
        self.native_arena.validate(&self.native)?;
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
        let token_state = TokenStateTxn::begin(&self.native, &self.native_arena)?;
        Ok(WholeTokenCandidate {
            base_epoch: self.commit_epoch,
            base_position: self.position,
            input_token_id,
            inactive_fixed_bank: 1 - self.active_fixed_bank,
            completed_layers: Vec::with_capacity(FULL_DEPTH_LAYERS.len()),
            predicted_token_id: None,
            staged_native: self.native.clone(),
            token_state,
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
    token_state: TokenStateTxn,
}

impl WholeTokenCandidate {
    pub fn inactive_fixed_bank(&self) -> u8 {
        self.inactive_fixed_bank
    }

    pub fn staged_native_mut(&mut self) -> &mut NativeState {
        &mut self.staged_native
    }

    /// 按 token position 写当前层的 window KV 与 compressor remainder。
    /// 真实已提交 arena 在 `commit` 前保持只读。
    pub fn stage_layer_state(
        &mut self,
        layer: u8,
        window_kv_bf16: &[u16],
        compressor_input: Position0CompressorInput<'_>,
    ) -> Result<(), WholeTokenError> {
        self.token_state.stage_layer(
            &self.staged_native,
            layer,
            window_kv_bf16,
            compressor_input,
        )?;
        Ok(())
    }

    /// 保留 position0 生产调用点的源码兼容入口。
    pub fn stage_position0_layer_state(
        &mut self,
        layer: u8,
        window_kv_bf16: &[u16],
        compressor_input: Position0CompressorInput<'_>,
    ) -> Result<(), WholeTokenError> {
        self.stage_layer_state(layer, window_kv_bf16, compressor_input)
    }

    pub fn position0_written_layer_count(&self) -> usize {
        self.token_state.written_layer_count()
    }

    pub fn position0_staged_bytes(&self) -> usize {
        self.token_state.staged_bytes()
    }

    pub fn stage_hc_state(&mut self, hc_streams_bf16: &[u16]) -> Result<(), WholeTokenError> {
        self.token_state
            .stage_hc_streams(&self.staged_native, hc_streams_bf16)?;
        Ok(())
    }

    /// 保留 position0 生产调用点的源码兼容入口。
    pub fn stage_position0_hc_state(
        &mut self,
        hc_streams_bf16: &[u16],
    ) -> Result<(), WholeTokenError> {
        self.stage_hc_state(hc_streams_bf16)
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
        if !self.token_state.has_layer(layer) {
            return Err(WholeTokenError::MissingPosition0LayerState(layer));
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
        self.commit_with_next_input(state, None)
    }

    /// 与 `commit` 相同的唯一原子提交点，但允许 prompt prefill 把下一输入
    /// 强制为下一枚真实 prompt token。当前 step 的模型预测仍原样写入 ledger。
    pub fn commit_with_next_input(
        self,
        state: &mut DecoderStateV1,
        forced_next_input: Option<u32>,
    ) -> Result<TokenRecord, WholeTokenError> {
        if let Some(token_id) = forced_next_input {
            validate_token(token_id)?;
        }
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
        let token_state = self.token_state;
        if !token_state.is_complete() {
            if !token_state.hc_written() {
                return Err(WholeTokenError::MissingPosition0HcState);
            }
            return Err(WholeTokenError::MissingPosition0LayerState(
                token_state.written_layer_count() as u8,
            ));
        }
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
        staged_native.validate_for(GraphProfile::FullDepth43NativeTop6)?;
        state.native_arena.validate(&staged_native)?;
        token_state.commit_into(&mut state.native_arena)?;
        state.native = staged_native;
        state.position = next_position;
        state.commit_epoch = next_epoch;
        state.active_fixed_bank = self.inactive_fixed_bank;
        state.input_token_id = forced_next_input.unwrap_or(predicted_token_id);
        state.committed_tokens.push(record.clone());
        debug_assert!(state.validate().is_ok());
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
    UnsupportedPosition(u32),
    EpochOverflow,
    LayerOrder { expected: u8, actual: u8 },
    DuplicateOrExtraLayer,
    IncompleteLayers,
    MissingFinal,
    DuplicateFinal,
    MissingPosition0LayerState(u8),
    MissingPosition0HcState,
    Position0State(Position0StateError),
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

impl From<Position0StateError> for WholeTokenError {
    fn from(value: Position0StateError) -> Self {
        Self::Position0State(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage_layer(candidate: &mut WholeTokenCandidate, layer: u8) {
        let position = candidate.base_position;
        let kv = vec![0x3f80 + layer as u16 + position as u16; 512];
        let position_bias = position as f32 * 100.0;
        let ratio = candidate.staged_native.kv[layer as usize].compress_ratio;
        match ratio {
            0 => candidate
                .stage_layer_state(layer, &kv, Position0CompressorInput::None)
                .unwrap(),
            4 if position % 4 == 3 => candidate
                .stage_layer_state(
                    layer,
                    &kv,
                    Position0CompressorInput::Ratio4Boundary {
                        main_kv: &vec![layer as f32 + 1.0 + position_bias; 1024],
                        main_score: &vec![layer as f32 + 2.0 + position_bias; 1024],
                        indexer_kv: &vec![layer as f32 + 3.0 + position_bias; 256],
                        indexer_score: &vec![layer as f32 + 4.0 + position_bias; 256],
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
                        main_kv: &vec![layer as f32 + 1.0 + position_bias; 1024],
                        main_score: &vec![layer as f32 + 2.0 + position_bias; 1024],
                        indexer_kv: &vec![layer as f32 + 3.0 + position_bias; 256],
                        indexer_score: &vec![layer as f32 + 4.0 + position_bias; 256],
                    },
                )
                .unwrap(),
            128 if position % 128 == 127 => candidate
                .stage_layer_state(
                    layer,
                    &kv,
                    Position0CompressorInput::Ratio128Boundary {
                        main_kv: &vec![layer as f32 + 1.0 + position_bias; 512],
                        main_score: &vec![layer as f32 + 2.0 + position_bias; 512],
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
                        main_kv: &vec![layer as f32 + 1.0 + position_bias; 512],
                        main_score: &vec![layer as f32 + 2.0 + position_bias; 512],
                    },
                )
                .unwrap(),
            _ => unreachable!(),
        }
    }

    fn complete_layers(candidate: &mut WholeTokenCandidate) {
        for &layer in &FULL_DEPTH_LAYERS {
            stage_layer(candidate, layer);
            candidate.complete_layer(layer).unwrap();
        }
        candidate
            .stage_hc_state(&vec![0x3f00 + candidate.base_position as u16; 4 * 4096])
            .unwrap();
    }

    fn decode_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn position0_through_position3_commit_first_ratio4_boundary_atomically() {
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

        let mut candidate = state.begin_token(1, 1, 5).unwrap();
        assert_eq!(candidate.inactive_fixed_bank(), 0);
        complete_layers(&mut candidate);
        candidate.complete_final(7).unwrap();
        let record = candidate.commit(&mut state).unwrap();
        assert_eq!(record.position, 1);
        assert_eq!(record.input_token_id, 5);
        assert_eq!(record.predicted_token_id, 7);
        assert_eq!(state.position, 2);
        assert_eq!(state.commit_epoch, 2);
        assert_eq!(state.active_fixed_bank, 0);
        assert_eq!(state.input_token_id, 7);
        assert_eq!(state.committed_tokens.len(), 2);
        assert_eq!(state.native.position, 2);
        state.validate().unwrap();

        for kv in &state.native.kv {
            let bytes = state.native_arena.slice_bytes(&kv.cache).unwrap();
            let row_bytes = 512 * 2;
            let expected0 = (0x3f80 + kv.layer as u16).to_le_bytes();
            let expected1 = (0x3f81 + kv.layer as u16).to_le_bytes();
            assert!(bytes[..row_bytes]
                .chunks_exact(2)
                .all(|chunk| chunk == expected0));
            assert!(bytes[row_bytes..2 * row_bytes]
                .chunks_exact(2)
                .all(|chunk| chunk == expected1));
        }
        for compressor in &state.native.compressors {
            let values = decode_f32(
                state
                    .native_arena
                    .slice_bytes(&compressor.kv_state)
                    .unwrap(),
            );
            let width = compressor.kv_state.shape[2] as usize;
            let row = if compressor.compress_ratio == 4 { 5 } else { 1 };
            assert!(values[row * width..(row + 1) * width]
                .iter()
                .all(|value| *value == compressor.layer as f32 + 101.0));
        }

        let mut candidate = state.begin_token(2, 2, 7).unwrap();
        complete_layers(&mut candidate);
        candidate.complete_final(9).unwrap();
        candidate.commit(&mut state).unwrap();
        let mut candidate = state.begin_token(3, 3, 9).unwrap();
        complete_layers(&mut candidate);
        candidate.complete_final(11).unwrap();
        candidate.commit(&mut state).unwrap();
        assert_eq!(state.position, 4);
        assert_eq!(state.commit_epoch, 4);

        for (position, predicted) in [(4u32, 13), (5, 15), (6, 17), (7, 19)] {
            let input = state.input_token_id;
            let mut candidate = state
                .begin_token(u64::from(position), position, input)
                .unwrap();
            complete_layers(&mut candidate);
            candidate.complete_final(predicted).unwrap();
            candidate.commit(&mut state).unwrap();
        }
        assert_eq!(state.position, 8);
        assert_eq!(state.commit_epoch, 8);

        for kv in state.native.kv.iter().filter(|kv| kv.compress_ratio == 4) {
            let cache = state.native_arena.slice_bytes(&kv.cache).unwrap();
            let block_offset = 128 * 512 * 2;
            let expected = (0x4100 + kv.layer as u16 + 3).to_le_bytes();
            assert!(cache[block_offset..block_offset + 512 * 2]
                .chunks_exact(2)
                .all(|chunk| chunk == expected));
            let indexer = state
                .native
                .indexers
                .iter()
                .find(|indexer| indexer.layer == kv.layer)
                .unwrap();
            let index_cache = state.native_arena.slice_bytes(&indexer.kv_cache).unwrap();
            let expected = (0x4200 + kv.layer as u16 + 3).to_le_bytes();
            assert!(index_cache[..128 * 2]
                .chunks_exact(2)
                .all(|chunk| chunk == expected));
            let expected_second = (0x4100 + kv.layer as u16 + 7).to_le_bytes();
            assert!(cache[block_offset + 512 * 2..block_offset + 2 * 512 * 2]
                .chunks_exact(2)
                .all(|chunk| chunk == expected_second));
            let expected_second_indexer = (0x4200 + kv.layer as u16 + 7).to_le_bytes();
            assert!(index_cache[128 * 2..2 * 128 * 2]
                .chunks_exact(2)
                .all(|chunk| chunk == expected_second_indexer));

            let compressor = state
                .native
                .compressors
                .iter()
                .find(|compressor| compressor.layer == kv.layer)
                .unwrap();
            let values = decode_f32(
                state
                    .native_arena
                    .slice_bytes(&compressor.kv_state)
                    .unwrap(),
            );
            for row in 0..4 {
                let expected = kv.layer as f32 + 1.0 + (row + 4) as f32 * 100.0;
                assert!(values[row * 1024..(row + 1) * 1024]
                    .iter()
                    .all(|value| *value == expected));
            }
        }
    }

    #[test]
    fn position127_commits_first_ratio128_block_atomically() {
        let mut state = DecoderStateV1::new(256, 9).unwrap();
        state.position = 127;
        state.native.position = 127;
        state.commit_epoch = 127;
        state.active_fixed_bank = 1;
        state.committed_tokens = (0..127)
            .map(|position| TokenRecord {
                position,
                input_token_id: 9,
                predicted_token_id: 9,
            })
            .collect();
        state.validate().unwrap();

        let mut candidate = state.begin_token(127, 127, 9).unwrap();
        complete_layers(&mut candidate);
        candidate.complete_final(11).unwrap();
        candidate.commit(&mut state).unwrap();

        assert_eq!(state.position, 128);
        assert_eq!(state.commit_epoch, 128);
        assert_eq!(state.active_fixed_bank, 0);
        for kv in state.native.kv.iter().filter(|kv| kv.compress_ratio == 128) {
            let cache = state.native_arena.slice_bytes(&kv.cache).unwrap();
            let compressed_row = 128 * 512 * 2;
            let expected = (0x4300 + kv.layer as u16 + 127).to_le_bytes();
            assert!(cache[compressed_row..compressed_row + 512 * 2]
                .chunks_exact(2)
                .all(|chunk| chunk == expected));
        }
        state.validate().unwrap();
    }

    #[test]
    fn forced_prefill_commits_prediction_but_uses_the_next_prompt_token_as_input() {
        let mut state = DecoderStateV1::new(16, 0).unwrap();
        let mut candidate = state.begin_token(0, 0, 0).unwrap();
        complete_layers(&mut candidate);
        candidate.complete_final(5).unwrap();
        let record = candidate
            .commit_with_next_input(&mut state, Some(128_803))
            .unwrap();

        assert_eq!(record.predicted_token_id, 5);
        assert_eq!(state.committed_tokens, [record]);
        assert_eq!(state.input_token_id, 128_803);
        assert_eq!(state.position, 1);
        assert_eq!(state.commit_epoch, 1);
        state.validate().unwrap();
    }

    #[test]
    fn position1_mid_token_failure_preserves_epoch1_bank1_and_one_record() {
        let mut state = DecoderStateV1::new(16, 0).unwrap();
        let mut position0 = state.begin_token(0, 0, 0).unwrap();
        complete_layers(&mut position0);
        position0.complete_final(5).unwrap();
        position0.commit(&mut state).unwrap();
        let snapshot = state.clone();

        let mut position1 = state.begin_token(1, 1, 5).unwrap();
        stage_layer(&mut position1, 0);
        position1.complete_layer(0).unwrap();
        assert_eq!(
            position1.complete_layer(1).unwrap_err(),
            WholeTokenError::MissingPosition0LayerState(1)
        );
        drop(position1);

        assert_eq!(state, snapshot);
        assert_eq!(state.commit_epoch, 1);
        assert_eq!(state.active_fixed_bank, 1);
        assert_eq!(state.committed_tokens.len(), 1);
    }

    #[test]
    fn dropping_candidates_at_l0_l42_or_final_preserves_committed_state() {
        let state = DecoderStateV1::new(16, 0).unwrap();
        let snapshot = state.clone();

        let mut after_l0 = state.begin_token(0, 0, 0).unwrap();
        stage_layer(&mut after_l0, 0);
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
            candidate.complete_layer(0).unwrap_err(),
            WholeTokenError::MissingPosition0LayerState(0)
        );
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
