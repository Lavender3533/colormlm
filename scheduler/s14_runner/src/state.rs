use crate::{GraphProfile, COMPRESS_RATIOS, HC_STREAMS, HIDDEN_SIZE};
use serde::{Deserialize, Serialize};
use std::fmt;

const HEAD_DIM: u32 = 512;
const INDEX_HEAD_DIM: u32 = 128;
const WINDOW_SIZE: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    Bf16,
    F32,
}

impl DType {
    fn bytes(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferSlice {
    pub arena_id: u64,
    pub offset: u64,
    pub bytes: u64,
    pub dtype: DType,
    pub shape: Vec<u32>,
}

impl BufferSlice {
    fn new(
        arena_id: u64,
        cursor: &mut u64,
        dtype: DType,
        shape: &[u32],
    ) -> Result<Self, StateLayoutError> {
        *cursor = align_up(*cursor, 256)?;
        let elements = shape.iter().try_fold(1u64, |acc, &dim| {
            acc.checked_mul(dim as u64)
                .ok_or(StateLayoutError::Overflow)
        })?;
        let bytes = elements
            .checked_mul(dtype.bytes())
            .ok_or(StateLayoutError::Overflow)?;
        let offset = *cursor;
        *cursor = cursor
            .checked_add(bytes)
            .ok_or(StateLayoutError::Overflow)?;
        Ok(Self {
            arena_id,
            offset,
            bytes,
            dtype,
            shape: shape.to_vec(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HcState {
    /// `[batch=1, token=1, hc=4, hidden=4096]` BF16.
    pub streams: BufferSlice,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvState {
    pub layer: u8,
    pub compress_ratio: u16,
    /// Official implementation stores one shared latent KV vector per position.
    pub cache: BufferSlice,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressorState {
    pub layer: u8,
    pub compress_ratio: u16,
    pub kv_state: BufferSlice,
    pub score_state: BufferSlice,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexerState {
    pub layer: u8,
    pub kv_cache: BufferSlice,
    pub compressor_kv_state: BufferSlice,
    pub compressor_score_state: BufferSlice,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeState {
    pub profile: GraphProfile,
    pub position: u32,
    pub max_seq_len: u32,
    pub hc: HcState,
    pub kv: Vec<KvState>,
    pub compressors: Vec<CompressorState>,
    pub indexers: Vec<IndexerState>,
    pub arena_bytes: u64,
    /// Any operator failure poisons the recursive state; it cannot yield a token.
    pub poisoned: bool,
}

impl NativeState {
    pub fn decode_layout(max_seq_len: u32) -> Result<Self, StateLayoutError> {
        Self::decode_layout_for(GraphProfile::S14Top6, max_seq_len)
    }

    pub fn decode_layout_for(
        profile: GraphProfile,
        max_seq_len: u32,
    ) -> Result<Self, StateLayoutError> {
        if max_seq_len == 0 || max_seq_len > 1_048_576 {
            return Err(StateLayoutError::InvalidMaxSeq(max_seq_len));
        }
        let arena_id = 1;
        let mut cursor = 0u64;
        let hc = HcState {
            streams: BufferSlice::new(
                arena_id,
                &mut cursor,
                DType::Bf16,
                &[1, 1, HC_STREAMS, HIDDEN_SIZE],
            )?,
        };
        let mut kv = Vec::with_capacity(profile.layers().len());
        let mut compressors = Vec::new();
        let mut indexers = Vec::new();

        for &layer in profile.layers() {
            let ratio = COMPRESS_RATIOS[layer as usize];
            let cache_tokens = WINDOW_SIZE
                + if ratio == 0 {
                    0
                } else {
                    max_seq_len.div_ceil(ratio as u32)
                };
            kv.push(KvState {
                layer,
                compress_ratio: ratio,
                cache: BufferSlice::new(
                    arena_id,
                    &mut cursor,
                    DType::Bf16,
                    &[1, cache_tokens, HEAD_DIM],
                )?,
            });

            if ratio != 0 {
                let coefficient = if ratio == 4 { 2 } else { 1 };
                let shape = [1, coefficient * ratio as u32, coefficient * HEAD_DIM];
                compressors.push(CompressorState {
                    layer,
                    compress_ratio: ratio,
                    kv_state: BufferSlice::new(arena_id, &mut cursor, DType::F32, &shape)?,
                    score_state: BufferSlice::new(arena_id, &mut cursor, DType::F32, &shape)?,
                });

                if ratio == 4 {
                    let index_shape = [1, max_seq_len.div_ceil(4), INDEX_HEAD_DIM];
                    let compressor_shape = [1, 8, 2 * INDEX_HEAD_DIM];
                    indexers.push(IndexerState {
                        layer,
                        kv_cache: BufferSlice::new(
                            arena_id,
                            &mut cursor,
                            DType::Bf16,
                            &index_shape,
                        )?,
                        compressor_kv_state: BufferSlice::new(
                            arena_id,
                            &mut cursor,
                            DType::F32,
                            &compressor_shape,
                        )?,
                        compressor_score_state: BufferSlice::new(
                            arena_id,
                            &mut cursor,
                            DType::F32,
                            &compressor_shape,
                        )?,
                    });
                }
            }
        }
        let arena_bytes = align_up(cursor, 256)?;
        Ok(Self {
            profile,
            position: 0,
            max_seq_len,
            hc,
            kv,
            compressors,
            indexers,
            arena_bytes,
            poisoned: false,
        })
    }

    pub fn validate(&self) -> Result<(), StateLayoutError> {
        self.validate_for(self.profile)
    }

    pub fn validate_for(&self, profile: GraphProfile) -> Result<(), StateLayoutError> {
        if self.profile != profile {
            return Err(StateLayoutError::ProfileDrift);
        }
        if self.hc.streams.shape != [1, 1, HC_STREAMS, HIDDEN_SIZE]
            || self.hc.streams.dtype != DType::Bf16
        {
            return Err(StateLayoutError::InvalidHcLayout);
        }
        let layers: Vec<u8> = self.kv.iter().map(|item| item.layer).collect();
        if layers.as_slice() != profile.layers() {
            return Err(StateLayoutError::LayerDrift);
        }
        if self.position >= self.max_seq_len {
            return Err(StateLayoutError::PositionOutOfRange);
        }
        Ok(())
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, StateLayoutError> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or(StateLayoutError::Overflow)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateLayoutError {
    Overflow,
    InvalidMaxSeq(u32),
    InvalidHcLayout,
    LayerDrift,
    ProfileDrift,
    PositionOutOfRange,
}

impl fmt::Display for StateLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for StateLayoutError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_state_has_four_hc_streams_and_all_selected_cache_kinds() {
        let state = NativeState::decode_layout(4096).unwrap();
        state.validate().unwrap();
        assert_eq!(state.hc.streams.bytes, 4 * 4096 * 2);
        assert_eq!(state.kv.len(), 14);
        assert_eq!(state.compressors.len(), 12);
        assert_eq!(state.indexers.len(), 7);
        assert_eq!(state.arena_bytes, 14_401_536);
    }

    #[test]
    fn full_depth_profile_has_all_recursive_state_containers() {
        let state = NativeState::decode_layout_for(GraphProfile::FullDepthTop1, 4096).unwrap();
        state.validate_for(GraphProfile::FullDepthTop1).unwrap();
        assert_eq!(state.kv.len(), 43);
        assert_eq!(state.compressors.len(), 41);
        assert_eq!(state.indexers.len(), 21);
        assert_eq!(state.arena_bytes, 46_055_424);
    }
}
