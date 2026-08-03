//! FullDepth43 token 的递归 KV/compressor/indexer 字节状态事务。
//!
//! `NativeState` 只描述稳定 ABI 布局；本模块拥有真实 arena。算子先把每层
//! 小型 write-set 写进 `TokenStateTxn`，完整 token 最终通过后
//! 才一次发布到已提交 arena。丢弃事务不会触碰已提交字节。

use crate::{BufferSlice, DType, GraphProfile, NativeState, FULL_DEPTH_LAYERS};
use serde::{Deserialize, Serialize};
use std::{fmt, ops::Range};

pub const POSITION0_KV_ELEMENTS: usize = 512;
pub const POSITION0_HC_ELEMENTS: usize = 4 * 4096;

/// position0 的 compressor 投影结果。这里只接收官方 compressor 的输入行，
/// 不在状态层伪造投影、pooling、norm、RoPE 或量化。
#[derive(Debug, Clone, Copy)]
pub enum Position0CompressorInput<'a> {
    None,
    Ratio4 {
        main_kv: &'a [f32],
        main_score: &'a [f32],
        indexer_kv: &'a [f32],
        indexer_score: &'a [f32],
    },
    /// ratio4 的 block 完成位置（position=3,7,...）。除了当前投影行，
    /// 还必须携带已经完成 norm/RoPE/量化的 main/indexer 压缩 KV。
    Ratio4Boundary {
        main_kv: &'a [f32],
        main_score: &'a [f32],
        indexer_kv: &'a [f32],
        indexer_score: &'a [f32],
        main_compressed_kv_bf16: &'a [u16],
        indexer_compressed_kv_bf16: &'a [u16],
    },
    Ratio128 {
        main_kv: &'a [f32],
        main_score: &'a [f32],
    },
    /// ratio128 的 block 完成位置（position=127,255,...）。除当前
    /// remainder 行外，还必须携带已完成 pooling/norm/RoPE 的主 KV。
    Ratio128Boundary {
        main_kv: &'a [f32],
        main_score: &'a [f32],
        main_compressed_kv_bf16: &'a [u16],
    },
}

/// `NativeState` 中所有 `BufferSlice` 的最小真实 owner。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeStateArena {
    arena_id: u64,
    bytes: Vec<u8>,
}

impl NativeStateArena {
    /// 建立 position0 之前的确定初态：KV/remainder 全零，所有未写 score 为
    /// `-inf`，与冻结 Python oracle 的 compressor 初始化完全一致。
    pub fn initialized(layout: &NativeState) -> Result<Self, Position0StateError> {
        validate_layout(layout)?;
        if layout.position != 0 {
            return Err(Position0StateError::NotPosition0(layout.position));
        }
        if layout.poisoned {
            return Err(Position0StateError::Layout(
                "poisoned NativeState 不得初始化 arena",
            ));
        }
        let len = usize::try_from(layout.arena_bytes)
            .map_err(|_| Position0StateError::ArenaTooLarge(layout.arena_bytes))?;
        let arena_id = layout.hc.streams.arena_id;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| Position0StateError::AllocationFailed(len))?;
        bytes.resize(len, 0);
        let mut arena = Self { arena_id, bytes };
        for compressor in &layout.compressors {
            arena.fill_f32(&compressor.score_state, f32::NEG_INFINITY)?;
        }
        for indexer in &layout.indexers {
            arena.fill_f32(&indexer.compressor_score_state, f32::NEG_INFINITY)?;
        }
        arena.validate(layout)?;
        Ok(arena)
    }

    /// 从已经由 production device checkpoint readback 完整覆盖的字节恢复 arena owner。
    ///
    /// 本入口只重建 host 所有权和布局身份，不替调用方证明 GPU producer/timeline。调用方必须
    /// 在调用前完成同一 checkpoint 的 timeline wait、范围闭合与传输回执校验；长度、arena id
    /// 或 `NativeState` 布局任一漂移仍会在这里 fail-closed。
    pub fn from_verified_checkpoint_bytes(
        layout: &NativeState,
        bytes: Vec<u8>,
    ) -> Result<Self, Position0StateError> {
        validate_layout(layout)?;
        let expected = usize::try_from(layout.arena_bytes)
            .map_err(|_| Position0StateError::ArenaTooLarge(layout.arena_bytes))?;
        if bytes.len() != expected {
            return Err(Position0StateError::ArenaLength {
                expected,
                actual: bytes.len(),
            });
        }
        let arena = Self {
            arena_id: layout.hc.streams.arena_id,
            bytes,
        };
        arena.validate(layout)?;
        Ok(arena)
    }

    pub fn arena_id(&self) -> u64 {
        self.arena_id
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn slice_bytes<'a>(&'a self, slice: &BufferSlice) -> Result<&'a [u8], Position0StateError> {
        let range = checked_range(slice, self.arena_id, self.bytes.len())?;
        Ok(&self.bytes[range])
    }

    pub fn validate(&self, layout: &NativeState) -> Result<(), Position0StateError> {
        validate_layout(layout)?;
        if self.arena_id != layout.hc.streams.arena_id {
            return Err(Position0StateError::ArenaIdDrift {
                expected: layout.hc.streams.arena_id,
                actual: self.arena_id,
            });
        }
        let expected = usize::try_from(layout.arena_bytes)
            .map_err(|_| Position0StateError::ArenaTooLarge(layout.arena_bytes))?;
        if self.bytes.len() != expected {
            return Err(Position0StateError::ArenaLength {
                expected,
                actual: self.bytes.len(),
            });
        }
        Ok(())
    }

    fn fill_f32(&mut self, slice: &BufferSlice, value: f32) -> Result<(), Position0StateError> {
        if slice.dtype != DType::F32 || !slice.bytes.is_multiple_of(4) {
            return Err(Position0StateError::Layout("F32 slice 契约漂移"));
        }
        let range = checked_range(slice, self.arena_id, self.bytes.len())?;
        let encoded = value.to_le_bytes();
        for chunk in self.bytes[range].chunks_exact_mut(4) {
            chunk.copy_from_slice(&encoded);
        }
        Ok(())
    }

    fn apply(&mut self, txn: &TokenStateTxn) -> Result<(), Position0StateError> {
        txn.validate_target(self)?;
        // 所有 range、重叠和完整性检查已经结束；下面只有等长内存复制，不再
        // 存在可恢复错误点，因此不会发布半个事务。
        for patch in &txn.patches {
            self.bytes[patch.range.clone()].copy_from_slice(&patch.bytes);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ArenaPatch {
    range: Range<usize>,
    bytes: Vec<u8>,
}

/// 连续 token 状态事务。每个 `position % 4 == 3` 的边界都会原子发布
/// ratio4 compressed KV/indexer block，并把 overlap 活动半区滚入下一组前半区。
/// ratio128 在每个 `position % 128 == 127` 边界原子发布一行
/// compressed KV；非边界仅更新当前 remainder 行。
#[derive(Debug, Clone)]
pub struct TokenStateTxn {
    arena_id: u64,
    arena_len: usize,
    position: u32,
    written_layers: [bool; 43],
    hc_written: bool,
    patches: Vec<ArenaPatch>,
}

/// 保留既有 position0 调用点的源码兼容别名。
pub type Position0StateTxn = TokenStateTxn;

impl TokenStateTxn {
    pub fn begin(
        layout: &NativeState,
        committed: &NativeStateArena,
    ) -> Result<Self, Position0StateError> {
        if layout.profile != GraphProfile::FullDepth43NativeTop6 {
            return Err(Position0StateError::Layout("事务 profile 漂移"));
        }
        committed.validate(layout)?;
        let mut txn = Self {
            arena_id: committed.arena_id,
            arena_len: committed.bytes.len(),
            position: layout.position,
            written_layers: [false; 43],
            hc_written: false,
            patches: Vec::with_capacity(FULL_DEPTH_LAYERS.len() * 5 + 1),
        };
        if layout.position % 4 == 3 {
            txn.stage_ratio4_overlap_prefix(layout, committed)?;
        }
        Ok(txn)
    }

    pub fn has_layer(&self, layer: u8) -> bool {
        self.written_layers
            .get(layer as usize)
            .copied()
            .unwrap_or(false)
    }

    pub fn written_layer_count(&self) -> usize {
        self.written_layers
            .iter()
            .filter(|&&written| written)
            .count()
    }

    /// 本 token 最终提交时实际复制的字节数；事务不复制整个 arena。
    pub fn staged_bytes(&self) -> usize {
        self.patches.iter().map(|patch| patch.bytes.len()).sum()
    }

    pub fn is_complete(&self) -> bool {
        self.hc_written && self.written_layers.iter().all(|&written| written)
    }

    pub fn hc_written(&self) -> bool {
        self.hc_written
    }

    /// 暂存本 token 的最终四流 HC 状态。它不是下一 token 的 embedding，但属于
    /// 原子 DecoderState 检查点；host/device bank 必须发布同一份字节。
    pub fn stage_hc_streams(
        &mut self,
        layout: &NativeState,
        hc_streams_bf16: &[u16],
    ) -> Result<(), Position0StateError> {
        validate_layout(layout)?;
        if layout.position != self.position {
            return Err(Position0StateError::PositionDrift {
                expected: self.position,
                actual: layout.position,
            });
        }
        if self.hc_written {
            return Err(Position0StateError::DuplicateHcStreams);
        }
        validate_bf16(hc_streams_bf16, POSITION0_HC_ELEMENTS, "hc_streams")?;
        let patch = patch_bf16_slice(
            &layout.hc.streams,
            hc_streams_bf16,
            self.arena_id,
            self.arena_len,
        )?;
        validate_patch_set(self.arena_len, self.patches.iter().chain([&patch]))?;
        self.patches.push(patch);
        self.hc_written = true;
        Ok(())
    }

    /// 暂存一层 token 状态。校验在任何 write-set 改变之前完成；非法输入
    /// 会 fail-closed，允许调用者修正后重试同一层。
    pub fn stage_layer(
        &mut self,
        layout: &NativeState,
        layer: u8,
        window_kv_bf16: &[u16],
        compressor_input: Position0CompressorInput<'_>,
    ) -> Result<(), Position0StateError> {
        validate_layout(layout)?;
        if layout.position != self.position {
            return Err(Position0StateError::PositionDrift {
                expected: self.position,
                actual: layout.position,
            });
        }
        if layer as usize >= self.written_layers.len() || !FULL_DEPTH_LAYERS.contains(&layer) {
            return Err(Position0StateError::InvalidLayer(layer));
        }
        if self.written_layers[layer as usize] {
            return Err(Position0StateError::DuplicateLayer(layer));
        }
        validate_bf16(window_kv_bf16, POSITION0_KV_ELEMENTS, "window_kv")?;

        let kv = layout
            .kv
            .iter()
            .find(|entry| entry.layer == layer)
            .ok_or(Position0StateError::InvalidLayer(layer))?;
        validate_kv_slice(&kv.cache)?;

        let mut additions = Vec::with_capacity(5);
        additions.push(patch_bf16_row(
            &kv.cache,
            (self.position % 128) as usize,
            window_kv_bf16,
            self.arena_id,
            self.arena_len,
        )?);

        match (kv.compress_ratio, compressor_input) {
            (0, Position0CompressorInput::None) => {}
            (
                4,
                input @ (Position0CompressorInput::Ratio4 { .. }
                | Position0CompressorInput::Ratio4Boundary { .. }),
            ) => {
                let (main_kv, main_score, indexer_kv, indexer_score, compressed) = match input {
                    Position0CompressorInput::Ratio4 {
                        main_kv,
                        main_score,
                        indexer_kv,
                        indexer_score,
                    } => (main_kv, main_score, indexer_kv, indexer_score, None),
                    Position0CompressorInput::Ratio4Boundary {
                        main_kv,
                        main_score,
                        indexer_kv,
                        indexer_score,
                        main_compressed_kv_bf16,
                        indexer_compressed_kv_bf16,
                    } => (
                        main_kv,
                        main_score,
                        indexer_kv,
                        indexer_score,
                        Some((main_compressed_kv_bf16, indexer_compressed_kv_bf16)),
                    ),
                    _ => unreachable!(),
                };
                let block_ready = self.position % 4 == 3;
                if block_ready != compressed.is_some() {
                    return Err(Position0StateError::CompressorBoundary {
                        layer,
                        position: self.position,
                        expected_ready: block_ready,
                    });
                }
                let compressor = layout
                    .compressors
                    .iter()
                    .find(|entry| entry.layer == layer && entry.compress_ratio == 4)
                    .ok_or(Position0StateError::Layout("ratio4 compressor 缺失"))?;
                let indexer = layout
                    .indexers
                    .iter()
                    .find(|entry| entry.layer == layer)
                    .ok_or(Position0StateError::Layout("ratio4 indexer 缺失"))?;
                validate_f32(main_kv, 1024, "ratio4 main_kv")?;
                validate_f32(main_score, 1024, "ratio4 main_score")?;
                validate_f32(indexer_kv, 256, "ratio4 indexer_kv")?;
                validate_f32(indexer_score, 256, "ratio4 indexer_score")?;
                // overlap remainder 的活动半区为 row=ratio+(position%ratio)。
                let remainder_row = 4 + (self.position % 4) as usize;
                additions.push(patch_f32_row(
                    &compressor.kv_state,
                    remainder_row,
                    main_kv,
                    self.arena_id,
                    self.arena_len,
                )?);
                additions.push(patch_f32_row(
                    &compressor.score_state,
                    remainder_row,
                    main_score,
                    self.arena_id,
                    self.arena_len,
                )?);
                additions.push(patch_f32_row(
                    &indexer.compressor_kv_state,
                    remainder_row,
                    indexer_kv,
                    self.arena_id,
                    self.arena_len,
                )?);
                additions.push(patch_f32_row(
                    &indexer.compressor_score_state,
                    remainder_row,
                    indexer_score,
                    self.arena_id,
                    self.arena_len,
                )?);
                if let Some((main_compressed, indexer_compressed)) = compressed {
                    validate_bf16(main_compressed, 512, "ratio4 main_compressed_kv")?;
                    validate_bf16(indexer_compressed, 128, "ratio4 indexer_compressed_kv")?;
                    let block = (self.position / 4) as usize;
                    additions.push(patch_bf16_row(
                        &kv.cache,
                        128 + block,
                        main_compressed,
                        self.arena_id,
                        self.arena_len,
                    )?);
                    additions.push(patch_bf16_row(
                        &indexer.kv_cache,
                        block,
                        indexer_compressed,
                        self.arena_id,
                        self.arena_len,
                    )?);
                    // 官方 overlap 状态机在完成块后执行 prefix = active half。
                    // 先前 row4..6 已由 begin() 从 committed arena 镜像到 row0..2；
                    // 当前 row7 同时发布到下一组的 row3。
                    additions.push(patch_f32_row(
                        &compressor.kv_state,
                        3,
                        main_kv,
                        self.arena_id,
                        self.arena_len,
                    )?);
                    additions.push(patch_f32_row(
                        &compressor.score_state,
                        3,
                        main_score,
                        self.arena_id,
                        self.arena_len,
                    )?);
                    additions.push(patch_f32_row(
                        &indexer.compressor_kv_state,
                        3,
                        indexer_kv,
                        self.arena_id,
                        self.arena_len,
                    )?);
                    additions.push(patch_f32_row(
                        &indexer.compressor_score_state,
                        3,
                        indexer_score,
                        self.arena_id,
                        self.arena_len,
                    )?);
                }
            }
            (
                128,
                input @ (Position0CompressorInput::Ratio128 { .. }
                | Position0CompressorInput::Ratio128Boundary { .. }),
            ) => {
                let (main_kv, main_score, compressed) = match input {
                    Position0CompressorInput::Ratio128 {
                        main_kv,
                        main_score,
                    } => (main_kv, main_score, None),
                    Position0CompressorInput::Ratio128Boundary {
                        main_kv,
                        main_score,
                        main_compressed_kv_bf16,
                    } => (main_kv, main_score, Some(main_compressed_kv_bf16)),
                    _ => unreachable!(),
                };
                let block_ready = self.position % 128 == 127;
                if block_ready != compressed.is_some() {
                    return Err(Position0StateError::CompressorBoundary {
                        layer,
                        position: self.position,
                        expected_ready: block_ready,
                    });
                }
                let compressor = layout
                    .compressors
                    .iter()
                    .find(|entry| entry.layer == layer && entry.compress_ratio == 128)
                    .ok_or(Position0StateError::Layout("ratio128 compressor 缺失"))?;
                validate_f32(main_kv, 512, "ratio128 main_kv")?;
                validate_f32(main_score, 512, "ratio128 main_score")?;
                let remainder_row = (self.position % 128) as usize;
                additions.push(patch_f32_row(
                    &compressor.kv_state,
                    remainder_row,
                    main_kv,
                    self.arena_id,
                    self.arena_len,
                )?);
                if let Some(main_compressed) = compressed {
                    validate_bf16(main_compressed, 512, "ratio128 main_compressed_kv")?;
                    let block = (self.position / 128) as usize;
                    additions.push(patch_bf16_row(
                        &kv.cache,
                        128 + block,
                        main_compressed,
                        self.arena_id,
                        self.arena_len,
                    )?);
                }
                additions.push(patch_f32_row(
                    &compressor.score_state,
                    remainder_row,
                    main_score,
                    self.arena_id,
                    self.arena_len,
                )?);
            }
            (expected, actual) => {
                return Err(Position0StateError::CompressorKind {
                    layer,
                    expected,
                    actual: actual.ratio_name(),
                });
            }
        }

        validate_patch_set(self.arena_len, self.patches.iter().chain(additions.iter()))?;
        self.patches.extend(additions);
        self.written_layers[layer as usize] = true;
        Ok(())
    }

    pub(crate) fn commit_into(
        self,
        target: &mut NativeStateArena,
    ) -> Result<(), Position0StateError> {
        if !self.written_layers.iter().all(|&written| written) {
            return Err(Position0StateError::IncompleteLayers {
                written: self.written_layer_count(),
            });
        }
        if !self.hc_written {
            return Err(Position0StateError::MissingHcStreams);
        }
        target.apply(&self)
    }

    fn validate_target(&self, target: &NativeStateArena) -> Result<(), Position0StateError> {
        if target.arena_id != self.arena_id {
            return Err(Position0StateError::ArenaIdDrift {
                expected: self.arena_id,
                actual: target.arena_id,
            });
        }
        if target.bytes.len() != self.arena_len {
            return Err(Position0StateError::ArenaLength {
                expected: self.arena_len,
                actual: target.bytes.len(),
            });
        }
        validate_patch_set(self.arena_len, self.patches.iter())?;
        Ok(())
    }

    fn stage_ratio4_overlap_prefix(
        &mut self,
        layout: &NativeState,
        committed: &NativeStateArena,
    ) -> Result<(), Position0StateError> {
        for compressor in layout
            .compressors
            .iter()
            .filter(|entry| entry.compress_ratio == 4)
        {
            let indexer = layout
                .indexers
                .iter()
                .find(|entry| entry.layer == compressor.layer)
                .ok_or(Position0StateError::Layout("ratio4 indexer 缺失"))?;
            for slice in [
                &compressor.kv_state,
                &compressor.score_state,
                &indexer.compressor_kv_state,
                &indexer.compressor_score_state,
            ] {
                for row in 0..3 {
                    self.patches
                        .push(copy_row_patch(slice, 4 + row, row, committed)?);
                }
            }
        }
        validate_patch_set(self.arena_len, self.patches.iter())
    }
}

impl Position0CompressorInput<'_> {
    fn ratio_name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ratio4 { .. } => "ratio4",
            Self::Ratio4Boundary { .. } => "ratio4-boundary",
            Self::Ratio128 { .. } => "ratio128",
            Self::Ratio128Boundary { .. } => "ratio128_boundary",
        }
    }
}

fn validate_layout(layout: &NativeState) -> Result<(), Position0StateError> {
    layout
        .validate_for(GraphProfile::FullDepth43NativeTop6)
        .map_err(Position0StateError::StateLayout)?;
    let expected =
        NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, layout.max_seq_len)
            .map_err(Position0StateError::StateLayout)?;
    if layout.hc != expected.hc
        || layout.kv != expected.kv
        || layout.compressors != expected.compressors
        || layout.indexers != expected.indexers
        || layout.arena_bytes != expected.arena_bytes
    {
        return Err(Position0StateError::Layout(
            "NativeState BufferSlice/shape/offset 契约漂移",
        ));
    }
    Ok(())
}

fn validate_kv_slice(slice: &BufferSlice) -> Result<(), Position0StateError> {
    if slice.dtype != DType::Bf16
        || slice.shape.len() != 3
        || slice.shape[0] != 1
        || slice.shape[1] < 128
        || slice.shape[2] != POSITION0_KV_ELEMENTS as u32
    {
        return Err(Position0StateError::Layout(
            "window KV slice shape/dtype 漂移",
        ));
    }
    Ok(())
}

fn validate_bf16(
    values: &[u16],
    expected: usize,
    label: &'static str,
) -> Result<(), Position0StateError> {
    if values.len() != expected {
        return Err(Position0StateError::Shape {
            label,
            expected,
            actual: values.len(),
        });
    }
    if values
        .iter()
        .any(|&bits| !f32::from_bits((bits as u32) << 16).is_finite())
    {
        return Err(Position0StateError::NonFinite(label));
    }
    Ok(())
}

fn validate_f32(
    values: &[f32],
    expected: usize,
    label: &'static str,
) -> Result<(), Position0StateError> {
    if values.len() != expected {
        return Err(Position0StateError::Shape {
            label,
            expected,
            actual: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Position0StateError::NonFinite(label));
    }
    Ok(())
}

fn patch_bf16_row(
    slice: &BufferSlice,
    row: usize,
    values: &[u16],
    arena_id: u64,
    arena_len: usize,
) -> Result<ArenaPatch, Position0StateError> {
    patch_row(
        slice,
        DType::Bf16,
        row,
        values.len() * 2,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        arena_id,
        arena_len,
    )
}

fn patch_bf16_slice(
    slice: &BufferSlice,
    values: &[u16],
    arena_id: u64,
    arena_len: usize,
) -> Result<ArenaPatch, Position0StateError> {
    if slice.dtype != DType::Bf16 || slice.bytes != (values.len() * 2) as u64 {
        return Err(Position0StateError::Layout("BF16 slice shape/dtype 漂移"));
    }
    let range = checked_range(slice, arena_id, arena_len)?;
    Ok(ArenaPatch {
        range,
        bytes: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    })
}

fn patch_f32_row(
    slice: &BufferSlice,
    row: usize,
    values: &[f32],
    arena_id: u64,
    arena_len: usize,
) -> Result<ArenaPatch, Position0StateError> {
    patch_row(
        slice,
        DType::F32,
        row,
        values.len() * 4,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        arena_id,
        arena_len,
    )
}

fn patch_row(
    slice: &BufferSlice,
    dtype: DType,
    row: usize,
    row_bytes: usize,
    bytes: Vec<u8>,
    arena_id: u64,
    arena_len: usize,
) -> Result<ArenaPatch, Position0StateError> {
    if slice.dtype != dtype || slice.shape.len() != 3 || slice.shape[0] != 1 {
        return Err(Position0StateError::Layout(
            "remainder row shape/dtype 漂移",
        ));
    }
    let rows = slice.shape[1] as usize;
    let width = slice.shape[2] as usize;
    let dtype_bytes = match dtype {
        DType::Bf16 => 2,
        DType::F32 => 4,
    };
    if row >= rows || width.checked_mul(dtype_bytes) != Some(row_bytes) || bytes.len() != row_bytes
    {
        return Err(Position0StateError::Layout(
            "remainder row width/offset 漂移",
        ));
    }
    let whole = checked_range(slice, arena_id, arena_len)?;
    let start = whole
        .start
        .checked_add(
            row.checked_mul(row_bytes)
                .ok_or(Position0StateError::Overflow)?,
        )
        .ok_or(Position0StateError::Overflow)?;
    let end = start
        .checked_add(row_bytes)
        .ok_or(Position0StateError::Overflow)?;
    if end > whole.end {
        return Err(Position0StateError::Layout("remainder row 越出 slice"));
    }
    Ok(ArenaPatch {
        range: start..end,
        bytes,
    })
}

fn copy_row_patch(
    slice: &BufferSlice,
    source_row: usize,
    target_row: usize,
    committed: &NativeStateArena,
) -> Result<ArenaPatch, Position0StateError> {
    if slice.shape.len() != 3 || slice.shape[0] != 1 {
        return Err(Position0StateError::Layout(
            "overlap rollover slice shape 漂移",
        ));
    }
    let dtype_bytes = match slice.dtype {
        DType::Bf16 => 2,
        DType::F32 => 4,
    };
    let row_bytes = (slice.shape[2] as usize)
        .checked_mul(dtype_bytes)
        .ok_or(Position0StateError::Overflow)?;
    let rows = slice.shape[1] as usize;
    if source_row >= rows || target_row >= rows {
        return Err(Position0StateError::Layout("overlap rollover row 越界"));
    }
    let whole = checked_range(slice, committed.arena_id, committed.bytes.len())?;
    let source_start = whole.start + source_row * row_bytes;
    let target_start = whole.start + target_row * row_bytes;
    Ok(ArenaPatch {
        range: target_start..target_start + row_bytes,
        bytes: committed.bytes[source_start..source_start + row_bytes].to_vec(),
    })
}

fn checked_range(
    slice: &BufferSlice,
    arena_id: u64,
    arena_len: usize,
) -> Result<Range<usize>, Position0StateError> {
    if slice.arena_id != arena_id {
        return Err(Position0StateError::ArenaIdDrift {
            expected: arena_id,
            actual: slice.arena_id,
        });
    }
    let start = usize::try_from(slice.offset).map_err(|_| Position0StateError::Overflow)?;
    let bytes = usize::try_from(slice.bytes).map_err(|_| Position0StateError::Overflow)?;
    let end = start
        .checked_add(bytes)
        .ok_or(Position0StateError::Overflow)?;
    if end > arena_len {
        return Err(Position0StateError::SliceOutOfBounds);
    }
    Ok(start..end)
}

fn validate_patch_set<'a>(
    arena_len: usize,
    patches: impl Iterator<Item = &'a ArenaPatch>,
) -> Result<(), Position0StateError> {
    let mut ranges: Vec<Range<usize>> = patches.map(|patch| patch.range.clone()).collect();
    for range in &ranges {
        if range.start >= range.end || range.end > arena_len {
            return Err(Position0StateError::SliceOutOfBounds);
        }
    }
    ranges.sort_unstable_by_key(|range| range.start);
    if ranges.windows(2).any(|pair| pair[0].end > pair[1].start) {
        return Err(Position0StateError::OverlappingPatch);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Position0StateError {
    StateLayout(crate::StateLayoutError),
    ArenaTooLarge(u64),
    AllocationFailed(usize),
    ArenaIdDrift {
        expected: u64,
        actual: u64,
    },
    ArenaLength {
        expected: usize,
        actual: usize,
    },
    NotPosition0(u32),
    UnsupportedPosition(u32),
    PositionDrift {
        expected: u32,
        actual: u32,
    },
    InvalidLayer(u8),
    DuplicateLayer(u8),
    DuplicateHcStreams,
    MissingHcStreams,
    CompressorKind {
        layer: u8,
        expected: u16,
        actual: &'static str,
    },
    CompressorBoundary {
        layer: u8,
        position: u32,
        expected_ready: bool,
    },
    Shape {
        label: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite(&'static str),
    IncompleteLayers {
        written: usize,
    },
    SliceOutOfBounds,
    OverlappingPatch,
    Overflow,
    Layout(&'static str),
}

impl fmt::Display for Position0StateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Position0StateError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_layout(max_seq_len: u32) -> NativeState {
        NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, max_seq_len).unwrap()
    }

    fn f32_values(count: usize, value: f32) -> Vec<f32> {
        vec![value; count]
    }

    fn stage_valid_layer(txn: &mut Position0StateTxn, layout: &NativeState, layer: u8) {
        let kv = vec![0x3f80 + layer as u16; POSITION0_KV_ELEMENTS];
        match layout.kv[layer as usize].compress_ratio {
            0 => txn
                .stage_layer(layout, layer, &kv, Position0CompressorInput::None)
                .unwrap(),
            4 => txn
                .stage_layer(
                    layout,
                    layer,
                    &kv,
                    Position0CompressorInput::Ratio4 {
                        main_kv: &f32_values(1024, layer as f32 + 1.0),
                        main_score: &f32_values(1024, layer as f32 + 2.0),
                        indexer_kv: &f32_values(256, layer as f32 + 3.0),
                        indexer_score: &f32_values(256, layer as f32 + 4.0),
                    },
                )
                .unwrap(),
            128 => txn
                .stage_layer(
                    layout,
                    layer,
                    &kv,
                    Position0CompressorInput::Ratio128 {
                        main_kv: &f32_values(512, layer as f32 + 1.0),
                        main_score: &f32_values(512, layer as f32 + 2.0),
                    },
                )
                .unwrap(),
            ratio => panic!("unexpected ratio {ratio}"),
        }
    }

    fn decode_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn arena_owns_exact_layout_and_initializes_score_remainders_to_negative_infinity() {
        let layout = full_layout(16);
        let arena = NativeStateArena::initialized(&layout).unwrap();
        assert_eq!(arena.len(), layout.arena_bytes as usize);
        for compressor in &layout.compressors {
            assert!(
                decode_f32(arena.slice_bytes(&compressor.score_state).unwrap())
                    .iter()
                    .all(|value| *value == f32::NEG_INFINITY)
            );
        }
        for indexer in &layout.indexers {
            assert!(
                decode_f32(arena.slice_bytes(&indexer.compressor_score_state).unwrap())
                    .iter()
                    .all(|value| *value == f32::NEG_INFINITY)
            );
        }
    }

    #[test]
    fn complete_position0_writes_windows_and_remainders_but_no_compressed_blocks() {
        let layout = full_layout(16);
        let mut arena = NativeStateArena::initialized(&layout).unwrap();
        let mut txn = Position0StateTxn::begin(&layout, &arena).unwrap();
        for &layer in &FULL_DEPTH_LAYERS {
            stage_valid_layer(&mut txn, &layout, layer);
        }
        let hc = vec![0x3f00; POSITION0_HC_ELEMENTS];
        txn.stage_hc_streams(&layout, &hc).unwrap();
        assert!(txn.is_complete());
        assert_eq!(txn.staged_bytes(), 373_760);
        txn.commit_into(&mut arena).unwrap();

        let expected_hc = 0x3f00u16.to_le_bytes();
        assert!(arena
            .slice_bytes(&layout.hc.streams)
            .unwrap()
            .chunks_exact(2)
            .all(|chunk| chunk == expected_hc));

        for kv in &layout.kv {
            let bytes = arena.slice_bytes(&kv.cache).unwrap();
            let expected = (0x3f80 + kv.layer as u16).to_le_bytes();
            assert!(bytes[..POSITION0_KV_ELEMENTS * 2]
                .chunks_exact(2)
                .all(|chunk| chunk == expected));
            assert!(bytes[POSITION0_KV_ELEMENTS * 2..]
                .iter()
                .all(|&byte| byte == 0));
        }
        for compressor in &layout.compressors {
            let row = if compressor.compress_ratio == 4 { 4 } else { 0 };
            let width = compressor.kv_state.shape[2] as usize;
            let kv = decode_f32(arena.slice_bytes(&compressor.kv_state).unwrap());
            let score = decode_f32(arena.slice_bytes(&compressor.score_state).unwrap());
            assert!(kv[row * width..(row + 1) * width]
                .iter()
                .all(|value| *value == compressor.layer as f32 + 1.0));
            assert!(score[row * width..(row + 1) * width]
                .iter()
                .all(|value| *value == compressor.layer as f32 + 2.0));
            for candidate_row in 0..compressor.kv_state.shape[1] as usize {
                if candidate_row != row {
                    assert!(kv[candidate_row * width..(candidate_row + 1) * width]
                        .iter()
                        .all(|value| *value == 0.0));
                    assert!(score[candidate_row * width..(candidate_row + 1) * width]
                        .iter()
                        .all(|value| *value == f32::NEG_INFINITY));
                }
            }
        }
        // Indexer kv_cache 只保存完成的 compressed block；position0 必须为零块。
        for indexer in &layout.indexers {
            assert!(arena
                .slice_bytes(&indexer.kv_cache)
                .unwrap()
                .iter()
                .all(|&byte| byte == 0));
            let kv = decode_f32(arena.slice_bytes(&indexer.compressor_kv_state).unwrap());
            let score = decode_f32(arena.slice_bytes(&indexer.compressor_score_state).unwrap());
            let width = indexer.compressor_kv_state.shape[2] as usize;
            assert!(kv[4 * width..5 * width]
                .iter()
                .all(|value| *value == indexer.layer as f32 + 3.0));
            assert!(score[4 * width..5 * width]
                .iter()
                .all(|value| *value == indexer.layer as f32 + 4.0));
            for row in [0usize, 1, 2, 3, 5, 6, 7] {
                assert!(kv[row * width..(row + 1) * width]
                    .iter()
                    .all(|value| *value == 0.0));
                assert!(score[row * width..(row + 1) * width]
                    .iter()
                    .all(|value| *value == f32::NEG_INFINITY));
            }
        }
    }

    #[test]
    fn invalid_input_is_fail_closed_and_does_not_consume_layer() {
        let layout = full_layout(16);
        let arena = NativeStateArena::initialized(&layout).unwrap();
        let mut txn = Position0StateTxn::begin(&layout, &arena).unwrap();
        let good_kv = vec![0x3f80; POSITION0_KV_ELEMENTS];

        assert_eq!(
            txn.stage_layer(&layout, 0, &good_kv[..511], Position0CompressorInput::None,)
                .unwrap_err(),
            Position0StateError::Shape {
                label: "window_kv",
                expected: 512,
                actual: 511,
            }
        );
        assert!(!txn.has_layer(0));

        let mut non_finite_kv = good_kv.clone();
        non_finite_kv[9] = 0x7f80;
        assert_eq!(
            txn.stage_layer(&layout, 0, &non_finite_kv, Position0CompressorInput::None,)
                .unwrap_err(),
            Position0StateError::NonFinite("window_kv")
        );
        assert!(!txn.has_layer(0));

        stage_valid_layer(&mut txn, &layout, 0);
        assert_eq!(
            txn.stage_layer(&layout, 0, &good_kv, Position0CompressorInput::None)
                .unwrap_err(),
            Position0StateError::DuplicateLayer(0)
        );
        assert_eq!(txn.written_layer_count(), 1);
        assert_eq!(
            txn.stage_layer(&layout, 43, &good_kv, Position0CompressorInput::None)
                .unwrap_err(),
            Position0StateError::InvalidLayer(43)
        );
    }

    #[test]
    fn ratio_kind_and_non_finite_projection_are_rejected_without_writes() {
        let layout = full_layout(16);
        let arena = NativeStateArena::initialized(&layout).unwrap();
        let mut txn = Position0StateTxn::begin(&layout, &arena).unwrap();
        let kv = vec![0x3f80; POSITION0_KV_ELEMENTS];
        let ratio4_layer = layout
            .kv
            .iter()
            .find(|entry| entry.compress_ratio == 4)
            .unwrap()
            .layer;
        assert!(matches!(
            txn.stage_layer(&layout, ratio4_layer, &kv, Position0CompressorInput::None,),
            Err(Position0StateError::CompressorKind { .. })
        ));
        assert!(!txn.has_layer(ratio4_layer));

        assert_eq!(
            txn.stage_layer(
                &layout,
                ratio4_layer,
                &kv,
                Position0CompressorInput::Ratio4Boundary {
                    main_kv: &f32_values(1024, 1.0),
                    main_score: &f32_values(1024, 2.0),
                    indexer_kv: &f32_values(256, 3.0),
                    indexer_score: &f32_values(256, 4.0),
                    main_compressed_kv_bf16: &vec![0x3f80; 512],
                    indexer_compressed_kv_bf16: &vec![0x3f80; 128],
                },
            )
            .unwrap_err(),
            Position0StateError::CompressorBoundary {
                layer: ratio4_layer,
                position: 0,
                expected_ready: false,
            }
        );
        assert!(!txn.has_layer(ratio4_layer));

        let mut bad = f32_values(1024, 1.0);
        bad[1] = f32::NAN;
        assert_eq!(
            txn.stage_layer(
                &layout,
                ratio4_layer,
                &kv,
                Position0CompressorInput::Ratio4 {
                    main_kv: &bad,
                    main_score: &f32_values(1024, 2.0),
                    indexer_kv: &f32_values(256, 3.0),
                    indexer_score: &f32_values(256, 4.0),
                },
            )
            .unwrap_err(),
            Position0StateError::NonFinite("ratio4 main_kv")
        );
        assert!(!txn.has_layer(ratio4_layer));
    }

    #[test]
    fn dropped_or_incomplete_transaction_never_changes_committed_arena() {
        let layout = full_layout(16);
        let mut arena = NativeStateArena::initialized(&layout).unwrap();
        let snapshot = arena.bytes().to_vec();
        let mut txn = Position0StateTxn::begin(&layout, &arena).unwrap();
        stage_valid_layer(&mut txn, &layout, 0);
        assert_eq!(
            txn.clone().commit_into(&mut arena).unwrap_err(),
            Position0StateError::IncompleteLayers { written: 1 }
        );
        assert_eq!(arena.bytes(), snapshot);
        drop(txn);
        assert_eq!(arena.bytes(), snapshot);
    }

    #[test]
    fn continuous_position_is_allowed_but_arena_reinitialization_and_tampered_layout_fail() {
        let mut layout = full_layout(16);
        let arena = NativeStateArena::initialized(&layout).unwrap();
        layout.position = 1;
        Position0StateTxn::begin(&layout, &arena).unwrap();
        assert_eq!(
            NativeStateArena::initialized(&layout).unwrap_err(),
            Position0StateError::NotPosition0(1)
        );

        layout.position = 0;
        let mut txn = Position0StateTxn::begin(&layout, &arena).unwrap();
        layout.kv[0].cache.offset += 256;
        let kv = vec![0x3f80; POSITION0_KV_ELEMENTS];
        assert_eq!(
            txn.stage_layer(&layout, 0, &kv, Position0CompressorInput::None)
                .unwrap_err(),
            Position0StateError::Layout("NativeState BufferSlice/shape/offset 契约漂移")
        );
        assert_eq!(txn.written_layer_count(), 0);
        assert_eq!(
            arena,
            NativeStateArena::initialized(&full_layout(16)).unwrap()
        );
    }
}
