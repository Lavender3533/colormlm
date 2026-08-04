//! S14 K-block production 数据面的纯 host 证据账本。
//!
//! 账本只保存已经通过 GPU fence、sticky status、route decode 与 prefix seal 验收的回执；
//! 不持有 Vulkan context、buffer、command、fence 或任何资源 owner。

use crate::{
    s14_causal_block_prefix_state::S14CausalBlockPrefixStateSealReceipt,
    s14_causal_block_ratio4_boundary::S14CausalBlockRatio4BoundaryRecordingReceipt,
};
use anyhow::{bail, Context, Result};
use polaris_s14_runner::{COMPRESS_RATIOS, FULL_DEPTH_LAYERS};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

/// 一个物理 K-block 的 ratio4 分段事务回执。状态边界天然以 B4 为单位，
/// 因而这里保存有序 segment 列表而不是把证据结构锁死在 K4 或 K8。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRatio4SegmentedRecordingReceipt {
    pub base_position: u32,
    pub block_size: usize,
    pub owner_epoch: u64,
    pub segments: Vec<S14CausalBlockRatio4BoundaryRecordingReceipt>,
    pub state_transaction_calls: u32,
    pub attention_dispatch_calls: u32,
    pub attention_rows: u32,
    pub serial_token_forward_calls: u32,
    pub cpu_fallback_calls: u32,
}

impl S14CausalBlockRatio4SegmentedRecordingReceipt {
    pub fn from_segments(
        base_position: u32,
        owner_epoch: u64,
        segments: Vec<S14CausalBlockRatio4BoundaryRecordingReceipt>,
    ) -> Result<Self> {
        let segment_count =
            u32::try_from(segments.len()).context("ratio4 segment count 超出 u32")?;
        let block_size = segments
            .len()
            .checked_mul(4)
            .context("ratio4 segmented block size overflow")?;
        let receipt = Self {
            base_position,
            block_size,
            owner_epoch,
            state_transaction_calls: segment_count,
            attention_dispatch_calls: segments.iter().try_fold(0u32, |total, segment| {
                total
                    .checked_add(segment.attention_dispatch_calls)
                    .context("ratio4 attention dispatch count overflow")
            })?,
            attention_rows: segments.iter().try_fold(0u32, |total, segment| {
                total
                    .checked_add(segment.attention_rows)
                    .context("ratio4 attention row count overflow")
            })?,
            serial_token_forward_calls: segments.iter().try_fold(0u32, |total, segment| {
                total
                    .checked_add(segment.serial_token_forward_calls)
                    .context("ratio4 serial-token count overflow")
            })?,
            cpu_fallback_calls: segments.iter().try_fold(0u32, |total, segment| {
                total
                    .checked_add(segment.cpu_fallback_calls)
                    .context("ratio4 CPU fallback count overflow")
            })?,
            segments,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<()> {
        if self.owner_epoch == 0
            || self.block_size < 4
            || self.block_size % 4 != 0
            || self.segments.len() != self.block_size / 4
            || self.state_transaction_calls != self.segments.len() as u32
            || self.attention_dispatch_calls != self.segments.len() as u32
            || self.attention_rows != self.block_size as u32
            || self.serial_token_forward_calls != 0
            || self.cpu_fallback_calls != 0
        {
            bail!("ratio4 segmented receipt 的物理 K/事务计数漂移");
        }
        self.base_position
            .checked_add(u32::try_from(self.block_size).context("ratio4 block size 超出 u32")?)
            .context("ratio4 block position overflow")?;
        for (segment_index, segment) in self.segments.iter().enumerate() {
            segment.validate()?;
            let expected_base = self
                .base_position
                .checked_add(
                    u32::try_from(
                        segment_index
                            .checked_mul(4)
                            .context("segment lane overflow")?,
                    )
                    .context("segment lane 超出 u32")?,
                )
                .context("segment base position overflow")?;
            if segment.positions[0] != expected_base {
                bail!("ratio4 segmented receipt 的 B4 顺序/绝对 position 漂移");
            }
        }
        Ok(())
    }

    fn finalize_writeback_rollover_complete(&self) -> bool {
        self.segments.iter().all(|segment| {
            segment.main_finalize_writeback_calls == 1
                && segment.indexer_finalize_writeback_calls == 1
                && segment.rollover_record_calls == 1
        })
    }

    fn position4_indexer_attention_complete(&self) -> bool {
        self.validate().is_ok()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRatio4LayerEvidence {
    pub layer: u8,
    pub receipt: S14CausalBlockRatio4SegmentedRecordingReceipt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14CausalBlockProductionEvidenceSnapshot {
    pub base_position: u32,
    pub block_size: usize,
    pub ratio4_owner_epoch: Option<u64>,
    pub expected_ratio4_layers: Vec<u8>,
    pub ratio4_layer_evidence: Vec<S14CausalBlockRatio4LayerEvidence>,
    pub position3_finalize_writeback_rollover_layers: Vec<u8>,
    pub position4_indexer_attention_layers: Vec<u8>,
    pub prefix_seal_receipt: Option<S14CausalBlockPrefixStateSealReceipt>,
    pub serial_token_forward_calls: u64,
    pub cpu_fallback_calls: u64,
    /// 只表示本账本覆盖的 ratio4 + prefix 两组证据闭合，不代表 terminal/head 或整模完成。
    pub ratio4_prefix_evidence_complete: bool,
}

#[derive(Debug)]
struct EvidenceState {
    base_position: u32,
    block_size: usize,
    expected_ratio4_layers: BTreeSet<u8>,
    ratio4_owner_epoch: Option<u64>,
    ratio4_layers: BTreeMap<u8, S14CausalBlockRatio4SegmentedRecordingReceipt>,
    prefix_seal_receipt: Option<S14CausalBlockPrefixStateSealReceipt>,
}

/// 可跨 production owner 共享的纯 host 账本。写入口保持 crate-private；外部只能读取快照。
#[derive(Clone, Debug)]
pub struct S14CausalBlockProductionEvidenceLedger {
    state: Arc<Mutex<EvidenceState>>,
}

impl S14CausalBlockProductionEvidenceLedger {
    pub(crate) fn new(base_position: u32, block_size: usize) -> Result<Self> {
        let expected_ratio4_layers = FULL_DEPTH_LAYERS
            .iter()
            .copied()
            .filter(|&layer| COMPRESS_RATIOS.get(usize::from(layer)).copied() == Some(4))
            .collect::<BTreeSet<_>>();
        if expected_ratio4_layers.is_empty() {
            bail!("production evidence 无法从真实 FullDepth43 ABI 派生 ratio4 layers");
        }
        Ok(Self {
            state: Arc::new(Mutex::new(EvidenceState {
                base_position,
                block_size,
                expected_ratio4_layers,
                ratio4_owner_epoch: None,
                ratio4_layers: BTreeMap::new(),
                prefix_seal_receipt: None,
            })),
        })
    }

    pub(crate) fn record_completed_ratio4_layer(
        &self,
        layer: u8,
        receipt: S14CausalBlockRatio4SegmentedRecordingReceipt,
    ) -> Result<()> {
        receipt.validate()?;
        let mut state = self.lock()?;
        if state.base_position != receipt.base_position || state.block_size != receipt.block_size {
            bail!("ratio4 production evidence base/K 与 prefix arena 漂移");
        }
        if !state.expected_ratio4_layers.contains(&layer) {
            bail!("L{layer} 不是 FullDepth43 production ratio4 layer");
        }
        if state.ratio4_layers.contains_key(&layer) {
            bail!("L{layer} ratio4 production evidence 重复发布");
        }
        match state.ratio4_owner_epoch {
            Some(epoch) if epoch != receipt.owner_epoch => {
                bail!("L{layer} ratio4 owner epoch 与同 block 其他层漂移")
            }
            None => state.ratio4_owner_epoch = Some(receipt.owner_epoch),
            _ => {}
        }
        state.ratio4_layers.insert(layer, receipt);
        Ok(())
    }

    pub(crate) fn record_prefix_seal(
        &self,
        receipt: S14CausalBlockPrefixStateSealReceipt,
    ) -> Result<()> {
        let mut state = self.lock()?;
        validate_prefix_receipt(&state, receipt)?;
        if state.prefix_seal_receipt.is_some() {
            bail!("production prefix seal evidence 重复发布");
        }
        state.prefix_seal_receipt = Some(receipt);
        Ok(())
    }

    pub fn snapshot(&self) -> Result<S14CausalBlockProductionEvidenceSnapshot> {
        let state = self.lock()?;
        let ratio4_layer_evidence = state
            .ratio4_layers
            .iter()
            .map(|(&layer, receipt)| S14CausalBlockRatio4LayerEvidence {
                layer,
                receipt: receipt.clone(),
            })
            .collect::<Vec<_>>();
        let position3_finalize_writeback_rollover_layers = ratio4_layer_evidence
            .iter()
            .filter(|entry| entry.receipt.finalize_writeback_rollover_complete())
            .map(|entry| entry.layer)
            .collect::<Vec<_>>();
        let position4_indexer_attention_layers = ratio4_layer_evidence
            .iter()
            .filter(|entry| entry.receipt.position4_indexer_attention_complete())
            .map(|entry| entry.layer)
            .collect::<Vec<_>>();
        let serial_token_forward_calls = ratio4_layer_evidence
            .iter()
            .map(|entry| u64::from(entry.receipt.serial_token_forward_calls))
            .sum::<u64>()
            + state
                .prefix_seal_receipt
                .map_or(0, |receipt| u64::from(receipt.serial_token_forward_calls));
        let cpu_fallback_calls = ratio4_layer_evidence
            .iter()
            .map(|entry| u64::from(entry.receipt.cpu_fallback_calls))
            .sum::<u64>();
        let expected_ratio4_layers = state
            .expected_ratio4_layers
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let completed_layers = state.ratio4_layers.keys().copied().collect::<Vec<_>>();
        let ratio4_prefix_evidence_complete = completed_layers == expected_ratio4_layers
            && position3_finalize_writeback_rollover_layers == expected_ratio4_layers
            && position4_indexer_attention_layers == expected_ratio4_layers
            && state.ratio4_owner_epoch.is_some_and(|epoch| epoch != 0)
            && state.prefix_seal_receipt.is_some()
            && serial_token_forward_calls == 0
            && cpu_fallback_calls == 0;

        Ok(S14CausalBlockProductionEvidenceSnapshot {
            base_position: state.base_position,
            block_size: state.block_size,
            ratio4_owner_epoch: state.ratio4_owner_epoch,
            expected_ratio4_layers,
            ratio4_layer_evidence,
            position3_finalize_writeback_rollover_layers,
            position4_indexer_attention_layers,
            prefix_seal_receipt: state.prefix_seal_receipt,
            serial_token_forward_calls,
            cpu_fallback_calls,
            ratio4_prefix_evidence_complete,
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, EvidenceState>> {
        self.state
            .lock()
            .map_err(|_| anyhow::anyhow!("production evidence ledger poisoned"))
    }
}

fn validate_prefix_receipt(
    state: &EvidenceState,
    receipt: S14CausalBlockPrefixStateSealReceipt,
) -> Result<()> {
    let full_depth_layers = FULL_DEPTH_LAYERS.len();
    let sealed_prefix_layers = state
        .block_size
        .checked_mul(full_depth_layers)
        .context("production evidence prefix layer count overflow")?;
    let cumulative_lane_applications = state
        .block_size
        .checked_mul(state.block_size + 1)
        .and_then(|value| value.checked_div(2))
        .and_then(|value| value.checked_mul(full_depth_layers))
        .context("production evidence prefix triangular count overflow")?;
    if receipt.base_position != state.base_position
        || receipt.block_size != state.block_size
        || receipt.sealed_prefixes != state.block_size
        || receipt.sealed_prefix_layers != sealed_prefix_layers
        || receipt.cumulative_lane_applications != cumulative_lane_applications
        || receipt.serial_token_forward_calls != 0
    {
        bail!("production prefix seal evidence 与真实 K×FullDepth43 覆盖漂移");
    }
    Ok(())
}
