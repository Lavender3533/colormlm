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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRatio4LayerEvidence {
    pub layer: u8,
    pub receipt: S14CausalBlockRatio4BoundaryRecordingReceipt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14CausalBlockProductionEvidenceSnapshot {
    pub base_position: u32,
    pub block_size: usize,
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
    ratio4_layers: BTreeMap<u8, S14CausalBlockRatio4BoundaryRecordingReceipt>,
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
                ratio4_layers: BTreeMap::new(),
                prefix_seal_receipt: None,
            })),
        })
    }

    pub(crate) fn record_completed_ratio4_layer(
        &self,
        layer: u8,
        receipt: S14CausalBlockRatio4BoundaryRecordingReceipt,
    ) -> Result<()> {
        receipt.validate()?;
        let mut state = self.lock()?;
        if state.base_position != receipt.positions[0]
            || state.block_size != receipt.positions.len()
        {
            bail!("ratio4 production evidence base/K 与 prefix arena 漂移");
        }
        if !state.expected_ratio4_layers.contains(&layer) {
            bail!("L{layer} 不是 FullDepth43 production ratio4 layer");
        }
        if state.ratio4_layers.contains_key(&layer) {
            bail!("L{layer} ratio4 production evidence 重复发布");
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
            .map(|(&layer, &receipt)| S14CausalBlockRatio4LayerEvidence { layer, receipt })
            .collect::<Vec<_>>();
        let position3_finalize_writeback_rollover_layers = ratio4_layer_evidence
            .iter()
            .filter(|entry| {
                entry.receipt.main_finalize_writeback_calls == 1
                    && entry.receipt.indexer_finalize_writeback_calls == 1
                    && entry.receipt.rollover_record_calls == 1
            })
            .map(|entry| entry.layer)
            .collect::<Vec<_>>();
        let position4_indexer_attention_layers = ratio4_layer_evidence
            .iter()
            .filter(|entry| {
                // receipt 自身按绝对 base 相位验证 boundary/tail/deferred post 与
                // 每-lane sparse index 数；不能继续把 base1/5 的固定“全部为1”
                // 当作任意连续 K4 的 production 证据。
                entry.receipt.validate().is_ok()
            })
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
            && state.prefix_seal_receipt.is_some()
            && serial_token_forward_calls == 0
            && cpu_fallback_calls == 0;

        Ok(S14CausalBlockProductionEvidenceSnapshot {
            base_position: state.base_position,
            block_size: state.block_size,
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
