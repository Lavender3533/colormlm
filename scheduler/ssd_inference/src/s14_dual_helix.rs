//! Polaris S14 双螺旋最小闭环。
//!
//! 生成链只产生未提交候选；约束链只消费外部神经/工具 provider 的数值证据。
//! 本模块不读取文本、不实现关键词规则，也没有整链回退旧模型的执行入口。

#![forbid(unsafe_code)]

use std::{collections::BTreeSet, error::Error, fmt};

pub const DUAL_HELIX_ABI_VERSION: u32 = 1;
pub const SCORE_SCALE: u32 = 1_000_000;

pub type TokenId = u32;
pub type Digest = [u8; 32];
pub type HelixResult<T> = Result<T, HelixError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HelixError(String);

impl HelixError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for HelixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for HelixError {}

/// 确定性的百万分比，避免多个 provider 聚合时产生浮点遍历顺序差异。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Score(u32);

impl Score {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(SCORE_SCALE);

    pub fn new(parts_per_million: u32) -> HelixResult<Self> {
        if parts_per_million > SCORE_SCALE {
            return Err(HelixError::new("score 必须位于 0..=1_000_000"));
        }
        Ok(Self(parts_per_million))
    }

    pub fn from_f32(value: f32) -> HelixResult<Self> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(HelixError::new("浮点 score 必须是 [0, 1] 内的有限数"));
        }
        Ok(Self((value * SCORE_SCALE as f32).round() as u32))
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransactionBase {
    pub position: u64,
    pub commit_epoch: u64,
    pub state_digest: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalBudget {
    pub max_positions: usize,
    pub max_transfer_bytes: u64,
    pub max_flops: u64,
    pub max_latency_micros: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationRequest {
    pub transaction_id: u64,
    pub base: TransactionBase,
    pub vocab_size: u32,
    pub requested_positions: usize,
    pub semantic_plan_digest: Digest,
    pub budget: PhysicalBudget,
}

impl GenerationRequest {
    pub fn validate(&self) -> HelixResult<()> {
        if self.vocab_size == 0
            || self.requested_positions == 0
            || self.requested_positions > self.budget.max_positions
        {
            return Err(HelixError::new(
                "generation request 的 vocab/position budget 非法",
            ));
        }
        self.base
            .position
            .checked_add(self.requested_positions as u64)
            .ok_or_else(|| HelixError::new("generation position 溢出"))?;
        self.base
            .commit_epoch
            .checked_add(self.requested_positions as u64)
            .ok_or_else(|| HelixError::new("generation epoch 溢出"))?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateCheckpoint {
    pub checkpoint_id: u64,
    pub position: u64,
    pub commit_epoch: u64,
    pub state_digest: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationCandidate {
    pub position_offset: usize,
    pub token_id: TokenId,
    pub confidence: Score,
    pub uncertainty: Score,
    pub checkpoint: CandidateCheckpoint,
    /// 仅供外部 scorer/router 使用；本模块不解析 ID 内容。
    pub capability_hints: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationCandidateBlock {
    pub abi_version: u32,
    pub transaction_id: u64,
    pub chain_id: String,
    pub base: TransactionBase,
    pub vocab_size: u32,
    /// 生产编排会拒绝该位；局部旧算子只能实现 evidence/capsule trait。
    pub uses_whole_chain_legacy_fallback: bool,
    pub candidates: Vec<GenerationCandidate>,
}

impl GenerationCandidateBlock {
    pub fn validate(&self) -> HelixResult<()> {
        if self.abi_version != DUAL_HELIX_ABI_VERSION
            || self.chain_id.trim().is_empty()
            || self.vocab_size == 0
            || self.candidates.is_empty()
            || self.uses_whole_chain_legacy_fallback
        {
            return Err(HelixError::new(
                "generation block ABI/chain/vocab 非法或启用了整链旧模型回退",
            ));
        }
        for (offset, candidate) in self.candidates.iter().enumerate() {
            let position = self
                .base
                .position
                .checked_add(offset as u64 + 1)
                .ok_or_else(|| HelixError::new("candidate position 溢出"))?;
            let epoch = self
                .base
                .commit_epoch
                .checked_add(offset as u64 + 1)
                .ok_or_else(|| HelixError::new("candidate epoch 溢出"))?;
            if candidate.position_offset != offset
                || candidate.token_id >= self.vocab_size
                || candidate.checkpoint.position != position
                || candidate.checkpoint.commit_epoch != epoch
                || candidate
                    .capability_hints
                    .iter()
                    .any(|hint| hint.trim().is_empty())
            {
                return Err(HelixError::new(format!(
                    "candidate {offset} 的 token/checkpoint/capability 合同不闭合"
                )));
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn token_ids(&self) -> Vec<TokenId> {
        self.candidates
            .iter()
            .map(|candidate| candidate.token_id)
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintScope {
    pub start: usize,
    pub end: usize,
}

impl ConstraintScope {
    pub fn new(start: usize, end: usize) -> HelixResult<Self> {
        if start >= end {
            return Err(HelixError::new("constraint scope 必须是非空半开区间"));
        }
        Ok(Self { start, end })
    }

    pub fn contains(self, position_offset: usize) -> bool {
        self.start <= position_offset && position_offset < self.end
    }

    fn validate(self, block_len: usize) -> HelixResult<()> {
        if self.start >= self.end || self.end > block_len {
            return Err(HelixError::new("constraint scope 越出 candidate block"));
        }
        Ok(())
    }

    fn intersects(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepairDirective {
    pub scope: ConstraintScope,
    pub replacement_token_ids: Vec<TokenId>,
    pub repair_digest: Digest,
}

impl RepairDirective {
    fn validate(&self, block: &GenerationCandidateBlock) -> HelixResult<()> {
        self.scope.validate(block.len())?;
        if self.replacement_token_ids.is_empty()
            || self
                .replacement_token_ids
                .iter()
                .any(|token| *token >= block.vocab_size)
        {
            return Err(HelixError::new("repair replacement token 非法"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapsuleRequest {
    pub request_id: u64,
    pub capsule_id: String,
    pub revision: String,
    pub scope: ConstraintScope,
    pub expected_information_gain: Score,
    pub input_digest: Digest,
    pub budget: PhysicalBudget,
}

impl CapsuleRequest {
    fn validate(&self, block_len: usize) -> HelixResult<()> {
        self.scope.validate(block_len)?;
        if self.capsule_id.trim().is_empty() || self.revision.trim().is_empty() {
            return Err(HelixError::new("capsule ID/revision 不能为空"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintDecision {
    Accept,
    Repair(RepairDirective),
    Rollback { keep_prefix_len: usize },
    InvokeCapsule(CapsuleRequest),
}

impl ConstraintDecision {
    fn applicable_at(&self, evidence_scope: ConstraintScope, position: usize) -> bool {
        if !evidence_scope.contains(position) {
            return false;
        }
        match self {
            Self::Accept => true,
            Self::Repair(repair) => repair.scope.contains(position),
            Self::Rollback { keep_prefix_len } => *keep_prefix_len <= position,
            Self::InvokeCapsule(request) => request.scope.contains(position),
        }
    }

    fn next_position(&self, current: usize, block_len: usize) -> usize {
        match self {
            Self::Accept => current + 1,
            Self::Repair(repair) => repair.scope.end.max(current + 1),
            Self::Rollback { .. } => block_len,
            Self::InvokeCapsule(request) => request.scope.end.max(current + 1),
        }
    }

    fn safe_prefix(&self, decision_position: usize) -> usize {
        match self {
            Self::Accept => decision_position + 1,
            Self::Repair(_) | Self::InvokeCapsule(_) => decision_position,
            Self::Rollback { keep_prefix_len } => *keep_prefix_len,
        }
    }
}

/// 未提交候选只能由生成链产生；repair 必须重建受影响 checkpoint。
pub trait GenerationChain {
    fn chain_id(&self) -> &str;

    fn generate_candidates(
        &mut self,
        request: &GenerationRequest,
    ) -> HelixResult<GenerationCandidateBlock>;

    fn repair_candidates(
        &mut self,
        original: &GenerationCandidateBlock,
        repair: &RepairDirective,
        capsule: Option<&CapsuleResult>,
    ) -> HelixResult<GenerationCandidateBlock>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum EvidenceSource {
    Neural = 0,
    Tool = 1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceProviderDescriptor {
    pub provider_id: String,
    pub revision: String,
    pub source: EvidenceSource,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintEvidence {
    pub evidence_id: u64,
    pub scope: ConstraintScope,
    pub support: Score,
    pub confidence: Score,
    pub decision: ConstraintDecision,
    pub evidence_digest: Digest,
}

impl ConstraintEvidence {
    fn validate(&self, block: &GenerationCandidateBlock) -> HelixResult<()> {
        self.scope.validate(block.len())?;
        if self.support == Score::ZERO || self.confidence == Score::ZERO {
            return Err(HelixError::new("evidence support/confidence 不能为 0"));
        }
        match &self.decision {
            ConstraintDecision::Accept => {}
            ConstraintDecision::Repair(repair) => {
                repair.validate(block)?;
                if !self.scope.intersects(repair.scope) {
                    return Err(HelixError::new("repair 与 evidence scope 不相交"));
                }
            }
            ConstraintDecision::Rollback { keep_prefix_len } => {
                if *keep_prefix_len > self.scope.start {
                    return Err(HelixError::new(
                        "rollback keep_prefix 不能越过 evidence scope 起点",
                    ));
                }
            }
            ConstraintDecision::InvokeCapsule(request) => {
                request.validate(block.len())?;
                if !self.scope.intersects(request.scope) {
                    return Err(HelixError::new("capsule 与 evidence scope 不相交"));
                }
            }
        }
        Ok(())
    }

    fn weighted_score(&self) -> u32 {
        (u64::from(self.support.0) * u64::from(self.confidence.0) / u64::from(SCORE_SCALE)) as u32
    }
}

pub struct ConstraintEvidenceRequest<'a> {
    pub candidate: &'a GenerationCandidateBlock,
    pub round: u32,
    pub prior_capsule_digest: Option<Digest>,
}

/// 唯一约束证据入口。局部旧算子可适配成 Tool provider，但不得接管整个生成链。
pub trait ConstraintEvidenceProvider {
    fn descriptor(&self) -> EvidenceProviderDescriptor;

    fn provide_evidence(
        &mut self,
        request: &ConstraintEvidenceRequest<'_>,
    ) -> HelixResult<Vec<ConstraintEvidence>>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub provider: EvidenceProviderDescriptor,
    pub evidence: ConstraintEvidence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseDecision {
    pub position_offset: usize,
    pub decision: ConstraintDecision,
    pub aggregate_score: Score,
    pub margin: Score,
    pub evidence_ids: Vec<(String, u64)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintRound {
    pub abi_version: u32,
    pub transaction_id: u64,
    pub base: TransactionBase,
    pub round: u32,
    pub candidate_len: usize,
    pub accepted_prefix_len: usize,
    pub evidence: Vec<EvidenceRecord>,
    /// Accept 为默认已证明状态；这里只保存 repair/rollback/capsule。
    pub sparse_decisions: Vec<SparseDecision>,
}

impl ConstraintRound {
    pub fn decision_at(&self, position: usize) -> Option<&ConstraintDecision> {
        if position >= self.candidate_len {
            return None;
        }
        self.sparse_decisions
            .iter()
            .find(|item| item.position_offset == position)
            .map(|item| &item.decision)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintPolicy {
    pub minimum_providers: usize,
    pub minimum_score: Score,
    pub minimum_margin: Score,
    pub max_evidence_items: usize,
}

impl Default for ConstraintPolicy {
    fn default() -> Self {
        Self {
            minimum_providers: 1,
            minimum_score: Score(600_000),
            minimum_margin: Score(50_000),
            max_evidence_items: 256,
        }
    }
}

impl ConstraintPolicy {
    fn validate(&self) -> HelixResult<()> {
        if self.minimum_providers == 0 || self.max_evidence_items == 0 {
            return Err(HelixError::new("constraint policy 容量不能为 0"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapsuleResult {
    pub request_id: u64,
    pub capsule_id: String,
    pub output_digest: Digest,
    pub information_gain: Score,
    pub repair: Option<RepairDirective>,
}

pub trait CapsuleInvoker {
    fn invoke(
        &mut self,
        request: &CapsuleRequest,
        candidate: &GenerationCandidateBlock,
    ) -> HelixResult<CapsuleResult>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrefixTerminal {
    Accepted,
    RepairRequired,
    RolledBack,
    CapsuleRequired,
}

/// 字段与现有 whole-token 最长前缀决定同构；checkpoint/base 字段用于 host/device 同源校验。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarwavePrefixCommit {
    pub abi_version: u32,
    pub transaction_id: u64,
    pub base: TransactionBase,
    pub accepted_prefix: Vec<TokenId>,
    pub fallback_token_id: Option<TokenId>,
    pub rejected_draft_suffix: Vec<TokenId>,
    pub committed_token_ids: Vec<TokenId>,
    pub mismatch_index: Option<usize>,
    pub checkpoint_index: Option<usize>,
    pub selected_checkpoint: Option<CandidateCheckpoint>,
    pub terminal: PrefixTerminal,
}

impl StarwavePrefixCommit {
    pub fn validate_against(&self, block: &GenerationCandidateBlock) -> HelixResult<()> {
        block.validate()?;
        if self.abi_version != DUAL_HELIX_ABI_VERSION
            || self.transaction_id != block.transaction_id
            || self.base != block.base
        {
            return Err(HelixError::new("prefix commit 事务身份漂移"));
        }
        let draft = block.token_ids();
        let accepted_len = self.accepted_prefix.len();
        if accepted_len > draft.len()
            || self.accepted_prefix != draft[..accepted_len]
            || self.rejected_draft_suffix != draft[accepted_len..]
        {
            return Err(HelixError::new("prefix accepted/rejected 切分不闭合"));
        }
        let mut committed = self.accepted_prefix.clone();
        if let Some(fallback) = self.fallback_token_id {
            if fallback >= block.vocab_size || accepted_len >= draft.len() {
                return Err(HelixError::new("prefix fallback token 非法"));
            }
            committed.push(fallback);
        }
        if committed != self.committed_token_ids
            || self.mismatch_index != (accepted_len < draft.len()).then_some(accepted_len)
            || self.checkpoint_index != committed.len().checked_sub(1)
        {
            return Err(HelixError::new("prefix token ledger/index 不闭合"));
        }
        match (&self.selected_checkpoint, committed.len()) {
            (None, 0) => Ok(()),
            (Some(checkpoint), count) if count > 0 => {
                let position = self.base.position + count as u64;
                let epoch = self.base.commit_epoch + count as u64;
                if checkpoint.position != position || checkpoint.commit_epoch != epoch {
                    return Err(HelixError::new("prefix checkpoint position/epoch 漂移"));
                }
                Ok(())
            }
            _ => Err(HelixError::new(
                "prefix checkpoint 与 committed count 不一致",
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixCommitReceipt {
    pub transaction_id: u64,
    pub committed_position: u64,
    pub committed_epoch: u64,
    pub committed_token_count: usize,
    pub checkpoint_index: Option<usize>,
}

/// 星潮 owner 在实现中复用既有 host/device 原子发布与 rollback 逻辑。
pub trait StarwavePrefixCommitSink {
    fn authoritative_base(&self) -> TransactionBase;

    fn commit_longest_prefix(
        &mut self,
        commit: &StarwavePrefixCommit,
    ) -> HelixResult<PrefixCommitReceipt>;
}

pub struct DualHelix {
    policy: ConstraintPolicy,
}

impl DualHelix {
    pub fn new(policy: ConstraintPolicy) -> HelixResult<Self> {
        policy.validate()?;
        Ok(Self { policy })
    }

    pub fn generate<G: GenerationChain>(
        &self,
        chain: &mut G,
        request: &GenerationRequest,
    ) -> HelixResult<GenerationCandidateBlock> {
        request.validate()?;
        let chain_id = chain.chain_id().to_owned();
        if chain_id.trim().is_empty() {
            return Err(HelixError::new("generation chain ID 不能为空"));
        }
        let block = chain.generate_candidates(request)?;
        block.validate()?;
        if block.transaction_id != request.transaction_id
            || block.base != request.base
            || block.vocab_size != request.vocab_size
            || block.chain_id != chain_id
            || block.len() > request.requested_positions
        {
            return Err(HelixError::new("generation output 与 request/chain 不同源"));
        }
        Ok(block)
    }

    pub fn constrain(
        &self,
        block: &GenerationCandidateBlock,
        round: u32,
        prior_capsule_digest: Option<Digest>,
        providers: &mut [&mut dyn ConstraintEvidenceProvider],
    ) -> HelixResult<ConstraintRound> {
        block.validate()?;
        if providers.len() < self.policy.minimum_providers {
            return Err(HelixError::new("constraint evidence provider 数量不足"));
        }
        let request = ConstraintEvidenceRequest {
            candidate: block,
            round,
            prior_capsule_digest,
        };
        let mut records = Vec::new();
        let mut provider_ids = BTreeSet::new();
        for provider in providers {
            let descriptor = provider.descriptor();
            if descriptor.provider_id.trim().is_empty()
                || descriptor.revision.trim().is_empty()
                || !provider_ids.insert(descriptor.provider_id.clone())
            {
                return Err(HelixError::new("evidence provider 身份为空或重复"));
            }
            let mut evidence_ids = BTreeSet::new();
            for evidence in provider.provide_evidence(&request)? {
                evidence.validate(block)?;
                if !evidence_ids.insert(evidence.evidence_id)
                    || records.len() >= self.policy.max_evidence_items
                {
                    return Err(HelixError::new("evidence ID 重复或数量超限"));
                }
                records.push(EvidenceRecord {
                    provider: descriptor.clone(),
                    evidence,
                });
            }
        }

        let sparse_decisions = self.decide(block, &records)?;
        let accepted_prefix_len = sparse_decisions
            .iter()
            .map(|item| item.decision.safe_prefix(item.position_offset))
            .min()
            .unwrap_or(block.len());
        Ok(ConstraintRound {
            abi_version: DUAL_HELIX_ABI_VERSION,
            transaction_id: block.transaction_id,
            base: block.base.clone(),
            round,
            candidate_len: block.len(),
            accepted_prefix_len,
            evidence: records,
            sparse_decisions,
        })
    }

    pub fn repair<G: GenerationChain>(
        &self,
        chain: &mut G,
        original: &GenerationCandidateBlock,
        directive: &RepairDirective,
        capsule: Option<&CapsuleResult>,
    ) -> HelixResult<GenerationCandidateBlock> {
        original.validate()?;
        directive.validate(original)?;
        if chain.chain_id() != original.chain_id {
            return Err(HelixError::new("repair generation chain 不同源"));
        }
        let repaired = chain.repair_candidates(original, directive, capsule)?;
        repaired.validate()?;
        if repaired.base != original.base
            || repaired.vocab_size != original.vocab_size
            || repaired.chain_id != original.chain_id
        {
            return Err(HelixError::new("repair 改变了事务基线/词表/chain"));
        }
        Ok(repaired)
    }

    pub fn invoke_capsule<I: CapsuleInvoker>(
        &self,
        invoker: &mut I,
        request: &CapsuleRequest,
        block: &GenerationCandidateBlock,
    ) -> HelixResult<CapsuleResult> {
        block.validate()?;
        request.validate(block.len())?;
        let result = invoker.invoke(request, block)?;
        if result.request_id != request.request_id
            || result.capsule_id != request.capsule_id
            || result.information_gain == Score::ZERO
        {
            return Err(HelixError::new("capsule result 身份或信息增益非法"));
        }
        if let Some(repair) = &result.repair {
            repair.validate(block)?;
        }
        Ok(result)
    }

    pub fn prefix_commit(
        &self,
        block: &GenerationCandidateBlock,
        round: &ConstraintRound,
    ) -> HelixResult<StarwavePrefixCommit> {
        block.validate()?;
        if round.abi_version != DUAL_HELIX_ABI_VERSION
            || round.transaction_id != block.transaction_id
            || round.base != block.base
            || round.candidate_len != block.len()
            || round.accepted_prefix_len > block.len()
        {
            return Err(HelixError::new(
                "constraint round 与 generation block 不同源",
            ));
        }
        let tokens = block.token_ids();
        let accepted_len = round.accepted_prefix_len;
        let terminal = match round
            .sparse_decisions
            .iter()
            .min_by_key(|item| item.decision.safe_prefix(item.position_offset))
            .map(|item| &item.decision)
        {
            None | Some(ConstraintDecision::Accept) => PrefixTerminal::Accepted,
            Some(ConstraintDecision::Repair(_)) => PrefixTerminal::RepairRequired,
            Some(ConstraintDecision::Rollback { .. }) => PrefixTerminal::RolledBack,
            Some(ConstraintDecision::InvokeCapsule(_)) => PrefixTerminal::CapsuleRequired,
        };
        let commit = StarwavePrefixCommit {
            abi_version: DUAL_HELIX_ABI_VERSION,
            transaction_id: block.transaction_id,
            base: block.base.clone(),
            accepted_prefix: tokens[..accepted_len].to_vec(),
            fallback_token_id: None,
            rejected_draft_suffix: tokens[accepted_len..].to_vec(),
            committed_token_ids: tokens[..accepted_len].to_vec(),
            mismatch_index: (accepted_len < block.len()).then_some(accepted_len),
            checkpoint_index: accepted_len.checked_sub(1),
            selected_checkpoint: accepted_len
                .checked_sub(1)
                .map(|index| block.candidates[index].checkpoint.clone()),
            terminal,
        };
        commit.validate_against(block)?;
        Ok(commit)
    }

    pub fn publish<S: StarwavePrefixCommitSink>(
        &self,
        sink: &mut S,
        block: &GenerationCandidateBlock,
        commit: &StarwavePrefixCommit,
    ) -> HelixResult<PrefixCommitReceipt> {
        commit.validate_against(block)?;
        if sink.authoritative_base() != commit.base {
            return Err(HelixError::new("权威状态已变化，拒绝 stale prefix commit"));
        }
        let receipt = sink.commit_longest_prefix(commit)?;
        let count = commit.committed_token_ids.len();
        if receipt.transaction_id != commit.transaction_id
            || receipt.committed_position != commit.base.position + count as u64
            || receipt.committed_epoch != commit.base.commit_epoch + count as u64
            || receipt.committed_token_count != count
            || receipt.checkpoint_index != commit.checkpoint_index
        {
            return Err(HelixError::new("Starwave prefix commit receipt 漂移"));
        }
        Ok(receipt)
    }

    fn decide(
        &self,
        block: &GenerationCandidateBlock,
        records: &[EvidenceRecord],
    ) -> HelixResult<Vec<SparseDecision>> {
        let mut decisions = Vec::new();
        let mut position = 0usize;
        while position < block.len() {
            let relevant = records
                .iter()
                .filter(|record| {
                    record
                        .evidence
                        .decision
                        .applicable_at(record.evidence.scope, position)
                })
                .collect::<Vec<_>>();
            let providers = relevant
                .iter()
                .map(|record| record.provider.provider_id.as_str())
                .collect::<BTreeSet<_>>();
            if providers.len() < self.policy.minimum_providers {
                return Err(HelixError::new(format!(
                    "position {position} 缺少外部神经/工具证据"
                )));
            }

            let mut ranked = relevant;
            ranked.sort_by(|left, right| {
                right
                    .evidence
                    .weighted_score()
                    .cmp(&left.evidence.weighted_score())
                    .then_with(|| left.provider.provider_id.cmp(&right.provider.provider_id))
                    .then_with(|| left.evidence.evidence_id.cmp(&right.evidence.evidence_id))
            });
            let winner = ranked[0];
            let winner_score = winner.evidence.weighted_score();
            let runner_up = ranked
                .iter()
                .skip(1)
                .find(|record| record.evidence.decision != winner.evidence.decision)
                .map(|record| record.evidence.weighted_score())
                .unwrap_or(0);
            let margin = winner_score.saturating_sub(runner_up);
            if winner_score < self.policy.minimum_score.0 || margin < self.policy.minimum_margin.0 {
                return Err(HelixError::new(format!(
                    "position {position} evidence 未达到 score/margin 门"
                )));
            }

            let decision = winner.evidence.decision.clone();
            if decision == ConstraintDecision::Accept {
                position += 1;
                continue;
            }
            let evidence_ids = ranked
                .iter()
                .filter(|record| record.evidence.decision == decision)
                .map(|record| {
                    (
                        record.provider.provider_id.clone(),
                        record.evidence.evidence_id,
                    )
                })
                .collect();
            let next = decision.next_position(position, block.len());
            let rollback = matches!(decision, ConstraintDecision::Rollback { .. });
            decisions.push(SparseDecision {
                position_offset: position,
                decision,
                aggregate_score: Score(winner_score),
                margin: Score(margin),
                evidence_ids,
            });
            if rollback {
                break;
            }
            position = next;
        }
        Ok(decisions)
    }
}
