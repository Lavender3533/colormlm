//! 无第二模型的 StarWave committed-transition navigator。
//!
//! 当前请求优先使用已经 atomic-commit 的 token 输入链做最长后缀匹配；请求成功归还
//! resident root 时，完整 committed 输入链才会写入进程级有界 transition atlas。草稿、
//! 未提交 lane、失败事务和跨请求边界永远不会成为候选来源。候选仍由 FullDepth43 target
//! 验证；不命中时 terminal 提交真实 fallback，因此本模块不改变目标模型数学。

use crate::s14_starwave_draft::{
    S14StarwaveCandidateEvidenceSource, S14StarwaveCandidateLaneCertificate,
    S14StarwaveDraftResult, S14StarwaveLanePhysicalEvidence, S14StarwaveNavigatorContext,
    S14StarwaveNoFallbackTelemetry, S14StarwaveProductionNavigator,
    S14StarwaveProductionNavigatorOutput, S14_STARWAVE_DRAFT_PHYSICAL_K,
};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

const DEFAULT_MAX_SUFFIX_TOKENS: usize = 64;
pub const S14_STARWAVE_TRANSITION_ATLAS_CONTRACT_VERSION: u32 = 1;
pub const S14_STARWAVE_TRANSITION_ATLAS_CAPACITY: usize = 65_536;
pub const S14_STARWAVE_TRANSITION_ATLAS_CONTEXT_TOKENS: usize = 32;
const ATLAS_VARIANTS_PER_CONTEXT: usize = 4;
const ATLAS_PRUNE_BATCH: usize = 4_096;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14StarwaveTransitionAtlasStats {
    pub observed_sequences: u64,
    pub observed_tokens: u64,
    pub indexed_transitions: u64,
    pub queries: u64,
    pub hits: u64,
    pub multi_token_hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub resident_contexts: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct AtlasKey {
    len: u8,
    tokens: [u32; S14_STARWAVE_TRANSITION_ATLAS_CONTEXT_TOKENS],
}

impl AtlasKey {
    fn from_suffix(tokens: &[u32], suffix_len: usize) -> Option<Self> {
        if suffix_len == 0
            || suffix_len > S14_STARWAVE_TRANSITION_ATLAS_CONTEXT_TOKENS
            || suffix_len > tokens.len()
        {
            return None;
        }
        let mut key_tokens = [0u32; S14_STARWAVE_TRANSITION_ATLAS_CONTEXT_TOKENS];
        key_tokens[..suffix_len].copy_from_slice(&tokens[tokens.len() - suffix_len..]);
        Some(Self {
            len: suffix_len as u8,
            tokens: key_tokens,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AtlasContinuation {
    len: u8,
    tokens: [u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
    support: u32,
    last_touch: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HistoryCandidateProposal {
    tokens: Vec<u32>,
    source: Option<S14StarwaveCandidateEvidenceSource>,
}

impl AtlasContinuation {
    fn new(tokens: &[u32], last_touch: u64) -> Self {
        let mut fixed = [0u32; S14_STARWAVE_DRAFT_PHYSICAL_K];
        fixed[..tokens.len()].copy_from_slice(tokens);
        Self {
            len: tokens.len() as u8,
            tokens: fixed,
            support: 1,
            last_touch,
        }
    }

    fn matches(&self, tokens: &[u32]) -> bool {
        self.len as usize == tokens.len() && self.tokens[..tokens.len()] == *tokens
    }

    fn as_vec(self) -> Vec<u32> {
        self.tokens[..self.len as usize].to_vec()
    }
}

#[derive(Clone, Debug)]
struct AtlasEntry {
    variants: [Option<AtlasContinuation>; ATLAS_VARIANTS_PER_CONTEXT],
    last_touch: u64,
}

impl AtlasEntry {
    fn new() -> Self {
        Self {
            variants: [None; ATLAS_VARIANTS_PER_CONTEXT],
            last_touch: 0,
        }
    }

    fn observe(&mut self, tokens: &[u32], touch: u64) {
        self.last_touch = touch;
        if let Some(existing) = self
            .variants
            .iter_mut()
            .flatten()
            .find(|variant| variant.matches(tokens))
        {
            existing.support = existing.support.saturating_add(1);
            existing.last_touch = touch;
            return;
        }
        if let Some(empty) = self.variants.iter_mut().find(|variant| variant.is_none()) {
            *empty = Some(AtlasContinuation::new(tokens, touch));
            return;
        }
        let victim = self
            .variants
            .iter()
            .enumerate()
            .min_by_key(|(_, variant)| {
                let variant = variant.expect("full atlas variant set");
                (variant.support, variant.last_touch)
            })
            .map(|(index, _)| index)
            .expect("atlas variant set 不能为空");
        self.variants[victim] = Some(AtlasContinuation::new(tokens, touch));
    }

    fn best(&self, suffix_len: usize) -> Option<AtlasContinuation> {
        self.variants
            .iter()
            .flatten()
            // 极短上下文歧义高，至少需要两次独立 committed 观察；较长上下文允许
            // 单次精确复现。错误候选仍只损失效率，不会越过 target verifier。
            .filter(|variant| suffix_len >= 4 || variant.support >= 2)
            .copied()
            .max_by_key(|variant| (variant.support, variant.len, variant.last_touch))
    }
}

#[derive(Debug)]
struct S14StarwaveTransitionAtlas {
    entries: HashMap<AtlasKey, AtlasEntry>,
    clock: u64,
    stats: S14StarwaveTransitionAtlasStats,
}

impl S14StarwaveTransitionAtlas {
    fn new() -> Self {
        Self {
            entries: HashMap::with_capacity(S14_STARWAVE_TRANSITION_ATLAS_CAPACITY),
            clock: 0,
            stats: S14StarwaveTransitionAtlasStats::default(),
        }
    }

    fn next_touch(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    fn observe_committed_sequence(&mut self, tokens: &[u32]) {
        self.stats.observed_sequences = self.stats.observed_sequences.saturating_add(1);
        self.stats.observed_tokens = self
            .stats
            .observed_tokens
            .saturating_add(tokens.len() as u64);
        if tokens.len() < 2 {
            return;
        }
        for next_index in 1..tokens.len() {
            let continuation_end = tokens.len().min(next_index + S14_STARWAVE_DRAFT_PHYSICAL_K);
            let continuation = &tokens[next_index..continuation_end];
            for suffix_len in 1..=S14_STARWAVE_TRANSITION_ATLAS_CONTEXT_TOKENS.min(next_index) {
                let context = &tokens[..next_index];
                let key = AtlasKey::from_suffix(context, suffix_len)
                    .expect("bounded non-empty committed suffix");
                let touch = self.next_touch();
                self.entries
                    .entry(key)
                    .or_insert_with(AtlasEntry::new)
                    .observe(continuation, touch);
                self.stats.indexed_transitions = self.stats.indexed_transitions.saturating_add(1);
                if self.entries.len() > S14_STARWAVE_TRANSITION_ATLAS_CAPACITY + ATLAS_PRUNE_BATCH {
                    self.prune_to(S14_STARWAVE_TRANSITION_ATLAS_CAPACITY - ATLAS_PRUNE_BATCH);
                }
            }
        }
        self.prune_to(S14_STARWAVE_TRANSITION_ATLAS_CAPACITY);
        self.stats.resident_contexts = self.entries.len();
    }

    fn propose(&mut self, history: &[u32]) -> Option<HistoryCandidateProposal> {
        self.stats.queries = self.stats.queries.saturating_add(1);
        let max_suffix = S14_STARWAVE_TRANSITION_ATLAS_CONTEXT_TOKENS.min(history.len());
        for suffix_len in (1..=max_suffix).rev() {
            let key = AtlasKey::from_suffix(history, suffix_len)?;
            let Some(candidate) = self
                .entries
                .get(&key)
                .and_then(|entry| entry.best(suffix_len))
            else {
                continue;
            };
            if candidate.len < 2 {
                continue;
            }
            let touch = self.next_touch();
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.last_touch = touch;
            }
            self.stats.hits = self.stats.hits.saturating_add(1);
            self.stats.multi_token_hits = self.stats.multi_token_hits.saturating_add(1);
            return Some(HistoryCandidateProposal {
                tokens: candidate.as_vec(),
                source: Some(
                    S14StarwaveCandidateEvidenceSource::ProcessCommittedTransitionAtlas {
                        context_tokens: suffix_len as u8,
                        committed_support: candidate.support,
                    },
                ),
            });
        }
        self.stats.misses = self.stats.misses.saturating_add(1);
        None
    }

    fn prune_to(&mut self, target_entries: usize) {
        let excess = self.entries.len().saturating_sub(target_entries);
        if excess == 0 {
            return;
        }
        let mut victims = self
            .entries
            .iter()
            .map(|(key, entry)| (*key, entry.last_touch))
            .collect::<Vec<_>>();
        victims.sort_unstable_by_key(|(_, touch)| *touch);
        for (key, _) in victims.into_iter().take(excess) {
            self.entries.remove(&key);
        }
        self.stats.evictions = self.stats.evictions.saturating_add(excess as u64);
    }
}

static PROCESS_TRANSITION_ATLAS: OnceLock<Mutex<S14StarwaveTransitionAtlas>> = OnceLock::new();

fn with_process_transition_atlas<T>(
    operation: impl FnOnce(&mut S14StarwaveTransitionAtlas) -> T,
) -> T {
    let atlas =
        PROCESS_TRANSITION_ATLAS.get_or_init(|| Mutex::new(S14StarwaveTransitionAtlas::new()));
    let mut guard = atlas
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    operation(&mut guard)
}

pub fn process_starwave_transition_atlas_stats() -> S14StarwaveTransitionAtlasStats {
    with_process_transition_atlas(|atlas| {
        let mut stats = atlas.stats;
        stats.resident_contexts = atlas.entries.len();
        stats
    })
}

/// 进程级 atlas 的唯一写入口。调用方必须位于 resident root 已完整归还并重新验签之后；
/// 失败事务、草稿 lane 和仍可能回滚的 session 不得调用。
pub(crate) fn observe_process_starwave_committed_sequence(tokens: &[u32]) {
    with_process_transition_atlas(|atlas| atlas.observe_committed_sequence(tokens));
}

/// 基于当前请求已提交 token 链和进程级 committed transition atlas 的导航核心。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarwaveHistoryNavigator {
    eos_token_id: u32,
    max_suffix_tokens: usize,
    proposals: u64,
    multi_token_proposals: u64,
    lane_physical_evidence:
        Option<[S14StarwaveLanePhysicalEvidence; S14_STARWAVE_DRAFT_PHYSICAL_K]>,
}

impl S14StarwaveHistoryNavigator {
    pub const fn new(eos_token_id: u32) -> Self {
        Self {
            eos_token_id,
            max_suffix_tokens: DEFAULT_MAX_SUFFIX_TOKENS,
            proposals: 0,
            multi_token_proposals: 0,
            lane_physical_evidence: None,
        }
    }

    pub fn with_max_suffix_tokens(mut self, max_suffix_tokens: usize) -> Self {
        self.max_suffix_tokens = max_suffix_tokens.max(1);
        self
    }

    /// 注入由当前 block plan 的 verified leases 签出的逐 lane 成本。没有该证据时，
    /// history/atlas 仍可提供 lane0 猜测，但 production adapter 必须退化为 commit_limit=1。
    pub fn with_lane_physical_evidence(
        mut self,
        lane_physical_evidence: [S14StarwaveLanePhysicalEvidence; S14_STARWAVE_DRAFT_PHYSICAL_K],
    ) -> Self {
        self.lane_physical_evidence = Some(lane_physical_evidence);
        self
    }

    pub const fn proposals(&self) -> u64 {
        self.proposals
    }

    pub const fn multi_token_proposals(&self) -> u64 {
        self.multi_token_proposals
    }

    fn committed_input_chain(context: S14StarwaveNavigatorContext<'_>) -> Vec<u32> {
        let authoritative = context.authoritative();
        // TokenRecord.input_token_id 保存真实执行输入；ForcedPrefill 时不能使用模型预测
        // 代替被强制提交的下一 prompt token。最后追加当前待执行 input，得到连续输入链。
        let mut history = authoritative
            .committed_tokens
            .iter()
            .map(|record| record.input_token_id)
            .collect::<Vec<_>>();
        history.push(authoritative.input_token_id);
        history
    }

    fn local_multi_token_candidates(&self, history: &[u32]) -> Option<HistoryCandidateProposal> {
        let end = history.len();
        let max_suffix = self.max_suffix_tokens.min(end.saturating_sub(1));
        for suffix_len in (1..=max_suffix).rev() {
            let suffix_start = end - suffix_len;
            for match_start in (0..suffix_start).rev() {
                let candidate_start = match_start + suffix_len;
                if candidate_start >= end
                    || history[match_start..candidate_start] != history[suffix_start..end]
                {
                    continue;
                }
                let candidate_end = end.min(candidate_start + S14_STARWAVE_DRAFT_PHYSICAL_K);
                let candidates = history[candidate_start..candidate_end].to_vec();
                if candidates.len() >= 2 {
                    return Some(HistoryCandidateProposal {
                        tokens: candidates,
                        source: Some(
                            S14StarwaveCandidateEvidenceSource::InRequestCommittedHistory {
                                matched_suffix_tokens: u32::try_from(suffix_len)
                                    .unwrap_or(u32::MAX),
                            },
                        ),
                    });
                }
            }
        }
        None
    }

    fn candidates(&self, context: S14StarwaveNavigatorContext<'_>) -> HistoryCandidateProposal {
        let history = Self::committed_input_chain(context);
        if let Some(candidates) = self.local_multi_token_candidates(&history) {
            return candidates;
        }
        if let Some(candidates) = with_process_transition_atlas(|atlas| atlas.propose(&history)) {
            return candidates;
        }
        // 没有已提交转移时仍给 target 一个明确 lane0 候选；mismatch 时由真实 head
        // fallback，绝不把重复值或未提交草稿冒充多 token 证书。
        HistoryCandidateProposal {
            tokens: vec![context.authoritative().input_token_id],
            source: None,
        }
    }
}

impl S14StarwaveProductionNavigator for S14StarwaveHistoryNavigator {
    fn propose_real_candidates(
        &mut self,
        context: S14StarwaveNavigatorContext<'_>,
    ) -> S14StarwaveDraftResult<S14StarwaveProductionNavigatorOutput> {
        let candidates = self.candidates(context);
        self.proposals = self.proposals.saturating_add(1);
        if candidates.tokens.len() >= 2 {
            self.multi_token_proposals = self.multi_token_proposals.saturating_add(1);
        }
        let horizon = (candidates.tokens.len() >= 2).then_some(candidates.tokens.len());
        let mut certificates = Vec::new();
        if let (Some(physical), Some(source)) = (self.lane_physical_evidence, candidates.source) {
            for (lane, &token_id) in candidates.tokens.iter().enumerate() {
                certificates.push(S14StarwaveCandidateLaneCertificate::from_verified_evidence(
                    context,
                    lane,
                    token_id,
                    source,
                    physical[lane],
                )?);
            }
        }
        S14StarwaveProductionNavigatorOutput::from_certified_candidates(
            context,
            &candidates.tokens,
            Some(self.eos_token_id),
            horizon,
            &certificates,
            S14StarwaveNoFallbackTelemetry::default(),
        )
    }
}
