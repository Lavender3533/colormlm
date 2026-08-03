//! 无第二模型的 StarWave history navigator。
//!
//! 它只在已经 atomic-commit 的 token 输入链中查找“最长后缀的最近一次历史出现”，
//! 将随后 1..=4 个 token 作为候选。候选仍由 FullDepth43 target 验证；不命中时
//! terminal 提交真实 fallback，因此本模块不会改变目标模型数学或绕过 checkpoint。

use crate::s14_starwave_draft::{
    S14StarwaveDraftResult, S14StarwaveNavigatorContext, S14StarwaveNoFallbackTelemetry,
    S14StarwaveProductionNavigator, S14StarwaveProductionNavigatorOutput,
    S14_STARWAVE_DRAFT_PHYSICAL_K,
};

const DEFAULT_MAX_SUFFIX_TOKENS: usize = 64;

/// 基于当前请求已提交 token 链的轻量导航核心；不拥有模型、Vulkan 或外部状态。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarwaveHistoryNavigator {
    eos_token_id: u32,
    max_suffix_tokens: usize,
    proposals: u64,
    multi_token_proposals: u64,
}

impl S14StarwaveHistoryNavigator {
    pub const fn new(eos_token_id: u32) -> Self {
        Self {
            eos_token_id,
            max_suffix_tokens: DEFAULT_MAX_SUFFIX_TOKENS,
            proposals: 0,
            multi_token_proposals: 0,
        }
    }

    pub fn with_max_suffix_tokens(mut self, max_suffix_tokens: usize) -> Self {
        self.max_suffix_tokens = max_suffix_tokens.max(1);
        self
    }

    pub const fn proposals(&self) -> u64 {
        self.proposals
    }

    pub const fn multi_token_proposals(&self) -> u64 {
        self.multi_token_proposals
    }

    fn candidates(&self, context: S14StarwaveNavigatorContext<'_>) -> Vec<u32> {
        let authoritative = context.authoritative();
        // TokenRecord.input_token_id 保存真实执行输入；ForcedPrefill 时不能使用模型预测
        // 代替被强制提交的下一 prompt token。最后追加当前待执行 input，得到连续输入链。
        let mut history = authoritative
            .committed_tokens
            .iter()
            .map(|record| record.input_token_id)
            .collect::<Vec<_>>();
        history.push(authoritative.input_token_id);

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
                if !candidates.is_empty() {
                    return candidates;
                }
            }
        }

        // 没有历史转移时仍给 target 一个明确的 lane0 候选；它只允许提交1 token，
        // mismatch 时由真实 head fallback，绝不把重复值冒充多 token 证书。
        vec![authoritative.input_token_id]
    }
}

impl S14StarwaveProductionNavigator for S14StarwaveHistoryNavigator {
    fn propose_real_candidates(
        &mut self,
        context: S14StarwaveNavigatorContext<'_>,
    ) -> S14StarwaveDraftResult<S14StarwaveProductionNavigatorOutput> {
        let candidates = self.candidates(context);
        self.proposals = self.proposals.saturating_add(1);
        if candidates.len() >= 2 {
            self.multi_token_proposals = self.multi_token_proposals.saturating_add(1);
        }
        let horizon = (candidates.len() >= 2).then_some(candidates.len());
        S14StarwaveProductionNavigatorOutput::from_real_candidates(
            context,
            &candidates,
            Some(self.eos_token_id),
            horizon,
            S14StarwaveNoFallbackTelemetry::default(),
        )
    }
}
