//! S14 StarWave 在没有常驻草稿模型时使用的因果安全 K4 draft policy。
//!
//! FullDepth43 的四个物理输入固定为 `[authoritative.input_token_id, draft0,
//! draft1, draft2]`，terminal 比较的草稿固定为 `[draft0, draft1, draft2, draft3]`。
//! lane0 只依赖 authoritative checkpoint，因此即使 `draft0` 在 mismatch0 被拒绝，GPU
//! head 仍会给出一个真实 fallback token，terminal 可提交 lane0 checkpoint，并把其中的
//! `input_token_id` 改成该 fallback。后续 filler 只服务物理 K4，不属于这个 checkpoint。
//!
//! 默认 deterministic policy 不把 filler 冒充成模型预测，并把 `effective_commit_limit`
//! 固定为 1。这样即使 filler 偶然匹配，也不会在一个 K4 内越过未观测的 EOS 隐藏提交。
//! 只有实现 [`S14StarwaveProductionNavigator`] 的真实导航器经
//! [`S14StarwaveProductionNavigatorAdapter`] 绑定候选/EOS/horizon 后，proposal 才能
//! 允许提交更长的匹配前缀。

use crate::{
    s14_starfold_k4_rebind::S14StarfoldK4BlockMode,
    s14_whole_token_device::WholeTokenDeviceCommittedCheckpointBinding,
};
use polaris_s14_runner::{DecoderStateV1, MaterializedTokenSource, VOCAB_SIZE};
use std::{error::Error, fmt};

pub const S14_STARWAVE_DRAFT_PHYSICAL_K: usize = 4;
pub const S14_STARWAVE_DRAFT_EXPECTED_MIN_COMMITTABLE_TOKENS: usize = 1;

pub type S14StarwaveDraftResult<T> = Result<T, S14StarwaveDraftError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarwaveDraftError {
    message: String,
}

impl S14StarwaveDraftError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for S14StarwaveDraftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for S14StarwaveDraftError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarwaveDraftPolicy {
    DeterministicRepeatAuthoritativeInput,
    DeterministicExplicitFiller,
    NavigatorLane0Only,
    EosAwareNavigator,
}

/// 必须保持全零的 no-fallback 审计计数。导航器若报告任何旧 runtime、第二模型或
/// CPU/Transformer fallback 活动，proposal 构造会失败。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14StarwaveNoFallbackTelemetry {
    pub secondary_model_loads: u64,
    pub legacy_runtime_starts: u64,
    pub cpu_fallback_forward_calls: u64,
    pub transformer_fallback_forward_calls: u64,
}

impl S14StarwaveNoFallbackTelemetry {
    pub const fn is_no_fallback(self) -> bool {
        self.secondary_model_loads == 0
            && self.legacy_runtime_starts == 0
            && self.cpu_fallback_forward_calls == 0
            && self.transformer_fallback_forward_calls == 0
    }
}

/// 导航器对“从 lane0 起最多可提交多少 token 且不会跨过 EOS”的显式证明声明。
///
/// 证书同时冻结 authoritative position/epoch、真实 navigator 给出的四个 draft token、
/// 其已验证预测 horizon、EOS token 身份和首个 EOS lane。`effective_limit` 永远是
/// `min(horizon, first_eos_lane + 1)`；
/// 因此 EOS 本身可以提交，但 EOS 之后的隐藏 lane 永远不会进入 terminal commit limit。
/// production 只能经 [`S14StarwaveProductionNavigatorAdapter`] 构造与 draft 绑定的声明。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarwaveEosAwarePrefixCap {
    authoritative_position: u32,
    authoritative_commit_epoch: u64,
    draft_token_ids: [u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
    eos_token_id: u32,
    navigator_horizon: usize,
    first_eos_lane: Option<usize>,
    effective_limit: usize,
}

impl S14StarwaveEosAwarePrefixCap {
    fn for_draft(
        context: S14StarwaveNavigatorContext<'_>,
        draft_token_ids: [u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
        eos_token_id: u32,
        navigator_horizon: usize,
    ) -> S14StarwaveDraftResult<Self> {
        if eos_token_id >= VOCAB_SIZE {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave EOS token 越出冻结 vocab",
            ));
        }
        if !(2..=S14_STARWAVE_DRAFT_PHYSICAL_K).contains(&navigator_horizon) {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave 真实 navigator horizon 必须为 2..=4",
            ));
        }
        if draft_token_ids
            .iter()
            .any(|&token_id| token_id >= VOCAB_SIZE)
        {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave EOS-aware draft token 越出冻结 vocab",
            ));
        }
        let first_eos_lane = draft_token_ids
            .iter()
            .take(navigator_horizon)
            .position(|&token_id| token_id == eos_token_id);
        let effective_limit = first_eos_lane
            .and_then(|lane| lane.checked_add(1))
            .map_or(navigator_horizon, |eos_prefix| {
                eos_prefix.min(navigator_horizon)
            });
        if !(1..=S14_STARWAVE_DRAFT_PHYSICAL_K).contains(&effective_limit) {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave EOS-aware effective limit 越界",
            ));
        }
        Ok(Self {
            authoritative_position: context.authoritative().position,
            authoritative_commit_epoch: context.authoritative().commit_epoch,
            draft_token_ids,
            eos_token_id,
            navigator_horizon,
            first_eos_lane,
            effective_limit,
        })
    }

    pub const fn limit(self) -> usize {
        self.effective_limit
    }

    pub const fn navigator_horizon(self) -> usize {
        self.navigator_horizon
    }

    pub const fn eos_token_id(self) -> u32 {
        self.eos_token_id
    }

    pub const fn first_eos_lane(self) -> Option<usize> {
        self.first_eos_lane
    }

    fn validate_for(
        self,
        authoritative: &DecoderStateV1,
        draft_token_ids: &[u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
    ) -> bool {
        if self.authoritative_position != authoritative.position
            || self.authoritative_commit_epoch != authoritative.commit_epoch
            || self.draft_token_ids != *draft_token_ids
            || self.eos_token_id >= VOCAB_SIZE
            || !(2..=S14_STARWAVE_DRAFT_PHYSICAL_K).contains(&self.navigator_horizon)
        {
            return false;
        }
        let first_eos_lane = draft_token_ids
            .iter()
            .take(self.navigator_horizon)
            .position(|&token_id| token_id == self.eos_token_id);
        let effective_limit = first_eos_lane
            .and_then(|lane| lane.checked_add(1))
            .map_or(self.navigator_horizon, |eos_prefix| {
                eos_prefix.min(self.navigator_horizon)
            });
        self.first_eos_lane == first_eos_lane && self.effective_limit == effective_limit
    }
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarwaveNavigatorContext<'a> {
    authoritative: &'a DecoderStateV1,
    position0_committed_origin: Option<S14StarwavePosition0CommittedOrigin>,
}

impl<'a> S14StarwaveNavigatorContext<'a> {
    pub const fn authoritative(self) -> &'a DecoderStateV1 {
        self.authoritative
    }

    pub const fn physical_k(self) -> usize {
        S14_STARWAVE_DRAFT_PHYSICAL_K
    }

    pub const fn position0_committed_origin(self) -> Option<S14StarwavePosition0CommittedOrigin> {
        self.position0_committed_origin
    }
}

/// 单-token prompt 从 position0 直接进入 generation 的 host/device 同源证明。
/// 只能由真实 committed device binding 与同一份 authoritative state 联合签发。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarwavePosition0CommittedOrigin {
    commit_epoch: u64,
    input_token_id: u32,
    active_bank: usize,
    state_bytes: u64,
}

impl S14StarwavePosition0CommittedOrigin {
    pub fn from_committed_checkpoint(
        authoritative: &DecoderStateV1,
        committed: &WholeTokenDeviceCommittedCheckpointBinding<'_>,
    ) -> S14StarwaveDraftResult<Self> {
        authoritative.validate().map_err(|error| {
            S14StarwaveDraftError::new(format!(
                "S14 StarWave position0 origin authoritative 非法: {error}",
            ))
        })?;
        let state_bytes = u64::try_from(authoritative.native_arena.len()).map_err(|_| {
            S14StarwaveDraftError::new("S14 StarWave position0 origin state bytes overflow")
        })?;
        if authoritative.position != 0
            || authoritative.commit_epoch != 0
            || authoritative.active_fixed_bank != 0
            || !authoritative.committed_tokens.is_empty()
            || committed.epoch() != authoritative.commit_epoch
            || committed.active_bank() != usize::from(authoritative.active_fixed_bank)
            || committed.state_bytes() != state_bytes
        {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave position0 origin 未闭合真实 host/device committed identity",
            ));
        }
        Ok(Self {
            commit_epoch: authoritative.commit_epoch,
            input_token_id: authoritative.input_token_id,
            active_bank: usize::from(authoritative.active_fixed_bank),
            state_bytes,
        })
    }

    fn validate_for(self, authoritative: &DecoderStateV1) -> S14StarwaveDraftResult<()> {
        let state_bytes = u64::try_from(authoritative.native_arena.len()).map_err(|_| {
            S14StarwaveDraftError::new("S14 StarWave position0 origin state bytes overflow")
        })?;
        if authoritative.position != 0
            || authoritative.commit_epoch != self.commit_epoch
            || authoritative.input_token_id != self.input_token_id
            || usize::from(authoritative.active_fixed_bank) != self.active_bank
            || state_bytes != self.state_bytes
            || !authoritative.committed_tokens.is_empty()
        {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave position0 committed-origin 证据已陈旧",
            ));
        }
        Ok(())
    }
}

/// SpeculativeGeneration 进入 adapter/session 前的唯一 committed-origin 判定。
/// base0 必须携带与 authoritative 匹配的强证据；后续 position 维持普通 committed-state
/// 路径，并拒绝复用只属于 base0 的证据。
pub fn validate_s14_starwave_generation_origin(
    authoritative: &DecoderStateV1,
    position0_committed_origin: Option<S14StarwavePosition0CommittedOrigin>,
) -> S14StarwaveDraftResult<()> {
    authoritative.validate().map_err(|error| {
        S14StarwaveDraftError::new(format!(
            "S14 StarWave generation origin authoritative 非法: {error}",
        ))
    })?;
    match (authoritative.position, position0_committed_origin) {
        (0, Some(origin)) => origin.validate_for(authoritative),
        (0, None) => Err(S14StarwaveDraftError::new(
            "S14 StarWave position0 generation 缺少真实 host/device committed-origin 证据",
        )),
        (_, Some(_)) => Err(S14StarwaveDraftError::new(
            "S14 StarWave position0 committed-origin 证据不得复用于后续 position",
        )),
        (_, None) => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum S14StarwaveCommitSafety {
    Lane0Only,
    EosAware(S14StarwaveEosAwarePrefixCap),
}

/// 可替换导航器的窄输出。它只给出 target 将验证的四个草稿 token 与提交上限证明；
/// 不拥有或启动任何模型/runtime。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarwaveNavigatorDraft {
    draft_token_ids: [u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
    policy: S14StarwaveDraftPolicy,
    commit_safety: S14StarwaveCommitSafety,
    no_fallback_telemetry: S14StarwaveNoFallbackTelemetry,
}

impl S14StarwaveNavigatorDraft {
    pub fn lane0_only(
        draft_token_ids: [u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
        no_fallback_telemetry: S14StarwaveNoFallbackTelemetry,
    ) -> Self {
        Self {
            draft_token_ids,
            policy: S14StarwaveDraftPolicy::NavigatorLane0Only,
            commit_safety: S14StarwaveCommitSafety::Lane0Only,
            no_fallback_telemetry,
        }
    }

    fn eos_aware(
        context: S14StarwaveNavigatorContext<'_>,
        draft_token_ids: [u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
        eos_token_id: u32,
        navigator_horizon: usize,
        no_fallback_telemetry: S14StarwaveNoFallbackTelemetry,
    ) -> S14StarwaveDraftResult<Self> {
        let prefix_cap = S14StarwaveEosAwarePrefixCap::for_draft(
            context,
            draft_token_ids,
            eos_token_id,
            navigator_horizon,
        )?;
        Ok(Self {
            draft_token_ids,
            policy: S14StarwaveDraftPolicy::EosAwareNavigator,
            commit_safety: S14StarwaveCommitSafety::EosAware(prefix_cap),
            no_fallback_telemetry,
        })
    }

    fn deterministic(
        filler_token_id: u32,
        policy: S14StarwaveDraftPolicy,
    ) -> S14StarwaveDraftResult<Self> {
        if !matches!(
            policy,
            S14StarwaveDraftPolicy::DeterministicRepeatAuthoritativeInput
                | S14StarwaveDraftPolicy::DeterministicExplicitFiller
        ) {
            return Err(S14StarwaveDraftError::new(
                "deterministic draft policy 类型非法",
            ));
        }
        Ok(Self {
            draft_token_ids: [filler_token_id; S14_STARWAVE_DRAFT_PHYSICAL_K],
            policy,
            commit_safety: S14StarwaveCommitSafety::Lane0Only,
            no_fallback_telemetry: S14StarwaveNoFallbackTelemetry::default(),
        })
    }
}

/// 真实 production navigator 的一次候选输出。`candidate_count` 只统计导航器实际产生的
/// token；物理 K4 多出的 lane 仅以最后一个真实候选做 execution padding，并永远不会
/// 超过证书 horizon 提交。字段私有，防止调用方事后把 padding 冒充真实候选。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarwaveProductionNavigatorOutput {
    authoritative_position: u32,
    authoritative_commit_epoch: u64,
    draft_token_ids: [u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
    candidate_count: usize,
    eos_token_id: Option<u32>,
    navigator_horizon: Option<usize>,
    no_fallback_telemetry: S14StarwaveNoFallbackTelemetry,
}

impl S14StarwaveProductionNavigatorOutput {
    /// `candidate_token_ids` 必须是本次 navigator 真正产生的连续候选，长度为1..=4。
    /// `navigator_horizon` 只有在 navigator 能证明前 N 个候选均属于同一因果轨迹时才传
    /// `Some(N)`；缺少 EOS、horizon 越界或 horizon 超过真实候选数时不报假成功，而是
    /// 在转换 proposal 时 fail-closed 为 lane0-only。
    pub fn from_real_candidates(
        context: S14StarwaveNavigatorContext<'_>,
        candidate_token_ids: &[u32],
        eos_token_id: Option<u32>,
        navigator_horizon: Option<usize>,
        no_fallback_telemetry: S14StarwaveNoFallbackTelemetry,
    ) -> S14StarwaveDraftResult<Self> {
        if candidate_token_ids.is_empty()
            || candidate_token_ids.len() > S14_STARWAVE_DRAFT_PHYSICAL_K
        {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave production navigator 必须返回1..=4个真实候选 token",
            ));
        }
        if candidate_token_ids
            .iter()
            .any(|&token_id| token_id >= VOCAB_SIZE)
        {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave production navigator 候选 token 越出冻结 vocab",
            ));
        }
        if !no_fallback_telemetry.is_no_fallback() {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave production navigator 触发禁止的模型/runtime fallback",
            ));
        }

        let candidate_count = candidate_token_ids.len();
        let mut draft_token_ids =
            [candidate_token_ids[candidate_count - 1]; S14_STARWAVE_DRAFT_PHYSICAL_K];
        draft_token_ids[..candidate_count].copy_from_slice(candidate_token_ids);

        // 错误或缺失的 EOS/horizon 只能降级为 lane0-only。这里不把调用方声明的
        // horizon 截断成一个看似有效的值，避免把“无法证明”伪装成较短证明。
        let eos_token_id = eos_token_id.filter(|&token_id| token_id < VOCAB_SIZE);
        let navigator_horizon = navigator_horizon.filter(|&horizon| {
            (2..=S14_STARWAVE_DRAFT_PHYSICAL_K).contains(&horizon)
                && horizon <= candidate_count
                && eos_token_id.is_some()
        });

        Ok(Self {
            authoritative_position: context.authoritative().position,
            authoritative_commit_epoch: context.authoritative().commit_epoch,
            draft_token_ids,
            candidate_count,
            eos_token_id,
            navigator_horizon,
            no_fallback_telemetry,
        })
    }

    pub const fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    pub const fn certified_horizon(&self) -> Option<usize> {
        self.navigator_horizon
    }

    pub const fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn into_navigator_draft(
        self,
        context: S14StarwaveNavigatorContext<'_>,
    ) -> S14StarwaveDraftResult<S14StarwaveNavigatorDraft> {
        if self.authoritative_position != context.authoritative().position
            || self.authoritative_commit_epoch != context.authoritative().commit_epoch
            || !(1..=S14_STARWAVE_DRAFT_PHYSICAL_K).contains(&self.candidate_count)
            || !self.no_fallback_telemetry.is_no_fallback()
        {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave production navigator output authoritative/candidate identity 已陈旧",
            ));
        }

        match (self.eos_token_id, self.navigator_horizon) {
            (Some(eos_token_id), Some(horizon))
                if (2..=self.candidate_count).contains(&horizon) =>
            {
                S14StarwaveNavigatorDraft::eos_aware(
                    context,
                    self.draft_token_ids,
                    eos_token_id,
                    horizon,
                    self.no_fallback_telemetry,
                )
            }
            _ => Ok(S14StarwaveNavigatorDraft::lane0_only(
                self.draft_token_ids,
                self.no_fallback_telemetry,
            )),
        }
    }
}

/// proposal builder 的通用窄接口；trait 本身不要求也不提供第二草稿模型。production
/// 的2--4提交只能通过 [`S14StarwaveProductionNavigatorAdapter`] 进入，普通实现者只能
/// 构造 lane0-only 输出，不能绕过 adapter 私自签发 EOS-aware 证书。
pub trait S14StarwaveNavigator {
    fn propose(
        &mut self,
        context: S14StarwaveNavigatorContext<'_>,
    ) -> S14StarwaveDraftResult<S14StarwaveNavigatorDraft>;
}

/// production navigator 的最窄接口。实现者只负责产生真实候选及其 horizon/EOS 证据；
/// adapter 统一负责 authoritative 绑定、物理 K4 padding 与 fail-closed commit limit。
pub trait S14StarwaveProductionNavigator {
    fn propose_real_candidates(
        &mut self,
        context: S14StarwaveNavigatorContext<'_>,
    ) -> S14StarwaveDraftResult<S14StarwaveProductionNavigatorOutput>;
}

/// 把真实 production navigator 接到现有 proposal builder，不允许 session 直接拼装
/// `S14StarwaveNavigatorDraft`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarwaveProductionNavigatorAdapter<N> {
    navigator: N,
}

impl<N> S14StarwaveProductionNavigatorAdapter<N> {
    pub const fn new(navigator: N) -> Self {
        Self { navigator }
    }

    pub fn navigator(&self) -> &N {
        &self.navigator
    }

    pub fn navigator_mut(&mut self) -> &mut N {
        &mut self.navigator
    }

    pub fn into_inner(self) -> N {
        self.navigator
    }
}

impl<N: S14StarwaveProductionNavigator> S14StarwaveNavigator
    for S14StarwaveProductionNavigatorAdapter<N>
{
    fn propose(
        &mut self,
        context: S14StarwaveNavigatorContext<'_>,
    ) -> S14StarwaveDraftResult<S14StarwaveNavigatorDraft> {
        self.navigator
            .propose_real_candidates(context)?
            .into_navigator_draft(context)
    }
}

/// 默认无模型实现。`None` 重复 authoritative input；`Some(token)` 使用调用方已验证语义
/// 的显式 filler。二者都只允许 terminal 提交 lane0。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14StarwaveDeterministicNavigator {
    explicit_filler_token_id: Option<u32>,
}

impl S14StarwaveDeterministicNavigator {
    pub const fn new(explicit_filler_token_id: Option<u32>) -> Self {
        Self {
            explicit_filler_token_id,
        }
    }
}

impl S14StarwaveNavigator for S14StarwaveDeterministicNavigator {
    fn propose(
        &mut self,
        context: S14StarwaveNavigatorContext<'_>,
    ) -> S14StarwaveDraftResult<S14StarwaveNavigatorDraft> {
        let authoritative_input = context.authoritative().input_token_id;
        let (filler_token_id, policy) = match self.explicit_filler_token_id {
            Some(token_id) => (
                token_id,
                S14StarwaveDraftPolicy::DeterministicExplicitFiller,
            ),
            None => (
                authoritative_input,
                S14StarwaveDraftPolicy::DeterministicRepeatAuthoritativeInput,
            ),
        };
        if filler_token_id >= VOCAB_SIZE {
            return Err(S14StarwaveDraftError::new(format!(
                "S14 StarWave filler token {filler_token_id} 越出 vocab {VOCAB_SIZE}",
            )));
        }
        S14StarwaveNavigatorDraft::deterministic(filler_token_id, policy)
    }
}

/// 已绑定 authoritative identity 的 K4 proposal。字段保持私有，避免调用方把 source
/// 改成 `ForcedPrefill`、放宽 deterministic commit limit 或打乱物理/terminal token 映射。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarwaveDraftProposal {
    authoritative_position: u32,
    authoritative_commit_epoch: u64,
    authoritative_max_seq_len: u32,
    position0_committed_origin: Option<S14StarwavePosition0CommittedOrigin>,
    input_token_ids: [u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
    draft_token_ids: [u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
    policy: S14StarwaveDraftPolicy,
    mode: S14StarfoldK4BlockMode,
    source: MaterializedTokenSource,
    physical_k: usize,
    expected_min_committable_tokens: usize,
    effective_commit_limit: usize,
    eos_aware_prefix_cap: Option<S14StarwaveEosAwarePrefixCap>,
    no_fallback_telemetry: S14StarwaveNoFallbackTelemetry,
}

impl S14StarwaveDraftProposal {
    pub const fn authoritative_position(&self) -> u32 {
        self.authoritative_position
    }

    pub const fn authoritative_commit_epoch(&self) -> u64 {
        self.authoritative_commit_epoch
    }

    pub const fn authoritative_max_seq_len(&self) -> u32 {
        self.authoritative_max_seq_len
    }

    pub const fn position0_committed_origin(&self) -> Option<S14StarwavePosition0CommittedOrigin> {
        self.position0_committed_origin
    }

    pub const fn input_token_ids(&self) -> &[u32; S14_STARWAVE_DRAFT_PHYSICAL_K] {
        &self.input_token_ids
    }

    pub const fn draft_token_ids(&self) -> &[u32; S14_STARWAVE_DRAFT_PHYSICAL_K] {
        &self.draft_token_ids
    }

    pub const fn policy(&self) -> S14StarwaveDraftPolicy {
        self.policy
    }

    pub const fn mode(&self) -> S14StarfoldK4BlockMode {
        self.mode
    }

    pub const fn source(&self) -> MaterializedTokenSource {
        self.source
    }

    pub const fn physical_k(&self) -> usize {
        self.physical_k
    }

    pub const fn expected_min_committable_tokens(&self) -> usize {
        self.expected_min_committable_tokens
    }

    /// production 必须把此值传给 terminal 的 `execute_sealed_k4_with_commit_limit`，
    /// 不能改走默认的无 cap K4 提交入口。
    pub const fn effective_commit_limit(&self) -> usize {
        self.effective_commit_limit
    }

    pub const fn eos_aware_prefix_cap(&self) -> Option<usize> {
        match self.eos_aware_prefix_cap {
            Some(cap) => Some(cap.limit()),
            None => None,
        }
    }

    pub const fn eos_aware_navigator_horizon(&self) -> Option<usize> {
        match self.eos_aware_prefix_cap {
            Some(cap) => Some(cap.navigator_horizon()),
            None => None,
        }
    }

    pub const fn eos_aware_first_eos_lane(&self) -> Option<usize> {
        match self.eos_aware_prefix_cap {
            Some(cap) => cap.first_eos_lane(),
            None => None,
        }
    }

    pub const fn eos_token_id(&self) -> Option<u32> {
        match self.eos_aware_prefix_cap {
            Some(cap) => Some(cap.eos_token_id()),
            None => None,
        }
    }

    pub const fn no_fallback_telemetry(&self) -> S14StarwaveNoFallbackTelemetry {
        self.no_fallback_telemetry
    }

    /// 在 launch 前重新绑定当前 authoritative state，拒绝 stale proposal。
    pub fn validate_for(&self, authoritative: &DecoderStateV1) -> S14StarwaveDraftResult<()> {
        validate_authoritative_k4(authoritative, self.position0_committed_origin)?;
        if self.authoritative_position != authoritative.position
            || self.authoritative_commit_epoch != authoritative.commit_epoch
            || self.authoritative_max_seq_len != authoritative.native.max_seq_len
        {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave draft proposal authoritative identity 已陈旧",
            ));
        }
        if self.mode != S14StarfoldK4BlockMode::SpeculativeGeneration
            || self.source != MaterializedTokenSource::SpeculativeDraft
            || self.mode.materialized_source() != self.source
        {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave draft 必须是 SpeculativeGeneration/SpeculativeDraft，禁止 ForcedPrefill",
            ));
        }
        if self.physical_k != S14_STARWAVE_DRAFT_PHYSICAL_K
            || self.expected_min_committable_tokens
                != S14_STARWAVE_DRAFT_EXPECTED_MIN_COMMITTABLE_TOKENS
            || self.input_token_ids[0] != authoritative.input_token_id
            || self.input_token_ids[1..] != self.draft_token_ids[..3]
            || self
                .input_token_ids
                .iter()
                .chain(self.draft_token_ids.iter())
                .any(|&token_id| token_id >= VOCAB_SIZE)
            || !self.no_fallback_telemetry.is_no_fallback()
        {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave draft K4/token/no-fallback 合同漂移",
            ));
        }

        let commit_contract_valid = match self.policy {
            S14StarwaveDraftPolicy::DeterministicRepeatAuthoritativeInput
            | S14StarwaveDraftPolicy::DeterministicExplicitFiller
            | S14StarwaveDraftPolicy::NavigatorLane0Only => {
                self.effective_commit_limit == 1 && self.eos_aware_prefix_cap.is_none()
            }
            S14StarwaveDraftPolicy::EosAwareNavigator => {
                (1..=S14_STARWAVE_DRAFT_PHYSICAL_K).contains(&self.effective_commit_limit)
                    && self.eos_aware_prefix_cap.is_some_and(|cap| {
                        cap.limit() == self.effective_commit_limit
                            && cap.validate_for(authoritative, &self.draft_token_ids)
                    })
            }
        };
        if !commit_contract_valid {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave draft EOS-aware/effective commit limit 合同漂移",
            ));
        }
        Ok(())
    }
}

/// 默认 production proposer：不加载第二模型，不启动旧 runtime，filler 不宣称预测质量。
pub fn propose_s14_starwave_draft(
    authoritative: &DecoderStateV1,
    position0_committed_origin: Option<S14StarwavePosition0CommittedOrigin>,
    explicit_filler_token_id: Option<u32>,
) -> S14StarwaveDraftResult<S14StarwaveDraftProposal> {
    let mut navigator = S14StarwaveDeterministicNavigator::new(explicit_filler_token_id);
    propose_s14_starwave_draft_with_navigator(
        authoritative,
        position0_committed_origin,
        &mut navigator,
    )
}

pub fn propose_s14_starwave_draft_with_navigator<N: S14StarwaveNavigator + ?Sized>(
    authoritative: &DecoderStateV1,
    position0_committed_origin: Option<S14StarwavePosition0CommittedOrigin>,
    navigator: &mut N,
) -> S14StarwaveDraftResult<S14StarwaveDraftProposal> {
    validate_authoritative_k4(authoritative, position0_committed_origin)?;
    let navigation = navigator.propose(S14StarwaveNavigatorContext {
        authoritative,
        position0_committed_origin,
    })?;
    if navigation
        .draft_token_ids
        .iter()
        .any(|&token_id| token_id >= VOCAB_SIZE)
    {
        return Err(S14StarwaveDraftError::new(
            "S14 StarWave navigator draft token 越出 vocab",
        ));
    }
    if !navigation.no_fallback_telemetry.is_no_fallback() {
        return Err(S14StarwaveDraftError::new(
            "S14 StarWave navigator 触发了禁止的模型/runtime fallback",
        ));
    }

    let (effective_commit_limit, eos_aware_prefix_cap) = match navigation.commit_safety {
        S14StarwaveCommitSafety::Lane0Only => (1, None),
        S14StarwaveCommitSafety::EosAware(prefix_cap) => {
            if !prefix_cap.validate_for(authoritative, &navigation.draft_token_ids) {
                return Err(S14StarwaveDraftError::new(
                    "S14 StarWave navigator EOS-aware 证书与 draft token 不同源",
                ));
            }
            (prefix_cap.limit(), Some(prefix_cap))
        }
    };
    let draft_token_ids = navigation.draft_token_ids;
    let proposal = S14StarwaveDraftProposal {
        authoritative_position: authoritative.position,
        authoritative_commit_epoch: authoritative.commit_epoch,
        authoritative_max_seq_len: authoritative.native.max_seq_len,
        position0_committed_origin,
        input_token_ids: [
            authoritative.input_token_id,
            draft_token_ids[0],
            draft_token_ids[1],
            draft_token_ids[2],
        ],
        draft_token_ids,
        policy: navigation.policy,
        mode: S14StarfoldK4BlockMode::SpeculativeGeneration,
        source: MaterializedTokenSource::SpeculativeDraft,
        physical_k: S14_STARWAVE_DRAFT_PHYSICAL_K,
        expected_min_committable_tokens: S14_STARWAVE_DRAFT_EXPECTED_MIN_COMMITTABLE_TOKENS,
        effective_commit_limit,
        eos_aware_prefix_cap,
        no_fallback_telemetry: navigation.no_fallback_telemetry,
    };
    proposal.validate_for(authoritative)?;
    Ok(proposal)
}

fn validate_authoritative_k4(
    authoritative: &DecoderStateV1,
    position0_committed_origin: Option<S14StarwavePosition0CommittedOrigin>,
) -> S14StarwaveDraftResult<()> {
    validate_s14_starwave_generation_origin(authoritative, position0_committed_origin)?;
    let physical_end = authoritative
        .position
        .checked_add(S14_STARWAVE_DRAFT_PHYSICAL_K as u32)
        .ok_or_else(|| S14StarwaveDraftError::new("S14 StarWave K4 position overflow"))?;
    if physical_end > authoritative.native.max_seq_len {
        return Err(S14StarwaveDraftError::new(format!(
            "S14 StarWave K4 position 越界: base={} end={} max_seq_len={}",
            authoritative.position, physical_end, authoritative.native.max_seq_len,
        )));
    }
    Ok(())
}
