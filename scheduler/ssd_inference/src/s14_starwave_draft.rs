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
    s14_starfold_cache::{
        StarfoldVerifiedMappedLease, STARFOLD_VERIFIED_LEASE_CACHE_CONTRACT_VERSION,
    },
    s14_starfold_k4_rebind::S14StarfoldK4BlockMode,
    s14_starwave_transaction::{S14StarwaveProofWriter, S14StarwaveSha256},
    s14_whole_token_device::WholeTokenDeviceCommittedCheckpointBinding,
};
use polaris_s14_runner::{DecoderStateV1, MaterializedTokenSource, VOCAB_SIZE};
use std::{cmp::Ordering, error::Error, fmt, sync::Arc};

pub const S14_STARWAVE_DRAFT_PHYSICAL_K: usize = 4;
pub const S14_STARWAVE_DRAFT_EXPECTED_MIN_COMMITTABLE_TOKENS: usize = 1;
pub const S14_STARWAVE_BLOCK_PROPOSAL_CONTRACT_VERSION: u32 = 1;

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

/// 一个 proposal 所消费的 authoritative base checkpoint 窄身份。
///
/// 它不替换现有 `DecoderStateV1` ABI，也不复制 native state；只冻结多位置候选必须共同
/// 消费的 position/epoch/input/bank/arena/layout 身份。证书在 proposal 构造和 launch 前
/// 都会重新与同一 authoritative state 比较。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarwaveBaseCheckpointIdentity {
    position: u32,
    commit_epoch: u64,
    input_token_id: u32,
    active_fixed_bank: u8,
    arena_id: u64,
    state_bytes: u64,
    max_seq_len: u32,
}

impl S14StarwaveBaseCheckpointIdentity {
    fn from_authoritative(authoritative: &DecoderStateV1) -> S14StarwaveDraftResult<Self> {
        let state_bytes = u64::try_from(authoritative.native_arena.len()).map_err(|_| {
            S14StarwaveDraftError::new("S14 StarWave base checkpoint state bytes overflow")
        })?;
        Ok(Self {
            position: authoritative.position,
            commit_epoch: authoritative.commit_epoch,
            input_token_id: authoritative.input_token_id,
            active_fixed_bank: authoritative.active_fixed_bank,
            arena_id: authoritative.native_arena.arena_id(),
            state_bytes,
            max_seq_len: authoritative.native.max_seq_len,
        })
    }

    fn validate_for(self, authoritative: &DecoderStateV1) -> bool {
        u64::try_from(authoritative.native_arena.len()).is_ok_and(|state_bytes| {
            self.position == authoritative.position
                && self.commit_epoch == authoritative.commit_epoch
                && self.input_token_id == authoritative.input_token_id
                && self.active_fixed_bank == authoritative.active_fixed_bank
                && self.arena_id == authoritative.native_arena.arena_id()
                && self.state_bytes == state_bytes
                && self.max_seq_len == authoritative.native.max_seq_len
        })
    }

    pub const fn position(self) -> u32 {
        self.position
    }

    pub const fn commit_epoch(self) -> u64 {
        self.commit_epoch
    }

    pub const fn input_token_id(self) -> u32 {
        self.input_token_id
    }

    pub const fn active_fixed_bank(self) -> u8 {
        self.active_fixed_bank
    }

    pub const fn state_bytes(self) -> u64 {
        self.state_bytes
    }

    pub const fn arena_id(self) -> u64 {
        self.arena_id
    }

    pub const fn max_seq_len(self) -> u32 {
        self.max_seq_len
    }
}

/// 候选连续性的 committed-only 来源。这里没有草稿模型或未提交状态入口。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarwaveCandidateEvidenceSource {
    InRequestCommittedHistory {
        matched_suffix_tokens: u32,
    },
    ProcessCommittedTransitionAtlas {
        context_tokens: u8,
        committed_support: u32,
    },
}

impl S14StarwaveCandidateEvidenceSource {
    fn is_committed_evidence(self) -> bool {
        match self {
            Self::InRequestCommittedHistory {
                matched_suffix_tokens,
            } => matched_suffix_tokens != 0,
            Self::ProcessCommittedTransitionAtlas {
                context_tokens,
                committed_support,
            } => context_tokens != 0 && committed_support != 0,
        }
    }
}

/// verified lease 绑定的单 lane 预计物理成本。分母严格是
/// `transfer_bytes / bandwidth + FLOPs / throughput + fixed_latency`，统一换算为 ns。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarwaveLanePhysicalEvidence {
    lease_cache_contract_version: u32,
    verified_lease_count: u32,
    verified_physical_bytes: u64,
    expected_transfer_bytes: u64,
    expected_flops: u64,
    bandwidth_bytes_per_second: u64,
    throughput_flops_per_second: u64,
    fixed_latency_ns: u64,
    expected_entropy_drop_nanobits: u64,
    transfer_latency_ns: u64,
    compute_latency_ns: u64,
    total_latency_ns: u64,
    lease_identity_sha256: S14StarwaveSha256,
}

impl S14StarwaveLanePhysicalEvidence {
    /// `leases` 只能来自 verified lease cache 的 `acquire_planned`；普通 mmap 无法构造
    /// `StarfoldVerifiedMappedLease`。带宽、FLOPs 与固定延迟是调用方当前 block plan 的
    /// 保守预计值，全部进入证书和路由分母。
    pub fn from_verified_leases(
        leases: &[Arc<StarfoldVerifiedMappedLease>],
        expected_transfer_bytes: u64,
        expected_flops: u64,
        bandwidth_bytes_per_second: u64,
        throughput_flops_per_second: u64,
        fixed_latency_ns: u64,
        expected_entropy_drop_nanobits: u64,
    ) -> S14StarwaveDraftResult<Self> {
        if leases.is_empty() {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave 多 lane 物理成本缺少 verified lease",
            ));
        }
        if expected_entropy_drop_nanobits == 0 {
            return Err(S14StarwaveDraftError::new("S14 StarWave 信息增益必须大于0"));
        }
        let mut physical_bytes = 0u64;
        let mut writer = S14StarwaveProofWriter::new("polaris-s14-starwave-verified-leases-v1");
        writer.write_u32(STARFOLD_VERIFIED_LEASE_CACHE_CONTRACT_VERSION);
        writer.write_u64(leases.len() as u64);
        for lease in leases {
            let identity = lease.identity();
            physical_bytes = physical_bytes
                .checked_add(identity.payload_bytes)
                .ok_or_else(|| {
                    S14StarwaveDraftError::new("S14 StarWave verified lease bytes overflow")
                })?;
            writer.write_str(&identity.tensor);
            writer.write_str(&identity.cache_key);
            writer.write_str(&identity.range_key);
            writer.write_u64(identity.payload_bytes);
            writer.write_str(&identity.payload_sha256);
            writer.write_str(&identity.proof_sha256);
            writer.write_u32(identity.expert_id.map_or(u32::MAX, u32::from));
        }
        if expected_transfer_bytes > physical_bytes {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave expected transfer bytes 超过 verified physical bytes",
            ));
        }
        let transfer_latency_ns = starwave_scaled_ceil_ns(
            expected_transfer_bytes,
            bandwidth_bytes_per_second,
            "transfer bandwidth",
        )?;
        let compute_latency_ns = starwave_scaled_ceil_ns(
            expected_flops,
            throughput_flops_per_second,
            "compute throughput",
        )?;
        let total_latency_ns = transfer_latency_ns
            .checked_add(compute_latency_ns)
            .and_then(|value| value.checked_add(fixed_latency_ns))
            .ok_or_else(|| S14StarwaveDraftError::new("S14 StarWave route latency overflow"))?;
        if total_latency_ns == 0 {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave route score 分母不能为0",
            ));
        }
        Ok(Self {
            lease_cache_contract_version: STARFOLD_VERIFIED_LEASE_CACHE_CONTRACT_VERSION,
            verified_lease_count: u32::try_from(leases.len()).map_err(|_| {
                S14StarwaveDraftError::new("S14 StarWave verified lease count overflow")
            })?,
            verified_physical_bytes: physical_bytes,
            expected_transfer_bytes,
            expected_flops,
            bandwidth_bytes_per_second,
            throughput_flops_per_second,
            fixed_latency_ns,
            expected_entropy_drop_nanobits,
            transfer_latency_ns,
            compute_latency_ns,
            total_latency_ns,
            lease_identity_sha256: writer.finish(),
        })
    }

    fn validate(self) -> bool {
        self.lease_cache_contract_version == STARFOLD_VERIFIED_LEASE_CACHE_CONTRACT_VERSION
            && self.verified_lease_count != 0
            && self.verified_physical_bytes != 0
            && self.expected_transfer_bytes <= self.verified_physical_bytes
            && self.expected_entropy_drop_nanobits != 0
            && self.total_latency_ns != 0
            && self.lease_identity_sha256 != S14StarwaveSha256::ZERO
            && starwave_scaled_ceil_ns(
                self.expected_transfer_bytes,
                self.bandwidth_bytes_per_second,
                "transfer bandwidth",
            )
            .is_ok_and(|value| value == self.transfer_latency_ns)
            && starwave_scaled_ceil_ns(
                self.expected_flops,
                self.throughput_flops_per_second,
                "compute throughput",
            )
            .is_ok_and(|value| value == self.compute_latency_ns)
            && self
                .transfer_latency_ns
                .checked_add(self.compute_latency_ns)
                .and_then(|value| value.checked_add(self.fixed_latency_ns))
                == Some(self.total_latency_ns)
    }

    pub const fn expected_entropy_drop_nanobits(self) -> u64 {
        self.expected_entropy_drop_nanobits
    }

    pub const fn verified_physical_bytes(self) -> u64 {
        self.verified_physical_bytes
    }

    pub const fn expected_transfer_bytes(self) -> u64 {
        self.expected_transfer_bytes
    }

    pub const fn expected_flops(self) -> u64 {
        self.expected_flops
    }

    pub const fn total_latency_ns(self) -> u64 {
        self.total_latency_ns
    }

    pub const fn transfer_latency_ns(self) -> u64 {
        self.transfer_latency_ns
    }

    pub const fn compute_latency_ns(self) -> u64 {
        self.compute_latency_ns
    }

    pub const fn fixed_latency_ns(self) -> u64 {
        self.fixed_latency_ns
    }

    pub const fn lease_identity_sha256(self) -> S14StarwaveSha256 {
        self.lease_identity_sha256
    }
}

/// `预计不确定性降低 / (字节搬运时间 + FLOPs计算时间 + 固定延迟)` 的无浮点分数。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarwaveInformationGainScore {
    numerator_nanobits: u64,
    denominator_ns: u64,
}

impl S14StarwaveInformationGainScore {
    pub const fn numerator_nanobits(self) -> u64 {
        self.numerator_nanobits
    }

    pub const fn denominator_ns(self) -> u64 {
        self.denominator_ns
    }

    /// 无浮点比较 `ΔH / physical_time`，供 navigator 在多个 committed 候选之间选择。
    pub fn compare_efficiency(self, other: Self) -> Ordering {
        (u128::from(self.numerator_nanobits) * u128::from(other.denominator_ns))
            .cmp(&(u128::from(other.numerator_nanobits) * u128::from(self.denominator_ns)))
    }
}

/// 单个候选 lane 的 production 证书。所有字段私有，调用方只能把 committed 来源、
/// verified lease 成本与当前 authoritative context 一次性绑定。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarwaveCandidateLaneCertificate {
    contract_version: u32,
    lane: u8,
    candidate_position: u32,
    candidate_commit_epoch: u64,
    token_id: u32,
    base_checkpoint: S14StarwaveBaseCheckpointIdentity,
    source: S14StarwaveCandidateEvidenceSource,
    physical: S14StarwaveLanePhysicalEvidence,
    score: S14StarwaveInformationGainScore,
    certificate_sha256: S14StarwaveSha256,
}

impl S14StarwaveCandidateLaneCertificate {
    pub fn from_verified_evidence(
        context: S14StarwaveNavigatorContext<'_>,
        lane: usize,
        token_id: u32,
        source: S14StarwaveCandidateEvidenceSource,
        physical: S14StarwaveLanePhysicalEvidence,
    ) -> S14StarwaveDraftResult<Self> {
        if lane >= S14_STARWAVE_DRAFT_PHYSICAL_K || token_id >= VOCAB_SIZE {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave candidate certificate lane/token 越界",
            ));
        }
        if !source.is_committed_evidence() || !physical.validate() {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave candidate certificate 缺少 committed/physical evidence",
            ));
        }
        let authoritative = context.authoritative();
        let candidate_position =
            authoritative
                .position
                .checked_add(lane as u32)
                .ok_or_else(|| {
                    S14StarwaveDraftError::new("S14 StarWave candidate position overflow")
                })?;
        let candidate_commit_epoch = authoritative
            .commit_epoch
            .checked_add(lane as u64 + 1)
            .ok_or_else(|| S14StarwaveDraftError::new("S14 StarWave candidate epoch overflow"))?;
        let base_checkpoint = S14StarwaveBaseCheckpointIdentity::from_authoritative(authoritative)?;
        let score = S14StarwaveInformationGainScore {
            numerator_nanobits: physical.expected_entropy_drop_nanobits,
            denominator_ns: physical.total_latency_ns,
        };
        let mut certificate = Self {
            contract_version: S14_STARWAVE_BLOCK_PROPOSAL_CONTRACT_VERSION,
            lane: lane as u8,
            candidate_position,
            candidate_commit_epoch,
            token_id,
            base_checkpoint,
            source,
            physical,
            score,
            certificate_sha256: S14StarwaveSha256::ZERO,
        };
        certificate.certificate_sha256 = starwave_lane_certificate_sha256(&certificate);
        Ok(certificate)
    }

    fn validate_for(self, authoritative: &DecoderStateV1, lane: usize, token_id: u32) -> bool {
        self.contract_version == S14_STARWAVE_BLOCK_PROPOSAL_CONTRACT_VERSION
            && usize::from(self.lane) == lane
            && self.token_id == token_id
            && self.base_checkpoint.validate_for(authoritative)
            && authoritative
                .position
                .checked_add(lane as u32)
                .is_some_and(|value| value == self.candidate_position)
            && authoritative
                .commit_epoch
                .checked_add(lane as u64 + 1)
                .is_some_and(|value| value == self.candidate_commit_epoch)
            && self.source.is_committed_evidence()
            && self.physical.validate()
            && self.score.numerator_nanobits == self.physical.expected_entropy_drop_nanobits
            && self.score.denominator_ns == self.physical.total_latency_ns
            && self.certificate_sha256 != S14StarwaveSha256::ZERO
            && self.certificate_sha256 == starwave_lane_certificate_sha256(&self)
    }

    pub const fn lane(self) -> u8 {
        self.lane
    }

    pub const fn candidate_position(self) -> u32 {
        self.candidate_position
    }

    pub const fn candidate_commit_epoch(self) -> u64 {
        self.candidate_commit_epoch
    }

    pub const fn token_id(self) -> u32 {
        self.token_id
    }

    pub const fn base_checkpoint(self) -> S14StarwaveBaseCheckpointIdentity {
        self.base_checkpoint
    }

    pub const fn source(self) -> S14StarwaveCandidateEvidenceSource {
        self.source
    }

    pub const fn physical(self) -> S14StarwaveLanePhysicalEvidence {
        self.physical
    }

    pub const fn score(self) -> S14StarwaveInformationGainScore {
        self.score
    }

    pub const fn certificate_sha256(self) -> S14StarwaveSha256 {
        self.certificate_sha256
    }
}

fn starwave_scaled_ceil_ns(
    quantity: u64,
    rate_per_second: u64,
    field: &'static str,
) -> S14StarwaveDraftResult<u64> {
    if quantity == 0 {
        return Ok(0);
    }
    if rate_per_second == 0 {
        return Err(S14StarwaveDraftError::new(format!(
            "S14 StarWave {field} 缺失",
        )));
    }
    let numerator = u128::from(quantity)
        .checked_mul(1_000_000_000)
        .ok_or_else(|| S14StarwaveDraftError::new("S14 StarWave route scale overflow"))?;
    let denominator = u128::from(rate_per_second);
    let value = numerator
        .checked_add(denominator - 1)
        .map(|scaled| scaled / denominator)
        .ok_or_else(|| S14StarwaveDraftError::new("S14 StarWave route ceil overflow"))?;
    u64::try_from(value)
        .map_err(|_| S14StarwaveDraftError::new("S14 StarWave route latency 超出 u64"))
}

fn starwave_lane_certificate_sha256(
    certificate: &S14StarwaveCandidateLaneCertificate,
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starwave-lane-certificate-v1");
    writer.write_u32(certificate.contract_version);
    writer.write_u8(certificate.lane);
    writer.write_u32(certificate.candidate_position);
    writer.write_u64(certificate.candidate_commit_epoch);
    writer.write_u32(certificate.token_id);
    writer.write_u32(certificate.base_checkpoint.position);
    writer.write_u64(certificate.base_checkpoint.commit_epoch);
    writer.write_u32(certificate.base_checkpoint.input_token_id);
    writer.write_u8(certificate.base_checkpoint.active_fixed_bank);
    writer.write_u64(certificate.base_checkpoint.arena_id);
    writer.write_u64(certificate.base_checkpoint.state_bytes);
    writer.write_u32(certificate.base_checkpoint.max_seq_len);
    match certificate.source {
        S14StarwaveCandidateEvidenceSource::InRequestCommittedHistory {
            matched_suffix_tokens,
        } => {
            writer.write_u8(0);
            writer.write_u32(matched_suffix_tokens);
        }
        S14StarwaveCandidateEvidenceSource::ProcessCommittedTransitionAtlas {
            context_tokens,
            committed_support,
        } => {
            writer.write_u8(1);
            writer.write_u8(context_tokens);
            writer.write_u32(committed_support);
        }
    }
    writer.write_u32(certificate.physical.lease_cache_contract_version);
    writer.write_u32(certificate.physical.verified_lease_count);
    writer.write_u64(certificate.physical.verified_physical_bytes);
    writer.write_u64(certificate.physical.expected_transfer_bytes);
    writer.write_u64(certificate.physical.expected_flops);
    writer.write_u64(certificate.physical.bandwidth_bytes_per_second);
    writer.write_u64(certificate.physical.throughput_flops_per_second);
    writer.write_u64(certificate.physical.fixed_latency_ns);
    writer.write_u64(certificate.physical.expected_entropy_drop_nanobits);
    writer.write_u64(certificate.physical.transfer_latency_ns);
    writer.write_u64(certificate.physical.compute_latency_ns);
    writer.write_u64(certificate.physical.total_latency_ns);
    writer.write_sha256(certificate.physical.lease_identity_sha256);
    writer.write_u64(certificate.score.numerator_nanobits);
    writer.write_u64(certificate.score.denominator_ns);
    writer.finish()
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
    lane_certificates: [Option<S14StarwaveCandidateLaneCertificate>; S14_STARWAVE_DRAFT_PHYSICAL_K],
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
            lane_certificates: [None; S14_STARWAVE_DRAFT_PHYSICAL_K],
            no_fallback_telemetry,
        }
    }

    fn lane0_only_with_certificates(
        draft_token_ids: [u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
        lane_certificates: [Option<S14StarwaveCandidateLaneCertificate>;
            S14_STARWAVE_DRAFT_PHYSICAL_K],
        no_fallback_telemetry: S14StarwaveNoFallbackTelemetry,
    ) -> Self {
        Self {
            draft_token_ids,
            policy: S14StarwaveDraftPolicy::NavigatorLane0Only,
            commit_safety: S14StarwaveCommitSafety::Lane0Only,
            lane_certificates,
            no_fallback_telemetry,
        }
    }

    fn eos_aware(
        context: S14StarwaveNavigatorContext<'_>,
        draft_token_ids: [u32; S14_STARWAVE_DRAFT_PHYSICAL_K],
        eos_token_id: u32,
        navigator_horizon: usize,
        lane_certificates: [Option<S14StarwaveCandidateLaneCertificate>;
            S14_STARWAVE_DRAFT_PHYSICAL_K],
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
            lane_certificates,
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
            lane_certificates: [None; S14_STARWAVE_DRAFT_PHYSICAL_K],
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
    lane_certificates: [Option<S14StarwaveCandidateLaneCertificate>; S14_STARWAVE_DRAFT_PHYSICAL_K],
    no_fallback_telemetry: S14StarwaveNoFallbackTelemetry,
}

impl S14StarwaveProductionNavigatorOutput {
    /// 无物理成本证书的兼容入口。候选仍可供 target 执行 lane0，但即使调用方声明了
    /// horizon，也必须 fail-closed 为 lane0-only；2..=4 只能走
    /// [`Self::from_certified_candidates`]。
    pub fn from_real_candidates(
        context: S14StarwaveNavigatorContext<'_>,
        candidate_token_ids: &[u32],
        eos_token_id: Option<u32>,
        navigator_horizon: Option<usize>,
        no_fallback_telemetry: S14StarwaveNoFallbackTelemetry,
    ) -> S14StarwaveDraftResult<Self> {
        Self::from_candidate_contract(
            context,
            candidate_token_ids,
            eos_token_id,
            navigator_horizon,
            &[],
            no_fallback_telemetry,
        )
    }

    /// 带 per-lane committed/lease/cost 证书的 production 入口。证书必须从 lane0 连续排列；
    /// 某 lane 缺失或陈旧只截断 longest reliable prefix，后续 lane 不会令整个 block 回退。
    pub fn from_certified_candidates(
        context: S14StarwaveNavigatorContext<'_>,
        candidate_token_ids: &[u32],
        eos_token_id: Option<u32>,
        navigator_horizon: Option<usize>,
        lane_certificates: &[S14StarwaveCandidateLaneCertificate],
        no_fallback_telemetry: S14StarwaveNoFallbackTelemetry,
    ) -> S14StarwaveDraftResult<Self> {
        Self::from_candidate_contract(
            context,
            candidate_token_ids,
            eos_token_id,
            navigator_horizon,
            lane_certificates,
            no_fallback_telemetry,
        )
    }

    fn from_candidate_contract(
        context: S14StarwaveNavigatorContext<'_>,
        candidate_token_ids: &[u32],
        eos_token_id: Option<u32>,
        navigator_horizon: Option<usize>,
        lane_certificates: &[S14StarwaveCandidateLaneCertificate],
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
        if lane_certificates.len() > candidate_token_ids.len() {
            return Err(S14StarwaveDraftError::new(
                "S14 StarWave lane certificate 数量超过真实候选",
            ));
        }

        let candidate_count = candidate_token_ids.len();
        let mut draft_token_ids =
            [candidate_token_ids[candidate_count - 1]; S14_STARWAVE_DRAFT_PHYSICAL_K];
        draft_token_ids[..candidate_count].copy_from_slice(candidate_token_ids);

        let authoritative = context.authoritative();
        let mut certified = [None; S14_STARWAVE_DRAFT_PHYSICAL_K];
        let mut reliable_prefix = 0usize;
        for (lane, certificate) in lane_certificates.iter().copied().enumerate() {
            if !certificate.validate_for(authoritative, lane, candidate_token_ids[lane]) {
                break;
            }
            certified[lane] = Some(certificate);
            reliable_prefix = lane + 1;
        }

        // EOS/horizon/cost 证据共同界定 longest reliable prefix。horizon 越界或 EOS
        // 缺失仍退化 lane0；证书中途缺失则保留此前已连续证明的2..=4，而非整块回退。
        let eos_token_id = eos_token_id.filter(|&token_id| token_id < VOCAB_SIZE);
        let requested_horizon = navigator_horizon.filter(|&horizon| {
            (2..=S14_STARWAVE_DRAFT_PHYSICAL_K).contains(&horizon)
                && horizon <= candidate_count
                && eos_token_id.is_some()
        });
        let navigator_horizon = requested_horizon
            .map(|horizon| horizon.min(reliable_prefix))
            .filter(|&horizon| horizon >= 2);

        Ok(Self {
            authoritative_position: context.authoritative().position,
            authoritative_commit_epoch: context.authoritative().commit_epoch,
            draft_token_ids,
            candidate_count,
            eos_token_id,
            navigator_horizon,
            lane_certificates: certified,
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

    pub const fn lane_certificates(
        &self,
    ) -> &[Option<S14StarwaveCandidateLaneCertificate>; S14_STARWAVE_DRAFT_PHYSICAL_K] {
        &self.lane_certificates
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
                    self.lane_certificates,
                    self.no_fallback_telemetry,
                )
            }
            _ => Ok(S14StarwaveNavigatorDraft::lane0_only_with_certificates(
                self.draft_token_ids,
                self.lane_certificates,
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
    lane_certificates: [Option<S14StarwaveCandidateLaneCertificate>; S14_STARWAVE_DRAFT_PHYSICAL_K],
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

    /// 每个真实候选的 authoritative/checkpoint/物理成本证书；未证明的 padding lane 为
    /// `None`。只有从 lane0 开始的连续证书前缀可能进入 `effective_commit_limit`。
    pub const fn lane_certificates(
        &self,
    ) -> &[Option<S14StarwaveCandidateLaneCertificate>; S14_STARWAVE_DRAFT_PHYSICAL_K] {
        &self.lane_certificates
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
                            && self.lane_certificates[..cap.navigator_horizon()]
                                .iter()
                                .enumerate()
                                .all(|(lane, certificate)| {
                                    certificate.is_some_and(|certificate| {
                                        certificate.validate_for(
                                            authoritative,
                                            lane,
                                            self.draft_token_ids[lane],
                                        )
                                    })
                                })
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
        lane_certificates: navigation.lane_certificates,
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
