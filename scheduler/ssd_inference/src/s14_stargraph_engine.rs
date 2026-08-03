//! 北极星 S14 最小真实编排器。
//!
//! 本模块只负责把既有信息增益路由、双螺旋约束与 Starwave 事务按单向流程连接起来：
//! 候选只由注入的生成链生产一次，适配器随后把同一候选块封装为 Starwave batch；
//! 所有提交状态仍由 [`S14StarwaveTransaction`] 独占维护。

#![forbid(unsafe_code)]

use std::{error::Error, fmt};

use crate::{
    s14_dual_helix::{
        CandidateCheckpoint, ConstraintDecision, ConstraintEvidence, ConstraintEvidenceProvider,
        ConstraintEvidenceRequest, ConstraintRound, Digest, DualHelix, EvidenceProviderDescriptor,
        GenerationCandidateBlock, GenerationChain, GenerationRequest, HelixError, HelixResult,
        PhysicalBudget, TransactionBase, SCORE_SCALE,
    },
    s14_information_gain_router::{
        S14CandidateExecution, S14InformationGainBudget, S14InformationGainCandidate,
        S14InformationGainError, S14InformationGainRouter, S14InformationGainRoutingReceipt,
    },
    s14_starwave_transaction::{
        S14StarwaveBranch, S14StarwaveCandidateBatch, S14StarwaveCandidateProducer,
        S14StarwaveCandidateRequest, S14StarwaveCandidateStep, S14StarwaveCapabilityDelta,
        S14StarwaveCapabilityVerdict, S14StarwaveCorrectionBatch, S14StarwaveError,
        S14StarwavePreparedTransaction, S14StarwaveProduceError, S14StarwaveProducedCandidates,
        S14StarwaveProof, S14StarwaveProofWriter, S14StarwaveRollbackReceipt,
        S14StarwaveRolledBackStep, S14StarwaveSha256, S14StarwaveStateProof,
        S14StarwaveTransaction, S14StarwaveTransactionOutcome, S14_STARWAVE_MAX_POSITIONS,
        S14_STARWAVE_MIN_POSITIONS,
    },
};

pub const S14_STARGRAPH_ENGINE_ABI_VERSION: u32 = 1;

pub type S14StarGraphResult<T> = Result<T, S14StarGraphError>;

#[derive(Debug)]
pub enum S14StarGraphError {
    MissingCandidateProducer,
    MissingScorer,
    MissingPhysicalCommitter,
    InvalidRequest(&'static str),
    NoCapsuleCandidates,
    NoSelectedCapsules,
    Generation(HelixError),
    Scoring(HelixError),
    Constraint(HelixError),
    Adapter(HelixError),
    InformationGain(S14InformationGainError),
    Starwave(S14StarwaveError),
}

impl fmt::Display for S14StarGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCandidateProducer => {
                formatter.write_str("StarGraph 缺少候选 producer，按 fail-closed 拒绝执行")
            }
            Self::MissingScorer => formatter
                .write_str("StarGraph 缺少 capsule/constraint scorer，按 fail-closed 拒绝执行"),
            Self::MissingPhysicalCommitter => {
                formatter.write_str("StarGraph 缺少外部物理 committer，按 fail-closed 拒绝执行")
            }
            Self::InvalidRequest(message) => write!(formatter, "StarGraph request 非法: {message}"),
            Self::NoCapsuleCandidates => {
                formatter.write_str("StarGraph scorer 未产生任何信息增益胶囊候选")
            }
            Self::NoSelectedCapsules => formatter.write_str("StarGraph 信息增益路由未选择任何胶囊"),
            Self::Generation(error) => write!(formatter, "StarGraph 候选生产失败: {error}"),
            Self::Scoring(error) => write!(formatter, "StarGraph 胶囊评分失败: {error}"),
            Self::Constraint(error) => write!(formatter, "StarGraph 双螺旋约束失败: {error}"),
            Self::Adapter(error) => write!(formatter, "StarGraph ABI 适配失败: {error}"),
            Self::InformationGain(error) => error.fmt(formatter),
            Self::Starwave(error) => error.fmt(formatter),
        }
    }
}

impl Error for S14StarGraphError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Generation(error)
            | Self::Scoring(error)
            | Self::Constraint(error)
            | Self::Adapter(error) => Some(error),
            Self::InformationGain(error) => Some(error),
            Self::Starwave(error) => Some(error),
            Self::MissingCandidateProducer
            | Self::MissingScorer
            | Self::MissingPhysicalCommitter
            | Self::InvalidRequest(_)
            | Self::NoCapsuleCandidates
            | Self::NoSelectedCapsules => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarGraphRequest {
    pub transaction_id: u64,
    pub vocab_size: u32,
    pub positions: u8,
    pub constraint_round: u32,
    pub semantic_plan_digest: Digest,
    pub generation_budget: PhysicalBudget,
    pub information_gain_budget: S14InformationGainBudget,
}

impl S14StarGraphRequest {
    fn validate(&self) -> S14StarGraphResult<()> {
        if self.vocab_size == 0 {
            return Err(S14StarGraphError::InvalidRequest("vocab_size 必须大于 0"));
        }
        if !(S14_STARWAVE_MIN_POSITIONS..=S14_STARWAVE_MAX_POSITIONS).contains(&self.positions) {
            return Err(S14StarGraphError::InvalidRequest(
                "positions 必须位于 Starwave 的 8..=32 合同内",
            ));
        }
        if usize::from(self.positions) > self.generation_budget.max_positions {
            return Err(S14StarGraphError::InvalidRequest(
                "positions 超出 generation physical budget",
            ));
        }
        Ok(())
    }
}

/// 同一个注入 scorer 同时产生胶囊评分和约束证据，确保约束阶段能看到最终路由收据。
pub trait S14StarGraphScorer {
    fn descriptor(&self) -> EvidenceProviderDescriptor;

    fn score_capsules(
        &mut self,
        request: &GenerationRequest,
        candidate: &GenerationCandidateBlock,
    ) -> HelixResult<Vec<S14InformationGainCandidate>>;

    fn score_constraints(
        &mut self,
        request: &ConstraintEvidenceRequest<'_>,
        routing: &S14InformationGainRoutingReceipt,
    ) -> HelixResult<Vec<ConstraintEvidence>>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum S14StarGraphPhysicalDecision {
    Commit(S14StarwaveProof),
    Abort { reason: String },
}

pub struct S14StarGraphPhysicalCommitRequest<'a> {
    pub routing: &'a S14InformationGainRoutingReceipt,
    pub constraint: &'a ConstraintRound,
    pub prepared: &'a S14StarwavePreparedTransaction<S14StarGraphCandidateState>,
}

/// 外部实现负责真实物理提交并返回 proof；拒绝或失败时由编排器走既有 prepared abort。
pub trait S14StarGraphPhysicalCommitter {
    fn commit_prepared(
        &mut self,
        request: &S14StarGraphPhysicalCommitRequest<'_>,
    ) -> HelixResult<S14StarGraphPhysicalDecision>;

    fn abort_prepared(
        &mut self,
        request: &S14StarGraphPhysicalCommitRequest<'_>,
        reason: &str,
    ) -> HelixResult<()>;
}

pub struct S14StarGraphDependencies<'a> {
    pub producer: Option<&'a mut dyn GenerationChain>,
    pub scorer: Option<&'a mut dyn S14StarGraphScorer>,
    pub physical_committer: Option<&'a mut dyn S14StarGraphPhysicalCommitter>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarGraphCandidateState {
    pub transaction_id: u64,
    pub position_offset: usize,
    pub checkpoint: CandidateCheckpoint,
    pub capsule_route_digest: Digest,
}

impl S14StarwaveStateProof for S14StarGraphCandidateState {
    fn write_state_proof(&self, writer: &mut S14StarwaveProofWriter) {
        writer.write_u32(S14_STARGRAPH_ENGINE_ABI_VERSION);
        writer.write_u64(self.transaction_id);
        writer.write_u64(self.position_offset as u64);
        writer.write_u64(self.checkpoint.checkpoint_id);
        writer.write_u64(self.checkpoint.position);
        writer.write_u64(self.checkpoint.commit_epoch);
        writer.write_bytes(&self.checkpoint.state_digest);
        writer.write_bytes(&self.capsule_route_digest);
    }
}

#[derive(Debug)]
pub enum S14StarGraphOutcome {
    Committed {
        routing: S14InformationGainRoutingReceipt,
        constraint: ConstraintRound,
        transaction: S14StarwaveTransactionOutcome<S14StarGraphCandidateState>,
    },
    Aborted {
        routing: S14InformationGainRoutingReceipt,
        constraint: ConstraintRound,
        rollback_receipt: S14StarwaveRollbackReceipt,
        rolled_back: Vec<S14StarwaveRolledBackStep<S14StarGraphCandidateState>>,
        reason: String,
    },
}

pub struct S14StarGraphEngine {
    dual_helix: DualHelix,
    starwave: S14StarwaveTransaction,
}

impl S14StarGraphEngine {
    pub const fn new(dual_helix: DualHelix, starwave: S14StarwaveTransaction) -> Self {
        Self {
            dual_helix,
            starwave,
        }
    }

    pub const fn starwave(&self) -> &S14StarwaveTransaction {
        &self.starwave
    }

    pub fn execute(
        &mut self,
        request: &S14StarGraphRequest,
        dependencies: S14StarGraphDependencies<'_>,
    ) -> S14StarGraphResult<S14StarGraphOutcome> {
        request.validate()?;
        let producer = dependencies
            .producer
            .ok_or(S14StarGraphError::MissingCandidateProducer)?;
        let scorer = dependencies
            .scorer
            .ok_or(S14StarGraphError::MissingScorer)?;
        let physical_committer = dependencies
            .physical_committer
            .ok_or(S14StarGraphError::MissingPhysicalCommitter)?;

        let generation_request = self.generation_request(request);
        let candidate = {
            let mut adapter = GenerationChainAdapter { inner: producer };
            self.dual_helix
                .generate(&mut adapter, &generation_request)
                .map_err(S14StarGraphError::Generation)?
        };
        if candidate.len() != usize::from(request.positions) {
            return Err(S14StarGraphError::Adapter(HelixError::new(
                "生成链必须为 Starwave adapter 产出请求长度的完整候选块",
            )));
        }

        let capsule_candidates = scorer
            .score_capsules(&generation_request, &candidate)
            .map_err(S14StarGraphError::Scoring)?;
        if capsule_candidates.is_empty() {
            return Err(S14StarGraphError::NoCapsuleCandidates);
        }
        let routing =
            S14InformationGainRouter::route(&capsule_candidates, request.information_gain_budget)
                .map_err(S14StarGraphError::InformationGain)?;
        if routing.selected.is_empty() {
            return Err(S14StarGraphError::NoSelectedCapsules);
        }
        debug_assert!(!routing.uses_whole_model_fallback());
        let capsule_route_digest = routing_digest(&routing);

        let constraint = {
            let mut adapter = RoutedScorerAdapter {
                inner: scorer,
                routing: &routing,
            };
            let mut providers: [&mut dyn ConstraintEvidenceProvider; 1] = [&mut adapter];
            self.dual_helix
                .constrain(
                    &candidate,
                    request.constraint_round,
                    Some(capsule_route_digest),
                    &mut providers,
                )
                .map_err(S14StarGraphError::Constraint)?
        };

        let batch = {
            let mut adapter = StarwaveCandidateAdapter {
                candidate: &candidate,
                capsule_route_digest,
            };
            self.starwave
                .produce_batch(request.positions, &mut adapter)
                .map_err(map_produce_error)?
        };
        let corrections = adapt_constraint_round(&batch, &constraint, &routing)?;
        let prepared = self
            .starwave
            .prepare(batch, corrections)
            .map_err(S14StarGraphError::Starwave)?;

        if !prepared.will_commit() {
            return Ok(abort_prepared(
                physical_committer,
                routing,
                constraint,
                prepared,
                "双螺旋与 Starwave 可靠性门未批准任何物理提交前缀".to_owned(),
            ));
        }

        let physical_decision = {
            let commit_request = S14StarGraphPhysicalCommitRequest {
                routing: &routing,
                constraint: &constraint,
                prepared: &prepared,
            };
            physical_committer.commit_prepared(&commit_request)
        };
        let proof = match physical_decision {
            Ok(S14StarGraphPhysicalDecision::Commit(proof)) => proof,
            Ok(S14StarGraphPhysicalDecision::Abort { reason }) => {
                return Ok(abort_prepared(
                    physical_committer,
                    routing,
                    constraint,
                    prepared,
                    reason,
                ));
            }
            Err(error) => {
                return Ok(abort_prepared(
                    physical_committer,
                    routing,
                    constraint,
                    prepared,
                    format!("外部物理提交失败: {error}"),
                ));
            }
        };
        if let Err(error) = proof.validate() {
            return Ok(abort_prepared(
                physical_committer,
                routing,
                constraint,
                prepared,
                format!("外部物理 commit proof 非法: {error}"),
            ));
        }

        let transaction = self
            .starwave
            .finalize(prepared, Some(proof))
            .map_err(S14StarGraphError::Starwave)?;
        Ok(S14StarGraphOutcome::Committed {
            routing,
            constraint,
            transaction,
        })
    }

    fn generation_request(&self, request: &S14StarGraphRequest) -> GenerationRequest {
        GenerationRequest {
            transaction_id: request.transaction_id,
            base: TransactionBase {
                position: self.starwave.next_position(),
                commit_epoch: self.starwave.commit_epoch(),
                state_digest: *self.starwave.last_commit_sha256().as_bytes(),
            },
            vocab_size: request.vocab_size,
            requested_positions: usize::from(request.positions),
            semantic_plan_digest: request.semantic_plan_digest,
            budget: request.generation_budget.clone(),
        }
    }
}

struct GenerationChainAdapter<'a> {
    inner: &'a mut dyn GenerationChain,
}

impl GenerationChain for GenerationChainAdapter<'_> {
    fn chain_id(&self) -> &str {
        self.inner.chain_id()
    }

    fn generate_candidates(
        &mut self,
        request: &GenerationRequest,
    ) -> HelixResult<GenerationCandidateBlock> {
        self.inner.generate_candidates(request)
    }

    fn repair_candidates(
        &mut self,
        original: &GenerationCandidateBlock,
        repair: &crate::s14_dual_helix::RepairDirective,
        capsule: Option<&crate::s14_dual_helix::CapsuleResult>,
    ) -> HelixResult<GenerationCandidateBlock> {
        self.inner.repair_candidates(original, repair, capsule)
    }
}

struct RoutedScorerAdapter<'a> {
    inner: &'a mut dyn S14StarGraphScorer,
    routing: &'a S14InformationGainRoutingReceipt,
}

impl ConstraintEvidenceProvider for RoutedScorerAdapter<'_> {
    fn descriptor(&self) -> EvidenceProviderDescriptor {
        self.inner.descriptor()
    }

    fn provide_evidence(
        &mut self,
        request: &ConstraintEvidenceRequest<'_>,
    ) -> HelixResult<Vec<ConstraintEvidence>> {
        self.inner.score_constraints(request, self.routing)
    }
}

struct StarwaveCandidateAdapter<'a> {
    candidate: &'a GenerationCandidateBlock,
    capsule_route_digest: Digest,
}

impl S14StarwaveCandidateProducer for StarwaveCandidateAdapter<'_> {
    type State = S14StarGraphCandidateState;
    type Error = HelixError;

    fn produce_candidates(
        &mut self,
        request: &S14StarwaveCandidateRequest,
    ) -> HelixResult<S14StarwaveProducedCandidates<Self::State>> {
        if self.candidate.len() != usize::from(request.positions)
            || self.candidate.base.position != request.base_position
            || self.candidate.base.commit_epoch != request.base_commit_epoch
            || self.candidate.base.state_digest != *request.anchor_sha256.as_bytes()
        {
            return Err(HelixError::new(
                "DualHelix candidate block 与 Starwave batch anchor 不同源",
            ));
        }

        let mut steps = Vec::with_capacity(self.candidate.len());
        for (offset, candidate) in self.candidate.candidates.iter().enumerate() {
            let position = request
                .base_position
                .checked_add(offset as u64)
                .ok_or_else(|| HelixError::new("StarGraph candidate position 溢出"))?;
            let state = S14StarGraphCandidateState {
                transaction_id: self.candidate.transaction_id,
                position_offset: offset,
                checkpoint: candidate.checkpoint.clone(),
                capsule_route_digest: self.capsule_route_digest,
            };
            steps.push(
                S14StarwaveCandidateStep::new(
                    position,
                    candidate.token_id,
                    candidate.confidence.get() as f32 / SCORE_SCALE as f32,
                    state,
                )
                .map_err(|error| HelixError::new(error.to_string()))?,
            );
        }
        let branch = S14StarwaveBranch::new(self.candidate.transaction_id, steps)
            .map_err(|error| HelixError::new(error.to_string()))?;
        let producer_proof = S14StarwaveProof::new(
            "polaris-s14-stargraph-generation-adapter-v1",
            generation_adapter_digest(self.candidate, request, self.capsule_route_digest)
                .as_bytes()
                .to_vec(),
        )
        .map_err(|error| HelixError::new(error.to_string()))?;
        Ok(S14StarwaveProducedCandidates::new(
            vec![branch],
            producer_proof,
        ))
    }
}

fn map_produce_error(error: S14StarwaveProduceError<HelixError>) -> S14StarGraphError {
    match error {
        S14StarwaveProduceError::Producer(error) => S14StarGraphError::Adapter(error),
        S14StarwaveProduceError::Core(error) => S14StarGraphError::Starwave(error),
    }
}

fn adapt_constraint_round(
    batch: &S14StarwaveCandidateBatch<S14StarGraphCandidateState>,
    constraint: &ConstraintRound,
    routing: &S14InformationGainRoutingReceipt,
) -> S14StarGraphResult<S14StarwaveCorrectionBatch> {
    if batch.branches().len() != 1
        || batch.positions() as usize != constraint.candidate_len
        || constraint.accepted_prefix_len > constraint.candidate_len
    {
        return Err(S14StarGraphError::Adapter(HelixError::new(
            "DualHelix constraint round 无法映射到单一 Starwave branch",
        )));
    }
    let branch = &batch.branches()[0];
    let mut deltas = Vec::with_capacity(branch.steps().len());
    for (offset, step) in branch.steps().iter().enumerate() {
        let verdict = if offset < constraint.accepted_prefix_len {
            S14StarwaveCapabilityVerdict::Reliable
        } else {
            S14StarwaveCapabilityVerdict::Unreliable
        };
        let proof = S14StarwaveProof::new(
            "polaris-s14-stargraph-dual-helix-delta-v1",
            constraint_delta_digest(constraint, routing, offset, step.token_id())
                .as_bytes()
                .to_vec(),
        )
        .map_err(S14StarGraphError::Starwave)?;
        deltas.push(
            S14StarwaveCapabilityDelta::new(
                branch.branch_id(),
                step.position(),
                step.token_id(),
                0.0,
                verdict,
                proof,
            )
            .map_err(S14StarGraphError::Starwave)?,
        );
    }
    let authority_proof = S14StarwaveProof::new(
        "polaris-s14-stargraph-dual-helix-authority-v1",
        constraint_authority_digest(constraint, routing)
            .as_bytes()
            .to_vec(),
    )
    .map_err(S14StarGraphError::Starwave)?;
    S14StarwaveCorrectionBatch::new(
        batch.batch_id(),
        batch.batch_sha256(),
        deltas,
        authority_proof,
    )
    .map_err(S14StarGraphError::Starwave)
}

fn abort_prepared(
    physical_committer: &mut dyn S14StarGraphPhysicalCommitter,
    routing: S14InformationGainRoutingReceipt,
    constraint: ConstraintRound,
    prepared: S14StarwavePreparedTransaction<S14StarGraphCandidateState>,
    mut reason: String,
) -> S14StarGraphOutcome {
    let rollback_receipt = prepared.rollback_receipt().clone();
    let cleanup = {
        let request = S14StarGraphPhysicalCommitRequest {
            routing: &routing,
            constraint: &constraint,
            prepared: &prepared,
        };
        physical_committer.abort_prepared(&request, &reason)
    };
    if let Err(error) = cleanup {
        reason.push_str("；外部 abort 清理失败: ");
        reason.push_str(&error.to_string());
    }
    let rolled_back = prepared.abort();
    S14StarGraphOutcome::Aborted {
        routing,
        constraint,
        rollback_receipt,
        rolled_back,
        reason,
    }
}

fn routing_digest(routing: &S14InformationGainRoutingReceipt) -> Digest {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-stargraph-routing-adapter-v1");
    writer.write_u32(S14_STARGRAPH_ENGINE_ABI_VERSION);
    writer.write_u32(routing.router_version);
    writer.write_u64(routing.input_fingerprint);
    writer.write_u64(routing.decision_fingerprint);
    writer.write_u64(routing.selected.len() as u64);
    for selected in &routing.selected {
        writer.write_u32(selected.ordinal);
        writer.write_u64(selected.capsule_id.0);
        match selected.execution {
            S14CandidateExecution::NativeCapsule => writer.write_u8(0),
            S14CandidateExecution::LegacyLocalOperator {
                boundary_key,
                operator_key,
            } => {
                writer.write_u8(1);
                writer.write_u32(boundary_key);
                writer.write_u64(operator_key);
            }
        }
        writer.write_u64(selected.marginal_entropy_drop_nanobits);
        writer.write_u64(selected.cost.transferred_bytes);
        writer.write_u64(selected.cost.working_vram_bytes);
        writer.write_u64(selected.cost.total_latency_ns);
    }
    *writer.finish().as_bytes()
}

fn generation_adapter_digest(
    candidate: &GenerationCandidateBlock,
    request: &S14StarwaveCandidateRequest,
    capsule_route_digest: Digest,
) -> S14StarwaveSha256 {
    let mut writer =
        S14StarwaveProofWriter::new("polaris-s14-stargraph-generation-adapter-binding-v1");
    writer.write_u32(S14_STARGRAPH_ENGINE_ABI_VERSION);
    writer.write_u64(candidate.transaction_id);
    writer.write_str(&candidate.chain_id);
    writer.write_u64(request.batch_id);
    writer.write_u64(request.base_commit_epoch);
    writer.write_u64(request.base_position);
    writer.write_u8(request.positions);
    writer.write_sha256(request.anchor_sha256);
    writer.write_bytes(&capsule_route_digest);
    writer.write_u64(candidate.candidates.len() as u64);
    for item in &candidate.candidates {
        writer.write_u64(item.position_offset as u64);
        writer.write_u32(item.token_id);
        writer.write_u32(item.confidence.get());
        writer.write_u32(item.uncertainty.get());
        writer.write_u64(item.checkpoint.checkpoint_id);
        writer.write_bytes(&item.checkpoint.state_digest);
    }
    writer.finish()
}

fn constraint_authority_digest(
    constraint: &ConstraintRound,
    routing: &S14InformationGainRoutingReceipt,
) -> S14StarwaveSha256 {
    let mut writer =
        S14StarwaveProofWriter::new("polaris-s14-stargraph-constraint-authority-binding-v1");
    writer.write_u32(S14_STARGRAPH_ENGINE_ABI_VERSION);
    writer.write_u64(constraint.transaction_id);
    writer.write_u32(constraint.round);
    writer.write_u64(constraint.candidate_len as u64);
    writer.write_u64(constraint.accepted_prefix_len as u64);
    writer.write_bytes(&routing_digest(routing));
    writer.write_u64(constraint.evidence.len() as u64);
    for record in &constraint.evidence {
        writer.write_str(&record.provider.provider_id);
        writer.write_str(&record.provider.revision);
        writer.write_u8(record.provider.source as u8);
        writer.write_u64(record.evidence.evidence_id);
        writer.write_u64(record.evidence.scope.start as u64);
        writer.write_u64(record.evidence.scope.end as u64);
        writer.write_u32(record.evidence.support.get());
        writer.write_u32(record.evidence.confidence.get());
        writer.write_bytes(&record.evidence.evidence_digest);
        write_constraint_decision(&mut writer, &record.evidence.decision);
    }
    writer.write_u64(constraint.sparse_decisions.len() as u64);
    for decision in &constraint.sparse_decisions {
        writer.write_u64(decision.position_offset as u64);
        writer.write_u32(decision.aggregate_score.get());
        writer.write_u32(decision.margin.get());
        write_constraint_decision(&mut writer, &decision.decision);
    }
    writer.finish()
}

fn constraint_delta_digest(
    constraint: &ConstraintRound,
    routing: &S14InformationGainRoutingReceipt,
    position_offset: usize,
    token_id: u32,
) -> S14StarwaveSha256 {
    let mut writer =
        S14StarwaveProofWriter::new("polaris-s14-stargraph-constraint-delta-binding-v1");
    writer.write_sha256(constraint_authority_digest(constraint, routing));
    writer.write_u64(position_offset as u64);
    writer.write_u32(token_id);
    writer.write_u8((position_offset < constraint.accepted_prefix_len) as u8);
    writer.finish()
}

fn write_constraint_decision(writer: &mut S14StarwaveProofWriter, decision: &ConstraintDecision) {
    match decision {
        ConstraintDecision::Accept => writer.write_u8(0),
        ConstraintDecision::Repair(repair) => {
            writer.write_u8(1);
            writer.write_u64(repair.scope.start as u64);
            writer.write_u64(repair.scope.end as u64);
            writer.write_u64(repair.replacement_token_ids.len() as u64);
            for token_id in &repair.replacement_token_ids {
                writer.write_u32(*token_id);
            }
            writer.write_bytes(&repair.repair_digest);
        }
        ConstraintDecision::Rollback { keep_prefix_len } => {
            writer.write_u8(2);
            writer.write_u64(*keep_prefix_len as u64);
        }
        ConstraintDecision::InvokeCapsule(request) => {
            writer.write_u8(3);
            writer.write_u64(request.request_id);
            writer.write_str(&request.capsule_id);
            writer.write_str(&request.revision);
            writer.write_u64(request.scope.start as u64);
            writer.write_u64(request.scope.end as u64);
            writer.write_u32(request.expected_information_gain.get());
            writer.write_bytes(&request.input_digest);
            writer.write_u64(request.budget.max_positions as u64);
            writer.write_u64(request.budget.max_transfer_bytes);
            writer.write_u64(request.budget.max_flops);
            writer.write_u64(request.budget.max_latency_micros);
        }
    }
}
