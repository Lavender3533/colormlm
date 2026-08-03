use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const S14_STARWAVE_MIN_POSITIONS: u8 = 8;
pub const S14_STARWAVE_MAX_POSITIONS: u8 = 32;
pub const S14_STARWAVE_RECEIPT_SCHEMA_VERSION: u32 = 1;

pub type S14StarwaveResult<T> = Result<T, S14StarwaveError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarwaveError {
    message: String,
}

impl S14StarwaveError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for S14StarwaveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for S14StarwaveError {}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct S14StarwaveSha256([u8; 32]);

impl S14StarwaveSha256 {
    pub const ZERO: Self = Self([0; 32]);

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(64);
        for byte in self.0 {
            encoded.push(HEX[usize::from(byte >> 4)] as char);
            encoded.push(HEX[usize::from(byte & 0x0f)] as char);
        }
        encoded
    }
}

impl fmt::Debug for S14StarwaveSha256 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("S14StarwaveSha256")
            .field(&self.to_hex())
            .finish()
    }
}

impl fmt::Display for S14StarwaveSha256 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

pub struct S14StarwaveProofWriter {
    sha256: Sha256,
}

impl S14StarwaveProofWriter {
    pub fn new(domain: &str) -> Self {
        let mut writer = Self {
            sha256: Sha256::new(),
        };
        writer.write_bytes(domain.as_bytes());
        writer
    }

    pub fn write_u8(&mut self, value: u8) {
        self.sha256.update(&[0x01, value]);
    }

    pub fn write_u32(&mut self, value: u32) {
        self.sha256.update(&[0x02]);
        self.sha256.update(&value.to_le_bytes());
    }

    pub fn write_u64(&mut self, value: u64) {
        self.sha256.update(&[0x03]);
        self.sha256.update(&value.to_le_bytes());
    }

    pub fn write_f32(&mut self, value: f32) {
        self.sha256.update(&[0x04]);
        self.sha256.update(&value.to_bits().to_le_bytes());
    }

    pub fn write_bytes(&mut self, value: &[u8]) {
        self.sha256.update(&[0x05]);
        self.sha256.update(&(value.len() as u64).to_le_bytes());
        self.sha256.update(value);
    }

    pub fn write_str(&mut self, value: &str) {
        self.sha256.update(&[0x06]);
        self.sha256.update(&(value.len() as u64).to_le_bytes());
        self.sha256.update(value.as_bytes());
    }

    pub fn write_sha256(&mut self, value: S14StarwaveSha256) {
        self.sha256.update(&[0x07]);
        self.sha256.update(value.as_bytes());
    }

    pub fn finish(self) -> S14StarwaveSha256 {
        let digest = self.sha256.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&digest);
        S14StarwaveSha256(bytes)
    }
}

pub trait S14StarwaveStateProof {
    fn write_state_proof(&self, writer: &mut S14StarwaveProofWriter);
}

impl S14StarwaveStateProof for () {
    fn write_state_proof(&self, writer: &mut S14StarwaveProofWriter) {
        writer.write_bytes(&[]);
    }
}

impl S14StarwaveStateProof for Vec<u8> {
    fn write_state_proof(&self, writer: &mut S14StarwaveProofWriter) {
        writer.write_bytes(self);
    }
}

impl S14StarwaveStateProof for Box<[u8]> {
    fn write_state_proof(&self, writer: &mut S14StarwaveProofWriter) {
        writer.write_bytes(self);
    }
}

impl<const N: usize> S14StarwaveStateProof for [u8; N] {
    fn write_state_proof(&self, writer: &mut S14StarwaveProofWriter) {
        writer.write_bytes(self);
    }
}

impl S14StarwaveStateProof for String {
    fn write_state_proof(&self, writer: &mut S14StarwaveProofWriter) {
        writer.write_str(self);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarwaveProof {
    scheme: String,
    payload: Vec<u8>,
    sha256: S14StarwaveSha256,
}

impl S14StarwaveProof {
    pub fn new(scheme: impl Into<String>, payload: Vec<u8>) -> S14StarwaveResult<Self> {
        let scheme = scheme.into();
        if scheme.is_empty() {
            return Err(S14StarwaveError::new("Starwave proof scheme 不能为空"));
        }
        let sha256 = proof_sha256(&scheme, &payload);
        Ok(Self {
            scheme,
            payload,
            sha256,
        })
    }

    pub fn scheme(&self) -> &str {
        &self.scheme
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub const fn sha256(&self) -> S14StarwaveSha256 {
        self.sha256
    }

    pub fn validate(&self) -> S14StarwaveResult<()> {
        if self.scheme.is_empty() || proof_sha256(&self.scheme, &self.payload) != self.sha256 {
            return Err(S14StarwaveError::new("Starwave proof SHA-256 漂移"));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct S14StarwaveCandidateStep<S> {
    position: u64,
    token_id: u32,
    latent_reliability: f32,
    state: S,
    state_sha256: S14StarwaveSha256,
}

impl<S: S14StarwaveStateProof> S14StarwaveCandidateStep<S> {
    pub fn new(
        position: u64,
        token_id: u32,
        latent_reliability: f32,
        state: S,
    ) -> S14StarwaveResult<Self> {
        if !latent_reliability.is_finite() {
            return Err(S14StarwaveError::new(
                "Starwave latent reliability 必须是有限数",
            ));
        }
        let state_sha256 = state_sha256(&state);
        Ok(Self {
            position,
            token_id,
            latent_reliability,
            state,
            state_sha256,
        })
    }

    pub const fn position(&self) -> u64 {
        self.position
    }

    pub const fn token_id(&self) -> u32 {
        self.token_id
    }

    pub const fn latent_reliability(&self) -> f32 {
        self.latent_reliability
    }

    pub const fn state_sha256(&self) -> S14StarwaveSha256 {
        self.state_sha256
    }

    pub fn state(&self) -> &S {
        &self.state
    }

    pub fn into_state(self) -> S {
        self.state
    }

    fn validate(&self) -> S14StarwaveResult<()> {
        if !self.latent_reliability.is_finite() {
            return Err(S14StarwaveError::new(
                "Starwave candidate latent reliability 漂移为非有限数",
            ));
        }
        if state_sha256(&self.state) != self.state_sha256 {
            return Err(S14StarwaveError::new(
                "Starwave candidate state proof SHA-256 漂移",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct S14StarwaveBranch<S> {
    branch_id: u64,
    steps: Vec<S14StarwaveCandidateStep<S>>,
    branch_sha256: S14StarwaveSha256,
}

impl<S: S14StarwaveStateProof> S14StarwaveBranch<S> {
    pub fn new(branch_id: u64, steps: Vec<S14StarwaveCandidateStep<S>>) -> S14StarwaveResult<Self> {
        if steps.is_empty() {
            return Err(S14StarwaveError::new("Starwave branch 不能为空"));
        }
        let branch_sha256 = branch_sha256(branch_id, &steps)?;
        Ok(Self {
            branch_id,
            steps,
            branch_sha256,
        })
    }

    pub const fn branch_id(&self) -> u64 {
        self.branch_id
    }

    pub fn steps(&self) -> &[S14StarwaveCandidateStep<S>] {
        &self.steps
    }

    pub const fn branch_sha256(&self) -> S14StarwaveSha256 {
        self.branch_sha256
    }

    fn validate(&self) -> S14StarwaveResult<()> {
        if branch_sha256(self.branch_id, &self.steps)? != self.branch_sha256 {
            return Err(S14StarwaveError::new("Starwave branch proof SHA-256 漂移"));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct S14StarwaveProducedCandidates<S> {
    branches: Vec<S14StarwaveBranch<S>>,
    producer_proof: S14StarwaveProof,
}

impl<S> S14StarwaveProducedCandidates<S> {
    pub fn new(branches: Vec<S14StarwaveBranch<S>>, producer_proof: S14StarwaveProof) -> Self {
        Self {
            branches,
            producer_proof,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarwaveCandidateRequest {
    pub batch_id: u64,
    pub base_commit_epoch: u64,
    pub base_position: u64,
    pub positions: u8,
    pub max_branches: u16,
    pub anchor_sha256: S14StarwaveSha256,
}

pub trait S14StarwaveCandidateProducer {
    type State: S14StarwaveStateProof;
    type Error;

    fn produce_candidates(
        &mut self,
        request: &S14StarwaveCandidateRequest,
    ) -> Result<S14StarwaveProducedCandidates<Self::State>, Self::Error>;
}

#[derive(Debug)]
pub enum S14StarwaveProduceError<E> {
    Producer(E),
    Core(S14StarwaveError),
}

impl<E: fmt::Display> fmt::Display for S14StarwaveProduceError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Producer(error) => write!(formatter, "Starwave candidate producer: {error}"),
            Self::Core(error) => error.fmt(formatter),
        }
    }
}

impl<E: Error + 'static> Error for S14StarwaveProduceError<E> {}

#[derive(Debug)]
pub struct S14StarwaveCandidateBatch<S> {
    batch_id: u64,
    base_commit_epoch: u64,
    base_position: u64,
    positions: u8,
    anchor_sha256: S14StarwaveSha256,
    branches: Vec<S14StarwaveBranch<S>>,
    producer_proof: S14StarwaveProof,
    batch_sha256: S14StarwaveSha256,
}

impl<S> S14StarwaveCandidateBatch<S> {
    pub const fn batch_id(&self) -> u64 {
        self.batch_id
    }

    pub const fn base_commit_epoch(&self) -> u64 {
        self.base_commit_epoch
    }

    pub const fn base_position(&self) -> u64 {
        self.base_position
    }

    pub const fn positions(&self) -> u8 {
        self.positions
    }

    pub const fn anchor_sha256(&self) -> S14StarwaveSha256 {
        self.anchor_sha256
    }

    pub fn branches(&self) -> &[S14StarwaveBranch<S>] {
        &self.branches
    }

    pub fn producer_proof(&self) -> &S14StarwaveProof {
        &self.producer_proof
    }

    pub const fn batch_sha256(&self) -> S14StarwaveSha256 {
        self.batch_sha256
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarwaveCapabilityVerdict {
    Reliable,
    Unreliable,
    Conflict { authoritative_token_id: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct S14StarwaveCapabilityDelta {
    branch_id: u64,
    position: u64,
    candidate_token_id: u32,
    reliability_delta: f32,
    verdict: S14StarwaveCapabilityVerdict,
    proof: S14StarwaveProof,
}

impl S14StarwaveCapabilityDelta {
    pub fn new(
        branch_id: u64,
        position: u64,
        candidate_token_id: u32,
        reliability_delta: f32,
        verdict: S14StarwaveCapabilityVerdict,
        proof: S14StarwaveProof,
    ) -> S14StarwaveResult<Self> {
        if !reliability_delta.is_finite() {
            return Err(S14StarwaveError::new(
                "Starwave capability reliability delta 必须是有限数",
            ));
        }
        proof.validate()?;
        Ok(Self {
            branch_id,
            position,
            candidate_token_id,
            reliability_delta,
            verdict,
            proof,
        })
    }

    pub const fn branch_id(&self) -> u64 {
        self.branch_id
    }

    pub const fn position(&self) -> u64 {
        self.position
    }

    pub const fn candidate_token_id(&self) -> u32 {
        self.candidate_token_id
    }

    pub const fn reliability_delta(&self) -> f32 {
        self.reliability_delta
    }

    pub const fn verdict(&self) -> S14StarwaveCapabilityVerdict {
        self.verdict
    }

    pub fn proof(&self) -> &S14StarwaveProof {
        &self.proof
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct S14StarwaveCorrectionBatch {
    batch_id: u64,
    candidate_batch_sha256: S14StarwaveSha256,
    deltas: Vec<S14StarwaveCapabilityDelta>,
    authority_proof: S14StarwaveProof,
    correction_sha256: S14StarwaveSha256,
}

impl S14StarwaveCorrectionBatch {
    pub fn new(
        batch_id: u64,
        candidate_batch_sha256: S14StarwaveSha256,
        mut deltas: Vec<S14StarwaveCapabilityDelta>,
        authority_proof: S14StarwaveProof,
    ) -> S14StarwaveResult<Self> {
        authority_proof.validate()?;
        deltas.sort_by_key(|delta| (delta.branch_id, delta.position));
        for pair in deltas.windows(2) {
            if (pair[0].branch_id, pair[0].position) == (pair[1].branch_id, pair[1].position) {
                return Err(S14StarwaveError::new(
                    "Starwave capability delta 出现重复 branch/position",
                ));
            }
        }
        for delta in &deltas {
            if !delta.reliability_delta.is_finite() {
                return Err(S14StarwaveError::new(
                    "Starwave capability delta 漂移为非有限数",
                ));
            }
            delta.proof.validate()?;
        }
        let correction_sha256 = correction_batch_sha256(
            batch_id,
            candidate_batch_sha256,
            &deltas,
            authority_proof.sha256(),
        );
        Ok(Self {
            batch_id,
            candidate_batch_sha256,
            deltas,
            authority_proof,
            correction_sha256,
        })
    }

    pub const fn batch_id(&self) -> u64 {
        self.batch_id
    }

    pub const fn candidate_batch_sha256(&self) -> S14StarwaveSha256 {
        self.candidate_batch_sha256
    }

    pub fn deltas(&self) -> &[S14StarwaveCapabilityDelta] {
        &self.deltas
    }

    pub fn authority_proof(&self) -> &S14StarwaveProof {
        &self.authority_proof
    }

    pub const fn correction_sha256(&self) -> S14StarwaveSha256 {
        self.correction_sha256
    }

    fn validate(&self) -> S14StarwaveResult<()> {
        self.authority_proof.validate()?;
        for delta in &self.deltas {
            if !delta.reliability_delta.is_finite() {
                return Err(S14StarwaveError::new(
                    "Starwave capability delta 漂移为非有限数",
                ));
            }
            delta.proof.validate()?;
        }
        if correction_batch_sha256(
            self.batch_id,
            self.candidate_batch_sha256,
            &self.deltas,
            self.authority_proof.sha256(),
        ) != self.correction_sha256
        {
            return Err(S14StarwaveError::new(
                "Starwave correction batch SHA-256 漂移",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum S14StarwavePrefixBoundary {
    Complete,
    MissingDelta {
        position: u64,
    },
    BelowReliability {
        position: u64,
        corrected_reliability: f32,
    },
    Unreliable {
        position: u64,
    },
    FirstConflict {
        position: u64,
        candidate_token_id: u32,
        authoritative_token_id: u32,
    },
}

impl S14StarwavePrefixBoundary {
    pub const fn position(&self) -> Option<u64> {
        match self {
            Self::Complete => None,
            Self::MissingDelta { position }
            | Self::BelowReliability { position, .. }
            | Self::Unreliable { position }
            | Self::FirstConflict { position, .. } => Some(*position),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct S14StarwaveBranchDecision {
    pub branch_id: u64,
    pub reliable_prefix_positions: u8,
    pub corrected_reliability_sum: f64,
    pub boundary: S14StarwavePrefixBoundary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum S14StarwaveRollbackReason {
    CompetingBranch,
    FirstConflict { authoritative_token_id: u32 },
    ReliabilityBoundary,
    AfterFirstConflict { conflict_position: u64 },
    AfterReliabilityBoundary { boundary_position: u64 },
    TransactionAborted,
}

#[derive(Debug)]
pub struct S14StarwaveRolledBackStep<S> {
    pub branch_id: u64,
    pub reason: S14StarwaveRollbackReason,
    pub step: S14StarwaveCandidateStep<S>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct S14StarwaveRollbackReceipt {
    pub batch_id: u64,
    pub selected_branch_id: Option<u64>,
    pub retained_positions: u8,
    pub rolled_back_positions: u64,
    pub branch_decisions: Vec<S14StarwaveBranchDecision>,
    pub rollback_sha256: S14StarwaveSha256,
}

impl S14StarwaveRollbackReceipt {
    pub fn validate(&self, candidate_batch_sha256: S14StarwaveSha256) -> S14StarwaveResult<()> {
        if rollback_receipt_sha256(
            self.batch_id,
            candidate_batch_sha256,
            self.selected_branch_id,
            self.retained_positions,
            self.rolled_back_positions,
            &self.branch_decisions,
        ) != self.rollback_sha256
        {
            return Err(S14StarwaveError::new(
                "Starwave rollback receipt SHA-256 漂移",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct S14StarwavePreparedTransaction<S> {
    batch_id: u64,
    base_commit_epoch: u64,
    base_position: u64,
    anchor_sha256: S14StarwaveSha256,
    candidate_batch_sha256: S14StarwaveSha256,
    correction_sha256: S14StarwaveSha256,
    selected_branch_id: Option<u64>,
    committed_prefix: Vec<S14StarwaveCandidateStep<S>>,
    rolled_back: Vec<S14StarwaveRolledBackStep<S>>,
    rollback_receipt: S14StarwaveRollbackReceipt,
    prefix_sha256: S14StarwaveSha256,
}

impl<S> S14StarwavePreparedTransaction<S> {
    pub const fn batch_id(&self) -> u64 {
        self.batch_id
    }

    pub const fn base_commit_epoch(&self) -> u64 {
        self.base_commit_epoch
    }

    pub const fn base_position(&self) -> u64 {
        self.base_position
    }

    pub const fn selected_branch_id(&self) -> Option<u64> {
        self.selected_branch_id
    }

    pub fn committed_prefix(&self) -> &[S14StarwaveCandidateStep<S>] {
        &self.committed_prefix
    }

    pub fn rolled_back(&self) -> &[S14StarwaveRolledBackStep<S>] {
        &self.rolled_back
    }

    pub fn rollback_receipt(&self) -> &S14StarwaveRollbackReceipt {
        &self.rollback_receipt
    }

    pub fn will_commit(&self) -> bool {
        !self.committed_prefix.is_empty()
    }

    pub fn abort(mut self) -> Vec<S14StarwaveRolledBackStep<S>> {
        let branch_id = self.selected_branch_id.unwrap_or(0);
        self.rolled_back
            .extend(
                self.committed_prefix
                    .drain(..)
                    .map(|step| S14StarwaveRolledBackStep {
                        branch_id,
                        reason: S14StarwaveRollbackReason::TransactionAborted,
                        step,
                    }),
            );
        self.rolled_back
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarwaveCommitReceipt {
    pub schema_version: u32,
    pub batch_id: u64,
    pub selected_branch_id: u64,
    pub start_position: u64,
    pub committed_positions: u8,
    pub end_position_exclusive: u64,
    pub previous_commit_epoch: u64,
    pub commit_epoch: u64,
    pub previous_commit_sha256: S14StarwaveSha256,
    pub candidate_batch_sha256: S14StarwaveSha256,
    pub correction_sha256: S14StarwaveSha256,
    pub prefix_sha256: S14StarwaveSha256,
    pub rollback_sha256: S14StarwaveSha256,
    pub backend_proof_sha256: S14StarwaveSha256,
    pub proof_sha256: S14StarwaveSha256,
    pub commit_sha256: S14StarwaveSha256,
}

impl S14StarwaveCommitReceipt {
    pub fn validate(&self) -> S14StarwaveResult<()> {
        if self.schema_version != S14_STARWAVE_RECEIPT_SCHEMA_VERSION
            || self.committed_positions == 0
            || self.committed_positions > S14_STARWAVE_MAX_POSITIONS
            || self.commit_epoch
                != self.previous_commit_epoch.checked_add(1).ok_or_else(|| {
                    S14StarwaveError::new("Starwave receipt commit epoch overflow")
                })?
            || self.end_position_exclusive
                != self
                    .start_position
                    .checked_add(u64::from(self.committed_positions))
                    .ok_or_else(|| S14StarwaveError::new("Starwave receipt position overflow"))?
        {
            return Err(S14StarwaveError::new(
                "Starwave commit receipt 单调 epoch/position 字段非法",
            ));
        }
        let proof_sha256 = commit_proof_sha256(
            self.candidate_batch_sha256,
            self.correction_sha256,
            self.prefix_sha256,
            self.rollback_sha256,
            self.backend_proof_sha256,
        );
        if proof_sha256 != self.proof_sha256 {
            return Err(S14StarwaveError::new(
                "Starwave commit receipt proof SHA-256 漂移",
            ));
        }
        let expected_commit = commit_receipt_sha256(
            self.batch_id,
            self.selected_branch_id,
            self.start_position,
            self.committed_positions,
            self.end_position_exclusive,
            self.previous_commit_epoch,
            self.commit_epoch,
            self.previous_commit_sha256,
            self.proof_sha256,
        );
        if expected_commit != self.commit_sha256 {
            return Err(S14StarwaveError::new(
                "Starwave commit receipt SHA-256 漂移",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct S14StarwaveTransactionOutcome<S> {
    pub committed_prefix: Vec<S14StarwaveCandidateStep<S>>,
    pub rolled_back: Vec<S14StarwaveRolledBackStep<S>>,
    pub rollback_receipt: S14StarwaveRollbackReceipt,
    pub commit_receipt: Option<S14StarwaveCommitReceipt>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct S14StarwaveTransactionConfig {
    pub minimum_corrected_reliability: f32,
    pub max_branches: u16,
}

impl S14StarwaveTransactionConfig {
    pub fn validate(self) -> S14StarwaveResult<Self> {
        if !self.minimum_corrected_reliability.is_finite() {
            return Err(S14StarwaveError::new(
                "Starwave minimum corrected reliability 必须是有限数",
            ));
        }
        if self.max_branches == 0 {
            return Err(S14StarwaveError::new("Starwave max_branches 必须大于 0"));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct S14StarwaveTransaction {
    config: S14StarwaveTransactionConfig,
    commit_epoch: u64,
    next_position: u64,
    last_commit_sha256: S14StarwaveSha256,
    next_batch_id: u64,
}

impl S14StarwaveTransaction {
    pub fn new(
        config: S14StarwaveTransactionConfig,
        initial_position: u64,
    ) -> S14StarwaveResult<Self> {
        let config = config.validate()?;
        let mut genesis = S14StarwaveProofWriter::new("polaris-s14-starwave-genesis-v1");
        genesis.write_u64(initial_position);
        Ok(Self {
            config,
            commit_epoch: 0,
            next_position: initial_position,
            last_commit_sha256: genesis.finish(),
            next_batch_id: 1,
        })
    }

    pub fn from_checkpoint(
        config: S14StarwaveTransactionConfig,
        commit_epoch: u64,
        next_position: u64,
        last_commit_sha256: S14StarwaveSha256,
        next_batch_id: u64,
    ) -> S14StarwaveResult<Self> {
        let config = config.validate()?;
        if next_batch_id == 0 || last_commit_sha256 == S14StarwaveSha256::ZERO {
            return Err(S14StarwaveError::new(
                "Starwave checkpoint batch id/commit SHA 非法",
            ));
        }
        Ok(Self {
            config,
            commit_epoch,
            next_position,
            last_commit_sha256,
            next_batch_id,
        })
    }

    pub const fn commit_epoch(&self) -> u64 {
        self.commit_epoch
    }

    pub const fn next_position(&self) -> u64 {
        self.next_position
    }

    pub const fn last_commit_sha256(&self) -> S14StarwaveSha256 {
        self.last_commit_sha256
    }

    pub const fn next_batch_id(&self) -> u64 {
        self.next_batch_id
    }

    pub fn produce_batch<P: S14StarwaveCandidateProducer>(
        &mut self,
        positions: u8,
        producer: &mut P,
    ) -> Result<S14StarwaveCandidateBatch<P::State>, S14StarwaveProduceError<P::Error>> {
        if !(S14_STARWAVE_MIN_POSITIONS..=S14_STARWAVE_MAX_POSITIONS).contains(&positions) {
            return Err(S14StarwaveProduceError::Core(S14StarwaveError::new(
                "Starwave candidate batch positions 必须位于 8..=32",
            )));
        }
        self.next_position
            .checked_add(u64::from(positions))
            .ok_or_else(|| {
                S14StarwaveProduceError::Core(S14StarwaveError::new(
                    "Starwave candidate batch position overflow",
                ))
            })?;
        let following_batch_id = self.next_batch_id.checked_add(1).ok_or_else(|| {
            S14StarwaveProduceError::Core(S14StarwaveError::new(
                "Starwave candidate batch id overflow",
            ))
        })?;
        let request = S14StarwaveCandidateRequest {
            batch_id: self.next_batch_id,
            base_commit_epoch: self.commit_epoch,
            base_position: self.next_position,
            positions,
            max_branches: self.config.max_branches,
            anchor_sha256: self.last_commit_sha256,
        };
        let produced = producer
            .produce_candidates(&request)
            .map_err(S14StarwaveProduceError::Producer)?;
        let batch =
            seal_candidate_batch(&request, produced).map_err(S14StarwaveProduceError::Core)?;
        self.next_batch_id = following_batch_id;
        Ok(batch)
    }

    pub fn prepare<S: S14StarwaveStateProof>(
        &self,
        batch: S14StarwaveCandidateBatch<S>,
        corrections: S14StarwaveCorrectionBatch,
    ) -> S14StarwaveResult<S14StarwavePreparedTransaction<S>> {
        validate_batch_anchor(self, &batch)?;
        validate_candidate_batch(&batch, self.config.max_branches)?;
        corrections.validate()?;
        if corrections.batch_id != batch.batch_id
            || corrections.candidate_batch_sha256 != batch.batch_sha256
        {
            return Err(S14StarwaveError::new(
                "Starwave correction batch 未绑定当前 candidate batch",
            ));
        }

        let candidate_steps = candidate_step_index(&batch)?;
        let mut deltas = BTreeMap::new();
        for delta in &corrections.deltas {
            let key = (delta.branch_id, delta.position);
            let candidate_token_id = candidate_steps.get(&key).ok_or_else(|| {
                S14StarwaveError::new("Starwave capability delta 引用了不存在的 branch/position")
            })?;
            if **candidate_token_id != delta.candidate_token_id {
                return Err(S14StarwaveError::new(
                    "Starwave capability delta candidate token 漂移",
                ));
            }
            if deltas.insert(key, delta).is_some() {
                return Err(S14StarwaveError::new("Starwave capability delta 重复"));
            }
        }

        let decisions = batch
            .branches
            .iter()
            .map(|branch| {
                analyze_branch(branch, &deltas, self.config.minimum_corrected_reliability)
            })
            .collect::<S14StarwaveResult<Vec<_>>>()?;
        let selected_index = decisions
            .iter()
            .enumerate()
            .filter(|(_, decision)| decision.reliable_prefix_positions > 0)
            .max_by(|(_, left), (_, right)| compare_decisions(left, right))
            .map(|(index, _)| index);
        let selected_branch_id = selected_index.map(|index| decisions[index].branch_id);
        let retained_positions = selected_index
            .map(|index| decisions[index].reliable_prefix_positions)
            .unwrap_or(0);
        let total_positions = (batch.branches.len() as u64)
            .checked_mul(u64::from(batch.positions))
            .ok_or_else(|| S14StarwaveError::new("Starwave rollback position count overflow"))?;
        let rolled_back_positions = total_positions
            .checked_sub(u64::from(retained_positions))
            .ok_or_else(|| S14StarwaveError::new("Starwave rollback position count underflow"))?;
        let rollback_sha256 = rollback_receipt_sha256(
            batch.batch_id,
            batch.batch_sha256,
            selected_branch_id,
            retained_positions,
            rolled_back_positions,
            &decisions,
        );
        let rollback_receipt = S14StarwaveRollbackReceipt {
            batch_id: batch.batch_id,
            selected_branch_id,
            retained_positions,
            rolled_back_positions,
            branch_decisions: decisions.clone(),
            rollback_sha256,
        };

        let mut committed_prefix = Vec::with_capacity(usize::from(retained_positions));
        let mut rolled_back = Vec::with_capacity(
            usize::try_from(rolled_back_positions)
                .map_err(|_| S14StarwaveError::new("Starwave rollback capacity overflow"))?,
        );
        for (branch_index, branch) in batch.branches.into_iter().enumerate() {
            let decision = &decisions[branch_index];
            if Some(branch_index) != selected_index {
                rolled_back.extend(branch.steps.into_iter().map(|step| {
                    S14StarwaveRolledBackStep {
                        branch_id: branch.branch_id,
                        reason: S14StarwaveRollbackReason::CompetingBranch,
                        step,
                    }
                }));
                continue;
            }
            let prefix_len = usize::from(decision.reliable_prefix_positions);
            let boundary_position = decision.boundary.position();
            for (step_index, step) in branch.steps.into_iter().enumerate() {
                if step_index < prefix_len {
                    committed_prefix.push(step);
                } else {
                    let reason = selected_rollback_reason(
                        &decision.boundary,
                        boundary_position,
                        step_index == prefix_len,
                    );
                    rolled_back.push(S14StarwaveRolledBackStep {
                        branch_id: branch.branch_id,
                        reason,
                        step,
                    });
                }
            }
        }
        let prefix_sha256 = committed_prefix_sha256(selected_branch_id, &committed_prefix);
        Ok(S14StarwavePreparedTransaction {
            batch_id: batch.batch_id,
            base_commit_epoch: batch.base_commit_epoch,
            base_position: batch.base_position,
            anchor_sha256: batch.anchor_sha256,
            candidate_batch_sha256: batch.batch_sha256,
            correction_sha256: corrections.correction_sha256,
            selected_branch_id,
            committed_prefix,
            rolled_back,
            rollback_receipt,
            prefix_sha256,
        })
    }

    pub fn finalize<S>(
        &mut self,
        prepared: S14StarwavePreparedTransaction<S>,
        backend_commit_proof: Option<S14StarwaveProof>,
    ) -> S14StarwaveResult<S14StarwaveTransactionOutcome<S>> {
        if prepared.base_commit_epoch != self.commit_epoch
            || prepared.base_position != self.next_position
            || prepared.anchor_sha256 != self.last_commit_sha256
        {
            return Err(S14StarwaveError::new(
                "Starwave prepared transaction 已过期，拒绝推进 commit epoch",
            ));
        }
        prepared
            .rollback_receipt
            .validate(prepared.candidate_batch_sha256)?;

        if prepared.committed_prefix.is_empty() {
            if backend_commit_proof.is_some() || prepared.selected_branch_id.is_some() {
                return Err(S14StarwaveError::new(
                    "Starwave 空可靠前缀不得携带 backend commit proof",
                ));
            }
            return Ok(S14StarwaveTransactionOutcome {
                committed_prefix: prepared.committed_prefix,
                rolled_back: prepared.rolled_back,
                rollback_receipt: prepared.rollback_receipt,
                commit_receipt: None,
            });
        }

        let backend_commit_proof = backend_commit_proof.ok_or_else(|| {
            S14StarwaveError::new("Starwave 非空可靠前缀缺少 backend commit proof")
        })?;
        backend_commit_proof.validate()?;
        let selected_branch_id = prepared
            .selected_branch_id
            .ok_or_else(|| S14StarwaveError::new("Starwave 非空可靠前缀缺少 selected branch"))?;
        let committed_positions = u8::try_from(prepared.committed_prefix.len())
            .map_err(|_| S14StarwaveError::new("Starwave committed prefix 长度超出 u8"))?;
        let commit_epoch = self
            .commit_epoch
            .checked_add(1)
            .ok_or_else(|| S14StarwaveError::new("Starwave commit epoch overflow"))?;
        let end_position_exclusive = self
            .next_position
            .checked_add(u64::from(committed_positions))
            .ok_or_else(|| S14StarwaveError::new("Starwave commit position overflow"))?;
        let proof_sha256 = commit_proof_sha256(
            prepared.candidate_batch_sha256,
            prepared.correction_sha256,
            prepared.prefix_sha256,
            prepared.rollback_receipt.rollback_sha256,
            backend_commit_proof.sha256(),
        );
        let commit_sha256 = commit_receipt_sha256(
            prepared.batch_id,
            selected_branch_id,
            self.next_position,
            committed_positions,
            end_position_exclusive,
            self.commit_epoch,
            commit_epoch,
            self.last_commit_sha256,
            proof_sha256,
        );
        let receipt = S14StarwaveCommitReceipt {
            schema_version: S14_STARWAVE_RECEIPT_SCHEMA_VERSION,
            batch_id: prepared.batch_id,
            selected_branch_id,
            start_position: self.next_position,
            committed_positions,
            end_position_exclusive,
            previous_commit_epoch: self.commit_epoch,
            commit_epoch,
            previous_commit_sha256: self.last_commit_sha256,
            candidate_batch_sha256: prepared.candidate_batch_sha256,
            correction_sha256: prepared.correction_sha256,
            prefix_sha256: prepared.prefix_sha256,
            rollback_sha256: prepared.rollback_receipt.rollback_sha256,
            backend_proof_sha256: backend_commit_proof.sha256(),
            proof_sha256,
            commit_sha256,
        };
        receipt.validate()?;
        self.commit_epoch = commit_epoch;
        self.next_position = end_position_exclusive;
        self.last_commit_sha256 = commit_sha256;
        Ok(S14StarwaveTransactionOutcome {
            committed_prefix: prepared.committed_prefix,
            rolled_back: prepared.rolled_back,
            rollback_receipt: prepared.rollback_receipt,
            commit_receipt: Some(receipt),
        })
    }
}

fn seal_candidate_batch<S: S14StarwaveStateProof>(
    request: &S14StarwaveCandidateRequest,
    mut produced: S14StarwaveProducedCandidates<S>,
) -> S14StarwaveResult<S14StarwaveCandidateBatch<S>> {
    produced.producer_proof.validate()?;
    if produced.branches.is_empty() || produced.branches.len() > usize::from(request.max_branches) {
        return Err(S14StarwaveError::new(
            "Starwave candidate branch 数量超出 1..=max_branches",
        ));
    }
    produced.branches.sort_by_key(S14StarwaveBranch::branch_id);
    let mut branch_ids = BTreeSet::new();
    for branch in &produced.branches {
        if !branch_ids.insert(branch.branch_id) {
            return Err(S14StarwaveError::new("Starwave candidate branch id 重复"));
        }
        branch.validate()?;
        if branch.steps.len() != usize::from(request.positions) {
            return Err(S14StarwaveError::new(
                "Starwave candidate branch 未覆盖请求的完整 8..=32 位置批",
            ));
        }
        for (offset, step) in branch.steps.iter().enumerate() {
            let expected_position = request
                .base_position
                .checked_add(offset as u64)
                .ok_or_else(|| S14StarwaveError::new("Starwave branch position overflow"))?;
            if step.position != expected_position {
                return Err(S14StarwaveError::new(
                    "Starwave candidate branch position 不连续",
                ));
            }
        }
    }
    let batch_sha256 = candidate_batch_sha256(
        request,
        &produced.branches,
        produced.producer_proof.sha256(),
    );
    Ok(S14StarwaveCandidateBatch {
        batch_id: request.batch_id,
        base_commit_epoch: request.base_commit_epoch,
        base_position: request.base_position,
        positions: request.positions,
        anchor_sha256: request.anchor_sha256,
        branches: produced.branches,
        producer_proof: produced.producer_proof,
        batch_sha256,
    })
}

fn validate_batch_anchor<S>(
    transaction: &S14StarwaveTransaction,
    batch: &S14StarwaveCandidateBatch<S>,
) -> S14StarwaveResult<()> {
    if batch.base_commit_epoch != transaction.commit_epoch
        || batch.base_position != transaction.next_position
        || batch.anchor_sha256 != transaction.last_commit_sha256
    {
        return Err(S14StarwaveError::new(
            "Starwave candidate batch 基于过期 commit anchor",
        ));
    }
    Ok(())
}

fn validate_candidate_batch<S: S14StarwaveStateProof>(
    batch: &S14StarwaveCandidateBatch<S>,
    max_branches: u16,
) -> S14StarwaveResult<()> {
    batch.producer_proof.validate()?;
    if !(S14_STARWAVE_MIN_POSITIONS..=S14_STARWAVE_MAX_POSITIONS).contains(&batch.positions)
        || batch.branches.is_empty()
        || batch.branches.len() > usize::from(max_branches)
    {
        return Err(S14StarwaveError::new(
            "Starwave candidate batch positions/branches 越界",
        ));
    }
    for branch in &batch.branches {
        branch.validate()?;
    }
    if candidate_batch_sha256(
        &S14StarwaveCandidateRequest {
            batch_id: batch.batch_id,
            base_commit_epoch: batch.base_commit_epoch,
            base_position: batch.base_position,
            positions: batch.positions,
            max_branches,
            anchor_sha256: batch.anchor_sha256,
        },
        &batch.branches,
        batch.producer_proof.sha256(),
    ) != batch.batch_sha256
    {
        return Err(S14StarwaveError::new(
            "Starwave candidate batch SHA-256 漂移",
        ));
    }
    Ok(())
}

fn candidate_step_index<'a, S>(
    batch: &'a S14StarwaveCandidateBatch<S>,
) -> S14StarwaveResult<BTreeMap<(u64, u64), &'a u32>> {
    let mut index = BTreeMap::new();
    for branch in &batch.branches {
        for step in &branch.steps {
            if index
                .insert((branch.branch_id, step.position), &step.token_id)
                .is_some()
            {
                return Err(S14StarwaveError::new(
                    "Starwave candidate batch branch/position 重复",
                ));
            }
        }
    }
    Ok(index)
}

fn analyze_branch<S>(
    branch: &S14StarwaveBranch<S>,
    deltas: &BTreeMap<(u64, u64), &S14StarwaveCapabilityDelta>,
    minimum_reliability: f32,
) -> S14StarwaveResult<S14StarwaveBranchDecision> {
    let mut reliable_prefix_positions = 0u8;
    let mut corrected_reliability_sum = 0.0f64;
    let mut boundary = S14StarwavePrefixBoundary::Complete;
    for step in &branch.steps {
        let Some(delta) = deltas.get(&(branch.branch_id, step.position)) else {
            boundary = S14StarwavePrefixBoundary::MissingDelta {
                position: step.position,
            };
            break;
        };
        match delta.verdict {
            S14StarwaveCapabilityVerdict::Conflict {
                authoritative_token_id,
            } => {
                boundary = S14StarwavePrefixBoundary::FirstConflict {
                    position: step.position,
                    candidate_token_id: step.token_id,
                    authoritative_token_id,
                };
                break;
            }
            S14StarwaveCapabilityVerdict::Unreliable => {
                boundary = S14StarwavePrefixBoundary::Unreliable {
                    position: step.position,
                };
                break;
            }
            S14StarwaveCapabilityVerdict::Reliable => {}
        }
        let corrected = step.latent_reliability + delta.reliability_delta;
        if !corrected.is_finite() {
            return Err(S14StarwaveError::new(
                "Starwave corrected reliability overflow",
            ));
        }
        if corrected < minimum_reliability {
            boundary = S14StarwavePrefixBoundary::BelowReliability {
                position: step.position,
                corrected_reliability: corrected,
            };
            break;
        }
        reliable_prefix_positions = reliable_prefix_positions
            .checked_add(1)
            .ok_or_else(|| S14StarwaveError::new("Starwave reliable prefix overflow"))?;
        corrected_reliability_sum += f64::from(corrected);
    }
    Ok(S14StarwaveBranchDecision {
        branch_id: branch.branch_id,
        reliable_prefix_positions,
        corrected_reliability_sum,
        boundary,
    })
}

fn compare_decisions(
    left: &S14StarwaveBranchDecision,
    right: &S14StarwaveBranchDecision,
) -> Ordering {
    left.reliable_prefix_positions
        .cmp(&right.reliable_prefix_positions)
        .then_with(|| {
            left.corrected_reliability_sum
                .partial_cmp(&right.corrected_reliability_sum)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| right.branch_id.cmp(&left.branch_id))
}

fn selected_rollback_reason(
    boundary: &S14StarwavePrefixBoundary,
    boundary_position: Option<u64>,
    is_boundary: bool,
) -> S14StarwaveRollbackReason {
    match (boundary, is_boundary) {
        (
            S14StarwavePrefixBoundary::FirstConflict {
                authoritative_token_id,
                ..
            },
            true,
        ) => S14StarwaveRollbackReason::FirstConflict {
            authoritative_token_id: *authoritative_token_id,
        },
        (S14StarwavePrefixBoundary::FirstConflict { position, .. }, false) => {
            S14StarwaveRollbackReason::AfterFirstConflict {
                conflict_position: *position,
            }
        }
        (_, true) => S14StarwaveRollbackReason::ReliabilityBoundary,
        (_, false) => S14StarwaveRollbackReason::AfterReliabilityBoundary {
            boundary_position: boundary_position.unwrap_or(0),
        },
    }
}

fn state_sha256<S: S14StarwaveStateProof>(state: &S) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starwave-state-v1");
    state.write_state_proof(&mut writer);
    writer.finish()
}

fn proof_sha256(scheme: &str, payload: &[u8]) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starwave-proof-v1");
    writer.write_str(scheme);
    writer.write_bytes(payload);
    writer.finish()
}

fn branch_sha256<S: S14StarwaveStateProof>(
    branch_id: u64,
    steps: &[S14StarwaveCandidateStep<S>],
) -> S14StarwaveResult<S14StarwaveSha256> {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starwave-branch-v1");
    writer.write_u64(branch_id);
    writer.write_u64(steps.len() as u64);
    for step in steps {
        step.validate()?;
        writer.write_u64(step.position);
        writer.write_u32(step.token_id);
        writer.write_f32(step.latent_reliability);
        writer.write_sha256(step.state_sha256);
    }
    Ok(writer.finish())
}

fn candidate_batch_sha256<S>(
    request: &S14StarwaveCandidateRequest,
    branches: &[S14StarwaveBranch<S>],
    producer_proof_sha256: S14StarwaveSha256,
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starwave-candidate-batch-v1");
    writer.write_u64(request.batch_id);
    writer.write_u64(request.base_commit_epoch);
    writer.write_u64(request.base_position);
    writer.write_u8(request.positions);
    writer.write_sha256(request.anchor_sha256);
    writer.write_sha256(producer_proof_sha256);
    writer.write_u64(branches.len() as u64);
    for branch in branches {
        writer.write_u64(branch.branch_id);
        writer.write_sha256(branch.branch_sha256);
    }
    writer.finish()
}

fn correction_batch_sha256(
    batch_id: u64,
    candidate_batch_sha256: S14StarwaveSha256,
    deltas: &[S14StarwaveCapabilityDelta],
    authority_proof_sha256: S14StarwaveSha256,
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starwave-correction-batch-v1");
    writer.write_u64(batch_id);
    writer.write_sha256(candidate_batch_sha256);
    writer.write_sha256(authority_proof_sha256);
    writer.write_u64(deltas.len() as u64);
    for delta in deltas {
        writer.write_u64(delta.branch_id);
        writer.write_u64(delta.position);
        writer.write_u32(delta.candidate_token_id);
        writer.write_f32(delta.reliability_delta);
        match delta.verdict {
            S14StarwaveCapabilityVerdict::Reliable => writer.write_u8(1),
            S14StarwaveCapabilityVerdict::Unreliable => writer.write_u8(2),
            S14StarwaveCapabilityVerdict::Conflict {
                authoritative_token_id,
            } => {
                writer.write_u8(3);
                writer.write_u32(authoritative_token_id);
            }
        }
        writer.write_sha256(delta.proof.sha256());
    }
    writer.finish()
}

fn rollback_receipt_sha256(
    batch_id: u64,
    candidate_batch_sha256: S14StarwaveSha256,
    selected_branch_id: Option<u64>,
    retained_positions: u8,
    rolled_back_positions: u64,
    decisions: &[S14StarwaveBranchDecision],
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starwave-rollback-v1");
    writer.write_u64(batch_id);
    writer.write_sha256(candidate_batch_sha256);
    match selected_branch_id {
        Some(branch_id) => {
            writer.write_u8(1);
            writer.write_u64(branch_id);
        }
        None => writer.write_u8(0),
    }
    writer.write_u8(retained_positions);
    writer.write_u64(rolled_back_positions);
    writer.write_u64(decisions.len() as u64);
    for decision in decisions {
        writer.write_u64(decision.branch_id);
        writer.write_u8(decision.reliable_prefix_positions);
        writer.write_u64(decision.corrected_reliability_sum.to_bits());
        match &decision.boundary {
            S14StarwavePrefixBoundary::Complete => writer.write_u8(0),
            S14StarwavePrefixBoundary::MissingDelta { position } => {
                writer.write_u8(1);
                writer.write_u64(*position);
            }
            S14StarwavePrefixBoundary::BelowReliability {
                position,
                corrected_reliability,
            } => {
                writer.write_u8(2);
                writer.write_u64(*position);
                writer.write_f32(*corrected_reliability);
            }
            S14StarwavePrefixBoundary::Unreliable { position } => {
                writer.write_u8(3);
                writer.write_u64(*position);
            }
            S14StarwavePrefixBoundary::FirstConflict {
                position,
                candidate_token_id,
                authoritative_token_id,
            } => {
                writer.write_u8(4);
                writer.write_u64(*position);
                writer.write_u32(*candidate_token_id);
                writer.write_u32(*authoritative_token_id);
            }
        }
    }
    writer.finish()
}

fn committed_prefix_sha256<S>(
    selected_branch_id: Option<u64>,
    steps: &[S14StarwaveCandidateStep<S>],
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starwave-committed-prefix-v1");
    writer.write_u64(selected_branch_id.unwrap_or(0));
    writer.write_u64(steps.len() as u64);
    for step in steps {
        writer.write_u64(step.position);
        writer.write_u32(step.token_id);
        writer.write_f32(step.latent_reliability);
        writer.write_sha256(step.state_sha256);
    }
    writer.finish()
}

fn commit_proof_sha256(
    candidate_batch_sha256: S14StarwaveSha256,
    correction_sha256: S14StarwaveSha256,
    prefix_sha256: S14StarwaveSha256,
    rollback_sha256: S14StarwaveSha256,
    backend_proof_sha256: S14StarwaveSha256,
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starwave-commit-proof-v1");
    writer.write_sha256(candidate_batch_sha256);
    writer.write_sha256(correction_sha256);
    writer.write_sha256(prefix_sha256);
    writer.write_sha256(rollback_sha256);
    writer.write_sha256(backend_proof_sha256);
    writer.finish()
}

#[allow(clippy::too_many_arguments)]
fn commit_receipt_sha256(
    batch_id: u64,
    selected_branch_id: u64,
    start_position: u64,
    committed_positions: u8,
    end_position_exclusive: u64,
    previous_commit_epoch: u64,
    commit_epoch: u64,
    previous_commit_sha256: S14StarwaveSha256,
    proof_sha256: S14StarwaveSha256,
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starwave-commit-receipt-v1");
    writer.write_u32(S14_STARWAVE_RECEIPT_SCHEMA_VERSION);
    writer.write_u64(batch_id);
    writer.write_u64(selected_branch_id);
    writer.write_u64(start_position);
    writer.write_u8(committed_positions);
    writer.write_u64(end_position_exclusive);
    writer.write_u64(previous_commit_epoch);
    writer.write_u64(commit_epoch);
    writer.write_sha256(previous_commit_sha256);
    writer.write_sha256(proof_sha256);
    writer.finish()
}
