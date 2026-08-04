//! StarGraph/StarWave 的 S14 StarFold K4 物理事务门。
//!
//! 本模块不执行模型，也不发布 token。它只接收已经由 production adapter 产生的唯一
//! B4 × FullDepth43 物理产物，把候选状态与当前 StarWave checkpoint 隔离，并在完整
//! 证据闭合后把 backend proof 交给 `S14StarwaveTransaction::finalize`。任何校验失败都
//! 只会 poison/abort 候选；上一条已提交 checkpoint 始终保持不变。
//!
//! 这是“物理 owner 直接 finalize”的单一提交路径，不实现旧的
//! `S14StarGraphPhysicalCommitter`（旧引擎会在 committer 返回 proof 后再次 finalize）。
//! 接线时必须由 StarGraph 选择本 owner 取代旧 finalize，不能把两条路径叠加。

#![forbid(unsafe_code)]

use crate::{
    s14_causal_block_hc_qkv_adapter::S14CausalBlockHcQkvLayerRecordingReceipt,
    s14_causal_block_layer::{S14CausalBlockHiddenBinding, S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE},
    s14_starfold_b4_owner::S14StarfoldB4RoutedLayerReceipt,
    s14_starfold_cache::{STARFOLD_B4_LANES, STARFOLD_TOP_K},
    s14_starfold_expert_schedule::S14StarfoldExpertProjection,
    s14_starfold_k4_adapter::{S14StarfoldK4FullDepthReceipt, S14StarfoldK4HiddenCommitReceipt},
    s14_starfold_routed_executor::S14StarfoldProjectionExecutionReceipt,
    s14_starfold_vulkan_windows::{S14StarfoldCompletedTimelines, S14StarfoldTimelinePoint},
    s14_starwave_transaction::{
        S14StarwavePreparedTransaction, S14StarwaveProof, S14StarwaveProofWriter,
        S14StarwaveSha256, S14StarwaveStateProof, S14StarwaveTransaction,
        S14StarwaveTransactionOutcome,
    },
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk::{self, Handle};
use polaris_s14_runner::{
    GraphProfile, RouteDecision, RouterKind, COMPRESS_RATIOS, FULL_DEPTH_LAYERS,
};

pub const S14_STARFOLD_K4_PHYSICAL_TRANSACTION_SCHEMA_VERSION: u32 = 2;
pub const S14_STARFOLD_PHYSICAL_PROOF_SCHEME: &str =
    "polaris-s14-stargraph-starfold-k4-fulldepth43-physical-v2";

const K4: usize = STARFOLD_B4_LANES;
const BF16_BYTES: u64 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4CommittedCheckpoint {
    pub commit_epoch: u64,
    pub next_position: u64,
    pub commit_sha256: S14StarwaveSha256,
}

impl S14StarfoldK4CommittedCheckpoint {
    fn capture(starwave: &S14StarwaveTransaction) -> Self {
        Self {
            commit_epoch: starwave.commit_epoch(),
            next_position: starwave.next_position(),
            commit_sha256: starwave.last_commit_sha256(),
        }
    }

    fn validate(self) -> Result<()> {
        if self.commit_sha256 == S14StarwaveSha256::ZERO {
            bail!("S14 StarFold K4 committed checkpoint SHA-256 不能为零");
        }
        Ok(())
    }
}

/// 由物理执行边界补充的禁止路径审计。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4PhysicalAudit {
    pub legacy_union_calls: u64,
    pub serial_token_forward_calls: u64,
    pub cpu_fallback_calls: u64,
    pub legacy_model_fallback_calls: u64,
    pub whole_model_fallback_calls: u64,
}

/// 唯一 GPU owner 在等待 fence/timeline 后签发的完成 seal。字段保持私有，普通调用方
/// 不能用几个自填计数冒充物理完成；构造时必须给出同一代真实 submitted/completed 点。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4PhysicalCompletionSeal {
    owner_id: u64,
    production_binding_sha256: S14StarwaveSha256,
    submitted_transfer: S14StarfoldTimelinePoint,
    submitted_compute: S14StarfoldTimelinePoint,
    completed: S14StarfoldCompletedTimelines,
    completed_queue_submit_calls: u64,
    completion_fence_checks: u32,
    seal_sha256: S14StarwaveSha256,
}

impl S14StarfoldK4PhysicalCompletionSeal {
    /// 只能由 crate 内持有真实 Vulkan owner 的 production 边界调用。
    pub(crate) fn seal(
        owner_id: u64,
        production_binding_sha256: S14StarwaveSha256,
        submitted_transfer: S14StarfoldTimelinePoint,
        submitted_compute: S14StarfoldTimelinePoint,
        completed: S14StarfoldCompletedTimelines,
        completed_queue_submit_calls: u64,
        completion_fence_checks: u32,
    ) -> Result<Self> {
        validate_completed_timelines(
            owner_id,
            production_binding_sha256,
            submitted_transfer,
            submitted_compute,
            completed,
            completed_queue_submit_calls,
            completion_fence_checks,
        )?;
        let seal_sha256 = completion_seal_sha256(
            owner_id,
            production_binding_sha256,
            submitted_transfer,
            submitted_compute,
            completed,
            completed_queue_submit_calls,
            completion_fence_checks,
        );
        Ok(Self {
            owner_id,
            production_binding_sha256,
            submitted_transfer,
            submitted_compute,
            completed,
            completed_queue_submit_calls,
            completion_fence_checks,
            seal_sha256,
        })
    }

    pub const fn owner_id(self) -> u64 {
        self.owner_id
    }

    pub const fn production_binding_sha256(self) -> S14StarwaveSha256 {
        self.production_binding_sha256
    }

    pub const fn completed_queue_submit_calls(self) -> u64 {
        self.completed_queue_submit_calls
    }

    fn validate_against(
        self,
        production_binding_sha256: S14StarwaveSha256,
        known_queue_submit_calls: u64,
    ) -> Result<()> {
        validate_completed_timelines(
            self.owner_id,
            self.production_binding_sha256,
            self.submitted_transfer,
            self.submitted_compute,
            self.completed,
            self.completed_queue_submit_calls,
            self.completion_fence_checks,
        )?;
        if self.production_binding_sha256 != production_binding_sha256
            || self.completed_queue_submit_calls < known_queue_submit_calls
            || self.seal_sha256
                != completion_seal_sha256(
                    self.owner_id,
                    self.production_binding_sha256,
                    self.submitted_transfer,
                    self.submitted_compute,
                    self.completed,
                    self.completed_queue_submit_calls,
                    self.completion_fence_checks,
                )
        {
            bail!("S14 StarFold K4 completion seal 未绑定当前 production/submit 集合");
        }
        Ok(())
    }
}

/// 完整物理产物。三个内容 SHA 分别绑定 routed B4 输出、43层 checkpoint chain 与最终
/// device hidden，补足结构回执本身不包含张量内容这一缺口。
#[derive(Clone, Debug)]
pub struct S14StarfoldK4PhysicalProduct {
    /// 必须原样回带 `begin_k4` 返回的 production binding，禁止复用旧 candidate 产物。
    pub production_binding_sha256: S14StarwaveSha256,
    pub full_depth: S14StarfoldK4FullDepthReceipt,
    pub routed_b4_output_sha256: S14StarwaveSha256,
    pub checkpoint_chain_sha256: S14StarwaveSha256,
    pub final_hidden_sha256: S14StarwaveSha256,
    pub completion: S14StarfoldK4PhysicalCompletionSeal,
    pub audit: S14StarfoldK4PhysicalAudit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldK4PhysicalTransactionPhase {
    Idle,
    Collecting,
    EvidenceSealed,
    Poisoned,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4PhysicalBeginReceipt {
    pub transaction_id: u64,
    pub batch_id: u64,
    pub selected_branch_id: u64,
    pub base_checkpoint: S14StarfoldK4CommittedCheckpoint,
    pub reliable_prefix_positions: u8,
    pub prepared_prefix_sha256: S14StarwaveSha256,
    pub production_binding_sha256: S14StarwaveSha256,
}

#[derive(Debug)]
pub struct S14StarfoldK4FinalizedTransaction<S> {
    pub transaction_id: u64,
    pub batch_id: u64,
    pub reliable_prefix_positions: u8,
    pub previous_checkpoint: S14StarfoldK4CommittedCheckpoint,
    pub committed_checkpoint: S14StarfoldK4CommittedCheckpoint,
    pub physical_evidence_sha256: S14StarwaveSha256,
    pub backend_proof: S14StarwaveProof,
    pub physical_product: S14StarfoldK4PhysicalProduct,
    pub starwave: S14StarwaveTransactionOutcome<S>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4PhysicalAbortReceipt {
    pub transaction_id: u64,
    pub batch_id: u64,
    pub reason: String,
    pub was_poisoned: bool,
    pub had_sealed_evidence: bool,
    pub discarded_physical_evidence_sha256: Option<S14StarwaveSha256>,
    pub previous_checkpoint: S14StarfoldK4CommittedCheckpoint,
    pub current_checkpoint: S14StarfoldK4CommittedCheckpoint,
    pub abort_sha256: S14StarwaveSha256,
}

impl S14StarfoldK4PhysicalAbortReceipt {
    pub fn validate(&self) -> Result<()> {
        if self.reason.is_empty()
            || self.previous_checkpoint != self.current_checkpoint
            || self.had_sealed_evidence != self.discarded_physical_evidence_sha256.is_some()
        {
            bail!("S14 StarFold K4 abort 未证明 checkpoint 前后完全相同");
        }
        let expected = abort_receipt_sha256(
            self.transaction_id,
            self.batch_id,
            &self.reason,
            self.was_poisoned,
            self.had_sealed_evidence,
            self.discarded_physical_evidence_sha256,
            self.previous_checkpoint,
        );
        if expected != self.abort_sha256 {
            bail!("S14 StarFold K4 abort receipt SHA-256 漂移");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateStatus {
    Collecting,
    EvidenceSealed,
    Poisoned,
}

#[derive(Debug)]
struct ActiveCandidate {
    transaction_id: u64,
    batch_id: u64,
    selected_branch_id: u64,
    anchor: S14StarfoldK4CommittedCheckpoint,
    reliable_prefix_positions: u8,
    prepared_prefix_sha256: S14StarwaveSha256,
    production_binding_sha256: S14StarwaveSha256,
    status: CandidateStatus,
    poison_reason: Option<String>,
    physical_product: Option<Box<S14StarfoldK4PhysicalProduct>>,
    physical_evidence_sha256: Option<S14StarwaveSha256>,
}

/// 单 owner 的事务状态机。唯一可改变 committed checkpoint 的入口是 `finalize_k4`，
/// 且该改变发生在 StarWave finalize 成功之后。
#[derive(Debug)]
pub struct S14StarfoldK4PhysicalTransactionOwner {
    committed: S14StarfoldK4CommittedCheckpoint,
    active: Option<ActiveCandidate>,
}

impl S14StarfoldK4PhysicalTransactionOwner {
    pub fn new(starwave: &S14StarwaveTransaction) -> Result<Self> {
        let committed = S14StarfoldK4CommittedCheckpoint::capture(starwave);
        committed.validate()?;
        Ok(Self {
            committed,
            active: None,
        })
    }

    pub const fn committed_checkpoint(&self) -> S14StarfoldK4CommittedCheckpoint {
        self.committed
    }

    pub fn phase(&self) -> S14StarfoldK4PhysicalTransactionPhase {
        match self.active.as_ref().map(|active| active.status) {
            None => S14StarfoldK4PhysicalTransactionPhase::Idle,
            Some(CandidateStatus::Collecting) => S14StarfoldK4PhysicalTransactionPhase::Collecting,
            Some(CandidateStatus::EvidenceSealed) => {
                S14StarfoldK4PhysicalTransactionPhase::EvidenceSealed
            }
            Some(CandidateStatus::Poisoned) => S14StarfoldK4PhysicalTransactionPhase::Poisoned,
        }
    }

    /// 把当前 prepared reliable prefix 锁到 committed checkpoint。StarFold 固定计算四个
    /// future，但 StarWave 仍可只提交其中 1..=4 个最长可靠前缀。
    pub fn begin_k4<S: S14StarwaveStateProof>(
        &mut self,
        transaction_id: u64,
        starwave: &S14StarwaveTransaction,
        prepared: &S14StarwavePreparedTransaction<S>,
    ) -> Result<S14StarfoldK4PhysicalBeginReceipt> {
        if self.active.is_some() {
            bail!("S14 StarFold K4 已有 active physical transaction");
        }
        if transaction_id == 0 {
            bail!("S14 StarFold K4 transaction id 不能为0");
        }
        let observed = S14StarfoldK4CommittedCheckpoint::capture(starwave);
        if observed != self.committed {
            bail!("S14 StarFold K4 owner 与 StarWave committed checkpoint 漂移");
        }
        validate_prepared_anchor(prepared, self.committed)?;
        let reliable_prefix_positions = u8::try_from(prepared.committed_prefix().len())
            .context("S14 StarFold K4 reliable prefix 长度超出 u8")?;
        if reliable_prefix_positions == 0 || usize::from(reliable_prefix_positions) > K4 {
            bail!("S14 StarFold K4 physical transaction 只接受 1..=4 个可靠位置");
        }
        let selected_branch_id = prepared
            .selected_branch_id()
            .context("S14 StarFold K4 非空 reliable prefix 缺少 selected branch")?;
        let prepared_prefix_sha256 = prepared_prefix_sha256(prepared)?;
        let batch_id = prepared.batch_id();
        let production_binding_sha256 = production_binding_sha256(
            transaction_id,
            batch_id,
            selected_branch_id,
            self.committed,
            reliable_prefix_positions,
            prepared_prefix_sha256,
        );
        self.active = Some(ActiveCandidate {
            transaction_id,
            batch_id,
            selected_branch_id,
            anchor: self.committed,
            reliable_prefix_positions,
            prepared_prefix_sha256,
            production_binding_sha256,
            status: CandidateStatus::Collecting,
            poison_reason: None,
            physical_product: None,
            physical_evidence_sha256: None,
        });
        Ok(S14StarfoldK4PhysicalBeginReceipt {
            transaction_id,
            batch_id,
            selected_branch_id,
            base_checkpoint: self.committed,
            reliable_prefix_positions,
            prepared_prefix_sha256,
            production_binding_sha256,
        })
    }

    /// 接收唯一 B4 的完整43层物理产物。任何缺层、跳层、重复层、内容 digest 缺失或
    /// 禁止路径计数非零都会 poison 候选；不会修改 committed checkpoint。
    pub fn record_full_depth_product(
        &mut self,
        product: S14StarfoldK4PhysicalProduct,
    ) -> Result<S14StarwaveSha256> {
        let active = self
            .active
            .as_ref()
            .context("S14 StarFold K4 当前没有 active physical transaction")?;
        if active.status == CandidateStatus::Poisoned {
            bail!(
                "S14 StarFold K4 candidate 已 poisoned，只能 abort: {}",
                active.poison_reason.as_deref().unwrap_or("unknown")
            );
        }
        if active.status != CandidateStatus::Collecting || active.physical_product.is_some() {
            let error = anyhow!("S14 StarFold K4 full-depth physical product 重复发布");
            self.poison(error.to_string());
            return Err(error);
        }
        let binding = CandidateBinding::from(active);
        let evidence_sha256 = match validate_and_hash_physical_product(&binding, &product) {
            Ok(digest) => digest,
            Err(error) => {
                self.poison(error.to_string());
                return Err(error);
            }
        };
        let active = self.active.as_mut().expect("active checked");
        active.physical_product = Some(Box::new(product));
        active.physical_evidence_sha256 = Some(evidence_sha256);
        active.status = CandidateStatus::EvidenceSealed;
        Ok(evidence_sha256)
    }

    /// 先生成 backend proof，再调用既有 StarWave finalize。只有 StarWave 成功推进 epoch
    /// 后，本 owner 才替换自己的 committed checkpoint；此前的任意错误都只 poison 候选。
    pub fn finalize_k4<S: S14StarwaveStateProof>(
        &mut self,
        starwave: &mut S14StarwaveTransaction,
        prepared: S14StarwavePreparedTransaction<S>,
    ) -> Result<S14StarfoldK4FinalizedTransaction<S>> {
        let active = self
            .active
            .as_ref()
            .context("S14 StarFold K4 当前没有 active physical transaction")?;
        if active.status == CandidateStatus::Poisoned {
            bail!(
                "S14 StarFold K4 candidate 已 poisoned，只能 abort: {}",
                active.poison_reason.as_deref().unwrap_or("unknown")
            );
        }
        if active.status != CandidateStatus::EvidenceSealed
            || active.physical_product.is_none()
            || active.physical_evidence_sha256.is_none()
        {
            bail!("S14 StarFold K4 禁止 finalize 未闭合的 B4 × FullDepth43 证据");
        }
        if let Err(error) = validate_finalize_binding(active, starwave, &prepared, self.committed) {
            self.poison(error.to_string());
            return Err(error);
        }

        let physical_evidence_sha256 = active
            .physical_evidence_sha256
            .expect("sealed digest checked");
        let backend_proof = match S14StarwaveProof::new(
            S14_STARFOLD_PHYSICAL_PROOF_SCHEME,
            physical_evidence_sha256.as_bytes().to_vec(),
        ) {
            Ok(proof) => proof,
            Err(error) => {
                self.poison(error.to_string());
                return Err(anyhow::Error::new(error)
                    .context("构造 S14 StarFold K4 StarWave backend proof"));
            }
        };

        let mut active = self.active.take().expect("active checked");
        let product = active
            .physical_product
            .take()
            .expect("sealed product checked");
        let starwave_outcome = match starwave.finalize(prepared, Some(backend_proof.clone())) {
            Ok(outcome) => outcome,
            Err(error) => {
                active.status = CandidateStatus::Poisoned;
                active.poison_reason = Some(error.to_string());
                active.physical_product = Some(product);
                self.active = Some(active);
                return Err(anyhow::Error::new(error).context("S14 StarFold K4 StarWave finalize"));
            }
        };

        let previous_checkpoint = self.committed;
        let committed_checkpoint = S14StarfoldK4CommittedCheckpoint::capture(starwave);
        // `S14StarwaveTransaction::finalize` 对非空 prefix 保证 Some(commit_receipt)，并且
        // 在返回 Ok 前已经完成所有可失败校验；这里不再增加可能造成“双状态”的后置错误。
        debug_assert_eq!(
            committed_checkpoint.commit_epoch,
            previous_checkpoint.commit_epoch + 1
        );
        debug_assert_eq!(
            committed_checkpoint.next_position,
            previous_checkpoint.next_position + u64::from(active.reliable_prefix_positions)
        );
        self.committed = committed_checkpoint;
        Ok(S14StarfoldK4FinalizedTransaction {
            transaction_id: active.transaction_id,
            batch_id: active.batch_id,
            reliable_prefix_positions: active.reliable_prefix_positions,
            previous_checkpoint,
            committed_checkpoint,
            physical_evidence_sha256,
            backend_proof,
            physical_product: *product,
            starwave: starwave_outcome,
        })
    }

    /// 丢弃 collecting、sealed 或 poisoned candidate。该方法没有 StarWave 可变引用，
    /// 因而在类型边界上就不能推进 epoch/position；abort receipt 再证明 checkpoint 前后相同。
    pub fn abort_k4(
        &mut self,
        reason: impl Into<String>,
    ) -> Result<S14StarfoldK4PhysicalAbortReceipt> {
        let reason = reason.into();
        if reason.is_empty() {
            bail!("S14 StarFold K4 abort reason 不能为空");
        }
        let active = self
            .active
            .take()
            .context("S14 StarFold K4 当前没有可 abort 的 physical transaction")?;
        let previous_checkpoint = self.committed;
        let current_checkpoint = self.committed;
        let was_poisoned = active.status == CandidateStatus::Poisoned;
        let had_sealed_evidence = active.physical_product.is_some();
        let abort_sha256 = abort_receipt_sha256(
            active.transaction_id,
            active.batch_id,
            &reason,
            was_poisoned,
            had_sealed_evidence,
            active.physical_evidence_sha256,
            previous_checkpoint,
        );
        let receipt = S14StarfoldK4PhysicalAbortReceipt {
            transaction_id: active.transaction_id,
            batch_id: active.batch_id,
            reason,
            was_poisoned,
            had_sealed_evidence,
            discarded_physical_evidence_sha256: active.physical_evidence_sha256,
            previous_checkpoint,
            current_checkpoint,
            abort_sha256,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    fn poison(&mut self, reason: String) {
        if let Some(active) = self.active.as_mut() {
            active.status = CandidateStatus::Poisoned;
            active.poison_reason = Some(reason);
        }
    }
}

#[derive(Clone, Copy)]
struct CandidateBinding {
    transaction_id: u64,
    batch_id: u64,
    selected_branch_id: u64,
    anchor: S14StarfoldK4CommittedCheckpoint,
    reliable_prefix_positions: u8,
    prepared_prefix_sha256: S14StarwaveSha256,
    production_binding_sha256: S14StarwaveSha256,
}

impl From<&ActiveCandidate> for CandidateBinding {
    fn from(active: &ActiveCandidate) -> Self {
        Self {
            transaction_id: active.transaction_id,
            batch_id: active.batch_id,
            selected_branch_id: active.selected_branch_id,
            anchor: active.anchor,
            reliable_prefix_positions: active.reliable_prefix_positions,
            prepared_prefix_sha256: active.prepared_prefix_sha256,
            production_binding_sha256: active.production_binding_sha256,
        }
    }
}

fn validate_prepared_anchor<S: S14StarwaveStateProof>(
    prepared: &S14StarwavePreparedTransaction<S>,
    committed: S14StarfoldK4CommittedCheckpoint,
) -> Result<()> {
    if prepared.base_commit_epoch() != committed.commit_epoch
        || prepared.base_position() != committed.next_position
        || !prepared.will_commit()
    {
        bail!("S14 StarFold K4 prepared transaction 未绑定当前 committed checkpoint");
    }
    let base_position = u32::try_from(prepared.base_position())
        .context("S14 StarFold K4 base position 超出 production u32 ABI")?;
    if base_position == 0 {
        bail!("S14 StarFold K4 production base position 不能为0");
    }
    u64::from(base_position)
        .checked_add((K4 - 1) as u64)
        .context("S14 StarFold K4 physical block position overflow")?;
    for (offset, step) in prepared.committed_prefix().iter().enumerate() {
        let expected = prepared
            .base_position()
            .checked_add(offset as u64)
            .context("S14 StarFold K4 reliable prefix position overflow")?;
        if step.position() != expected {
            bail!("S14 StarFold K4 prepared reliable prefix position 不连续");
        }
    }
    Ok(())
}

fn validate_finalize_binding<S: S14StarwaveStateProof>(
    active: &ActiveCandidate,
    starwave: &S14StarwaveTransaction,
    prepared: &S14StarwavePreparedTransaction<S>,
    committed: S14StarfoldK4CommittedCheckpoint,
) -> Result<()> {
    if committed != active.anchor
        || S14StarfoldK4CommittedCheckpoint::capture(starwave) != active.anchor
    {
        bail!("S14 StarFold K4 finalize anchor 已过期");
    }
    validate_prepared_anchor(prepared, committed)?;
    let prefix_positions = u8::try_from(prepared.committed_prefix().len())
        .context("S14 StarFold K4 finalize prefix 长度超出 u8")?;
    if prepared.batch_id() != active.batch_id
        || prepared.selected_branch_id() != Some(active.selected_branch_id)
        || prefix_positions != active.reliable_prefix_positions
        || prepared_prefix_sha256(prepared)? != active.prepared_prefix_sha256
    {
        bail!("S14 StarFold K4 finalize prepared transaction 与 begin binding 漂移");
    }
    Ok(())
}

fn prepared_prefix_sha256<S: S14StarwaveStateProof>(
    prepared: &S14StarwavePreparedTransaction<S>,
) -> Result<S14StarwaveSha256> {
    let selected_branch_id = prepared
        .selected_branch_id()
        .context("S14 StarFold K4 prepared prefix 缺少 selected branch")?;
    let mut writer =
        S14StarwaveProofWriter::new("polaris-s14-starfold-k4-prepared-prefix-binding-v1");
    writer.write_u64(prepared.batch_id());
    writer.write_u64(prepared.base_commit_epoch());
    writer.write_u64(prepared.base_position());
    writer.write_u64(selected_branch_id);
    writer.write_u64(prepared.committed_prefix().len() as u64);
    for step in prepared.committed_prefix() {
        writer.write_u64(step.position());
        writer.write_u32(step.token_id());
        writer.write_sha256(step.state_sha256());
    }
    Ok(writer.finish())
}

#[allow(clippy::too_many_arguments)]
fn production_binding_sha256(
    transaction_id: u64,
    batch_id: u64,
    selected_branch_id: u64,
    anchor: S14StarfoldK4CommittedCheckpoint,
    reliable_prefix_positions: u8,
    prepared_prefix_sha256: S14StarwaveSha256,
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starfold-k4-production-binding-v2");
    writer.write_u32(S14_STARFOLD_K4_PHYSICAL_TRANSACTION_SCHEMA_VERSION);
    writer.write_u64(transaction_id);
    writer.write_u64(batch_id);
    writer.write_u64(selected_branch_id);
    writer.write_u64(anchor.commit_epoch);
    writer.write_u64(anchor.next_position);
    writer.write_sha256(anchor.commit_sha256);
    writer.write_u8(reliable_prefix_positions);
    writer.write_sha256(prepared_prefix_sha256);
    writer.finish()
}

#[allow(clippy::too_many_arguments)]
fn validate_completed_timelines(
    owner_id: u64,
    production_binding_sha256: S14StarwaveSha256,
    submitted_transfer: S14StarfoldTimelinePoint,
    submitted_compute: S14StarfoldTimelinePoint,
    completed: S14StarfoldCompletedTimelines,
    completed_queue_submit_calls: u64,
    completion_fence_checks: u32,
) -> Result<()> {
    if owner_id == 0
        || production_binding_sha256 == S14StarwaveSha256::ZERO
        || submitted_transfer.semaphore == vk::Semaphore::null()
        || submitted_compute.semaphore == vk::Semaphore::null()
        || submitted_transfer.semaphore == submitted_compute.semaphore
        || submitted_transfer.generation == 0
        || submitted_transfer.generation != submitted_compute.generation
        || submitted_transfer.generation != completed.generation
        || submitted_transfer.value == 0
        || submitted_compute.value == 0
        || completed.transfer < submitted_transfer.value
        || completed.compute < submitted_compute.value
        || completed_queue_submit_calls == 0
        || completion_fence_checks == 0
    {
        bail!("S14 StarFold K4 completion seal 没有证明同一代 submitted timelines 已完成");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn completion_seal_sha256(
    owner_id: u64,
    production_binding_sha256: S14StarwaveSha256,
    submitted_transfer: S14StarfoldTimelinePoint,
    submitted_compute: S14StarfoldTimelinePoint,
    completed: S14StarfoldCompletedTimelines,
    completed_queue_submit_calls: u64,
    completion_fence_checks: u32,
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starfold-k4-completion-seal-v1");
    writer.write_u64(owner_id);
    writer.write_sha256(production_binding_sha256);
    write_timeline_point(&mut writer, submitted_transfer);
    write_timeline_point(&mut writer, submitted_compute);
    writer.write_u64(completed.generation);
    writer.write_u64(completed.transfer);
    writer.write_u64(completed.compute);
    writer.write_u64(completed_queue_submit_calls);
    writer.write_u32(completion_fence_checks);
    writer.finish()
}

fn validate_and_hash_physical_product(
    binding: &CandidateBinding,
    product: &S14StarfoldK4PhysicalProduct,
) -> Result<S14StarwaveSha256> {
    let known_queue_submit_calls = validate_full_depth_receipt(binding, &product.full_depth)?;
    if product.production_binding_sha256 != binding.production_binding_sha256 {
        bail!("S14 StarFold K4 物理产物未绑定当前 prepared prefix/anchor");
    }
    if product.routed_b4_output_sha256 == S14StarwaveSha256::ZERO
        || product.checkpoint_chain_sha256 == S14StarwaveSha256::ZERO
        || product.final_hidden_sha256 == S14StarwaveSha256::ZERO
    {
        bail!("S14 StarFold K4 物理产物缺少非零内容 SHA-256");
    }
    product
        .completion
        .validate_against(binding.production_binding_sha256, known_queue_submit_calls)?;
    if product.audit.legacy_union_calls != 0
        || product.audit.serial_token_forward_calls != 0
        || product.audit.cpu_fallback_calls != 0
        || product.audit.legacy_model_fallback_calls != 0
        || product.audit.whole_model_fallback_calls != 0
    {
        bail!("S14 StarFold K4 物理产物命中旧 union/串行 token/CPU/整模 fallback");
    }
    Ok(physical_evidence_sha256(binding, product))
}

fn validate_full_depth_receipt(
    binding: &CandidateBinding,
    receipt: &S14StarfoldK4FullDepthReceipt,
) -> Result<u64> {
    let expected_base = u32::try_from(binding.anchor.next_position)
        .context("S14 StarFold K4 anchor position 超出 u32")?;
    if receipt.base_position != expected_base
        || receipt.block_size != K4
        || receipt.completed_layers != FULL_DEPTH_LAYERS.len()
        || receipt.layers.len() != FULL_DEPTH_LAYERS.len()
        || receipt.routes_by_position.len() != K4
        || receipt.serial_token_forward_calls != 0
        || !receipt.terminal_ready
        || receipt.token_committed
    {
        bail!("S14 StarFold K4 full-depth receipt 未形成未发布的 K4 × FullDepth43 future");
    }
    if receipt.checkpoint_seal.base_position != expected_base
        || receipt.checkpoint_seal.block_size != K4
        || receipt.checkpoint_seal.completed_layers != FULL_DEPTH_LAYERS.len()
        || receipt.checkpoint_seal.checkpoint_count != K4
        || receipt.checkpoint_seal.prefix_program_seal_calls != 1
        || receipt.checkpoint_seal.checkpoint_commit_calls != 0
        || receipt.checkpoint_seal.serial_token_forward_calls != 0
    {
        bail!("S14 StarFold K4 checkpoint seal 在 StarWave finalize 前非法或不完整");
    }

    let mut packed_uploads = 0u32;
    let mut packed_upload_bytes = 0u64;
    let mut lane_dispatches = 0u32;
    let mut known_queue_submit_calls = 0u64;
    let mut previous_hidden: Option<S14CausalBlockHiddenBinding> = None;
    for (layer_index, (&expected_layer, layer)) in
        FULL_DEPTH_LAYERS.iter().zip(&receipt.layers).enumerate()
    {
        if layer.layer != expected_layer
            || layer.base_position != expected_base
            || layer.routes.len() != K4
        {
            bail!("S14 StarFold K4 L{expected_layer} layer/base/B4 identity 漂移");
        }
        for route in &layer.routes {
            route
                .validate_for(GraphProfile::FullDepth43NativeTop6)
                .map_err(anyhow::Error::new)
                .with_context(|| format!("校验 S14 StarFold K4 L{expected_layer} top-6 route"))?;
            if route.layer != expected_layer {
                bail!("S14 StarFold K4 L{expected_layer} route layer 漂移");
            }
        }
        validate_hc_qkv_layer(
            expected_base,
            expected_layer,
            layer.checkpoint,
            previous_hidden,
        )?;
        let layer_submits =
            validate_b4_layer(expected_base, expected_layer, &layer.routes, &layer.expert)?;
        validate_hidden_commit(
            expected_base,
            expected_layer,
            layer.hidden_commit,
            layer.checkpoint,
        )?;
        previous_hidden = Some(layer.hidden_commit.next_hidden);
        known_queue_submit_calls = known_queue_submit_calls
            .checked_add(layer_submits)
            .and_then(|value| {
                value.checked_add(u64::from(layer.checkpoint.command_graph_submit_calls))
            })
            .and_then(|value| value.checked_add(u64::from(layer.hidden_commit.queue_submit_calls)))
            .context("S14 StarFold K4 known queue submit count overflow")?;
        packed_uploads = packed_uploads
            .checked_add(layer.expert.packed_uploads)
            .context("S14 StarFold K4 packed upload count overflow")?;
        packed_upload_bytes = packed_upload_bytes
            .checked_add(layer.expert.packed_upload_bytes)
            .context("S14 StarFold K4 packed upload bytes overflow")?;
        lane_dispatches = lane_dispatches
            .checked_add(layer.expert.lane_dispatches)
            .context("S14 StarFold K4 lane dispatch count overflow")?;

        for position in 0..K4 {
            let position_routes = &receipt.routes_by_position[position];
            if position_routes.len() != FULL_DEPTH_LAYERS.len()
                || position_routes[layer_index] != layer.routes[position]
            {
                bail!("S14 StarFold K4 position-major route chain 与 layer receipt 漂移");
            }
        }
    }
    if receipt.packed_uploads == 0
        || receipt.packed_upload_bytes == 0
        || receipt.lane_dispatches == 0
        || receipt.packed_uploads != packed_uploads
        || receipt.packed_upload_bytes != packed_upload_bytes
        || receipt.lane_dispatches != lane_dispatches
        || Some(receipt.final_hidden) != previous_hidden
    {
        bail!("S14 StarFold K4 full-depth aggregate/最终 hidden 漂移");
    }
    validate_hidden_binding(receipt.final_hidden)?;
    Ok(known_queue_submit_calls)
}

fn validate_hc_qkv_layer(
    base_position: u32,
    layer: u8,
    receipt: S14CausalBlockHcQkvLayerRecordingReceipt,
    previous_hidden: Option<S14CausalBlockHiddenBinding>,
) -> Result<()> {
    let ratio = *COMPRESS_RATIOS
        .get(usize::from(layer))
        .context("S14 StarFold K4 HC/QKV ratio layer 越界")?;
    validate_hidden_binding(receipt.input_hidden)?;
    validate_hidden_binding(receipt.post_attention_hidden)?;
    // ratio4 的边界 lane 由 `base_position mod 4` 决定；任意连续 base 都必须走
    // ratio4 recorder，而不是只把早期 base1/base5 当作特殊值。这里是证据审计层，
    // 必须与动态 phase 执行层保持同一合同，否则真实 base2/base3/base0 会在完成后
    // 被旧回执规则误判为 contiguous fallback。
    let ratio4_boundary = ratio == 4;
    let expected_contiguous_dispatches = u32::from(!ratio4_boundary);
    // ratio4 recorder 使用一次动态 K4 boundary dispatch；base 只改变 boundary lane，
    // 不把同一 shader 拆成两个虚构的物理提交。
    let expected_ratio4_dispatches = u32::from(ratio4_boundary);
    let expected_ratio4_transition = u32::from(ratio4_boundary);
    let expected_post_generation = receipt
        .input_hidden
        .generation
        .checked_add(1)
        .context("S14 StarFold K4 HC/QKV generation overflow")?;
    if receipt.base_position != base_position
        || receipt.block_size != K4
        || receipt.layer != layer
        || previous_hidden.is_some_and(|hidden| hidden != receipt.input_hidden)
        || receipt.post_attention_hidden.generation != expected_post_generation
        || (receipt.post_attention_hidden.buffer == receipt.input_hidden.buffer
            && hidden_ranges_overlap(receipt.post_attention_hidden, receipt.input_hidden)?)
        || receipt.layer_record_calls != 1
        || receipt.command_graph_submit_calls != 1
        || receipt.hc_qkv_projection_record_calls != 1
        || receipt.attention_recording_calls != 1
        || receipt.contiguous_attention_dispatch_calls != expected_contiguous_dispatches
        || receipt.ratio4_boundary_attention_dispatch_calls != expected_ratio4_dispatches
        || receipt.ratio4_state_transition_record_calls != expected_ratio4_transition
        || receipt.attention_output_post_record_calls != 1
        || receipt.ffn_hc_router_input_record_calls != 1
        || receipt.router_recording_calls != 1
        || receipt.serial_token_forward_calls != 0
        || !receipt.hc_hidden_integration_complete
    {
        bail!("S14 StarFold K4 L{layer} HC/QKV/attention/router 回执不完整");
    }
    Ok(())
}

fn validate_b4_layer(
    base_position: u32,
    layer: u8,
    routes: &[RouteDecision],
    receipt: &S14StarfoldB4RoutedLayerReceipt,
) -> Result<u64> {
    let unique_experts = routes
        .iter()
        .flat_map(|route| route.expert_ids.iter().copied())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if receipt.layer != u16::from(layer)
        || receipt.base_position != u64::from(base_position)
        || receipt.unique_experts as usize != unique_experts
        || receipt.serial_token_forward_calls != 0
    {
        bail!("S14 StarFold K4 L{layer} B4 receipt identity 漂移");
    }
    validate_projection(receipt, receipt.w1, S14StarfoldExpertProjection::W1)?;
    validate_projection(receipt, receipt.w3, S14StarfoldExpertProjection::W3)?;
    validate_projection(receipt, receipt.w2, S14StarfoldExpertProjection::W2)?;
    if receipt.prepare.layer != receipt.layer
        || receipt.prepare.base_position != receipt.base_position
        || receipt.prepare.exact_route_weights != (K4 * STARFOLD_TOP_K) as u32
        || receipt.prepare.update_buffer_calls != 1
        || receipt.prepare.prepare_dispatch_calls != 1
        || receipt.prepare.queue_submit_calls != 1
        || receipt.prepare.serial_token_forward_calls != 0
    {
        bail!("S14 StarFold K4 L{layer} batched prepare 回执漂移");
    }
    let packed_uploads = receipt
        .w1
        .packed_uploads
        .checked_add(receipt.w3.packed_uploads)
        .and_then(|value| value.checked_add(receipt.w2.packed_uploads))
        .context("S14 StarFold K4 layer packed upload count overflow")?;
    let packed_upload_bytes = receipt
        .w1
        .packed_upload_bytes
        .checked_add(receipt.w3.packed_upload_bytes)
        .and_then(|value| value.checked_add(receipt.w2.packed_upload_bytes))
        .context("S14 StarFold K4 layer packed upload bytes overflow")?;
    let lane_dispatches = receipt
        .w1
        .lane_dispatches
        .checked_add(receipt.w3.lane_dispatches)
        .and_then(|value| value.checked_add(receipt.w2.lane_dispatches))
        .context("S14 StarFold K4 layer lane dispatch count overflow")?;
    if receipt.packed_uploads != packed_uploads
        || receipt.packed_upload_bytes != packed_upload_bytes
        || receipt.lane_dispatches != lane_dispatches
    {
        bail!("S14 StarFold K4 L{layer} projection aggregate 漂移");
    }
    Ok(u64::from(receipt.w1.queue_submit_calls)
        + u64::from(receipt.w3.queue_submit_calls)
        + u64::from(receipt.prepare.queue_submit_calls)
        + u64::from(receipt.w2.queue_submit_calls))
}

fn validate_projection(
    layer: &S14StarfoldB4RoutedLayerReceipt,
    receipt: S14StarfoldProjectionExecutionReceipt,
    projection: S14StarfoldExpertProjection,
) -> Result<()> {
    if receipt.layer != layer.layer
        || receipt.base_position != layer.base_position
        || receipt.projection != projection
        || receipt.unique_experts != layer.unique_experts
        || receipt.packed_uploads == 0
        || receipt.packed_upload_bytes == 0
        || receipt.lane_dispatches == 0
        || receipt.queue_submit_calls != receipt.packed_uploads
        || (projection != S14StarfoldExpertProjection::W3
            && receipt.source_projection_fallbacks != 0)
        || receipt.source_projection_fallbacks > receipt.unique_experts
        || receipt.serial_token_forward_calls != 0
    {
        bail!(
            "S14 StarFold K4 {:?} projection physical receipt 漂移",
            projection
        );
    }
    Ok(())
}

fn validate_hidden_commit(
    base_position: u32,
    layer: u8,
    receipt: S14StarfoldK4HiddenCommitReceipt,
    hc_qkv: S14CausalBlockHcQkvLayerRecordingReceipt,
) -> Result<()> {
    validate_hidden_binding(receipt.next_hidden)?;
    if receipt.base_position != base_position
        || receipt.layer != layer
        || receipt.routed_reduce_dispatch_calls != 1
        || receipt.hc_post_dispatch_calls != 1
        || receipt.queue_submit_calls != 1
        || receipt.serial_token_forward_calls != 0
    {
        bail!("S14 StarFold K4 L{layer} hidden commit 回执漂移");
    }
    let expected_generation = hc_qkv
        .input_hidden
        .generation
        .checked_add(2)
        .context("S14 StarFold K4 hidden generation overflow")?;
    if receipt.next_hidden.generation != expected_generation
        || (receipt.next_hidden.buffer == hc_qkv.post_attention_hidden.buffer
            && hidden_ranges_overlap(receipt.next_hidden, hc_qkv.post_attention_hidden)?)
    {
        bail!("S14 StarFold K4 L{layer} hidden chain 未按 attention/experts 两阶段推进");
    }
    Ok(())
}

fn hidden_ranges_overlap(
    left: S14CausalBlockHiddenBinding,
    right: S14CausalBlockHiddenBinding,
) -> Result<bool> {
    if left.buffer != right.buffer {
        return Ok(false);
    }
    let left_end = left
        .offset
        .checked_add(left.bytes)
        .context("S14 StarFold K4 hidden left range overflow")?;
    let right_end = right
        .offset
        .checked_add(right.bytes)
        .context("S14 StarFold K4 hidden right range overflow")?;
    Ok(left.offset < right_end && right.offset < left_end)
}

fn validate_hidden_binding(binding: S14CausalBlockHiddenBinding) -> Result<()> {
    let expected_bytes = (K4 as u64)
        .checked_mul(S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64)
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .context("S14 StarFold K4 hidden bytes overflow")?;
    if binding.buffer == vk::Buffer::null()
        || binding.offset % 4 != 0
        || binding.bytes != expected_bytes
        || binding.block_size != K4
    {
        bail!("S14 StarFold K4 hidden binding 不是精确 device [4,4,4096] BF16");
    }
    Ok(())
}

fn physical_evidence_sha256(
    binding: &CandidateBinding,
    product: &S14StarfoldK4PhysicalProduct,
) -> S14StarwaveSha256 {
    let receipt = &product.full_depth;
    let mut writer =
        S14StarwaveProofWriter::new("polaris-s14-stargraph-starfold-k4-physical-evidence-v2");
    writer.write_u32(S14_STARFOLD_K4_PHYSICAL_TRANSACTION_SCHEMA_VERSION);
    writer.write_u64(binding.transaction_id);
    writer.write_u64(binding.batch_id);
    writer.write_u64(binding.selected_branch_id);
    writer.write_u64(binding.anchor.commit_epoch);
    writer.write_u64(binding.anchor.next_position);
    writer.write_sha256(binding.anchor.commit_sha256);
    writer.write_u8(binding.reliable_prefix_positions);
    writer.write_sha256(binding.prepared_prefix_sha256);
    writer.write_sha256(binding.production_binding_sha256);
    writer.write_sha256(product.production_binding_sha256);
    writer.write_u32(receipt.base_position);
    writer.write_u64(receipt.block_size as u64);
    writer.write_u64(receipt.completed_layers as u64);
    writer.write_u64(receipt.layers.len() as u64);
    for layer in &receipt.layers {
        writer.write_u8(layer.layer);
        writer.write_u32(layer.base_position);
        writer.write_u64(layer.routes.len() as u64);
        for route in &layer.routes {
            write_route(&mut writer, route);
        }
        write_hc_qkv_layer(&mut writer, layer.checkpoint);
        write_b4_layer(&mut writer, &layer.expert);
        write_hidden_commit(&mut writer, layer.hidden_commit);
    }
    write_hidden_binding(&mut writer, receipt.final_hidden);
    writer.write_u32(receipt.checkpoint_seal.base_position);
    writer.write_u64(receipt.checkpoint_seal.block_size as u64);
    writer.write_u64(receipt.checkpoint_seal.completed_layers as u64);
    writer.write_u64(receipt.checkpoint_seal.checkpoint_count as u64);
    writer.write_u32(receipt.checkpoint_seal.prefix_program_seal_calls);
    writer.write_u32(receipt.checkpoint_seal.checkpoint_commit_calls);
    writer.write_u32(receipt.checkpoint_seal.serial_token_forward_calls);
    writer.write_u32(receipt.packed_uploads);
    writer.write_u64(receipt.packed_upload_bytes);
    writer.write_u32(receipt.lane_dispatches);
    writer.write_u32(receipt.serial_token_forward_calls);
    writer.write_u8(receipt.terminal_ready as u8);
    writer.write_u8(receipt.token_committed as u8);
    writer.write_sha256(product.routed_b4_output_sha256);
    writer.write_sha256(product.checkpoint_chain_sha256);
    writer.write_sha256(product.final_hidden_sha256);
    writer.write_u64(product.completion.owner_id);
    writer.write_sha256(product.completion.production_binding_sha256);
    write_timeline_point(&mut writer, product.completion.submitted_transfer);
    write_timeline_point(&mut writer, product.completion.submitted_compute);
    writer.write_u64(product.completion.completed.generation);
    writer.write_u64(product.completion.completed.transfer);
    writer.write_u64(product.completion.completed.compute);
    writer.write_u64(product.completion.completed_queue_submit_calls);
    writer.write_u32(product.completion.completion_fence_checks);
    writer.write_sha256(product.completion.seal_sha256);
    writer.write_u64(product.audit.legacy_union_calls);
    writer.write_u64(product.audit.serial_token_forward_calls);
    writer.write_u64(product.audit.cpu_fallback_calls);
    writer.write_u64(product.audit.legacy_model_fallback_calls);
    writer.write_u64(product.audit.whole_model_fallback_calls);
    writer.finish()
}

fn write_timeline_point(writer: &mut S14StarwaveProofWriter, point: S14StarfoldTimelinePoint) {
    writer.write_u64(point.semaphore.as_raw());
    writer.write_u64(point.generation);
    writer.write_u64(point.value);
}

fn write_route(writer: &mut S14StarwaveProofWriter, route: &RouteDecision) {
    writer.write_u8(route.layer);
    writer.write_u8(match route.kind {
        RouterKind::Hash => 0,
        RouterKind::Score => 1,
    });
    writer.write_u64(route.expert_ids.len() as u64);
    for expert_id in &route.expert_ids {
        writer.write_u32(u32::from(*expert_id));
    }
    writer.write_u64(route.weights.len() as u64);
    for weight in &route.weights {
        writer.write_f32(*weight);
    }
}

fn write_hc_qkv_layer(
    writer: &mut S14StarwaveProofWriter,
    receipt: S14CausalBlockHcQkvLayerRecordingReceipt,
) {
    writer.write_u32(receipt.base_position);
    writer.write_u8(receipt.layer);
    writer.write_u64(receipt.block_size as u64);
    write_hidden_binding(writer, receipt.input_hidden);
    write_hidden_binding(writer, receipt.post_attention_hidden);
    writer.write_u32(receipt.layer_record_calls);
    writer.write_u32(receipt.command_graph_submit_calls);
    writer.write_u32(receipt.hc_qkv_projection_record_calls);
    writer.write_u32(receipt.attention_recording_calls);
    writer.write_u32(receipt.contiguous_attention_dispatch_calls);
    writer.write_u32(receipt.ratio4_boundary_attention_dispatch_calls);
    writer.write_u32(receipt.ratio4_state_transition_record_calls);
    writer.write_u32(receipt.attention_output_post_record_calls);
    writer.write_u32(receipt.ffn_hc_router_input_record_calls);
    writer.write_u32(receipt.router_recording_calls);
    writer.write_u32(receipt.serial_token_forward_calls);
    writer.write_u8(receipt.hc_hidden_integration_complete as u8);
}

fn write_b4_layer(writer: &mut S14StarwaveProofWriter, receipt: &S14StarfoldB4RoutedLayerReceipt) {
    writer.write_u32(u32::from(receipt.layer));
    writer.write_u64(receipt.base_position);
    writer.write_u32(receipt.unique_experts);
    write_projection(writer, receipt.w1);
    write_projection(writer, receipt.w3);
    writer.write_u32(u32::from(receipt.prepare.layer));
    writer.write_u64(receipt.prepare.base_position);
    writer.write_u32(receipt.prepare.exact_route_weights);
    writer.write_u32(receipt.prepare.update_buffer_calls);
    writer.write_u32(receipt.prepare.prepare_dispatch_calls);
    writer.write_u32(receipt.prepare.queue_submit_calls);
    writer.write_u32(receipt.prepare.serial_token_forward_calls);
    write_projection(writer, receipt.w2);
    writer.write_u32(receipt.packed_uploads);
    writer.write_u64(receipt.packed_upload_bytes);
    writer.write_u32(receipt.lane_dispatches);
    writer.write_u32(receipt.serial_token_forward_calls);
}

fn write_projection(
    writer: &mut S14StarwaveProofWriter,
    receipt: S14StarfoldProjectionExecutionReceipt,
) {
    writer.write_u32(u32::from(receipt.layer));
    writer.write_u64(receipt.base_position);
    writer.write_u8(match receipt.projection {
        S14StarfoldExpertProjection::W1 => 0,
        S14StarfoldExpertProjection::W3 => 1,
        S14StarfoldExpertProjection::W2 => 2,
    });
    writer.write_u32(receipt.unique_experts);
    writer.write_u32(receipt.packed_uploads);
    writer.write_u64(receipt.packed_upload_bytes);
    writer.write_u32(receipt.lane_dispatches);
    writer.write_u32(receipt.queue_submit_calls);
    writer.write_u32(receipt.source_projection_fallbacks);
    writer.write_u32(receipt.serial_token_forward_calls);
}

fn write_hidden_commit(
    writer: &mut S14StarwaveProofWriter,
    receipt: S14StarfoldK4HiddenCommitReceipt,
) {
    writer.write_u32(receipt.base_position);
    writer.write_u8(receipt.layer);
    write_hidden_binding(writer, receipt.next_hidden);
    writer.write_u32(receipt.routed_reduce_dispatch_calls);
    writer.write_u32(receipt.hc_post_dispatch_calls);
    writer.write_u32(receipt.queue_submit_calls);
    writer.write_u32(receipt.serial_token_forward_calls);
}

fn write_hidden_binding(writer: &mut S14StarwaveProofWriter, binding: S14CausalBlockHiddenBinding) {
    writer.write_u64(binding.buffer.as_raw());
    writer.write_u64(binding.offset);
    writer.write_u64(binding.bytes);
    writer.write_u64(binding.block_size as u64);
    writer.write_u64(binding.generation);
}

fn abort_receipt_sha256(
    transaction_id: u64,
    batch_id: u64,
    reason: &str,
    was_poisoned: bool,
    had_sealed_evidence: bool,
    discarded_physical_evidence_sha256: Option<S14StarwaveSha256>,
    checkpoint: S14StarfoldK4CommittedCheckpoint,
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starfold-k4-physical-abort-v2");
    writer.write_u32(S14_STARFOLD_K4_PHYSICAL_TRANSACTION_SCHEMA_VERSION);
    writer.write_u64(transaction_id);
    writer.write_u64(batch_id);
    writer.write_str(reason);
    writer.write_u8(was_poisoned as u8);
    writer.write_u8(had_sealed_evidence as u8);
    match discarded_physical_evidence_sha256 {
        Some(digest) => {
            writer.write_u8(1);
            writer.write_sha256(digest);
        }
        None => writer.write_u8(0),
    }
    writer.write_u64(checkpoint.commit_epoch);
    writer.write_u64(checkpoint.next_position);
    writer.write_sha256(checkpoint.commit_sha256);
    writer.finish()
}
