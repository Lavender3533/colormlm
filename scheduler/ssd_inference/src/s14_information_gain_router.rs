//! S14 信息增益路由器：在物理预算内选择最值得装载的能力胶囊。
//!
//! 本模块只生成确定性的事务候选集合，不执行 I/O、GPU 工作或旧模型整链回退。

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

pub const S14_INFORMATION_GAIN_ROUTER_VERSION: u32 = 1;
pub const S14_INFORMATION_GAIN_MAX_CANDIDATES: usize = 4_096;
pub const S14_INFORMATION_GAIN_MAX_RELATIONS: usize = 1_000_000;

const NANOSECONDS_PER_SECOND: u128 = 1_000_000_000;
const FINGERPRINT_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FINGERPRINT_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct S14CapsuleId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum S14ResidencyTier {
    Vram = 0,
    Ram = 1,
    Ssd = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum S14RouteRequirement {
    Demand = 0,
    Opportunistic = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum S14CandidateExecution {
    NativeCapsule,
    LegacyLocalOperator {
        boundary_key: u32,
        operator_key: u64,
    },
}

impl S14CandidateExecution {
    pub const fn is_legacy_local(self) -> bool {
        matches!(self, Self::LegacyLocalOperator { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct S14Complement {
    pub capsule_id: S14CapsuleId,
    pub extra_entropy_drop_nanobits: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14InformationGainCandidate {
    pub capsule_id: S14CapsuleId,
    pub requirement: S14RouteRequirement,
    pub execution: S14CandidateExecution,
    pub expected_entropy_drop_nanobits: u64,
    pub physical_bytes: u64,
    pub working_vram_bytes: u64,
    pub bandwidth_bytes_per_second: u64,
    pub flops: u64,
    pub throughput_flops_per_second: u64,
    pub fixed_latency_ns: u64,
    pub residency: S14ResidencyTier,
    pub duplicate_group: Option<u64>,
    pub complements: Vec<S14Complement>,
    pub conflicts: Vec<S14CapsuleId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14InformationGainBudget {
    pub max_vram_bytes: u64,
    pub max_transfer_bytes: u64,
    pub max_latency_ns: u64,
    pub max_legacy_local_operators: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14InformationGainCost {
    pub physical_bytes: u64,
    pub transferred_bytes: u64,
    pub working_vram_bytes: u64,
    pub transfer_latency_ns: u64,
    pub compute_latency_ns: u64,
    pub fixed_latency_ns: u64,
    pub total_latency_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14BudgetDimension {
    VramBytes,
    TransferBytes,
    LatencyNanoseconds,
    LegacyLocalOperators,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum S14RejectionReason {
    DuplicateCapability {
        duplicate_group: u64,
        selected_capsule_id: S14CapsuleId,
    },
    ConflictsWithSelected {
        selected_capsule_id: S14CapsuleId,
    },
    BudgetExceeded {
        dimension: S14BudgetDimension,
        already_used: u64,
        candidate_charge: u64,
        limit: u64,
    },
    NoPositiveMarginalInformationGain,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14SelectedCapsuleReceipt {
    pub ordinal: u32,
    pub capsule_id: S14CapsuleId,
    pub requirement: S14RouteRequirement,
    pub execution: S14CandidateExecution,
    pub residency: S14ResidencyTier,
    pub base_entropy_drop_nanobits: u64,
    pub complement_entropy_drop_nanobits: u64,
    pub marginal_entropy_drop_nanobits: u64,
    pub score_numerator_nanobits: u64,
    pub score_denominator_ns: u64,
    pub triggered_complements: Vec<S14CapsuleId>,
    pub cost: S14InformationGainCost,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14RejectedCapsuleReceipt {
    pub capsule_id: S14CapsuleId,
    pub requirement: S14RouteRequirement,
    pub final_marginal_entropy_drop_nanobits: u64,
    pub cost: S14InformationGainCost,
    pub reason: S14RejectionReason,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14InformationGainRoutingReceipt {
    pub router_version: u32,
    pub budget: S14InformationGainBudget,
    pub input_fingerprint: u64,
    pub decision_fingerprint: u64,
    pub demand_count: u32,
    pub total_entropy_drop_nanobits: u64,
    pub total_physical_bytes: u64,
    pub total_transfer_bytes: u64,
    pub total_vram_bytes: u64,
    pub total_latency_ns: u64,
    pub selected_legacy_local_operators: u32,
    pub selected: Vec<S14SelectedCapsuleReceipt>,
    pub rejected: Vec<S14RejectedCapsuleReceipt>,
}

impl S14InformationGainRoutingReceipt {
    pub const fn uses_whole_model_fallback(&self) -> bool {
        false
    }

    pub fn verify_against(
        &self,
        candidates: &[S14InformationGainCandidate],
    ) -> S14InformationGainResult<()> {
        let replay = S14InformationGainRouter::route(candidates, self.budget)?;
        if replay != *self {
            return Err(S14InformationGainError::new(
                S14InformationGainErrorKind::ReceiptMismatch,
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum S14InformationGainErrorKind {
    TooManyCandidates {
        actual: usize,
        maximum: usize,
    },
    TooManyRelations {
        actual: usize,
        maximum: usize,
    },
    DuplicateCapsuleId {
        capsule_id: S14CapsuleId,
    },
    UnknownRelationTarget {
        capsule_id: S14CapsuleId,
        target_capsule_id: S14CapsuleId,
    },
    SelfRelation {
        capsule_id: S14CapsuleId,
    },
    DuplicateRelation {
        capsule_id: S14CapsuleId,
        target_capsule_id: S14CapsuleId,
    },
    InconsistentComplement {
        left_capsule_id: S14CapsuleId,
        right_capsule_id: S14CapsuleId,
        first_extra_nanobits: u64,
        second_extra_nanobits: u64,
    },
    ConflictAndComplement {
        left_capsule_id: S14CapsuleId,
        right_capsule_id: S14CapsuleId,
    },
    MissingBandwidth {
        capsule_id: S14CapsuleId,
    },
    MissingThroughput {
        capsule_id: S14CapsuleId,
    },
    ArithmeticOverflow {
        capsule_id: Option<S14CapsuleId>,
        field: &'static str,
    },
    UnsatisfiedDemand {
        capsule_id: S14CapsuleId,
        reason: S14RejectionReason,
    },
    ReceiptMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14InformationGainError {
    kind: S14InformationGainErrorKind,
}

impl S14InformationGainError {
    fn new(kind: S14InformationGainErrorKind) -> Self {
        Self { kind }
    }

    pub const fn kind(&self) -> &S14InformationGainErrorKind {
        &self.kind
    }
}

impl fmt::Display for S14InformationGainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "S14 信息增益路由失败: {:?}", self.kind)
    }
}

impl Error for S14InformationGainError {}

pub type S14InformationGainResult<T> = Result<T, S14InformationGainError>;

#[derive(Debug)]
struct PreparedCandidate<'a> {
    candidate: &'a S14InformationGainCandidate,
    cost: S14InformationGainCost,
}

#[derive(Debug, Default)]
struct CanonicalRelations {
    complements: BTreeMap<(S14CapsuleId, S14CapsuleId), u64>,
    conflicts: BTreeSet<(S14CapsuleId, S14CapsuleId)>,
}

#[derive(Debug, Default)]
struct SelectionState {
    selected_ids: BTreeSet<S14CapsuleId>,
    selected_duplicate_groups: BTreeMap<u64, S14CapsuleId>,
    selected: Vec<S14SelectedCapsuleReceipt>,
    total_entropy_drop_nanobits: u64,
    total_physical_bytes: u64,
    total_transfer_bytes: u64,
    total_vram_bytes: u64,
    total_latency_ns: u64,
    selected_legacy_local_operators: u32,
}

#[derive(Debug)]
struct MarginalEvaluation {
    capsule_id: S14CapsuleId,
    base_gain: u64,
    complement_gain: u64,
    marginal_gain: u64,
    triggered_complements: Vec<S14CapsuleId>,
    total_latency_ns: u64,
    transferred_bytes: u64,
    working_vram_bytes: u64,
}

pub struct S14InformationGainRouter;

impl S14InformationGainRouter {
    pub fn route(
        candidates: &[S14InformationGainCandidate],
        budget: S14InformationGainBudget,
    ) -> S14InformationGainResult<S14InformationGainRoutingReceipt> {
        if candidates.len() > S14_INFORMATION_GAIN_MAX_CANDIDATES {
            return Err(S14InformationGainError::new(
                S14InformationGainErrorKind::TooManyCandidates {
                    actual: candidates.len(),
                    maximum: S14_INFORMATION_GAIN_MAX_CANDIDATES,
                },
            ));
        }

        let mut ordered = candidates.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|candidate| candidate.capsule_id);
        for pair in ordered.windows(2) {
            if pair[0].capsule_id == pair[1].capsule_id {
                return Err(S14InformationGainError::new(
                    S14InformationGainErrorKind::DuplicateCapsuleId {
                        capsule_id: pair[0].capsule_id,
                    },
                ));
            }
        }

        let known_ids = ordered
            .iter()
            .map(|candidate| candidate.capsule_id)
            .collect::<BTreeSet<_>>();
        let relations = normalize_relations(&ordered, &known_ids)?;
        validate_possible_gain(&ordered, &relations)?;

        let mut prepared = Vec::with_capacity(ordered.len());
        for candidate in ordered {
            prepared.push(PreparedCandidate {
                cost: prepare_cost(candidate)?,
                candidate,
            });
        }
        validate_aggregate_accounting(&prepared)?;

        let input_fingerprint = fingerprint_input(&prepared, &relations, budget);
        let demand_count = u32::try_from(
            prepared
                .iter()
                .filter(|candidate| candidate.candidate.requirement == S14RouteRequirement::Demand)
                .count(),
        )
        .map_err(|_| {
            S14InformationGainError::new(S14InformationGainErrorKind::ArithmeticOverflow {
                capsule_id: None,
                field: "demand_count",
            })
        })?;
        let mut state = SelectionState::default();

        for candidate in prepared
            .iter()
            .filter(|candidate| candidate.candidate.requirement == S14RouteRequirement::Demand)
        {
            if let Some(reason) = rejection_reason(candidate, &state, &relations, budget) {
                return Err(S14InformationGainError::new(
                    S14InformationGainErrorKind::UnsatisfiedDemand {
                        capsule_id: candidate.candidate.capsule_id,
                        reason,
                    },
                ));
            }
            let evaluation = marginal_evaluation(candidate, &state.selected_ids, &relations)?;
            select_candidate(candidate, evaluation, &mut state)?;
        }

        loop {
            let mut best: Option<(&PreparedCandidate<'_>, MarginalEvaluation)> = None;
            for candidate in prepared.iter().filter(|candidate| {
                candidate.candidate.requirement == S14RouteRequirement::Opportunistic
                    && !state.selected_ids.contains(&candidate.candidate.capsule_id)
            }) {
                if rejection_reason(candidate, &state, &relations, budget).is_some() {
                    continue;
                }
                let evaluation = marginal_evaluation(candidate, &state.selected_ids, &relations)?;
                if evaluation.marginal_gain == 0 {
                    continue;
                }
                let replace = best.as_ref().is_none_or(|(_, current)| {
                    compare_marginal(&evaluation, current) == Ordering::Greater
                });
                if replace {
                    best = Some((candidate, evaluation));
                }
            }
            let Some((candidate, evaluation)) = best else {
                break;
            };
            select_candidate(candidate, evaluation, &mut state)?;
        }

        let mut rejected = Vec::with_capacity(prepared.len().saturating_sub(state.selected.len()));
        for candidate in &prepared {
            if state.selected_ids.contains(&candidate.candidate.capsule_id) {
                continue;
            }
            let evaluation = marginal_evaluation(candidate, &state.selected_ids, &relations)?;
            let reason = rejection_reason(candidate, &state, &relations, budget)
                .unwrap_or(S14RejectionReason::NoPositiveMarginalInformationGain);
            rejected.push(S14RejectedCapsuleReceipt {
                capsule_id: candidate.candidate.capsule_id,
                requirement: candidate.candidate.requirement,
                final_marginal_entropy_drop_nanobits: evaluation.marginal_gain,
                cost: candidate.cost,
                reason,
            });
        }

        let mut receipt = S14InformationGainRoutingReceipt {
            router_version: S14_INFORMATION_GAIN_ROUTER_VERSION,
            budget,
            input_fingerprint,
            decision_fingerprint: 0,
            demand_count,
            total_entropy_drop_nanobits: state.total_entropy_drop_nanobits,
            total_physical_bytes: state.total_physical_bytes,
            total_transfer_bytes: state.total_transfer_bytes,
            total_vram_bytes: state.total_vram_bytes,
            total_latency_ns: state.total_latency_ns,
            selected_legacy_local_operators: state.selected_legacy_local_operators,
            selected: state.selected,
            rejected,
        };
        receipt.decision_fingerprint = fingerprint_decision(&receipt);
        Ok(receipt)
    }
}

fn prepare_cost(
    candidate: &S14InformationGainCandidate,
) -> S14InformationGainResult<S14InformationGainCost> {
    let transferred_bytes = match candidate.residency {
        S14ResidencyTier::Vram => 0,
        S14ResidencyTier::Ram | S14ResidencyTier::Ssd => candidate.physical_bytes,
    };
    if transferred_bytes != 0 && candidate.bandwidth_bytes_per_second == 0 {
        return Err(S14InformationGainError::new(
            S14InformationGainErrorKind::MissingBandwidth {
                capsule_id: candidate.capsule_id,
            },
        ));
    }
    if candidate.flops != 0 && candidate.throughput_flops_per_second == 0 {
        return Err(S14InformationGainError::new(
            S14InformationGainErrorKind::MissingThroughput {
                capsule_id: candidate.capsule_id,
            },
        ));
    }

    let transfer_latency_ns = scaled_ceil_div(
        transferred_bytes,
        candidate.bandwidth_bytes_per_second,
        candidate.capsule_id,
        "transfer_latency_ns",
    )?;
    let compute_latency_ns = scaled_ceil_div(
        candidate.flops,
        candidate.throughput_flops_per_second,
        candidate.capsule_id,
        "compute_latency_ns",
    )?;
    let total_latency_ns = transfer_latency_ns
        .checked_add(compute_latency_ns)
        .and_then(|value| value.checked_add(candidate.fixed_latency_ns))
        .ok_or_else(|| {
            S14InformationGainError::new(S14InformationGainErrorKind::ArithmeticOverflow {
                capsule_id: Some(candidate.capsule_id),
                field: "total_latency_ns",
            })
        })?;
    Ok(S14InformationGainCost {
        physical_bytes: candidate.physical_bytes,
        transferred_bytes,
        working_vram_bytes: candidate.working_vram_bytes,
        transfer_latency_ns,
        compute_latency_ns,
        fixed_latency_ns: candidate.fixed_latency_ns,
        total_latency_ns,
    })
}

fn scaled_ceil_div(
    quantity: u64,
    rate_per_second: u64,
    capsule_id: S14CapsuleId,
    field: &'static str,
) -> S14InformationGainResult<u64> {
    if quantity == 0 {
        return Ok(0);
    }
    debug_assert_ne!(rate_per_second, 0);
    let numerator = u128::from(quantity) * NANOSECONDS_PER_SECOND;
    let result = numerator.div_ceil(u128::from(rate_per_second));
    u64::try_from(result).map_err(|_| {
        S14InformationGainError::new(S14InformationGainErrorKind::ArithmeticOverflow {
            capsule_id: Some(capsule_id),
            field,
        })
    })
}

fn normalize_relations(
    candidates: &[&S14InformationGainCandidate],
    known_ids: &BTreeSet<S14CapsuleId>,
) -> S14InformationGainResult<CanonicalRelations> {
    let relation_count = candidates.iter().try_fold(0usize, |total, candidate| {
        total
            .checked_add(candidate.complements.len())
            .and_then(|value| value.checked_add(candidate.conflicts.len()))
    });
    let Some(relation_count) = relation_count else {
        return Err(S14InformationGainError::new(
            S14InformationGainErrorKind::TooManyRelations {
                actual: usize::MAX,
                maximum: S14_INFORMATION_GAIN_MAX_RELATIONS,
            },
        ));
    };
    if relation_count > S14_INFORMATION_GAIN_MAX_RELATIONS {
        return Err(S14InformationGainError::new(
            S14InformationGainErrorKind::TooManyRelations {
                actual: relation_count,
                maximum: S14_INFORMATION_GAIN_MAX_RELATIONS,
            },
        ));
    }

    let mut relations = CanonicalRelations::default();
    for candidate in candidates {
        let mut local_targets = BTreeSet::new();
        for complement in &candidate.complements {
            validate_relation_target(
                candidate.capsule_id,
                complement.capsule_id,
                known_ids,
                &mut local_targets,
            )?;
            let pair = canonical_pair(candidate.capsule_id, complement.capsule_id);
            if let Some(previous) = relations
                .complements
                .insert(pair, complement.extra_entropy_drop_nanobits)
            {
                if previous != complement.extra_entropy_drop_nanobits {
                    return Err(S14InformationGainError::new(
                        S14InformationGainErrorKind::InconsistentComplement {
                            left_capsule_id: pair.0,
                            right_capsule_id: pair.1,
                            first_extra_nanobits: previous,
                            second_extra_nanobits: complement.extra_entropy_drop_nanobits,
                        },
                    ));
                }
            }
        }
        for &conflict in &candidate.conflicts {
            validate_relation_target(
                candidate.capsule_id,
                conflict,
                known_ids,
                &mut local_targets,
            )?;
            relations
                .conflicts
                .insert(canonical_pair(candidate.capsule_id, conflict));
        }
    }
    if let Some(pair) = relations
        .conflicts
        .iter()
        .find(|pair| relations.complements.contains_key(pair))
        .copied()
    {
        return Err(S14InformationGainError::new(
            S14InformationGainErrorKind::ConflictAndComplement {
                left_capsule_id: pair.0,
                right_capsule_id: pair.1,
            },
        ));
    }
    Ok(relations)
}

fn validate_relation_target(
    capsule_id: S14CapsuleId,
    target_capsule_id: S14CapsuleId,
    known_ids: &BTreeSet<S14CapsuleId>,
    local_targets: &mut BTreeSet<S14CapsuleId>,
) -> S14InformationGainResult<()> {
    if capsule_id == target_capsule_id {
        return Err(S14InformationGainError::new(
            S14InformationGainErrorKind::SelfRelation { capsule_id },
        ));
    }
    if !known_ids.contains(&target_capsule_id) {
        return Err(S14InformationGainError::new(
            S14InformationGainErrorKind::UnknownRelationTarget {
                capsule_id,
                target_capsule_id,
            },
        ));
    }
    if !local_targets.insert(target_capsule_id) {
        return Err(S14InformationGainError::new(
            S14InformationGainErrorKind::DuplicateRelation {
                capsule_id,
                target_capsule_id,
            },
        ));
    }
    Ok(())
}

fn canonical_pair(left: S14CapsuleId, right: S14CapsuleId) -> (S14CapsuleId, S14CapsuleId) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

fn validate_possible_gain(
    candidates: &[&S14InformationGainCandidate],
    relations: &CanonicalRelations,
) -> S14InformationGainResult<()> {
    let base = candidates.iter().try_fold(0u64, |total, candidate| {
        total.checked_add(candidate.expected_entropy_drop_nanobits)
    });
    let total = base.and_then(|base| {
        relations
            .complements
            .values()
            .try_fold(base, |value, extra| value.checked_add(*extra))
    });
    if total.is_none() {
        return Err(S14InformationGainError::new(
            S14InformationGainErrorKind::ArithmeticOverflow {
                capsule_id: None,
                field: "possible_entropy_drop_nanobits",
            },
        ));
    }
    Ok(())
}

fn validate_aggregate_accounting(
    candidates: &[PreparedCandidate<'_>],
) -> S14InformationGainResult<()> {
    for (field, values) in [
        (
            "aggregate_physical_bytes",
            candidates
                .iter()
                .map(|candidate| candidate.cost.physical_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            "aggregate_transfer_bytes",
            candidates
                .iter()
                .map(|candidate| candidate.cost.transferred_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            "aggregate_vram_bytes",
            candidates
                .iter()
                .map(|candidate| candidate.cost.working_vram_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            "aggregate_latency_ns",
            candidates
                .iter()
                .map(|candidate| candidate.cost.total_latency_ns)
                .collect::<Vec<_>>(),
        ),
    ] {
        if values
            .into_iter()
            .try_fold(0u64, |total, value| total.checked_add(value))
            .is_none()
        {
            return Err(S14InformationGainError::new(
                S14InformationGainErrorKind::ArithmeticOverflow {
                    capsule_id: None,
                    field,
                },
            ));
        }
    }
    Ok(())
}

fn rejection_reason(
    candidate: &PreparedCandidate<'_>,
    state: &SelectionState,
    relations: &CanonicalRelations,
    budget: S14InformationGainBudget,
) -> Option<S14RejectionReason> {
    if let Some(group) = candidate.candidate.duplicate_group {
        if let Some(&selected_capsule_id) = state.selected_duplicate_groups.get(&group) {
            return Some(S14RejectionReason::DuplicateCapability {
                duplicate_group: group,
                selected_capsule_id,
            });
        }
    }
    if let Some(selected_capsule_id) = state.selected_ids.iter().find(|selected| {
        relations
            .conflicts
            .contains(&canonical_pair(candidate.candidate.capsule_id, **selected))
    }) {
        return Some(S14RejectionReason::ConflictsWithSelected {
            selected_capsule_id: *selected_capsule_id,
        });
    }
    if candidate.candidate.execution.is_legacy_local()
        && state.selected_legacy_local_operators >= budget.max_legacy_local_operators
    {
        return Some(S14RejectionReason::BudgetExceeded {
            dimension: S14BudgetDimension::LegacyLocalOperators,
            already_used: u64::from(state.selected_legacy_local_operators),
            candidate_charge: 1,
            limit: u64::from(budget.max_legacy_local_operators),
        });
    }
    budget_rejection(
        S14BudgetDimension::VramBytes,
        state.total_vram_bytes,
        candidate.cost.working_vram_bytes,
        budget.max_vram_bytes,
    )
    .or_else(|| {
        budget_rejection(
            S14BudgetDimension::TransferBytes,
            state.total_transfer_bytes,
            candidate.cost.transferred_bytes,
            budget.max_transfer_bytes,
        )
    })
    .or_else(|| {
        budget_rejection(
            S14BudgetDimension::LatencyNanoseconds,
            state.total_latency_ns,
            candidate.cost.total_latency_ns,
            budget.max_latency_ns,
        )
    })
}

fn budget_rejection(
    dimension: S14BudgetDimension,
    already_used: u64,
    candidate_charge: u64,
    limit: u64,
) -> Option<S14RejectionReason> {
    let exceeds = already_used
        .checked_add(candidate_charge)
        .is_none_or(|total| total > limit);
    exceeds.then_some(S14RejectionReason::BudgetExceeded {
        dimension,
        already_used,
        candidate_charge,
        limit,
    })
}

fn marginal_evaluation(
    candidate: &PreparedCandidate<'_>,
    selected_ids: &BTreeSet<S14CapsuleId>,
    relations: &CanonicalRelations,
) -> S14InformationGainResult<MarginalEvaluation> {
    let mut complement_gain = 0u64;
    let mut triggered_complements = Vec::new();
    for &selected in selected_ids {
        if let Some(&extra) = relations
            .complements
            .get(&canonical_pair(candidate.candidate.capsule_id, selected))
        {
            complement_gain = complement_gain.checked_add(extra).ok_or_else(|| {
                S14InformationGainError::new(S14InformationGainErrorKind::ArithmeticOverflow {
                    capsule_id: Some(candidate.candidate.capsule_id),
                    field: "complement_entropy_drop_nanobits",
                })
            })?;
            triggered_complements.push(selected);
        }
    }
    let marginal_gain = candidate
        .candidate
        .expected_entropy_drop_nanobits
        .checked_add(complement_gain)
        .ok_or_else(|| {
            S14InformationGainError::new(S14InformationGainErrorKind::ArithmeticOverflow {
                capsule_id: Some(candidate.candidate.capsule_id),
                field: "marginal_entropy_drop_nanobits",
            })
        })?;
    Ok(MarginalEvaluation {
        capsule_id: candidate.candidate.capsule_id,
        base_gain: candidate.candidate.expected_entropy_drop_nanobits,
        complement_gain,
        marginal_gain,
        triggered_complements,
        total_latency_ns: candidate.cost.total_latency_ns,
        transferred_bytes: candidate.cost.transferred_bytes,
        working_vram_bytes: candidate.cost.working_vram_bytes,
    })
}

fn compare_marginal(left: &MarginalEvaluation, right: &MarginalEvaluation) -> Ordering {
    compare_ratio(
        left.marginal_gain,
        left.total_latency_ns,
        right.marginal_gain,
        right.total_latency_ns,
    )
    .then_with(|| left.marginal_gain.cmp(&right.marginal_gain))
    .then_with(|| right.total_latency_ns.cmp(&left.total_latency_ns))
    .then_with(|| right.transferred_bytes.cmp(&left.transferred_bytes))
    .then_with(|| right.working_vram_bytes.cmp(&left.working_vram_bytes))
    .then_with(|| right.capsule_id.cmp(&left.capsule_id))
}

fn compare_ratio(
    left_numerator: u64,
    left_denominator: u64,
    right_numerator: u64,
    right_denominator: u64,
) -> Ordering {
    match (left_denominator == 0, right_denominator == 0) {
        (true, true) => left_numerator.cmp(&right_numerator),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => (u128::from(left_numerator) * u128::from(right_denominator))
            .cmp(&(u128::from(right_numerator) * u128::from(left_denominator))),
    }
}

fn select_candidate(
    candidate: &PreparedCandidate<'_>,
    evaluation: MarginalEvaluation,
    state: &mut SelectionState,
) -> S14InformationGainResult<()> {
    let capsule_id = candidate.candidate.capsule_id;
    state.total_entropy_drop_nanobits = checked_add(
        state.total_entropy_drop_nanobits,
        evaluation.marginal_gain,
        capsule_id,
        "selected_entropy_drop_nanobits",
    )?;
    state.total_physical_bytes = checked_add(
        state.total_physical_bytes,
        candidate.cost.physical_bytes,
        capsule_id,
        "selected_physical_bytes",
    )?;
    state.total_transfer_bytes = checked_add(
        state.total_transfer_bytes,
        candidate.cost.transferred_bytes,
        capsule_id,
        "selected_transfer_bytes",
    )?;
    state.total_vram_bytes = checked_add(
        state.total_vram_bytes,
        candidate.cost.working_vram_bytes,
        capsule_id,
        "selected_vram_bytes",
    )?;
    state.total_latency_ns = checked_add(
        state.total_latency_ns,
        candidate.cost.total_latency_ns,
        capsule_id,
        "selected_latency_ns",
    )?;
    if candidate.candidate.execution.is_legacy_local() {
        state.selected_legacy_local_operators = state
            .selected_legacy_local_operators
            .checked_add(1)
            .ok_or_else(|| {
                S14InformationGainError::new(S14InformationGainErrorKind::ArithmeticOverflow {
                    capsule_id: Some(capsule_id),
                    field: "selected_legacy_local_operators",
                })
            })?;
    }
    state.selected_ids.insert(capsule_id);
    if let Some(group) = candidate.candidate.duplicate_group {
        state.selected_duplicate_groups.insert(group, capsule_id);
    }
    let ordinal = u32::try_from(state.selected.len()).map_err(|_| {
        S14InformationGainError::new(S14InformationGainErrorKind::ArithmeticOverflow {
            capsule_id: Some(capsule_id),
            field: "selection_ordinal",
        })
    })?;
    state.selected.push(S14SelectedCapsuleReceipt {
        ordinal,
        capsule_id,
        requirement: candidate.candidate.requirement,
        execution: candidate.candidate.execution,
        residency: candidate.candidate.residency,
        base_entropy_drop_nanobits: evaluation.base_gain,
        complement_entropy_drop_nanobits: evaluation.complement_gain,
        marginal_entropy_drop_nanobits: evaluation.marginal_gain,
        score_numerator_nanobits: evaluation.marginal_gain,
        score_denominator_ns: candidate.cost.total_latency_ns,
        triggered_complements: evaluation.triggered_complements,
        cost: candidate.cost,
    });
    Ok(())
}

fn checked_add(
    left: u64,
    right: u64,
    capsule_id: S14CapsuleId,
    field: &'static str,
) -> S14InformationGainResult<u64> {
    left.checked_add(right).ok_or_else(|| {
        S14InformationGainError::new(S14InformationGainErrorKind::ArithmeticOverflow {
            capsule_id: Some(capsule_id),
            field,
        })
    })
}

#[derive(Clone, Copy)]
struct Fingerprint64(u64);

impl Fingerprint64 {
    fn new() -> Self {
        Self(FINGERPRINT_OFFSET_BASIS)
    }

    fn write_u8(&mut self, value: u8) {
        self.0 ^= u64::from(value);
        self.0 = self.0.wrapping_mul(FINGERPRINT_PRIME);
    }

    fn write_u32(&mut self, value: u32) {
        for byte in value.to_le_bytes() {
            self.write_u8(byte);
        }
    }

    fn write_u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.write_u8(byte);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}

fn fingerprint_input(
    candidates: &[PreparedCandidate<'_>],
    relations: &CanonicalRelations,
    budget: S14InformationGainBudget,
) -> u64 {
    let mut hash = Fingerprint64::new();
    hash.write_u32(S14_INFORMATION_GAIN_ROUTER_VERSION);
    hash_budget(&mut hash, budget);
    hash.write_u64(candidates.len() as u64);
    for prepared in candidates {
        let candidate = prepared.candidate;
        hash.write_u64(candidate.capsule_id.0);
        hash.write_u8(candidate.requirement as u8);
        match candidate.execution {
            S14CandidateExecution::NativeCapsule => hash.write_u8(0),
            S14CandidateExecution::LegacyLocalOperator {
                boundary_key,
                operator_key,
            } => {
                hash.write_u8(1);
                hash.write_u32(boundary_key);
                hash.write_u64(operator_key);
            }
        }
        hash.write_u64(candidate.expected_entropy_drop_nanobits);
        hash.write_u64(candidate.physical_bytes);
        hash.write_u64(candidate.working_vram_bytes);
        hash.write_u64(candidate.bandwidth_bytes_per_second);
        hash.write_u64(candidate.flops);
        hash.write_u64(candidate.throughput_flops_per_second);
        hash.write_u64(candidate.fixed_latency_ns);
        hash.write_u8(candidate.residency as u8);
        match candidate.duplicate_group {
            Some(group) => {
                hash.write_u8(1);
                hash.write_u64(group);
            }
            None => hash.write_u8(0),
        }
    }
    hash.write_u64(relations.complements.len() as u64);
    for (&(left, right), &extra) in &relations.complements {
        hash.write_u64(left.0);
        hash.write_u64(right.0);
        hash.write_u64(extra);
    }
    hash.write_u64(relations.conflicts.len() as u64);
    for &(left, right) in &relations.conflicts {
        hash.write_u64(left.0);
        hash.write_u64(right.0);
    }
    hash.finish()
}

fn fingerprint_decision(receipt: &S14InformationGainRoutingReceipt) -> u64 {
    let mut hash = Fingerprint64::new();
    hash.write_u64(receipt.input_fingerprint);
    hash.write_u64(receipt.selected.len() as u64);
    for selected in &receipt.selected {
        hash.write_u32(selected.ordinal);
        hash.write_u64(selected.capsule_id.0);
        hash.write_u64(selected.marginal_entropy_drop_nanobits);
        hash.write_u64(selected.cost.transferred_bytes);
        hash.write_u64(selected.cost.working_vram_bytes);
        hash.write_u64(selected.cost.total_latency_ns);
        hash.write_u64(selected.triggered_complements.len() as u64);
        for complement in &selected.triggered_complements {
            hash.write_u64(complement.0);
        }
    }
    hash.write_u64(receipt.rejected.len() as u64);
    for rejected in &receipt.rejected {
        hash.write_u64(rejected.capsule_id.0);
        hash.write_u64(rejected.final_marginal_entropy_drop_nanobits);
        hash_rejection_reason(&mut hash, &rejected.reason);
    }
    hash.write_u64(receipt.total_entropy_drop_nanobits);
    hash.write_u64(receipt.total_physical_bytes);
    hash.write_u64(receipt.total_transfer_bytes);
    hash.write_u64(receipt.total_vram_bytes);
    hash.write_u64(receipt.total_latency_ns);
    hash.write_u32(receipt.selected_legacy_local_operators);
    hash.finish()
}

fn hash_budget(hash: &mut Fingerprint64, budget: S14InformationGainBudget) {
    hash.write_u64(budget.max_vram_bytes);
    hash.write_u64(budget.max_transfer_bytes);
    hash.write_u64(budget.max_latency_ns);
    hash.write_u32(budget.max_legacy_local_operators);
}

fn hash_rejection_reason(hash: &mut Fingerprint64, reason: &S14RejectionReason) {
    match reason {
        S14RejectionReason::DuplicateCapability {
            duplicate_group,
            selected_capsule_id,
        } => {
            hash.write_u8(0);
            hash.write_u64(*duplicate_group);
            hash.write_u64(selected_capsule_id.0);
        }
        S14RejectionReason::ConflictsWithSelected {
            selected_capsule_id,
        } => {
            hash.write_u8(1);
            hash.write_u64(selected_capsule_id.0);
        }
        S14RejectionReason::BudgetExceeded {
            dimension,
            already_used,
            candidate_charge,
            limit,
        } => {
            hash.write_u8(2);
            hash.write_u8(match dimension {
                S14BudgetDimension::VramBytes => 0,
                S14BudgetDimension::TransferBytes => 1,
                S14BudgetDimension::LatencyNanoseconds => 2,
                S14BudgetDimension::LegacyLocalOperators => 3,
            });
            hash.write_u64(*already_used);
            hash.write_u64(*candidate_charge);
            hash.write_u64(*limit);
        }
        S14RejectionReason::NoPositiveMarginalInformationGain => hash.write_u8(3),
    }
}
