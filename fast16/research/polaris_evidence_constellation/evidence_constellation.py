#!/usr/bin/env python3
"""Polaris 证据耦合星座动力学的最小结构内核。

这个模块不执行模型，也不将神经置信度当成事实。它只实现：

1. 器官提出局部变化；
2. 可交换变化合并，冲突变化分支；
3. 证据按独立来源组更新候选；
4. 只提交所有存活分支中共有、已证实且依赖闭合的局部子图。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    expected: Any
    threshold: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("证据阈值必须位于 (0, 1]")


@dataclass(frozen=True)
class Claim:
    claim_id: str
    slot: str
    value: Any
    organ_id: str
    requirements: tuple[EvidenceRequirement, ...] = ()
    depends_on: tuple[str, ...] = ()
    anchor: bool = False

    def __post_init__(self) -> None:
        if not self.claim_id or not self.slot or not self.organ_id:
            raise ValueError("claim_id、slot 和 organ_id 不能为空")
        if self.anchor and self.requirements:
            raise ValueError("不可变锚点不应依赖后续证据")


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    organ_id: str
    claims: tuple[Claim, ...]
    goal_alignment: float
    novelty: float
    expected_gain: float
    physical_cost: float

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.organ_id or not self.claims:
            raise ValueError("提议必须有身份、器官与至少一个候选")
        if self.physical_cost < 0.0:
            raise ValueError("物理成本不能为负")
        slots = [claim.slot for claim in self.claims]
        if len(slots) != len(set(slots)):
            raise ValueError("同一提议不能对同一 slot 写入多次")
        for claim in self.claims:
            if claim.organ_id != self.organ_id:
                raise ValueError("候选的 organ_id 必须与提议一致")

    @property
    def exploration_priority(self) -> float:
        """只用于调度探索，不参与事实认证。"""
        numerator = max(self.goal_alignment, 0.0) + max(self.novelty, 0.0) + max(
            self.expected_gain, 0.0
        )
        return numerator / (1.0 + self.physical_cost)


@dataclass(frozen=True)
class EvidenceReceipt:
    receipt_id: str
    requirement_id: str
    observed: Any
    source_group: str
    reliability: float
    match_strength: float
    epoch: int

    def __post_init__(self) -> None:
        if not self.receipt_id or not self.requirement_id or not self.source_group:
            raise ValueError("证据回执身份不能为空")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability 必须位于 [0, 1]")
        if not 0.0 <= self.match_strength <= 1.0:
            raise ValueError("match_strength 必须位于 [0, 1]")


@dataclass
class Branch:
    branch_id: str
    assignments: dict[str, str]
    parent_branch_id: str | None
    born_from: str


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: str
    status: str
    certification: float
    contradiction: float
    requirement_support: dict[str, float]
    requirement_refute: dict[str, float]


@dataclass(frozen=True)
class TransitionReceipt:
    epoch: int
    action: str
    detail: dict[str, Any]


@dataclass
class ConstellationField:
    """版本化的候选图、分支和证据状态。"""

    field_id: str
    claims: dict[str, Claim] = field(default_factory=dict)
    proposals: dict[str, Proposal] = field(default_factory=dict)
    branches: dict[str, Branch] = field(default_factory=dict)
    committed: dict[str, str] = field(default_factory=dict)
    evidence_log: list[EvidenceReceipt] = field(default_factory=list)
    transitions: list[TransitionReceipt] = field(default_factory=list)
    epoch: int = 0
    _branch_counter: int = 0
    _proposal_round_sealed: bool = False
    _latest_evidence: dict[tuple[str, str], EvidenceReceipt] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("field_id 不能为空")
        if not self.branches:
            self.branches["branch.0"] = Branch(
                branch_id="branch.0",
                assignments={},
                parent_branch_id=None,
                born_from="field_birth",
            )

    def _record(self, action: str, **detail: Any) -> TransitionReceipt:
        receipt = TransitionReceipt(self.epoch, action, detail)
        self.transitions.append(receipt)
        return receipt

    def _validate_claim_registration(self, claim: Claim) -> None:
        current = self.claims.get(claim.claim_id)
        if current is not None and current != claim:
            raise ValueError(f"候选身份冲突: {claim.claim_id}")
        for other in self.claims.values():
            if other.anchor and other.slot == claim.slot and other.value != claim.value:
                raise ValueError(f"提议试图覆盖北极星锚点: {claim.slot}")

    def _register_claim(self, claim: Claim) -> None:
        self._validate_claim_registration(claim)
        self.claims[claim.claim_id] = claim

    def add_anchor(self, claim: Claim) -> None:
        if not claim.anchor:
            raise ValueError("add_anchor 只接受 anchor=True 的候选")
        for branch in self.branches.values():
            existing = branch.assignments.get(claim.slot)
            if existing is not None and existing != claim.claim_id:
                raise ValueError(f"锚点与已有分支冲突: {claim.slot}")
        self._validate_claim_registration(claim)
        self.epoch += 1
        self.claims[claim.claim_id] = claim
        for branch in self.branches.values():
            branch.assignments[claim.slot] = claim.claim_id
        self.committed[claim.slot] = claim.claim_id
        self._record("anchor", claim_id=claim.claim_id, slot=claim.slot)

    def propose(self, branch_id: str, proposal: Proposal) -> TransitionReceipt:
        """合并可交换局部变化；同 slot 异值时保留原分支并生成新分支。"""
        if branch_id not in self.branches:
            raise KeyError(f"未知分支: {branch_id}")
        if proposal.proposal_id in self.proposals:
            raise ValueError(f"提议已存在: {proposal.proposal_id}")
        for claim in proposal.claims:
            self._validate_claim_registration(claim)
            committed_claim_id = self.committed.get(claim.slot)
            if committed_claim_id is not None and committed_claim_id != claim.claim_id:
                committed_claim = self.claims[committed_claim_id]
                if committed_claim.anchor:
                    raise ValueError(f"提议试图覆盖北极星锚点: {claim.slot}")
                raise ValueError(f"已提交 slot 必须先因证据失效而回滚: {claim.slot}")

        self.epoch += 1
        for claim in proposal.claims:
            self.claims[claim.claim_id] = claim
        self.proposals[proposal.proposal_id] = proposal
        target = self.branches[branch_id]
        conflicts = {
            claim.slot: (target.assignments[claim.slot], claim.claim_id)
            for claim in proposal.claims
            if claim.slot in target.assignments
            and target.assignments[claim.slot] != claim.claim_id
        }
        self._proposal_round_sealed = False

        if not conflicts:
            for claim in proposal.claims:
                target.assignments[claim.slot] = claim.claim_id
            return self._record(
                "merge",
                proposal_id=proposal.proposal_id,
                branch_id=branch_id,
                priority=proposal.exploration_priority,
                written_slots=sorted(claim.slot for claim in proposal.claims),
            )

        self._branch_counter += 1
        new_branch_id = f"branch.{self._branch_counter}"
        assignments = dict(target.assignments)
        for claim in proposal.claims:
            assignments[claim.slot] = claim.claim_id
        self.branches[new_branch_id] = Branch(
            branch_id=new_branch_id,
            assignments=assignments,
            parent_branch_id=branch_id,
            born_from=proposal.proposal_id,
        )
        return self._record(
            "branch",
            proposal_id=proposal.proposal_id,
            source_branch_id=branch_id,
            new_branch_id=new_branch_id,
            priority=proposal.exploration_priority,
            conflicts=conflicts,
        )

    def seal_proposals(self) -> TransitionReceipt:
        """声明本轮器官提议已到齐，防止在替代分支出生前早提交。"""
        self.epoch += 1
        self._proposal_round_sealed = True
        return self._record(
            "seal_proposals",
            proposal_count=len(self.proposals),
            branch_count=len(self.branches),
        )

    @staticmethod
    def _noisy_or(strengths: Iterable[float]) -> float:
        remaining = 1.0
        seen = False
        for strength in strengths:
            seen = True
            remaining *= 1.0 - min(max(strength, 0.0), 1.0)
        return 1.0 - remaining if seen else 0.0

    def assess_claim(self, claim_id: str) -> ClaimAssessment:
        claim = self.claims[claim_id]
        if claim.anchor:
            return ClaimAssessment(claim_id, "certified", 1.0, 0.0, {}, {})
        if not claim.requirements:
            return ClaimAssessment(claim_id, "pending", 0.0, 0.0, {}, {})

        support_by_requirement: dict[str, float] = {}
        refute_by_requirement: dict[str, float] = {}
        contradicted = False
        certified = True
        for requirement in claim.requirements:
            receipts = [
                receipt
                for (requirement_id, _), receipt in self._latest_evidence.items()
                if requirement_id == requirement.requirement_id
            ]
            support = self._noisy_or(
                receipt.reliability * receipt.match_strength
                for receipt in receipts
                if receipt.observed == requirement.expected
            )
            refute = self._noisy_or(
                receipt.reliability * receipt.match_strength
                for receipt in receipts
                if receipt.observed != requirement.expected
            )
            support_by_requirement[requirement.requirement_id] = support
            refute_by_requirement[requirement.requirement_id] = refute
            contradicted |= refute >= requirement.threshold
            certified &= support >= requirement.threshold

        certification = min(support_by_requirement.values())
        contradiction = max(refute_by_requirement.values())
        status = "contradicted" if contradicted else "certified" if certified else "pending"
        return ClaimAssessment(
            claim_id=claim_id,
            status=status,
            certification=certification,
            contradiction=contradiction,
            requirement_support=support_by_requirement,
            requirement_refute=refute_by_requirement,
        )

    def _rollback_invalid_commits(self) -> tuple[str, ...]:
        rolled_back: list[str] = []
        changed = True
        while changed:
            changed = False
            committed_claim_ids = set(self.committed.values())
            for slot, claim_id in list(self.committed.items()):
                claim = self.claims[claim_id]
                if claim.anchor:
                    continue
                assessment = self.assess_claim(claim_id)
                dependencies_closed = set(claim.depends_on).issubset(committed_claim_ids)
                if assessment.status != "certified" or not dependencies_closed:
                    del self.committed[slot]
                    rolled_back.append(claim_id)
                    changed = True
        return tuple(rolled_back)

    def attach_evidence(
        self,
        *,
        receipt_id: str,
        requirement_id: str,
        observed: Any,
        source_group: str,
        reliability: float = 1.0,
        match_strength: float = 1.0,
    ) -> TransitionReceipt:
        """同一 requirement/source_group 的新回执取代旧回执，不重复投票。"""
        self.epoch += 1
        receipt = EvidenceReceipt(
            receipt_id=receipt_id,
            requirement_id=requirement_id,
            observed=observed,
            source_group=source_group,
            reliability=reliability,
            match_strength=match_strength,
            epoch=self.epoch,
        )
        self.evidence_log.append(receipt)
        self._latest_evidence[(requirement_id, source_group)] = receipt
        affected_claims = sorted(
            claim.claim_id
            for claim in self.claims.values()
            if any(
                requirement.requirement_id == requirement_id
                for requirement in claim.requirements
            )
        )
        rolled_back = self._rollback_invalid_commits()
        return self._record(
            "attach_evidence",
            receipt=asdict(receipt),
            affected_claims=affected_claims,
            rolled_back_claims=list(rolled_back),
        )

    def branch_is_viable(self, branch_id: str) -> bool:
        branch = self.branches[branch_id]
        return all(
            self.assess_claim(claim_id).status != "contradicted"
            for claim_id in branch.assignments.values()
        )

    def viable_branch_ids(self) -> tuple[str, ...]:
        return tuple(
            branch_id
            for branch_id in sorted(self.branches)
            if self.branch_is_viable(branch_id)
        )

    def _common_assignments(self, branch_ids: tuple[str, ...]) -> dict[str, str]:
        if not branch_ids:
            return {}
        common = dict(self.branches[branch_ids[0]].assignments)
        for branch_id in branch_ids[1:]:
            assignments = self.branches[branch_id].assignments
            common = {
                slot: claim_id
                for slot, claim_id in common.items()
                if assignments.get(slot) == claim_id
            }
        return common

    def partial_commit(self) -> TransitionReceipt:
        if not self._proposal_round_sealed:
            raise RuntimeError("提案轮尚未 seal，禁止在替代分支出生前早提交")
        self.epoch += 1
        viable = self.viable_branch_ids()
        common = self._common_assignments(viable)
        newly_committed: list[str] = []

        progress = True
        while progress:
            progress = False
            committed_claim_ids = set(self.committed.values())
            for slot, claim_id in sorted(common.items()):
                if self.committed.get(slot) == claim_id:
                    continue
                if slot in self.committed and self.committed[slot] != claim_id:
                    continue
                claim = self.claims[claim_id]
                assessment = self.assess_claim(claim_id)
                if assessment.status != "certified":
                    continue
                if not set(claim.depends_on).issubset(committed_claim_ids):
                    continue
                self.committed[slot] = claim_id
                newly_committed.append(claim_id)
                progress = True

        return self._record(
            "partial_commit",
            viable_branches=list(viable),
            common_slots=sorted(common),
            newly_committed_claims=newly_committed,
            committed=dict(sorted(self.committed.items())),
            blocked=len(viable) == 0,
        )

    def snapshot(self) -> dict[str, Any]:
        assessments = {
            claim_id: asdict(self.assess_claim(claim_id))
            for claim_id in sorted(self.claims)
        }
        return {
            "field_id": self.field_id,
            "epoch": self.epoch,
            "proposal_round_sealed": self._proposal_round_sealed,
            "claims": {claim_id: asdict(claim) for claim_id, claim in sorted(self.claims.items())},
            "proposals": {
                proposal_id: {
                    **asdict(proposal),
                    "exploration_priority": proposal.exploration_priority,
                }
                for proposal_id, proposal in sorted(self.proposals.items())
            },
            "branches": {
                branch_id: {
                    **asdict(branch),
                    "viable": self.branch_is_viable(branch_id),
                }
                for branch_id, branch in sorted(self.branches.items())
            },
            "assessments": assessments,
            "committed": dict(sorted(self.committed.items())),
            "evidence_log": [asdict(receipt) for receipt in self.evidence_log],
            "transitions": [asdict(receipt) for receipt in self.transitions],
        }


def make_claim(
    claim_id: str,
    slot: str,
    value: Any,
    organ_id: str,
    *,
    requirements: Iterable[EvidenceRequirement] = (),
    depends_on: Iterable[str] = (),
    anchor: bool = False,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        slot=slot,
        value=value,
        organ_id=organ_id,
        requirements=tuple(requirements),
        depends_on=tuple(depends_on),
        anchor=anchor,
    )


def finite_snapshot(snapshot: dict[str, Any]) -> bool:
    """回执写盘前的最小有限值守卫。"""

    def walk(value: Any) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(walk(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(walk(item) for item in value)
        return True

    return walk(snapshot)
