#!/usr/bin/env python3
"""北极星“星云相”的最小 Reframe（重构问题坐标）实验。

输入不是候选答案，而是多个器官各自看到的局部活动碎片。内核只根据碎片中的
上下文/断言关系搜索一个更短的描述：若把某个上下文维度提升为问题坐标，能够
显著减少互相冲突的全局断言，同时没有把每个碎片都切成孤岛，则形成一个暂态
Frame。Frame 仍然不是事实；它必须进入 evidence_constellation 的凝固相后才能提交。
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from evidence_constellation import (
    ConstellationField,
    EvidenceRequirement,
    Proposal,
    make_claim,
)


PairTuple = tuple[tuple[str, str], ...]


def _canonical_pairs(values: dict[str, str] | Iterable[tuple[str, str]]) -> PairTuple:
    pairs = tuple(sorted(dict(values).items()))
    if not pairs or any(not key or not value for key, value in pairs):
        raise ValueError("上下文和断言必须包含非空键值")
    return pairs


@dataclass(frozen=True)
class ActivationFragment:
    """一个器官的局部活动；它不允许携带完整 Frame 或候选列表。"""

    fragment_id: str
    organ_id: str
    context: PairTuple
    assertions: PairTuple

    def __post_init__(self) -> None:
        if not self.fragment_id or not self.organ_id:
            raise ValueError("fragment_id 和 organ_id 不能为空")
        if not self.context or not self.assertions:
            raise ValueError("局部活动必须同时含有上下文和断言")
        if tuple(sorted(dict(self.context).items())) != self.context:
            raise ValueError("context 必须规范化且键唯一")
        if tuple(sorted(dict(self.assertions).items())) != self.assertions:
            raise ValueError("assertions 必须规范化且键唯一")

    @property
    def context_map(self) -> dict[str, str]:
        return dict(self.context)

    @property
    def assertion_map(self) -> dict[str, str]:
        return dict(self.assertions)


def make_fragment(
    fragment_id: str,
    organ_id: str,
    *,
    context: dict[str, str] | Iterable[tuple[str, str]],
    assertions: dict[str, str] | Iterable[tuple[str, str]],
) -> ActivationFragment:
    return ActivationFragment(
        fragment_id=fragment_id,
        organ_id=organ_id,
        context=_canonical_pairs(context),
        assertions=_canonical_pairs(assertions),
    )


@dataclass(frozen=True)
class FrameGroup:
    coordinates: PairTuple
    fragment_ids: tuple[str, ...]
    organ_ids: tuple[str, ...]
    consensus: PairTuple


@dataclass(frozen=True)
class FrameCandidate:
    partition_keys: tuple[str, ...]
    groups: tuple[FrameGroup, ...]
    residual_conflicts: int
    singleton_groups: int
    description_length: float
    gain_over_flat: float
    emergent: bool


@dataclass(frozen=True)
class ReframeDecision:
    flat_conflicts: int
    flat_description_length: float
    selected: FrameCandidate | None
    alternatives: tuple[FrameCandidate, ...]
    reason: str

    def snapshot(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceBinding:
    claim_id: str
    slot: str
    requirement_id: str
    expected: str
    coordinates: PairTuple
    predicate: str


class NebulaReframer:
    """使用最小描述长度搜索问题划分，而不是给预设候选投票。"""

    def __init__(
        self,
        *,
        conflict_cost: float = 4.0,
        partition_cost: float = 1.0,
        key_cost: float = 1.0,
        singleton_cost: float = 2.0,
        min_gain: float = 1.0,
        max_partition_keys: int = 2,
    ) -> None:
        if min(conflict_cost, partition_cost, key_cost, singleton_cost) < 0.0:
            raise ValueError("MDL 成本不能为负")
        if min_gain < 0.0 or max_partition_keys < 1:
            raise ValueError("min_gain/max_partition_keys 无效")
        self.conflict_cost = conflict_cost
        self.partition_cost = partition_cost
        self.key_cost = key_cost
        self.singleton_cost = singleton_cost
        self.min_gain = min_gain
        self.max_partition_keys = max_partition_keys

    @staticmethod
    def _conflicts(groups: Iterable[tuple[ActivationFragment, ...]]) -> int:
        conflicts = 0
        for group in groups:
            by_predicate: dict[str, list[str]] = defaultdict(list)
            for fragment in group:
                for predicate, value in fragment.assertions:
                    by_predicate[predicate].append(value)
            for values in by_predicate.values():
                counts = Counter(values)
                conflicts += len(values) - max(counts.values())
        return conflicts

    def _description_length(
        self,
        *,
        conflicts: int,
        group_count: int,
        key_count: int,
        singleton_count: int,
    ) -> float:
        return (
            conflicts * self.conflict_cost
            + max(group_count - 1, 0) * self.partition_cost
            + key_count * self.key_cost
            + singleton_count * self.singleton_cost
        )

    @staticmethod
    def _consensus(group: tuple[ActivationFragment, ...]) -> PairTuple:
        by_predicate: dict[str, list[str]] = defaultdict(list)
        for fragment in group:
            for predicate, value in fragment.assertions:
                by_predicate[predicate].append(value)
        return tuple(
            sorted(
                (predicate, values[0])
                for predicate, values in by_predicate.items()
                if len(set(values)) == 1
            )
        )

    @staticmethod
    def _is_emergent(groups: tuple[FrameGroup, ...]) -> bool:
        """没有任何单一器官覆盖全部新分区，才算跨器官形成的结构。"""
        if len(groups) < 2:
            return False
        contributors = [set(group.organ_ids) for group in groups]
        return not set.intersection(*contributors)

    def _build_candidate(
        self,
        fragments: tuple[ActivationFragment, ...],
        partition_keys: tuple[str, ...],
        flat_length: float,
    ) -> FrameCandidate:
        grouped: dict[tuple[str, ...], list[ActivationFragment]] = defaultdict(list)
        for fragment in fragments:
            context = fragment.context_map
            grouped[tuple(context[key] for key in partition_keys)].append(fragment)

        raw_groups = tuple(
            tuple(sorted(group, key=lambda item: item.fragment_id))
            for _, group in sorted(grouped.items())
        )
        frame_groups = tuple(
            FrameGroup(
                coordinates=tuple(
                    zip(
                        partition_keys,
                        tuple(group[0].context_map[key] for key in partition_keys),
                    )
                ),
                fragment_ids=tuple(fragment.fragment_id for fragment in group),
                organ_ids=tuple(sorted({fragment.organ_id for fragment in group})),
                consensus=self._consensus(group),
            )
            for group in raw_groups
        )
        conflicts = self._conflicts(raw_groups)
        singleton_count = sum(len(group) == 1 for group in raw_groups)
        length = self._description_length(
            conflicts=conflicts,
            group_count=len(raw_groups),
            key_count=len(partition_keys),
            singleton_count=singleton_count,
        )
        return FrameCandidate(
            partition_keys=partition_keys,
            groups=frame_groups,
            residual_conflicts=conflicts,
            singleton_groups=singleton_count,
            description_length=length,
            gain_over_flat=flat_length - length,
            emergent=self._is_emergent(frame_groups),
        )

    def discover(self, fragments: Iterable[ActivationFragment]) -> ReframeDecision:
        materialized = tuple(fragments)
        if len(materialized) < 2:
            raise ValueError("至少需要两个局部活动碎片")
        fragment_ids = [fragment.fragment_id for fragment in materialized]
        if len(fragment_ids) != len(set(fragment_ids)):
            raise ValueError("fragment_id 必须唯一")

        common_context_keys = set(materialized[0].context_map)
        for fragment in materialized[1:]:
            common_context_keys &= set(fragment.context_map)
        eligible_keys = tuple(
            sorted(
                key
                for key in common_context_keys
                if len({fragment.context_map[key] for fragment in materialized}) > 1
            )
        )

        flat_conflicts = self._conflicts((materialized,))
        flat_length = self._description_length(
            conflicts=flat_conflicts,
            group_count=1,
            key_count=0,
            singleton_count=0,
        )
        candidates = [
            self._build_candidate(materialized, keys, flat_length)
            for width in range(1, min(self.max_partition_keys, len(eligible_keys)) + 1)
            for keys in itertools.combinations(eligible_keys, width)
        ]
        candidates.sort(
            key=lambda candidate: (
                candidate.description_length,
                len(candidate.partition_keys),
                candidate.partition_keys,
            )
        )

        selected = candidates[0] if candidates else None
        if flat_conflicts == 0:
            selected = None
            reason = "全局描述没有冲突，无需为了新颖而强行重构"
        elif selected is None:
            reason = "没有跨碎片共有且发生变化的上下文维度"
        elif selected.gain_over_flat < self.min_gain:
            selected = None
            reason = "冲突减少不足以抵偿分区复杂度"
        elif not selected.emergent:
            selected = None
            reason = "单一器官覆盖全部分区，不构成跨器官涌现结构"
        else:
            reason = "重构降低描述长度，且没有单一器官提出完整新 Frame"

        return ReframeDecision(
            flat_conflicts=flat_conflicts,
            flat_description_length=flat_length,
            selected=selected,
            alternatives=tuple(candidates),
            reason=reason,
        )


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")
    return slug or "value"


def install_frame_for_evidence(
    field: ConstellationField,
    frame: FrameCandidate,
    *,
    goal_claim_id: str,
) -> tuple[EvidenceBinding, ...]:
    """把暂态 Frame 送入已有凝固相；本函数不提供或伪造任何证据。"""
    claims = []
    bindings = []
    for group in frame.groups:
        coordinate_slug = "__".join(
            f"{_slug(key)}_{_slug(value)}" for key, value in group.coordinates
        )
        for predicate, value in group.consensus:
            predicate_slug = _slug(predicate)
            claim_id = f"claim.reframe.{coordinate_slug}.{predicate_slug}"
            slot = f"frame.{coordinate_slug}.{predicate_slug}"
            requirement_id = f"probe.reframe.{coordinate_slug}.{predicate_slug}"
            claims.append(
                make_claim(
                    claim_id,
                    slot,
                    {
                        "coordinates": dict(group.coordinates),
                        "predicate": predicate,
                        "value": value,
                        "source_fragments": list(group.fragment_ids),
                    },
                    "nebula.reframe",
                    requirements=(EvidenceRequirement(requirement_id, value),),
                    depends_on=(goal_claim_id,),
                )
            )
            bindings.append(
                EvidenceBinding(
                    claim_id=claim_id,
                    slot=slot,
                    requirement_id=requirement_id,
                    expected=value,
                    coordinates=group.coordinates,
                    predicate=predicate,
                )
            )
    if not claims:
        raise ValueError("Frame 没有可送入证据相的组内共识")
    field.propose(
        "branch.0",
        Proposal(
            proposal_id="proposal.nebula.reframe",
            organ_id="nebula.reframe",
            claims=tuple(claims),
            goal_alignment=1.0,
            novelty=max(frame.gain_over_flat, 0.0),
            expected_gain=max(frame.gain_over_flat, 0.0),
            physical_cost=0.0,
        ),
    )
    return tuple(bindings)
