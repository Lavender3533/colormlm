#!/usr/bin/env python3
"""Blind Composite-Axis Gate 的共享表达式语言与确定性搜索器。"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PrimitiveRelation:
    organ_id: str
    left: str
    right: str

    def evaluate(self, features: dict[str, float]) -> bool:
        return float(features[self.left]) < float(features[self.right])

    @property
    def canonical(self) -> str:
        return f"lt({self.left},{self.right})"

    def snapshot(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_snapshot(cls, value: dict[str, str]) -> "PrimitiveRelation":
        return cls(
            organ_id=value["organ_id"],
            left=value["left"],
            right=value["right"],
        )


@dataclass(frozen=True)
class CompositeAxis:
    """由多个器官的局部关系组成的布尔潜坐标。"""

    terms: tuple[PrimitiveRelation, ...]
    negate: bool = False

    def __post_init__(self) -> None:
        if len(self.terms) < 2:
            raise ValueError("复合坐标至少需要两个局部关系")
        organs = [term.organ_id for term in self.terms]
        if len(organs) != len(set(organs)):
            raise ValueError("同一复合坐标每个器官最多贡献一个局部关系")

    def evaluate(self, features: dict[str, float]) -> bool:
        parity = False
        for term in self.terms:
            parity ^= term.evaluate(features)
        return parity ^ self.negate

    @property
    def required_organs(self) -> tuple[str, ...]:
        return tuple(sorted(term.organ_id for term in self.terms))

    @property
    def canonical(self) -> str:
        body = " xor ".join(term.canonical for term in self.terms)
        return f"not({body})" if self.negate else body

    @property
    def description_length(self) -> int:
        return 1 + 3 * len(self.terms) + int(self.negate)

    def snapshot(self) -> dict[str, object]:
        return {
            "operator": "parity",
            "negate": self.negate,
            "terms": [term.snapshot() for term in self.terms],
            "canonical": self.canonical,
            "required_organs": list(self.required_organs),
            "description_length": self.description_length,
        }

    @classmethod
    def from_snapshot(cls, value: dict[str, object]) -> "CompositeAxis":
        if value.get("operator") != "parity":
            raise ValueError("未知复合算子")
        raw_terms = value.get("terms")
        if not isinstance(raw_terms, list):
            raise ValueError("Frame terms 无效")
        return cls(
            terms=tuple(PrimitiveRelation.from_snapshot(term) for term in raw_terms),
            negate=bool(value.get("negate", False)),
        )


@dataclass(frozen=True)
class AxisFit:
    axis: CompositeAxis
    correct: int
    total: int
    candidate_count: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True)
class PrimitiveFit:
    primitive: PrimitiveRelation
    invert: bool
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    def predict(self, features: dict[str, float]) -> bool:
        return self.primitive.evaluate(features) ^ self.invert


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根必须是对象: {path}")
    return value


def feature_schema() -> dict[str, tuple[str, str]]:
    return {
        "time_organ": ("time.deadline", "time.duration"),
        "interaction_organ": (
            "interaction.feedback_count",
            "interaction.turn_interval",
        ),
        "state_organ": ("state.lifetime", "state.mutation_rate"),
        "resource_organ": ("resource.burst_size", "resource.memory_pressure"),
    }


def schema_snapshot(schema: dict[str, tuple[str, str]]) -> dict[str, list[str]]:
    return {organ: list(features) for organ, features in sorted(schema.items())}


def schema_from_snapshot(value: object) -> dict[str, tuple[str, str]]:
    if not isinstance(value, dict):
        raise ValueError("schema 必须是对象")
    schema: dict[str, tuple[str, str]] = {}
    for organ, raw_features in value.items():
        if not isinstance(organ, str) or not isinstance(raw_features, list):
            raise ValueError("schema 条目无效")
        features = tuple(str(item) for item in raw_features)
        if len(features) != 2 or len(set(features)) != 2:
            raise ValueError("v1 每个器官必须有两个不同的原始量")
        schema[organ] = (features[0], features[1])
    if len(schema) < 2:
        raise ValueError("至少需要两个器官")
    return schema


def primitive_relations(
    schema: dict[str, tuple[str, str]],
) -> tuple[PrimitiveRelation, ...]:
    """每个器官只保留一个规范方向；反向关系由最终取反等价表示。

    对 parity 而言，翻转任意奇数个 term 等价于翻转整个表达式，偶数个翻转则
    不改变表达式。保留双向会把同一布尔函数重复 2**k 次，并扭曲置换基线。
    """
    primitives = []
    for organ_id, features in sorted(schema.items()):
        left, right = features
        primitives.append(PrimitiveRelation(organ_id, left, right))
    return tuple(primitives)


def composite_candidates(
    schema: dict[str, tuple[str, str]],
    *,
    min_terms: int = 2,
    max_terms: int = 3,
) -> tuple[CompositeAxis, ...]:
    by_organ: dict[str, list[PrimitiveRelation]] = {}
    for primitive in primitive_relations(schema):
        by_organ.setdefault(primitive.organ_id, []).append(primitive)

    candidates = []
    organs = tuple(sorted(by_organ))
    for width in range(min_terms, min(max_terms, len(organs)) + 1):
        for selected_organs in itertools.combinations(organs, width):
            for terms in itertools.product(*(by_organ[organ] for organ in selected_organs)):
                for negate in (False, True):
                    candidates.append(CompositeAxis(tuple(terms), negate))
    return tuple(candidates)


def _row_features(row: dict[str, object]) -> dict[str, float]:
    raw = row.get("features")
    if not isinstance(raw, dict):
        raise ValueError("episode.features 必须是对象")
    return {str(key): float(value) for key, value in raw.items()}


def _row_outcome(row: dict[str, object]) -> bool:
    value = row.get("outcome")
    if not isinstance(value, bool):
        raise ValueError("episode.outcome 必须是布尔值")
    return value


def accuracy_of(axis: CompositeAxis, rows: Iterable[dict[str, object]]) -> tuple[int, int]:
    correct = 0
    total = 0
    for row in rows:
        total += 1
        correct += axis.evaluate(_row_features(row)) == _row_outcome(row)
    if total == 0:
        raise ValueError("不能评估空数据集")
    return correct, total


def synthesize_axis(
    schema: dict[str, tuple[str, str]],
    rows: Iterable[dict[str, object]],
) -> AxisFit:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("发现集不能为空")
    candidates = composite_candidates(schema)
    ranked: list[tuple[int, int, str, CompositeAxis]] = []
    for axis in candidates:
        correct, _ = accuracy_of(axis, materialized)
        ranked.append((-correct, axis.description_length, axis.canonical, axis))
    ranked.sort()
    best = ranked[0][3]
    correct, total = accuracy_of(best, materialized)
    return AxisFit(best, correct, total, len(candidates))


def fit_best_primitive(
    schema: dict[str, tuple[str, str]],
    rows: Iterable[dict[str, object]],
) -> PrimitiveFit:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("发现集不能为空")
    ranked: list[tuple[int, str, bool, PrimitiveRelation]] = []
    for primitive in primitive_relations(schema):
        for invert in (False, True):
            correct = sum(
                (primitive.evaluate(_row_features(row)) ^ invert) == _row_outcome(row)
                for row in materialized
            )
            ranked.append((-correct, primitive.canonical, invert, primitive))
    ranked.sort()
    _, _, invert, primitive = ranked[0]
    correct = sum(
        (primitive.evaluate(_row_features(row)) ^ invert) == _row_outcome(row)
        for row in materialized
    )
    return PrimitiveFit(primitive, invert, correct, len(materialized))
