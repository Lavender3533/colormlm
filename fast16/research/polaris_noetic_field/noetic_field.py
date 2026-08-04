#!/usr/bin/env python3
"""Polaris Noetic Field 的最小真实激活回放。

这不是聊天模型，也不把离线 capture 冒充实时权重执行。
它只验证一个可证伪的计算骨架：多个权重岛意见在共享场中反复竞争，
约束链保留冲突，只在稳定后提交，否则回退权威路径。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


EPS = 1e-12


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _entropy(probability: np.ndarray) -> float:
    if probability.size <= 1:
        return 0.0
    return float(
        -np.sum(probability * np.log(np.maximum(probability, EPS)))
        / math.log(probability.size)
    )


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    mixture = 0.5 * (left + right)
    return float(
        0.5
        * np.sum(left * (np.log(np.maximum(left, EPS)) - np.log(np.maximum(mixture, EPS))))
        + 0.5
        * np.sum(right * (np.log(np.maximum(right, EPS)) - np.log(np.maximum(mixture, EPS))))
    )


@dataclass(frozen=True)
class IslandOpinion:
    island_id: str
    probability: np.ndarray
    confidence: float
    physical_cost: float


@dataclass(frozen=True)
class CycleReceipt:
    cycle: int
    top_token: int
    top_probability: float
    movement_l1: float
    conflict_js: float
    base_expression: float
    donor_expression: float
    stable_cycles: int


@dataclass(frozen=True)
class CommitReceipt:
    row: int
    task_id: str
    sample_id: str
    base_token: int
    donor_token: int
    committed_token: int
    label: int
    committed: bool
    rolled_back_to_base: bool
    reason: str
    cycles: int
    final_conflict_js: float
    trace: tuple[CycleReceipt, ...]


def _confidence(probability: np.ndarray) -> float:
    if probability.size == 1:
        return 1.0
    top = np.partition(probability, -2)[-2:]
    top.sort()
    margin = float(top[1] - top[0])
    return float(np.clip(0.55 * (1.0 - _entropy(probability)) + 0.45 * margin, 0.0, 1.0))


def _candidate_union(
    base_logits: np.ndarray,
    donor_logits: np.ndarray,
    donor_base_ids: np.ndarray,
    top_k: int,
) -> np.ndarray:
    base_top = np.argpartition(base_logits, -top_k)[-top_k:]
    donor_local_top = np.argpartition(donor_logits, -top_k)[-top_k:]
    donor_top = donor_base_ids[donor_local_top]
    return np.unique(np.concatenate((base_top, donor_top))).astype(np.int64, copy=False)


def _donor_on_support(
    donor_logits: np.ndarray,
    donor_base_ids: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(donor_base_ids)
    sorted_ids = donor_base_ids[order]
    positions = np.searchsorted(sorted_ids, support)
    valid = positions < sorted_ids.size
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= sorted_ids[positions[valid_indices]] == support[valid_indices]
    if not np.any(valid):
        return np.full(support.size, -80.0, dtype=np.float64), valid
    floor = float(np.min(donor_logits) - 20.0)
    projected = np.full(support.size, floor, dtype=np.float64)
    projected[valid] = donor_logits[order[positions[valid]]]
    return projected, valid


def evolve_field(
    base_logits: np.ndarray,
    donor_logits: np.ndarray,
    donor_base_ids: np.ndarray,
    *,
    top_k: int,
    max_cycles: int,
    donor_enabled: bool,
) -> tuple[int, bool, str, tuple[CycleReceipt, ...], float]:
    """令两个岛在共享候选空间内产生连续影响。

    不读文本、task id 或 label。donor 无法表达的 token 保留为一等候选，
    约束链会依照表达覆盖与冲突自动压低 donor 发言权。
    """
    base_token = int(np.argmax(base_logits))
    if not donor_enabled:
        return base_token, True, "donor_disabled_exact_noop", (), 0.0

    support = _candidate_union(base_logits, donor_logits, donor_base_ids, top_k)
    base_probability = _softmax(base_logits[support].astype(np.float64))
    donor_projected, donor_coverage = _donor_on_support(
        donor_logits, donor_base_ids, support
    )
    donor_probability = _softmax(donor_projected)

    opinions = (
        IslandOpinion("authority-base", base_probability, _confidence(base_probability), 1.0),
        IslandOpinion("donor-0", donor_probability, _confidence(donor_probability), 1.35),
    )
    conflict = _js_divergence(base_probability, donor_probability)
    coverage = float(np.count_nonzero(donor_coverage) / support.size)

    # 不把 base 当语言主干；它只是可精确回滚的权威意见之一。
    field = np.full(support.size, 1.0 / support.size, dtype=np.float64)
    trace: list[CycleReceipt] = []
    previous_top = -1
    stable_cycles = 0

    for cycle in range(1, max_cycles + 1):
        bids: list[float] = []
        for opinion in opinions:
            novelty = _js_divergence(field, opinion.probability)
            information_gain = max(_entropy(field) - _entropy(opinion.probability), 0.0)
            bid = (0.20 + opinion.confidence + information_gain + 0.25 * novelty) / opinion.physical_cost
            bids.append(bid)

        # Helix 约束链：高冲突和低覆盖不做硬否决，只降低 donor 表达，保留候选。
        bids[1] *= max(0.05, coverage * math.exp(-2.5 * conflict))
        expression = np.asarray(bids, dtype=np.float64)
        expression /= np.sum(expression)

        target_log = sum(
            expression[index] * np.log(np.maximum(opinion.probability, EPS))
            for index, opinion in enumerate(opinions)
        )
        target = _softmax(target_log)
        next_field = 0.35 * field + 0.65 * target
        next_field /= np.sum(next_field)
        movement = float(np.sum(np.abs(next_field - field)))
        field = next_field

        top_index = int(np.argmax(field))
        top_token = int(support[top_index])
        if top_token == previous_top and movement < 0.02:
            stable_cycles += 1
        else:
            stable_cycles = 0
        previous_top = top_token
        trace.append(
            CycleReceipt(
                cycle=cycle,
                top_token=top_token,
                top_probability=float(field[top_index]),
                movement_l1=movement,
                conflict_js=conflict,
                base_expression=float(expression[0]),
                donor_expression=float(expression[1]),
                stable_cycles=stable_cycles,
            )
        )
        if stable_cycles >= 2:
            break

    top_token = int(support[int(np.argmax(field))])
    stable = stable_cycles >= 2
    ordered = np.sort(field)
    margin = float(ordered[-1] - ordered[-2]) if field.size > 1 else 1.0
    if not stable:
        return base_token, False, "field_not_stable_rollback", tuple(trace), conflict
    if conflict > 0.62 and margin < 0.10:
        return base_token, False, "unresolved_conflict_rollback", tuple(trace), conflict
    return top_token, True, "stable_attractor_commit", tuple(trace), conflict


def run_replay(
    capture_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    top_k: int,
    max_cycles: int,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_sha = _sha256(capture_path)
    expected_sha = str(manifest.get("capture_sha256", ""))
    if observed_sha != expected_sha:
        raise ValueError(f"capture SHA-256 不匹配: expected={expected_sha}, observed={observed_sha}")

    with np.load(capture_path, allow_pickle=False) as data:
        base_logits = np.asarray(data["base_logits"], dtype=np.float64)
        donor_logits = np.asarray(data["donor_0_logits"], dtype=np.float64)
        donor_base_ids = np.asarray(data["donor_0_base_ids"], dtype=np.int64)
        labels = np.asarray(data["labels"], dtype=np.int64)
        task_ids = np.asarray(data["task_ids"], dtype=np.str_)
        sample_ids = np.asarray(data["sample_ids"], dtype=np.str_)

    receipts: list[CommitReceipt] = []
    noop_exact = True
    for row in range(labels.size):
        token, committed, reason, trace, conflict = evolve_field(
            base_logits[row],
            donor_logits[row],
            donor_base_ids,
            top_k=top_k,
            max_cycles=max_cycles,
            donor_enabled=True,
        )
        noop_token, _, _, _, _ = evolve_field(
            base_logits[row],
            donor_logits[row],
            donor_base_ids,
            top_k=top_k,
            max_cycles=max_cycles,
            donor_enabled=False,
        )
        base_token = int(np.argmax(base_logits[row]))
        donor_token = int(donor_base_ids[int(np.argmax(donor_logits[row]))])
        noop_exact &= noop_token == base_token
        receipts.append(
            CommitReceipt(
                row=row,
                task_id=str(task_ids[row]),
                sample_id=str(sample_ids[row]),
                base_token=base_token,
                donor_token=donor_token,
                committed_token=token,
                label=int(labels[row]),
                committed=committed,
                rolled_back_to_base=token == base_token and reason != "stable_attractor_commit",
                reason=reason,
                cycles=len(trace),
                final_conflict_js=conflict,
                trace=trace,
            )
        )

    base_correct = sum(item.base_token == item.label for item in receipts)
    donor_correct = sum(item.donor_token == item.label for item in receipts)
    field_correct = sum(item.committed_token == item.label for item in receipts)
    changed = [item for item in receipts if item.committed_token != item.base_token]
    report = {
        "format": "polaris-noetic-field-replay-v0",
        "status": "structural_experiment_not_live_weight_execution",
        "source": {
            "capture": str(capture_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "capture_sha256": observed_sha,
            "rows": int(labels.size),
        },
        "contract": {
            "label_visible_to_field": False,
            "task_id_visible_to_field": False,
            "text_visible_to_field": False,
            "fixed_task_router": False,
            "iterative_cycles": True,
            "exact_noop_when_donor_disabled": bool(noop_exact),
        },
        "summary": {
            "base_top1_correct": base_correct,
            "donor_top1_correct": donor_correct,
            "field_top1_correct": field_correct,
            "field_changed_from_base": len(changed),
            "field_changes_won": sum(
                item.committed_token == item.label and item.base_token != item.label for item in changed
            ),
            "field_changes_lost": sum(
                item.committed_token != item.label and item.base_token == item.label for item in changed
            ),
            "rolled_back": sum(item.rolled_back_to_base for item in receipts),
            "stable_commits": sum(item.committed for item in receipts),
            "mean_cycles": float(np.mean([item.cycles for item in receipts])),
            "max_cycles": max(item.cycles for item in receipts),
        },
        "receipts": [asdict(item) for item in receipts],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    _configure_utf8()
    root = Path(__file__).resolve().parents[1] / "inference_arch"
    parser = argparse.ArgumentParser(description="回放真实激活的 Polaris 意识场最小实验")
    parser.add_argument("--capture", type=Path, default=root / "capture_dev60b.npz")
    parser.add_argument("--manifest", type=Path, default=root / "capture_dev60b.manifest.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("replay_receipt.json"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-cycles", type=int, default=12)
    args = parser.parse_args()
    if args.top_k < 2 or args.max_cycles < 3:
        parser.error("--top-k 必须>=2，--max-cycles 必须>=3")
    report = run_replay(
        args.capture,
        args.manifest,
        args.output,
        top_k=args.top_k,
        max_cycles=args.max_cycles,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"回执: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
