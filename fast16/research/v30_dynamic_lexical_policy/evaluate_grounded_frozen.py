"""用预先冻结的v30权重评估全新、可由输入唯一推出的工具状态任务。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, os.fspath(HERE))
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402
from probe_dynamic_lexical_head import (  # noqa: E402
    VOCAB,
    load_capture,
    read_jsonl,
    recent_unique,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v30冻结动态词汇头独立门")
    parser.add_argument("--contract", type=Path, default=HERE / "grounded_gate_contract.json")
    parser.add_argument("--teacher", type=Path, default=HERE / "grounded-teacher.jsonl")
    parser.add_argument("--capture", type=Path, default=HERE / "grounded-states.cnob")
    parser.add_argument("--report", type=Path, default=HERE / "grounded_gate_report.json")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def corrected_nll(
    raw: np.ndarray,
    target_id: int,
    candidates: np.ndarray,
    delta: np.ndarray,
) -> tuple[float, float]:
    peak = max(float(np.max(raw)), float(np.max(raw[candidates] + delta)))
    scaled = np.exp(raw.astype(np.float64) - peak)
    total = float(np.sum(scaled))
    total -= float(np.sum(scaled[candidates]))
    total += float(np.sum(np.exp(raw[candidates].astype(np.float64) + delta - peak)))
    position = np.flatnonzero(candidates == target_id)
    target = float(raw[target_id])
    if len(position) == 1:
        target += float(delta[int(position[0])])
    return peak + math.log(total) - target, float(position[0]) if len(position) == 1 else -1.0


def main() -> int:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    tasks_path = resolve(contract["tasks"])
    weights_path = resolve(contract["candidate"]["weights"])
    if sha256_file(tasks_path) != contract["tasks_sha256"]:
        raise ValueError("独立任务SHA-256与冻结契约不一致")
    if sha256_file(weights_path) != contract["candidate"]["weights_sha256"]:
        raise ValueError("候选权重SHA-256与冻结契约不一致")
    tasks = read_jsonl(tasks_path)
    teacher = read_jsonl(args.teacher)
    task_by_id = {row["id"]: row for row in tasks}
    if len(tasks) != contract["task_count"] or len(task_by_id) != len(tasks):
        raise ValueError("独立任务数量或ID不符合冻结契约")
    if any(row["task_id"] not in task_by_id for row in teacher):
        raise ValueError("teacher引用了冻结任务之外的ID")
    teacher_manifest_path = args.teacher.with_suffix(args.teacher.suffix + ".manifest.json")
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    if (
        teacher_manifest["source_tasks_sha256"] != contract["tasks_sha256"]
        or teacher_manifest["teacher_sha256"] != sha256_file(args.teacher)
        or teacher_manifest["task_count"] != contract["task_count"]
        or teacher_manifest["max_target_tokens"] != contract["teacher"]["max_target_tokens"]
    ):
        raise ValueError("teacher清单与冻结独立门不一致")

    hidden, logits = load_capture(args.capture, len(teacher))
    hidden /= np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), np.float32(1e-8))
    with np.load(weights_path, allow_pickle=False) as package:
        left = np.asarray(package["left"], dtype=np.float32)
        right = np.asarray(package["right"], dtype=np.float32)
        static_ids = [int(value) for value in package["static_token_ids"]]
    rank = int(contract["candidate"]["rank"])
    if left.shape != (rank, 2048) or right.shape != (rank, 2048):
        raise ValueError("冻结权重形状错误")

    maximum = int(contract["candidate"]["maximum_dynamic_tokens"])
    candidate_ids: list[list[int]] = []
    union_ids: set[int] = set(static_ids)
    for row in teacher:
        dynamic = recent_unique(row["prefix_tokens"], maximum - len(static_ids))
        combined = list(static_ids)
        present = set(combined)
        for token in dynamic:
            if token not in present:
                present.add(token)
                combined.append(token)
        combined = combined[:maximum]
        candidate_ids.append(combined)
        union_ids.update(combined)

    model_path = ROOT / "fast16" / "models" / "ColorLM-v6-Q3Router-Fused-A1.gguf"
    reader = GGUFReader(os.fspath(model_path), "r")
    embedding_tensor = {tensor.name: tensor for tensor in reader.tensors}["token_embd.weight"]
    union = np.asarray(sorted(union_ids), dtype=np.int64)
    union_lookup = {int(token): index for index, token in enumerate(union)}
    embedding = np.asarray(
        dequantize(embedding_tensor.data[union], embedding_tensor.tensor_type),
        dtype=np.float32,
    )
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), np.float32(1e-8))
    hidden_rank = hidden @ left.T
    embedding_rank = embedding @ right.T
    correction_min = float(contract["candidate"]["correction_min"])
    correction_max = float(contract["candidate"]["correction_max"])

    evaluated: list[dict[str, Any]] = []
    by_task: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(teacher):
        ids = np.asarray(candidate_ids[index], dtype=np.int64)
        selected = np.asarray([union_lookup[int(token)] for token in ids], dtype=np.int64)
        delta = embedding_rank[selected] @ hidden_rank[index]
        delta = np.clip(delta, correction_min, correction_max).astype(np.float64)
        raw = logits[index]
        target_id = int(row["target_token_id"])
        peak = float(np.max(raw))
        native = peak + math.log(float(np.exp(raw.astype(np.float64) - peak).sum())) - float(raw[target_id])
        corrected, position = corrected_nll(raw, target_id, ids, delta)
        change = corrected - native
        task = task_by_id[row["task_id"]]
        by_task[row["task_id"]].append(change)
        evaluated.append(
            {
                "sample_id": row["sample_id"],
                "task_id": row["task_id"],
                "family": task["family"],
                "token_index": int(row["token_index"]),
                "target_token_id": target_id,
                "target_in_dynamic_pool": position >= 0,
                "candidate_count": len(ids),
                "native_target_nll": native,
                "corrected_target_nll": corrected,
                "target_nll_delta": change,
            }
        )

    per_task = [
        {
            "task_id": task_id,
            "family": task_by_id[task_id]["family"],
            "samples": len(values),
            "mean_target_nll_delta": float(np.mean(values)),
        }
        for task_id, values in sorted(by_task.items())
    ]
    mean_delta = float(np.mean([row["target_nll_delta"] for row in evaluated]))
    coverage = float(np.mean([row["target_in_dynamic_pool"] for row in evaluated]))
    wins = sum(row["mean_target_nll_delta"] < 0.0 for row in per_task)
    regressions = sum(row["mean_target_nll_delta"] > 0.0 for row in per_task)
    maximum_regression = max(row["mean_target_nll_delta"] for row in per_task)
    gate = contract["pass_gate"]
    passed = (
        mean_delta <= float(gate["mean_target_nll_delta_max"])
        and coverage >= float(gate["target_coverage_min"])
        and wins >= int(gate["task_wins_min"])
        and regressions <= int(gate["task_regressions_max"])
        and maximum_regression <= float(gate["maximum_task_mean_nll_regression"])
    )
    worst_samples = sorted(
        evaluated, key=lambda row: row["target_nll_delta"], reverse=True
    )[:12]
    report = {
        "format": "colorlm-v30-grounded-independent-gate-report-v1",
        "status": "passed_for_runtime_prototype" if passed else "stopped_before_runtime",
        "contract": args.contract.as_posix(),
        "contract_sha256": sha256_file(args.contract),
        "tasks_sha256": sha256_file(tasks_path),
        "teacher_sha256": sha256_file(args.teacher),
        "capture_sha256": sha256_file(args.capture),
        "weights_sha256": sha256_file(weights_path),
        "sample_count": len(evaluated),
        "task_count": len(per_task),
        "candidate_pool": {
            "static_tokens": len(static_ids),
            "union_tokens": len(union),
            "mean_tokens_per_sample": float(np.mean([len(ids) for ids in candidate_ids])),
            "target_coverage": coverage,
        },
        "metrics": {
            "mean_target_nll_delta": mean_delta,
            "task_wins": wins,
            "task_regressions": regressions,
            "maximum_task_mean_nll_regression": maximum_regression,
        },
        "per_task": per_task,
        "worst_samples": worst_samples,
        "gate_passed": passed,
        "decision": contract["success_action"] if passed else contract["failure_action"],
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
