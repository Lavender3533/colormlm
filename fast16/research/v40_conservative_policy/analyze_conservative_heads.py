"""Compare conservative v36 policy-head combinations on the consumed v29 dev set.

This script is diagnostic only.  It deliberately labels the old 20-task set as
development data; any v40 advancement requires a newly frozen blind set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_fit_module(path: Path):
    spec = importlib.util.spec_from_file_location("colorlm_v29_fit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载拟合实现: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_weights(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    with np.load(path, allow_pickle=False) as package:
        return (
            package["token_ids"].astype(np.int64),
            package["weight"].astype(np.float64),
            package["bias"].astype(np.float64),
            float(package["correction_min"][0]),
            float(package["correction_max"][0]),
        )


def summarize(
    name: str,
    correction: np.ndarray,
    logits: list[np.ndarray],
    teacher: list[dict[str, Any]],
    task_by_id: dict[str, dict[str, Any]],
    token_ids: np.ndarray,
    fit_module,
) -> dict[str, Any]:
    lookup = {int(token): index for index, token in enumerate(token_ids)}
    rows: list[dict[str, Any]] = []
    by_task: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(teacher):
        target_id = int(row["target_token_id"])
        native = fit_module.exact_nll(logits[index], target_id)
        corrected = fit_module.corrected_nll(
            logits[index], target_id, token_ids, correction[index], lookup
        )
        delta = float(corrected - native)
        task = task_by_id[row["task_id"]]
        rows.append(
            {
                "task_id": row["task_id"],
                "split": task["split"],
                "target_is_candidate": target_id in lookup,
                "delta": delta,
            }
        )
        by_task[row["task_id"]].append(delta)

    task_rows = [
        {
            "task_id": task_id,
            "split": task_by_id[task_id]["split"],
            "family": task_by_id[task_id]["family"],
            "mean_target_nll_delta": float(np.mean(values)),
        }
        for task_id, values in sorted(by_task.items())
    ]
    candidate_rows = [row for row in rows if row["target_is_candidate"]]
    return {
        "name": name,
        "role": "consumed-development-only",
        "mean_target_nll_delta": float(np.mean([row["delta"] for row in rows])),
        "candidate_win_rate": float(np.mean([row["delta"] < 0 for row in candidate_rows])),
        "improved_tasks": sum(row["mean_target_nll_delta"] < 0 for row in task_rows),
        "regressed_tasks": sum(row["mean_target_nll_delta"] > 0 for row in task_rows),
        "worst_task_mean_nll_regression": max(
            row["mean_target_nll_delta"] for row in task_rows
        ),
        "mean_abs_correction": float(np.mean(np.abs(correction))),
        "active_fraction": float(np.mean(np.abs(correction) > 1e-8)),
        "per_task": task_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="v40保守策略头开发集诊断")
    parser.add_argument("--fit-implementation", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--old-weights", type=Path, required=True)
    parser.add_argument("--native-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fit_module = load_fit_module(args.fit_implementation)
    capture = fit_module.load_capture(args.capture)
    teacher = read_jsonl(args.teacher)
    tasks = read_jsonl(args.tasks)
    task_by_id = {row["id"]: row for row in tasks}
    if sorted(capture) != list(range(len(teacher))):
        raise ValueError("CNOB与teacher数量不一致")

    old_ids, old_weight, old_bias, old_min, old_max = load_weights(args.old_weights)
    native_ids, native_weight, native_bias, native_min, native_max = load_weights(
        args.native_weights
    )
    if not np.array_equal(old_ids, native_ids):
        raise ValueError("旧头与原生头候选token不一致")

    hidden = np.stack(
        [capture[index][fit_module.BASE_HIDDEN] for index in range(len(teacher))]
    ).astype(np.float64)
    hidden /= np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), 1e-8)
    logits = [capture[index][fit_module.BASE_LOGITS] for index in range(len(teacher))]

    old = np.clip(hidden @ old_weight.T + old_bias, old_min, old_max)
    native = np.clip(
        hidden @ native_weight.T + native_bias, native_min, native_max
    )
    anchored = np.clip(0.75 * old + 0.25 * native, old_min, old_max)
    norm_ratio = float(np.linalg.norm(old_weight) / np.linalg.norm(native_weight))
    norm_matched_native = np.clip(norm_ratio * native, old_min, old_max)
    consensus = np.where(
        (old > 0.0) & (native > 0.0), np.minimum(old, native), 0.0
    )

    candidates = {
        "v38_old_cross_backbone": old,
        "v39_native_failed": native,
        "anchored_75_old_25_native": anchored,
        "native_weight_norm_matched": norm_matched_native,
        "positive_consensus_min": consensus,
    }
    summaries = [
        summarize(name, correction, logits, teacher, task_by_id, old_ids, fit_module)
        for name, correction in candidates.items()
    ]
    report = {
        "format": "colorlm-v40-conservative-policy-dev-analysis-v1",
        "warning": "The v29 20-task set is consumed development data, not v40 holdout evidence.",
        "sample_count": len(teacher),
        "task_count": len(tasks),
        "candidate_token_count": int(len(old_ids)),
        "old_weight_l2": float(np.linalg.norm(old_weight)),
        "native_weight_l2": float(np.linalg.norm(native_weight)),
        "native_norm_match_ratio": norm_ratio,
        "candidates": summaries,
        "advancement_rule": "choose at most one rule here, then freeze a newly authored blind task set before runtime evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
