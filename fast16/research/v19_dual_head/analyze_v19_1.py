"""Compare paired v17 and v19.1 exact NLL shards at task level."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"duplicate sample_id at {path}:{line_number}: {sample_id}")
        if row.get("exact") is not True:
            raise ValueError(f"non-exact sample at {path}:{line_number}")
        rows[sample_id] = row
    if not rows:
        raise ValueError(f"empty shard: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_jsonl(args.baseline)
    candidate = read_jsonl(args.candidate)
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(f"sample mismatch: missing={missing}, extra={extra}")

    task_ids = sorted({str(row["task_id"]) for row in baseline.values()})
    tasks: list[dict[str, Any]] = []
    for task_id in task_ids:
        sample_ids = [
            sample_id
            for sample_id, row in baseline.items()
            if row["task_id"] == task_id
        ]
        deltas = [
            float(candidate[sample_id]["target_nll"])
            - float(baseline[sample_id]["target_nll"])
            for sample_id in sample_ids
        ]
        tasks.append(
            {
                "task": task_id,
                "group": "tool" if task_id.startswith("tool-") else "code",
                "sample_count": len(deltas),
                "mean_delta": statistics.mean(deltas),
                "median_delta": statistics.median(deltas),
                "improved_tokens": sum(delta < 0 for delta in deltas),
                "regressed_tokens": sum(delta > 0 for delta in deltas),
                "equal_tokens": sum(delta == 0 for delta in deltas),
                "worst_regression": max(deltas),
                "best_improvement": min(deltas),
            }
        )

    all_deltas = [
        float(candidate[sample_id]["target_nll"])
        - float(baseline[sample_id]["target_nll"])
        for sample_id in baseline
    ]
    task_means = [float(task["mean_delta"]) for task in tasks]
    groups: dict[str, dict[str, Any]] = {}
    for group in ("code", "tool"):
        group_tasks = [task for task in tasks if task["group"] == group]
        group_means = [float(task["mean_delta"]) for task in group_tasks]
        groups[group] = {
            "task_count": len(group_tasks),
            "mean_task_delta": statistics.mean(group_means),
            "median_task_delta": statistics.median(group_means),
            "task_wins": sum(delta < 0 for delta in group_means),
            "task_losses": sum(delta > 0 for delta in group_means),
            "task_equal": sum(delta == 0 for delta in group_means),
        }

    loto = []
    for left_out in tasks:
        remaining = [
            float(task["mean_delta"])
            for task in tasks
            if task["task"] != left_out["task"]
        ]
        loto.append(
            {
                "left_out": left_out["task"],
                "remaining_task_mean_delta": statistics.mean(remaining),
            }
        )

    result = {
        "format": "colorlm-v19.1-paired-task-comparison-v1",
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "sample_count": len(baseline),
        "task_count": len(tasks),
        "overall_mean_delta": statistics.mean(all_deltas),
        "overall_median_delta": statistics.median(all_deltas),
        "token_wins": sum(delta < 0 for delta in all_deltas),
        "token_losses": sum(delta > 0 for delta in all_deltas),
        "token_equal": sum(delta == 0 for delta in all_deltas),
        "task_wins": sum(delta < 0 for delta in task_means),
        "task_losses": sum(delta > 0 for delta in task_means),
        "task_equal": sum(delta == 0 for delta in task_means),
        "worst_task": max(tasks, key=lambda task: float(task["mean_delta"])),
        "best_task": min(tasks, key=lambda task: float(task["mean_delta"])),
        "groups": groups,
        "tasks": tasks,
        "leave_one_task_out": loto,
        "all_loto_positive_direction": all(
            row["remaining_task_mean_delta"] < 0 for row in loto
        ),
        "smoke_only": True,
        "decision_note": (
            "The first six consecutive target tokens per task are only a smoke test. "
            "Promotion also requires preselected decision-token and generation gates."
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
