"""Diagnose v19.1 smoke-set regressions without evaluating a new candidate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            row = json.loads(raw)
            result[str(row["sample_id"])] = row
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["delta"]) for row in rows]
    if not deltas:
        return {
            "count": 0,
            "mean_delta": None,
            "median_delta": None,
            "improved": 0,
            "regressed": 0,
            "equal": 0,
            "worst_regression": None,
            "best_improvement": None,
        }
    return {
        "count": len(rows),
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "improved": sum(delta < 0 for delta in deltas),
        "regressed": sum(delta > 0 for delta in deltas),
        "equal": sum(delta == 0 for delta in deltas),
        "worst_regression": max(deltas),
        "best_improvement": min(deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--donor-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_jsonl(args.baseline)
    candidate = read_jsonl(args.candidate)
    if set(baseline) != set(candidate):
        raise ValueError("paired shard sample IDs differ")

    donor_to_base = np.fromfile(args.donor_map, dtype="<i4")
    mapped_base_ids = set(int(value) for value in donor_to_base if value >= 0)
    paired: list[dict[str, Any]] = []
    for sample_id, base_row in baseline.items():
        candidate_row = candidate[sample_id]
        target_id = int(base_row["target_token_id"])
        paired.append(
            {
                "sample_id": sample_id,
                "task_id": str(base_row["task_id"]),
                "group": "tool" if str(base_row["task_id"]).startswith("tool-") else "code",
                "token_index": int(base_row["token_index"]),
                "target_token_id": target_id,
                "target_is_mapped": target_id in mapped_base_ids,
                "baseline_nll": float(base_row["target_nll"]),
                "candidate_nll": float(candidate_row["target_nll"]),
                "delta": float(candidate_row["target_nll"]) - float(base_row["target_nll"]),
            }
        )

    mapped_rows = [row for row in paired if row["target_is_mapped"]]
    unmapped_rows = [row for row in paired if not row["target_is_mapped"]]
    by_index = {
        str(index): summarize([row for row in paired if row["token_index"] == index])
        for index in sorted({int(row["token_index"]) for row in paired})
    }
    largest_regressions = sorted(paired, key=lambda row: float(row["delta"]), reverse=True)[:12]
    largest_improvements = sorted(paired, key=lambda row: float(row["delta"]))[:12]
    result = {
        "format": "colorlm-v19.1-smoke-diagnostic-v1",
        "candidate_point": 0.03,
        "candidate_status": "failed_smoke_point",
        "overall": summarize(paired),
        "mapped_targets": summarize(mapped_rows),
        "unmapped_targets": summarize(unmapped_rows),
        "by_token_index": by_index,
        "largest_regressions": largest_regressions,
        "largest_improvements": largest_improvements,
        "interpretation_rule": (
            "This report diagnoses the failed smoke point only. It must not be used "
            "to cherry-pick decision tokens, which were frozen separately before any new candidate."
        ),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
