"""校验真实v38草稿数据契约，并先做oracle/validator完整轨迹shortlist必要覆盖。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from draft_core import DraftDataset, ShortlistConfig, build_shortlist, load_contract, train_frequent_ids


def target_coverage(
    dataset: DraftDataset,
    indices: list[int],
    candidates: list[list[int]],
    field: str,
) -> dict[str, Any]:
    tokens = covered = complete = available = 0
    for index in indices:
        target = dataset.rows[index].get(field)
        if not isinstance(target, list) or not target:
            continue
        available += 1
        values = [int(value) for value in target[:4]]
        flags = [values[0] == int(dataset.native_top_ids[index, 0])]
        candidate_set = set(candidates[index])
        flags.extend(value in candidate_set for value in values[1:])
        tokens += len(flags)
        covered += sum(flags)
        terminated = bool(
            dataset.rows[index]["oracle_terminated" if field == "oracle_token_ids" else "validator_terminated"]
        )
        complete += int((len(values) == 4 or terminated) and all(flags))
    return {
        "anchors": len(indices),
        "available": available,
        "availability": available / max(len(indices), 1),
        "tokens": tokens,
        "covered": covered,
        "token_coverage": covered / max(tokens, 1),
        "complete_trajectory_coverage": complete / max(available, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("拒绝覆盖已有数据校验报告")
    contract = load_contract(args.contract)
    dataset = DraftDataset.load(args.dataset)
    config = ShortlistConfig.from_contract(contract)
    frequent = train_frequent_ids(dataset.rows, config.train_frequent)
    candidates = [
        build_shortlist(dataset.native_top_ids[index], row["recent_token_ids"], frequent, config)
        for index, row in enumerate(dataset.rows)
    ]
    split_counts = Counter(str(row["split"]) for row in dataset.rows)
    context_counts = Counter(
        str(row["context_bucket"])
        for row in dataset.rows
        if row["split"] in contract["coverage_gate"]["required_splits"]
    )
    coverage = {}
    coverage_failures = []
    for split in ("train", "validation", "test"):
        indices = [index for index, row in enumerate(dataset.rows) if row["split"] == split]
        coverage[split] = {
            "oracle": target_coverage(dataset, indices, candidates, "oracle_token_ids"),
            "v38_validator": target_coverage(dataset, indices, candidates, "validator_token_ids"),
        }
        if split in contract["coverage_gate"]["required_splits"]:
            for name, metrics in coverage[split].items():
                if metrics["availability"] < 1.0:
                    coverage_failures.append(f"{split}.{name}.availability")
                if metrics["token_coverage"] < float(contract["coverage_gate"]["minimum_token_coverage"]):
                    coverage_failures.append(f"{split}.{name}.token_coverage")
                if metrics["complete_trajectory_coverage"] < float(
                    contract["coverage_gate"]["minimum_complete_trajectory_coverage"]
                ):
                    coverage_failures.append(f"{split}.{name}.complete_trajectory_coverage")
    evaluation_count = split_counts["validation"] + split_counts["test"]
    contract_gates = {
        "v38_anchor_capture": dataset.manifest["base_model"] == "ColorLM-v38-Qwen36-Shared-Sequence-Policy",
        "train_validation_test_present": all(split_counts[split] > 0 for split in ("train", "validation", "test")),
        "full_oracle_token_sequences": all(
            isinstance(row.get("oracle_token_ids"), list) and row["oracle_token_ids"] for row in dataset.rows
        ),
        "v38_free_roll_validator_sequences": all(
            row["validator_source"] == "v38-free-greedy-from-anchor" for row in dataset.rows
        ),
        "minimum_evaluation_anchors": evaluation_count >= int(contract["acceptance_gate"]["minimum_evaluation_anchors"]),
        "short_medium_long_context_buckets": set(context_counts) >= {"lt_2k", "2k_8k", "gt_8k"},
        "complete_trajectory_shortlist_gate": not coverage_failures,
    }
    passed = all(contract_gates.values())
    report = {
        "format": "colorlm-v47-parallel-draft-dataset-audit-v1",
        "dataset": str(args.dataset.resolve()),
        "split_counts": dict(split_counts),
        "evaluation_anchors": evaluation_count,
        "evaluation_context_buckets": dict(context_counts),
        "train_frequent_token_ids": frequent,
        "candidate_count": {
            "mean": sum(map(len, candidates)) / len(candidates),
            "maximum": max(map(len, candidates)),
        },
        "coverage": coverage,
        "coverage_failures": coverage_failures,
        "contract_gates": contract_gates,
        "passed": passed,
        "decision": "may_train_then_free_roll" if passed else "stop_before_training",
        "claim_limit": "oracle/validator shortlist覆盖只是训练资格，不是head自由滚动命中、v38接受长度或加速。"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": report["decision"], "gates": contract_gates}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
