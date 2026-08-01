"""区分oracle覆盖、自由滚动命中与v38验证接受长度的离线模拟器。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from draft_core import (
    CascadedBlockHead,
    DraftDataset,
    ShortlistConfig,
    build_shortlist,
    load_contract,
    sha256_file,
    token_keys,
    write_jsonl,
)


def sequence_is_complete(row: dict[str, Any], field: str) -> bool:
    tokens = row.get(field)
    if not isinstance(tokens, list) or not tokens:
        return False
    if len(tokens) == 4:
        return True
    terminated_field = "oracle_terminated" if field == "oracle_token_ids" else "validator_terminated"
    return bool(row.get(terminated_field, False))


def summarize_coverage(
    rows: list[dict[str, Any]],
    indices: list[int],
    candidates: list[list[int]],
    native_top_ids: np.ndarray,
    target_field: str,
) -> dict[str, Any]:
    token_count = 0
    covered_tokens = 0
    complete = 0
    available = 0
    first_token_matches = 0
    for index in indices:
        target = rows[index].get(target_field)
        if not isinstance(target, list) or not target:
            continue
        available += 1
        target = [int(value) for value in target]
        candidate_set = set(candidates[index])
        token_flags = [target[0] == int(native_top_ids[index, 0])]
        token_flags.extend(token in candidate_set for token in target[1:4])
        token_count += len(token_flags)
        covered_tokens += sum(token_flags)
        first_token_matches += int(token_flags[0])
        complete += int(sequence_is_complete(rows[index], target_field) and all(token_flags))
    return {
        "anchors": len(indices),
        "target_available_anchors": available,
        "target_availability": available / max(len(indices), 1),
        "tokens": token_count,
        "covered_tokens": covered_tokens,
        "token_coverage": covered_tokens / max(token_count, 1),
        "first_token_native_match_rate": first_token_matches / max(available, 1),
        "complete_trajectory_anchors": complete,
        "complete_trajectory_coverage": complete / max(available, 1),
        "warning": "这是anchor shortlist的oracle/validator必要覆盖，不是自由滚动接受率。",
    }


def group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"anchors": 0}
    acceptance = [int(item["accepted_draft_tokens"]) for item in items]
    matched_future = [int(item["matched_future_tokens"]) for item in items]
    candidate_hits = [int(item["validator_future_tokens_in_shortlist"]) for item in items]
    future_count = [int(item["future_tokens"]) for item in items]
    rejection = Counter(str(item["rejection_position"]) for item in items)
    return {
        "anchors": len(items),
        "mean_accepted_draft_tokens": float(np.mean(acceptance)),
        "accepted_draft_tokens_sum": sum(acceptance),
        "acceptance_histogram": {str(value): acceptance.count(value) for value in range(5)},
        "mean_matched_future_tokens": float(np.mean(matched_future)),
        "free_roll_validator_candidate_hit_rate": sum(candidate_hits) / max(sum(future_count), 1),
        "rejection_position_histogram": dict(sorted(rejection.items())),
    }


def coverage_gate(
    report_by_split: dict[str, dict[str, Any]], contract: dict[str, Any]
) -> tuple[bool, list[str]]:
    gate = contract["coverage_gate"]
    failures: list[str] = []
    for split in gate["required_splits"]:
        row = report_by_split.get(split)
        if not row:
            failures.append(f"缺少{split} split")
            continue
        for target_name in ("oracle", "v38_validator"):
            target = row[target_name]
            if target["target_availability"] < 1.0:
                failures.append(f"{split}.{target_name}完整目标不可用")
            if target["token_coverage"] < float(gate["minimum_token_coverage"]):
                failures.append(f"{split}.{target_name} token覆盖不足")
            if target["complete_trajectory_coverage"] < float(gate["minimum_complete_trajectory_coverage"]):
                failures.append(f"{split}.{target_name}完整轨迹覆盖不足")
    return not failures, failures


def simulate(
    dataset: DraftDataset,
    head: CascadedBlockHead,
    model_metadata: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shortlist_config = ShortlistConfig.from_contract(contract)
    frequent = [int(value) for value in model_metadata["frequent_train_token_ids"]]
    candidates = [
        build_shortlist(dataset.native_top_ids[index], row["recent_token_ids"], frequent, shortlist_config)
        for index, row in enumerate(dataset.rows)
    ]
    splits = sorted({str(row["split"]) for row in dataset.rows})
    coverage_by_split: dict[str, Any] = {}
    for split in splits:
        indices = [index for index, row in enumerate(dataset.rows) if row["split"] == split]
        coverage_by_split[split] = {
            "oracle": summarize_coverage(
                dataset.rows, indices, candidates, dataset.native_top_ids, "oracle_token_ids"
            ),
            "v38_validator": summarize_coverage(
                dataset.rows, indices, candidates, dataset.native_top_ids, "validator_token_ids"
            ),
        }
    passed, coverage_failures = coverage_gate(coverage_by_split, contract)
    base_report: dict[str, Any] = {
        "format": "colorlm-v47-parallel-draft-acceptance-report-v1",
        "scope": "offline-free-roll-replay",
        "model_architecture": model_metadata.get("format"),
        "first_token": "直接取anchor的v38原生top-1；head没有第1位参数或输出",
        "shortlist_mode": "NanoSpec-inspired-anchor-dynamic-tiny-vocabulary",
        "offline_mismatch_control": "Draft-OPD约束：oracle覆盖仅为必要条件；只有下方自由滚动拒绝位置可估接受长度",
        "block_mode": "FastEagle-inspired-one-cascaded-block-with-layerwise-supervision",
        "coverage": {"by_split": coverage_by_split, "gate_passed": passed, "failures": coverage_failures},
        "simulation_executed": False,
        "v38_verifier_proven": all(
            row.get("validator_source") == "v38-free-greedy-from-anchor" for row in dataset.rows
        ),
    }
    if not passed:
        base_report["decision"] = {
            "status": "stop_before_acceptance_prediction",
            "reason": "完整oracle与v38 validator轨迹的anchor shortlist覆盖未过门；禁止用teacher-forced局部覆盖预测接受长度。",
        }
        return base_report, []

    states = head.states(dataset.hidden)
    replay_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for record, row in enumerate(dataset.rows):
        validator = [int(value) for value in row["validator_token_ids"]]
        available_future = min(len(validator) - 1, head.future_positions)
        keys = token_keys(candidates[record], head.rank)
        future_proposals: list[int] = []
        for position in range(available_future):
            scores = keys @ states[record, position]
            future_proposals.append(int(candidates[record][int(np.argmax(scores))]))
        proposals = [int(dataset.native_top_ids[record, 0])] + future_proposals
        accepted = 0
        rejection_position: int | str = "none"
        for position, (proposal, target) in enumerate(zip(proposals, validator), start=1):
            if proposal != target:
                rejection_position = position
                break
            accepted += 1
        future_targets = validator[1 : 1 + available_future]
        candidate_set = set(candidates[record])
        candidate_hits = sum(target in candidate_set for target in future_targets)
        matched_future = sum(
            proposal == target for proposal, target in zip(future_proposals, future_targets)
        )
        detail = {
            "record": record,
            "anchor_id": row["anchor_id"],
            "split": row["split"],
            "context_bucket": row["context_bucket"],
            "proposals": proposals,
            "validator": validator[: len(proposals)],
            "accepted_draft_tokens": accepted,
            "future_tokens": available_future,
            "matched_future_tokens": matched_future,
            "validator_future_tokens_in_shortlist": candidate_hits,
            "rejection_position": rejection_position,
        }
        details.append(detail)
        if isinstance(rejection_position, int):
            pos = rejection_position - 1
            target = validator[pos]
            replay_rows.append(
                {
                    "format": "colorlm-v47-parallel-draft-error-replay-v1",
                    "record": record,
                    "anchor_id": row["anchor_id"],
                    "split": row["split"],
                    "rejection_position": rejection_position,
                    "proposed_token_id": proposals[pos],
                    "validator_token_id": target,
                    "validator_target_in_shortlist": target in candidate_set,
                    "proposal_prefix": proposals[: rejection_position],
                    "error_type": "ranking" if target in candidate_set else "shortlist_coverage",
                }
            )

    by_split = {
        split: group_summary([item for item in details if item["split"] == split]) for split in splits
    }
    buckets = sorted({str(item["context_bucket"]) for item in details})
    by_context = {
        bucket: group_summary([item for item in details if item["context_bucket"] == bucket])
        for bucket in buckets
    }
    evaluation = [item for item in details if item["split"] in contract["coverage_gate"]["required_splits"]]
    base_report.update(
        {
            "simulation_executed": True,
            "free_roll": {
                "candidate_policy": "候选集在anchor一次构造；模型预测不会被teacher token覆盖",
                "by_split": by_split,
                "by_context_bucket": by_context,
                "evaluation": group_summary(evaluation),
                "details": details,
            },
            "error_replay": {
                "records": len(replay_rows),
                "train_records": sum(row["split"] == "train" for row in replay_rows),
                "training_rule": "只允许train拒绝位置回灌；validation/test只报告",
            },
            "decision": {
                "status": "acceptance_predicted_needs_cost_gate",
                "mean_accepted_draft_tokens": group_summary(evaluation).get("mean_accepted_draft_tokens"),
                "reason": "完整覆盖已过，接受长度来自v38自由滚动逐token比较；仍须解析加速下界硬门。",
            },
        }
    )
    return base_report, replay_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-replay-output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.error_replay_output and args.error_replay_output.exists()):
        raise FileExistsError("拒绝覆盖已有回放产物")
    contract = load_contract(args.contract)
    dataset = DraftDataset.load(args.dataset)
    head, metadata = CascadedBlockHead.load(args.model)
    dataset_sha = sha256_file(args.dataset)
    if metadata.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("模型绑定的数据manifest SHA-256与回放数据不一致")
    if head.rank != int(contract["future_head"]["rank"]):
        raise ValueError("模型rank与冻结合同不一致")
    if head.future_positions != len(contract["future_head"]["positions"]):
        raise ValueError("模型未来位置数与冻结合同不一致")
    report, replay_rows = simulate(dataset, head, metadata, contract)
    report["dataset_manifest"] = str(args.dataset.resolve())
    report["dataset_manifest_sha256"] = dataset_sha
    report["model"] = str(args.model.resolve())
    report["model_sha256"] = sha256_file(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.error_replay_output:
        # 落盘只保留train，防止后续训练器误读已消费留出拒绝位置。
        write_jsonl(args.error_replay_output, (row for row in replay_rows if row["split"] == "train"))
    print(json.dumps({"output": str(args.output), "decision": report["decision"]}, ensure_ascii=False, indent=2))
    return 0 if report["simulation_executed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
