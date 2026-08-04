#!/usr/bin/env python3
"""独立揭示 sealed holdout，验收 Frame 的泛化、置换、删器官与干预。"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from axis_core import (
    CompositeAxis,
    accuracy_of,
    fit_best_primitive,
    read_json,
    schema_from_snapshot,
    sha256_file,
    synthesize_axis,
    write_json,
)


def _features(row: dict[str, object]) -> dict[str, float]:
    raw = row["features"]
    assert isinstance(raw, dict)
    return {str(key): float(value) for key, value in raw.items()}


def _outcome(row: dict[str, object]) -> bool:
    value = row["outcome"]
    if not isinstance(value, bool):
        raise ValueError("outcome 无效")
    return value


def _accuracy(predictions: list[bool], rows: list[dict[str, object]]) -> float:
    if len(predictions) != len(rows) or not rows:
        raise ValueError("预测长度无效")
    return sum(pred == _outcome(row) for pred, row in zip(predictions, rows)) / len(rows)


def _majority(discovery_rows: list[dict[str, object]]) -> bool:
    return sum(_outcome(row) for row in discovery_rows) * 2 >= len(discovery_rows)


def _identity_memory_accuracy(
    discovery_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> float:
    default = _majority(discovery_rows)
    memory = {
        tuple(sorted(_features(row).items())): _outcome(row)
        for row in discovery_rows
    }
    predictions = [memory.get(tuple(sorted(_features(row).items())), default) for row in holdout_rows]
    return _accuracy(predictions, holdout_rows)


def _random_accuracy(rows: list[dict[str, object]]) -> float:
    predictions = [
        bool(int(hashlib.sha256(str(row["episode_id"]).encode("utf-8")).hexdigest(), 16) & 1)
        for row in rows
    ]
    return _accuracy(predictions, rows)


def _permuted_axis_metrics(
    schema: dict[str, tuple[str, str]],
    discovery_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
    discovery_sha: str,
    *,
    repeats: int = 64,
) -> dict[str, object]:
    """多次置换；0% 与 100% 都代表携带完整目标信息，不能只设上界。"""
    base_labels = [_outcome(row) for row in discovery_rows]
    rng = random.Random(int(discovery_sha[:16], 16))
    trials = []
    for repeat in range(repeats):
        labels = list(base_labels)
        rng.shuffle(labels)
        permuted = [
            {**row, "outcome": label}
            for row, label in zip(discovery_rows, labels)
        ]
        fit = synthesize_axis(schema, permuted)
        correct, total = accuracy_of(fit.axis, holdout_rows)
        accuracy = correct / total
        trials.append(
            {
                "repeat": repeat,
                "accuracy": accuracy,
                "information_accuracy": max(accuracy, 1.0 - accuracy),
                "axis": fit.axis.canonical,
            }
        )
    information_values = [float(trial["information_accuracy"]) for trial in trials]
    candidate_count = synthesize_axis(schema, discovery_rows).candidate_count
    # “信息准确率>=90%”同时把真实函数和其取反视为偶然恢复，因此机会率为 2/N。
    chance_recovery_rate = min(2.0 / candidate_count, 1.0)
    standard_error = (
        chance_recovery_rate * (1.0 - chance_recovery_rate) / repeats
    ) ** 0.5
    recovery_rate_limit_3sigma = min(
        chance_recovery_rate + 3.0 * standard_error,
        1.0,
    )
    return {
        "repeats": repeats,
        "unique_candidate_function_count": candidate_count,
        "mean_accuracy": sum(float(trial["accuracy"]) for trial in trials) / repeats,
        "mean_information_accuracy": sum(information_values) / repeats,
        "target_recovery_rate": sum(value >= 0.90 for value in information_values) / repeats,
        "chance_recovery_rate": chance_recovery_rate,
        "recovery_rate_limit_3sigma": recovery_rate_limit_3sigma,
        "trials": trials,
    }


def _leave_one_organ_out(
    axis: CompositeAxis,
    schema: dict[str, tuple[str, str]],
    discovery_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> dict[str, float]:
    neutral: dict[str, float] = {}
    for organ, names in schema.items():
        values = [
            value
            for row in discovery_rows
            for name, value in _features(row).items()
            if name in names
        ]
        neutral[organ] = sum(values) / len(values)

    result = {}
    for organ in axis.required_organs:
        masked_rows = []
        for row in holdout_rows:
            features = _features(row)
            for name in schema[organ]:
                features[name] = neutral[organ]
            masked_rows.append({**row, "features": features})
        correct, total = accuracy_of(axis, masked_rows)
        result[organ] = correct / total
    return result


def _intervention_accuracy(
    axis: CompositeAxis,
    interventions: list[dict[str, object]],
) -> float:
    checks = 0
    correct = 0
    for pair in interventions:
        before = pair["before"]
        after = pair["after"]
        assert isinstance(before, dict) and isinstance(after, dict)
        before_prediction = axis.evaluate(_features(before))
        after_prediction = axis.evaluate(_features(after))
        before_expected = _outcome(before)
        after_expected = _outcome(after)
        correct += before_prediction == before_expected
        correct += after_prediction == after_expected
        correct += (before_prediction != after_prediction) == (before_expected != after_expected)
        checks += 3
    return correct / checks


def evaluate(
    *,
    discovery_path: Path,
    frame_path: Path,
    manifest_path: Path,
    holdout_path: Path,
    output_path: Path,
) -> dict[str, object]:
    discovery = read_json(discovery_path)
    frame = read_json(frame_path)
    manifest = read_json(manifest_path)
    holdout = read_json(holdout_path)
    if sha256_file(discovery_path) != manifest.get("discovery_sha256"):
        raise ValueError("discovery SHA 与封存清单不符")
    if sha256_file(holdout_path) != manifest.get("sealed_holdout_sha256"):
        raise ValueError("holdout SHA 与封存清单不符")
    if frame.get("discovery_sha256") != manifest.get("discovery_sha256"):
        raise ValueError("Frame 不是由当前 discovery 冻结得到")
    if frame.get("format") != "polaris-blind-composite-frame-v1":
        raise ValueError("Frame 格式错误")

    schema = schema_from_snapshot(discovery.get("schema"))
    discovery_rows = discovery.get("episodes")
    holdout_rows = holdout.get("episodes")
    interventions = holdout.get("interventions")
    if not isinstance(discovery_rows, list) or not isinstance(holdout_rows, list):
        raise ValueError("episode 列表无效")
    if not isinstance(interventions, list):
        raise ValueError("intervention 列表无效")
    axis_snapshot = frame.get("axis")
    if not isinstance(axis_snapshot, dict):
        raise ValueError("Frame axis 无效")
    axis = CompositeAxis.from_snapshot(axis_snapshot)

    correct, total = accuracy_of(axis, holdout_rows)
    holdout_accuracy = correct / total
    majority = _majority(discovery_rows)
    majority_accuracy = _accuracy([majority] * len(holdout_rows), holdout_rows)
    primitive_fit = fit_best_primitive(schema, discovery_rows)
    primitive_predictions = [primitive_fit.predict(_features(row)) for row in holdout_rows]
    best_single_accuracy = _accuracy(primitive_predictions, holdout_rows)
    identity_accuracy = _identity_memory_accuracy(discovery_rows, holdout_rows)
    random_accuracy = _random_accuracy(holdout_rows)
    permutation = _permuted_axis_metrics(
        schema,
        discovery_rows,
        holdout_rows,
        str(manifest["discovery_sha256"]),
    )
    leave_one = _leave_one_organ_out(axis, schema, discovery_rows, holdout_rows)
    intervention_accuracy = _intervention_accuracy(axis, interventions)
    minimum_leave_one_drop = min(holdout_accuracy - value for value in leave_one.values())

    assertions = {
        "sealed_hashes_match": True,
        "frame_frozen_from_current_discovery": True,
        "frame_uses_at_least_two_organs": len(axis.required_organs) >= 2,
        "heldout_accuracy_at_least_98pct": holdout_accuracy >= 0.98,
        "best_single_organ_at_most_62pct": best_single_accuracy <= 0.62,
        "joint_gain_over_single_at_least_30pp": holdout_accuracy - best_single_accuracy >= 0.30,
        "permuted_labels_have_low_mean_information": permutation["mean_information_accuracy"] <= 0.62,
        "permuted_labels_do_not_exceed_chance_recovery_bound": (
            permutation["target_recovery_rate"]
            <= permutation["recovery_rate_limit_3sigma"]
        ),
        "leave_one_required_organ_drops_at_least_30pp": minimum_leave_one_drop >= 0.30,
        "intervention_consistency_at_least_98pct": intervention_accuracy >= 0.98,
    }
    private_audit = holdout.get("private_audit")
    receipt = {
        "format": "polaris-blind-composite-evaluation-v1",
        "passed": all(assertions.values()),
        "assertions": assertions,
        "frame": axis.snapshot(),
        "metrics": {
            "holdout_accuracy": holdout_accuracy,
            "best_single_organ_accuracy": best_single_accuracy,
            "joint_gain_over_single": holdout_accuracy - best_single_accuracy,
            "majority_accuracy": majority_accuracy,
            "identity_memory_accuracy": identity_accuracy,
            "random_partition_accuracy": random_accuracy,
            "permuted_label_metrics": permutation,
            "leave_one_organ_out_accuracy": leave_one,
            "minimum_leave_one_organ_drop": minimum_leave_one_drop,
            "intervention_consistency": intervention_accuracy,
        },
        "provenance": {
            "discovery_sha256": manifest["discovery_sha256"],
            "sealed_holdout_sha256": manifest["sealed_holdout_sha256"],
            "frame_sha256": sha256_file(frame_path),
            "holdout_revealed_only_by_evaluator": True,
            "observed_copied_from_frame_expected": False,
        },
        "private_audit_revealed_after_scoring": private_audit,
        "truth_boundary": {
            "uses_synthetic_world": True,
            "proves_model_intelligence": False,
            "proves_new_coordinate_outside_search_grammar": False,
            "tests_unseen_composite_relation_generalization": True,
        },
    }
    write_json(output_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = evaluate(
        discovery_path=args.discovery,
        frame_path=args.frame,
        manifest_path=args.manifest,
        holdout_path=args.holdout,
        output_path=args.output,
    )
    print("pass" if receipt["passed"] else "fail")


if __name__ == "__main__":
    main()
