"""纯 CPU 核验 v43 运行包布局并回放最终 F32 离线报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
F32 = 0
BASE_LOGITS = 1
BASE_HIDDEN = 4
VOCAB = 248320
HIDDEN = 2048


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / np.sum(values)


def stable_softmax_f32(logits: np.ndarray) -> np.ndarray:
    shifted = (logits - np.max(logits)).astype(np.float32)
    values = np.exp(shifted).astype(np.float32)
    return (values / np.sum(values, dtype=np.float32)).astype(np.float32)


def exact_nll(raw: np.ndarray, target_id: int) -> float:
    peak = float(np.max(raw))
    return peak + math.log(float(np.exp(raw.astype(np.float64) - peak).sum())) - float(raw[target_id])


def corrected_nll(
    raw: np.ndarray,
    target_id: int,
    candidate_ids: np.ndarray,
    delta: np.ndarray,
    candidate_lookup: dict[int, int],
) -> float:
    native_rows = raw[candidate_ids].astype(np.float64)
    corrected_rows = native_rows + delta
    peak = max(float(np.max(raw)), float(np.max(corrected_rows)))
    total = float(np.exp(raw.astype(np.float64) - peak).sum())
    total -= float(np.exp(native_rows - peak).sum())
    total += float(np.exp(corrected_rows - peak).sum())
    target_logit = float(raw[target_id])
    position = candidate_lookup.get(target_id)
    if position is not None:
        target_logit += float(delta[position])
    return peak + math.log(total) - target_logit


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    task_values: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        task_values[row["task_id"]].append(float(row["target_nll_delta"]))
    task_means = {task: float(np.mean(values)) for task, values in task_values.items()}
    deltas = np.asarray([row["target_nll_delta"] for row in selected], dtype=np.float64)
    return {
        "sample_count": len(selected),
        "task_count": len(task_values),
        "mean_target_nll_delta": float(np.mean(deltas)),
        "token_wins": int(np.sum(deltas < -1e-12)),
        "token_losses": int(np.sum(deltas > 1e-12)),
        "token_equal": int(np.sum(np.abs(deltas) <= 1e-12)),
        "task_wins": sum(value < -1e-12 for value in task_means.values()),
        "task_losses": sum(value > 1e-12 for value in task_means.values()),
        "task_equal": sum(abs(value) <= 1e-12 for value in task_means.values()),
        "worst_task_mean_nll_regression": max(task_means.values()),
        "exact_no_op_rate": float(np.mean([row["exact_no_op"] for row in selected])),
        "class_accuracy": float(np.mean([row["class_correct"] for row in selected])),
        "per_task": task_means,
    }


def compare_metrics(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, float, list[str]]:
    errors: list[str] = []
    maximum = 0.0
    exact_fields = (
        "sample_count",
        "task_count",
        "token_wins",
        "token_losses",
        "token_equal",
        "task_wins",
        "task_losses",
        "task_equal",
    )
    numeric_fields = (
        "mean_target_nll_delta",
        "worst_task_mean_nll_regression",
        "exact_no_op_rate",
        "class_accuracy",
    )
    for name in exact_fields:
        if actual[name] != expected[name]:
            errors.append(f"{name}: actual={actual[name]!r}, expected={expected[name]!r}")
    for name in numeric_fields:
        difference = abs(float(actual[name]) - float(expected[name]))
        maximum = max(maximum, difference)
        if difference > 1e-12:
            errors.append(f"{name}: abs_diff={difference:.17g}")
    if set(actual["per_task"]) != set(expected["per_task"]):
        errors.append("per_task 键集合不一致")
    else:
        for task_id, value in actual["per_task"].items():
            difference = abs(float(value) - float(expected["per_task"][task_id]))
            maximum = max(maximum, difference)
            if difference > 1e-12:
                errors.append(f"per_task[{task_id}]: abs_diff={difference:.17g}")
    return not errors, maximum, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=HERE / "runtime-v1")
    parser.add_argument("--weights", type=Path, default=HERE / "policy-weights-f32.npz")
    parser.add_argument("--report", type=Path, default=HERE / "policy-report-f32.json")
    parser.add_argument("--tasks", type=Path, default=HERE / "trajectory_tasks_v1.jsonl")
    parser.add_argument("--teacher", type=Path, default=HERE / "teacher.jsonl")
    parser.add_argument("--capture", type=Path, default=HERE / "v36-states.cnob")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.runtime / "policy.json"
    blob_path = args.runtime / "weights.bin"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_report = json.loads(args.report.read_text(encoding="utf-8"))
    tasks = read_jsonl(args.tasks)
    teacher = read_jsonl(args.teacher)
    task_by_id = {row["id"]: row for row in tasks}

    blob = blob_path.read_bytes()
    records = {row["name"]: row for row in manifest["tensors"]}
    expected_names = {
        "policy.base_ids",
        "policy.hidden_mean",
        "policy.pca_components",
        "policy.classifier",
        "policy.classifier_bias",
        "policy.correction_strength",
    }
    layout_errors: list[str] = []
    if set(records) != expected_names or len(records) != int(manifest["tensor_count"]):
        layout_errors.append("运行包张量名或数量不一致")
    if len(blob) != int(manifest["weights"]["bytes"]) or sha256_bytes(blob) != manifest["weights"]["sha256"]:
        layout_errors.append("weights.bin 大小或总 SHA-256 不一致")
    if sha256_file(args.weights) != manifest["source_weights_sha256"]:
        layout_errors.append("NPZ SHA-256 与运行包来源声明不一致")
    if sha256_file(args.report) != manifest["source_report_sha256"]:
        layout_errors.append("F32 报告 SHA-256 与运行包来源声明不一致")

    previous_end = 0
    payloads: dict[str, bytes] = {}
    for record in sorted(manifest["tensors"], key=lambda row: int(row["offset"])):
        offset = int(record["offset"])
        size = int(record["bytes"])
        if offset % int(manifest["alignment"]) != 0:
            layout_errors.append(f"{record['name']} 未按64字节对齐")
        if offset < previous_end or offset + size > len(blob):
            layout_errors.append(f"{record['name']} 偏移重叠或越界")
            continue
        if any(blob[previous_end:offset]):
            layout_errors.append(f"{record['name']} 前置padding不是全零")
        payload = blob[offset : offset + size]
        payloads[record["name"]] = payload
        if sha256_bytes(payload) != record["sha256"]:
            layout_errors.append(f"{record['name']} payload SHA-256不一致")
        previous_end = offset + size
    if previous_end != len(blob):
        layout_errors.append("weights.bin 末端存在未声明字节")

    rank = int(manifest["pca_rank"])
    candidate_count = int(manifest["candidate_token_count"])
    class_count = int(manifest["class_count"])
    base_ids = np.frombuffer(payloads["policy.base_ids"], dtype="<i8").copy()
    hidden_mean = np.frombuffer(payloads["policy.hidden_mean"], dtype="<f4").copy()
    pca = np.frombuffer(payloads["policy.pca_components"], dtype="<f4").reshape(rank, HIDDEN).copy()
    classifier_weight = np.frombuffer(payloads["policy.classifier"], dtype="<f4").reshape(class_count, rank).copy()
    classifier_bias = np.frombuffer(payloads["policy.classifier_bias"], dtype="<f4").copy()
    strength = np.frombuffer(payloads["policy.correction_strength"], dtype="<f4").copy()

    with np.load(args.weights, allow_pickle=False) as package:
        npz = {name: package[name].copy() for name in package.files}
    tensor_checks = {
        "candidate_ids_bitwise": bool(np.array_equal(base_ids, npz["candidate_ids"])),
        "hidden_mean_bitwise": bool(np.array_equal(hidden_mean, npz["hidden_mean"])),
        "pca_components_bitwise": bool(np.array_equal(pca, npz["pca_components"])),
        "classifier_weight_transpose_bitwise": bool(
            np.array_equal(classifier_weight, npz["classifier"][1:, :].T)
        ),
        "classifier_bias_bitwise": bool(np.array_equal(classifier_bias, npz["classifier"][0, :])),
        "correction_strength_bitwise": bool(np.array_equal(strength, npz["correction_strength"])),
    }
    pca_direction_cosines = np.sum(
        pca.astype(np.float64) * npz["pca_components"].astype(np.float64), axis=1
    ) / np.maximum(
        np.linalg.norm(pca.astype(np.float64), axis=1)
        * np.linalg.norm(npz["pca_components"].astype(np.float64), axis=1),
        1e-30,
    )
    pca_orthogonality_error = float(
        np.max(np.abs(pca.astype(np.float64) @ pca.astype(np.float64).T - np.eye(rank)))
    )
    if not all(tensor_checks.values()):
        layout_errors.extend(name for name, passed in tensor_checks.items() if not passed)
    if base_ids.shape != (candidate_count,) or hidden_mean.shape != (HIDDEN,):
        layout_errors.append("ID或hidden_mean解码形状错误")
    if pca.shape != (rank, HIDDEN) or classifier_weight.shape != (class_count, rank):
        layout_errors.append("PCA或classifier解码形状错误")
    if classifier_bias.shape != (class_count,) or strength.shape != (1,):
        layout_errors.append("classifier_bias或strength解码形状错误")

    candidate_lookup = {int(token): index for index, token in enumerate(base_ids)}
    evaluated: list[dict[str, Any]] = []
    always_active_evaluated: list[dict[str, Any]] = []
    formula_errors: list[str] = []
    classifier_logit_max_abs_difference = 0.0
    f32_predicted_class_mismatches = 0
    f32_no_op_gate_mismatches = 0
    f32_probability_max_abs_difference = 0.0
    minimum_argmax_margin = math.inf
    minimum_no_op_boundary_margin = math.inf
    record_index = 0
    pending: tuple[int, np.ndarray, np.ndarray, np.ndarray, int, bool, bool] | None = None
    with args.capture.open("rb") as source:
        while True:
            encoded = source.read(HEADER.size)
            if not encoded:
                break
            if len(encoded) != HEADER.size:
                raise ValueError("CNOB头被截断")
            magic, version, kind, record, dtype, reserved, *tail = HEADER.unpack(encoded)
            ne = tuple(tail[:4])
            payload_bytes = int(tail[4])
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError("CNOB payload被截断")
            if magic != MAGIC or version != 1 or dtype != F32 or reserved != 0:
                raise ValueError("CNOB头、版本或dtype错误")
            if kind == BASE_HIDDEN:
                if pending is not None or record != record_index or ne != (HIDDEN, 1, 1, 1):
                    raise ValueError(f"hidden记录顺序或形状错误: record={record}")
                raw_hidden = np.frombuffer(payload, dtype="<f4")
                normalized = raw_hidden.astype(np.float64)
                normalized /= max(float(np.linalg.norm(normalized)), 1e-8)
                reduced = (normalized - hidden_mean.astype(np.float64)) @ pca.astype(np.float64).T
                runtime_logits = reduced @ classifier_weight.astype(np.float64).T + classifier_bias.astype(np.float64)
                design_logits = np.concatenate(([1.0], reduced)) @ npz["classifier"].astype(np.float64)
                classifier_logit_max_abs_difference = max(
                    classifier_logit_max_abs_difference,
                    float(np.max(np.abs(runtime_logits - design_logits))),
                )
                probabilities = stable_softmax(runtime_logits)
                predicted = int(np.argmax(probabilities))
                strict_active = bool(np.any(runtime_logits[1:] > runtime_logits[0]))
                ordered = np.sort(runtime_logits)
                minimum_argmax_margin = min(
                    minimum_argmax_margin, float(ordered[-1] - ordered[-2])
                )
                minimum_no_op_boundary_margin = min(
                    minimum_no_op_boundary_margin,
                    abs(float(np.max(runtime_logits[1:]) - runtime_logits[0])),
                )
                if strict_active != (predicted != 0):
                    formula_errors.append(f"record={record}: strict gate与argmax不一致")
                norm_squared_f32 = np.sum(raw_hidden * raw_hidden, dtype=np.float32)
                norm_f32 = max(float(np.float32(np.sqrt(norm_squared_f32))), 1e-8)
                normalized_f32 = (raw_hidden / np.float32(norm_f32)).astype(np.float32)
                reduced_f32 = (
                    pca @ (normalized_f32 - hidden_mean).astype(np.float32)
                ).astype(np.float32)
                runtime_logits_f32 = (
                    classifier_weight @ reduced_f32 + classifier_bias
                ).astype(np.float32)
                probabilities_f32 = stable_softmax_f32(runtime_logits_f32)
                predicted_f32 = int(np.argmax(probabilities_f32))
                active_f32 = bool(np.any(runtime_logits_f32[1:] > runtime_logits_f32[0]))
                f32_predicted_class_mismatches += int(predicted_f32 != predicted)
                f32_no_op_gate_mismatches += int(active_f32 != strict_active)
                f32_probability_max_abs_difference = max(
                    f32_probability_max_abs_difference,
                    float(np.max(np.abs(probabilities_f32.astype(np.float64) - probabilities))),
                )
                correction = float(strength[0]) * probabilities[1:]
                intended = correction if strict_active else np.zeros_like(correction)
                pending = (record, intended, correction, probabilities, predicted, not strict_active, strict_active)
            elif kind == BASE_LOGITS:
                if pending is None or pending[0] != record or ne != (VOCAB, 1, 1, 1):
                    raise ValueError(f"logits记录顺序或形状错误: record={record}")
                raw = np.frombuffer(payload, dtype="<f4")
                row = teacher[record]
                task = task_by_id[row["task_id"]]
                target_id = int(row["target_token_id"])
                label = candidate_lookup.get(target_id, -1) + 1
                native = exact_nll(raw, target_id)
                for destination, delta, exact_no_op in (
                    (evaluated, pending[1], pending[5]),
                    (always_active_evaluated, pending[2], False),
                ):
                    adjusted = corrected_nll(raw, target_id, base_ids, delta, candidate_lookup)
                    destination.append(
                        {
                            "record": record,
                            "task_id": row["task_id"],
                            "split": task["split"],
                            "target_nll_delta": adjusted - native,
                            "exact_no_op": exact_no_op,
                            "class_correct": pending[4] == label,
                        }
                    )
                pending = None
                record_index += 1
            else:
                raise ValueError(f"CNOB出现意外kind={kind}")
    if pending is not None or record_index != len(teacher):
        raise ValueError("CNOB记录数与teacher不一致")

    replay_metrics = {split: summarize(evaluated, split) for split in ("train", "validation", "test")}
    always_active_metrics = {
        split: summarize(always_active_evaluated, split) for split in ("train", "validation", "test")
    }
    metric_checks: dict[str, Any] = {}
    replay_errors: list[str] = []
    maximum_metric_difference = 0.0
    for split in ("train", "validation", "test"):
        passed, maximum, errors = compare_metrics(replay_metrics[split], expected_report["metrics"][split])
        maximum_metric_difference = max(maximum_metric_difference, maximum)
        metric_checks[split] = {"passed": passed, "maximum_abs_difference": maximum, "errors": errors}
        replay_errors.extend(f"{split}: {error}" for error in errors)

    qwen_path = PROJECT / "llama.cpp/src/models/qwen35moe.cpp"
    cpu_step_path = PROJECT / "llama.cpp/ggml/src/ggml-cpu/unary-ops.cpp"
    vulkan_step_path = PROJECT / "llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/step.comp"
    qwen_source = qwen_path.read_text(encoding="utf-8")
    cpu_step_source = cpu_step_path.read_text(encoding="utf-8")
    vulkan_step_source = vulkan_step_path.read_text(encoding="utf-8")
    source_evidence = {
        "qwen_gate_has_nested_step": (
            "ggml_step(\n                ctx0, ggml_sum_rows(ctx0, candidate_beats_no_op))" in qwen_source
        ),
        "cpu_step_is_strict_gt_zero": "return (x > 0.f) ? 1.f : 0.f;" in cpu_step_source,
        "vulkan_step_is_greater_or_equal_zero": "x >= 0.0f ? 1.0f : 0.0f" in vulkan_step_source,
        "qwen_source_sha256": sha256_file(qwen_path),
        "cpu_step_source_sha256": sha256_file(cpu_step_path),
        "vulkan_step_source_sha256": sha256_file(vulkan_step_path),
    }
    intended_no_op_samples = int(sum(row["exact_no_op"] for row in evaluated))
    always_active_metric_changes = {
        split: {
            "mean_target_nll_delta_change": (
                always_active_metrics[split]["mean_target_nll_delta"]
                - replay_metrics[split]["mean_target_nll_delta"]
            ),
            "token_wins_change": always_active_metrics[split]["token_wins"] - replay_metrics[split]["token_wins"],
            "token_losses_change": always_active_metrics[split]["token_losses"] - replay_metrics[split]["token_losses"],
            "task_wins_change": always_active_metrics[split]["task_wins"] - replay_metrics[split]["task_wins"],
            "task_losses_change": always_active_metrics[split]["task_losses"] - replay_metrics[split]["task_losses"],
        }
        for split in ("train", "validation", "test")
    }
    layout_passed = not layout_errors
    offline_replay_passed = not replay_errors and not formula_errors
    vulkan_no_op_contract_passed = bool(
        source_evidence["qwen_gate_has_nested_step"]
        and source_evidence["cpu_step_is_strict_gt_zero"]
        and not source_evidence["vulkan_step_is_greater_or_equal_zero"]
    )
    result = {
        "format": "colorlm-v43-runtime-package-cpu-selfcheck-v1",
        "status": (
            "passed" if layout_passed and offline_replay_passed and vulkan_no_op_contract_passed
            else "layout_and_offline_replay_passed_but_vulkan_noop_mismatch"
            if layout_passed and offline_replay_passed
            else "failed"
        ),
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "blob": str(blob_path),
            "blob_sha256": sha256_file(blob_path),
            "npz": str(args.weights),
            "npz_sha256": sha256_file(args.weights),
            "report": str(args.report),
            "report_sha256": sha256_file(args.report),
            "capture": str(args.capture),
            "capture_sha256": sha256_file(args.capture),
        },
        "layout": {
            "passed": layout_passed,
            "errors": layout_errors,
            "tensor_checks": tensor_checks,
            "pca_direction_cosines": pca_direction_cosines.tolist(),
            "pca_orthogonality_max_abs_error": pca_orthogonality_error,
        },
        "offline_replay": {
            "passed": offline_replay_passed,
            "sample_count": len(evaluated),
            "strict_candidate_gt_noop_matches_argmax": not formula_errors,
            "formula_errors": formula_errors,
            "classifier_npz_vs_runtime_max_abs_logit_difference": classifier_logit_max_abs_difference,
            "f32_accumulation": {
                "predicted_class_mismatches_vs_report_replay": f32_predicted_class_mismatches,
                "no_op_gate_mismatches_vs_report_replay": f32_no_op_gate_mismatches,
                "probability_max_abs_difference": f32_probability_max_abs_difference,
                "minimum_argmax_margin": minimum_argmax_margin,
                "minimum_no_op_boundary_margin": minimum_no_op_boundary_margin,
            },
            "report_max_abs_metric_difference": maximum_metric_difference,
            "metric_checks": metric_checks,
            "metrics": replay_metrics,
        },
        "vulkan_no_op_audit": {
            "passed": vulkan_no_op_contract_passed,
            "source_evidence": source_evidence,
            "reason": (
                "当前CPU与Vulkan均满足step(0)=0，嵌套step门与严格candidate>no-op/argmax语义一致。"
                if vulkan_no_op_contract_passed
                else "图对非负的候选胜出计数再次调用ggml_step；若Vulkan使用step(0)=1，active_mask将恒为1。"
            ),
            "offline_intended_no_op_samples": intended_no_op_samples,
            "samples_whose_exact_noop_is_disabled_by_current_backend": (
                0 if vulkan_no_op_contract_passed else intended_no_op_samples
            ),
            "counterfactual_samples_affected_if_step_zero_were_active": intended_no_op_samples,
            "always_active_metric_changes": always_active_metric_changes,
        },
        "conclusion": (
            "weights.bin与NPZ布局逐字节一致，按严格argmax/no-op公式可复现720样本F32离线报告；"
            "当前CPU与Vulkan的step(0)=0语义也一致，未发现布局或门公式不一致。"
            if vulkan_no_op_contract_passed
            else "weights.bin与NPZ布局逐字节一致，按离线严格argmax/no-op公式可复现720样本报告；"
            "但当前Vulkan嵌套step实现不能兑现exact no-op，必须修正图内门后再做GPU运行验收。"
        ),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if layout_passed and offline_replay_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
