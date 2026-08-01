"""拟合 v43 低秩多类策略头；类别0是精确 no-op。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
V29_FIT = PROJECT / "fast16/research/v29_sequence_policy_head/fit_sequence_policy.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def import_fit():
    spec = importlib.util.spec_from_file_location("colorlm_v29_fit", V29_FIT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载v29解析器: {V29_FIT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_candidate_allowlist(package: Path, expected_manifest_sha: str, expected_weights_sha: str) -> np.ndarray:
    manifest_path = package / "policy.json"
    weights_path = package / "weights.bin"
    if sha256_file(manifest_path) != expected_manifest_sha or sha256_file(weights_path) != expected_weights_sha:
        raise ValueError("固定v29候选源SHA-256不匹配")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["tensors"] if item["name"] == "policy.base_ids")
    payload = weights_path.read_bytes()[int(record["offset"]): int(record["offset"]) + int(record["bytes"])]
    values = np.frombuffer(payload, dtype="<i8").copy()
    if values.shape != (int(manifest["candidate_token_count"]),):
        raise ValueError("固定候选token形状错误")
    return values.astype(np.int64)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate-package", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()

    for output in (args.report, args.weights):
        if output.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if sha256_file(args.tasks) != contract["source_tasks_sha256"]:
        raise ValueError("任务文件与预注册合同SHA-256不一致")
    tasks = read_jsonl(args.tasks)
    teacher = read_jsonl(args.teacher)
    task_by_id = {row["id"]: row for row in tasks}
    if len(task_by_id) != len(tasks) or any(row["task_id"] not in task_by_id for row in teacher):
        raise ValueError("任务ID重复或teacher引用未知任务")

    fit = import_fit()
    capture = fit.load_capture(args.capture)
    if sorted(capture) != list(range(len(teacher))):
        raise ValueError("CNOB编号与teacher不一致")
    if any(set(bucket) != {fit.BASE_HIDDEN, fit.BASE_LOGITS} for bucket in capture.values()):
        raise ValueError("每个样本必须恰有hidden与base logits")

    allowlist = load_candidate_allowlist(
        args.candidate_package,
        contract["source_candidate_policy_sha256"],
        contract["source_candidate_weights_sha256"],
    )
    minimum_groups = int(contract["fit"]["minimum_distinct_train_groups_per_candidate"])
    token_groups: dict[int, set[str]] = defaultdict(set)
    for row in teacher:
        task = task_by_id[row["task_id"]]
        if task["split"] == "train" and int(row["target_token_id"]) in allowlist:
            token_groups[int(row["target_token_id"])].add(task["group_id"])
    candidate_ids = np.asarray(
        sorted(token for token in allowlist if len(token_groups[int(token)]) >= minimum_groups),
        dtype=np.int64,
    )
    if candidate_ids.size < 4:
        raise ValueError(f"可泛化候选token不足: {candidate_ids.size}")
    candidate_lookup = {int(token): index for index, token in enumerate(candidate_ids)}

    hidden = np.stack([capture[index][fit.BASE_HIDDEN] for index in range(len(teacher))]).astype(np.float64)
    hidden /= np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), 1e-8)
    splits = np.asarray([task_by_id[row["task_id"]]["split"] for row in teacher])
    train = np.flatnonzero(splits == "train")
    hidden_mean = hidden[train].mean(axis=0)
    centered_train = hidden[train] - hidden_mean
    rank = int(contract["fit"]["pca_rank"])
    _, _, right = np.linalg.svd(centered_train, full_matrices=False)
    if right.shape[0] < rank:
        raise ValueError("训练样本不足以满足冻结PCA rank")
    # 判门必须使用最终部署精度，不能先用F64过门再把参数截成F32。
    hidden_mean_f32 = hidden_mean.astype("<f4")
    components_f32 = right[:rank].astype("<f4")
    hidden_mean = hidden_mean_f32.astype(np.float64)
    components = components_f32.astype(np.float64)
    reduced = (hidden - hidden_mean) @ components.T
    design = np.column_stack((np.ones(len(teacher), dtype=np.float64), reduced))

    labels = np.asarray(
        [candidate_lookup.get(int(row["target_token_id"]), -1) + 1 for row in teacher],
        dtype=np.int64,
    )
    class_count = int(candidate_ids.size) + 1
    target = np.eye(class_count, dtype=np.float64)[labels[train]]
    ridge = float(contract["fit"]["ridge_lambda"])
    gram = design[train].T @ design[train] + ridge * np.eye(rank + 1, dtype=np.float64)
    gram[0, 0] -= ridge
    classifier_f32 = np.linalg.solve(gram, design[train].T @ target).astype("<f4")
    classifier = classifier_f32.astype(np.float64)
    probabilities = softmax(design @ classifier)
    predicted_class = np.argmax(probabilities, axis=1)
    exact_no_op = predicted_class == 0
    correction_strength_f32 = np.asarray(
        [contract["fit"]["correction_strength"]], dtype="<f4"
    )
    correction = float(correction_strength_f32[0]) * probabilities[:, 1:]
    correction[exact_no_op] = 0.0

    evaluated: list[dict[str, Any]] = []
    for index, row in enumerate(teacher):
        raw = capture[index][fit.BASE_LOGITS]
        target_id = int(row["target_token_id"])
        native = fit.exact_nll(raw, target_id)
        corrected = fit.corrected_nll(raw, target_id, candidate_ids, correction[index], candidate_lookup)
        task = task_by_id[row["task_id"]]
        evaluated.append(
            {
                "record": index,
                "sample_id": row["sample_id"],
                "task_id": row["task_id"],
                "group_id": task["group_id"],
                "split": task["split"],
                "capability": task["capability"],
                "state_label": task["label"],
                "target_token_id": target_id,
                "target_class": int(labels[index]),
                "predicted_class": int(predicted_class[index]),
                "class_correct": bool(predicted_class[index] == labels[index]),
                "exact_no_op": bool(exact_no_op[index]),
                "native_target_nll": native,
                "corrected_target_nll": corrected,
                "target_nll_delta": corrected - native,
            }
        )

    metrics = {split: summarize(evaluated, split) for split in ("train", "validation", "test")}
    gate = contract["offline_gate"]
    heldout = (metrics["validation"], metrics["test"])
    passed = bool(
        metrics["validation"]["mean_target_nll_delta"] <= float(gate["validation_mean_target_nll_delta_max"])
        and metrics["test"]["mean_target_nll_delta"] <= float(gate["test_mean_target_nll_delta_max"])
        and all(part["task_wins"] - part["task_losses"] >= int(gate["minimum_task_wins_minus_losses"]) for part in heldout)
        and all(part["worst_task_mean_nll_regression"] <= float(gate["maximum_task_mean_nll_regression"]) for part in heldout)
        and all(float(gate["minimum_exact_no_op_rate"]) <= part["exact_no_op_rate"] <= float(gate["maximum_exact_no_op_rate"]) for part in heldout)
    )

    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.weights,
        candidate_ids=candidate_ids.astype("<i8"),
        hidden_mean=hidden_mean_f32,
        pca_components=components_f32,
        classifier=classifier_f32,
        correction_strength=correction_strength_f32,
    )
    report = {
        "format": "colorlm-v43-noop-policy-report-v1",
        "contract": str(args.contract),
        "contract_sha256": sha256_file(args.contract),
        "tasks_sha256": sha256_file(args.tasks),
        "teacher_sha256": sha256_file(args.teacher),
        "capture_sha256": sha256_file(args.capture),
        "fit_implementation_sha256": sha256_file(Path(__file__)),
        "sample_count": len(teacher),
        "task_count": len(tasks),
        "candidate_token_count": int(candidate_ids.size),
        "candidate_token_ids": candidate_ids.tolist(),
        "no_op_class": 0,
        "pca_rank": rank,
        "classifier_parameters": int(classifier.size),
        "evaluation_precision": "final-deployment-f32-parameters-with-f64-accumulation",
        "metrics": metrics,
        "gate_passed": passed,
        "decision": "allow runtime v43 prototype" if passed else contract["failure_action"],
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
