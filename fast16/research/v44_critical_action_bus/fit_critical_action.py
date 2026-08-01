"""拟合 v44 关键动作稀疏头；类0为精确 no-op，其余类直接对应判别性 token。"""

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
ROOT = HERE.parents[2]
V29_FIT = ROOT / "fast16/research/v29_sequence_policy_head/fit_sequence_policy.py"


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
        raise RuntimeError(f"无法加载v29数值实现: {V29_FIT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    candidates = [row for row in selected if row["target_is_candidate"]]
    task_base: defaultdict[str, int] = defaultdict(int)
    task_corrected: defaultdict[str, int] = defaultdict(int)
    task_samples: defaultdict[str, int] = defaultdict(int)
    for row in candidates:
        task_samples[row["task_id"]] += 1
        task_base[row["task_id"]] += int(row["native_candidate_correct"])
        task_corrected[row["task_id"]] += int(row["corrected_candidate_correct"])
    task_delta = {task: task_corrected[task] - task_base[task] for task in task_samples}
    nll = np.asarray([row["target_nll_delta"] for row in selected], dtype=np.float64)
    candidate_nll = np.asarray([row["target_nll_delta"] for row in candidates], dtype=np.float64)
    native_correct = sum(row["native_candidate_correct"] for row in candidates)
    corrected_correct = sum(row["corrected_candidate_correct"] for row in candidates)
    return {
        "sample_count": len(selected),
        "candidate_sample_count": len(candidates),
        "task_count": len({row["task_id"] for row in selected}),
        "candidate_task_count": len(task_samples),
        "native_candidate_accuracy": native_correct / len(candidates) if candidates else None,
        "corrected_candidate_accuracy": corrected_correct / len(candidates) if candidates else None,
        "candidate_accuracy_gain": (corrected_correct - native_correct) / len(candidates) if candidates else None,
        "rescues": sum(not row["native_candidate_correct"] and row["corrected_candidate_correct"] for row in candidates),
        "regressions": sum(row["native_candidate_correct"] and not row["corrected_candidate_correct"] for row in candidates),
        "mean_target_nll_delta": float(np.mean(nll)),
        "mean_candidate_target_nll_delta": float(np.mean(candidate_nll)) if len(candidate_nll) else None,
        "exact_no_op_rate": float(np.mean([row["exact_no_op"] for row in selected])),
        "label_accuracy": float(np.mean([row["predicted_class"] == row["target_class"] for row in selected])),
        "task_wins": sum(value > 0 for value in task_delta.values()),
        "task_regressions": sum(value < 0 for value in task_delta.values()),
        "task_net_wins": sum(value > 0 for value in task_delta.values()) - sum(value < 0 for value in task_delta.values()),
        "per_task_candidate_correct_delta": task_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=HERE / "critical-states-dev-v1.cnob")
    parser.add_argument("--teacher", type=Path, default=HERE / "critical-teacher-dev-v1.jsonl")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "fast16/research/v43_policy_dataset/trajectory_tasks_v1.jsonl",
    )
    parser.add_argument("--contract", type=Path, default=HERE / "v44_dev_contract.json")
    parser.add_argument("--report", type=Path, default=HERE / "critical-policy-dev-report.json")
    parser.add_argument("--weights", type=Path, default=HERE / "critical-policy-dev-weights.npz")
    args = parser.parse_args()

    for path in (args.capture, args.teacher, args.tasks, args.contract):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.report.exists() or args.weights.exists():
        raise FileExistsError("策略报告或权重已存在，拒绝覆盖")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if (
        contract.get("format") != "colorlm-v44-critical-action-dev-contract-v1"
        or contract.get("critical_teacher_sha256") != sha256_file(args.teacher)
        or contract.get("source_tasks_sha256") != sha256_file(args.tasks)
        or contract["fit"].get("parameter_scan_after_capture") is not False
    ):
        raise ValueError("v44 dev合同格式、哈希或禁止扫参字段无效")

    teacher = read_jsonl(args.teacher)
    tasks = read_jsonl(args.tasks)
    task_by_id = {row["id"]: row for row in tasks}
    if any(row.get("split") == "test" or row["task_id"] not in task_by_id for row in teacher):
        raise ValueError("v44 dev teacher泄漏test或引用未知任务")
    fit = import_fit()
    capture = fit.load_capture(args.capture)
    if sorted(capture) != list(range(len(teacher))):
        raise ValueError("CNOB record与teacher数量不对齐")
    if any(set(bucket) != {fit.BASE_HIDDEN, fit.BASE_LOGITS} for bucket in capture.values()):
        raise ValueError("每个CNOB record必须恰有hidden和logits")

    minimum_groups = int(contract["fit"]["minimum_distinct_train_groups_per_candidate"])
    token_groups: defaultdict[int, set[str]] = defaultdict(set)
    for row in teacher:
        task = task_by_id[row["task_id"]]
        if row["split"] == "train":
            token_groups[int(row["target_token_id"])].add(task["group_id"])
    candidate_ids = np.asarray(
        sorted(token for token, groups in token_groups.items() if len(groups) >= minimum_groups),
        dtype=np.int64,
    )
    if candidate_ids.size < 8 or candidate_ids.size > 64:
        raise ValueError(f"训练独立组筛选后候选token数异常: {candidate_ids.size}")
    candidate_lookup = {int(token): index for index, token in enumerate(candidate_ids)}

    hidden = np.stack([capture[index][fit.BASE_HIDDEN] for index in range(len(teacher))]).astype(np.float64)
    hidden /= np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), 1e-8)
    split = np.asarray([row["split"] for row in teacher])
    train = np.flatnonzero(split == "train")
    hidden_mean = hidden[train].mean(axis=0)
    centered_train = hidden[train] - hidden_mean
    rank = int(contract["fit"]["pca_rank"])
    _, _, right = np.linalg.svd(centered_train, full_matrices=False)
    if right.shape[0] < rank:
        raise ValueError("样本数不足以拟合冻结PCA rank")
    hidden_mean_f32 = hidden_mean.astype("<f4")
    components_f32 = right[:rank].astype("<f4")
    hidden_mean = hidden_mean_f32.astype(np.float64)
    components = components_f32.astype(np.float64)
    reduced = (hidden - hidden_mean) @ components.T
    design = np.column_stack((np.ones(len(teacher), dtype=np.float64), reduced))

    native_candidate_position = []
    labels = []
    for index, row in enumerate(teacher):
        raw = capture[index][fit.BASE_LOGITS]
        native_position = int(np.argmax(raw[candidate_ids]))
        native_candidate_position.append(native_position)
        target_position = candidate_lookup.get(int(row["target_token_id"]))
        labels.append(0 if target_position is None or target_position == native_position else target_position + 1)
    labels_array = np.asarray(labels, dtype=np.int64)
    class_count = int(candidate_ids.size) + 1
    target = np.eye(class_count, dtype=np.float64)[labels_array[train]]
    ridge = float(contract["fit"]["ridge_lambda"])
    gram = design[train].T @ design[train] + ridge * np.eye(rank + 1, dtype=np.float64)
    gram[0, 0] -= ridge
    classifier_f32 = np.linalg.solve(gram, design[train].T @ target).astype("<f4")
    classifier = classifier_f32.astype(np.float64)
    probabilities = softmax(design @ classifier)
    predicted_class = np.argmax(probabilities, axis=1)
    exact_no_op = predicted_class == 0
    strength_f32 = np.asarray([contract["fit"]["correction_strength"]], dtype="<f4")
    correction = float(strength_f32[0]) * probabilities[:, 1:]
    correction[exact_no_op] = 0.0

    evaluated: list[dict[str, Any]] = []
    for index, row in enumerate(teacher):
        raw = capture[index][fit.BASE_LOGITS]
        target_id = int(row["target_token_id"])
        target_position = candidate_lookup.get(target_id)
        native_position = native_candidate_position[index]
        corrected_position = int(np.argmax(raw[candidate_ids].astype(np.float64) + correction[index]))
        native_nll = fit.exact_nll(raw, target_id)
        corrected_nll = fit.corrected_nll(raw, target_id, candidate_ids, correction[index], candidate_lookup)
        evaluated.append(
            {
                "record": index,
                "sample_id": row["sample_id"],
                "task_id": row["task_id"],
                "split": row["split"],
                "capability": row["capability"],
                "critical_roles": row["critical_roles"],
                "target_token_id": target_id,
                "target_is_candidate": target_position is not None,
                "target_class": int(labels_array[index]),
                "predicted_class": int(predicted_class[index]),
                "exact_no_op": bool(exact_no_op[index]),
                "native_candidate_token_id": int(candidate_ids[native_position]),
                "corrected_candidate_token_id": int(candidate_ids[corrected_position]),
                "native_candidate_correct": bool(target_position is not None and native_position == target_position),
                "corrected_candidate_correct": bool(target_position is not None and corrected_position == target_position),
                "native_target_nll": native_nll,
                "corrected_target_nll": corrected_nll,
                "target_nll_delta": corrected_nll - native_nll,
            }
        )

    metrics = {name: summarize(evaluated, name) for name in ("train", "validation")}
    validation = metrics["validation"]
    gate = contract["development_gate"]
    passed = bool(
        validation["candidate_accuracy_gain"] >= float(gate["minimum_validation_candidate_accuracy_gain"])
        and validation["rescues"] >= int(gate["minimum_validation_rescues"])
        and validation["regressions"] <= int(gate["maximum_validation_regressions"])
        and validation["mean_target_nll_delta"] <= float(gate["maximum_validation_mean_target_nll_delta"])
        and validation["task_net_wins"] >= int(gate["minimum_validation_task_net_wins"])
        and float(gate["minimum_validation_exact_no_op_rate"])
        <= validation["exact_no_op_rate"]
        <= float(gate["maximum_validation_exact_no_op_rate"])
    )

    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.weights,
        candidate_ids=candidate_ids.astype("<i8"),
        hidden_mean=hidden_mean_f32,
        pca_components=components_f32,
        classifier=classifier_f32,
        correction_strength=strength_f32,
    )
    report = {
        "format": "colorlm-v44-critical-action-dev-report-v1",
        "claim_scope": contract["claim_scope"],
        "contract": str(args.contract),
        "contract_sha256": sha256_file(args.contract),
        "teacher_sha256": sha256_file(args.teacher),
        "capture_sha256": sha256_file(args.capture),
        "fit_implementation_sha256": sha256_file(Path(__file__)),
        "sample_count": len(teacher),
        "candidate_token_count": int(candidate_ids.size),
        "candidate_token_ids": candidate_ids.tolist(),
        "class_count": class_count,
        "pca_rank": rank,
        "classifier_parameters": int(classifier.size),
        "metrics": metrics,
        "development_gate_passed": passed,
        "decision": "allow v44 dev runtime prototype; blind still required" if passed else contract["failure_action"],
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
