"""Fit the pre-registered v29 sparse sequence policy head from CNOB records."""

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


HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
F32 = 0
BASE_LOGITS = 1
BASE_HIDDEN = 4
VOCAB = 248320
WIDTH = 2048


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def manifest_path(path: Path) -> str:
    return path.as_posix()


def load_capture(path: Path) -> dict[int, dict[int, np.ndarray]]:
    grouped: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    with path.open("rb") as source:
        while True:
            encoded = source.read(HEADER.size)
            if not encoded:
                break
            if len(encoded) != HEADER.size:
                raise ValueError("CNOB头被截断")
            magic, version, kind, record, dtype, reserved, *tail = HEADER.unpack(encoded)
            ne = tuple(tail[:4])
            payload_bytes = int(tail[4])
            if magic != MAGIC or version != 1 or dtype != F32 or reserved != 0:
                raise ValueError("CNOB头、版本或dtype不符合v29契约")
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError("CNOB payload被截断")
            if kind not in (BASE_LOGITS, BASE_HIDDEN):
                raise ValueError(f"v29采集出现意外kind={kind}")
            width = VOCAB if kind == BASE_LOGITS else WIDTH
            if ne != (width, 1, 1, 1):
                raise ValueError(f"kind={kind} shape错误: {ne}")
            if kind in grouped[record]:
                raise ValueError(f"kind={kind} record={record}重复")
            tensor = np.frombuffer(payload, dtype="<f4").copy()
            if not np.isfinite(tensor).all():
                raise ValueError(f"kind={kind} record={record}包含非有限值")
            grouped[record][kind] = tensor
    return dict(grouped)


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


def split_metrics(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    policy = [row for row in selected if row["target_is_candidate"]]
    return {
        "sample_count": len(selected),
        "candidate_target_count": len(policy),
        "mean_target_nll_delta": float(np.mean([row["target_nll_delta"] for row in selected])),
        "selected_token_win_rate": (
            float(np.mean([row["target_nll_delta"] < 0.0 for row in policy])) if policy else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="拟合v29稀疏序列执行策略头")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if sha256_file(args.tasks) != contract["source_tasks_sha256"]:
        raise ValueError("v29任务文件SHA-256与冻结契约不一致")
    tasks = read_jsonl(args.tasks)
    teacher = read_jsonl(args.teacher)
    task_by_id = {row["id"]: row for row in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("v29任务ID重复")
    if any(row["task_id"] not in task_by_id for row in teacher):
        raise ValueError("teacher包含冻结任务之外的样本")

    capture = load_capture(args.capture)
    if sorted(capture) != list(range(len(teacher))):
        raise ValueError("CNOB record编号与teacher样本数不一致")
    if any(set(bucket) != {BASE_HIDDEN, BASE_LOGITS} for bucket in capture.values()):
        raise ValueError("每个teacher样本必须恰有一个hidden和一个logits记录")

    minimum_tasks = int(contract["candidate_rows"]["minimum_distinct_train_tasks"])
    token_tasks: dict[int, set[str]] = defaultdict(set)
    for row in teacher:
        task = task_by_id[row["task_id"]]
        if task["split"] == contract["candidate_rows"]["source_split"]:
            token_tasks[int(row["target_token_id"])].add(row["task_id"])
    candidate_ids = np.array(
        sorted(token for token, task_ids in token_tasks.items() if len(task_ids) >= minimum_tasks),
        dtype=np.int64,
    )
    if len(candidate_ids) < int(contract["candidate_rows"]["minimum_candidate_tokens"]):
        raise ValueError(f"候选token不足: {len(candidate_ids)}")
    candidate_lookup = {int(token): index for index, token in enumerate(candidate_ids)}

    hidden = np.stack([capture[index][BASE_HIDDEN] for index in range(len(teacher))]).astype(np.float64)
    hidden /= np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), 1e-8)
    target = np.zeros((len(teacher), len(candidate_ids)), dtype=np.float64)
    margin = float(contract["target"]["margin_over_native_top1"])
    maximum = float(contract["target"]["maximum_positive_correction"])
    for index, row in enumerate(teacher):
        task = task_by_id[row["task_id"]]
        if task["split"] != contract["fit"]["fit_split"]:
            continue
        token_id = int(row["target_token_id"])
        position = candidate_lookup.get(token_id)
        if position is None:
            continue
        logits = capture[index][BASE_LOGITS]
        need = max(0.0, float(np.max(logits)) - float(logits[token_id]) + margin)
        target[index, position] = min(maximum, need)

    train_indices = np.array(
        [index for index, row in enumerate(teacher) if task_by_id[row["task_id"]]["split"] == contract["fit"]["fit_split"]]
    )
    x_train = hidden[train_indices]
    y_train = target[train_indices]
    x_mean = x_train.mean(axis=0)
    y_mean = y_train.mean(axis=0)
    xc = x_train - x_mean
    yc = y_train - y_mean
    ridge_lambda = float(contract["fit"]["ridge_lambda"])
    dual = np.linalg.solve(
        xc @ xc.T + ridge_lambda * np.eye(len(train_indices), dtype=np.float64),
        yc,
    )
    weight = xc.T @ dual
    bias = y_mean - x_mean @ weight
    predicted = hidden @ weight + bias
    predicted = np.clip(
        predicted,
        float(contract["target"]["minimum_runtime_correction"]),
        float(contract["target"]["maximum_runtime_correction"]),
    )

    evaluated: list[dict[str, Any]] = []
    for index, row in enumerate(teacher):
        task = task_by_id[row["task_id"]]
        target_id = int(row["target_token_id"])
        logits = capture[index][BASE_LOGITS]
        native = exact_nll(logits, target_id)
        corrected = corrected_nll(
            logits, target_id, candidate_ids, predicted[index], candidate_lookup
        )
        evaluated.append(
            {
                "sample_id": row["sample_id"],
                "task_id": row["task_id"],
                "split": task["split"],
                "family": task["family"],
                "token_index": int(row["token_index"]),
                "target_token_id": target_id,
                "target_is_candidate": target_id in candidate_lookup,
                "native_target_nll": native,
                "corrected_target_nll": corrected,
                "target_nll_delta": corrected - native,
            }
        )

    by_task: dict[str, list[float]] = defaultdict(list)
    for row in evaluated:
        by_task[row["task_id"]].append(float(row["target_nll_delta"]))
    task_metrics = [
        {
            "task_id": task_id,
            "split": task_by_id[task_id]["split"],
            "family": task_by_id[task_id]["family"],
            "mean_target_nll_delta": float(np.mean(values)),
        }
        for task_id, values in sorted(by_task.items())
    ]
    metrics = {
        split: split_metrics(evaluated, split)
        for split in ("train", "validation", "test")
    }
    heldout_task_regression = max(
        row["mean_target_nll_delta"]
        for row in task_metrics
        if row["split"] != "train"
    )
    gate = contract["pass_gate"]
    passed = (
        metrics["validation"]["mean_target_nll_delta"]
        <= float(gate["validation_mean_target_nll_delta_max"])
        and metrics["test"]["mean_target_nll_delta"]
        <= float(gate["test_mean_target_nll_delta_max"])
        and metrics["validation"]["selected_token_win_rate"] is not None
        and metrics["validation"]["selected_token_win_rate"]
        >= float(gate["validation_selected_token_win_rate_min"])
        and metrics["test"]["selected_token_win_rate"] is not None
        and metrics["test"]["selected_token_win_rate"]
        >= float(gate["test_selected_token_win_rate_min"])
        and heldout_task_regression
        <= float(gate["maximum_heldout_task_mean_nll_regression"])
    )

    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.weights,
        token_ids=candidate_ids.astype("<i4"),
        weight=weight.T.astype("<f4"),
        bias=bias.astype("<f4"),
        correction_min=np.array(
            [contract["target"]["minimum_runtime_correction"]], dtype="<f4"
        ),
        correction_max=np.array(
            [contract["target"]["maximum_runtime_correction"]], dtype="<f4"
        ),
    )
    report = {
        "format": "colorlm-v29-sequence-policy-report-v1",
        "contract": manifest_path(args.contract),
        "contract_sha256": sha256_file(args.contract),
        "tasks": manifest_path(args.tasks),
        "tasks_sha256": sha256_file(args.tasks),
        "teacher": manifest_path(args.teacher),
        "teacher_sha256": sha256_file(args.teacher),
        "capture": manifest_path(args.capture),
        "capture_sha256": sha256_file(args.capture),
        "candidate_token_count": int(len(candidate_ids)),
        "candidate_token_ids": [int(token) for token in candidate_ids],
        "fit": {
            **contract["fit"],
            "train_sample_count": int(len(train_indices)),
            "weight_l2": float(np.linalg.norm(weight)),
        },
        "metrics": metrics,
        "maximum_heldout_task_mean_nll_regression": heldout_task_regression,
        "per_task": task_metrics,
        "gate_passed": passed,
        "decision": (
            "allow v29 runtime sequence policy prototype"
            if passed
            else contract["failure_action"]
        ),
        "weights": manifest_path(args.weights),
        "weights_sha256": sha256_file(args.weights),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
