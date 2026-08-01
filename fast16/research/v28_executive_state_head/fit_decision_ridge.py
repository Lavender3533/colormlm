"""Fit and evaluate the pre-registered v28 binary decision ridge probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
F32 = 0
BASE_LOGITS = 1
BASE_HIDDEN = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(path: Path) -> str:
    """Keep reports portable and independent of the host code page."""
    return path.as_posix()


def load_capture(path: Path) -> dict[int, dict[int, np.ndarray]]:
    grouped: dict[int, dict[int, np.ndarray]] = {}
    with path.open("rb") as source:
        while True:
            encoded = source.read(HEADER.size)
            if not encoded:
                break
            if len(encoded) != HEADER.size:
                raise ValueError("CNOB头被截断")
            magic, version, kind, record, dtype, reserved, *tail = HEADER.unpack(encoded)
            ne = tail[:4]
            payload_bytes = tail[4]
            if magic != MAGIC or version != 1 or reserved != 0 or dtype != F32:
                raise ValueError("CNOB头、版本或dtype不符合v28契约")
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError("CNOB payload被截断")
            if kind not in (BASE_HIDDEN, BASE_LOGITS):
                raise ValueError(f"v28采集出现意外kind={kind}")
            expected_width = 2048 if kind == BASE_HIDDEN else 248320
            if ne != [expected_width, 1, 1, 1] and tuple(ne) != (expected_width, 1, 1, 1):
                raise ValueError(f"kind={kind} shape错误: {ne}")
            tensor = np.frombuffer(payload, dtype="<f4").copy()
            if not np.isfinite(tensor).all():
                raise ValueError(f"kind={kind} record={record}包含非有限值")
            bucket = grouped.setdefault(record, {})
            if kind in bucket:
                raise ValueError(f"kind={kind} record={record}重复")
            bucket[kind] = tensor
    return grouped


def accuracy(rows: list[dict[str, Any]], split: str) -> float:
    selected = [row for row in rows if row["split"] == split]
    return sum(row["correct"] for row in selected) / len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description="拟合固定v28二元ridge探针")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--gate-contract", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()

    index = json.loads(args.index.read_text(encoding="utf-8"))
    gate = json.loads(args.gate_contract.read_text(encoding="utf-8"))
    if sha256_file(args.capture) != index["capture_sha256"]:
        raise ValueError("capture SHA-256与索引不一致")
    if index["contract_sha256"] != gate["source_contract_sha256"]:
        raise ValueError("冻结任务契约与ridge契约不一致")
    grouped = load_capture(args.capture)
    records = index["records"]
    if sorted(grouped) != list(range(len(records))):
        raise ValueError("CNOB record编号不连续或与索引不一致")
    if any(set(grouped[i]) != {BASE_HIDDEN, BASE_LOGITS} for i in grouped):
        raise ValueError("每个任务必须恰有一个hidden和一个logits记录")

    x = np.stack([grouped[row["record"]][BASE_HIDDEN] for row in records]).astype(np.float64)
    eps = 1e-8
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    y = np.array([1.0 if row["label"] == "continue_tool" else -1.0 for row in records])
    train_indices = np.array([i for i, row in enumerate(records) if row["split"] == "train"])
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_mean = x_train.mean(axis=0)
    y_mean = float(y_train.mean())
    xc = x_train - x_mean
    yc = y_train - y_mean
    ridge_lambda = float(gate["probe"]["ridge_lambda"])
    dual = np.linalg.solve(
        xc @ xc.T + ridge_lambda * np.eye(len(train_indices), dtype=np.float64), yc
    )
    weight = xc.T @ dual
    bias = y_mean - float(x_mean @ weight)
    scores = x @ weight + bias
    threshold = float(gate["probe"]["decision_threshold"])
    predictions = np.where(scores >= threshold, 1.0, -1.0)

    evaluated: list[dict[str, Any]] = []
    for row, expected, score, predicted in zip(records, y, scores, predictions, strict=True):
        evaluated.append(
            {
                "id": row["id"],
                "split": row["split"],
                "label": row["label"],
                "score": float(score),
                "prediction": "continue_tool" if predicted > 0 else "finish",
                "correct": bool(predicted == expected),
                "absolute_margin": float(abs(score - threshold)),
            }
        )

    train_accuracy = accuracy(evaluated, "train")
    validation_accuracy = accuracy(evaluated, "validation")
    test_accuracy = accuracy(evaluated, "test")
    heldout = [row for row in evaluated if row["split"] != "train"]
    minimum_heldout_margin = min(row["absolute_margin"] for row in heldout)
    requirements = gate["pass_gate"]
    passed = (
        train_accuracy >= float(requirements["train_accuracy"])
        and validation_accuracy >= float(requirements["validation_accuracy"])
        and test_accuracy >= float(requirements["test_accuracy"])
        and minimum_heldout_margin >= float(requirements["minimum_absolute_heldout_margin"])
    )

    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.weights,
        weight=weight.astype("<f4"),
        bias=np.array([bias], dtype="<f4"),
        threshold=np.array([threshold], dtype="<f4"),
    )
    report = {
        "format": "colorlm-v28-decision-ridge-report-v1",
        "gate_contract": manifest_path(args.gate_contract),
        "gate_contract_sha256": sha256_file(args.gate_contract),
        "capture": manifest_path(args.capture),
        "capture_sha256": index["capture_sha256"],
        "probe": gate["probe"],
        "metrics": {
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy,
            "minimum_absolute_heldout_margin": minimum_heldout_margin,
            "weight_l2": float(np.linalg.norm(weight)),
            "bias": bias,
        },
        "evaluated": evaluated,
        "gate_passed": passed,
        "decision": (
            "allow runtime executive-state-head prototype"
            if passed
            else gate["failure_action"]
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
