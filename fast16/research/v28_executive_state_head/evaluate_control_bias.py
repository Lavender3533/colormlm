"""Evaluate the pre-registered two-token v28 control correction offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np


HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
BASE_LOGITS = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(path: Path) -> str:
    """Keep reports portable and independent of the host code page."""
    return path.as_posix()


def load_logits(path: Path) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
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
            if magic != MAGIC or version != 1 or reserved != 0 or dtype != 0:
                raise ValueError("CNOB头不符合v28控制门")
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError("CNOB payload被截断")
            if kind == BASE_LOGITS:
                if ne != [248320, 1, 1, 1]:
                    raise ValueError(f"base logits shape错误: {ne}")
                if record in result:
                    raise ValueError(f"base logits record重复: {record}")
                logits = np.frombuffer(payload, dtype="<f4").copy()
                if not np.isfinite(logits).all():
                    raise ValueError(f"base logits record={record}存在非有限值")
                result[record] = logits
    return result


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def split_accuracy(rows: list[dict[str, Any]], split: str) -> float:
    selected = [row for row in rows if row["split"] == split]
    return sum(row["corrected_top1_correct"] for row in selected) / len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description="离线验证v28双控制token修正")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--ridge-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    index = json.loads(args.index.read_text(encoding="utf-8"))
    ridge = json.loads(args.ridge_report.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if sha256_file(args.capture) != index["capture_sha256"]:
        raise ValueError("capture SHA-256不匹配")
    if ridge["capture_sha256"] != index["capture_sha256"]:
        raise ValueError("ridge报告与capture不匹配")
    if ridge["gate_contract_sha256"] != contract["source_ridge_contract_sha256"]:
        raise ValueError("控制契约引用的ridge契约不匹配")
    logits = load_logits(args.capture)
    records = index["records"]
    if sorted(logits) != list(range(len(records))):
        raise ValueError("base logits记录与任务索引不一致")

    ridge_rows = {row["id"]: row for row in ridge["evaluated"]}
    sharpness = float(contract["gate"]["sharpness"])
    target_margin = float(contract["calibration"]["target_margin"])
    token_ids = {
        "continue_tool": int(contract["control_tokens"]["continue_tool"]["base_token_id"]),
        "finish": int(contract["control_tokens"]["finish"]["base_token_id"]),
    }
    prepared: list[dict[str, Any]] = []
    required_beta: list[float] = []
    for item in records:
        raw = logits[item["record"]]
        score = float(ridge_rows[item["id"]]["score"])
        continue_weight = sigmoid(sharpness * score)
        desired_weight = continue_weight if item["label"] == "continue_tool" else 1.0 - continue_weight
        desired_id = token_ids[item["label"]]
        native_top_id = int(np.argmax(raw))
        native_top = float(raw[native_top_id])
        desired_logit = float(raw[desired_id])
        need = max(0.0, (native_top - desired_logit + target_margin) / desired_weight)
        if item["split"] == contract["calibration"]["split"]:
            required_beta.append(need)
        prepared.append(
            {
                "id": item["id"],
                "record": item["record"],
                "split": item["split"],
                "label": item["label"],
                "ridge_score": score,
                "continue_weight": continue_weight,
                "desired_gate_weight": desired_weight,
                "desired_token_id": desired_id,
                "native_top_token_id": native_top_id,
                "native_top_logit": native_top,
                "native_desired_logit": desired_logit,
                "training_required_beta": need,
            }
        )
    beta = max(required_beta)

    evaluated: list[dict[str, Any]] = []
    for row in prepared:
        corrected = logits[row["record"]].copy()
        continue_weight = row["continue_weight"]
        corrected[token_ids["continue_tool"]] += beta * continue_weight
        corrected[token_ids["finish"]] += beta * (1.0 - continue_weight)
        top_two = np.argpartition(corrected, -2)[-2:]
        top_two = top_two[np.argsort(corrected[top_two])[::-1]]
        top_id = int(top_two[0])
        runner_up_id = int(top_two[1])
        corrected_margin = float(corrected[top_id] - corrected[runner_up_id])
        evaluated.append(
            {
                **row,
                "beta": beta,
                "corrected_top_token_id": top_id,
                "corrected_runner_up_token_id": runner_up_id,
                "corrected_top1_correct": top_id == row["desired_token_id"],
                "corrected_margin": corrected_margin,
            }
        )

    metrics = {
        "beta": beta,
        "train_control_top1_accuracy": split_accuracy(evaluated, "train"),
        "validation_control_top1_accuracy": split_accuracy(evaluated, "validation"),
        "test_control_top1_accuracy": split_accuracy(evaluated, "test"),
        "minimum_corrected_heldout_margin": min(
            row["corrected_margin"] for row in evaluated if row["split"] != "train"
        ),
    }
    gate = contract["pass_gate"]
    passed = (
        metrics["beta"] <= float(gate["maximum_beta"])
        and metrics["train_control_top1_accuracy"] >= float(gate["train_control_top1_accuracy"])
        and metrics["validation_control_top1_accuracy"] >= float(gate["validation_control_top1_accuracy"])
        and metrics["test_control_top1_accuracy"] >= float(gate["test_control_top1_accuracy"])
        and metrics["minimum_corrected_heldout_margin"] >= float(gate["minimum_corrected_heldout_margin"])
    )
    report = {
        "format": "colorlm-v28-control-bias-report-v1",
        "contract": manifest_path(args.contract),
        "contract_sha256": sha256_file(args.contract),
        "capture_sha256": index["capture_sha256"],
        "metrics": metrics,
        "evaluated": evaluated,
        "gate_passed": passed,
        "decision": (
            "allow two-token runtime executive control prototype"
            if passed
            else contract["failure_action"]
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
