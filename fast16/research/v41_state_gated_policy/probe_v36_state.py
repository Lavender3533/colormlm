"""Probe whether v36 terminal hidden separates continue-tool from finish states."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def import_fit(path: Path):
    spec = importlib.util.spec_from_file_location("v29_fit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-implementation", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.01)
    args = parser.parse_args()

    fit = import_fit(args.fit_implementation)
    capture = fit.load_capture(args.capture)
    teacher = read_jsonl(args.teacher)
    tasks = read_jsonl(args.tasks)
    task_by_id = {row["id"]: row for row in tasks}
    selected = [
        (index, row)
        for index, row in enumerate(teacher)
        if int(row["token_index"]) == 0
    ]
    if len(selected) != len(tasks):
        raise ValueError("每个任务必须恰有一个首决策状态")

    x = np.stack([capture[index][fit.BASE_HIDDEN] for index, _ in selected]).astype(np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    y = np.asarray([
        1.0 if task_by_id[row["task_id"]]["family"] == "continue" else -1.0
        for _, row in selected
    ])
    splits = [task_by_id[row["task_id"]]["split"] for _, row in selected]
    train = np.asarray([index for index, split in enumerate(splits) if split == "train"])
    xt = x[train]
    yt = y[train]
    x_mean = xt.mean(axis=0)
    y_mean = float(yt.mean())
    xc = xt - x_mean
    yc = yt - y_mean
    dual = np.linalg.solve(
        xc @ xc.T + args.ridge * np.eye(len(train), dtype=np.float64), yc
    )
    weight = xc.T @ dual
    bias = y_mean - float(x_mean @ weight)
    score = x @ weight + bias
    prediction = np.where(score >= 0.0, 1.0, -1.0)
    rows = []
    for (_, teacher_row), split, expected, value, predicted in zip(
        selected, splits, y, score, prediction, strict=True
    ):
        rows.append(
            {
                "task_id": teacher_row["task_id"],
                "split": split,
                "expected": "continue" if expected > 0 else "finish",
                "score": float(value),
                "prediction": "continue" if predicted > 0 else "finish",
                "correct": bool(predicted == expected),
            }
        )
    metrics = {}
    for split in ("train", "validation", "test"):
        part = [row for row in rows if row["split"] == split]
        metrics[split] = {
            "correct": sum(row["correct"] for row in part),
            "total": len(part),
            "accuracy": float(np.mean([row["correct"] for row in part])),
            "minimum_absolute_margin": min(abs(row["score"]) for row in part),
        }
    report = {
        "format": "colorlm-v41-v36-state-probe-v1",
        "role": "consumed-development-only",
        "ridge_lambda": args.ridge,
        "train_samples": len(train),
        "weight_l2": float(np.linalg.norm(weight)),
        "bias": bias,
        "metrics": metrics,
        "rows": rows,
        "decision_rule": "continue iff normalized_hidden dot weight + bias >= 0",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
