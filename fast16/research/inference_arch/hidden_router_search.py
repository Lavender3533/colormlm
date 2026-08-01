#!/usr/bin/env python3
"""用完整 donor hidden 做嵌套 LOTO 的动态 alpha/no-op 路由搜索。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from offline_bus import Capture, historical_branch, load_capture, softmax, token_nll


ALPHA_GRID = np.asarray(
    (0.0, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.02, 0.03),
    dtype=np.float64,
)
L2_GRID = (0.1, 1.0, 10.0)
THRESHOLD_GRID = (0.1, 0.3, 0.5)
SHRINK_GRID = (0.5, 1.0)


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    target_mean: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _base_summary(logits: np.ndarray) -> np.ndarray:
    probabilities = softmax(logits)
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1
    ) / math.log(logits.shape[1])
    top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
    top_two.sort(axis=1)
    margin = top_two[:, 1] - top_two[:, 0]
    return np.column_stack((entropy, margin))


def build_router_features(capture: Capture, donor_index: int) -> dict[str, np.ndarray]:
    donor = capture.donors[donor_index]
    if donor.hidden is None:
        raise ValueError("动态hidden路由要求donor hidden")
    hidden = np.asarray(donor.hidden, dtype=np.float64)
    hidden_rms = np.sqrt(np.mean(np.square(hidden), axis=1, keepdims=True))
    hidden_unit = hidden / np.maximum(hidden_rms, 1e-12)
    base = _base_summary(capture.base_logits)
    return {
        "base-only": base,
        "hidden-unit": hidden_unit,
        "base+hidden-unit": np.column_stack((base, hidden_unit)),
    }


def counterfactual_curve(
    capture: Capture, donor_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch = historical_branch(capture, donor_index)
    losses = np.column_stack(
        [
            token_nll(capture.base_logits + alpha * branch, capture.labels)
            for alpha in ALPHA_GRID
        ]
    )
    best = np.argmin(losses, axis=1)
    target = ALPHA_GRID[best] / ALPHA_GRID[-1]
    gain = losses[:, 0] - losses[np.arange(losses.shape[0]), best]
    target[gain < 1e-6] = 0.0
    return losses, target, branch


def fit_ridge(
    features: np.ndarray, target: np.ndarray, indices: np.ndarray, l2: float
) -> RidgeModel:
    train = features[indices]
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (train - mean) / scale
    centered_target = target[indices] - float(np.mean(target[indices]))
    # p远大于n时走对偶闭式解，只解至多54×54矩阵。
    kernel = normalized @ normalized.T
    dual = np.linalg.solve(
        kernel + np.eye(kernel.shape[0], dtype=np.float64) * float(l2),
        centered_target,
    )
    weights = normalized.T @ dual
    return RidgeModel(mean, scale, weights, float(np.mean(target[indices])))


def predict_ridge(model: RidgeModel, features: np.ndarray) -> np.ndarray:
    normalized = (features - model.mean) / model.scale
    return normalized @ model.weights + model.target_mean


def route_alpha(prediction: np.ndarray, threshold: float, shrink: float) -> np.ndarray:
    mass = np.clip(prediction, 0.0, 1.0)
    mass[mass < threshold] = 0.0
    return mass * ALPHA_GRID[-1] * shrink


def interpolate_losses(
    losses: np.ndarray, indices: np.ndarray, alphas: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            np.interp(alpha, ALPHA_GRID, losses[index])
            for index, alpha in zip(indices, alphas)
        ],
        dtype=np.float64,
    )


def _task_score(
    delta: np.ndarray, task_ids: np.ndarray
) -> tuple[float, float, int, int]:
    means = np.asarray(
        [np.mean(delta[task_ids == task]) for task in np.unique(task_ids)]
    )
    return (
        float(np.mean(delta)),
        float(np.max(means)),
        int(np.sum(means < 0.0)),
        int(np.sum(means > 0.0)),
    )


def select_config(
    features_by_name: dict[str, np.ndarray],
    target: np.ndarray,
    losses: np.ndarray,
    task_ids: np.ndarray,
    candidate_indices: np.ndarray,
) -> dict[str, object]:
    candidate_tasks = np.unique(task_ids[candidate_indices])
    best: tuple[tuple[float, float, int, float, float], dict[str, object]] | None = None
    for feature_name, features in features_by_name.items():
        for l2 in L2_GRID:
            prediction = np.zeros(candidate_indices.size, dtype=np.float64)
            for task in candidate_tasks:
                validation = candidate_indices[task_ids[candidate_indices] == task]
                training = candidate_indices[task_ids[candidate_indices] != task]
                model = fit_ridge(features, target, training, l2)
                positions = np.flatnonzero(task_ids[candidate_indices] == task)
                prediction[positions] = predict_ridge(model, features[validation])
            for threshold in THRESHOLD_GRID:
                for shrink in SHRINK_GRID:
                    alphas = route_alpha(prediction, threshold, shrink)
                    routed = interpolate_losses(losses, candidate_indices, alphas)
                    delta = routed - losses[candidate_indices, 0]
                    mean, worst, wins, task_losses = _task_score(
                        delta, task_ids[candidate_indices]
                    )
                    # 先最小化平均NLL，再偏向最坏任务较小、任务胜场较多的配置。
                    score = (
                        mean,
                        max(worst, 0.0),
                        task_losses,
                        -wins,
                        float(np.mean(alphas > 0.0)),
                    )
                    config = {
                        "feature": feature_name,
                        "l2": float(l2),
                        "threshold": float(threshold),
                        "shrink": float(shrink),
                        "inner_mean_delta": mean,
                        "inner_worst_task_delta": worst,
                        "inner_task_wins": wins,
                        "inner_task_losses": task_losses,
                    }
                    if best is None or score < best[0]:
                        best = (score, config)
    assert best is not None
    return best[1]


def nested_loto(capture: Capture, donor_index: int) -> dict[str, object]:
    losses, target, branch = counterfactual_curve(capture, donor_index)
    features_by_name = build_router_features(capture, donor_index)
    tasks = np.unique(capture.task_ids)
    predicted_alpha = np.zeros(capture.labels.size, dtype=np.float64)
    fold_reports: list[dict[str, object]] = []

    for task in tasks:
        test = np.flatnonzero(capture.task_ids == task)
        train = np.flatnonzero(capture.task_ids != task)
        config = select_config(
            features_by_name, target, losses, capture.task_ids, train
        )
        model = fit_ridge(
            features_by_name[str(config["feature"])],
            target,
            train,
            float(config["l2"]),
        )
        prediction = predict_ridge(
            model, features_by_name[str(config["feature"])][test]
        )
        alpha = route_alpha(
            prediction, float(config["threshold"]), float(config["shrink"])
        )
        predicted_alpha[test] = alpha
        fold_reports.append(
            {
                "held_out_task": str(task),
                "sample_count": int(test.size),
                "active_rate": float(np.mean(alpha > 0.0)),
                "mean_alpha": float(np.mean(alpha)),
                "config": config,
            }
        )

    routed_logits = capture.base_logits + predicted_alpha[:, None] * branch
    base_nll = losses[:, 0]
    routed_nll = token_nll(routed_logits, capture.labels)
    delta = routed_nll - base_nll
    mean, worst, task_wins, task_losses = _task_score(delta, capture.task_ids)
    task_mean = {
        str(task): float(np.mean(delta[capture.task_ids == task])) for task in tasks
    }

    all_indices = np.arange(capture.labels.size)
    final_config = select_config(
        features_by_name, target, losses, capture.task_ids, all_indices
    )
    final_features = features_by_name[str(final_config["feature"])]
    final_model = fit_ridge(
        final_features, target, all_indices, float(final_config["l2"])
    )
    return {
        "format": "colorlm-hidden-dynamic-alpha-nested-loto-v1",
        "sample_count": int(capture.labels.size),
        "task_count": int(tasks.size),
        "alpha_grid": ALPHA_GRID.tolist(),
        "mean_delta": mean,
        "median_delta": float(np.median(delta)),
        "worst_task_delta": worst,
        "token_wins": int(np.sum(delta < -1e-12)),
        "token_losses": int(np.sum(delta > 1e-12)),
        "token_equal": int(np.sum(np.abs(delta) <= 1e-12)),
        "task_wins": task_wins,
        "task_losses": task_losses,
        "task_mean_delta": task_mean,
        "exact_no_op_rate": float(np.mean(predicted_alpha == 0.0)),
        "mean_active_alpha": (
            float(np.mean(predicted_alpha[predicted_alpha > 0.0]))
            if np.any(predicted_alpha > 0.0)
            else 0.0
        ),
        "development_signal": bool(
            mean < 0.0 and task_wins > task_losses and worst <= 0.0
        ),
        "folds": fold_reports,
        "final_model": {
            "feature": final_config["feature"],
            "feature_count": int(final_model.weights.size),
            "l2": final_config["l2"],
            "threshold": final_config["threshold"],
            "shrink": final_config["shrink"],
            "mean": final_model.mean.tolist(),
            "scale": final_model.scale.tolist(),
            "weights": final_model.weights.tolist(),
            "target_mean": final_model.target_mean,
        },
        "feature_contract": (
            "continuous base-logit summary and/or donor hidden only; text, task id and label "
            "are forbidden at inference"
        ),
    }


def _json_dump(value: object, output: Path | None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
    else:
        output.write_text(rendered, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--donor", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    capture = load_capture(args.capture)
    report = nested_loto(capture, args.donor)
    report["capture"] = str(args.capture.resolve())
    report["capture_sha256"] = _sha256(args.capture)
    _json_dump(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
