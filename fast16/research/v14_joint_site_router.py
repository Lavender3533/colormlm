"""Audit a joint four-state Neural Bus router on existing v13 counterfactual data.

This is an offline research gate. It does not emit a runtime plan and never promotes
an artifact without strict leave-one-task-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from v13_counterfactual_router import (
    V13Error,
    sha256_file,
    unit_l2,
    validate_route_shard,
)


ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "fast16" / "research" / "v13_live"
ROUTE_NAMES = ("no_op", "l12_k3", "l28_k3", "both_k3")
BASELINE_ROUTE = 1


@dataclass(frozen=True, order=True)
class Config:
    kernel: str
    scale: float
    alpha: float
    margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ColorLM v14联合站点路由离线审计")
    parser.add_argument("--no-op", type=Path, default=LIVE / "no_op_v2.jsonl")
    parser.add_argument("--l12", type=Path, default=LIVE / "l12_k3.jsonl")
    parser.add_argument("--l28", type=Path, default=LIVE / "l28_k3.jsonl")
    parser.add_argument("--both", type=Path, default=LIVE / "k3_v2.jsonl")
    parser.add_argument(
        "--feature-sidecar",
        type=Path,
        help="可选的同sample隐藏态NPZ；默认使用no-op shard记录的sidecar",
    )
    parser.add_argument("--feature-trajectory", default="no_op")
    parser.add_argument(
        "--residual-sidecar",
        type=Path,
        help="使用L28的6维实际K3残差特征替代完整隐藏态。",
    )
    parser.add_argument(
        "--decision-mode",
        choices=("joint4", "l28_after_l12"),
        default="joint4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fast16" / "research" / "v14_joint_site_router_report.json",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise V13Error(message)


def load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, str]]:
    paths = (args.no_op, args.l12, args.l28, args.both)
    expected_routes = ("no_op", "k3", "k3", "k3")
    validated = [validate_route_shard(route, path) for route, path in zip(expected_routes, paths)]
    rows = [item[0] for item in validated]
    manifests = [item[1] for item in validated]
    manifest_paths = [item[2] for item in validated]

    sample_ids = [str(row["sample_id"]) for row in rows[0]]
    require(len(sample_ids) > 0 and len(set(sample_ids)) == len(sample_ids), "no-op样本为空或重复")
    task_ids = [str(row["task_id"]) for row in rows[0]]
    for route_rows in rows[1:]:
        require([str(row["sample_id"]) for row in route_rows] == sample_ids, "四条路径样本顺序不一致")
        require([str(row["task_id"]) for row in route_rows] == task_ids, "四条路径task顺序不一致")
    require(len(set(task_ids)) >= 3, "联合路由审计至少需要三个任务组")

    nll = np.asarray(
        [[float(row["target_nll"]) for row in route_rows] for route_rows in rows],
        dtype=np.float64,
    ).T
    require(nll.shape == (len(sample_ids), len(ROUTE_NAMES)), "NLL矩阵形状错误")
    require(np.isfinite(nll).all() and np.all(nll >= 0), "NLL矩阵包含无效值")

    hidden_record = manifests[0].get("hidden")
    require(isinstance(hidden_record, dict), "no-op manifest缺少隐藏状态记录")
    hidden_file = hidden_record.get("file")
    require(isinstance(hidden_file, str) and hidden_file, "no-op manifest隐藏状态路径无效")
    hidden_path = (
        args.residual_sidecar.resolve()
        if args.residual_sidecar is not None
        else args.feature_sidecar.resolve()
        if args.feature_sidecar is not None
        else (manifest_paths[0].parent / hidden_file).resolve()
    )
    require(hidden_path.is_file(), f"特征sidecar不存在: {hidden_path}")
    if args.residual_sidecar is None and args.feature_sidecar is None:
        require(sha256_file(hidden_path) == hidden_record.get("sha256"), "隐藏状态sidecar SHA-256不匹配")

    with np.load(hidden_path, allow_pickle=False) as archive:
        archived_ids = [str(value) for value in archive["sample_ids"].tolist()]
        require(archived_ids == sample_ids, "特征sample顺序与路径样本不一致")
        if args.residual_sidecar is not None:
            features = np.asarray(archive["layer_28"], dtype=np.float64)
            names = [str(value) for value in archive["feature_names"].tolist()]
            require(
                names == [
                    "hidden_rms", "native_rms", "k3_delta_rms",
                    "hidden_delta_cos", "native_delta_cos", "energy_gate",
                ],
                "K3残差特征名称或顺序错误",
            )
            require(features.shape == (len(sample_ids), 6), "K3残差特征矩阵形状错误")
        else:
            l12 = np.asarray(archive["layer_12"], dtype=np.float32)
            l28 = np.asarray(archive["layer_28"], dtype=np.float32)
            require(l12.shape == l28.shape == (len(sample_ids), 2048), "隐藏状态矩阵形状错误")
            features = np.concatenate((unit_l2(l12), unit_l2(l28)), axis=1).astype(np.float64)
            features /= math.sqrt(2.0)
    require(np.isfinite(features).all(), "路由特征包含NaN或Inf")

    hashes = {name: sha256_file(path) for name, path in zip(ROUTE_NAMES, paths)}
    hashes["feature_hidden"] = sha256_file(hidden_path)
    return features, nll, sample_ids, task_ids, hashes


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1, keepdims=True).T
        - 2.0 * (left @ right.T)
    )
    return np.maximum(distances, 0.0)


def kernel_matrices(
    train: np.ndarray,
    test: np.ndarray,
    kernel: str,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if kernel == "linear":
        return train @ train.T, test @ train.T
    train_dist = squared_distances(train, train)
    off_diagonal = train_dist[np.triu_indices(len(train), 1)]
    positive = off_diagonal[off_diagonal > 1e-12]
    median = float(np.median(positive)) if len(positive) else 1.0
    gamma = scale / max(median, 1e-12)
    return np.exp(-gamma * train_dist), np.exp(-gamma * squared_distances(test, train))


def predict_deltas(
    train_x: np.ndarray,
    train_delta: np.ndarray,
    test_x: np.ndarray,
    config: Config,
    standardize: bool,
) -> np.ndarray:
    if standardize:
        mean = np.mean(train_x, axis=0, keepdims=True)
        scale = np.std(train_x, axis=0, keepdims=True)
        scale = np.where(scale > 1e-8, scale, 1.0)
        train_x = (train_x - mean) / scale
        test_x = (test_x - mean) / scale
    train_kernel, test_kernel = kernel_matrices(train_x, test_x, config.kernel, config.scale)
    regularized = train_kernel + config.alpha * np.eye(len(train_kernel), dtype=np.float64)
    try:
        coefficients = np.linalg.solve(regularized, train_delta)
    except np.linalg.LinAlgError as error:
        raise V13Error(f"联合路由核回归求解失败: {error}") from error
    prediction = test_kernel @ coefficients
    require(np.isfinite(prediction).all(), "联合路由预测包含NaN或Inf")
    prediction[:, BASELINE_ROUTE] = 0.0
    return prediction


def select_routes(
    predicted_delta: np.ndarray,
    margin: float,
    allowed_routes: tuple[int, ...],
) -> np.ndarray:
    require(BASELINE_ROUTE in allowed_routes, "允许路径必须包含固定L12安全基线")
    allowed = np.asarray(allowed_routes, dtype=np.int64)
    selected = allowed[np.argmin(predicted_delta[:, allowed], axis=1)]
    predicted_gain = -predicted_delta[np.arange(len(selected)), selected]
    selected = np.where(predicted_gain > margin, selected, BASELINE_ROUTE)
    return selected.astype(np.int64)


def configurations() -> list[Config]:
    result: list[Config] = []
    for kernel, scales in (("linear", (1.0,)), ("rbf", (0.5, 1.0, 2.0))):
        for scale in scales:
            for alpha in (0.1, 1.0, 10.0):
                for margin in (0.0, 0.01):
                    result.append(Config(kernel, scale, alpha, margin))
    return result


def choose_config_nested(
    features: np.ndarray,
    deltas: np.ndarray,
    nll: np.ndarray,
    task_ids: np.ndarray,
    outer_train: np.ndarray,
    allowed_routes: tuple[int, ...],
    standardize: bool,
) -> tuple[Config, float]:
    scores = {config: [0.0, 0] for config in configurations()}
    inner_tasks = sorted(set(task_ids[outer_train].tolist()))
    require(len(inner_tasks) >= 2, "内层交叉验证任务不足")
    for heldout_task in inner_tasks:
        inner_test = outer_train[task_ids[outer_train] == heldout_task]
        inner_train = outer_train[task_ids[outer_train] != heldout_task]
        prediction_cache: dict[tuple[str, float, float], np.ndarray] = {}
        for config in scores:
            key = (config.kernel, config.scale, config.alpha)
            if key not in prediction_cache:
                prediction_cache[key] = predict_deltas(
                    features[inner_train],
                    deltas[inner_train],
                    features[inner_test],
                    config,
                    standardize,
                )
            selected = select_routes(prediction_cache[key], config.margin, allowed_routes)
            scores[config][0] += float(np.sum(nll[inner_test, selected]))
            scores[config][1] += len(inner_test)
    ranked = sorted(
        (total / count, config) for config, (total, count) in scores.items() if count > 0
    )
    require(bool(ranked), "内层交叉验证没有有效配置")
    return ranked[0][1], float(ranked[0][0])


def task_means(values: np.ndarray, task_ids: np.ndarray) -> dict[str, float]:
    return {
        task: float(np.mean(values[task_ids == task]))
        for task in sorted(set(task_ids.tolist()))
    }


def main() -> int:
    args = parse_args()
    try:
        features, nll, sample_ids, raw_task_ids, hashes = load_inputs(args)
        task_ids = np.asarray(raw_task_ids)
        allowed_routes = (
            tuple(range(len(ROUTE_NAMES)))
            if args.decision_mode == "joint4"
            else (BASELINE_ROUTE, 3)
        )
        standardize = args.residual_sidecar is not None
        deltas = nll - nll[:, [BASELINE_ROUTE]]
        selected = np.full(len(sample_ids), BASELINE_ROUTE, dtype=np.int64)
        fold_records: list[dict[str, Any]] = []

        for heldout_task in sorted(set(raw_task_ids)):
            outer_test = np.flatnonzero(task_ids == heldout_task)
            outer_train = np.flatnonzero(task_ids != heldout_task)
            config, inner_nll = choose_config_nested(
                features, deltas, nll, task_ids, outer_train, allowed_routes, standardize
            )
            prediction = predict_deltas(
                features[outer_train], deltas[outer_train], features[outer_test], config,
                standardize,
            )
            fold_selected = select_routes(prediction, config.margin, allowed_routes)
            selected[outer_test] = fold_selected
            fold_records.append(
                {
                    "heldout_task": heldout_task,
                    "tokens": len(outer_test),
                    "selected_config": {
                        "kernel": config.kernel,
                        "scale": config.scale,
                        "alpha": config.alpha,
                        "margin": config.margin,
                    },
                    "inner_mean_nll": inner_nll,
                    "heldout_routed_mean_nll": float(np.mean(nll[outer_test, fold_selected])),
                    "heldout_l12_mean_nll": float(np.mean(nll[outer_test, BASELINE_ROUTE])),
                }
            )

        routed_nll = nll[np.arange(len(nll)), selected]
        fixed_means = np.mean(nll, axis=0)
        token_oracle = np.min(nll, axis=1)
        allowed_token_oracle = np.min(nll[:, allowed_routes], axis=1)
        task_oracle = np.empty(len(nll), dtype=np.float64)
        task_oracle_routes: dict[str, str] = {}
        for task in sorted(set(raw_task_ids)):
            indices = np.flatnonzero(task_ids == task)
            route = int(np.argmin(np.mean(nll[indices], axis=0)))
            task_oracle[indices] = nll[indices, route]
            task_oracle_routes[task] = ROUTE_NAMES[route]

        baseline_nll = nll[:, BASELINE_ROUTE]
        task_gain = {
            task: float(np.mean(baseline_nll[task_ids == task] - routed_nll[task_ids == task]))
            for task in sorted(set(raw_task_ids))
        }
        gains = np.asarray(list(task_gain.values()), dtype=np.float64)
        rng = np.random.default_rng(20260730)
        bootstrap = np.mean(
            gains[rng.integers(0, len(gains), size=(10000, len(gains)))], axis=1
        )
        ci_low, ci_high = (float(value) for value in np.quantile(bootstrap, (0.05, 0.95)))
        mean_routed = float(np.mean(routed_nll))
        mean_baseline = float(fixed_means[BASELINE_ROUTE])
        relative_improvement = (mean_baseline - mean_routed) / mean_baseline
        tasks_improved = int(np.sum(gains > 0))
        promotable = relative_improvement >= 0.002 and tasks_improved >= 7 and ci_low > 0

        interaction = nll[:, 3] - nll[:, 1] - nll[:, 2] + nll[:, 0]
        best_counts = np.bincount(np.argmin(nll, axis=1), minlength=len(ROUTE_NAMES))
        report = {
            "format": (
                "colorlm-v15-residual-aware-router-audit-v1"
                if args.residual_sidecar is not None
                else "colorlm-v14-joint-site-router-audit-v1"
            ),
            "date": "2026-07-30",
            "status": "candidate_for_fresh_holdout" if promotable else "rejected_control",
            "hypothesis": (
                "A joint four-state router can model destructive L12/L28 interaction better "
                "than two independent site routers."
                if args.decision_mode == "joint4"
                else "An L12-conditioned binary gate can safely decide whether L28 should be added."
            ),
            "data": {
                "samples": len(sample_ids),
                "tasks": len(set(raw_task_ids)),
                "feature": (
                    "six standardized L28 K3 residual features"
                    if args.residual_sidecar is not None
                    else "concatenated unit-L2 attn_post_norm from L12 and L28"
                ),
                "feature_trajectory": args.feature_trajectory,
                "decision_mode": args.decision_mode,
                "allowed_routes": [ROUTE_NAMES[index] for index in allowed_routes],
                "source_sha256": hashes,
            },
            "counterfactual_surface": {
                "route_mean_nll": {
                    route: float(value) for route, value in zip(ROUTE_NAMES, fixed_means)
                },
                "token_oracle_mean_nll": float(np.mean(token_oracle)),
                "allowed_token_oracle_mean_nll": float(np.mean(allowed_token_oracle)),
                "allowed_token_oracle_relative_improvement_vs_l12": float(
                    (mean_baseline - np.mean(allowed_token_oracle)) / mean_baseline
                ),
                "token_oracle_relative_improvement_vs_l12": float(
                    (mean_baseline - np.mean(token_oracle)) / mean_baseline
                ),
                "token_oracle_best_route_counts": {
                    route: int(value) for route, value in zip(ROUTE_NAMES, best_counts)
                },
                "task_oracle_mean_nll": float(np.mean(task_oracle)),
                "task_oracle_routes": task_oracle_routes,
                "mean_additive_interaction_nll": float(np.mean(interaction)),
                "both_isolated_good_but_joint_bad_tokens": int(
                    np.sum((nll[:, 1] < nll[:, 0]) & (nll[:, 2] < nll[:, 0]) & (nll[:, 3] > nll[:, 0]))
                ),
            },
            "nested_loto": {
                "protocol": "outer leave-one-task-out; inner leave-one-task-out selects kernel/regularization/margin",
                "objective": "actual routed next-token NLL, not route classification accuracy",
                "routed_mean_nll": mean_routed,
                "fixed_l12_mean_nll": mean_baseline,
                "relative_improvement_vs_l12": relative_improvement,
                "tasks_improved": tasks_improved,
                "tasks_total": len(gains),
                "task_mean_gain_nll": task_gain,
                "task_bootstrap_90pct_gain_ci": [ci_low, ci_high],
                "selected_route_counts": {
                    route: int(value)
                    for route, value in zip(
                        ROUTE_NAMES,
                        np.bincount(selected, minlength=len(ROUTE_NAMES)),
                    )
                },
                "folds": fold_records,
            },
            "promotion_gate": {
                "relative_nll_improvement_at_least": 0.002,
                "tasks_improved_at_least": 7,
                "bootstrap_90pct_lower_bound_above_zero": True,
                "passed": promotable,
            },
            "decision": (
                "Proceed only to a fresh-task holdout; do not emit a runtime plan yet."
                if promotable
                else "Stop the joint kernel router; retain v13 fixed L12 as the formal graph."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, V13Error) as error:
        print(f"v14联合站点路由审计失败: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
