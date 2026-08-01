#!/usr/bin/env python3
"""ColorLM Neural Output Bus 的纯 CPU 离线回放与多 donor 验证原型。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class DonorCapture:
    """一个 donor 在 base 词表上的精确映射输出及末端隐藏态。"""

    logits: np.ndarray  # [sample, mapped_token]
    base_ids: np.ndarray  # [mapped_token]
    hidden: np.ndarray | None = None  # [sample, donor_hidden]


@dataclass(frozen=True)
class Capture:
    """同一批 teacher-forced 前缀的一次采集结果。"""

    base_logits: np.ndarray  # [sample, base_vocab]
    labels: np.ndarray  # [sample]
    task_ids: np.ndarray  # [sample], Unicode string
    donors: tuple[DonorCapture, ...]

    def validate(self) -> None:
        if self.base_logits.ndim != 2:
            raise ValueError("base_logits 必须是 [sample, vocab]")
        sample_count, vocab_size = self.base_logits.shape
        if self.labels.shape != (sample_count,):
            raise ValueError("labels 形状不匹配")
        if self.task_ids.shape != (sample_count,):
            raise ValueError("task_ids 形状不匹配")
        if np.any(self.labels < 0) or np.any(self.labels >= vocab_size):
            raise ValueError("labels 超出 base 词表")
        for index, donor in enumerate(self.donors):
            if donor.logits.ndim != 2 or donor.logits.shape[0] != sample_count:
                raise ValueError(f"donor_{index}_logits 形状不匹配")
            if donor.base_ids.shape != (donor.logits.shape[1],):
                raise ValueError(f"donor_{index}_base_ids 形状不匹配")
            if np.any(donor.base_ids < 0) or np.any(donor.base_ids >= vocab_size):
                raise ValueError(f"donor_{index}_base_ids 超出 base 词表")
            if np.unique(donor.base_ids).size != donor.base_ids.size:
                raise ValueError(f"donor_{index}_base_ids 存在碰撞")
            if donor.hidden is not None:
                if donor.hidden.ndim != 2 or donor.hidden.shape[0] != sample_count:
                    raise ValueError(f"donor_{index}_hidden 形状不匹配")


@dataclass(frozen=True)
class GateModel:
    donor_indices: tuple[int, ...]
    temperatures: tuple[float, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray  # [no-op + donors, feature]


def _reconfigure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def load_capture(path: Path) -> Capture:
    with np.load(path, allow_pickle=False) as data:
        base_logits = np.asarray(data["base_logits"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.int64)
        task_ids = np.asarray(data["task_ids"], dtype=np.str_)
        donors: list[DonorCapture] = []
        index = 0
        while f"donor_{index}_logits" in data.files:
            hidden_key = f"donor_{index}_hidden"
            donors.append(
                DonorCapture(
                    logits=np.asarray(data[f"donor_{index}_logits"], dtype=np.float64),
                    base_ids=np.asarray(data[f"donor_{index}_base_ids"], dtype=np.int64),
                    hidden=(
                        np.asarray(data[hidden_key], dtype=np.float64)
                        if hidden_key in data.files
                        else None
                    ),
                )
            )
            index += 1
    capture = Capture(base_logits, labels, task_ids, tuple(donors))
    capture.validate()
    return capture


def subset_capture(capture: Capture, indices: np.ndarray) -> Capture:
    donors = tuple(
        DonorCapture(
            logits=donor.logits[indices],
            base_ids=donor.base_ids,
            hidden=donor.hidden[indices] if donor.hidden is not None else None,
        )
        for donor in capture.donors
    )
    result = Capture(
        capture.base_logits[indices],
        capture.labels[indices],
        capture.task_ids[indices],
        donors,
    )
    result.validate()
    return result


def logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(values - maximum), axis=axis)
    )


def softmax(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(values - maximum)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def token_nll(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rows = np.arange(logits.shape[0])
    return logsumexp(logits) - logits[rows, labels]


def sparsemax(scores: np.ndarray) -> np.ndarray:
    """Martins & Astudillo sparsemax；base-only 支持集即为精确 no-op。"""

    ordered = np.sort(scores, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)
    ranks = np.arange(1, scores.shape[1] + 1, dtype=np.float64)[None, :]
    support = 1.0 + ranks * ordered > cumulative
    support_size = np.sum(support, axis=1)
    tau = (cumulative[np.arange(scores.shape[0]), support_size - 1] - 1.0) / support_size
    return np.maximum(scores - tau[:, None], 0.0)


def historical_branch(capture: Capture, donor_index: int) -> np.ndarray:
    """重建 v19 的 mapped donor logits 减行均值、再 scatter 的分支。"""

    donor = capture.donors[donor_index]
    centered = donor.logits - np.mean(donor.logits, axis=1, keepdims=True)
    branch = np.zeros_like(capture.base_logits, dtype=np.float64)
    branch[:, donor.base_ids] = centered
    return branch


def replay_alphas(
    capture: Capture, donor_index: int, alphas: Sequence[float]
) -> dict[str, object]:
    branch = historical_branch(capture, donor_index)
    baseline = token_nll(capture.base_logits, capture.labels)
    points: list[dict[str, object]] = []
    for alpha in alphas:
        current = token_nll(capture.base_logits + float(alpha) * branch, capture.labels)
        delta = current - baseline
        points.append(
            {
                "alpha": float(alpha),
                "mean_nll": float(np.mean(current)),
                "median_nll": float(np.median(current)),
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "wins": int(np.sum(delta < -1e-12)),
                "losses": int(np.sum(delta > 1e-12)),
                "equal": int(np.sum(np.abs(delta) <= 1e-12)),
            }
        )
    return {
        "format": "colorlm-offline-alpha-replay-v1",
        "sample_count": int(capture.labels.size),
        "donor_index": int(donor_index),
        "alpha_zero_exact": bool(
            np.array_equal(capture.base_logits + 0.0 * branch, capture.base_logits)
        ),
        "points": points,
    }


def donor_residual(capture: Capture, donor_index: int, temperature: float) -> np.ndarray:
    """构造 shift-invariant、base 概率加权零均值的 donor log-odds 残差。"""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature 必须为正有限数")
    donor = capture.donors[donor_index]
    base_mapped = capture.base_logits[:, donor.base_ids]
    raw = donor.logits / temperature - base_mapped
    raw -= np.mean(raw, axis=1, keepdims=True)
    base_prob = softmax(capture.base_logits)[:, donor.base_ids]
    covered_mass = np.sum(base_prob, axis=1, keepdims=True)
    correction = np.sum(base_prob * raw, axis=1, keepdims=True) / np.maximum(
        covered_mass, EPS
    )
    raw -= correction
    residual = np.zeros_like(capture.base_logits, dtype=np.float64)
    residual[:, donor.base_ids] = raw
    return residual


def _entropy_and_margin(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = softmax(logits)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, EPS)), axis=1)
    entropy /= max(math.log(logits.shape[1]), EPS)
    top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
    top_two.sort(axis=1)
    margin = top_two[:, 1] - top_two[:, 0]
    return entropy, margin


def build_features(capture: Capture, donor_indices: Sequence[int]) -> np.ndarray:
    """只从数值状态构造特征；不读取文本、task_id 或标签。"""

    columns: list[np.ndarray] = [np.ones(capture.labels.size, dtype=np.float64)]
    base_entropy, base_margin = _entropy_and_margin(capture.base_logits)
    columns.extend((base_entropy, base_margin))

    donor_probabilities: list[tuple[np.ndarray, np.ndarray]] = []
    for donor_index in donor_indices:
        donor = capture.donors[donor_index]
        donor_entropy, donor_margin = _entropy_and_margin(donor.logits)
        base_conditional = softmax(capture.base_logits[:, donor.base_ids])
        donor_conditional = softmax(donor.logits)
        mixture = 0.5 * (base_conditional + donor_conditional)
        js = 0.5 * np.sum(
            base_conditional
            * (np.log(np.maximum(base_conditional, EPS)) - np.log(np.maximum(mixture, EPS))),
            axis=1,
        )
        js += 0.5 * np.sum(
            donor_conditional
            * (np.log(np.maximum(donor_conditional, EPS)) - np.log(np.maximum(mixture, EPS))),
            axis=1,
        )
        residual_rms = np.sqrt(
            np.mean(
                np.square(
                    (donor.logits - np.mean(donor.logits, axis=1, keepdims=True))
                    - (
                        capture.base_logits[:, donor.base_ids]
                        - np.mean(
                            capture.base_logits[:, donor.base_ids], axis=1, keepdims=True
                        )
                    )
                ),
                axis=1,
            )
        )
        if donor.hidden is None:
            hidden_rms = np.zeros(capture.labels.size, dtype=np.float64)
            hidden_shape = np.zeros(capture.labels.size, dtype=np.float64)
        else:
            hidden_rms = np.sqrt(np.mean(np.square(donor.hidden), axis=1))
            hidden_shape = np.mean(np.abs(donor.hidden), axis=1) / np.maximum(hidden_rms, EPS)
        columns.extend(
            (donor_entropy, donor_margin, js, residual_rms, hidden_rms, hidden_shape)
        )
        donor_probabilities.append((donor.base_ids, donor_conditional))

    for left in range(len(donor_probabilities)):
        for right in range(left + 1, len(donor_probabilities)):
            left_ids, left_prob = donor_probabilities[left]
            right_ids, right_prob = donor_probabilities[right]
            common, left_pos, right_pos = np.intersect1d(
                left_ids, right_ids, assume_unique=True, return_indices=True
            )
            if common.size < 2:
                columns.append(np.ones(capture.labels.size, dtype=np.float64))
                continue
            left_common = left_prob[:, left_pos]
            right_common = right_prob[:, right_pos]
            left_common /= np.sum(left_common, axis=1, keepdims=True)
            right_common /= np.sum(right_common, axis=1, keepdims=True)
            agreement = np.sum(np.sqrt(left_common * right_common), axis=1)
            columns.append(agreement)
    return np.column_stack(columns)


def _normalization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    mean[0] = 0.0
    scale[0] = 1.0
    scale[scale < 1e-8] = 1.0
    return mean, scale


def calibrate_temperatures(
    capture: Capture,
    donor_indices: Sequence[int],
    grid: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0),
) -> tuple[float, ...]:
    result: list[float] = []
    for donor_index in donor_indices:
        candidates: list[tuple[float, float]] = []
        for temperature in grid:
            residual = donor_residual(capture, donor_index, float(temperature))
            loss = float(np.mean(token_nll(capture.base_logits + residual, capture.labels)))
            candidates.append((loss, float(temperature)))
        result.append(min(candidates)[1])
    return tuple(result)


def fit_gate(
    capture: Capture,
    donor_indices: Sequence[int],
    *,
    epochs: int = 500,
    learning_rate: float = 0.08,
    conservatism: float = 0.2,
    l2: float = 1e-4,
) -> GateModel:
    if not donor_indices:
        raise ValueError("至少需要一个 donor")
    donor_indices = tuple(int(index) for index in donor_indices)
    temperatures = calibrate_temperatures(capture, donor_indices)
    residuals = [
        donor_residual(capture, donor_index, temperature)
        for donor_index, temperature in zip(donor_indices, temperatures)
    ]
    raw_features = build_features(capture, donor_indices)
    feature_mean, feature_scale = _normalization(raw_features)
    features = (raw_features - feature_mean) / feature_scale
    weights = np.zeros((len(donor_indices) + 1, features.shape[1]), dtype=np.float64)
    rows = np.arange(capture.labels.size)

    for epoch in range(epochs):
        gate = sparsemax(features @ weights.T)
        fused = capture.base_logits.copy()
        for local_index, residual in enumerate(residuals, start=1):
            fused += gate[:, local_index, None] * residual
        probabilities = softmax(fused)
        probabilities[rows, capture.labels] -= 1.0
        probabilities /= capture.labels.size

        grad_gate = np.zeros_like(gate)
        for local_index, residual in enumerate(residuals, start=1):
            grad_gate[:, local_index] = np.sum(probabilities * residual, axis=1)
            grad_gate[:, local_index] += conservatism / capture.labels.size

        active = gate > 0.0
        grad_scores = np.zeros_like(gate)
        for row in range(gate.shape[0]):
            support = active[row]
            supported_gradient = grad_gate[row, support]
            grad_scores[row, support] = supported_gradient - np.mean(supported_gradient)
        grad_weights = grad_scores.T @ features + l2 * weights
        step = learning_rate / math.sqrt(1.0 + epoch / 100.0)
        weights -= step * grad_weights

    return GateModel(
        donor_indices=donor_indices,
        temperatures=temperatures,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=weights,
    )


def fit_counterfactual_single_gate(
    capture: Capture,
    donor_index: int,
    *,
    l2: float = 1e-3,
    minimum_gain: float = 1e-5,
    mass_grid: Sequence[float] = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
) -> GateModel:
    """用反事实 NLL 标签闭式拟合单 donor 门，避免在全词表上迭代数百轮。"""

    if donor_index < 0 or donor_index >= len(capture.donors):
        raise ValueError("donor_index 超出范围")
    masses = np.asarray(tuple(float(value) for value in mass_grid), dtype=np.float64)
    if masses.size < 2 or masses[0] != 0.0 or np.any(np.diff(masses) <= 0.0):
        raise ValueError("mass_grid 必须从0开始并严格递增")
    if masses[-1] > 1.0:
        raise ValueError("mass_grid 不能超过1")

    residual = donor_residual(capture, donor_index, 1.0)
    losses = np.column_stack(
        [
            token_nll(capture.base_logits + mass * residual, capture.labels)
            for mass in masses
        ]
    )
    best_index = np.argmin(losses, axis=1)
    best_mass = masses[best_index]
    gain = losses[:, 0] - losses[np.arange(capture.labels.size), best_index]
    best_mass[gain < minimum_gain] = 0.0

    raw_features = build_features(capture, (donor_index,))
    feature_mean, feature_scale = _normalization(raw_features)
    features = (raw_features - feature_mean) / feature_scale
    # 二分类 sparsemax 中 donor 质量为 clip((score_diff + 1) / 2, 0, 1)。
    target_difference = 2.0 * best_mass - 1.0
    target_difference[best_mass == 0.0] = -1.25
    target_difference[best_mass == 1.0] = 1.25
    gram = features.T @ features
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    difference_weights = np.linalg.solve(
        gram + penalty, features.T @ target_difference
    )
    weights = np.vstack((-0.5 * difference_weights, 0.5 * difference_weights))
    return GateModel(
        donor_indices=(int(donor_index),),
        temperatures=(1.0,),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=weights,
    )


def apply_gate(capture: Capture, model: GateModel) -> tuple[np.ndarray, np.ndarray]:
    raw_features = build_features(capture, model.donor_indices)
    features = (raw_features - model.feature_mean) / model.feature_scale
    gate = sparsemax(features @ model.weights.T)
    fused = capture.base_logits.copy()
    for local_index, (donor_index, temperature) in enumerate(
        zip(model.donor_indices, model.temperatures), start=1
    ):
        residual = donor_residual(capture, donor_index, temperature)
        fused += gate[:, local_index, None] * residual
    return fused, gate


def _bootstrap_task_ci(
    deltas: np.ndarray, task_ids: np.ndarray, *, seed: int = 19, draws: int = 2000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    tasks = np.unique(task_ids)
    task_means = np.asarray([np.mean(deltas[task_ids == task]) for task in tasks])
    samples = rng.choice(task_means, size=(draws, task_means.size), replace=True)
    means = np.mean(samples, axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def validate_complementarity(
    capture: Capture, *, epochs: int = 500, seed: int = 19
) -> dict[str, object]:
    """LOTO 比较 base+donor1 与 base+donor1+其余 donor。"""

    if len(capture.donors) < 2:
        raise ValueError("条件互补性验证至少需要两个 donor")
    tasks = np.unique(capture.task_ids)
    if tasks.size < 2:
        raise ValueError("LOTO 至少需要两个 task")

    all_delta: list[np.ndarray] = []
    all_tasks: list[np.ndarray] = []
    conflict_delta: list[np.ndarray] = []
    fold_reports: list[dict[str, object]] = []
    no_op_count = 0
    sample_count = 0
    conflict_count = 0
    conflict_first_correct = 0
    conflict_second_correct = 0
    conflict_fused_correct = 0
    resolvable_conflict_count = 0
    resolvable_fused_correct = 0

    for task in tasks:
        test_indices = np.flatnonzero(capture.task_ids == task)
        train_indices = np.flatnonzero(capture.task_ids != task)
        train = subset_capture(capture, train_indices)
        test = subset_capture(capture, test_indices)
        first_model = fit_gate(train, (0,), epochs=epochs)
        full_model = fit_gate(train, tuple(range(len(capture.donors))), epochs=epochs)
        first_logits, _ = apply_gate(test, first_model)
        full_logits, full_gate = apply_gate(test, full_model)
        first_nll = token_nll(first_logits, test.labels)
        full_nll = token_nll(full_logits, test.labels)
        delta = full_nll - first_nll

        first_residual = donor_residual(test, 0, full_model.temperatures[0])
        second_residual = donor_residual(test, 1, full_model.temperatures[1])
        first_choice = np.argmax(test.base_logits + first_residual, axis=1)
        second_choice = np.argmax(test.base_logits + second_residual, axis=1)
        fused_choice = np.argmax(full_logits, axis=1)
        conflict = first_choice != second_choice
        first_correct = first_choice == test.labels
        second_correct = second_choice == test.labels
        fused_correct = fused_choice == test.labels
        resolvable = conflict & np.logical_xor(first_correct, second_correct)

        all_delta.append(delta)
        all_tasks.append(test.task_ids)
        conflict_delta.append(delta[conflict])
        no_op_count += int(np.sum(full_gate[:, 0] >= 1.0 - 1e-12))
        sample_count += int(test.labels.size)
        conflict_count += int(np.sum(conflict))
        conflict_first_correct += int(np.sum(first_correct & conflict))
        conflict_second_correct += int(np.sum(second_correct & conflict))
        conflict_fused_correct += int(np.sum(fused_correct & conflict))
        resolvable_conflict_count += int(np.sum(resolvable))
        resolvable_fused_correct += int(np.sum(fused_correct & resolvable))
        fold_reports.append(
            {
                "held_out_task": str(task),
                "sample_count": int(test.labels.size),
                "mean_delta_vs_base_plus_donor1": float(np.mean(delta)),
                "wins": int(np.sum(delta < -1e-12)),
                "losses": int(np.sum(delta > 1e-12)),
                "conflicts": int(np.sum(conflict)),
                "conflict_mean_delta": (
                    float(np.mean(delta[conflict])) if np.any(conflict) else None
                ),
            }
        )

    deltas = np.concatenate(all_delta)
    task_ids = np.concatenate(all_tasks)
    conflicts = np.concatenate([value for value in conflict_delta if value.size])
    ci_low, ci_high = _bootstrap_task_ci(deltas, task_ids, seed=seed)
    task_mean = {
        str(task): float(np.mean(deltas[task_ids == task])) for task in np.unique(task_ids)
    }
    return {
        "format": "colorlm-multidonor-complementarity-v1",
        "comparison": "base+donor1+additional_donors minus base+donor1",
        "sample_count": int(deltas.size),
        "task_count": int(tasks.size),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "task_bootstrap_95_ci": [ci_low, ci_high],
        "task_mean_delta": task_mean,
        "task_wins": int(sum(value < 0.0 for value in task_mean.values())),
        "task_losses": int(sum(value > 0.0 for value in task_mean.values())),
        "no_op_rate": float(no_op_count / max(sample_count, 1)),
        "conflict_count": conflict_count,
        "conflict_mean_delta": float(np.mean(conflicts)) if conflicts.size else None,
        "conflict_accuracy": {
            "donor1": (
                float(conflict_first_correct / conflict_count) if conflict_count else None
            ),
            "donor2": (
                float(conflict_second_correct / conflict_count) if conflict_count else None
            ),
            "fused": (
                float(conflict_fused_correct / conflict_count) if conflict_count else None
            ),
            "resolvable_count": resolvable_conflict_count,
            "fused_resolvable_accuracy": (
                float(resolvable_fused_correct / resolvable_conflict_count)
                if resolvable_conflict_count
                else None
            ),
        },
        "conditionally_complementary": bool(
            np.mean(deltas) < 0.0
            and ci_high < 0.0
            and conflicts.size > 0
            and np.mean(conflicts) <= 0.0
        ),
        "folds": fold_reports,
        "feature_contract": "numeric logits/hidden only; task_ids used only for split/report",
    }


def _gate_model_report(model: GateModel) -> dict[str, object]:
    """把小门导出为可审计、可直接移植到运行时的 JSON 数据。"""

    return {
        "donor_indices": list(model.donor_indices),
        "temperatures": list(model.temperatures),
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "weights": model.weights.tolist(),
    }


def validate_single_donor(
    capture: Capture,
    *,
    donor_index: int = 0,
    epochs: int = 500,
    seed: int = 19,
) -> dict[str, object]:
    """用 leave-one-task-out 检查单 donor 的显式 no-op 门是否可泛化。"""

    if donor_index < 0 or donor_index >= len(capture.donors):
        raise ValueError("donor_index 超出范围")
    tasks = np.unique(capture.task_ids)
    if tasks.size < 2:
        raise ValueError("LOTO 至少需要两个 task")

    all_delta: list[np.ndarray] = []
    all_tasks: list[np.ndarray] = []
    fold_reports: list[dict[str, object]] = []
    no_op_count = 0
    sample_count = 0
    donor_mass = 0.0

    for task in tasks:
        test_indices = np.flatnonzero(capture.task_ids == task)
        train_indices = np.flatnonzero(capture.task_ids != task)
        train = subset_capture(capture, train_indices)
        test = subset_capture(capture, test_indices)
        model = fit_counterfactual_single_gate(train, donor_index)
        fused_logits, gate = apply_gate(test, model)
        base_nll = token_nll(test.base_logits, test.labels)
        fused_nll = token_nll(fused_logits, test.labels)
        delta = fused_nll - base_nll
        exact_no_op = gate[:, 0] >= 1.0 - 1e-12

        all_delta.append(delta)
        all_tasks.append(test.task_ids)
        no_op_count += int(np.sum(exact_no_op))
        sample_count += int(test.labels.size)
        donor_mass += float(np.sum(gate[:, 1]))
        fold_reports.append(
            {
                "held_out_task": str(task),
                "sample_count": int(test.labels.size),
                "mean_delta_vs_base": float(np.mean(delta)),
                "wins": int(np.sum(delta < -1e-12)),
                "losses": int(np.sum(delta > 1e-12)),
                "equal": int(np.sum(np.abs(delta) <= 1e-12)),
                "exact_no_op_rate": float(np.mean(exact_no_op)),
                "mean_donor_mass": float(np.mean(gate[:, 1])),
            }
        )

    deltas = np.concatenate(all_delta)
    task_ids = np.concatenate(all_tasks)
    ci_low, ci_high = _bootstrap_task_ci(deltas, task_ids, seed=seed)
    task_mean = {
        str(task): float(np.mean(deltas[task_ids == task])) for task in tasks
    }
    task_wins = int(sum(value < 0.0 for value in task_mean.values()))
    task_losses = int(sum(value > 0.0 for value in task_mean.values()))
    full_model = fit_counterfactual_single_gate(capture, donor_index)
    no_op_rate = float(no_op_count / max(sample_count, 1))
    mean_delta = float(np.mean(deltas))

    return {
        "format": "colorlm-single-donor-noop-loto-v1",
        "comparison": "base+sparse-noop-donor minus base",
        "donor_index": int(donor_index),
        "sample_count": int(deltas.size),
        "task_count": int(tasks.size),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(deltas)),
        "token_wins": int(np.sum(deltas < -1e-12)),
        "token_losses": int(np.sum(deltas > 1e-12)),
        "token_equal": int(np.sum(np.abs(deltas) <= 1e-12)),
        "task_bootstrap_95_ci": [ci_low, ci_high],
        "task_mean_delta": task_mean,
        "task_wins": task_wins,
        "task_losses": task_losses,
        "worst_task_delta": float(max(task_mean.values())),
        "exact_no_op_rate": no_op_rate,
        "mean_donor_mass": float(donor_mass / max(sample_count, 1)),
        "development_signal": bool(
            mean_delta < 0.0
            and task_wins > task_losses
            and no_op_rate > 0.0
        ),
        "strict_signal": bool(
            mean_delta < 0.0
            and ci_high < 0.0
            and task_wins > task_losses
            and max(task_mean.values()) <= 0.0
            and no_op_rate > 0.0
        ),
        "folds": fold_reports,
        "full_model": _gate_model_report(full_model),
        "feature_contract": "numeric logits/hidden only; task_ids used only for split/report",
    }


def make_synthetic_capture(seed: int = 7) -> Capture:
    """构造 donor2 对 donor1 条件互补、且含 base-only 区域的合成数据。"""

    rng = np.random.default_rng(seed)
    sample_count, vocab_size, hidden_size = 480, 17, 24
    labels = rng.integers(0, vocab_size, size=sample_count)
    task_ids = np.asarray([f"task-{index % 6}" for index in range(sample_count)])
    base = rng.normal(0.0, 0.35, size=(sample_count, vocab_size))
    base[np.arange(sample_count), labels] += 1.4
    state = rng.choice(3, size=sample_count, p=(0.38, 0.38, 0.24))
    wrong = (labels + rng.integers(1, vocab_size, size=sample_count)) % vocab_size

    donor_logits: list[np.ndarray] = []
    donor_hidden: list[np.ndarray] = []
    for donor_index in range(2):
        reliable_state = donor_index
        current = base + rng.normal(0.0, 0.08, size=base.shape)
        reliable = state == reliable_state
        unreliable = ~reliable
        current[reliable, labels[reliable]] += 2.8
        current[unreliable, wrong[unreliable]] += 2.3
        scale = np.where(reliable, 2.2, 0.35)
        hidden = rng.normal(size=(sample_count, hidden_size)) * scale[:, None]
        donor_logits.append(current)
        donor_hidden.append(hidden)

    donors = tuple(
        DonorCapture(
            logits=donor_logits[index],
            base_ids=np.arange(vocab_size, dtype=np.int64),
            hidden=donor_hidden[index],
        )
        for index in range(2)
    )
    capture = Capture(base, labels, task_ids, donors)
    capture.validate()
    return capture


def _json_dump(value: object, output: Path | None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
    else:
        output.write_text(rendered, encoding="utf-8")


def _parse_alphas(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("alpha 列表不能为空")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay", help="离线回放旧式全局 alpha")
    replay.add_argument("capture", type=Path)
    replay.add_argument("--donor", type=int, default=0)
    replay.add_argument(
        "--alphas", type=_parse_alphas, default=_parse_alphas("0,0.001,0.003,0.01,0.03")
    )
    replay.add_argument("--output", type=Path)

    validate = subparsers.add_parser("validate", help="LOTO 多 donor 条件互补性验证")
    validate.add_argument("capture", type=Path)
    validate.add_argument("--epochs", type=int, default=500)
    validate.add_argument("--output", type=Path)

    validate_single = subparsers.add_parser(
        "validate-single", help="LOTO 单 donor 显式 no-op 门验证"
    )
    validate_single.add_argument("capture", type=Path)
    validate_single.add_argument("--donor", type=int, default=0)
    validate_single.add_argument("--epochs", type=int, default=500)
    validate_single.add_argument("--output", type=Path)

    selftest = subparsers.add_parser("selftest", help="运行内置纯 CPU 合成验证")
    selftest.add_argument("--epochs", type=int, default=500)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    _reconfigure_stdio()
    args = build_parser().parse_args(argv)
    if args.command == "replay":
        capture = load_capture(args.capture)
        _json_dump(replay_alphas(capture, args.donor, args.alphas), args.output)
        return 0
    if args.command == "validate":
        capture = load_capture(args.capture)
        _json_dump(validate_complementarity(capture, epochs=args.epochs), args.output)
        return 0
    if args.command == "validate-single":
        capture = load_capture(args.capture)
        _json_dump(
            validate_single_donor(
                capture, donor_index=args.donor, epochs=args.epochs
            ),
            args.output,
        )
        return 0
    if args.command == "selftest":
        capture = make_synthetic_capture()
        report = validate_complementarity(capture, epochs=args.epochs)
        replay = replay_alphas(capture, 0, (0.0, 0.01, 0.03))
        _json_dump({"replay": replay, "complementarity": report}, None)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
