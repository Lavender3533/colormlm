"""按完整请求留出，拟合带衰减状态的低秩 Neural Island 蒸馏候选。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
V24_DIR = HERE.parent / "v24_speed_quality_bus"
if str(V24_DIR) not in sys.path:
    sys.path.insert(0, str(V24_DIR))

from fit_micro_island import metrics, read_pairs  # noqa: E402


def group_sequences(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """用每次prefill的多token记录作为请求边界，decode记录归入其后。"""
    groups: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for pair in pairs:
        token_count = int(pair[0].shape[0])
        if token_count > 1:
            groups.append([])
        if not groups:
            raise RuntimeError(
                "dump以单token记录开头，无法无歧义恢复请求边界；请用--no-warmup重新采集"
            )
        groups[-1].append(pair)
    return [
        (
            np.concatenate([pair[0] for pair in group], axis=0),
            np.concatenate([pair[1] for pair in group], axis=0),
        )
        for group in groups
    ]


def flatten(sequences: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([row[0] for row in sequences], axis=0),
        np.concatenate([row[1] for row in sequences], axis=0),
    )


def build_features(
    sequences: list[tuple[np.ndarray, np.ndarray]],
    x_mean: np.ndarray,
    basis: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    spans: list[slice] = []
    offset = 0
    for x_value, y_value in sequences:
        projected = (x_value - x_mean) @ basis.T
        state = np.zeros(projected.shape[1], dtype=np.float32)
        state_rows = np.empty_like(projected)
        for index, row in enumerate(projected):
            state = rho * state + row
            state_rows[index] = state
        feature = np.concatenate([projected, state_rows], axis=1)
        features.append(feature)
        targets.append(y_value)
        spans.append(slice(offset, offset + len(x_value)))
        offset += len(x_value)
    return np.concatenate(features), np.concatenate(targets), spans


def prepare_basis(
    train: list[tuple[np.ndarray, np.ndarray]], max_rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train, _ = flatten(train)
    x_mean = x_train.mean(axis=0)
    _, singular, vt = np.linalg.svd(x_train - x_mean, full_matrices=False)
    return x_mean, np.asarray(vt[: min(max_rank, vt.shape[0])], dtype=np.float32), singular


def fit_candidate(
    train: list[tuple[np.ndarray, np.ndarray]],
    evaluate: list[tuple[np.ndarray, np.ndarray]],
    rank: int,
    rho: float,
    ridge: float,
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    x_train, y_train = flatten(train)
    y_mean = y_train.mean(axis=0)
    if prepared is None:
        prepared = prepare_basis(train, rank)
    x_mean, basis_max, singular = prepared
    actual_rank = min(rank, basis_max.shape[0])
    basis = basis_max[:actual_rank]

    train_features, train_targets, _ = build_features(train, x_mean, basis, rho)
    gram = train_features.T @ train_features
    scale = float(np.trace(gram) / max(gram.shape[0], 1))
    regularizer = max(scale * ridge, 1e-8)
    gram.flat[:: gram.shape[0] + 1] += regularizer
    rhs = train_features.T @ (train_targets - y_mean)
    output_projection = np.linalg.solve(gram, rhs)

    eval_features, eval_targets, spans = build_features(evaluate, x_mean, basis, rho)
    prediction = eval_features @ output_projection + y_mean
    aggregate = metrics(eval_targets, prediction)
    per_sequence = [
        metrics(eval_targets[span], prediction[span]) for span in spans
    ]
    return {
        "requested_rank": rank,
        "rank": actual_rank,
        "rho": rho,
        "ridge": ridge,
        "ridge_absolute": regularizer,
        "weights_bytes_f16": int(
            (basis.size + output_projection.size + x_mean.size + y_mean.size) * 2
        ),
        "input_projection_shape": list(basis.shape),
        "output_projection_shape": list(output_projection.shape),
        "singular_values_kept_fraction": float(
            np.sum(singular[:actual_rank] ** 2) / max(np.sum(singular**2), 1e-12)
        ),
        "metrics": aggregate,
        "per_sequence": per_sequence,
        "basis": basis.astype(np.float16),
        "output_projection": output_projection.astype(np.float16),
        "x_mean": x_mean.astype(np.float16),
        "y_mean": y_mean.astype(np.float16),
    }


def public_row(candidate: dict[str, Any]) -> dict[str, Any]:
    hidden = {"basis", "output_projection", "x_mean", "y_mean"}
    return {key: value for key, value in candidate.items() if key not in hidden}


def split_sequences(
    sequences: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray]],
]:
    if len(sequences) < 8:
        raise RuntimeError(
            f"仅恢复出{len(sequences)}条请求；有状态微岛要求至少8条独立请求"
        )
    test_count = max(2, len(sequences) // 4)
    validation_count = max(2, len(sequences) // 4)
    train_count = len(sequences) - validation_count - test_count
    if train_count < 4:
        raise RuntimeError("训练请求不足4条")
    return (
        sequences[:train_count],
        sequences[train_count : train_count + validation_count],
        sequences[train_count + validation_count :],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="拟合v25有状态低秩微岛")
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-sequences", type=int)
    parser.add_argument(
        "--drop-leading-sequences",
        type=int,
        default=0,
        help="显式丢弃服务内部产生的前导微型序列；不会自动猜测",
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    if args.ridge <= 0:
        raise RuntimeError("--ridge必须大于0")

    all_sequences = group_sequences(read_pairs(args.dump))
    if not 0 <= args.drop_leading_sequences < len(all_sequences):
        raise RuntimeError("--drop-leading-sequences超出可用序列范围")
    dropped = all_sequences[: args.drop_leading_sequences]
    sequences = all_sequences[args.drop_leading_sequences :]
    if args.expected_sequences is not None and len(sequences) != args.expected_sequences:
        raise RuntimeError(
            f"请求数不匹配: actual={len(sequences)}, expected={args.expected_sequences}"
        )
    train, validation, test = split_sequences(sequences)

    validation_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    validation_basis = prepare_basis(train, 128)
    for rank in (32, 64, 128):
        for rho in (0.0, 0.5, 0.8, 0.95):
            candidate = fit_candidate(
                train, validation, rank, rho, args.ridge, validation_basis
            )
            candidates.append(candidate)
            validation_rows.append(public_row(candidate))
            print(
                f"validation rank={candidate['rank']} rho={rho:.2f} "
                f"cos={candidate['metrics']['cosine_median']:.4f} "
                f"rel_rmse={candidate['metrics']['relative_rmse']:.4f}",
                flush=True,
            )

    # rho=0是匹配参数量的无状态负对照，不允许被选作有状态候选。
    stateful = [candidate for candidate in candidates if candidate["rho"] > 0]
    selected_validation = min(
        stateful,
        key=lambda row: (
            row["metrics"]["relative_rmse"],
            -row["metrics"]["cosine_median"],
            row["weights_bytes_f16"],
        ),
    )
    selected_rank = int(selected_validation["requested_rank"])
    selected_rho = float(selected_validation["rho"])

    train_validation = train + validation
    test_basis = prepare_basis(train_validation, selected_rank)
    selected_test = fit_candidate(
        train_validation,
        test,
        selected_rank,
        selected_rho,
        args.ridge,
        test_basis,
    )
    stateless_test = fit_candidate(
        train_validation, test, selected_rank, 0.0, args.ridge, test_basis
    )
    test_metrics = selected_test["metrics"]
    stateless_metrics = stateless_test["metrics"]
    per_sequence_passes = sum(
        row["relative_rmse"] < 1.0 and row["cosine_median"] > 0.0
        for row in selected_test["per_sequence"]
    )
    stateful_gain = (
        stateless_metrics["relative_rmse"] - test_metrics["relative_rmse"]
    )
    passed = bool(
        test_metrics["relative_rmse"] <= 0.90
        and test_metrics["cosine_median"] >= 0.40
        and stateful_gain >= 0.02
        and per_sequence_passes == len(test)
    )

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.artifact,
        basis=selected_test["basis"],
        output_projection=selected_test["output_projection"],
        x_mean=selected_test["x_mean"],
        y_mean=selected_test["y_mean"],
        rho=np.asarray([selected_rho], dtype=np.float32),
    )
    report = {
        "format": "colorlm-v25-stateful-micro-island-fit-v1",
        "dump": str(args.dump.resolve()),
        "sequence_boundary": "每个ne1>1的prefill开始新请求；状态在请求边界清零",
        "dropped_leading_sequence_tokens": [len(row[0]) for row in dropped],
        "architecture": "q_t=U(h_t-mean); z_t=rho*z_(t-1)+q_t; delta_hat=[q_t,z_t]V+y_mean",
        "split": {
            "train_sequences": len(train),
            "validation_sequences": len(validation),
            "test_sequences": len(test),
            "train_tokens": sum(len(row[0]) for row in train),
            "validation_tokens": sum(len(row[0]) for row in validation),
            "test_tokens": sum(len(row[0]) for row in test),
        },
        "validation_candidates": validation_rows,
        "selected_on_validation": {
            "requested_rank": selected_rank,
            "rho": selected_rho,
        },
        "test": {
            "stateful": public_row(selected_test),
            "matched_stateless": public_row(stateless_test),
            "stateful_relative_rmse_gain": stateful_gain,
            "per_sequence_passes": per_sequence_passes,
        },
        "runtime_integration_gate": {
            "passed": passed,
            "requirements": {
                "test_relative_rmse_max": 0.90,
                "test_cosine_median_min": 0.40,
                "relative_rmse_gain_over_matched_stateless_min": 0.02,
                "all_test_sequences_beat_zero": True,
            },
            "next_action": (
                "允许实现默认关闭的C++运行图候选"
                if passed
                else "停止运行图接入；扩大独立教师轨迹或改变状态结构"
            ),
        },
        "artifact": str(args.artifact.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["runtime_integration_gate"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
