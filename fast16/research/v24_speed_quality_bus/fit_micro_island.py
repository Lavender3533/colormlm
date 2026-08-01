"""从v17教师dump拟合低秩、无状态微岛候选并报告可压缩性。"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


HEADER = struct.Struct("<IIiIQQQ")
MAGIC = 0x544C4349


def read_pairs(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    data = path.read_bytes()
    offset = 0
    pending: list[tuple[int, np.ndarray]] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    while offset < len(data):
        if len(data) - offset < HEADER.size:
            raise RuntimeError("教师dump头部不完整")
        magic, version, layer, kind, ne0, ne1, payload_bytes = HEADER.unpack_from(data, offset)
        offset += HEADER.size
        if magic != MAGIC or version != 1 or kind not in (0, 1) or ne0 <= 0 or ne1 <= 0:
            raise RuntimeError("教师dump头部无效")
        expected = ne0 * ne1 * 4
        if payload_bytes != expected or len(data) - offset < payload_bytes:
            raise RuntimeError("教师dump载荷长度不匹配")
        array = np.frombuffer(data, dtype="<f4", count=ne0 * ne1, offset=offset)
        array = np.array(array.reshape((ne1, ne0)), dtype=np.float32, copy=True)
        offset += payload_bytes
        if kind == 0:
            pending.append((layer, array))
        else:
            if not pending:
                raise RuntimeError("delta记录没有对应input")
            input_layer, input_array = pending.pop(0)
            if input_layer != layer or input_array.shape != array.shape:
                raise RuntimeError("input/delta层或shape不匹配")
            pairs.append((input_array, array))
    if pending:
        raise RuntimeError("存在没有delta的input记录")
    if not pairs:
        raise RuntimeError("教师dump没有成对记录")
    return pairs


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - reference
    ref_norm = np.linalg.norm(reference, axis=1)
    pred_norm = np.linalg.norm(prediction, axis=1)
    denom = np.maximum(ref_norm * np.maximum(pred_norm, 1e-12), 1e-12)
    cosine = np.sum(reference * prediction, axis=1) / denom
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "reference_rms": float(np.sqrt(np.mean(reference * reference))),
        "relative_rmse": float(np.sqrt(np.mean(error * error)) / max(np.sqrt(np.mean(reference * reference)), 1e-12)),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_median": float(np.median(cosine)),
        "cosine_positive_fraction": float(np.mean(cosine > 0.0)),
    }


def fit_rank(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray, rank: int) -> dict[str, Any]:
    x_mean = x_train.mean(axis=0)
    y_mean = y_train.mean(axis=0)
    x_centered = x_train - x_mean
    _, singular, vt = np.linalg.svd(x_centered, full_matrices=False)
    actual_rank = min(rank, vt.shape[0])
    basis = vt[:actual_rank]
    z_train = x_centered @ basis.T
    output_projection, *_ = np.linalg.lstsq(z_train, y_train - y_mean, rcond=1e-5)
    prediction = (x_eval - x_mean) @ basis.T @ output_projection + y_mean
    zero = np.zeros_like(y_eval)
    return {
        "rank": actual_rank,
        "weights_bytes_f16": int((basis.size + output_projection.size) * 2),
        "input_projection_shape": list(basis.shape),
        "output_projection_shape": list(output_projection.shape),
        "singular_values_kept_fraction": float(np.sum(singular[:actual_rank] ** 2) / max(np.sum(singular ** 2), 1e-12)),
        "zero_baseline": metrics(y_eval, zero),
        "micro_island": metrics(y_eval, prediction),
        "prediction": prediction,
        "basis": basis.astype(np.float16),
        "output_projection": output_projection.astype(np.float16),
        "x_mean": x_mean.astype(np.float16),
        "y_mean": y_mean.astype(np.float16),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="拟合v17低秩微岛候选")
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    pairs = read_pairs(args.dump)
    # 旧版只按图记录顺序切分；它不是严格的按请求留出，仅用于无状态可压缩性探针。
    split = max(1, int(len(pairs) * 0.8))
    train_pairs = pairs[:split]
    eval_pairs = pairs[split:]
    if not eval_pairs:
        raise RuntimeError("教师记录不足以形成留出图")
    x_train = np.concatenate([row[0] for row in train_pairs], axis=0)
    y_train = np.concatenate([row[1] for row in train_pairs], axis=0)
    x_eval = np.concatenate([row[0] for row in eval_pairs], axis=0)
    y_eval = np.concatenate([row[1] for row in eval_pairs], axis=0)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "format": "colorlm-v25-micro-island-fit-v1",
        "dump": str(args.dump.resolve()),
        "pair_records": len(pairs),
        "train_records": len(train_pairs),
        "eval_records": len(eval_pairs),
        "train_tokens": int(x_train.shape[0]),
        "eval_tokens": int(x_eval.shape[0]),
        "input_width": int(x_train.shape[1]),
        "target_width": int(y_train.shape[1]),
        "ranks": {},
        "decision_rule": "candidate must beat zero baseline on held-out cosine and relative RMSE before runtime integration",
    }
    for rank in (32, 64, 128, 256):
        fit = fit_rank(x_train, y_train, x_eval, y_eval, rank)
        artifact = args.artifact_dir / f"micro_island_r{fit['rank']}.npz"
        np.savez_compressed(
            artifact,
            basis=fit["basis"],
            output_projection=fit["output_projection"],
            x_mean=fit["x_mean"],
            y_mean=fit["y_mean"],
        )
        row = {key: value for key, value in fit.items() if key not in {"prediction", "basis", "output_projection", "x_mean", "y_mean"}}
        row["artifact"] = str(artifact.resolve())
        summary["ranks"][str(fit["rank"])] = row
        print(
            f"rank={fit['rank']} bytes={fit['weights_bytes_f16']} "
            f"cos={fit['micro_island']['cosine_median']:.4f} "
            f"rel_rmse={fit['micro_island']['relative_rmse']:.4f} "
            f"zero_cos={fit['zero_baseline']['cosine_median']:.4f}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
