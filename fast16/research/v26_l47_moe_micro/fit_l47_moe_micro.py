"""严格按请求留出，拟合L47 MoE+shared专家支路的低秩候选。"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
V25_DIR = HERE.parent / "v25_stateful_micro_island"
if str(V25_DIR) not in sys.path:
    sys.path.insert(0, str(V25_DIR))

from fit_stateful_micro_island import (  # noqa: E402
    fit_candidate,
    group_sequences,
    prepare_basis,
    public_row,
    split_sequences,
)


HEADER = struct.Struct("<IIiIQQQ")
MAGIC = 0x4D4C4349


def read_pairs(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    data = path.read_bytes()
    offset = 0
    pending: list[np.ndarray] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    while offset < len(data):
        if len(data) - offset < HEADER.size:
            raise RuntimeError("L47 MoE dump头部不完整")
        magic, version, layer, kind, ne0, ne1, payload_bytes = HEADER.unpack_from(
            data, offset
        )
        offset += HEADER.size
        if magic != MAGIC or version != 1 or layer != 47 or kind not in (0, 1):
            raise RuntimeError("L47 MoE dump头部无效")
        expected = ne0 * ne1 * 4
        if payload_bytes != expected or len(data) - offset < payload_bytes:
            raise RuntimeError("L47 MoE dump载荷长度不匹配")
        array = np.frombuffer(
            data, dtype="<f4", count=ne0 * ne1, offset=offset
        ).reshape((ne1, ne0))
        array = np.array(array, dtype=np.float32, copy=True)
        offset += payload_bytes
        if kind == 0:
            pending.append(array)
        else:
            if not pending:
                raise RuntimeError("MoE残差没有对应FFN输入")
            input_array = pending.pop(0)
            if input_array.shape != array.shape:
                raise RuntimeError("FFN输入/MoE残差shape不匹配")
            pairs.append((input_array, array))
    if pending or not pairs:
        raise RuntimeError("L47 MoE dump没有形成完整成对记录")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="拟合L47微MoE")
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-sequences", type=int, default=8)
    parser.add_argument("--drop-leading-sequences", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    all_sequences = group_sequences(read_pairs(args.dump))
    dropped = all_sequences[: args.drop_leading_sequences]
    sequences = all_sequences[args.drop_leading_sequences :]
    if len(sequences) != args.expected_sequences:
        raise RuntimeError(
            f"请求数不匹配: actual={len(sequences)}, expected={args.expected_sequences}"
        )
    train, validation, test = split_sequences(sequences)
    prepared_validation = prepare_basis(train, 256)
    candidates = [
        fit_candidate(
            train, validation, rank, 0.0, args.ridge, prepared_validation
        )
        for rank in (32, 64, 128, 256)
    ]
    selected = min(
        candidates,
        key=lambda row: (
            row["metrics"]["relative_rmse"],
            -row["metrics"]["cosine_median"],
            row["weights_bytes_f16"],
        ),
    )
    train_validation = train + validation
    prepared_test = prepare_basis(train_validation, selected["requested_rank"])
    test_candidate = fit_candidate(
        train_validation,
        test,
        selected["requested_rank"],
        0.0,
        args.ridge,
        prepared_test,
    )
    metrics = test_candidate["metrics"]
    passed = bool(
        metrics["relative_rmse"] <= 0.80
        and metrics["cosine_median"] >= 0.50
        and all(
            row["relative_rmse"] < 1.0 and row["cosine_median"] > 0.0
            for row in test_candidate["per_sequence"]
        )
    )
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.artifact,
        basis=test_candidate["basis"],
        output_projection=test_candidate["output_projection"],
        x_mean=test_candidate["x_mean"],
        y_mean=test_candidate["y_mean"],
        rho=np.asarray([0.0], dtype=np.float32),
    )
    report = {
        "format": "colorlm-v26-l47-moe-micro-fit-v1",
        "dump": str(args.dump.resolve()),
        "target": "L47 post-attention norm -> exact MoE + shared expert residual",
        "dropped_leading_sequence_tokens": [len(row[0]) for row in dropped],
        "split_tokens": {
            "train": sum(len(row[0]) for row in train),
            "validation": sum(len(row[0]) for row in validation),
            "test": sum(len(row[0]) for row in test),
        },
        "validation_candidates": [public_row(row) for row in candidates],
        "selected": {
            "requested_rank": selected["requested_rank"],
            "rho": 0.0,
        },
        "test": public_row(test_candidate),
        "runtime_gate": {
            "passed": passed,
            "requirements": {
                "relative_rmse_max": 0.80,
                "cosine_median_min": 0.50,
                "all_test_sequences_beat_zero": True,
            },
            "next_action": (
                "允许接入保留Attention/KV的decode微MoE候选"
                if passed
                else "停止线性微MoE；只允许再试一次门控非线性支路"
            ),
        },
        "artifact": str(args.artifact.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"rank={test_candidate['rank']} cos={metrics['cosine_median']:.4f} "
        f"rel_rmse={metrics['relative_rmse']:.4f} pass={passed}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
