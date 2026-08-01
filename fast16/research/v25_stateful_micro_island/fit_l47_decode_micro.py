"""只用单token decode记录拟合L47微层；prefill将继续走完整供体层。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fit_stage_micro_islands import read_stage_pairs
from fit_stateful_micro_island import (
    fit_candidate,
    prepare_basis,
    public_row,
    split_sequences,
)


def decode_sequences(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for pair in pairs:
        if pair[0].shape[0] > 1:
            groups.append([])
            continue
        if not groups:
            raise RuntimeError("decode记录出现在首个prefill之前")
        groups[-1].append(pair)
    return [
        (
            np.concatenate([row[0] for row in group]),
            np.concatenate([row[1] for row in group]),
        )
        for group in groups
        if group
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="拟合decode-only L47微层")
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-sequences", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    by_layer = read_stage_pairs(args.dump)
    sequences = decode_sequences(by_layer[47])
    if len(sequences) != args.expected_sequences:
        raise RuntimeError(
            f"decode请求数不匹配: actual={len(sequences)}, expected={args.expected_sequences}"
        )
    train, validation, test = split_sequences(sequences)
    prepared_validation = prepare_basis(train, 32)
    candidates = [
        fit_candidate(
            train, validation, rank, 0.0, args.ridge, prepared_validation
        )
        for rank in (8, 16, 32)
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
        metrics["relative_rmse"] <= 0.85
        and metrics["cosine_median"] >= 0.40
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
        "format": "colorlm-v25-l47-decode-micro-fit-v1",
        "dump": str(args.dump.resolve()),
        "contract": "prefill始终走完整L47；微层仅替换n_tokens==1的decode",
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
        "gate_passed": passed,
        "artifact": str(args.artifact.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"rank={test_candidate['rank']} cos={metrics['cosine_median']:.4f} "
        f"rel_rmse={metrics['relative_rmse']:.4f} pass={passed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
