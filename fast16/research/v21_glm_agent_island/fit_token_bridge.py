"""用两个 GGUF 的共享 token 嵌入拟合 GLM→ColorLM 正交先验桥。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import svd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def embedding_rows(reader: GGUFReader, ids: np.ndarray) -> np.ndarray:
    tensor = next(item for item in reader.tensors if item.name == "token_embd.weight")
    rows = dequantize(tensor.data[ids], tensor.tensor_type)
    return np.asarray(rows, dtype=np.float32).reshape(len(ids), -1)


def main() -> int:
    parser = argparse.ArgumentParser(description="拟合 GLM 到 ColorLM 的共享 token 正交桥")
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "fast16/models/ColorLM-v6-Q3Router-Fused-A1.gguf",
    )
    parser.add_argument(
        "--donor",
        type=Path,
        default=ROOT / "fast16/models/GLM-4.7-Flash-Q5_K_M.gguf",
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--train", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("glm_to_colorlm_token_orthogonal_f32.npy"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).with_name("glm_token_bridge_report.json"),
    )
    args = parser.parse_args()
    if not 0 < args.train < args.samples:
        raise ValueError("必须满足 0 < train < samples")

    base = GGUFReader(os.fspath(args.base), "r")
    donor = GGUFReader(os.fspath(args.donor), "r")
    base_tokens = base.fields["tokenizer.ggml.tokens"].contents()
    donor_tokens = donor.fields["tokenizer.ggml.tokens"].contents()

    base_ids: dict[bytes | str, int] = {}
    duplicates: set[bytes | str] = set()
    for token_id, token in enumerate(base_tokens):
        if token in base_ids:
            duplicates.add(token)
        else:
            base_ids[token] = token_id
    pairs = [
        (donor_id, base_ids[token])
        for donor_id, token in enumerate(donor_tokens)
        if token in base_ids and token not in duplicates
    ]
    if len(pairs) < args.samples:
        raise RuntimeError(f"共享 token 不足: {len(pairs)} < {args.samples}")

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(pairs), size=args.samples, replace=False)
    donor_ids = np.asarray([pairs[index][0] for index in chosen], dtype=np.int64)
    base_row_ids = np.asarray([pairs[index][1] for index in chosen], dtype=np.int64)
    donor_rows = normalize(embedding_rows(donor, donor_ids))
    base_rows = normalize(embedding_rows(base, base_row_ids))

    order = rng.permutation(args.samples)
    train = order[: args.train]
    test = order[args.train :]
    cross = donor_rows[train].T @ base_rows[train]
    left, singular_values, right_t = svd(
        cross,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    transport = np.ascontiguousarray(left @ right_t, dtype=np.float32)
    before = np.sum(donor_rows[test] * base_rows[test], axis=1)
    after = np.sum(normalize(donor_rows[test] @ transport) * base_rows[test], axis=1)
    identity = np.eye(transport.shape[0], dtype=np.float32)
    orthogonality_rmse = float(
        np.linalg.norm(transport.T @ transport - identity) / np.sqrt(identity.size)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, transport, allow_pickle=False)
    report = {
        "format": "colorlm-gguf-token-transport-v1",
        "source": str(args.donor.resolve()),
        "target": str(args.base.resolve()),
        "contract": "donor_row @ T = colorlm_row",
        "shared_unique_tokens": len(pairs),
        "sampled_tokens": args.samples,
        "train_tokens": len(train),
        "heldout_tokens": len(test),
        "heldout_cosine_before": summary(before),
        "heldout_cosine_after": summary(after),
        "orthogonality_rmse": orthogonality_rmse,
        "singular_values": {
            "min": float(np.min(singular_values)),
            "median": float(np.median(singular_values)),
            "max": float(np.max(singular_values)),
        },
        "matrix": str(args.out.resolve()),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
