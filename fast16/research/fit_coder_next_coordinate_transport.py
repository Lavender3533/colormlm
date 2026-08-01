"""Fit and validate an activation-free hidden-space transport for Coder-Next."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import svd


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from analyze_coder_next_alignment import bf16, normalize_rows, summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    extracted = (
        RESEARCH
        / "biopsy_cache"
        / "Qwen_Qwen3-Coder-Next"
        / "master"
        / "extracted"
    )
    metadata = extracted.parent / "metadata"
    parser = argparse.ArgumentParser(description="拟合Coder-Next到ColorLM的正交坐标运输")
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "fast16" / "models" / "ColorLM-v6-Q3Router-Fused-A1.gguf",
    )
    parser.add_argument(
        "--donor-embedding",
        type=Path,
        default=extracted / "model.embed_tokens.weight.axis0-0-4096.bin",
    )
    parser.add_argument(
        "--donor-router",
        type=Path,
        default=extracted / "model.layers.47.mlp.gate.weight.bin",
    )
    parser.add_argument(
        "--donor-tokenizer", type=Path, default=metadata / "tokenizer.json"
    )
    parser.add_argument("--donor-rows", type=int, default=4096)
    parser.add_argument("--train-count", type=int, default=3072)
    parser.add_argument("--base-layer", type=int, default=39)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--output-matrix",
        type=Path,
        default=RESEARCH / "coder_next_to_colorlm_orthogonal_f32.npy",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=RESEARCH / "coder_next_coordinate_transport_report.json",
    )
    return parser.parse_args()


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(normalize_rows(left) * normalize_rows(right), axis=1)


def main() -> int:
    args = parse_args()
    base = GGUFReader(os.fspath(args.base), "r")
    tensors = {tensor.name: tensor for tensor in base.tensors}

    base_tokens = base.fields["tokenizer.ggml.tokens"].contents()
    base_token_ids: dict[str, int] = {}
    for token_id, token in enumerate(base_tokens):
        base_token_ids.setdefault(token, token_id)

    tokenizer = json.loads(args.donor_tokenizer.read_text(encoding="utf-8"))
    donor_tokens = {
        int(token_id): token
        for token, token_id in tokenizer["model"]["vocab"].items()
    }
    donor_tokens.update(
        {
            int(item["id"]): item["content"]
            for item in tokenizer.get("added_tokens", [])
        }
    )
    pairs = [
        (donor_id, base_token_ids[token])
        for donor_id, token in sorted(donor_tokens.items())
        if donor_id < args.donor_rows and token in base_token_ids
    ]
    if not 0 < args.train_count < len(pairs):
        raise ValueError(f"train-count越界: {args.train_count} vs {len(pairs)}")

    donor_ids = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    base_ids = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    embedding_tensor = tensors["token_embd.weight"]
    base_embedding = np.asarray(
        dequantize(embedding_tensor.data[base_ids], embedding_tensor.tensor_type),
        dtype=np.float32,
    )
    donor_embedding = bf16(args.donor_embedding, (args.donor_rows, 2048))[donor_ids]
    base_embedding = normalize_rows(base_embedding)
    donor_embedding = normalize_rows(donor_embedding)

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(pairs))
    train = permutation[: args.train_count]
    test = permutation[args.train_count :]
    cross = donor_embedding[train].T @ base_embedding[train]
    left, singular_values, right_t = svd(
        cross,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    transport = np.ascontiguousarray(left @ right_t, dtype=np.float32)

    train_before = row_cosine(donor_embedding[train], base_embedding[train])
    train_after = row_cosine(
        donor_embedding[train] @ transport, base_embedding[train]
    )
    test_before = row_cosine(donor_embedding[test], base_embedding[test])
    mapped_test = normalize_rows(donor_embedding[test] @ transport)
    target_test = normalize_rows(base_embedding[test])
    test_after = np.sum(mapped_test * target_test, axis=1)
    retrieval = mapped_test @ target_test.T
    top1 = float(np.mean(np.argmax(retrieval, axis=1) == np.arange(len(test))))

    base_router = np.asarray(
        tensors[f"blk.{args.base_layer}.ffn_gate_inp.weight"].data,
        dtype=np.float32,
    )
    donor_router = bf16(args.donor_router, (512, 2048))
    base_router_unit = normalize_rows(base_router)
    raw_router = normalize_rows(donor_router) @ base_router_unit.T
    mapped_router_weights = donor_router @ transport
    mapped_router = normalize_rows(mapped_router_weights) @ base_router_unit.T
    raw_best = np.max(raw_router, axis=1)
    mapped_target = np.argmax(mapped_router, axis=1)
    mapped_best = mapped_router[np.arange(mapped_router.shape[0]), mapped_target]
    order = np.argsort(mapped_best)[::-1]
    unique_pairs = []
    used_targets: set[int] = set()
    for donor_index in order:
        target_index = int(mapped_target[donor_index])
        if target_index in used_targets:
            continue
        used_targets.add(target_index)
        unique_pairs.append(
            {
                "donor_expert": int(donor_index),
                "target_expert": target_index,
                "router_cosine_before": float(raw_router[donor_index, target_index]),
                "router_cosine_after": float(mapped_best[donor_index]),
            }
        )
        if len(unique_pairs) == 16:
            break

    identity = np.eye(transport.shape[0], dtype=np.float32)
    orthogonality_rmse = float(
        np.linalg.norm(transport.T @ transport - identity) / np.sqrt(transport.size)
    )
    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_matrix, transport, allow_pickle=False)
    report = {
        "format": "colorlm-coordinate-transport-v1",
        "source": "Qwen/Qwen3-Coder-Next",
        "target": args.base.name,
        "method": "shared-token orthogonal Procrustes",
        "matched_tokens": len(pairs),
        "train_tokens": len(train),
        "test_tokens": len(test),
        "embedding_cosine": {
            "train_before": summary(train_before),
            "train_after": summary(train_after),
            "test_before": summary(test_before),
            "test_after": summary(test_after),
        },
        "test_token_retrieval_top1": top1,
        "router_best_cosine": {
            "before": summary(raw_best),
            "after_transport": summary(mapped_best),
        },
        "top_unique_router_pairs": unique_pairs,
        "transport": {
            "shape": list(transport.shape),
            "dtype": str(transport.dtype),
            "orthogonality_rmse": orthogonality_rmse,
            "singular_value_min": float(np.min(singular_values)),
            "singular_value_median": float(np.median(singular_values)),
            "singular_value_max": float(np.max(singular_values)),
            "file": args.output_matrix.name,
            "bytes": args.output_matrix.stat().st_size,
        },
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
