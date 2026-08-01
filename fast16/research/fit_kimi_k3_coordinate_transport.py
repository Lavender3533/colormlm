"""Fit a rectangular Kimi K3 (7168) to ColorLM (2048) residual transport."""

from __future__ import annotations

import argparse
import base64
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
    cache = RESEARCH / "biopsy_cache" / "moonshotai_Kimi-K3" / "master"
    extracted = cache / "extracted"
    parser = argparse.ArgumentParser(description="拟合Kimi K3到ColorLM的跨宽度坐标运输")
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "fast16" / "models" / "ColorLM-v6-Q3Router-Fused-A1.gguf",
    )
    parser.add_argument(
        "--donor-embedding",
        type=Path,
        default=extracted / "language_model.model.embed_tokens.weight.axis0-0-4096.bin",
    )
    parser.add_argument(
        "--donor-router",
        type=Path,
        default=extracted / "language_model.model.layers.92.block_sparse_moe.gate.weight.bin",
    )
    parser.add_argument(
        "--donor-tokenizer", type=Path, default=cache / "metadata" / "tiktoken.model"
    )
    parser.add_argument("--donor-rows", type=int, default=4096)
    parser.add_argument("--donor-width", type=int, default=7168)
    parser.add_argument("--train-count", type=int, default=3072)
    parser.add_argument("--base-layer", type=int, default=39)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--output-matrix",
        type=Path,
        default=RESEARCH / "kimi_k3_to_colorlm_semiorthogonal_f32.npy",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=RESEARCH / "kimi_k3_coordinate_transport_report.json",
    )
    return parser.parse_args()


def gpt2_byte_decoder() -> dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(byte_values)
    offset = 0
    for value in range(256):
        if value in byte_values:
            continue
        byte_values.append(value)
        codepoints.append(256 + offset)
        offset += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}


def base_token_bytes(token: str, decoder: dict[str, int]) -> bytes:
    try:
        return bytes(decoder[character] for character in token)
    except KeyError:
        return token.encode("utf-8")


def load_kimi_tokens(path: Path, row_limit: int) -> dict[int, bytes]:
    tokens: dict[int, bytes] = {}
    with path.open("rt", encoding="utf-8") as stream:
        for line in stream:
            encoded, raw_rank = line.split()
            rank = int(raw_rank)
            if rank < row_limit:
                tokens[rank] = base64.b64decode(encoded)
    return tokens


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(normalize_rows(left) * normalize_rows(right), axis=1)


def main() -> int:
    args = parse_args()
    base = GGUFReader(os.fspath(args.base), "r")
    tensors = {tensor.name: tensor for tensor in base.tensors}
    decoder = gpt2_byte_decoder()
    base_tokens = base.fields["tokenizer.ggml.tokens"].contents()
    base_ids: dict[bytes, int] = {}
    for token_id, token in enumerate(base_tokens):
        base_ids.setdefault(base_token_bytes(token, decoder), token_id)
    donor_tokens = load_kimi_tokens(args.donor_tokenizer, args.donor_rows)
    pairs = [
        (donor_id, base_ids[token])
        for donor_id, token in sorted(donor_tokens.items())
        if token in base_ids
    ]
    print(f"共享token: {len(pairs)}", flush=True)
    if not 0 < args.train_count < len(pairs):
        raise ValueError(f"train-count越界: {args.train_count} vs {len(pairs)}")

    donor_ids = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    target_ids = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    target_tensor = tensors["token_embd.weight"]
    target_embedding = np.asarray(
        dequantize(target_tensor.data[target_ids], target_tensor.tensor_type),
        dtype=np.float32,
    )
    donor_all = bf16(
        args.donor_embedding, (args.donor_rows, args.donor_width)
    )
    donor_embedding = donor_all[donor_ids]
    target_embedding = normalize_rows(target_embedding)
    donor_embedding = normalize_rows(donor_embedding)

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(pairs))
    train = permutation[: args.train_count]
    test = permutation[args.train_count :]
    cross = donor_embedding[train].T @ target_embedding[train]
    left, singular_values, right_t = svd(
        cross,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    transport = np.ascontiguousarray(left @ right_t, dtype=np.float32)

    train_before = row_cosine(
        donor_embedding[train, : target_embedding.shape[1]], target_embedding[train]
    )
    train_after = row_cosine(
        donor_embedding[train] @ transport, target_embedding[train]
    )
    test_before = row_cosine(
        donor_embedding[test, : target_embedding.shape[1]], target_embedding[test]
    )
    mapped_test = normalize_rows(donor_embedding[test] @ transport)
    target_test = normalize_rows(target_embedding[test])
    test_after = np.sum(mapped_test * target_test, axis=1)
    retrieval = mapped_test @ target_test.T
    top1 = float(np.mean(np.argmax(retrieval, axis=1) == np.arange(len(test))))

    donor_router = bf16(args.donor_router, (896, args.donor_width))
    base_router = np.asarray(
        tensors[f"blk.{args.base_layer}.ffn_gate_inp.weight"].data,
        dtype=np.float32,
    )
    base_router_unit = normalize_rows(base_router)
    mapped_router_weights = donor_router @ transport
    mapped_router = normalize_rows(mapped_router_weights) @ base_router_unit.T
    mapped_targets = np.argmax(mapped_router, axis=1)
    mapped_best = mapped_router[np.arange(mapped_router.shape[0]), mapped_targets]
    shuffled_router = donor_router[:, rng.permutation(args.donor_width)] @ transport
    shuffled_best = np.max(normalize_rows(shuffled_router) @ base_router_unit.T, axis=1)

    order = np.argsort(mapped_best)[::-1]
    pairs_report = []
    used_targets: set[int] = set()
    for donor_index in order:
        target_index = int(mapped_targets[donor_index])
        if target_index in used_targets:
            continue
        used_targets.add(target_index)
        pairs_report.append(
            {
                "donor_expert": int(donor_index),
                "target_expert": target_index,
                "router_cosine": float(mapped_best[donor_index]),
            }
        )
        if len(pairs_report) == 16:
            break

    identity = np.eye(transport.shape[1], dtype=np.float32)
    column_orthogonality_rmse = float(
        np.linalg.norm(transport.T @ transport - identity) / np.sqrt(identity.size)
    )
    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_matrix, transport, allow_pickle=False)
    report = {
        "format": "colorlm-rectangular-coordinate-transport-v1",
        "source": "moonshotai/Kimi-K3",
        "target": args.base.name,
        "method": "shared-token rectangular Procrustes",
        "matched_tokens": len(pairs),
        "train_tokens": len(train),
        "test_tokens": len(test),
        "embedding_cosine": {
            "train_unaligned_first2048": summary(train_before),
            "train_after": summary(train_after),
            "test_unaligned_first2048": summary(test_before),
            "test_after": summary(test_after),
        },
        "test_token_retrieval_top1": top1,
        "router_best_cosine": {
            "after_transport": summary(mapped_best),
            "shuffled_donor_axes": summary(shuffled_best),
        },
        "top_unique_router_pairs": pairs_report,
        "transport": {
            "shape": list(transport.shape),
            "dtype": str(transport.dtype),
            "column_orthogonality_rmse": column_orthogonality_rmse,
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
