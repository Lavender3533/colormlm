"""Measure whether Coder-Next and ColorLM share a usable hidden coordinate system."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def parse_args() -> argparse.Namespace:
    extracted = (
        RESEARCH
        / "biopsy_cache"
        / "Qwen_Qwen3-Coder-Next"
        / "master"
        / "extracted"
    )
    parser = argparse.ArgumentParser(description="分析跨模型隐藏坐标兼容性")
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
        "--donor-tokenizer",
        type=Path,
        default=(
            RESEARCH
            / "biopsy_cache"
            / "Qwen_Qwen3-Coder-Next"
            / "master"
            / "metadata"
            / "tokenizer.json"
        ),
    )
    parser.add_argument("--token-count", type=int, default=4096)
    parser.add_argument("--geometry-sample", type=int, default=1024)
    parser.add_argument("--base-layer", type=int, default=39)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESEARCH / "coder_next_alignment_report.json",
    )
    return parser.parse_args()


def bf16(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape, dtype=np.int64)) * 2
    if path.stat().st_size != expected:
        raise RuntimeError(f"BF16文件大小不匹配: {path}: {path.stat().st_size} vs {expected}")
    bits = np.fromfile(path, dtype="<u2").astype(np.uint32) << 16
    return np.ascontiguousarray(bits.view(np.float32).reshape(shape))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, np.float32(1e-12))


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def centered_gram(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    gram = centered @ centered.T
    gram -= np.mean(gram, axis=0, keepdims=True)
    gram -= np.mean(gram, axis=1, keepdims=True)
    gram += np.float32(np.mean(gram))
    return gram


def main() -> int:
    args = parse_args()
    if args.token_count < args.geometry_sample:
        raise ValueError("token-count不能小于geometry-sample")

    base = GGUFReader(os.fspath(args.base), "r")
    tensors = {tensor.name: tensor for tensor in base.tensors}
    base_tokens = base.fields["tokenizer.ggml.tokens"].contents()
    base_token_ids: dict[str, int] = {}
    for token_id, token in enumerate(base_tokens):
        base_token_ids.setdefault(token, token_id)

    donor_tokenizer = json.loads(args.donor_tokenizer.read_text(encoding="utf-8"))
    donor_tokens = {
        int(token_id): token
        for token, token_id in donor_tokenizer["model"]["vocab"].items()
    }
    donor_tokens.update(
        {
            int(item["id"]): item["content"]
            for item in donor_tokenizer.get("added_tokens", [])
        }
    )
    pairs = [
        (donor_id, base_token_ids[token])
        for donor_id, token in sorted(donor_tokens.items())
        if donor_id < args.token_count and token in base_token_ids
    ]
    if len(pairs) < args.geometry_sample:
        raise RuntimeError(f"共享token不足: {len(pairs)}")
    donor_ids = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    base_ids = np.asarray([pair[1] for pair in pairs], dtype=np.int64)

    embedding_tensor = tensors["token_embd.weight"]
    base_embedding = np.asarray(
        dequantize(embedding_tensor.data[base_ids], embedding_tensor.tensor_type),
        dtype=np.float32,
    )
    donor_embedding_all = bf16(args.donor_embedding, (args.token_count, 2048))
    donor_embedding = donor_embedding_all[donor_ids]

    base_unit = normalize_rows(base_embedding)
    donor_unit = normalize_rows(donor_embedding)
    token_cosine = np.sum(base_unit * donor_unit, axis=1)

    indices = np.linspace(
        0, len(pairs) - 1, args.geometry_sample, dtype=np.int64
    )
    base_gram = centered_gram(base_unit[indices])
    donor_gram = centered_gram(donor_unit[indices])
    gram_denominator = np.linalg.norm(base_gram) * np.linalg.norm(donor_gram)
    linear_cka = float(
        np.sum(base_gram * donor_gram) / max(float(gram_denominator), 1e-12)
    )
    upper = np.triu_indices(args.geometry_sample, 1)
    geometry_correlation = float(
        np.corrcoef(base_gram[upper], donor_gram[upper])[0, 1]
    )

    base_router_tensor = tensors[f"blk.{args.base_layer}.ffn_gate_inp.weight"]
    base_router = np.asarray(base_router_tensor.data, dtype=np.float32)
    donor_router = bf16(args.donor_router, (512, 2048))
    base_router_unit = normalize_rows(base_router)
    donor_router_unit = normalize_rows(donor_router)
    router_cosine = donor_router_unit @ base_router_unit.T
    best_target = np.argmax(router_cosine, axis=1)
    best_cosine = router_cosine[np.arange(router_cosine.shape[0]), best_target]
    order = np.argsort(best_cosine)[::-1]
    unique_pairs = []
    claimed_targets: set[int] = set()
    for donor_index in order:
        target_index = int(best_target[donor_index])
        if target_index in claimed_targets:
            continue
        claimed_targets.add(target_index)
        unique_pairs.append(
            {
                "donor_expert": int(donor_index),
                "target_expert": target_index,
                "router_cosine": float(best_cosine[donor_index]),
            }
        )
        if len(unique_pairs) == 16:
            break

    rng = np.random.default_rng(20260729)
    shuffled = donor_router_unit[:, rng.permutation(donor_router_unit.shape[1])]
    shuffled_best = np.max(shuffled @ base_router_unit.T, axis=1)

    report = {
        "format": "colorlm-hidden-coordinate-audit-v1",
        "base": args.base.name,
        "donor": "Qwen/Qwen3-Coder-Next",
        "donor_embedding_rows": args.token_count,
        "matched_token_count": len(pairs),
        "same_id_count": int(np.sum(donor_ids == base_ids)),
        "geometry_sample": args.geometry_sample,
        "embedding": {
            "same_token_direct_cosine": summary(token_cosine),
            "linear_cka": linear_cka,
            "pairwise_geometry_correlation": geometry_correlation,
            "base_rms": float(np.sqrt(np.mean(np.square(base_embedding)))),
            "donor_rms": float(np.sqrt(np.mean(np.square(donor_embedding)))),
        },
        "router": {
            "base_layer": args.base_layer,
            "direct_best_cosine": summary(best_cosine),
            "shuffled_axis_best_cosine": summary(shuffled_best),
            "top_unique_pairs": unique_pairs,
        },
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
