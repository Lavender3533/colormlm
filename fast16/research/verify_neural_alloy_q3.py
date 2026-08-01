"""Verify Neural Alloy q3_g64 reconstruction against source GGUF tensors."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))
sys.path.insert(0, os.fspath(RESEARCH))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from neural_alloy_codec import AlloyReader  # noqa: E402


def tensor_f32(tensor) -> np.ndarray:
    if tensor.data.dtype == np.float32:
        return np.asarray(tensor.data, dtype=np.float32)
    return np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("container", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    alloy = AlloyReader(args.container)
    base = GGUFReader(os.fspath(args.base), "r")
    target = GGUFReader(os.fspath(args.target), "r")
    base_map = {tensor.name: tensor for tensor in base.tensors}
    target_map = {tensor.name: tensor for tensor in target.tensors}

    delta_error_sq = 0.0
    delta_sq = 0.0
    target_error_sq = 0.0
    target_sq = 0.0
    delta_dot = 0.0
    reconstructed_sq = 0.0
    for index, item in enumerate(alloy.manifest["tensors"], start=1):
        name = item["name"]
        base_values = tensor_f32(base_map[name])
        target_values = tensor_f32(target_map[name])
        reconstructed_delta = alloy.read_delta(name)
        exact_delta = target_values - base_values
        delta_error = exact_delta - reconstructed_delta
        target_error = target_values - (base_values + reconstructed_delta)
        delta_error_sq += float(np.dot(delta_error.ravel(), delta_error.ravel()))
        delta_sq += float(np.dot(exact_delta.ravel(), exact_delta.ravel()))
        target_error_sq += float(np.dot(target_error.ravel(), target_error.ravel()))
        target_sq += float(np.dot(target_values.ravel(), target_values.ravel()))
        delta_dot += float(np.dot(exact_delta.ravel(), reconstructed_delta.ravel()))
        reconstructed_sq += float(np.dot(reconstructed_delta.ravel(), reconstructed_delta.ravel()))
        print(f"[{index}/{alloy.header.tensor_count}] {name}", flush=True)

    cosine = delta_dot / np.sqrt(delta_sq * reconstructed_sq) if delta_sq and reconstructed_sq else 1.0
    print(f"delta_relative_error={np.sqrt(delta_error_sq / delta_sq):.9f}")
    print(f"delta_cosine_similarity={cosine:.9f}")
    print(f"target_relative_error={np.sqrt(target_error_sq / target_sq):.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
