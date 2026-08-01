"""Measure low-bit full-rank checkpoint deltas for Neural Alloy.

Low-rank decomposition loses most of the observed router delta. This probe keeps
the full delta topology and quantizes only its values. It estimates whether one
base checkpoint plus compact capability deltas can fit the 30 GiB budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))
sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))

from gguf import GGUFReader  # noqa: E402
from neural_alloy_probe import (  # noqa: E402
    ROUTER_SUFFIX,
    SAMPLE_TENSORS,
    tensor_f32,
    tensor_map,
    validate_compatibility,
)


SCHEMES = (
    (2, 32),
    (2, 64),
    (2, 128),
    (3, 32),
    (3, 64),
    (3, 128),
    (4, 32),
    (4, 64),
    (4, 128),
)


def quantize_residual(
    delta: np.ndarray, bits: int, group_size: int
) -> tuple[np.ndarray, int]:
    flat = np.asarray(delta, dtype=np.float32).reshape(-1)
    original_size = flat.size
    padding = (-original_size) % group_size
    if padding:
        flat = np.pad(flat, (0, padding))
    groups = flat.reshape(-1, group_size)

    # Two bits use a signed ternary code (-1, 0, +1). Higher bit widths use
    # ordinary symmetric integer levels. One 2-byte scale is stored per group.
    qmax = 1 if bits == 2 else (1 << (bits - 1)) - 1
    absmax = np.max(np.abs(groups), axis=1, keepdims=True)
    scale = np.where(absmax > 0.0, absmax / qmax, 1.0).astype(np.float32)
    quantized = np.clip(np.rint(groups / scale), -qmax, qmax)
    reconstructed = (quantized * scale).reshape(-1)[:original_size]

    code_bytes = (original_size * bits + 7) // 8
    scale_bytes = groups.shape[0] * 2
    return reconstructed.reshape(delta.shape), code_bytes + scale_bytes


class MetricAccumulator:
    def __init__(self) -> None:
        self.delta_error_sq = 0.0
        self.delta_sq = 0.0
        self.reconstructed_delta_sq = 0.0
        self.delta_dot = 0.0
        self.target_error_sq = 0.0
        self.target_sq = 0.0
        self.full_delta_bytes = 0
        self.compressed_bytes = 0

    def add(
        self,
        target: np.ndarray,
        base: np.ndarray,
        reconstructed_delta: np.ndarray,
        compressed_bytes: int,
    ) -> None:
        target64 = np.asarray(target, dtype=np.float64).reshape(-1)
        base64 = np.asarray(base, dtype=np.float64).reshape(-1)
        delta64 = target64 - base64
        reconstructed64 = np.asarray(reconstructed_delta, dtype=np.float64).reshape(-1)
        delta_error = delta64 - reconstructed64
        target_error = target64 - (base64 + reconstructed64)

        self.delta_error_sq += float(np.dot(delta_error, delta_error))
        self.delta_sq += float(np.dot(delta64, delta64))
        self.reconstructed_delta_sq += float(
            np.dot(reconstructed64, reconstructed64)
        )
        self.delta_dot += float(np.dot(delta64, reconstructed64))
        self.target_error_sq += float(np.dot(target_error, target_error))
        self.target_sq += float(np.dot(target64, target64))
        self.full_delta_bytes += delta64.size * 2
        self.compressed_bytes += compressed_bytes

    def result(self) -> dict:
        cosine_denominator = np.sqrt(self.delta_sq * self.reconstructed_delta_sq)
        return {
            "delta_relative_error": (
                float(np.sqrt(self.delta_error_sq / self.delta_sq))
                if self.delta_sq
                else 0.0
            ),
            "delta_cosine_similarity": (
                float(self.delta_dot / cosine_denominator)
                if cosine_denominator
                else 1.0
            ),
            "target_relative_error": (
                float(np.sqrt(self.target_error_sq / self.target_sq))
                if self.target_sq
                else 0.0
            ),
            "compression_ratio_vs_bf16_delta": (
                self.full_delta_bytes / self.compressed_bytes
                if self.compressed_bytes
                else 0.0
            ),
            "compressed_bytes": self.compressed_bytes,
        }


def logical_parameter_count(tensors: dict) -> int:
    return sum(int(np.prod(tensor.shape, dtype=np.int64)) for tensor in tensors.values())


def estimate_full_delta_bytes(parameter_count: int, bits: int, group_size: int) -> int:
    code_bytes = (parameter_count * bits + 7) // 8
    scale_bytes = ((parameter_count + group_size - 1) // group_size) * 2
    return code_bytes + scale_bytes


def probe_routers(a_tensors: dict, b_tensors: dict) -> dict:
    router_names = sorted(
        name
        for name in set(a_tensors) & set(b_tensors)
        if name.endswith(ROUTER_SUFFIX)
        and a_tensors[name].data.dtype == np.float32
        and b_tensors[name].data.dtype == np.float32
        and tuple(a_tensors[name].shape) == tuple(b_tensors[name].shape)
    )
    accumulators = {scheme: MetricAccumulator() for scheme in SCHEMES}
    for index, name in enumerate(router_names, start=1):
        target = np.asarray(a_tensors[name].data, dtype=np.float32)
        base = np.asarray(b_tensors[name].data, dtype=np.float32)
        delta = target - base
        for scheme in SCHEMES:
            reconstructed, compressed_bytes = quantize_residual(delta, *scheme)
            accumulators[scheme].add(target, base, reconstructed, compressed_bytes)
        print(f"  路由层 {index:02d}/{len(router_names)}", end="\r", flush=True)
    print(" " * 32, end="\r")
    return {
        f"q{bits}_g{group_size}": accumulator.result()
        for (bits, group_size), accumulator in accumulators.items()
    }


def probe_samples(a_tensors: dict, b_tensors: dict) -> dict:
    results = defaultdict(dict)
    sample_schemes = ((2, 64), (3, 64), (4, 64))
    for name in SAMPLE_TENSORS:
        if name not in a_tensors or name not in b_tensors:
            continue
        if tuple(a_tensors[name].shape) != tuple(b_tensors[name].shape):
            continue
        target = tensor_f32(a_tensors[name])
        base = tensor_f32(b_tensors[name])
        delta = target - base
        for scheme in sample_schemes:
            reconstructed, compressed_bytes = quantize_residual(delta, *scheme)
            accumulator = MetricAccumulator()
            accumulator.add(target, base, reconstructed, compressed_bytes)
            results[name][f"q{scheme[0]}_g{scheme[1]}"] = accumulator.result()
    return dict(results)


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Neural Alloy 全秩低比特差分探针",
        "",
        f"- 目标供体：`{report['model_a']}`",
        f"- 基座供体：`{report['model_b']}`",
        f"- 逻辑参数量：{report['logical_parameter_count']:,}",
        "",
        "## 40 层路由器差分",
        "",
        "| 方案 | 差分余弦相似度 | 差分相对误差 | 目标权重相对误差 | 相对 BF16 差分压缩 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in report["router_residual_probe"].items():
        lines.append(
            f"| {name} | {values['delta_cosine_similarity']:.6f} | "
            f"{values['delta_relative_error']:.6f} | "
            f"{values['target_relative_error']:.6f} | "
            f"{values['compression_ratio_vs_bf16_delta']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## 整模型差分体积估算",
            "",
            "| 方案 | 差分估算 | 基座 A + 差分 | 基座 B + 差分 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in report["size_estimates"].items():
        lines.append(
            f"| {name} | {values['delta_gib']:.2f} GiB | "
            f"{values['model_a_plus_delta_gib']:.2f} GiB | "
            f"{values['model_b_plus_delta_gib']:.2f} GiB |"
        )
    lines.extend(
        [
            "",
            "## 解读规则",
            "",
            "- 差分余弦相似度表示供体能力变化方向被保留的程度。",
            "- 目标权重误差小不代表行为已完全保留；必须再生成可推理原型验证。",
            "- 本探针不会修改或生成模型文件。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a", type=Path)
    parser.add_argument("model_b", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fast16" / "research" / "neural_alloy_residual_probe.json",
    )
    args = parser.parse_args()

    print(f"读取目标供体: {args.model_a}", flush=True)
    reader_a = GGUFReader(os.fspath(args.model_a))
    print(f"读取基座供体: {args.model_b}", flush=True)
    reader_b = GGUFReader(os.fspath(args.model_b))
    tensors_a = tensor_map(reader_a)
    tensors_b = tensor_map(reader_b)
    compatibility = validate_compatibility(tensors_a, tensors_b)
    if compatibility["shape_mismatch_count"]:
        print("张量形状不兼容，停止。", file=sys.stderr)
        return 2

    print("量化 40 层路由差分...", flush=True)
    router_results = probe_routers(tensors_a, tensors_b)
    print("量化抽样神经张量差分...", flush=True)
    sample_results = probe_samples(tensors_a, tensors_b)

    parameter_count = logical_parameter_count(tensors_a)
    a_bytes = args.model_a.stat().st_size
    b_bytes = args.model_b.stat().st_size
    size_estimates = {}
    for bits, group_size in SCHEMES:
        delta_bytes = estimate_full_delta_bytes(parameter_count, bits, group_size)
        name = f"q{bits}_g{group_size}"
        size_estimates[name] = {
            "delta_gib": delta_bytes / (1024**3),
            "model_a_plus_delta_gib": (a_bytes + delta_bytes) / (1024**3),
            "model_b_plus_delta_gib": (b_bytes + delta_bytes) / (1024**3),
        }

    report = {
        "model_a": args.model_a.name,
        "model_b": args.model_b.name,
        "compatibility": compatibility,
        "logical_parameter_count": parameter_count,
        "router_residual_probe": router_results,
        "sample_tensor_residual_probe": sample_results,
        "size_estimates": size_estimates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path = args.output.with_suffix(".md")
    write_markdown(report, markdown_path)
    print(f"结果: {args.output}")
    print(f"摘要: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
