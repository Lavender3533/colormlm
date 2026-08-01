"""Probe whether two compatible GGUF checkpoints have compressible weight deltas.

The experiment is intentionally weight-only. It does not run inference, alter a
checkpoint, or claim a capability gain. It answers the first Neural Alloy
question: can checkpoint differences be represented by a small atom basis?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


ROUTER_SUFFIX = "ffn_gate_inp.weight"
SAMPLE_TENSORS = (
    "blk.0.attn_gate.weight",
    "blk.0.ffn_gate_shexp.weight",
    "blk.0.ssm_out.weight",
)
ATOM_COUNTS = (1, 2, 4, 8, 16, 24, 32)
MATRIX_RANKS = (1, 2, 4, 8, 16, 32, 64, 128)


def field_value(reader: GGUFReader, name: str):
    field = reader.fields.get(name)
    if field is None:
        return None
    value = field.contents()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def tensor_map(reader: GGUFReader):
    return {tensor.name: tensor for tensor in reader.tensors}


def tensor_f32(tensor) -> np.ndarray:
    if tensor.data.dtype == np.float32:
        return np.asarray(tensor.data, dtype=np.float32)
    return np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)


def cumulative_energy(eigenvalues: np.ndarray, counts) -> dict[str, float]:
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    values = np.sort(values)[::-1]
    total = float(values.sum())
    if total == 0.0:
        return {str(count): 1.0 for count in counts if count <= len(values)}
    cumulative = np.cumsum(values) / total
    return {
        str(count): float(cumulative[count - 1])
        for count in counts
        if count <= len(values)
    }


def rank_for_energy(eigenvalues: np.ndarray, threshold: float) -> int:
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    values = np.sort(values)[::-1]
    total = float(values.sum())
    if total == 0.0:
        return 0
    return int(np.searchsorted(np.cumsum(values) / total, threshold) + 1)


def architecture_summary(reader: GGUFReader) -> dict:
    arch = field_value(reader, "general.architecture")
    return {
        "architecture": arch,
        "block_count": field_value(reader, f"{arch}.block_count"),
        "embedding_length": field_value(reader, f"{arch}.embedding_length"),
        "expert_count": field_value(reader, f"{arch}.expert_count"),
        "expert_used_count": field_value(reader, f"{arch}.expert_used_count"),
        "tensor_count": len(reader.tensors),
    }


def validate_compatibility(a_tensors: dict, b_tensors: dict) -> dict:
    common = sorted(set(a_tensors) & set(b_tensors))
    only_a = sorted(set(a_tensors) - set(b_tensors))
    only_b = sorted(set(b_tensors) - set(a_tensors))
    shape_mismatches = []
    for name in common:
        a_shape = tuple(int(x) for x in a_tensors[name].shape)
        b_shape = tuple(int(x) for x in b_tensors[name].shape)
        if a_shape != b_shape:
            shape_mismatches.append({"name": name, "a": a_shape, "b": b_shape})
    return {
        "common_tensor_count": len(common),
        "only_a_count": len(only_a),
        "only_b_count": len(only_b),
        "shape_mismatch_count": len(shape_mismatches),
        "only_a_preview": only_a[:20],
        "only_b_preview": only_b[:20],
        "shape_mismatch_preview": shape_mismatches[:20],
    }


def f32_change_summary(a_tensors: dict, b_tensors: dict) -> dict:
    common = sorted(set(a_tensors) & set(b_tensors))
    f32_names = [
        name
        for name in common
        if a_tensors[name].data.dtype == np.float32
        and b_tensors[name].data.dtype == np.float32
        and tuple(a_tensors[name].shape) == tuple(b_tensors[name].shape)
    ]
    exact = 0
    squared_delta = 0.0
    squared_base = 0.0
    for name in f32_names:
        a = np.asarray(a_tensors[name].data, dtype=np.float32)
        b = np.asarray(b_tensors[name].data, dtype=np.float32)
        exact += int(np.array_equal(a, b))
        delta = a.astype(np.float64) - b.astype(np.float64)
        squared_delta += float(np.dot(delta.ravel(), delta.ravel()))
        b64 = b.astype(np.float64)
        squared_base += float(np.dot(b64.ravel(), b64.ravel()))
    return {
        "shared_f32_tensor_count": len(f32_names),
        "exact_f32_tensor_count": exact,
        "changed_f32_tensor_count": len(f32_names) - exact,
        "global_relative_delta_l2": (
            float(np.sqrt(squared_delta / squared_base)) if squared_base else 0.0
        ),
    }


def router_atom_probe(a_tensors: dict, b_tensors: dict) -> dict:
    router_names = sorted(
        name
        for name in set(a_tensors) & set(b_tensors)
        if name.endswith(ROUTER_SUFFIX)
        and a_tensors[name].data.dtype == np.float32
        and b_tensors[name].data.dtype == np.float32
        and tuple(a_tensors[name].shape) == tuple(b_tensors[name].shape)
    )
    if not router_names:
        raise RuntimeError("No compatible F32 router tensors were found")

    deltas = []
    layer_norms = []
    for name in router_names:
        a = np.asarray(a_tensors[name].data, dtype=np.float32)
        b = np.asarray(b_tensors[name].data, dtype=np.float32)
        delta = (a - b).reshape(-1)
        deltas.append(delta)
        layer_norms.append(float(np.linalg.norm(delta)))
    delta_stack = np.stack(deltas, axis=0)

    # The 40 x 40 Gram matrix gives the exact singular-value spectrum without
    # materializing the enormous right-singular-vector matrix.
    gram = delta_stack.astype(np.float64) @ delta_stack.astype(np.float64).T
    cross_layer_eigenvalues = np.linalg.eigvalsh(gram)

    sample_delta = (
        np.asarray(a_tensors[router_names[0]].data, dtype=np.float32)
        - np.asarray(b_tensors[router_names[0]].data, dtype=np.float32)
    )
    sample_singular_values = np.linalg.svd(sample_delta, compute_uv=False)
    sample_eigenvalues = np.square(sample_singular_values.astype(np.float64))
    matrix_shape = tuple(int(x) for x in sample_delta.shape)
    full_values = int(np.prod(matrix_shape))

    rank_storage = {}
    for rank in MATRIX_RANKS:
        if rank <= min(matrix_shape):
            factor_values = rank * (matrix_shape[0] + matrix_shape[1])
            rank_storage[str(rank)] = {
                "energy": cumulative_energy(sample_eigenvalues, (rank,))[str(rank)],
                "compression_ratio_vs_full_delta": full_values / factor_values,
            }

    return {
        "router_tensor_count": len(router_names),
        "router_shape": matrix_shape,
        "layer_delta_l2_min": min(layer_norms),
        "layer_delta_l2_mean": float(np.mean(layer_norms)),
        "layer_delta_l2_max": max(layer_norms),
        "cross_layer_atom_energy": cumulative_energy(
            cross_layer_eigenvalues, ATOM_COUNTS
        ),
        "cross_layer_atoms_for_90pct": rank_for_energy(
            cross_layer_eigenvalues, 0.90
        ),
        "cross_layer_atoms_for_95pct": rank_for_energy(
            cross_layer_eigenvalues, 0.95
        ),
        "cross_layer_atoms_for_99pct": rank_for_energy(
            cross_layer_eigenvalues, 0.99
        ),
        "first_layer_matrix_rank_probe": rank_storage,
    }


def sampled_tensor_probe(a_tensors: dict, b_tensors: dict) -> list[dict]:
    results = []
    for name in SAMPLE_TENSORS:
        if name not in a_tensors or name not in b_tensors:
            continue
        a_tensor = a_tensors[name]
        b_tensor = b_tensors[name]
        if tuple(a_tensor.shape) != tuple(b_tensor.shape):
            continue
        a = tensor_f32(a_tensor).reshape(-1).astype(np.float64)
        b = tensor_f32(b_tensor).reshape(-1).astype(np.float64)
        delta = a - b
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        denominator = a_norm * b_norm
        results.append(
            {
                "name": name,
                "shape": tuple(int(x) for x in a_tensor.shape),
                "a_quantization": a_tensor.tensor_type.name,
                "b_quantization": b_tensor.tensor_type.name,
                "cosine_similarity": (
                    float(np.dot(a, b) / denominator) if denominator else 1.0
                ),
                "relative_delta_l2_vs_b": (
                    float(np.linalg.norm(delta) / b_norm) if b_norm else 0.0
                ),
            }
        )
    return results


def write_markdown(report: dict, path: Path) -> None:
    router = report["router_atom_probe"]
    lines = [
        "# Neural Alloy 权重原子探针",
        "",
        f"- 供体 A：`{report['model_a']}`",
        f"- 供体 B：`{report['model_b']}`",
        f"- 共享张量：{report['compatibility']['common_tensor_count']}",
        f"- 形状冲突：{report['compatibility']['shape_mismatch_count']}",
        f"- 变化的 F32 张量：{report['f32_changes']['changed_f32_tensor_count']} / {report['f32_changes']['shared_f32_tensor_count']}",
        "",
        "## 跨层路由权重原子",
        "",
        "| 原子数 | 捕获差异能量 |",
        "|---:|---:|",
    ]
    for count, energy in router["cross_layer_atom_energy"].items():
        lines.append(f"| {count} | {energy:.4%} |")
    lines.extend(
        [
            "",
            f"- 90% 能量所需原子：{router['cross_layer_atoms_for_90pct']}",
            f"- 95% 能量所需原子：{router['cross_layer_atoms_for_95pct']}",
            f"- 99% 能量所需原子：{router['cross_layer_atoms_for_99pct']}",
            "",
            "## 单层路由矩阵低秩分解",
            "",
            "| 秩 | 捕获差异能量 | 相对完整差分压缩率 |",
            "|---:|---:|---:|",
        ]
    )
    for rank, values in router["first_layer_matrix_rank_probe"].items():
        lines.append(
            f"| {rank} | {values['energy']:.4%} | "
            f"{values['compression_ratio_vs_full_delta']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## 解读规则",
            "",
            "- 少量原子即可捕获高比例能量：支持继续研究共享权重原子。",
            "- 需要接近完整秩才能捕获差异：低秩权重合金不适合这两个供体。",
            "- 权重重建只是第一关，不等于已保留或提高模型能力。",
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
        default=ROOT / "fast16" / "research" / "neural_alloy_probe.json",
    )
    args = parser.parse_args()

    print(f"读取供体 A: {args.model_a}", flush=True)
    reader_a = GGUFReader(os.fspath(args.model_a))
    print(f"读取供体 B: {args.model_b}", flush=True)
    reader_b = GGUFReader(os.fspath(args.model_b))
    tensors_a = tensor_map(reader_a)
    tensors_b = tensor_map(reader_b)

    compatibility = validate_compatibility(tensors_a, tensors_b)
    if compatibility["shape_mismatch_count"]:
        print("存在张量形状冲突，停止原子分解。", file=sys.stderr)
        return 2

    print("统计 F32 差异...", flush=True)
    f32_changes = f32_change_summary(tensors_a, tensors_b)
    print("分解路由器差异...", flush=True)
    router_probe = router_atom_probe(tensors_a, tensors_b)
    print("抽样反量化共享张量...", flush=True)
    sampled_tensors = sampled_tensor_probe(tensors_a, tensors_b)

    report = {
        "model_a": args.model_a.name,
        "model_b": args.model_b.name,
        "architecture_a": architecture_summary(reader_a),
        "architecture_b": architecture_summary(reader_b),
        "compatibility": compatibility,
        "f32_changes": f32_changes,
        "router_atom_probe": router_probe,
        "sampled_tensors": sampled_tensors,
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
