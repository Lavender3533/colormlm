"""Build a full-rank router delta adapter for the Neural Alloy experiment.

This is a mechanism probe, not a low-rank approximation. For every MoE router
matrix D with shape [experts, hidden], it writes A=D and B=I, so B@A equals
the complete donor-minus-base delta (apart from the selected storage dtype).
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

from gguf import GGUFReader, GGUFWriter  # noqa: E402
from gguf.constants import Keys  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


ROUTER_SUFFIX = "ffn_gate_inp.weight"


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


def tensor_f32(tensor) -> np.ndarray:
    if tensor.data.dtype == np.float32:
        return np.asarray(tensor.data, dtype=np.float32)
    return np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)


def router_map(reader: GGUFReader) -> dict[str, object]:
    return {
        tensor.name: tensor
        for tensor in reader.tensors
        if tensor.name.endswith(ROUTER_SUFFIX)
    }


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(
        description="生成 Neural Alloy 满秩路由差分 adapter"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=models / "ColorLM-v4-SMoE.gguf",
        help="运行时加载的公共基座",
    )
    parser.add_argument(
        "--donor",
        type=Path,
        default=models / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        help="提供目标路由权重的供体",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=models / "ColorLM-Neural-Alloy-Router-F16.gguf",
    )
    parser.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.resolve() in (args.base.resolve(), args.donor.resolve()):
        raise ValueError("输出路径不能覆盖基座或供体")

    print(f"读取基座: {args.base}", flush=True)
    base_reader = GGUFReader(os.fspath(args.base), "r")
    print(f"读取供体: {args.donor}", flush=True)
    donor_reader = GGUFReader(os.fspath(args.donor), "r")

    base_routers = router_map(base_reader)
    donor_routers = router_map(donor_reader)
    names = sorted(set(base_routers) & set(donor_routers))
    if not names or set(base_routers) != set(donor_routers):
        raise RuntimeError("基座与供体的路由张量集合不兼容")

    first_shape = tuple(int(x) for x in base_routers[names[0]].data.shape)
    if len(first_shape) != 2:
        raise RuntimeError(f"路由张量不是矩阵: {first_shape}")
    output_dim, input_dim = first_shape
    rank = output_dim
    storage_dtype = np.float16 if args.dtype == "f16" else np.float32

    architecture = field_value(base_reader, "general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise RuntimeError("基座缺少 general.architecture")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(args.output, arch=architecture)
    writer.add_type("adapter")
    writer.add_string(Keys.Adapter.TYPE, "lora")
    # llama.cpp applies adapter_scale * alpha / rank. Setting alpha=rank
    # makes the command-line scale exactly equal to Neural Alloy alpha.
    writer.add_float32(Keys.Adapter.LORA_ALPHA, float(rank))
    writer.add_name("ColorLM Neural Alloy Router Full-Rank Delta")
    writer.add_description(
        "Full-rank donor-minus-base MoE router deltas; A=delta, B=identity"
    )

    squared_delta = 0.0
    squared_storage_error = 0.0
    max_abs_storage_error = 0.0
    identity = np.eye(rank, dtype=storage_dtype)

    for index, name in enumerate(names, start=1):
        base = tensor_f32(base_routers[name])
        donor = tensor_f32(donor_routers[name])
        if base.shape != donor.shape or base.shape != (output_dim, input_dim):
            raise RuntimeError(
                f"路由张量形状不一致: {name}: {base.shape} vs {donor.shape}"
            )

        delta_f32 = donor - base
        delta_stored = np.ascontiguousarray(delta_f32, dtype=storage_dtype)
        restored = delta_stored.astype(np.float32)
        error = restored - delta_f32
        squared_delta += float(np.dot(delta_f32.ravel(), delta_f32.ravel()))
        squared_storage_error += float(np.dot(error.ravel(), error.ravel()))
        max_abs_storage_error = max(
            max_abs_storage_error, float(np.max(np.abs(error)))
        )

        writer.add_tensor(f"{name}.lora_a", delta_stored)
        writer.add_tensor(f"{name}.lora_b", identity.copy())
        print(f"[{index:02d}/{len(names):02d}] {name}", flush=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    relative_error = (
        float(np.sqrt(squared_storage_error / squared_delta))
        if squared_delta
        else 0.0
    )
    manifest = {
        "format": "neural-alloy-router-v1",
        "base": args.base.name,
        "donor": args.donor.name,
        "adapter": args.output.name,
        "architecture": architecture,
        "tensor_count": len(names),
        "rank": rank,
        "storage_dtype": args.dtype,
        "runtime_scale_semantics": "effective_weight = base + scale * delta",
        "delta_storage_relative_l2_error": relative_error,
        "delta_storage_max_abs_error": max_abs_storage_error,
        "adapter_bytes": args.output.stat().st_size,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"adapter: {args.output}")
    print(f"清单: {manifest_path}")
    print(f"F16 差分相对误差: {relative_error:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
