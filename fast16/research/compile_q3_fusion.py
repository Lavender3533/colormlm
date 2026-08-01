"""Compile Neural Alloy q3_g64 deltas into a standard GGUF checkpoint.

The resulting model has no runtime adapter or custom CPU operation. Selected
weights are materialized as ``base + alpha * decoded_delta`` while every other
base tensor is copied without requantization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))
sys.path.insert(0, os.fspath(RESEARCH))

from gguf import GGUFReader, GGUFWriter  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from build_heterogeneous_slices import copy_metadata  # noqa: E402
from neural_alloy_codec import AlloyReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(
        description="把q3_g64 Neural Alloy差分直接融合进单一GGUF"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=models / "ColorLM-v5-GLM-SynapticGraft.gguf",
    )
    parser.add_argument(
        "--container",
        type=Path,
        default=models / "ColorLM-v5-to-Qwen36-router-q3g64.nal",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=models / "ColorLM-v6-Q3Router-Fused-A1.gguf",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="差分融合强度，1表示q3可表示范围内的完整目标差分",
    )
    return parser.parse_args()


def tensor_f32(tensor) -> np.ndarray:
    if tensor.data.dtype == np.float32:
        return np.asarray(tensor.data, dtype=np.float32)
    return np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)


def add_source_tensor_info(writer: GGUFWriter, tensor) -> None:
    writer.add_tensor_info(
        tensor.name,
        tensor.data.shape,
        tensor.data.dtype,
        tensor.data.nbytes,
        raw_dtype=tensor.tensor_type,
    )


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not -2.0 <= args.alpha <= 2.0:
        raise ValueError("alpha必须在-2到2之间")

    protected = {args.base.resolve(), args.container.resolve()}
    if args.output.resolve() in protected:
        raise ValueError("输出不能覆盖基座或q3容器")
    if args.output.exists():
        raise FileExistsError(f"输出已存在: {args.output}")

    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"上次的临时文件仍存在: {partial}")

    print(f"读取基座: {args.base}", flush=True)
    base = GGUFReader(os.fspath(args.base), "r")
    alloy = AlloyReader(args.container)
    manifest_base = alloy.manifest.get("base", {})
    if manifest_base.get("name") != args.base.name:
        raise RuntimeError(
            f"q3基座名称不匹配: {manifest_base.get('name')} vs {args.base.name}"
        )
    if manifest_base.get("bytes") != args.base.stat().st_size:
        raise RuntimeError("q3容器记录的基座大小与当前文件不一致")

    base_tensors = {tensor.name: tensor for tensor in base.tensors}
    fused_names = set(alloy.tensors)
    missing = sorted(fused_names - set(base_tensors))
    if missing:
        raise RuntimeError(f"基座缺少q3张量: {missing[:3]}")

    for name in fused_names:
        tensor = base_tensors[name]
        item = alloy.tensors[name]
        if list(map(int, tensor.shape)) != list(map(int, item["shape"])):
            raise RuntimeError(f"q3张量形状不匹配: {name}")

    architecture = str(base.fields["general.architecture"].contents())
    output_name = f"ColorLM-v6-Q3Router-Fused-alpha{args.alpha:g}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(partial, arch=architecture)
    copy_metadata(writer, base, {"general.name": output_name})
    writer.add_string("colorlm.neural_alloy.format", "q3-g64-compiled-v1")
    writer.add_string("colorlm.neural_alloy.container", args.container.name)
    writer.add_float32("colorlm.neural_alloy.alpha", float(args.alpha))
    writer.add_uint32("colorlm.neural_alloy.tensor_count", len(fused_names))

    for tensor in base.tensors:
        if tensor.name not in fused_names:
            add_source_tensor_info(writer, tensor)
            continue
        # GGUF logical dimensions are reversed relative to NumPy row-major
        # matrix storage. Materialized alloy tensors are kept in F32 because
        # MoE routers in the base checkpoint are F32.
        matrix_shape = tuple(int(value) for value in reversed(tensor.shape))
        matrix_bytes = int(np.prod(matrix_shape, dtype=np.int64)) * 4
        writer.add_tensor_info(
            tensor.name,
            matrix_shape,
            np.dtype(np.float32),
            matrix_bytes,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    fused_report = []
    for index, tensor in enumerate(base.tensors, start=1):
        if tensor.name not in fused_names:
            writer.write_tensor_data(tensor.data)
            continue

        base_matrix = tensor_f32(tensor)
        delta = alloy.read_delta(tensor.name)
        if base_matrix.shape != delta.shape:
            raise RuntimeError(
                f"q3解码形状不匹配: {tensor.name}: "
                f"{base_matrix.shape} vs {delta.shape}"
            )
        fused = np.ascontiguousarray(
            base_matrix + np.float32(args.alpha) * delta,
            dtype=np.float32,
        )
        writer.write_tensor_data(fused)
        fused_report.append(
            {
                "name": tensor.name,
                "shape": list(map(int, fused.shape)),
                "delta_rms": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))),
                "delta_max_abs": float(np.max(np.abs(delta))),
            }
        )
        print(
            f"[{len(fused_report):02d}/{len(fused_names):02d}] 融合 {tensor.name}",
            flush=True,
        )

    writer.close()
    partial.replace(args.output)

    report = {
        "format": "neural-alloy-q3-compiled-v1",
        "base": args.base.name,
        "container": args.container.name,
        "target": alloy.manifest.get("target"),
        "output": args.output.name,
        "alpha": args.alpha,
        "tensor_count": len(fused_report),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
        "runtime": "standard GGUF tensors; no adapter and no custom q3 CPU op",
        "tensors": fused_report,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"融合模型: {args.output}")
    print(f"完整性报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
