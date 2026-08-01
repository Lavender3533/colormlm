"""Export q3_g64 router deltas as a temporary full-rank llama.cpp adapter."""

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
sys.path.insert(0, os.fspath(RESEARCH))

from gguf import GGUFReader, GGUFWriter  # noqa: E402
from gguf.constants import Keys  # noqa: E402

from neural_alloy_codec import AlloyReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "container", type=Path, default=models / "ColorLM-v5-to-Qwen36-router-q3g64.nal", nargs="?"
    )
    parser.add_argument(
        "--base", type=Path, default=models / "ColorLM-v5-GLM-SynapticGraft.gguf"
    )
    parser.add_argument(
        "--output", type=Path, default=models / "ColorLM-v5-Q3Router-Bridge.gguf"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alloy = AlloyReader(args.container)
    base = GGUFReader(os.fspath(args.base), "r")
    architecture = base.fields["general.architecture"].contents()
    ranks = set()
    tensors = []

    for item in alloy.manifest["tensors"]:
        delta = alloy.read_delta(item["name"])
        if delta.ndim != 2:
            raise RuntimeError(f"临时桥只支持二维矩阵: {item['name']}")
        output_dim, input_dim = delta.shape
        if output_dim > input_dim:
            raise RuntimeError(f"临时桥要求输出维度不大于输入维度: {item['name']}")
        rank = output_dim
        ranks.add(rank)
        tensors.append(
            (
                item["name"],
                np.ascontiguousarray(delta, dtype=np.float16),
                np.eye(rank, dtype=np.float16),
            )
        )
    if len(ranks) != 1:
        raise RuntimeError("llama.cpp adapter alpha 是全局值，临时桥要求所有矩阵秩相同")
    rank = ranks.pop()

    writer = GGUFWriter(args.output, arch=str(architecture))
    writer.add_type("adapter")
    writer.add_string(Keys.Adapter.TYPE, "lora")
    writer.add_float32(Keys.Adapter.LORA_ALPHA, float(rank))
    writer.add_name("ColorLM q3_g64 Full-Rank Router Runtime Bridge")
    writer.add_description(
        "Temporary runtime bridge for exact full-rank factors of decoded q3_g64 deltas"
    )
    for name, matrix_a, matrix_b in tensors:
        writer.add_tensor(f"{name}.lora_a", matrix_a)
        writer.add_tensor(f"{name}.lora_b", matrix_b)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    report = {
        "format": "neural-alloy-q3-runtime-bridge-v1",
        "container": args.container.name,
        "base": args.base.name,
        "adapter": args.output.name,
        "tensor_count": len(tensors),
        "rank": rank,
        "adapter_bytes": args.output.stat().st_size,
        "factorization": "A=decoded_q3_delta,B=identity",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"q3运行时桥: {args.output}")
    print(f"清单: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
