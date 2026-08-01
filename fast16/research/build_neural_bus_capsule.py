"""Extract one transported expert from a fused GGUF as a Neural Bus capsule."""

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

from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402


TENSOR_NAMES = {
    "gate": "ffn_gate_exps.weight",
    "up": "ffn_up_exps.weight",
    "down": "ffn_down_exps.weight",
}


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(
        description="从已运输的GGUF中提取一个Neural Bus残差专家胶囊"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=models / "ColorLM-v8-CoderNext-Transport-E471.gguf",
    )
    parser.add_argument("--layer", type=int, default=39)
    parser.add_argument("--expert", type=int, default=201)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESEARCH / "neural_bus_capsules" / "coder_next_l47_e471_q4_0",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_bytes(data: memoryview | bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"缺少运输模型: {args.source}")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise FileExistsError(f"输出目录非空: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(os.fspath(args.source), "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    records: dict[str, dict[str, object]] = {}

    for role, suffix in TENSOR_NAMES.items():
        name = f"blk.{args.layer}.{suffix}"
        if name not in tensors:
            raise RuntimeError(f"源张量不存在: {name}")
        tensor = tensors[name]
        if tensor.tensor_type != GGMLQuantizationType.Q4_0:
            raise RuntimeError(
                f"胶囊v1只接受Q4_0，{name}实际为{tensor.tensor_type.name}"
            )
        expert_count = int(tensor.shape[2])
        if not 0 <= args.expert < expert_count:
            raise ValueError(f"专家越界: {args.expert} vs {expert_count}")

        payload = np.ascontiguousarray(tensor.data[args.expert]).view(np.uint8)
        raw = memoryview(payload).cast("B")
        output = args.output / f"{role}.q4_0"
        output.write_bytes(raw)
        records[role] = {
            "file": output.name,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "ggml_shape": (
                [2048, 512] if role in {"gate", "up"} else [512, 2048]
            ),
        }

    manifest = {
        "format": "colorlm-neural-bus-capsule-v1",
        "source": args.source.name,
        "source_layer": args.layer,
        "source_expert_slot": args.expert,
        "donor": "Qwen/Qwen3-Coder-Next layer 47 expert 471",
        "transport": "shared-token-orthogonal-procrustes, baked into weights",
        "dtype": "Q4_0",
        "input_width": 2048,
        "intermediate_width": 512,
        "output_width": 2048,
        "activation": "SwiGLU",
        "tensors": records,
    }
    manifest_path = args.output / "capsule.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total = sum(int(record["bytes"]) for record in records.values())
    print(f"胶囊已生成: {args.output}")
    print(f"权重大小: {total / (1024 * 1024):.2f} MiB")
    print(f"契约: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
