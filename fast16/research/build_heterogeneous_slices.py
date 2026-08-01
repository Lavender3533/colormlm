"""Compile GLM shared-expert slices into a single ColorLM GGUF checkpoint.

The two models use different architectures but share a 2048-wide residual
stream. A GLM shared expert is a self-contained 2048 -> 1536 -> 2048 neural
module, so it can replace the ColorLM shared expert while the ColorLM router
gate still controls its contribution for every token.
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
from gguf.constants import GGUFValueType  # noqa: E402


SLICE_SUFFIXES = (
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
)
TARGET_LAYERS = 40
DONOR_FIRST_MOE_LAYER = 1
DONOR_LAST_LAYER = 46
SHARED_EXPERT_WIDTH = 1536


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(
        description="把 GLM 共享专家神经切片编译进单一 ColorLM GGUF"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=models / "ColorLM-v4-SMoE.gguf",
    )
    parser.add_argument(
        "--donor",
        type=Path,
        default=models / "GLM-4.7-Flash-Q5_K_M.gguf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=models / "ColorLM-v5-GLM-DepthSlices.gguf",
    )
    parser.add_argument(
        "--schedule",
        choices=("depth", "bands", "alternating"),
        default="depth",
        help="GLM 层到 ColorLM 层的映射方式",
    )
    return parser.parse_args()


def scalar(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def copy_metadata(
    writer: GGUFWriter,
    reader: GGUFReader,
    overrides: dict[str, object],
) -> None:
    skip = {"general.architecture", *overrides}
    for key, field in reader.fields.items():
        # GGUFReader exposes header bookkeeping as synthetic fields. The
        # writer emits those values in the file header, so copying them as KV
        # metadata creates duplicate keys and an unreadable checkpoint.
        if key in skip or key.startswith(("GGUF.", "split.")):
            continue
        value = scalar(field.contents())
        if field.types[0] == GGUFValueType.ARRAY:
            writer.add_key_value(
                key,
                value,
                GGUFValueType.ARRAY,
                field.types[1],
            )
        else:
            writer.add_key_value(key, value, field.types[0])

    for key, value in overrides.items():
        original = reader.fields.get(key)
        if original is None:
            raise RuntimeError(f"基座缺少需要覆盖的元数据: {key}")
        writer.add_key_value(key, value, original.types[0])


def source_layer_map(schedule: str) -> list[int]:
    if schedule == "depth":
        span = DONOR_LAST_LAYER - DONOR_FIRST_MOE_LAYER
        return [
            DONOR_FIRST_MOE_LAYER + round(layer * span / (TARGET_LAYERS - 1))
            for layer in range(TARGET_LAYERS)
        ]
    if schedule == "bands":
        # Low-level GLM slices, then middle slices, then repeated high-level
        # slices. This directly represents a 10/10/20 staggered alloy.
        return [
            (1 + layer)
            if layer < 10
            else (19 + layer - 10)
            if layer < 20
            else (37 if layer % 2 == 0 else 46)
            for layer in range(TARGET_LAYERS)
        ]
    return [
        DONOR_FIRST_MOE_LAYER if layer % 2 == 0 else DONOR_LAST_LAYER
        for layer in range(TARGET_LAYERS)
    ]


def replacement_map(
    base: GGUFReader,
    donor: GGUFReader,
    schedule: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    base_tensors = {tensor.name: tensor for tensor in base.tensors}
    donor_tensors = {tensor.name: tensor for tensor in donor.tensors}
    replacements: dict[str, object] = {}
    manifest: list[dict[str, object]] = []

    for target_layer, donor_layer in enumerate(source_layer_map(schedule)):
        item = {
            "target_layer": target_layer,
            "donor_layer": donor_layer,
            "modules": [],
        }
        for suffix in SLICE_SUFFIXES:
            target_name = f"blk.{target_layer}.{suffix}"
            donor_name = f"blk.{donor_layer}.{suffix}"
            if target_name not in base_tensors:
                raise RuntimeError(f"基座张量不存在: {target_name}")
            if donor_name not in donor_tensors:
                raise RuntimeError(f"GLM 切片不存在: {donor_name}")
            replacements[target_name] = donor_tensors[donor_name]
            item["modules"].append(
                {
                    "target": target_name,
                    "source": donor_name,
                    "shape": [int(x) for x in donor_tensors[donor_name].shape],
                    "type": int(donor_tensors[donor_name].tensor_type),
                }
            )
        manifest.append(item)
    return replacements, manifest


def add_tensor_info(writer: GGUFWriter, name: str, tensor) -> None:
    data = tensor.data
    writer.add_tensor_info(
        name,
        data.shape,
        data.dtype,
        data.nbytes,
        raw_dtype=tensor.tensor_type,
    )


def main() -> int:
    args = parse_args()
    if args.output.resolve() in (args.base.resolve(), args.donor.resolve()):
        raise ValueError("输出不能覆盖供体模型")

    print(f"读取主干: {args.base}", flush=True)
    base = GGUFReader(os.fspath(args.base), "r")
    print(f"读取 GLM: {args.donor}", flush=True)
    donor = GGUFReader(os.fspath(args.donor), "r")

    base_arch = base.fields["general.architecture"].contents()
    donor_arch = donor.fields["general.architecture"].contents()
    base_width = base.fields[f"{base_arch}.embedding_length"].contents()
    donor_width = donor.fields[f"{donor_arch}.embedding_length"].contents()
    if base_width != donor_width:
        raise RuntimeError(
            f"v1 直接切片要求残差宽度相同: {base_width} vs {donor_width}"
        )

    replacements, layers = replacement_map(base, donor, args.schedule)
    overrides = {
        "general.name": f"ColorLM-v5-GLM-{args.schedule}-slices",
        f"{base_arch}.expert_shared_feed_forward_length": SHARED_EXPERT_WIDTH,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(args.output, arch=str(base_arch))
    copy_metadata(writer, base, overrides)

    selected = []
    original_bytes = 0
    replacement_bytes = 0
    for tensor in base.tensors:
        source = replacements.get(tensor.name, tensor)
        selected.append((tensor.name, source))
        add_tensor_info(writer, tensor.name, source)
        if source is tensor:
            original_bytes += source.data.nbytes
        else:
            replacement_bytes += source.data.nbytes

    print(
        f"写入 {len(selected)} 个张量，其中 {len(replacements)} 个来自 GLM...",
        flush=True,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for index, (_, source) in enumerate(selected, start=1):
        writer.write_tensor_data(source.data)
        if index % 40 == 0 or index == len(selected):
            print(f"[{index}/{len(selected)}]", flush=True)
    writer.close()

    report = {
        "format": "neural-slice-abi-v1",
        "model": args.output.name,
        "base": args.base.name,
        "donor": args.donor.name,
        "base_architecture": base_arch,
        "donor_architecture": donor_arch,
        "residual_width": int(base_width),
        "schedule": args.schedule,
        "slice_type": "gated_shared_expert",
        "replacement_tensor_count": len(replacements),
        "copied_base_tensor_bytes": original_bytes,
        "copied_donor_tensor_bytes": replacement_bytes,
        "output_bytes": args.output.stat().st_size,
        "layers": layers,
    }
    report_path = args.output.with_suffix(".slices.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"异构模型: {args.output}")
    print(f"切片清单: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
