"""Graft high-importance GLM neurons into ColorLM shared experts.

The graft keeps the original 512-neuron shared-expert width, so inference
compute does not grow. Low-importance ColorLM neurons are replaced with a
small, energy-normalized set of GLM neurons selected from compatible 2048-wide
shared experts.
"""

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
from gguf.quants import dequantize  # noqa: E402

from build_heterogeneous_slices import (  # noqa: E402
    SLICE_SUFFIXES,
    copy_metadata,
    source_layer_map,
)


TARGET_WIDTH = 512


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(
        description="按神经元重要度把 GLM 微切片嫁接进 ColorLM"
    )
    parser.add_argument(
        "--base", type=Path, default=models / "ColorLM-v4-SMoE.gguf"
    )
    parser.add_argument(
        "--donor", type=Path, default=models / "GLM-4.7-Flash-Q5_K_M.gguf"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=models / "ColorLM-v5-GLM-SynapticGraft.gguf",
    )
    parser.add_argument(
        "--schedule",
        choices=("depth", "bands", "alternating"),
        default="depth",
    )
    parser.add_argument(
        "--graft-neurons",
        type=int,
        default=32,
        help="每层512个槽位中由 GLM 提供的神经元数量",
    )
    parser.add_argument(
        "--target-energy",
        type=float,
        default=0.03,
        help="GLM嫁接支路相对保留ColorLM支路的目标能量",
    )
    return parser.parse_args()


def tensor_f32(tensor) -> np.ndarray:
    if tensor.data.dtype == np.float32:
        return np.asarray(tensor.data, dtype=np.float32)
    return np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)


def importance(gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    gate_rms = np.sqrt(np.mean(np.square(gate), axis=1, dtype=np.float64))
    up_rms = np.sqrt(np.mean(np.square(up), axis=1, dtype=np.float64))
    down_rms = np.sqrt(np.mean(np.square(down), axis=0, dtype=np.float64))
    return gate_rms * up_rms * down_rms


def top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    indices = np.argpartition(scores, -count)[-count:]
    return np.sort(indices.astype(np.int64))


def layer_graft(
    base_tensors: dict[str, object],
    donor_tensors: dict[str, object],
    target_layer: int,
    donor_layer: int,
    graft_neurons: int,
    target_energy: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    prefix = f"blk.{target_layer}."
    donor_prefix = f"blk.{donor_layer}."

    color_gate = tensor_f32(base_tensors[prefix + "ffn_gate_shexp.weight"])
    color_up = tensor_f32(base_tensors[prefix + "ffn_up_shexp.weight"])
    color_down = tensor_f32(base_tensors[prefix + "ffn_down_shexp.weight"])
    glm_gate = tensor_f32(donor_tensors[donor_prefix + "ffn_gate_shexp.weight"])
    glm_up = tensor_f32(donor_tensors[donor_prefix + "ffn_up_shexp.weight"])
    glm_down = tensor_f32(donor_tensors[donor_prefix + "ffn_down_shexp.weight"])

    keep_color = TARGET_WIDTH - graft_neurons
    color_score = importance(color_gate, color_up, color_down)
    glm_score = importance(glm_gate, glm_up, glm_down)
    color_indices = top_indices(color_score, keep_color)
    glm_indices = top_indices(glm_score, graft_neurons)

    color_energy = float(np.linalg.norm(color_score[color_indices]))
    glm_energy = float(np.linalg.norm(glm_score[glm_indices]))
    beta = target_energy * color_energy / max(glm_energy, 1e-12)
    beta = float(np.clip(beta, 1e-5, 0.25))

    gate = np.concatenate(
        (color_gate[color_indices], glm_gate[glm_indices]), axis=0
    ).astype(np.float16)
    up = np.concatenate(
        (color_up[color_indices], glm_up[glm_indices]), axis=0
    ).astype(np.float16)
    down = np.concatenate(
        (color_down[:, color_indices], beta * glm_down[:, glm_indices]), axis=1
    ).astype(np.float16)

    tensors = {
        prefix + "ffn_gate_shexp.weight": np.ascontiguousarray(gate),
        prefix + "ffn_up_shexp.weight": np.ascontiguousarray(up),
        prefix + "ffn_down_shexp.weight": np.ascontiguousarray(down),
    }
    report = {
        "target_layer": target_layer,
        "donor_layer": donor_layer,
        "kept_color_neurons": keep_color,
        "grafted_glm_neurons": graft_neurons,
        "target_energy": target_energy,
        "beta": beta,
        "color_importance_retained": float(
            np.sum(color_score[color_indices]) / max(np.sum(color_score), 1e-12)
        ),
        "glm_indices": glm_indices.tolist(),
    }
    return tensors, report


def add_source_info(writer: GGUFWriter, name: str, tensor) -> None:
    writer.add_tensor_info(
        name,
        tensor.data.shape,
        tensor.data.dtype,
        tensor.data.nbytes,
        raw_dtype=tensor.tensor_type,
    )


def main() -> int:
    args = parse_args()
    if not 1 <= args.graft_neurons < TARGET_WIDTH:
        raise ValueError("graft-neurons 必须在 1 到 511 之间")
    if not 0.0 < args.target_energy <= 1.0:
        raise ValueError("target-energy 必须在 0 到 1 之间")
    if args.output.resolve() in (args.base.resolve(), args.donor.resolve()):
        raise ValueError("输出不能覆盖供体模型")

    print(f"读取主干: {args.base}", flush=True)
    base = GGUFReader(os.fspath(args.base), "r")
    print(f"读取 GLM: {args.donor}", flush=True)
    donor = GGUFReader(os.fspath(args.donor), "r")
    base_tensors = {tensor.name: tensor for tensor in base.tensors}
    donor_tensors = {tensor.name: tensor for tensor in donor.tensors}
    donor_layers = source_layer_map(args.schedule)

    base_arch = str(base.fields["general.architecture"].contents())
    donor_arch = str(donor.fields["general.architecture"].contents())
    base_width = int(base.fields[f"{base_arch}.embedding_length"].contents())
    donor_width = int(donor.fields[f"{donor_arch}.embedding_length"].contents())
    if base_width != donor_width:
        raise RuntimeError(
            f"当前嫁接器要求残差宽度相同: {base_width} vs {donor_width}"
        )

    replacement_names = {
        f"blk.{layer}.{suffix}"
        for layer in range(len(donor_layers))
        for suffix in SLICE_SUFFIXES
    }
    output_name = (
        f"ColorLM-v5-GLM-graft{args.graft_neurons}-"
        f"e{args.target_energy:g}-{args.schedule}"
    )
    writer = GGUFWriter(args.output, arch=base_arch)
    copy_metadata(writer, base, {"general.name": output_name})

    selected = []
    for tensor in base.tensors:
        selected.append(tensor)
        if tensor.name in replacement_names:
            # GGUF logical dimensions are the reverse of NumPy's row-major
            # matrix layout. Gate/up are [512, 2048] arrays while down is
            # [2048, 512], even though all contain the same value count.
            f16_shape = tuple(int(x) for x in reversed(tensor.shape))
            f16_bytes = int(np.prod(f16_shape)) * np.dtype(np.float16).itemsize
            writer.add_tensor_info(
                tensor.name,
                f16_shape,
                np.dtype(np.float16),
                f16_bytes,
            )
        else:
            add_source_info(writer, tensor.name, tensor)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    cache: dict[str, np.ndarray] = {}
    layer_reports: list[dict[str, object]] = []
    built_layers: set[int] = set()
    for index, tensor in enumerate(selected, start=1):
        if tensor.name not in replacement_names:
            writer.write_tensor_data(tensor.data)
        else:
            target_layer = int(tensor.name.split(".")[1])
            if target_layer not in built_layers:
                grafts, report = layer_graft(
                    base_tensors,
                    donor_tensors,
                    target_layer,
                    donor_layers[target_layer],
                    args.graft_neurons,
                    args.target_energy,
                )
                cache.update(grafts)
                layer_reports.append(report)
                built_layers.add(target_layer)
                print(
                    f"层 {target_layer:02d} <- GLM {donor_layers[target_layer]:02d}, "
                    f"beta={report['beta']:.6f}",
                    flush=True,
                )
            writer.write_tensor_data(cache.pop(tensor.name))
        if index % 80 == 0:
            print(f"[{index}/{len(selected)}]", flush=True)
    writer.close()

    report = {
        "format": "neural-slice-abi-v1-synaptic-graft",
        "model": args.output.name,
        "base": args.base.name,
        "donor": args.donor.name,
        "schedule": args.schedule,
        "residual_width": base_width,
        "shared_expert_width": TARGET_WIDTH,
        "graft_neurons_per_layer": args.graft_neurons,
        "target_energy": args.target_energy,
        "replacement_tensor_count": len(replacement_names),
        "output_bytes": args.output.stat().st_size,
        "layers": sorted(layer_reports, key=lambda item: item["target_layer"]),
    }
    report_path = args.output.with_suffix(".graft.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"嫁接模型: {args.output}")
    print(f"嫁接清单: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
