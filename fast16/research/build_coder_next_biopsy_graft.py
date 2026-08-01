"""Compile a remotely biopsied Qwen3-Coder-Next expert into ColorLM."""

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
from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402

from build_heterogeneous_slices import copy_metadata  # noqa: E402


GATE_NAME = "ffn_gate_exps.weight"
UP_NAME = "ffn_up_exps.weight"
DOWN_NAME = "ffn_down_exps.weight"
ROUTER_NAME = "ffn_gate_inp.weight"
GRAFT_QUANT = GGMLQuantizationType.Q4_0
Q4_0_BLOCK_SIZE = 32
Q4_0_BLOCK_BYTES = 18


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    extracted = (
        RESEARCH
        / "biopsy_cache"
        / "Qwen_Qwen3-Coder-Next"
        / "master"
        / "extracted"
    )
    parser = argparse.ArgumentParser(
        description="把Qwen3-Coder-Next远程活检专家编译进ColorLM"
    )
    parser.add_argument(
        "--base", type=Path, default=models / "ColorLM-v6-Q3Router-Fused-A1.gguf"
    )
    parser.add_argument(
        "--output", type=Path, default=models / "ColorLM-v7-CoderNext-Biopsy-E1.gguf"
    )
    parser.add_argument("--target-layer", type=int, default=39)
    parser.add_argument("--target-expert", type=int, default=255)
    parser.add_argument("--donor-layer", type=int, default=47)
    parser.add_argument("--donor-expert", type=int, default=0)
    parser.add_argument(
        "--gate",
        type=Path,
        default=extracted / "model.layers.47.mlp.experts.0.gate_proj.weight.bin",
    )
    parser.add_argument(
        "--up",
        type=Path,
        default=extracted / "model.layers.47.mlp.experts.0.up_proj.weight.bin",
    )
    parser.add_argument(
        "--down",
        type=Path,
        default=extracted / "model.layers.47.mlp.experts.0.down_proj.weight.bin",
    )
    parser.add_argument(
        "--router-row",
        type=Path,
        default=extracted / "model.layers.47.mlp.gate.weight.axis0-0-1.bin",
    )
    parser.add_argument(
        "--transport",
        type=Path,
        help="可选的Coder-Next到ColorLM 2048维正交运输矩阵(.npy)",
    )
    parser.add_argument("--model-name", help="写入GGUF的general.name")
    return parser.parse_args()


def bf16_matrix(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape, dtype=np.int64)) * 2
    if path.stat().st_size != expected:
        raise RuntimeError(f"BF16文件大小不匹配: {path}")
    bits = np.fromfile(path, dtype="<u2").astype(np.uint32) << 16
    return np.ascontiguousarray(bits.view(np.float32).reshape(shape))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def importance_energy(gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> float:
    gate_rms = np.sqrt(np.mean(np.square(gate), axis=1, dtype=np.float64))
    up_rms = np.sqrt(np.mean(np.square(up), axis=1, dtype=np.float64))
    down_rms = np.sqrt(np.mean(np.square(down), axis=0, dtype=np.float64))
    score = gate_rms * up_rms * down_rms
    return float(np.linalg.norm(score))


def add_source_tensor_info(writer: GGUFWriter, tensor) -> None:
    writer.add_tensor_info(
        tensor.name,
        tensor.data.shape,
        tensor.data.dtype,
        tensor.data.nbytes,
        raw_dtype=tensor.tensor_type,
    )


def q4_0_byte_shape(tensor) -> tuple[int, ...]:
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    if logical_shape[-1] % Q4_0_BLOCK_SIZE:
        raise RuntimeError(f"Q4_0行长度不对齐: {tensor.name}: {logical_shape}")
    return (
        *logical_shape[:-1],
        logical_shape[-1] // Q4_0_BLOCK_SIZE * Q4_0_BLOCK_BYTES,
    )


def main() -> int:
    args = parse_args()
    sources = (args.gate, args.up, args.down, args.router_row)
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"缺少活检切片: {path}")
    if args.output.exists():
        raise FileExistsError(f"输出已存在: {args.output}")
    if args.output.resolve() == args.base.resolve():
        raise ValueError("输出不能覆盖基座")

    donor_gate = bf16_matrix(args.gate, (512, 2048))
    donor_up = bf16_matrix(args.up, (512, 2048))
    donor_down = bf16_matrix(args.down, (2048, 512))
    if args.router_row.stat().st_size == 512 * 2048 * 2:
        donor_router = bf16_matrix(args.router_row, (512, 2048))[args.donor_expert]
    else:
        donor_router = bf16_matrix(args.router_row, (1, 2048))[0]

    transport_sha256 = None
    if args.transport is not None:
        if not args.transport.is_file():
            raise FileNotFoundError(f"缺少坐标运输矩阵: {args.transport}")
        transport = np.asarray(np.load(args.transport, allow_pickle=False), dtype=np.float32)
        if transport.shape != (2048, 2048):
            raise RuntimeError(f"坐标运输矩阵形状不匹配: {transport.shape}")
        donor_gate = np.ascontiguousarray(donor_gate @ transport)
        donor_up = np.ascontiguousarray(donor_up @ transport)
        donor_down = np.ascontiguousarray(transport.T @ donor_down)
        donor_router = np.ascontiguousarray(donor_router @ transport)
        transport_sha256 = sha256_file(args.transport)

    print(f"读取基座: {args.base}", flush=True)
    base = GGUFReader(os.fspath(args.base), "r")
    tensors = {tensor.name: tensor for tensor in base.tensors}
    prefix = f"blk.{args.target_layer}."
    names = {
        "gate": prefix + GATE_NAME,
        "up": prefix + UP_NAME,
        "down": prefix + DOWN_NAME,
        "router": prefix + ROUTER_NAME,
    }
    for name in names.values():
        if name not in tensors:
            raise RuntimeError(f"基座张量不存在: {name}")

    base_gate_tensor = tensors[names["gate"]]
    base_up_tensor = tensors[names["up"]]
    base_down_tensor = tensors[names["down"]]
    base_router_tensor = tensors[names["router"]]
    expert_count = int(base_gate_tensor.shape[2])
    if not 0 <= args.target_expert < expert_count:
        raise ValueError(f"目标专家越界: {args.target_expert} vs {expert_count}")

    slot = slice(args.target_expert, args.target_expert + 1)
    base_gate = np.asarray(
        dequantize(base_gate_tensor.data[slot], base_gate_tensor.tensor_type),
        dtype=np.float32,
    )[0]
    base_up = np.asarray(
        dequantize(base_up_tensor.data[slot], base_up_tensor.tensor_type),
        dtype=np.float32,
    )[0]
    base_down = np.asarray(
        dequantize(base_down_tensor.data[slot], base_down_tensor.tensor_type),
        dtype=np.float32,
    )[0]

    base_energy = importance_energy(base_gate, base_up, base_down)
    donor_energy = importance_energy(donor_gate, donor_up, donor_down)
    beta = float(np.clip(base_energy / max(donor_energy, 1e-12), 0.25, 4.0))
    donor_down = np.ascontiguousarray(donor_down * np.float32(beta))

    base_router = np.asarray(base_router_tensor.data, dtype=np.float32)
    row_rms = np.sqrt(np.mean(np.square(base_router), axis=1, dtype=np.float64))
    target_router_rms = float(np.median(row_rms))
    donor_router_rms = float(np.sqrt(np.mean(np.square(donor_router), dtype=np.float64)))
    router_scale = float(
        np.clip(target_router_rms / max(donor_router_rms, 1e-12), 0.25, 4.0)
    )
    donor_router = np.ascontiguousarray(donor_router * np.float32(router_scale))

    donor_by_name = {
        names["gate"]: donor_gate,
        names["up"]: donor_up,
        names["down"]: donor_down,
    }

    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"临时文件已存在: {partial}")
    architecture = str(base.fields["general.architecture"].contents())
    writer = GGUFWriter(partial, arch=architecture)
    model_name = args.model_name or args.output.stem
    copy_metadata(writer, base, {"general.name": model_name})
    writer.add_string("colorlm.biopsy.repo", "Qwen/Qwen3-Coder-Next")
    writer.add_uint32("colorlm.biopsy.donor_layer", args.donor_layer)
    writer.add_uint32("colorlm.biopsy.donor_expert", args.donor_expert)
    writer.add_uint32("colorlm.biopsy.target_layer", args.target_layer)
    writer.add_uint32("colorlm.biopsy.target_expert", args.target_expert)
    writer.add_float32("colorlm.biopsy.expert_beta", beta)
    writer.add_float32("colorlm.biopsy.router_scale", router_scale)
    if args.transport is not None:
        writer.add_string("colorlm.biopsy.transport", "shared-token-orthogonal-procrustes")
        writer.add_string("colorlm.biopsy.transport_sha256", transport_sha256)

    for tensor in base.tensors:
        if tensor.name not in donor_by_name:
            add_source_tensor_info(writer, tensor)
            continue
        byte_shape = q4_0_byte_shape(tensor)
        writer.add_tensor_info(
            tensor.name,
            byte_shape,
            np.dtype(np.uint8),
            int(np.prod(byte_shape, dtype=np.int64)),
            raw_dtype=GRAFT_QUANT,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    patch_names = set(donor_by_name) | {names["router"]}
    for tensor in base.tensors:
        if tensor.name not in patch_names:
            writer.write_tensor_data(tensor.data)
            continue
        if tensor.name == names["router"]:
            patched = np.array(tensor.data, copy=True)
            patched[args.target_expert] = donor_router
            output_data = np.ascontiguousarray(patched)
        else:
            # IQ3_S/IQ3_XXS have no Python encoder. Re-encode only this
            # layer's expert bank as Q4_0, which remains Vulkan-native and
            # increases the checkpoint by roughly 0.16 GiB.
            decoded = np.asarray(
                dequantize(tensor.data, tensor.tensor_type), dtype=np.float32
            )
            decoded[args.target_expert] = donor_by_name[tensor.name]
            output_data = quantize(decoded, GRAFT_QUANT)
            expected_shape = q4_0_byte_shape(tensor)
            if output_data.shape != expected_shape:
                raise RuntimeError(
                    f"Q4_0形状不匹配: {tensor.name}: "
                    f"{output_data.shape} vs {expected_shape}"
                )
        writer.write_tensor_data(output_data)
        print(f"已嫁接: {tensor.name}[专家{args.target_expert}]", flush=True)

    writer.close()
    partial.replace(args.output)
    report = {
        "format": (
            "colorlm-coder-next-transport-graft-v1"
            if args.transport is not None
            else "colorlm-coder-next-biopsy-graft-v1"
        ),
        "base": args.base.name,
        "output": args.output.name,
        "donor": {
            "repo": "Qwen/Qwen3-Coder-Next",
            "layer": args.donor_layer,
            "expert": args.donor_expert,
            "source_sha256": {path.name: sha256_file(path) for path in sources},
        },
        "target": {"layer": args.target_layer, "expert": args.target_expert},
        "expert_beta": beta,
        "router_scale": router_scale,
        "transport": (
            {
                "method": "shared-token-orthogonal-procrustes",
                "file": args.transport.name,
                "sha256": transport_sha256,
            }
            if args.transport is not None
            else None
        ),
        "grafted_layer_quantization": "Q4_0",
        "base_energy": base_energy,
        "donor_energy_before_scale": donor_energy,
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"输出模型: {args.output}")
    print(f"报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
