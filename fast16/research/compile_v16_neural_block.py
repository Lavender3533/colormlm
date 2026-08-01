"""Compile extracted BF16 source ranges into a contiguous Q4_0 Neural Block."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf.quants import quantize  # noqa: E402


DEFAULT_PLAN = RESEARCH / "v16_coder_neural_block_plan.json"
DEFAULT_SOURCE = RESEARCH / "neural_blocks" / "qwen3_coder_next_l47" / "source"
DEFAULT_OUTPUT = RESEARCH / "neural_blocks" / "qwen3_coder_next_l47" / "q4_0"
DEFAULT_TRANSPORT = RESEARCH / "coder_next_to_colorlm_orthogonal_f32.npy"
ALIGNMENT = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把v16完整神经块的BF16 Range编译为连续Q4_0运行包"
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transport", type=Path, default=DEFAULT_TRANSPORT)
    parser.add_argument(
        "--max-groups",
        type=int,
        help="仅用于编译器烟测；正式组数由供体层类型决定",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def align(value: int, boundary: int = ALIGNMENT) -> int:
    return (value + boundary - 1) // boundary * boundary


def bf16_to_f32(raw: np.ndarray, shape: list[int]) -> np.ndarray:
    bits = np.asarray(raw, dtype="<u2").astype(np.uint32) << 16
    return np.ascontiguousarray(bits.view(np.float32).reshape(shape))


class SegmentReader:
    def __init__(self, source: Path, plan: dict) -> None:
        self.maps: dict[str, np.memmap] = {}
        self.sizes: dict[str, int] = {}
        for segment in plan["source_ranges"]:
            path = source / segment["file"]
            expected = int(segment["bytes"])
            if not path.is_file() or path.stat().st_size != expected:
                raise RuntimeError(f"缺少完整源Range: {path}")
            self.maps[segment["file"]] = np.memmap(path, mode="r", dtype=np.uint8)
            self.sizes[segment["file"]] = expected

    def tensor(self, record: dict) -> np.ndarray:
        segment_file = record["segment_file"]
        offset = int(record["segment_offset"])
        size = int(record["bytes"])
        if offset < 0 or offset + size > self.sizes[segment_file]:
            raise RuntimeError(f"张量越过Range边界: {record['name']}")
        return self.maps[segment_file][offset : offset + size]


def encode_tensor(reader: SegmentReader, record: dict) -> tuple[str, bytes, list[int]]:
    if record["dtype"] != "BF16":
        raise RuntimeError(f"首版编译器只接受BF16: {record['name']}")
    shape = [int(value) for value in record["shape"]]
    raw = reader.tensor(record)
    if math.prod(shape) * 2 != raw.size:
        raise RuntimeError(f"BF16源尺寸不匹配: {record['name']}")
    source = bf16_to_f32(raw.view("<u2"), shape)

    # Keep the runtime tensors mathematically identical to llama.cpp's
    # Qwen3Next GGUF conversion.  The Range extractor deliberately stores the
    # untouched HF tensors, so these transformations belong here rather than
    # in the downloader.
    name = record["name"]
    if name.endswith(".A_log"):
        source = -np.exp(source)
    elif ".conv1d." in name:
        source = np.squeeze(source)
    elif name.endswith("norm.weight") and not name.endswith(
        "linear_attn.norm.weight"
    ):
        # Qwen3Next uses residual RMSNorm parameters in HF, while ggml's RMS
        # norm operator consumes the full multiplicative weight.
        source = source + np.float32(1.0)

    source = np.ascontiguousarray(source)
    runtime_shape = list(source.shape)
    if len(runtime_shape) == 1 or ".conv1d." in name:
        encoded = source.astype("<f4", copy=False).tobytes(order="C")
        return "F32", encoded, list(reversed(runtime_shape))
    if runtime_shape[-1] % 32:
        encoded = source.astype("<f2", copy=False).tobytes(order="C")
        return "F16", encoded, list(reversed(runtime_shape))
    encoded_array = quantize(source, GGMLQuantizationType.Q4_0)
    expected_bytes = (
        math.prod(runtime_shape[:-1]) * (runtime_shape[-1] // 32) * 18
    )
    if encoded_array.nbytes != expected_bytes:
        raise RuntimeError(
            f"Q4_0尺寸异常: {record['name']}: {encoded_array.nbytes} vs {expected_bytes}"
        )
    return (
        "Q4_0",
        encoded_array.tobytes(order="C"),
        list(reversed(runtime_shape)),
    )


def runtime_groups(plan: dict) -> list[dict]:
    source = plan["tensors"]
    source_layer = int(plan["source"]["layer"])
    groups = [
        {
            "name": record["name"],
            "role": record["role"],
            "records": [record],
            "source_shape": record["shape"],
            "ggml_shape": list(reversed(record["shape"])),
        }
        for record in source
        if record["role"] != "routed_expert"
    ]
    expected_experts = int(plan["moe"]["expert_count"])
    pattern = re.compile(r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    by_projection: dict[str, dict[int, dict]] = {
        "gate_proj": {},
        "up_proj": {},
        "down_proj": {},
    }
    for record in source:
        if record["role"] != "routed_expert":
            continue
        match = pattern.search(record["name"])
        if not match:
            raise RuntimeError(f"无法分组的专家张量: {record['name']}")
        expert = int(match.group(1))
        projection = match.group(2)
        if expert in by_projection[projection]:
            raise RuntimeError(f"重复专家张量: {record['name']}")
        by_projection[projection][expert] = record
    for projection in ("gate_proj", "up_proj", "down_proj"):
        records = by_projection[projection]
        if sorted(records) != list(range(expected_experts)):
            raise RuntimeError(f"{projection}专家银行不完整")
        ordered = [records[index] for index in range(expected_experts)]
        shape = ordered[0]["shape"]
        if any(record["shape"] != shape for record in ordered):
            raise RuntimeError(f"{projection}专家形状不一致")
        groups.append(
            {
                "name": f"model.layers.{source_layer}.mlp.experts.{projection}.weight",
                "role": "routed_expert_bank",
                "records": ordered,
                "source_shape": [expected_experts, *shape],
                "ggml_shape": [*reversed(shape), expected_experts],
            }
        )
    groups.extend(
        [
            {
                "name": "colorlm.neural_block.transport_in.weight",
                "role": "input_transport",
                "records": [],
                "source_shape": [2048, 2048],
                "ggml_shape": [2048, 2048],
                "transport": "input",
            },
            {
                "name": "colorlm.neural_block.transport_out.weight",
                "role": "output_transport",
                "records": [],
                "source_shape": [2048, 2048],
                "ggml_shape": [2048, 2048],
                "transport": "output",
            },
        ]
    )
    expected_groups = sum(record["role"] != "routed_expert" for record in source) + 5
    if len(groups) != expected_groups:
        raise RuntimeError(
            f"运行张量组数量异常: {len(groups)} vs {expected_groups}"
        )
    return groups


def load_checkpoint(path: Path, plan_sha256: str, tensor_count: int) -> dict:
    if not path.is_file():
        return {
            "format": "colorlm-neural-block-compile-checkpoint-v1",
            "plan_sha256": plan_sha256,
            "tensor_count": tensor_count,
            "completed": [],
            "output_bytes": 0,
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if (
        checkpoint.get("format") != "colorlm-neural-block-compile-checkpoint-v1"
        or checkpoint.get("plan_sha256") != plan_sha256
        or int(checkpoint.get("tensor_count", -1)) != tensor_count
    ):
        raise RuntimeError("编译断点与当前ABI计划不匹配")
    return checkpoint


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("format") != "colorlm-neural-block-abi-v1":
        raise RuntimeError("不支持的Neural Block ABI")
    source_receipt = args.source / "source.json"
    if not source_receipt.is_file():
        raise RuntimeError("源Range尚未完成，缺少source.json")
    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    plan_sha256 = sha256_file(args.plan)
    if receipt.get("plan_sha256") != plan_sha256:
        raise RuntimeError("源Range收据与当前ABI计划不一致")

    groups = runtime_groups(plan)
    expected_transport = plan["interface"]["transport"]
    if (
        not args.transport.is_file()
        or sha256_file(args.transport) != expected_transport["source_to_target_sha256"]
    ):
        raise RuntimeError("坐标桥与ABI计划不一致")
    transport = np.asarray(np.load(args.transport, allow_pickle=False), dtype=np.float32)
    if transport.shape != (2048, 2048) or not np.all(np.isfinite(transport)):
        raise RuntimeError("坐标桥必须是有限值2048x2048矩阵")
    formal = args.max_groups is None
    if args.max_groups is not None:
        if args.max_groups <= 0:
            raise ValueError("max-groups必须为正数")
        groups = groups[: args.max_groups]
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = "" if formal else f".smoke-{len(groups)}"
    weights = args.output / f"weights{suffix}.bin"
    partial = weights.with_suffix(weights.suffix + ".partial")
    checkpoint_path = args.output / f"compile{suffix}.checkpoint.json"
    manifest_path = args.output / f"block{suffix}.json"
    if weights.exists() or manifest_path.exists():
        raise FileExistsError(f"目标运行包已存在: {weights} / {manifest_path}")

    checkpoint = load_checkpoint(checkpoint_path, plan_sha256, len(groups))
    completed: list[dict] = checkpoint["completed"]
    if len(completed) > len(groups):
        raise RuntimeError("编译断点张量组数量越界")
    expected_partial = int(checkpoint["output_bytes"])
    actual_partial = partial.stat().st_size if partial.is_file() else 0
    if actual_partial != expected_partial:
        raise RuntimeError(
            f"编译断点文件长度不匹配: {actual_partial} vs {expected_partial}"
        )
    for index, record in enumerate(completed):
        if record["name"] != groups[index]["name"]:
            raise RuntimeError("编译断点张量组顺序漂移")

    reader = SegmentReader(args.source, plan)
    with partial.open("ab") as stream:
        for index in range(len(completed), len(groups)):
            group = groups[index]
            offset = align(stream.tell())
            if offset > stream.tell():
                stream.write(b"\0" * (offset - stream.tell()))
            digest = hashlib.sha256()
            group_bytes = 0
            dtype = None
            single_runtime_shape = None
            if group.get("transport"):
                matrix = transport if group["transport"] == "input" else transport.T
                encoded = np.ascontiguousarray(matrix, dtype="<f2").tobytes(order="C")
                dtype = "F16"
                stream.write(encoded)
                digest.update(encoded)
                group_bytes += len(encoded)
            else:
                for source_record in group["records"]:
                    item_dtype, encoded, item_shape = encode_tensor(
                        reader, source_record
                    )
                    if len(group["records"]) == 1:
                        single_runtime_shape = item_shape
                    if dtype is None:
                        dtype = item_dtype
                    elif dtype != item_dtype:
                        raise RuntimeError(f"张量组dtype不一致: {group['name']}")
                    stream.write(encoded)
                    digest.update(encoded)
                    group_bytes += len(encoded)
            compiled = {
                "name": group["name"],
                "role": group["role"],
                "dtype": dtype,
                "source_shape": group["source_shape"],
                "ggml_shape": single_runtime_shape or group["ggml_shape"],
                "offset": offset,
                "bytes": group_bytes,
                "sha256": digest.hexdigest(),
            }
            completed.append(compiled)
            checkpoint["output_bytes"] = stream.tell()
            stream.flush()
            checkpoint["completed"] = completed
            save_checkpoint(checkpoint_path, checkpoint)
            print(
                f"{len(completed)}/{len(groups)}  "
                f"{stream.tell() / 1024**2:.1f} MiB  {group['name']}",
                flush=True,
            )

    partial.replace(weights)
    weights_sha256 = sha256_file(weights)
    manifest = {
        "format": "colorlm-neural-block-runtime-v1",
        "name": plan["name"],
        "formal": formal,
        "source_abi": os.fspath(args.plan.resolve()),
        "source_abi_sha256": plan_sha256,
        "runtime_layout": "single-aligned-tensor-blob-v1",
        "alignment": ALIGNMENT,
        "weights": {
            "file": weights.name,
            "bytes": weights.stat().st_size,
            "sha256": weights_sha256,
        },
        "source": plan["source"],
        "interface": plan["interface"],
        "attention": plan["attention"],
        "moe": plan["moe"],
        "target": plan["target"],
        "transport": expected_transport,
        "tensor_count": len(completed),
        "tensors": completed,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint_path.unlink()
    print(f"运行包: {manifest_path}")
    print(f"权重: {weights} ({weights.stat().st_size / 1024**3:.3f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
