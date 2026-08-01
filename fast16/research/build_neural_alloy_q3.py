"""Stream a full-rank q3_g64 Neural Alloy delta from two compatible GGUFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))
sys.path.insert(0, os.fspath(RESEARCH))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from neural_alloy_codec import (  # noqa: E402
    BLOCK_BYTES,
    GROUP_SIZE,
    TENSOR_ALIGNMENT,
    align_up,
    encode_blocks,
    encode_manifest,
    write_header,
)


MAX_CHUNK_VALUES = 4 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(description="生成 q3_g64 全秩 Neural Alloy 差分")
    parser.add_argument(
        "--base", type=Path, default=models / "ColorLM-v5-GLM-SynapticGraft.gguf"
    )
    parser.add_argument(
        "--target", type=Path, default=models / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    )
    parser.add_argument(
        "--output", type=Path, default=models / "ColorLM-v5-to-Qwen36-q3g64.nal"
    )
    parser.add_argument("--include-regex", default=".*")
    parser.add_argument("--max-tensors", type=int, default=0)
    return parser.parse_args()


def field_value(reader: GGUFReader, name: str):
    field = reader.fields.get(name)
    return None if field is None else field.contents()


def tensor_rows(tensor, start: int, end: int) -> np.ndarray:
    logical_row = int(tensor.shape[0])
    rows = np.asarray(tensor.data).reshape(-1, tensor.data.shape[-1])[start:end]
    if tensor.data.dtype == np.float32:
        values = np.asarray(rows, dtype=np.float32)
    else:
        values = np.asarray(dequantize(rows, tensor.tensor_type), dtype=np.float32)
    return values.reshape(end - start, logical_row)


def delta_group_chunks(base_tensor, target_tensor):
    logical_row = int(base_tensor.shape[0])
    row_count = int(np.prod(base_tensor.shape[1:], dtype=np.int64))
    rows_per_chunk = max(1, MAX_CHUNK_VALUES // logical_row)
    carry = np.empty(0, dtype=np.float32)
    logical_seen = 0

    for start in range(0, row_count, rows_per_chunk):
        end = min(row_count, start + rows_per_chunk)
        base = tensor_rows(base_tensor, start, end)
        target = tensor_rows(target_tensor, start, end)
        values = (target - base).reshape(-1)
        logical_seen += values.size
        if carry.size:
            values = np.concatenate((carry, values))
        complete = values.size // GROUP_SIZE * GROUP_SIZE
        if complete:
            yield values[:complete].reshape(-1, GROUP_SIZE)
        carry = values[complete:].copy()

    if carry.size:
        padded = np.zeros(GROUP_SIZE, dtype=np.float32)
        padded[: carry.size] = carry
        yield padded.reshape(1, GROUP_SIZE)
    if logical_seen != int(np.prod(base_tensor.shape, dtype=np.int64)):
        raise RuntimeError("streamed logical value count mismatch")


def build_records(tensors: list[tuple[str, object, object]]) -> list[dict]:
    records = []
    offset = 0
    for name, base_tensor, _ in tensors:
        logical_count = int(np.prod(base_tensor.shape, dtype=np.int64))
        group_count = (logical_count + GROUP_SIZE - 1) // GROUP_SIZE
        offset = align_up(offset, TENSOR_ALIGNMENT)
        records.append(
            {
                "name": name,
                "shape": [int(x) for x in base_tensor.shape],
                "logical_count": logical_count,
                "group_count": group_count,
                "offset": offset,
                "bytes": group_count * BLOCK_BYTES,
                "base_type": int(base_tensor.tensor_type),
            }
        )
        offset += group_count * BLOCK_BYTES
    return records


def main() -> int:
    args = parse_args()
    if args.output.resolve() in (args.base.resolve(), args.target.resolve()):
        raise ValueError("输出不能覆盖模型")
    matcher = re.compile(args.include_regex)

    print(f"读取基座: {args.base}", flush=True)
    base = GGUFReader(os.fspath(args.base), "r")
    print(f"读取目标: {args.target}", flush=True)
    target = GGUFReader(os.fspath(args.target), "r")
    base_map = {tensor.name: tensor for tensor in base.tensors}
    target_map = {tensor.name: tensor for tensor in target.tensors}

    selected = []
    for tensor in base.tensors:
        if not matcher.search(tensor.name):
            continue
        other = target_map.get(tensor.name)
        if other is None or tuple(tensor.shape) != tuple(other.shape):
            raise RuntimeError(f"目标张量缺失或形状不同: {tensor.name}")
        selected.append((tensor.name, tensor, other))
        if args.max_tensors and len(selected) >= args.max_tensors:
            break
    if not selected:
        raise RuntimeError("没有选中任何兼容张量")

    records = build_records(selected)
    manifest = {
        "format": "neural-alloy-q3-g64-v1",
        "base": {
            "name": args.base.name,
            "bytes": args.base.stat().st_size,
            "architecture": field_value(base, "general.architecture"),
        },
        "target": {
            "name": args.target.name,
            "bytes": args.target.stat().st_size,
            "architecture": field_value(target, "general.architecture"),
        },
        "bits": 3,
        "group_size": GROUP_SIZE,
        "block_bytes": BLOCK_BYTES,
        "tensor_count": len(records),
        "logical_parameter_count": sum(item["logical_count"] for item in records),
        "tensors": records,
    }
    manifest_bytes = encode_manifest(manifest)
    record_map = {item["name"]: item for item in records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_hashes = {}

    with args.output.open("wb") as stream:
        data_start = write_header(stream, manifest_bytes, len(records))
        for index, (name, base_tensor, target_tensor) in enumerate(selected, start=1):
            record = record_map[name]
            wanted = data_start + record["offset"]
            if stream.tell() > wanted:
                raise RuntimeError("tensor offsets overlap")
            stream.write(b"\0" * (wanted - stream.tell()))
            digest = hashlib.sha256()
            written = 0
            for groups in delta_group_chunks(base_tensor, target_tensor):
                encoded = encode_blocks(groups)
                stream.write(encoded)
                digest.update(encoded)
                written += len(encoded)
            if written != record["bytes"]:
                raise RuntimeError(f"写入大小不一致: {name}: {written} vs {record['bytes']}")
            tensor_hashes[name] = digest.hexdigest()
            print(f"[{index}/{len(selected)}] {name} ({written / 1048576:.2f} MiB)", flush=True)

    sidecar = {
        "container": args.output.name,
        "container_bytes": args.output.stat().st_size,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "tensor_sha256": tensor_hashes,
    }
    sidecar_path = args.output.with_suffix(args.output.suffix + ".json")
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"q3差分: {args.output}")
    print(f"完整性清单: {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
