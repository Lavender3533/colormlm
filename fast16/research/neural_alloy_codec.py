"""Codec and container primitives for Neural Alloy q3_g64 deltas."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


MAGIC = b"CLMALY01"
VERSION = 1
BITS = 3
GROUP_SIZE = 64
PACKED_BYTES = 24
BLOCK_BYTES = 2 + PACKED_BYTES
DATA_ALIGNMENT = 4096
TENSOR_ALIGNMENT = 64
HEADER = struct.Struct("<8sIIIIQQ32s")


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def pack_q3(codes: np.ndarray) -> np.ndarray:
    """Pack uint3 codes with shape [groups, 64] into [groups, 24]."""
    values = np.asarray(codes, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != GROUP_SIZE:
        raise ValueError(f"q3 codes must have shape [N, {GROUP_SIZE}]")
    if values.size and int(values.max()) > 7:
        raise ValueError("q3 code exceeds three bits")

    chunks = values.reshape(-1, 8, 8).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * BITS).reshape(1, 1, 8)
    words = np.bitwise_or.reduce(chunks << shifts, axis=2)
    packed = np.empty((*words.shape, 3), dtype=np.uint8)
    packed[..., 0] = words & 0xFF
    packed[..., 1] = (words >> 8) & 0xFF
    packed[..., 2] = (words >> 16) & 0xFF
    return np.ascontiguousarray(packed.reshape(-1, PACKED_BYTES))


def unpack_q3(packed: np.ndarray) -> np.ndarray:
    """Unpack [groups, 24] uint3 bytes into [groups, 64] codes."""
    data = np.asarray(packed, dtype=np.uint8)
    if data.ndim != 2 or data.shape[1] != PACKED_BYTES:
        raise ValueError(f"packed q3 data must have shape [N, {PACKED_BYTES}]")
    triples = data.reshape(-1, 8, 3).astype(np.uint32)
    words = triples[..., 0] | (triples[..., 1] << 8) | (triples[..., 2] << 16)
    shifts = (np.arange(8, dtype=np.uint32) * BITS).reshape(1, 1, 8)
    return np.ascontiguousarray(((words[..., None] >> shifts) & 7).reshape(-1, 64).astype(np.uint8))


def quantize_groups(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(groups, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != GROUP_SIZE:
        raise ValueError(f"values must have shape [N, {GROUP_SIZE}]")
    absmax = np.max(np.abs(values), axis=1)
    scales = (absmax / 3.0).astype("<f2")
    safe = np.where(absmax > 0.0, absmax / 3.0, 1.0).astype(np.float32)
    quantized = np.clip(np.rint(values / safe[:, None]), -3, 3).astype(np.int8)
    codes = (quantized + 3).astype(np.uint8)
    return scales, pack_q3(codes)


def encode_blocks(groups: np.ndarray) -> bytes:
    scales, packed = quantize_groups(groups)
    blocks = np.empty((groups.shape[0], BLOCK_BYTES), dtype=np.uint8)
    blocks[:, :2] = scales.view(np.uint8).reshape(-1, 2)
    blocks[:, 2:] = packed
    return blocks.tobytes()


def decode_blocks(blocks: bytes | np.ndarray) -> np.ndarray:
    raw = np.frombuffer(blocks, dtype=np.uint8) if isinstance(blocks, bytes) else np.asarray(blocks, dtype=np.uint8).reshape(-1)
    if raw.size % BLOCK_BYTES:
        raise ValueError("q3 block byte count is not divisible by 26")
    rows = raw.reshape(-1, BLOCK_BYTES)
    scales = np.ascontiguousarray(rows[:, :2]).view("<f2").astype(np.float32).reshape(-1)
    codes = unpack_q3(rows[:, 2:]).astype(np.int8) - 3
    return codes.astype(np.float32) * scales[:, None]


@dataclass(frozen=True)
class AlloyHeader:
    tensor_count: int
    manifest_bytes: int
    manifest_sha256: bytes
    data_start: int


def encode_manifest(manifest: dict) -> bytes:
    return json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def write_header(stream: BinaryIO, manifest_bytes: bytes, tensor_count: int) -> int:
    digest = hashlib.sha256(manifest_bytes).digest()
    stream.write(
        HEADER.pack(
            MAGIC,
            VERSION,
            BITS,
            GROUP_SIZE,
            BLOCK_BYTES,
            tensor_count,
            len(manifest_bytes),
            digest,
        )
    )
    stream.write(manifest_bytes)
    data_start = align_up(HEADER.size + len(manifest_bytes), DATA_ALIGNMENT)
    stream.write(b"\0" * (data_start - stream.tell()))
    return data_start


def read_manifest(path: Path) -> tuple[AlloyHeader, dict]:
    with path.open("rb") as stream:
        raw_header = stream.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise ValueError("truncated Neural Alloy header")
        magic, version, bits, group_size, block_bytes, tensor_count, manifest_size, digest = HEADER.unpack(raw_header)
        if magic != MAGIC:
            raise ValueError("invalid Neural Alloy magic")
        if (version, bits, group_size, block_bytes) != (VERSION, BITS, GROUP_SIZE, BLOCK_BYTES):
            raise ValueError("unsupported Neural Alloy codec parameters")
        manifest_bytes = stream.read(manifest_size)
        if hashlib.sha256(manifest_bytes).digest() != digest:
            raise ValueError("Neural Alloy manifest checksum mismatch")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if len(manifest.get("tensors", [])) != tensor_count:
        raise ValueError("tensor count does not match manifest")
    return (
        AlloyHeader(
            tensor_count=tensor_count,
            manifest_bytes=manifest_size,
            manifest_sha256=digest,
            data_start=align_up(HEADER.size + manifest_size, DATA_ALIGNMENT),
        ),
        manifest,
    )


class AlloyReader:
    def __init__(self, path: Path):
        self.path = path
        self.header, self.manifest = read_manifest(path)
        self.tensors = {item["name"]: item for item in self.manifest["tensors"]}

    def read_delta(self, name: str) -> np.ndarray:
        item = self.tensors[name]
        with self.path.open("rb") as stream:
            stream.seek(self.header.data_start + item["offset"])
            blocks = stream.read(item["group_count"] * BLOCK_BYTES)
        values = decode_blocks(blocks).reshape(-1)[: item["logical_count"]]
        return values.reshape(tuple(reversed(item["shape"])))
