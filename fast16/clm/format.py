"""CLM v0 binary container.

The format keeps a UTF-8 JSON manifest followed by aligned raw tensor data.
Tensor offsets are relative to the payload start, so readers can mmap weights.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .memory import compile_memory
from .transplant import transplant_utf8_byte


MAGIC = b"CLMZERO1"
VERSION = 1
HEADER = struct.Struct("<8sIIQQ32s")
HEADER_SIZE = HEADER.size
MANIFEST_ALIGNMENT = 4096
TENSOR_ALIGNMENT = 64

DTYPE_NAMES = {
    torch.float16: "f16",
    torch.float32: "f32",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "u8",
}
NAME_DTYPES = {name: dtype for dtype, name in DTYPE_NAMES.items()}


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_tensor(name: str) -> bool:
    return (
        name in {"token_embed.weight", "pos_embed.weight", "final_norm.weight"}
        or name.startswith("layers.")
        or name.startswith("temperature_head.")
        or name.startswith("token_pred_head.")
    )


def _storage_tensor(tensor: torch.Tensor, storage_dtype: str) -> torch.Tensor:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.is_floating_point():
        target = NAME_DTYPES[storage_dtype]
        if target not in (torch.float16, torch.float32):
            raise ValueError("floating weights support f16 or f32 storage")
        return tensor.to(target)
    if tensor.dtype not in DTYPE_NAMES:
        return tensor.to(torch.int64)
    return tensor


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.numpy().tobytes(order="C")


def write_clm(path: str | Path, metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Write a CLM atomically and return its manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    blobs: list[tuple[int, bytes]] = []
    payload_size = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        dtype_name = DTYPE_NAMES.get(tensor.dtype)
        if dtype_name is None:
            raise ValueError(f"unsupported tensor dtype for {name}: {tensor.dtype}")
        payload_size = _align(payload_size, TENSOR_ALIGNMENT)
        blob = _tensor_bytes(tensor)
        records.append(
            {
                "name": name,
                "dtype": dtype_name,
                "shape": list(tensor.shape),
                "offset": payload_size,
                "nbytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
        blobs.append((payload_size, blob))
        payload_size += len(blob)

    manifest = {
        "format": "CLM-ZeroTrain",
        "version": VERSION,
        "metadata": _json_safe(metadata),
        "tensors": records,
        "payload_bytes": payload_size,
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload_offset = _align(HEADER_SIZE + len(manifest_bytes), MANIFEST_ALIGNMENT)
    manifest_hash = hashlib.sha256(manifest_bytes).digest()
    header = HEADER.pack(MAGIC, VERSION, 0, len(manifest_bytes), payload_offset, manifest_hash)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as output:
        output.write(header)
        output.write(manifest_bytes)
        output.write(b"\0" * (payload_offset - output.tell()))
        cursor = 0
        for offset, blob in blobs:
            output.write(b"\0" * (offset - cursor))
            output.write(blob)
            cursor = offset + len(blob)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp_path, path)
    return manifest


def pack_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    memory_paths: Iterable[str | Path] = (),
    storage_dtype: str = "f16",
    tokenizer_mode: str = "utf8-byte",
    max_seq_len: int | None = None,
) -> dict[str, Any]:
    """Convert the local ColorLM v3 checkpoint into a standalone CLM."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("checkpoint must contain model_state")

    source_state = checkpoint["model_state"]
    config = dict(checkpoint.get("config", {}))
    char2id = {str(k): int(v) for k, v in checkpoint.get("char2id", {}).items()}
    id2char = {str(k): str(v) for k, v in checkpoint.get("id2char", {}).items()}
    if not char2id or "token_pred_head.weight" not in source_state:
        raise ValueError("checkpoint is missing a text tokenizer or token prediction head")

    tokenizer: dict[str, Any]
    transplant: dict[str, Any]
    architecture: str
    if tokenizer_mode == "utf8-byte":
        target_seq_len = int(max_seq_len or config.get("max_seq_len", 64))
        source_state, config, tokenizer, transplant = transplant_utf8_byte(
            source_state,
            config,
            char2id,
            target_seq_len=target_seq_len,
        )
        architecture = (
            "colormlm_zerotrain_v2"
            if target_seq_len > int(checkpoint.get("config", {}).get("max_seq_len", 64))
            else "colormlm_zerotrain_v1"
        )
    elif tokenizer_mode == "character":
        tokenizer = {"type": "character", "char2id": char2id, "id2char": id2char}
        transplant = {"method": "identity"}
        architecture = "colormlm_zerotrain_v0"
    else:
        raise ValueError(f"unsupported tokenizer mode: {tokenizer_mode}")

    tensors = {
        name: _storage_tensor(tensor, storage_dtype)
        for name, tensor in source_state.items()
        if isinstance(tensor, torch.Tensor) and _runtime_tensor(name)
    }

    memory_tensors, memory_meta = compile_memory(
        memory_paths,
        tokenizer=tokenizer,
        token_embedding=source_state["token_embed.weight"].float(),
        max_value_tokens=int(config.get("max_seq_len", 64)),
    )
    tensors.update({name: _storage_tensor(tensor, storage_dtype) for name, tensor in memory_tensors.items()})

    n_layers = int(config["n_layers"])
    metadata = {
        "architecture": architecture,
        "model_name": (
            "ColorLM-ZeroTrain-v2"
            if architecture == "colormlm_zerotrain_v2"
            else "ColorLM-ZeroTrain-v1"
            if tokenizer_mode == "utf8-byte"
            else "ColorLM-ZeroTrain-v0"
        ),
        "source": {
            "path": checkpoint_path.name,
            "sha256": _source_sha256(checkpoint_path),
            "tensor_count": len(source_state),
        },
        "config": config,
        "tokenizer": tokenizer,
        "transplant": transplant,
        "runtime": {
            "recursive_start_layer": max(0, n_layers - 2),
            "recurrence_steps": 2,
            "recurrence_alpha": 0.25,
            "memory_top_k": 4,
            "memory_temperature": 0.02,
            "memory_hidden_scale": 0.12,
            "memory_logit_scale": 24.0,
        },
        "memory": memory_meta,
        "storage_dtype": storage_dtype,
        "zero_training": True,
    }
    return write_clm(output_path, metadata, tensors)


class ClmReader:
    """Memory-mapped reader for CLM files."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)
        if len(self._mmap) < HEADER_SIZE:
            self.close()
            raise ValueError("CLM file is shorter than its header")

        magic, version, flags, manifest_len, payload_offset, expected_hash = HEADER.unpack_from(self._mmap, 0)
        if magic != MAGIC:
            self.close()
            raise ValueError("invalid CLM magic")
        if version != VERSION:
            self.close()
            raise ValueError(f"unsupported CLM version: {version}")

        manifest_start = HEADER_SIZE
        manifest_end = manifest_start + manifest_len
        manifest_bytes = self._mmap[manifest_start:manifest_end]
        if hashlib.sha256(manifest_bytes).digest() != expected_hash:
            self.close()
            raise ValueError("CLM manifest checksum mismatch")

        self.flags = flags
        self.payload_offset = payload_offset
        self.manifest = json.loads(manifest_bytes.decode("utf-8"))
        self.metadata = self.manifest["metadata"]
        self._records = {record["name"]: record for record in self.manifest["tensors"]}

    def tensor_names(self) -> list[str]:
        return sorted(self._records)

    def has_tensor(self, name: str) -> bool:
        return name in self._records

    def tensor(self, name: str, *, copy: bool = False) -> torch.Tensor:
        record = self._records[name]
        dtype = NAME_DTYPES[record["dtype"]]
        count = 1
        for size in record["shape"]:
            count *= int(size)
        tensor = torch.frombuffer(
            self._mmap,
            dtype=dtype,
            count=count,
            offset=self.payload_offset + int(record["offset"]),
        ).reshape(record["shape"])
        return tensor.clone() if copy else tensor

    def verify_tensors(self) -> list[str]:
        bad: list[str] = []
        for name, record in self._records.items():
            start = self.payload_offset + int(record["offset"])
            end = start + int(record["nbytes"])
            digest = hashlib.sha256(self._mmap[start:end]).hexdigest()
            if digest != record["sha256"]:
                bad.append(name)
        return bad

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_bytes": self.path.stat().st_size,
            "format": self.manifest["format"],
            "version": self.manifest["version"],
            "model_name": self.metadata.get("model_name"),
            "architecture": self.metadata.get("architecture"),
            "tensor_count": len(self._records),
            "payload_bytes": self.manifest["payload_bytes"],
            "storage_dtype": self.metadata.get("storage_dtype"),
            "memory": self.metadata.get("memory", {}),
            "runtime": self.metadata.get("runtime", {}),
        }

    def close(self) -> None:
        mm = getattr(self, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                pass
            self._mmap = None
        file_obj = getattr(self, "_file", None)
        if file_obj is not None:
            file_obj.close()
            self._file = None

    def __enter__(self) -> "ClmReader":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
