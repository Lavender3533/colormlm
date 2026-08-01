"""DeepSeek-V4 原生状态 CNOB v2 的流式写入、读取与严格校验。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
VERSION = 2
F32 = 0
I32 = 2
BF16 = 3
LAYERS = (39, 40, 41, 42)
HIDDEN = 4096
HC_MULT = 4
TOP_K = 6
KIND_TOKEN_IDS = 2000
KIND_TOKEN_STATS = 2001


def kind_hidden(layer: int) -> int:
    return 1000 + _checked_layer(layer)


def kind_mhc_streams(layer: int) -> int:
    return 1100 + _checked_layer(layer)


def kind_mhc_attn(layer: int) -> int:
    return 1200 + _checked_layer(layer)


def kind_mhc_ffn(layer: int) -> int:
    return 1300 + _checked_layer(layer)


def kind_router_ids(layer: int) -> int:
    return 1400 + _checked_layer(layer)


def kind_router_weights(layer: int) -> int:
    return 1500 + _checked_layer(layer)


def _checked_layer(layer: int) -> int:
    value = int(layer)
    if value not in LAYERS:
        raise ValueError(f"capture layer 必须是 {LAYERS}，实际为 {value}")
    return value


def expected_kinds() -> list[int]:
    result: list[int] = []
    for layer in LAYERS:
        result.extend(
            [
                kind_hidden(layer),
                kind_mhc_streams(layer),
                kind_mhc_attn(layer),
                kind_mhc_ffn(layer),
                kind_router_ids(layer),
                kind_router_weights(layer),
            ]
        )
    result.extend([KIND_TOKEN_IDS, KIND_TOKEN_STATS])
    return result


def kind_spec(kind: int) -> tuple[int, tuple[int, int, int, int]]:
    if kind == KIND_TOKEN_IDS:
        return I32, (3, 1, 1, 1)
    if kind == KIND_TOKEN_STATS:
        return F32, (2, 1, 1, 1)
    layer = kind % 100
    _checked_layer(layer)
    family = kind - layer
    if family == 1000:
        return BF16, (HIDDEN, 1, 1, 1)
    if family == 1100:
        return BF16, (HC_MULT, HIDDEN, 1, 1)
    if family in (1200, 1300):
        return F32, (20, 1, 1, 1)
    if family == 1400:
        return I32, (TOP_K, 1, 1, 1)
    if family == 1500:
        return F32, (TOP_K, 1, 1, 1)
    raise ValueError(f"未知 kind={kind}")


def dtype_item_bytes(dtype: int) -> int:
    try:
        return {F32: 4, I32: 4, BF16: 2}[int(dtype)]
    except KeyError as exc:
        raise ValueError(f"未知 dtype={dtype}") from exc


def element_count(shape: Sequence[int]) -> int:
    count = 1
    for value in shape:
        if int(value) <= 0:
            raise ValueError(f"shape 必须为正数: {shape}")
        count *= int(value)
    return count


def pack_f32(values: Iterable[float]) -> bytes:
    materialized = [float(value) for value in values]
    return struct.pack(f"<{len(materialized)}f", *materialized)


def pack_i32(values: Iterable[int]) -> bytes:
    materialized = [int(value) for value in values]
    return struct.pack(f"<{len(materialized)}i", *materialized)


def pack_bf16_from_f32(values: Iterable[float]) -> bytes:
    """将 F32 以 round-to-nearest-even 转成 BF16 原始 little-endian bits。"""

    output = bytearray()
    for value in values:
        raw = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        rounded = (raw + 0x7FFF + ((raw >> 16) & 1)) >> 16
        output.extend(struct.pack("<H", rounded & 0xFFFF))
    return bytes(output)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CnobChunk:
    kind: int
    record: int
    dtype: int
    shape: tuple[int, int, int, int]
    payload: bytes


def iter_cnob(path: Path) -> Iterator[CnobChunk]:
    with path.open("rb") as source:
        while True:
            encoded = source.read(HEADER.size)
            if not encoded:
                return
            if len(encoded) != HEADER.size:
                raise ValueError("CNOB header 被截断")
            magic, version, kind, record, dtype, reserved, *tail = HEADER.unpack(encoded)
            if magic != MAGIC or version != VERSION or reserved != 0:
                raise ValueError("CNOB magic/version/reserved 不符合 v2 契约")
            shape = tuple(int(value) for value in tail[:4])
            payload_bytes = int(tail[4])
            expected_dtype, expected_shape = kind_spec(kind)
            if dtype != expected_dtype or shape != expected_shape:
                raise ValueError(
                    f"record={record} kind={kind} dtype/shape 错误: {(dtype, shape)} != "
                    f"{(expected_dtype, expected_shape)}"
                )
            expected_bytes = element_count(shape) * dtype_item_bytes(dtype)
            if payload_bytes != expected_bytes:
                raise ValueError(
                    f"record={record} kind={kind} payload_bytes={payload_bytes}，期望 {expected_bytes}"
                )
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError(f"record={record} kind={kind} payload 被截断")
            yield CnobChunk(kind, record, dtype, shape, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("JSONL 禁止 UTF-8 BOM")
    text = raw.decode("utf-8")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"JSONL 第 {line_number} 行为空")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 第 {line_number} 行必须是 object")
        rows.append(value)
    return rows


def validate_capture(cnob: Path, sidecar: Path, *, require_real: bool = False) -> dict[str, Any]:
    groups: dict[int, list[int]] = {}
    chunk_count = 0
    for chunk in iter_cnob(cnob):
        groups.setdefault(chunk.record, []).append(chunk.kind)
        chunk_count += 1
    expected_records = list(range(len(groups)))
    if sorted(groups) != expected_records:
        raise ValueError("CNOB record 必须从 0 开始连续")
    kinds = expected_kinds()
    for record in expected_records:
        if groups[record] != kinds:
            raise ValueError(f"record={record} kind 顺序/完整性错误")

    rows = read_jsonl(sidecar)
    if len(rows) != len(groups):
        raise ValueError(f"CNOB/JSONL 记录数不一致: {len(groups)} != {len(rows)}")
    required = {
        "format",
        "record",
        "sequence_id",
        "phase",
        "token_position",
        "input_token_id",
        "input_token_text",
        "target_token_id",
        "target_token_text",
        "predicted_token_id",
        "predicted_token_text",
        "target_logprob",
        "target_nll",
        "prefix_sha256",
        "synthetic",
        "native_forward_completed",
    }
    for record, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"JSONL record={record} 缺少字段: {sorted(missing)}")
        if row["format"] != "polaris-deepseek-native-step-v1" or int(row["record"]) != record:
            raise ValueError(f"JSONL record={record} format/record 错误")
        if row["phase"] not in {"prefill", "decode"}:
            raise ValueError(f"JSONL record={record} phase 错误")
        if not isinstance(row["prefix_sha256"], str) or len(row["prefix_sha256"]) != 64:
            raise ValueError(f"JSONL record={record} prefix_sha256 错误")
        target = int(row["target_token_id"])
        logprob = row["target_logprob"]
        nll = row["target_nll"]
        if target < 0:
            if logprob is not None or nll is not None:
                raise ValueError(f"JSONL record={record} 无 target 时 logprob/nll 必须为 null")
        else:
            if not (math.isfinite(float(logprob)) and math.isfinite(float(nll)) and float(nll) >= 0):
                raise ValueError(f"JSONL record={record} NLL 非法")
            if abs(float(nll) + float(logprob)) > 1e-5:
                raise ValueError(f"JSONL record={record} NLL != -logprob")
        if require_real and (bool(row["synthetic"]) or not bool(row["native_forward_completed"])):
            raise ValueError("正式 capture 禁止 synthetic，且必须完成原生 forward")

    return {
        "format": "polaris-deepseek-capture-validation-v1",
        "ok": True,
        "real_capture_required": require_real,
        "records": len(rows),
        "cnob_chunks": chunk_count,
        "chunks_per_record": len(kinds),
        "cnob_bytes": cnob.stat().st_size,
        "cnob_sha256": sha256_file(cnob),
        "sidecar_bytes": sidecar.stat().st_size,
        "sidecar_sha256": sha256_file(sidecar),
    }


class CaptureWriter:
    """以 `.part` 原子写 CNOB/JSONL；正式状态需要显式原生适配器证明。"""

    def __init__(self, cnob: Path, sidecar: Path):
        self.cnob = cnob
        self.sidecar = sidecar
        self.cnob_part = cnob.with_name(cnob.name + ".part")
        self.sidecar_part = sidecar.with_name(sidecar.name + ".part")
        for path in (cnob, sidecar, self.cnob_part, self.sidecar_part):
            if path.exists():
                raise FileExistsError(f"拒绝覆盖已有 capture 文件: {path}")
        cnob.parent.mkdir(parents=True, exist_ok=True)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        self._binary = self.cnob_part.open("xb")
        self._text = self.sidecar_part.open("x", encoding="utf-8", newline="\n")
        self._record = 0
        self._closed = False

    @property
    def records(self) -> int:
        return self._record

    def _chunk(self, record: int, kind: int, payload: bytes) -> None:
        dtype, shape = kind_spec(kind)
        expected = element_count(shape) * dtype_item_bytes(dtype)
        if len(payload) != expected:
            raise ValueError(f"record={record} kind={kind} payload 长度 {len(payload)} != {expected}")
        self._binary.write(HEADER.pack(MAGIC, VERSION, kind, record, dtype, 0, *shape, len(payload)))
        self._binary.write(payload)

    def write_step(self, layers: Mapping[int, Mapping[str, Any]], token: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("CaptureWriter 已关闭")
        record = self._record
        if sorted(int(layer) for layer in layers) != list(LAYERS):
            raise ValueError(f"必须精确提供 L39-L42，实际为 {sorted(layers)}")
        for layer in LAYERS:
            value = layers[layer]
            hidden = _as_payload(value["hidden_mean_bf16"], BF16, HIDDEN)
            streams = _as_payload(value["mhc_streams_bf16"], BF16, HC_MULT * HIDDEN)
            attn = _as_payload(value["mhc_attention_post_comb"], F32, 20)
            ffn = _as_payload(value["mhc_ffn_post_comb"], F32, 20)
            ids = _as_payload(value["router_topk_ids"], I32, TOP_K)
            weights = _as_payload(value["router_topk_weights"], F32, TOP_K)
            self._chunk(record, kind_hidden(layer), hidden)
            self._chunk(record, kind_mhc_streams(layer), streams)
            self._chunk(record, kind_mhc_attn(layer), attn)
            self._chunk(record, kind_mhc_ffn(layer), ffn)
            self._chunk(record, kind_router_ids(layer), ids)
            self._chunk(record, kind_router_weights(layer), weights)

        target_id = int(token.get("target_token_id", -1))
        logprob = token.get("target_logprob")
        nll = token.get("target_nll")
        if target_id < 0:
            logprob = float("nan")
            nll = float("nan")
        else:
            if logprob is None and nll is not None:
                logprob = -float(nll)
            if nll is None and logprob is not None:
                nll = -float(logprob)
            if logprob is None or nll is None:
                raise ValueError("有 target_token_id 时必须提供 logprob 或 nll")
            if not (math.isfinite(float(logprob)) and math.isfinite(float(nll)) and float(nll) >= 0):
                raise ValueError("target logprob/nll 非法")
            if abs(float(nll) + float(logprob)) > 1e-5:
                raise ValueError("target_nll 必须等于 -target_logprob")

        input_id = int(token["input_token_id"])
        predicted_id = int(token["predicted_token_id"])
        self._chunk(record, KIND_TOKEN_IDS, pack_i32([input_id, target_id, predicted_id]))
        self._chunk(record, KIND_TOKEN_STATS, pack_f32([float(logprob), float(nll)]))

        row = {
            "format": "polaris-deepseek-native-step-v1",
            "record": record,
            "sequence_id": str(token["sequence_id"]),
            "phase": str(token["phase"]),
            "token_position": int(token["token_position"]),
            "input_token_id": input_id,
            "input_token_text": str(token.get("input_token_text", "")),
            "target_token_id": target_id,
            "target_token_text": str(token.get("target_token_text", "")),
            "predicted_token_id": predicted_id,
            "predicted_token_text": str(token.get("predicted_token_text", "")),
            "target_logprob": None if target_id < 0 else float(logprob),
            "target_nll": None if target_id < 0 else float(nll),
            "prefix_sha256": str(token["prefix_sha256"]),
            "synthetic": bool(token["synthetic"]),
            "native_forward_completed": bool(token["native_forward_completed"]),
        }
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        self._text.write(encoded + "\n")
        self._record += 1

    def finish(self) -> None:
        if self._closed:
            return
        self._binary.flush()
        os.fsync(self._binary.fileno())
        self._text.flush()
        os.fsync(self._text.fileno())
        self._binary.close()
        self._text.close()
        os.replace(self.cnob_part, self.cnob)
        os.replace(self.sidecar_part, self.sidecar)
        self._closed = True

    def abort(self) -> None:
        if not self._closed:
            self._binary.close()
            self._text.close()
            self._closed = True


def _as_payload(value: Any, dtype: int, count: int) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    elif dtype == F32:
        payload = pack_f32(value)
    elif dtype == I32:
        payload = pack_i32(value)
    elif dtype == BF16:
        payload = pack_bf16_from_f32(value)
    else:
        raise ValueError(f"未知 dtype={dtype}")
    expected = count * dtype_item_bytes(dtype)
    if len(payload) != expected:
        raise ValueError(f"payload 长度 {len(payload)} != {expected}")
    return payload
