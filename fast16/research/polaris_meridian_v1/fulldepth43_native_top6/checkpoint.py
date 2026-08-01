"""FullDepth43 ``DecoderState`` 的安全、原子 checkpoint 格式。

格式不使用 pickle。UTF-8 manifest 绑定固定 model/profile/tokenizer、
committed ledger、forced cursor 与所有 tensor 的 shape/dtype/offset/SHA-256。
binary payload 先以内容寻址文件原子发布，manifest 最后原子替换。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, TYPE_CHECKING

import torch

from fast16.research.polaris_meridian_v1.s14_chat_encoding.forced_prefill import (
    S14_TOKENIZER_SHA256,
)
from fast16.research.polaris_meridian_v1.s14_first_real_token import executor as s14

from .profile import FULLDEPTH43_NATIVE_TOP6, ExecutionProfile

if TYPE_CHECKING:
    from .executor import DecoderState


CHECKPOINT_FORMAT = "polaris-fulldepth43-decoder-checkpoint-v1"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TENSOR_FIELDS = (
    "main_kv_state",
    "main_score_state",
    "main_compressed_kv",
    "indexer_kv_state",
    "indexer_score_state",
    "indexer_compressed_kv",
)
_DTYPE_TO_NAME = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


class CheckpointError(RuntimeError):
    pass


def _canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def catalog_fingerprint(catalog: Mapping[str, Any]) -> str:
    """将已验证的 FullDepth catalog 绑定到生产 resume provenance。"""

    if not isinstance(catalog, Mapping):
        raise CheckpointError("catalog 必须是 object")
    try:
        return hashlib.sha256(_canonical_bytes(dict(catalog))).hexdigest()
    except Exception as exc:
        raise CheckpointError(f"catalog 无法产生确定指纹: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointError(f"checkpoint manifest 不存在: {path}")
    size = path.stat().st_size
    if not 0 < size <= MAX_MANIFEST_BYTES:
        raise CheckpointError("checkpoint manifest 字节数非法")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CheckpointError(f"checkpoint JSON 重复键: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=no_duplicates,
        )
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"checkpoint manifest 不是严格 UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError("checkpoint manifest 顶层必须是 object")
    return value


def _json_object(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise CheckpointError(f"{label} 必须是 object")
    source = {} if value is None else dict(value)
    try:
        encoded = json.dumps(source, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
    except Exception as exc:
        raise CheckpointError(f"{label} 必须是有限 UTF-8 JSON object") from exc
    if not isinstance(decoded, dict):
        raise CheckpointError(f"{label} 必须是 object")
    return decoded


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CheckpointError(f"{label} 必须是 >= {minimum} 的 int")
    return value


def _validate_token(token_id: Any, label: str) -> int:
    value = _require_int(token_id, label)
    if value >= s14.VOCAB_SIZE:
        raise CheckpointError(f"{label} 越出固定 vocab")
    return value


def _validate_ledger(state: Any) -> None:
    position = _require_int(state.position, "decoder.position")
    input_token_id = _validate_token(state.input_token_id, "decoder.input_token_id")
    if not isinstance(state.committed_tokens, list) or len(state.committed_tokens) != position:
        raise CheckpointError("committed ledger 长度与 decoder.position 不一致")
    if position == 0 and input_token_id != s14.BOS_TOKEN_ID:
        raise CheckpointError("position0 checkpoint 必须从 BOS 开始")
    expected_input: int | None = None
    for index, row in enumerate(state.committed_tokens):
        if not isinstance(row, Mapping) or row.get("position") != index:
            raise CheckpointError(f"committed ledger position 不连续: {index}")
        row_input = _validate_token(row.get("input_token_id"), f"ledger[{index}].input")
        row_output = _validate_token(row.get("output_token_id"), f"ledger[{index}].output")
        if expected_input is not None and row_input != expected_input:
            raise CheckpointError(f"committed ledger token 链断裂: {index}")
        next_input = row.get("next_input_token_id", row_output)
        expected_input = _validate_token(next_input, f"ledger[{index}].next_input")
    if position and state.committed_tokens[0].get("input_token_id") != s14.BOS_TOKEN_ID:
        raise CheckpointError("committed ledger 必须从 BOS 开始")
    if expected_input is not None and expected_input != input_token_id:
        raise CheckpointError("decoder pending token 与 committed ledger 末尾不一致")

    queue = state.forced_queue
    if queue is None:
        if any(row.get("input_source") == "forced_prefill" for row in state.committed_tokens):
            raise CheckpointError("forced ledger 缺少必要 token queue provenance")
        return
    try:
        queue.validate()
    except Exception as exc:
        raise CheckpointError(f"forced queue 合同失败: {exc}") from exc
    expected_cursor = min(position, len(queue.token_ids))
    if queue.cursor != expected_cursor:
        raise CheckpointError("forced cursor 与 decoder.position 不一致")
    forced_count = min(position, len(queue.token_ids))
    for index in range(forced_count):
        row = state.committed_tokens[index]
        if (
            row.get("input_source") != "forced_prefill"
            or row.get("forced_cursor") != index
            or row.get("input_token_id") != queue.token_ids[index]
        ):
            raise CheckpointError(f"forced ledger 与 token queue 不一致: {index}")
    if queue.active and input_token_id != queue.current_token_id:
        raise CheckpointError("active forced queue 与 decoder pending token 不一致")


def _expect_tensor(
    tensor: Any,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    label: str,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise CheckpointError(f"{label} 不是 tensor")
    if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device.type != "cpu":
        raise CheckpointError(
            f"{label} shape/dtype/device 漂移: "
            f"{tuple(tensor.shape)}/{tensor.dtype}/{tensor.device.type}"
        )


def validate_decoder_state(
    state: Any,
    *,
    profile: ExecutionProfile = FULLDEPTH43_NATIVE_TOP6,
) -> None:
    """验证可恢复的 FullDepth 连续状态，不接受摘要或部分层。"""

    profile.validate()
    _validate_ledger(state)
    position = state.position
    if position == 0:
        if state.layer_states:
            raise CheckpointError("position0 不得携带旧层状态")
        return
    if not isinstance(state.layer_states, Mapping) or set(state.layer_states) != set(profile.layers):
        raise CheckpointError("非零 checkpoint 必须完整包含 43 层状态")
    state_position = position - 1
    compressed_count_by_ratio = {
        ratio: (state_position + 1) // ratio for ratio in (4, 128)
    }
    for layer in profile.layers:
        layer_state = state.layer_states[layer]
        if layer_state.layer != layer or layer_state.position != state_position:
            raise CheckpointError(f"L{layer} layer/position 漂移")
        _expect_tensor(
            layer_state.window_kv,
            shape=(1, s14.WINDOW_SIZE, 512),
            dtype=torch.bfloat16,
            label=f"L{layer}.window_kv",
        )
        ratio = profile.ratio_for(layer)
        compressor = layer_state.compressor
        if ratio == 0:
            if compressor is not None:
                raise CheckpointError(f"L{layer} ratio0 不得携带 compressor")
            continue
        if (
            compressor is None
            or compressor.ratio != ratio
            or not isinstance(compressor.overlap, bool)
            or compressor.overlap != (ratio == 4)
        ):
            raise CheckpointError(f"L{layer} compressor ratio/overlap 漂移")
        coff = 2 if ratio == 4 else 1
        projected = 1024 if ratio == 4 else 512
        remainder_shape = (1, coff * ratio, projected)
        _expect_tensor(
            compressor.main_kv_state,
            shape=remainder_shape,
            dtype=torch.float32,
            label=f"L{layer}.main_kv_state",
        )
        _expect_tensor(
            compressor.main_score_state,
            shape=remainder_shape,
            dtype=torch.float32,
            label=f"L{layer}.main_score_state",
        )
        _expect_tensor(
            compressor.main_compressed_kv,
            shape=(1, compressed_count_by_ratio[ratio], 512),
            dtype=torch.bfloat16,
            label=f"L{layer}.main_compressed_kv",
        )
        indexers = (
            compressor.indexer_kv_state,
            compressor.indexer_score_state,
            compressor.indexer_compressed_kv,
        )
        if ratio == 128:
            if any(value is not None for value in indexers):
                raise CheckpointError(f"L{layer} ratio128 不得携带 indexer")
            continue
        _expect_tensor(
            compressor.indexer_kv_state,
            shape=(1, 8, 256),
            dtype=torch.float32,
            label=f"L{layer}.indexer_kv_state",
        )
        _expect_tensor(
            compressor.indexer_score_state,
            shape=(1, 8, 256),
            dtype=torch.float32,
            label=f"L{layer}.indexer_score_state",
        )
        _expect_tensor(
            compressor.indexer_compressed_kv,
            shape=(1, compressed_count_by_ratio[ratio], 128),
            dtype=torch.bfloat16,
            label=f"L{layer}.indexer_compressed_kv",
        )


def _tensor_bytes(tensor: torch.Tensor, label: str) -> tuple[torch.Tensor, bytes, str]:
    if tensor.layout != torch.strided:
        raise CheckpointError(f"{label} 必须是 strided tensor")
    cpu = tensor.detach().to(device="cpu").contiguous()
    dtype_name = _DTYPE_TO_NAME.get(cpu.dtype)
    if dtype_name is None:
        raise CheckpointError(f"{label} dtype 不受支持: {cpu.dtype}")
    payload = cpu.view(torch.uint8).numpy().tobytes(order="C")
    return cpu, payload, dtype_name


def _iter_layer_tensors(layer_state: Any) -> Iterable[tuple[str, torch.Tensor | None]]:
    yield "window_kv", layer_state.window_kv
    compressor = layer_state.compressor
    if compressor is None:
        return
    for field in _TENSOR_FIELDS:
        yield field, getattr(compressor, field)


def _forced_document(queue: s14.ForcedTokenQueue | None) -> dict[str, Any] | None:
    if queue is None:
        return None
    return {
        "token_ids": list(queue.token_ids),
        "cursor": queue.cursor,
        "artifact_sha256": queue.artifact_sha256,
    }


def _old_payload_name(manifest_path: Path) -> str | None:
    try:
        document = _strict_json(manifest_path)
        payload = document.get("payload")
        name = payload.get("file") if isinstance(payload, Mapping) else None
        if isinstance(name, str) and Path(name).name == name:
            return name
    except CheckpointError:
        pass
    return None


def save_decoder_checkpoint(
    state: "DecoderState",
    manifest_path: str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
    profile: ExecutionProfile = FULLDEPTH43_NATIVE_TOP6,
    tokenizer_sha256: str = S14_TOKENIZER_SHA256,
) -> dict[str, Any]:
    """先发布内容寻址 payload，再原子替换 UTF-8 manifest。"""

    profile.validate()
    if not _SHA256.fullmatch(tokenizer_sha256):
        raise CheckpointError("tokenizer SHA-256 非法")
    validate_decoder_state(state, profile=profile)
    manifest = Path(manifest_path).resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    old_payload = _old_payload_name(manifest) if manifest.exists() else None
    temporary_payload = manifest.parent / f".{manifest.name}.{uuid.uuid4().hex}.state.tmp"
    payload_digest = hashlib.sha256()
    offset = 0
    tensor_count = 0
    layer_documents: list[dict[str, Any]] = []
    try:
        with temporary_payload.open("xb") as handle:
            for layer in profile.layers:
                layer_state = state.layer_states.get(layer) if state.position else None
                if layer_state is None:
                    continue
                tensor_documents: dict[str, Any] = {}
                for field, tensor in _iter_layer_tensors(layer_state):
                    if tensor is None:
                        tensor_documents[field] = None
                        continue
                    cpu, raw, dtype_name = _tensor_bytes(tensor, f"L{layer}.{field}")
                    tensor_digest = hashlib.sha256(raw).hexdigest()
                    tensor_documents[field] = {
                        "offset": offset,
                        "bytes": len(raw),
                        "dtype": dtype_name,
                        "shape": list(cpu.shape),
                        "sha256": tensor_digest,
                    }
                    handle.write(raw)
                    payload_digest.update(raw)
                    offset += len(raw)
                    if offset > MAX_PAYLOAD_BYTES:
                        raise CheckpointError("checkpoint payload 超过安全上限")
                    tensor_count += 1
                compressor = layer_state.compressor
                layer_documents.append(
                    {
                        "layer": layer,
                        "position": layer_state.position,
                        "window_kv": tensor_documents.pop("window_kv"),
                        "compressor": (
                            None
                            if compressor is None
                            else {
                                "ratio": compressor.ratio,
                                "overlap": compressor.overlap,
                                **tensor_documents,
                            }
                        ),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        payload_sha256 = payload_digest.hexdigest()
        payload_name = f"{manifest.stem}.{payload_sha256}.state.bin"
        payload_path = manifest.parent / payload_name
        if payload_path.exists():
            if payload_path.stat().st_size != offset or _sha256_file(payload_path) != payload_sha256:
                raise CheckpointError("已有内容寻址 payload 与 SHA 不一致")
            temporary_payload.unlink()
        else:
            os.replace(temporary_payload, payload_path)

        document: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": {
                "repo": profile.repo,
                "revision": profile.revision,
                "profile": profile.as_dict(),
                "vocab_size": s14.VOCAB_SIZE,
                "tokenizer_sha256": tokenizer_sha256,
            },
            "decoder": {
                "position": state.position,
                "input_token_id": state.input_token_id,
                "committed_tokens": [dict(row) for row in state.committed_tokens],
                "forced_queue": _forced_document(state.forced_queue),
            },
            "payload": {
                "file": payload_name,
                "bytes": offset,
                "sha256": payload_sha256,
                "tensor_count": tensor_count,
            },
            "layers": layer_documents,
            "provenance": _json_object(provenance, "checkpoint provenance"),
        }
        document["checkpoint_sha256"] = hashlib.sha256(_canonical_bytes(document)).hexdigest()
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n"
        temporary_manifest = manifest.parent / f".{manifest.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary_manifest.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_manifest, manifest)
        finally:
            if temporary_manifest.exists():
                temporary_manifest.unlink()
        if old_payload and old_payload != payload_name:
            old_path = manifest.parent / old_payload
            if old_path.is_file() and old_path.parent == manifest.parent:
                try:
                    old_path.unlink()
                except OSError:
                    # 旧 payload 只是孤儿；新 manifest 已原子发布，
                    # 不能因为清理失败否定新 checkpoint 事务。
                    pass
        return {
            "format": CHECKPOINT_FORMAT,
            "manifest": str(manifest),
            "manifest_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "checkpoint_sha256": document["checkpoint_sha256"],
            "payload": str(payload_path),
            "payload_sha256": payload_sha256,
            "payload_bytes": offset,
            "tensor_count": tensor_count,
            "position": state.position,
        }
    finally:
        if temporary_payload.exists():
            temporary_payload.unlink()


def _verify_model(
    document: Mapping[str, Any],
    *,
    profile: ExecutionProfile,
    tokenizer_sha256: str,
) -> None:
    if document.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError("拒绝非 FullDepth43 decoder checkpoint 格式")
    model = document.get("model")
    if not isinstance(model, Mapping):
        raise CheckpointError("checkpoint 缺少 model provenance")
    if (
        model.get("repo") != profile.repo
        or model.get("revision") != profile.revision
        or model.get("profile") != profile.as_dict()
        or model.get("vocab_size") != s14.VOCAB_SIZE
        or model.get("tokenizer_sha256") != tokenizer_sha256
    ):
        raise CheckpointError("checkpoint model/revision/profile/tokenizer 与当前运行时不匹配")


def _tensor_refs(layer_document: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    yield "window_kv", layer_document.get("window_kv")
    compressor = layer_document.get("compressor")
    if compressor is None:
        return
    if not isinstance(compressor, Mapping):
        raise CheckpointError("compressor manifest 必须是 object/null")
    for field in _TENSOR_FIELDS:
        yield field, compressor.get(field)


def _validate_ref(reference: Any, expected_offset: int, label: str) -> tuple[int, torch.dtype, tuple[int, ...], str]:
    if not isinstance(reference, Mapping):
        raise CheckpointError(f"{label} tensor ref 缺失")
    offset = _require_int(reference.get("offset"), f"{label}.offset")
    size = _require_int(reference.get("bytes"), f"{label}.bytes")
    if offset != expected_offset:
        raise CheckpointError(f"{label} payload offset 不连续")
    dtype = _NAME_TO_DTYPE.get(reference.get("dtype"))
    shape_value = reference.get("shape")
    if dtype is None or not isinstance(shape_value, list):
        raise CheckpointError(f"{label} dtype/shape 非法")
    shape = tuple(_require_int(value, f"{label}.shape", minimum=0) for value in shape_value)
    if not shape:
        raise CheckpointError(f"{label} shape 非法")
    expected_size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if size != expected_size:
        raise CheckpointError(f"{label} bytes 与 dtype/shape 不闭合")
    digest = reference.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise CheckpointError(f"{label} SHA-256 非法")
    return size, dtype, shape, digest


def _read_tensor(
    handle: Any,
    reference: Mapping[str, Any],
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    expected_sha256: str,
    label: str,
) -> torch.Tensor:
    size = int(reference["bytes"])
    raw = handle.read(size)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise CheckpointError(f"{label} tensor payload 截断或 SHA 不一致")
    if size == 0:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()


def load_decoder_checkpoint(
    manifest_path: str | Path,
    *,
    profile: ExecutionProfile = FULLDEPTH43_NATIVE_TOP6,
    tokenizer_sha256: str = S14_TOKENIZER_SHA256,
) -> tuple["DecoderState", dict[str, Any]]:
    """完整验证 manifest/payload/模型/状态后才构造 DecoderState。"""

    profile.validate()
    if not _SHA256.fullmatch(tokenizer_sha256):
        raise CheckpointError("期望 tokenizer SHA-256 非法")
    manifest = Path(manifest_path).resolve()
    document = _strict_json(manifest)
    _verify_model(document, profile=profile, tokenizer_sha256=tokenizer_sha256)
    checkpoint_sha256 = document.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or not _SHA256.fullmatch(checkpoint_sha256):
        raise CheckpointError("checkpoint 自身 SHA-256 缺失")
    unsigned = dict(document)
    unsigned.pop("checkpoint_sha256", None)
    if hashlib.sha256(_canonical_bytes(unsigned)).hexdigest() != checkpoint_sha256:
        raise CheckpointError("checkpoint manifest 元数据 SHA-256 不一致")

    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint payload 合同缺失")
    payload_sha256 = payload.get("sha256")
    payload_bytes = _require_int(payload.get("bytes"), "payload.bytes")
    tensor_count = _require_int(payload.get("tensor_count"), "payload.tensor_count")
    if payload_bytes > MAX_PAYLOAD_BYTES:
        raise CheckpointError("checkpoint payload 超过安全上限")
    if not isinstance(payload_sha256, str) or not _SHA256.fullmatch(payload_sha256):
        raise CheckpointError("payload SHA-256 非法")
    payload_name = payload.get("file")
    expected_name = f"{manifest.stem}.{payload_sha256}.state.bin"
    if not isinstance(payload_name, str) or payload_name != expected_name or Path(payload_name).name != payload_name:
        raise CheckpointError("payload 必须是同目录内容寻址文件")
    payload_path = manifest.parent / payload_name
    if not payload_path.is_file() or payload_path.stat().st_size != payload_bytes:
        raise CheckpointError("payload 缺失或字节数不一致")
    if _sha256_file(payload_path) != payload_sha256:
        raise CheckpointError("payload 整体 SHA-256 不一致")

    decoder_document = document.get("decoder")
    layers_document = document.get("layers")
    if not isinstance(decoder_document, Mapping) or not isinstance(layers_document, list):
        raise CheckpointError("decoder/layers manifest 缺失")
    position = _require_int(decoder_document.get("position"), "decoder.position")
    input_token_id = _validate_token(decoder_document.get("input_token_id"), "decoder.input_token_id")
    committed = decoder_document.get("committed_tokens")
    if not isinstance(committed, list):
        raise CheckpointError("committed_tokens 必须是 array")
    forced_document = decoder_document.get("forced_queue")
    forced_queue = None
    if forced_document is not None:
        if not isinstance(forced_document, Mapping) or not isinstance(forced_document.get("token_ids"), list):
            raise CheckpointError("forced_queue manifest 非法")
        token_ids = tuple(
            _validate_token(value, "forced_queue.token_id")
            for value in forced_document["token_ids"]
        )
        artifact = forced_document.get("artifact_sha256")
        if not isinstance(artifact, str) or not _SHA256.fullmatch(artifact):
            raise CheckpointError("forced artifact SHA-256 非法")
        forced_queue = s14.ForcedTokenQueue(
            token_ids=token_ids,
            cursor=_require_int(forced_document.get("cursor"), "forced_queue.cursor"),
            artifact_sha256=artifact,
        )

    if position == 0 and layers_document:
        raise CheckpointError("position0 checkpoint 不得包含层 tensor")
    if position and len(layers_document) != len(profile.layers):
        raise CheckpointError("非零 checkpoint 必须精确包含 43 层")
    expected_offset = 0
    observed_tensors = 0
    layer_states: dict[int, s14.LayerRuntimeState] = {}
    expected_layers = () if position == 0 else profile.layers
    with payload_path.open("rb") as handle:
        for expected_layer, layer_document in zip(expected_layers, layers_document, strict=True):
            if not isinstance(layer_document, Mapping):
                raise CheckpointError("layer manifest 必须是 object")
            layer = _require_int(layer_document.get("layer"), "layer")
            layer_position = _require_int(layer_document.get("position"), f"L{layer}.position")
            if layer != expected_layer:
                raise CheckpointError("layer manifest 顺序/覆盖漂移")
            tensors: dict[str, torch.Tensor | None] = {}
            for field, reference in _tensor_refs(layer_document):
                if reference is None:
                    tensors[field] = None
                    continue
                size, dtype, shape, digest = _validate_ref(
                    reference, expected_offset, f"L{layer}.{field}"
                )
                tensors[field] = _read_tensor(
                    handle,
                    reference,
                    dtype=dtype,
                    shape=shape,
                    expected_sha256=digest,
                    label=f"L{layer}.{field}",
                )
                expected_offset += size
                observed_tensors += 1
            compressor_document = layer_document.get("compressor")
            compressor = None
            if compressor_document is not None:
                assert isinstance(compressor_document, Mapping)
                compressor = s14.CompressorRemainderState(
                    ratio=_require_int(compressor_document.get("ratio"), f"L{layer}.ratio", minimum=1),
                    overlap=compressor_document.get("overlap"),
                    main_kv_state=tensors["main_kv_state"],
                    main_score_state=tensors["main_score_state"],
                    main_compressed_kv=tensors["main_compressed_kv"],
                    indexer_kv_state=tensors["indexer_kv_state"],
                    indexer_score_state=tensors["indexer_score_state"],
                    indexer_compressed_kv=tensors["indexer_compressed_kv"],
                )
            layer_states[layer] = s14.LayerRuntimeState(
                layer=layer,
                position=layer_position,
                window_kv=tensors["window_kv"],
                compressor=compressor,
            )
        if handle.read(1):
            raise CheckpointError("payload 存在未引用尾部字节")
    if expected_offset != payload_bytes or observed_tensors != tensor_count:
        raise CheckpointError("payload tensor 覆盖/count 不闭合")

    from .executor import DecoderState

    state = DecoderState(
        position=position,
        input_token_id=input_token_id,
        layer_states=layer_states,
        committed_tokens=[dict(row) if isinstance(row, Mapping) else row for row in committed],
        forced_queue=forced_queue,
    )
    validate_decoder_state(state, profile=profile)
    return state, {
        "format": CHECKPOINT_FORMAT,
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "checkpoint_sha256": checkpoint_sha256,
        "payload": str(payload_path),
        "payload_sha256": payload_sha256,
        "payload_bytes": payload_bytes,
        "tensor_count": tensor_count,
        "position": position,
        "provenance": _json_object(document.get("provenance"), "checkpoint provenance"),
    }


__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointError",
    "catalog_fingerprint",
    "load_decoder_checkpoint",
    "save_decoder_checkpoint",
    "validate_decoder_state",
]
