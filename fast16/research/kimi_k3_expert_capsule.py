"""Range-extract and decode one Kimi K3 MXFP4 routed expert.

The planning and decoding paths are strictly local.  Only the ``extract``
subcommand performs an HTTP request, and it refuses a non-206 response so a
server cannot silently return the multi-gigabyte Safetensors shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

from remote_neural_biopsy import RemoteSafetensors, safe_slug


RESEARCH = Path(__file__).resolve().parent
DEFAULT_CACHE = RESEARCH / "biopsy_cache"
DEFAULT_REPO = "moonshotai/Kimi-K3"
DEFAULT_REVISION = "master"
GROUP_SIZE = 32
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
DECODER_CONTRACT = "colorlm-kimi-k3-mxfp4-decoder-v1"

# OCP MXFP4 E2M1 values.  A scale byte s multiplies the value by 2**(s - 127).
MXFP4_VALUES = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

COMPONENTS = (
    ("w1", "weight_packed", "gate"),
    ("w1", "weight_scale", "gate"),
    ("w2", "weight_packed", "down"),
    ("w2", "weight_scale", "down"),
    ("w3", "weight_packed", "up"),
    ("w3", "weight_scale", "up"),
)
ROLE_TO_WEIGHT = {"gate": "w1", "up": "w3", "down": "w2"}


def model_cache_dir(cache_dir: Path, repo: str, revision: str) -> Path:
    return cache_dir / safe_slug(repo) / safe_slug(revision)


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"本地缓存缺失: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON顶层必须是对象: {path}")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def plan_identity(plan: dict) -> dict:
    return {
        key: plan.get(key)
        for key in ("repo", "revision", "layer", "expert", "source_shard", "http_range")
    }


def lock_source_plan(output_dir: Path, plan: dict) -> Path:
    path = output_dir / "source-plan.json"
    if path.is_file():
        existing = read_json(path)
        if plan_identity(existing) != plan_identity(plan):
            raise RuntimeError(
                "输出目录已绑定到另一个K3专家或Range，请更换目录: "
                f"{output_dir}"
            )
        return path

    occupied = [
        output_dir / "expert.mxfp4.bin",
        output_dir / "expert.mxfp4.bin.part",
        output_dir / "expert.mxfp4.bin.part.json",
        output_dir / "expert.mxfp4.bin.source.json",
        output_dir / "gate.f16.npy",
        output_dir / "up.f16.npy",
        output_dir / "down.f16.npy",
        output_dir / "gate.f32.npy",
        output_dir / "up.f32.npy",
        output_dir / "down.f32.npy",
    ]
    if any(candidate.exists() for candidate in occupied):
        raise RuntimeError(f"输出目录有数据但缺少source-plan.json: {output_dir}")
    write_json(path, plan)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_content_range(value: str, expected_start: int, expected_end: int) -> int:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
    if (
        match is None
        or int(match.group(1)) != expected_start
        or int(match.group(2)) != expected_end
    ):
        raise RuntimeError(f"Content-Range不匹配: {value}")
    return int(match.group(3))


def response_validator(headers) -> dict | None:
    etag = headers.get("ETag")
    if etag and not etag.startswith("W/"):
        return {"header": "ETag", "value": etag}
    last_modified = headers.get("Last-Modified")
    if last_modified:
        return {"header": "Last-Modified", "value": last_modified}
    return None


def tensor_name(layer: int, expert: int, weight: str, component: str) -> str:
    return (
        f"language_model.model.layers.{layer}.block_sparse_moe."
        f"experts.{expert}.{weight}.{component}"
    )


def build_local_plan(
    cache_dir: Path,
    repo: str,
    revision: str,
    layer: int,
    expert: int,
) -> dict:
    if layer < 1:
        raise ValueError("layer必须大于等于1")
    if expert < 0:
        raise ValueError("expert必须大于等于0")

    local = model_cache_dir(cache_dir, repo, revision)
    index_path = local / "model.safetensors.index.json"
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"模型索引缺少weight_map: {index_path}")

    names = [tensor_name(layer, expert, weight, part) for weight, part, _ in COMPONENTS]
    missing = [name for name in names if name not in weight_map]
    if missing:
        raise RuntimeError(
            f"第{layer}层专家{expert}的MXFP4张量不完整; 缺失: {missing[0]}"
        )

    shards = {str(weight_map[name]) for name in names}
    if len(shards) != 1:
        raise RuntimeError(f"同一专家横跨多个分片，本工具拒绝合并: {sorted(shards)}")
    shard = next(iter(shards))
    header_path = local / "headers" / f"{safe_slug(shard)}.json"
    header = read_json(header_path)
    header_bytes = int(header.get("header_bytes", -1))
    tensors = header.get("tensors")
    if header_bytes <= 2 or not isinstance(tensors, dict):
        raise RuntimeError(f"Safetensors头缓存无效: {header_path}")

    data_start = 8 + header_bytes
    entries = []
    for (weight, part, role), name in zip(COMPONENTS, names, strict=True):
        item = tensors.get(name)
        if not isinstance(item, dict):
            raise RuntimeError(f"分片头中缺失张量: {name}")
        dtype = str(item.get("dtype"))
        shape = [int(value) for value in item.get("shape", [])]
        offsets = [int(value) for value in item.get("data_offsets", [])]
        if dtype != "U8" or len(shape) != 2 or len(offsets) != 2:
            raise RuntimeError(f"K3 MXFP4张量布局异常: {name} {item}")
        byte_count = math.prod(shape)
        if offsets[1] - offsets[0] != byte_count:
            raise RuntimeError(f"张量字节数与U8 shape不匹配: {name}")
        entries.append(
            {
                "name": name,
                "weight": weight,
                "role": role,
                "component": part,
                "dtype": dtype,
                "shape": shape,
                "bytes": byte_count,
                "absolute_start": data_start + offsets[0],
                "absolute_end": data_start + offsets[1],
            }
        )

    by_weight = {}
    for entry in entries:
        by_weight.setdefault(entry["weight"], {})[entry["component"]] = entry
    matrix_shapes = {}
    for weight, role in (("w1", "gate"), ("w2", "down"), ("w3", "up")):
        packed = by_weight[weight]["weight_packed"]
        scale = by_weight[weight]["weight_scale"]
        rows, packed_columns = packed["shape"]
        scale_rows, groups = scale["shape"]
        if rows != scale_rows or packed_columns * 2 != groups * GROUP_SIZE:
            raise RuntimeError(
                f"{weight}的packed/scale布局不是group_size={GROUP_SIZE}: "
                f"{packed['shape']} vs {scale['shape']}"
            )
        matrix_shapes[role] = [rows, packed_columns * 2]

    if matrix_shapes["gate"] != matrix_shapes["up"]:
        raise RuntimeError("K3 gate/up形状不一致")
    if matrix_shapes["down"] != list(reversed(matrix_shapes["gate"])):
        raise RuntimeError("K3 down形状不是gate的转置形状")

    ordered = sorted(entries, key=lambda item: item["absolute_start"])
    for left, right in zip(ordered, ordered[1:]):
        if left["absolute_end"] != right["absolute_start"]:
            raise RuntimeError(
                "六个专家张量不连续，拒绝把无关数据包进Range: "
                f"{left['name']} -> {right['name']}"
            )

    span_start = ordered[0]["absolute_start"]
    span_end = ordered[-1]["absolute_end"]
    for entry in entries:
        entry["capsule_offset_start"] = entry["absolute_start"] - span_start
        entry["capsule_offset_end"] = entry["absolute_end"] - span_start

    download_bytes = span_end - span_start
    f16_bytes = sum(math.prod(shape) * 2 for shape in matrix_shapes.values())
    f32_bytes = sum(math.prod(shape) * 4 for shape in matrix_shapes.values())
    return {
        "format": "colorlm-kimi-k3-expert-range-plan-v1",
        "repo": repo,
        "revision": revision,
        "layer": layer,
        "expert": expert,
        "source_shard": shard,
        "header_cache": os.fspath(header_path),
        "header_bytes": header_bytes,
        "header_tensors_sha256": canonical_json_sha256(tensors),
        "http_range": {
            "start": span_start,
            "end_inclusive": span_end - 1,
            "bytes": download_bytes,
        },
        "download_bytes": download_bytes,
        "download_mib": download_bytes / 1024**2,
        "group_size": GROUP_SIZE,
        "quantization": "MXFP4 E2M1 + E8M0 scale",
        "nibble_order": "low_then_high_interleaved",
        "scale_equation": "value * 2**(scale_u8 - 127)",
        "matrix_mapping": {"w1": "gate", "w3": "up", "w2": "down"},
        "matrix_orientation": "pytorch-linear-out-in",
        "latent_input_width": matrix_shapes["gate"][1],
        "intermediate_width": matrix_shapes["gate"][0],
        "activation": {
            "name": "situ",
            "beta": SITU_BETA,
            "linear_beta": SITU_LINEAR_BETA,
            "equation": "beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)",
        },
        "matrix_shapes": matrix_shapes,
        "decoded_f16_bytes": f16_bytes,
        "decoded_f32_bytes": f32_bytes,
        "tensors": entries,
    }


def default_output_dir(cache_dir: Path, repo: str, revision: str, layer: int, expert: int) -> Path:
    return (
        model_cache_dir(cache_dir, repo, revision)
        / "expert_capsules"
        / f"layer-{layer:02d}"
        / f"expert-{expert:03d}"
    )


def validate_remote_header(
    plan: dict,
    remote: RemoteSafetensors,
) -> dict:
    expected = 8 + int(plan["header_bytes"])
    end = expected - 1
    response = remote.session.get(
        remote.file_url(plan["source_shard"]),
        headers={"Range": f"bytes=0-{end}", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(15, 60),
    )
    try:
        if response.status_code != 206:
            raise RuntimeError(
                f"远端header校验未返回HTTP 206: {response.status_code}"
            )
        shard_bytes = parse_content_range(
            response.headers.get("Content-Range", ""),
            0,
            end,
        )
        validator = response_validator(response.headers)
        if validator is None:
            raise RuntimeError(
                "远端分片未提供强ETag或Last-Modified，"
                "无法绑定缓存header与专家Range"
            )
        raw = bytearray()
        for chunk in response.iter_content(1024 * 1024):
            if not chunk:
                continue
            if len(raw) + len(chunk) > expected:
                raise RuntimeError("远端header响应超过声明长度")
            raw.extend(chunk)
    finally:
        response.close()

    if len(raw) != expected:
        raise RuntimeError(f"远端header长度不匹配: {len(raw)} vs {expected}")
    remote_header_bytes = struct.unpack("<Q", raw[:8])[0]
    if remote_header_bytes != int(plan["header_bytes"]):
        raise RuntimeError(
            f"远端header长度已变化: {remote_header_bytes} vs {plan['header_bytes']}"
        )
    parsed = json.loads(bytes(raw[8:]).decode("utf-8"))
    remote_tensors = {name: value for name, value in parsed.items() if name != "__metadata__"}
    remote_hash = canonical_json_sha256(remote_tensors)
    if remote_hash != plan["header_tensors_sha256"]:
        raise RuntimeError("远端分片header与本地缓存不是同一版本")
    return {
        "validator": validator,
        "shard_bytes": shard_bytes,
        "header_tensors_sha256": remote_hash,
        "header_validation_bytes": expected,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
    }


def download_range(
    plan: dict,
    cache_dir: Path,
    output: Path,
    max_download_bytes: int,
    min_free_bytes: int,
) -> dict:
    expected = int(plan["download_bytes"])
    if expected > max_download_bytes:
        raise RuntimeError(
            f"计划下载{expected / 1024**2:.3f}MiB，超过配额"
            f"{max_download_bytes / 1024**2:.3f}MiB"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial_meta_path = partial.with_suffix(partial.suffix + ".json")
    source_meta_path = output.with_suffix(output.suffix + ".source.json")
    if output.is_file():
        if output.stat().st_size != expected:
            raise RuntimeError(f"已有原始胶囊大小不匹配: {output}")
        source_meta = read_json(source_meta_path)
        if source_meta.get("plan_identity") != plan_identity(plan):
            raise RuntimeError(f"已有原始胶囊的来源身份不匹配: {output}")
        digest = sha256_file(output)
        if source_meta.get("raw_sha256") != digest:
            raise RuntimeError(f"已有原始胶囊的SHA-256不匹配: {output}")
        partial_meta_path.unlink(missing_ok=True)
        return {
            "path": output.name,
            "bytes": expected,
            "sha256": digest,
            "downloaded": False,
            "source_validator": source_meta.get("validator"),
            "source_shard_bytes": source_meta.get("shard_bytes"),
        }

    completed = partial.stat().st_size if partial.is_file() else 0
    if completed > expected:
        raise RuntimeError(f"临时文件大于计划Range: {partial}")
    remaining = expected - completed
    used_network = remaining > 0
    resume_meta = None
    if completed:
        resume_meta = read_json(partial_meta_path)
        if resume_meta.get("plan_identity") != plan_identity(plan):
            raise RuntimeError(f"断点续传元数据与当前计划不匹配: {partial}")
        if remaining and not resume_meta.get("validator"):
            raise RuntimeError(
                f"服务端未提供ETag/Last-Modified，无法安全续传: {partial}"
            )
    free = shutil.disk_usage(output.parent).free
    if free - remaining < min_free_bytes:
        raise RuntimeError(
            f"下载后剩余空间将低于{min_free_bytes / 1024**3:.1f}GiB安全线"
        )

    if remaining:
        remote = RemoteSafetensors(plan["repo"], plan["revision"], cache_dir)
        source_guard = validate_remote_header(plan, remote)
        if resume_meta is not None:
            if resume_meta.get("validator") != source_guard["validator"]:
                raise RuntimeError("断点续传的远端分片版本已变化")
            if resume_meta.get("shard_bytes") != source_guard["shard_bytes"]:
                raise RuntimeError("断点续传的远端分片长度已变化")
        range_start = int(plan["http_range"]["start"]) + completed
        range_end = int(plan["http_range"]["end_inclusive"])
        request_headers = {
            "Range": f"bytes={range_start}-{range_end}",
            "Accept-Encoding": "identity",
            "If-Range": str(source_guard["validator"]["value"]),
        }
        response = remote.session.get(
            remote.file_url(plan["source_shard"]),
            headers=request_headers,
            stream=True,
            timeout=(15, 120),
        )
        if response.status_code != 206:
            response.close()
            raise RuntimeError(
                f"服务端未执行Range请求，已中止以避免整分片下载: "
                f"HTTP {response.status_code}"
            )
        try:
            shard_bytes = parse_content_range(
                response.headers.get("Content-Range", ""),
                range_start,
                range_end,
            )
        except RuntimeError:
            response.close()
            raise
        if shard_bytes != source_guard["shard_bytes"]:
            response.close()
            raise RuntimeError("header校验与专家Range的分片总长度不一致")
        current_validator = response.headers.get(source_guard["validator"]["header"])
        if current_validator != source_guard["validator"]["value"]:
            response.close()
            raise RuntimeError("header校验与专家Range的远端验证器不一致")
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) != remaining:
            response.close()
            raise RuntimeError(
                f"Content-Length与Range长度不匹配: {content_length} vs {remaining}"
            )

        response_meta = {
            "format": "colorlm-kimi-k3-range-source-v1",
            "plan_identity": plan_identity(plan),
            "etag": source_guard["etag"],
            "last_modified": source_guard["last_modified"],
            "validator": source_guard["validator"],
            "shard_bytes": shard_bytes,
            "header_tensors_sha256": source_guard["header_tensors_sha256"],
            "header_validation_bytes": source_guard["header_validation_bytes"],
        }
        if resume_meta is not None:
            if resume_meta.get("shard_bytes") != shard_bytes:
                response.close()
                raise RuntimeError("续传时分片总长度发生变化，已中止")
            validator_header = str(resume_meta["validator"]["header"])
            current_validator = response.headers.get(validator_header)
            if current_validator and current_validator != resume_meta["validator"]["value"]:
                response.close()
                raise RuntimeError("续传时远端分片验证器发生变化，已中止")
            response_meta = resume_meta
        else:
            write_json(partial_meta_path, response_meta)

        written = completed
        with partial.open("ab") as stream:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    if written + len(chunk) > expected:
                        response.close()
                        raise RuntimeError("Range响应超过声明长度，已中止")
                    stream.write(chunk)
                    written += len(chunk)
        response.close()
        resume_meta = response_meta
    if partial.stat().st_size != expected:
        raise RuntimeError(
            f"Range下载字节数不匹配: {partial.stat().st_size} vs {expected}"
        )
    if resume_meta is None:
        resume_meta = read_json(partial_meta_path)
    digest = sha256_file(partial)
    resume_meta["raw_sha256"] = digest
    write_json(source_meta_path, resume_meta)
    partial.replace(output)
    partial_meta_path.unlink(missing_ok=True)

    return {
        "path": output.name,
        "bytes": expected,
        "sha256": digest,
        "downloaded": used_network,
        "source_validator": resume_meta.get("validator"),
        "source_shard_bytes": resume_meta.get("shard_bytes"),
    }


def decode_mxfp4_chunk(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if packed.ndim != 2 or scales.ndim != 2:
        raise ValueError("packed和scales必须是二维数组")
    rows, packed_columns = packed.shape
    scale_rows, groups = scales.shape
    if rows != scale_rows or packed_columns * 2 != groups * GROUP_SIZE:
        raise ValueError(
            f"MXFP4 packed/scale形状不匹配: {packed.shape} vs {scales.shape}"
        )
    if np.any(np.asarray(scales, dtype=np.uint8) == 0xFF):
        raise ValueError("E8M0 scale包含0xFF NaN编码")

    blocks = np.asarray(packed, dtype=np.uint8).reshape(rows, groups, GROUP_SIZE // 2)
    decoded = np.empty((rows, groups, GROUP_SIZE), dtype=np.float32)
    decoded[..., 0::2] = MXFP4_VALUES[blocks & 0x0F]
    decoded[..., 1::2] = MXFP4_VALUES[blocks >> 4]
    exponents = np.asarray(scales, dtype=np.int16) - 127
    np.ldexp(decoded, exponents[..., None], out=decoded)
    return decoded.reshape(rows, groups * GROUP_SIZE)


def find_component(plan: dict, weight: str, component: str) -> dict:
    matches = [
        item
        for item in plan["tensors"]
        if item["weight"] == weight and item["component"] == component
    ]
    if len(matches) != 1:
        raise RuntimeError(f"计划中{weight}.{component}数量异常: {len(matches)}")
    return matches[0]


def raw_component(raw: np.memmap, entry: dict) -> np.ndarray:
    start = int(entry["capsule_offset_start"])
    end = int(entry["capsule_offset_end"])
    shape = tuple(int(value) for value in entry["shape"])
    view = raw[start:end]
    if view.size != math.prod(shape):
        raise RuntimeError(f"原始胶囊中张量长度异常: {entry['name']}")
    return view.reshape(shape)


def decode_capsule(
    plan: dict,
    raw_path: Path,
    output_dir: Path,
    output_dtype: str,
    chunk_rows: int,
    min_free_bytes: int = 0,
) -> dict:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows必须大于0")
    dtype = np.dtype(output_dtype)
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("只支持float16或float32输出")
    if raw_path.stat().st_size != int(plan["download_bytes"]):
        raise RuntimeError(f"原始胶囊大小与计划不一致: {raw_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    decoded_bytes = sum(
        math.prod(shape) * dtype.itemsize
        for shape in plan["matrix_shapes"].values()
    )
    free = shutil.disk_usage(output_dir).free
    if free - decoded_bytes < min_free_bytes:
        raise RuntimeError(
            f"解码后剩余空间将低于{min_free_bytes / 1024**3:.1f}GiB安全线"
        )
    raw_sha256 = sha256_file(raw_path)
    raw = np.memmap(raw_path, dtype=np.uint8, mode="r")
    matrices = {}
    suffix = "f16" if dtype == np.dtype("float16") else "f32"

    for role in ("gate", "up", "down"):
        weight = ROLE_TO_WEIGHT[role]
        packed_entry = find_component(plan, weight, "weight_packed")
        scale_entry = find_component(plan, weight, "weight_scale")
        packed = raw_component(raw, packed_entry)
        scales = raw_component(raw, scale_entry)
        shape = tuple(int(value) for value in plan["matrix_shapes"][role])
        output = output_dir / f"{role}.{suffix}.npy"

        partial = output.with_suffix(output.suffix + ".part")
        partial.unlink(missing_ok=True)
        destination = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        minimum = math.inf
        maximum = -math.inf
        square_sum = 0.0
        nonzero = 0
        for first in range(0, shape[0], chunk_rows):
            stop = min(first + chunk_rows, shape[0])
            decoded = decode_mxfp4_chunk(packed[first:stop], scales[first:stop])
            if not np.isfinite(decoded).all():
                raise RuntimeError(f"{role}解码出现NaN/Inf")
            if dtype == np.dtype("float16") and np.max(np.abs(decoded)) > np.finfo(np.float16).max:
                raise RuntimeError(f"{role}数值超出float16范围，请使用--dtype float32")
            destination[first:stop] = decoded.astype(dtype, copy=False)
            minimum = min(minimum, float(decoded.min()))
            maximum = max(maximum, float(decoded.max()))
            values64 = decoded.astype(np.float64, copy=False)
            square_sum += float(np.square(values64).sum())
            nonzero += int(np.count_nonzero(decoded))
        destination.flush()
        del destination
        partial.replace(output)
        stats = {
            "min": minimum,
            "max": maximum,
            "rms": math.sqrt(square_sum / math.prod(shape)),
            "nonzero": nonzero,
        }

        matrices[role] = {
            "source_weight": weight,
            "file": output.name,
            "dtype": dtype.name,
            "shape": list(shape),
            "orientation": "pytorch-linear-out-in",
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "raw_sha256": raw_sha256,
            "decoder_contract": DECODER_CONTRACT,
            "decode_stats": stats,
        }

    del raw
    return matrices


def build_manifest(plan: dict, raw_report: dict, matrices: dict | None) -> dict:
    manifest = {
        "format": "colorlm-kimi-k3-expert-capsule-v1",
        "repo": plan["repo"],
        "revision": plan["revision"],
        "layer": plan["layer"],
        "expert": plan["expert"],
        "source_shard": plan["source_shard"],
        "source_http_range": plan["http_range"],
        "source_quantization": plan["quantization"],
        "group_size": plan["group_size"],
        "nibble_order": plan["nibble_order"],
        "scale_equation": plan["scale_equation"],
        "raw_capsule": raw_report,
        "matrix_mapping": plan["matrix_mapping"],
        "matrix_orientation": plan["matrix_orientation"],
        "latent_input_width": plan["latent_input_width"],
        "intermediate_width": plan["intermediate_width"],
        "activation": plan["activation"],
        "computation_contract": (
            "down @ ((beta*tanh((gate@x)/beta)*sigmoid(gate@x)) * "
            "(linear_beta*tanh((up@x)/linear_beta)))"
        ),
        "requires_outer_latent_bridge": True,
        "matrices": matrices,
    }
    return manifest


def run_self_test() -> dict:
    packed = np.asarray(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2],
        dtype=np.uint8,
    )
    scales = np.asarray([[127]], dtype=np.uint8)
    decoded = decode_mxfp4_chunk(packed, scales)
    expected_first = np.asarray(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=np.float32,
    )
    expected = np.concatenate((expected_first, expected_first))[None, :]
    np.testing.assert_array_equal(decoded, expected)
    doubled = decode_mxfp4_chunk(packed, np.asarray([[128]], dtype=np.uint8))
    np.testing.assert_array_equal(doubled, expected * 2.0)

    header_tensors = {
        "test.tensor": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}
    }
    header_json = json.dumps(header_tensors, separators=(",", ":")).encode("utf-8")
    header_raw = struct.pack("<Q", len(header_json)) + header_json

    class FakeResponse:
        status_code = 206
        headers = {
            "Content-Range": f"bytes 0-{len(header_raw) - 1}/999",
            "ETag": '"header-etag"',
        }

        def iter_content(self, _chunk_size):
            yield header_raw

        def close(self):
            return None

    class FakeSession:
        def get(self, *_args, **_kwargs):
            return FakeResponse()

    class FakeRemote:
        session = FakeSession()

        def file_url(self, path):
            return path

    header_guard = validate_remote_header(
        {
            "header_bytes": len(header_json),
            "header_tensors_sha256": canonical_json_sha256(header_tensors),
            "source_shard": "test.safetensors",
        },
        FakeRemote(),
    )
    if header_guard["validator"]["value"] != '"header-etag"':
        raise AssertionError("远端header同版校验失败")

    with tempfile.TemporaryDirectory(prefix="kimi-k3-capsule-test-") as directory:
        temporary = Path(directory)

        # A fully received .part file must be finalized without touching the network.
        resume_output = temporary / "resume.bin"
        resume_plan = {
            "download_bytes": 3,
            "repo": DEFAULT_REPO,
            "revision": DEFAULT_REVISION,
            "layer": 92,
            "expert": 0,
            "source_shard": "test.safetensors",
            "http_range": {"start": 10, "end_inclusive": 12, "bytes": 3},
        }
        resume_output.with_suffix(".bin.part").write_bytes(b"abc")
        write_json(
            resume_output.with_suffix(".bin.part.json"),
            {
                "format": "colorlm-kimi-k3-range-source-v1",
                "plan_identity": plan_identity(resume_plan),
                "etag": '"test-etag"',
                "last_modified": None,
                "validator": {"header": "ETag", "value": '"test-etag"'},
                "shard_bytes": 99,
            },
        )
        resume_report = download_range(
            resume_plan,
            temporary,
            resume_output,
            max_download_bytes=3,
            min_free_bytes=0,
        )
        if resume_output.read_bytes() != b"abc" or resume_report["downloaded"] is not False:
            raise AssertionError("已完整的续传文件未正确收尾")
        reused_report = download_range(
            resume_plan,
            temporary,
            resume_output,
            max_download_bytes=3,
            min_free_bytes=0,
        )
        if reused_report["sha256"] != resume_report["sha256"]:
            raise AssertionError("原始胶囊SHA-256复用校验失败")

        identity_dir = temporary / "identity"
        identity_dir.mkdir()
        identity_plan = {
            "repo": DEFAULT_REPO,
            "revision": DEFAULT_REVISION,
            "layer": 92,
            "expert": 0,
            "source_shard": "test.safetensors",
            "http_range": {"start": 1, "end_inclusive": 2, "bytes": 2},
        }
        lock_source_plan(identity_dir, identity_plan)
        conflicting = dict(identity_plan)
        conflicting["expert"] = 1
        try:
            lock_source_plan(identity_dir, conflicting)
        except RuntimeError:
            pass
        else:
            raise AssertionError("输出目录未拒绝冲突的专家计划")

        # Exercise the file-level chunk decoder and NPY metadata with a tiny capsule.
        tensors = []
        raw_parts = []
        offset = 0
        for weight, role in (("w1", "gate"), ("w2", "down"), ("w3", "up")):
            for component, values, shape in (
                ("weight_packed", packed.tobytes(), [1, 16]),
                ("weight_scale", scales.tobytes(), [1, 1]),
            ):
                raw_parts.append(values)
                tensors.append(
                    {
                        "weight": weight,
                        "role": role,
                        "component": component,
                        "name": f"test.{weight}.{component}",
                        "shape": shape,
                        "capsule_offset_start": offset,
                        "capsule_offset_end": offset + len(values),
                    }
                )
                offset += len(values)
        raw_path = temporary / "expert.mxfp4.bin"
        raw_path.write_bytes(b"".join(raw_parts))
        tiny_plan = {
            "download_bytes": offset,
            "matrix_shapes": {"gate": [1, 32], "up": [1, 32], "down": [1, 32]},
            "tensors": tensors,
        }
        matrices = decode_capsule(
            tiny_plan,
            raw_path,
            temporary,
            output_dtype="float16",
            chunk_rows=1,
        )
        for role in ("gate", "up", "down"):
            matrix = np.load(temporary / matrices[role]["file"], allow_pickle=False)
            np.testing.assert_array_equal(matrix.astype(np.float32), expected)

    return {
        "ok": True,
        "decoded_shape": list(decoded.shape),
        "nibble_order": "low_then_high_interleaved",
        "scale_127": 1.0,
        "scale_128": 2.0,
        "remote_header_binding": True,
        "resume_finalize": True,
        "raw_sha256_binding": True,
        "source_plan_lock": True,
        "npy_roundtrip": True,
    }


def add_expert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--layer", type=int, default=92)
    parser.add_argument("--expert", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kimi K3 MXFP4专家Range活检与解码")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="纯离线计算专家张量与HTTP Range")
    add_expert_arguments(plan)
    plan.add_argument("--output", type=Path)

    extract = commands.add_parser("extract", help="一次Range下载专家并默认解码")
    add_expert_arguments(extract)
    extract.add_argument("--output-dir", type=Path)
    extract.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    extract.add_argument("--chunk-rows", type=int, default=128)
    extract.add_argument("--raw-only", action="store_true")
    extract.add_argument("--max-download-mib", type=float, default=32.0)
    extract.add_argument("--min-free-gib", type=float, default=2.0)

    decode = commands.add_parser("decode", help="将已下载的原始MXFP4胶囊离线解码")
    decode.add_argument("--capsule-dir", type=Path, required=True)
    decode.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    decode.add_argument("--chunk-rows", type=int, default=128)
    decode.add_argument("--min-free-gib", type=float, default=2.0)

    commands.add_parser("self-test", help="用已知E2M1/E8M0向量做零网络自检")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "self-test":
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "decode":
        plan_path = args.capsule_dir / "source-plan.json"
        raw_path = args.capsule_dir / "expert.mxfp4.bin"
        plan = read_json(plan_path)
        source_meta = read_json(raw_path.with_suffix(raw_path.suffix + ".source.json"))
        raw_sha256 = sha256_file(raw_path)
        if source_meta.get("raw_sha256") != raw_sha256:
            raise RuntimeError(f"原始胶囊的SHA-256与来源元数据不一致: {raw_path}")
        raw_report = {
            "path": raw_path.name,
            "bytes": raw_path.stat().st_size,
            "sha256": raw_sha256,
            "downloaded": False,
            "source_validator": source_meta.get("validator"),
            "source_shard_bytes": source_meta.get("shard_bytes"),
        }
        matrices = decode_capsule(
            plan,
            raw_path,
            args.capsule_dir,
            args.dtype,
            args.chunk_rows,
            int(args.min_free_gib * 1024**3),
        )
        manifest = build_manifest(plan, raw_report, matrices)
        write_json(args.capsule_dir / "capsule.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    plan = build_local_plan(
        args.cache_dir,
        args.repo,
        args.revision,
        args.layer,
        args.expert,
    )
    if args.command == "plan":
        if args.output is not None:
            write_json(args.output, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = default_output_dir(
            args.cache_dir,
            args.repo,
            args.revision,
            args.layer,
            args.expert,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_source_plan(output_dir, plan)
    raw_path = output_dir / "expert.mxfp4.bin"
    raw_report = download_range(
        plan,
        args.cache_dir,
        raw_path,
        int(args.max_download_mib * 1024**2),
        int(args.min_free_gib * 1024**3),
    )
    matrices = None
    if not args.raw_only:
        matrices = decode_capsule(
            plan,
            raw_path,
            output_dir,
            args.dtype,
            args.chunk_rows,
            int(args.min_free_gib * 1024**3),
        )
    manifest = build_manifest(plan, raw_report, matrices)
    write_json(output_dir / "capsule.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
