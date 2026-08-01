#!/usr/bin/env python3
"""Polaris S14 route-first 在线 Range 状态机。

本模块保留 :mod:`range_pack` 的完整 route trace/batch v1，不改变其格式或语义。
``catalog`` 子命令只读取固定 revision 的本地 index/header/manifest，绝不请求或下载
payload。真正的 Range 读取只通过显式构造 ``RangeCache(allow_fetch=True, ...)`` 的
Python 调用发生，便于 runtime 注入 executor 和预算。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Protocol

try:
    from . import range_pack as rp
except ImportError:  # 兼容直接执行 online_range.py 的 CLI 入口。
    import range_pack as rp


CATALOG_FORMAT = "polaris-s14-route-first-catalog-v1"
CACHE_META_FORMAT = "polaris-s14-range-cache-entry-v1"
AUTHORITATIVE_LOCK_FORMAT = "polaris-s14-authoritative-range-hash-lock-v1"
LOCAL_METADATA_FORMAT = "polaris-s14-local-metadata-v1"
SKELETON_FORMAT = "polaris-s14-local-skeleton-ranges-v1"
EXPERT_COUNT = 256
TOP_K = 6
CONTENT_RANGE_RE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return rp.sha256_bytes(_canonical_bytes(value))


def _safe_source_file(filename: str) -> str:
    """只允许冻结 contract 中使用的单层 basename，拒绝路径逃逸。"""
    if not isinstance(filename, str) or not filename or "\\" in filename or "\x00" in filename:
        raise rp.ContractError(f"非法 source file 路径：{filename!r}")
    parsed = PurePosixPath(filename)
    if parsed.is_absolute() or len(parsed.parts) != 1 or parsed.name in {".", ".."}:
        raise rp.ContractError(f"source file 必须是 basename：{filename!r}")
    return filename


def _safe_local_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise rp.ContractError(f"非法 metadata 本地路径：{relative!r}")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise rp.ContractError(f"metadata 路径越界：{relative!r}")
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*parsed.parts).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise rp.ContractError(f"metadata 路径越界：{relative!r}") from exc
    return resolved


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        json.dump(value, sink, ensure_ascii=False, indent=2)
        sink.write("\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)


def _header_set_identity(source_files: Mapping[str, Mapping[str, Any]]) -> str:
    frozen = {
        filename: {
            "file_bytes": int(row["file_bytes"]),
            "header_length": int(row["header_length"]),
            "data_start": int(row["data_start"]),
            "header_sha256": str(row["header_sha256"]),
            "tensor_table_sha256": str(row["tensor_table_sha256"]),
        }
        for filename, row in sorted(source_files.items())
    }
    return _canonical_sha256(frozen)


def validate_local_metadata(path: Path, index_path: Path) -> dict[str, Any]:
    """校验主线程已有的 local metadata v1；不访问网络。"""
    manifest = rp.read_json(path)
    if manifest.get("format") != LOCAL_METADATA_FORMAT:
        raise rp.ContractError("metadata_manifest format 不兼容")
    if manifest.get("repo") != rp.REPO or manifest.get("revision") != rp.REVISION:
        raise rp.ContractError("metadata_manifest repo/revision 不匹配")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise rp.ContractError("metadata_manifest.files 必须是非空数组")
    root = path.resolve().parent
    verified: list[dict[str, Any]] = []
    index_resolved = index_path.resolve()
    index_seen = False
    for raw in files:
        if not isinstance(raw, dict):
            raise rp.ContractError("metadata_manifest file row 必须是 object")
        name = str(raw.get("file", ""))
        local_path = _safe_local_path(root, name)
        if not local_path.is_file():
            raise rp.ContractError(f"metadata_manifest 本地文件缺失：{name}")
        size = int(raw.get("bytes", -1))
        digest = str(raw.get("sha256", ""))
        if size < 0 or not SHA256_RE.fullmatch(digest):
            raise rp.ContractError(f"metadata_manifest 字节/SHA 非法：{name}")
        if local_path.stat().st_size != size or rp.sha256_file(local_path) != digest:
            raise rp.ContractError(f"metadata_manifest 本地文件校验失败：{name}")
        if raw.get("expected_match") is not True:
            raise rp.ContractError(f"metadata_manifest 未确认 expected_match：{name}")
        if local_path == index_resolved:
            index_seen = True
        verified.append({"file": name, "bytes": size, "sha256": digest, "expected_match": True})
    if not index_seen:
        raise rp.ContractError("metadata_manifest 未覆盖传入的官方 index")
    if not isinstance(manifest.get("weights_downloaded"), bool):
        raise rp.ContractError("metadata_manifest.weights_downloaded 必须是 boolean")
    return {
        "format": LOCAL_METADATA_FORMAT,
        "manifest_sha256": rp.sha256_file(path),
        "weights_downloaded": manifest["weights_downloaded"],
        "files": verified,
    }


def _make_entry(
    *,
    name: str,
    filename: str,
    header: Mapping[str, Any],
    meta: Mapping[str, Any],
    kind: str,
    layer: int | None,
    expert_id: int | None = None,
) -> dict[str, Any]:
    filename = _safe_source_file(filename)
    begin, end = map(int, meta["data_offsets"])
    start = int(header["data_start"]) + begin
    end_inclusive = int(header["data_start"]) + end - 1
    entry: dict[str, Any] = {
        "tensor": name,
        "kind": kind,
        "layer": layer,
        "file": filename,
        "file_bytes": int(header["file_bytes"]),
        "header_tensor_table_sha256": str(header["tensor_table_sha256"]),
        "start": start,
        "end": end_inclusive,
        "bytes": end_inclusive - start + 1,
        "dtype": meta.get("dtype"),
        "shape": meta.get("shape"),
        "range_key": f"{filename}:{start}-{end_inclusive}",
    }
    if expert_id is not None:
        entry["expert_id"] = expert_id
    return entry


def _flatten_prerequisites(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = list(catalog["boundary"]["embedding"]) + list(catalog["boundary"]["final"])
    for layer in catalog["selected_layers"]:
        row = catalog["layers"][str(layer)]
        entries.extend(row["non_expert"])
        entries.extend(row["router"])
        entries.extend(row["shared"])
    return entries


def validate_skeleton_manifest(path: Path, catalog: Mapping[str, Any]) -> dict[str, Any]:
    """验证已有 509-range skeleton 与新 catalog 的先路由所需集合完全一致。

    skeleton 的 null ``expected_range_sha256`` 只表示尚未观测；本函数绝不把它升级成
    authoritative lock。
    """
    skeleton = rp.read_json(path)
    if skeleton.get("format") != SKELETON_FORMAT:
        raise rp.ContractError("s14_skeleton_tofu_manifest format 不兼容")
    if skeleton.get("repo") != rp.REPO or skeleton.get("revision") != rp.REVISION:
        raise rp.ContractError("skeleton repo/revision 不匹配")
    if skeleton.get("selected_layers") != catalog["selected_layers"]:
        raise rp.ContractError("skeleton S14 层集合漂移")
    if skeleton.get("download_authorized") is not False:
        raise rp.ContractError("skeleton 不能授予下载权限")
    integrity_mode = str(skeleton.get("integrity_mode", ""))
    if integrity_mode != "TOFU fixed revision; not an authoritative range hash lock":
        raise rp.ContractError("skeleton 必须明确标记 TOFU/non-authoritative")
    raw_ranges = skeleton.get("ranges")
    if not isinstance(raw_ranges, list):
        raise rp.ContractError("skeleton.ranges 必须是数组")
    expected = {
        (row["file"], row["tensor"]): row
        for row in _flatten_prerequisites(catalog)
    }
    seen: set[tuple[str, str]] = set()
    total = 0
    for raw in raw_ranges:
        if not isinstance(raw, dict):
            raise rp.ContractError("skeleton range row 必须是 object")
        key = (str(raw.get("file", "")), str(raw.get("tensor", "")))
        if key in seen or key not in expected:
            raise rp.ContractError(f"skeleton 含重复/非 route-first prerequisite：{key}")
        seen.add(key)
        row = expected[key]
        for field, wanted in (("start", row["start"]), ("end", row["end"]), ("bytes", row["bytes"])):
            if int(raw.get(field, -1)) != int(wanted):
                raise rp.ContractError(f"skeleton Range 与 header 不一致：{key} / {field}")
        compatible_kind = "boundary" if row["kind"] == "boundary" else "layer_non_routed"
        if raw.get("kind") != compatible_kind:
            raise rp.ContractError(f"skeleton kind 不兼容：{key}")
        if raw.get("expected_range_sha256") is not None:
            raise rp.ContractError("TOFU skeleton 的 range hash 不能冒充权威 lock")
        if str(raw.get("integrity", "")) != "tofu_fixed_revision_not_authoritative":
            raise rp.ContractError("skeleton range 缺少明确 TOFU/non-authoritative 标记")
        total += int(raw["bytes"])
    if seen != set(expected):
        missing = sorted(set(expected) - seen)[:5]
        raise rp.ContractError(f"skeleton 缺少 route-first prerequisite：{missing}")
    if int(skeleton.get("range_count", -1)) != len(raw_ranges) or int(skeleton.get("total_bytes", -1)) != total:
        raise rp.ContractError("skeleton 汇总字节/count 自相矛盾")
    return {
        "format": SKELETON_FORMAT,
        "manifest_sha256": rp.sha256_file(path),
        "range_count": len(raw_ranges),
        "total_bytes": total,
        "integrity": "tofu_fixed_revision_not_authoritative",
        "authoritative": False,
        "download_authorized": False,
    }


def build_catalog(
    source: Mapping[str, Any],
    index_path: Path,
    header_dir: Path,
    *,
    metadata_manifest_path: Path | None = None,
    skeleton_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """从固定 revision 的本地真实 header 枚举 route-first catalog。"""
    source_dict = dict(source)
    rp.validate_source_contract(source_dict)
    weight_map = rp.validate_index(index_path, source_dict)
    contracts = rp.source_file_contracts(source_dict)
    headers = rp.load_headers(header_dir, source_dict, set(contracts))
    source_files: dict[str, dict[str, Any]] = {}
    for filename, header in sorted(headers.items()):
        if not SHA256_RE.fullmatch(str(header.get("header_sha256", ""))):
            raise rp.ContractError(f"header 缺少有效 header_sha256：{filename}")
        source_files[filename] = {
            "file_bytes": int(header["file_bytes"]),
            "header_length": int(header["header_length"]),
            "data_start": int(header["data_start"]),
            "header_sha256": str(header["header_sha256"]),
            "tensor_table_sha256": str(header["tensor_table_sha256"]),
            "integrity": "tofu_fixed_revision_not_authoritative",
            "authoritative": False,
        }

    selected_layers = list(source_dict["selected_layers"])
    boundary: dict[str, list[dict[str, Any]]] = {"embedding": [], "final": []}
    layers: dict[str, dict[str, Any]] = {
        str(layer): {"non_expert": [], "router": [], "shared": [], "experts": {}}
        for layer in selected_layers
    }
    boundary_names = set(source_dict["boundary_tensors"])
    for name, filename in sorted(weight_map.items()):
        layer: int | None = None
        kind: str | None = None
        expert_id: int | None = None
        target: list[dict[str, Any]] | None = None
        if name in boundary_names:
            expected_file = source_dict["boundary_shards"][name]["file"]
            if filename != expected_file:
                raise rp.ContractError(f"boundary tensor shard 漂移：{name}")
            kind = "boundary"
            target = boundary["embedding" if name == "embed.weight" else "final"]
        else:
            layer_match = rp.LAYER_RE.match(name)
            if layer_match is None or int(layer_match.group(1)) not in selected_layers:
                continue
            layer = int(layer_match.group(1))
            expected_file = source_dict["layer_shards"][str(layer)]["file"]
            if filename != expected_file:
                raise rp.ContractError(f"L{layer} tensor shard 漂移：{name}")
            row = layers[str(layer)]
            if ".ffn.experts." in name:
                match = rp.EXPERT_RE.match(name)
                if match is None or int(match.group(1)) != layer:
                    raise rp.ContractError(f"无法从 header 精确解析 expert：{name}")
                expert_id = int(match.group(2))
                if not 0 <= expert_id < EXPERT_COUNT:
                    raise rp.ContractError(f"expert ID 越界：{name}")
                kind = "routed_expert"
                target = row["experts"].setdefault(str(expert_id), [])
            elif name.startswith(f"layers.{layer}.ffn.shared_experts."):
                kind = "shared"
                target = row["shared"]
            elif name.startswith(f"layers.{layer}.ffn.gate."):
                kind = "router"
                target = row["router"]
            else:
                kind = "non_expert"
                target = row["non_expert"]
        if filename not in headers:
            raise rp.ContractError(f"index 引用了未冻结 header：{filename}")
        meta = headers[filename]["tensors"].get(name)
        if not isinstance(meta, dict):
            raise rp.ContractError(f"header 中缺少 index tensor：{name}")
        assert kind is not None and target is not None
        target.append(_make_entry(
            name=name,
            filename=filename,
            header=headers[filename],
            meta=meta,
            kind=kind,
            layer=layer,
            expert_id=expert_id,
        ))

    if [row["tensor"] for row in boundary["embedding"]] != ["embed.weight"]:
        raise rp.ContractError("catalog 必须且只能有原生 embed.weight")
    required_final = {
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "norm.weight",
        "head.weight",
    }
    if {row["tensor"] for row in boundary["final"]} != required_final:
        raise rp.ContractError("catalog 缺少原生 HC head/norm/head")
    for layer in selected_layers:
        row = layers[str(layer)]
        if not row["non_expert"] or not row["router"] or not row["shared"]:
            raise rp.ContractError(f"L{layer} 缺少 non-expert/router/shared path")
        expert_ids = {int(value) for value in row["experts"]}
        if expert_ids != set(range(EXPERT_COUNT)):
            raise rp.ContractError(f"L{layer} header 未精确枚举 0..255 experts；禁止猜 ID")
        if any(not tensors for tensors in row["experts"].values()):
            raise rp.ContractError(f"L{layer} 含空 expert tensor 集合")

    all_entries = list(boundary["embedding"]) + list(boundary["final"])
    for layer in selected_layers:
        row = layers[str(layer)]
        all_entries.extend(row["non_expert"])
        all_entries.extend(row["router"])
        all_entries.extend(row["shared"])
        for expert_id in range(EXPERT_COUNT):
            all_entries.extend(row["experts"][str(expert_id)])
    catalog: dict[str, Any] = {
        "format": CATALOG_FORMAT,
        "repo": rp.REPO,
        "revision": rp.REVISION,
        "selected_layers": selected_layers,
        "top_k": TOP_K,
        "expert_id_range": [0, EXPERT_COUNT - 1],
        "download_authorized": False,
        "index": {
            "file": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": rp.sha256_file(index_path),
            "authoritative": True,
        },
        "headers": {
            "set_sha256": _header_set_identity(source_files),
            "integrity": "tofu_fixed_revision_not_authoritative",
            "authoritative": False,
            "files": source_files,
        },
        "boundary": boundary,
        "layers": layers,
        "summary": {
            "range_count": len(all_entries),
            "range_bytes": sum(int(row["bytes"]) for row in all_entries),
            "prerequisite_range_count": len(_flatten_prerequisites({
                "boundary": boundary,
                "selected_layers": selected_layers,
                "layers": layers,
            })),
            "route_policy": "current token/current layer exact top-6; no guessed expert",
        },
        "integrity_policy": {
            "first_observed_range_hash": "TOFU/non-authoritative",
            "formal_reproduction": AUTHORITATIVE_LOCK_FORMAT,
            "tofu_must_not_be_promoted_without_external_lock": True,
        },
    }
    if metadata_manifest_path is not None:
        catalog["local_metadata"] = validate_local_metadata(metadata_manifest_path, index_path)
    if skeleton_manifest_path is not None:
        catalog["skeleton_compatibility"] = validate_skeleton_manifest(skeleton_manifest_path, catalog)
    validate_catalog(catalog, source_dict)
    return catalog


def _iter_catalog_entries(catalog: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    yield from catalog["boundary"]["embedding"]
    yield from catalog["boundary"]["final"]
    for layer in catalog["selected_layers"]:
        row = catalog["layers"][str(layer)]
        yield from row["non_expert"]
        yield from row["router"]
        yield from row["shared"]
        for expert_id in range(EXPERT_COUNT):
            yield from row["experts"][str(expert_id)]


def validate_catalog(catalog: Mapping[str, Any], source: Mapping[str, Any] | None = None) -> None:
    source_dict = dict(source) if source is not None else rp.read_json(rp.SOURCE_CONTRACT)
    rp.validate_source_contract(source_dict)
    if catalog.get("format") != CATALOG_FORMAT:
        raise rp.ContractError("route-first catalog format 不兼容")
    if catalog.get("repo") != rp.REPO or catalog.get("revision") != rp.REVISION:
        raise rp.ContractError("route-first catalog repo/revision 不匹配")
    if catalog.get("selected_layers") != source_dict["selected_layers"]:
        raise rp.ContractError("route-first catalog S14 层集合漂移")
    if catalog.get("top_k") != TOP_K or catalog.get("expert_id_range") != [0, EXPERT_COUNT - 1]:
        raise rp.ContractError("route-first catalog router 契约漂移")
    if catalog.get("download_authorized") is not False:
        raise rp.ContractError("catalog 文件本身不得授予 payload 下载权限")
    index = catalog.get("index")
    if not isinstance(index, dict) or int(index.get("bytes", -1)) != int(source_dict["index"]["bytes"]):
        raise rp.ContractError("catalog index 字节不匹配")
    if index.get("sha256") != source_dict["index"]["sha256"] or index.get("authoritative") is not True:
        raise rp.ContractError("catalog index 不是冻结官方 lock")
    headers = catalog.get("headers")
    if not isinstance(headers, dict) or headers.get("authoritative") is not False:
        raise rp.ContractError("catalog header 必须明确为 fixed-revision TOFU/non-authoritative")
    if headers.get("integrity") != "tofu_fixed_revision_not_authoritative":
        raise rp.ContractError("catalog header integrity 标记错误")
    files = headers.get("files")
    if not isinstance(files, dict):
        raise rp.ContractError("catalog headers.files 必须是 object")
    contracts = rp.source_file_contracts(source_dict)
    if set(files) != set(contracts):
        raise rp.ContractError("catalog header file 集合不完整")
    for filename, row in files.items():
        _safe_source_file(filename)
        if not isinstance(row, dict) or int(row.get("file_bytes", -1)) != int(contracts[filename]["bytes"]):
            raise rp.ContractError(f"catalog source file 字节漂移：{filename}")
        if row.get("authoritative") is not False or row.get("integrity") != "tofu_fixed_revision_not_authoritative":
            raise rp.ContractError(f"catalog header 不得冒充权威：{filename}")
        for field in ("header_sha256", "tensor_table_sha256"):
            if not SHA256_RE.fullmatch(str(row.get(field, ""))):
                raise rp.ContractError(f"catalog header {field} 非法：{filename}")
        if int(row.get("data_start", -1)) != 8 + int(row.get("header_length", -9)):
            raise rp.ContractError(f"catalog header data_start 错误：{filename}")
    if headers.get("set_sha256") != _header_set_identity(files):
        raise rp.ContractError("catalog header set SHA-256 错误")

    boundary = catalog.get("boundary")
    layers = catalog.get("layers")
    if not isinstance(boundary, dict) or not isinstance(layers, dict):
        raise rp.ContractError("catalog boundary/layers 必须是 object")
    if set(layers) != {str(layer) for layer in source_dict["selected_layers"]}:
        raise rp.ContractError("catalog layers 集合不完整")
    if [row.get("tensor") for row in boundary.get("embedding", [])] != ["embed.weight"]:
        raise rp.ContractError("catalog embedding 边界错误")
    required_final = {
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "norm.weight",
        "head.weight",
    }
    if {row.get("tensor") for row in boundary.get("final", [])} != required_final:
        raise rp.ContractError("catalog final HC 边界错误")
    seen_tensors: set[str] = set()
    range_count = 0
    range_bytes = 0

    def validate_entry(row: Any, *, kind: str, layer: int | None, expert_id: int | None = None) -> None:
        nonlocal range_count, range_bytes
        if not isinstance(row, dict):
            raise rp.ContractError("catalog entry 必须是 object")
        name = str(row.get("tensor", ""))
        if not name or name in seen_tensors:
            raise rp.ContractError(f"catalog tensor 重复/为空：{name!r}")
        seen_tensors.add(name)
        if row.get("kind") != kind or row.get("layer") != layer:
            raise rp.ContractError(f"catalog entry 分类错误：{name}")
        if expert_id is not None and row.get("expert_id") != expert_id:
            raise rp.ContractError(f"catalog expert ID 与分组不符：{name}")
        filename = _safe_source_file(str(row.get("file", "")))
        if filename not in files:
            raise rp.ContractError(f"catalog entry 引用未冻结文件：{name}")
        expected_file = (
            source_dict["boundary_shards"][name]["file"]
            if layer is None
            else source_dict["layer_shards"][str(layer)]["file"]
        )
        if filename != expected_file:
            raise rp.ContractError(f"catalog entry shard 漂移：{name}")
        file_row = files[filename]
        start = int(row.get("start", -1))
        end = int(row.get("end", -1))
        size = int(row.get("bytes", -1))
        if int(row.get("file_bytes", -1)) != int(file_row["file_bytes"]):
            raise rp.ContractError(f"catalog entry file_bytes 漂移：{name}")
        if row.get("header_tensor_table_sha256") != file_row["tensor_table_sha256"]:
            raise rp.ContractError(f"catalog entry header identity 漂移：{name}")
        if not int(file_row["data_start"]) <= start <= end < int(file_row["file_bytes"]):
            raise rp.ContractError(f"catalog entry Range 越界：{name}")
        if size != end - start + 1 or row.get("range_key") != f"{filename}:{start}-{end}":
            raise rp.ContractError(f"catalog entry Range 自相矛盾：{name}")
        range_count += 1
        range_bytes += size

    for row in boundary["embedding"] + boundary["final"]:
        validate_entry(row, kind="boundary", layer=None)
    for layer in source_dict["selected_layers"]:
        layer_row = layers[str(layer)]
        if not isinstance(layer_row, dict) or set(layer_row) != {"non_expert", "router", "shared", "experts"}:
            raise rp.ContractError(f"L{layer} catalog 分组错误")
        for group in ("non_expert", "router", "shared"):
            if not isinstance(layer_row[group], list) or not layer_row[group]:
                raise rp.ContractError(f"L{layer} 缺少 {group}")
            for row in layer_row[group]:
                validate_entry(row, kind=group, layer=layer)
        experts = layer_row["experts"]
        if not isinstance(experts, dict) or set(experts) != {str(value) for value in range(EXPERT_COUNT)}:
            raise rp.ContractError(f"L{layer} 必须由 header 枚举全部 0..255 experts")
        for expert_id in range(EXPERT_COUNT):
            expert_rows = experts[str(expert_id)]
            if not isinstance(expert_rows, list) or not expert_rows:
                raise rp.ContractError(f"L{layer} E{expert_id} tensor 为空")
            for row in expert_rows:
                validate_entry(row, kind="routed_expert", layer=layer, expert_id=expert_id)
    summary = catalog.get("summary")
    if not isinstance(summary, dict) or int(summary.get("range_count", -1)) != range_count:
        raise rp.ContractError("catalog summary.range_count 错误")
    if int(summary.get("range_bytes", -1)) != range_bytes:
        raise rp.ContractError("catalog summary.range_bytes 错误")
    if int(summary.get("prerequisite_range_count", -1)) != len(_flatten_prerequisites(catalog)):
        raise rp.ContractError("catalog prerequisite range count 错误")


def load_authoritative_hash_lock(path: Path) -> dict[str, str]:
    """只接受显式 external authoritative lock；拒绝 TOFU skeleton/缓存元数据。"""
    lock = rp.read_json(path)
    if lock.get("format") != AUTHORITATIVE_LOCK_FORMAT:
        raise rp.ContractError("hash lock 必须是显式 authoritative 格式；TOFU manifest 不可代替")
    if lock.get("repo") != rp.REPO or lock.get("revision") != rp.REVISION:
        raise rp.ContractError("authoritative hash lock repo/revision 不匹配")
    if lock.get("authoritative") is not True or lock.get("hash_authority") != "official_lock":
        raise rp.ContractError("hash lock 缺少 official_lock/authoritative 证明")
    raw_ranges = lock.get("ranges")
    if not isinstance(raw_ranges, dict):
        raise rp.ContractError("authoritative hash lock.ranges 必须是 object")
    result: dict[str, str] = {}
    for key, digest in raw_ranges.items():
        if not isinstance(key, str) or not SHA256_RE.fullmatch(str(digest)):
            raise rp.ContractError(f"authoritative range lock 非法：{key!r}")
        result[key] = str(digest)
    return result


class ResponseLike(Protocol):
    status: int
    headers: Any

    def read(self, size: int = -1) -> bytes: ...
    def geturl(self) -> str: ...
    def __enter__(self) -> "ResponseLike": ...
    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...


class RangeTransport(Protocol):
    def open_range(self, url: str, start: int, end: int, timeout: float) -> ResponseLike: ...


class UrllibHttpsRangeTransport:
    """生产 HTTPS transport；最终重定向 URL 仍由 RangeCache 做 HTTPS 校验。"""

    def open_range(self, url: str, start: int, end: int, timeout: float) -> ResponseLike:
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        )
        return urllib.request.urlopen(request, timeout=timeout)  # type: ignore[return-value]


def _require_https(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise rp.ContractError(f"Range URL 必须是无 userinfo 的 HTTPS：{url!r}")


def _header(headers: Any, name: str) -> str | None:
    value = headers.get(name) if hasattr(headers, "get") else None
    if value is not None:
        return str(value)
    if isinstance(headers, Mapping):
        lowered = name.lower()
        for key, candidate in headers.items():
            if str(key).lower() == lowered:
                return str(candidate)
    return None


@dataclass(frozen=True)
class CachedRange:
    entry: Mapping[str, Any]
    path: Path
    proof: Mapping[str, Any]
    cache_hit: bool


_KEY_LOCKS_GUARD = threading.Lock()
_KEY_LOCKS: dict[tuple[str, str], threading.Lock] = {}


def _key_lock(cache_root: Path, key: str) -> threading.Lock:
    identity = (str(cache_root), key)
    with _KEY_LOCKS_GUARD:
        return _KEY_LOCKS.setdefault(identity, threading.Lock())


class RangeCache:
    """带 SHA proof、断点续传、预算和 keyed concurrency 的内容缓存。"""

    def __init__(
        self,
        cache_dir: Path,
        *,
        endpoint: str = "https://huggingface.co",
        transport: RangeTransport | None = None,
        allow_fetch: bool = False,
        download_budget_bytes: int = 0,
        cache_budget_bytes: int | None = None,
        authoritative_hashes: Mapping[str, str] | None = None,
        require_authoritative: bool = False,
        timeout: float = 300.0,
        chunk_bytes: int = 8 << 20,
    ) -> None:
        endpoint = endpoint.rstrip("/")
        _require_https(endpoint)
        if download_budget_bytes < 0 or cache_budget_bytes is not None and cache_budget_bytes < 0:
            raise rp.ContractError("Range budget 不能为负数")
        if chunk_bytes <= 0:
            raise rp.ContractError("chunk_bytes 必须为正数")
        self.root = cache_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint
        self.transport = transport or UrllibHttpsRangeTransport()
        self.allow_fetch = allow_fetch
        self.download_budget_bytes = int(download_budget_bytes)
        self.cache_budget_bytes = int(cache_budget_bytes) if cache_budget_bytes is not None else None
        self.authoritative_hashes = dict(authoritative_hashes or {})
        for key, digest in self.authoritative_hashes.items():
            if not isinstance(key, str) or not SHA256_RE.fullmatch(str(digest)):
                raise rp.ContractError(f"非法 authoritative range hash：{key!r}")
        self.require_authoritative = require_authoritative
        self.timeout = timeout
        self.chunk_bytes = chunk_bytes
        self._budget_lock = threading.Lock()
        self._download_used = 0
        self._download_reserved = 0
        self._cache_used = sum(
            path.stat().st_size
            for path in self.root.iterdir()
            if path.is_file() and (path.suffix == ".bin" or path.suffix == ".part" and path.name.endswith(".bin.part"))
        )
        self._cache_reserved = 0

    @property
    def downloaded_bytes(self) -> int:
        with self._budget_lock:
            return self._download_used

    def _identity(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        filename = _safe_source_file(str(entry["file"]))
        start = int(entry["start"])
        end = int(entry["end"])
        size = int(entry["bytes"])
        file_bytes = int(entry["file_bytes"])
        table_hash = str(entry["header_tensor_table_sha256"])
        if not SHA256_RE.fullmatch(table_hash):
            raise rp.ContractError("entry header tensor table SHA-256 非法")
        if not 0 <= start <= end < file_bytes or size != end - start + 1:
            raise rp.ContractError("entry Range 越过固定 shard 或字节不一致")
        if entry.get("range_key") != f"{filename}:{start}-{end}":
            raise rp.ContractError("entry range_key 与精确 Range 不一致")
        return {
            "repo": rp.REPO,
            "revision": rp.REVISION,
            "source_file": filename,
            "source_file_bytes": file_bytes,
            "start": start,
            "end": end,
            "header_tensor_table_sha256": table_hash,
        }

    def _paths(self, identity: Mapping[str, Any]) -> tuple[str, Path, Path, Path]:
        key = _canonical_sha256(identity)
        if not SHA256_RE.fullmatch(key):
            raise AssertionError("cache key 必须是 SHA-256")
        payload = (self.root / f"{key}.bin").resolve()
        partial = (self.root / f"{key}.bin.part").resolve()
        meta = (self.root / f"{key}.json").resolve()
        for path in (payload, partial, meta):
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise rp.ContractError("cache path 越过 cache root") from exc
        return key, payload, partial, meta

    def _url(self, filename: str) -> str:
        filename = _safe_source_file(filename)
        repo = urllib.parse.quote(rp.REPO, safe="/")
        revision = urllib.parse.quote(rp.REVISION, safe="")
        encoded_file = urllib.parse.quote(filename, safe="")
        url = f"{self.endpoint}/{repo}/resolve/{revision}/{encoded_file}"
        _require_https(url)
        return url

    def _reserve(self, download_bytes: int, cache_bytes: int) -> None:
        with self._budget_lock:
            if self.download_budget_bytes == 0:
                raise rp.ContractError("download budget 为 0；拒绝任何 cache miss Range")
            if self._download_used + self._download_reserved + download_bytes > self.download_budget_bytes:
                raise rp.ContractError(
                    f"download budget 超限：used={self._download_used}, requested={download_bytes}, "
                    f"limit={self.download_budget_bytes}"
                )
            if self.cache_budget_bytes is not None and (
                self._cache_used + self._cache_reserved + cache_bytes > self.cache_budget_bytes
            ):
                raise rp.ContractError(
                    f"cache budget 超限：used={self._cache_used}, requested={cache_bytes}, "
                    f"limit={self.cache_budget_bytes}"
                )
            self._download_reserved += download_bytes
            self._cache_reserved += cache_bytes

    def _settle(self, reserved: int, received: int, cache_written: int) -> None:
        with self._budget_lock:
            self._download_reserved -= reserved
            self._cache_reserved -= reserved
            self._download_used += received
            self._cache_used += cache_written

    def _proof(
        self,
        *,
        key: str,
        identity: Mapping[str, Any],
        digest: str,
        expected: str | None,
    ) -> dict[str, Any]:
        authoritative = expected is not None
        return {
            "format": CACHE_META_FORMAT,
            "cache_key": key,
            "identity": dict(identity),
            "bytes": int(identity["end"]) - int(identity["start"]) + 1,
            "observed_sha256": digest,
            "expected_sha256": expected,
            "hash_authority": "official_lock" if authoritative else "tofu",
            "authoritative": authoritative,
            "verified_transport": "HTTPS/206/exact-Content-Range",
        }

    def _load_hit(
        self,
        *,
        key: str,
        identity: Mapping[str, Any],
        payload: Path,
        meta_path: Path,
        expected: str | None,
    ) -> CachedRange | None:
        if not payload.exists() and not meta_path.exists():
            return None
        if not payload.is_file():
            raise rp.ContractError("cache metadata 存在但 payload 缺失")
        size = int(identity["end"]) - int(identity["start"]) + 1
        if payload.stat().st_size != size:
            raise rp.ContractError("cache payload 长度错误")
        observed = rp.sha256_file(payload)
        if meta_path.exists():
            meta = rp.read_json(meta_path)
            if meta.get("format") != CACHE_META_FORMAT or meta.get("cache_key") != key:
                raise rp.ContractError("cache metadata format/key 错误")
            if meta.get("identity") != dict(identity) or int(meta.get("bytes", -1)) != size:
                raise rp.ContractError("cache metadata 身份漂移")
            if meta.get("observed_sha256") != observed:
                raise rp.ContractError("cache payload SHA-256 与 metadata 不一致")
            authoritative = meta.get("authoritative") is True
            if authoritative:
                if meta.get("hash_authority") != "official_lock" or meta.get("expected_sha256") != observed:
                    raise rp.ContractError("cache authoritative proof 自相矛盾")
            elif meta.get("hash_authority") != "tofu" or meta.get("expected_sha256") is not None:
                raise rp.ContractError("cache TOFU proof 试图冒充权威")
        else:
            # 崩溃可能发生在 payload 原子 rename 后、metadata rename 前；本地重哈希即可恢复，
            # 但没有 external lock 时仍只能是 TOFU。
            meta = self._proof(key=key, identity=identity, digest=observed, expected=expected)
            _atomic_write_json(meta_path, meta)
        if expected is not None:
            if observed != expected:
                raise rp.ContractError("cache payload 与 authoritative hash lock 不匹配")
            if meta.get("authoritative") is not True:
                meta = self._proof(key=key, identity=identity, digest=observed, expected=expected)
                _atomic_write_json(meta_path, meta)
        elif self.require_authoritative and meta.get("authoritative") is not True:
            raise rp.ContractError("正式复现模式拒绝 TOFU cache entry")
        return CachedRange(entry={}, path=payload, proof=meta, cache_hit=True)

    def fetch(self, entry: Mapping[str, Any]) -> CachedRange:
        identity = self._identity(entry)
        key, payload, partial, meta_path = self._paths(identity)
        expected = self.authoritative_hashes.get(str(entry["range_key"]))
        if self.require_authoritative and expected is None:
            raise rp.ContractError(f"正式复现缺少 authoritative range lock：{entry['range_key']}")
        with _key_lock(self.root, key):
            hit = self._load_hit(
                key=key,
                identity=identity,
                payload=payload,
                meta_path=meta_path,
                expected=expected,
            )
            if hit is not None:
                return CachedRange(entry=dict(entry), path=hit.path, proof=hit.proof, cache_hit=True)
            if not self.allow_fetch:
                raise rp.ContractError("cache miss；必须显式 allow_fetch=True 才能发起 Range")
            size = int(entry["bytes"])
            done = partial.stat().st_size if partial.exists() else 0
            if not 0 <= done <= size:
                raise rp.ContractError("cache .part 长度越界")
            if done < size:
                remaining = size - done
                self._reserve(remaining, remaining)
                received = 0
                cache_written = 0
                original_done = done
                url = self._url(str(entry["file"]))
                request_start = int(entry["start"]) + done
                request_end = int(entry["end"])
                try:
                    response = self.transport.open_range(url, request_start, request_end, self.timeout)
                    with response:
                        final_url = response.geturl()
                        _require_https(final_url)
                        content_range = _header(response.headers, "Content-Range") or ""
                        match = CONTENT_RANGE_RE.fullmatch(content_range)
                        wanted = (request_start, request_end, int(entry["file_bytes"]))
                        got = tuple(map(int, match.groups())) if match else None
                        if response.status != 206 or got != wanted:
                            raise rp.ContractError(
                                f"严格 Range 契约失败：status={response.status}, Content-Range={content_range!r}"
                            )
                        content_length = _header(response.headers, "Content-Length")
                        if content_length is not None and (
                            not content_length.isdigit() or int(content_length) != remaining
                        ):
                            raise rp.ContractError("Content-Length 与精确 Range 不一致")
                        mode = "ab" if done else "wb"
                        with partial.open(mode) as sink:
                            try:
                                left = remaining
                                while left:
                                    chunk = response.read(min(self.chunk_bytes, left))
                                    if not chunk:
                                        raise rp.ContractError("Range payload 提前结束")
                                    if len(chunk) > left:
                                        raise rp.ContractError("Range response 单次 read 超过请求边界")
                                    sink.write(chunk)
                                    received += len(chunk)
                                    cache_written += len(chunk)
                                    left -= len(chunk)
                                extra = response.read(1)
                                if extra:
                                    received += len(extra)
                                    raise rp.ContractError("Range payload 超过 Content-Range")
                            finally:
                                # 中断也必须把可恢复 prefix 刷入 .part，下一次按实际长度续传。
                                sink.flush()
                                os.fsync(sink.fileno())
                except rp.ContractError:
                    if partial.exists():
                        with partial.open("r+b") as sink:
                            sink.truncate(original_done)
                    self._settle(remaining, received, 0)
                    raise
                except Exception:
                    # 连接中断保留已 fsync/close 的 prefix；下次从 .part 长度续传并重哈希。
                    self._settle(remaining, received, cache_written)
                    raise
                else:
                    self._settle(remaining, received, cache_written)
            if not partial.is_file() or partial.stat().st_size != size:
                raise rp.ContractError("Range .part 完成长度错误")
            observed = rp.sha256_file(partial)
            if expected is not None and observed != expected:
                partial.unlink()
                raise rp.ContractError("Range SHA-256 与 authoritative lock 不匹配")
            proof = self._proof(key=key, identity=identity, digest=observed, expected=expected)
            os.replace(partial, payload)
            _atomic_write_json(meta_path, proof)
            return CachedRange(entry=dict(entry), path=payload, proof=proof, cache_hit=False)


class SessionPhase(str, Enum):
    INIT = "init"
    AWAITING_LAYER = "awaiting_layer"
    LAYER_BASE_READY = "layer_base_ready"
    ROUTED = "routed"
    LAYER_READY = "layer_ready"
    FINAL_PENDING = "final_pending"
    COMPLETE = "complete"


@dataclass(frozen=True)
class LayerPrerequisites:
    layer: int
    token_id: str | int
    non_expert: tuple[CachedRange, ...]
    router: tuple[CachedRange, ...]


@dataclass(frozen=True)
class RoutedLayer:
    layer: int
    token_id: str | int
    expert_ids: tuple[int, ...]
    experts: Mapping[int, tuple[CachedRange, ...]]
    shared: tuple[CachedRange, ...]


class RouteFirstSession:
    """单 token 逐 S14 层状态机；路由前没有任何专家读取入口。"""

    def __init__(self, catalog: Mapping[str, Any], cache: RangeCache) -> None:
        validate_catalog(catalog)
        self.catalog = catalog
        self.cache = cache
        self.phase = SessionPhase.INIT
        self._layer_index = 0
        self._token_id: str | int | None = None
        self._route: tuple[int, ...] | None = None

    @property
    def current_layer(self) -> int | None:
        if self._layer_index >= len(self.catalog["selected_layers"]):
            return None
        return int(self.catalog["selected_layers"][self._layer_index])

    def _fetch_all(self, entries: Iterable[Mapping[str, Any]]) -> tuple[CachedRange, ...]:
        return tuple(self.cache.fetch(entry) for entry in entries)

    @staticmethod
    def _validate_token(token_id: str | int) -> None:
        if isinstance(token_id, bool) or not isinstance(token_id, (str, int)) or isinstance(token_id, str) and not token_id:
            raise rp.ContractError("token_id 必须是非空 string 或 integer")

    def prepare_embedding(self) -> tuple[CachedRange, ...]:
        if self.phase is not SessionPhase.INIT:
            raise rp.ContractError(f"embedding 只能在 init 获取；当前={self.phase.value}")
        result = self._fetch_all(self.catalog["boundary"]["embedding"])
        self.phase = SessionPhase.AWAITING_LAYER
        return result

    def prepare_layer(self, layer: int, token_id: str | int) -> LayerPrerequisites:
        self._validate_token(token_id)
        if self.phase is not SessionPhase.AWAITING_LAYER:
            raise rp.ContractError(f"当前状态不能获取层前置张量：{self.phase.value}")
        if layer != self.current_layer:
            raise rp.ContractError(f"错误 layer：expected={self.current_layer}, got={layer}")
        row = self.catalog["layers"][str(layer)]
        non_expert = self._fetch_all(row["non_expert"])
        router = self._fetch_all(row["router"])
        self._token_id = token_id
        self._route = None
        self.phase = SessionPhase.LAYER_BASE_READY
        return LayerPrerequisites(layer=layer, token_id=token_id, non_expert=non_expert, router=router)

    def submit_top6(self, layer: int, token_id: str | int, expert_ids: Iterable[int]) -> tuple[int, ...]:
        if self.phase is not SessionPhase.LAYER_BASE_READY:
            raise rp.ContractError("必须先提供当前层 non-expert/router，才能提交 route")
        if layer != self.current_layer or token_id != self._token_id:
            raise rp.ContractError("route 的 layer/token 与当前状态不匹配")
        ids = list(expert_ids)
        if len(ids) != TOP_K or any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
            raise rp.ContractError("当前 token route 必须是恰好 6 个 integer expert IDs")
        if len(set(ids)) != TOP_K:
            raise rp.ContractError("当前 token top-6 expert IDs 不得重复")
        available = self.catalog["layers"][str(layer)]["experts"]
        if any(not 0 <= value < EXPERT_COUNT or str(value) not in available for value in ids):
            raise rp.ContractError("当前 token expert ID 越界/不在固定 header；禁止猜 expert")
        self._route = tuple(ids)
        self.phase = SessionPhase.ROUTED
        return self._route

    def fetch_routed(self, layer: int, token_id: str | int) -> RoutedLayer:
        if self.phase is not SessionPhase.ROUTED or self._route is None:
            raise rp.ContractError("路由前拒绝专家 Range；必须先提交当前 token top-6")
        if layer != self.current_layer or token_id != self._token_id:
            raise rp.ContractError("专家请求的 layer/token 与当前 route 不匹配")
        row = self.catalog["layers"][str(layer)]
        shared = self._fetch_all(row["shared"])
        experts = {
            expert_id: self._fetch_all(row["experts"][str(expert_id)])
            for expert_id in self._route
        }
        self.phase = SessionPhase.LAYER_READY
        return RoutedLayer(
            layer=layer,
            token_id=token_id,
            expert_ids=self._route,
            experts=experts,
            shared=shared,
        )

    def finish_layer(self, layer: int, token_id: str | int) -> None:
        if self.phase is not SessionPhase.LAYER_READY:
            raise rp.ContractError("当前层专家/shared 尚未 ready，不能完成层")
        if layer != self.current_layer or token_id != self._token_id:
            raise rp.ContractError("完成层的 layer/token 与当前状态不匹配")
        self._layer_index += 1
        self._token_id = None
        self._route = None
        self.phase = (
            SessionPhase.FINAL_PENDING
            if self._layer_index == len(self.catalog["selected_layers"])
            else SessionPhase.AWAITING_LAYER
        )

    def prepare_final(self) -> tuple[CachedRange, ...]:
        if self.phase is not SessionPhase.FINAL_PENDING:
            raise rp.ContractError("必须完成全部固定 S14 层后才能获取 norm/head")
        result = self._fetch_all(self.catalog["boundary"]["final"])
        self.phase = SessionPhase.COMPLETE
        return result


def command_catalog(args: argparse.Namespace) -> None:
    source = rp.read_json(args.source)
    catalog = build_catalog(
        source,
        args.index,
        args.header_dir,
        metadata_manifest_path=args.metadata_manifest,
        skeleton_manifest_path=args.skeleton_manifest,
    )
    rp.write_json(args.output, catalog, force=args.force)
    print(json.dumps({
        "status": "catalog_ready_no_payload_download",
        "output": str(args.output),
        "range_count": catalog["summary"]["range_count"],
        "prerequisite_range_count": catalog["summary"]["prerequisite_range_count"],
        "download_authorized": False,
    }, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    catalog = sub.add_parser("catalog", help="只读本地固定 index/header，生成 route-first catalog；不取 payload")
    catalog.add_argument("--source", type=Path, default=rp.SOURCE_CONTRACT)
    catalog.add_argument("--index", type=Path, required=True)
    catalog.add_argument("--header-dir", type=Path, required=True)
    catalog.add_argument("--metadata-manifest", type=Path)
    catalog.add_argument("--skeleton-manifest", type=Path)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--force", action="store_true")
    catalog.set_defaults(func=command_catalog)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
