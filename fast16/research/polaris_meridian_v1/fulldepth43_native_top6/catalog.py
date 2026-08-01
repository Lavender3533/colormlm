"""从固定 index 与本地真实 header 构建 43 层 Range catalog。

只读 JSON/header，不读取或下载任何权重 payload。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .profile import FULLDEPTH43_NATIVE_TOP6, ExecutionProfile


FORMAT = "polaris-fulldepth43-native-top6-catalog-v1"
BOUNDARY = {
    "embed.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "norm.weight",
    "head.weight",
}
LAYER_RE = re.compile(r"^layers\.(\d+)\.")
EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DTYPE_BYTES = {"F32": 4, "BF16": 2, "I64": 8, "F8_E8M0": 1, "F8_E4M3": 1, "I8": 1}


class CatalogError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except FileNotFoundError as exc:
        raise CatalogError(f"缺少 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CatalogError(f"JSON 顶层必须是 object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(name: str, filename: str, header: Mapping[str, Any], meta: Mapping[str, Any], kind: str) -> dict:
    begin, end = map(int, meta["data_offsets"])
    start = int(header["data_start"]) + begin
    layer_match = LAYER_RE.match(name)
    expert_match = EXPERT_RE.match(name)
    row: dict[str, Any] = {
        "tensor": name,
        "kind": kind,
        "layer": None if layer_match is None else int(layer_match.group(1)),
        "file": filename,
        "file_bytes": int(header["file_bytes"]),
        "header_tensor_table_sha256": str(header["tensor_table_sha256"]),
        "start": start,
        "end": int(header["data_start"]) + end - 1,
        "bytes": end - begin,
        "dtype": meta["dtype"],
        "shape": meta["shape"],
        "range_key": f"{filename}:{start}-{int(header['data_start']) + end - 1}",
    }
    if expert_match is not None:
        row["expert_id"] = int(expert_match.group(2))
    return row


def _suffix(name: str) -> str:
    match = LAYER_RE.match(name)
    if match is None:
        return name
    return name[match.end() :]


def _expert_suffix(name: str) -> str:
    match = EXPERT_RE.match(name)
    if match is None:
        raise CatalogError(f"非 expert tensor: {name}")
    return name[match.end() :]


def _schema_exemplar(layer: int) -> int:
    if layer < 3:
        return layer
    return 6 if layer % 2 == 0 else 7


def _iter_entries(catalog: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    yield from catalog["boundary"]["embedding"]
    yield from catalog["boundary"]["final"]
    for layer in catalog["profile"]["layers"]:
        row = catalog["layers"][str(layer)]
        yield from row["non_expert"]
        yield from row["router"]
        yield from row["shared"]
        for expert_id in range(256):
            yield from row["experts"][str(expert_id)]


def validate_catalog(catalog: Mapping[str, Any], *, template: Mapping[str, Any] | None = None) -> None:
    profile = FULLDEPTH43_NATIVE_TOP6
    profile.validate()
    if catalog.get("format") != FORMAT or catalog.get("repo") != profile.repo:
        raise CatalogError("FullDepth catalog format/repo 错误")
    if catalog.get("revision") != profile.revision or catalog.get("profile") != profile.as_dict():
        raise CatalogError("FullDepth catalog revision/profile 漂移")
    if catalog.get("download_authorized") is not False:
        raise CatalogError("catalog 不得授予下载权限")
    boundary = catalog.get("boundary")
    if not isinstance(boundary, dict):
        raise CatalogError("boundary 缺失")
    if [row["tensor"] for row in boundary.get("embedding", [])] != ["embed.weight"]:
        raise CatalogError("必须且只能有原生 embedding")
    if {row["tensor"] for row in boundary.get("final", [])} != BOUNDARY - {"embed.weight"}:
        raise CatalogError("原生 HC/norm/head boundary 不完整")
    layers = catalog.get("layers")
    if not isinstance(layers, dict) or set(layers) != {str(value) for value in profile.layers}:
        raise CatalogError("禁止缺层或额外层")

    template_layers = {} if template is None else template.get("layers", {})
    for layer in profile.layers:
        row = layers[str(layer)]
        for group in ("non_expert", "router", "shared"):
            if not isinstance(row.get(group), list) or not row[group]:
                raise CatalogError(f"L{layer} 缺少 {group}")
        experts = row.get("experts")
        if not isinstance(experts, dict) or set(experts) != {str(value) for value in range(256)}:
            raise CatalogError(f"L{layer} 必须完整枚举 256 experts")
        expert_suffix = {_expert_suffix(item["tensor"]) for item in experts["0"]}
        if len(expert_suffix) != 6 or any(
            {_expert_suffix(item["tensor"]) for item in experts[str(expert)]} != expert_suffix
            for expert in range(256)
        ):
            raise CatalogError(f"L{layer} expert FP4 weight/scale schema 不闭合")
        if template_layers:
            exemplar = template_layers[str(_schema_exemplar(layer))]
            for group in ("non_expert", "router", "shared"):
                expected = {_suffix(item["tensor"]) for item in exemplar[group]}
                actual = {_suffix(item["tensor"]) for item in row[group]}
                if actual != expected:
                    raise CatalogError(f"L{layer} {group} 与同类官方层 schema 不一致")

    entries = list(_iter_entries(catalog))
    seen: set[tuple[str, str]] = set()
    for row in entries:
        key = (str(row["file"]), str(row["tensor"]))
        if key in seen:
            raise CatalogError(f"重复 tensor Range: {key}")
        seen.add(key)
        dtype = row.get("dtype")
        shape = row.get("shape")
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or not shape:
            raise CatalogError(f"dtype/shape 不受支持: {key}")
        if math.prod(shape) * DTYPE_BYTES[dtype] != row.get("bytes"):
            raise CatalogError(f"tensor 字节不闭合: {key}")
        if not SHA256_RE.fullmatch(str(row.get("header_tensor_table_sha256", ""))):
            raise CatalogError(f"header tensor table SHA 非法: {key}")
        if row.get("range_key") != f"{row['file']}:{row['start']}-{row['end']}":
            raise CatalogError(f"Range key 漂移: {key}")
    summary = catalog.get("summary", {})
    if summary.get("range_count") != len(entries) or summary.get("range_bytes") != sum(
        int(row["bytes"]) for row in entries
    ):
        raise CatalogError("catalog summary 不闭合")


def build_catalog(
    *,
    asset_root: Path,
    template_catalog_path: Path | None = None,
    profile: ExecutionProfile = FULLDEPTH43_NATIVE_TOP6,
) -> dict[str, Any]:
    profile.validate()
    root = asset_root.resolve()
    index_path = root / "model.safetensors.index.json"
    header_dir = root / "headers"
    template_path = template_catalog_path or root / "route_first_catalog.json"
    template = read_json(template_path)
    if template.get("repo") != profile.repo or template.get("revision") != profile.revision:
        raise CatalogError("S14 template catalog donor 身份漂移")
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CatalogError("官方 index.weight_map 缺失")
    if sha256_file(index_path) != template.get("index", {}).get("sha256"):
        raise CatalogError("官方 index SHA-256 与已验证 S14 catalog 不一致")

    needed_files = {
        filename
        for name, filename in weight_map.items()
        if name in BOUNDARY or (LAYER_RE.match(name) and int(LAYER_RE.match(name).group(1)) in profile.layers)
    }
    headers: dict[str, dict[str, Any]] = {}
    for filename in sorted(needed_files):
        header = read_json(header_dir / f"{filename}.header.json")
        if (
            header.get("repo") != profile.repo
            or header.get("revision") != profile.revision
            or header.get("file") != filename
            or not SHA256_RE.fullmatch(str(header.get("header_sha256", "")))
            or not SHA256_RE.fullmatch(str(header.get("tensor_table_sha256", "")))
        ):
            raise CatalogError(f"header 身份或 SHA 非法: {filename}")
        headers[filename] = header

    boundary: dict[str, list[dict[str, Any]]] = {"embedding": [], "final": []}
    layers = {
        str(layer): {"non_expert": [], "router": [], "shared": [], "experts": {}}
        for layer in profile.layers
    }
    for name, filename in sorted(weight_map.items()):
        if name in BOUNDARY:
            kind = "boundary"
            target = boundary["embedding" if name == "embed.weight" else "final"]
        else:
            layer_match = LAYER_RE.match(name)
            if layer_match is None or int(layer_match.group(1)) not in profile.layers:
                continue
            layer = int(layer_match.group(1))
            row = layers[str(layer)]
            expert_match = EXPERT_RE.match(name)
            if expert_match is not None:
                expert_id = int(expert_match.group(2))
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
        header = headers.get(filename)
        if header is None or name not in header.get("tensors", {}):
            raise CatalogError(f"index/header 不一致: {name} -> {filename}")
        target.append(_entry(name, filename, header, header["tensors"][name], kind))

    all_entries = list(boundary["embedding"]) + list(boundary["final"])
    for layer in profile.layers:
        row = layers[str(layer)]
        all_entries.extend(row["non_expert"])
        all_entries.extend(row["router"])
        all_entries.extend(row["shared"])
        for expert_id in range(256):
            all_entries.extend(row["experts"].get(str(expert_id), []))
    catalog = {
        "format": FORMAT,
        "repo": profile.repo,
        "revision": profile.revision,
        "profile": profile.as_dict(),
        "download_authorized": False,
        "index": {
            "file": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
            "authoritative": True,
        },
        "headers": {
            "count": len(headers),
            "files": {
                filename: {
                    "file_bytes": int(header["file_bytes"]),
                    "header_sha256": header["header_sha256"],
                    "tensor_table_sha256": header["tensor_table_sha256"],
                    "file_lfs_sha256": header.get("file_lfs_sha256"),
                }
                for filename, header in headers.items()
            },
        },
        "boundary": boundary,
        "layers": layers,
        "summary": {
            "range_count": len(all_entries),
            "range_bytes": sum(int(row["bytes"]) for row in all_entries),
            "layer_count": len(profile.layers),
            "route_policy": "current token/current layer native top-6; no skipped layer; no guessed expert",
        },
        "claim_limit": "header/index catalog only; no payload read, no model token executed",
    }
    validate_catalog(catalog, template=template)
    return catalog


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
