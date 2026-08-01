"""Read-only audit of the real Polaris-S14 metadata assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


LAYERS = 43
EXPERTS_PER_LAYER = 256
TOP_K = 6
S14_LAYERS = (0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42)
EXPECTED_HEADER_NUMBERS = tuple(range(1, 46))
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")
_HEADER_RE = re.compile(r"^model-(\d{5})-of-00048\.safetensors\.header\.json$")


class AssetContractError(ValueError):
    """Raised when local metadata no longer satisfies the frozen contract."""


@dataclass(frozen=True)
class TensorLocation:
    file: str
    start: int
    end: int
    bytes: int


@dataclass(frozen=True)
class AssetAudit:
    asset_root: str
    budget_sha256: str
    route_catalog_sha256: str
    tokenizer_sha256: str
    header_count: int
    header_set_sha256: str
    tensor_count: int
    full_base_shard_bytes: int
    full_base_payload_bytes: int
    shard_container_overhead_bytes: int
    full_non_routed_bytes: int
    head_bytes: int
    embedding_bytes: int
    other_boundary_bytes: int
    expert_page_bytes: int
    expert_page_count: int
    expert_bank_bytes: int
    route_catalog_range_count: int
    route_catalog_range_bytes: int
    route_catalog_selected_layers: tuple[int, ...]
    weights_downloaded: bool
    budget_vram_fit_nonrouted_head_one_layer_top6: bool

    @property
    def fixed_decode_scan_bytes(self) -> int:
        return self.full_non_routed_bytes + self.head_bytes

    @property
    def non_expert_base_payload_bytes(self) -> int:
        return self.full_base_payload_bytes - self.expert_bank_bytes

    @property
    def non_expert_packed_storage_bytes(self) -> int:
        return self.non_expert_base_payload_bytes + self.shard_container_overhead_bytes

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["route_catalog_selected_layers"] = list(self.route_catalog_selected_layers)
        result["fixed_decode_scan_bytes"] = self.fixed_decode_scan_bytes
        result["non_expert_base_payload_bytes"] = self.non_expert_base_payload_bytes
        result["non_expert_packed_storage_bytes"] = self.non_expert_packed_storage_bytes
        return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetContractError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AssetContractError(f"JSON 顶层必须是 object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_header_sha256(paths: Iterable[Path]) -> str:
    """Hash local JSON headers, including names, in deterministic order."""

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssetContractError(message)


def _walk_catalog_ranges(catalog: dict[str, Any]) -> Iterable[dict[str, Any]]:
    boundary = catalog.get("boundary")
    _expect(isinstance(boundary, dict), "route catalog 缺少 boundary")
    for group in ("embedding", "final"):
        rows = boundary.get(group)
        _expect(isinstance(rows, list), f"route catalog boundary.{group} 必须是 list")
        yield from rows

    layers = catalog.get("layers")
    _expect(isinstance(layers, dict), "route catalog 缺少 layers")
    for layer in catalog["selected_layers"]:
        row = layers.get(str(layer))
        _expect(isinstance(row, dict), f"route catalog 缺少 L{layer}")
        for group in ("non_expert", "router", "shared"):
            ranges = row.get(group)
            _expect(isinstance(ranges, list), f"route catalog L{layer}.{group} 必须是 list")
            yield from ranges
        experts = row.get("experts")
        _expect(isinstance(experts, dict), f"route catalog L{layer}.experts 必须是 object")
        _expect(len(experts) == EXPERTS_PER_LAYER, f"route catalog L{layer} 不是 256 个专家")
        for expert_id in range(EXPERTS_PER_LAYER):
            ranges = experts.get(str(expert_id))
            _expect(isinstance(ranges, list), f"route catalog L{layer}/E{expert_id} 缺失")
            yield from ranges


def audit_assets(asset_root: str | Path) -> AssetAudit:
    """Cross-check budget, all 45 base headers and route-first ranges.

    No payload or model code is opened.  Every byte count is reconstructed from
    safetensors offsets and then checked against the frozen budget/catalog.
    """

    root = Path(asset_root)
    budget_path = root / "fulldepth_kadaptive_budget.json"
    catalog_path = root / "route_first_catalog.json"
    tokenizer_path = root / "tokenizer.json"
    budget = _read_json(budget_path)
    catalog = _read_json(catalog_path)

    header_dir = root / "headers"
    header_paths = sorted(header_dir.glob("*.safetensors.header.json"))
    numbers: list[int] = []
    for path in header_paths:
        match = _HEADER_RE.match(path.name)
        _expect(match is not None, f"非预期 header 文件名: {path.name}")
        numbers.append(int(match.group(1)))
    _expect(tuple(numbers) == EXPECTED_HEADER_NUMBERS, "必须恰好提供 base shard 00001..00045 的 45 个 header")

    tensor_locations: dict[str, TensorLocation] = {}
    layer_non_routed = [0] * LAYERS
    expert_pages: dict[tuple[int, int], int] = {}
    boundary: dict[str, int] = {}
    full_payload_bytes = 0
    full_shard_bytes = 0

    header_metadata: dict[str, dict[str, Any]] = {}
    for path in header_paths:
        header = _read_json(path)
        file = header.get("file")
        data_start = header.get("data_start")
        tensors = header.get("tensors")
        _expect(isinstance(file, str), f"{path.name} 缺少 file")
        _expect(isinstance(data_start, int), f"{path.name} 缺少 data_start")
        _expect(isinstance(tensors, dict), f"{path.name} 缺少 tensors")
        file_bytes = header.get("file_bytes")
        _expect(isinstance(file_bytes, int) and file_bytes > 0, f"{path.name} 缺少 file_bytes")
        full_shard_bytes += file_bytes
        header_metadata[file] = header
        for name, metadata in tensors.items():
            _expect(name not in tensor_locations, f"重复 tensor: {name}")
            offsets = metadata.get("data_offsets")
            _expect(
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(isinstance(item, int) for item in offsets)
                and 0 <= offsets[0] < offsets[1],
                f"tensor offset 非法: {name}",
            )
            tensor_bytes = offsets[1] - offsets[0]
            location = TensorLocation(
                file=file,
                start=data_start + offsets[0],
                end=data_start + offsets[1] - 1,
                bytes=tensor_bytes,
            )
            tensor_locations[name] = location
            full_payload_bytes += tensor_bytes

            expert_match = _EXPERT_RE.match(name)
            if expert_match:
                key = (int(expert_match.group(1)), int(expert_match.group(2)))
                _expect(0 <= key[0] < LAYERS and 0 <= key[1] < EXPERTS_PER_LAYER, f"专家索引越界: {name}")
                expert_pages[key] = expert_pages.get(key, 0) + tensor_bytes
                continue
            layer_match = _LAYER_RE.match(name)
            if layer_match:
                layer = int(layer_match.group(1))
                _expect(0 <= layer < LAYERS, f"层索引越界: {name}")
                layer_non_routed[layer] += tensor_bytes
            else:
                boundary[name] = boundary.get(name, 0) + tensor_bytes

    _expect(len(expert_pages) == LAYERS * EXPERTS_PER_LAYER, "header 没有覆盖 43x256 个专家页")
    page_sizes = set(expert_pages.values())
    _expect(len(page_sizes) == 1, f"专家页字节不统一: {sorted(page_sizes)}")
    expert_page_bytes = next(iter(page_sizes))
    expert_bank_bytes = sum(expert_pages.values())
    full_non_routed_bytes = sum(layer_non_routed)
    head_bytes = boundary.get("head.weight", 0)
    embedding_bytes = boundary.get("embed.weight", 0)
    other_boundary_bytes = sum(boundary.values()) - head_bytes - embedding_bytes

    _expect(budget.get("layers") == LAYERS, "budget layers 不是 43")
    _expect(budget.get("full_non_routed_bytes") == full_non_routed_bytes, "budget full_non_routed_bytes 与 header 不符")
    _expect(budget.get("head_bf16_bytes") == head_bytes, "budget head_bf16_bytes 与 header 不符")
    _expect(budget.get("expert_page_bytes") == expert_page_bytes, "budget expert_page_bytes 与 header 不符")
    layer_rows = budget.get("layer_rows")
    _expect(isinstance(layer_rows, list) and len(layer_rows) == LAYERS, "budget layer_rows 不完整")
    for layer, row in enumerate(layer_rows):
        _expect(row.get("layer") == layer, f"budget layer_rows[{layer}] 层号错误")
        _expect(row.get("non_routed_bytes") == layer_non_routed[layer], f"budget L{layer} 非路由字节不符")
        _expect(row.get("expert_bytes") == expert_page_bytes, f"budget L{layer} 专家页字节不符")
    for top_k in (1, 2, TOP_K):
        expected_expert_bytes = LAYERS * top_k * expert_page_bytes
        _expect(budget.get(f"top{top_k}_expert_bytes_per_token") == expected_expert_bytes, f"budget top{top_k} 字节不符")
        expected_scan = full_non_routed_bytes + head_bytes + expected_expert_bytes
        _expect(budget.get("decode_scan_bytes", {}).get(f"top{top_k}") == expected_scan, f"budget top{top_k} scan 不符")
    vram_fit_flag = budget.get("vram_8gib_fit_nonrouted_plus_head_plus_one_layer_top6")
    _expect(isinstance(vram_fit_flag, bool), "budget 缺少 8GiB fit 布尔值")
    computed_vram_fit = full_non_routed_bytes + head_bytes + TOP_K * expert_page_bytes <= 8 * 1024**3
    _expect(vram_fit_flag == computed_vram_fit, "budget 8GiB fit 标志与 header 字节不符")

    selected_layers = tuple(catalog.get("selected_layers", ()))
    _expect(selected_layers == S14_LAYERS, "route catalog 冻结 S14 层集漂移")
    _expect(catalog.get("top_k") == TOP_K, "route catalog top_k 不是 6")
    _expect(catalog.get("expert_id_range") == [0, EXPERTS_PER_LAYER - 1], "route catalog 专家范围不符")
    _expect(all(isinstance(layer, int) and 0 <= layer < LAYERS for layer in selected_layers), "route catalog selected_layers 非法")

    range_count = 0
    range_bytes = 0
    range_keys: set[str] = set()
    for row in _walk_catalog_ranges(catalog):
        _expect(isinstance(row, dict), "route catalog range 必须是 object")
        tensor = row.get("tensor")
        _expect(isinstance(tensor, str) and tensor in tensor_locations, f"route catalog tensor 不在 header: {tensor}")
        location = tensor_locations[tensor]
        _expect(row.get("file") == location.file, f"route catalog tensor shard 不符: {tensor}")
        _expect(row.get("start") == location.start and row.get("end") == location.end, f"route catalog range 不符: {tensor}")
        _expect(row.get("bytes") == location.bytes, f"route catalog bytes 不符: {tensor}")
        expected_key = f"{location.file}:{location.start}-{location.end}"
        _expect(row.get("range_key") == expected_key, f"route catalog range_key 不符: {tensor}")
        _expect(expected_key not in range_keys, f"route catalog 重复 range: {expected_key}")
        range_keys.add(expected_key)
        range_count += 1
        range_bytes += location.bytes

    summary = catalog.get("summary", {})
    _expect(summary.get("range_count") == range_count, "route catalog summary.range_count 不符")
    _expect(summary.get("range_bytes") == range_bytes, "route catalog summary.range_bytes 不符")

    # The catalog carries raw safetensors-header metadata for the shards it uses.
    catalog_headers = catalog.get("headers", {}).get("files", {})
    _expect(isinstance(catalog_headers, dict), "route catalog headers.files 缺失")
    for file, metadata in catalog_headers.items():
        _expect(file in header_metadata, f"route catalog 引用未知 shard: {file}")
        header = header_metadata[file]
        for key in ("file_bytes", "header_length", "data_start", "header_sha256", "tensor_table_sha256"):
            _expect(metadata.get(key) == header.get(key), f"route catalog {file}.{key} 与 header 不符")

    weights_downloaded = bool(catalog.get("local_metadata", {}).get("weights_downloaded", False))
    return AssetAudit(
        asset_root=str(root.resolve()),
        budget_sha256=_sha256(budget_path),
        route_catalog_sha256=_sha256(catalog_path),
        tokenizer_sha256=_sha256(tokenizer_path),
        header_count=len(header_paths),
        header_set_sha256=_combined_header_sha256(header_paths),
        tensor_count=len(tensor_locations),
        full_base_shard_bytes=full_shard_bytes,
        full_base_payload_bytes=full_payload_bytes,
        shard_container_overhead_bytes=full_shard_bytes - full_payload_bytes,
        full_non_routed_bytes=full_non_routed_bytes,
        head_bytes=head_bytes,
        embedding_bytes=embedding_bytes,
        other_boundary_bytes=other_boundary_bytes,
        expert_page_bytes=expert_page_bytes,
        expert_page_count=len(expert_pages),
        expert_bank_bytes=expert_bank_bytes,
        route_catalog_range_count=range_count,
        route_catalog_range_bytes=range_bytes,
        route_catalog_selected_layers=selected_layers,
        weights_downloaded=weights_downloaded,
        budget_vram_fit_nonrouted_head_one_layer_top6=vram_fit_flag,
    )
