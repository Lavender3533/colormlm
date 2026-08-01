#!/usr/bin/env python3
"""为冻结的 Polaris S14 生成 safetensors HTTP Range 精确打包计划。

默认命令只读本地 metadata/header/route trace；不会下载权重。只有显式使用
``fetch-headers --execute-metadata-fetch`` 才会请求 safetensors header（不是 tensor
payload），只有 ``materialize --execute`` 且所有 range SHA 已预先锁定时才会取 payload。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO


HERE = Path(__file__).resolve().parent
SOURCE_CONTRACT = HERE / "source_contract.json"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
ALIGNMENT = 64
SEGMENT_BYTES = 1 << 30
GIB = 1 << 30
EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")
LAYER_RE = re.compile(r"^layers\.(\d+)\.")


class ContractError(ValueError):
    """输入不满足冻结契约。"""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} 必须是 JSON object")
    return value


def write_json(path: Path, value: Any, *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"拒绝覆盖 {path}；显式使用 --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def validate_source_contract(source: dict[str, Any]) -> None:
    if source.get("repo") != REPO or source.get("revision") != REVISION:
        raise ContractError("拒绝非冻结 DeepSeek repo/revision")
    expected = [0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42]
    if source.get("selected_layers") != expected:
        raise ContractError("S14 层集合发生漂移")
    layer_total = sum(int(row["bytes"]) for row in source["layer_shards"].values())
    boundary_files = {row["file"]: int(row["bytes"]) for row in source["boundary_shards"].values()}
    total = layer_total + sum(boundary_files.values())
    frozen = source["shard_union_budget"]
    if layer_total != 50_112_879_344 or total != 52_231_273_716:
        raise ContractError("冻结 shard-union 字节预算不一致")
    if int(frozen["total_bytes"]) != total:
        raise ContractError("source_contract 的总字节自相矛盾")


def validate_index(index_path: Path, source: dict[str, Any], *, strict_hash: bool = True) -> dict[str, str]:
    raw = index_path.read_bytes()
    expected = source["index"]
    if strict_hash and (len(raw) != int(expected["bytes"]) or sha256_bytes(raw) != expected["sha256"]):
        raise ContractError("index 不是固定 revision 的官方索引")
    index = json.loads(raw.decode("utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ContractError("index 缺少 weight_map")
    return {str(key): str(value) for key, value in weight_map.items()}


def validate_route_trace(trace: dict[str, Any], selected_layers: list[int]) -> dict[int, set[int]]:
    checks = {
        "format": "polaris-s14-route-trace-v1",
        "repo": REPO,
        "revision": REVISION,
        "status": "approved_native_trace",
        "coverage_complete": True,
        "top_k": 6,
    }
    for key, expected in checks.items():
        if trace.get(key) != expected:
            raise ContractError(f"route trace 拒绝门：{key} 必须为 {expected!r}")
    if int(trace.get("observed_tokens", 0)) <= 0 or int(trace.get("task_groups", 0)) <= 0:
        raise ContractError("route trace 必须来自至少一个完整任务和一个 token")
    if not re.fullmatch(r"[0-9a-f]{64}", str(trace.get("capture_manifest_sha256", ""))):
        raise ContractError("route trace 必须引用原生 capture manifest SHA-256")
    raw_layers = trace.get("layers")
    if not isinstance(raw_layers, dict):
        raise ContractError("route trace.layers 必须是 object")
    result: dict[int, set[int]] = {}
    for layer in selected_layers:
        row = raw_layers.get(str(layer))
        if not isinstance(row, dict):
            raise ContractError(f"route trace 缺少 S14 L{layer}；禁止猜 expert")
        ids = row.get("expert_ids")
        if not isinstance(ids, list) or not ids:
            raise ContractError(f"L{layer} expert_ids 为空；禁止默认 0..5")
        parsed = {int(item) for item in ids}
        if len(parsed) != len(ids) or min(parsed) < 0 or max(parsed) >= 256:
            raise ContractError(f"L{layer} expert_ids 越界或重复")
        if int(row.get("events", 0)) <= 0:
            raise ContractError(f"L{layer} 缺少 route event")
        result[layer] = parsed
    unexpected = set(raw_layers) - {str(layer) for layer in selected_layers}
    if unexpected:
        raise ContractError(f"route trace 含非 S14 层：{sorted(unexpected)}")
    return result


def source_file_contracts(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in source["layer_shards"].values():
        rows[row["file"]] = dict(row)
    for row in source["boundary_shards"].values():
        old = rows.setdefault(row["file"], dict(row))
        if old["bytes"] != row["bytes"]:
            raise ContractError(f"同一 shard 字节冲突：{row['file']}")
    return rows


def hf_url(filename: str) -> str:
    # 国内容器可显式指向兼容 Hugging Face resolve/Range 语义的镜像；
    # revision、Content-Range 与哈希门仍保持不变，不能用 main 漂移替代。
    endpoint = os.environ.get("POLARIS_HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    if not endpoint.startswith("https://"):
        raise ContractError("POLARIS_HF_ENDPOINT 必须使用 HTTPS")
    return f"{endpoint}/{REPO}/resolve/{REVISION}/{filename}"


def _range_request(url: str, start: int, end: int) -> tuple[bytes, Any]:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    response = urllib.request.urlopen(request, timeout=120)
    data = response.read()
    content_range = response.headers.get("Content-Range", "")
    if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
        raise ContractError(f"服务器未遵守精确 Range：status={response.status}, Content-Range={content_range!r}")
    if len(data) != end - start + 1:
        raise ContractError("Range 返回长度不匹配")
    return data, response.headers


def fetch_header(filename: str, file_bytes: int) -> dict[str, Any]:
    url = hf_url(filename)
    prefix, prefix_headers = _range_request(url, 0, 7)
    header_len = struct.unpack("<Q", prefix)[0]
    if not 2 <= header_len <= 256 * 1024 * 1024:
        raise ContractError(f"异常 safetensors header 长度：{header_len}")
    raw_header, headers = _range_request(url, 8, 7 + header_len)
    total = int(headers["Content-Range"].rsplit("/", 1)[1])
    if total != file_bytes:
        raise ContractError(f"{filename} 固定字节不匹配：{total} != {file_bytes}")
    decoded = json.loads(raw_header.decode("utf-8"))
    tensors = {key: value for key, value in decoded.items() if key != "__metadata__"}
    return {
        "format": "polaris-safetensors-header-v1",
        "repo": REPO,
        "revision": REVISION,
        "file": filename,
        "file_bytes": file_bytes,
        "header_length": header_len,
        "header_sha256": sha256_bytes(raw_header),
        "tensor_table_sha256": sha256_bytes(
            json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "data_start": 8 + header_len,
        "etag": headers.get("ETag") or prefix_headers.get("ETag"),
        "tensors": tensors,
    }


def validate_header(header: dict[str, Any], filename: str, file_bytes: int) -> None:
    if header.get("repo") != REPO or header.get("revision") != REVISION or header.get("file") != filename:
        raise ContractError(f"header cache 身份不匹配：{filename}")
    if int(header.get("file_bytes", -1)) != file_bytes:
        raise ContractError(f"header cache 字节不匹配：{filename}")
    if int(header.get("data_start", -1)) != 8 + int(header.get("header_length", -9)):
        raise ContractError(f"header cache data_start 错误：{filename}")
    tensors = header.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ContractError(f"header cache 无 tensor：{filename}")
    table_hash = sha256_bytes(json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if header.get("tensor_table_sha256") != table_hash:
        raise ContractError(f"header cache tensor 表摘要错误：{filename}")
    data_bytes = file_bytes - int(header["data_start"])
    for name, meta in tensors.items():
        offsets = meta.get("data_offsets") if isinstance(meta, dict) else None
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ContractError(f"tensor offset 非法：{name}")
        begin, end = map(int, offsets)
        if not 0 <= begin < end <= data_bytes:
            raise ContractError(f"tensor 越过 shard：{name}")


def load_headers(header_dir: Path, source: dict[str, Any], files: set[str]) -> dict[str, dict[str, Any]]:
    contracts = source_file_contracts(source)
    result = {}
    for filename in sorted(files):
        if filename not in contracts:
            raise ContractError(f"index 引用了未冻结 shard：{filename}")
        path = header_dir / f"{filename}.header.json"
        if not path.exists():
            raise ContractError(f"缺少 header cache：{path}")
        row = read_json(path)
        validate_header(row, filename, int(contracts[filename]["bytes"]))
        result[filename] = row
    return result


def select_names(weight_map: dict[str, str], source: dict[str, Any], routes: dict[int, set[int]]) -> list[str]:
    selected_layers = set(source["selected_layers"])
    selected: list[str] = []
    boundary = set(source["boundary_tensors"])
    missing_boundary = boundary - set(weight_map)
    if missing_boundary:
        raise ContractError(f"索引缺少原生边界 tensor：{sorted(missing_boundary)}")
    expert_layers_seen: set[int] = set()
    for name in sorted(weight_map):
        if name in boundary:
            selected.append(name)
            continue
        layer_match = LAYER_RE.match(name)
        if not layer_match:
            continue
        layer = int(layer_match.group(1))
        if layer not in selected_layers:
            continue
        if ".ffn.experts." in name:
            expert_match = EXPERT_RE.match(name)
            if expert_match is None:
                raise ContractError(f"无法解析 routed expert tensor，拒绝猜测：{name}")
            expert_layers_seen.add(layer)
            if int(expert_match.group(2)) not in routes[layer]:
                continue
        selected.append(name)
    for layer in selected_layers:
        if not any(name.startswith(f"layers.{layer}.") for name in selected):
            raise ContractError(f"L{layer} 没有被选中任何 tensor")
    # Trace 必须覆盖所有层；即便某层 index 不出现 experts，也不允许静默删掉该层 trace。
    if set(routes) != selected_layers:
        raise ContractError("route trace 层集合不完整")
    return selected


def load_hash_lock(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    lock = read_json(path)
    if lock.get("repo") != REPO or lock.get("revision") != REVISION:
        raise ContractError("range hash lock 身份不匹配")
    ranges = lock.get("ranges")
    if not isinstance(ranges, dict):
        raise ContractError("range hash lock 缺少 ranges")
    result = {}
    for key, value in ranges.items():
        digest = str(value)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError(f"非法 range SHA-256：{key}")
        result[str(key)] = digest
    return result


def load_asset_lock(path: Path | None, required: list[str]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    lock = read_json(path)
    if lock.get("repo") != REPO or lock.get("revision") != REVISION:
        raise ContractError("tokenizer asset lock 身份不匹配")
    assets = lock.get("assets")
    if not isinstance(assets, dict):
        raise ContractError("tokenizer asset lock 缺少 assets")
    result: dict[str, dict[str, Any]] = {}
    for name in required:
        row = assets.get(name)
        if not isinstance(row, dict) or int(row.get("bytes", 0)) <= 0:
            raise ContractError(f"tokenizer asset 未锁定字节：{name}")
        digest = str(row.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError(f"tokenizer asset 未锁定 SHA-256：{name}")
        result[name] = {"bytes": int(row["bytes"]), "sha256": digest}
    return result


def make_plan(
    source: dict[str, Any],
    weight_map: dict[str, str],
    headers: dict[str, dict[str, Any]],
    routes: dict[int, set[int]],
    hash_lock: dict[str, str] | None = None,
    asset_lock: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hash_lock = hash_lock or {}
    asset_lock = asset_lock or {}
    names = select_names(weight_map, source, routes)
    entries: list[dict[str, Any]] = []
    segment = 0
    cursor = 0
    payload_bytes = 0
    padding_bytes = 0
    segment_sizes: list[int] = []
    for name in names:
        filename = weight_map[name]
        if filename not in headers:
            raise ContractError(f"缺少 {filename} header")
        header = headers[filename]
        meta = header["tensors"].get(name)
        if not isinstance(meta, dict):
            raise ContractError(f"header 中缺少 index tensor：{name}")
        begin, end = map(int, meta["data_offsets"])
        size = end - begin
        if size > SEGMENT_BYTES:
            raise ContractError(f"单 tensor 超过 1 GiB segment，需先实现 split：{name}")
        aligned = align_up(cursor)
        if aligned + size > SEGMENT_BYTES and cursor:
            segment_sizes.append(cursor)
            segment += 1
            cursor = 0
            aligned = 0
        padding_bytes += aligned - cursor
        source_start = int(header["data_start"]) + begin
        source_end = int(header["data_start"]) + end - 1
        range_key = f"{filename}:{source_start}-{source_end}"
        entries.append({
            "tensor": name,
            "dtype": meta.get("dtype"),
            "shape": meta.get("shape"),
            "payload_bytes": size,
            "source_file": filename,
            "source_start": source_start,
            "source_end_inclusive": source_end,
            "range_key": range_key,
            "expected_sha256": hash_lock.get(range_key),
            "segment": segment,
            "segment_offset": aligned,
        })
        cursor = aligned + size
        payload_bytes += size
    segment_sizes.append(cursor)
    binary_bytes = sum(segment_sizes)
    locked = sum(row["expected_sha256"] is not None for row in entries)
    missing = len(entries) - locked
    missing_assets = [name for name in source["tokenizer_required"] if name not in asset_lock]
    asset_bytes = sum(int(row["bytes"]) for row in asset_lock.values())
    source_files = sorted({row["source_file"] for row in entries})
    return {
        "format": "polaris-s14-range-pack-plan-v1",
        "status": "ready_to_materialize" if missing == 0 and not missing_assets else "blocked_missing_integrity_locks",
        "repo": REPO,
        "revision": REVISION,
        "selected_layers": source["selected_layers"],
        "route_trace_required": True,
        "tokenizer_assets": {
            "required": source["tokenizer_required"],
            "locked": asset_lock,
            "missing": missing_assets,
        },
        "source_files": source_files,
        "source_file_count": len(source_files),
        "tensor_count": len(entries),
        "layout": {
            "alignment": ALIGNMENT,
            "segment_limit_bytes": SEGMENT_BYTES,
            "segment_count": segment + 1,
            "segment_sizes": segment_sizes,
            "payload_bytes": payload_bytes,
            "padding_bytes": binary_bytes - payload_bytes,
            "binary_bytes": binary_bytes,
            "tokenizer_asset_bytes": asset_bytes,
            "package_bytes": binary_bytes + asset_bytes,
        },
        "integrity": {
            "range_sha256_locked": locked,
            "range_sha256_missing": missing,
            "materialize_requires_all_locked": True,
            "transport": "HTTPS + exact Content-Range + fixed revision + prelocked range SHA-256",
        },
        "entries": entries,
        "claim_limit": "该计划只证明精确字节选择与布局；不证明 S14 能运行、质量提升或达到 Claude/GPT。",
    }


def make_environment_budget(source: dict[str, Any], *, overlay_gib: float, ram_gib: float, hbm_gib: float) -> dict[str, Any]:
    package = int(source["shard_union_budget"]["total_bytes"])
    overlay = int(overlay_gib * GIB)
    ram = int(ram_gib * GIB)
    hbm = int(hbm_gib * GIB)
    safety = 2 * GIB
    direct_fit = overlay >= package
    safe_fit = overlay >= package + safety
    return {
        "format": "polaris-s14-gitcode-budget-v1",
        "package_upper_bound_bytes": package,
        "package_upper_bound_gib": package / GIB,
        "environment": {
            "overlay_bytes": overlay,
            "overlay_gib": overlay_gib,
            "ram_bytes": ram,
            "ram_gib": ram_gib,
            "hbm_bytes": hbm,
            "hbm_gib": hbm_gib,
        },
        "overlay": {
            "raw_fit": direct_fit,
            "remaining_bytes_if_raw_fit": overlay - package,
            "required_safety_reserve_bytes": safety,
            "safe_fit": safe_fit,
            "decision": "reject_overlay_output" if not safe_fit else "allow_after_statvfs_probe",
        },
        "ram_stream": {
            "full_package_fit": ram >= package + 8 * GIB,
            "recommended": True,
            "mode": "1 GiB immutable segments in anonymous RAM, consume-by-layer, release after HBM upload",
            "must_not_use_dev_shm": True,
            "peak_hbm_policy": "only current layer non-expert tensors + traced hot expert pages; never full 52.231 GB",
        },
        "external_output": {
            "allowed": True,
            "minimum_free_bytes": package + safety,
            "requirements": ["statvfs probe", "atomic segment rename", "resume journal", "all range SHA-256 locked"],
        },
        "stop_conditions": [
            "actual overlay/output free bytes < exact binary_bytes + manifest/journal + 2 GiB reserve",
            "anonymous RAM available < current segment + runtime reserve",
            "any response is not HTTP 206 with exact Content-Range",
            "fixed revision/index/header/route trace/range SHA-256 mismatch",
            "any selected routed expert lacks native route-trace evidence",
            "runtime attempts to place the full package in 32 GiB HBM"
        ],
    }


def iter_verified_tensors(plan: dict[str, Any]):
    """逐 tensor 下载到匿名 RAM，校验后交给 runtime adapter。

    该接口不落盘、不保留全包；调用者必须在请求下一项前消费并释放当前 bytes。
    它故意不提供“未校验就边到边执行”，避免损坏 payload 先进入 NPU。
    """
    if plan.get("status") != "ready_to_materialize":
        raise ContractError("manifest 尚未锁定全部 range SHA-256")
    for row in plan["entries"]:
        request = urllib.request.Request(
            hf_url(row["source_file"]),
            headers={
                "Range": f"bytes={row['source_start']}-{row['source_end_inclusive']}",
                "Accept-Encoding": "identity",
            },
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            expected = f"bytes {row['source_start']}-{row['source_end_inclusive']}/"
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                raise ContractError(f"RAM stream Range 契约失败：{row['range_key']}")
            payload = bytearray()
            remaining = int(row["payload_bytes"])
            while remaining:
                chunk = response.read(min(8 << 20, remaining))
                if not chunk:
                    raise ContractError(f"RAM stream 提前结束：{row['range_key']}")
                payload.extend(chunk)
                remaining -= len(chunk)
        if len(payload) != int(row["payload_bytes"]):
            raise ContractError(f"RAM stream 长度错误：{row['range_key']}")
        if sha256_bytes(payload) != row["expected_sha256"]:
            raise ContractError(f"RAM stream SHA-256 错误：{row['range_key']}")
        yield row, payload


def materialize(plan: dict[str, Any], output_dir: Path) -> None:
    if plan.get("status") != "ready_to_materialize":
        raise ContractError("manifest 尚未锁定全部 range SHA-256")
    output_dir.mkdir(parents=True, exist_ok=True)
    required = int(plan["layout"].get("package_bytes", plan["layout"]["binary_bytes"])) + 2 * GIB
    free = shutil.disk_usage(output_dir).free
    if free < required:
        raise ContractError(f"输出卷空间不足：free={free}, required={required}")
    journal_path = output_dir / "resume.json"
    journal = read_json(journal_path) if journal_path.exists() else {
        "format": "polaris-s14-resume-v1",
        "completed": {},
        "partial": {},
    }
    completed = journal["completed"]
    partial = journal.setdefault("partial", {})

    def save_journal() -> None:
        temporary = journal_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(journal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, journal_path)

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    for name, asset in plan.get("tokenizer_assets", {}).get("locked", {}).items():
        final_asset = assets_dir / name
        if final_asset.exists() and final_asset.stat().st_size == int(asset["bytes"]) and sha256_file(final_asset) == asset["sha256"]:
            continue
        request = urllib.request.Request(hf_url(name), headers={"Accept-Encoding": "identity"})
        partial_asset = final_asset.with_suffix(final_asset.suffix + ".part")
        digest = hashlib.sha256()
        received = 0
        with urllib.request.urlopen(request, timeout=300) as response, partial_asset.open("wb") as sink:
            if response.status != 200:
                raise ContractError(f"tokenizer asset HTTP 状态错误：{name} / {response.status}")
            while chunk := response.read(1 << 20):
                sink.write(chunk)
                digest.update(chunk)
                received += len(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        if received != int(asset["bytes"]) or digest.hexdigest() != asset["sha256"]:
            raise ContractError(f"tokenizer asset 字节/SHA-256 错误：{name}")
        os.replace(partial_asset, final_asset)

    for row in plan["entries"]:
        key = row["range_key"]
        segment_path = output_dir / f"pack-{int(row['segment']):03d}.bin.part"
        if completed.get(key) == row["expected_sha256"]:
            finished_path = output_dir / f"pack-{int(row['segment']):03d}.bin"
            verified_path = segment_path if segment_path.exists() else finished_path
            if not verified_path.exists():
                raise ContractError(f"完成账本存在但 segment 缺失：{key}")
            digest = hashlib.sha256()
            with verified_path.open("rb") as verified_file:
                verified_file.seek(int(row["segment_offset"]))
                remaining = int(row["payload_bytes"])
                while remaining:
                    chunk = verified_file.read(min(8 << 20, remaining))
                    if not chunk:
                        raise ContractError(f"完成账本 segment 太短：{key}")
                    digest.update(chunk)
                    remaining -= len(chunk)
            if digest.hexdigest() != row["expected_sha256"]:
                raise ContractError(f"完成账本 payload SHA-256 不一致：{key}")
            continue
        size = int(row["payload_bytes"])
        state = partial.get(key, {"completed_bytes": 0, "sha256_prefix": hashlib.sha256(b"").hexdigest()})
        done = int(state.get("completed_bytes", 0))
        if not 0 <= done <= size:
            raise ContractError(f"恢复账本 offset 越界：{key}")
        digest = hashlib.sha256()
        if done:
            if not segment_path.exists():
                raise ContractError(f"恢复账本存在但 segment 缺失：{key}")
            with segment_path.open("rb") as existing:
                existing.seek(int(row["segment_offset"]))
                remaining_prefix = done
                while remaining_prefix:
                    chunk = existing.read(min(8 << 20, remaining_prefix))
                    if not chunk:
                        raise ContractError(f"恢复 segment 短于账本：{key}")
                    digest.update(chunk)
                    remaining_prefix -= len(chunk)
            if digest.hexdigest() != state.get("sha256_prefix"):
                raise ContractError(f"恢复前缀 SHA-256 不一致：{key}")
        if done == size:
            if digest.hexdigest() != row["expected_sha256"]:
                raise ContractError(f"完整恢复 range SHA-256 不匹配：{key}")
            completed[key] = row["expected_sha256"]
            partial.pop(key, None)
            save_journal()
            continue
        request_start = int(row["source_start"]) + done
        request = urllib.request.Request(
            hf_url(row["source_file"]),
            headers={
                "Range": f"bytes={request_start}-{row['source_end_inclusive']}",
                "Accept-Encoding": "identity",
            },
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            expected_range = f"bytes {request_start}-{row['source_end_inclusive']}/"
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected_range):
                raise ContractError(f"payload Range 契约失败：{key}")
            mode = "r+b" if segment_path.exists() else "w+b"
            with segment_path.open(mode) as sink:
                sink.seek(int(row["segment_offset"]) + done)
                remaining = size - done
                while remaining:
                    chunk = response.read(min(8 << 20, remaining))
                    if not chunk:
                        raise ContractError(f"payload 提前结束：{key}")
                    sink.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                    done += len(chunk)
                    sink.flush()
                    os.fsync(sink.fileno())
                    partial[key] = {"completed_bytes": done, "sha256_prefix": digest.hexdigest()}
                    save_journal()
            if digest.hexdigest() != row["expected_sha256"]:
                raise ContractError(f"range SHA-256 不匹配：{key}")
        completed[key] = row["expected_sha256"]
        partial.pop(key, None)
        save_journal()
    for index, size in enumerate(plan["layout"]["segment_sizes"]):
        partial_path = output_dir / f"pack-{index:03d}.bin.part"
        final_path = output_dir / f"pack-{index:03d}.bin"
        segment_path = partial_path if partial_path.exists() else final_path
        if not segment_path.exists():
            raise ContractError(f"完成阶段缺少 segment：pack-{index:03d}.bin[.part]")
        with segment_path.open("r+b") as segment_file:
            segment_file.truncate(int(size))
        if segment_path == partial_path:
            os.replace(partial_path, final_path)
    (output_dir / "COMPLETE").write_text(sha256_bytes(json.dumps(plan, sort_keys=True).encode("utf-8")) + "\n", encoding="ascii")


def command_budget(args: argparse.Namespace) -> None:
    source = read_json(args.source)
    validate_source_contract(source)
    report = make_environment_budget(source, overlay_gib=args.overlay_gib, ram_gib=args.ram_gib, hbm_gib=args.hbm_gib)
    write_json(args.output, report, force=args.force)
    print(json.dumps({"status": "pass", "overlay_decision": report["overlay"]["decision"], "output": str(args.output)}, ensure_ascii=False))


def command_fetch_headers(args: argparse.Namespace) -> None:
    if not args.execute_metadata_fetch:
        raise ContractError("必须显式提供 --execute-metadata-fetch；本命令只取 header，不取 tensor payload")
    source = read_json(args.source)
    validate_source_contract(source)
    contracts = source_file_contracts(source)
    args.header_dir.mkdir(parents=True, exist_ok=True)
    for filename, row in sorted(contracts.items()):
        output = args.header_dir / f"{filename}.header.json"
        if output.exists() and not args.force:
            continue
        write_json(output, fetch_header(filename, int(row["bytes"])), force=args.force)


def command_plan(args: argparse.Namespace) -> None:
    source = read_json(args.source)
    validate_source_contract(source)
    weight_map = validate_index(args.index, source)
    routes = validate_route_trace(read_json(args.route_trace), source["selected_layers"])
    names = select_names(weight_map, source, routes)
    files = {weight_map[name] for name in names}
    headers = load_headers(args.header_dir, source, files)
    plan = make_plan(
        source,
        weight_map,
        headers,
        routes,
        load_hash_lock(args.hash_lock),
        load_asset_lock(args.asset_lock, source["tokenizer_required"]),
    )
    plan["route_trace_file_sha256"] = sha256_file(args.route_trace)
    write_json(args.output, plan, force=args.force)
    print(json.dumps({"status": plan["status"], "tensor_count": plan["tensor_count"], "binary_bytes": plan["layout"]["binary_bytes"]}, ensure_ascii=False))


def command_materialize(args: argparse.Namespace) -> None:
    if not args.execute:
        raise ContractError("materialize 是真实权重传输；必须显式提供 --execute")
    materialize(read_json(args.plan), args.output_dir)
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    budget = sub.add_parser("budget", help="生成 GitCode 资源/停止门报告，不访问网络")
    budget.add_argument("--source", type=Path, default=SOURCE_CONTRACT)
    budget.add_argument("--overlay-gib", type=float, default=50.0)
    budget.add_argument("--ram-gib", type=float, default=1536.0)
    budget.add_argument("--hbm-gib", type=float, default=32.0)
    budget.add_argument("--output", type=Path, default=HERE / "environment_budget.json")
    budget.add_argument("--force", action="store_true")
    budget.set_defaults(func=command_budget)

    fetch = sub.add_parser("fetch-headers", help="只获取 safetensors header；绝不获取 tensor payload")
    fetch.add_argument("--source", type=Path, default=SOURCE_CONTRACT)
    fetch.add_argument("--header-dir", type=Path, required=True)
    fetch.add_argument("--execute-metadata-fetch", action="store_true")
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=command_fetch_headers)

    plan = sub.add_parser("plan", help="由 index/header/native route trace 生成精确 Range 计划")
    plan.add_argument("--source", type=Path, default=SOURCE_CONTRACT)
    plan.add_argument("--index", type=Path, required=True)
    plan.add_argument("--header-dir", type=Path, required=True)
    plan.add_argument("--route-trace", type=Path, required=True)
    plan.add_argument("--hash-lock", type=Path)
    plan.add_argument("--asset-lock", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--force", action="store_true")
    plan.set_defaults(func=command_plan)

    material = sub.add_parser("materialize", help="显式批准后按 Range 写 1 GiB 可恢复 segments")
    material.add_argument("--plan", type=Path, required=True)
    material.add_argument("--output-dir", type=Path, required=True)
    material.add_argument("--execute", action="store_true")
    material.set_defaults(func=command_materialize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
