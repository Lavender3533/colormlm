#!/usr/bin/env python3
"""只读审计 Polaris S14 Range cache，并规划不可变 pack 索引。

本工具绝不读取 payload 内容、下载权重或创建 packfile。它只读取已有 JSON
sidecar，并 stat 对应 ``.bin``，使用 sidecar 中已经提交的 SHA-256 生成确定性
JSONL 索引。真正打包必须由后续独立 writer 按此索引顺序流式复制并重新校验。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CACHE_META_FORMAT = "polaris-s14-range-cache-entry-v1"
INDEX_FORMAT = "polaris-s14-readonly-pack-index-v1"
REPORT_FORMAT = "polaris-s14-range-cache-pack-audit-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SOURCE_FILE_RE = re.compile(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors")
GIB = 1024**3
MIB = 1024**2


class AuditError(RuntimeError):
    """缓存或索引契约不成立。"""


def _json_without_duplicate_keys(text: str, source: Path) -> Any:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise AuditError(f"{source.name}: JSON 字段重复: {key}")
            value[key] = item
        return value

    try:
        return json.loads(text, object_pairs_hook=hook)
    except AuditError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{source.name}: JSON/UTF-8 无法解析: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _require_int(value: Any, field: str, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditError(f"{source.name}: {field} 必须为整数")
    return value


def _require_sha(value: Any, field: str, source: Path) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AuditError(f"{source.name}: {field} 必须为64位小写 SHA-256")
    return value


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    payload_path: Path
    bytes: int
    observed_sha256: str
    authoritative: bool
    repo: str
    revision: str
    source_file: str
    source_file_bytes: int
    start: int
    end: int
    header_tensor_table_sha256: str


@dataclass(frozen=True)
class FileRecord:
    name: str
    bytes: int
    allocated_bytes_estimate: int


def _validate_sidecar(path: Path, payload_path: Path, payload_bytes: int) -> CacheEntry:
    raw = _json_without_duplicate_keys(path.read_text(encoding="utf-8"), path)
    if not isinstance(raw, dict) or raw.get("format") != CACHE_META_FORMAT:
        raise AuditError(f"{path.name}: cache metadata format 漂移")
    cache_key = _require_sha(raw.get("cache_key"), "cache_key", path)
    if path.stem != cache_key or payload_path.stem != cache_key:
        raise AuditError(f"{path.name}: 文件名与 cache_key 不一致")

    identity = raw.get("identity")
    if not isinstance(identity, dict):
        raise AuditError(f"{path.name}: identity 必须为对象")
    required_identity = {
        "repo",
        "revision",
        "source_file",
        "source_file_bytes",
        "start",
        "end",
        "header_tensor_table_sha256",
    }
    if set(identity) != required_identity:
        raise AuditError(f"{path.name}: identity 字段集合漂移")
    repo = identity["repo"]
    revision = identity["revision"]
    source_file = identity["source_file"]
    if not isinstance(repo, str) or not repo or not isinstance(revision, str) or not revision:
        raise AuditError(f"{path.name}: repo/revision 为空或类型错误")
    if not isinstance(source_file, str) or SOURCE_FILE_RE.fullmatch(source_file) is None:
        raise AuditError(f"{path.name}: source_file 不是冻结 shard basename")
    source_file_bytes = _require_int(
        identity["source_file_bytes"], "identity.source_file_bytes", path
    )
    start = _require_int(identity["start"], "identity.start", path)
    end = _require_int(identity["end"], "identity.end", path)
    table_sha = _require_sha(
        identity["header_tensor_table_sha256"],
        "identity.header_tensor_table_sha256",
        path,
    )
    declared_bytes = _require_int(raw.get("bytes"), "bytes", path)
    if not 0 <= start <= end < source_file_bytes:
        raise AuditError(f"{path.name}: Range 越过固定 shard")
    if declared_bytes != end - start + 1 or payload_bytes != declared_bytes:
        raise AuditError(
            f"{path.name}: Range/sidecar/payload 字节不一致 "
            f"({end - start + 1}/{declared_bytes}/{payload_bytes})"
        )
    expected_key = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    if cache_key != expected_key:
        raise AuditError(f"{path.name}: cache_key 不是 identity canonical SHA-256")

    observed = _require_sha(raw.get("observed_sha256"), "observed_sha256", path)
    authoritative = raw.get("authoritative") is True
    if authoritative:
        expected = _require_sha(raw.get("expected_sha256"), "expected_sha256", path)
        if raw.get("hash_authority") != "official_lock" or expected != observed:
            raise AuditError(f"{path.name}: authoritative proof 自相矛盾")
    elif raw.get("expected_sha256") is not None or raw.get("hash_authority") != "tofu":
        raise AuditError(f"{path.name}: TOFU proof 试图冒充 authoritative")
    if raw.get("verified_transport") != "HTTPS/206/exact-Content-Range":
        raise AuditError(f"{path.name}: verified_transport 不成立")

    return CacheEntry(
        cache_key=cache_key,
        payload_path=payload_path,
        bytes=declared_bytes,
        observed_sha256=observed,
        authoritative=authoritative,
        repo=repo,
        revision=revision,
        source_file=source_file,
        source_file_bytes=source_file_bytes,
        start=start,
        end=end,
        header_tensor_table_sha256=table_sha,
    )


def _cluster_size(path: Path) -> tuple[int, str]:
    if os.name != "nt":
        try:
            return int(os.statvfs(path).f_frsize), "statvfs"
        except (AttributeError, OSError):
            return 4096, "fallback_4096"
    try:
        import ctypes
        from ctypes import wintypes

        sectors_per_cluster = wintypes.DWORD()
        bytes_per_sector = wintypes.DWORD()
        free_clusters = wintypes.DWORD()
        total_clusters = wintypes.DWORD()
        root = f"{path.resolve().drive}\\"
        ok = ctypes.windll.kernel32.GetDiskFreeSpaceW(
            root,
            ctypes.byref(sectors_per_cluster),
            ctypes.byref(bytes_per_sector),
            ctypes.byref(free_clusters),
            ctypes.byref(total_clusters),
        )
        if ok:
            return (
                int(sectors_per_cluster.value * bytes_per_sector.value),
                "GetDiskFreeSpaceW",
            )
    except (AttributeError, OSError, ValueError):
        pass
    return 4096, "fallback_4096"


def _allocated_estimate(size: int, cluster_size: int) -> int:
    return 0 if size == 0 else math.ceil(size / cluster_size) * cluster_size


def _read_hot_keys(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise AuditError("hot key JSON 必须为数组")
        rows = raw
    else:
        rows = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    keys = set()
    for value in rows:
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise AuditError(f"hot key 非法: {value!r}")
        keys.add(value)
    return keys


def _scan_root(
    cache_root: Path, cluster_size: int
) -> tuple[dict[str, FileRecord], float]:
    started = time.perf_counter()
    files: dict[str, FileRecord] = {}
    with os.scandir(cache_root) as iterator:
        for row in iterator:
            if row.is_symlink():
                raise AuditError(f"cache root 拒绝符号链接: {row.name}")
            if not row.is_file(follow_symlinks=False):
                continue
            stat = row.stat(follow_symlinks=False)
            files[row.name] = FileRecord(
                name=row.name,
                bytes=stat.st_size,
                allocated_bytes_estimate=_allocated_estimate(stat.st_size, cluster_size),
            )
    return files, time.perf_counter() - started


def _pack_rows(
    entries: Iterable[CacheEntry], hot_keys: set[str], pack_target_bytes: int
) -> tuple[list[dict[str, Any]], list[int], str]:
    rows: list[dict[str, Any]] = []
    pack_sizes: list[int] = []
    pack_id = -1
    pack_offset = 0
    entries_digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.cache_key):
        if entry.cache_key in hot_keys:
            storage: dict[str, Any] = {
                "kind": "loose_hot",
                "relative_path": f"{entry.cache_key}.bin",
            }
        else:
            if pack_id < 0 or (pack_offset > 0 and pack_offset + entry.bytes > pack_target_bytes):
                pack_id += 1
                pack_offset = 0
                pack_sizes.append(0)
            storage = {
                "kind": "immutable_pack",
                "pack": f"range-pack-{pack_id:05d}.bin",
                "offset": pack_offset,
            }
            pack_offset += entry.bytes
            pack_sizes[pack_id] = pack_offset
        row = {
            "kind": "entry",
            "cache_key": entry.cache_key,
            "bytes": entry.bytes,
            "observed_sha256": entry.observed_sha256,
            "authoritative": entry.authoritative,
            "identity": {
                "repo": entry.repo,
                "revision": entry.revision,
                "source_file": entry.source_file,
                "source_file_bytes": entry.source_file_bytes,
                "start": entry.start,
                "end": entry.end,
                "header_tensor_table_sha256": entry.header_tensor_table_sha256,
            },
            "storage": storage,
        }
        line = _canonical_bytes(row)
        entries_digest.update(line)
        entries_digest.update(b"\n")
        rows.append(row)
    return rows, pack_sizes, entries_digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    if path.exists():
        raise AuditError(f"拒绝覆盖已有输出: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_index_atomic(
    path: Path,
    header: dict[str, Any],
    rows: list[dict[str, Any]],
    footer: dict[str, Any],
) -> int:
    if path.exists():
        raise AuditError(f"拒绝覆盖已有输出: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    written = 0
    try:
        with temporary.open("xb") as handle:
            for value in (header, *rows, footer):
                line = _canonical_bytes(value) + b"\n"
                handle.write(line)
                written += len(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return written
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--index-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--hot-keys", type=Path)
    parser.add_argument("--pack-target-bytes", type=int, default=4 * GIB)
    parser.add_argument("--min-free-bytes", type=int, default=20 * GIB)
    parser.add_argument("--assumed-hash-mib-s", type=float, default=1024.0)
    parser.add_argument("--max-errors", type=int, default=32)
    args = parser.parse_args()

    if args.pack_target_bytes <= 0 or args.min_free_bytes < 0:
        raise AuditError("pack target/min free 参数非法")
    if args.assumed_hash_mib_s <= 0 or args.max_errors < 1:
        raise AuditError("hash 吞吐/max errors 参数非法")
    cache_root = args.cache_root.resolve()
    if not cache_root.is_dir():
        raise AuditError(f"cache root 不存在: {cache_root}")
    cluster_size, cluster_source = _cluster_size(cache_root)
    records, scan_seconds = _scan_root(cache_root, cluster_size)

    json_names = sorted(name for name in records if name.endswith(".json"))
    bin_names = {name for name in records if name.endswith(".bin")}
    lock_names = {name for name in records if name.endswith(".lock")}
    part_names = {name for name in records if ".part" in name}
    errors: list[str] = []
    entries: list[CacheEntry] = []
    started = time.perf_counter()
    for name in json_names:
        key = name[:-5]
        bin_name = f"{key}.bin"
        if bin_name not in records:
            errors.append(f"{name}: 配对 payload 缺失")
            continue
        try:
            entries.append(
                _validate_sidecar(
                    cache_root / name,
                    cache_root / bin_name,
                    records[bin_name].bytes,
                )
            )
        except (AuditError, OSError) as exc:
            errors.append(str(exc))
    sidecar_seconds = time.perf_counter() - started
    sidecar_keys = {entry.cache_key for entry in entries}
    all_json_stems = {name[:-5] for name in json_names}
    orphan_bins = sorted(name for name in bin_names if name[:-4] not in all_json_stems)
    errors.extend(f"{name}: payload 缺少 sidecar" for name in orphan_bins)
    if part_names:
        errors.extend(f"{name}: 存在未提交 partial" for name in sorted(part_names))

    repo_revisions = sorted({(entry.repo, entry.revision) for entry in entries})
    if len(repo_revisions) != 1:
        errors.append(f"缓存混入多个或零个 repo/revision: {repo_revisions!r}")
    hot_keys = _read_hot_keys(args.hot_keys)
    unknown_hot = sorted(hot_keys - sidecar_keys)
    if unknown_hot:
        errors.append(f"hot key 不在有效缓存中: {unknown_hot[:8]!r}")

    planning_started = time.perf_counter()
    rows, pack_sizes, entries_sha = _pack_rows(
        entries, hot_keys & sidecar_keys, args.pack_target_bytes
    )
    planning_seconds = time.perf_counter() - planning_started
    payload_bytes = sum(entry.bytes for entry in entries)
    pack_bytes = sum(pack_sizes)
    authoritative_count = sum(entry.authoritative for entry in entries)
    logical_total = sum(record.bytes for record in records.values())
    allocated_total = sum(record.allocated_bytes_estimate for record in records.values())
    metadata_bytes = sum(records[name].bytes for name in json_names)
    lock_bytes = sum(records[name].bytes for name in lock_names)
    disk = shutil.disk_usage(cache_root)
    singleton_source = (
        {"repo": repo_revisions[0][0], "revision": repo_revisions[0][1]}
        if len(repo_revisions) == 1
        else None
    )

    header = {
        "format": INDEX_FORMAT,
        "kind": "header",
        "cache_meta_format": CACHE_META_FORMAT,
        "entry_count": len(rows),
        "source": singleton_source,
        "pack_target_bytes": args.pack_target_bytes,
        "payload_sha_policy": "sidecar_observed_sha256_not_rehashed",
        "lookup_order": ["loose_hot", "immutable_pack", "remote_cold_miss"],
    }
    footer = {
        "format": INDEX_FORMAT,
        "kind": "footer",
        "entry_count": len(rows),
        "entries_jsonl_sha256": entries_sha,
        "pack_count": len(pack_sizes),
        "pack_sizes": pack_sizes,
        "packed_payload_bytes": pack_bytes,
        "loose_hot_payload_bytes": payload_bytes - pack_bytes,
    }
    estimated_index_bytes = sum(
        len(_canonical_bytes(value)) + 1 for value in (header, *rows, footer)
    )
    packed_allocated = sum(
        _allocated_estimate(size, cluster_size) for size in pack_sizes
    ) + _allocated_estimate(estimated_index_bytes, cluster_size)
    index_allocated = _allocated_estimate(estimated_index_bytes, cluster_size)
    free_after_pack = disk.free - packed_allocated
    pack_transition_safe = free_after_pack >= args.min_free_bytes
    index_publish_safe = disk.free - index_allocated >= args.min_free_bytes
    open_proxy_seconds = scan_seconds / max(1, len(records))
    hash_stream_seconds = payload_bytes / (args.assumed_hash_mib_s * MIB)
    hash_open_proxy_seconds = len(entries) * open_proxy_seconds

    report = {
        "format": REPORT_FORMAT,
        "status": "pass" if not errors else "fail",
        "read_only_audit": True,
        "payload_bytes_read": 0,
        "payload_sha_rehashed": False,
        "cache_root": str(cache_root),
        "counts": {
            "directory_files": len(records),
            "valid_entries": len(entries),
            "json_sidecars": len(json_names),
            "payload_bins": len(bin_names),
            "lock_files": len(lock_names),
            "partial_files": len(part_names),
            "authoritative_entries": authoritative_count,
            "tofu_entries": len(entries) - authoritative_count,
            "loose_hot_entries": len(hot_keys & sidecar_keys),
            "planned_pack_entries": len(entries) - len(hot_keys & sidecar_keys),
            "planned_pack_files": len(pack_sizes),
        },
        "bytes": {
            "payload_logical": payload_bytes,
            "cache_all_files_logical": logical_total,
            "cache_all_files_allocated_estimate": allocated_total,
            "sidecar_logical": metadata_bytes,
            "lock_logical": lock_bytes,
            "index_estimate": estimated_index_bytes,
            "index_allocated_estimate": index_allocated,
            "planned_pack_payload": pack_bytes,
            "planned_pack_allocated_with_index": packed_allocated,
            "initial_pack_write_amplification": (
                packed_allocated / pack_bytes if pack_bytes else 0.0
            ),
            "eventual_allocated_reduction_after_verified_loose_retirement": (
                max(0, allocated_total - packed_allocated)
            ),
        },
        "disk_guard": {
            "disk_total": disk.total,
            "disk_free_now": disk.free,
            "allocation_cluster_bytes": cluster_size,
            "allocation_cluster_source": cluster_source,
            "required_free_reserve": args.min_free_bytes,
            "estimated_free_after_index_only": disk.free - index_allocated,
            "index_publish_safe": index_publish_safe,
            "estimated_free_after_non_destructive_pack_build": free_after_pack,
            "pack_transition_safe": pack_transition_safe,
            "note": "本次未创建 pack；估算假定保留全部 loose 文件并只新增一份 pack。",
        },
        "cost": {
            "directory_enumerate_and_stat_seconds": scan_seconds,
            "sidecar_open_parse_validate_seconds": sidecar_seconds,
            "index_planning_seconds": planning_seconds,
            "full_payload_hash_performed": False,
            "assumed_hash_throughput_mib_s": args.assumed_hash_mib_s,
            "estimated_full_payload_hash_stream_seconds": hash_stream_seconds,
            "estimated_payload_file_open_proxy_seconds": hash_open_proxy_seconds,
            "estimated_full_hash_total_seconds": hash_stream_seconds
            + hash_open_proxy_seconds,
            "open_proxy_basis": "directory enumerate+stat wall / directory file count；不是 payload 实读基准",
        },
        "pack_plan": {
            "format": INDEX_FORMAT,
            "entries_jsonl_sha256": entries_sha,
            "pack_target_bytes": args.pack_target_bytes,
            "pack_sizes": pack_sizes,
            "runtime_lookup": "loose-hot 优先；命中不可变 pack 索引则按 offset/bytes 读取；否则远端 cold miss",
            "update_policy": "新增页只写 loose-hot；达到阈值后封成新 delta pack，禁止为单页重写 base pack",
            "writer_gate": "流式复制时必须重算每项 SHA、核对长度，并在完整 pack+index fsync 后原子发布",
        },
        "repo_revisions": [
            {"repo": repo, "revision": revision}
            for repo, revision in repo_revisions
        ],
        "errors_total": len(errors),
        "errors": errors[: args.max_errors],
        "errors_truncated": len(errors) > args.max_errors,
    }
    if args.report_output is not None:
        _write_json_atomic(args.report_output.resolve(), report)
    if errors:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    if args.index_output is not None:
        if not index_publish_safe:
            raise AuditError(
                "索引自身会突破磁盘保留线；拒绝发布索引计划"
            )
        actual_index_bytes = _write_index_atomic(
            args.index_output.resolve(), header, rows, footer
        )
        if actual_index_bytes != estimated_index_bytes:
            raise AuditError("索引估算字节与实际写入不一致")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
