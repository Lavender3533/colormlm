#!/usr/bin/env python3
"""将 Polaris S14 最近使用的 loose Range 合并为不可变 SSD pack。

该工具只复制已提交的 ``<cache_key>.bin + .json`` 页，不下载、不删除
loose 文件。每个 payload 在写入 pack 时重新计算 SHA-256，并与 sidecar
的证明链逐字段绑定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator


INDEX_FORMAT = "polaris-s14-range-pack-index-v1"
PROOF_FORMAT = "polaris-s14-range-cache-entry-v1"
VERIFIED_TRANSPORT = "HTTPS/206/exact-Content-Range"
MODEL_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
PACK_MAGIC = b"PS14PACK"
PACK_VERSION = 1
PACK_HEADER_BYTES = 4096
PACK_ALIGNMENT = 4096
MAX_PACK_BYTES = 4 * 1024**3
MAX_PROOF_BYTES = 1024 * 1024
MAX_PACKS = 256
MAX_ENTRIES = 250_000
COPY_CHUNK_BYTES = 8 * 1024**2
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PACK_NAME_RE = re.compile(r"^range-pack-(\d+)\.bin$")
SOURCE_FILE_RE = re.compile(r"^[^/\\]+\.safetensors$")


class PackWriterError(RuntimeError):
    pass


@dataclass(frozen=True)
class LooseEntry:
    cache_key: str
    payload_path: Path
    proof_path: Path
    bytes: int
    mtime_ns: int
    observed_sha256: str
    proof_sha256: str
    hash_authority: str
    authoritative: bool
    identity: dict[str, Any]

    def to_index_entry(self, pack_name: str, offset: int) -> dict[str, Any]:
        return {
            "pack": pack_name,
            "offset": offset,
            "bytes": self.bytes,
            "observed_sha256": self.observed_sha256,
            "proof_sha256": self.proof_sha256,
            "hash_authority": self.hash_authority,
            "authoritative": self.authoritative,
            "identity": self.identity,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把最近使用的 Polaris S14 loose Range 写入不可变 SSD pack。"
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(r"D:\models\Polaris-S14\range_cache"),
        help="loose Range cache 目录",
    )
    parser.add_argument(
        "--pack-root",
        type=Path,
        default=Path(r"D:\models\Polaris-S14\range_cache_pack"),
        help="pack 与 index.v1.json 输出目录",
    )
    parser.add_argument(
        "--max-pack-gib",
        type=float,
        default=4.0,
        help="本次新 pack 的总字节上限（包含头和对齐），范围 (0, 4] GiB",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只选页和校验 sidecar，不哈希 payload、不写 pack/index",
    )
    return parser.parse_args()


def is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PackWriterError(f"{label} 必须是 64 位小写 SHA-256")
    return value


def require_positive_int(value: object, label: str) -> int:
    if not is_plain_int(value) or value <= 0:
        raise PackWriterError(f"{label} 必须是正整数")
    return value


def canonical_cache_key(identity: dict[str, Any]) -> str:
    raw = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_identity(value: object, expected_bytes: int, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PackWriterError(f"{label}.identity 必须是 JSON object")
    required = {
        "repo",
        "revision",
        "source_file",
        "source_file_bytes",
        "start",
        "end",
        "header_tensor_table_sha256",
    }
    if set(value) != required:
        raise PackWriterError(f"{label}.identity 字段集合漂移")
    repo = value["repo"]
    revision = value["revision"]
    source_file = value["source_file"]
    source_file_bytes = value["source_file_bytes"]
    start = value["start"]
    end = value["end"]
    header_sha256 = value["header_tensor_table_sha256"]
    if repo != MODEL_REPO or revision != MODEL_REVISION:
        raise PackWriterError(f"{label}.identity repo/revision 漂移")
    if not isinstance(source_file, str) or SOURCE_FILE_RE.fullmatch(source_file) is None:
        raise PackWriterError(f"{label}.identity source_file 非法")
    if not is_plain_int(source_file_bytes) or source_file_bytes <= 0:
        raise PackWriterError(f"{label}.identity source_file_bytes 非法")
    if (
        not is_plain_int(start)
        or not is_plain_int(end)
        or start < 0
        or end < start
        or end >= source_file_bytes
        or end - start + 1 != expected_bytes
    ):
        raise PackWriterError(f"{label}.identity Range bounds/bytes 漂移")
    require_sha256(header_sha256, f"{label}.identity.header_tensor_table_sha256")
    # 用新 dict 固定索引中的字段，避免转发非预期 JSON 对象子类。
    return {
        "repo": repo,
        "revision": revision,
        "source_file": source_file,
        "source_file_bytes": source_file_bytes,
        "start": start,
        "end": end,
        "header_tensor_table_sha256": header_sha256,
    }


def read_proof(proof_path: Path, cache_key: str, payload_bytes: int) -> LooseEntry:
    try:
        proof_stat_before = proof_path.stat()
    except FileNotFoundError as error:
        raise PackWriterError(f"缺少 Range sidecar: {proof_path}") from error
    if not proof_path.is_file() or not (0 < proof_stat_before.st_size <= MAX_PROOF_BYTES):
        raise PackWriterError(f"Range sidecar 字节越界: {proof_path}")
    proof_bytes = proof_path.read_bytes()
    proof_stat_after = proof_path.stat()
    if (
        proof_stat_before.st_size != proof_stat_after.st_size
        or proof_stat_before.st_mtime_ns != proof_stat_after.st_mtime_ns
        or len(proof_bytes) != proof_stat_before.st_size
    ):
        raise PackWriterError(f"Range sidecar 读取期间发生漂移: {proof_path}")
    try:
        proof = json.loads(proof_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackWriterError(f"解析 Range sidecar 失败: {proof_path}: {error}") from error
    if not isinstance(proof, dict):
        raise PackWriterError(f"Range sidecar 不是 JSON object: {proof_path}")
    if proof.get("format") != PROOF_FORMAT or proof.get("cache_key") != cache_key:
        raise PackWriterError(f"Range sidecar format/cache_key 漂移: {proof_path}")
    declared_bytes = require_positive_int(proof.get("bytes"), f"{proof_path}.bytes")
    if declared_bytes != payload_bytes:
        raise PackWriterError(
            f"Range sidecar/payload bytes 漂移: {proof_path} "
            f"declared={declared_bytes} actual={payload_bytes}"
        )
    identity = validate_identity(proof.get("identity"), declared_bytes, str(proof_path))
    if canonical_cache_key(identity) != cache_key:
        raise PackWriterError(f"Range sidecar cache_key 不是 canonical identity 派生: {proof_path}")
    observed_sha256 = require_sha256(
        proof.get("observed_sha256"), f"{proof_path}.observed_sha256"
    )
    authoritative = proof.get("authoritative")
    if not isinstance(authoritative, bool):
        raise PackWriterError(f"{proof_path}.authoritative 必须是 bool")
    hash_authority = proof.get("hash_authority")
    expected_sha256 = proof.get("expected_sha256")
    if authoritative:
        if hash_authority != "official_lock" or expected_sha256 != observed_sha256:
            raise PackWriterError(f"authoritative Range sidecar 自相矛盾: {proof_path}")
    elif hash_authority != "tofu" or expected_sha256 is not None:
        raise PackWriterError(f"TOFU Range sidecar 试图冒充权威: {proof_path}")
    if proof.get("verified_transport") != VERIFIED_TRANSPORT:
        raise PackWriterError(f"Range sidecar verified_transport 漂移: {proof_path}")
    payload_path = proof_path.with_suffix(".bin")
    payload_stat = payload_path.stat()
    return LooseEntry(
        cache_key=cache_key,
        payload_path=payload_path,
        proof_path=proof_path,
        bytes=declared_bytes,
        mtime_ns=payload_stat.st_mtime_ns,
        observed_sha256=observed_sha256,
        proof_sha256=hashlib.sha256(proof_bytes).hexdigest(),
        hash_authority=hash_authority,
        authoritative=authoritative,
        identity=identity,
    )


def align_up(value: int, alignment: int = PACK_ALIGNMENT) -> int:
    return (value + alignment - 1) & -alignment


def validate_pack_name(name: object) -> str:
    if not isinstance(name, str) or PACK_NAME_RE.fullmatch(name) is None:
        raise PackWriterError(f"pack 文件名非法: {name!r}")
    return name


def empty_index(cache_root: Path) -> dict[str, Any]:
    return {
        "format": INDEX_FORMAT,
        "generation": 0,
        "cache_root": str(cache_root),
        "alignment": PACK_ALIGNMENT,
        "packs": {},
        "entries": {},
    }


def load_existing_index(index_path: Path, cache_root: Path, pack_root: Path) -> dict[str, Any]:
    if not index_path.exists():
        return empty_index(cache_root)
    try:
        raw = index_path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackWriterError(f"读取已有 pack index 失败: {index_path}: {error}") from error
    if not isinstance(document, dict):
        raise PackWriterError("已有 pack index 不是 JSON object")
    if document.get("format") != INDEX_FORMAT:
        raise PackWriterError("已有 pack index format 漂移")
    generation = document.get("generation")
    if not is_plain_int(generation) or generation <= 0:
        raise PackWriterError("已有 pack index generation 非法")
    declared_root = document.get("cache_root")
    if not isinstance(declared_root, str):
        raise PackWriterError("已有 pack index cache_root 非法")
    try:
        resolved_declared_root = Path(declared_root).resolve(strict=True)
    except OSError as error:
        raise PackWriterError("已有 pack index cache_root 无法解析") from error
    if resolved_declared_root != cache_root or document.get("alignment") != PACK_ALIGNMENT:
        raise PackWriterError("已有 pack index cache_root/alignment 与本次不一致")
    packs = document.get("packs")
    entries = document.get("entries")
    if not isinstance(packs, dict) or not packs or not isinstance(entries, dict) or not entries:
        raise PackWriterError("已有 pack index packs/entries 为空或非 object")
    if len(packs) > MAX_PACKS or len(entries) > MAX_ENTRIES:
        raise PackWriterError("已有 pack index packs/entries 超出 runtime 上限")
    counted: dict[str, int] = {name: 0 for name in packs}
    ranges: dict[str, list[tuple[int, int, str]]] = {name: [] for name in packs}
    for name, record in packs.items():
        validate_pack_name(name)
        if not isinstance(record, dict):
            raise PackWriterError(f"已有 pack record 非 object: {name}")
        pack_bytes = require_positive_int(record.get("bytes"), f"packs.{name}.bytes")
        pack_entries = require_positive_int(record.get("entries"), f"packs.{name}.entries")
        require_sha256(record.get("sha256"), f"packs.{name}.sha256")
        pack_path = pack_root / name
        try:
            actual_bytes = pack_path.stat().st_size
        except FileNotFoundError as error:
            raise PackWriterError(f"已有 pack 文件缺失: {pack_path}") from error
        if not pack_path.is_file() or actual_bytes != pack_bytes or pack_entries <= 0:
            raise PackWriterError(f"已有 pack bytes/entries 漂移: {pack_path}")
        with pack_path.open("rb") as handle:
            header = handle.read(24)
        if len(header) != 24:
            raise PackWriterError(f"已有 pack header 截断: {pack_path}")
        magic, version, header_bytes, _pack_id = struct.unpack("<8sIIQ", header)
        if magic != PACK_MAGIC or version != PACK_VERSION or header_bytes != PACK_HEADER_BYTES:
            raise PackWriterError(f"已有 pack header 漂移: {pack_path}")
    for cache_key, record in entries.items():
        require_sha256(cache_key, "entries.cache_key")
        if not isinstance(record, dict):
            raise PackWriterError(f"已有 pack entry 非 object: {cache_key}")
        pack_name = validate_pack_name(record.get("pack"))
        if pack_name not in packs:
            raise PackWriterError(f"已有 entry 引用未知 pack: {cache_key}")
        offset = require_positive_int(record.get("offset"), f"entries.{cache_key}.offset")
        payload_bytes = require_positive_int(record.get("bytes"), f"entries.{cache_key}.bytes")
        if offset < PACK_HEADER_BYTES or offset % PACK_ALIGNMENT != 0:
            raise PackWriterError(f"已有 entry offset 非法: {cache_key}")
        if offset + payload_bytes > packs[pack_name]["bytes"]:
            raise PackWriterError(f"已有 entry 越出 pack: {cache_key}")
        require_sha256(record.get("observed_sha256"), f"entries.{cache_key}.observed_sha256")
        require_sha256(record.get("proof_sha256"), f"entries.{cache_key}.proof_sha256")
        authoritative = record.get("authoritative")
        authority = record.get("hash_authority")
        if not isinstance(authoritative, bool) or (
            authoritative != (authority == "official_lock")
        ) or authority not in {"tofu", "official_lock"}:
            raise PackWriterError(f"已有 entry authority 非法: {cache_key}")
        identity = validate_identity(record.get("identity"), payload_bytes, cache_key)
        if canonical_cache_key(identity) != cache_key:
            raise PackWriterError(f"已有 entry canonical cache_key 漂移: {cache_key}")
        counted[pack_name] += 1
        ranges[pack_name].append((offset, offset + payload_bytes, cache_key))
    for name, count in counted.items():
        if count != packs[name]["entries"]:
            raise PackWriterError(f"已有 pack entry count 漂移: {name}")
        rows = sorted(ranges[name])
        for left, right in zip(rows, rows[1:]):
            if left[1] > right[0]:
                raise PackWriterError(
                    f"已有 pack entry 重叠: pack={name} left={left[2]} right={right[2]}"
                )
    return document


def choose_pack_id(document: dict[str, Any], pack_root: Path) -> int:
    used_ids: set[int] = set()
    for name in document["packs"]:
        match = PACK_NAME_RE.fullmatch(name)
        assert match is not None
        used_ids.add(int(match.group(1)))
    for path in pack_root.glob("range-pack-*.bin"):
        match = PACK_NAME_RE.fullmatch(path.name)
        if match is not None:
            used_ids.add(int(match.group(1)))
    return max(used_ids, default=-1) + 1


def select_recent_entries(
    cache_root: Path, existing_keys: set[str], max_pack_bytes: int
) -> tuple[list[LooseEntry], int]:
    candidates: list[tuple[int, str, Path, int]] = []
    with os.scandir(cache_root) as iterator:
        for item in iterator:
            if not item.is_file(follow_symlinks=False) or not item.name.endswith(".bin"):
                continue
            cache_key = item.name[:-4]
            if cache_key in existing_keys:
                continue
            require_sha256(cache_key, f"loose filename {item.name}")
            stat = item.stat(follow_symlinks=False)
            if stat.st_size <= 0:
                raise PackWriterError(f"loose payload 为空: {item.path}")
            candidates.append((-stat.st_mtime_ns, item.name, Path(item.path), stat.st_size))
    candidates.sort()

    selected: list[LooseEntry] = []
    cursor = PACK_HEADER_BYTES
    for neg_mtime_ns, _name, payload_path, payload_bytes in candidates:
        offset = align_up(cursor)
        end = offset + payload_bytes
        if end > max_pack_bytes:
            continue
        cache_key = payload_path.stem
        entry = read_proof(payload_path.with_suffix(".json"), cache_key, payload_bytes)
        if entry.payload_path != payload_path or entry.mtime_ns != -neg_mtime_ns:
            raise PackWriterError(f"loose payload 在选页期间发生漂移: {payload_path}")
        selected.append(entry)
        cursor = end
        if cursor == max_pack_bytes:
            break
    return selected, cursor


def write_zeros(handle: BinaryIO, hasher: "hashlib._Hash", count: int) -> None:
    zero_block = bytes(min(COPY_CHUNK_BYTES, max(count, 1)))
    remaining = count
    while remaining:
        chunk = zero_block[: min(len(zero_block), remaining)]
        handle.write(chunk)
        hasher.update(chunk)
        remaining -= len(chunk)


def copy_verified_payload(
    handle: BinaryIO, pack_hasher: "hashlib._Hash", entry: LooseEntry
) -> None:
    refreshed = read_proof(entry.proof_path, entry.cache_key, entry.bytes)
    if refreshed != entry:
        raise PackWriterError(f"Range sidecar/payload identity 在 pack 写入前漂移: {entry.cache_key}")
    before = entry.payload_path.stat()
    payload_hasher = hashlib.sha256()
    copied = 0
    with entry.payload_path.open("rb") as source:
        while True:
            chunk = source.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            handle.write(chunk)
            pack_hasher.update(chunk)
            payload_hasher.update(chunk)
            copied += len(chunk)
    after = entry.payload_path.stat()
    if (
        copied != entry.bytes
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise PackWriterError(f"Range payload 读取期间发生漂移: {entry.payload_path}")
    observed = payload_hasher.hexdigest()
    if observed != entry.observed_sha256:
        raise PackWriterError(
            f"Range payload SHA-256 漂移: {entry.cache_key} "
            f"expected={entry.observed_sha256} actual={observed}"
        )


def write_pack(
    part_path: Path,
    pack_id: int,
    pack_name: str,
    selected: list[LooseEntry],
) -> tuple[int, str, dict[str, dict[str, Any]]]:
    pack_hasher = hashlib.sha256()
    index_entries: dict[str, dict[str, Any]] = {}
    with part_path.open("xb") as handle:
        header = struct.pack("<8sIIQ", PACK_MAGIC, PACK_VERSION, PACK_HEADER_BYTES, pack_id)
        handle.write(header)
        pack_hasher.update(header)
        write_zeros(handle, pack_hasher, PACK_HEADER_BYTES - len(header))
        for entry in selected:
            offset = align_up(handle.tell())
            write_zeros(handle, pack_hasher, offset - handle.tell())
            copy_verified_payload(handle, pack_hasher, entry)
            index_entries[entry.cache_key] = entry.to_index_entry(pack_name, offset)
        handle.flush()
        os.fsync(handle.fileno())
        pack_bytes = handle.tell()
    return pack_bytes, pack_hasher.hexdigest(), index_entries


def write_index_atomic(index_path: Path, document: dict[str, Any]) -> None:
    raw = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    part_path = index_path.with_name(f"{index_path.name}.{os.getpid()}.part")
    try:
        with part_path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part_path, index_path)
    finally:
        part_path.unlink(missing_ok=True)


@contextmanager
def writer_lock(pack_root: Path) -> Iterator[None]:
    lock_path = pack_root / ".range-pack-writer.lock"
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise PackWriterError(f"已有 Range pack writer 锁: {lock_path}") from error
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if not (0.0 < args.max_pack_gib <= 4.0):
        raise PackWriterError("--max-pack-gib 必须在 (0, 4] 之间")
    max_pack_bytes = min(int(args.max_pack_gib * 1024**3), MAX_PACK_BYTES)
    if max_pack_bytes <= PACK_HEADER_BYTES:
        raise PackWriterError("pack 上限小于固定头")
    cache_root = args.cache_root.resolve(strict=True)
    if not cache_root.is_dir():
        raise PackWriterError(f"Range cache root 不是目录: {cache_root}")
    pack_root = args.pack_root.resolve(strict=False)
    pack_root.mkdir(parents=True, exist_ok=True)
    pack_root = pack_root.resolve(strict=True)
    index_path = pack_root / "index.v1.json"

    with writer_lock(pack_root):
        document = load_existing_index(index_path, cache_root, pack_root)
        selected, predicted_bytes = select_recent_entries(
            cache_root, set(document["entries"]), max_pack_bytes
        )
        if not selected:
            print(
                json.dumps(
                    {
                        "status": "no_new_entries",
                        "index": str(index_path),
                        "existing_entries": len(document["entries"]),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if len(document["packs"]) >= MAX_PACKS:
            raise PackWriterError(f"pack 数量已达 runtime 上限 {MAX_PACKS}")
        if len(document["entries"]) + len(selected) > MAX_ENTRIES:
            allowed = MAX_ENTRIES - len(document["entries"])
            if allowed <= 0:
                raise PackWriterError(f"entry 数量已达 runtime 上限 {MAX_ENTRIES}")
            selected = selected[:allowed]
            predicted_bytes = PACK_HEADER_BYTES
            for entry in selected:
                predicted_bytes = align_up(predicted_bytes) + entry.bytes
        pack_id = choose_pack_id(document, pack_root)
        pack_name = f"range-pack-{pack_id:05d}.bin"
        pack_path = pack_root / pack_name
        if pack_path.exists():
            raise PackWriterError(f"拒绝覆盖已有 pack: {pack_path}")
        free_bytes = shutil.disk_usage(pack_root).free
        if not args.dry_run and free_bytes < predicted_bytes + 1024**3:
            raise PackWriterError(
                f"pack 输出盘不足：free={free_bytes} required={predicted_bytes + 1024**3}"
            )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry_run",
                        "pack": str(pack_path),
                        "entries": len(selected),
                        "predicted_bytes": predicted_bytes,
                        "max_pack_bytes": max_pack_bytes,
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        part_path = pack_root / f"{pack_name}.{os.getpid()}.part"
        try:
            pack_bytes, pack_sha256, new_entries = write_pack(
                part_path, pack_id, pack_name, selected
            )
            if pack_bytes != predicted_bytes or pack_bytes > max_pack_bytes:
                raise PackWriterError(
                    f"pack 写入字节与预计漂移: predicted={predicted_bytes} actual={pack_bytes}"
                )
            os.replace(part_path, pack_path)
            document["generation"] = document["generation"] + 1
            document["packs"][pack_name] = {
                "bytes": pack_bytes,
                "sha256": pack_sha256,
                "entries": len(new_entries),
            }
            document["entries"].update(new_entries)
            write_index_atomic(index_path, document)
        finally:
            part_path.unlink(missing_ok=True)

        print(
            json.dumps(
                {
                    "status": "committed",
                    "pack": str(pack_path),
                    "index": str(index_path),
                    "generation": document["generation"],
                    "entries_added": len(new_entries),
                    "pack_bytes": pack_bytes,
                    "pack_sha256": pack_sha256,
                    "loose_files_deleted": 0,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except (OSError, PackWriterError) as error:
        print(f"Range pack 构建失败: {error}", file=sys.stderr)
        raise SystemExit(1) from error
