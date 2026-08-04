#!/usr/bin/env python3
"""Fetch only an explicit dynamic S14 Range manifest through RangeCache."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import email.utils
import http.client
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

import requests
import urllib3

try:
    from . import online_range
except ImportError:
    import online_range


MANIFEST_FORMAT = "polaris-s14-dynamic-page-fetch-manifest-v1"
# 单个 token/lane 是36项；causal-block K=8 会把同层8条 route 去重后合并，
# 上限仍是精确的 8 * 36，不接受无界 manifest。
MAX_RANGE_COUNT = 8 * 36
DEFAULT_CACHE_BUDGET_BYTES = 64 << 30
MIN_DISK_FREE_RESERVE_BYTES = 20 << 30
L3_CACHE_HYSTERESIS_BYTES = 4 << 30
_CACHE_PAIR_NAME_RE = re.compile(r"([0-9a-f]{64})\.(bin|json)")
_L3_BUDGET_LOCK_KEY = "_s14_dynamic_page_l3_budget"
_STARTUP_TRIM_GUARD = threading.Lock()
_STARTUP_TRIMMED_ROOTS: set[str] = set()
_STDOUT_PROTOCOL_LOCK = threading.Lock()
PROGRESS_FORMAT = "polaris-s14-dynamic-page-fetch-progress-v1"
PROGRESS_HEARTBEAT_SECONDS = 2.0
DEFAULT_ORIGIN_DEADLINE_SECONDS = 75.0
MIN_ORIGIN_DEADLINE_SECONDS = 5.0
MAX_ORIGIN_DEADLINE_SECONDS = 240.0
ORIGIN_IO_TIMEOUT_SECONDS = 15.0
ORIGIN_PROGRESS_CHUNK_BYTES = 256 << 10
MODELSCOPE_LFS_SNAPSHOT_FORMAT = "polaris-deepseek-fixed-revision-lfs-v1"
MODELSCOPE_RESOLVE_REVISION = "master"


class _OriginDeadlineExceeded(TimeoutError):
    pass


def _load_modelscope_lfs_snapshot(path: Path) -> dict[str, tuple[int, str]]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if (
        snapshot.get("format") != MODELSCOPE_LFS_SNAPSHOT_FORMAT
        or snapshot.get("repo") != online_range.rp.REPO
        or snapshot.get("revision") != online_range.rp.REVISION
    ):
        raise online_range.rp.ContractError(
            "ModelScope LFS snapshot format/repo/revision 不匹配"
        )
    raw_files = snapshot.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise online_range.rp.ContractError("ModelScope LFS snapshot.files 为空")
    result: dict[str, tuple[int, str]] = {}
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise online_range.rp.ContractError("ModelScope LFS snapshot file entry 非法")
        filename = raw.get("file")
        byte_count = raw.get("bytes")
        digest = raw.get("lfs_sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or "/" in filename
            or "\\" in filename
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or filename in result
        ):
            raise online_range.rp.ContractError(
                f"ModelScope LFS snapshot file identity 非法: {filename!r}"
            )
        result[filename] = (byte_count, digest)
    return result


class _RequestDeadlineExceeded(TimeoutError):
    pass


class _RequestCancelled(TimeoutError):
    pass


class _RequestProgress:
    """Bounded request snapshot shared by fetch threads and the JSONL heartbeat."""

    def __init__(
        self,
        request_id: int,
        *,
        ranges_total: int,
        requested_bytes: int,
        origin_deadline_seconds: float,
        worker_deadline_seconds: float,
    ) -> None:
        self.request_id = request_id
        self.ranges_total = ranges_total
        self.requested_bytes = requested_bytes
        self.origin_deadline_seconds = origin_deadline_seconds
        self.worker_deadline_seconds = worker_deadline_seconds
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._request_started = time.monotonic()
        self._worker_deadline = self._request_started + worker_deadline_seconds
        self._stage_started = self._request_started
        self._stage = "manifest_validated"
        self._revision = 1
        self._heartbeat_sequence = 0
        self._ranges_started = 0
        self._ranges_completed = 0
        self._ranges_failed = 0
        self._active_ranges = 0
        self._transport_bytes_read = 0
        self._last_action = "request_accepted"
        self._last_range_key: str | None = None
        self._last_origin: str | None = None
        self._last_error: str | None = None

    def set_stage(self, stage: str, action: str) -> None:
        now = time.monotonic()
        with self._lock:
            if self._stage != stage:
                self._stage = stage
                self._stage_started = now
            self._last_action = action
            self._revision += 1

    def range_started(self, range_key: str) -> None:
        with self._lock:
            self._ranges_started += 1
            self._active_ranges += 1
            self._last_action = "range_started"
            self._last_range_key = range_key
            self._revision += 1

    def transport_activity(self, action: str, range_key: str, origin: str) -> None:
        with self._lock:
            self._last_action = action
            self._last_range_key = range_key
            self._last_origin = origin
            self._revision += 1

    def add_transport_bytes(self, range_key: str, origin: str, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self._transport_bytes_read += count
            self._last_action = "range_body_read"
            self._last_range_key = range_key
            self._last_origin = origin
            self._revision += 1

    def retry_backoff(self, range_key: str) -> None:
        with self._lock:
            self._last_action = "range_retry_backoff"
            self._last_range_key = range_key
            self._revision += 1

    def range_completed(self, range_key: str, *, cache_hit: bool) -> None:
        with self._lock:
            self._ranges_completed += 1
            self._active_ranges = max(self._active_ranges - 1, 0)
            self._last_action = "range_cache_hit" if cache_hit else "range_committed"
            self._last_range_key = range_key
            self._revision += 1

    def range_failed(self, range_key: str, error: BaseException) -> None:
        with self._lock:
            self._ranges_failed += 1
            self._active_ranges = max(self._active_ranges - 1, 0)
            self._last_action = "range_failed"
            self._last_range_key = range_key
            if not self._cancelled.is_set():
                self._last_error = f"{type(error).__name__}: {error}"[:512]
            self._revision += 1

    def cancel(self, error: BaseException) -> None:
        with self._lock:
            if self._cancelled.is_set():
                return
            self._cancelled.set()
            self._last_action = "request_cancelling"
            self._last_error = f"{type(error).__name__}: {error}"[:512]
            self._revision += 1

    def enforce_worker_deadline(self) -> None:
        if time.monotonic() >= self._worker_deadline:
            self.cancel(
                _RequestDeadlineExceeded(
                    f"worker request deadline 已耗尽: seconds={self.worker_deadline_seconds}"
                )
            )

    def raise_if_cancelled(self) -> None:
        self.enforce_worker_deadline()
        if self._cancelled.is_set():
            with self._lock:
                reason = self._last_error or "unknown"
            raise _RequestCancelled(f"dynamic Range request 已取消: {reason}")

    def wait_cancelled(self, timeout: float) -> bool:
        return self._cancelled.wait(timeout)

    def snapshot(self, *, heartbeat: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if heartbeat:
                self._heartbeat_sequence += 1
            return {
                "format": PROGRESS_FORMAT,
                "stage": self._stage,
                "request_elapsed_ms": round(
                    (now - self._request_started) * 1000.0, 3
                ),
                "stage_elapsed_ms": round((now - self._stage_started) * 1000.0, 3),
                "progress_revision": self._revision,
                "heartbeat_sequence": self._heartbeat_sequence,
                "ranges_total": self.ranges_total,
                "ranges_started": self._ranges_started,
                "ranges_completed": self._ranges_completed,
                "ranges_failed": self._ranges_failed,
                "active_ranges": self._active_ranges,
                "requested_bytes": self.requested_bytes,
                "origin_deadline_seconds": self.origin_deadline_seconds,
                "worker_deadline_seconds": self.worker_deadline_seconds,
                "transport_bytes_read": self._transport_bytes_read,
                "last_action": self._last_action,
                "last_range_key": self._last_range_key,
                "last_origin": self._last_origin,
                "last_error": self._last_error,
            }


def _emit_protocol(value: dict[str, Any]) -> None:
    # One lock binds each JSON object and its newline into an indivisible frame;
    # heartbeat and terminal response are otherwise produced by different threads.
    encoded = json.dumps(value, ensure_ascii=True)
    with _STDOUT_PROTOCOL_LOCK:
        print(encoded, flush=True)


class _ProgressHeartbeat:
    def __init__(self, progress: _RequestProgress) -> None:
        self._progress = progress
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"s14-range-heartbeat-{progress.request_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        _emit_protocol(
            {
                "request_id": self._progress.request_id,
                "event": "progress",
                "progress": self._progress.snapshot(heartbeat=True),
            }
        )

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=PROGRESS_HEARTBEAT_SECONDS + 1.0)

    def _run(self) -> None:
        while not self._stop.wait(PROGRESS_HEARTBEAT_SECONDS):
            try:
                self._progress.enforce_worker_deadline()
                _emit_protocol(
                    {
                        "request_id": self._progress.request_id,
                        "event": "progress",
                        "progress": self._progress.snapshot(heartbeat=True),
                    }
                )
            except (BrokenPipeError, OSError):
                self._stop.set()
                return


def _complete_cache_pair(cache_root: Path, key: str) -> dict[str, Any] | None:
    """只把结构闭合且 proof 身份一致的 ``.bin/.json`` 视为可淘汰项。"""

    payload = cache_root / f"{key}.bin"
    proof_path = cache_root / f"{key}.json"
    try:
        if (
            payload.is_symlink()
            or proof_path.is_symlink()
            or not payload.is_file()
            or not proof_path.is_file()
        ):
            return None
        payload_stat = payload.stat()
        proof_stat = proof_path.stat()
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        proof_bytes = proof.get("bytes") if isinstance(proof, dict) else None
        if (
            not isinstance(proof, dict)
            or proof.get("format") != online_range.CACHE_META_FORMAT
            or proof.get("cache_key") != key
            or isinstance(proof_bytes, bool)
            or not isinstance(proof_bytes, int)
            or proof_bytes <= 0
            or proof_bytes != payload_stat.st_size
        ):
            return None
        return {
            "key": key,
            "payload": payload,
            "proof": proof_path,
            "payload_bytes": payload_stat.st_size,
            "pair_bytes": payload_stat.st_size + proof_stat.st_size,
            # Windows/挂载选项可能不更新 atime；mtime 仅作为从未命中记录时的
            # 稳定回退。绝不 touch mtime，因为 Rust lease 把它纳入强身份。
            "last_access_ns": max(
                payload_stat.st_atime_ns,
                payload_stat.st_mtime_ns,
            ),
        }
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _scan_complete_cache_pairs(cache_root: Path) -> tuple[list[dict[str, Any]], int]:
    keys: set[str] = set()
    try:
        children = tuple(cache_root.iterdir())
    except OSError as error:
        raise online_range.rp.ContractError(
            f"Range L3 cache root 无法枚举: {cache_root}: {error}"
        ) from error
    for path in children:
        match = _CACHE_PAIR_NAME_RE.fullmatch(path.name)
        if match is not None:
            keys.add(match.group(1))
    complete: list[dict[str, Any]] = []
    incomplete_or_unknown = 0
    for key in keys:
        pair = _complete_cache_pair(cache_root, key)
        if pair is None:
            incomplete_or_unknown += 1
        else:
            complete.append(pair)
    return complete, incomplete_or_unknown


@contextlib.contextmanager
def _try_process_cache_key_lock(cache_root: Path, key: str) -> Any:
    """非阻塞取得既有 key 锁；活跃下载或其他淘汰者一律跳过。"""

    local_lock = online_range._key_lock(cache_root, key)
    if not local_lock.acquire(blocking=False):
        yield False
        return
    lock_file = None
    os_locked = False
    try:
        lock_path = (cache_root / f"{key}.lock").resolve()
        try:
            lock_path.relative_to(cache_root)
            lock_file = lock_path.open("a+b")
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            os_locked = True
        except (OSError, ValueError):
            if lock_file is not None:
                lock_file.close()
                lock_file = None
            yield False
            return
        yield True
    finally:
        if os_locked and lock_file is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()
        local_lock.release()


def _evict_complete_cache_pair(
    cache_root: Path,
    pair: dict[str, Any],
) -> tuple[bool, str | None]:
    """先把 proof/payload 成对移出命名空间，再删除；共享冲突不破坏原对。"""

    key = str(pair["key"])
    current = _complete_cache_pair(cache_root, key)
    if current is None:
        return False, "changed_or_incomplete"
    nonce = f"{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
    proof_tomb = cache_root / f".s14-l3-evict.{key}.{nonce}.json"
    payload_tomb = cache_root / f".s14-l3-evict.{key}.{nonce}.bin"
    try:
        os.replace(current["proof"], proof_tomb)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) in (32, 33):
            return False, "windows_sharing_violation"
        return False, "proof_rename_conflict"
    try:
        os.replace(current["payload"], payload_tomb)
    except OSError as error:
        try:
            os.replace(proof_tomb, current["proof"])
        except OSError as restore_error:
            raise online_range.rp.ContractError(
                f"Range L3 淘汰回滚 proof 失败: key={key}: {restore_error}"
            ) from restore_error
        if os.name == "nt" and getattr(error, "winerror", None) in (32, 33):
            return False, "windows_sharing_violation"
        return False, "payload_rename_conflict"

    cleanup_errors: list[str] = []
    for tomb in (payload_tomb, proof_tomb):
        try:
            tomb.unlink()
        except OSError as error:
            cleanup_errors.append(f"{tomb.name}:{type(error).__name__}")
    return True, ",".join(cleanup_errors) or None


def _trim_l3_cache(
    *,
    cache_root: Path,
    cache_budget_bytes: int,
    disk_free_reserve_bytes: int,
    projected_download_bytes: int,
    protected_keys: set[str],
    phase: str,
    target_cache_bytes: int | None = None,
) -> dict[str, Any]:
    """在调用者持有根级锁时按 LRU 收缩完整缓存对。"""

    if target_cache_bytes is None:
        target_cache_bytes = cache_budget_bytes
    if not 0 < target_cache_bytes <= cache_budget_bytes:
        raise online_range.rp.ContractError(
            "Range L3 target_cache_bytes 必须位于 1..=cache_budget_bytes"
        )
    pairs, incomplete_or_unknown = _scan_complete_cache_pairs(cache_root)
    cache_bytes = sum(int(pair["pair_bytes"]) for pair in pairs)
    free_bytes = shutil.disk_usage(cache_root).free
    telemetry: dict[str, Any] = {
        "phase": phase,
        "cache_bytes_before": cache_bytes,
        "disk_free_bytes_before": free_bytes,
        "cache_budget_bytes": cache_budget_bytes,
        "target_cache_bytes": target_cache_bytes,
        "disk_free_reserve_bytes": disk_free_reserve_bytes,
        "projected_download_bytes": projected_download_bytes,
        "protected_keys": len(protected_keys),
        "complete_pairs_before": len(pairs),
        "incomplete_or_unknown_pairs_ignored": incomplete_or_unknown,
        "evicted_pairs": 0,
        "evicted_payload_bytes": 0,
        "evicted_pair_bytes": 0,
        "skipped_protected": 0,
        "skipped_locked_or_active": 0,
        "skipped_changed_or_incomplete": 0,
        "tombstone_cleanup_errors": [],
    }

    def constraints_satisfied() -> bool:
        return (
            cache_bytes + projected_download_bytes <= target_cache_bytes
            and free_bytes - projected_download_bytes >= disk_free_reserve_bytes
        )

    for pair in sorted(pairs, key=lambda row: (int(row["last_access_ns"]), str(row["key"]))):
        if constraints_satisfied():
            break
        key = str(pair["key"])
        if key in protected_keys:
            telemetry["skipped_protected"] += 1
            continue
        with _try_process_cache_key_lock(cache_root, key) as acquired:
            if not acquired:
                telemetry["skipped_locked_or_active"] += 1
                continue
            current = _complete_cache_pair(cache_root, key)
            if current is None:
                telemetry["skipped_changed_or_incomplete"] += 1
                continue
            evicted, detail = _evict_complete_cache_pair(cache_root, current)
            if not evicted:
                if detail == "changed_or_incomplete":
                    telemetry["skipped_changed_or_incomplete"] += 1
                else:
                    telemetry["skipped_locked_or_active"] += 1
                continue
            telemetry["evicted_pairs"] += 1
            telemetry["evicted_payload_bytes"] += int(current["payload_bytes"])
            telemetry["evicted_pair_bytes"] += int(current["pair_bytes"])
            if detail:
                telemetry["tombstone_cleanup_errors"].append(detail)
            cache_bytes = max(0, cache_bytes - int(current["pair_bytes"]))
            free_bytes = shutil.disk_usage(cache_root).free

    telemetry["cache_bytes_after"] = cache_bytes
    telemetry["disk_free_bytes_after"] = free_bytes
    telemetry["projected_cache_bytes_after"] = (
        cache_bytes + projected_download_bytes
    )
    telemetry["projected_disk_free_bytes_after"] = (
        free_bytes - projected_download_bytes
    )
    telemetry["constraints_satisfied"] = constraints_satisfied()
    if not telemetry["constraints_satisfied"]:
        raise online_range.rp.ContractError(
            "Range L3 cache 无法满足 projected budget；"
            + json.dumps(telemetry, ensure_ascii=False, sort_keys=True)
        )
    return telemetry


def _touch_cache_pair_access(cache_root: Path, keys: set[str]) -> int:
    """只更新 atime，保留被 Rust 强身份门使用的 mtime。"""

    touched = 0
    now = time.time_ns()
    for key in keys:
        pair = _complete_cache_pair(cache_root, key)
        if pair is None:
            continue
        pair_touched = True
        for path in (pair["payload"], pair["proof"]):
            try:
                stat = path.stat()
                os.utime(
                    path,
                    ns=(now, stat.st_mtime_ns),
                )
            except (OSError, NotImplementedError):
                pair_touched = False
                break
        if pair_touched:
            touched += 1
    return touched


def _fast_l3_budget_snapshot(
    cache: online_range.RangeCache,
    *,
    projected_download_bytes: int,
    cache_budget_bytes: int,
    disk_free_reserve_bytes: int,
) -> dict[str, Any]:
    """热路径只读 RangeCache 计数与磁盘空闲量，不枚举数万缓存文件。"""

    with cache._budget_lock:
        cache_used_bytes = cache._cache_used
        cache_reserved_bytes = cache._cache_reserved
    disk_free_bytes = shutil.disk_usage(cache.root).free
    projected_cache_bytes = (
        cache_used_bytes + cache_reserved_bytes + projected_download_bytes
    )
    projected_disk_free_bytes = disk_free_bytes - projected_download_bytes
    return {
        "phase": "predownload",
        "scan": "skipped_no_pressure",
        "cache_used_bytes": cache_used_bytes,
        "cache_reserved_bytes": cache_reserved_bytes,
        "disk_free_bytes": disk_free_bytes,
        "projected_download_bytes": projected_download_bytes,
        "projected_cache_bytes_after": projected_cache_bytes,
        "projected_disk_free_bytes_after": projected_disk_free_bytes,
        "cache_budget_bytes": cache_budget_bytes,
        "disk_free_reserve_bytes": disk_free_reserve_bytes,
        "constraints_satisfied": (
            projected_cache_bytes <= cache_budget_bytes
            and projected_disk_free_bytes >= disk_free_reserve_bytes
        ),
    }


def _l3_low_water_bytes(cache_budget_bytes: int) -> int:
    hysteresis = min(L3_CACHE_HYSTERESIS_BYTES, cache_budget_bytes // 8)
    return max(1, cache_budget_bytes - hysteresis)


@contextlib.contextmanager
def _l3_root_budget_lock(cache_root: Path) -> Any:
    """所有启动 trim、projected trim 与同一 manifest 下载共享一把根锁。"""

    with online_range._key_lock(
        cache_root, _L3_BUDGET_LOCK_KEY
    ), online_range._process_key_lock(cache_root, _L3_BUDGET_LOCK_KEY):
        yield


def _sync_range_cache_storage_accounting(cache: online_range.RangeCache) -> None:
    """外部 trim 后同步 RangeCache 的硬门计数，保留其第二道预算防线。"""

    active_bytes = sum(
        path.stat().st_size
        for path in cache.root.iterdir()
        if path.is_file()
        and (
            path.suffix == ".bin"
            or path.suffix == ".part"
            and path.name.endswith(".bin.part")
        )
    )
    with cache._budget_lock:
        if cache._download_reserved != 0 or cache._cache_reserved != 0:
            raise online_range.rp.ContractError(
                "Range L3 trim 后仍有未结算 reservation；拒绝重置缓存计数"
            )
        cache._cache_used = active_bytes


def _validate_manifest(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("format") != MANIFEST_FORMAT:
        raise online_range.rp.ContractError("dynamic Range fetch manifest format 漂移")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) > MAX_RANGE_COUNT:
        raise online_range.rp.ContractError(
            f"dynamic Range fetch entries 必须为至多{MAX_RANGE_COUNT}项数组"
        )
    if not all(isinstance(entry, dict) for entry in entries):
        raise online_range.rp.ContractError("dynamic Range fetch entry 必须为对象")
    keys = [str(entry.get("range_key", "")) for entry in entries]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise online_range.rp.ContractError("dynamic Range fetch range_key 为空或重复")
    return entries


def _manifest_request_identity(value: Any) -> tuple[list[dict[str, Any]], int]:
    entries = _validate_manifest(value)
    sizes = [entry.get("bytes") for entry in entries]
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in sizes
    ):
        raise online_range.rp.ContractError("dynamic Range fetch bytes 必须为正整数")
    return entries, sum(sizes)


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(value)
    return value


class _RequestsResponse:
    """Expose the narrow ``online_range.ResponseLike`` contract."""

    def __init__(
        self,
        response: requests.Response,
        *,
        progress: _RequestProgress | None,
        range_key: str,
        origin: str,
        origin_deadline: float,
    ) -> None:
        self._response = response
        self._progress = progress
        self._range_key = range_key
        self._origin = origin
        self._origin_deadline = origin_deadline
        self.status = response.status_code
        self.headers = response.headers

    def read(self, size: int = -1) -> bytes:
        if self._progress is not None:
            self._progress.raise_if_cancelled()
        if time.monotonic() >= self._origin_deadline:
            raise _OriginDeadlineExceeded(
                f"Range origin deadline 已耗尽: origin={self._origin}"
            )
        try:
            bounded_size = size
            if bounded_size < 0 or bounded_size > ORIGIN_PROGRESS_CHUNK_BYTES:
                bounded_size = ORIGIN_PROGRESS_CHUNK_BYTES
            chunk = self._response.raw.read(bounded_size, decode_content=False)
        except urllib3.exceptions.HTTPError as error:
            # Preserve RangeCache's resumable-prefix retry path.
            raise ConnectionError(str(error)) from error
        if self._progress is not None:
            self._progress.add_transport_bytes(
                self._range_key, self._origin, len(chunk)
            )
        return chunk

    def geturl(self) -> str:
        return self._response.url

    def __enter__(self) -> "_RequestsResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._response.close()


class _ThreadLocalRequestsRangeTransport:
    """Persistent per-worker-thread HTTPS pools with normal proxy support.

    ``urllib.request.urlopen`` creates no reusable application-level pool.  The
    resident Python process therefore still paid repeated TLS setup.  A stable
    executor thread now owns one ``requests.Session`` and reuses connections to
    both the Hugging Face resolve host and its redirected CDN.  RangeCache
    remains the sole owner of the exact 206/Content-Range/length/SHA checks.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self._sessions = 0
        self._requests = 0
        self._rate_limit_events = 0
        self._rate_limit_wait_seconds = 0.0
        self._blocked_until_by_origin: dict[str, float] = {}
        self._fallback_attempts = 0
        self._fallback_successes = 0
        self._modelscope_attestations = 0
        self._origin_deadline_seconds = float(
            os.environ.get(
                "S14_DYNAMIC_PAGE_FETCH_ORIGIN_DEADLINE_SECONDS",
                str(DEFAULT_ORIGIN_DEADLINE_SECONDS),
            )
        )
        if not (
            MIN_ORIGIN_DEADLINE_SECONDS
            <= self._origin_deadline_seconds
            <= MAX_ORIGIN_DEADLINE_SECONDS
        ):
            raise online_range.rp.ContractError(
                "S14_DYNAMIC_PAGE_FETCH_ORIGIN_DEADLINE_SECONDS 必须在5..240"
            )
        self._primary_endpoint = os.environ.get(
            "S14_DYNAMIC_PAGE_FETCH_ENDPOINT", "https://huggingface.co"
        ).rstrip("/")
        fallback_raw = os.environ.get(
            "S14_DYNAMIC_PAGE_FETCH_FALLBACK_ENDPOINTS", "https://huggingface.co"
        )
        fallback_endpoints: list[str] = []
        for candidate in fallback_raw.split(","):
            endpoint = candidate.strip().rstrip("/")
            if not endpoint or endpoint == self._primary_endpoint:
                continue
            online_range._require_https(endpoint)
            if endpoint not in fallback_endpoints:
                fallback_endpoints.append(endpoint)
        self._fallback_endpoints = tuple(fallback_endpoints)
        modelscope_endpoint = os.environ.get(
            "S14_DYNAMIC_PAGE_FETCH_MODELSCOPE_ENDPOINT", ""
        ).strip().rstrip("/")
        self._modelscope_endpoint = modelscope_endpoint or None
        self._modelscope_lfs: dict[str, tuple[int, str]] = {}
        if self._modelscope_endpoint is not None:
            online_range._require_https(self._modelscope_endpoint)
            snapshot_raw = os.environ.get(
                "S14_DYNAMIC_PAGE_FETCH_LFS_SNAPSHOT", ""
            ).strip()
            if not snapshot_raw:
                raise online_range.rp.ContractError(
                    "启用 ModelScope Range 必须提供 pinned LFS snapshot"
                )
            snapshot_path = Path(snapshot_raw).expanduser().resolve(strict=True)
            if not snapshot_path.is_file():
                raise online_range.rp.ContractError(
                    "ModelScope LFS snapshot 不是文件"
                )
            self._modelscope_lfs = _load_modelscope_lfs_snapshot(snapshot_path)
        self._rate_limit_retries = int(
            os.environ.get("S14_DYNAMIC_PAGE_FETCH_429_RETRIES", "8")
        )
        if not 1 <= self._rate_limit_retries <= 16:
            raise online_range.rp.ContractError(
                "S14_DYNAMIC_PAGE_FETCH_429_RETRIES 必须在1..16"
            )

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=4,
                pool_maxsize=1,
                max_retries=0,
                pool_block=True,
            )
            session.mount("https://", adapter)
            self._local.session = session
            with self._stats_lock:
                self._sessions += 1
        return session

    @contextlib.contextmanager
    def progress_scope(
        self, progress: _RequestProgress | None, range_key: str
    ) -> Any:
        if getattr(self._local, "progress_scope", None) is not None:
            raise online_range.rp.ContractError("Range transport progress scope 不允许嵌套")
        scope = {
            "progress": progress,
            "range_key": range_key,
            "origin_deadlines": {},
        }
        self._local.progress_scope = scope
        try:
            yield
        finally:
            self._local.progress_scope = None

    def open_range(
        self, url: str, start: int, end: int, timeout: float
    ) -> _RequestsResponse:
        online_range._require_https(url)
        candidates = self._candidate_urls(url)
        last_error: BaseException | None = None
        for index, candidate in enumerate(candidates):
            is_last = index + 1 == len(candidates)
            origin = self._origin(candidate)
            origin_deadline = self._origin_deadline(origin)
            progress, range_key = self._active_progress()
            if progress is not None:
                progress.raise_if_cancelled()
                progress.transport_activity(
                    "origin_attempt", range_key, origin
                )
            # 某个镜像已明确要求退避时，不让它的等待窗口阻塞健康 fallback。
            # 最后一个 origin 仍执行正常等待，避免所有 endpoint 同时被打爆。
            if not is_last and self._origin_blocked(origin):
                with self._stats_lock:
                    self._fallback_attempts += 1
                continue
            try:
                modelscope_identity = self._modelscope_identity(candidate)
                response = self._open_single_range(
                    candidate,
                    start,
                    end,
                    timeout,
                    origin_deadline=origin_deadline,
                    progress=progress,
                    range_key=range_key,
                    allow_extended_retry=is_last,
                    modelscope_identity=modelscope_identity,
                )
            except urllib.error.HTTPError as error:
                last_error = error
                if is_last or error.code not in {429, 500, 502, 503, 504}:
                    raise
                with self._stats_lock:
                    self._fallback_attempts += 1
                continue
            except _OriginDeadlineExceeded as error:
                last_error = error
                if is_last:
                    raise
                with self._stats_lock:
                    self._fallback_attempts += 1
                continue
            except (ConnectionError, requests.ConnectionError, requests.Timeout) as error:
                last_error = error
                if is_last:
                    raise
                with self._stats_lock:
                    self._fallback_attempts += 1
                continue
            if index > 0:
                with self._stats_lock:
                    self._fallback_successes += 1
            return response
        if last_error is not None:
            raise last_error
        raise AssertionError("Range endpoint candidate list must not be empty")

    def _candidate_urls(self, url: str) -> tuple[str, ...]:
        prefix = f"{self._primary_endpoint}/"
        if not url.startswith(prefix):
            return (url,)
        suffix = url[len(self._primary_endpoint) :]
        candidates: list[str] = []
        if self._modelscope_endpoint is not None:
            revision = urllib.parse.quote(online_range.rp.REVISION, safe="")
            marker = f"/resolve/{revision}/"
            if marker not in suffix:
                raise online_range.rp.ContractError(
                    "无法把 pinned Hugging Face URL 映射到 ModelScope master"
                )
            modelscope_suffix = suffix.replace(
                marker, f"/resolve/{MODELSCOPE_RESOLVE_REVISION}/", 1
            )
            candidates.append(f"{self._modelscope_endpoint}{modelscope_suffix}")
        candidates.append(url)
        candidates.extend(
            f"{endpoint}{suffix}" for endpoint in self._fallback_endpoints
        )
        return tuple(dict.fromkeys(candidates))

    def _modelscope_identity(self, url: str) -> tuple[int, str] | None:
        if self._modelscope_endpoint is None:
            return None
        prefix = f"{self._modelscope_endpoint}/"
        if not url.startswith(prefix):
            return None
        filename = urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1])
        identity = self._modelscope_lfs.get(filename)
        if identity is None:
            raise online_range.rp.ContractError(
                f"ModelScope Range 文件不在 pinned LFS snapshot: {filename}"
            )
        return identity

    def _active_progress(self) -> tuple[_RequestProgress | None, str]:
        scope = getattr(self._local, "progress_scope", None)
        if scope is None:
            return None, "single-shot"
        return scope["progress"], str(scope["range_key"])

    def _origin_deadline(self, origin: str) -> float:
        scope = getattr(self._local, "progress_scope", None)
        if scope is None:
            return time.monotonic() + self._origin_deadline_seconds
        deadlines = scope["origin_deadlines"]
        deadline = deadlines.get(origin)
        if deadline is None:
            deadline = time.monotonic() + self._origin_deadline_seconds
            deadlines[origin] = deadline
        return float(deadline)

    def _open_single_range(
        self,
        url: str,
        start: int,
        end: int,
        timeout: float,
        *,
        origin_deadline: float,
        progress: _RequestProgress | None,
        range_key: str,
        allow_extended_retry: bool,
        modelscope_identity: tuple[int, str] | None,
    ) -> _RequestsResponse:
        origin = self._origin(url)
        connection_attempt = 0
        rate_limit_attempt = 0
        while True:
            if progress is not None:
                progress.raise_if_cancelled()
            remaining = origin_deadline - time.monotonic()
            if remaining <= 0.0:
                raise _OriginDeadlineExceeded(
                    f"Range origin deadline 已耗尽: origin={origin} range_key={range_key}"
                )
            self._wait_for_origin_rate_limit(
                origin,
                origin_deadline=origin_deadline,
                progress=progress,
                range_key=range_key,
            )
            try:
                request_timeout = min(
                    float(timeout),
                    max(origin_deadline - time.monotonic(), 0.001),
                    ORIGIN_IO_TIMEOUT_SECONDS,
                )
                if progress is not None:
                    progress.transport_activity(
                        "https_range_open", range_key, origin
                    )
                response = self._session().get(
                    url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                    },
                    allow_redirects=True,
                    stream=True,
                    timeout=(request_timeout, request_timeout),
                )
                with self._stats_lock:
                    self._requests += 1
                online_range._require_https(response.url)
                if response.status_code == 429:
                    retry_after = self._retry_after_seconds(
                        response.headers.get("Retry-After"), rate_limit_attempt
                    )
                    status = response.status_code
                    reason = response.reason
                    headers = response.headers
                    final_url = response.url
                    response.close()
                    self._extend_origin_rate_limit(origin, retry_after)
                    if not allow_extended_retry or rate_limit_attempt >= self._rate_limit_retries:
                        raise urllib.error.HTTPError(
                            final_url, status, reason, headers, None
                        )
                    rate_limit_attempt += 1
                    continue
                if response.status_code >= 400:
                    status = response.status_code
                    reason = response.reason
                    headers = response.headers
                    final_url = response.url
                    response.close()
                    raise urllib.error.HTTPError(
                        final_url, status, reason, headers, None
                    )
                if modelscope_identity is not None:
                    expected_bytes, expected_lfs_sha256 = modelscope_identity
                    linked_etag = (
                        response.headers.get("X-Linked-ETag", "")
                        .strip()
                        .strip('"')
                        .lower()
                    )
                    content_range = response.headers.get("Content-Range", "")
                    match = online_range.CONTENT_RANGE_RE.fullmatch(content_range)
                    observed_file_bytes = int(match.group(3)) if match else -1
                    if (
                        response.status_code != 206
                        or linked_etag != expected_lfs_sha256
                        or observed_file_bytes != expected_bytes
                    ):
                        response.close()
                        raise online_range.rp.ContractError(
                            "ModelScope Range 未绑定 pinned LFS object: "
                            f"linked_etag={linked_etag!r}, file_bytes={observed_file_bytes}"
                        )
                    with self._stats_lock:
                        self._modelscope_attestations += 1
                return _RequestsResponse(
                    response,
                    progress=progress,
                    range_key=range_key,
                    origin=origin,
                    origin_deadline=origin_deadline,
                )
            except urllib.error.HTTPError:
                raise
            except (requests.ConnectionError, requests.Timeout) as error:
                max_connection_attempts = 3 if allow_extended_retry else 0
                if connection_attempt >= max_connection_attempts:
                    raise ConnectionError(str(error)) from error
                retry_sleep = 0.5 * (2**connection_attempt)
                if time.monotonic() + retry_sleep >= origin_deadline:
                    raise _OriginDeadlineExceeded(
                        f"Range origin deadline 不足以继续连接重试: origin={origin}"
                    ) from error
                time.sleep(retry_sleep)
                connection_attempt += 1

    @staticmethod
    def _origin(url: str) -> str:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise online_range.rp.ContractError("Range endpoint origin 必须为 HTTPS")
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def _retry_after_seconds(value: str | None, attempt: int) -> float:
        fallback = float(min(2**min(attempt, 6), 60))
        if not value:
            return fallback
        stripped = value.strip()
        if stripped.isdigit():
            return min(max(float(stripped), 1.0), 120.0)
        try:
            retry_at = email.utils.parsedate_to_datetime(stripped)
            return min(max(retry_at.timestamp() - time.time(), 1.0), 120.0)
        except (TypeError, ValueError, OverflowError):
            return fallback

    def _extend_origin_rate_limit(self, origin: str, delay_seconds: float) -> None:
        with self._stats_lock:
            self._rate_limit_events += 1
            self._blocked_until_by_origin[origin] = max(
                self._blocked_until_by_origin.get(origin, 0.0),
                time.monotonic() + delay_seconds,
            )

    def _wait_for_origin_rate_limit(
        self,
        origin: str,
        *,
        origin_deadline: float,
        progress: _RequestProgress | None,
        range_key: str,
    ) -> None:
        while True:
            if progress is not None:
                progress.raise_if_cancelled()
            with self._stats_lock:
                remaining = (
                    self._blocked_until_by_origin.get(origin, 0.0) - time.monotonic()
                )
            if remaining <= 0.0:
                return
            deadline_remaining = origin_deadline - time.monotonic()
            if deadline_remaining <= 0.0:
                raise _OriginDeadlineExceeded(
                    f"Range origin deadline 在429退避期间耗尽: origin={origin}"
                )
            if progress is not None:
                progress.transport_activity(
                    "origin_rate_limit_wait", range_key, origin
                )
            slept = min(remaining, deadline_remaining, 1.0)
            time.sleep(slept)
            with self._stats_lock:
                self._rate_limit_wait_seconds += slept

    def _origin_blocked(self, origin: str) -> bool:
        with self._stats_lock:
            return self._blocked_until_by_origin.get(origin, 0.0) > time.monotonic()

    @property
    def telemetry(self) -> dict[str, int | float | str]:
        with self._stats_lock:
            return {
                "kind": "thread_local_requests_session_pool",
                "sessions": self._sessions,
                "requests": self._requests,
                "rate_limit_events": self._rate_limit_events,
                "rate_limit_wait_seconds": round(
                    self._rate_limit_wait_seconds, 3
                ),
                "fallback_endpoints": len(self._fallback_endpoints),
                "fallback_attempts": self._fallback_attempts,
                "fallback_successes": self._fallback_successes,
                "modelscope_attestations": self._modelscope_attestations,
                "origin_deadline_seconds": self._origin_deadline_seconds,
            }

    @property
    def origin_deadline_seconds(self) -> float:
        return self._origin_deadline_seconds


def _runtime_limits() -> tuple[int, int, int, int]:
    retries = int(os.environ.get("S14_DYNAMIC_PAGE_FETCH_RETRIES", "5"))
    if retries < 1 or retries > 8:
        raise online_range.rp.ContractError(
            "S14_DYNAMIC_PAGE_FETCH_RETRIES 必须在1..8"
        )
    workers = int(os.environ.get("S14_DYNAMIC_PAGE_FETCH_WORKERS", "6"))
    if workers < 1 or workers > 12:
        raise online_range.rp.ContractError(
            "S14_DYNAMIC_PAGE_FETCH_WORKERS 必须在1..12"
        )
    cache_budget_bytes = int(
        os.environ.get(
            "S14_DYNAMIC_PAGE_CACHE_BUDGET_BYTES",
            str(DEFAULT_CACHE_BUDGET_BYTES),
        )
    )
    if cache_budget_bytes < 1 << 30 or cache_budget_bytes > 2 << 40:
        raise online_range.rp.ContractError(
            "S14_DYNAMIC_PAGE_CACHE_BUDGET_BYTES 必须在1GiB..2TiB"
        )
    disk_free_reserve_bytes = int(
        os.environ.get(
            "S14_DYNAMIC_PAGE_DISK_RESERVE_BYTES",
            str(MIN_DISK_FREE_RESERVE_BYTES),
        )
    )
    if disk_free_reserve_bytes < MIN_DISK_FREE_RESERVE_BYTES:
        raise online_range.rp.ContractError(
            "S14_DYNAMIC_PAGE_DISK_RESERVE_BYTES 不得低于20GiB"
        )
    return retries, workers, cache_budget_bytes, disk_free_reserve_bytes


def _execute_request(
    *,
    manifest: dict[str, Any],
    cache_root: Path,
    download_budget_bytes: int,
    cache_pool: dict[tuple[str, str], online_range.RangeCache],
    executor: concurrent.futures.ThreadPoolExecutor,
    transport: _ThreadLocalRequestsRangeTransport,
    retries: int,
    cache_budget_bytes: int,
    disk_free_reserve_bytes: int,
    progress: _RequestProgress,
) -> dict[str, Any]:
    request_started = time.perf_counter()
    entries, requested_bytes = _manifest_request_identity(manifest)
    if requested_bytes <= 0 or download_budget_bytes != requested_bytes:
        raise online_range.rp.ContractError(
            "download budget 必须精确等于 manifest 未就绪 Range 字节和"
        )
    if (
        progress.ranges_total != len(entries)
        or progress.requested_bytes != requested_bytes
    ):
        raise online_range.rp.ContractError("dynamic Range progress identity 漂移")
    progress.set_stage("cache_prepare", "range_cache_open")
    endpoint = os.environ.get(
        "S14_DYNAMIC_PAGE_FETCH_ENDPOINT", "https://huggingface.co"
    )
    pool_key = (str(cache_root.resolve()), endpoint.rstrip("/"))
    cache = cache_pool.get(pool_key)
    if cache is None:
        cache = online_range.RangeCache(
            cache_root,
            allow_fetch=True,
            download_budget_bytes=download_budget_bytes,
            cache_budget_bytes=cache_budget_bytes,
            cache_free_reserve_bytes=disk_free_reserve_bytes,
            endpoint=endpoint,
            transport=transport,
        )
        cache_pool[pool_key] = cache
    else:
        cache.begin_request_budget(download_budget_bytes)
    protected_keys = {
        cache._paths(cache._identity(entry))[0]
        for entry in entries
    }
    retryable = (
        TimeoutError,
        ConnectionError,
        urllib.error.URLError,
        http.client.RemoteDisconnected,
    )

    def fetch_one(entry: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
        range_key = str(entry["range_key"])
        progress.raise_if_cancelled()
        progress.range_started(range_key)
        try:
            cached = None
            with transport.progress_scope(progress, range_key):
                for attempt in range(retries):
                    progress.raise_if_cancelled()
                    try:
                        cached = cache.fetch(entry)
                        break
                    except urllib.error.HTTPError:
                        # An explicit remote HTTP status is not a transient transport
                        # break; do not multiply a 4xx/5xx across the outer retry loop.
                        raise
                    except _OriginDeadlineExceeded:
                        # The deadline is cumulative across this range's retries;
                        # starting the same origin again cannot make progress.
                        raise
                    except retryable:
                        if attempt + 1 >= retries:
                            raise
                        retry_sleep = min(1 << attempt, 8)
                        progress.retry_backoff(range_key)
                        if progress.wait_cancelled(retry_sleep):
                            progress.raise_if_cancelled()
            if cached is None:
                raise online_range.rp.ContractError("dynamic Range retry 状态未闭合")
            proof = cached.proof
            if (
                proof.get("format") != online_range.CACHE_META_FORMAT
                or proof.get("verified_transport") != "HTTPS/206/exact-Content-Range"
                or not cached.path.is_file()
                or cached.path.stat().st_size != int(entry["bytes"])
            ):
                raise online_range.rp.ContractError(
                    f"dynamic Range fetch postcondition 失败: {range_key}"
                )
            progress.range_completed(range_key, cache_hit=bool(cached.cache_hit))
            return (
                int(cached.cache_hit),
                int(not cached.cache_hit),
                {
                    "range_key": range_key,
                    "cache_hit": cached.cache_hit,
                    "path": str(cached.path),
                    "observed_sha256": proof["observed_sha256"],
                },
            )
        except Exception as error:
            progress.range_failed(range_key, error)
            raise

    # 每个 manifest 至多288项；RangeCache 内部已有预算锁、proof锁与 keyed
    # file lock。受控并发只重叠 HTTPS Range I/O，仍逐项执行精确206/长度/SHA门。
    startup_trim: dict[str, Any] | None = None
    cache_root_identity = str(cache.root)
    if os.name == "nt":
        cache_root_identity = cache_root_identity.casefold()
    # 根锁覆盖 trim 与本 manifest 的全部并行 fetch。这样多个 resident
    # worker/进程不能在分别通过 projected 检查后同时把磁盘写爆；锁序恒为
    # root -> 非阻塞候选 key，正式 fetch 则为 root -> 当前 key，不形成反向等待。
    progress.set_stage("root_budget_lock", "wait_l3_root_budget_lock")
    with _l3_root_budget_lock(cache.root):
        with _STARTUP_TRIM_GUARD:
            needs_startup_trim = cache_root_identity not in _STARTUP_TRIMMED_ROOTS
        if needs_startup_trim:
            progress.set_stage("startup_trim", "scan_complete_cache_pairs")
            startup_trim = _trim_l3_cache(
                cache_root=cache.root,
                cache_budget_bytes=cache_budget_bytes,
                disk_free_reserve_bytes=disk_free_reserve_bytes,
                projected_download_bytes=0,
                protected_keys=protected_keys,
                phase="startup",
                target_cache_bytes=_l3_low_water_bytes(cache_budget_bytes),
            )
            _sync_range_cache_storage_accounting(cache)
            with _STARTUP_TRIM_GUARD:
                _STARTUP_TRIMMED_ROOTS.add(cache_root_identity)
        progress.set_stage("predownload_check", "l3_budget_snapshot")
        predownload_trim = _fast_l3_budget_snapshot(
            cache,
            projected_download_bytes=download_budget_bytes,
            cache_budget_bytes=cache_budget_bytes,
            disk_free_reserve_bytes=disk_free_reserve_bytes,
        )
        if not predownload_trim["constraints_satisfied"]:
            progress.set_stage("predownload_trim", "evict_complete_cache_pairs")
            predownload_trim = _trim_l3_cache(
                cache_root=cache.root,
                cache_budget_bytes=cache_budget_bytes,
                disk_free_reserve_bytes=disk_free_reserve_bytes,
                projected_download_bytes=download_budget_bytes,
                protected_keys=protected_keys,
                phase="predownload",
                target_cache_bytes=_l3_low_water_bytes(cache_budget_bytes),
            )
            _sync_range_cache_storage_accounting(cache)
        progress.set_stage("fetch_ranges", "submit_range_futures")
        futures = [executor.submit(fetch_one, entry) for entry in entries]
        try:
            result_slots: list[tuple[int, int, dict[str, Any]] | None] = [
                None
            ] * len(futures)
            future_indexes = {future: index for index, future in enumerate(futures)}
            for future in concurrent.futures.as_completed(futures):
                result_slots[future_indexes[future]] = future.result()
            if any(result is None for result in result_slots):
                raise online_range.rp.ContractError(
                    "dynamic Range future result 未形成完整有序闭包"
                )
            results = [result for result in result_slots if result is not None]
        except Exception as error:
            progress.cancel(error)
            for future in futures:
                future.cancel()
            # Running futures observe the cancellation event before retry/open/read.
            # Waiting keeps the persistent worker request boundary one-shot: no task
            # from a failed request may leak into the following JSONL request.
            concurrent.futures.wait(futures)
            raise
        progress.set_stage("cache_finalize", "touch_committed_cache_pairs")
        touched_pairs = _touch_cache_pair_access(cache.root, protected_keys)
        post_request_storage = cache.cache_storage_telemetry
    hits = sum(row[0] for row in results)
    misses = sum(row[1] for row in results)
    rows = [row[2] for row in results]
    downloaded_bytes = sum(
        int(entry["bytes"])
        for entry, row in zip(entries, rows, strict=True)
        if not bool(row["cache_hit"])
    )
    request_wall_ms = (time.perf_counter() - request_started) * 1000.0
    progress.set_stage("complete", "terminal_result_ready")
    return {
        "format": "polaris-s14-dynamic-page-fetch-result-v1",
        "range_count": len(rows),
        "requested_bytes": requested_bytes,
        "cache_hits": hits,
        "cache_misses": misses,
        "downloaded_bytes": downloaded_bytes,
        "request_wall_ms": request_wall_ms,
        "effective_download_mib_s": (
            downloaded_bytes / (1024.0 * 1024.0) / (request_wall_ms / 1000.0)
            if downloaded_bytes and request_wall_ms > 0.0
            else 0.0
        ),
        "ranges": rows,
        "proof_cache": cache.proof_cache_telemetry,
        "cache_storage": cache.cache_storage_telemetry,
        "l3_cache": {
            "policy": "complete-pair-lru/root-serialized/projected-budget-v1",
            "cache_budget_env": "S14_DYNAMIC_PAGE_CACHE_BUDGET_BYTES",
            "disk_reserve_env": "S14_DYNAMIC_PAGE_DISK_RESERVE_BYTES",
            "cache_budget_bytes": cache_budget_bytes,
            "disk_free_reserve_bytes": disk_free_reserve_bytes,
            "manifest_protected_keys": len(protected_keys),
            "startup_trim": startup_trim,
            "predownload_trim": predownload_trim,
            "touched_pairs": touched_pairs,
            "post_request": post_request_storage,
        },
        "transport": transport.telemetry,
    }


def _serve() -> int:
    (
        retries,
        workers,
        cache_budget_bytes,
        disk_free_reserve_bytes,
    ) = _runtime_limits()
    cache_pool: dict[tuple[str, str], online_range.RangeCache] = {}
    transport = _ThreadLocalRequestsRangeTransport()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for raw in sys.stdin:
            request_id: Any = None
            progress: _RequestProgress | None = None
            heartbeat: _ProgressHeartbeat | None = None
            try:
                request = json.loads(raw)
                if not isinstance(request, dict) or request.get("op") != "fetch_manifest":
                    raise online_range.rp.ContractError("worker request op 必须为 fetch_manifest")
                request_id = request.get("request_id")
                if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
                    raise online_range.rp.ContractError("worker request_id 必须为非负整数")
                manifest = request.get("manifest")
                cache_root = request.get("cache_root")
                budget = request.get("download_budget_bytes")
                progress_format = request.get("progress_format")
                worker_deadline_seconds = request.get("worker_deadline_seconds")
                if progress_format != PROGRESS_FORMAT:
                    raise online_range.rp.ContractError(
                        "worker progress_format 漂移"
                    )
                if (
                    isinstance(worker_deadline_seconds, bool)
                    or not isinstance(worker_deadline_seconds, (int, float))
                    or not 1.0 <= float(worker_deadline_seconds) <= 86_400.0
                ):
                    raise online_range.rp.ContractError(
                        "worker_deadline_seconds 必须在1..86400"
                    )
                if not isinstance(cache_root, str) or not cache_root:
                    raise online_range.rp.ContractError("worker cache_root 必须为非空UTF-8路径")
                if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
                    raise online_range.rp.ContractError("worker download budget 必须为正整数")
                entries, requested_bytes = _manifest_request_identity(manifest)
                progress = _RequestProgress(
                    request_id,
                    ranges_total=len(entries),
                    requested_bytes=requested_bytes,
                    origin_deadline_seconds=transport.origin_deadline_seconds,
                    worker_deadline_seconds=float(worker_deadline_seconds),
                )
                heartbeat = _ProgressHeartbeat(progress)
                heartbeat.start()
                result = _execute_request(
                    manifest=manifest,
                    cache_root=Path(cache_root),
                    download_budget_bytes=budget,
                    cache_pool=cache_pool,
                    executor=executor,
                    transport=transport,
                    retries=retries,
                    cache_budget_bytes=cache_budget_bytes,
                    disk_free_reserve_bytes=disk_free_reserve_bytes,
                    progress=progress,
                )
                response = {
                    "request_id": request_id,
                    "ok": True,
                    "result": result,
                    "progress": progress.snapshot(),
                }
            except Exception as error:
                response = {
                    "request_id": request_id,
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "transport": transport.telemetry,
                }
                if progress is not None:
                    response["progress"] = progress.snapshot()
            finally:
                if heartbeat is not None:
                    heartbeat.stop()
            # JSONL stdout is a control-plane byte protocol.  ASCII escaping is
            # an independent safety belt around PYTHONUTF8/PYTHONIOENCODING so
            # localized exception text cannot depend on a Windows code page.
            _emit_protocol(response)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--download-budget-bytes", type=int)
    args = parser.parse_args()
    if args.serve:
        if any(value is not None for value in (args.manifest, args.cache_root, args.download_budget_bytes)):
            raise online_range.rp.ContractError("--serve 不接受单次manifest参数")
        return _serve()
    if args.manifest is None or args.cache_root is None or args.download_budget_bytes is None:
        raise online_range.rp.ContractError("单次模式必须提供manifest/cache-root/download-budget")
    (
        retries,
        workers,
        cache_budget_bytes,
        disk_free_reserve_bytes,
    ) = _runtime_limits()
    manifest = _load_manifest(args.manifest)
    entries, requested_bytes = _manifest_request_identity(manifest)
    transport = _ThreadLocalRequestsRangeTransport()
    progress = _RequestProgress(
        0,
        ranges_total=len(entries),
        requested_bytes=requested_bytes,
        origin_deadline_seconds=transport.origin_deadline_seconds,
        worker_deadline_seconds=86_400.0,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        result = _execute_request(
            manifest=manifest,
            cache_root=args.cache_root,
            download_budget_bytes=args.download_budget_bytes,
            cache_pool={},
            executor=executor,
            transport=transport,
            retries=retries,
            cache_budget_bytes=cache_budget_bytes,
            disk_free_reserve_bytes=disk_free_reserve_bytes,
            progress=progress,
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
