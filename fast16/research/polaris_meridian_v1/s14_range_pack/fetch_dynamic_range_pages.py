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


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(value)
    return value


class _RequestsResponse:
    """Expose the narrow ``online_range.ResponseLike`` contract."""

    def __init__(self, response: requests.Response) -> None:
        self._response = response
        self.status = response.status_code
        self.headers = response.headers

    def read(self, size: int = -1) -> bytes:
        try:
            return self._response.raw.read(size, decode_content=False)
        except urllib3.exceptions.HTTPError as error:
            # Preserve RangeCache's resumable-prefix retry path.
            raise ConnectionError(str(error)) from error

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
        self._blocked_until = 0.0
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

    def open_range(
        self, url: str, start: int, end: int, timeout: float
    ) -> _RequestsResponse:
        online_range._require_https(url)
        connection_attempt = 0
        rate_limit_attempt = 0
        while True:
            self._wait_for_global_rate_limit()
            try:
                response = self._session().get(
                    url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                    },
                    allow_redirects=True,
                    stream=True,
                    timeout=timeout,
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
                    if rate_limit_attempt >= self._rate_limit_retries:
                        raise urllib.error.HTTPError(
                            final_url, status, reason, headers, None
                        )
                    rate_limit_attempt += 1
                    self._extend_global_rate_limit(retry_after)
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
                return _RequestsResponse(response)
            except urllib.error.HTTPError:
                raise
            except (requests.ConnectionError, requests.Timeout) as error:
                if connection_attempt >= 3:
                    raise ConnectionError(str(error)) from error
                time.sleep(0.5 * (2**connection_attempt))
                connection_attempt += 1

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

    def _extend_global_rate_limit(self, delay_seconds: float) -> None:
        with self._stats_lock:
            self._rate_limit_events += 1
            self._blocked_until = max(
                self._blocked_until, time.monotonic() + delay_seconds
            )

    def _wait_for_global_rate_limit(self) -> None:
        while True:
            with self._stats_lock:
                remaining = self._blocked_until - time.monotonic()
            if remaining <= 0.0:
                return
            slept = min(remaining, 1.0)
            time.sleep(slept)
            with self._stats_lock:
                self._rate_limit_wait_seconds += slept

    @property
    def telemetry(self) -> dict[str, int | str]:
        with self._stats_lock:
            return {
                "kind": "thread_local_requests_session_pool",
                "sessions": self._sessions,
                "requests": self._requests,
                "rate_limit_events": self._rate_limit_events,
                "rate_limit_wait_seconds": round(
                    self._rate_limit_wait_seconds, 3
                ),
            }


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
) -> dict[str, Any]:
    request_started = time.perf_counter()
    entries = _validate_manifest(manifest)
    sizes = [entry.get("bytes") for entry in entries]
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in sizes
    ):
        raise online_range.rp.ContractError("dynamic Range fetch bytes 必须为正整数")
    requested_bytes = sum(sizes)
    if requested_bytes <= 0 or download_budget_bytes != requested_bytes:
        raise online_range.rp.ContractError(
            "download budget 必须精确等于 manifest 未就绪 Range 字节和"
        )
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
        cached = None
        for attempt in range(retries):
            try:
                cached = cache.fetch(entry)
                break
            except urllib.error.HTTPError:
                # An explicit remote HTTP status is not a transient transport
                # break; do not multiply a 4xx/5xx across the outer retry loop.
                raise
            except retryable:
                if attempt + 1 >= retries:
                    raise
                time.sleep(min(1 << attempt, 8))
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
                f"dynamic Range fetch postcondition 失败: {entry.get('range_key')}"
            )
        return (
            int(cached.cache_hit),
            int(not cached.cache_hit),
            {
                "range_key": entry["range_key"],
                "cache_hit": cached.cache_hit,
                "path": str(cached.path),
                "observed_sha256": proof["observed_sha256"],
            },
        )

    # 每个 manifest 至多288项；RangeCache 内部已有预算锁、proof锁与 keyed
    # file lock。受控并发只重叠 HTTPS Range I/O，仍逐项执行精确206/长度/SHA门。
    startup_trim: dict[str, Any] | None = None
    cache_root_identity = str(cache.root)
    if os.name == "nt":
        cache_root_identity = cache_root_identity.casefold()
    # 根锁覆盖 trim 与本 manifest 的全部并行 fetch。这样多个 resident
    # worker/进程不能在分别通过 projected 检查后同时把磁盘写爆；锁序恒为
    # root -> 非阻塞候选 key，正式 fetch 则为 root -> 当前 key，不形成反向等待。
    with _l3_root_budget_lock(cache.root):
        with _STARTUP_TRIM_GUARD:
            needs_startup_trim = cache_root_identity not in _STARTUP_TRIMMED_ROOTS
        if needs_startup_trim:
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
        predownload_trim = _fast_l3_budget_snapshot(
            cache,
            projected_download_bytes=download_budget_bytes,
            cache_budget_bytes=cache_budget_bytes,
            disk_free_reserve_bytes=disk_free_reserve_bytes,
        )
        if not predownload_trim["constraints_satisfied"]:
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
        results = list(executor.map(fetch_one, entries))
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
                _validate_manifest(manifest)
                if not isinstance(cache_root, str) or not cache_root:
                    raise online_range.rp.ContractError("worker cache_root 必须为非空UTF-8路径")
                if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
                    raise online_range.rp.ContractError("worker download budget 必须为正整数")
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
                )
                response = {"request_id": request_id, "ok": True, "result": result}
            except Exception as error:
                response = {
                    "request_id": request_id,
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            # JSONL stdout is a control-plane byte protocol.  ASCII escaping is
            # an independent safety belt around PYTHONUTF8/PYTHONIOENCODING so
            # localized exception text cannot depend on a Windows code page.
            print(json.dumps(response, ensure_ascii=True), flush=True)
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
    transport = _ThreadLocalRequestsRangeTransport()
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
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
