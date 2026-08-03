#!/usr/bin/env python3
"""Fetch only an explicit dynamic S14 Range manifest through RangeCache."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
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
        for attempt in range(4):
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
                if attempt == 3:
                    raise ConnectionError(str(error)) from error
                time.sleep(0.5 * (2**attempt))
        raise AssertionError("unreachable")

    @property
    def telemetry(self) -> dict[str, int | str]:
        with self._stats_lock:
            return {
                "kind": "thread_local_requests_session_pool",
                "sessions": self._sessions,
                "requests": self._requests,
            }


def _runtime_limits() -> tuple[int, int]:
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
    return retries, workers


def _execute_request(
    *,
    manifest: dict[str, Any],
    cache_root: Path,
    download_budget_bytes: int,
    cache_pool: dict[tuple[str, str], online_range.RangeCache],
    executor: concurrent.futures.ThreadPoolExecutor,
    transport: _ThreadLocalRequestsRangeTransport,
    retries: int,
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
            endpoint=endpoint,
            transport=transport,
        )
        cache_pool[pool_key] = cache
    else:
        cache.begin_request_budget(download_budget_bytes)
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
    results = list(executor.map(fetch_one, entries))
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
        "transport": transport.telemetry,
    }


def _serve() -> int:
    retries, workers = _runtime_limits()
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
    retries, workers = _runtime_limits()
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
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
