#!/usr/bin/env python3
"""Fetch only an explicit dynamic S14 Range manifest through RangeCache."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import time
import urllib.error
from pathlib import Path
from typing import Any

try:
    from . import online_range
except ImportError:
    import online_range


MANIFEST_FORMAT = "polaris-s14-dynamic-page-fetch-manifest-v1"
# 单个 token/lane 是36项；causal-block K=8 会把同层8条 route 去重后合并，
# 上限仍是精确的 8 * 36，不接受无界 manifest。
MAX_RANGE_COUNT = 8 * 36


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != MANIFEST_FORMAT:
        raise online_range.rp.ContractError("dynamic Range fetch manifest format 漂移")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) > MAX_RANGE_COUNT:
        raise online_range.rp.ContractError("dynamic Range fetch entries 必须为至多36项数组")
    if not all(isinstance(entry, dict) for entry in entries):
        raise online_range.rp.ContractError("dynamic Range fetch entry 必须为对象")
    keys = [str(entry.get("range_key", "")) for entry in entries]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise online_range.rp.ContractError("dynamic Range fetch range_key 为空或重复")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--download-budget-bytes", type=int, required=True)
    args = parser.parse_args()

    entries = _load_manifest(args.manifest)
    sizes = [entry.get("bytes") for entry in entries]
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in sizes
    ):
        raise online_range.rp.ContractError("dynamic Range fetch bytes 必须为正整数")
    requested_bytes = sum(sizes)
    if requested_bytes <= 0 or args.download_budget_bytes != requested_bytes:
        raise online_range.rp.ContractError(
            "download budget 必须精确等于 manifest 未就绪 Range 字节和"
        )
    cache = online_range.RangeCache(
        args.cache_root,
        allow_fetch=True,
        download_budget_bytes=args.download_budget_bytes,
        endpoint=os.environ.get(
            "S14_DYNAMIC_PAGE_FETCH_ENDPOINT", "https://huggingface.co"
        ),
    )
    retries = int(os.environ.get("S14_DYNAMIC_PAGE_FETCH_RETRIES", "5"))
    if retries < 1 or retries > 8:
        raise online_range.rp.ContractError(
            "S14_DYNAMIC_PAGE_FETCH_RETRIES 必须在1..8"
        )
    retryable = (
        TimeoutError,
        ConnectionError,
        urllib.error.URLError,
        http.client.RemoteDisconnected,
    )
    workers = int(os.environ.get("S14_DYNAMIC_PAGE_FETCH_WORKERS", "6"))
    if workers < 1 or workers > 12:
        raise online_range.rp.ContractError(
            "S14_DYNAMIC_PAGE_FETCH_WORKERS 必须在1..12"
        )

    def fetch_one(entry: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
        cached = None
        for attempt in range(retries):
            try:
                cached = cache.fetch(entry)
                break
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
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, max(1, len(entries)))
    ) as executor:
        results = list(executor.map(fetch_one, entries))
    hits = sum(row[0] for row in results)
    misses = sum(row[1] for row in results)
    rows = [row[2] for row in results]
    print(
        json.dumps(
            {
                "format": "polaris-s14-dynamic-page-fetch-result-v1",
                "range_count": len(rows),
                "requested_bytes": requested_bytes,
                "cache_hits": hits,
                "cache_misses": misses,
                "ranges": rows,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
