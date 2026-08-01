"""对一个已建立的真实页目录做短读取与热命中基准。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from page_cache import ExpertPageCache, PageSource, RuntimePage, RuntimeSpan, load_catalog


def evenly_spaced(items: list[dict], count: int) -> list[dict]:
    count = min(max(1, count), len(items))
    if count == 1:
        return [items[0]]
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description="真实专家页目录短基准")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hot-iterations", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    selected = evenly_spaced(catalog["entries"], args.pages)
    keys = [entry["key"] for entry in selected]
    selected_bytes = sum(int(entry["nbytes"]) for entry in selected)

    started = time.perf_counter()
    with ExpertPageCache(args.catalog, selected_bytes, workers=args.workers) as cache:
        futures = cache.prefetch(keys)
        cache.wait(futures.values())
        prefetch_seconds = time.perf_counter() - started
        prefetch_stats = cache.stats()

        hot_key = keys[len(keys) // 2]
        expected_id = id(cache.get(hot_key))
        started_hot = time.perf_counter_ns()
        stable_identity = True
        for _ in range(args.hot_iterations):
            stable_identity &= id(cache.get(hot_key)) == expected_id
        hot_nanoseconds = time.perf_counter_ns() - started_hot

        # 独立文件句柄复读均匀抽样页，防止目录 offset 错位或缓存串页。
        source = PageSource()
        checks: list[dict[str, object]] = []
        for entry in evenly_spaced(selected, min(8, len(selected))):
            page = RuntimePage(
                entry["key"],
                int(entry["nbytes"]),
                tuple(
                    RuntimeSpan(Path(span["file"]), int(span["offset"]), int(span["length"]))
                    for span in entry["spans"]
                ),
            )
            direct = source.read(page)
            resident = cache.get(page.key)
            checks.append(
                {
                    "key": page.key,
                    "bytes": len(direct),
                    "sha256_equal": hashlib.sha256(direct).digest() == hashlib.sha256(resident).digest(),
                }
            )
        source.close()

    report = {
        "format": "polaris-real-page-cache-benchmark-v1",
        "catalog": str(args.catalog.resolve()),
        "source_format": catalog["source_format"],
        "catalog_summary": catalog["summary"],
        "selection": {
            "pages": len(selected),
            "bytes": selected_bytes,
            "method": "evenly spaced over (layer, expert)",
        },
        "prefetch": {
            "workers": args.workers,
            "wall_seconds": prefetch_seconds,
            "wall_mib_per_second": selected_bytes / 1024**2 / prefetch_seconds,
            "os_cache_state": "uncontrolled",
            "stats": prefetch_stats,
        },
        "hot_path": {
            "iterations": args.hot_iterations,
            "nanoseconds_per_get": hot_nanoseconds / args.hot_iterations,
            "gets_per_second": args.hot_iterations / (hot_nanoseconds / 1e9),
            "same_object_no_payload_copy": stable_identity,
        },
        "content_checks": checks,
        "all_passed": stable_identity and all(bool(item["sha256_equal"]) for item in checks),
        "claim_boundary": "仅证明本地 GGUF 专家页 offset、并发读取和 RAM 热命中；不代表端到端 tok/s。",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
