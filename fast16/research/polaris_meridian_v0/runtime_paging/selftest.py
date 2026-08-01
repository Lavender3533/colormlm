"""无需模型权重的页目录、缓存正确性和短吞吐自检。"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from page_cache import ExpertPageCache, PageSource, RuntimePage, RuntimeSpan
from page_catalog import build_safetensors_catalog, validate_catalog, write_catalog


HERE = Path(__file__).resolve().parent


def _write_synthetic_safetensors(path: Path, pages: int, page_bytes: int) -> None:
    component_bytes = page_bytes // 3
    sizes = [component_bytes, component_bytes, page_bytes - component_bytes * 2]
    header: dict[str, object] = {}
    offset = 0
    for expert in range(pages):
        for component, size in zip(("w1.weight_packed", "w2.weight_packed", "w3.weight_packed"), sizes):
            name = f"language_model.model.layers.92.block_sparse_moe.experts.{expert}.{component}"
            header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
            offset += size
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    chunk_bytes = 1024 * 1024
    with path.open("wb", buffering=0) as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        remaining = offset
        chunk_index = 0
        while remaining:
            size = min(chunk_bytes, remaining)
            handle.write(bytes([chunk_index % 251]) * size)
            remaining -= size
            chunk_index += 1


def _throughput(total_bytes: int, elapsed: float) -> float:
    return total_bytes / (1024**2) / elapsed


def run(page_mib: int, pages: int, output: Path) -> dict[str, object]:
    page_bytes = page_mib * 1024**2
    with tempfile.TemporaryDirectory(prefix="polaris-page-selftest-") as temporary:
        root = Path(temporary)
        weights = root / "synthetic-k3.safetensors"
        catalog_path = root / "catalog.json"
        _write_synthetic_safetensors(weights, pages, page_bytes)
        catalog = build_safetensors_catalog([weights], "synthetic-k3")
        validate_catalog(catalog)
        write_catalog(catalog, catalog_path)
        keys = [entry["key"] for entry in catalog["entries"]]

        assertions: dict[str, bool] = {
            "page_count_exact": len(keys) == pages,
            "page_size_exact": all(entry["nbytes"] == page_bytes for entry in catalog["entries"]),
            "one_contiguous_span_per_expert": all(len(entry["spans"]) == 1 for entry in catalog["entries"]),
        }

        # 进程内未缓存的随机页读取；操作系统页缓存状态不受本脚本控制，因此不称“冷盘”。
        source = PageSource()
        runtime_pages = [
            RuntimePage(
                entry["key"],
                entry["nbytes"],
                tuple(RuntimeSpan(Path(span["file"]), span["offset"], span["length"]) for span in entry["spans"]),
            )
            for entry in catalog["entries"]
        ]
        order = runtime_pages[::2] + runtime_pages[1::2]
        started = time.perf_counter()
        raw_payloads = [source.read(page) for page in order]
        raw_elapsed = time.perf_counter() - started
        source.close()
        assertions["raw_read_lengths_exact"] = all(len(data) == page_bytes for data in raw_payloads)

        # 四路异步预取。容量足够，完成后全部应驻留。
        prefetch_capacity = pages * page_bytes
        started = time.perf_counter()
        with ExpertPageCache(catalog_path, prefetch_capacity, workers=4) as cache:
            futures = cache.prefetch(keys)
            cache.wait(futures.values())
            prefetch_elapsed = time.perf_counter() - started
            prefetch_stats = cache.stats()
            assertions["prefetch_all_resident"] = prefetch_stats["resident_pages"] == pages

            # 热路径只反复访问一个已驻留页，避免复制 payload。
            hit_loops = 100_000
            started_hits = time.perf_counter_ns()
            identity = id(cache.get(keys[0]))
            stable_identity = True
            for _ in range(hit_loops):
                stable_identity &= id(cache.get(keys[0])) == identity
            hit_elapsed_ns = time.perf_counter_ns() - started_hits
            assertions["cache_hit_returns_same_object"] = stable_identity

        # 8 个并发请求同一个冷页，物理读取必须只有一次。
        with ExpertPageCache(catalog_path, page_bytes * 2, workers=4) as cache:
            with ThreadPoolExecutor(max_workers=8) as callers:
                results = list(callers.map(lambda _: cache.get(keys[-1]), range(8)))
            singleflight_stats = cache.stats()
            assertions["singleflight_one_physical_load"] = singleflight_stats["physical_loads"] == 1
            assertions["singleflight_payload_equal"] = len({result for result in results}) == 1

        # 两页容量加载四页，必须发生精确淘汰且不超预算。
        with ExpertPageCache(catalog_path, page_bytes * 2, workers=2) as cache:
            for key in keys[:4]:
                cache.get(key)
            eviction_stats = cache.stats()
            assertions["lru_evicted"] = eviction_stats["evictions"] == 2
            assertions["capacity_respected"] = eviction_stats["resident_bytes"] <= page_bytes * 2

        report: dict[str, object] = {
            "format": "polaris-runtime-paging-selftest-v1",
            "fixture": {
                "pages": pages,
                "page_mib": page_mib,
                "payload_mib": pages * page_mib,
                "note": "合成 Safetensors；不启动模型、不下载权重。",
            },
            "catalog_summary": catalog["summary"],
            "assertions": assertions,
            "all_passed": all(assertions.values()),
            "benchmarks": {
                "process_uncached_os_cache_uncontrolled": {
                    "elapsed_seconds": raw_elapsed,
                    "mib_per_second": _throughput(pages * page_bytes, raw_elapsed),
                    "physical_read_mib": pages * page_mib,
                },
                "four_worker_prefetch_os_cache_uncontrolled": {
                    "elapsed_seconds": prefetch_elapsed,
                    "wall_mib_per_second": _throughput(pages * page_bytes, prefetch_elapsed),
                    "cache_stats": prefetch_stats,
                },
                "resident_hot_path": {
                    "iterations": hit_loops,
                    "nanoseconds_per_get": hit_elapsed_ns / hit_loops,
                    "gets_per_second": hit_loops / (hit_elapsed_ns / 1e9),
                    "payload_copy": False,
                },
            },
            "singleflight_stats": singleflight_stats,
            "eviction_stats": eviction_stats,
            "limitations": [
                "Windows 用户态无法由本脚本可靠清空系统文件缓存，磁盘读吞吐必须标为 OS cache uncontrolled。",
                "这是 SSD/RAM 页缓存原型，不包含 Vulkan 上传和 GGML 图内 remap。",
                "合成页验证字节边界与缓存并发，不证明 300B+ donor 的能力。",
            ],
        }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="北极星专家页缓存快速自检")
    parser.add_argument("--page-mib", type=int, default=2)
    parser.add_argument("--pages", type=int, default=32)
    parser.add_argument("--output", type=Path, default=HERE / "selftest_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.page_mib <= 0 or args.pages < 8:
        raise SystemExit("--page-mib 必须 >0，--pages 必须 >=8")
    report = run(args.page_mib, args.pages, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
