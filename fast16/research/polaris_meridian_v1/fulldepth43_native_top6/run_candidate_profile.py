"""运行一次不超过单 token 的 FullDepth43 Vulkan candidate 细分剖析。"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate_profile import CandidateProfiler, MaterializedFp8Cache, write_json
from .executor import ExecutionConfig, FullDepthError, execute
from .preflight import DEFAULT_ASSET_ROOT, DEFAULT_CATALOG


RUN_FORMAT = "polaris-fulldepth43-profiled-candidate-run-v1"


def run_profiled_candidate(
    *,
    worker: Path,
    output_root: Path,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    catalog_path: Path = DEFAULT_CATALOG,
    timeout_seconds: float = 30.0,
    fp8_cache_bytes: int = 0,
) -> dict[str, Any]:
    worker = worker.resolve(strict=True)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    if isinstance(fp8_cache_bytes, bool) or not isinstance(fp8_cache_bytes, int) or fp8_cache_bytes < 0:
        raise ValueError("fp8_cache_bytes 必须为非负整数")

    profiler = CandidateProfiler()
    cache = MaterializedFp8Cache(fp8_cache_bytes) if fp8_cache_bytes else None
    cache_scope = nullcontext() if cache is None else cache
    with profiler:
        with cache_scope:
            model_report = execute(
                ExecutionConfig(
                    asset_root=asset_root,
                    catalog_path=catalog_path,
                    report_path=output_root / "model_report.json",
                    token_count=1,
                    vulkan_bridge_capture=output_root / "captures",
                    vulkan_writeback_worker=worker,
                    vulkan_writeback_timeout_seconds=timeout_seconds,
                    vulkan_writeback_all_layers=True,
                    vulkan_writeback_verify_cpu=False,
                    vulkan_writeback_cpu_fallback=False,
                    vulkan_writeback_fast_production=True,
                )
            )
    runtime_profile = profiler.snapshot()
    runtime_profile["materialized_fp8_cache"] = None if cache is None else cache.stats()
    write_json(output_root / "runtime_profile.json", runtime_profile)

    if model_report.get("status") != "complete":
        raise FullDepthError(f"profiled candidate 未完成: {model_report.get('error')}")
    tokens = model_report.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != 1:
        raise FullDepthError("profiled candidate 必须精确执行一个 token")
    layers = tokens[0].get("layers")
    if not isinstance(layers, list) or len(layers) != 43:
        raise FullDepthError("profiled candidate 没有覆盖 43 层")
    if model_report.get("vulkan_writeback_fallbacks"):
        raise FullDepthError("profiled candidate 发生 CPU fallback")
    execution_seconds = float(model_report["execution_seconds"])
    if execution_seconds <= 0:
        raise FullDepthError("profiled candidate execution_seconds 非法")
    token_id = int(model_report["committed_tokens"][0]["output_token_id"])
    summary: dict[str, Any] = {
        "format": RUN_FORMAT,
        "status": "complete",
        "worker": str(worker),
        "output_token_id": token_id,
        "execution_seconds": execution_seconds,
        "tokens_per_second": 1.0 / execution_seconds,
        "gc_removed": runtime_profile["gc_status"]["gc_removed"],
        "vulkan_boundary": runtime_profile["vulkan_boundary"],
        "materialized_fp8_cache": runtime_profile["materialized_fp8_cache"],
        "reports": {
            "model": str((output_root / "model_report.json").resolve()),
            "runtime_profile": str((output_root / "runtime_profile.json").resolve()),
        },
        "claim_limit": (
            "这是单 token 端到端 candidate；局部 phase/kernel 时间不等于整 token TPS，"
            "单 token 的 FP8 cache 命中也不能外推 token2+。"
        ),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--fp8-cache-gib", type=float, default=0.0)
    args = parser.parse_args(argv)
    if not 0.0 <= args.fp8_cache_gib <= 12.0:
        parser.error("--fp8-cache-gib 必须在 0..12")
    result = run_profiled_candidate(
        worker=args.worker,
        output_root=args.output_root,
        asset_root=args.asset_root,
        catalog_path=args.catalog,
        timeout_seconds=args.timeout_seconds,
        fp8_cache_bytes=int(args.fp8_cache_gib * 1024**3),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
