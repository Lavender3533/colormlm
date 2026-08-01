"""运行 1..16 个连续 FullDepth43 Vulkan candidate token 并细分剖析。"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate_profile import CandidateProfiler, MaterializedFp8Cache, write_json
from .executor import ExecutionConfig, FullDepthError, execute
from .preflight import DEFAULT_ASSET_ROOT, DEFAULT_CATALOG


RUN_FORMAT = "polaris-fulldepth43-profiled-candidate-run-v1"


def _per_token_wall_seconds(
    profiler: CandidateProfiler,
    *,
    expected_count: int,
) -> list[float]:
    """读取 profiler 已保存的每次 ``token_total`` 原始观测。"""

    observations = getattr(profiler, "_observations", None)
    if not isinstance(observations, list):
        raise FullDepthError("candidate profiler 缺少原始 token_total 观测")
    values = [
        float(row.inclusive_seconds)
        for row in observations
        if getattr(row, "label", None) == "token_total"
    ]
    if len(values) != expected_count:
        raise FullDepthError(
            "candidate profiler token_total 次数漂移: "
            f"expected={expected_count} actual={len(values)}"
        )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise FullDepthError("candidate profiler token_total wall timing 非法")
    return values


def run_profiled_candidate(
    *,
    worker: Path,
    output_root: Path,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    catalog_path: Path = DEFAULT_CATALOG,
    timeout_seconds: float = 30.0,
    fp8_cache_bytes: int = 0,
    download_budget_bytes: int = 0,
    token_count: int = 1,
    vulkan_attention_worker: Path | None = None,
    vulkan_attention_scratch: Path | None = None,
    vulkan_attention_shared_batch: bool = False,
    vulkan_final_head_worker: Path | None = None,
    vulkan_final_head_scratch: Path | None = None,
) -> dict[str, Any]:
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or not 1 <= token_count <= 16
    ):
        raise ValueError("token_count 必须是 1..16 的整数")
    worker = worker.resolve(strict=True)
    if vulkan_attention_worker is not None:
        vulkan_attention_worker = vulkan_attention_worker.resolve(strict=True)
    if vulkan_attention_scratch is not None:
        vulkan_attention_scratch = vulkan_attention_scratch.resolve()
    if vulkan_final_head_worker is not None:
        vulkan_final_head_worker = vulkan_final_head_worker.resolve(strict=True)
    if vulkan_final_head_scratch is not None:
        vulkan_final_head_scratch = vulkan_final_head_scratch.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    if isinstance(fp8_cache_bytes, bool) or not isinstance(fp8_cache_bytes, int) or fp8_cache_bytes < 0:
        raise ValueError("fp8_cache_bytes 必须为非负整数")
    if (
        isinstance(download_budget_bytes, bool)
        or not isinstance(download_budget_bytes, int)
        or download_budget_bytes < 0
    ):
        raise ValueError("download_budget_bytes 必须为非负整数")

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
                    token_count=token_count,
                    allow_fetch=download_budget_bytes > 0,
                    download_budget_bytes=download_budget_bytes,
                    vulkan_bridge_capture=output_root / "captures",
                    vulkan_writeback_worker=worker,
                    vulkan_writeback_timeout_seconds=timeout_seconds,
                    vulkan_writeback_all_layers=True,
                    vulkan_writeback_verify_cpu=False,
                    vulkan_writeback_cpu_fallback=False,
                    vulkan_writeback_fast_production=True,
                    vulkan_attention_worker=vulkan_attention_worker,
                    vulkan_attention_scratch=vulkan_attention_scratch,
                    vulkan_attention_shared_batch=vulkan_attention_shared_batch,
                    vulkan_final_head_worker=vulkan_final_head_worker,
                    vulkan_final_head_scratch=vulkan_final_head_scratch,
                )
            )
    runtime_profile = profiler.snapshot()
    if model_report.get("status") != "complete":
        error = model_report.get("error")
        raise FullDepthError(f"profiled candidate 未完成: {error}")
    per_token_wall_seconds = _per_token_wall_seconds(
        profiler,
        expected_count=token_count,
    )
    runtime_profile["per_token_wall_seconds"] = per_token_wall_seconds
    runtime_profile["materialized_fp8_cache"] = None if cache is None else cache.stats()
    write_json(output_root / "runtime_profile.json", runtime_profile)

    tokens = model_report.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != token_count:
        raise FullDepthError("profiled candidate token 数量漂移")
    for position, token in enumerate(tokens):
        if not isinstance(token, Mapping) or token.get("position") != position:
            raise FullDepthError("profiled candidate token position 不连续")
        layers = token.get("layers")
        if not isinstance(layers, list) or len(layers) != 43:
            raise FullDepthError(f"profiled candidate position {position} 没有覆盖 43 层")
    if model_report.get("vulkan_writeback_fallbacks"):
        raise FullDepthError("profiled candidate 发生 CPU fallback")
    execution_seconds = float(model_report["execution_seconds"])
    if execution_seconds <= 0:
        raise FullDepthError("profiled candidate execution_seconds 非法")
    committed = model_report.get("committed_tokens")
    if not isinstance(committed, list) or len(committed) != token_count:
        raise FullDepthError("profiled candidate committed token 数量漂移")
    token_rows: list[dict[str, Any]] = []
    for position, (token, wall_seconds) in enumerate(
        zip(committed, per_token_wall_seconds, strict=True)
    ):
        if not isinstance(token, Mapping) or token.get("position") != position:
            raise FullDepthError("profiled candidate committed position 不连续")
        token_rows.append(
            {
                "position": position,
                "input_token_id": int(token["input_token_id"]),
                "output_token_id": int(token["output_token_id"]),
                "wall_seconds": wall_seconds,
            }
        )
    token_id = token_rows[-1]["output_token_id"]
    effective_tokens_per_second = token_count / execution_seconds
    summary: dict[str, Any] = {
        "format": RUN_FORMAT,
        "status": "complete",
        "worker": str(worker),
        "output_token_id": token_id,
        "committed_token_count": token_count,
        "tokens": token_rows,
        "per_token_wall_seconds": per_token_wall_seconds,
        "execution_seconds": execution_seconds,
        "tokens_per_second": effective_tokens_per_second,
        "effective_tokens_per_second": effective_tokens_per_second,
        "gc_removed": runtime_profile["gc_status"]["gc_removed"],
        "vulkan_boundary": runtime_profile["vulkan_boundary"],
        "materialized_fp8_cache": runtime_profile["materialized_fp8_cache"],
        "range_proof_cache": model_report.get("range_proof_cache"),
        "reports": {
            "model": str((output_root / "model_report.json").resolve()),
            "runtime_profile": str((output_root / "runtime_profile.json").resolve()),
        },
        "claim_limit": (
            "这是连续端到端 candidate；effective_tokens_per_second 仅来自本次完整 execution，"
            "局部 phase/kernel 时间不能冒充整 token TPS。"
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
    parser.add_argument("--token-count", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--vulkan-attention-worker", type=Path)
    parser.add_argument("--vulkan-attention-scratch", type=Path)
    parser.add_argument("--vulkan-attention-shared-batch", action="store_true")
    parser.add_argument("--vulkan-final-head-worker", type=Path)
    parser.add_argument("--vulkan-final-head-scratch", type=Path)
    parser.add_argument("--fp8-cache-gib", type=float, default=0.0)
    parser.add_argument("--download-budget-gib", type=float, default=0.0)
    args = parser.parse_args(argv)
    if not 0.0 <= args.fp8_cache_gib <= 12.0:
        parser.error("--fp8-cache-gib 必须在 0..12")
    if not 0.0 <= args.download_budget_gib <= 8.0:
        parser.error("--download-budget-gib 必须在 0..8")
    result = run_profiled_candidate(
        worker=args.worker,
        output_root=args.output_root,
        asset_root=args.asset_root,
        catalog_path=args.catalog,
        timeout_seconds=args.timeout_seconds,
        fp8_cache_bytes=int(args.fp8_cache_gib * 1024**3),
        download_budget_bytes=int(args.download_budget_gib * 1024**3),
        token_count=args.token_count,
        vulkan_attention_worker=args.vulkan_attention_worker,
        vulkan_attention_scratch=args.vulkan_attention_scratch,
        vulkan_attention_shared_batch=args.vulkan_attention_shared_batch,
        vulkan_final_head_worker=args.vulkan_final_head_worker,
        vulkan_final_head_scratch=args.vulkan_final_head_scratch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
