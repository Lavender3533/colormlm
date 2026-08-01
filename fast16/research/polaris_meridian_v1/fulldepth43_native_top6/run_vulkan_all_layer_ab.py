"""运行 FullDepth43 全层 Vulkan 精确验证与同进程相邻速度 A/B。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import write_json
from .executor import ExecutionConfig, FullDepthError, execute
from .preflight import DEFAULT_ASSET_ROOT, DEFAULT_CATALOG


FORMAT = "polaris-fulldepth43-vulkan-all-layer-ab-v1"


def _require_complete(report: Mapping[str, Any], *, label: str) -> None:
    if report.get("status") != "complete":
        error = report.get("error")
        raise FullDepthError(f"{label} 未完成: {error}")
    committed = report.get("committed_tokens")
    if not isinstance(committed, list) or len(committed) != 1:
        raise FullDepthError(f"{label} 必须精确提交一个 token")


def _output_token(report: Mapping[str, Any]) -> int:
    return int(report["committed_tokens"][0]["output_token_id"])


def _verified_layer_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tokens = report.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != 1:
        raise FullDepthError("全层验证报告 token 数量漂移")
    layers = tokens[0].get("layers")
    if not isinstance(layers, list) or len(layers) != 43:
        raise FullDepthError("全层验证报告必须包含 43 层")
    return layers


def run_all_layer_ab(
    *,
    worker: Path,
    output_root: Path,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    catalog_path: Path = DEFAULT_CATALOG,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    worker = worker.resolve(strict=True)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    verified = execute(
        ExecutionConfig(
            asset_root=asset_root,
            catalog_path=catalog_path,
            report_path=output_root / "01_verified_all_layers.json",
            token_count=1,
            vulkan_bridge_capture=output_root / "01_verified_captures",
            vulkan_writeback_worker=worker,
            vulkan_writeback_timeout_seconds=timeout_seconds,
            vulkan_writeback_all_layers=True,
            vulkan_writeback_verify_cpu=True,
            vulkan_writeback_cpu_fallback=False,
        )
    )
    _require_complete(verified, label="逐层 CPU/GPU 验证")
    if verified.get("vulkan_writeback_layers") != list(range(43)):
        raise FullDepthError("逐层验证没有覆盖完整 0..42")
    for row in _verified_layer_rows(verified):
        evidence = row.get("vulkan_writeback")
        comparison = None if not isinstance(evidence, Mapping) else evidence.get("comparison")
        if not isinstance(comparison, Mapping) or comparison.get("exact_bf16_equal") is not True:
            raise FullDepthError(f"L{row.get('layer')} 未通过 BF16 逐位验证")

    baseline = execute(
        ExecutionConfig(
            asset_root=asset_root,
            catalog_path=catalog_path,
            report_path=output_root / "02_cpu_baseline.json",
            token_count=1,
        )
    )
    _require_complete(baseline, label="CPU baseline")

    candidate = execute(
        ExecutionConfig(
            asset_root=asset_root,
            catalog_path=catalog_path,
            report_path=output_root / "03_vulkan_candidate.json",
            token_count=1,
            vulkan_bridge_capture=output_root / "03_candidate_captures",
            vulkan_writeback_worker=worker,
            vulkan_writeback_timeout_seconds=timeout_seconds,
            vulkan_writeback_all_layers=True,
            vulkan_writeback_verify_cpu=False,
            vulkan_writeback_cpu_fallback=False,
        )
    )
    _require_complete(candidate, label="Vulkan candidate")
    if candidate.get("vulkan_writeback_layers") != list(range(43)):
        raise FullDepthError("速度候选没有覆盖完整 0..42")
    if candidate.get("vulkan_writeback_fallbacks"):
        raise FullDepthError("速度候选发生 CPU fallback")

    token_ids = {
        "verified": _output_token(verified),
        "baseline": _output_token(baseline),
        "candidate": _output_token(candidate),
    }
    if len(set(token_ids.values())) != 1:
        raise FullDepthError(f"全层 Vulkan 输出 token 与 CPU 漂移: {token_ids}")
    baseline_seconds = float(baseline["execution_seconds"])
    candidate_seconds = float(candidate["execution_seconds"])
    if baseline_seconds <= 0 or candidate_seconds <= 0:
        raise FullDepthError("A/B execution_seconds 非法")
    speedup = baseline_seconds / candidate_seconds
    result = {
        "format": FORMAT,
        "status": "complete",
        "worker": str(worker),
        "token_ids": token_ids,
        "exact_bf16_layers": 43,
        "cpu_fallbacks": 0,
        "same_process_adjacent_ab": True,
        "timing": {
            "verified_seconds": float(verified["execution_seconds"]),
            "baseline_seconds": baseline_seconds,
            "candidate_seconds": candidate_seconds,
            "speedup_vs_cpu": speedup,
            "candidate_tokens_per_second": 1.0 / candidate_seconds,
            "total_wall_seconds": time.perf_counter() - started,
        },
        "reports": {
            "verified": str((output_root / "01_verified_all_layers.json").resolve()),
            "baseline": str((output_root / "02_cpu_baseline.json").resolve()),
            "candidate": str((output_root / "03_vulkan_candidate.json").resolve()),
        },
        "passed_speed_smoke": speedup > 1.0,
        "claim_limit": (
            "证明同一 BOS 输入的 43 个 MoE 分支逐 BF16 位对齐及相邻单-token "
            "速度；attention/HC/router/head 仍在 CPU，不证明 20/50 token/s、质量或长上下文。"
        ),
    }
    write_json(output_root / "all_layer_ab_report.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    result = run_all_layer_ab(
        worker=args.worker,
        output_root=args.output_root,
        asset_root=args.asset_root,
        catalog_path=args.catalog,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
