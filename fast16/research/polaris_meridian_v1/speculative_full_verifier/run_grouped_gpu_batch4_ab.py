"""对 K=4 grouped GPU 路径执行 43 层 warm→measure 相邻短门。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cached_layer_replay import PersistentCachedLayerReplay


REPORT_FORMAT = "polaris-causal-block-k4-grouped-gpu-adjacent-ab-v1"
LAYERS = 43
POSITIONS = 4
SINGLE_OUTPUT = "vulkan_moe_branch.bf16le.bin"


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} 顶层必须是 JSON 对象")
    return document


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是数值")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} 必须是有限非负数值")
    return result


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} 必须是非负整数")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifests(root: Path, layer: int) -> tuple[Path, ...]:
    result = tuple(
        root
        / f"position-{position:06d}"
        / f"layer-{layer:02d}"
        / "bridge_manifest.json"
        for position in range(POSITIONS)
    )
    for path in result:
        path.resolve(strict=True)
    return result


def _single_sha256s(manifests: Sequence[Path]) -> list[str]:
    result: list[str] = []
    for manifest in manifests:
        output = manifest.parent / SINGLE_OUTPUT
        if output.stat().st_size != 8192:
            raise ValueError(f"{output} BF16 字节数漂移")
        result.append(_sha256(output))
    return result


def _response_metrics(response: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    payload_cache = response.get("payload_cache")
    arena = response.get("gpu_union_arena")
    if not isinstance(payload_cache, Mapping) or not isinstance(arena, Mapping):
        raise ValueError(f"{label} 缺少 cache/arena telemetry")
    if response.get("speed_eligible_verifier") is not False:
        raise ValueError(f"{label} 错误声明 speed eligible")
    return {
        "wall_ms": _number(response.get("wall_ms"), label=f"{label} wall_ms"),
        "gpu_kernel_ms": _number(
            response.get("gpu_kernel_ms"), label=f"{label} gpu_kernel_ms"
        ),
        "host_disk_bytes": _integer(
            payload_cache.get("request_disk_bytes_read"),
            label=f"{label} host disk bytes",
        ),
        "staging_allocations": _integer(
            arena.get("staging_allocations"),
            label=f"{label} staging allocations",
        ),
        "device_allocations": _integer(
            arena.get("device_allocations"),
            label=f"{label} device allocations",
        ),
        "actual_uploaded_bytes": _integer(
            arena.get("actual_uploaded_bytes"),
            label=f"{label} uploaded bytes",
        ),
        "host_pack_ms": _number(
            arena.get("host_pack_ms"), label=f"{label} host_pack_ms"
        ),
        "arena_upload_ms": _number(
            arena.get("arena_upload_ms"), label=f"{label} arena_upload_ms"
        ),
        "unique_routed_identities": _integer(
            arena.get("unique_routed_identities"),
            label=f"{label} unique routed identities",
        ),
        "reused_routed_references": _integer(
            arena.get("reused_routed_references"),
            label=f"{label} reused routed references",
        ),
    }


def run(
    *,
    worker: Path,
    run_root: Path,
    baseline_report_path: Path,
    report_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    worker = worker.resolve(strict=True)
    run_root = run_root.resolve(strict=True)
    warm_root = (run_root / "warm").resolve(strict=True)
    measure_root = (run_root / "measure").resolve(strict=True)
    baseline_report_path = baseline_report_path.resolve(strict=True)
    report_path = report_path.resolve()
    if report_path.exists():
        raise ValueError("拒绝覆盖已有 grouped GPU 报告")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须为正数")

    baseline = _read_json(baseline_report_path)
    old_layers = baseline.get("layers")
    if (
        baseline.get("status") != "complete"
        or baseline.get("all_exact_bf16_equal") is not True
        or not isinstance(old_layers, list)
        or len(old_layers) != LAYERS
    ):
        raise ValueError("旧 persistent-arena 基线不完整")

    rows: list[dict[str, Any]] = []
    command = (str(worker), "--fulldepth43-production-worker")
    with PersistentCachedLayerReplay(command, timeout_seconds=timeout_seconds) as replay:
        hello = replay.hello
        if (
            not isinstance(hello, Mapping)
            or hello.get("causal_block_grouped_gpu_batch4") is not True
            or hello.get("causal_block_grouped_gpu_dispatches") != 9
        ):
            raise ValueError("worker 未声明 K=4 grouped GPU 9-dispatch 合同")

        for layer in range(LAYERS):
            warm_manifests = _manifests(warm_root, layer)
            measure_manifests = _manifests(measure_root, layer)
            warm_outputs, warm_evidence = replay.execute(warm_manifests)
            measure_outputs, measure_evidence = replay.execute(measure_manifests)
            warm_sha = list(warm_evidence["output_sha256s"])
            measure_sha = list(measure_evidence["output_sha256s"])
            warm_single_sha = _single_sha256s(warm_manifests)
            measure_single_sha = _single_sha256s(measure_manifests)
            exact_rows = [
                warm_sha[position]
                == measure_sha[position]
                == warm_single_sha[position]
                == measure_single_sha[position]
                for position in range(POSITIONS)
            ]
            if not all(exact_rows):
                raise ValueError(f"L{layer} grouped GPU BF16 首次不精确")
            if len(warm_outputs) != POSITIONS or len(measure_outputs) != POSITIONS:
                raise ValueError(f"L{layer} 输出行数漂移")

            warm_response = warm_evidence["worker_response"]
            measure_response = measure_evidence["worker_response"]
            if not isinstance(warm_response, Mapping) or not isinstance(
                measure_response, Mapping
            ):
                raise ValueError(f"L{layer} worker response 非对象")
            warm = _response_metrics(warm_response, label=f"L{layer} warm")
            measure = _response_metrics(measure_response, label=f"L{layer} measure")
            old = old_layers[layer]
            if not isinstance(old, Mapping) or old.get("layer") != layer:
                raise ValueError(f"旧基线 L{layer} 身份漂移")
            old_wall_ms = _number(
                old.get("block_wall_ms"), label=f"旧基线 L{layer} wall"
            )
            old_kernel_ms = _number(
                old.get("block_gpu_kernel_ms"), label=f"旧基线 L{layer} kernel"
            )
            row = {
                "layer": layer,
                "all_exact_bf16_equal": True,
                "exact_rows": exact_rows,
                "output_sha256": measure_sha,
                "warm": warm,
                "measure": measure,
                "old_block_wall_ms": old_wall_ms,
                "old_block_gpu_kernel_ms": old_kernel_ms,
                "wall_speedup": old_wall_ms / measure["wall_ms"],
                "gpu_kernel_speedup": old_kernel_ms / measure["gpu_kernel_ms"],
                "speed_eligible_verifier": False,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "completed_layer": layer,
                        "exact_rows": sum(exact_rows),
                        "measure_wall_ms": measure["wall_ms"],
                        "measure_kernel_ms": measure["gpu_kernel_ms"],
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                flush=True,
            )

    old_wall_ms = sum(row["old_block_wall_ms"] for row in rows)
    old_kernel_ms = sum(row["old_block_gpu_kernel_ms"] for row in rows)
    measure_wall_ms = sum(row["measure"]["wall_ms"] for row in rows)
    measure_kernel_ms = sum(row["measure"]["gpu_kernel_ms"] for row in rows)
    wall_speedups = [row["wall_speedup"] for row in rows]
    kernel_speedups = [row["gpu_kernel_speedup"] for row in rows]
    report = {
        "format": REPORT_FORMAT,
        "status": "complete",
        "worker": str(worker),
        "device": baseline.get("device"),
        "run_root": str(run_root),
        "source_baseline_report": str(baseline_report_path),
        "block_size": POSITIONS,
        "layer_count": LAYERS,
        "comparison_rows": LAYERS * POSITIONS,
        "exact_bf16_rows": sum(sum(row["exact_rows"]) for row in rows),
        "all_exact_bf16_equal": all(row["all_exact_bf16_equal"] for row in rows),
        "grouped_gpu_compute_submissions_per_layer": 1,
        "grouped_gpu_dispatches_per_layer": 9,
        "old_block_wall_ms": old_wall_ms,
        "measure_wall_ms": measure_wall_ms,
        "worker_wall_speedup": old_wall_ms / measure_wall_ms,
        "minimum_layer_wall_speedup": min(wall_speedups),
        "median_layer_wall_speedup": statistics.median(wall_speedups),
        "old_block_gpu_kernel_ms": old_kernel_ms,
        "measure_gpu_kernel_ms": measure_kernel_ms,
        "gpu_kernel_speedup": old_kernel_ms / measure_kernel_ms,
        "minimum_layer_gpu_kernel_speedup": min(kernel_speedups),
        "median_layer_gpu_kernel_speedup": statistics.median(kernel_speedups),
        "measure_staging_allocations": sum(
            row["measure"]["staging_allocations"] for row in rows
        ),
        "measure_device_allocations": sum(
            row["measure"]["device_allocations"] for row in rows
        ),
        "measure_host_disk_bytes": sum(
            row["measure"]["host_disk_bytes"] for row in rows
        ),
        "layers": rows,
        "speed_eligible_verifier": False,
        "claim_limit": (
            "43-layer K=4 same-layer MoE replay only; not an end-to-end token/s or "
            "quality result."
        ),
    }
    if (
        report["all_exact_bf16_equal"] is not True
        or report["measure_staging_allocations"] != 0
        or report["measure_device_allocations"] != 0
        or report["measure_host_disk_bytes"] != 0
    ):
        raise ValueError("43 层 grouped GPU 晋级门未通过")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(
        worker=args.worker,
        run_root=args.run_root,
        baseline_report_path=args.baseline_report,
        report_path=args.report,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
