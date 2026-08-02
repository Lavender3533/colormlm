"""对四位置 FullDepth43 capture 执行 43 层 causal-block replay A/B。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .cached_layer_replay import PersistentCachedLayerReplay


REPORT_FORMAT = "polaris-causal-block-k4-layer-replay-ab-v1"
LAYER_COUNT = 43
BLOCK_SIZE = 4
SINGLE_OUTPUT_FILE = "vulkan_moe_branch.bf16le.bin"


def _read_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path} 含重复 JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"{path} 含非有限 JSON 常量: {value}")

    document = json.loads(
        path.read_text(encoding="utf-8", errors="strict"),
        object_pairs_hook=unique,
        parse_constant=invalid_constant,
    )
    if not isinstance(document, dict):
        raise ValueError(f"{path} 顶层必须是 JSON 对象")
    return document


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} 必须是有限非负数值")
    return result


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} 必须是非负整数")
    return value


def _baseline_rows(model_report: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    if model_report.get("status") != "complete":
        raise ValueError("model report 尚未 complete")
    tokens = model_report.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != BLOCK_SIZE:
        raise ValueError("model report 必须恰含四个 token")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for position, token in enumerate(tokens):
        if not isinstance(token, Mapping) or token.get("position") != position:
            raise ValueError("model report token position 漂移")
        layers = token.get("layers")
        if not isinstance(layers, list) or len(layers) != LAYER_COUNT:
            raise ValueError(f"position {position} 未覆盖 43 层")
        for layer, row in enumerate(layers):
            if not isinstance(row, Mapping) or row.get("layer") != layer:
                raise ValueError(f"position {position} layer {layer} 身份漂移")
            receipt = row.get("vulkan_writeback")
            if not isinstance(receipt, Mapping):
                raise ValueError(f"position {position} layer {layer} 缺少 Vulkan receipt")
            slot = receipt.get("reusable_gpu_slot")
            if not isinstance(slot, Mapping):
                raise ValueError(f"position {position} layer {layer} 缺少上传槽 receipt")
            result[(position, layer)] = {
                "worker_wall_ms": _finite_number(
                    receipt.get("worker_wall_ms"),
                    label=f"P{position}/L{layer} worker_wall_ms",
                ),
                "gpu_kernel_ms": _finite_number(
                    receipt.get("gpu_kernel_ms"),
                    label=f"P{position}/L{layer} gpu_kernel_ms",
                ),
                "uploaded_bytes": _integer(
                    slot.get("request_uploaded_bytes"),
                    label=f"P{position}/L{layer} uploaded_bytes",
                ),
                "output_sha256": receipt.get("output_sha256"),
            }
    return result


def run(
    *,
    worker: Path,
    capture_root: Path,
    model_report_path: Path,
    report_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    worker = worker.resolve(strict=True)
    capture_root = capture_root.resolve(strict=True)
    model_report_path = model_report_path.resolve(strict=True)
    report_path = report_path.resolve()
    if report_path.exists():
        raise ValueError("拒绝覆盖既有 causal-block A/B report")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须为正数")

    model_report = _read_json(model_report_path)
    baseline = _baseline_rows(model_report)
    command = (str(worker), "--fulldepth43-production-worker")
    layer_rows: list[dict[str, Any]] = []
    with PersistentCachedLayerReplay(command, timeout_seconds=timeout_seconds) as replay:
        for layer in range(LAYER_COUNT):
            manifests = tuple(
                capture_root
                / f"position-{position:06d}"
                / f"layer-{layer:02d}"
                / "bridge_manifest.json"
                for position in range(BLOCK_SIZE)
            )
            outputs, evidence = replay.execute(manifests)
            response = evidence["worker_response"]
            comparisons: list[dict[str, Any]] = []
            for position, block in enumerate(outputs):
                single_path = manifests[position].parent / SINGLE_OUTPUT_FILE
                single_bytes = single_path.read_bytes()
                if len(single_bytes) != 8192:
                    raise ValueError(f"P{position}/L{layer} 单层 BF16 字节数漂移")
                single = (
                    torch.frombuffer(bytearray(single_bytes), dtype=torch.bfloat16)
                    .clone()
                    .reshape(1, 1, 4096)
                )
                exact = bool(torch.equal(block, single))
                baseline_sha = baseline[(position, layer)]["output_sha256"]
                observed_sha = _sha256_bytes(single_bytes)
                if baseline_sha != observed_sha:
                    raise ValueError(f"P{position}/L{layer} 基线文件 SHA 与 receipt 漂移")
                comparisons.append(
                    {
                        "position": position,
                        "exact_bf16_equal": exact,
                        "single_sha256": observed_sha,
                        "block_sha256": evidence["output_sha256s"][position],
                    }
                )

            gpu_cache = response.get("gpu_payload_cache")
            payload_cache = response.get("payload_cache")
            if not isinstance(gpu_cache, Mapping) or not isinstance(payload_cache, Mapping):
                raise ValueError(f"L{layer} causal-block cache telemetry 缺失")
            baseline_rows = [baseline[(position, layer)] for position in range(BLOCK_SIZE)]
            row = {
                "layer": layer,
                "all_exact_bf16_equal": all(
                    item["exact_bf16_equal"] for item in comparisons
                ),
                "comparisons": comparisons,
                "baseline_four_single_worker_wall_ms": sum(
                    item["worker_wall_ms"] for item in baseline_rows
                ),
                "block_worker_wall_ms": _finite_number(
                    response.get("wall_ms"), label=f"L{layer} block wall_ms"
                ),
                "baseline_four_single_gpu_kernel_ms": sum(
                    item["gpu_kernel_ms"] for item in baseline_rows
                ),
                "block_gpu_kernel_ms": _finite_number(
                    response.get("gpu_kernel_ms"), label=f"L{layer} block gpu_kernel_ms"
                ),
                "baseline_four_single_uploaded_bytes": sum(
                    item["uploaded_bytes"] for item in baseline_rows
                ),
                "block_uploaded_bytes": _integer(
                    gpu_cache.get("request_uploaded_bytes"),
                    label=f"L{layer} block uploaded_bytes",
                ),
                "total_routed_references": _integer(
                    response.get("total_routed_references"),
                    label=f"L{layer} total_routed_references",
                ),
                "unique_routed_experts": _integer(
                    response.get("unique_routed_experts"),
                    label=f"L{layer} unique_routed_experts",
                ),
                "reused_routed_references": _integer(
                    response.get("reused_routed_references"),
                    label=f"L{layer} reused_routed_references",
                ),
                "shared_payload_uploads": _integer(
                    response.get("shared_payload_uploads"),
                    label=f"L{layer} shared_payload_uploads",
                ),
                "gpu_cache_request_hits": _integer(
                    gpu_cache.get("request_hits"),
                    label=f"L{layer} gpu cache hits",
                ),
                "gpu_cache_request_misses": _integer(
                    gpu_cache.get("request_misses"),
                    label=f"L{layer} gpu cache misses",
                ),
                "host_cache_disk_bytes_read": _integer(
                    payload_cache.get("request_disk_bytes_read"),
                    label=f"L{layer} disk bytes",
                ),
                "speed_eligible_verifier": response.get("speed_eligible_verifier"),
            }
            if row["speed_eligible_verifier"] is not False:
                raise ValueError(f"L{layer} 错误声明 speed eligible")
            layer_rows.append(row)
            print(
                json.dumps(
                    {
                        "completed_layer": layer,
                        "exact": row["all_exact_bf16_equal"],
                        "single_ms": row["baseline_four_single_worker_wall_ms"],
                        "block_ms": row["block_worker_wall_ms"],
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                flush=True,
            )

    baseline_wall_ms = sum(row["baseline_four_single_worker_wall_ms"] for row in layer_rows)
    block_wall_ms = sum(row["block_worker_wall_ms"] for row in layer_rows)
    baseline_kernel_ms = sum(
        row["baseline_four_single_gpu_kernel_ms"] for row in layer_rows
    )
    block_kernel_ms = sum(row["block_gpu_kernel_ms"] for row in layer_rows)
    baseline_uploaded_bytes = sum(
        row["baseline_four_single_uploaded_bytes"] for row in layer_rows
    )
    block_uploaded_bytes = sum(row["block_uploaded_bytes"] for row in layer_rows)
    report = {
        "format": REPORT_FORMAT,
        "status": "complete",
        "worker": str(worker),
        "capture_root": str(capture_root),
        "source_model_report": str(model_report_path),
        "block_size": BLOCK_SIZE,
        "layer_count": LAYER_COUNT,
        "all_exact_bf16_equal": all(row["all_exact_bf16_equal"] for row in layer_rows),
        "baseline_four_single_worker_wall_ms": baseline_wall_ms,
        "block_worker_wall_ms": block_wall_ms,
        "worker_wall_speedup": baseline_wall_ms / block_wall_ms,
        "baseline_four_single_gpu_kernel_ms": baseline_kernel_ms,
        "block_gpu_kernel_ms": block_kernel_ms,
        "baseline_four_single_uploaded_bytes": baseline_uploaded_bytes,
        "block_uploaded_bytes": block_uploaded_bytes,
        "uploaded_bytes_reduction_ratio": 1.0 - block_uploaded_bytes / baseline_uploaded_bytes,
        "total_routed_references": sum(
            row["total_routed_references"] for row in layer_rows
        ),
        "unique_routed_experts_sum": sum(
            row["unique_routed_experts"] for row in layer_rows
        ),
        "reused_routed_references": sum(
            row["reused_routed_references"] for row in layer_rows
        ),
        "shared_payload_uploads": sum(
            row["shared_payload_uploads"] for row in layer_rows
        ),
        "layers": layer_rows,
        "speed_eligible_verifier": False,
        "claim_limit": (
            "这是43层离线同层K=4 MoE replay A/B；不含attention/router/KV/HC/final head，"
            "不能冒充端到端token/s。"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args(argv)
    report = run(
        worker=args.worker,
        capture_root=args.capture_root,
        model_report_path=args.model_report,
        report_path=args.report,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "all_exact_bf16_equal",
                    "worker_wall_speedup",
                    "uploaded_bytes_reduction_ratio",
                    "reused_routed_references",
                    "speed_eligible_verifier",
                    "claim_limit",
                )
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
