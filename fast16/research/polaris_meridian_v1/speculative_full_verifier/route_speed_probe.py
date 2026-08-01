"""从真实 FullDepth 路由轨迹计算专家复用与质量质量覆盖的速度边界。

该模块只分析已经产生的原生 top-6 路由；它不把静态投影冒充实测
token/s，也不允许部分运行报告被标记为完整证据。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


TOP_K = 6
DEFAULT_THRESHOLDS = (0.8, 0.9, 0.95, 0.99)


class RouteSpeedProbeError(ValueError):
    """路由报告不满足分析合同。"""


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RouteSpeedProbeError(f"{label} 必须是有限数字")
    result = float(value)
    if not math.isfinite(result):
        raise RouteSpeedProbeError(f"{label} 必须是有限数字")
    return result


def _validate_thresholds(values: Iterable[float]) -> tuple[float, ...]:
    thresholds = tuple(_finite_number(value, label="threshold") for value in values)
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise RouteSpeedProbeError("threshold 必须位于 (0, 1]")
    if len(set(thresholds)) != len(thresholds):
        raise RouteSpeedProbeError("threshold 不得重复")
    return thresholds


def _route_row(token: Mapping[str, Any], layer: Mapping[str, Any]) -> dict[str, Any]:
    position = token.get("position")
    layer_id = layer.get("layer")
    expert_ids = layer.get("expert_ids")
    weights = layer.get("route_weights")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise RouteSpeedProbeError("token position 非法")
    if isinstance(layer_id, bool) or not isinstance(layer_id, int) or layer_id < 0:
        raise RouteSpeedProbeError("layer ID 非法")
    if not isinstance(expert_ids, list) or len(expert_ids) != TOP_K:
        raise RouteSpeedProbeError(f"P{position}/L{layer_id} 不是原生 top-{TOP_K}")
    if len(set(expert_ids)) != TOP_K or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in expert_ids
    ):
        raise RouteSpeedProbeError(f"P{position}/L{layer_id} expert IDs 非法")
    if not isinstance(weights, list) or len(weights) != TOP_K:
        raise RouteSpeedProbeError(f"P{position}/L{layer_id} route weights 不完整")
    numeric_weights = tuple(
        _finite_number(value, label=f"P{position}/L{layer_id} route weight")
        for value in weights
    )
    if any(value < 0.0 for value in numeric_weights) or sum(numeric_weights) <= 0.0:
        raise RouteSpeedProbeError(f"P{position}/L{layer_id} route weights 非法")
    return {
        "position": position,
        "layer": layer_id,
        "expert_ids": tuple(expert_ids),
        "weights": numeric_weights,
    }


def _required_k(weights: Sequence[float], threshold: float) -> int:
    total = sum(weights)
    cumulative = 0.0
    for index, value in enumerate(sorted(weights, reverse=True), start=1):
        cumulative += value
        if cumulative / total >= threshold:
            return index
    return len(weights)


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.floor((len(ordered) - 1) * fraction)]


def analyze_report(
    report: Mapping[str, Any],
    *,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """分析完整或运行中的 FullDepth JSON 报告。

    ``evidence_status`` 会保留源报告的完整性，调用者不能把 running
    报告的部分路由升级成完成证据。
    """

    checked_thresholds = _validate_thresholds(thresholds)
    tokens = report.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise RouteSpeedProbeError("报告没有 token 路由")

    rows: list[dict[str, Any]] = []
    for token in tokens:
        if not isinstance(token, Mapping):
            raise RouteSpeedProbeError("token row 必须是对象")
        layers = token.get("layers")
        if not isinstance(layers, list):
            raise RouteSpeedProbeError("token layers 必须是列表")
        rows.extend(_route_row(token, layer) for layer in layers)
    if not rows:
        raise RouteSpeedProbeError("报告尚无已完成路由")

    by_key = {(row["position"], row["layer"]): row for row in rows}
    adjacent: list[dict[str, int]] = []
    positions = sorted({row["position"] for row in rows})
    layers = sorted({row["layer"] for row in rows})
    for left, right in zip(positions, positions[1:]):
        if right != left + 1:
            continue
        for layer_id in layers:
            first = by_key.get((left, layer_id))
            second = by_key.get((right, layer_id))
            if first is None or second is None:
                continue
            first_ids = set(first["expert_ids"])
            second_ids = set(second["expert_ids"])
            adjacent.append(
                {
                    "left_position": left,
                    "right_position": right,
                    "layer": layer_id,
                    "overlap": len(first_ids & second_ids),
                    "union": len(first_ids | second_ids),
                }
            )

    top1_mass = [max(row["weights"]) / sum(row["weights"]) for row in rows]
    coverage: list[dict[str, Any]] = []
    dynamic_top6_bytes = (
        report.get("preflight", {})
        .get("dynamic_routed_experts", {})
        .get("native_top6_cold_bytes_per_token")
    )
    if dynamic_top6_bytes is not None:
        if (
            isinstance(dynamic_top6_bytes, bool)
            or not isinstance(dynamic_top6_bytes, int)
            or dynamic_top6_bytes <= 0
        ):
            raise RouteSpeedProbeError("native_top6_cold_bytes_per_token 非法")

    for threshold in checked_thresholds:
        counts = [_required_k(row["weights"], threshold) for row in rows]
        mean_k = fmean(counts)
        item: dict[str, Any] = {
            "threshold": threshold,
            "mean_selected_experts": mean_k,
            "minimum_selected_experts": min(counts),
            "maximum_selected_experts": max(counts),
            "expert_io_fraction_vs_top6": mean_k / TOP_K,
        }
        if dynamic_top6_bytes is not None:
            item["projected_dynamic_bytes_per_token"] = round(
                dynamic_top6_bytes * mean_k / TOP_K
            )
        coverage.append(item)

    source_status = report.get("status")
    complete = source_status == "complete"
    result: dict[str, Any] = {
        "format": "polaris-route-speed-probe-v1",
        "evidence_status": "complete_trace" if complete else "partial_live_trace",
        "source_status": source_status,
        "sampled_positions": positions,
        "route_rows": len(rows),
        "top1_mass": {
            "mean": fmean(top1_mass),
            "p10": _quantile(top1_mass, 0.10),
            "minimum": min(top1_mass),
            "layers_at_or_above_90pct": sum(value >= 0.90 for value in top1_mass),
        },
        "mass_adaptive_candidates": coverage,
        "adjacent_route_reuse": {
            "comparisons": len(adjacent),
            "mean_overlap": fmean(item["overlap"] for item in adjacent) if adjacent else None,
            "mean_union": fmean(item["union"] for item in adjacent) if adjacent else None,
            "expert_io_fraction_vs_serial": (
                fmean(item["union"] for item in adjacent) / (2 * TOP_K)
                if adjacent
                else None
            ),
        },
        "claim_limit": (
            "路由质量质量与专家字节投影；不证明截断路由的 NLL/生成质量，"
            "也不证明任何 token/s。"
        ),
    }
    if not complete:
        result["incomplete_reason"] = "源 FullDepth 报告仍在运行，只能用于方向筛查"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        dest="thresholds",
        help="可重复；默认 0.8/0.9/0.95/0.99",
    )
    args = parser.parse_args(argv)
    document = json.loads(args.report.read_text(encoding="utf-8"))
    result = analyze_report(
        document,
        thresholds=DEFAULT_THRESHOLDS if args.thresholds is None else args.thresholds,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8", newline="\n")
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
