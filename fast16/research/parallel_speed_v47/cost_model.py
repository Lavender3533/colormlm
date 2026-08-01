#!/usr/bin/env python3
"""ColorLM v47 三路线的纯 CPU、无第三方依赖成本计算器。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def _mib(value: float) -> float:
    return value / (1024.0**2)


def _gib(value: float) -> float:
    return value / (1024.0**3)


def calculate(manifest: dict[str, Any]) -> dict[str, Any]:
    base = manifest["base"]
    draft = manifest["draft_head"]
    recursion = manifest["latent_recursion"]
    paging = manifest["paging"]
    d = int(base["hidden_size"])
    vocab = int(base["vocab_size"])

    taps = len(draft["tap_layers"])
    fusion_rank = int(draft["fusion_rank"])
    position_rank = int(draft["position_rank"])
    draft_tokens = int(draft["draft_tokens"])
    # 每个tap有d->r；拼接后有(taps*r)->d。两项恰为2*taps*d*r。
    fusion_params = 2 * taps * d * fusion_rank
    # 共享A: d->r，每个未来位置各有B_j: r->d。
    position_params = (draft_tokens + 1) * d * position_rank
    position_embeddings = draft_tokens * d
    draft_params = fusion_params + position_params + position_embeddings
    draft_weight_bytes = draft_params * int(draft["weight_dtype_bytes"])
    adapter_flops = 2 * draft_params
    # 复用lm_head省参数而不省矩阵乘；K个位置均需投到全词表。
    vocabulary_projection_flops = 2 * draft_tokens * d * vocab
    draft_flops = adapter_flops + vocabulary_projection_flops
    base_step_flops = 2 * int(base["active_parameters_per_token"])
    accept = float(draft["uniform_acceptance_assumption"])
    expected_accepted_prefix = sum(accept**i for i in range(1, draft_tokens + 1))
    expected_advanced_tokens = 1.0 + expected_accepted_prefix
    verification_factor = float(draft["target_verification_pass_factor"])
    analytical_upper_speedup = expected_advanced_tokens / (
        verification_factor + draft_flops / base_step_flops
    )

    bottleneck = int(recursion["bottleneck_rank"])
    router_hidden = int(recursion["router_hidden"])
    loop_params = 2 * d * bottleneck
    router_params = d * router_hidden + router_hidden * 5 + 5
    recursion_params = loop_params + router_params
    distribution = [float(x) for x in recursion["k_distribution_assumption"]]
    distribution_sum = sum(distribution)
    if not math.isclose(distribution_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"K分布之和必须为1，实际为{distribution_sum}")
    mean_k = sum(k * p for k, p in enumerate(distribution))
    cumulative = 0.0
    p95_k = 4
    for k, probability in enumerate(distribution):
        cumulative += probability
        if cumulative >= 0.95:
            p95_k = k
            break
    mean_recursion_flops = 2 * loop_params * mean_k + 2 * router_params

    layer_count = len(paging["layers"])
    expert_bytes = int(paging["expert_page_bytes"])
    full_bank_bytes = layer_count * int(paging["experts_per_layer"]) * expert_bytes
    gpu_pool_bytes = layer_count * int(paging["gpu_slots_per_layer"]) * expert_bytes
    cpu_pool_bytes = layer_count * int(paging["cpu_warm_slots_per_layer"]) * expert_bytes
    active_bytes_per_token = layer_count * int(paging["active_experts_per_token"]) * expert_bytes
    ssd_bytes_per_second = float(paging["ssd_bandwidth_gib_per_second"]) * 1024**3
    pcie_bytes_per_second = float(paging["pcie_bandwidth_gib_per_second"]) * 1024**3
    no_cache_ssd_ms = 1000.0 * active_bytes_per_token / ssd_bytes_per_second
    no_cache_pcie_ms = 1000.0 * active_bytes_per_token / pcie_bytes_per_second
    overlap = float(paging["residual_overlap_fraction_assumption"])

    return {
        "format": "colorlm-parallel-speed-v47-cost-v1",
        "assumption_boundary": "解析成本模型；不是端到端测速，也不包含训练后接受率或真实内核效率",
        "draft_head": {
            "trainable_parameters": draft_params,
            "fusion_parameters": fusion_params,
            "position_parameters": position_params + position_embeddings,
            "f16_weight_mib": _mib(draft_weight_bytes),
            "adapter_flops_per_base_step": adapter_flops,
            "vocabulary_projection_flops_per_base_step": vocabulary_projection_flops,
            "total_draft_flops_per_base_step": draft_flops,
            "draft_to_base_step_flops_ratio": draft_flops / base_step_flops,
            "uniform_acceptance_assumption": accept,
            "expected_advanced_tokens_per_cycle": expected_advanced_tokens,
            "analytical_upper_bound_speedup": analytical_upper_speedup,
            "warning": "上界依赖批量验证因子与接受率；未训练头时不可当成预期实测速度"
        },
        "latent_recursion": {
            "trainable_parameters": recursion_params,
            "f16_weight_mib": _mib(recursion_params * 2),
            "mean_k_assumption": mean_k,
            "p95_k_assumption": p95_k,
            "mean_extra_flops_per_token": mean_recursion_flops,
            "extra_to_base_step_flops_ratio": mean_recursion_flops / base_step_flops,
            "k0_fraction_assumption": distribution[0],
            "warning": "小FLOPs不代表可提速；该块默认增加计算，只能用质量/路由收益或替代原层证明价值"
        },
        "paging": {
            "v17_four_layer_full_expert_bank_gib": _gib(full_bank_bytes),
            "gpu_pool_gib": _gib(gpu_pool_bytes),
            "cpu_warm_pool_gib": _gib(cpu_pool_bytes),
            "active_expert_mib_per_token": _mib(active_bytes_per_token),
            "no_cache_ideal_ssd_ms_per_token": no_cache_ssd_ms,
            "no_cache_ideal_pcie_ms_per_token": no_cache_pcie_ms,
            "residual_prefetch_exposed_ssd_ms_lower_bound": no_cache_ssd_ms * (1.0 - overlap),
            "warning": "理想值不含随机I/O、解压、Vulkan提交、CPU专家计算和同步尾延迟"
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="计算ColorLM v47并行速度方案的解析成本")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = calculate(manifest)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
