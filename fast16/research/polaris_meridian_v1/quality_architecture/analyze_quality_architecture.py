#!/usr/bin/env python3
"""Offline physical-budget analyzer for the Polaris quality architecture.

The analyzer never loads model weights.  It turns the frozen metadata and S14
schedule into auditable storage, active-byte, bandwidth and page-hit budgets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "architecture_spec.json"
REPORT_PATH = ROOT / "budget_report.json"


def _required_hit_rate(
    page_uses: int,
    page_bytes: int,
    storage_bytes_per_second: float,
    target_tps: float,
) -> dict[str, float]:
    allowed_bytes = storage_bytes_per_second / target_tps
    allowed_pages = allowed_bytes / page_bytes
    hit_rate = max(0.0, 1.0 - allowed_pages / page_uses)
    return {
        "target_tps": target_tps,
        "allowed_miss_bytes_per_token": allowed_bytes,
        "allowed_miss_pages_per_token": allowed_pages,
        "required_page_hit_rate": hit_rate,
    }


def analyze(spec: dict) -> dict:
    donor = spec["deepseek_v4_0731"]
    candidate = spec["preregistered_candidate"]
    local = spec["authoritative_local_evidence"]

    layers = candidate["selected_layers"]
    shard_sizes = donor["layer_shard_bytes"]
    layer_count = donor["num_hidden_layers"]

    assert len(shard_sizes) == layer_count == 43
    assert layers == sorted(set(layers))
    assert all(0 <= layer < layer_count for layer in layers)
    assert layers[:3] == [0, 1, 2]
    assert layers[-3:] == [40, 41, 42]
    assert len(donor["compress_ratios_main_layers"]) == layer_count

    selected_shard_bytes = sum(shard_sizes[layer] for layer in layers)
    storage_bytes = (
        donor["embedding_file_bytes"]
        + donor["output_head_file_bytes"]
        + selected_shard_bytes
    )

    # Each selected shard contains all 256 experts.  At decode, the score below
    # keeps all non-expert bytes in that shard and replaces the full expert bank
    # with the exact top-6 routed payload.  Headers make this a slight upper bound.
    active_layer_bytes_upper = sum(
        shard_sizes[layer]
        - donor["full_expert_bank_per_layer_bytes"]
        + donor["top6_expert_bytes"]
        for layer in layers
    )
    active_weight_bytes_upper = (
        active_layer_bytes_upper + donor["output_head_file_bytes"]
    )

    head_params_b = donor["output_head_parameters"] / 1e9
    active_params_b_estimate = head_params_b + (
        donor["active_parameters_reported"] / 1e9 - head_params_b
    ) * len(layers) / layer_count

    gaps = [right - left for left, right in zip(layers, layers[1:])]
    compression_pairs = [
        [layer, donor["compress_ratios_main_layers"][layer]] for layer in layers
    ]

    bandwidth = local["rx5700xt_nominal_bandwidth_gb_s"] * 1e9
    bandwidth_scenarios = []
    for efficiency in (1.0, 0.5, 0.3):
        effective = bandwidth * efficiency
        bandwidth_scenarios.append(
            {
                "bandwidth_efficiency": efficiency,
                "effective_gb_s": effective / 1e9,
                "weight_scan_tps_upper_bound": effective
                / active_weight_bytes_upper,
            }
        )

    page_uses = len(layers) * donor["num_experts_per_token"]
    measured_bps = local["measured_random_multispan_mib_s"] * 1024 * 1024
    optimistic_bps = local["optimistic_sequential_mib_s"] * 1024 * 1024
    target_tps = spec["objective"]["target_tokens_per_second"]

    report = {
        "schema_version": "polaris.quality_budget.v1",
        "candidate_id": candidate["id"],
        "donor": {
            "repo": donor["repo"],
            "revision": donor["revision"],
            "checkpoint_parameters": donor["checkpoint_parameters"],
        },
        "schedule": {
            "selected_layers": layers,
            "selected_layer_count": len(layers),
            "retained_depth_fraction": len(layers) / layer_count,
            "maximum_layer_index_gap": max(gaps),
            "maximum_consecutive_skipped_layers": max(gaps) - 1,
            "selected_layer_compress_ratios": compression_pairs,
            "residual_rescaling": candidate["residual_rescaling"],
        },
        "storage": {
            "selected_layer_shards_bytes": selected_shard_bytes,
            "embedding_file_bytes": donor["embedding_file_bytes"],
            "output_head_file_bytes": donor["output_head_file_bytes"],
            "total_bytes": storage_bytes,
            "total_gb_decimal": storage_bytes / 1e9,
            "total_gib": storage_bytes / (1024**3),
            "fits_50_to_70_gb": 50e9 <= storage_bytes <= 70e9,
        },
        "decode_physics": {
            "active_parameters_b_estimate": active_params_b_estimate,
            "active_selected_layers_bytes_upper": active_layer_bytes_upper,
            "active_weight_bytes_per_token_upper": active_weight_bytes_upper,
            "active_weight_gb_per_token_upper": active_weight_bytes_upper / 1e9,
            "required_gb_s": [
                {
                    "target_tps": tps,
                    "weight_scan_gb_s": active_weight_bytes_upper * tps / 1e9,
                }
                for tps in target_tps
            ],
            "rx5700xt_weight_scan_bounds": bandwidth_scenarios,
            "note": "These are weight-scan ceilings, not measured model speed; kernels, quant decode, attention, KV, synchronization and CPU/GPU split only reduce throughput.",
        },
        "expert_paging": {
            "page_bytes": donor["single_expert_bytes"],
            "page_uses_per_token": page_uses,
            "routed_expert_bytes_per_token": page_uses
            * donor["single_expert_bytes"],
            "measured_random_multispan": [
                _required_hit_rate(
                    page_uses,
                    donor["single_expert_bytes"],
                    measured_bps,
                    tps,
                )
                for tps in target_tps
            ],
            "optimistic_sequential": [
                _required_hit_rate(
                    page_uses,
                    donor["single_expert_bytes"],
                    optimistic_bps,
                    tps,
                )
                for tps in target_tps
            ],
        },
        "decision": {
            "physical_budget_pass": (
                50e9 <= storage_bytes <= 70e9
                and 3.0 <= active_params_b_estimate <= 5.0
                and bandwidth_scenarios[-1]["weight_scan_tps_upper_bound"] >= 20
            ),
            "quality_pass": None,
            "quality_status": "unverified; requires the preregistered native sparse-depth experiment",
            "may_claim_claude_gpt_quality_now": False,
        },
    }
    return report


def self_test(spec: dict, report: dict) -> None:
    assert report["storage"]["total_bytes"] == 52_231_273_716
    assert report["storage"]["fits_50_to_70_gb"] is True
    assert report["schedule"]["selected_layer_count"] == 14
    assert math.isclose(
        report["decode_physics"]["active_parameters_b_estimate"],
        4.589683616744186,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert report["decode_physics"]["active_weight_bytes_per_token_upper"] == 4_379_507_860
    assert report["expert_paging"]["page_uses_per_token"] == 84
    assert report["decision"]["physical_budget_pass"] is True
    assert report["decision"]["quality_pass"] is None
    assert report["decision"]["may_claim_claude_gpt_quality_now"] is False


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    report = analyze(spec)
    self_test(spec, report)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "pass",
        "report": REPORT_PATH.name,
        "storage_gb": report["storage"]["total_gb_decimal"],
        "active_parameters_b_estimate": report["decode_physics"]["active_parameters_b_estimate"],
        "active_weight_gb_per_token_upper": report["decode_physics"]["active_weight_gb_per_token_upper"],
        "quality_status": report["decision"]["quality_status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
