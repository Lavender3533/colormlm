"""Calculate exact expert paging memory and SSD bandwidth budgets from a GGUF."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算MoE专家分页的内存和SSD预算")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "fast16" / "models" / "ColorLM-v6-Q3Router-Fused-A1.gguf",
    )
    parser.add_argument("--ssd-gib-per-second", type=float, default=3.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESEARCH / "expert_paging_budget_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reader = GGUFReader(os.fspath(args.model), "r")
    fields = reader.fields
    architecture = str(fields["general.architecture"].contents())
    expert_count = int(fields[f"{architecture}.expert_count"].contents())
    active_count = int(fields[f"{architecture}.expert_used_count"].contents())
    layer_count = int(fields[f"{architecture}.block_count"].contents())
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    layer_bytes: list[int] = []
    for layer in range(layer_count):
        names = [
            f"blk.{layer}.ffn_gate_exps.weight",
            f"blk.{layer}.ffn_up_exps.weight",
            f"blk.{layer}.ffn_down_exps.weight",
        ]
        layer_bytes.append(sum(int(tensors[name].data.nbytes) for name in names))
    full_expert_bytes = sum(layer_bytes)
    per_expert_layer = [value / expert_count for value in layer_bytes]
    mean_expert_bytes = sum(per_expert_layer) / len(per_expert_layer)
    bandwidth = args.ssd_gib_per_second * 1024**3

    slots = []
    for hot_count in (8, 16, 32, 64):
        resident = sum(value * hot_count / expert_count for value in layer_bytes)
        rolling_two_layers = sum(layer_bytes[:2]) * hot_count / expert_count
        slots.append(
            {
                "hot_experts_per_layer": hot_count,
                "all_layer_pool_gib": resident / 1024**3,
                "rolling_two_layer_pool_mib": rolling_two_layers / 1024**2,
                "fraction_of_full_expert_bank": hot_count / expert_count,
            }
        )

    misses = []
    selected_bytes_per_token = mean_expert_bytes * active_count * layer_count
    for hit_rate in (0.0, 0.5, 0.9, 0.95, 0.99):
        miss_bytes = selected_bytes_per_token * (1.0 - hit_rate)
        misses.append(
            {
                "cache_hit_rate": hit_rate,
                "ssd_read_mib_per_token": miss_bytes / 1024**2,
                "ideal_ssd_time_ms_per_token": miss_bytes / bandwidth * 1000.0,
            }
        )

    report = {
        "format": "colorlm-expert-paging-budget-v1",
        "model": args.model.name,
        "architecture": architecture,
        "layers": layer_count,
        "experts_per_layer": expert_count,
        "active_experts_per_token": active_count,
        "full_expert_bank_gib": full_expert_bytes / 1024**3,
        "mean_expert_mib_per_layer": mean_expert_bytes / 1024**2,
        "slot_pool": slots,
        "ssd_assumption_gib_per_second": args.ssd_gib_per_second,
        "cache_miss_budget": misses,
        "notes": [
            "all_layer_pool keeps K hot experts for every layer",
            "rolling pool is a lower bound and requires correct asynchronous prefetch",
            "SSD times exclude random I/O, synchronization, decompression and GPU upload",
        ],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
