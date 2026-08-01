"""把实测页吞吐换算为全深度巨型 donor 的硬速度预算。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="北极星巨型 donor 页命中与带宽预算")
    parser.add_argument("--cache-report", type=Path, default=HERE / "real_gguf_cache_report.json")
    parser.add_argument("--output", type=Path, default=HERE / "speed_budget_report.json")
    parser.add_argument("--expert-bytes", type=int, default=17_547_264, help="Kimi K3 单 routed expert")
    parser.add_argument("--moe-layers", type=int, default=92)
    parser.add_argument("--active-experts", type=int, default=16)
    parser.add_argument("--active-parameters", type=float, default=104e9)
    parser.add_argument("--gpu-bandwidth-gb-s", type=float, default=448.0)
    args = parser.parse_args()

    measured = json.loads(args.cache_report.read_text(encoding="utf-8"))
    measured_mib_s = float(measured["prefetch"]["wall_mib_per_second"])
    expert_mib = args.expert_bytes / 1024**2
    uses = args.moe_layers * args.active_experts
    active_expert_mib = expert_mib * uses

    def miss_budget(target_tps: float, storage_mib_s: float) -> dict[str, float]:
        allowed_mib_per_token = storage_mib_s / target_tps
        allowed_pages = allowed_mib_per_token / expert_mib
        return {
            "target_tokens_per_second": target_tps,
            "storage_mib_per_second": storage_mib_s,
            "allowed_miss_mib_per_token": allowed_mib_per_token,
            "allowed_miss_pages_per_token": allowed_pages,
            "required_page_hit_rate": max(0.0, 1.0 - allowed_pages / uses),
        }

    bytes_per_parameter = 0.5
    active_weight_gb = args.active_parameters * bytes_per_parameter / 1e9
    theoretical_tps = args.gpu_bandwidth_gb_s / active_weight_gb
    active_caps: list[dict[str, float]] = []
    for target in (20.0, 50.0):
        max_active_b = args.gpu_bandwidth_gb_s / target / bytes_per_parameter
        active_caps.append(
            {
                "target_tokens_per_second": target,
                "q4_active_parameters_b_theoretical_100pct_bandwidth": max_active_b,
                "q4_active_parameters_b_at_50pct_bandwidth": max_active_b * 0.5,
            }
        )

    report = {
        "format": "polaris-meridian-speed-budget-v1",
        "donor_case": {
            "name": "Kimi K3 original routed path",
            "moe_layers": args.moe_layers,
            "active_experts_per_layer": args.active_experts,
            "expert_mib": expert_mib,
            "expert_page_uses_per_token": uses,
            "routed_expert_payload_mib_per_token": active_expert_mib,
            "reported_total_active_parameters": args.active_parameters,
        },
        "ssd_miss_budget": {
            "measured_source": str(args.cache_report.resolve()),
            "measured_os_cache_state": measured["prefetch"]["os_cache_state"],
            "measured_random_multispan": [
                miss_budget(20.0, measured_mib_s),
                miss_budget(50.0, measured_mib_s),
            ],
            "optimistic_3500_mib_s_sequential_ceiling": [
                miss_budget(20.0, 3500.0),
                miss_budget(50.0, 3500.0),
            ],
        },
        "vram_bandwidth_floor": {
            "rx5700xt_nominal_gb_per_second": args.gpu_bandwidth_gb_s,
            "q4_active_weight_gb_per_token": active_weight_gb,
            "original_path_theoretical_upper_tokens_per_second": theoretical_tps,
            "note": "只算每 token 扫描一次 Q4 权重；未计量化 scales、计算、attention、KV、同步，真实值只会更低。",
            "active_parameter_caps": active_caps,
        },
        "decision": {
            "original_k3_path_can_reach_20_50_tps_on_rx5700xt": False,
            "why": "104B 激活参数的 Q4 权重带宽下限已超过 RX 5700 XT；SSD miss 还要求接近 100% 页命中。",
            "runtime_target": "总容量可为 300B+，但每 token 必须压到约 3--5B 活跃参数，并让绝大多数活跃页在 VRAM/RAM 热集。",
            "research_risk": "未经训练把 K3 top-16 直接裁成 top-1/微块不会自动保留 K3 质量，必须用真实反事实 NLL/任务门证明。",
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
