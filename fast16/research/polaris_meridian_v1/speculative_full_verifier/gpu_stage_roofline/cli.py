"""命令行入口：生成冻结报告或运行可证伪的阶段场景。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .anchors import load_real_anchors
from .model import CommandBufferMode, SimulationConfig, build_roofline_report, simulate_token


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Polaris 草稿 + FullDepth43 exact verifier 阶段级 roofline（不加载模型权重）"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser("report", help="从真实元数据与冻结计时锚点生成完整报告")
    report.add_argument("--asset-root", default="D:/models/Polaris-S14")
    report.add_argument("--repo-root")
    report.add_argument("--output", type=Path, help="写入 UTF-8 JSON；省略则输出到 stdout")

    scenario = commands.add_parser("simulate", help="运行一个显式参数的单 token 阶段场景")
    scenario.add_argument("--asset-root", default="D:/models/Polaris-S14")
    scenario.add_argument("--repo-root")
    scenario.add_argument("--target-tps", type=float, required=True)
    scenario.add_argument("--expert-hit-rate", type=float, required=True)
    scenario.add_argument("--pcie-gb-s", type=float, default=22.03)
    scenario.add_argument("--batch-size", type=int, default=1)
    scenario.add_argument("--batch-efficiency", type=float, default=1.0)
    scenario.add_argument("--kernel-fusion-speedup", type=float, default=1.0)
    scenario.add_argument("--attention-remainder-ms-per-layer", type=float, default=0.0)
    scenario.add_argument("--hc-ms-per-layer", type=float, default=0.0)
    scenario.add_argument("--router-ms-per-layer", type=float, default=0.0)
    scenario.add_argument("--norm-head-ms-per-token", type=float, default=0.0)
    scenario.add_argument(
        "--command-buffer-mode",
        choices=[mode.value for mode in CommandBufferMode],
        default=CommandBufferMode.ROUTE_SPLIT_PER_LAYER.value,
    )
    scenario.add_argument("--submit-overhead-us", type=float, default=0.0)
    scenario.add_argument("--output", type=Path, help="写入 UTF-8 JSON；省略则输出到 stdout")
    return parser


def _emit(value: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "report":
        result = build_roofline_report(args.asset_root, args.repo_root)
    elif args.command == "simulate":
        anchors = load_real_anchors(args.asset_root, args.repo_root)
        config = SimulationConfig(
            target_tps=args.target_tps,
            expert_cache_hit_rate=args.expert_hit_rate,
            pcie_bytes_per_second=args.pcie_gb_s * 1e9,
            batch_size=args.batch_size,
            batch_efficiency=args.batch_efficiency,
            kernel_fusion_speedup=args.kernel_fusion_speedup,
            attention_remainder_ms_per_layer=args.attention_remainder_ms_per_layer,
            hc_ms_per_layer=args.hc_ms_per_layer,
            router_ms_per_layer=args.router_ms_per_layer,
            norm_head_ms_per_token=args.norm_head_ms_per_token,
            command_buffer_mode=CommandBufferMode(args.command_buffer_mode),
            submit_overhead_us=args.submit_overhead_us,
        )
        result = {
            "format": "polaris-s14-stage-scenario-v1",
            "anchors": anchors.to_dict(),
            "simulation": simulate_token(anchors, config).to_dict(include_events=True),
            "claim_limit": "显式假设下的阶段调度结果，不是完整 GPU token 实测。",
        }
    else:  # pragma: no cover - argparse 已封闭该分支
        raise AssertionError(args.command)
    _emit(result, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
