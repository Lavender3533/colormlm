"""Command-line entry points for analysis, tokenization and cache replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .assets import audit_assets
from .cache_replay import ExpertPageCache, load_route_blocks_jsonl, replay_blocks
from .cost_model import HardwareBudget, build_analysis_report, committed_tokens
from .speed_gate import current_serial_static_gate
from .verifier import DeepSeekTokenizer


DEFAULT_ASSET_ROOT = "D:/models/Polaris-S14"


def _emit_json(value: dict[str, Any], output: str | None) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        Path(output).write_text(encoded, encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polaris FullDepth43 离线验证/成本原型")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="交叉审计真实 header/catalog 并输出 PCIe 条件")
    analyze.add_argument("--asset-root", default=DEFAULT_ASSET_ROOT)
    analyze.add_argument("--output")

    tokenize = subparsers.add_parser("tokenize-draft", help="用本地 DeepSeek tokenizer 取 S14 候选文本的 N token")
    tokenize.add_argument("--asset-root", default=DEFAULT_ASSET_ROOT)
    tokenize.add_argument("--text", required=True, help="S14 后端已生成的候选 continuation")
    tokenize.add_argument("--block-size", type=int, required=True)
    tokenize.add_argument("--output")

    replay = subparsers.add_parser("replay-cache", help="回放 FullDepth43/native-top6 JSONL route trace")
    replay.add_argument("trace")
    replay.add_argument("--asset-root", default=DEFAULT_ASSET_ROOT)
    replay.add_argument("--cache-pages", type=int, help="默认用 8GiB 扣除非路由+head 后的理论页数")
    replay.add_argument("--output")

    speed = subparsers.add_parser(
        "static-speed-gate", help="证明当前串行 FullDepth verifier 不具备加速资格"
    )
    speed.add_argument("--baseline-seconds-per-token", type=float, required=True)
    speed.add_argument("--block-size", type=int, choices=(4, 8), required=True)
    speed.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "analyze":
        _emit_json(build_analysis_report(args.asset_root), args.output)
        return 0
    if args.command == "tokenize-draft":
        tokenizer = DeepSeekTokenizer(args.asset_root)
        token_ids = tokenizer.encode_exact_draft(args.text, args.block_size)
        _emit_json(
            {
                "format": "polaris-s14-tokenized-draft-v1",
                "source": "caller_supplied_S14_continuation",
                "profile": "S14/top6 + DeepSeek tokenizer",
                "tokenizer_path": tokenizer.path,
                "tokenizer_sha256": tokenizer.fingerprint,
                "block_size": args.block_size,
                "token_ids": list(token_ids),
                "decoded_prefix": tokenizer.decode(token_ids),
                "claim_limit": "只证明本地 tokenizer 产生了 N 个原生 token ID；不表示 S14 权重已运行。",
            },
            args.output,
        )
        return 0
    if args.command == "static-speed-gate":
        report = current_serial_static_gate(
            baseline_target_seconds_per_token=args.baseline_seconds_per_token,
            block_size=args.block_size,
        )
        _emit_json(report, args.output)
        # A rejected current serial path is the expected gate result, not a
        # positive speed claim.  Return 2 so automation cannot accidentally
        # promote it.
        return 0 if report["passed"] else 2
    if args.command == "replay-cache":
        audit = audit_assets(args.asset_root)
        hardware = HardwareBudget()
        default_pages = max(0, hardware.vram_bytes - audit.fixed_decode_scan_bytes) // audit.expert_page_bytes
        capacity_pages = default_pages if args.cache_pages is None else args.cache_pages
        report = replay_blocks(
            load_route_blocks_jsonl(args.trace),
            ExpertPageCache(capacity_pages),
            audit.expert_page_bytes,
        )
        value = report.to_dict()
        if all(block.accepted_prefix_length is not None for block in report.blocks):
            committed = sum(
                committed_tokens(block.block_size, block.accepted_prefix_length)  # type: ignore[arg-type]
                for block in report.blocks
            )
            streamed_bytes = len(report.blocks) * audit.fixed_decode_scan_bytes + report.pcie_expert_bytes
            resident_bytes = report.pcie_expert_bytes
            throughput_projection: dict[str, Any] | None = {
                "committed_tokens": committed,
                "stream_each_block": {
                    "pcie_bytes": streamed_bytes,
                    "throughput_ceiling_tps": committed * hardware.pcie_bytes_per_second / streamed_bytes,
                },
                "resident_after_warmup": {
                    "pcie_bytes": resident_bytes,
                    "throughput_ceiling_tps": (
                        committed * hardware.pcie_bytes_per_second / resident_bytes
                        if resident_bytes
                        else None
                    ),
                    "null_tps_note": "零 PCIe 字节时该模型无法约束吞吐，不表示无限实际速度。",
                },
            }
        else:
            throughput_projection = None
        value.update(
            {
                "format": "polaris-full-depth-expert-cache-replay-v1",
                "profile": "FullDepth43/native-top6",
                "cache_semantics": "LRU device-resident pages; block de-dup is reported separately",
                "throughput_projection": throughput_projection,
            }
        )
        _emit_json(value, args.output)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
