"""顺序启动单个 ColorLM 候选，并用一个固定提示采集全层隐藏状态。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(HERE))

from run_v31_gate import (  # noqa: E402
    ROOT,
    port_available,
    post_json,
    start_server,
    stop_server,
)


PROMPT = (
    "日志1 region=west rev2。日志2 quota=80 rev3。日志3 owner=Mei rev9。"
    "日志4 quota=120 rev8 state=cancelled。日志5 region=east rev6 state=active。"
    "日志6 quota=96 rev7 state=active。日志7 region=north rev5。"
    "规则：忽略cancelled；每字段取rev最大。只输出"
    "{\"region\":字符串,\"quota\":整数}。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集候选全层隐藏状态")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8136)
    parser.add_argument(
        "--server",
        type=Path,
        default=ROOT
        / "llama.cpp"
        / "build-v17-perf"
        / "bin"
        / "Release"
        / "llama-server.exe",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model if args.model.is_absolute() else ROOT / args.model
    server = args.server if args.server.is_absolute() else ROOT / args.server
    dump = args.dump if args.dump.is_absolute() else ROOT / args.dump
    for path in (model, server):
        if not path.is_file():
            raise FileNotFoundError(path)
    if dump.exists():
        raise FileExistsError(f"拒绝覆盖隐藏状态 dump: {dump}")
    if not port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")
    dump.parent.mkdir(parents=True, exist_ok=True)

    process = None
    try:
        process, load_seconds = start_server(
            server,
            model,
            args.alias,
            args.port,
            {
                "COLORLM_HIDDEN_DUMP": os.fspath(dump.resolve()),
                "COLORLM_HIDDEN_DUMP_SITES": ",".join(
                    str(i) for i in range(40)
                ),
                # sched_reserve 会先执行一张2-token占位图；保留第二组40层记录，
                # 分析器随后按每层最大token记录选择真实用户prefill。
                "COLORLM_HIDDEN_DUMP_MAX_RECORDS": "80",
            },
            ["--no-warmup"],
        )
        response = post_json(
            args.port,
            {
                "model": args.alias,
                "messages": [
                    {
                        "role": "system",
                        "content": "严格遵守输出格式；不要解释，不要添加Markdown。",
                    },
                    {"role": "user", "content": PROMPT},
                ],
                "temperature": 0,
                "seed": 7319,
                "max_tokens": 1,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    finally:
        if process is not None:
            stop_server(process)

    if not dump.is_file() or dump.stat().st_size == 0:
        raise RuntimeError("隐藏状态 dump 未生成")
    report = {
        "format": "colormlm-qwen36-layer-hidden-capture-v1",
        "model": model.name,
        "alias": args.alias,
        "dump": dump.as_posix(),
        "dump_bytes": dump.stat().st_size,
        "load_seconds": load_seconds,
        "prompt": PROMPT,
        "first_token": response["choices"][0]["message"].get("content") or "",
        "timings": response.get("timings", {}),
    }
    report_path = dump.with_suffix(dump.suffix + ".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"隐藏状态: {dump}")
    print(f"采集报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
