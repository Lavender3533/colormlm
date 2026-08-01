"""检查 v38 无tools物理旁路与显式tools固定请求速度。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, os.fspath(HERE))

from run_v31_gate import port_available, post_json, start_server, stop_server  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 v38 最小运行契约检查")
    parser.add_argument("--port", type=int, default=8138)
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
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT
        / "fast16"
        / "models"
        / "ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT
        / "fast16"
        / "research"
        / "v29_sequence_policy_head"
        / "runtime-v1",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "v38_runtime_check.json"
    )
    return parser.parse_args()


def policy_environment(path: Path) -> dict[str, str]:
    manifest = path / "policy.json"
    return {
        "COLORLM_SEQUENCE_POLICY_PACKAGE": str(path.resolve()),
        "COLORLM_SEQUENCE_POLICY_MANIFEST_SHA256": hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
        "COLORLM_SEQUENCE_POLICY_VERIFY_WEIGHTS": "1",
    }


def request_pair(port: int, alias: str) -> dict[str, object]:
    base = {
        "model": alias,
        "messages": [
            {
                "role": "user",
                "content": "从1开始输出连续整数，用一个空格分隔，不要解释，直到200。",
            }
        ],
        "temperature": 0,
        "seed": 7319,
        "max_tokens": 96,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    no_tools = post_json(port, dict(base))
    with_tools_payload = dict(base)
    with_tools_payload["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "noop",
                "description": "只有明确要求时才调用；当前不要调用。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }
    ]
    with_tools_payload["tool_choice"] = "auto"
    with_tools = post_json(port, with_tools_payload)
    return {"no_tools": no_tools, "with_tools": with_tools}


def response_digest(response: dict[str, object]) -> dict[str, object]:
    choice = response["choices"][0]  # type: ignore[index]
    message = choice["message"]  # type: ignore[index]
    return {
        "content": message.get("content") or "",  # type: ignore[union-attr]
        "tool_calls": message.get("tool_calls") or [],  # type: ignore[union-attr]
        "finish_reason": choice.get("finish_reason"),  # type: ignore[union-attr]
        "usage": response.get("usage", {}),
        "timings": response.get("timings", {}),
    }


def main() -> int:
    args = parse_args()
    for path in (
        args.server,
        args.model,
        args.policy / "policy.json",
        args.policy / "weights.bin",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")

    variants = [
        ("ColorLM-v36-Qwen36-Global-Shared-Backbone", {}),
        (
            "ColorLM-v38-Qwen36-Shared-Sequence-Policy",
            policy_environment(args.policy),
        ),
    ]
    rows = []
    for alias, environment in variants:
        print(f"启动 {alias}", flush=True)
        process, load_seconds = start_server(
            args.server,
            args.model,
            alias,
            args.port,
            extra_environment=environment,
        )
        try:
            pair = request_pair(args.port, alias)
            rows.append(
                {
                    "alias": alias,
                    "load_seconds": load_seconds,
                    "no_tools": response_digest(pair["no_tools"]),
                    "with_tools": response_digest(pair["with_tools"]),
                }
            )
        finally:
            stop_server(process)
        time.sleep(1)

    base, candidate = rows
    no_tools_exact = all(
        base["no_tools"][key] == candidate["no_tools"][key]
        for key in ("content", "tool_calls", "finish_reason", "usage")
    )
    base_tps = float(base["with_tools"]["timings"].get("predicted_per_second", 0.0))
    candidate_tps = float(
        candidate["with_tools"]["timings"].get("predicted_per_second", 0.0)
    )
    report = {
        "format": "colormlm-v38-runtime-check-v1",
        "models": rows,
        "no_tools_exact": no_tools_exact,
        "with_tools_speed": {
            "v36_tokens_per_second": base_tps,
            "v38_tokens_per_second": candidate_tps,
            "relative_delta": candidate_tps / base_tps - 1.0 if base_tps else None,
        },
        "passes": bool(
            no_tools_exact
            and base_tps
            and candidate_tps / base_tps - 1.0 >= -0.05
        ),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: report[key] for key in ("no_tools_exact", "with_tools_speed", "passes")},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"报告: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
