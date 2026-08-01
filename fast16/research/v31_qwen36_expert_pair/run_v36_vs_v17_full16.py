"""在既有冻结16题上比较正式 v17 与新 v36 shared-backbone。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARALLEL_B = ROOT / "fast16" / "research" / "parallel_b"
sys.path.insert(0, os.fspath(HERE))
sys.path.insert(0, os.fspath(PARALLEL_B))

from run_multicap_base_gate import extract_record, post_json, tool_schema  # noqa: E402
from run_v31_gate import port_available, run_speed, start_server, stop_server  # noqa: E402
from validate_multicap_gate import compare  # noqa: E402


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(description="运行 v36 与 v17 的冻结16题比较")
    parser.add_argument("--port", type=int, default=8136)
    parser.add_argument(
        "--tasks", type=Path, default=PARALLEL_B / "multicap_short_gate_v1.json"
    )
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
        "--v17-model",
        type=Path,
        default=models / "ColorLM-v6-Q3Router-Fused-A1.gguf",
    )
    parser.add_argument(
        "--v36-model",
        type=Path,
        default=models / "ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf",
    )
    parser.add_argument(
        "--island",
        type=Path,
        default=ROOT
        / "fast16"
        / "research"
        / "v17_coder_island"
        / "runtime-v3"
        / "island.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "v36_vs_v17_full16_report.json",
    )
    return parser.parse_args()


def run_bank(port: int, alias: str, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, task in enumerate(tasks, 1):
        payload: dict[str, Any] = {
            "model": alias,
            "messages": [
                {
                    "role": "system",
                    "content": "严格遵守用户要求的输出格式；不要解释，不要添加 Markdown。",
                },
                {"role": "user", "content": task["prompt"]},
            ],
            "temperature": 0,
            "seed": 3407,
            "max_tokens": task["max_output_tokens"],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        tools = tool_schema(task)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required"
        started = time.perf_counter()
        response = post_json(
            f"http://127.0.0.1:{port}/v1/chat/completions", payload, 180.0
        )
        row = extract_record(task, response)
        row["wall_ms"] = round((time.perf_counter() - started) * 1000, 3)
        row["model"] = alias
        row["seed"] = 3407
        rows.append(row)
        print(f"  [{index:02d}/{len(tasks):02d}] {task['id']}", flush=True)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def island_environment(path: Path) -> dict[str, str]:
    return {
        "COLORLM_NEURAL_ISLAND_MANIFEST": str(path.resolve()),
        "COLORLM_NEURAL_ISLAND_ALPHA": "0.02",
        "COLORLM_NEURAL_ISLAND_SITE": "35",
        "COLORLM_NEURAL_ISLAND_TARGET_RATIO": "0.04",
        "COLORLM_NEURAL_ISLAND_SHARPNESS": "4.0",
        "COLORLM_NEURAL_ISLAND_EXPERT_CACHE_SLOTS": "32",
        "COLORLM_NEURAL_ISLAND_EXPERT_CACHE_POLICY": "lru",
    }


def main() -> int:
    args = parse_args()
    for path in (args.tasks, args.server, args.v17_model, args.v36_model, args.island):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")
    bank = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = bank["items"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    variants = [
        (
            "ColorLM-v17-Coder-Neural-Island",
            args.v17_model,
            island_environment(args.island),
        ),
        ("ColorLM-v36-Qwen36-Global-Shared-Backbone", args.v36_model, {}),
    ]
    response_paths = []
    runtime_rows = []
    for alias, model, environment in variants:
        print(f"启动 {alias}", flush=True)
        process, load_seconds = start_server(
            args.server,
            model,
            alias,
            args.port,
            extra_environment=environment,
        )
        try:
            rows = run_bank(args.port, alias, tasks)
            response_path = HERE / f"{alias}.full16.responses.jsonl"
            write_jsonl(response_path, rows)
            response_paths.append(response_path)
            runtime_rows.append(
                {
                    "alias": alias,
                    "model": model.name,
                    "model_bytes": model.stat().st_size,
                    "load_seconds": load_seconds,
                    "speed": run_speed(args.port, alias),
                }
            )
        finally:
            stop_server(process)
        time.sleep(1)

    dimensions = list(bank["dimensions"])
    comparison = compare(
        args.tasks,
        response_paths[0],
        response_paths[1],
        dimensions,
    )
    comparison["runtime"] = runtime_rows
    base_tps = float(runtime_rows[0]["speed"]["timings"]["predicted_per_second"])
    candidate_tps = float(
        runtime_rows[1]["speed"]["timings"]["predicted_per_second"]
    )
    comparison["speed"] = {
        "v17_tokens_per_second": base_tps,
        "v36_tokens_per_second": candidate_tps,
        "relative_delta": candidate_tps / base_tps - 1.0,
    }
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "v17": comparison["baseline"]["summary"],
                "v36": comparison["candidate"]["summary"],
                "paired": comparison["paired"],
                "speed": comparison["speed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"报告: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
