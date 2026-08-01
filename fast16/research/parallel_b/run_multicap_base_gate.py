"""通过 OpenAI 兼容接口采集多能力短门响应。

该脚本只负责冻结参数下的 G2 生成，不承担 G0/G1，也不自动晋级模型。
输出格式与 validate_multicap_gate.py 的 response_record_schema 兼容。
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_TASKS = Path(__file__).with_name("multicap_short_gate_v1.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tool_schema(task: dict[str, Any]) -> list[dict[str, Any]] | None:
    validator = task["validator"]
    if validator["type"] != "tool_call":
        return None

    name = validator["expected_name"]
    arguments = validator["expected_arguments"]
    properties = {key: {"type": "string"} for key in arguments}
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": task["prompt"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(arguments),
                    "additionalProperties": False,
                },
            },
        }
    ]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def extract_record(task: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    message = choice["message"]
    record: dict[str, Any] = {
        "id": task["id"],
        "output": message.get("content") or "",
        "finish_reason": choice.get("finish_reason"),
    }
    calls = message.get("tool_calls") or []
    if calls:
        record["tool_calls"] = calls
    record["usage"] = response.get("usage", {})
    record["timings"] = response.get("timings", {})
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="采集 ColorLM 多能力短门响应")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--one-per-dimension",
        action="store_true",
        help="只跑每个维度第一题，用于方向性冒烟门",
    )
    args = parser.parse_args()

    bank = load_json(args.tasks)
    tasks = bank["items"]
    if args.one_per_dimension:
        seen: set[str] = set()
        tasks = [
            task
            for task in tasks
            if task["dimension"] not in seen and not seen.add(task["dimension"])
        ]

    endpoint = normalize_endpoint(args.endpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    rows: list[dict[str, Any]] = []

    for index, task in enumerate(tasks, 1):
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": [
                {
                    "role": "system",
                    "content": "严格遵守用户要求的输出格式；不要解释，不要添加 Markdown。",
                },
                {"role": "user", "content": task["prompt"]},
            ],
            "temperature": 0,
            "seed": args.seed,
            "max_tokens": task["max_output_tokens"],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        tools = tool_schema(task)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required"

        started = time.perf_counter()
        response = post_json(endpoint, payload, args.timeout)
        record = extract_record(task, response)
        record["wall_ms"] = round((time.perf_counter() - started) * 1000, 3)
        record["model"] = args.model
        record["seed"] = args.seed
        rows.append(record)
        partial.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(
            f"[{index:02d}/{len(tasks):02d}] {task['id']} "
            f"{record['wall_ms'] / 1000:.2f}s",
            flush=True,
        )

    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    partial.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
