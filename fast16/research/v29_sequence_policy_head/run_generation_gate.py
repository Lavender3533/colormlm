"""运行v29冻结20题的严格生成门，并保存逐题可复核证据。"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


TOOL_TARGET = re.compile(r"^<tool_call>\s*(\{.*\})\s*</tool_call>$", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", type=Path, default=Path(__file__).with_name("policy_tasks_v1.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def normalize_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw_arguments = function.get("arguments", "")
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            arguments = {"__invalid_json__": raw_arguments}
        normalized.append({"name": function.get("name"), "arguments": arguments})
    return normalized


def judge(task: dict[str, Any], response: dict[str, Any]) -> tuple[bool, str]:
    choice = response["choices"][0]
    message = choice["message"]
    finish_reason = choice.get("finish_reason")
    content = message.get("content") or ""
    calls = normalize_calls(message)
    target = str(task["target"])
    match = TOOL_TARGET.fullmatch(target.strip())
    if match:
        expected = json.loads(match.group(1))
        if finish_reason != "tool_calls":
            return False, f"结束原因不是tool_calls: {finish_reason}"
        if content.strip():
            return False, "工具调用前后含额外文本"
        if len(calls) != 1:
            return False, f"工具调用数不是1: {len(calls)}"
        actual = calls[0]
        if actual["name"] != expected["name"]:
            return False, f"工具名错误: {actual['name']}"
        if actual["arguments"] != expected["arguments"]:
            return False, "工具参数与冻结目标不一致"
        return True, "严格工具调用匹配"

    if calls:
        return False, "结束任务错误地产生工具调用"
    if finish_reason != "stop":
        return False, f"结束原因不是stop: {finish_reason}"
    try:
        actual_json = json.loads(content.strip())
        expected_json = json.loads(target)
    except json.JSONDecodeError:
        return False, "输出不是裸JSON"
    if actual_json != expected_json:
        return False, "JSON与冻结目标不一致"
    return True, "严格JSON匹配"


def main() -> int:
    args = parse_args()
    tasks = [
        json.loads(line)
        for line in args.tasks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    endpoint = args.endpoint.rstrip("/") + "/v1/chat/completions"
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        payload = {
            "model": args.model,
            "messages": task["messages"],
            "tools": task["tools"],
            "temperature": 0,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            response = post_json(endpoint, payload)
            passed, reason = judge(task, response)
            error = None
        except (OSError, KeyError, IndexError, TypeError, ValueError, urllib.error.HTTPError) as exc:
            response = None
            passed, reason, error = False, "请求或响应解析失败", repr(exc)
        wall_ms = (time.perf_counter() - started) * 1000
        row = {
            "id": task["id"],
            "split": task["split"],
            "family": task["family"],
            "passed": passed,
            "reason": reason,
            "wall_ms": wall_ms,
            "target": task["target"],
            "response": response,
            "error": error,
        }
        rows.append(row)
        print(f"[{index:02d}/{len(tasks):02d}] {task['id']}: {'PASS' if passed else 'FAIL'} - {reason}", flush=True)

    by_split: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, int]] = {}
    for row in rows:
        for key, bucket in ((row["split"], by_split), (row["family"], by_family)):
            stats = bucket.setdefault(key, {"passed": 0, "total": 0})
            stats["total"] += 1
            stats["passed"] += int(row["passed"])
    report = {
        "format": "colorlm-v29-generation-gate-v1",
        "model": args.model,
        "endpoint": args.endpoint,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "passed": sum(int(row["passed"]) for row in rows),
        "total": len(rows),
        "by_split": by_split,
        "by_family": by_family,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "total", "by_split", "by_family")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
