#!/usr/bin/env python3
"""调用 OpenAI 兼容端点并生成 capability-response-v1 JSONL。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "capability-response-v1"
TEMPLATE_ID = "openai-chat-completions-server-template-v1"
INPUT_KEYS = frozenset({"messages", "tools", "temperature", "max_output_tokens"})
JSON_TYPES = frozenset({"string", "integer", "number", "boolean", "object", "array"})


class GateRunnerError(ValueError):
    """输入或端点响应不符合运行契约。"""


def read_task_rows(path: Path) -> list[dict[str, Any]]:
    """读取题目，但只向后续请求构造暴露 task id 与 input 白名单。"""
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise GateRunnerError(f"无法读取题目文件 {path}: {error}") from error
    if payload.startswith(b"\xef\xbb\xbf"):
        raise GateRunnerError(f"{path}: 禁止 UTF-8 BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GateRunnerError(f"{path}: 不是合法 UTF-8: {error}") from error

    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise GateRunnerError(f"{path}:{line_number}: JSONL 不允许空行")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise GateRunnerError(f"{path}:{line_number}: 非法 JSON: {error}") from error
        if not isinstance(raw, dict):
            raise GateRunnerError(f"{path}:{line_number}: 每行必须是 JSON 对象")

        task_id = raw.get("id")
        model_input = raw.get("input")
        if not isinstance(task_id, str) or not task_id:
            raise GateRunnerError(f"{path}:{line_number}: id 必须是非空字符串")
        if task_id in seen_ids:
            raise GateRunnerError(f"{path}:{line_number}: 重复 task id {task_id!r}")
        if not isinstance(model_input, dict):
            raise GateRunnerError(f"{path}:{line_number}: input 必须是对象")
        unknown = set(model_input) - INPUT_KEYS
        if unknown:
            raise GateRunnerError(f"{path}:{line_number}: input 含未知字段 {sorted(unknown)}")

        # 只复制明确允许进入模型请求的 input 字段，绝不传播题目的其他字段。
        safe_input = {key: model_input[key] for key in INPUT_KEYS if key in model_input}
        tasks.append({"id": task_id, "input": safe_input})
        seen_ids.add(task_id)
    if not tasks:
        raise GateRunnerError(f"{path}: 空 JSONL")
    return tasks


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise GateRunnerError("--endpoint 必须是合法的 http:// 或 https:// URL")
    if parsed.query or parsed.fragment:
        raise GateRunnerError("--endpoint 不允许 query 或 fragment")
    if parsed.path.endswith("/chat/completions"):
        return endpoint
    if parsed.path.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def _copy_messages(value: Any, task_id: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise GateRunnerError(f"{task_id}: input.messages 必须是非空数组")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise GateRunnerError(f"{task_id}: messages[{index}] 字段必须严格为 role/content")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str) or not content:
            raise GateRunnerError(f"{task_id}: messages[{index}] 的 role/content 无效")
        messages.append({"role": role, "content": content})
    return messages


def _json_schema_for_parameters(parameters: Any, task_id: str, tool_name: str) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        raise GateRunnerError(f"{task_id}: 工具 {tool_name!r} 的 parameters 必须是对象")
    properties: dict[str, Any] = {}
    for name, type_name in parameters.items():
        if not isinstance(name, str) or not name or type_name not in JSON_TYPES:
            raise GateRunnerError(f"{task_id}: 工具 {tool_name!r} 的参数声明无效")
        properties[name] = {"type": type_name}
    return {
        "type": "object",
        "properties": properties,
        "required": list(parameters),
        "additionalProperties": False,
    }


def _convert_tools(value: Any, task_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise GateRunnerError(f"{task_id}: input.tools 必须是非空数组")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, tool in enumerate(value):
        if not isinstance(tool, dict) or set(tool) != {"name", "parameters"}:
            raise GateRunnerError(f"{task_id}: tools[{index}] 字段必须严格为 name/parameters")
        name = tool.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise GateRunnerError(f"{task_id}: tools[{index}] 名称为空或重复")
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": _json_schema_for_parameters(tool.get("parameters"), task_id, name),
                },
            }
        )
        names.add(name)
    return result


def build_request(task: dict[str, Any], model: str, seed: int) -> dict[str, Any]:
    task_id = task["id"]
    model_input = task["input"]
    temperature = model_input.get("temperature")
    max_tokens = model_input.get("max_output_tokens")
    if temperature != 0:
        raise GateRunnerError(f"{task_id}: temperature 必须为 0")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 256:
        raise GateRunnerError(f"{task_id}: max_output_tokens 必须是 1..256 的整数")
    request_body: dict[str, Any] = {
        "model": model,
        "messages": _copy_messages(model_input.get("messages"), task_id),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
    }
    if "tools" in model_input:
        request_body["tools"] = _convert_tools(model_input["tools"], task_id)
    return request_body


def _integer_usage(usage: Any, key: str) -> int:
    if not isinstance(usage, dict):
        return 0
    value = usage.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise GateRunnerError("响应 message.tool_calls 不是数组")
    calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict) or not isinstance(raw_call.get("function"), dict):
            raise GateRunnerError(f"响应 tool_calls[{index}] 缺少 function")
        function = raw_call["function"]
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise GateRunnerError(f"响应 tool_calls[{index}] 缺少函数名")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise GateRunnerError(f"响应 tool_calls[{index}] arguments 不是合法 JSON") from error
        if not isinstance(arguments, dict):
            raise GateRunnerError(f"响应 tool_calls[{index}] arguments 必须是对象")
        calls.append({"name": name, "arguments": arguments})
    return calls


def response_record(
    payload: Any,
    *,
    run_id: str,
    model: str,
    task_id: str,
    seed: int,
    latency_ms: float,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GateRunnerError("端点响应必须是 JSON 对象")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise GateRunnerError("端点响应缺少 choices[0]")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise GateRunnerError("端点响应缺少 choices[0].message")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise GateRunnerError("响应 message.content 必须是字符串或 null")
    calls = _parse_tool_calls(message)
    finish_reason = choice.get("finish_reason")
    if finish_reason not in {"stop", "tool_calls", "length"}:
        raise GateRunnerError(f"响应 finish_reason 不受支持: {finish_reason!r}")
    usage = payload.get("usage")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": model,
        "task_id": task_id,
        "output": content,
        "tool_calls": calls,
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": _integer_usage(usage, "prompt_tokens"),
            "completion_tokens": _integer_usage(usage, "completion_tokens"),
        },
        "latency_ms": round(latency_ms, 3),
        "generation": {
            "temperature": 0,
            "seed": seed,
            "template_id": TEMPLATE_ID,
            "runtime_id": "openai-compatible-http-v1",
        },
    }


def error_record(
    error: Exception,
    *,
    run_id: str,
    model: str,
    task_id: str,
    seed: int,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": model,
        "task_id": task_id,
        "output": None,
        "tool_calls": [],
        "finish_reason": "error",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "latency_ms": round(latency_ms, 3),
        "generation": {
            "temperature": 0,
            "seed": seed,
            "template_id": TEMPLATE_ID,
            "runtime_id": "openai-compatible-http-v1",
        },
        "error": f"{type(error).__name__}: {error}",
    }


def post_json(url: str, body: dict[str, Any], timeout: float) -> Any:
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read(2048).decode("utf-8", errors="replace")
        raise GateRunnerError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise GateRunnerError(f"请求失败: {error.reason}") from error
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateRunnerError("端点返回的不是合法 UTF-8 JSON") from error


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise GateRunnerError(f"输出文件已存在，拒绝覆盖冻结运行: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_tasks(
    *,
    tasks_path: Path,
    endpoint: str,
    model: str,
    run_id: str,
    output_path: Path,
    seed: int,
    timeout: float,
) -> tuple[int, int]:
    if not model:
        raise GateRunnerError("--model 不得为空")
    if not run_id:
        raise GateRunnerError("--run-id 不得为空")
    if timeout <= 0:
        raise GateRunnerError("--timeout 必须大于 0")
    if tasks_path.resolve() == output_path.resolve():
        raise GateRunnerError("--output 不得覆盖 --tasks")

    tasks = read_task_rows(tasks_path)
    url = normalize_endpoint(endpoint)
    rows: list[dict[str, Any]] = []
    failures = 0
    for task in tasks:
        started = time.perf_counter()
        try:
            request_body = build_request(task, model, seed)
            payload = post_json(url, request_body, timeout)
            elapsed = (time.perf_counter() - started) * 1000
            row = response_record(
                payload,
                run_id=run_id,
                model=model,
                task_id=task["id"],
                seed=seed,
                latency_ms=elapsed,
            )
        except Exception as error:  # 每题都必须留下 schema 合法的失败记录。
            elapsed = (time.perf_counter() - started) * 1000
            row = error_record(
                error,
                run_id=run_id,
                model=model,
                task_id=task["id"],
                seed=seed,
                latency_ms=elapsed,
            )
            failures += 1
        rows.append(row)
    write_jsonl_atomic(output_path, rows)
    return len(rows), failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path, help="capability-task-v1 JSONL")
    parser.add_argument("--endpoint", required=True, help="服务根地址、/v1 或完整 chat/completions URL")
    parser.add_argument("--model", required=True, help="请求与响应记录使用的模型 ID")
    parser.add_argument("--run-id", required=True, help="本次冻结运行的唯一 ID")
    parser.add_argument("--output", required=True, type=Path, help="新建的 capability-response-v1 JSONL")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=float, default=120.0, help="每个 HTTP 请求的超时秒数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        total, failures = run_tasks(
            tasks_path=args.tasks,
            endpoint=args.endpoint,
            model=args.model,
            run_id=args.run_id,
            output_path=args.output,
            seed=args.seed,
            timeout=args.timeout,
        )
    except GateRunnerError as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"ok": failures == 0, "responses": total, "request_errors": failures, "output": str(args.output.resolve())},
            ensure_ascii=False,
        )
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
