"""Run a small deterministic code and tool-use decision benchmark."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(__file__).resolve().parent
HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/"
    "data/HumanEval.jsonl.gz"
)
SELECTED_HUMANEVAL = [0, 2, 8, 11, 20, 29, 40, 62]
LOCAL_CODE8 = [
    {
        "task_id": "Code8/close_elements",
        "entry_point": "has_close_elements",
        "prompt": """from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Return True when two distinct numbers are closer than threshold.\"\"\"\n""",
        "test": """def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) is False\n    assert candidate([1.0, 2.0, 2.2], 0.5) is True\n    assert candidate([], 1.0) is False\n    assert candidate([1.0, 1.0], 0.01) is True\n""",
    },
    {
        "task_id": "Code8/paren_groups",
        "entry_point": "separate_paren_groups",
        "prompt": """from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"Split a string of balanced parenthesis groups; ignore spaces.\"\"\"\n""",
        "test": """def check(candidate):\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n    assert candidate('()()') == ['()', '()']\n    assert candidate('((()))') == ['((()))']\n    assert candidate('') == []\n""",
    },
    {
        "task_id": "Code8/truncate",
        "entry_point": "truncate_number",
        "prompt": """def truncate_number(number: float) -> float:\n    \"\"\"Return the fractional part of a positive number.\"\"\"\n""",
        "test": """def check(candidate):\n    assert abs(candidate(3.5) - 0.5) < 1e-9\n    assert abs(candidate(1.25) - 0.25) < 1e-9\n    assert abs(candidate(7.0)) < 1e-9\n    assert abs(candidate(123.875) - 0.875) < 1e-9\n""",
    },
    {
        "task_id": "Code8/below_zero",
        "entry_point": "below_zero",
        "prompt": """from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\"Return whether a running account balance ever becomes negative.\"\"\"\n""",
        "test": """def check(candidate):\n    assert candidate([1, 2, -4, 5]) is True\n    assert candidate([1, 2, -3, 1]) is False\n    assert candidate([]) is False\n    assert candidate([-1]) is True\n""",
    },
    {
        "task_id": "Code8/intersperse",
        "entry_point": "intersperse",
        "prompt": """from typing import List\n\ndef intersperse(numbers: List[int], delimiter: int) -> List[int]:\n    \"\"\"Insert delimiter between every two adjacent input values.\"\"\"\n""",
        "test": """def check(candidate):\n    assert candidate([], 4) == []\n    assert candidate([1], 4) == [1]\n    assert candidate([1, 2, 3], 4) == [1, 4, 2, 4, 3]\n    assert candidate([0, 0], 9) == [0, 9, 0]\n""",
    },
    {
        "task_id": "Code8/rolling_max",
        "entry_point": "rolling_max",
        "prompt": """from typing import List\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\"Return the maximum value seen up to and including each position.\"\"\"\n""",
        "test": """def check(candidate):\n    assert candidate([]) == []\n    assert candidate([1, 2, 3, 2, 1]) == [1, 2, 3, 3, 3]\n    assert candidate([5, 4, 6]) == [5, 5, 6]\n    assert candidate([-3, -5, -2]) == [-3, -3, -2]\n""",
    },
    {
        "task_id": "Code8/string_xor",
        "entry_point": "string_xor",
        "prompt": """def string_xor(a: str, b: str) -> str:\n    \"\"\"XOR two equal-length strings containing only '0' and '1'.\"\"\"\n""",
        "test": """def check(candidate):\n    assert candidate('010', '110') == '100'\n    assert candidate('1111', '0000') == '1111'\n    assert candidate('', '') == ''\n    assert candidate('10101', '00111') == '10010'\n""",
    },
    {
        "task_id": "Code8/longest",
        "entry_point": "longest",
        "prompt": """from typing import List, Optional\n\ndef longest(strings: List[str]) -> Optional[str]:\n    \"\"\"Return the first longest string, or None for an empty list.\"\"\"\n""",
        "test": """def check(candidate):\n    assert candidate([]) is None\n    assert candidate(['a', 'bb', 'ccc']) == 'ccc'\n    assert candidate(['same', 'size', 'tiny']) == 'same'\n    assert candidate(['']) == ''\n""",
    },
]
ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "re",
    "statistics",
    "string",
    "typing",
}
BLOCKED_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
    "__import__",
}
BLOCKED_ATTRIBUTES = {
    "chmod",
    "connect",
    "kill",
    "popen",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "system",
    "unlink",
}

TOOL_TASKS = [
    {
        "id": "weather",
        "prompt": "查询杭州当前摄氏温度。必须使用工具，不要用文字猜测。",
        "tools": [
            {
                "name": "get_weather",
                "description": "查询指定城市天气",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                },
            },
            {
                "name": "get_time",
                "description": "查询指定城市时间",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        ],
        "expected_name": "get_weather",
        "expected_input": {"location": "杭州", "unit": "celsius"},
    },
    {
        "id": "read_file",
        "prompt": "读取文件 src/main.rs。必须调用工具。",
        "tools": [
            {
                "name": "read_file",
                "description": "读取文本文件",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "search_files",
                "description": "按模式搜索文件",
                "input_schema": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
        ],
        "expected_name": "read_file",
        "expected_input": {"path": "src/main.rs"},
    },
    {
        "id": "search_java",
        "prompt": "搜索 src 目录下所有 Java 文件。必须调用工具。",
        "tools": [
            {
                "name": "glob_files",
                "description": "按glob模式列出文件",
                "input_schema": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
            {
                "name": "read_file",
                "description": "读取一个文件",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        ],
        "expected_name": "glob_files",
        "expected_input": {"pattern": "src/**/*.java"},
    },
    {
        "id": "calculate",
        "prompt": "用计算器计算 17 乘以 23。必须调用工具。",
        "tools": [
            {
                "name": "calculator",
                "description": "执行基础算术",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                        },
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            }
        ],
        "expected_name": "calculator",
        "expected_input": {"operation": "multiply", "a": 17, "b": 23},
    },
]


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(description="ColorLM最小决策基准")
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("ALIAS", "PATH"),
        default=None,
        help="可重复传入；默认比较v6和v8",
    )
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESEARCH / "minimal_decision_benchmark_report.json",
    )
    parser.add_argument("--skip-code", action="store_true")
    parser.add_argument("--skip-tools", action="store_true")
    args = parser.parse_args()
    if args.model is None:
        args.model = [
            ("ColorLM-v6-Q3-Fused", os.fspath(models / "ColorLM-v6-Q3Router-Fused-A1.gguf")),
            (
                "ColorLM-v8-CoderNext-Transport-E471",
                os.fspath(models / "ColorLM-v8-CoderNext-Transport-E471.gguf"),
            ),
        ]
    return args


def load_code_tasks(cache: Path) -> tuple[list[dict], str]:
    if not cache.is_file():
        try:
            response = requests.get(HUMANEVAL_URL, timeout=(10, 30))
            response.raise_for_status()
            if len(response.content) > 1024 * 1024:
                raise RuntimeError("HumanEval数据异常地超过1MiB")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(response.content)
        except requests.RequestException:
            return LOCAL_CODE8, "ColorLM-Code8-v1 (HumanEval-style fallback)"
    wanted = {f"HumanEval/{index}" for index in SELECTED_HUMANEVAL}
    with gzip.open(cache, "rt", encoding="utf-8") as stream:
        tasks = [json.loads(line) for line in stream if json.loads(line)["task_id"] in wanted]
    by_id = {task["task_id"]: task for task in tasks}
    return [by_id[f"HumanEval/{index}"] for index in SELECTED_HUMANEVAL], "HumanEval-fixed8"


def server_ready(port: int, alias: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            if json.load(response).get("status") != "ok":
                return False
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
            models = json.load(response).get("data", [])
        return any(model.get("id") == alias for model in models)
    except Exception:
        return False


def start_server(model: Path, alias: str, port: int, log_dir: Path) -> subprocess.Popen:
    server = ROOT / "build" / "bin" / "Release" / "llama-server.exe"
    try:
        model_argument = os.fspath(model.relative_to(ROOT))
    except ValueError:
        model_argument = os.fspath(model)
    args = [
        os.fspath(server),
        "--model",
        model_argument,
        "--alias",
        alias,
        "--n-gpu-layers",
        "99",
        "--n-cpu-moe",
        "29",
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--batch-size",
        "256",
        "--ubatch-size",
        "256",
        "--no-mmap",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--jinja",
        "--reasoning",
        "off",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "COLORLM_V4_KERNEL_LAYERS": "0",
            "COLORLM_V4_RECURRENCE_ALPHA": "0",
            "COLORLM_V4_COGNITIVE_ROUNDS": "0",
            "COLORLM_V4_SEMANTIC_ALPHA": "0",
            "COLORLM_V4_PLASTIC_RANK": "0",
        }
    )
    environment.pop("COLORLM_ALLOY_Q3_PATH", None)
    environment.pop("COLORLM_ALLOY_ALPHA", None)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / f"{alias}.stdout.log").open("wb")
    stderr = (log_dir / f"{alias}.stderr.log").open("wb")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(
        args,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        env=environment,
        creationflags=flags,
    )
    stdout.close()
    stderr.close()
    for _ in range(360):
        if process.poll() is not None:
            raise RuntimeError(f"服务提前退出: {alias}: {process.returncode}")
        if server_ready(port, alias):
            return process
        time.sleep(0.5)
    process.kill()
    raise RuntimeError(f"服务启动超时: {alias}")


def post_json(port: int, path: str, payload: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": "local-benchmark",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def extract_text(response: dict) -> str:
    return "".join(
        item.get("text", "")
        for item in response.get("content", [])
        if item.get("type") == "text"
    )


def extract_python(text: str, task: dict) -> str:
    fences = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    code = fences[0].strip() if fences else text.strip()
    marker = f"def {task['entry_point']}"
    if marker in code:
        prompt_prefix = task["prompt"][: task["prompt"].index(marker)]
        code = prompt_prefix + code[code.index(marker) :]
    else:
        code = task["prompt"] + code
    return code


def validate_candidate(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in ALLOWED_IMPORTS for alias in node.names):
                raise ValueError("包含非白名单import")
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError("包含非白名单from import")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                raise ValueError(f"包含危险调用: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_ATTRIBUTES:
                raise ValueError(f"包含危险属性调用: {node.func.attr}")


def judge_code(source: str, task: dict) -> tuple[bool, str]:
    try:
        validate_candidate(source)
    except Exception as error:
        return False, f"安全/语法拒绝: {error}"
    program = source + "\n" + task["test"] + f"\ncheck({task['entry_point']})\n"
    try:
        environment = os.environ.copy()
        environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        result = subprocess.run(
            [sys.executable, "-I", "-c", program],
            cwd=RESEARCH,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return False, "执行超时"
    detail = (result.stderr or result.stdout).strip()[-1000:]
    return result.returncode == 0, detail


def run_code_tasks(port: int, alias: str, tasks: list[dict]) -> list[dict]:
    results = []
    for task in tasks:
        prompt = (
            "完成下面的Python函数。只输出完整函数代码，不要Markdown，不要解释。\n\n"
            + task["prompt"]
        )
        started = time.perf_counter()
        response = post_json(
            port,
            "/v1/messages",
            {
                "model": alias,
                "max_tokens": 256,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        elapsed = time.perf_counter() - started
        text = extract_text(response)
        source = extract_python(text, task)
        passed, detail = judge_code(source, task)
        results.append(
            {
                "id": task["task_id"],
                "passed": passed,
                "seconds": elapsed,
                "output_tokens": response.get("usage", {}).get("output_tokens"),
                "detail": detail,
                "response": text,
            }
        )
        print(f"  code {task['task_id']}: {'PASS' if passed else 'FAIL'}", flush=True)
    return results


def run_tool_tasks(port: int, alias: str) -> list[dict]:
    results = []
    for task in TOOL_TASKS:
        started = time.perf_counter()
        response = post_json(
            port,
            "/v1/messages",
            {
                "model": alias,
                "max_tokens": 128,
                "temperature": 0,
                "tools": task["tools"],
                "messages": [{"role": "user", "content": task["prompt"]}],
            },
        )
        elapsed = time.perf_counter() - started
        calls = [item for item in response.get("content", []) if item.get("type") == "tool_use"]
        call = calls[0] if len(calls) == 1 else {}
        passed = (
            call.get("name") == task["expected_name"]
            and call.get("input") == task["expected_input"]
        )
        results.append(
            {
                "id": task["id"],
                "passed": passed,
                "seconds": elapsed,
                "output_tokens": response.get("usage", {}).get("output_tokens"),
                "expected": {
                    "name": task["expected_name"],
                    "input": task["expected_input"],
                },
                "actual": call,
                "response": response.get("content", []),
            }
        )
        print(f"  tool {task['id']}: {'PASS' if passed else 'FAIL'}", flush=True)
    return results


def summarize(results: list[dict]) -> dict:
    if not results:
        return {"passed": 0, "total": 0, "mean_seconds": 0.0}
    return {
        "passed": sum(bool(item["passed"]) for item in results),
        "total": len(results),
        "mean_seconds": sum(float(item["seconds"]) for item in results) / len(results),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in results),
    }


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    args = parse_args()
    cache = RESEARCH / "benchmark_cache" / "HumanEval.jsonl.gz"
    if args.skip_code:
        tasks, code_suite = [], "skipped"
    else:
        tasks, code_suite = load_code_tasks(cache)
    report = {
        "format": "colorlm-minimal-decision-benchmark-v1",
        "code_suite": code_suite,
        "code_task_ids": [task["task_id"] for task in tasks],
        "tool_task_ids": [] if args.skip_tools else [item["id"] for item in TOOL_TASKS],
        "models": [],
    }
    log_dir = ROOT / "fast16" / "runtime" / "minimal-benchmark"
    for alias, raw_path in args.model:
        model = Path(raw_path)
        if not model.is_absolute():
            model = ROOT / model
        print(f"启动 {alias}", flush=True)
        process = start_server(model, alias, args.port, log_dir)
        try:
            code = run_code_tasks(args.port, alias, tasks) if tasks else []
            tools = [] if args.skip_tools else run_tool_tasks(args.port, alias)
            item = {
                "alias": alias,
                "model": model.name,
                "code": code,
                "tools": tools,
                "summary": {"code": summarize(code), "tools": summarize(tools)},
            }
            report["models"].append(item)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        finally:
            stop_server(process)
        time.sleep(1)
    print(json.dumps({item["alias"]: item["summary"] for item in report["models"]}, ensure_ascii=False, indent=2))
    print(f"报告: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
