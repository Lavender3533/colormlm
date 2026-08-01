"""顺序启动 v6/v31，执行冻结八维短门和一次相邻速度检查。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PARALLEL_B = ROOT / "fast16" / "research" / "parallel_b"
sys.path.insert(0, os.fspath(PARALLEL_B))

from validate_multicap_gate import grade_row  # noqa: E402


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(description="运行 v31 独立短门")
    parser.add_argument("--port", type=int, default=8131)
    parser.add_argument("--tasks", type=Path, default=HERE / "v31_gate.json")
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
        "--baseline",
        type=Path,
        default=models / "ColorLM-v6-Q3Router-Fused-A1.gguf",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=models / "ColorLM-v31-Qwen36-L39-Expert-Pair.gguf",
    )
    parser.add_argument(
        "--candidate-alias",
        default="ColorLM-v31-Qwen36-L39-Pair",
    )
    parser.add_argument("--output", type=Path, default=HERE / "gate_report.json")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def ready(port: int, alias: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            if json.load(response).get("status") != "ok":
                return False
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
            models = json.load(response).get("data", [])
        return any(item.get("id") == alias for item in models)
    except Exception:
        return False


def clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("COLORLM_") or name.startswith("GGML_SCHED_"):
            environment.pop(name, None)
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "GGML_SCHED_MERGE_CPU_SYNC": "1",
            "GGML_SCHED_SKIP_CPU_FINAL_SYNC": "1",
            "GGML_SCHED_BATCH_CPU_READ": "1",
            "COLORLM_V4_KERNEL_LAYERS": "0",
            "COLORLM_V4_RECURRENCE_ALPHA": "0",
            "COLORLM_V4_COGNITIVE_ROUNDS": "0",
            "COLORLM_V4_SEMANTIC_ALPHA": "0",
            "COLORLM_V4_PLASTIC_RANK": "0",
        }
    )
    environment.pop("GGML_VK_SPIN_FENCE", None)
    return environment


def start_server(
    server: Path,
    model: Path,
    alias: str,
    port: int,
    extra_environment: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> tuple[subprocess.Popen[bytes], float]:
    try:
        model_argument = os.fspath(model.relative_to(ROOT))
    except ValueError:
        model_argument = os.fspath(model)
    command = [
        os.fspath(server),
        "--model",
        model_argument,
        "--alias",
        alias,
        "--n-gpu-layers",
        "99",
        "--n-cpu-moe",
        "29",
        "--threads",
        "8",
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--no-mmap",
        "--cache-ram",
        "0",
        "--ctx-checkpoints",
        "0",
        "--spec-type",
        "none",
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
    if extra_args:
        command.extend(extra_args)
    logs = HERE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / f"{alias}.stdout.log").open("wb")
    stderr = (logs / f"{alias}.stderr.log").open("wb")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    started = time.perf_counter()
    environment = clean_environment()
    if extra_environment:
        environment.update(extra_environment)
    process = subprocess.Popen(
        command,
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
            tail = (logs / f"{alias}.stderr.log").read_text(
                encoding="utf-8", errors="replace"
            )[-4000:]
            raise RuntimeError(f"服务提前退出 {alias}: {process.returncode}\n{tail}")
        if ready(port, alias):
            return process, time.perf_counter() - started
        time.sleep(0.5)
    process.kill()
    raise RuntimeError(f"服务启动超时: {alias}")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=12)


def tool_schema(task: dict[str, Any]) -> list[dict[str, Any]] | None:
    validator = task["validator"]
    if validator["type"] != "tool_call":
        return None
    properties = {
        key: {"type": "string"} for key in validator["expected_arguments"]
    }
    return [
        {
            "type": "function",
            "function": {
                "name": validator["expected_name"],
                "description": task["prompt"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
        }
    ]


def post_json(port: int, payload: dict[str, Any], timeout: float = 180) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def response_row(task: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    message = choice["message"]
    row = {
        "id": task["id"],
        "output": message.get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage", {}),
        "timings": response.get("timings", {}),
    }
    if message.get("tool_calls"):
        row["tool_calls"] = message["tool_calls"]
    passed, detail = grade_row(task, row)
    row["passed"] = passed
    row["detail"] = detail
    return row


def run_tasks(port: int, alias: str, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, task in enumerate(tasks, 1):
        payload: dict[str, Any] = {
            "model": alias,
            "messages": [
                {
                    "role": "system",
                    "content": "严格遵守输出格式；不要解释，不要添加Markdown。",
                },
                {"role": "user", "content": task["prompt"]},
            ],
            "temperature": 0,
            "seed": 7319,
            "max_tokens": task["max_output_tokens"],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        tools = tool_schema(task)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required"
        row = response_row(task, post_json(port, payload))
        rows.append(row)
        print(
            f"  [{index}/{len(tasks)}] {task['id']}: "
            f"{'PASS' if row['passed'] else 'FAIL'}",
            flush=True,
        )
    return rows


def run_speed(port: int, alias: str) -> dict[str, Any]:
    payload = {
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
    response = post_json(port, payload)
    choice = response["choices"][0]
    return {
        "output": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage", {}),
        "timings": response.get("timings", {}),
    }


def main() -> int:
    args = parse_args()
    for path in (args.tasks, args.server, args.baseline, args.candidate):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")

    bank = json.loads(args.tasks.read_text(encoding="utf-8"))
    if bank.get("frozen_before_generation") is not True:
        raise RuntimeError("短门没有冻结标记")
    tasks = bank["items"]
    contract_sha256 = sha256_file(args.tasks)
    report: dict[str, Any] = {
        "format": "colormlm-qwen36-candidate-short-gate-report-v1",
        "contract": args.tasks.as_posix(),
        "contract_sha256": contract_sha256,
        "models": [],
    }

    variants = [
        ("ColorLM-v6-v31-control", args.baseline),
        (args.candidate_alias, args.candidate),
    ]
    for alias, model in variants:
        print(f"启动 {alias}", flush=True)
        process, load_seconds = start_server(args.server, model, alias, args.port)
        try:
            rows = run_tasks(args.port, alias, tasks)
            speed = run_speed(args.port, alias)
            item = {
                "alias": alias,
                "model": model.name,
                "model_bytes": model.stat().st_size,
                "load_seconds": load_seconds,
                "passed": sum(bool(row["passed"]) for row in rows),
                "total": len(rows),
                "responses": rows,
                "speed": speed,
            }
            report["models"].append(item)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        finally:
            stop_server(process)
        time.sleep(1)

    base, candidate = report["models"]
    base_by_id = {row["id"]: row for row in base["responses"]}
    candidate_by_id = {row["id"]: row for row in candidate["responses"]}
    wins = [
        task_id
        for task_id in base_by_id
        if not base_by_id[task_id]["passed"] and candidate_by_id[task_id]["passed"]
    ]
    regressions = [
        task_id
        for task_id in base_by_id
        if base_by_id[task_id]["passed"] and not candidate_by_id[task_id]["passed"]
    ]
    base_tps = float(base["speed"]["timings"].get("predicted_per_second", 0.0))
    candidate_tps = float(
        candidate["speed"]["timings"].get("predicted_per_second", 0.0)
    )
    speed_delta = candidate_tps / base_tps - 1.0 if base_tps else None
    report["comparison"] = {
        "wins": wins,
        "regressions": regressions,
        "score_delta": candidate["passed"] - base["passed"],
        "base_tokens_per_second": base_tps,
        "candidate_tokens_per_second": candidate_tps,
        "speed_delta": speed_delta,
        "generation_gate_pass": bool(wins and not regressions),
        "speed_gate_pass": bool(speed_delta is not None and speed_delta >= -0.05),
        "prototype_advances": bool(
            wins and not regressions and speed_delta is not None and speed_delta >= -0.05
        ),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    print(f"报告: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
