"""隔离启动Fara1.5-27B文本+视觉教师服务。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "llama.cpp/build-v17-perf/bin/Release/llama-server.exe"
MODEL = ROOT / "fast16/models/donor/fara15_27b/Fara1.5-27B-Q5_K_M.gguf"
MMPROJ = ROOT / "fast16/models/donor/fara15_27b/mmproj-Fara1.5-27B-f16.gguf"
RUNTIME = ROOT / "fast16/runtime"
ALIAS = "Fara1.5-27B-v23-Teacher"
EXPECTED = {
    MODEL: 20_513_802_048,
    MMPROJ: 927_607_456,
}


def relative_native(path: Path) -> str:
    return os.fspath(path.relative_to(ROOT))


def read_json(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"响应不是JSON对象: {url}")
    return value


def ready(port: int) -> bool:
    try:
        health = read_json(f"http://127.0.0.1:{port}/health")
        models = read_json(f"http://127.0.0.1:{port}/v1/models")
        ids = {
            str(item.get("id"))
            for item in models.get("data", [])
            if isinstance(item, dict)
        }
        return health.get("status") == "ok" and ALIAS in ids
    except Exception:
        return False


def port_occupied(port: int) -> bool:
    try:
        read_json(f"http://127.0.0.1:{port}/health")
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="启动Fara v23隔离教师服务")
    parser.add_argument("--port", type=int, default=8125)
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--startup-timeout", type=float, default=420.0)
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.ctx_size < 4096 or args.threads <= 0:
        print("端口、上下文或线程参数无效", file=sys.stderr)
        return 2

    if ready(args.port):
        print(f"Fara教师已在线: http://127.0.0.1:{args.port}/v1")
        return 0
    if port_occupied(args.port):
        print(f"端口{args.port}已被其他服务占用，拒绝覆盖", file=sys.stderr)
        return 1
    if not SERVER.is_file():
        print(f"缺少推理程序: {SERVER}", file=sys.stderr)
        return 1
    for path, expected_size in EXPECTED.items():
        if not path.is_file():
            print(f"缺少文件: {path}", file=sys.stderr)
            return 1
        if path.stat().st_size != expected_size:
            print(
                f"文件大小不符: {path}，实际={path.stat().st_size}，预期={expected_size}",
                file=sys.stderr,
            )
            return 1

    RUNTIME.mkdir(parents=True, exist_ok=True)
    command = [
        str(SERVER),
        "--model", relative_native(MODEL),
        "--mmproj", relative_native(MMPROJ),
        "--alias", ALIAS,
        "--ctx-size", str(args.ctx_size),
        "--parallel", "1",
        "--threads", str(args.threads),
        "--threads-batch", str(args.threads),
        "--batch-size", "512",
        "--ubatch-size", "256",
        "--cache-ram", "0",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--jinja",
        "--reasoning", "off",
        "--fit", "on",
        "--fit-target", "1024",
        "--no-warmup",
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "GGML_SCHED_MERGE_CPU_SYNC": "1",
            "GGML_SCHED_SKIP_CPU_FINAL_SYNC": "1",
            "GGML_SCHED_BATCH_CPU_READ": "1",
        }
    )
    environment.pop("GGML_VK_SPIN_FENCE", None)
    for key in list(environment):
        if key.startswith("COLORLM_"):
            environment.pop(key, None)

    stdout_path = RUNTIME / "fara15-v23.stdout.log"
    stderr_path = RUNTIME / "fara15-v23.stderr.log"
    stdout = stdout_path.open("ab")
    stderr = stderr_path.open("ab")
    creation_flags = 0
    if sys.platform == "win32" and not args.foreground:
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdin=None if args.foreground else subprocess.DEVNULL,
        stdout=None if args.foreground else stdout,
        stderr=None if args.foreground else stderr,
        creationflags=creation_flags,
        close_fds=not args.foreground,
    )
    stdout.close()
    stderr.close()
    if args.foreground:
        return process.wait()

    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(
                f"Fara提前退出，退出码={process.returncode}；查看 {stderr_path}",
                file=sys.stderr,
            )
            return 1
        if ready(args.port):
            print(f"Fara教师已启动: http://127.0.0.1:{args.port}/v1")
            return 0
        time.sleep(0.5)
    print(f"Fara启动超时；查看 {stderr_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
