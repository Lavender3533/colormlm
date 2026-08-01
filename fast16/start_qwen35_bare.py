"""启动原生 Qwen3.5-35B-A3B，作为 ColorLM 谱系安全对照。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


PORT = 8091
HEALTH = f"http://127.0.0.1:{PORT}/health"


def healthy() -> bool:
    try:
        with urllib.request.urlopen(HEALTH, timeout=2) as response:
            return json.load(response).get("status") == "ok"
    except Exception:
        return False


def main() -> int:
    if healthy():
        print(f"Qwen3.5 原生基座已在 http://127.0.0.1:{PORT} 运行")
        return 0

    root = Path(__file__).resolve().parent.parent
    runtime = root / "fast16" / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    server = root / "llama.cpp" / "build-v17-perf" / "bin" / "Release" / "llama-server.exe"
    model = (
        root
        / "fast16"
        / "models"
        / "donor"
        / "Qwen3.5-35B-A3B-UD-IQ3_S.gguf"
    )
    if not server.is_file() or not model.is_file():
        print(f"缺少推理程序或模型: {server}, {model}", file=sys.stderr)
        return 1

    print("预热 Qwen3.5 原生基座页缓存...", flush=True)
    with model.open("rb") as handle:
        while handle.read(64 * 1024 * 1024):
            pass

    args = [
        str(server),
        "--model",
        "fast16/models/donor/Qwen3.5-35B-A3B-UD-IQ3_S.gguf",
        "--alias",
        "Qwen3.5-35B-A3B-Bare",
        "--n-gpu-layers",
        "99",
        "--n-cpu-moe",
        "29",
        "--ctx-size",
        "16384",
        "--parallel",
        "1",
        "--threads",
        "8",
        "--threads-batch",
        "8",
        "--cache-ram",
        "0",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--jinja",
        "--spec-type",
        "ngram-mod",
        "--spec-ngram-mod-n-match",
        "16",
        "--spec-ngram-mod-n-min",
        "4",
        "--spec-ngram-mod-n-max",
        "16",
        "--fit",
        "off",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]

    env = dict(os.environ)
    # 与 v17 正式运行时保持相同的已验证调度优化，但清除所有神经岛/胶囊变量。
    env["GGML_SCHED_MERGE_CPU_SYNC"] = "1"
    env["GGML_SCHED_SKIP_CPU_FINAL_SYNC"] = "1"
    env["GGML_SCHED_BATCH_CPU_READ"] = "1"
    for key in list(env):
        if key.startswith("COLORLM_"):
            env.pop(key)

    stdout = (runtime / "qwen35-bare.stdout.log").open("ab")
    stderr = (runtime / "qwen35-bare.stderr.log").open("ab")
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    subprocess.Popen(
        args,
        cwd=root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        creationflags=creation_flags,
        close_fds=True,
    )
    stdout.close()
    stderr.close()

    for _ in range(360):
        if healthy():
            print(f"Qwen3.5 原生基座已启动: http://127.0.0.1:{PORT}")
            return 0
        time.sleep(0.5)
    print("Qwen3.5 启动失败，查看 fast16/runtime/qwen35-bare.stderr.log", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
