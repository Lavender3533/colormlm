"""Start the unmodified GLM-4.7-Flash baseline on the Vulkan runtime.

This launcher deliberately excludes all ColorLM memory and cognition features.
It exists to establish the coding baseline that later ColorLM changes must beat.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


PORT = 8093
HEALTH = f"http://127.0.0.1:{PORT}/health"

# Conservative initial split for an 8 GiB GPU. Tune only after the first
# successful load and a VRAM reading; model quality is independent of the split.
CPU_MOE_LAYERS = 38


def healthy() -> bool:
    try:
        with urllib.request.urlopen(HEALTH, timeout=2) as response:
            return json.load(response).get("status") == "ok"
    except Exception:
        return False


def main() -> int:
    if healthy():
        print(f"GLM-4.7-Flash 已在 http://127.0.0.1:{PORT} 运行")
        return 0

    root = Path(__file__).resolve().parent.parent
    runtime = root / "fast16" / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    server = root / "build" / "bin" / "Release" / "llama-server.exe"
    model = root / "fast16" / "models" / "GLM-4.7-Flash-Q5_K_M.gguf"

    if not server.is_file():
        print(f"推理程序不存在: {server}", file=sys.stderr)
        return 1
    if not model.is_file():
        print(f"模型尚未下载完成: {model}", file=sys.stderr)
        return 1

    print("预热模型页缓存...", flush=True)
    with model.open("rb") as fh:
        while fh.read(64 * 1024 * 1024):
            pass

    args = [
        str(server),
        "--model",
        "fast16/models/GLM-4.7-Flash-Q5_K_M.gguf",
        "--alias",
        "GLM-4.7-Flash",
        "--n-gpu-layers",
        "99",
        "--n-cpu-moe",
        str(CPU_MOE_LAYERS),
        "--ctx-size",
        "32768",
        "--parallel",
        "1",
        "--cache-ram",
        "0",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--jinja",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]

    env = dict(os.environ)
    env.setdefault("GGML_VK_SPIN_FENCE", "1")

    stdout = (runtime / "glm47-server.stdout.log").open("ab")
    stderr = (runtime / "glm47-server.stderr.log").open("ab")
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
            print(f"GLM-4.7-Flash 已启动: http://127.0.0.1:{PORT}")
            return 0
        time.sleep(0.5)

    print(
        "GLM-4.7-Flash 启动失败，查看 fast16/runtime/glm47-server.stderr.log",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
