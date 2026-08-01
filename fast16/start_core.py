"""Start the persistent llama.cpp Vulkan server with UTF-8-safe paths."""

from __future__ import annotations

import json
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
        return 0

    root = Path(__file__).resolve().parent.parent
    runtime = root / "fast16" / "runtime"
    (runtime / "kv").mkdir(parents=True, exist_ok=True)
    server = root / "build" / "bin" / "Release" / "llama-server.exe"
    args = [
        str(server),
        "-m",
        "fast16/models/seed/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "-ngl",
        "99",
        "-c",
        "4096",
        "--jinja",
        "--slot-save-path",
        "fast16/runtime/kv",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]
    stdout = (runtime / "core-v3.stdout.log").open("ab")
    stderr = (runtime / "core-v3.stderr.log").open("ab")
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    subprocess.Popen(
        args,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        creationflags=creation_flags,
        close_fds=True,
    )
    stdout.close()
    stderr.close()
    for _ in range(120):
        if healthy():
            return 0
        time.sleep(0.5)
    print("ColorLM core failed to start. See fast16/runtime/core-v3.stderr.log", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
