"""Start the bare Qwen3.6-35B-A3B donor server (no ColorLM machinery).

Baseline arm for the donor swap: same speed stack as start_v4.py but zero
ColorLM env vars, so eval scores/speed measure the naked donor.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


PORT = 8092
HEALTH = f"http://127.0.0.1:{PORT}/health"

# Q4_K_M routed experts are ~1.7x larger per layer than the v4 custom mix, so
# more layers must live in RAM to keep ~1 GiB of VRAM free.
CPU_MOE_LAYERS = 34


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
    runtime.mkdir(parents=True, exist_ok=True)
    server = root / "build" / "bin" / "Release" / "llama-server.exe"
    model = root / "fast16" / "models" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    if not model.is_file():
        print(f"模型不存在: {model}", file=sys.stderr)
        return 1

    # Pre-warm: one sequential pass primes the OS page cache (~15 s at
    # 1.6 GB/s) so mmap page faults hit RAM instead of 4K random SSD reads.
    print("预热页缓存...", flush=True)
    with model.open("rb") as fh:
        while fh.read(64 * 1024 * 1024):
            pass

    args = [
        str(server),
        "--model",
        "fast16/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "--alias",
        "Qwen3.6-35B-A3B",
        "--n-gpu-layers",
        "99",
        "--n-cpu-moe",
        str(CPU_MOE_LAYERS),
        # 32K so Claude Code's system prompt + tools fit; hybrid linear
        # attention keeps the KV cost of this tiny.
        "--ctx-size",
        "32768",
        "--parallel",
        "1",
        # NOTE: --no-mmap fails here ("unable to allocate Vulkan_Host buffer"):
        # the 21 GB file needs ~16 GB of pinned host memory in one shot, which
        # the AMD Windows driver refuses. Instead we mmap (default) and pre-warm
        # the OS page cache with one sequential read before boot — same effect
        # (no cold 4K random reads), no pinned allocation.
        "--cache-ram",
        "0",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--jinja",
        # Draft-free self-speculation, same tuning as v4.
        "--spec-type",
        "ngram-mod",
        "--spec-ngram-mod-n-match",
        "16",
        "--spec-ngram-mod-n-min",
        "4",
        "--spec-ngram-mod-n-max",
        "16",
        # NOTE: --poll 100 was A/B'd 2026-07-26: no gain over the default, reverted.
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]
    env = dict(os.environ)
    # A/B verified 2026-07-26 (fast16/SYNC_OVERHEAD_AUDIT.md): skipping the WDDM
    # kernel fence wait is worth ~14 ms/token at decode (82.7 -> 68.5 ms/tok).
    env.setdefault("GGML_VK_SPIN_FENCE", "1")

    stdout = (runtime / "qwen36-server.stdout.log").open("ab")
    stderr = (runtime / "qwen36-server.stderr.log").open("ab")
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
            return 0
        time.sleep(0.5)
    print("Qwen3.6 启动失败, 查看 fast16/runtime/qwen36-server.stderr.log", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
