"""Start the persistent Neural Alloy server for Claude Code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


PORT = 8094
ALIAS = "ColorLM-v6-Q3-Fused"
BASE_URL = f"http://127.0.0.1:{PORT}"
CPU_MOE_LAYERS = 29


def server_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as response:
            if json.load(response).get("status") != "ok":
                return False
        with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=2) as response:
            models = json.load(response).get("data", [])
        return any(model.get("id") == ALIAS for model in models)
    except Exception:
        return False


def main() -> int:
    if server_ready():
        return 0

    root = Path(__file__).resolve().parent.parent
    runtime = root / "fast16" / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    server = root / "build" / "bin" / "Release" / "llama-server.exe"
    model = root / "fast16" / "models" / "ColorLM-v6-Q3Router-Fused-A1.gguf"

    for path, label in ((server, "llama-server"), (model, "异构模型")):
        if not path.is_file():
            print(f"找不到{label}: {path}", file=sys.stderr)
            return 1

    args = [
        str(server),
        "--model",
        "fast16/models/ColorLM-v6-Q3Router-Fused-A1.gguf",
        "--alias",
        ALIAS,
        "--n-gpu-layers",
        "99",
        "--n-cpu-moe",
        str(CPU_MOE_LAYERS),
        "--ctx-size",
        "32768",
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--no-mmap",
        "--cache-ram",
        "0",
        "--spec-type",
        "ngram-mod",
        "--spec-ngram-mod-n-match",
        "16",
        "--spec-ngram-mod-n-min",
        "4",
        "--spec-ngram-mod-n-max",
        "16",
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
        str(PORT),
    ]

    stdout = (runtime / "neural-alloy-claude.stdout.log").open("ab")
    stderr = (runtime / "neural-alloy-claude.stderr.log").open("ab")
    environment = os.environ.copy()
    # The q3 router delta is already materialized in this checkpoint. Never
    # apply the experimental runtime container a second time.
    environment.pop("COLORLM_ALLOY_Q3_PATH", None)
    environment.pop("COLORLM_ALLOY_ALPHA", None)
    # Keep this endpoint focused on the compiled neural alloy checkpoint.
    environment["COLORLM_V4_KERNEL_LAYERS"] = "0"
    environment["COLORLM_V4_RECURRENCE_ALPHA"] = "0"
    environment["COLORLM_V4_COGNITIVE_ROUNDS"] = "0"
    environment["COLORLM_V4_SEMANTIC_ALPHA"] = "0"
    environment["COLORLM_V4_PLASTIC_RANK"] = "0"

    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    subprocess.Popen(
        args,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        env=environment,
        creationflags=creation_flags,
        close_fds=True,
    )
    stdout.close()
    stderr.close()

    for _ in range(360):
        if server_ready():
            print(f"Neural Alloy Claude 服务已就绪: {BASE_URL}")
            return 0
        time.sleep(0.5)

    print(
        "Neural Alloy Claude 服务启动失败，请查看 "
        "fast16/runtime/neural-alloy-claude.stderr.log",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
