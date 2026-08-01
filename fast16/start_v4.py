"""Start the persistent ColorLM v4 Vulkan server."""

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

# ColorLM-v4 has 274 MiB of quantized routed-expert weights per layer. Keeping
# the first 29 layers on the CPU places the final 11 layers on the RX 5700 XT,
# leaves room for the always-on 256-rank ColorPlasticity tensors, and keeps
# ~1.1 GiB of VRAM free so the Vulkan driver never demotes buffers to PCIe.
CPU_MOE_LAYERS = 29


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
    kv_path = runtime / "kv-v4"
    kv_path.mkdir(parents=True, exist_ok=True)
    server = root / "build" / "bin" / "Release" / "llama-server.exe"
    model = root / "fast16" / "models" / "ColorLM-v4-SMoE.gguf"
    if not model.is_file():
        print(f"ColorLM v4 模型不存在: {model}", file=sys.stderr)
        return 1

    args = [
        str(server),
        "--model",
        "fast16/models/ColorLM-v4-SMoE.gguf",
        "--alias",
        "ColorLM-v4-SMoE",
        "--n-gpu-layers",
        "99",
        "--n-cpu-moe",
        str(CPU_MOE_LAYERS),
        "--ctx-size",
        "8192",
        "--parallel",
        "1",
        # mmap + CPU-expert overrides means every cold page is a 4K random SSD
        # read (2-7 t/s for the first ~6000 tokens). Load everything up front
        # instead: ~10 s slower boot, full speed from the first request.
        # NOTE: --mlock crashes this fork (recurrent-kernel migration allocates
        # outside the mlock init path, llama-mmap.cpp:744), so no-mmap only.
        "--no-mmap",
        # Hybrid/recurrent memory invalidates the prompt cache on every new
        # request anyway; don't reserve 8 GiB of RAM for cache that never hits.
        "--cache-ram",
        "0",
        # Draft-free self-speculation: batches k tokens per expert-weight read,
        # attacking the CPU-MoE bandwidth ceiling. Output distribution is
        # unchanged by construction (greedy eval stays bit-identical).
        "--spec-type",
        "ngram-mod",
        # Defaults (n_match=24, n_min=48) only fire on huge verbatim repeats
        # (whole-file re-emission). General text/code needs short chains too;
        # wrong drafts are rejected by verification, so this is risk-free.
        "--spec-ngram-mod-n-match",
        "16",
        "--spec-ngram-mod-n-min",
        "4",
        "--spec-ngram-mod-n-max",
        "16",
        "--embeddings",
        "--pooling",
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
        "--reasoning-format",
        "deepseek",
        "--reasoning-budget-message",
        "\nNow stop reasoning and give only the final answer.",
        "--slot-save-path",
        "fast16/runtime/kv-v4",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]
    stdout = (runtime / "v4-server.stdout.log").open("ab")
    stderr = (runtime / "v4-server.stderr.log").open("ab")
    environment = os.environ.copy()
    environment["COLORLM_V4_KERNEL_LAYERS"] = environment.get(
        "COLORLM_V4_KERNEL_LAYERS", "1"
    )
    environment["COLORLM_V4_RECURRENCE_ALPHA"] = environment.get(
        "COLORLM_V4_RECURRENCE_ALPHA", "0.15"
    )
    environment["COLORLM_V4_COGNITIVE_ROUNDS"] = environment.get(
        "COLORLM_V4_COGNITIVE_ROUNDS", "0"
    )
    environment["COLORLM_V4_COGNITIVE_THRESHOLD"] = environment.get(
        "COLORLM_V4_COGNITIVE_THRESHOLD", "0.08"
    )
    environment["COLORLM_V4_COGNITIVE_SHARPNESS"] = environment.get(
        "COLORLM_V4_COGNITIVE_SHARPNESS", "4.0"
    )
    environment["COLORLM_V4_SEMANTIC_PATH"] = environment.get(
        "COLORLM_V4_SEMANTIC_PATH", "fast16/runtime/colorlm-v4-semantic.bin"
    )
    environment["COLORLM_V4_SEMANTIC_ALPHA"] = environment.get(
        "COLORLM_V4_SEMANTIC_ALPHA", "0.0"
    )
    environment["COLORLM_V4_PLASTIC_RANK"] = environment.get(
        "COLORLM_V4_PLASTIC_RANK", "256"
    )
    environment["COLORLM_V4_PLASTIC_PATH"] = environment.get(
        "COLORLM_V4_PLASTIC_PATH", "fast16/runtime/colorlm-v4-plastic.bin"
    )
    environment["COLORLM_V4_PLASTIC_ALPHA"] = environment.get(
        "COLORLM_V4_PLASTIC_ALPHA", "1.0"
    )
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
    for _ in range(240):
        if healthy():
            return 0
        time.sleep(0.5)
    print("ColorLM v4 启动失败，请查看 fast16/runtime/v4-server.stderr.log", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
