"""Run one short decode-speed A/B for Fast16 v6 with Vulkan spin fence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "fast16" / "models" / "ColorLM-v6-Q3Router-Fused-A1.gguf"
BENCH = ROOT / "build" / "bin" / "Release" / "llama-bench.exe"
OUTPUT = Path(__file__).resolve().parent / "fast16_spin_benchmark_report.json"


def run_arm(label: str, spin: bool) -> dict:
    environment = os.environ.copy()
    if spin:
        environment["GGML_VK_SPIN_FENCE"] = "1"
    else:
        environment.pop("GGML_VK_SPIN_FENCE", None)
    command = [
        os.fspath(BENCH),
        "--model",
        os.fspath(MODEL.relative_to(ROOT)),
        "--n-prompt",
        "0",
        "--n-gen",
        "64",
        "--repetitions",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--n-gpu-layers",
        "99",
        "--n-cpu-moe",
        "29",
        "--flash-attn",
        "1",
        "--mmap",
        "0",
        "--output",
        "json",
    ]
    print(f"运行 {label}...", flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=360,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label}失败:\n{result.stderr[-4000:]}")
    payload = json.loads(result.stdout)
    rows = payload if isinstance(payload, list) else [payload]
    generation = next((row for row in rows if int(row.get("n_gen", 0)) == 64), None)
    if generation is None:
        raise RuntimeError(f"{label}没有tg64结果")
    return {
        "label": label,
        "spin_fence": spin,
        "tokens_per_second": float(generation["avg_ts"]),
        "stddev_tokens_per_second": float(generation.get("stddev_ts", 0.0)),
        "raw": generation,
    }


def main() -> int:
    if not MODEL.is_file() or not BENCH.is_file():
        print("缺少模型或llama-bench", file=sys.stderr)
        return 1
    baseline = run_arm("baseline", False)
    spin = run_arm("spin_fence", True)
    gain = (spin["tokens_per_second"] / baseline["tokens_per_second"] - 1.0) * 100.0
    report = {
        "format": "fast16-spin-fence-ab-v1",
        "model": MODEL.name,
        "decode_tokens": 64,
        "baseline": baseline,
        "spin_fence": spin,
        "speedup_percent": gain,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
