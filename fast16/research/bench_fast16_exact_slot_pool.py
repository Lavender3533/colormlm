"""Short A/B for the exact qwen35moe expert slot pool."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "fast16" / "models" / "ColorLM-v6-Q3Router-Fused-A1.gguf"
BENCH = ROOT / "build" / "bin" / "Release" / "llama-bench.exe"
OUTPUT = Path(__file__).resolve().parent / "fast16_exact_slot_pool_ab_report.json"


def run_arm(label: str, layers: int = 0, slots: int = 0) -> dict:
    environment = os.environ.copy()
    environment["GGML_SCHED_MERGE_CPU_SYNC"] = "1"
    environment["GGML_SCHED_SKIP_CPU_FINAL_SYNC"] = "1"
    environment.pop("GGML_VK_SPIN_FENCE", None)
    if layers > 0 and slots > 0:
        environment["COLORLM_MOE_SLOT_POOL"] = "1"
        environment["COLORLM_MOE_SLOT_LAYERS"] = str(layers)
        environment["COLORLM_MOE_SLOT_K"] = str(slots)
    else:
        environment.pop("COLORLM_MOE_SLOT_POOL", None)
        environment.pop("COLORLM_MOE_SLOT_LAYERS", None)
        environment.pop("COLORLM_MOE_SLOT_K", None)

    command = [
        os.fspath(BENCH), "--model", os.fspath(MODEL.relative_to(ROOT)),
        "--n-prompt", "0", "--n-gen", "64", "--repetitions", "3",
        "--threads", "8", "--batch-size", "512", "--ubatch-size", "512",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--n-gpu-layers", "99", "--n-cpu-moe", "29", "--flash-attn", "1",
        "--mmap", "0", "--output", "json",
    ]
    print(f"运行 {label}...", flush=True)
    result = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=360,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label}失败:\n{result.stderr[-4000:]}")
    payload = json.loads(result.stdout)
    rows = payload if isinstance(payload, list) else [payload]
    row = next((item for item in rows if int(item.get("n_gen", 0)) == 64), None)
    if row is None:
        raise RuntimeError(f"{label}没有tg64结果")
    return {
        "label": label,
        "layers": layers,
        "slots_per_layer": slots,
        "tokens_per_second": float(row["avg_ts"]),
        "stddev_tokens_per_second": float(row["stddev_ts"]),
        "raw": row,
    }


def main() -> int:
    if not MODEL.is_file() or not BENCH.is_file():
        print("缺少模型或llama-bench", file=sys.stderr)
        return 1
    baseline = run_arm("baseline")
    slot_pool = run_arm("exact_slot_pool_8x28", layers=8, slots=28)
    gain = (slot_pool["tokens_per_second"] / baseline["tokens_per_second"] - 1.0) * 100.0
    report = {
        "format": "fast16-exact-slot-pool-ab-v1",
        "model": MODEL.name,
        "decode_tokens": 64,
        "repetitions": 3,
        "baseline": baseline,
        "exact_slot_pool": slot_pool,
        "speedup_percent": gain,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
