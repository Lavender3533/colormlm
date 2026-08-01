"""比较 v36 shared-backbone 与接入 v17 Coder 神经岛后的 v37。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_v31_gate import (
    ROOT,
    port_available,
    run_speed,
    run_tasks,
    sha256_file,
    start_server,
    stop_server,
)


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    parser = argparse.ArgumentParser(description="运行 v37 Coder岛组合短门")
    parser.add_argument("--port", type=int, default=8137)
    parser.add_argument("--tasks", type=Path, default=HERE / "v36_gate.json")
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
        "--model",
        type=Path,
        default=models / "ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf",
    )
    parser.add_argument(
        "--island",
        type=Path,
        default=ROOT
        / "fast16"
        / "research"
        / "v17_coder_island"
        / "runtime-v3"
        / "island.json",
    )
    parser.add_argument("--output", type=Path, default=HERE / "v37_gate_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.tasks, args.server, args.model, args.island):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")
    bank = json.loads(args.tasks.read_text(encoding="utf-8"))
    if bank.get("frozen_before_generation") is not True:
        raise RuntimeError("短门没有冻结标记")
    tasks = bank["items"]
    report: dict[str, Any] = {
        "format": "colormlm-v37-coder-island-composition-gate-v1",
        "contract": args.tasks.as_posix(),
        "contract_sha256": sha256_file(args.tasks),
        "models": [],
    }
    variants = [
        ("ColorLM-v36-Qwen36-Global-Shared-Backbone", {}),
        (
            "ColorLM-v37-Qwen36-Shared-Coder-Island",
            {
                "COLORLM_NEURAL_ISLAND_MANIFEST": str(args.island.resolve()),
                "COLORLM_NEURAL_ISLAND_ALPHA": "0.02",
                "COLORLM_NEURAL_ISLAND_SITE": "35",
                "COLORLM_NEURAL_ISLAND_TARGET_RATIO": "0.04",
                "COLORLM_NEURAL_ISLAND_SHARPNESS": "4.0",
                "COLORLM_NEURAL_ISLAND_EXPERT_CACHE_SLOTS": "32",
                "COLORLM_NEURAL_ISLAND_EXPERT_CACHE_POLICY": "lru",
            },
        ),
    ]
    for alias, extra_environment in variants:
        print(f"启动 {alias}", flush=True)
        process, load_seconds = start_server(
            args.server,
            args.model,
            alias,
            args.port,
            extra_environment=extra_environment,
        )
        try:
            rows = run_tasks(args.port, alias, tasks)
            report["models"].append(
                {
                    "alias": alias,
                    "model": args.model.name,
                    "load_seconds": load_seconds,
                    "passed": sum(bool(row["passed"]) for row in rows),
                    "total": len(rows),
                    "responses": rows,
                    "speed": run_speed(args.port, alias),
                }
            )
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
        "composition_safe": not regressions,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    print(f"报告: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
