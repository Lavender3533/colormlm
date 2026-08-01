"""比较 v36 与仅接入 v29 工具策略头后的 v38。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POLICY_DIR = ROOT / "fast16" / "research" / "v29_sequence_policy_head"
sys.path.insert(0, os.fspath(HERE))
sys.path.insert(0, os.fspath(POLICY_DIR))

from run_generation_gate import judge, post_json  # noqa: E402
from run_v31_gate import port_available, start_server, stop_server  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 v38 工具策略组合门")
    parser.add_argument("--port", type=int, default=8138)
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
        default=ROOT
        / "fast16"
        / "models"
        / "ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf",
    )
    parser.add_argument(
        "--policy", type=Path, default=POLICY_DIR / "runtime-v1"
    )
    parser.add_argument(
        "--tasks", type=Path, default=POLICY_DIR / "policy_tasks_v1.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "v38_policy_gate_report.json"
    )
    return parser.parse_args()


def run_tasks(port: int, alias: str, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
    for index, task in enumerate(tasks, 1):
        payload = {
            "model": alias,
            "messages": task["messages"],
            "tools": task["tools"],
            "temperature": 0,
            "seed": 3407,
            "max_tokens": 96,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            response = post_json(endpoint, payload)
            passed, reason = judge(task, response)
            error = None
        except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
            response = None
            passed, reason, error = False, "请求或响应解析失败", repr(exc)
        rows.append(
            {
                "id": task["id"],
                "split": task["split"],
                "family": task["family"],
                "passed": passed,
                "reason": reason,
                "wall_ms": (time.perf_counter() - started) * 1000,
                "target": task["target"],
                "response": response,
                "error": error,
            }
        )
        print(
            f"  [{index:02d}/{len(tasks):02d}] {task['id']}: "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, int]] = {}
    for row in rows:
        for key, bucket in ((row["split"], by_split), (row["family"], by_family)):
            value = bucket.setdefault(key, {"passed": 0, "total": 0})
            value["total"] += 1
            value["passed"] += int(row["passed"])
    return {
        "passed": sum(int(row["passed"]) for row in rows),
        "total": len(rows),
        "by_split": by_split,
        "by_family": by_family,
    }


def main() -> int:
    args = parse_args()
    manifest = args.policy / "policy.json"
    weights = args.policy / "weights.bin"
    for path in (args.server, args.model, manifest, weights, args.tasks):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")
    tasks = [
        json.loads(line)
        for line in args.tasks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    variants = [
        ("ColorLM-v36-Qwen36-Global-Shared-Backbone", {}),
        (
            "ColorLM-v38-Qwen36-Shared-Sequence-Policy",
            {
                "COLORLM_SEQUENCE_POLICY_PACKAGE": str(args.policy.resolve()),
                "COLORLM_SEQUENCE_POLICY_MANIFEST_SHA256": manifest_sha256,
                "COLORLM_SEQUENCE_POLICY_VERIFY_WEIGHTS": "1",
            },
        ),
    ]
    results = []
    for alias, environment in variants:
        print(f"启动 {alias}", flush=True)
        process, load_seconds = start_server(
            args.server,
            args.model,
            alias,
            args.port,
            extra_environment=environment,
        )
        try:
            rows = run_tasks(args.port, alias, tasks)
            results.append(
                {
                    "alias": alias,
                    "load_seconds": load_seconds,
                    "summary": summary(rows),
                    "rows": rows,
                }
            )
        finally:
            stop_server(process)
        time.sleep(1)

    base, candidate = results
    base_by_id = {row["id"]: row for row in base["rows"]}
    candidate_by_id = {row["id"]: row for row in candidate["rows"]}
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
    report = {
        "format": "colormlm-v38-policy-composition-gate-v1",
        "model": args.model.name,
        "policy_manifest_sha256": manifest_sha256,
        "models": results,
        "comparison": {
            "wins": wins,
            "regressions": regressions,
            "score_delta": candidate["summary"]["passed"]
            - base["summary"]["passed"],
            "advances": bool(wins and not regressions),
        },
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "v36": base["summary"],
                "v38": candidate["summary"],
                "comparison": report["comparison"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"报告: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
