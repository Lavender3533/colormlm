"""在冻结的24条test轨迹上比较v36与v43真实生成。"""

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
V29 = ROOT / "fast16/research/v29_sequence_policy_head"
V31 = ROOT / "fast16/research/v31_qwen36_expert_pair"
sys.path.insert(0, os.fspath(V29))
sys.path.insert(0, os.fspath(V31))

from run_generation_gate import judge, post_json  # noqa: E402
from run_v31_gate import port_available, start_server, stop_server  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_tasks(port: int, alias: str, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
    rows = []
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
                "capability": task["capability"],
                "label": task["label"],
                "passed": passed,
                "reason": reason,
                "wall_ms": (time.perf_counter() - started) * 1000,
                "target": task["target"],
                "response": response,
                "error": error,
            }
        )
        print(f"  [{index:02d}/{len(tasks):02d}] {task['id']}: {'PASS' if passed else 'FAIL'}", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    capabilities = sorted({row["capability"] for row in rows})
    return {
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "by_capability": {
            capability: {
                "passed": sum(row["passed"] for row in rows if row["capability"] == capability),
                "total": sum(row["capability"] == capability for row in rows),
            }
            for capability in capabilities
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8143)
    parser.add_argument("--server", type=Path, default=ROOT / "llama.cpp/build-v17-perf/bin/Release/llama-server.exe")
    parser.add_argument("--model", type=Path, default=ROOT / "fast16/models/ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf")
    parser.add_argument("--policy", type=Path, default=HERE / "runtime-v1")
    parser.add_argument("--tasks", type=Path, default=HERE / "trajectory_tasks_v1.jsonl")
    parser.add_argument("--contract", type=Path, default=HERE / "policy_contract.json")
    parser.add_argument("--output", type=Path, default=HERE / "v43-generation-gate.json")
    args = parser.parse_args()

    manifest = args.policy / "policy.json"
    weights = args.policy / "weights.bin"
    for path in (args.server, args.model, manifest, weights, args.tasks, args.contract):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    contract_sha = sha256_file(args.contract)
    policy_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    if sha256_file(args.tasks) != contract["source_tasks_sha256"]:
        raise ValueError("生成门任务与冻结合同不一致")
    if policy_manifest.get("source_contract_sha256") != contract_sha:
        raise ValueError("v43策略包与冻结合同不匹配")
    tasks = [
        json.loads(line)
        for line in args.tasks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = [task for task in tasks if task["split"] == "test"]
    if len(tasks) != int(contract["runtime_gate"]["test_tasks"]):
        raise ValueError("冻结test任务数量错误")
    task_ids = [task["id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("冻结test任务ID不唯一")

    manifest_sha = sha256_file(manifest)
    variants = [
        ("ColorLM-v36-Qwen36-Global-Shared-Backbone", {}),
        (
            "ColorLM-v43-PCA-Noop-Policy",
            {
                "COLORLM_SEQUENCE_POLICY_PACKAGE": str(args.policy.resolve()),
                "COLORLM_SEQUENCE_POLICY_MANIFEST_SHA256": manifest_sha,
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
            results.append({"alias": alias, "load_seconds": load_seconds, "summary": summarize(rows), "rows": rows})
        finally:
            stop_server(process)
        time.sleep(1)

    base, candidate = results
    base_by_id = {row["id"]: row for row in base["rows"]}
    candidate_by_id = {row["id"]: row for row in candidate["rows"]}
    wins = [task_id for task_id in base_by_id if not base_by_id[task_id]["passed"] and candidate_by_id[task_id]["passed"]]
    regressions = [task_id for task_id in base_by_id if base_by_id[task_id]["passed"] and not candidate_by_id[task_id]["passed"]]
    gate = contract["runtime_gate"]
    net_fixes = len(wins) - len(regressions)
    passed = net_fixes >= int(gate["minimum_net_fixes"]) and len(regressions) <= int(gate["maximum_regressions"])
    report = {
        "format": "colorlm-v43-generation-gate-v1",
        "tasks_sha256": sha256_file(args.tasks),
        "contract_sha256": sha256_file(args.contract),
        "policy_manifest_sha256": manifest_sha,
        "policy_weights_sha256": sha256_file(weights),
        "models": results,
        "comparison": {
            "wins": wins,
            "regressions": regressions,
            "score_delta": net_fixes,
            "gate_passed": passed,
        },
        "decision": "allow bypass and speed gates" if passed else "stop v43 runtime candidate",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"v36": base["summary"], "v43": candidate["summary"], "comparison": report["comparison"]}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
