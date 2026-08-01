"""在新跨模板 validation 上一次性筛选 v45 主脑，不接触 blind。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V29_GATE = ROOT / "fast16/research/v29_sequence_policy_head/run_generation_gate.py"
V31_GATE = ROOT / "fast16/research/v31_qwen36_expert_pair/run_v31_gate.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    capabilities = sorted({row["capability"] for row in rows})
    return {
        "passed": sum(bool(row["passed"]) for row in rows),
        "total": len(rows),
        "wall_seconds": sum(float(row["wall_ms"]) for row in rows) / 1000.0,
        "by_capability": {
            capability: {
                "passed": sum(
                    bool(row["passed"])
                    for row in rows
                    if row["capability"] == capability
                ),
                "total": sum(row["capability"] == capability for row in rows),
            }
            for capability in capabilities
        },
    }


def run_tasks(
    post_json,
    judge,
    port: int,
    alias: str,
    tasks: list[dict[str, Any]],
    seed: int,
    maximum_output_tokens: int,
) -> list[dict[str, Any]]:
    endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, 1):
        payload = {
            "model": alias,
            "messages": task["messages"],
            "tools": task["tools"],
            "temperature": 0,
            "seed": seed,
            "max_tokens": min(
                int(task.get("max_output_tokens", maximum_output_tokens)),
                maximum_output_tokens,
            ),
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
                "template_cluster_id": task["template_cluster_id"],
                "passed": bool(passed),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8145)
    parser.add_argument(
        "--server",
        type=Path,
        default=ROOT / "build/bin/Release/llama-server.exe",
    )
    parser.add_argument("--contract", type=Path, default=HERE / "v45_screen_contract.json")
    parser.add_argument("--output", type=Path, default=HERE / "v45_backbone_screen_report.json")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("format") != "colorlm-v45-backbone-screen-contract-v1":
        raise ValueError("v45筛选合同格式错误")
    dataset = ROOT / contract["dataset"]
    if not args.server.is_file() or not dataset.is_file():
        raise FileNotFoundError(args.server if not args.server.is_file() else dataset)
    if sha256_file(dataset) != contract["dataset_sha256"]:
        raise ValueError("v45数据集哈希与冻结合同不一致")
    tasks = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = [row for row in tasks if row["split"] == contract["allowed_split"]]
    if len(tasks) != int(contract["task_count"]):
        raise ValueError(f"validation任务数错误: {len(tasks)}")
    split_counts = Counter(row["split"] for row in tasks)
    if split_counts != Counter({contract["allowed_split"]: len(tasks)}):
        raise ValueError("筛选误触及禁止split")

    v29 = import_module("colorlm_v29_generation_gate", V29_GATE)
    v31 = import_module("colorlm_v31_gate", V31_GATE)
    if not v31.port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")

    results: list[dict[str, Any]] = []
    for variant in contract["variants"]:
        model = ROOT / variant["model"]
        if not model.is_file():
            raise FileNotFoundError(model)
        environment: dict[str, str] = {}
        policy_details = None
        if "policy" in variant:
            policy = ROOT / variant["policy"]
            manifest = policy / "policy.json"
            weights = policy / "weights.bin"
            if not manifest.is_file() or not weights.is_file():
                raise FileNotFoundError(policy)
            manifest_sha256 = sha256_file(manifest)
            environment = {
                "COLORLM_SEQUENCE_POLICY_PACKAGE": os.fspath(policy.resolve()),
                "COLORLM_SEQUENCE_POLICY_MANIFEST_SHA256": manifest_sha256,
                "COLORLM_SEQUENCE_POLICY_VERIFY_WEIGHTS": "1",
            }
            policy_details = {
                "manifest_sha256": manifest_sha256,
                "weights_sha256": sha256_file(weights),
            }
        print(f"启动 {variant['alias']}", flush=True)
        process, load_seconds = v31.start_server(
            args.server,
            model,
            variant["alias"],
            args.port,
            extra_environment=environment,
        )
        try:
            rows = run_tasks(
                v29.post_json,
                v29.judge,
                args.port,
                variant["alias"],
                tasks,
                int(contract["seed"]),
                int(contract["maximum_output_tokens"]),
            )
        finally:
            v31.stop_server(process)
        results.append(
            {
                "id": variant["id"],
                "alias": variant["alias"],
                "model": variant["model"],
                "model_bytes": model.stat().st_size,
                "load_seconds": load_seconds,
                "policy": policy_details,
                "summary": summarize(rows),
                "rows": rows,
            }
        )
        time.sleep(1)

    by_id = {row["id"]: row for row in results}
    gate = contract["promotion_gate"]
    baseline = by_id[gate["baseline_id"]]
    baseline_rows = {row["id"]: row for row in baseline["rows"]}
    comparisons = []
    for candidate in results:
        if candidate["id"] == baseline["id"]:
            continue
        candidate_rows = {row["id"]: row for row in candidate["rows"]}
        wins = [
            task_id
            for task_id, base_row in baseline_rows.items()
            if not base_row["passed"] and candidate_rows[task_id]["passed"]
        ]
        regressions = [
            task_id
            for task_id, base_row in baseline_rows.items()
            if base_row["passed"] and not candidate_rows[task_id]["passed"]
        ]
        capability_regressions = {
            capability: (
                candidate["summary"]["by_capability"][capability]["passed"]
                - baseline["summary"]["by_capability"][capability]["passed"]
            )
            for capability in baseline["summary"]["by_capability"]
        }
        score_gain = candidate["summary"]["passed"] - baseline["summary"]["passed"]
        paired_net_wins = len(wins) - len(regressions)
        passed = bool(
            score_gain >= int(gate["minimum_score_gain"])
            and paired_net_wins >= int(gate["minimum_paired_net_wins"])
            and min(capability_regressions.values())
            >= -int(gate["maximum_per_capability_regression"])
            and min(
                bucket["passed"]
                for bucket in candidate["summary"]["by_capability"].values()
            )
            >= int(gate["minimum_passes_per_capability"])
        )
        comparisons.append(
            {
                "candidate_id": candidate["id"],
                "score_gain": score_gain,
                "paired_wins": wins,
                "paired_regressions": regressions,
                "paired_net_wins": paired_net_wins,
                "capability_score_deltas": capability_regressions,
                "gate_passed": passed,
            }
        )

    passing = [row for row in comparisons if row["gate_passed"]]
    if passing:
        passing.sort(
            key=lambda row: (
                by_id[row["candidate_id"]]["summary"]["passed"],
                row["paired_net_wins"],
                -by_id[row["candidate_id"]]["summary"]["wall_seconds"],
            ),
            reverse=True,
        )
        selected = passing[0]["candidate_id"]
        decision = f"allow {selected} as v45 backbone candidate; blind and composition gates still required"
    else:
        selected = baseline["id"]
        decision = contract["failure_action"]

    report = {
        "format": "colorlm-v45-backbone-screen-report-v1",
        "contract": os.fspath(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "dataset_sha256": sha256_file(dataset),
        "split": contract["allowed_split"],
        "blind_touched": False,
        "results": results,
        "comparisons": comparisons,
        "selected_id": selected,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summaries": {row["id"]: row["summary"] for row in results},
                "comparisons": comparisons,
                "selected_id": selected,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
