"""在已消费的 validation 上运行 v46 开发门；结果不能直接晋级正式模型。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V29 = ROOT / "fast16/research/v29_sequence_policy_head/run_generation_gate.py"
V31 = ROOT / "fast16/research/v31_qwen36_expert_pair/run_v31_gate.py"
V45 = ROOT / "fast16/research/v45_backbone_screen/run_v45_backbone_screen.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8146)
    parser.add_argument("--server", type=Path, default=ROOT / "build/bin/Release/llama-server.exe")
    parser.add_argument("--contract", type=Path, default=HERE / "v46_dev_contract.json")
    parser.add_argument("--output", type=Path, default=HERE / "v46_dev_gate_report.json")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    contract_format = contract.get("format")
    if contract_format not in {
        "colorlm-v46-mid-cortex-dev-contract-v1",
        "colorlm-v46-mid-cortex-blind-contract-v1",
    }:
        raise ValueError("v46开发合同格式错误")
    dataset = ROOT / contract["dataset"]
    baseline_model = ROOT / contract["base"]
    candidate_model = ROOT / contract["candidate"]
    policy = ROOT / contract["policy"]
    manifest = policy / "policy.json"
    weights = policy / "weights.bin"
    for path in (args.server, dataset, baseline_model, candidate_model, manifest, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(dataset) != contract["dataset_sha256"]:
        raise ValueError("数据集哈希与冻结合同不一致")
    tasks = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = [row for row in tasks if row["split"] == contract["split"]]
    if len(tasks) != int(contract["task_count"]):
        raise ValueError(f"开发任务数错误: {len(tasks)}")

    v29 = import_module("colorlm_v29_gate_for_v46", V29)
    v31 = import_module("colorlm_v31_gate_for_v46", V31)
    v45 = import_module("colorlm_v45_helpers_for_v46", V45)
    if not v31.port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")
    policy_environment = {
        "COLORLM_SEQUENCE_POLICY_PACKAGE": os.fspath(policy.resolve()),
        "COLORLM_SEQUENCE_POLICY_MANIFEST_SHA256": sha256_file(manifest),
        "COLORLM_SEQUENCE_POLICY_VERIFY_WEIGHTS": "1",
    }
    variants = [
        (
            "v38",
            "ColorLM-v38-Qwen36-Shared-Sequence-Policy-v46-Control",
            baseline_model,
        ),
        (
            "v46",
            "ColorLM-v46-Qwen36-Mid-Cortex-L16-L31",
            candidate_model,
        ),
    ]
    results = []
    for variant_id, alias, model in variants:
        print(f"启动 {alias}", flush=True)
        process, load_seconds = v31.start_server(
            args.server,
            model,
            alias,
            args.port,
            extra_environment=policy_environment,
        )
        try:
            rows = v45.run_tasks(
                v29.post_json,
                v29.judge,
                args.port,
                alias,
                tasks,
                int(contract["seed"]),
                int(contract["maximum_output_tokens"]),
            )
        finally:
            v31.stop_server(process)
        results.append(
            {
                "id": variant_id,
                "alias": alias,
                "model": model.name,
                "model_bytes": model.stat().st_size,
                "load_seconds": load_seconds,
                "summary": v45.summarize(rows),
                "rows": rows,
            }
        )
        time.sleep(1)

    baseline, candidate = results
    base_rows = {row["id"]: row for row in baseline["rows"]}
    candidate_rows = {row["id"]: row for row in candidate["rows"]}
    wins = [
        task_id
        for task_id, row in base_rows.items()
        if not row["passed"] and candidate_rows[task_id]["passed"]
    ]
    regressions = [
        task_id
        for task_id, row in base_rows.items()
        if row["passed"] and not candidate_rows[task_id]["passed"]
    ]
    capability_deltas = {
        capability: (
            candidate["summary"]["by_capability"][capability]["passed"]
            - baseline["summary"]["by_capability"][capability]["passed"]
        )
        for capability in baseline["summary"]["by_capability"]
    }
    score_gain = candidate["summary"]["passed"] - baseline["summary"]["passed"]
    speed_regression = (
        candidate["summary"]["wall_seconds"] / baseline["summary"]["wall_seconds"] - 1.0
    )
    gate = contract["development_gate"]
    passed = bool(
        score_gain >= int(gate["minimum_score_gain_over_v38"])
        and len(wins) - len(regressions) >= int(gate.get("minimum_paired_net_wins", 0))
        and len(regressions) <= int(gate["maximum_regressions"])
        and min(capability_deltas.values()) >= -int(gate["maximum_per_capability_regression"])
        and min(
            bucket["passed"]
            for bucket in candidate["summary"]["by_capability"].values()
        )
        >= int(gate["minimum_passes_per_capability"])
        and speed_regression <= float(gate["maximum_speed_regression"])
    )
    is_blind = contract_format == "colorlm-v46-mid-cortex-blind-contract-v1"
    report = {
        "format": (
            "colorlm-v46-mid-cortex-blind-gate-report-v1"
            if is_blind
            else "colorlm-v46-mid-cortex-dev-gate-report-v1"
        ),
        "claim_scope": contract["claim_scope"],
        "contract_sha256": sha256_file(args.contract),
        "dataset_sha256": sha256_file(dataset),
        "split": contract["split"],
        "results": results,
        "comparison": {
            "score_gain": score_gain,
            "wins": wins,
            "regressions": regressions,
            "capability_score_deltas": capability_deltas,
            "speed_regression": speed_regression,
            "gate_passed": passed,
        },
        "decision": (
            (
                "promote v46 as the formal ColorLM model"
                if is_blind
                else "allow a newly generated blind dataset; not formal promotion"
            )
            if passed
            else contract["failure_action"]
        ),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "v38": baseline["summary"],
                "v46": candidate["summary"],
                "comparison": report["comparison"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
