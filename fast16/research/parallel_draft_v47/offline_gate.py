"""汇总证据、覆盖、自由滚动与成本门；任何缺项都显式停止。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_optional(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-audit", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("拒绝覆盖已有停止门报告")
    audit = json.loads(args.asset_audit.read_text(encoding="utf-8"))
    acceptance = load_optional(args.acceptance)
    cost = json.loads(args.cost.read_text(encoding="utf-8"))
    dataset_keys = (
        "v38_anchor_capture",
        "train_validation_test_present",
        "full_oracle_token_sequences",
        "v38_free_roll_validator_sequences",
    )
    gates = {
        "real_v38_dataset_contract": all(bool(audit["contract_gates"].get(key)) for key in dataset_keys),
        "oracle_full_trajectory_coverage": bool(
            acceptance and acceptance.get("coverage", {}).get("gate_passed")
        ),
        "free_roll_v38_acceptance": bool(
            acceptance and acceptance.get("simulation_executed") and acceptance.get("v38_verifier_proven")
        ),
        "analytical_speedup_lower_bound_at_least_1_08": bool(cost.get("stop_gate", {}).get("passed")),
    }
    passed = all(gates.values())
    missing = [name for name, value in gates.items() if not value]
    report = {
        "format": "colorlm-v47-parallel-draft-offline-gate-v1",
        "gates": gates,
        "passed": passed,
        "decision": "offline_gate_passed_runtime_design_may_start" if passed else "stop_no_cpp",
        "failed_or_missing": missing,
        "reason": (
            "全部必要证据与8%解析下界已过；本报告仍不等于端到端加速。"
            if passed else
            "现有真实资产不足以做v38完整轨迹自由滚动接受预测；按合同停止，不改C++。"
        ),
        "inputs": {
            "asset_audit": str(args.asset_audit.resolve()),
            "acceptance": None if args.acceptance is None else str(args.acceptance.resolve()),
            "cost": str(args.cost.resolve())
        },
        "claim_limit": "只有全部硬门通过才允许讨论运行时；解析通过后仍需真实相邻A/B与零回归门。"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
