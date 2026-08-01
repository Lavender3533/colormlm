#!/usr/bin/env python3
"""v47工具的标准库自检，不读取模型权重。"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import cost_model
import offline_gate


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="运行v47纯CPU自检")
    parser.add_argument("--output", type=Path, default=HERE / "selftest_report.json")
    args = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    errors = offline_gate.validate_manifest(manifest)
    assert not errors, errors
    cost = cost_model.calculate(manifest)
    assert 0 < cost["draft_head"]["trainable_parameters"] < 10_000_000
    assert cost["latent_recursion"]["trainable_parameters"] < 3_000_000
    assert math.isclose(cost["latent_recursion"]["mean_k_assumption"], 1.33, abs_tol=1e-9)
    records = offline_gate.read_records(
        offline_gate.WORKSPACE / "fast16/research/v24_speed_quality_bus/routes-none-20260801.bin"
    )
    assert len(records) == 512
    assert {record["layer"] for record in records} == {44, 45, 46, 47}
    replay = offline_gate.replay(records, manifest["paging"])
    assert replay["aggregate"]["lru"]["requests"] == 2560
    assert replay["aggregate"]["dali"]["requests"] == 2560
    assert replay["aggregate"]["dali"]["cold_misses"] <= replay["aggregate"]["lru"]["cold_misses"]
    for schema in sorted((HERE / "schemas").glob("*.json")):
        parsed = json.loads(schema.read_text(encoding="utf-8"))
        assert parsed["$schema"].endswith("2020-12/schema")
    contract = json.loads((HERE / "short_gate_contract.json").read_text(encoding="utf-8"))
    assert contract["offline_only"] is True
    assert not offline_gate.validate_utf8_no_bom(HERE)
    # 验证成本脚本输出可JSON往返，且临时文件不会进入工作区。
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cost.json"
        path.write_text(json.dumps(cost, ensure_ascii=False), encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8"))["format"].endswith("cost-v1")
    report = {
        "format": "colorlm-parallel-speed-v47-selftest-v1",
        "passed": True,
        "tests": 13,
        "records": len(records),
        "cpu_only": True,
        "gpu_used": False
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
