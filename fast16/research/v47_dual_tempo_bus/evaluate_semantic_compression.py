"""评估“短 Design Genome → 确定性编译”是否同时提升质量与端到端速度。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


FORMAT = "colorlm-v47-semantic-compression-ab-v1"


def finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} 必须是数值")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} 必须是有限非负数")
    return value


def required_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} 必须是布尔值")
    return value


def normalize_case(raw: dict[str, Any], index: int) -> dict[str, Any]:
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"cases[{index}].id 无效")
    split = raw.get("split")
    if split not in {"train", "validation", "blind"}:
        raise ValueError(f"{case_id}.split 无效")
    direct = raw.get("direct")
    compressed = raw.get("compressed")
    if not isinstance(direct, dict) or not isinstance(compressed, dict):
        raise ValueError(f"{case_id} 缺少 direct/compressed")

    direct_wall = finite_nonnegative(direct.get("wall_seconds"), f"{case_id}.direct.wall_seconds")
    model_wall = finite_nonnegative(
        compressed.get("model_wall_seconds"), f"{case_id}.compressed.model_wall_seconds"
    )
    compile_wall = finite_nonnegative(
        compressed.get("compile_wall_seconds"), f"{case_id}.compressed.compile_wall_seconds"
    )
    ir_bytes = finite_nonnegative(compressed.get("ir_bytes"), f"{case_id}.compressed.ir_bytes")
    html_bytes = finite_nonnegative(
        compressed.get("output_bytes"), f"{case_id}.compressed.output_bytes"
    )
    direct_score = finite_nonnegative(direct.get("score"), f"{case_id}.direct.score")
    compressed_score = finite_nonnegative(
        compressed.get("score"), f"{case_id}.compressed.score"
    )
    if direct_score > 100 or compressed_score > 100:
        raise ValueError(f"{case_id} score 必须位于 0..100")
    if ir_bytes <= 0 or html_bytes <= 0:
        raise ValueError(f"{case_id} IR/HTML 字节数必须大于0")
    total_wall = model_wall + compile_wall
    if direct_wall <= 0 or total_wall <= 0:
        raise ValueError(f"{case_id} 墙钟必须大于0")

    direct_passed = required_bool(direct.get("passed"), f"{case_id}.direct.passed")
    compressed_passed = required_bool(compressed.get("passed"), f"{case_id}.compressed.passed")
    deterministic = required_bool(
        compressed.get("deterministic"), f"{case_id}.compressed.deterministic"
    )
    critical_regression = direct_passed and not compressed_passed
    return {
        "id": case_id,
        "split": split,
        "direct_passed": direct_passed,
        "compressed_passed": compressed_passed,
        "deterministic": deterministic,
        "critical_regression": critical_regression,
        "score_delta": compressed_score - direct_score,
        "end_to_end_speedup": direct_wall / total_wall,
        "compiler_expansion_ratio": html_bytes / ir_bytes,
        "ir_bytes": int(ir_bytes),
        "html_bytes": int(html_bytes),
        "direct_wall_seconds": direct_wall,
        "compressed_wall_seconds": total_wall,
    }


def evaluate(payload: dict[str, Any], split: str) -> dict[str, Any]:
    if payload.get("format") != FORMAT:
        raise ValueError(f"输入 format 必须为 {FORMAT}")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases 必须是非空数组")
    cases = [normalize_case(item, index) for index, item in enumerate(raw_cases)]
    chosen = [item for item in cases if item["split"] == split]
    if not chosen:
        raise ValueError(f"没有 split={split} 的样本")

    scores = [item["score_delta"] for item in chosen]
    speedups = [item["end_to_end_speedup"] for item in chosen]
    expansion = [item["compiler_expansion_ratio"] for item in chosen]
    passed = sum(item["compressed_passed"] for item in chosen)
    regressions = [item["id"] for item in chosen if item["critical_regression"]]
    nondeterministic = [item["id"] for item in chosen if not item["deterministic"]]
    minimum_passes = math.ceil(len(chosen) * 0.75)
    metrics = {
        "sample_count": len(chosen),
        "compressed_pass_count": passed,
        "compressed_pass_rate": passed / len(chosen),
        "minimum_required_passes": minimum_passes,
        "median_score_delta": statistics.median(scores),
        "median_end_to_end_speedup": statistics.median(speedups),
        "median_compiler_expansion_ratio": statistics.median(expansion),
        "critical_regressions": regressions,
        "nondeterministic_cases": nondeterministic,
    }
    gates = {
        "pass_rate": passed >= minimum_passes,
        "median_score_delta_at_least_12": metrics["median_score_delta"] >= 12.0,
        "median_end_to_end_speedup_at_least_2_5x": metrics["median_end_to_end_speedup"] >= 2.5,
        "median_expansion_at_least_8x": metrics["median_compiler_expansion_ratio"] >= 8.0,
        "zero_critical_regression": not regressions,
        "deterministic_compilation": not nondeterministic,
    }
    return {
        "format": "colorlm-v47-semantic-compression-evaluation-v1",
        "source_format": FORMAT,
        "split": split,
        "status": "passed" if all(gates.values()) else "failed",
        "metrics": metrics,
        "gates": gates,
        "cases": chosen,
        "claim_limit": (
            "该门只证明短IR编译路径的端到端收益；只有模型独立生成IR并在冻结validation/blind通过，"
            "才可写成模型能力提升。"
        ),
    }


def selftest() -> None:
    cases = []
    for index in range(8):
        cases.append(
            {
                "id": f"fixture-{index}",
                "split": "validation",
                "direct": {"wall_seconds": 30.0, "score": 55.0, "passed": index < 2},
                "compressed": {
                    "model_wall_seconds": 7.0,
                    "compile_wall_seconds": 0.1,
                    "ir_bytes": 400,
                    "output_bytes": 8000,
                    "score": 75.0,
                    "passed": index < 7,
                    "deterministic": True,
                },
            }
        )
    report = evaluate({"format": FORMAT, "cases": cases}, "validation")
    assert report["status"] == "passed", report
    cases[0]["compressed"]["deterministic"] = False
    report = evaluate({"format": FORMAT, "cases": cases}, "validation")
    assert report["status"] == "failed", report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "blind"), default="validation")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print(json.dumps({"selftest": "passed"}, ensure_ascii=False))
        return 0
    if args.input is None:
        parser.error("非 selftest 模式必须提供 --input")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = evaluate(payload, args.split)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"拒绝覆盖已有报告: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
