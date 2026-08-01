"""按冻结前端短门比较固定三卡片基线与 v47 IR/编译候选。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FRONTEND = HERE.parent / "parallel_frontend_v47"
sys.path.insert(0, str(FRONTEND))

from validate_gates import evaluate_one  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("拒绝覆盖 A/B 报告")
    rows = [row for row in read_jsonl(args.tasks) if row.get("id") == args.task_id]
    if len(rows) != 1:
        raise ValueError(f"找不到唯一任务 {args.task_id}")
    baseline = evaluate_one(rows[0], args.baseline)
    candidate = evaluate_one(rows[0], args.candidate)
    base_score = float(baseline["audit"]["summary"]["final_score"])
    candidate_score = float(candidate["audit"]["summary"]["final_score"])
    base_penalty = float(baseline["audit"]["summary"]["template_penalty"])
    candidate_penalty = float(candidate["audit"]["summary"]["template_penalty"])
    result = {
        "format": "colorlm-v47-frontend-ir-single-task-ab-v1",
        "status": "train_task_development_hybrid",
        "task_id": args.task_id,
        "baseline": {
            "file": str(args.baseline.resolve()),
            "final_score": base_score,
            "template_penalty": base_penalty,
            "passed": baseline["passed"],
            "critical": baseline["critical"],
            "score_checks": baseline["score_checks"],
            "signals": baseline["signals"],
        },
        "candidate": {
            "file": str(args.candidate.resolve()),
            "final_score": candidate_score,
            "template_penalty": candidate_penalty,
            "passed": candidate["passed"],
            "critical": candidate["critical"],
            "score_checks": candidate["score_checks"],
            "signals": candidate["signals"],
            "dimensions": {
                name: value["score"] for name, value in candidate["audit"]["dimensions"].items()
            },
        },
        "delta": {
            "final_score": candidate_score - base_score,
            "template_penalty": candidate_penalty - base_penalty,
        },
        "decision": {
            "single_task_direction_positive": bool(
                candidate["passed"]
                and candidate_score - base_score >= 12.0
                and candidate_penalty <= 6.0
                and all(candidate["critical"].values())
            ),
            "allow_general_claim": False,
            "next_gate": "validation_8_after_embedded_short_ir_decoder_and_length_controller",
        },
        "claim_limit": "候选含确定性尾部编译，且只是一条 train 开发题；不能宣称 v38 或 v47 模型能力已经晋级。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

