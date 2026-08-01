"""把外部对照臂(如 Opus 无工具作答)的答案judged成 run_eval.py 兼容格式。

判分复用 run_eval.py 的 normalize/grade,不另写一套,避免"自己给自己打分"。
输入 JSON: [{"id": "q000", "answer": "52", "confidence": "high", "work": "..."}, ...]
输出与 run_eval.py run 子命令一致,可直接喂给 run_eval.py compare。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_eval import grade, wilson_interval


def main() -> int:
    parser = argparse.ArgumentParser(description="外部对照臂判分")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--arm", type=str, required=True)
    parser.add_argument(
        "--items",
        type=Path,
        default=Path(__file__).resolve().parent / "frozen_v1.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bank = json.loads(args.items.read_text(encoding="utf-8"))
    truth = {item["id"]: item for item in bank["items"]}
    raw = json.loads(args.answers.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("answers", [])
    given = {row["id"]: row for row in raw}

    results = []
    correct_count = 0
    for item_id, item in truth.items():
        row = given.get(item_id)
        got = row.get("answer") if row else None
        is_correct = grade(item["answer"], got, item["answer_type"])
        correct_count += is_correct
        results.append(
            {
                "id": item_id,
                "family": item["family"],
                "expected": item["answer"],
                "got": got,
                "correct": is_correct,
                "finish_reason": "stop" if row else "missing",
                "completion_tokens": None,
                "seconds": None,
                "content": (row or {}).get("work", ""),
                "reasoning": "",
                "confidence": (row or {}).get("confidence"),
            }
        )

    total = len(results)
    low, high = wilson_interval(correct_count, total)
    families: dict[str, list[bool]] = {}
    for row in results:
        families.setdefault(row["family"], []).append(row["correct"])
    summary = {
        "mode": args.arm,
        "budget": None,
        "temperature": None,
        "total": total,
        "correct": correct_count,
        "accuracy": round(correct_count / total, 4) if total else None,
        "wilson_95": [round(low, 4), round(high, 4)],
        "per_family": {
            family: f"{sum(marks)}/{len(marks)}"
            for family, marks in sorted(families.items())
        },
        "seconds_total": None,
    }
    args.out.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
