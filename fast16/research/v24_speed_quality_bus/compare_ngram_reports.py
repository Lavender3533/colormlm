"""比较none与ngram-mod短门，等价输出是速度结论的前置条件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_CONTRACT_SHA256 = "e42a27c855825187cfb31b15b71d10f8ac10b61f00c9979f5992bb5f8f81fdaa"


def load(path: Path, mode: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("format") != "colorlm-v24-ngram-gate-report-v1"
        or value.get("mode") != mode
        or value.get("contract_sha256") != EXPECTED_CONTRACT_SHA256
    ):
        raise RuntimeError(f"报告契约不匹配: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="比较v24 n-gram短门报告")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline, "none")
    candidate = load(args.candidate, "ngram-mod")
    base_rows = {row["id"]: row for row in baseline["tasks"]}
    cand_rows = {row["id"]: row for row in candidate["tasks"]}
    if base_rows.keys() != cand_rows.keys():
        raise RuntimeError("任务集合不一致")
    tasks = []
    exact = True
    for task_id in base_rows:
        base = base_rows[task_id]
        cand = cand_rows[task_id]
        output_equal = (
            base["message_sha256"] == cand["message_sha256"]
            and base["finish_reason"] == cand["finish_reason"]
            and base["completion_tokens"] == cand["completion_tokens"]
        )
        exact = exact and output_equal
        base_tps = float(base["client_tokens_per_second"])
        cand_tps = float(cand["client_tokens_per_second"])
        tasks.append(
            {
                "id": task_id,
                "output_exact": output_equal,
                "baseline_tokens_per_second": base_tps,
                "candidate_tokens_per_second": cand_tps,
                "relative_percent": 100.0 * (cand_tps / base_tps - 1.0),
            }
        )

    base_tps = float(baseline["summary"]["client_tokens_per_second"])
    cand_tps = float(candidate["summary"]["client_tokens_per_second"])
    relative = 100.0 * (cand_tps / base_tps - 1.0)
    # 单次短门只允许保留候选，不能证明稳定提速。低于3%视作没有值得保留的工程信号。
    decision = "retain_candidate" if exact and relative >= 3.0 else "reject"
    report = {
        "format": "colorlm-v24-ngram-ab-report-v1",
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "output_exact": exact,
        "tasks": tasks,
        "summary": {
            "baseline_tokens_per_second": base_tps,
            "candidate_tokens_per_second": cand_tps,
            "relative_percent": relative,
            "decision": decision,
            "claim_limit": "single adjacent short A/B; not a stable broad speed claim",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"] | {"output_exact": exact}, ensure_ascii=False))
    return 0 if decision == "retain_candidate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
