"""Offline exact validator for the frozen v28 post-tool-result state gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_call(row: dict[str, Any]) -> dict[str, Any] | None:
    calls = row.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    function = calls[0].get("function")
    if not isinstance(function, dict):
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return {"name": function.get("name"), "arguments": arguments}


def grade(item: dict[str, Any], row: dict[str, Any]) -> tuple[bool, str]:
    validator = item["validator"]
    if validator["type"] == "exact_json":
        if row.get("tool_calls"):
            return False, "应结束任务却继续调用工具"
        try:
            actual = json.loads(row.get("output", ""))
        except json.JSONDecodeError as error:
            return False, f"输出不是纯JSON: {error.msg}"
        return (actual == validator["expected"], f"actual={actual!r}")
    call = normalize_call(row)
    if call is None:
        return False, "缺少唯一合法工具调用"
    expected = {
        "name": validator["expected_name"],
        "arguments": validator["expected_arguments"],
    }
    if call != expected:
        return False, f"工具调用不匹配: actual={call!r}"
    if row.get("finish_reason") != validator["expected_finish_reason"]:
        return False, f"finish_reason={row.get('finish_reason')!r}"
    if row.get("output") not in (None, ""):
        return False, "工具调用附带额外文本"
    return True, "exact_tool_call"


def main() -> int:
    parser = argparse.ArgumentParser(description="离线判分v28工具结果状态门")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in load_jsonl(args.responses)}
    results: list[dict[str, Any]] = []
    splits: dict[str, list[bool]] = defaultdict(list)
    for item in contract["items"]:
        row = rows.get(item["id"])
        passed, detail = (False, "缺少响应") if row is None else grade(item, row)
        splits[item["split"]].append(passed)
        results.append({"id": item["id"], "split": item["split"], "passed": passed, "detail": detail})
    report = {
        "format": "colorlm-agent-state-gate-score-v1",
        "summary": {"passed": sum(r["passed"] for r in results), "total": len(results)},
        "per_split": {name: {"passed": sum(marks), "total": len(marks)} for name, marks in sorted(splits.items())},
        "results": results,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
