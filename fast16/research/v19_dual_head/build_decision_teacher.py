"""Build a predeclared decision-token teacher before evaluating new candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def detokenize(endpoint: str, token_id: int) -> str:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/detokenize",
        data=json.dumps({"tokens": [token_id]}).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.load(response)
    return str(value.get("content", ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8105")
    args = parser.parse_args()

    rows = read_jsonl(args.teacher)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    for task_rows in by_task.values():
        task_rows.sort(key=lambda row: int(row["token_index"]))

    token_text: dict[int, str] = {}
    for row in rows:
        token_id = int(row["target_token_id"])
        if token_id not in token_text:
            token_text[token_id] = detokenize(args.endpoint, token_id)

    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(by_task.items()):
        is_tool = task_id.startswith("tool-")
        rendered = [token_text[int(row["target_token_id"])] for row in task_rows]
        cumulative = ""
        target_text = "".join(rendered)
        spans: list[tuple[int, int, str, str]] = []
        if is_tool:
            tool_name = task_id.removeprefix("tool-").replace("-", "_")
            field_names = ["arguments", "path", "pattern", "command", "query", "content"]
            values = ["src/main.rs", "src/**/*.ts", "notes.txt", "cargo check", "TODO", "hello"]
            for value, category, reason in [
                (tool_name, "tool_name", "identifies the required tool"),
                ("</tool_call>", "termination_marker", "closes the structured tool call"),
            ]:
                start = target_text.find(value)
                if start >= 0:
                    spans.append((start, start + len(value), category, reason))
            for value in field_names:
                start = target_text.find(f'"{value}"')
                if start >= 0:
                    spans.append((start + 1, start + 1 + len(value), "argument_name", "selects a required schema field"))
            for value in values:
                start = target_text.find(value)
                if start >= 0:
                    spans.append((start, start + len(value), "argument_value", "forms a required argument value"))
        offset = 0
        for row, text in zip(task_rows, rendered):
            token_start = offset
            token_end = offset + len(text)
            offset = token_end
            cumulative += text
            category: str | None = None
            reason: str | None = None
            if is_tool:
                matches = [span for span in spans if token_start < span[1] and token_end > span[0]]
                if matches:
                    priority = {"tool_name": 0, "argument_name": 1, "argument_value": 2, "termination_marker": 3}
                    match = min(matches, key=lambda span: priority[span[2]])
                    category = match[2]
                    reason = match[3]
            else:
                stripped = text.strip()
                if any(symbol in text for symbol in ("<=", "<", ">", "+", "-", "//")):
                    category = "operator"
                    reason = "implements a core comparison or arithmetic decision"
                elif any(keyword in text for keyword in ("while", "if", "else")):
                    category = "boundary_condition"
                    reason = "controls a boundary or branch"
                elif "return" in text or stripped in {"True", "False", "-1"}:
                    category = "return_value"
                    reason = "determines the returned result"
                elif any(api in text for api in ("max", "min", "iter", "copied", "Set")):
                    category = "critical_api"
                    reason = "selects the key implementation API"
            if category is not None:
                annotated = dict(row)
                annotated["decision_category"] = category
                annotated["token_text"] = text
                annotated["selection_reason"] = reason
                selected.append(annotated)
                audit.append(
                    {
                        "sample_id": row["sample_id"],
                        "task_id": task_id,
                        "token_index": row["token_index"],
                        "target_token_id": row["target_token_id"],
                        "token_text": text,
                        "category": category,
                        "reason": reason,
                    }
                )

    required_code = {"operator", "boundary_condition", "return_value", "critical_api"}
    required_tool = {"tool_name", "argument_name", "argument_value", "termination_marker"}
    found_code = {row["decision_category"] for row in selected if not str(row["task_id"]).startswith("tool-")}
    found_tool = {row["decision_category"] for row in selected if str(row["task_id"]).startswith("tool-")}
    if not required_code.issubset(found_code):
        raise RuntimeError(f"missing code categories: {sorted(required_code - found_code)}")
    if not required_tool.issubset(found_tool):
        raise RuntimeError(f"missing tool categories: {sorted(required_tool - found_tool)}")

    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )
    report = {
        "format": "colorlm-v19.1-decision-token-selection-v1",
        "source_teacher": str(args.teacher.resolve()),
        "selection_timing": "before evaluating any candidate after alpha=0.03 smoke failure",
        "candidate_results_used": False,
        "sample_count": len(selected),
        "task_count": len({row["task_id"] for row in selected}),
        "code_categories": sorted(found_code),
        "tool_categories": sorted(found_tool),
        "teacher_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "audit": audit,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "audit"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
