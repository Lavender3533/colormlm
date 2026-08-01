"""ColorLM 多能力短门的纯离线精确判分器。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TASKS = Path(__file__).with_name("multicap_short_gate_v1.json")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: 非法 JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: 每行必须是 JSON 对象")
        rows.append(row)
    return rows


def exact_json(output: Any, expected: Any) -> tuple[bool, str]:
    if not isinstance(output, str):
        return False, "缺少文本 output"
    try:
        actual = json.loads(output)
    except json.JSONDecodeError as error:
        return False, f"output 不是纯 JSON: {error.msg}"
    if actual != expected:
        return False, f"JSON 不匹配: actual={actual!r}"
    return True, "exact_json"


def exact_text(output: Any, expected: str) -> tuple[bool, str]:
    if not isinstance(output, str):
        return False, "缺少文本 output"
    actual = output.rstrip("\r\n")
    if actual != expected:
        return False, f"文本不匹配: actual={actual!r}"
    return True, "exact_text"


def normalize_tool_call(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if isinstance(row.get("tool_call"), dict):
        return row["tool_call"], "tool_call"
    calls = row.get("tool_calls")
    if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict):
        call = calls[0]
        if isinstance(call.get("function"), dict):
            function = call["function"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return None, "tool_calls[0].function.arguments 不是合法 JSON"
            return {"name": function.get("name"), "arguments": arguments}, "openai"
        return call, "tool_calls"
    return None, "缺少唯一工具调用"


def tool_call(row: dict[str, Any], validator: dict[str, Any]) -> tuple[bool, str]:
    call, source = normalize_tool_call(row)
    if call is None:
        return False, source
    expected = {
        "name": validator["expected_name"],
        "arguments": validator["expected_arguments"],
    }
    actual = {"name": call.get("name"), "arguments": call.get("arguments")}
    if actual != expected:
        return False, f"工具调用不匹配: actual={actual!r}"
    if row.get("finish_reason") != validator["expected_finish_reason"]:
        return False, f"finish_reason={row.get('finish_reason')!r}"
    if row.get("output") not in (None, ""):
        return False, "工具调用前后出现了额外文本"
    return True, f"tool_call:{source}"


def grade_row(task: dict[str, Any], row: dict[str, Any]) -> tuple[bool, str]:
    validator = task["validator"]
    kind = validator["type"]
    if kind == "exact_json":
        return exact_json(row.get("output"), validator["expected"])
    if kind == "exact_text":
        return exact_text(row.get("output"), validator["expected"])
    if kind == "tool_call":
        return tool_call(row, validator)
    raise ValueError(f"未知 validator: {kind}")


def select_tasks(
    items: list[dict[str, Any]], one_per_dimension: bool
) -> list[dict[str, Any]]:
    if not one_per_dimension:
        return items
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if item["dimension"] in seen:
            continue
        seen.add(item["dimension"])
        selected.append(item)
    return selected


def score(
    tasks_path: Path,
    responses_path: Path,
    one_per_dimension: bool = False,
) -> dict[str, Any]:
    bank = load_json(tasks_path)
    selected = select_tasks(bank["items"], one_per_dimension)
    tasks = {item["id"]: item for item in selected}
    input_rows = load_jsonl(responses_path)
    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    for row in input_rows:
        task_id = row.get("id")
        if task_id not in tasks:
            unknown.append(str(task_id))
            continue
        if task_id in rows_by_id:
            duplicates.append(task_id)
            continue
        rows_by_id[task_id] = row

    results: list[dict[str, Any]] = []
    per_dimension: dict[str, list[bool]] = defaultdict(list)
    for task_id, task in tasks.items():
        row = rows_by_id.get(task_id)
        if row is None:
            passed, detail = False, "缺少响应记录"
        else:
            passed, detail = grade_row(task, row)
        per_dimension[task["dimension"]].append(passed)
        results.append(
            {
                "id": task_id,
                "dimension": task["dimension"],
                "passed": passed,
                "detail": detail,
            }
        )

    passed_count = sum(row["passed"] for row in results)
    return {
        "format": "colorlm-multicap-short-gate-score-v1",
        "selection": "first_per_dimension" if one_per_dimension else "full",
        "tasks": str(tasks_path.resolve()),
        "responses": str(responses_path.resolve()),
        "complete": not duplicates and not unknown and len(rows_by_id) == len(tasks),
        "summary": {
            "passed": passed_count,
            "total": len(results),
            "per_dimension": {
                dimension: {"passed": sum(marks), "total": len(marks)}
                for dimension, marks in sorted(per_dimension.items())
            },
        },
        "input_errors": {"duplicates": sorted(duplicates), "unknown": sorted(unknown)},
        "results": results,
    }


def compare(
    tasks_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    target_dimensions: list[str],
    one_per_dimension: bool = False,
) -> dict[str, Any]:
    baseline = score(tasks_path, baseline_path, one_per_dimension)
    candidate = score(tasks_path, candidate_path, one_per_dimension)
    base_rows = {row["id"]: row for row in baseline["results"]}
    cand_rows = {row["id"]: row for row in candidate["results"]}

    wins: list[str] = []
    regressions: list[str] = []
    unchanged_pass: list[str] = []
    unchanged_fail: list[str] = []
    dimension_delta: dict[str, int] = {}
    dimensions = baseline["summary"]["per_dimension"]
    for dimension, base_value in dimensions.items():
        cand_value = candidate["summary"]["per_dimension"][dimension]
        dimension_delta[dimension] = cand_value["passed"] - base_value["passed"]
    for task_id in base_rows:
        before = base_rows[task_id]["passed"]
        after = cand_rows[task_id]["passed"]
        if not before and after:
            wins.append(task_id)
        elif before and not after:
            regressions.append(task_id)
        elif before:
            unchanged_pass.append(task_id)
        else:
            unchanged_fail.append(task_id)

    missing_target_dimensions = sorted(set(target_dimensions) - set(dimensions))
    target_nonregression = not missing_target_dimensions and all(
        dimension_delta[name] >= 0 for name in target_dimensions
    )
    target_gain = not missing_target_dimensions and sum(
        dimension_delta[name] for name in target_dimensions
    ) >= 1
    complete = baseline["complete"] and candidate["complete"]
    generation_gate_pass = (
        complete
        and not regressions
        and len(wins) >= 1
        and target_nonregression
        and target_gain
    )
    multidimensional_claim = (
        generation_gate_pass
        and sum(delta > 0 for delta in dimension_delta.values()) >= 3
    )

    return {
        "format": "colorlm-multicap-short-gate-comparison-v1",
        "baseline": baseline,
        "candidate": candidate,
        "paired": {
            "wins": wins,
            "regressions": regressions,
            "unchanged_pass": unchanged_pass,
            "unchanged_fail": unchanged_fail,
            "dimension_delta": dimension_delta,
        },
        "target_dimensions": target_dimensions,
        "missing_target_dimensions": missing_target_dimensions,
        "decision": {
            "generation_gate_pass": generation_gate_pass,
            "multidimensional_claim": multidimensional_claim,
            "external_requirements_still_needed": [
                "关键 token 反事实 NLL 门",
                "leave-one-task-out 方向门",
                "alpha=0 物理旁路逐 token 等价",
                "能力通过后的相邻资源和速度检查"
            ],
        },
    }


def emit(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="ColorLM 多能力短门离线判分")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scorer = subparsers.add_parser("score")
    scorer.add_argument("--responses", type=Path, required=True)
    scorer.add_argument("--one-per-dimension", action="store_true")
    scorer.add_argument("--out", type=Path)

    comparer = subparsers.add_parser("compare")
    comparer.add_argument("--baseline", type=Path, required=True)
    comparer.add_argument("--candidate", type=Path, required=True)
    comparer.add_argument("--one-per-dimension", action="store_true")
    comparer.add_argument(
        "--target-dimension",
        action="append",
        default=[],
        help="供体负责的目标维度；K3 建议分别传 tools 和 planning",
    )
    comparer.add_argument("--out", type=Path)

    args = parser.parse_args()
    if args.command == "score":
        emit(score(args.tasks, args.responses, args.one_per_dimension), args.out)
    else:
        targets = args.target_dimension or ["tools", "planning"]
        emit(
            compare(
                args.tasks,
                args.baseline,
                args.candidate,
                targets,
                args.one_per_dimension,
            ),
            args.out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
