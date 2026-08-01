#!/usr/bin/env python3
"""不调用模型的 capability_eval schema、validator 与晋级规则自测。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import validate as gate


ROOT = Path(__file__).resolve().parent


def response_for(task: dict[str, Any], run_id: str, model_id: str, pass_task: bool = True) -> dict[str, Any]:
    validator = task["validator"]
    if validator["type"] == "exact_json":
        output = json.dumps(validator["expected"], ensure_ascii=False, separators=(",", ":")) if pass_task else "null"
        calls: list[dict[str, Any]] = []
        finish_reason = "stop"
    elif validator["type"] == "exact_text":
        output = validator["expected"] if pass_task else "__wrong__"
        calls = []
        finish_reason = "stop"
    else:
        output = "" if pass_task else "不应出现的文本"
        calls = [{"name": validator["expected_name"], "arguments": validator["expected_arguments"]}]
        finish_reason = "tool_calls"
    return {
        "schema_version": "capability-response-v1",
        "run_id": run_id,
        "model_id": model_id,
        "task_id": task["id"],
        "output": output,
        "tool_calls": calls,
        "finish_reason": finish_reason,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "latency_ms": 1.0,
        "generation": {
            "temperature": 0,
            "seed": 17,
            "template_id": "selftest-template-v1",
            "binary_id": "selftest-binary",
            "runtime_id": "cpu-selftest",
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def assert_json_schema(instance: Any, schema_name: str) -> str:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return "internal-strict-checks"
    schema = gate.read_json(ROOT / "schemas" / schema_name)
    jsonschema.Draft202012Validator(schema).validate(instance)
    return "jsonschema+internal-strict-checks"


def main() -> int:
    gate.check_schemas()
    all_paths = [ROOT / "data" / f"{split}.jsonl" for split in gate.SPLIT_COUNTS]
    all_tasks = gate.validate_task_files(all_paths, require_full_bank=True)
    assert len(all_tasks) == 48
    schema_backend = "internal-strict-checks"
    for task in all_tasks:
        schema_backend = assert_json_schema(task, "task.schema.json")

    with tempfile.TemporaryDirectory(prefix="capability-eval-selftest-") as temporary:
        temp = Path(temporary)
        task_tasks = gate.read_jsonl(ROOT / "data" / "task_holdout.jsonl")
        final_tasks = gate.read_jsonl(ROOT / "data" / "final_holdout.jsonl")

        baseline_task_failures = {
            "task_holdout_reasoning_01",
            "task_holdout_knowledge_01",
            "task_holdout_long_context_01",
            "task_holdout_coding_01",
        }
        baseline_final_failures = {"final_holdout_reasoning_01", "final_holdout_knowledge_01"}

        paths = {
            "task_base": temp / "task_base.jsonl",
            "task_candidate": temp / "task_candidate.jsonl",
            "final_base": temp / "final_base.jsonl",
            "final_candidate": temp / "final_candidate.jsonl",
        }
        write_jsonl(
            paths["task_base"],
            [response_for(task, "base-task", "ColorLM-v17", task["id"] not in baseline_task_failures) for task in task_tasks],
        )
        write_jsonl(
            paths["task_candidate"],
            [response_for(task, "candidate-task", "ColorLM-candidate", True) for task in task_tasks],
        )
        write_jsonl(
            paths["final_base"],
            [response_for(task, "base-final", "ColorLM-v17", task["id"] not in baseline_final_failures) for task in final_tasks],
        )
        write_jsonl(
            paths["final_candidate"],
            [response_for(task, "candidate-final", "ColorLM-candidate", True) for task in final_tasks],
        )

        sample_response = gate.read_jsonl(paths["task_candidate"])[0]
        gate.validate_response(sample_response, paths["task_candidate"], 1)
        schema_backend = assert_json_schema(sample_response, "response.schema.json")

        task_report = gate.compare_report(ROOT / "data" / "task_holdout.jsonl", paths["task_base"], paths["task_candidate"])
        gate.validate_report(task_report)
        assert task_report["decision"]["status"] == "task_holdout_pass"
        assert task_report["split_reports"][0]["loto"]["all_folds_positive"] is True
        assert_json_schema(task_report, "report.schema.json")

        promotion = gate.promotion_report(
            paths["task_base"], paths["task_candidate"], paths["final_base"], paths["final_candidate"]
        )
        gate.validate_report(promotion)
        assert promotion["decision"]["promotable"] is True
        assert promotion["decision"]["status"] == "promote"
        assert_json_schema(promotion, "report.schema.json")

        regression_rows = [response_for(task, "regression", "ColorLM-regression", True) for task in task_tasks]
        regression_target = next(task for task in task_tasks if task["id"] == "task_holdout_tools_01")
        regression_rows[task_tasks.index(regression_target)] = response_for(
            regression_target, "regression", "ColorLM-regression", False
        )
        regression_path = temp / "regression.jsonl"
        write_jsonl(regression_path, regression_rows)
        rejected = gate.compare_report(ROOT / "data" / "task_holdout.jsonl", paths["task_candidate"], regression_path)
        assert rejected["decision"]["status"] == "task_holdout_fail"
        assert rejected["split_reports"][0]["paired"]["regressions"] == ["task_holdout_tools_01"]

        incomplete_path = temp / "incomplete.jsonl"
        write_jsonl(incomplete_path, gate.read_jsonl(paths["task_candidate"])[:-1])
        incomplete = gate.score_run(ROOT / "data" / "task_holdout.jsonl", incomplete_path)
        assert incomplete["complete"] is False
        assert incomplete["passed"] == 15

        tool_task = next(task for task in task_tasks if task["id"] == "task_holdout_tools_01")
        polluted = response_for(tool_task, "polluted", "ColorLM-candidate", True)
        polluted["output"] = "正在调用工具"
        passed, detail = gate.grade(tool_task, polluted)
        assert passed is False and detail == "工具调用夹带文本"

        duplicate_path = temp / "duplicate.jsonl"
        good_rows = gate.read_jsonl(paths["task_candidate"])
        write_jsonl(duplicate_path, good_rows + [good_rows[0]])
        duplicate = gate.score_run(ROOT / "data" / "task_holdout.jsonl", duplicate_path)
        assert duplicate["complete"] is False
        assert duplicate["input_errors"]["duplicates"] == [good_rows[0]["task_id"]]

    print(
        json.dumps(
            {
                "ok": True,
                "tests": 8,
                "task_count": len(all_tasks),
                "schema_backend": schema_backend,
                "model_started": False,
                "gpu_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
