#!/usr/bin/env python3
"""ColorLM 八维短门的纯 CPU、离线、确定性校验与判分器。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SCHEMAS = ROOT / "schemas"
DIMENSIONS = (
    "reasoning",
    "knowledge",
    "long_context",
    "coding",
    "tools",
    "planning",
    "computer_use",
    "communication",
)
SPLIT_COUNTS = {"dev": 3, "task_holdout": 2, "final_holdout": 1}
TASK_REQUIRED = {
    "schema_version",
    "id",
    "dimension",
    "split",
    "family",
    "input",
    "reference_answer",
    "validator",
    "critical_decision_tokens",
    "failure_conditions",
}
RESPONSE_REQUIRED = {
    "schema_version",
    "run_id",
    "model_id",
    "task_id",
    "output",
    "tool_calls",
    "finish_reason",
    "usage",
    "latency_ms",
    "generation",
}
RESPONSE_ALLOWED = RESPONSE_REQUIRED | {"critical_token_observations", "error"}
REPORT_REQUIRED = {"schema_version", "created_utc", "mode", "split_reports", "decision"}


class ValidationError(ValueError):
    """可直接向操作者报告的数据错误。"""


def read_utf8(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValidationError(f"无法读取 {path}: {error}") from error
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValidationError(f"{path}: 禁止 UTF-8 BOM")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError(f"{path}: 不是合法 UTF-8: {error}") from error


def read_json(path: Path) -> Any:
    try:
        return json.loads(read_utf8(path))
    except json.JSONDecodeError as error:
        raise ValidationError(f"{path}: 非法 JSON: {error}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(read_utf8(path).splitlines(), 1):
        if not line.strip():
            raise ValidationError(f"{path}:{line_number}: JSONL 不允许空行")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValidationError(f"{path}:{line_number}: 非法 JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValidationError(f"{path}:{line_number}: 每行必须是 JSON 对象")
        rows.append(row)
    if not rows:
        raise ValidationError(f"{path}: 空 JSONL")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(condition: bool, message: str) -> None:
    if condition:
        raise ValidationError(message)


def validate_task(task: dict[str, Any], source: Path, line: int) -> None:
    prefix = f"{source}:{line}"
    fail(set(task) != TASK_REQUIRED, f"{prefix}: task 字段集合不符合 schema")
    fail(task["schema_version"] != "capability-task-v1", f"{prefix}: schema_version 错误")
    fail(task["dimension"] not in DIMENSIONS, f"{prefix}: 未知维度 {task['dimension']!r}")
    fail(task["split"] not in SPLIT_COUNTS, f"{prefix}: 未知 split {task['split']!r}")
    fail(not task["id"].startswith(task["split"] + "_"), f"{prefix}: id 与 split 不一致")
    fail(not isinstance(task["family"], str) or not task["family"], f"{prefix}: family 为空")

    model_input = task["input"]
    fail(not isinstance(model_input, dict), f"{prefix}: input 必须是对象")
    fail(set(model_input) - {"messages", "max_output_tokens", "temperature", "tools"}, f"{prefix}: input 有未知字段")
    fail(not isinstance(model_input.get("messages"), list) or not model_input["messages"], f"{prefix}: messages 为空")
    for message in model_input["messages"]:
        fail(set(message) != {"role", "content"}, f"{prefix}: message 字段错误")
        fail(message["role"] not in {"system", "user"}, f"{prefix}: message role 错误")
        fail(not isinstance(message["content"], str) or not message["content"], f"{prefix}: message content 为空")
    fail(model_input.get("temperature") != 0, f"{prefix}: temperature 必须为 0")
    max_tokens = model_input.get("max_output_tokens")
    fail(not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 256, f"{prefix}: max_output_tokens 越界")

    validator = task["validator"]
    fail(not isinstance(validator, dict), f"{prefix}: validator 必须是对象")
    kind = validator.get("type")
    fail(kind not in {"exact_json", "exact_text", "tool_call"}, f"{prefix}: validator type 错误")
    if kind in {"exact_json", "exact_text"}:
        fail(set(validator) != {"type", "expected"}, f"{prefix}: {kind} 字段错误")
        fail(validator["expected"] != task["reference_answer"], f"{prefix}: reference_answer 与 validator 不一致")
        if kind == "exact_text":
            fail(not isinstance(validator["expected"], str), f"{prefix}: exact_text expected 必须是字符串")
    else:
        expected_fields = {"type", "expected_name", "expected_arguments", "expected_finish_reason"}
        fail(set(validator) != expected_fields, f"{prefix}: tool_call 字段错误")
        expected_call = {"name": validator["expected_name"], "arguments": validator["expected_arguments"]}
        fail(task["reference_answer"] != expected_call, f"{prefix}: tool reference_answer 不一致")
        fail(validator["expected_finish_reason"] != "tool_calls", f"{prefix}: 工具调用必须以 tool_calls 结束")
        fail("tools" not in model_input, f"{prefix}: 工具题未声明 tools")

    tokens = task["critical_decision_tokens"]
    fail(not isinstance(tokens, list) or not tokens, f"{prefix}: critical_decision_tokens 为空")
    token_ids: set[str] = set()
    for token in tokens:
        fail(set(token) != {"id", "token_text", "rationale"}, f"{prefix}: decision token 字段错误")
        fail(any(not isinstance(token[key], str) or not token[key] for key in token), f"{prefix}: decision token 值为空")
        fail(token["id"] in token_ids, f"{prefix}: decision token id 重复")
        token_ids.add(token["id"])
    conditions = task["failure_conditions"]
    fail(not isinstance(conditions, list) or not conditions, f"{prefix}: failure_conditions 为空")
    fail(any(not isinstance(item, str) or not item for item in conditions), f"{prefix}: failure condition 为空")


def validate_task_files(paths: Iterable[Path], require_full_bank: bool) -> list[dict[str, Any]]:
    paths = list(paths)
    fail(not paths, "至少需要一个 task 文件")
    tasks: list[dict[str, Any]] = []
    ids: set[str] = set()
    families: set[str] = set()
    counts: dict[tuple[str, str], int] = defaultdict(int)
    seen_splits: set[str] = set()
    for path in paths:
        rows = read_jsonl(path)
        file_splits = {row.get("split") for row in rows}
        fail(len(file_splits) != 1, f"{path}: 一个文件只能包含一个 split")
        split = next(iter(file_splits))
        seen_splits.add(str(split))
        expected_name = f"{split}.jsonl"
        fail(path.name != expected_name, f"{path}: 文件名必须为 {expected_name}")
        for line, task in enumerate(rows, 1):
            validate_task(task, path, line)
            fail(task["id"] in ids, f"{path}:{line}: 重复 task id {task['id']}")
            fail(task["family"] in families, f"{path}:{line}: 跨 split 重复 family {task['family']}")
            ids.add(task["id"])
            families.add(task["family"])
            counts[(task["split"], task["dimension"])] += 1
            tasks.append(task)
    for split in seen_splits:
        for dimension in DIMENSIONS:
            expected = SPLIT_COUNTS[split]
            actual = counts[(split, dimension)]
            fail(actual != expected, f"{split}/{dimension}: 应有 {expected} 题，实际 {actual} 题")
    if require_full_bank:
        fail(seen_splits != set(SPLIT_COUNTS), f"完整题库必须包含 {sorted(SPLIT_COUNTS)}")
        for dimension in DIMENSIONS:
            total = sum(counts[(split, dimension)] for split in SPLIT_COUNTS)
            fail(not 5 <= total <= 10, f"{dimension}: 每维总题数必须为 5-10，实际 {total}")
    return tasks


def validate_response(row: dict[str, Any], source: Path, line: int) -> None:
    prefix = f"{source}:{line}"
    fail(not RESPONSE_REQUIRED <= set(row), f"{prefix}: 缺少 response 必填字段")
    fail(set(row) - RESPONSE_ALLOWED, f"{prefix}: response 有未知字段")
    fail(row["schema_version"] != "capability-response-v1", f"{prefix}: schema_version 错误")
    for key in ("run_id", "model_id", "task_id"):
        fail(not isinstance(row[key], str) or not row[key], f"{prefix}: {key} 为空")
    fail(row["output"] is not None and not isinstance(row["output"], str), f"{prefix}: output 类型错误")
    fail(not isinstance(row["tool_calls"], list), f"{prefix}: tool_calls 必须是数组")
    for call in row["tool_calls"]:
        fail(set(call) != {"name", "arguments"}, f"{prefix}: tool_call 字段错误")
        fail(not isinstance(call["name"], str) or not isinstance(call["arguments"], dict), f"{prefix}: tool_call 类型错误")
    fail(row["finish_reason"] not in {"stop", "tool_calls", "length", "error"}, f"{prefix}: finish_reason 错误")
    usage = row["usage"]
    fail(set(usage) != {"prompt_tokens", "completion_tokens"}, f"{prefix}: usage 字段错误")
    for key in usage:
        fail(not isinstance(usage[key], int) or isinstance(usage[key], bool) or usage[key] < 0, f"{prefix}: usage 值错误")
    fail(not isinstance(row["latency_ms"], (int, float)) or isinstance(row["latency_ms"], bool) or row["latency_ms"] < 0, f"{prefix}: latency_ms 错误")
    generation = row["generation"]
    required_generation = {"temperature", "seed", "template_id"}
    allowed_generation = required_generation | {"binary_id", "runtime_id"}
    fail(not required_generation <= set(generation) or set(generation) - allowed_generation, f"{prefix}: generation 字段错误")
    fail(generation["temperature"] != 0, f"{prefix}: response temperature 必须为 0")
    fail(not isinstance(generation["seed"], int) or isinstance(generation["seed"], bool), f"{prefix}: seed 必须为整数")
    fail(not isinstance(generation["template_id"], str) or not generation["template_id"], f"{prefix}: template_id 为空")
    observations = row.get("critical_token_observations")
    if observations is not None:
        fail(not isinstance(observations, list), f"{prefix}: critical_token_observations 必须是数组")
        for item in observations:
            fail(set(item) != {"id", "token_text", "token_id", "nll"}, f"{prefix}: token observation 字段错误")
            fail(not isinstance(item["token_id"], int) or item["token_id"] < 0, f"{prefix}: token_id 错误")
            fail(not isinstance(item["nll"], (int, float)) or item["nll"] < 0, f"{prefix}: nll 错误")


def validate_report(report: dict[str, Any]) -> None:
    fail(set(report) != REPORT_REQUIRED, "report 字段集合不符合 schema")
    fail(report["schema_version"] != "capability-report-v1", "report schema_version 错误")
    fail(report["mode"] not in {"score", "compare", "promote"}, "report mode 错误")
    fail(not isinstance(report["created_utc"], str) or not report["created_utc"].endswith("Z"), "created_utc 错误")
    splits = report["split_reports"]
    fail(not isinstance(splits, list) or not splits, "split_reports 为空")
    split_fields = {"split", "task_file", "task_sha256", "complete", "baseline", "candidate", "paired", "loto", "gate"}
    for split in splits:
        fail(set(split) != split_fields, "split_report 字段集合不符合 schema")
        fail(split["split"] not in SPLIT_COUNTS, "split_report split 错误")
        fail(not isinstance(split["task_file"], str), "split_report task_file 错误")
        digest = split["task_sha256"]
        fail(not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest), "task_sha256 错误")
        fail(not isinstance(split["complete"], bool), "split_report complete 错误")
        fail(split["baseline"] is not None and not isinstance(split["baseline"], dict), "baseline 类型错误")
        fail(not isinstance(split["candidate"], dict), "candidate 类型错误")
        fail(split["paired"] is not None and not isinstance(split["paired"], dict), "paired 类型错误")
        fail(split["loto"] is not None and not isinstance(split["loto"], dict), "loto 类型错误")
        fail(not isinstance(split["gate"], dict), "gate 类型错误")
    decision = report["decision"]
    fail(set(decision) != {"promotable", "status", "reasons", "claim_ceiling"}, "decision 字段集合错误")
    fail(not isinstance(decision["promotable"], bool), "decision promotable 错误")
    statuses = {
        "score_only",
        "dev_only",
        "task_holdout_pass",
        "task_holdout_fail",
        "final_holdout_pass",
        "final_holdout_fail",
        "promote",
        "reject",
    }
    fail(decision["status"] not in statuses, "decision status 错误")
    fail(not isinstance(decision["reasons"], list) or any(not isinstance(item, str) for item in decision["reasons"]), "decision reasons 错误")
    fail(not isinstance(decision["claim_ceiling"], str), "decision claim_ceiling 错误")


def grade(task: dict[str, Any], response: dict[str, Any]) -> tuple[bool, str]:
    validator = task["validator"]
    kind = validator["type"]
    if response["finish_reason"] in {"length", "error"}:
        return False, f"finish_reason={response['finish_reason']}"
    if kind == "exact_json":
        if response["tool_calls"]:
            return False, "普通回答出现工具调用"
        if not isinstance(response["output"], str):
            return False, "缺少文本 output"
        try:
            actual = json.loads(response["output"])
        except json.JSONDecodeError as error:
            return False, f"output 不是纯 JSON: {error.msg}"
        if actual != validator["expected"]:
            return False, f"JSON 不匹配: actual={actual!r}"
        if response["finish_reason"] != "stop":
            return False, f"finish_reason={response['finish_reason']!r}"
        return True, "exact_json"
    if kind == "exact_text":
        if response["tool_calls"]:
            return False, "普通回答出现工具调用"
        if not isinstance(response["output"], str):
            return False, "缺少文本 output"
        actual = response["output"].rstrip("\r\n")
        if actual != validator["expected"]:
            return False, f"文本不匹配: actual={actual!r}"
        if response["finish_reason"] != "stop":
            return False, f"finish_reason={response['finish_reason']!r}"
        return True, "exact_text"

    if len(response["tool_calls"]) != 1:
        return False, f"工具调用数必须为 1，实际 {len(response['tool_calls'])}"
    call = response["tool_calls"][0]
    expected = {"name": validator["expected_name"], "arguments": validator["expected_arguments"]}
    if call != expected:
        return False, f"工具调用不匹配: actual={call!r}"
    if response["output"] not in (None, ""):
        return False, "工具调用夹带文本"
    if response["finish_reason"] != validator["expected_finish_reason"]:
        return False, f"finish_reason={response['finish_reason']!r}"
    return True, "tool_call"


def score_run(tasks_path: Path, responses_path: Path) -> dict[str, Any]:
    tasks = validate_task_files([tasks_path], require_full_bank=False)
    task_map = {task["id"]: task for task in tasks}
    rows = read_jsonl(responses_path)
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    unknown: list[str] = []
    validation_errors: list[str] = []
    run_ids: set[str] = set()
    model_ids: set[str] = set()
    generation_fingerprints: set[str] = set()
    for line, row in enumerate(rows, 1):
        try:
            validate_response(row, responses_path, line)
        except ValidationError as error:
            validation_errors.append(str(error))
            continue
        run_ids.add(row["run_id"])
        model_ids.add(row["model_id"])
        generation_fingerprints.add(json.dumps(row["generation"], ensure_ascii=False, sort_keys=True))
        task_id = row["task_id"]
        if task_id not in task_map:
            unknown.append(task_id)
        elif task_id in by_id:
            duplicates.append(task_id)
        else:
            by_id[task_id] = row

    results: list[dict[str, Any]] = []
    dimension_marks: dict[str, list[bool]] = defaultdict(list)
    for task in tasks:
        response = by_id.get(task["id"])
        if response is None:
            passed, detail = False, "缺少有效响应"
        else:
            passed, detail = grade(task, response)
        result = {"task_id": task["id"], "dimension": task["dimension"], "passed": passed, "detail": detail}
        results.append(result)
        dimension_marks[task["dimension"]].append(passed)

    consistency_errors: list[str] = []
    if len(run_ids) != 1:
        consistency_errors.append(f"有效记录应只有一个 run_id，实际 {sorted(run_ids)}")
    if len(model_ids) != 1:
        consistency_errors.append(f"有效记录应只有一个 model_id，实际 {sorted(model_ids)}")
    if len(generation_fingerprints) != 1:
        consistency_errors.append("同一 run 的 generation 参数不一致")
    input_errors = {
        "duplicates": sorted(set(duplicates)),
        "unknown": sorted(set(unknown)),
        "validation_errors": validation_errors,
        "consistency_errors": consistency_errors,
    }
    complete = not any(input_errors.values()) and len(by_id) == len(tasks)
    passed_count = sum(item["passed"] for item in results)
    return {
        "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None,
        "model_id": next(iter(model_ids)) if len(model_ids) == 1 else None,
        "generation": json.loads(next(iter(generation_fingerprints))) if len(generation_fingerprints) == 1 else None,
        "responses_file": str(responses_path.resolve()),
        "responses_sha256": sha256_file(responses_path),
        "complete": complete,
        "passed": passed_count,
        "total": len(tasks),
        "score": passed_count / len(tasks),
        "per_dimension": {
            dimension: {"passed": sum(dimension_marks[dimension]), "total": len(dimension_marks[dimension])}
            for dimension in DIMENSIONS
        },
        "input_errors": input_errors,
        "results": results,
    }


def compare_split(tasks_path: Path, baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    tasks = validate_task_files([tasks_path], require_full_bank=False)
    split = tasks[0]["split"]
    baseline = score_run(tasks_path, baseline_path)
    candidate = score_run(tasks_path, candidate_path)
    base_results = {item["task_id"]: item for item in baseline["results"]}
    cand_results = {item["task_id"]: item for item in candidate["results"]}
    wins: list[str] = []
    regressions: list[str] = []
    unchanged_pass: list[str] = []
    unchanged_fail: list[str] = []
    for task in tasks:
        task_id = task["id"]
        before = base_results[task_id]["passed"]
        after = cand_results[task_id]["passed"]
        if not before and after:
            wins.append(task_id)
        elif before and not after:
            regressions.append(task_id)
        elif before:
            unchanged_pass.append(task_id)
        else:
            unchanged_fail.append(task_id)
    dimension_delta = {
        dimension: candidate["per_dimension"][dimension]["passed"] - baseline["per_dimension"][dimension]["passed"]
        for dimension in DIMENSIONS
    }
    folds: list[dict[str, Any]] = []
    for held_out in tasks:
        remaining_delta = sum(
            int(cand_results[task["id"]]["passed"]) - int(base_results[task["id"]]["passed"])
            for task in tasks
            if task["id"] != held_out["id"]
        )
        folds.append({"held_out_task": held_out["id"], "remaining_score_delta": remaining_delta, "passed": remaining_delta > 0})
    loto = {
        "applicable": split == "task_holdout",
        "all_folds_positive": all(fold["passed"] for fold in folds) if split == "task_holdout" else None,
        "folds": folds if split == "task_holdout" else [],
    }
    complete = baseline["complete"] and candidate["complete"]
    improved_dimensions = sum(delta > 0 for delta in dimension_delta.values())
    nonregression_by_dimension = all(delta >= 0 for delta in dimension_delta.values())
    controlled_keys = ("temperature", "seed", "template_id")
    matched_generation_controls = (
        baseline["generation"] is not None
        and candidate["generation"] is not None
        and all(baseline["generation"].get(key) == candidate["generation"].get(key) for key in controlled_keys)
    )
    checks: dict[str, bool | None] = {
        "complete_pairs": complete,
        "matched_generation_controls": matched_generation_controls,
        "zero_regressions": not regressions,
        "dimension_nonregression": nonregression_by_dimension,
        "minimum_wins": None,
        "minimum_improved_dimensions": None,
        "all_loto_folds_positive": None,
    }
    if split == "task_holdout":
        checks["minimum_wins"] = len(wins) >= 4
        checks["minimum_improved_dimensions"] = improved_dimensions >= 4
        checks["all_loto_folds_positive"] = bool(loto["all_folds_positive"])
        gate_pass = all(value is True for value in checks.values())
    elif split == "final_holdout":
        checks["minimum_wins"] = len(wins) >= 2
        checks["minimum_improved_dimensions"] = improved_dimensions >= 2
        gate_pass = all(value is True for key, value in checks.items() if key != "all_loto_folds_positive")
    else:
        gate_pass = False
    return {
        "split": split,
        "task_file": str(tasks_path.resolve()),
        "task_sha256": sha256_file(tasks_path),
        "complete": complete,
        "baseline": baseline,
        "candidate": candidate,
        "paired": {
            "wins": wins,
            "regressions": regressions,
            "unchanged_pass": unchanged_pass,
            "unchanged_fail": unchanged_fail,
            "dimension_delta": dimension_delta,
            "improved_dimensions": improved_dimensions,
        },
        "loto": loto,
        "gate": {"passed": gate_pass, "checks": checks},
    }


def created_utc() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        moment = datetime.fromtimestamp(int(epoch), timezone.utc)
    else:
        moment = datetime.now(timezone.utc)
    return moment.isoformat().replace("+00:00", "Z")


def score_report(tasks: Path, responses: Path) -> dict[str, Any]:
    task_rows = validate_task_files([tasks], require_full_bank=False)
    split = task_rows[0]["split"]
    candidate = score_run(tasks, responses)
    return {
        "schema_version": "capability-report-v1",
        "created_utc": created_utc(),
        "mode": "score",
        "split_reports": [{
            "split": split,
            "task_file": str(tasks.resolve()),
            "task_sha256": sha256_file(tasks),
            "complete": candidate["complete"],
            "baseline": None,
            "candidate": candidate,
            "paired": None,
            "loto": None,
            "gate": {"passed": False, "checks": {"score_only": True}},
        }],
        "decision": {
            "promotable": False,
            "status": "score_only",
            "reasons": ["单模型分数不能证明相对 v17 的提升"],
            "claim_ceiling": "仅可报告该运行在指定 split 的原始正确数。",
        },
    }


def compare_report(tasks: Path, baseline: Path, candidate: Path) -> dict[str, Any]:
    split_report = compare_split(tasks, baseline, candidate)
    split = split_report["split"]
    passed = split_report["gate"]["passed"]
    if split == "dev":
        status = "dev_only"
        reasons = ["开发集只用于调试，不参与晋级"]
    elif split == "task_holdout":
        status = "task_holdout_pass" if passed else "task_holdout_fail"
        reasons = [] if passed else failed_check_reasons(split_report)
    else:
        status = "final_holdout_pass" if passed else "final_holdout_fail"
        reasons = [] if passed else failed_check_reasons(split_report)
    return {
        "schema_version": "capability-report-v1",
        "created_utc": created_utc(),
        "mode": "compare",
        "split_reports": [split_report],
        "decision": {
            "promotable": False,
            "status": status,
            "reasons": reasons,
            "claim_ceiling": "单个 split 通过不构成最终晋级；必须运行 promote 同时复核任务级与最终留出集。",
        },
    }


def failed_check_reasons(split_report: dict[str, Any]) -> list[str]:
    reasons = [key for key, value in split_report["gate"]["checks"].items() if value is False]
    return reasons or ["split gate 未通过"]


def promotion_report(
    task_holdout_baseline: Path,
    task_holdout_candidate: Path,
    final_baseline: Path,
    final_candidate: Path,
) -> dict[str, Any]:
    task_report = compare_split(DATA / "task_holdout.jsonl", task_holdout_baseline, task_holdout_candidate)
    fail(task_report["split"] != "task_holdout", "内部错误：task_holdout split 不一致")
    final_report = compare_split(DATA / "final_holdout.jsonl", final_baseline, final_candidate)
    fail(final_report["split"] != "final_holdout", "内部错误：final_holdout split 不一致")
    passed = task_report["gate"]["passed"] and final_report["gate"]["passed"]
    reasons: list[str] = []
    if not task_report["gate"]["passed"]:
        reasons.extend(f"task_holdout:{reason}" for reason in failed_check_reasons(task_report))
    if not final_report["gate"]["passed"]:
        reasons.extend(f"final_holdout:{reason}" for reason in failed_check_reasons(final_report))
    return {
        "schema_version": "capability-report-v1",
        "created_utc": created_utc(),
        "mode": "promote",
        "split_reports": [task_report, final_report],
        "decision": {
            "promotable": passed,
            "status": "promote" if passed else "reject",
            "reasons": reasons,
            "claim_ceiling": (
                "仅说明候选通过该八维短门且在冻结样本上优于 v17；不得声称长榜、真实桌面或全面能力提升。"
                if passed
                else "不得声称候选优于 v17。"
            ),
        },
    }


def emit(payload: dict[str, Any], output: Path | None) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(serialized)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8", newline="\n")


def check_schemas() -> None:
    schema_paths = [SCHEMAS / name for name in ("task.schema.json", "response.schema.json", "report.schema.json")]
    for path in schema_paths:
        schema = read_json(path)
        fail(not isinstance(schema, dict), f"{path}: schema 必须是对象")
        fail(schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema", f"{path}: 非 Draft 2020-12")
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return
    validator = jsonschema.Draft202012Validator
    for path in schema_paths:
        validator.check_schema(read_json(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ColorLM 八维短门离线校验与判分")
    subparsers = parser.add_subparsers(dest="command", required=True)

    checker = subparsers.add_parser("check", help="检查 schema 与题库，不运行模型")
    checker.add_argument(
        "--task-file",
        action="append",
        type=Path,
        default=[],
        help="可重复；省略时检查 dev/task_holdout/final_holdout 全部题库",
    )

    scorer = subparsers.add_parser("score", help="判定一个 split 的单模型响应")
    scorer.add_argument("--tasks", type=Path, required=True)
    scorer.add_argument("--responses", type=Path, required=True)
    scorer.add_argument("--out", type=Path)

    comparer = subparsers.add_parser("compare", help="在一个 split 上配对比较 v17 与候选")
    comparer.add_argument("--tasks", type=Path, required=True)
    comparer.add_argument("--baseline", type=Path, required=True)
    comparer.add_argument("--candidate", type=Path, required=True)
    comparer.add_argument("--out", type=Path)

    promoter = subparsers.add_parser("promote", help="复核任务级和最终留出集，给出最终短门决策")
    promoter.add_argument("--task-holdout-baseline", type=Path, required=True)
    promoter.add_argument("--task-holdout-candidate", type=Path, required=True)
    promoter.add_argument("--final-baseline", type=Path, required=True)
    promoter.add_argument("--final-candidate", type=Path, required=True)
    promoter.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        if args.command == "check":
            paths = args.task_file or [DATA / f"{split}.jsonl" for split in SPLIT_COUNTS]
            check_schemas()
            tasks = validate_task_files(paths, require_full_bank=not args.task_file)
            result = {
                "ok": True,
                "task_count": len(tasks),
                "dimensions": list(DIMENSIONS),
                "files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths],
            }
            emit(result, None)
        elif args.command == "score":
            report = score_report(args.tasks, args.responses)
            validate_report(report)
            emit(report, args.out)
        elif args.command == "compare":
            report = compare_report(args.tasks, args.baseline, args.candidate)
            validate_report(report)
            emit(report, args.out)
        else:
            report = promotion_report(
                args.task_holdout_baseline,
                args.task_holdout_candidate,
                args.final_baseline,
                args.final_candidate,
            )
            validate_report(report)
            emit(report, args.out)
        return 0
    except (ValidationError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
