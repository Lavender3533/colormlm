#!/usr/bin/env python3
"""验证 24 条冻结短门，并对已有 HTML 响应做离线评估/比较。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from score_html import CONTRACT_VERSION, SCORER_VERSION, audit_file, write_json


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SPLITS = ("train", "validation", "blind")
EXPECTED_KEYS = {
    "schema_version", "id", "split", "template_family", "title", "prompt",
    "required_regex", "required_any_regex", "forbidden_regex", "min_scores", "critical",
}
EXPECTED_MIN_KEYS = {
    "final", "structure", "responsive", "interaction", "visual_complexity",
    "dependency_safety", "accessibility", "max_template_penalty",
}


class GateError(ValueError):
    pass


def read_utf8_no_bom(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise GateError(f"{path}: 禁止 UTF-8 BOM")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GateError(f"{path}: 非法 UTF-8: {error}") from error


def load_split(split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise GateError(f"未知 split: {split}")
    path = DATA / f"{split}.jsonl"
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(read_utf8_no_bom(path).splitlines(), 1):
        if not line.strip():
            raise GateError(f"{path}:{line_no}: 不允许空行")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise GateError(f"{path}:{line_no}: {error}") from error
        validate_row(row, split, line_no)
        rows.append(row)
    return rows


def validate_row(row: Any, split: str, line_no: int) -> None:
    if not isinstance(row, dict) or set(row) != EXPECTED_KEYS:
        raise GateError(f"{split}:{line_no}: 字段不严格匹配契约")
    if row["schema_version"] != "parallel-frontend-gate-v1" or row["split"] != split:
        raise GateError(f"{split}:{line_no}: schema_version/split 错误")
    if not re.fullmatch(rf"pf47-{split}-\d{{2}}", row["id"]):
        raise GateError(f"{split}:{line_no}: id 格式错误")
    if not all(isinstance(row[key], str) and row[key].strip() for key in ("template_family", "title", "prompt")):
        raise GateError(f"{row['id']}: 文本字段为空")
    if len(row["prompt"]) < 40:
        raise GateError(f"{row['id']}: prompt 太短")
    for key in ("required_regex", "forbidden_regex"):
        if not isinstance(row[key], list) or not all(isinstance(x, str) and x for x in row[key]):
            raise GateError(f"{row['id']}: {key} 非字符串数组")
        for pattern in row[key]:
            re.compile(pattern, re.I | re.S)
    if not isinstance(row["required_any_regex"], list):
        raise GateError(f"{row['id']}: required_any_regex 非数组")
    for group in row["required_any_regex"]:
        if not isinstance(group, list) or len(group) < 2:
            raise GateError(f"{row['id']}: required_any_regex 每组至少两个模式")
        for pattern in group:
            re.compile(pattern, re.I | re.S)
    mins = row["min_scores"]
    if not isinstance(mins, dict) or set(mins) != EXPECTED_MIN_KEYS or not all(isinstance(v, (int, float)) for v in mins.values()):
        raise GateError(f"{row['id']}: min_scores 错误")
    if row["critical"] != ["valid_utf8", "no_bom", "single_html", "task_signals", "no_forbidden"]:
        raise GateError(f"{row['id']}: critical 必须采用冻结顺序")


def check_all() -> dict[str, Any]:
    rows = [row for split in SPLITS for row in load_split(split)]
    ids = [row["id"] for row in rows]
    families_by_split = {split: {row["template_family"] for row in rows if row["split"] == split} for split in SPLITS}
    overlaps = {
        f"{a}:{b}": sorted(families_by_split[a] & families_by_split[b])
        for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]
    }
    errors: list[str] = []
    if len(rows) != 24:
        errors.append(f"题数不是 24，而是 {len(rows)}")
    if len(set(ids)) != len(ids):
        errors.append("存在重复 id")
    if any(len(families_by_split[x]) != 8 for x in SPLITS):
        errors.append("每个 split 必须有 8 个不同模板族")
    if any(overlaps.values()):
        errors.append(f"模板族跨 split 泄漏：{overlaps}")
    for split in SPLITS:
        expected_ids = [f"pf47-{split}-{i:02d}" for i in range(1, 9)]
        actual = [row["id"] for row in rows if row["split"] == split]
        if actual != expected_ids:
            errors.append(f"{split} id 或顺序不连续")
    prompt_hashes = [hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest() for row in rows]
    if len(set(prompt_hashes)) != 24:
        errors.append("存在重复 prompt")
    return {
        "schema_version": "parallel-frontend-gate-check-v1",
        "ok": not errors,
        "gate_count": len(rows),
        "split_counts": {split: sum(row["split"] == split for row in rows) for split in SPLITS},
        "family_counts": {split: len(families_by_split[split]) for split in SPLITS},
        "cross_split_family_overlaps": overlaps,
        "prompt_sha256_unique": len(set(prompt_hashes)),
        "errors": errors,
    }


def task_signal_result(row: dict[str, Any], text: str) -> dict[str, Any]:
    required = [{"pattern": p, "matched": bool(re.search(p, text, re.I | re.S))} for p in row["required_regex"]]
    any_groups = []
    for patterns in row["required_any_regex"]:
        matches = [p for p in patterns if re.search(p, text, re.I | re.S)]
        any_groups.append({"patterns": patterns, "matched_patterns": matches, "passed": bool(matches)})
    forbidden = [{"pattern": p, "matched": bool(re.search(p, text, re.I | re.S))} for p in row["forbidden_regex"]]
    return {
        "required": required,
        "required_any": any_groups,
        "forbidden": forbidden,
        "task_signals_pass": all(x["matched"] for x in required) and all(x["passed"] for x in any_groups),
        "no_forbidden_pass": not any(x["matched"] for x in forbidden),
    }


def evaluate_one(row: dict[str, Any], path: Path) -> dict[str, Any]:
    audit = audit_file(path)
    text = path.read_bytes().decode("utf-8", errors="replace")
    signals = task_signal_result(row, text)
    mins = row["min_scores"]
    score_checks = {
        "final": audit["summary"]["final_score"] >= mins["final"],
        "structure": audit["dimensions"]["structure"]["score"] >= mins["structure"],
        "responsive": audit["dimensions"]["responsive"]["score"] >= mins["responsive"],
        "interaction": audit["dimensions"]["interaction"]["score"] >= mins["interaction"],
        "visual_complexity": audit["dimensions"]["visual_complexity"]["score"] >= mins["visual_complexity"],
        "dependency_safety": audit["dimensions"]["dependency_safety"]["score"] >= mins["dependency_safety"],
        "accessibility": audit["dimensions"]["accessibility"]["score"] >= mins["accessibility"],
        "max_template_penalty": audit["summary"]["template_penalty"] <= mins["max_template_penalty"],
    }
    critical = {
        "valid_utf8": audit["encoding"]["valid_utf8"],
        "no_bom": not audit["encoding"]["bom"],
        "single_html": audit["dimensions"]["structure"]["checks"][1]["passed"],
        "task_signals": signals["task_signals_pass"],
        "no_forbidden": signals["no_forbidden_pass"],
    }
    return {
        "task_id": row["id"], "file": str(path.resolve()), "audit": audit,
        "signals": signals, "score_checks": score_checks, "critical": critical,
        "passed": all(score_checks.values()) and all(critical.values()),
    }


def evaluate(split: str, responses_dir: Path | None = None, shared_html: Path | None = None) -> dict[str, Any]:
    if (responses_dir is None) == (shared_html is None):
        raise GateError("必须且只能提供 responses_dir 或 shared_html")
    rows = load_split(split)
    results = []
    missing = []
    for row in rows:
        path = shared_html if shared_html else responses_dir / f"{row['id']}.html"  # type: ignore[operator]
        if not path.is_file():
            missing.append(row["id"])
            continue
        results.append(evaluate_one(row, path))
    scores = [x["audit"]["summary"]["final_score"] for x in results]
    penalties = [x["audit"]["summary"]["template_penalty"] for x in results]
    return {
        "schema_version": "parallel-frontend-gate-evaluation-v1",
        "contract_version": CONTRACT_VERSION,
        "scorer_version": SCORER_VERSION,
        "split": split,
        "expected": len(rows),
        "evaluated": len(results),
        "missing": missing,
        "passed": sum(x["passed"] for x in results),
        "pass_rate": round(sum(x["passed"] for x in results) / len(rows), 4),
        "median_final_score": round(statistics.median(scores), 2) if scores else None,
        "median_template_penalty": round(statistics.median(penalties), 2) if penalties else None,
        "results": results,
    }


def compare(split: str, baseline_html: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline = evaluate(split, shared_html=baseline_html)
    candidate = evaluate(split, responses_dir=candidate_dir)
    by_id_base = {x["task_id"]: x for x in baseline["results"]}
    pairs = []
    for item in candidate["results"]:
        base = by_id_base[item["task_id"]]
        gain = round(item["audit"]["summary"]["final_score"] - base["audit"]["summary"]["final_score"], 2)
        pairs.append({"task_id": item["task_id"], "score_gain": gain, "candidate_passed": item["passed"], "baseline_passed": base["passed"]})
    gains = [x["score_gain"] for x in pairs]
    decision_checks = {
        "complete_8_of_8": candidate["evaluated"] == 8 and not candidate["missing"],
        "all_critical_pass": all(all(x["critical"].values()) for x in candidate["results"]) and len(candidate["results"]) == 8,
        "at_least_6_gate_passes": candidate["passed"] >= 6,
        "median_gain_at_least_12": bool(gains) and statistics.median(gains) >= 12,
        "at_least_7_pairwise_wins_by_10": sum(x["score_gain"] >= 10 for x in pairs) >= 7,
        "no_pairwise_regression": all(x["score_gain"] >= 0 for x in pairs),
        "median_template_penalty_at_most_6": candidate["median_template_penalty"] is not None and candidate["median_template_penalty"] <= 6,
    }
    return {
        "schema_version": "parallel-frontend-gate-comparison-v1",
        "split": split,
        "baseline_kind": "fixed_ordinary_three_cards",
        "baseline_sha256": hashlib.sha256(baseline_html.read_bytes()).hexdigest(),
        "baseline": baseline,
        "candidate": candidate,
        "pairs": pairs,
        "decision": {
            "better_than_ordinary_three_cards": all(decision_checks.values()),
            "checks": decision_checks,
            "scope": "只说明冻结短门中优于本固定普通三卡片基线；不等价于通用前端能力。",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--output", type=Path)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--split", required=True, choices=SPLITS)
    group = evaluate_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--responses-dir", type=Path)
    group.add_argument("--shared-html", type=Path)
    evaluate_parser.add_argument("--output", type=Path)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--split", required=True, choices=SPLITS)
    compare_parser.add_argument("--baseline-html", type=Path, default=ROOT / "fixtures" / "ordinary_three_cards.html")
    compare_parser.add_argument("--candidate-dir", required=True, type=Path)
    compare_parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "check":
            payload = check_all()
        elif args.command == "evaluate":
            payload = evaluate(args.split, args.responses_dir, args.shared_html)
        else:
            payload = compare(args.split, args.baseline_html, args.candidate_dir)
        if args.output:
            write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload.get("ok", True) else 1
    except (OSError, GateError, re.error) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
