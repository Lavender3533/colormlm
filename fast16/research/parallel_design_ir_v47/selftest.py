#!/usr/bin/env python3
"""只消费 8 条 train 的 Design Genome/编译器轻量静态自检。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
FRONTEND_ROOT = ROOT.parent / "parallel_frontend_v47"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FRONTEND_ROOT))

from compile_design_genome import compile_genome  # noqa: E402
from decode_genome import decode_text  # noqa: E402
from ir_core import (  # noqa: E402
    HARD_MAX_BYTES,
    PREFERRED_MAX_BYTES,
    PREFERRED_MIN_BYTES,
    canonical_text,
    load_ir,
    load_slots,
    read_utf8_no_bom,
    utf8_size,
    validate_ir,
    write_utf8_no_bom,
)
from validate_gates import evaluate_one, load_split  # noqa: E402


def walk_schema(node: Any, path: str = "$", parent_key: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(node, bool) and parent_key != "additionalProperties":
        errors.append(f"{path}: llama.cpp 兼容 schema 禁止布尔子 schema")
    elif isinstance(node, dict):
        if "prefixItems" in node:
            errors.append(f"{path}: 兼容 schema 禁止 prefixItems；本地 README 标记其语义损坏")
        if isinstance(node.get("items"), list):
            length = len(node["items"])
            if node.get("minItems") != length or node.get("maxItems") != length:
                errors.append(f"{path}: tuple items 长度 {length} 必须有同值 minItems/maxItems")
        for key, value in node.items():
            errors.extend(walk_schema(value, f"{path}.{key}", key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            errors.extend(walk_schema(value, f"{path}[{index}]", parent_key))
    return errors


def record(checks: list[dict[str, Any]], name: str, passed: bool, evidence: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})


def main() -> int:
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    compiled_dir = ROOT / "compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)

    strict_schema = json.loads(read_utf8_no_bom(ROOT / "design_ir.schema.json"))
    compat_schema = json.loads(read_utf8_no_bom(ROOT / "design_ir.llamacpp.schema.json"))
    slot_schema = json.loads(read_utf8_no_bom(ROOT / "copy_slots.schema.json"))
    record(checks, "schemas_parse", all(isinstance(x, dict) for x in (strict_schema, compat_schema, slot_schema)), [x.get("title") for x in (strict_schema, compat_schema, slot_schema)])
    compat_errors = walk_schema(compat_schema)
    record(checks, "llamacpp_schema_no_boolean_subschema_or_prefixitems", not compat_errors, compat_errors)

    converter = PROJECT_ROOT / "llama.cpp" / "examples" / "json_schema_to_grammar.py"
    converted = subprocess.run([sys.executable, str(converter), str(ROOT / "design_ir.llamacpp.schema.json")], capture_output=True, timeout=20)
    converted_stdout = converted.stdout.decode("utf-8", errors="replace")
    converted_stderr = converted.stderr.decode("utf-8", errors="replace")
    record(checks, "llamacpp_schema_converts_to_grammar", converted.returncode == 0 and "root ::=" in converted_stdout, {"returncode": converted.returncode, "stderr": converted_stderr[-1000:], "grammar_bytes": len(converted.stdout)})

    gbnf = read_utf8_no_bom(ROOT / "design_genome.gbnf")
    required_rules = ["primary ::=", "controls ::=", "content ::=", "detail ::=", "support ::=", "data-action ::=", "view-action ::=", "commit-action ::=", "state-action ::=", "main-transform ::=", "overlay-transform ::="]
    gbnf_errors = []
    if "*" in gbnf or "+" in gbnf:
        gbnf_errors.append("出现无界 * 或 +")
    gbnf_errors.extend(f"缺少 {rule}" for rule in required_rules if rule not in gbnf)
    root_line = gbnf.splitlines()[0]
    for symbol in ("primary", "controls", "content", "detail", "support", "data-action", "view-action", "commit-action", "state-action", "main-transform", "overlay-transform"):
        if root_line.count(symbol) != 1:
            gbnf_errors.append(f"root 中 {symbol} 不是恰好一次")
    record(checks, "bounded_role_ordered_gbnf", not gbnf_errors, {"errors": gbnf_errors, "bytes": len(gbnf.encode("utf-8")), "lines": len(gbnf.splitlines())})

    # 这里只读 train.jsonl；不得调用 check_all 或加载另外两个 split。
    rows = load_split("train")
    record(checks, "train_only_exactly_8", len(rows) == 8 and all(row["split"] == "train" for row in rows), [row["id"] for row in rows])
    compile_reports: list[dict[str, Any]] = []
    gate_results: list[dict[str, Any]] = []
    sizes: list[int] = []
    for row in rows:
        task_id = row["id"]
        genome_path = ROOT / "teachers" / f"{task_id}.genome.json"
        slots_path = ROOT / "teachers" / f"{task_id}.slots.json"
        try:
            ir = load_ir(genome_path, enforce_target_size=True)
            if genome_path.read_text(encoding="utf-8").strip() != canonical_text(ir):
                raise ValueError("教师 Genome 不是规范单行 JSON")
            slots = load_slots(slots_path, prompt=row["prompt"])
            size = utf8_size(ir)
            sizes.append(size)
            page, report = compile_genome(ir, slots)
            page_again, report_again = compile_genome(ir, slots)
            if page != page_again or report != report_again:
                raise ValueError("编译结果不确定")
            output = compiled_dir / f"{task_id}.html"
            write_utf8_no_bom(output, page)
            write_utf8_no_bom(compiled_dir / f"{task_id}.compile.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            evaluation = evaluate_one(row, output)
            compile_reports.append({"task_id": task_id, **report})
            gate_results.append({
                "task_id": task_id,
                "passed": evaluation["passed"],
                "final_score": evaluation["audit"]["summary"]["final_score"],
                "template_penalty": evaluation["audit"]["summary"]["template_penalty"],
                "critical": evaluation["critical"],
                "score_checks": evaluation["score_checks"],
                "failed_required": [x["pattern"] for x in evaluation["signals"]["required"] if not x["matched"]],
                "failed_any": [x["patterns"] for x in evaluation["signals"]["required_any"] if not x["passed"]],
                "matched_forbidden": [x["pattern"] for x in evaluation["signals"]["forbidden"] if x["matched"]],
            })
            if not evaluation["passed"]:
                failures.append({"kind": "train_static_gate", "task_id": task_id, "details": gate_results[-1]})
        except Exception as error:  # 聚合失败，不让首个错误遮住其他任务。
            failures.append({"kind": "teacher_or_compile_contract", "task_id": task_id, "error": str(error)})

    size_ok = len(sizes) == 8 and all(PREFERRED_MIN_BYTES <= size <= PREFERRED_MAX_BYTES for size in sizes)
    record(checks, "genome_preferred_150_to_600_bytes", size_ok, {"sizes": sizes, "hard_max": HARD_MAX_BYTES})
    record(checks, "compile_deterministic_and_all_train_static_gates", len(gate_results) == 8 and all(item["passed"] for item in gate_results), gate_results)
    ratios = [item["expansion_ratio_html_per_genome_byte"] for item in compile_reports]
    record(checks, "compiler_expansion_measured", len(ratios) == 8 and all(value > 1 for value in ratios), {"per_task": ratios, "min": min(ratios) if ratios else None, "max": max(ratios) if ratios else None})

    first = load_ir(ROOT / "teachers" / "pf47-train-01.genome.json", enforce_target_size=True)
    canonical = canonical_text(first)
    decoder_cases = {
        "canonical": decode_text(canonical),
        "fenced": decode_text(f"```json\n{canonical}\n```"),
        "duplicate": decode_text(canonical + canonical),
        "closure": decode_text(canonical[:-1]),
        "early_truncation": decode_text(canonical[:110]),
    }
    decoder_ok = (
        decoder_cases["canonical"].status == "ok"
        and decoder_cases["fenced"].genome == first
        and decoder_cases["duplicate"].genome == first
        and decoder_cases["closure"].status == "recovered_closure"
        and decoder_cases["early_truncation"].status == "needs_resume"
        and decoder_cases["early_truncation"].resume_prefix is not None
    )
    record(checks, "decoder_repeat_truncation_recovery", decoder_ok, {name: result.as_dict() for name, result in decoder_cases.items()})

    bad = json.loads(canonical)
    bad["c"][1] = ["sidebar", "docs"]
    bad_errors = validate_ir(bad, enforce_target_size=True)
    record(checks, "semantic_role_mismatch_rejected", any("controls" in item for item in bad_errors), bad_errors)

    compiler_source = read_utf8_no_bom(ROOT / "compile_design_genome.py")
    forbidden_literals = ["pf47-train", "边缘节点运维台", "独立杂志商店", "工业设计案例集", "社区工坊预约页", "API 文档工作台", "隐私设置中心", "多舞台音乐节日程", "城市树冠数据故事"]
    leaked = [value for value in forbidden_literals if value in compiler_source]
    record(checks, "compiler_has_no_task_id_or_title_special_case", not leaked, leaked)

    text_files = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in {".py", ".json", ".md", ".gbnf", ".html"}]
    encoding_errors: list[str] = []
    for path in text_files:
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            encoding_errors.append(f"{path}: BOM")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as error:
            encoding_errors.append(f"{path}: {error}")
    record(checks, "all_text_utf8_without_bom", not encoding_errors, {"checked": len(text_files), "errors": encoding_errors})

    ok = all(item["passed"] for item in checks) and not failures
    report = {
        "schema_version": "parallel-design-genome-selftest-v1",
        "ok": ok,
        "cpu_static_only": True,
        "splits_read": ["train"],
        "model_or_gpu_used": False,
        "checks": checks,
        "teacher_summary": {
            "count": len(sizes),
            "genome_bytes": sizes,
            "min_bytes": min(sizes) if sizes else None,
            "max_bytes": max(sizes) if sizes else None,
            "mean_bytes": round(sum(sizes) / len(sizes), 2) if sizes else None,
            "compile_expansion_ratios": ratios,
            "train_static_gate_passes": sum(item["passed"] for item in gate_results),
        },
    }
    failure_report = {
        "schema_version": "parallel-design-genome-failure-report-v1",
        "status": "no_observed_static_failures" if not failures else "observed_failures",
        "scope": "仅 8 条 train 的 schema、编译、静态门与合成恢复测试；未读取或运行 validation/blind。",
        "failures": failures,
        "unproven": [
            "静态检查不证明浏览器交互、焦点陷阱、视觉审美或 WCAG 对比度真实通过。",
            "8 条教师 Genome 不证明 terminal-hidden Genome Head 可泛化。",
            "copy slots 当前由教师标注；自动候选抽取与未见专有名词复制尚未证明。",
            "llama.cpp 兼容 schema 只通过本地 converter；真实 v38 已知更可靠路径是有界 GBNF，最终目标是并行分类头。",
        ],
    }
    write_utf8_no_bom(ROOT / "SELFTEST_REPORT.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_utf8_no_bom(ROOT / "FAILURE_REPORT.json", json.dumps(failure_report, ensure_ascii=False, indent=2) + "\n")
    write_utf8_no_bom(compiled_dir / "compile_manifest.json", json.dumps(compile_reports, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
