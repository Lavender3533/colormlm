#!/usr/bin/env python3
"""不调用模型、不联网、不使用 GPU 的评分器与短门自测。"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

from score_html import DIMENSION_MAX, audit_file, build_report
from validate_gates import ROOT, check_all, load_split


def main() -> int:
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    baseline_path = ROOT / "fixtures" / "ordinary_three_cards.html"
    advanced_path = ROOT / "fixtures" / "advanced_reference.html"
    baseline = audit_file(baseline_path, include_source=False)
    advanced = audit_file(advanced_path, include_source=False)
    baseline_again = audit_file(baseline_path, include_source=False)
    record("deterministic", baseline == baseline_again, baseline["sha256"])
    record("dimension_bounds", all(0 <= advanced["dimensions"][name]["score"] <= maximum for name, maximum in DIMENSION_MAX.items()), {name: advanced["dimensions"][name]["score"] for name in DIMENSION_MAX})
    record("final_formula", advanced["summary"]["final_score"] == round(max(0, min(100, advanced["summary"]["gross_score"] - advanced["summary"]["template_penalty"])), 2), advanced["summary"])
    record("ordinary_template_detected", baseline["summary"]["template_penalty"] >= 9, baseline["summary"])
    record("advanced_beats_template", advanced["summary"]["final_score"] >= baseline["summary"]["final_score"] + 12, {"advanced": advanced["summary"], "baseline": baseline["summary"]})
    record("advanced_low_template_penalty", advanced["summary"]["template_penalty"] <= 4, advanced["summary"]["template_penalty"])
    gate_check = check_all()
    record("gate_contract", gate_check["ok"], gate_check)
    rows = [row for split in ("train", "validation", "blind") for row in load_split(split)]
    record("exactly_24_gates", len(rows) == 24, len(rows))
    record("split_isolation", all(not value for value in gate_check["cross_split_family_overlaps"].values()), gate_check["cross_split_family_overlaps"])
    report = build_report([baseline_path, advanced_path], source_root="selftest-fixtures")
    record("report_ranking", report["ranking"][0]["file"] == "advanced_reference.html", report["ranking"])
    sample_report = json.loads((ROOT / "sample_audit_report.json").read_text(encoding="utf-8"))
    record("six_sample_report", sample_report.get("sample_count") == 6 and len(sample_report.get("items", [])) == 6, {"sample_count": sample_report.get("sample_count"), "items": len(sample_report.get("items", []))})
    schemas = [json.loads((ROOT / name).read_text(encoding="utf-8")) for name in ("scoring_contract.json", "audit_report.schema.json", "gate.schema.json")]
    record("json_contracts_parse", len(schemas) == 3 and schemas[1].get("$schema", "").endswith("2020-12/schema"), [x.get("schema_version", x.get("$schema")) for x in schemas])
    manifest_path = ROOT / "MANIFEST.json"
    manifest_ok = manifest_path.is_file()
    manifest_errors = []
    if manifest_ok:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("files", []):
            data = (ROOT / entry["path"]).read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if data.startswith(b"\xef\xbb\xbf") or actual != entry["sha256"]:
                manifest_errors.append(entry["path"])
    record("frozen_manifest", manifest_ok and not manifest_errors, manifest_errors)

    payload = {
        "schema_version": "parallel-frontend-selftest-v1",
        "ok": all(bool(x["passed"]) for x in checks),
        "cpu_static_only": True,
        "tests": len(checks),
        "checks": checks,
    }
    output = ROOT / "SELFTEST_REPORT.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
