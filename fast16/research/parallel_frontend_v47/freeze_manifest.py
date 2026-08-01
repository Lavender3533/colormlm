#!/usr/bin/env python3
"""为冻结契约与 24 条短门生成确定性 SHA-256 清单。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FILES = [
    "score_html.py", "validate_gates.py", "selftest.py",
    "scoring_contract.json", "audit_report.schema.json", "gate.schema.json",
    "data/train.jsonl", "data/validation.jsonl", "data/blind.jsonl",
    "fixtures/ordinary_three_cards.html", "fixtures/advanced_reference.html",
    "sample_audit_report.json", "RESEARCH_NOTES.md", "DISTILLATION_PLAN.md",
]


def main() -> int:
    entries = []
    for name in FILES:
        data = (ROOT / name).read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            raise ValueError(f"{name}: 禁止 UTF-8 BOM")
        data.decode("utf-8")
        entries.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    payload = {
        "schema_version": "parallel-frontend-manifest-v1",
        "encoding": "UTF-8 without BOM",
        "scorer_version": "parallel-frontend-static-v1.0.0",
        "gate_counts": {"train": 8, "validation": 8, "blind": 8, "total": 24},
        "files": entries,
    }
    (ROOT / "MANIFEST.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
