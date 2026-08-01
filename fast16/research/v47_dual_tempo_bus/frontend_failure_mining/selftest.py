#!/usr/bin/env python3
"""frontend_failure_mining 的离线确定性与隐私自检。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mine_frontend_failures import (
    CATEGORY_ORDER,
    HERE,
    assert_private_payload,
    build_artifacts,
    pretty_bytes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()

    report_a, contract_a = build_artifacts(args.input_dir)
    report_b, contract_b = build_artifacts(args.input_dir)
    report_payload = pretty_bytes(report_a)
    contract_payload = pretty_bytes(contract_a)
    assert report_payload == pretty_bytes(report_b), "report 非确定性"
    assert contract_payload == pretty_bytes(contract_b), "contract 非确定性"
    assert len(report_a["samples"]) == 6
    assert [item["id"] for item in contract_a["constraints"]] == CATEGORY_ORDER
    assert set(contract_a["teacher_screening"]["required_failure_state"]) == set(CATEGORY_ORDER)
    assert set(report_a["aggregate_support"][index]["id"] for index in range(len(CATEGORY_ORDER))) == set(CATEGORY_ORDER)
    for sample in report_a["samples"]:
        assert list(sample["failures"]) == CATEGORY_ORDER
        assert len(sample["source_sha256"]) == 64
        assert len(sample["failure_fingerprint_sha256"]) == 64
        assert not sample["privacy"]["source_html_retained"]
        assert not sample["privacy"]["remote_urls_retained"]
    assert_private_payload(report_payload)
    assert_private_payload(contract_payload)

    expected = {
        "report.json": report_payload,
        "negative_contract.json": contract_payload,
    }
    for name, payload in expected.items():
        path = args.output_dir / name
        assert path.is_file(), f"缺少产物：{name}"
        assert path.read_bytes() == payload, f"产物不是当前输入的确定性结果：{name}"
        assert not path.read_bytes().startswith(b"\xef\xbb\xbf")

    readme = (args.output_dir / "README.md").read_bytes()
    assert_private_payload(readme)
    assert not readme.startswith(b"\xef\xbb\xbf")
    print(
        json.dumps(
            {
                "ok": True,
                "sample_count": len(report_a["samples"]),
                "category_count": len(CATEGORY_ORDER),
                "deterministic_rebuild": True,
                "raw_html_or_url_leak": False,
                "utf8_bom": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
