"""Inspect a partial GGUF header and validate the ColorLM donor plan."""

from __future__ import annotations

import argparse
import json

from .donor import build_qwen35_donor_plan
from .gguf_header import inspect_header, validate_plan_sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("header")
    args = parser.parse_args()
    reader = inspect_header(args.header)
    validation = validate_plan_sources(build_qwen35_donor_plan(), reader)
    print(
        json.dumps(
            {
                "header": reader.summary(),
                "plan": {
                    "ok": validation.ok,
                    "required_sources": validation.required_sources,
                    "found_sources": validation.found_sources,
                    "missing_sources": validation.missing_sources,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

