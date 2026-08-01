"""CLI for compiling the knowledge donor into ColorLM v4."""

from __future__ import annotations

import argparse

from .transplant import transplant_qwen35_to_colorlm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()
    result = transplant_qwen35_to_colorlm(
        args.donor,
        args.output,
        verify_sha256=not args.skip_sha256,
    )
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

