"""Write the deterministic ColorLM v4 donor compilation plan."""

from __future__ import annotations

import argparse

from .donor import build_qwen35_donor_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--donor-sha256", default="pending")
    args = parser.parse_args()
    path = build_qwen35_donor_plan(donor_hash=args.donor_sha256).write(args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

