#!/usr/bin/env python3
"""只读取 discovery.json，冻结复合 Frame；CLI 没有 holdout 参数。"""

from __future__ import annotations

import argparse
from pathlib import Path

from axis_core import (
    read_json,
    schema_from_snapshot,
    sha256_file,
    synthesize_axis,
    write_json,
)


def synthesize(discovery_path: Path, output_path: Path) -> dict[str, object]:
    discovery = read_json(discovery_path)
    if discovery.get("format") != "polaris-blind-composite-discovery-v1":
        raise ValueError("发现集格式错误")
    schema = schema_from_snapshot(discovery.get("schema"))
    rows = discovery.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("发现集 episodes 无效")
    fit = synthesize_axis(schema, rows)
    frame = {
        "format": "polaris-blind-composite-frame-v1",
        "discovery_sha256": sha256_file(discovery_path),
        "axis": fit.axis.snapshot(),
        "training": {
            "correct": fit.correct,
            "total": fit.total,
            "accuracy": fit.accuracy,
            "candidate_count": fit.candidate_count,
        },
        "process_contract": {
            "accepted_input_files": [str(discovery_path)],
            "holdout_argument_exists": False,
            "frame_frozen_before_evaluation": True,
        },
    }
    write_json(output_path, frame)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = synthesize(args.discovery, args.output)
    print(frame["axis"]["canonical"])


if __name__ == "__main__":
    main()
