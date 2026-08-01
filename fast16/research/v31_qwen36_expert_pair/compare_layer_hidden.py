"""比较同提示下两个 ColorLM 候选的逐层隐藏状态差异。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_qwen36_routes import read_dump


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较两个全层隐藏状态 dump")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "layer_attribution.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control = read_dump(args.control)
    candidate = read_dump(args.candidate)
    if set(control) != set(candidate):
        raise RuntimeError(
            f"层集合不一致: control={sorted(control)}, candidate={sorted(candidate)}"
        )

    rows = []
    for layer in sorted(control):
        left_records = control[layer]
        right_records = candidate[layer]
        left = max(left_records, key=lambda values: values.shape[0]).astype(np.float64)
        right = max(right_records, key=lambda values: values.shape[0]).astype(np.float64)
        if left.shape != right.shape:
            raise RuntimeError(f"L{layer} 形状不一致: {left.shape} != {right.shape}")
        diff = right - left
        left_rms = float(np.sqrt(np.mean(left * left)))
        diff_rms = float(np.sqrt(np.mean(diff * diff)))
        dots = np.sum(left * right, axis=1)
        norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        cosine = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)
        rows.append(
            {
                "layer": layer,
                "control_records_seen": len(left_records),
                "candidate_records_seen": len(right_records),
                "tokens": int(left.shape[0]),
                "hidden_rms": left_rms,
                "delta_rms": diff_rms,
                "relative_delta_rms": diff_rms / max(left_rms, 1e-12),
                "cosine_mean": float(np.mean(cosine)),
                "cosine_min": float(np.min(cosine)),
            }
        )

    report = {
        "format": "colormlm-qwen36-layer-hidden-attribution-v1",
        "control": args.control.as_posix(),
        "candidate": args.candidate.as_posix(),
        "layers": rows,
        "largest_relative_delta_layers": [
            row["layer"]
            for row in sorted(
                rows, key=lambda item: item["relative_delta_rms"], reverse=True
            )[:10]
        ],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"最大相对差异层: {report['largest_relative_delta_layers']}")
    print(f"报告: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
