"""在完全独立的激活语料上评估已经冻结的深层坐标桥。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BRIDGE_TOOLS = ROOT / "fast16/research/v18_activation_bridge"
sys.path.insert(0, str(BRIDGE_TOOLS))

from activation_bridge import concatenate, pair_receipts  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    return np.sum(left * right, axis=1) / np.maximum(left_norm * right_norm, 1e-12)


def metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, object]:
    cosine = cosine_rows(predicted, target)
    return {
        "cosine": {
            "mean": float(np.mean(cosine)),
            "median": float(np.median(cosine)),
            "p05": float(np.percentile(cosine, 5)),
            "p95": float(np.percentile(cosine, 95)),
            "min": float(np.min(cosine)),
            "max": float(np.max(cosine)),
        },
        "relative_rmse": float(
            np.linalg.norm(predicted - target) / max(np.linalg.norm(target), 1e-12)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在未参与拟合的外部语料上评估冻结GLM桥"
    )
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--donor-receipt", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--baseline-scale", type=float, required=True)
    parser.add_argument("--input-weight", type=Path, required=True)
    parser.add_argument("--output-weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.baseline_scale <= 0:
        raise RuntimeError("--baseline-scale必须为正数")
    groups, pairing = pair_receipts(args.base_receipt, args.donor_receipt)
    base, donor = concatenate(groups)
    donor_to_base = np.asarray(np.load(args.prior, allow_pickle=False), dtype=np.float32)
    input_weight = np.asarray(
        np.load(args.input_weight, allow_pickle=False), dtype=np.float32
    )
    output_weight = np.asarray(
        np.load(args.output_weight, allow_pickle=False), dtype=np.float32
    )
    width = base.shape[1]
    for label, matrix in (
        ("prior", donor_to_base),
        ("input_weight", input_weight),
        ("output_weight", output_weight),
    ):
        if matrix.shape != (width, width):
            raise RuntimeError(f"{label}形状错误: {matrix.shape}")

    baseline_matrix = donor_to_base.T * np.float32(args.baseline_scale)
    candidate_matrix = input_weight.T
    baseline = metrics(base @ baseline_matrix, donor)
    candidate = metrics(base @ candidate_matrix, donor)
    cosine_lift = float(candidate["cosine"]["median"]) - float(
        baseline["cosine"]["median"]
    )
    nrmse_ratio = float(candidate["relative_rmse"]) / max(
        float(baseline["relative_rmse"]), 1e-12
    )

    per_prompt: list[dict[str, object]] = []
    for group in groups:
        before = float(np.median(cosine_rows(group.base @ baseline_matrix, group.donor)))
        after = float(np.median(cosine_rows(group.base @ candidate_matrix, group.donor)))
        per_prompt.append(
            {
                "prompt_id": group.prompt_id,
                "tokens": len(group.base),
                "baseline_median_cosine": before,
                "candidate_median_cosine": after,
                "lift": after - before,
            }
        )
    positive_prompts = sum(float(row["lift"]) > 0 for row in per_prompt)
    cycle = output_weight @ input_weight
    cycle_rmse = float(
        np.linalg.norm(cycle - np.eye(width, dtype=np.float32)) / math.sqrt(width * width)
    )
    gates = {
        "all_prompts_paired": len(groups) == 16,
        "enough_tokens": len(base) >= 900,
        "cosine_lift": cosine_lift >= 0.30,
        "nrmse_ratio": nrmse_ratio <= 0.95,
        "positive_prompts": positive_prompts >= 12,
        "cycle_rmse": cycle_rmse <= 5e-5,
    }
    report = {
        "format": "colorlm-external-activation-bridge-evaluation-v1",
        "evaluation_policy": "frozen bridge; external data is never used for fitting or scaling",
        "pairing": pairing,
        "artifacts": {
            "prior": {"path": str(args.prior.resolve()), "sha256": sha256(args.prior)},
            "input_weight": {
                "path": str(args.input_weight.resolve()),
                "sha256": sha256(args.input_weight),
            },
            "output_weight": {
                "path": str(args.output_weight.resolve()),
                "sha256": sha256(args.output_weight),
            },
        },
        "baseline_scale_frozen": args.baseline_scale,
        "external": {
            "prompts": len(groups),
            "paired_tokens": len(base),
            "baseline": baseline,
            "candidate": candidate,
            "median_cosine_lift": cosine_lift,
            "nrmse_ratio": nrmse_ratio,
            "positive_prompts": positive_prompts,
            "per_prompt": per_prompt,
        },
        "stability": {"cycle_rmse": cycle_rmse},
        "promotion": {
            "gates": gates,
            "decision": "candidate" if all(gates.values()) else "reject",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
