"""只用v21语料拟合带偏置锚定岭桥，并在v22外部语料上一次验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve, svd


ROOT = Path(__file__).resolve().parents[3]
BRIDGE_TOOLS = ROOT / "fast16/research/v18_activation_bridge"
sys.path.insert(0, str(BRIDGE_TOOLS))

from activation_bridge import concatenate, pair_receipts  # noqa: E402


FACTORS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)


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


def clipped_invertible(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, singular, right_t = svd(
        matrix,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    clipped = np.clip(singular, 0.25, 8.0)
    stable = np.asarray((left * clipped) @ right_t, dtype=np.float32)
    inverse = np.asarray((right_t.T * (1.0 / clipped)) @ left.T, dtype=np.float32)
    return stable, inverse, singular


def centered_scale(base: np.ndarray, donor: np.ndarray, matrix: np.ndarray) -> float:
    predicted = base @ matrix
    ratios = np.linalg.norm(donor, axis=1) / np.maximum(
        np.linalg.norm(predicted, axis=1), 1e-12
    )
    return float(np.clip(np.median(ratios), 0.25, 8.0))


def baseline_scale(base: np.ndarray, donor: np.ndarray, prior: np.ndarray) -> float:
    predicted = base @ prior
    ratios = np.linalg.norm(donor, axis=1) / np.maximum(
        np.linalg.norm(predicted, axis=1), 1e-12
    )
    return float(np.clip(np.median(ratios), 0.25, 8.0))


def fit_affine(
    base: np.ndarray,
    donor: np.ndarray,
    prior: np.ndarray,
    factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_base = np.asarray(np.mean(base, axis=0), dtype=np.float32)
    mean_donor = np.asarray(np.mean(donor, axis=0), dtype=np.float32)
    centered_base = np.asarray(base - mean_base, dtype=np.float32)
    centered_donor = np.asarray(donor - mean_donor, dtype=np.float32)
    residual = centered_donor - centered_base @ prior
    gram = centered_base @ centered_base.T
    regularization_scale = max(float(np.trace(gram)) / max(len(base), 1), 1e-8)
    system = gram + np.eye(len(base), dtype=np.float32) * (
        factor * regularization_scale
    )
    dual = solve(system, residual, assume_a="pos", check_finite=False)
    raw = np.asarray(prior + centered_base.T @ dual, dtype=np.float32)
    stable, inverse, singular = clipped_invertible(raw)
    scale = centered_scale(centered_base, centered_donor, stable)
    stable = np.asarray(stable * scale, dtype=np.float32)
    inverse = np.asarray(inverse / scale, dtype=np.float32)
    bias = np.asarray(mean_donor - mean_base @ stable, dtype=np.float32)
    return stable, inverse, bias, singular, np.asarray([scale], dtype=np.float32)


def predict(base: np.ndarray, matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(base @ matrix + bias, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="拟合并外部验证GLM仿射深层桥")
    parser.add_argument("--train-base-receipt", type=Path, required=True)
    parser.add_argument("--train-donor-receipt", type=Path, required=True)
    parser.add_argument("--external-base-receipt", type=Path, required=True)
    parser.add_argument("--external-donor-receipt", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()

    train_groups, train_pairing = pair_receipts(
        args.train_base_receipt, args.train_donor_receipt
    )
    external_groups, external_pairing = pair_receipts(
        args.external_base_receipt, args.external_donor_receipt
    )
    if len(train_groups) < 6:
        raise RuntimeError("旧训练语料至少需要6个配对提示")
    train_base, train_donor = concatenate(train_groups)
    external_base, external_donor = concatenate(external_groups)
    donor_to_base = np.asarray(np.load(args.prior, allow_pickle=False), dtype=np.float32)
    prior = donor_to_base.T
    width = train_base.shape[1]
    if prior.shape != (width, width) or external_base.shape[1] != width:
        raise RuntimeError("先验桥或外部激活宽度不匹配")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_groups))
    validation_ids = set(int(value) for value in order[:3])
    inner_train_groups = [
        group for index, group in enumerate(train_groups) if index not in validation_ids
    ]
    inner_validation_groups = [
        group for index, group in enumerate(train_groups) if index in validation_ids
    ]
    inner_base, inner_donor = concatenate(inner_train_groups)
    validation_base, validation_donor = concatenate(inner_validation_groups)

    selection: list[dict[str, float]] = []
    for factor in FACTORS:
        matrix, _inverse, bias, _singular, _scale = fit_affine(
            inner_base, inner_donor, prior, factor
        )
        result = metrics(predict(validation_base, matrix, bias), validation_donor)
        selection.append(
            {
                "factor": factor,
                "relative_rmse": float(result["relative_rmse"]),
                "median_cosine": float(result["cosine"]["median"]),
            }
        )
    chosen = min(selection, key=lambda row: (row["relative_rmse"], -row["median_cosine"]))

    matrix, inverse, bias, raw_singular, scale_array = fit_affine(
        train_base, train_donor, prior, float(chosen["factor"])
    )
    frozen_baseline_scale = baseline_scale(train_base, train_donor, prior)
    baseline_matrix = prior * np.float32(frozen_baseline_scale)
    baseline_result = metrics(external_base @ baseline_matrix, external_donor)
    candidate_result = metrics(predict(external_base, matrix, bias), external_donor)
    cosine_lift = float(candidate_result["cosine"]["median"]) - float(
        baseline_result["cosine"]["median"]
    )
    nrmse_ratio = float(candidate_result["relative_rmse"]) / max(
        float(baseline_result["relative_rmse"]), 1e-12
    )

    per_prompt: list[dict[str, object]] = []
    for group in external_groups:
        before = float(np.median(cosine_rows(group.base @ baseline_matrix, group.donor)))
        after = float(np.median(cosine_rows(predict(group.base, matrix, bias), group.donor)))
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
    input_weight = np.ascontiguousarray(matrix.T, dtype=np.float32)
    output_weight = np.ascontiguousarray(inverse.T, dtype=np.float32)
    input_bias = np.ascontiguousarray(bias, dtype=np.float32)
    cycle = output_weight @ input_weight
    cycle_rmse = float(
        np.linalg.norm(cycle - np.eye(width, dtype=np.float32)) / math.sqrt(width * width)
    )
    gates = {
        "all_prompts_paired": len(external_groups) == 16,
        "enough_tokens": len(external_base) >= 900,
        "cosine_lift": cosine_lift >= 0.30,
        "nrmse_ratio": nrmse_ratio <= 0.95,
        "positive_prompts": positive_prompts >= 12,
        "cycle_rmse": cycle_rmse <= 5e-5,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    input_path = args.out / "glm_affine_input_weight_f32.npy"
    output_path = args.out / "glm_affine_output_weight_f32.npy"
    bias_path = args.out / "glm_affine_input_bias_f32.npy"
    np.save(input_path, input_weight, allow_pickle=False)
    np.save(output_path, output_weight, allow_pickle=False)
    np.save(bias_path, input_bias, allow_pickle=False)
    report = {
        "format": "colorlm-affine-anchored-ridge-external-v1",
        "policy": {
            "fit_and_selection": "v21 corpus only",
            "external": "v22 corpus used once for final metrics only",
            "seed": args.seed,
        },
        "train_pairing": train_pairing,
        "external_pairing": external_pairing,
        "selection": {
            "train_prompts": [group.prompt_id for group in inner_train_groups],
            "validation_prompts": [group.prompt_id for group in inner_validation_groups],
            "candidates": selection,
            "chosen_factor": chosen["factor"],
        },
        "fit": {
            "all_old_prompts": len(train_groups),
            "all_old_tokens": len(train_base),
            "baseline_scale": frozen_baseline_scale,
            "candidate_centered_scale": float(scale_array[0]),
            "raw_singular_min": float(np.min(raw_singular)),
            "raw_singular_median": float(np.median(raw_singular)),
            "raw_singular_max": float(np.max(raw_singular)),
            "runtime_singular_clip": [0.25, 8.0],
        },
        "external": {
            "prompts": len(external_groups),
            "paired_tokens": len(external_base),
            "baseline": baseline_result,
            "candidate": candidate_result,
            "median_cosine_lift": cosine_lift,
            "nrmse_ratio": nrmse_ratio,
            "positive_prompts": positive_prompts,
            "per_prompt": per_prompt,
        },
        "runtime": {
            "input_weight": {"file": input_path.name, "sha256": sha256(input_path)},
            "output_weight": {"file": output_path.name, "sha256": sha256(output_path)},
            "input_bias": {"file": bias_path.name, "sha256": sha256(bias_path)},
            "cycle_rmse": cycle_rmse,
        },
        "promotion": {
            "gates": gates,
            "decision": "candidate" if all(gates.values()) else "reject",
        },
    }
    report_path = args.out / "affine_external_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
