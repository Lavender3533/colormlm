"""拟合带共享-token锚点的低秩岭回归深层坐标桥。

行向量契约：``donor = base @ M``。修正项仅位于训练隐藏态张成的子空间；
其正交补严格退回共享-token先验桥。正则强度只在外层训练提示中选择。
"""

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
        },
        "relative_rmse": float(
            np.linalg.norm(predicted - target) / max(np.linalg.norm(target), 1e-12)
        ),
    }


def train_scale(base: np.ndarray, donor: np.ndarray, prior: np.ndarray) -> float:
    mapped = base @ prior
    ratios = np.linalg.norm(donor, axis=1) / np.maximum(
        np.linalg.norm(mapped, axis=1), 1e-12
    )
    return float(np.clip(np.median(ratios), 0.25, 8.0))


def ridge_map(
    base: np.ndarray,
    donor: np.ndarray,
    prior: np.ndarray,
    factor: float,
) -> np.ndarray:
    residual = donor - base @ prior
    gram = base @ base.T
    scale = max(float(np.trace(gram)) / max(len(base), 1), 1e-8)
    system = gram + np.eye(len(base), dtype=np.float32) * (factor * scale)
    dual = solve(system, residual, assume_a="pos", check_finite=False)
    return np.asarray(prior + base.T @ dual, dtype=np.float32)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="拟合锚定岭回归 GLM 深层桥")
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--donor-receipt", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()

    groups, pairing = pair_receipts(args.base_receipt, args.donor_receipt)
    if len(groups) < 6:
        raise RuntimeError("至少需要 6 个配对提示")
    donor_to_base = np.asarray(np.load(args.prior, allow_pickle=False), dtype=np.float32)
    prior = donor_to_base.T
    width = groups[0].base.shape[1]
    if prior.shape != (width, width):
        raise RuntimeError(f"先验桥形状错误: {prior.shape}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(groups))
    outer_test_ids = set(int(value) for value in order[:3])
    outer_train = [group for index, group in enumerate(groups) if index not in outer_test_ids]
    outer_test = [group for index, group in enumerate(groups) if index in outer_test_ids]

    inner_order = rng.permutation(len(outer_train))
    inner_test_ids = set(int(value) for value in inner_order[:2])
    inner_train = [group for index, group in enumerate(outer_train) if index not in inner_test_ids]
    inner_test = [group for index, group in enumerate(outer_train) if index in inner_test_ids]
    inner_x, inner_y = concatenate(inner_train)
    inner_vx, inner_vy = concatenate(inner_test)

    factors = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
    selection: list[dict[str, float]] = []
    for factor in factors:
        candidate = ridge_map(inner_x, inner_y, prior, factor)
        result = metrics(inner_vx @ candidate, inner_vy)
        selection.append(
            {
                "factor": factor,
                "relative_rmse": float(result["relative_rmse"]),
                "median_cosine": float(result["cosine"]["median"]),
            }
        )
    chosen = min(selection, key=lambda row: (row["relative_rmse"], -row["median_cosine"]))

    train_x, train_y = concatenate(outer_train)
    test_x, test_y = concatenate(outer_test)
    raw = ridge_map(train_x, train_y, prior, chosen["factor"])
    stable, inverse, raw_singular = clipped_invertible(raw)
    # 截断奇异值只约束条件数；随后用外层训练集恢复两个模型真实的整体幅度。
    # 这个标量不改变方向或条件数，逆桥同步除以同一标量。
    candidate_scale = train_scale(train_x, train_y, stable)
    stable = np.asarray(stable * candidate_scale, dtype=np.float32)
    inverse = np.asarray(inverse / candidate_scale, dtype=np.float32)
    baseline_scale = train_scale(train_x, train_y, prior)
    baseline_result = metrics(test_x @ (prior * baseline_scale), test_y)
    candidate_result = metrics(test_x @ stable, test_y)
    nrmse_ratio = float(candidate_result["relative_rmse"]) / max(
        float(baseline_result["relative_rmse"]), 1e-12
    )
    cosine_lift = float(candidate_result["cosine"]["median"]) - float(
        baseline_result["cosine"]["median"]
    )

    per_prompt = []
    for group in outer_test:
        base_prediction = group.base @ (prior * baseline_scale)
        candidate_prediction = group.base @ stable
        before = float(np.median(cosine_rows(base_prediction, group.donor)))
        after = float(np.median(cosine_rows(candidate_prediction, group.donor)))
        per_prompt.append(
            {
                "prompt_id": group.prompt_id,
                "tokens": len(group.base),
                "baseline_median_cosine": before,
                "candidate_median_cosine": after,
                "lift": after - before,
            }
        )
    positive_rate = sum(row["lift"] > 0 for row in per_prompt) / len(per_prompt)

    input_weight = np.ascontiguousarray(stable.T, dtype=np.float32)
    output_weight = np.ascontiguousarray(inverse.T, dtype=np.float32)
    cycle = output_weight @ input_weight
    cycle_rmse = float(
        np.linalg.norm(cycle - np.eye(width, dtype=np.float32)) / math.sqrt(width * width)
    )
    gates = {
        "cosine_lift": cosine_lift >= 0.03,
        "nrmse_ratio": nrmse_ratio <= 0.95,
        "positive_prompt_rate": positive_rate >= 2 / 3,
        "cycle_rmse": cycle_rmse <= 5e-5,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    input_path = args.out / "glm_activation_input_weight_f32.npy"
    output_path = args.out / "glm_activation_output_weight_f32.npy"
    np.save(input_path, input_weight, allow_pickle=False)
    np.save(output_path, output_weight, allow_pickle=False)
    report = {
        "format": "colorlm-anchored-ridge-activation-bridge-v1",
        "pairing": pairing,
        "split": {
            "seed": args.seed,
            "outer_train_prompts": [group.prompt_id for group in outer_train],
            "outer_test_prompts": [group.prompt_id for group in outer_test],
            "inner_train_prompts": [group.prompt_id for group in inner_train],
            "inner_test_prompts": [group.prompt_id for group in inner_test],
        },
        "regularization_selection": selection,
        "chosen_factor": chosen["factor"],
        "baseline_train_scale": baseline_scale,
        "candidate_train_scale": candidate_scale,
        "heldout": {
            "baseline": baseline_result,
            "candidate": candidate_result,
            "median_cosine_lift": cosine_lift,
            "nrmse_ratio": nrmse_ratio,
            "positive_prompt_rate": positive_rate,
            "per_prompt": per_prompt,
        },
        "stability": {
            "raw_singular_min": float(np.min(raw_singular)),
            "raw_singular_median": float(np.median(raw_singular)),
            "raw_singular_max": float(np.max(raw_singular)),
            "runtime_singular_clip": [0.25, 8.0],
            "runtime_global_scale": candidate_scale,
            "cycle_rmse": cycle_rmse,
        },
        "runtime": {
            "input_weight": {"file": input_path.name, "sha256": sha256(input_path)},
            "output_weight": {"file": output_path.name, "sha256": sha256(output_path)},
        },
        "promotion": {"gates": gates, "decision": "candidate" if all(gates.values()) else "reject"},
    }
    report_path = args.out / "anchored_ridge_bridge_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
