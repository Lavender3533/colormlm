"""Offline whole-prompt stability audit for the ColorLM v18 activation bridge.

This script reuses existing capture receipts only.  It never starts a model,
downloads weights, or changes an island/runtime package.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from activation_bridge import (
    BridgeError,
    PairedPrompt,
    concatenate,
    evaluate_map,
    fit_bridge,
    heldout_record_lifts,
    optimal_positive_scale,
    pair_receipts,
    relative_rmse,
    row_cosine,
    sha256_file,
    subsample_pair,
    write_json,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE = (
    ROOT / "fast16" / "research" / "coder_next_to_colorlm_orthogonal_f32.npy"
)
DEFAULT_BASE_RECEIPT = (
    Path(__file__).resolve().parent
    / "captures"
    / "run-20260731"
    / "base.receipt.json"
)
DEFAULT_DONOR_RECEIPT = (
    Path(__file__).resolve().parent
    / "captures"
    / "run-20260731b"
    / "donor.receipt.json"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "stability_report.json"
REPORT_FORMAT = "colorlm-activation-bridge-stability-v1"


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds 必须是逗号分隔的整数") from error
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("至少需要 3 个互不重复的 seed")
    return seeds


def summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def fit_split(
    train_groups: list[PairedPrompt],
    heldout_groups: list[PairedPrompt],
    baseline_donor_to_base: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(seed)
    train_base, train_donor = concatenate(train_groups)
    test_base, test_donor = concatenate(heldout_groups)
    train_base, train_donor = subsample_pair(
        train_base, train_donor, args.max_train_tokens, rng
    )
    test_base, test_donor = subsample_pair(
        test_base, test_donor, args.max_heldout_tokens, rng
    )
    (
        _input_weight,
        _output_weight,
        donor_to_base,
        median_norm_scale,
        raw_median_norm_scale,
        singular_values,
    ) = fit_bridge(
        train_base,
        train_donor,
        baseline_donor_to_base,
        args.prior_samples,
        args.min_scale,
        args.max_scale,
        args.bridge_strategy,
        args.rank_relative_tolerance,
    )
    candidate_base_to_donor = donor_to_base.T
    baseline_base_to_donor = baseline_donor_to_base.T
    if args.scale_strategy == "train_least_squares":
        raw_scale = optimal_positive_scale(
            train_base @ candidate_base_to_donor, train_donor
        )
    else:
        raw_scale = raw_median_norm_scale
    scale = float(np.clip(raw_scale, args.min_scale, args.max_scale))
    raw_baseline_scale = optimal_positive_scale(
        train_base @ baseline_base_to_donor, train_donor
    )
    baseline_scale = float(
        np.clip(raw_baseline_scale, args.min_scale, args.max_scale)
    )
    candidate_prediction = scale * (test_base @ candidate_base_to_donor)
    baseline_prediction = baseline_scale * (test_base @ baseline_base_to_donor)
    baseline_native_prediction = test_base @ baseline_base_to_donor
    candidate_metrics = evaluate_map(
        test_base, test_donor, candidate_base_to_donor, scale
    )
    baseline_metrics = evaluate_map(
        test_base, test_donor, baseline_base_to_donor, baseline_scale
    )
    baseline_native_metrics = evaluate_map(
        test_base, test_donor, baseline_base_to_donor, 1.0
    )
    candidate_optimal_metrics = evaluate_map(
        test_base, test_donor, candidate_base_to_donor
    )
    per_prompt, positive_rate = heldout_record_lifts(
        heldout_groups, baseline_base_to_donor, candidate_base_to_donor
    )
    cosine_lift = (
        candidate_metrics["cosine"]["median"]
        - baseline_metrics["cosine"]["median"]
    )
    nrmse_ratio = candidate_metrics["relative_rmse"] / max(
        baseline_metrics["relative_rmse"], 1e-12
    )
    split_gates = {
        "cosine_lift": cosine_lift >= args.min_cosine_lift,
        "nrmse_ratio": nrmse_ratio <= args.max_nrmse_ratio,
        "absolute_nrmse": (
            candidate_metrics["relative_rmse"] <= args.max_absolute_nrmse
        ),
        "positive_prompt_rate": (
            positive_rate >= args.min_positive_prompt_rate
        ),
        "scale_in_contract": args.min_scale <= raw_scale <= args.max_scale,
    }
    effective_rank = int(
        np.count_nonzero(
            singular_values
            > max(float(np.max(singular_values)) * args.rank_relative_tolerance, 1e-12)
        )
    )
    result = {
        "seed": seed,
        "train_prompt_ids": [item.prompt_id for item in train_groups],
        "heldout_prompt_ids": [item.prompt_id for item in heldout_groups],
        "train_tokens": len(train_base),
        "heldout_tokens": len(test_base),
        "scale": scale,
        "raw_scale": raw_scale,
        "median_norm_scale_diagnostic": median_norm_scale,
        "raw_median_norm_scale_diagnostic": raw_median_norm_scale,
        "baseline_train_scale": baseline_scale,
        "raw_baseline_train_scale": raw_baseline_scale,
        "effective_rank": effective_rank,
        "candidate": candidate_metrics,
        "candidate_optimal_scale_diagnostic": candidate_optimal_metrics,
        "baseline_train_calibrated": baseline_metrics,
        "baseline_native_scale_diagnostic": baseline_native_metrics,
        "median_cosine_lift_vs_embedding": cosine_lift,
        "nrmse_ratio_vs_train_calibrated_embedding": nrmse_ratio,
        "positive_prompt_rate": positive_rate,
        "per_prompt": per_prompt,
        "gates": split_gates,
        "decision": "pass" if all(split_gates.values()) else "fail",
        "_candidate_prediction": candidate_prediction,
        "_baseline_prediction": baseline_prediction,
        "_baseline_native_prediction": baseline_native_prediction,
        "_target": test_donor,
    }
    return result, candidate_base_to_donor


def public_split(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def materialize_runtime_candidate(
    output_dir: Path,
    base_to_donor: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    if output_dir.exists():
        raise BridgeError(f"候选输出目录已存在，拒绝覆盖: {output_dir}")
    output_dir.mkdir(parents=True)
    width = base_to_donor.shape[0]
    input_weight = np.ascontiguousarray(scale * base_to_donor.T, dtype=np.float32)
    output_weight = np.ascontiguousarray(
        (1.0 / scale) * base_to_donor,
        dtype=np.float32,
    )
    donor_to_base = np.ascontiguousarray(base_to_donor.T, dtype=np.float32)
    identity = np.eye(width, dtype=np.float32)
    orthogonality_rmse = float(
        np.linalg.norm(base_to_donor.T @ base_to_donor - identity)
        / np.sqrt(identity.size)
    )
    cycle_rmse = float(
        np.linalg.norm(output_weight @ input_weight - identity)
        / np.sqrt(identity.size)
    )
    if orthogonality_rmse > 5e-5 or cycle_rmse > 5e-5:
        raise BridgeError(
            "稳定候选运行矩阵不可逆: "
            f"orthogonality_rmse={orthogonality_rmse}, cycle_rmse={cycle_rmse}"
        )

    input_path = output_dir / "coder_activation_input_weight_f32.npy"
    output_path = output_dir / "coder_activation_output_weight_f32.npy"
    compatibility_path = output_dir / "coder_to_colorlm_activation_orthogonal_f32.npy"
    np.save(input_path, input_weight, allow_pickle=False)
    np.save(output_path, output_weight, allow_pickle=False)
    np.save(compatibility_path, donor_to_base, allow_pickle=False)
    return {
        "scale": scale,
        "orthogonality_rmse": orthogonality_rmse,
        "cycle_rmse": cycle_rmse,
        "input_weight": {
            "file": input_path.name,
            "sha256": sha256_file(input_path),
            "contract": "donor_column = W_in @ colorlm_column",
        },
        "output_weight": {
            "file": output_path.name,
            "sha256": sha256_file(output_path),
            "contract": "colorlm_delta_column = W_out @ donor_delta_column",
        },
        "orthogonal_compatibility": {
            "file": compatibility_path.name,
            "sha256": sha256_file(compatibility_path),
            "contract": "donor_row @ T = colorlm_row",
        },
    }


def aggregate_predictions(results: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = np.concatenate(
        [item["_candidate_prediction"] for item in results], axis=0
    )
    baseline = np.concatenate(
        [item["_baseline_prediction"] for item in results], axis=0
    )
    baseline_native = np.concatenate(
        [item["_baseline_native_prediction"] for item in results], axis=0
    )
    target = np.concatenate([item["_target"] for item in results], axis=0)
    candidate_cosine = row_cosine(candidate, target)
    baseline_cosine = row_cosine(baseline, target)
    cosine_lift = float(np.median(candidate_cosine) - np.median(baseline_cosine))
    candidate_nrmse = relative_rmse(candidate, target)
    baseline_nrmse = relative_rmse(baseline, target)
    return {
        "tokens": len(target),
        "candidate_median_cosine": float(np.median(candidate_cosine)),
        "baseline_median_cosine": float(np.median(baseline_cosine)),
        "median_cosine_lift_vs_embedding": cosine_lift,
        "candidate_relative_rmse": candidate_nrmse,
        "baseline_train_calibrated_relative_rmse": baseline_nrmse,
        "baseline_native_scale_relative_rmse_diagnostic": relative_rmse(
            baseline_native, target
        ),
        "nrmse_ratio_vs_train_calibrated_embedding": candidate_nrmse
        / max(baseline_nrmse, 1e-12),
    }


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    groups, pairing = pair_receipts(args.base_receipt, args.donor_receipt)
    if len(groups) < 3:
        raise BridgeError("稳定性检查至少需要 3 条完整配对 prompt")
    width = groups[0].base.shape[1]
    if any(item.base.shape[1] != width for item in groups):
        raise BridgeError("配对隐藏态宽度不一致")
    baseline = np.asarray(np.load(args.baseline, allow_pickle=False), dtype=np.float32)
    if baseline.shape != (width, width) or not np.isfinite(baseline).all():
        raise BridgeError(
            f"基线桥形状或数值无效: {baseline.shape}, expected={(width, width)}"
        )

    all_base, _all_donor = concatenate(groups)
    full_result, full_map = fit_split(
        groups,
        groups,
        baseline,
        args,
        seed=args.seeds[0],
    )

    loto_results: list[dict[str, Any]] = []
    loto_prediction_stability: list[float] = []
    for heldout_index, heldout in enumerate(groups):
        train = [item for index, item in enumerate(groups) if index != heldout_index]
        split, candidate_map = fit_split(
            train,
            [heldout],
            baseline,
            args,
            seed=args.seeds[0] + heldout_index,
        )
        prediction_stability = float(
            np.median(
                row_cosine(all_base @ candidate_map, all_base @ full_map)
            )
        )
        split["prediction_cosine_vs_full_fit"] = prediction_stability
        loto_prediction_stability.append(prediction_stability)
        loto_results.append(split)

    loto_aggregate = aggregate_predictions(loto_results)
    loto_positive_rate = sum(
        item["median_cosine_lift_vs_embedding"] > 0 for item in loto_results
    ) / len(loto_results)
    loto_aggregate["positive_prompt_rate"] = loto_positive_rate
    loto_aggregate["prediction_cosine_vs_full_fit"] = summary(
        loto_prediction_stability
    )
    loto_gates = {
        "cosine_lift": (
            loto_aggregate["median_cosine_lift_vs_embedding"]
            >= args.min_cosine_lift
        ),
        "nrmse_ratio": (
            loto_aggregate["nrmse_ratio_vs_train_calibrated_embedding"]
            <= args.max_nrmse_ratio
        ),
        "absolute_nrmse": (
            loto_aggregate["candidate_relative_rmse"]
            <= args.max_absolute_nrmse
        ),
        "positive_prompt_rate": (
            loto_positive_rate >= args.min_positive_prompt_rate
        ),
        "median_prediction_stability": (
            loto_aggregate["prediction_cosine_vs_full_fit"]["median"]
            >= args.min_median_prediction_stability
        ),
        "worst_prediction_stability": (
            loto_aggregate["prediction_cosine_vs_full_fit"]["min"]
            >= args.min_worst_prediction_stability
        ),
        "all_scales_in_contract": all(
            args.min_scale <= item["raw_scale"] <= args.max_scale
            for item in loto_results
        ),
    }
    loto_aggregate["gates"] = loto_gates
    loto_aggregate["decision"] = "pass" if all(loto_gates.values()) else "fail"

    seed_results: list[dict[str, Any]] = []
    heldout_signatures: list[tuple[str, ...]] = []
    heldout_count = max(1, int(round(len(groups) * args.heldout_fraction)))
    heldout_count = min(heldout_count, len(groups) - 1)
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(groups))
        heldout_indices = set(int(index) for index in order[:heldout_count])
        heldout = [item for index, item in enumerate(groups) if index in heldout_indices]
        train = [item for index, item in enumerate(groups) if index not in heldout_indices]
        heldout_signatures.append(tuple(item.prompt_id for item in heldout))
        split, candidate_map = fit_split(train, heldout, baseline, args, seed)
        split["prediction_cosine_vs_full_fit"] = float(
            np.median(row_cosine(all_base @ candidate_map, all_base @ full_map))
        )
        seed_results.append(split)

    seed_pass_rate = sum(item["decision"] == "pass" for item in seed_results) / len(
        seed_results
    )
    seed_gates = {
        "unique_heldout_splits": len(set(heldout_signatures)) == len(args.seeds),
        "pass_rate": seed_pass_rate >= args.min_seed_pass_rate,
        "no_cosine_regression": min(
            item["median_cosine_lift_vs_embedding"] for item in seed_results
        )
        > 0,
        "no_nrmse_regression": max(
            item["nrmse_ratio_vs_train_calibrated_embedding"] for item in seed_results
        )
        <= 1.0,
        "worst_prediction_stability": min(
            item["prediction_cosine_vs_full_fit"] for item in seed_results
        )
        >= args.min_worst_prediction_stability,
    }
    seed_summary = {
        "seeds": args.seeds,
        "heldout_fraction": args.heldout_fraction,
        "heldout_prompts_per_seed": heldout_count,
        "unique_heldout_splits": len(set(heldout_signatures)),
        "pass_rate": seed_pass_rate,
        "cosine_lift": summary(
            item["median_cosine_lift_vs_embedding"] for item in seed_results
        ),
        "nrmse_ratio": summary(
            item["nrmse_ratio_vs_train_calibrated_embedding"] for item in seed_results
        ),
        "candidate_absolute_nrmse": summary(
            item["candidate"]["relative_rmse"] for item in seed_results
        ),
        "scale": summary(item["scale"] for item in seed_results),
        "prediction_cosine_vs_full_fit": summary(
            item["prediction_cosine_vs_full_fit"] for item in seed_results
        ),
        "gates": seed_gates,
        "decision": "pass" if all(seed_gates.values()) else "fail",
    }

    data_gates = {
        "enough_prompts": len(groups) >= args.min_prompts,
        "enough_tokens": len(all_base) >= args.min_tokens,
        "all_prompts_pair": len(groups) == pairing["common_prompts"],
    }
    gates = {
        "data_contract": all(data_gates.values()),
        "loto": all(loto_gates.values()),
        "multi_seed": all(seed_gates.values()),
    }
    decision = "stable_candidate" if all(gates.values()) else "reject"
    report = {
        "format": REPORT_FORMAT,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "inputs": {
            "base_receipt": str(args.base_receipt.resolve()),
            "base_receipt_sha256": sha256_file(args.base_receipt),
            "donor_receipt": str(args.donor_receipt.resolve()),
            "donor_receipt_sha256": sha256_file(args.donor_receipt),
            "baseline": str(args.baseline.resolve()),
            "baseline_sha256": sha256_file(args.baseline),
            "paired_prompts": len(groups),
            "paired_tokens": len(all_base),
            "hidden_width": width,
            "mean_match_coverage": pairing["mean_match_coverage"],
        },
        "method": {
            "split_unit": "whole prompt",
            "bridge_strategy": args.bridge_strategy,
            "prior_equivalent_samples": args.prior_samples,
            "candidate_scale_source": args.scale_strategy,
            "baseline_scale_source": "training-prompt least squares",
            "loto_folds": len(groups),
            "multi_seed_values": args.seeds,
            "rank_relative_tolerance": args.rank_relative_tolerance,
        },
        "thresholds": {
            "min_prompts": args.min_prompts,
            "min_tokens": args.min_tokens,
            "min_cosine_lift": args.min_cosine_lift,
            "max_nrmse_ratio": args.max_nrmse_ratio,
            "max_absolute_nrmse": args.max_absolute_nrmse,
            "min_positive_prompt_rate": args.min_positive_prompt_rate,
            "min_seed_pass_rate": args.min_seed_pass_rate,
            "min_median_prediction_stability": (
                args.min_median_prediction_stability
            ),
            "min_worst_prediction_stability": args.min_worst_prediction_stability,
        },
        "data_gates": data_gates,
        "full_fit_diagnostic": {
            "train_tokens": len(all_base),
            "scale": full_result["scale"],
            "effective_rank": full_result["effective_rank"],
            "rank_fraction_of_hidden_width": full_result["effective_rank"] / width,
            "in_sample_candidate_relative_rmse": full_result["candidate"][
                "relative_rmse"
            ],
        },
        "loto": {
            "aggregate": loto_aggregate,
            "folds": [public_split(item) for item in loto_results],
        },
        "multi_seed": {
            "summary": seed_summary,
            "runs": [public_split(item) for item in seed_results],
        },
        "gates": gates,
    }
    if args.candidate_output_dir is not None:
        if decision != "stable_candidate":
            raise BridgeError(
                "稳定性总门未通过，拒绝生成运行矩阵: "
                f"decision={decision}"
            )
        report["runtime"] = materialize_runtime_candidate(
            args.candidate_output_dir,
            full_map,
            float(full_result["scale"]),
        )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if decision == "stable_candidate" or args.allow_rejected else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读运行整提示 LOTO 与多 seed 的 v18 激活桥稳定性短检查"
    )
    parser.add_argument("--base-receipt", type=Path, default=DEFAULT_BASE_RECEIPT)
    parser.add_argument("--donor-receipt", type=Path, default=DEFAULT_DONOR_RECEIPT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--candidate-output-dir",
        type=Path,
        help="仅在总决策stable_candidate时，用全量配对状态生成带哈希的运行矩阵。",
    )
    parser.add_argument("--prior-samples", type=float, default=0.0)
    parser.add_argument(
        "--bridge-strategy",
        choices=("full_procrustes", "nullspace_anchored"),
        default="full_procrustes",
    )
    parser.add_argument(
        "--scale-strategy",
        choices=("median_norm_ratio", "train_least_squares"),
        default="median_norm_ratio",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("18,19,20,21,22"))
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--max-train-tokens", type=int, default=4096)
    parser.add_argument("--max-heldout-tokens", type=int, default=2048)
    parser.add_argument("--min-prompts", type=int, default=6)
    parser.add_argument("--min-tokens", type=int, default=512)
    parser.add_argument("--min-cosine-lift", type=float, default=0.03)
    parser.add_argument("--max-nrmse-ratio", type=float, default=0.95)
    parser.add_argument("--max-absolute-nrmse", type=float, default=0.95)
    parser.add_argument("--min-positive-prompt-rate", type=float, default=0.67)
    parser.add_argument("--min-seed-pass-rate", type=float, default=0.8)
    parser.add_argument("--min-median-prediction-stability", type=float, default=0.95)
    parser.add_argument("--min-worst-prediction-stability", type=float, default=0.90)
    parser.add_argument("--min-scale", type=float, default=0.25)
    parser.add_argument("--max-scale", type=float, default=4.0)
    parser.add_argument("--rank-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--allow-rejected", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.heldout_fraction < 0.5:
        raise BridgeError("heldout-fraction 必须在 (0, 0.5) 内")
    if args.prior_samples < 0:
        raise BridgeError("prior-samples 不能为负")
    if args.bridge_strategy == "nullspace_anchored" and args.prior_samples > 0:
        raise BridgeError("nullspace_anchored要求--prior-samples 0")
    if args.rank_relative_tolerance <= 0:
        raise BridgeError("rank-relative-tolerance 必须为正数")
    if args.min_scale <= 0 or args.max_scale <= args.min_scale:
        raise BridgeError("scale 边界无效")
    for name in (
        "min_positive_prompt_rate",
        "min_seed_pass_rate",
        "min_median_prediction_stability",
        "min_worst_prediction_stability",
    ):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise BridgeError(f"{name} 必须位于 [0, 1]")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        return run(args)
    except BridgeError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
