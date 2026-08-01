"""编译跨模型原生状态门户，不做反向传播或 epoch 训练。

输入是同一批 token/任务在北极星与巨型 donor 中采集的成对隐藏态。工具只在
训练任务张成的子空间里，对已有 anchor 做闭式核岭残差修正；未观测子空间仍由
anchor 决定。评估严格按完整任务 leave-one-group-out，并可检查 donor router
top-k 是否在映射后保持。

该工具通过只表示“坐标门户几何可进入器官 A/B”，不表示模型质量已提升。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FORMAT = "polaris-native-state-portal-v1"
REQUIRED_KEYS = (
    "sample_ids",
    "base_input",
    "donor_input",
    "donor_output",
    "base_output",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def group_id(sample_id: str) -> str:
    head, separator, tail = sample_id.rpartition(":")
    if separator and tail.isdigit():
        return head
    return sample_id


def load_pairs(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"成对激活缺少字段：{missing}")
        result = {key: np.asarray(archive[key]) for key in REQUIRED_KEYS}

    sample_ids = result["sample_ids"]
    if sample_ids.ndim != 1:
        raise ValueError("sample_ids 必须是一维数组")
    count = sample_ids.shape[0]
    normalized_ids = np.asarray([str(value) for value in sample_ids], dtype=np.str_)
    if len(set(normalized_ids.tolist())) != count:
        raise ValueError("sample_ids 必须唯一")
    result["sample_ids"] = normalized_ids

    for key in REQUIRED_KEYS[1:]:
        value = np.asarray(result[key], dtype=np.float32)
        if value.ndim != 2 or value.shape[0] != count:
            raise ValueError(f"{key} 形状异常：{value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} 含 NaN/Inf")
        result[key] = value
    return result


def load_anchors(
    path: Path | None,
    base_input_width: int,
    donor_input_width: int,
    donor_output_width: int,
    base_output_width: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    expected_input = (base_input_width, donor_input_width)
    expected_output = (donor_output_width, base_output_width)
    if path is None:
        return (
            np.zeros(expected_input, dtype=np.float32),
            np.zeros(expected_output, dtype=np.float32),
            "zero_anchor_for_diagnostics_only",
        )
    with np.load(path, allow_pickle=False) as archive:
        if "input_anchor" not in archive or "output_anchor" not in archive:
            raise ValueError("anchor NPZ 必须含 input_anchor/output_anchor")
        input_anchor = np.asarray(archive["input_anchor"], dtype=np.float32)
        output_anchor = np.asarray(archive["output_anchor"], dtype=np.float32)
    if input_anchor.shape != expected_input:
        raise ValueError(f"input_anchor 形状异常：{input_anchor.shape} != {expected_input}")
    if output_anchor.shape != expected_output:
        raise ValueError(f"output_anchor 形状异常：{output_anchor.shape} != {expected_output}")
    if not np.all(np.isfinite(input_anchor)) or not np.all(np.isfinite(output_anchor)):
        raise ValueError("anchor 含 NaN/Inf")
    return input_anchor, output_anchor, "provided_anchor"


@dataclass
class Portal:
    anchor: np.ndarray
    source_mean: np.ndarray
    source_scale: float
    residual_mean: np.ndarray
    left: np.ndarray
    right: np.ndarray

    def predict(self, source: np.ndarray) -> np.ndarray:
        source = np.asarray(source, dtype=np.float32)
        normalized = (source - self.source_mean[None, :]) / self.source_scale
        correction = (normalized @ self.left) @ self.right
        return source @ self.anchor + self.residual_mean[None, :] + correction


def factor_product(left: np.ndarray, right: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对 ``left @ right`` 做小矩阵 SVD，不物化宽×宽修正矩阵。"""
    q_left, r_left = np.linalg.qr(left, mode="reduced")
    q_right, r_right = np.linalg.qr(right.T, mode="reduced")
    middle = r_left @ r_right.T
    u, singular, vt = np.linalg.svd(middle, full_matrices=False)
    keep = min(rank, singular.shape[0])
    root = np.sqrt(np.maximum(singular[:keep], 0.0)).astype(np.float32)
    compressed_left = (q_left @ u[:, :keep]).astype(np.float32)
    compressed_left *= root[None, :]
    compressed_right = (vt[:keep] @ q_right.T).astype(np.float32)
    compressed_right *= root[:, None]
    return compressed_left, compressed_right, singular.astype(np.float32)


def fit_portal(
    source: np.ndarray,
    target: np.ndarray,
    anchor: np.ndarray,
    ridge: float,
    rank: int,
) -> tuple[Portal, dict]:
    if source.shape[0] < 2:
        raise ValueError("门户拟合至少需要两个样本")
    source_mean = np.mean(source, axis=0, dtype=np.float64).astype(np.float32)
    centered = source - source_mean[None, :]
    source_scale = float(np.sqrt(np.mean(np.square(centered), dtype=np.float64)))
    source_scale = max(source_scale, 1.0e-8)
    normalized = centered / source_scale

    residual = target - source @ anchor
    residual_mean = np.mean(residual, axis=0, dtype=np.float64).astype(np.float32)
    residual_centered = residual - residual_mean[None, :]
    gram = normalized @ normalized.T
    regularizer = ridge * max(float(np.trace(gram)) / source.shape[0], 1.0e-8)
    coefficients = np.linalg.solve(
        gram + regularizer * np.eye(source.shape[0], dtype=np.float32),
        residual_centered,
    ).astype(np.float32)
    left, right, singular = factor_product(normalized.T, coefficients, rank)
    portal = Portal(
        anchor=np.asarray(anchor, dtype=np.float32),
        source_mean=source_mean,
        source_scale=source_scale,
        residual_mean=residual_mean,
        left=left,
        right=right,
    )
    total_energy = float(np.sum(np.square(singular), dtype=np.float64))
    kept_energy = float(np.sum(np.square(singular[: left.shape[1]]), dtype=np.float64))
    return portal, {
        "requested_rank": rank,
        "effective_rank": int(left.shape[1]),
        "ridge": ridge,
        "scaled_regularizer": regularizer,
        "correction_singular_max": float(singular[0]) if singular.size else 0.0,
        "correction_energy_retained": kept_energy / max(total_energy, 1.0e-30),
    }


def sample_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, np.ndarray]:
    reference_norm = np.linalg.norm(reference, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    cosine = np.sum(reference * candidate, axis=1) / np.maximum(
        reference_norm * candidate_norm, 1.0e-20
    )
    nrmse = np.linalg.norm(candidate - reference, axis=1) / np.maximum(
        reference_norm, 1.0e-20
    )
    return {"cosine": cosine, "nrmse": nrmse}


def metric_summary(values: dict[str, np.ndarray]) -> dict:
    return {
        "cosine_mean": float(np.mean(values["cosine"])),
        "cosine_median": float(np.median(values["cosine"])),
        "cosine_p10": float(np.quantile(values["cosine"], 0.10)),
        "nrmse_mean": float(np.mean(values["nrmse"])),
        "nrmse_median": float(np.median(values["nrmse"])),
    }


def topk_recall(reference_scores: np.ndarray, candidate_scores: np.ndarray, k: int) -> np.ndarray:
    experts = reference_scores.shape[1]
    if k < 1 or k > experts:
        raise ValueError(f"router top-k 越界：{k} vs {experts}")
    reference = np.argpartition(reference_scores, experts - k, axis=1)[:, -k:]
    candidate = np.argpartition(candidate_scores, experts - k, axis=1)[:, -k:]
    recalls = []
    for expected, actual in zip(reference, candidate, strict=True):
        recalls.append(len(set(expected.tolist()) & set(actual.tolist())) / k)
    return np.asarray(recalls, dtype=np.float32)


def cross_validate(
    pairs: dict[str, np.ndarray],
    input_anchor: np.ndarray,
    output_anchor: np.ndarray,
    ridge: float,
    rank: int,
    router: np.ndarray | None,
    router_topk: int,
) -> dict:
    ids = pairs["sample_ids"]
    groups = np.asarray([group_id(value) for value in ids], dtype=np.str_)
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] < 3:
        raise ValueError("严格整任务留出至少需要三个不同 group")

    folds = []
    input_reference_all = []
    input_anchor_all = []
    input_candidate_all = []
    output_reference_all = []
    output_anchor_all = []
    output_candidate_all = []
    router_anchor_all = []
    router_candidate_all = []

    for held_out in unique_groups:
        test = groups == held_out
        train = ~test
        input_portal, _ = fit_portal(
            pairs["base_input"][train],
            pairs["donor_input"][train],
            input_anchor,
            ridge,
            rank,
        )
        output_portal, _ = fit_portal(
            pairs["donor_output"][train],
            pairs["base_output"][train],
            output_anchor,
            ridge,
            rank,
        )
        input_reference = pairs["donor_input"][test]
        input_baseline = pairs["base_input"][test] @ input_anchor
        input_candidate = input_portal.predict(pairs["base_input"][test])
        output_reference = pairs["base_output"][test]
        output_baseline = pairs["donor_output"][test] @ output_anchor
        output_candidate = output_portal.predict(pairs["donor_output"][test])
        input_base_metrics = sample_metrics(input_reference, input_baseline)
        input_new_metrics = sample_metrics(input_reference, input_candidate)
        output_base_metrics = sample_metrics(output_reference, output_baseline)
        output_new_metrics = sample_metrics(output_reference, output_candidate)
        fold = {
            "group": str(held_out),
            "train_samples": int(np.sum(train)),
            "test_samples": int(np.sum(test)),
            "input_anchor": metric_summary(input_base_metrics),
            "input_portal": metric_summary(input_new_metrics),
            "output_anchor": metric_summary(output_base_metrics),
            "output_portal": metric_summary(output_new_metrics),
            "input_cosine_gain": float(
                np.mean(input_new_metrics["cosine"] - input_base_metrics["cosine"])
            ),
            "output_cosine_gain": float(
                np.mean(output_new_metrics["cosine"] - output_base_metrics["cosine"])
            ),
        }
        if router is not None:
            true_scores = input_reference @ router.T
            anchor_scores = input_baseline @ router.T
            candidate_scores = input_candidate @ router.T
            anchor_recall = topk_recall(true_scores, anchor_scores, router_topk)
            candidate_recall = topk_recall(true_scores, candidate_scores, router_topk)
            fold["router_topk"] = {
                "k": router_topk,
                "anchor_recall_mean": float(np.mean(anchor_recall)),
                "portal_recall_mean": float(np.mean(candidate_recall)),
                "recall_gain": float(np.mean(candidate_recall - anchor_recall)),
            }
            router_anchor_all.append(anchor_recall)
            router_candidate_all.append(candidate_recall)
        folds.append(fold)
        input_reference_all.append(input_reference)
        input_anchor_all.append(input_baseline)
        input_candidate_all.append(input_candidate)
        output_reference_all.append(output_reference)
        output_anchor_all.append(output_baseline)
        output_candidate_all.append(output_candidate)

    input_reference = np.concatenate(input_reference_all)
    input_baseline = np.concatenate(input_anchor_all)
    input_candidate = np.concatenate(input_candidate_all)
    output_reference = np.concatenate(output_reference_all)
    output_baseline = np.concatenate(output_anchor_all)
    output_candidate = np.concatenate(output_candidate_all)
    input_base_metrics = sample_metrics(input_reference, input_baseline)
    input_new_metrics = sample_metrics(input_reference, input_candidate)
    output_base_metrics = sample_metrics(output_reference, output_baseline)
    output_new_metrics = sample_metrics(output_reference, output_candidate)

    aggregate = {
        "input_anchor": metric_summary(input_base_metrics),
        "input_portal": metric_summary(input_new_metrics),
        "output_anchor": metric_summary(output_base_metrics),
        "output_portal": metric_summary(output_new_metrics),
        "input_cosine_gain": float(
            np.mean(input_new_metrics["cosine"] - input_base_metrics["cosine"])
        ),
        "output_cosine_gain": float(
            np.mean(output_new_metrics["cosine"] - output_base_metrics["cosine"])
        ),
        "input_nrmse_ratio": float(
            np.mean(input_new_metrics["nrmse"])
            / max(float(np.mean(input_base_metrics["nrmse"])), 1.0e-20)
        ),
        "output_nrmse_ratio": float(
            np.mean(output_new_metrics["nrmse"])
            / max(float(np.mean(output_base_metrics["nrmse"])), 1.0e-20)
        ),
        "positive_input_groups": int(sum(fold["input_cosine_gain"] > 0 for fold in folds)),
        "positive_output_groups": int(sum(fold["output_cosine_gain"] > 0 for fold in folds)),
        "groups": len(folds),
    }
    if router is not None:
        router_anchor = np.concatenate(router_anchor_all)
        router_candidate = np.concatenate(router_candidate_all)
        aggregate["router_topk"] = {
            "k": router_topk,
            "anchor_recall_mean": float(np.mean(router_anchor)),
            "portal_recall_mean": float(np.mean(router_candidate)),
            "recall_gain": float(np.mean(router_candidate - router_anchor)),
        }
    return {"folds": folds, "aggregate": aggregate}


def gate_decision(cross_validation: dict, anchor_kind: str, router_present: bool) -> dict:
    aggregate = cross_validation["aggregate"]
    groups = aggregate["groups"]
    checks = {
        "nonzero_anchor_required": anchor_kind == "provided_anchor",
        "input_cosine_gain_at_least_0_10": aggregate["input_cosine_gain"] >= 0.10,
        "output_cosine_gain_at_least_0_10": aggregate["output_cosine_gain"] >= 0.10,
        "input_nrmse_ratio_at_most_0_90": aggregate["input_nrmse_ratio"] <= 0.90,
        "output_nrmse_ratio_at_most_0_90": aggregate["output_nrmse_ratio"] <= 0.90,
        "positive_input_groups_at_least_80_percent": aggregate["positive_input_groups"] >= math.ceil(0.8 * groups),
        "positive_output_groups_at_least_80_percent": aggregate["positive_output_groups"] >= math.ceil(0.8 * groups),
    }
    if router_present:
        route = aggregate["router_topk"]
        checks["router_recall_at_least_0_70"] = route["portal_recall_mean"] >= 0.70
        checks["router_recall_gain_at_least_0_10"] = route["recall_gain"] >= 0.10
    passed = all(checks.values())
    return {
        "status": "portal_geometry_pass" if passed else "portal_geometry_rejected",
        "checks": checks,
        "passed": passed,
        "claim_boundary": (
            "通过仅允许进入连续器官短 NLL/任务 A/B；不代表北极星能力提升，"
            "更不代表追上 Claude/GPT。"
        ),
    }


def save_array(path: Path, array: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array, dtype="<f2")
    path.write_bytes(contiguous.tobytes(order="C"))
    return {
        "file": path.name,
        "shape": list(contiguous.shape),
        "dtype": "float16-le",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def export_portal(
    output_dir: Path,
    input_portal: Portal,
    output_portal: Portal,
    input_fit: dict,
    output_fit: dict,
    source: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "input_anchor": save_array(output_dir / "input_anchor.f16", input_portal.anchor),
        "input_source_mean": save_array(output_dir / "input_source_mean.f16", input_portal.source_mean),
        "input_residual_mean": save_array(output_dir / "input_residual_mean.f16", input_portal.residual_mean),
        "input_left": save_array(output_dir / "input_left.f16", input_portal.left),
        "input_right": save_array(output_dir / "input_right.f16", input_portal.right),
        "output_anchor": save_array(output_dir / "output_anchor.f16", output_portal.anchor),
        "output_source_mean": save_array(output_dir / "output_source_mean.f16", output_portal.source_mean),
        "output_residual_mean": save_array(output_dir / "output_residual_mean.f16", output_portal.residual_mean),
        "output_left": save_array(output_dir / "output_left.f16", output_portal.left),
        "output_right": save_array(output_dir / "output_right.f16", output_portal.right),
    }
    input_macs = int(np.prod(input_portal.anchor.shape)) + int(
        input_portal.left.shape[0] * input_portal.left.shape[1]
        + input_portal.right.shape[0] * input_portal.right.shape[1]
    )
    output_macs = int(np.prod(output_portal.anchor.shape)) + int(
        output_portal.left.shape[0] * output_portal.left.shape[1]
        + output_portal.right.shape[0] * output_portal.right.shape[1]
    )
    manifest = {
        "format": FORMAT,
        "source": source,
        "execution": {
            "input": "base @ anchor + residual_mean + (((base-source_mean)/scale) @ left) @ right",
            "output": "donor @ anchor + residual_mean + (((donor-source_mean)/scale) @ left) @ right",
            "input_source_scale": input_portal.source_scale,
            "output_source_scale": output_portal.source_scale,
        },
        "fit": {"input": input_fit, "output": output_fit},
        "arrays": arrays,
        "runtime_budget": {
            "input_macs": input_macs,
            "output_macs": output_macs,
            "total_macs_per_organ_invocation": input_macs + output_macs,
            "resident_f16_bytes": sum(item["bytes"] for item in arrays.values()),
        },
    }
    write_json(output_dir / "portal.json", manifest)
    return manifest


def fit_command(args: argparse.Namespace) -> dict:
    pairs = load_pairs(args.pairs)
    input_anchor, output_anchor, anchor_kind = load_anchors(
        args.anchor,
        pairs["base_input"].shape[1],
        pairs["donor_input"].shape[1],
        pairs["donor_output"].shape[1],
        pairs["base_output"].shape[1],
    )
    router = None
    if args.router is not None:
        router = np.asarray(np.load(args.router, allow_pickle=False), dtype=np.float32)
        expected_width = pairs["donor_input"].shape[1]
        if router.ndim != 2 or router.shape[1] != expected_width:
            raise ValueError(f"router 形状异常：{router.shape}，宽度应为 {expected_width}")
        if not np.all(np.isfinite(router)):
            raise ValueError("router 含 NaN/Inf")

    cross_validation = cross_validate(
        pairs,
        input_anchor,
        output_anchor,
        args.ridge,
        args.rank,
        router,
        args.router_topk,
    )
    decision = gate_decision(cross_validation, anchor_kind, router is not None)
    report = {
        "format": "polaris-native-state-portal-evaluation-v1",
        "source": {
            "pairs": str(args.pairs.resolve()),
            "anchor": str(args.anchor.resolve()) if args.anchor else None,
            "anchor_kind": anchor_kind,
            "router": str(args.router.resolve()) if args.router else None,
            "samples": int(pairs["sample_ids"].shape[0]),
            "groups": len(set(group_id(value) for value in pairs["sample_ids"])),
            "dimensions": {key: list(pairs[key].shape) for key in REQUIRED_KEYS[1:]},
        },
        "method": {
            "kind": "closed_form_nullspace_anchored_kernel_ridge_residual",
            "backpropagation": False,
            "epochs": 0,
            "rank": args.rank,
            "ridge": args.ridge,
            "split": "leave-one-complete-group-out",
        },
        "cross_validation": cross_validation,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "evaluation.json", report)
    if decision["passed"]:
        input_portal, input_fit = fit_portal(
            pairs["base_input"], pairs["donor_input"], input_anchor, args.ridge, args.rank
        )
        output_portal, output_fit = fit_portal(
            pairs["donor_output"], pairs["base_output"], output_anchor, args.ridge, args.rank
        )
        report["runtime_package"] = export_portal(
            args.output_dir / "runtime",
            input_portal,
            output_portal,
            input_fit,
            output_fit,
            report["source"],
        )
        write_json(args.output_dir / "evaluation.json", report)
    return report


def make_fixture(directory: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(20260801)
    groups = 8
    per_group = 10
    count = groups * per_group
    base_width = 32
    donor_width = 48
    latent_width = 14
    base_input = rng.standard_normal((count, base_width), dtype=np.float32)
    true_input = rng.standard_normal((base_width, donor_width), dtype=np.float32) / np.sqrt(base_width)
    input_anchor = true_input + 0.24 * rng.standard_normal(true_input.shape, dtype=np.float32)
    donor_input = base_input @ true_input + 0.01 * rng.standard_normal((count, donor_width), dtype=np.float32)

    donor_output = rng.standard_normal((count, donor_width), dtype=np.float32)
    true_output = rng.standard_normal((donor_width, base_width), dtype=np.float32) / np.sqrt(donor_width)
    output_anchor = true_output + 0.24 * rng.standard_normal(true_output.shape, dtype=np.float32)
    base_output = donor_output @ true_output + 0.01 * rng.standard_normal((count, base_width), dtype=np.float32)
    # 共享低秩结构让整任务留出可复现，但每组保留独立噪声。
    latent = rng.standard_normal((count, latent_width), dtype=np.float32)
    base_input += 0.08 * latent @ rng.standard_normal((latent_width, base_width), dtype=np.float32)
    donor_output += 0.08 * latent @ rng.standard_normal((latent_width, donor_width), dtype=np.float32)
    donor_input = base_input @ true_input + 0.01 * rng.standard_normal((count, donor_width), dtype=np.float32)
    base_output = donor_output @ true_output + 0.01 * rng.standard_normal((count, base_width), dtype=np.float32)

    sample_ids = np.asarray(
        [f"fixture-task-{group:02d}:{token:04d}" for group in range(groups) for token in range(per_group)],
        dtype=np.str_,
    )
    pairs_path = directory / "pairs.npz"
    anchor_path = directory / "anchor.npz"
    router_path = directory / "router.npy"
    np.savez(
        pairs_path,
        sample_ids=sample_ids,
        base_input=base_input,
        donor_input=donor_input,
        donor_output=donor_output,
        base_output=base_output,
    )
    np.savez(anchor_path, input_anchor=input_anchor, output_anchor=output_anchor)
    router = rng.standard_normal((24, donor_width), dtype=np.float32)
    np.save(router_path, router, allow_pickle=False)
    return pairs_path, anchor_path, router_path


def selftest_command(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="polaris-portal-") as temporary:
        pairs, anchor, router = make_fixture(Path(temporary))
        fit_args = argparse.Namespace(
            pairs=pairs,
            anchor=anchor,
            router=router,
            router_topk=4,
            ridge=1.0e-4,
            rank=32,
            output_dir=args.output_dir / "fixture_run",
        )
        evaluation = fit_command(fit_args)
    checks = {
        "geometry_gate_passed": bool(evaluation["decision"]["passed"]),
        "runtime_manifest_written": (args.output_dir / "fixture_run/runtime/portal.json").is_file(),
        "eight_group_loto": evaluation["cross_validation"]["aggregate"]["groups"] == 8,
        "router_evaluated": "router_topk" in evaluation["cross_validation"]["aggregate"],
        "no_backpropagation": evaluation["method"]["backpropagation"] is False,
        "zero_epochs": evaluation["method"]["epochs"] == 0,
    }
    report = {
        "format": "polaris-native-state-portal-selftest-v1",
        "checks": checks,
        "all_passed": all(checks.values()),
        "fixture_decision": evaluation["decision"],
        "aggregate": evaluation["cross_validation"]["aggregate"],
        "claim_boundary": "合成测试只验证门户编译和留出门，不是巨型 donor 能力证据。",
    }
    write_json(args.output_dir / "selftest_report.json", report)
    if not report["all_passed"]:
        raise RuntimeError(f"门户自检失败：{checks}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit", help="拟合并整任务留出评估真实成对激活")
    fit_parser.add_argument("--pairs", type=Path, required=True)
    fit_parser.add_argument("--anchor", type=Path)
    fit_parser.add_argument("--router", type=Path)
    fit_parser.add_argument("--router-topk", type=int, default=6)
    fit_parser.add_argument("--ridge", type=float, default=0.01)
    fit_parser.add_argument("--rank", type=int, default=64)
    fit_parser.add_argument("--output-dir", type=Path, required=True)

    selftest_parser = subparsers.add_parser("selftest", help="运行无模型、无下载的合成自检")
    selftest_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "selftest",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "fit":
        report = fit_command(args)
    else:
        report = selftest_command(args)
    print(json.dumps(report.get("decision", report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
