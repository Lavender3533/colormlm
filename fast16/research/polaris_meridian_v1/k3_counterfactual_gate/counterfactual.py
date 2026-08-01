"""用强制 no-op/donor next-token NLL 标定 FullDepth 连续态能力门。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .runtime import GATE_FORMAT, ContractError


REPORT_FORMAT = "polaris-k3-counterfactual-calibration-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def target_nll(logits: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    target_ids = np.asarray(target_ids)
    if logits.ndim != 2 or target_ids.shape != (logits.shape[0],):
        raise ContractError("logits/target_ids 形状异常")
    if not np.issubdtype(target_ids.dtype, np.integer):
        raise ContractError("target_ids 必须是整数")
    if np.any(target_ids < 0) or np.any(target_ids >= logits.shape[1]):
        raise ContractError("target_ids 越界")
    if not np.all(np.isfinite(logits)):
        raise ContractError("logits 含 NaN/Inf")
    maximum = np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(logits - maximum), axis=1)) + maximum[:, 0]
    return logsumexp - logits[np.arange(logits.shape[0]), target_ids.astype(np.int64)]


@dataclass(frozen=True)
class CounterfactualBatch:
    hidden: np.ndarray
    no_op_logits: np.ndarray
    donor_logits: np.ndarray
    target_ids: np.ndarray
    task_ids: tuple[str, ...]

    def validate(self) -> None:
        hidden = np.asarray(self.hidden)
        if hidden.ndim != 2 or hidden.shape[0] < 4 or hidden.shape[1] < 1:
            raise ContractError("hidden 必须是 [N,D]，N>=4")
        if not np.all(np.isfinite(hidden)):
            raise ContractError("hidden 含 NaN/Inf")
        if self.no_op_logits.shape != self.donor_logits.shape:
            raise ContractError("no-op/donor logits 形状不一致")
        if self.no_op_logits.ndim != 2 or self.no_op_logits.shape[0] != hidden.shape[0]:
            raise ContractError("logits 必须是 [N,V]")
        if self.target_ids.shape != (hidden.shape[0],) or len(self.task_ids) != hidden.shape[0]:
            raise ContractError("target/task 数量与 hidden 不一致")
        if any(not isinstance(task, str) or not task for task in self.task_ids):
            raise ContractError("task_id 必须是非空字符串")
        if len(set(self.task_ids)) < 2:
            raise ContractError("leave-one-task-out 至少需要 2 个任务")
        target_nll(self.no_op_logits, self.target_ids)
        target_nll(self.donor_logits, self.target_ids)


@dataclass(frozen=True)
class CalibrationResult:
    weights: np.ndarray
    bias: float
    threshold: float
    approved: bool
    report: dict[str, Any]


def _fit_ridge_dual(hidden: np.ndarray, advantage: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    x = np.asarray(hidden, dtype=np.float64)
    y = np.asarray(advantage, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] < 2:
        raise ContractError("ridge 输入形状异常")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ContractError("ridge 必须为正有限数")
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x @ centered_x.T
    dual = np.linalg.solve(gram + np.eye(x.shape[0]) * ridge, centered_y)
    weights = centered_x.T @ dual
    bias = y_mean - float(x_mean @ weights)
    return weights.astype(np.float32), float(bias)


def fit_counterfactual_gate(
    batch: CounterfactualBatch,
    *,
    frozen_contract_sha256: str | None,
    ridge: float = 1.0,
    threshold: float = 0.0,
    min_forced_advantage: float = 1.0e-6,
    min_loto_policy_advantage: float = 1.0e-6,
) -> CalibrationResult:
    """拟合 predicted NLL advantage，并严格用 leave-one-complete-task-out 晋级。"""

    batch.validate()
    if not all(
        math.isfinite(value)
        for value in (ridge, threshold, min_forced_advantage, min_loto_policy_advantage)
    ):
        raise ContractError("标定阈值必须有限")
    if min_forced_advantage < 0 or min_loto_policy_advantage < 0:
        raise ContractError("标定最小收益不得为负")

    hidden = np.asarray(batch.hidden, dtype=np.float64)
    no_op_nll = target_nll(batch.no_op_logits, batch.target_ids)
    donor_nll = target_nll(batch.donor_logits, batch.target_ids)
    advantage = no_op_nll - donor_nll
    task_array = np.asarray(batch.task_ids)
    tasks = sorted(set(batch.task_ids))
    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        mask = task_array == task
        task_advantage = float(np.mean(advantage[mask]))
        task_rows.append(
            {
                "task_id": task,
                "tokens": int(mask.sum()),
                "forced_donor_advantage": task_advantage,
                "forced_donor_improved": task_advantage > 0,
            }
        )

    loto_rows: list[dict[str, Any]] = []
    for held_out in tasks:
        train = task_array != held_out
        test = ~train
        weights, bias = _fit_ridge_dual(hidden[train], advantage[train], ridge)
        predicted = hidden[test] @ weights.astype(np.float64) + bias
        selected = predicted > threshold
        policy_nll = np.where(selected, donor_nll[test], no_op_nll[test])
        policy_advantage = float(np.mean(no_op_nll[test] - policy_nll))
        loto_rows.append(
            {
                "held_out_task": held_out,
                "tokens": int(test.sum()),
                "selected_tokens": int(selected.sum()),
                "policy_advantage": policy_advantage,
                "passed": bool(selected.any() and policy_advantage > min_loto_policy_advantage),
            }
        )

    weights, bias = _fit_ridge_dual(hidden, advantage, ridge)
    forced_advantage = float(np.mean(advantage))
    improved_tasks = sum(row["forced_donor_improved"] for row in task_rows)
    counterfactual_passed = bool(
        forced_advantage > min_forced_advantage and improved_tasks > len(tasks) / 2
    )
    loto_passed = bool(all(row["passed"] for row in loto_rows))
    frozen_contract = bool(
        isinstance(frozen_contract_sha256, str)
        and len(frozen_contract_sha256) == 64
        and all(character in "0123456789abcdef" for character in frozen_contract_sha256.lower())
    )
    approved = bool(counterfactual_passed and loto_passed and frozen_contract)
    report = {
        "format": REPORT_FORMAT,
        "samples": int(hidden.shape[0]),
        "input_width": int(hidden.shape[1]),
        "tasks": len(tasks),
        "forced_donor_advantage": forced_advantage,
        "improved_tasks": improved_tasks,
        "task_metrics": task_rows,
        "leave_one_task_out": loto_rows,
        "threshold": threshold,
        "ridge": ridge,
        "evidence": {
            "counterfactual_nll_passed": counterfactual_passed,
            "leave_one_task_out_passed": loto_passed,
            "frozen_contract": frozen_contract,
            "frozen_contract_sha256": frozen_contract_sha256,
        },
        "approved": approved,
        "claim_limit": "calibration evidence only; no K3 frontend or model-quality claim",
    }
    return CalibrationResult(weights, bias, threshold, approved, report)


def save_calibration(result: CalibrationResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = output_dir / "gate_weights.f32.npy"
    np.save(weight_path, np.asarray(result.weights, dtype=np.float32), allow_pickle=False)
    evidence = result.report["evidence"]
    manifest = {
        "format": GATE_FORMAT,
        "approved": result.approved,
        "input_width": int(result.weights.shape[0]),
        "bias": result.bias,
        "threshold": result.threshold,
        "weights": {
            "file": weight_path.name,
            "shape": [int(result.weights.shape[0])],
            "dtype": "float32",
            "sha256": _sha256(weight_path),
        },
        "evidence": evidence,
        "claim_limit": "gate may trigger only when approved=true; capability still requires generation gate",
    }
    manifest_path = output_dir / "gate.json"
    _write_json(manifest_path, manifest)
    _write_json(output_dir / "calibration_report.json", result.report)
    return manifest_path, weight_path


def _load_batch(path: Path) -> CounterfactualBatch:
    archive = np.load(path, allow_pickle=False)
    required = {"hidden", "no_op_logits", "donor_logits", "target_ids", "task_ids"}
    if missing := required - set(archive.files):
        raise ContractError(f"counterfactual NPZ 缺少: {sorted(missing)}")
    task_values = np.asarray(archive["task_ids"])
    if task_values.ndim != 1 or task_values.dtype.kind not in {"U", "S"}:
        raise ContractError("task_ids 必须是无 pickle 字符串数组")
    return CounterfactualBatch(
        hidden=np.asarray(archive["hidden"]),
        no_op_logits=np.asarray(archive["no_op_logits"]),
        donor_logits=np.asarray(archive["donor_logits"]),
        target_ids=np.asarray(archive["target_ids"]),
        task_ids=tuple(str(value) for value in task_values.tolist()),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="预先冻结的 counterfactual NPZ")
    parser.add_argument("--frozen-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.frozen_contract.is_file():
        raise SystemExit("冻结合同不存在")
    batch = _load_batch(args.input)
    result = fit_counterfactual_gate(
        batch,
        frozen_contract_sha256=_sha256(args.frozen_contract),
        ridge=args.ridge,
        threshold=args.threshold,
    )
    manifest, _ = save_calibration(result, args.output_dir)
    print(
        json.dumps(
            {
                "approved": result.approved,
                "manifest": str(manifest.resolve()),
                "forced_donor_advantage": result.report["forced_donor_advantage"],
                "loto_passed": result.report["evidence"]["leave_one_task_out_passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.approved else 2


if __name__ == "__main__":
    raise SystemExit(main())
