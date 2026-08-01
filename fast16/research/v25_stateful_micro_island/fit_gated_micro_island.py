"""在严格请求留出上训练一个小型门控非线性微岛，墙钟默认不超过25秒。"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from fit_stateful_micro_island import group_sequences, split_sequences

import sys


HERE = Path(__file__).resolve().parent
V24_DIR = HERE.parent / "v24_speed_quality_bus"
if str(V24_DIR) not in sys.path:
    sys.path.insert(0, str(V24_DIR))

from fit_micro_island import metrics, read_pairs  # noqa: E402


class GatedMicroIsland(nn.Module):
    def __init__(self, input_rank: int, hidden_rank: int, target_rank: int) -> None:
        super().__init__()
        self.up = nn.Linear(input_rank, hidden_rank)
        self.gate = nn.Linear(input_rank, hidden_rank)
        self.memory_gate = nn.Linear(input_rank, hidden_rank)
        self.output = nn.Linear(hidden_rank, target_rank)
        self.decay_logit = nn.Parameter(torch.zeros(hidden_rank))
        nn.init.constant_(self.memory_gate.bias, -4.0)

    def forward(
        self,
        values: torch.Tensor,
        active: torch.Tensor,
        disable_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = values.shape
        state = values.new_zeros((batch, self.up.out_features))
        outputs: list[torch.Tensor] = []
        gate_sum = values.new_zeros(())
        gate_count = values.new_zeros(())
        decay = torch.sigmoid(self.decay_logit)
        for index in range(steps):
            row = values[:, index]
            is_active = active[:, index]
            value = torch.nn.functional.silu(self.gate(row)) * self.up(row)
            memory_gate = torch.sigmoid(self.memory_gate(row))
            fused = value if disable_memory else value + memory_gate * state
            outputs.append(self.output(fused))
            updated = decay * state + value
            state = torch.where(is_active[:, None], updated, state)
            gate_sum = gate_sum + (memory_gate * is_active[:, None]).sum()
            gate_count = gate_count + is_active.sum() * memory_gate.shape[1]
        return torch.stack(outputs, dim=1), gate_sum / torch.clamp(gate_count, min=1)


def flatten(sequences: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([row[0] for row in sequences]),
        np.concatenate([row[1] for row in sequences]),
    )


def pca_basis(values: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    _, _, vt = np.linalg.svd(values - mean, full_matrices=False)
    return mean.astype(np.float32), vt[: min(rank, vt.shape[0])].astype(np.float32)


def build_transform(
    train: list[tuple[np.ndarray, np.ndarray]], input_rank: int, target_rank: int
) -> dict[str, np.ndarray]:
    x_value, y_value = flatten(train)
    x_mean, x_basis = pca_basis(x_value, input_rank)
    y_mean, y_basis = pca_basis(y_value, target_rank)
    q_value = (x_value - x_mean) @ x_basis.T
    coefficient = (y_value - y_mean) @ y_basis.T
    q_scale = np.maximum(q_value.std(axis=0), 1e-4).astype(np.float32)
    y_scale = np.maximum(coefficient.std(axis=0), 1e-4).astype(np.float32)
    return {
        "x_mean": x_mean,
        "x_basis": x_basis,
        "q_scale": q_scale,
        "y_mean": y_mean,
        "y_basis": y_basis,
        "y_scale": y_scale,
    }


def tensorize(
    sequences: list[tuple[np.ndarray, np.ndarray]], transform: dict[str, np.ndarray]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    lengths = [len(row[0]) for row in sequences]
    max_length = max(lengths)
    input_rank = transform["x_basis"].shape[0]
    target_rank = transform["y_basis"].shape[0]
    q_batch = np.zeros((len(sequences), max_length, input_rank), dtype=np.float32)
    y_batch = np.zeros((len(sequences), max_length, target_rank), dtype=np.float32)
    mask = np.zeros((len(sequences), max_length), dtype=np.bool_)
    for index, (x_value, y_value) in enumerate(sequences):
        length = len(x_value)
        q_batch[index, :length] = (
            (x_value - transform["x_mean"]) @ transform["x_basis"].T
        ) / transform["q_scale"]
        y_batch[index, :length] = (
            (y_value - transform["y_mean"]) @ transform["y_basis"].T
        ) / transform["y_scale"]
        mask[index, :length] = True
    return torch.from_numpy(q_batch), torch.from_numpy(y_batch), torch.from_numpy(mask), lengths


def reconstruct(
    normalized: torch.Tensor,
    mask: torch.Tensor,
    transform: dict[str, np.ndarray],
) -> np.ndarray:
    coefficient = normalized.detach().cpu().numpy()[mask.cpu().numpy()]
    coefficient = coefficient * transform["y_scale"]
    return coefficient @ transform["y_basis"] + transform["y_mean"]


def evaluate(
    model: GatedMicroIsland,
    sequences: list[tuple[np.ndarray, np.ndarray]],
    transform: dict[str, np.ndarray],
    disable_memory: bool = False,
) -> dict[str, Any]:
    q_value, _, mask, lengths = tensorize(sequences, transform)
    model.eval()
    with torch.no_grad():
        prediction, memory_gate = model(q_value, mask, disable_memory=disable_memory)
    reconstructed = reconstruct(prediction, mask, transform)
    reference = np.concatenate([row[1] for row in sequences])
    rows: list[dict[str, float]] = []
    offset = 0
    for length in lengths:
        rows.append(metrics(reference[offset : offset + length], reconstructed[offset : offset + length]))
        offset += length
    return {
        "metrics": metrics(reference, reconstructed),
        "per_sequence": rows,
        "memory_gate_mean": float(memory_gate),
    }


def train_model(
    model: GatedMicroIsland,
    train: list[tuple[np.ndarray, np.ndarray]],
    validation: list[tuple[np.ndarray, np.ndarray]],
    transform: dict[str, np.ndarray],
    max_seconds: float,
    max_epochs: int,
) -> tuple[int, float, dict[str, Any]]:
    train_q, train_y, train_mask, _ = tensorize(train, transform)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = float("inf")
    best_validation = evaluate(model, validation, transform)
    started = time.monotonic()
    stale_checks = 0
    completed_epoch = 0
    for epoch in range(1, max_epochs + 1):
        if time.monotonic() - started >= max_seconds:
            break
        model.train()
        prediction, _ = model(train_q, train_mask)
        loss = torch.mean((prediction[train_mask] - train_y[train_mask]) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_epoch = epoch
        if epoch == 1 or epoch % 5 == 0:
            validation_row = evaluate(model, validation, transform)
            score = validation_row["metrics"]["relative_rmse"]
            if score < best_score - 1e-4:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_validation = validation_row
                stale_checks = 0
            else:
                stale_checks += 1
            if stale_checks >= 8:
                break
    elapsed = time.monotonic() - started
    model.load_state_dict(best_state)
    return completed_epoch, elapsed, {"epoch": best_epoch, **best_validation}


def main() -> int:
    parser = argparse.ArgumentParser(description="训练v25门控非线性微岛")
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-sequences", type=int, default=8)
    parser.add_argument("--drop-leading-sequences", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=25.0)
    parser.add_argument("--max-epochs", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 30:
        raise RuntimeError("小训练墙钟必须在[1,30]秒内")

    torch.manual_seed(3407)
    torch.set_num_threads(min(6, torch.get_num_threads()))
    all_sequences = group_sequences(read_pairs(args.dump))
    dropped = all_sequences[: args.drop_leading_sequences]
    sequences = all_sequences[args.drop_leading_sequences :]
    if len(sequences) != args.expected_sequences:
        raise RuntimeError(
            f"请求数不匹配: actual={len(sequences)}, expected={args.expected_sequences}"
        )
    train, validation, test = split_sequences(sequences)
    transform = build_transform(train, input_rank=64, target_rank=128)
    model = GatedMicroIsland(input_rank=64, hidden_rank=128, target_rank=128)
    epochs, elapsed, validation_result = train_model(
        model, train, validation, transform, args.max_seconds, args.max_epochs
    )
    test_result = evaluate(model, test, transform)
    memory_off = evaluate(model, test, transform, disable_memory=True)

    test_metrics = test_result["metrics"]
    per_sequence_passes = sum(
        row["relative_rmse"] < 1.0 and row["cosine_median"] > 0.0
        for row in test_result["per_sequence"]
    )
    previous_linear_relative_rmse = 0.942562460899353
    passed = bool(
        test_metrics["relative_rmse"] <= 0.85
        and test_metrics["cosine_median"] >= 0.50
        and previous_linear_relative_rmse - test_metrics["relative_rmse"] >= 0.05
        and per_sequence_passes == len(test)
    )

    arrays: dict[str, np.ndarray] = {
        **{key: value.astype(np.float16) for key, value in transform.items()},
    }
    for key, value in model.state_dict().items():
        arrays["model." + key] = value.detach().cpu().numpy().astype(np.float16)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.artifact, **arrays)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    transform_count = sum(value.size for value in transform.values())
    report = {
        "format": "colorlm-v25-gated-micro-island-fit-v1",
        "dump": str(args.dump.resolve()),
        "dropped_leading_sequence_tokens": [len(row[0]) for row in dropped],
        "split": {
            "train_sequences": len(train),
            "validation_sequences": len(validation),
            "test_sequences": len(test),
            "train_tokens": sum(len(row[0]) for row in train),
            "validation_tokens": sum(len(row[0]) for row in validation),
            "test_tokens": sum(len(row[0]) for row in test),
        },
        "architecture": {
            "input_pca_rank": 64,
            "gated_hidden_rank": 128,
            "target_pca_rank": 128,
            "state": "z_t=sigmoid(decay)*z_(t-1)+SiLU(gate(q_t))*up(q_t)",
            "output": "target_basis @ output(value_t + memory_gate(q_t)*z_(t-1))",
            "trainable_parameters": parameter_count,
            "total_f16_bytes_including_fixed_transforms": 2
            * (parameter_count + transform_count),
        },
        "training": {
            "seed": 3407,
            "max_seconds": args.max_seconds,
            "elapsed_seconds": elapsed,
            "epochs_completed": epochs,
            "best_validation": validation_result,
        },
        "test": {
            "gated_state": test_result,
            "same_model_memory_disabled": memory_off,
            "previous_linear_relative_rmse": previous_linear_relative_rmse,
        },
        "runtime_integration_gate": {
            "passed": passed,
            "requirements": {
                "test_relative_rmse_max": 0.85,
                "test_cosine_median_min": 0.50,
                "gain_over_previous_linear_relative_rmse_min": 0.05,
                "all_test_sequences_beat_zero": True,
            },
            "next_action": (
                "允许实现默认关闭的C++门控微岛候选"
                if passed
                else "不接运行图；保留教师dump并转向分专家/分站蒸馏"
            ),
        },
        "artifact": str(args.artifact.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"epochs={epochs} elapsed={elapsed:.2f}s "
        f"test_cos={test_metrics['cosine_median']:.4f} "
        f"test_rel_rmse={test_metrics['relative_rmse']:.4f}",
        flush=True,
    )
    print(json.dumps(report["runtime_integration_gate"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
