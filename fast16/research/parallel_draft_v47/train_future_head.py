"""训练只预测未来第2..4个token的rank-64 cascaded block head。"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from draft_core import (
    CascadedBlockHead,
    DraftDataset,
    ShortlistConfig,
    build_shortlist,
    load_contract,
    read_jsonl,
    sha256_file,
    token_keys,
    train_frequent_ids,
)


@dataclass
class Example:
    record: int
    position: int
    keys: np.ndarray
    target_index: int
    weight: float


def load_replay_weights(path: Path | None, dataset: DraftDataset, replay_weight: float) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = {}
    if path is None:
        return weights
    for row in read_jsonl(path):
        if row.get("format") != "colorlm-v47-parallel-draft-error-replay-v1":
            raise ValueError("error replay format不匹配")
        if row.get("split") != "train":
            raise ValueError("严禁把validation/test error replay回灌训练")
        record = int(row["record"])
        position = int(row["rejection_position"]) - 2
        if not 0 <= record < len(dataset.rows) or not 0 <= position < 3:
            continue
        source = dataset.rows[record]
        if source["split"] != "train":
            raise ValueError("error replay引用了非train原始记录")
        validator = [int(value) for value in source["validator_token_ids"]]
        if position + 1 >= len(validator) or int(row["validator_token_id"]) != validator[position + 1]:
            raise ValueError("error replay的validator token与冻结数据不一致")
        weights[(record, position)] = weights.get((record, position), 1.0) + replay_weight
    return weights


def make_examples(
    dataset: DraftDataset,
    split: str,
    frequent: list[int],
    shortlist_config: ShortlistConfig,
    future_positions: int,
    rank: int,
    replay_weights: dict[tuple[int, int], float] | None = None,
) -> tuple[list[Example], dict[str, Any]]:
    examples: list[Example] = []
    attempted = 0
    covered = 0
    complete_anchors = 0
    anchors = 0
    replay_weights = replay_weights or {}
    for record, row in enumerate(dataset.rows):
        if row["split"] != split:
            continue
        anchors += 1
        candidates = build_shortlist(
            dataset.native_top_ids[record], row["recent_token_ids"], frequent, shortlist_config
        )
        index_by_token = {token: index for index, token in enumerate(candidates)}
        keys = token_keys(candidates, rank)
        validator = [int(value) for value in row["validator_token_ids"]]
        anchor_complete = True
        for position in range(min(future_positions, max(len(validator) - 1, 0))):
            attempted += 1
            target = validator[position + 1]
            if target not in index_by_token:
                anchor_complete = False
                continue
            covered += 1
            examples.append(
                Example(
                    record=record,
                    position=position,
                    keys=keys,
                    target_index=index_by_token[target],
                    weight=float(replay_weights.get((record, position), 1.0)),
                )
            )
        if len(validator) - 1 >= future_positions and anchor_complete:
            complete_anchors += 1
    return examples, {
        "split": split,
        "anchors": anchors,
        "attempted_future_tokens": attempted,
        "covered_future_tokens": covered,
        "candidate_coverage": covered / max(attempted, 1),
        "complete_four_token_anchors": complete_anchors,
    }


def loss_and_state_gradients(
    head: CascadedBlockHead,
    states: np.ndarray,
    examples: list[Example],
    loss_weights: list[float],
    with_gradients: bool,
) -> tuple[float, np.ndarray | None, float]:
    gradients = np.zeros_like(states) if with_gradients else None
    total_loss = 0.0
    denominator = 0.0
    for example in examples:
        state = states[example.record, example.position]
        scores = example.keys @ state
        shifted = scores - float(np.max(scores))
        probabilities = np.exp(shifted, dtype=np.float32)
        probabilities /= np.sum(probabilities)
        scale = float(loss_weights[example.position]) * example.weight
        total_loss -= scale * math.log(max(float(probabilities[example.target_index]), 1e-30))
        denominator += scale
        if gradients is not None:
            probabilities[example.target_index] -= 1.0
            gradients[example.record, example.position] += scale * (example.keys.T @ probabilities)
    return total_loss / max(denominator, 1e-12), gradients, denominator


def evaluate(
    head: CascadedBlockHead,
    dataset: DraftDataset,
    examples: list[Example],
    loss_weights: list[float],
) -> dict[str, Any]:
    states = head.states(dataset.hidden)
    loss, _, denominator = loss_and_state_gradients(head, states, examples, loss_weights, False)
    correct = 0
    by_position: dict[str, dict[str, int]] = {}
    for example in examples:
        scores = example.keys @ states[example.record, example.position]
        hit = int(np.argmax(scores)) == example.target_index
        correct += int(hit)
        key = str(example.position + 2)
        row = by_position.setdefault(key, {"samples": 0, "correct": 0})
        row["samples"] += 1
        row["correct"] += int(hit)
    for row in by_position.values():
        row["accuracy"] = row["correct"] / max(row["samples"], 1)
    return {
        "examples": len(examples),
        "weighted_denominator": denominator,
        "cross_entropy": loss,
        "accuracy": correct / max(len(examples), 1),
        "by_future_token_position": by_position,
    }


def train(
    dataset: DraftDataset,
    contract: dict[str, Any],
    epochs: int,
    learning_rate: float,
    seed: int,
    replay_path: Path | None = None,
    replay_weight: float = 2.0,
) -> tuple[CascadedBlockHead, dict[str, Any]]:
    rank = int(contract["future_head"]["rank"])
    future_positions = len(contract["future_head"]["positions"])
    loss_weights = [float(value) for value in contract["future_head"]["loss_weights"]]
    shortlist_config = ShortlistConfig.from_contract(contract)
    frequent = train_frequent_ids(dataset.rows, shortlist_config.train_frequent)
    replay_weights = load_replay_weights(replay_path, dataset, replay_weight)
    train_examples, train_coverage = make_examples(
        dataset, "train", frequent, shortlist_config, future_positions, rank, replay_weights
    )
    validation_examples, validation_coverage = make_examples(
        dataset, "validation", frequent, shortlist_config, future_positions, rank
    )
    if not train_examples:
        raise ValueError("train没有可训练的未来token样本")
    head = CascadedBlockHead.initialize(dataset.hidden.shape[1], rank, future_positions, seed)
    initial = evaluate(head, dataset, train_examples, loss_weights)

    # 每个epoch做一次全批量Adam；候选数量可变，但梯度严格只来自train。
    m_input = np.zeros_like(head.input_weight)
    v_input = np.zeros_like(head.input_weight)
    m_cascade = np.zeros_like(head.cascade_weight)
    v_cascade = np.zeros_like(head.cascade_weight)
    m_bias = np.zeros_like(head.bias)
    v_bias = np.zeros_like(head.bias)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    normalized = dataset.hidden / np.maximum(
        np.linalg.norm(dataset.hidden, axis=1, keepdims=True), np.float32(1e-12)
    )
    history = []
    for epoch in range(1, epochs + 1):
        states = head.states(dataset.hidden)
        loss, state_grad, denominator = loss_and_state_gradients(
            head, states, train_examples, loss_weights, True
        )
        assert state_grad is not None
        grad_cascade = np.zeros_like(head.cascade_weight)
        grad_bias = np.zeros_like(head.bias)
        for position in range(future_positions - 1, 0, -1):
            pre_grad = state_grad[:, position] * (1.0 - states[:, position] ** 2)
            grad_cascade[position - 1] = states[:, position - 1].T @ pre_grad / denominator
            grad_bias[position] = np.sum(pre_grad, axis=0) / denominator
            state_grad[:, position - 1] += pre_grad @ head.cascade_weight[position - 1].T
        pre_grad0 = state_grad[:, 0] * (1.0 - states[:, 0] ** 2)
        grad_input = normalized.T @ pre_grad0 / denominator
        grad_bias[0] = np.sum(pre_grad0, axis=0) / denominator

        for parameter, gradient, first, second in (
            (head.input_weight, grad_input, m_input, v_input),
            (head.cascade_weight, grad_cascade, m_cascade, v_cascade),
            (head.bias, grad_bias, m_bias, v_bias),
        ):
            first *= beta1
            first += (1.0 - beta1) * gradient
            second *= beta2
            second += (1.0 - beta2) * gradient * gradient
            first_hat = first / (1.0 - beta1**epoch)
            second_hat = second / (1.0 - beta2**epoch)
            parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        if epoch == 1 or epoch == epochs or epoch % max(epochs // 10, 1) == 0:
            history.append({"epoch": epoch, "train_cross_entropy_before_update": loss})

    return head, {
        "format": "colorlm-v47-parallel-draft-fit-report-v1",
        "scope": "offline-prototype-only",
        "first_token_trainable": False,
        "trained_positions": [2, 3, 4],
        "architecture": contract["future_head"]["architecture"],
        "rank": rank,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "error_replay": None if replay_path is None else str(replay_path.resolve()),
        "error_replay_examples": len(replay_weights),
        "frequent_train_token_ids": frequent,
        "coverage_before_fit": {"train": train_coverage, "validation": validation_coverage},
        "initial_train": initial,
        "final_train": evaluate(head, dataset, train_examples, loss_weights),
        "final_validation": evaluate(head, dataset, validation_examples, loss_weights),
        "history": history,
        "claim_limit": "teacher-forced分层CE仅训练未来token；必须另跑自由滚动v38接受模拟，不能由本报告宣称加速。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="数据manifest.json")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="输出head.npz")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--error-replay", type=Path)
    parser.add_argument("--error-replay-weight", type=float, default=2.0)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".report.json").exists():
        raise FileExistsError("拒绝覆盖已有训练产物")
    if args.epochs <= 0 or args.learning_rate <= 0 or args.error_replay_weight < 0:
        raise ValueError("训练参数必须为正数")
    started = time.perf_counter()
    contract = load_contract(args.contract)
    dataset = DraftDataset.load(args.dataset)
    head, report = train(
        dataset,
        contract,
        args.epochs,
        args.learning_rate,
        args.seed,
        args.error_replay,
        args.error_replay_weight,
    )
    metadata = {
        "format": "colorlm-v47-cascaded-block-head-v1",
        "dataset_manifest": str(args.dataset.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset),
        "contract_sha256": sha256_file(args.contract or Path(__file__).resolve().parent / "frozen_contract.json"),
        "first_token": "v38-native-logits-untrained",
        "future_positions": [2, 3, 4],
        "rank": int(contract["future_head"]["rank"]),
        "frequent_train_token_ids": report["frequent_train_token_ids"],
    }
    head.save(args.output, metadata)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["model"] = str(args.output.resolve())
    report["model_sha256"] = sha256_file(args.output)
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(args.output), "report": str(report_path), "final": report["final_validation"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
