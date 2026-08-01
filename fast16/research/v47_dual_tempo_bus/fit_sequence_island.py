"""训练 terminal hidden → 完整短序列的 v47 GRU 能力岛原型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
VERSION = 1
F32 = 0
BASE_HIDDEN = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_hidden(path: Path) -> dict[int, np.ndarray]:
    hidden: dict[int, np.ndarray] = {}
    with path.open("rb") as source:
        while True:
            encoded = source.read(HEADER.size)
            if not encoded:
                break
            if len(encoded) != HEADER.size:
                raise ValueError("CNOB 头被截断")
            magic, version, kind, record, dtype, reserved, *tail = HEADER.unpack(encoded)
            ne = tuple(int(value) for value in tail[:4])
            payload_bytes = int(tail[4])
            if magic != MAGIC or version != VERSION or dtype != F32 or reserved != 0:
                raise ValueError("CNOB 头/版本/dtype 不符合契约")
            if payload_bytes < 0:
                raise ValueError("CNOB payload 大小为负")
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError("CNOB payload 被截断")
            if kind != BASE_HIDDEN:
                continue
            if ne[1:] != (1, 1, 1) or ne[0] <= 0 or payload_bytes != ne[0] * 4:
                raise ValueError(f"hidden shape 错误: record={record}, ne={ne}, bytes={payload_bytes}")
            if record in hidden:
                raise ValueError(f"record={record} 有重复 hidden")
            array = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f"record={record} hidden 包含 NaN/Inf")
            hidden[int(record)] = array
    if not hidden:
        raise ValueError("CNOB 中没有 terminal hidden")
    widths = {len(value) for value in hidden.values()}
    if len(widths) != 1:
        raise ValueError(f"hidden width 不一致: {sorted(widths)}")
    return hidden


def verify_splits(rows: list[dict[str, Any]]) -> None:
    for key in ("group_id", "template_cluster_id"):
        owner: dict[str, str] = {}
        for row in rows:
            split = str(row["split"])
            value = str(row[key])
            previous = owner.setdefault(value, split)
            if previous != split:
                raise ValueError(f"{key}={value!r} 跨 {previous}/{split} 泄漏")


class SequenceIsland(nn.Module):
    def __init__(self, input_width: int, latent_width: int, output_classes: int) -> None:
        super().__init__()
        self.context = nn.Linear(input_width, latent_width)
        self.embedding = nn.Embedding(output_classes + 1, latent_width)  # 最后一行是 BOS
        self.cell = nn.GRUCell(latent_width * 2, latent_width)
        self.output = nn.Linear(latent_width, output_classes)

    @property
    def bos_index(self) -> int:
        return self.output.out_features

    def initial(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = F.normalize(raw, p=2, dim=-1, eps=1e-8)
        context = torch.tanh(self.context(normalized))
        return context, context

    def step(self, previous: torch.Tensor, context: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(previous)
        state = self.cell(torch.cat([embedded, context], dim=-1), state)
        return self.output(state), state


def encode_targets(
    rows: list[dict[str, Any]], token_to_local: dict[int, int], eos: int, unk: int
) -> list[list[int]]:
    result: list[list[int]] = []
    for row in rows:
        encoded = [token_to_local.get(int(token), unk) for token in row["target_token_ids"]]
        result.append(encoded + [eos])
    return result


def batch_loss(
    model: SequenceIsland,
    raw_hidden: torch.Tensor,
    sequences: list[list[int]],
) -> torch.Tensor:
    context, state = model.initial(raw_hidden)
    previous = torch.full((len(sequences),), model.bos_index, dtype=torch.long)
    loss_sum = torch.zeros((), dtype=torch.float32)
    count = 0
    maximum = max(len(sequence) for sequence in sequences)
    for position in range(maximum):
        logits, state = model.step(previous, context, state)
        active = [index for index, sequence in enumerate(sequences) if position < len(sequence)]
        if not active:
            break
        active_tensor = torch.tensor(active, dtype=torch.long)
        targets = torch.tensor([sequences[index][position] for index in active], dtype=torch.long)
        loss_sum = loss_sum + F.cross_entropy(logits[active_tensor], targets, reduction="sum")
        count += len(active)
        next_previous = previous.clone()
        next_previous[active_tensor] = targets
        previous = next_previous
    return loss_sum / max(count, 1)


@torch.no_grad()
def greedy(model: SequenceIsland, raw_hidden: torch.Tensor, eos: int, maximum: int) -> list[list[int]]:
    context, state = model.initial(raw_hidden)
    previous = torch.full((len(raw_hidden),), model.bos_index, dtype=torch.long)
    outputs: list[list[int]] = [[] for _ in range(len(raw_hidden))]
    finished = [False] * len(outputs)
    for _ in range(maximum):
        logits, state = model.step(previous, context, state)
        current = torch.argmax(logits, dim=-1)
        previous = current
        for index, value in enumerate(current.tolist()):
            if finished[index]:
                continue
            if value == eos:
                finished[index] = True
            else:
                outputs[index].append(int(value))
        if all(finished):
            break
    return outputs


def edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


@torch.no_grad()
def evaluate(
    model: SequenceIsland,
    rows: list[dict[str, Any]],
    hidden: dict[int, np.ndarray],
    encoded: list[list[int]],
    eos: int,
    unk: int,
    local_to_token: list[int],
) -> dict[str, Any]:
    if not rows:
        return {"evaluated": False, "sample_count": 0}
    raw = torch.from_numpy(np.stack([hidden[int(row["capture_record"])] for row in rows]))
    predicted = greedy(model, raw, eos, max(len(value) for value in encoded) + 8)
    references = [value[:-1] for value in encoded]
    exact = [predicted[index] == references[index] for index in range(len(rows))]
    total_tokens = sum(len(value) for value in references)
    aligned = sum(
        sum(a == b for a, b in zip(predicted[index], references[index]))
        for index in range(len(rows))
    )
    distances = [edit_distance(predicted[index], references[index]) for index in range(len(rows))]
    raw_token_count = sum(len(row["target_token_ids"]) for row in rows)
    oov_count = sum(local == unk for sequence in references for local in sequence)
    examples = []
    for index, row in enumerate(rows[:8]):
        examples.append(
            {
                "task_id": row["task_id"],
                "reference_token_ids": [
                    int(local_to_token[value]) if value < len(local_to_token) else -1 for value in references[index]
                ],
                "predicted_token_ids": [
                    int(local_to_token[value]) if value < len(local_to_token) else -1 for value in predicted[index]
                ],
                "exact": exact[index],
            }
        )
    return {
        "evaluated": True,
        "sample_count": len(rows),
        "exact_sequence_rate": float(np.mean(exact)),
        "aligned_token_accuracy": aligned / max(total_tokens, 1),
        "mean_normalized_edit_distance": float(
            np.mean([distance / max(len(reference), 1) for distance, reference in zip(distances, references)])
        ),
        "oov_token_rate": oov_count / max(raw_token_count, 1),
        "examples": examples,
    }


def save_npz(path: Path, model: SequenceIsland, local_to_token: list[int], eos: int, unk: int) -> None:
    state = model.state_dict()
    arrays = {key.replace(".", "__"): tensor.detach().cpu().numpy().astype("<f4") for key, tensor in state.items()}
    arrays.update(
        {
            "local_to_token": np.asarray(local_to_token, dtype="<i4"),
            "eos_local": np.asarray([eos], dtype="<i4"),
            "unk_local": np.asarray([unk], dtype="<i4"),
            "bos_local": np.asarray([model.bos_index], dtype="<i4"),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--latent-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--wall-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--evaluate-blind", action="store_true")
    args = parser.parse_args()

    if args.output.exists() or args.report.exists():
        raise FileExistsError("拒绝覆盖已有权重或报告")
    if args.latent_width <= 0 or args.epochs <= 0 or args.batch_size <= 0 or args.wall_seconds <= 0:
        raise ValueError("训练预算必须为正数")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    rows = read_jsonl(args.dataset)
    if not rows:
        raise ValueError("dataset 为空")
    verify_splits(rows)
    hidden = load_hidden(args.capture)
    expected_records = {int(row["capture_record"]) for row in rows}
    if expected_records != set(hidden):
        raise ValueError(f"dataset/CNOB record 不一致: dataset={len(expected_records)}, capture={len(hidden)}")
    widths = {len(hidden[record]) for record in hidden}
    input_width = widths.pop()

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
    if not by_split["train"] or not by_split["validation"]:
        raise ValueError("必须同时有 train 和 validation")

    train_tokens = sorted({int(token) for row in by_split["train"] for token in row["target_token_ids"]})
    token_to_local = {token: index for index, token in enumerate(train_tokens)}
    eos = len(train_tokens)
    unk = eos + 1
    local_to_token = train_tokens + [-1, -2]
    output_classes = len(local_to_token)
    model = SequenceIsland(input_width, args.latent_width, output_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    train_rows = by_split["train"]
    encoded_train = encode_targets(train_rows, token_to_local, eos, unk)
    started = time.perf_counter()
    history: list[float] = []
    completed_epochs = 0
    indices = np.arange(len(train_rows))
    for epoch in range(args.epochs):
        if time.perf_counter() - started >= args.wall_seconds:
            break
        np.random.shuffle(indices)
        losses = []
        model.train()
        for offset in range(0, len(indices), args.batch_size):
            selected = indices[offset : offset + args.batch_size].tolist()
            raw = torch.from_numpy(
                np.stack([hidden[int(train_rows[index]["capture_record"])] for index in selected])
            )
            sequences = [encoded_train[index] for index in selected]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, raw, sequences)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if time.perf_counter() - started >= args.wall_seconds:
                break
        history.append(float(np.mean(losses)))
        completed_epochs = epoch + 1

    model.eval()
    split_metrics: dict[str, Any] = {}
    for split in ("train", "validation", "blind"):
        split_rows = by_split[split]
        if split == "blind" and split_rows and not args.evaluate_blind:
            split_metrics[split] = {
                "evaluated": False,
                "sample_count": len(split_rows),
                "reason": "blind_locked_until_candidate_frozen",
            }
            continue
        encoded = encode_targets(split_rows, token_to_local, eos, unk)
        split_metrics[split] = evaluate(
            model, split_rows, hidden, encoded, eos, unk, local_to_token
        )

    validation = split_metrics["validation"]
    gate_passed = bool(
        validation.get("evaluated")
        and validation["oov_token_rate"] <= 0.02
        and validation["aligned_token_accuracy"] >= 0.70
        and validation["exact_sequence_rate"] >= 0.50
    )
    save_npz(args.output, model, local_to_token, eos, unk)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "format": "colorlm-v47-sequence-island-fit-report-v1",
        "status": "cpu_prototype_only",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "capture": str(args.capture.resolve()),
        "capture_sha256": sha256_file(args.capture),
        "weights": str(args.output.resolve()),
        "weights_sha256": sha256_file(args.output),
        "input_width": input_width,
        "latent_width": args.latent_width,
        "train_vocabulary_size": len(train_tokens),
        "parameter_count": int(parameter_count),
        "weight_mebibytes_f32": parameter_count * 4 / (1024 * 1024),
        "completed_epochs": completed_epochs,
        "wall_seconds": time.perf_counter() - started,
        "loss_first": history[0] if history else None,
        "loss_last": history[-1] if history else None,
        "metrics": split_metrics,
        "prototype_gate": {
            "validation_oov_token_rate_max": 0.02,
            "validation_aligned_token_accuracy_min": 0.70,
            "validation_exact_sequence_rate_min": 0.50,
            "passed": gate_passed,
        },
        "decision": (
            "allow_real_sequence_island_runtime_design"
            if gate_passed
            else "stop_or_add_pointer_copy_before_runtime"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

