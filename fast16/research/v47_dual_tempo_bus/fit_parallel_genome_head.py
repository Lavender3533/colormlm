"""训练 terminal hidden → 全字段并行 Design Genome 的小型多任务头。"""

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
DATA_FORMAT = "colorlm-v47-parallel-genome-record-v1"
ONTOLOGY_FORMAT = "colorlm-v47-parallel-genome-ontology-v1"


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
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError("CNOB payload 被截断")
            if kind != BASE_HIDDEN:
                continue
            if ne[1:] != (1, 1, 1) or ne[0] <= 0 or payload_bytes != ne[0] * 4:
                raise ValueError(f"hidden shape 错误: record={record}, ne={ne}")
            if int(record) in hidden:
                raise ValueError(f"record={record} 重复")
            array = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f"record={record} 包含 NaN/Inf")
            hidden[int(record)] = array
    if not hidden:
        raise ValueError("CNOB 中没有 terminal hidden")
    if len({len(value) for value in hidden.values()}) != 1:
        raise ValueError("hidden width 不一致")
    return hidden


def load_ontology(path: Path) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != ONTOLOGY_FORMAT:
        raise ValueError("ontology format 错误")
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise ValueError("ontology.fields 为空")
    names: list[str] = []
    values: list[list[str]] = []
    for index, field in enumerate(raw_fields):
        if not isinstance(field, dict):
            raise ValueError(f"ontology.fields[{index}] 必须是对象")
        name = field.get("name")
        choices = field.get("values")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"ontology field name 无效: {name!r}")
        if not isinstance(choices, list) or len(choices) < 2 or not all(isinstance(x, str) for x in choices):
            raise ValueError(f"ontology field={name} values 无效")
        if len(choices) != len(set(choices)):
            raise ValueError(f"ontology field={name} values 重复")
        names.append(name)
        values.append(choices)
    return names, values, payload


def verify_rows(
    rows: list[dict[str, Any]], names: list[str], values: list[list[str]], hidden: dict[int, np.ndarray]
) -> np.ndarray:
    if not rows:
        raise ValueError("dataset 为空")
    records: set[int] = set()
    isolation: dict[tuple[str, str], str] = {}
    encoded = np.empty((len(rows), len(names)), dtype=np.int64)
    lookups = [{value: index for index, value in enumerate(choices)} for choices in values]
    for row_index, row in enumerate(rows):
        if row.get("format") != DATA_FORMAT:
            raise ValueError(f"row={row_index} format 错误")
        split = row.get("split")
        if split not in {"train", "validation", "blind"}:
            raise ValueError(f"row={row_index} split 错误")
        record = row.get("capture_record")
        if isinstance(record, bool) or not isinstance(record, int) or record not in hidden:
            raise ValueError(f"row={row_index} capture_record 错误")
        if record in records:
            raise ValueError(f"capture_record={record} 重复")
        records.add(record)
        for key in ("group_id", "template_cluster_id"):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"row={row_index} 缺少 {key}")
            pair = (key, value)
            previous = isolation.setdefault(pair, split)
            if previous != split:
                raise ValueError(f"{key}={value!r} 跨 split 泄漏")
        labels = row.get("labels")
        if not isinstance(labels, dict) or set(labels) != set(names):
            raise ValueError(f"row={row_index} labels 字段必须精确匹配 ontology")
        for field_index, name in enumerate(names):
            label = labels[name]
            if label not in lookups[field_index]:
                raise ValueError(f"row={row_index} field={name} label={label!r} 不在 ontology")
            encoded[row_index, field_index] = lookups[field_index][label]
    if records != set(hidden):
        raise ValueError(f"dataset/CNOB record 不一致: rows={len(records)} hidden={len(hidden)}")
    return encoded


class ParallelGenomeHead(nn.Module):
    def __init__(self, input_width: int, latent_width: int, classes: list[int], dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(input_width, latent_width)
        self.norm = nn.LayerNorm(latent_width)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(nn.Linear(latent_width, count) for count in classes)

    def forward(self, raw: torch.Tensor) -> list[torch.Tensor]:
        raw = F.normalize(raw, p=2, dim=-1, eps=1e-8)
        latent = self.dropout(F.gelu(self.norm(self.projection(raw))))
        return [head(latent) for head in self.heads]


def multitask_loss(logits: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [F.cross_entropy(field_logits, target[:, index]) for index, field_logits in enumerate(logits)]
    ).mean()


@torch.no_grad()
def evaluate(
    model: ParallelGenomeHead,
    rows: list[dict[str, Any]],
    row_indices: list[int],
    hidden: dict[int, np.ndarray],
    target: np.ndarray,
    names: list[str],
    values: list[list[str]],
) -> dict[str, Any]:
    if not row_indices:
        return {"evaluated": False, "sample_count": 0}
    raw = torch.from_numpy(np.stack([hidden[int(rows[index]["capture_record"])] for index in row_indices]))
    prediction = torch.stack([torch.argmax(item, dim=-1) for item in model(raw)], dim=1).cpu().numpy()
    reference = target[row_indices]
    correct = prediction == reference
    exact = correct.all(axis=1)
    field_accuracy = {
        name: float(correct[:, index].mean()) for index, name in enumerate(names)
    }
    examples = []
    for local_index, row_index in enumerate(row_indices[:8]):
        examples.append(
            {
                "task_id": rows[row_index]["task_id"],
                "exact": bool(exact[local_index]),
                "reference": {
                    name: values[field][int(reference[local_index, field])]
                    for field, name in enumerate(names)
                },
                "predicted": {
                    name: values[field][int(prediction[local_index, field])]
                    for field, name in enumerate(names)
                },
            }
        )
    return {
        "evaluated": True,
        "sample_count": len(row_indices),
        "exact_genome_rate": float(exact.mean()),
        "field_accuracy": float(correct.mean()),
        "minimum_field_accuracy": min(field_accuracy.values()),
        "per_field_accuracy": field_accuracy,
        "examples": examples,
    }


def save_weights(
    path: Path, model: ParallelGenomeHead, names: list[str], values: list[list[str]], metadata: dict[str, Any]
) -> None:
    arrays = {
        key.replace(".", "__"): value.detach().cpu().numpy().astype("<f4")
        for key, value in model.state_dict().items()
    }
    arrays["metadata_utf8"] = np.frombuffer(
        json.dumps(
            {"format": "colorlm-v47-parallel-genome-head-weights-v1", "fields": dict(zip(names, values)), **metadata},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"),
        dtype=np.uint8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def selftest() -> None:
    torch.manual_seed(47)
    classes = [3, 4, 2]
    width = sum(classes)
    samples = 180
    target = torch.stack(
        [torch.arange(samples) % classes[0], (torch.arange(samples) // 3) % classes[1], (torch.arange(samples) // 12) % classes[2]],
        dim=1,
    )
    raw = torch.zeros(samples, width)
    offset = 0
    for field, count in enumerate(classes):
        raw[torch.arange(samples), offset + target[:, field]] = 4.0
        offset += count
    raw += torch.randn_like(raw) * 0.02
    model = ParallelGenomeHead(width, 32, classes, 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = multitask_loss(model(raw), target)
        loss.backward()
        optimizer.step()
    predicted = torch.stack([item.argmax(dim=-1) for item in model(raw)], dim=1)
    if float((predicted == target).float().mean()) < 0.99:
        raise AssertionError("并行 Genome Head 合成自检未收敛")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--ontology", type=Path, default=Path(__file__).with_name("genome_head_ontology.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--latent-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--wall-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--minimum-train-trajectories", type=int, default=128)
    parser.add_argument("--evaluate-blind", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print(json.dumps({"selftest": "passed"}, ensure_ascii=False))
        return 0
    required = {"dataset": args.dataset, "capture": args.capture, "output": args.output, "report": args.report}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("缺少参数: " + ", ".join(missing))
    assert args.dataset and args.capture and args.output and args.report
    if args.output.exists() or args.report.exists():
        raise FileExistsError("拒绝覆盖已有权重或报告")
    if args.latent_width <= 0 or args.epochs <= 0 or args.batch_size <= 0 or args.wall_seconds <= 0:
        raise ValueError("训练预算必须为正数")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout 必须位于 [0,1)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    names, values, ontology = load_ontology(args.ontology)
    hidden = load_hidden(args.capture)
    rows = read_jsonl(args.dataset)
    target = verify_rows(rows, names, values, hidden)
    split_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        split_indices[str(row["split"])].append(index)
    train_indices = split_indices["train"]
    validation_indices = split_indices["validation"]
    if len(train_indices) < args.minimum_train_trajectories:
        raise ValueError(
            f"train 只有 {len(train_indices)} 条，低于 {args.minimum_train_trajectories}；"
            "拒绝用少量自由样本制造记忆型成功"
        )
    if not validation_indices:
        raise ValueError("validation 为空")

    input_width = len(next(iter(hidden.values())))
    model = ParallelGenomeHead(input_width, args.latent_width, [len(item) for item in values], args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    started = time.perf_counter()
    history: list[float] = []
    completed_epochs = 0
    shuffled = np.asarray(train_indices, dtype=np.int64)
    for epoch in range(args.epochs):
        if time.perf_counter() - started >= args.wall_seconds:
            break
        np.random.shuffle(shuffled)
        losses: list[float] = []
        model.train()
        for offset in range(0, len(shuffled), args.batch_size):
            selected = shuffled[offset : offset + args.batch_size].tolist()
            raw = torch.from_numpy(np.stack([hidden[int(rows[index]["capture_record"])] for index in selected]))
            labels = torch.from_numpy(target[selected])
            optimizer.zero_grad(set_to_none=True)
            loss = multitask_loss(model(raw), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if time.perf_counter() - started >= args.wall_seconds:
                break
        history.append(float(np.mean(losses)))
        completed_epochs = epoch + 1

    model.eval()
    metrics: dict[str, Any] = {}
    for split in ("train", "validation", "blind"):
        indices = split_indices[split]
        if split == "blind" and indices and not args.evaluate_blind:
            metrics[split] = {"evaluated": False, "sample_count": len(indices), "reason": "blind_locked"}
        else:
            metrics[split] = evaluate(model, rows, indices, hidden, target, names, values)
    validation = metrics["validation"]
    gate_passed = bool(
        validation.get("evaluated")
        and validation["exact_genome_rate"] >= 0.75
        and validation["field_accuracy"] >= 0.90
        and validation["minimum_field_accuracy"] >= 0.50
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_weights(
        args.output,
        model,
        names,
        values,
        {"input_width": input_width, "latent_width": args.latent_width, "parameter_count": parameter_count},
    )
    report = {
        "format": "colorlm-v47-parallel-genome-head-fit-report-v1",
        "status": "cpu_prototype_only",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "capture": str(args.capture.resolve()),
        "capture_sha256": sha256_file(args.capture),
        "ontology": str(args.ontology.resolve()),
        "ontology_sha256": sha256_file(args.ontology),
        "weights": str(args.output.resolve()),
        "weights_sha256": sha256_file(args.output),
        "input_width": input_width,
        "latent_width": args.latent_width,
        "field_count": len(names),
        "parameter_count": parameter_count,
        "weight_mebibytes_f32": parameter_count * 4 / (1024 * 1024),
        "completed_epochs": completed_epochs,
        "wall_seconds": time.perf_counter() - started,
        "loss_first": history[0] if history else None,
        "loss_last": history[-1] if history else None,
        "metrics": metrics,
        "prototype_gate": {
            "validation_exact_genome_rate_min": 0.75,
            "validation_field_accuracy_min": 0.90,
            "validation_minimum_field_accuracy_min": 0.50,
            "passed": gate_passed,
        },
        "decision": "allow_compiler_ab" if gate_passed else "stop_or_expand_train_only_variants",
        "claim_limit": "Genome字段准确不等于网页质量；还必须编译并通过冻结前端门。",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
