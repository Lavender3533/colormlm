"""验证隐藏状态能否对当前上下文中的动态词元做可泛化残差修正。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
F32 = 0
BASE_LOGITS = 1
BASE_HIDDEN = 4
VOCAB = 248320
WIDTH = 2048


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="v30动态词汇策略头离线探针")
    parser.add_argument("--contract", type=Path, default=here / "probe_contract.json")
    parser.add_argument("--report", type=Path, default=here / "probe_report.json")
    parser.add_argument("--weights", type=Path, default=here / "probe_weights.npz")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_capture(path: Path, records: int) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.empty((records, WIDTH), dtype=np.float32)
    logits = np.empty((records, VOCAB), dtype=np.float32)
    seen: dict[int, set[int]] = defaultdict(set)
    with path.open("rb") as source:
        while True:
            encoded = source.read(HEADER.size)
            if not encoded:
                break
            if len(encoded) != HEADER.size:
                raise ValueError("CNOB头被截断")
            magic, version, kind, record, dtype, reserved, *tail = HEADER.unpack(encoded)
            ne = tuple(tail[:4])
            payload_bytes = int(tail[4])
            if magic != MAGIC or version != 1 or dtype != F32 or reserved != 0:
                raise ValueError("CNOB格式不符合v30探针契约")
            if record < 0 or record >= records or kind not in (BASE_LOGITS, BASE_HIDDEN):
                raise ValueError(f"CNOB记录非法: record={record}, kind={kind}")
            width = VOCAB if kind == BASE_LOGITS else WIDTH
            if ne != (width, 1, 1, 1) or payload_bytes != width * 4:
                raise ValueError(f"CNOB张量形状错误: record={record}, kind={kind}, ne={ne}")
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError("CNOB payload被截断")
            if kind in seen[record]:
                raise ValueError(f"CNOB记录重复: record={record}, kind={kind}")
            seen[record].add(kind)
            destination = logits[record] if kind == BASE_LOGITS else hidden[record]
            destination[:] = np.frombuffer(payload, dtype="<f4")
    if any(seen[index] != {BASE_LOGITS, BASE_HIDDEN} for index in range(records)):
        raise ValueError("CNOB记录不完整")
    if not np.isfinite(hidden).all() or not np.isfinite(logits).all():
        raise ValueError("CNOB包含非有限值")
    return hidden, logits


def recent_unique(tokens: list[int], maximum: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for token in reversed(tokens):
        value = int(token)
        if value < 0 or value >= VOCAB or value in seen:
            continue
        seen.add(value)
        selected.append(value)
        if len(selected) == maximum:
            break
    selected.reverse()
    return selected


def exact_nll(raw: np.ndarray, target_id: int) -> float:
    peak = float(np.max(raw))
    return peak + math.log(float(np.exp(raw.astype(np.float64) - peak).sum())) - float(raw[target_id])


def main() -> int:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    sources = contract["sources"]
    tasks_path = resolve_source(sources["tasks"])
    teacher_path = resolve_source(sources["teacher"])
    capture_path = resolve_source(sources["capture"])
    model_path = resolve_source(sources["model"])
    for key, path in (("tasks", tasks_path), ("teacher", teacher_path), ("capture", capture_path)):
        if sha256_file(path) != sources[f"{key}_sha256"]:
            raise ValueError(f"{key}的SHA-256与冻结契约不一致")

    tasks = read_jsonl(tasks_path)
    teacher = read_jsonl(teacher_path)
    task_by_id = {row["id"]: row for row in tasks}
    if len(task_by_id) != len(tasks) or any(row["task_id"] not in task_by_id for row in teacher):
        raise ValueError("任务ID重复或teacher引用未知任务")

    hidden, logits = load_capture(capture_path, len(teacher))
    hidden /= np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), np.float32(1e-8))

    pool_contract = contract["candidate_pool"]
    static_tasks: dict[int, set[str]] = defaultdict(set)
    for row in teacher:
        task = task_by_id[row["task_id"]]
        if task["split"] == pool_contract["static_source_split"]:
            static_tasks[int(row["target_token_id"])].add(row["task_id"])
    minimum = int(pool_contract["static_minimum_distinct_tasks"])
    static_ids = sorted(token for token, ids in static_tasks.items() if len(ids) >= minimum)
    maximum = int(pool_contract["maximum_tokens"])
    if len(static_ids) >= maximum:
        raise ValueError("静态协议token已占满动态候选池")

    candidate_ids: list[list[int]] = []
    target_positions: list[int] = []
    union_ids: set[int] = set(static_ids)
    for row in teacher:
        dynamic = recent_unique(row["prefix_tokens"], maximum - len(static_ids))
        combined = list(static_ids)
        present = set(combined)
        for token in dynamic:
            if token not in present:
                present.add(token)
                combined.append(token)
        combined = combined[:maximum]
        target_id = int(row["target_token_id"])
        target_positions.append(combined.index(target_id) if target_id in present else -1)
        candidate_ids.append(combined)
        union_ids.update(combined)

    union = np.asarray(sorted(union_ids), dtype=np.int64)
    union_lookup = {int(token): index for index, token in enumerate(union)}
    reader = GGUFReader(os.fspath(model_path), "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    embedding_tensor = tensors[sources["embedding_tensor"]]
    embedding = np.asarray(
        dequantize(embedding_tensor.data[union], embedding_tensor.tensor_type),
        dtype=np.float32,
    )
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), np.float32(1e-8))

    count = len(teacher)
    padded_ids = np.full((count, maximum), -1, dtype=np.int64)
    padded_union = np.zeros((count, maximum), dtype=np.int64)
    padded_raw = np.zeros((count, maximum), dtype=np.float32)
    padded_mask = np.zeros((count, maximum), dtype=np.bool_)
    base_scaled_partition = np.empty(count, dtype=np.float64)
    other_scaled_partition = np.empty(count, dtype=np.float64)
    peaks = np.empty(count, dtype=np.float32)
    native_nll = np.empty(count, dtype=np.float64)
    target_raw = np.empty(count, dtype=np.float32)
    for index, (row, ids) in enumerate(zip(teacher, candidate_ids, strict=True)):
        length = len(ids)
        ids_array = np.asarray(ids, dtype=np.int64)
        union_array = np.asarray([union_lookup[int(token)] for token in ids], dtype=np.int64)
        padded_ids[index, :length] = ids_array
        padded_union[index, :length] = union_array
        padded_mask[index, :length] = True
        raw = logits[index]
        peak = float(np.max(raw))
        scaled = np.exp(raw.astype(np.float64) - peak)
        candidate_scaled = scaled[ids_array]
        partition = float(np.sum(scaled))
        other = partition - float(np.sum(candidate_scaled))
        if other <= 0.0:
            raise ValueError("动态候选外分区非正")
        padded_raw[index, :length] = raw[ids_array]
        base_scaled_partition[index] = partition
        other_scaled_partition[index] = other
        peaks[index] = peak
        target_id = int(row["target_token_id"])
        target_raw[index] = raw[target_id]
        native_nll[index] = math.log(partition) + peak - float(raw[target_id])

    torch.manual_seed(int(contract["fit"]["seed"]))
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cpu")
    h = torch.from_numpy(hidden).to(device)
    e = torch.from_numpy(embedding).to(device)
    candidates = torch.from_numpy(padded_union).to(device)
    mask = torch.from_numpy(padded_mask).to(device)
    raw_rows = torch.from_numpy(padded_raw).to(device)
    other_z = torch.from_numpy(other_scaled_partition.astype(np.float32)).to(device)
    peak_tensor = torch.from_numpy(peaks).to(device)
    target_logits = torch.from_numpy(target_raw).to(device)
    target_position = torch.from_numpy(np.asarray(target_positions, dtype=np.int64)).to(device)
    split_names = [task_by_id[row["task_id"]]["split"] for row in teacher]
    train_mask = torch.tensor([value == contract["fit"]["split"] for value in split_names], device=device)
    present = target_position >= 0
    train_present = train_mask & present
    train_missing = train_mask & ~present

    rank = int(contract["model"]["rank"])
    left = torch.nn.Parameter(torch.empty((rank, WIDTH), device=device))
    right = torch.nn.Parameter(torch.empty((rank, WIDTH), device=device))
    torch.nn.init.normal_(left, mean=0.0, std=0.002)
    torch.nn.init.normal_(right, mean=0.0, std=0.002)
    optimizer = torch.optim.Adam(
        [left, right], lr=float(contract["fit"]["learning_rate"])
    )
    correction_min = float(contract["model"]["correction_min"])
    correction_max = float(contract["model"]["correction_max"])

    def corrections() -> torch.Tensor:
        hidden_rank = h @ left.T
        embedding_rank = e @ right.T
        selected = embedding_rank[candidates]
        value = torch.einsum("nr,nkr->nk", hidden_rank, selected)
        value = torch.clamp(value, correction_min, correction_max)
        return torch.where(mask, value, torch.zeros_like(value))

    initial_loss = None
    steps = int(contract["fit"]["steps"])
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        delta = corrections()
        corrected_scaled = torch.exp(raw_rows + delta - peak_tensor[:, None]) * mask
        log_z = torch.log(other_z + corrected_scaled.sum(dim=1)) + peak_tensor
        safe_position = torch.clamp(target_position, min=0)
        target_delta = delta.gather(1, safe_position[:, None]).squeeze(1)
        corrected_target = target_logits + torch.where(present, target_delta, 0.0)
        nll = log_z - corrected_target
        loss = nll[train_present].mean()
        loss = loss + float(contract["fit"]["l2_delta"]) * delta[train_mask].square().mean()
        if bool(train_missing.any()):
            loss = loss + float(contract["fit"]["missing_target_zero_penalty"]) * delta[train_missing].square().mean()
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        delta = corrections()
        corrected_scaled = torch.exp(raw_rows + delta - peak_tensor[:, None]) * mask
        log_z = torch.log(other_z + corrected_scaled.sum(dim=1)) + peak_tensor
        safe_position = torch.clamp(target_position, min=0)
        target_delta = delta.gather(1, safe_position[:, None]).squeeze(1)
        corrected_target = target_logits + torch.where(present, target_delta, 0.0)
        corrected_nll = (log_z - corrected_target).cpu().numpy().astype(np.float64)
        learned_delta = delta.cpu().numpy()

    evaluated: list[dict[str, Any]] = []
    per_task_values: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(teacher):
        task = task_by_id[row["task_id"]]
        nll_delta = float(corrected_nll[index] - native_nll[index])
        per_task_values[row["task_id"]].append(nll_delta)
        evaluated.append(
            {
                "sample_id": row["sample_id"],
                "task_id": row["task_id"],
                "split": task["split"],
                "family": task["family"],
                "token_index": int(row["token_index"]),
                "target_token_id": int(row["target_token_id"]),
                "target_in_dynamic_pool": target_positions[index] >= 0,
                "candidate_count": len(candidate_ids[index]),
                "native_target_nll": float(native_nll[index]),
                "corrected_target_nll": float(corrected_nll[index]),
                "target_nll_delta": nll_delta,
                "maximum_abs_correction": float(np.max(np.abs(learned_delta[index]))),
            }
        )

    def split_metrics(split: str) -> dict[str, Any]:
        selected = [row for row in evaluated if row["split"] == split]
        covered = [row for row in selected if row["target_in_dynamic_pool"]]
        return {
            "samples": len(selected),
            "target_coverage": len(covered) / len(selected),
            "mean_target_nll_delta": float(np.mean([row["target_nll_delta"] for row in selected])),
            "covered_target_win_rate": float(
                np.mean([row["target_nll_delta"] < 0.0 for row in covered])
            ) if covered else None,
        }

    metrics = {split: split_metrics(split) for split in ("train", "validation", "test")}
    per_task = [
        {
            "task_id": task_id,
            "split": task_by_id[task_id]["split"],
            "family": task_by_id[task_id]["family"],
            "mean_target_nll_delta": float(np.mean(values)),
        }
        for task_id, values in sorted(per_task_values.items())
    ]
    heldout_max = max(row["mean_target_nll_delta"] for row in per_task if row["split"] != "train")
    gate = contract["pass_gate"]
    passed = (
        metrics["validation"]["mean_target_nll_delta"] <= gate["validation_mean_target_nll_delta_max"]
        and metrics["test"]["mean_target_nll_delta"] <= gate["test_mean_target_nll_delta_max"]
        and metrics["validation"]["target_coverage"] >= gate["validation_target_coverage_min"]
        and metrics["test"]["target_coverage"] >= gate["test_target_coverage_min"]
        and heldout_max <= gate["maximum_heldout_task_mean_nll_regression"]
    )

    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.weights,
        left=left.detach().cpu().numpy().astype("<f4"),
        right=right.detach().cpu().numpy().astype("<f4"),
        static_token_ids=np.asarray(static_ids, dtype="<i4"),
    )
    report = {
        "format": "colorlm-v30-dynamic-lexical-probe-report-v1",
        "status": "passed_for_runtime_prototype" if passed else "stopped_offline",
        "retrospective_only": True,
        "contract": args.contract.as_posix(),
        "contract_sha256": sha256_file(args.contract),
        "candidate_pool": {
            "static_tokens": len(static_ids),
            "maximum_tokens_per_sample": maximum,
            "union_tokens": len(union),
            "mean_tokens_per_sample": float(np.mean([len(ids) for ids in candidate_ids])),
        },
        "fit": {
            **contract["fit"],
            "initial_loss": initial_loss,
            "final_loss": float(loss.detach()),
            "parameter_count": int(left.numel() + right.numel()),
        },
        "metrics": metrics,
        "maximum_heldout_task_mean_nll_regression": heldout_max,
        "per_task": per_task,
        "gate_passed": passed,
        "decision": (
            "允许实现v30动态候选运行时，但必须用全新冻结任务晋级"
            if passed
            else "停止动态词汇头；不得进入运行时"
        ),
        "weights": args.weights.as_posix(),
        "weights_sha256": sha256_file(args.weights),
        "claim_boundary": contract["claim_boundary"],
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
