"""v47 多token草稿原型的共享数据、shortlist与block head实现。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_unique(output: list[int], seen: set[int], values: Iterable[int], limit: int) -> None:
    for raw in values:
        value = int(raw)
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
        if len(output) >= limit:
            return


@dataclass(frozen=True)
class ShortlistConfig:
    native_top_k: int
    recent_tokens: int
    train_frequent: int
    candidate_limit: int

    @classmethod
    def from_contract(cls, contract: dict[str, Any]) -> "ShortlistConfig":
        row = contract["shortlist"]
        return cls(
            native_top_k=int(row["native_top_k"]),
            recent_tokens=int(row["recent_tokens"]),
            train_frequent=int(row["train_frequent"]),
            candidate_limit=int(row["candidate_limit"]),
        )


def build_shortlist(
    native_top_ids: Iterable[int],
    recent_token_ids: Iterable[int],
    frequent_train_ids: Iterable[int],
    config: ShortlistConfig,
) -> list[int]:
    """严格按冻结优先级构造anchor级shortlist，不窥视未来token。"""
    output: list[int] = []
    seen: set[int] = set()
    append_unique(output, seen, list(native_top_ids)[: config.native_top_k], config.candidate_limit)
    recent = list(recent_token_ids)[-config.recent_tokens :]
    append_unique(output, seen, reversed(recent), config.candidate_limit)
    append_unique(output, seen, list(frequent_train_ids)[: config.train_frequent], config.candidate_limit)
    return output


def train_frequent_ids(rows: list[dict[str, Any]], limit: int) -> list[int]:
    """只统计train的v38 validator目标；oracle/validation/test均不得参与。"""
    counter: Counter[int] = Counter()
    for row in rows:
        if row["split"] != "train":
            continue
        counter.update(int(value) for value in row["validator_token_ids"][1:])
    return [token for token, _ in counter.most_common(limit)]


def _splitmix64(values: np.ndarray) -> np.ndarray:
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        values = (values + np.uint64(0x9E3779B97F4A7C15)) & mask
        values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
        values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
        return values ^ (values >> np.uint64(31))


def token_keys(token_ids: Iterable[int], rank: int) -> np.ndarray:
    """按token ID即时生成固定Rademacher key；不保存、也不投影完整词表。"""
    ids = np.asarray(list(token_ids), dtype=np.uint64)[:, None]
    dims = np.arange(rank, dtype=np.uint64)[None, :]
    mixed = _splitmix64(ids ^ (dims * np.uint64(0xD6E8FEB86659FD93)))
    signs = np.where((mixed >> np.uint64(63)) == 0, -1.0, 1.0).astype(np.float32)
    return signs / np.float32(math.sqrt(rank))


@dataclass
class DraftDataset:
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    hidden: np.ndarray
    native_top_ids: np.ndarray
    native_top_logits: np.ndarray
    root: Path

    @classmethod
    def load(cls, manifest_path: Path, verify_hashes: bool = True) -> "DraftDataset":
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
        rows_path = root / manifest["rows"]["path"]
        arrays_path = root / manifest["arrays"]["path"]
        if verify_hashes:
            for path, expected in (
                (rows_path, manifest["rows"]["sha256"]),
                (arrays_path, manifest["arrays"]["sha256"]),
            ):
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError(f"SHA-256不匹配: {path.name}: {actual} != {expected}")
        rows = read_jsonl(rows_path)
        with np.load(arrays_path, allow_pickle=False) as arrays:
            hidden = np.asarray(arrays["hidden"], dtype=np.float32)
            native_top_ids = np.asarray(arrays["native_top_ids"], dtype=np.int32)
            native_top_logits = np.asarray(arrays["native_top_logits"], dtype=np.float32)
        dataset = cls(manifest, rows, hidden, native_top_ids, native_top_logits, root)
        dataset.validate()
        return dataset

    def validate(self) -> None:
        errors: list[str] = []
        count = len(self.rows)
        synthetic = bool(self.manifest.get("synthetic_fixture", False))
        if self.manifest.get("format") != "colorlm-v47-parallel-draft-dataset-manifest-v1":
            errors.append("manifest format不匹配")
        if self.manifest.get("base_model") != "ColorLM-v38-Qwen36-Shared-Sequence-Policy":
            errors.append("base_model必须是v38")
        if int(self.manifest["rows"]["count"]) != count:
            errors.append("rows.count与JSONL不一致")
        if self.hidden.ndim != 2 or self.hidden.shape[0] != count:
            errors.append("hidden必须是[N,D]")
        if self.native_top_ids.shape != self.native_top_logits.shape or self.native_top_ids.shape[0] != count:
            errors.append("native_top_ids/logits必须同形且首维为N")
        if not synthetic and (self.hidden.ndim != 2 or self.hidden.shape[1] != 2048):
            errors.append("真实v38 hidden宽度必须为2048")
        if not synthetic and (self.native_top_ids.ndim != 2 or self.native_top_ids.shape[1] != 32):
            errors.append("真实v38 native top宽度必须为32")
        if self.native_top_ids.size and (int(self.native_top_ids.min()) < 0 or int(self.native_top_ids.max()) >= 248320):
            errors.append("native token ID超出v38词表")
        if not np.isfinite(self.hidden).all() or not np.isfinite(self.native_top_logits).all():
            errors.append("数组含NaN/Inf")
        records = [int(row.get("record", -1)) for row in self.rows]
        if records != list(range(count)):
            errors.append("record必须从0连续递增")
        allowed = {"train", "validation", "test"}
        required_fields = {
            "anchor_id", "trajectory_id", "group_id", "template_cluster_id", "split",
            "context_bucket", "recent_token_ids", "validator_token_ids", "validator_terminated",
            "validator_source", "oracle_token_ids", "oracle_terminated"
        }
        for index, row in enumerate(self.rows):
            missing = required_fields - set(row)
            if missing:
                errors.append(f"record={index}缺字段: {sorted(missing)}")
                continue
            if row["split"] not in allowed:
                errors.append(f"record={index} split无效")
            validator = [int(value) for value in row["validator_token_ids"]]
            if not 1 <= len(validator) <= 4:
                errors.append(f"record={index} validator_token_ids长度必须为1..4")
            elif int(self.native_top_ids[index, 0]) != validator[0]:
                errors.append(f"record={index}原生top-1与v38 verifier首token不一致")
            if row["validator_source"] != "v38-free-greedy-from-anchor":
                errors.append(f"record={index} validator_source不是v38自由滚动")
            oracle = [int(value) for value in row.get("oracle_token_ids", [])]
            if not 1 <= len(oracle) <= 4:
                errors.append(f"record={index} oracle_token_ids长度必须为1..4")
            if len(row.get("recent_token_ids", [])) > 96:
                errors.append(f"record={index} recent_token_ids超过96")
        capture = self.manifest.get("capture", {})
        if capture.get("mode") != "one-anchor-free-greedy-rollout" or capture.get("temperature") != 0:
            errors.append("capture必须是同anchor温度0自由滚动")
        if capture.get("validator") != "v38-native" or capture.get("first_token_native_logits") is not True:
            errors.append("capture validator/首token来源不符合v38契约")
        for field in ("group_id", "template_cluster_id"):
            owner: dict[str, str] = {}
            for row in self.rows:
                key, split = str(row.get(field)), str(row.get("split"))
                if key in owner and owner[key] != split:
                    errors.append(f"{field}跨split泄漏: {key}")
                owner[key] = split
        if errors:
            raise ValueError("数据契约失败:\n- " + "\n- ".join(errors))


@dataclass
class CascadedBlockHead:
    input_weight: np.ndarray
    cascade_weight: np.ndarray
    bias: np.ndarray
    rank: int

    @classmethod
    def initialize(cls, hidden_size: int, rank: int, future_positions: int, seed: int) -> "CascadedBlockHead":
        rng = np.random.default_rng(seed)
        scale_in = 1.0 / math.sqrt(hidden_size)
        scale_hidden = 1.0 / math.sqrt(rank)
        return cls(
            input_weight=rng.normal(0.0, scale_in, size=(hidden_size, rank)).astype(np.float32),
            cascade_weight=rng.normal(
                0.0, scale_hidden, size=(max(future_positions - 1, 0), rank, rank)
            ).astype(np.float32),
            bias=np.zeros((future_positions, rank), dtype=np.float32),
            rank=rank,
        )

    @property
    def future_positions(self) -> int:
        return int(self.bias.shape[0])

    def states(self, hidden: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float32)
        if hidden.ndim == 1:
            hidden = hidden[None, :]
        norms = np.linalg.norm(hidden, axis=1, keepdims=True)
        normalized = hidden / np.maximum(norms, np.float32(1e-12))
        output = np.empty((len(hidden), self.future_positions, self.rank), dtype=np.float32)
        output[:, 0, :] = np.tanh(normalized @ self.input_weight + self.bias[0])
        for position in range(1, self.future_positions):
            output[:, position, :] = np.tanh(
                output[:, position - 1, :] @ self.cascade_weight[position - 1] + self.bias[position]
            )
        return output

    def propose_future(self, hidden: np.ndarray, candidates: list[int]) -> tuple[list[int], list[list[float]]]:
        states = self.states(hidden)[0]
        keys = token_keys(candidates, self.rank)
        proposals: list[int] = []
        scores_out: list[list[float]] = []
        for state in states:
            scores = keys @ state
            best = int(np.argmax(scores))
            proposals.append(int(candidates[best]))
            scores_out.append([float(value) for value in scores])
        return proposals, scores_out

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            input_weight=self.input_weight,
            cascade_weight=self.cascade_weight,
            bias=self.bias,
            metadata=np.array(json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))),
        )

    @classmethod
    def load(cls, path: Path) -> tuple["CascadedBlockHead", dict[str, Any]]:
        with np.load(path, allow_pickle=False) as arrays:
            input_weight = np.asarray(arrays["input_weight"], dtype=np.float32)
            cascade_weight = np.asarray(arrays["cascade_weight"], dtype=np.float32)
            bias = np.asarray(arrays["bias"], dtype=np.float32)
            metadata = json.loads(str(arrays["metadata"].item()))
        head = cls(input_weight, cascade_weight, bias, int(input_weight.shape[1]))
        return head, metadata


def load_contract(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or HERE / "frozen_contract.json").read_text(encoding="utf-8"))
