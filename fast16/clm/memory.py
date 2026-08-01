"""Training-free associative memory compiled into a CLM file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


def encode_text(text: str, tokenizer: Mapping[str, Any]) -> list[int]:
    tokenizer_type = tokenizer["type"]
    if tokenizer_type == "utf8_byte":
        return list(text.encode("utf-8"))
    if tokenizer_type == "character":
        char2id = {str(k): int(v) for k, v in tokenizer["char2id"].items()}
        return [char2id[ch] for ch in text if ch in char2id and char2id[ch] > 1]
    raise ValueError(f"unsupported tokenizer: {tokenizer_type}")


def _embedding_key(token_ids: list[int], embedding: torch.Tensor) -> torch.Tensor:
    ids = torch.tensor(token_ids, dtype=torch.long)
    vectors = embedding[ids]
    position_weights = torch.linspace(0.75, 1.25, vectors.shape[0]).unsqueeze(1)
    key = (vectors * position_weights).sum(dim=0) / position_weights.sum()
    return F.normalize(key.float(), dim=0)


def _hash_key(token_ids: list[int], width: int = 128) -> torch.Tensor:
    features = torch.zeros(width, dtype=torch.float32)
    for ngram_size in (1, 2, 3):
        for start in range(max(0, len(token_ids) - ngram_size + 1)):
            value = 1469598103934665603
            for token_id in token_ids[start : start + ngram_size]:
                value ^= int(token_id) + 1
                value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            index = value % width
            sign = 1.0 if (value >> 63) == 0 else -1.0
            features[index] += sign / ngram_size
    if torch.count_nonzero(features) == 0:
        features[0] = 1.0
    return F.normalize(features, dim=0)


def make_key_vector(
    token_ids: list[int],
    embedding: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    neural = _embedding_key(token_ids, embedding)
    if mode == "embedding_weighted_v0":
        return neural
    if mode == "hybrid_hash_v1":
        hashed = _hash_key(token_ids, width=128)
        return F.normalize(torch.cat((neural * 0.5, hashed * 1.5)), dim=0)
    raise ValueError(f"unsupported memory key mode: {mode}")


def _load_records(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = str(record.get("key", ""))
                value = str(record.get("value", ""))
                if not key or not value:
                    raise ValueError(f"{path}:{line_no}: memory record needs key and value")
                records.append({"key": key, "value": value})
    return records


def compile_memory(
    paths: Iterable[str | Path],
    *,
    tokenizer: Mapping[str, Any],
    token_embedding: torch.Tensor,
    max_value_tokens: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int | str]]:
    records = _load_records(paths)
    key_mode = "hybrid_hash_v1"
    keys: list[torch.Tensor] = []
    values: list[list[int]] = []
    skipped = 0
    for record in records:
        key_ids = encode_text(record["key"], tokenizer)
        value_ids = encode_text(record["value"], tokenizer)[:max_value_tokens]
        if not key_ids or not value_ids:
            skipped += 1
            continue
        keys.append(make_key_vector(key_ids, token_embedding, mode=key_mode))
        values.append(value_ids)

    if not keys:
        width = int(token_embedding.shape[1])
        return {}, {
            "kind": "embedding_knn",
            "key_mode": key_mode,
            "count": 0,
            "skipped": skipped,
            "max_value_tokens": 0,
            "key_width": width + 128,
            "value_layout": "flat_u8_v1",
        }

    flat_values = [token_id for value in values for token_id in value]
    value_dtype = torch.uint8 if max(flat_values) <= 255 else torch.int16
    value_tensor = torch.tensor(flat_values, dtype=value_dtype)
    offsets = [0]
    for value in values:
        offsets.append(offsets[-1] + len(value))
    offset_tensor = torch.tensor(offsets, dtype=torch.int64)

    tensors = {
        "memory.keys": torch.stack(keys),
        "memory.values": value_tensor,
        "memory.offsets": offset_tensor,
    }
    metadata: dict[str, int | str] = {
        "kind": "embedding_knn",
        "key_mode": key_mode,
        "count": len(values),
        "skipped": skipped,
        "max_value_tokens": max(len(value) for value in values),
        "key_width": int(keys[0].numel()),
        "value_layout": "flat_u8_v1" if value_dtype == torch.uint8 else "flat_i16_v1",
        "value_tokens": len(flat_values),
    }
    return tensors, metadata


class AssociativeMemory:
    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        lengths: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        *,
        top_k: int,
        temperature: float,
        key_mode: str = "embedding_weighted_v0",
    ):
        self.keys = F.normalize(keys.float(), dim=1)
        self.values = values.long()
        self.lengths = lengths.long() if lengths is not None else None
        self.offsets = offsets.long() if offsets is not None else None
        self.top_k = max(1, top_k)
        self.temperature = max(temperature, 1e-4)
        self.key_mode = key_mode

    def value_tokens(self, index: int) -> torch.Tensor:
        if self.offsets is not None:
            start = int(self.offsets[index])
            end = int(self.offsets[index + 1])
            return self.values[start:end]
        if self.lengths is None:
            raise RuntimeError("memory has neither offsets nor lengths")
        length = int(self.lengths[index])
        return self.values[index, :length]

    def value_length(self, index: int) -> int:
        if self.offsets is not None:
            return int(self.offsets[index + 1] - self.offsets[index])
        if self.lengths is None:
            raise RuntimeError("memory has neither offsets nor lengths")
        return int(self.lengths[index])

    def query_vector(self, token_ids: list[int], token_embedding: torch.Tensor) -> torch.Tensor:
        return make_key_vector(token_ids, token_embedding, mode=self.key_mode)

    def retrieve(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = F.normalize(query.float(), dim=0)
        scores = self.keys @ query
        count = min(self.top_k, scores.numel())
        top_scores, indices = torch.topk(scores, count)
        weights = torch.softmax(top_scores / self.temperature, dim=0)
        return indices, weights

    def hidden_context(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        token_embedding: torch.Tensor,
    ) -> torch.Tensor:
        contexts = []
        for index in indices.tolist():
            token_ids = self.value_tokens(index)
            contexts.append(token_embedding[token_ids].float().mean(dim=0))
        return torch.stack(contexts).mul(weights.unsqueeze(1)).sum(dim=0)

    def logit_prior(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        *,
        positions: int,
        vocab_size: int,
    ) -> torch.Tensor:
        prior = torch.zeros(positions, vocab_size, dtype=torch.float32)
        for index, weight in zip(indices.tolist(), weights.tolist()):
            length = min(self.value_length(index), positions)
            tokens = self.value_tokens(index)[:length]
            rows = torch.arange(length)
            prior[rows, tokens] += float(weight)
        return prior
