"""Independent CLM v0 runtime with recurrence and associative memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .format import ClmReader
from .memory import AssociativeMemory, encode_text


class BidirectionalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = x.shape
        query = self.q_proj(x).view(batch, sequence, self.n_heads, self.head_size).transpose(1, 2)
        key = self.k_proj(x).view(batch, sequence, self.n_heads, self.head_size).transpose(1, 2)
        value = self.v_proj(x).view(batch, sequence, self.n_heads, self.head_size).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / self.head_size**0.5
        attention = torch.softmax(scores, dim=-1)
        output = (attention @ value).transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.out_proj(output)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = BidirectionalAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Identity(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class TemperatureHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    steps: int
    memory_records: int


class ZeroTrainModel(nn.Module):
    """CLM v0 neural core.

    The model runs lower layers once, then reuses upper layers with a damped
    residual recurrence. Associative memory contributes both a hidden-state
    context and a continuous logit prior.
    """

    def __init__(self, config: dict[str, Any], runtime: dict[str, Any]):
        super().__init__()
        self.config = config
        self.runtime = runtime
        self.vocab_size = int(config["vocab_size"])
        self.token_count = int(config.get("token_count", self.vocab_size + 1))
        self.max_seq_len = int(config["max_seq_len"])
        self.pad_token_id = int(config.get("pad_token_id", 0))
        self.mask_token_id = int(config.get("mask_token_id", 1))
        d_model = int(config["d_model"])

        self.token_embed = nn.Embedding(self.token_count, d_model)
        self.pos_embed = nn.Embedding(self.max_seq_len, d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, int(config["n_heads"]), int(config["d_ff"]))
            for _ in range(int(config["n_layers"]))
        )
        self.temperature_head = TemperatureHead(d_model)
        self.final_norm = RMSNorm(d_model)
        self.token_pred_head = nn.Linear(d_model, self.token_count)

        self.tokenizer: dict[str, Any] = {}
        self.char2id: dict[str, int] = {}
        self.id2char: dict[int, str] = {}
        self.memory: AssociativeMemory | None = None

    @classmethod
    def from_clm(cls, path: str | Path) -> "ZeroTrainModel":
        with ClmReader(path) as reader:
            metadata = reader.metadata
            if metadata.get("architecture") not in {
                "colormlm_zerotrain_v0",
                "colormlm_zerotrain_v1",
                "colormlm_zerotrain_v2",
            }:
                raise ValueError(f"unsupported CLM architecture: {metadata.get('architecture')}")
            model = cls(dict(metadata["config"]), dict(metadata["runtime"]))

            expected = model.state_dict()
            loaded: dict[str, torch.Tensor] = {}
            missing: list[str] = []
            for name, target in expected.items():
                if not reader.has_tensor(name):
                    missing.append(name)
                    continue
                loaded[name] = reader.tensor(name, copy=True).to(target.dtype)
            if missing:
                raise ValueError(f"CLM is missing runtime tensors: {missing}")
            model.load_state_dict(loaded, strict=True)

            tokenizer = metadata["tokenizer"]
            model.tokenizer = dict(tokenizer)
            if tokenizer["type"] == "character":
                model.char2id = {str(k): int(v) for k, v in tokenizer["char2id"].items()}
                model.id2char = {int(k): str(v) for k, v in tokenizer["id2char"].items()}

            if reader.has_tensor("memory.keys"):
                memory_meta = metadata.get("memory", {})
                model.memory = AssociativeMemory(
                    reader.tensor("memory.keys", copy=True),
                    reader.tensor("memory.values", copy=True),
                    reader.tensor("memory.lengths", copy=True)
                    if reader.has_tensor("memory.lengths")
                    else None,
                    reader.tensor("memory.offsets", copy=True)
                    if reader.has_tensor("memory.offsets")
                    else None,
                    top_k=int(model.runtime["memory_top_k"]),
                    temperature=float(model.runtime["memory_temperature"]),
                    key_mode=str(memory_meta.get("key_mode", "embedding_weighted_v0")),
                )
        model.eval()
        return model

    def encode(self, text: str) -> list[int]:
        if self.tokenizer["type"] == "character":
            return [self.char2id.get(character, self.pad_token_id) for character in text]
        return encode_text(text, self.tokenizer)

    def decode(self, token_ids: list[int]) -> str:
        if self.tokenizer["type"] == "utf8_byte":
            raw = bytes(
                token_id
                for token_id in token_ids
                if 0 <= token_id <= 255
            )
            return raw.decode("utf-8", errors="replace")
        return "".join(
            self.id2char.get(token_id, "")
            for token_id in token_ids
            if token_id not in (self.pad_token_id, self.mask_token_id)
        )

    def _query_vector(self, prompt_ids: list[int]) -> torch.Tensor:
        if self.memory is not None:
            return self.memory.query_vector(prompt_ids, self.token_embed.weight)
        specials = {self.pad_token_id, self.mask_token_id}
        valid = [token_id for token_id in prompt_ids if token_id not in specials]
        if not valid:
            valid = [self.pad_token_id]
        ids = torch.tensor(valid, dtype=torch.long, device=self.token_embed.weight.device)
        vectors = self.token_embed(ids).float()
        weights = torch.linspace(0.75, 1.25, vectors.shape[0], device=vectors.device).unsqueeze(1)
        query = (vectors * weights).sum(dim=0) / weights.sum()
        return F.normalize(query, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        prompt_length: int,
        memory_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = input_ids.shape[1]
        memory_mask = torch.zeros((1, sequence, 1), dtype=torch.float32, device=input_ids.device)
        if memory_context is not None and prompt_length < sequence:
            memory_mask[:, prompt_length:, :] = 1.0
        context = memory_context
        if context is None:
            context = torch.zeros(
                self.token_embed.embedding_dim,
                dtype=torch.float32,
                device=input_ids.device,
            )
        return self.forward_with_memory_mask(input_ids, context, memory_mask)

    def forward_with_memory_mask(
        self,
        input_ids: torch.Tensor,
        memory_context: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, sequence = input_ids.shape
        positions = torch.arange(sequence, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        recursive_start = int(self.runtime["recursive_start_layer"])
        for layer in self.layers[:recursive_start]:
            x = layer(x)

        scale = float(self.runtime["memory_hidden_scale"])
        x = x + memory_mask.to(x.dtype) * memory_context.to(x.dtype).view(1, 1, -1) * scale

        for layer in self.layers[recursive_start:]:
            x = layer(x)
        first_pass = x

        recurrence_steps = max(1, int(self.runtime["recurrence_steps"]))
        alpha = float(self.runtime["recurrence_alpha"])
        recurrent = first_pass
        for _ in range(1, recurrence_steps):
            proposal = recurrent
            for layer in self.layers[recursive_start:]:
                proposal = layer(proposal)
            recurrent = torch.lerp(first_pass, proposal, alpha)

        hidden = self.final_norm(recurrent)
        return self.token_pred_head(hidden), self.temperature_head(hidden)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        new_tokens: int | None = None,
        refinement_steps: int = 8,
        core_backend: Any = None,
    ) -> GenerationResult:
        prompt_ids = self.encode(prompt)
        memory_context = None
        memory_prior = None
        memory_records = 0
        retrieved = None
        if self.memory is not None:
            query = self._query_vector(prompt_ids)
            indices, weights = self.memory.retrieve(query)
            retrieved = (indices, weights)
            memory_context = self.memory.hidden_context(indices, weights, self.token_embed.weight)
            if new_tokens is None:
                new_tokens = self.memory.value_length(int(indices[0]))
            memory_records = indices.numel()

        if new_tokens is None:
            new_tokens = min(40, self.max_seq_len - len(prompt_ids))
        total_length = len(prompt_ids) + new_tokens
        if total_length > self.max_seq_len:
            raise ValueError(f"requested sequence {total_length} exceeds max_seq_len={self.max_seq_len}")

        input_ids = torch.full((1, total_length), self.mask_token_id, dtype=torch.long)
        if prompt_ids:
            input_ids[0, : len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long)
        determined = torch.zeros(total_length, dtype=torch.bool)
        determined[: len(prompt_ids)] = True

        if self.memory is not None and retrieved is not None:
            indices, weights = retrieved
            memory_prior = self.memory.logit_prior(
                indices,
                weights,
                positions=new_tokens,
                vocab_size=self.token_count,
            )

        steps_used = 0
        for step in range(max(1, refinement_steps)):
            if core_backend is None:
                logits, temperature = self.forward(
                    input_ids,
                    prompt_length=len(prompt_ids),
                    memory_context=memory_context,
                )
            else:
                logits, temperature = core_backend.forward(
                    input_ids,
                    prompt_length=len(prompt_ids),
                    memory_context=memory_context,
                )
            if memory_prior is not None:
                logits = logits.clone()
                logits[:, len(prompt_ids):, :] += (
                    memory_prior.unsqueeze(0) * float(self.runtime["memory_logit_scale"])
                )

            unknown = ~determined
            remaining = int(unknown.sum())
            if remaining == 0:
                break
            scores = temperature[0, :, 0].clone()
            scores[determined] = -1
            ratio = min(1.0, 0.30 + 0.10 * step)
            count = min(remaining, max(1, int(remaining * ratio)))
            selected = torch.topk(scores, count).indices
            predicted = logits[0, selected].argmax(dim=-1)
            input_ids[0, selected] = predicted
            determined[selected] = True
            steps_used = step + 1

        if (~determined).any():
            if core_backend is None:
                logits, _ = self.forward(
                    input_ids,
                    prompt_length=len(prompt_ids),
                    memory_context=memory_context,
                )
            else:
                logits, _ = core_backend.forward(
                    input_ids,
                    prompt_length=len(prompt_ids),
                    memory_context=memory_context,
                )
            unknown_indices = torch.where(~determined)[0]
            input_ids[0, unknown_indices] = logits[0, unknown_indices].argmax(dim=-1)

        tokens = input_ids[0].tolist()
        return GenerationResult(
            text=self.decode(tokens),
            token_ids=tokens,
            steps=steps_used,
            memory_records=memory_records,
        )
