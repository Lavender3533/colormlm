"""Deterministic, data-free weight surgery for CLM architectures."""

from __future__ import annotations

from typing import Any, Mapping

import torch


BYTE_TOKEN_COUNT = 258
BYTE_PAD_ID = 256
BYTE_MASK_ID = 257


def _extend_positions(source: torch.Tensor, target_length: int) -> torch.Tensor:
    source = source.detach().float().cpu()
    source_length = int(source.shape[0])
    if target_length <= source_length:
        return source[:target_length].clone()
    output = torch.empty((target_length, source.shape[1]), dtype=torch.float32)
    output[:source_length] = source
    trend = (source[-1] - source[0]) / max(source_length - 1, 1)
    for position in range(source_length, target_length):
        phase = position % source_length
        cycle = position // source_length
        output[position] = source[phase] + trend * (cycle * source_length) * 0.08
    return output


def _single_byte_rows(char2id: Mapping[str, int]) -> dict[int, int]:
    rows: dict[int, int] = {}
    for character, source_id in char2id.items():
        if character.startswith("<") and character.endswith(">"):
            continue
        encoded = character.encode("utf-8")
        if len(encoded) == 1:
            rows[encoded[0]] = int(source_id)
    return rows


def _spectral_expand(
    source: torch.Tensor,
    known_rows: Mapping[int, int],
    *,
    pad_source_id: int,
    mask_source_id: int,
) -> torch.Tensor:
    """Expand a learned row table with deterministic bit-coded spectral rows."""

    source = source.detach().float().cpu()
    known_source_ids = sorted(set(known_rows.values()))
    learned = source[known_source_ids]
    center = learned.mean(dim=0)
    centered = learned - center
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[: min(8, vh.shape[0])]
    projected = centered @ components.T
    scales = projected.std(dim=0).clamp_min(1e-4)

    output = torch.empty((BYTE_TOKEN_COUNT, source.shape[1]), dtype=torch.float32)
    for byte_id in range(256):
        signs = torch.tensor(
            [1.0 if byte_id & (1 << bit) else -1.0 for bit in range(components.shape[0])]
        )
        output[byte_id] = center + (signs * scales * 0.35) @ components
    for byte_id, source_id in known_rows.items():
        output[byte_id] = source[source_id]
    output[BYTE_PAD_ID] = source[pad_source_id]
    output[BYTE_MASK_ID] = source[mask_source_id]
    return output


def _expand_bias(
    source: torch.Tensor,
    known_rows: Mapping[int, int],
    *,
    pad_source_id: int,
    mask_source_id: int,
) -> torch.Tensor:
    source = source.detach().float().cpu()
    known_values = source[list(sorted(set(known_rows.values())))]
    center = known_values.mean()
    scale = known_values.std().clamp_min(1e-4)
    output = torch.empty(BYTE_TOKEN_COUNT, dtype=torch.float32)
    for byte_id in range(256):
        signed_popcount = (byte_id.bit_count() - 4) / 4.0
        output[byte_id] = center + scale * 0.15 * signed_popcount
    for byte_id, source_id in known_rows.items():
        output[byte_id] = source[source_id]
    output[BYTE_PAD_ID] = source[pad_source_id]
    output[BYTE_MASK_ID] = source[mask_source_id]
    return output


def transplant_utf8_byte(
    source_state: Mapping[str, torch.Tensor],
    source_config: Mapping[str, Any],
    source_char2id: Mapping[str, int],
    *,
    target_seq_len: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create a UTF-8 byte model without examples, loss, or optimization."""

    pad_source_id = int(source_char2id["<pad>"])
    mask_source_id = int(source_char2id["<mask>"])
    known_rows = _single_byte_rows(source_char2id)

    state = {name: tensor.detach().cpu() for name, tensor in source_state.items()}
    state["token_embed.weight"] = _spectral_expand(
        source_state["token_embed.weight"],
        known_rows,
        pad_source_id=pad_source_id,
        mask_source_id=mask_source_id,
    )
    state["token_pred_head.weight"] = _spectral_expand(
        source_state["token_pred_head.weight"],
        known_rows,
        pad_source_id=pad_source_id,
        mask_source_id=mask_source_id,
    )
    state["token_pred_head.bias"] = _expand_bias(
        source_state["token_pred_head.bias"],
        known_rows,
        pad_source_id=pad_source_id,
        mask_source_id=mask_source_id,
    )
    source_seq_len = int(source_state["pos_embed.weight"].shape[0])
    state["pos_embed.weight"] = _extend_positions(
        source_state["pos_embed.weight"], target_seq_len
    )

    config = dict(source_config)
    config.update(
        {
            "vocab_size": BYTE_TOKEN_COUNT - 1,
            "token_count": BYTE_TOKEN_COUNT,
            "pad_token_id": BYTE_PAD_ID,
            "mask_token_id": BYTE_MASK_ID,
            "max_seq_len": target_seq_len,
        }
    )
    tokenizer = {
        "type": "utf8_byte",
        "byte_tokens": 256,
        "pad_token_id": BYTE_PAD_ID,
        "mask_token_id": BYTE_MASK_ID,
    }
    stats = {
        "method": "spectral_byte_transplant",
        "known_byte_rows": len(known_rows),
        "synthesized_byte_rows": 256 - len(known_rows),
        "source_token_rows": int(source_state["token_embed.weight"].shape[0]),
        "target_token_rows": BYTE_TOKEN_COUNT,
        "source_seq_len": source_seq_len,
        "target_seq_len": target_seq_len,
        "position_method": "cyclic_residual_extension",
    }
    return state, config, tokenizer, stats
