"""按原始token bytes审计Fara与ColorLM词表身份，不读取大权重payload。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402


def raw_token_bytes(reader: GGUFReader) -> list[bytes]:
    field = reader.fields.get("tokenizer.ggml.tokens")
    if field is None or len(field.types) < 2:
        raise RuntimeError("GGUF缺少tokenizer.ggml.tokens")
    return [bytes(memoryview(field.parts[index]).cast("B")) for index in field.data]


def token_types(reader: GGUFReader, expected: int) -> list[int]:
    field = reader.fields.get("tokenizer.ggml.token_type")
    if field is None:
        raise RuntimeError("GGUF缺少tokenizer.ggml.token_type")
    values = [int(value) for value in field.contents()]
    if len(values) != expected:
        raise RuntimeError("token_type长度与词表不一致")
    return values


def scalar_int(reader: GGUFReader, key: str) -> int | None:
    field = reader.fields.get(key)
    return None if field is None else int(field.contents())


def display(value: bytes) -> str:
    return value.decode("utf-8", errors="backslashreplace")


def main() -> int:
    parser = argparse.ArgumentParser(description="审计Fara/ColorLM token id是否逐行同义")
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "fast16/models/ColorLM-v6-Q3Router-Fused-A1.gguf",
    )
    parser.add_argument(
        "--donor",
        type=Path,
        default=ROOT / "fast16/models/donor/fara15_27b/Fara1.5-27B-Q5_K_M.gguf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("tokenizer_identity_report.json"),
    )
    args = parser.parse_args()
    if not args.base.is_file() or not args.donor.is_file():
        raise RuntimeError(f"模型不存在: base={args.base}, donor={args.donor}")

    base = GGUFReader(os.fspath(args.base), "r")
    donor = GGUFReader(os.fspath(args.donor), "r")
    base_tokens = raw_token_bytes(base)
    donor_tokens = raw_token_bytes(donor)
    base_types = token_types(base, len(base_tokens))
    donor_types = token_types(donor, len(donor_tokens))
    paired = min(len(base_tokens), len(donor_tokens))
    identical_bytes = sum(base_tokens[i] == donor_tokens[i] for i in range(paired))
    identical_types = sum(base_types[i] == donor_types[i] for i in range(paired))
    first_mismatches: list[dict[str, Any]] = []
    for token_id in range(paired):
        if base_tokens[token_id] != donor_tokens[token_id] or base_types[token_id] != donor_types[token_id]:
            first_mismatches.append(
                {
                    "id": token_id,
                    "base": display(base_tokens[token_id]),
                    "donor": display(donor_tokens[token_id]),
                    "base_type": base_types[token_id],
                    "donor_type": donor_types[token_id],
                }
            )
            if len(first_mismatches) == 20:
                break

    base_ids: dict[bytes, list[int]] = defaultdict(list)
    for token_id, token in enumerate(base_tokens):
        base_ids[token].append(token_id)
    mapping = [
        base_ids[token][0] if len(base_ids.get(token, [])) == 1 else -1
        for token in donor_tokens
    ]
    mapped = [value for value in mapping if value >= 0]
    collisions = sum(value > 1 for value in Counter(mapped).values())
    metadata_keys = (
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
    )
    metadata = {
        key: {"base": scalar_int(base, key), "donor": scalar_int(donor, key)}
        for key in metadata_keys
    }
    exact_identity = (
        len(base_tokens) == len(donor_tokens)
        and identical_bytes == len(base_tokens)
        and identical_types == len(base_tokens)
        and all(row["base"] == row["donor"] for row in metadata.values())
    )
    report = {
        "format": "colorlm-fara-tokenizer-identity-v1",
        "method": "tokenizer.ggml.tokens raw bytes + token_type + control ids",
        "base": {"path": str(args.base.resolve()), "vocab_size": len(base_tokens)},
        "donor": {"path": str(args.donor.resolve()), "vocab_size": len(donor_tokens)},
        "paired_ids": paired,
        "identical_token_bytes_by_id": identical_bytes,
        "identical_token_types_by_id": identical_types,
        "unique_bytes_mapped": len(mapped),
        "mapping_collisions": collisions,
        "metadata": metadata,
        "first_mismatches": first_mismatches,
        "exact_identity": exact_identity,
        "decision": "direct_logit_rows_allowed" if exact_identity else "explicit_map_required",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
