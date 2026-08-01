"""Build an exact donor-token to base-token map from two GGUF vocabularies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


BASE_DEFAULT = ROOT / "fast16" / "models" / "ColorLM-v6-Q3Router-Fused-A1.gguf"
DONOR_DEFAULT = (
    ROOT
    / "fast16"
    / "models"
    / "donor"
    / "qwen3-coder-next-iq3s"
    / "Qwen3-Coder-Next-UD-IQ3_S.gguf"
)
DONOR_CONFIG_DEFAULT = (
    ROOT
    / "fast16"
    / "research"
    / "biopsy_cache"
    / "Qwen_Qwen3-Coder-Next"
    / "master"
    / "metadata"
    / "config.json"
)
DONOR_TOKENIZER_DEFAULT = DONOR_CONFIG_DEFAULT.with_name("tokenizer.json")
FORMAT = "colorlm-donor-token-map-v1"


class MappingError(RuntimeError):
    """A token mapping contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def raw_token_bytes(reader: GGUFReader) -> list[bytes]:
    field = reader.fields.get("tokenizer.ggml.tokens")
    if field is None or len(field.types) < 2:
        raise MappingError("GGUF缺少tokenizer.ggml.tokens字符串数组")
    result: list[bytes] = []
    for part_index in field.data:
        part = field.parts[part_index]
        result.append(bytes(memoryview(part).cast("B")))
    return result


def scalar_int(reader: GGUFReader, key: str) -> int | None:
    field = reader.fields.get(key)
    return None if field is None else int(field.contents())


def token_types(reader: GGUFReader, expected: int) -> list[int]:
    field = reader.fields.get("tokenizer.ggml.token_type")
    if field is None:
        raise MappingError("GGUF缺少tokenizer.ggml.token_type")
    values = [int(value) for value in field.contents()]
    if len(values) != expected:
        raise MappingError("token_type长度与tokens长度不一致")
    return values


def token_display(value: bytes) -> str:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        return "hex:" + value.hex()
    return text.encode("unicode_escape").decode("ascii")


def metadata_control_ids(
    donor_tokens: list[bytes],
    donor_types: list[int],
    config_path: Path,
    tokenizer_path: Path,
) -> dict[int, set[str]]:
    controls: dict[int, set[str]] = defaultdict(set)
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for role in ("bos_token_id", "eos_token_id", "pad_token_id"):
            raw = config.get(role)
            if isinstance(raw, int) and 0 <= raw < len(donor_tokens):
                controls[raw].add("config." + role)
    if tokenizer_path.is_file():
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        for item in tokenizer.get("added_tokens", []):
            token_id = item.get("id")
            content = item.get("content")
            if not isinstance(token_id, int) or not 0 <= token_id < len(donor_tokens):
                continue
            if not isinstance(content, str) or content.encode("utf-8") != donor_tokens[token_id]:
                raise MappingError(f"tokenizer.json added_token与GGUF不一致: donor_id={token_id}")
            controls[token_id].add("tokenizer.json.added_token")
            if item.get("special") is True:
                controls[token_id].add("tokenizer.json.special")
    for token_id, token_type in enumerate(donor_types):
        if token_type != 1:
            controls[token_id].add(f"gguf.token_type.{token_type}")
    return controls


def tensor_record(reader: GGUFReader, name: str) -> dict[str, Any]:
    matches = [tensor for tensor in reader.tensors if tensor.name == name]
    if len(matches) != 1:
        raise MappingError(f"GGUF内{name}数量不是1")
    tensor = matches[0]
    payload = memoryview(tensor.data).cast("B")
    return {
        "name": name,
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.tensor_type),
        "bytes": int(tensor.n_bytes),
        "data_offset": int(tensor.data_offset),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按GGUF原始token bytes构建donor-to-base精确映射")
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--donor", type=Path, default=DONOR_DEFAULT)
    parser.add_argument("--donor-config", type=Path, default=DONOR_CONFIG_DEFAULT)
    parser.add_argument("--donor-tokenizer", type=Path, default=DONOR_TOKENIZER_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path, label in ((args.base, "base GGUF"), (args.donor, "donor GGUF")):
        if not path.is_file():
            raise MappingError(f"找不到{label}: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_path = args.output_dir / "donor_to_base.i32"
    report_path = args.output_dir / "token_map_report.json"
    if map_path.exists() or report_path.exists():
        raise MappingError("输出已存在，拒绝覆盖；请先选择新的output-dir")

    base = GGUFReader(os.fspath(args.base), "r")
    donor = GGUFReader(os.fspath(args.donor), "r")
    base_tokens = raw_token_bytes(base)
    donor_tokens = raw_token_bytes(donor)
    base_types = token_types(base, len(base_tokens))
    donor_types = token_types(donor, len(donor_tokens))

    base_ids_by_bytes: dict[bytes, list[int]] = defaultdict(list)
    donor_ids_by_bytes: dict[bytes, list[int]] = defaultdict(list)
    for token_id, token in enumerate(base_tokens):
        base_ids_by_bytes[token].append(token_id)
    for token_id, token in enumerate(donor_tokens):
        donor_ids_by_bytes[token].append(token_id)

    mapping = np.full(len(donor_tokens), -1, dtype="<i4")
    ambiguous_base: list[dict[str, Any]] = []
    for donor_id, token in enumerate(donor_tokens):
        candidates = base_ids_by_bytes.get(token, [])
        if len(candidates) == 1:
            mapping[donor_id] = candidates[0]
        elif len(candidates) > 1:
            ambiguous_base.append(
                {
                    "donor_id": donor_id,
                    "base_ids": candidates,
                    "token": token_display(token),
                    "token_sha256": hashlib.sha256(token).hexdigest(),
                }
            )

    mapped_ids = np.flatnonzero(mapping >= 0)
    mapped_base_ids = mapping[mapped_ids]
    mapped_base_counts = Counter(int(value) for value in mapped_base_ids)
    runtime_collisions = [
        {
            "base_id": base_id,
            "donor_ids": [
                int(donor_id) for donor_id in mapped_ids if int(mapping[donor_id]) == base_id
            ],
            "token": token_display(base_tokens[base_id]),
        }
        for base_id, count in sorted(mapped_base_counts.items())
        if count > 1
    ]
    if runtime_collisions:
        raise MappingError(
            "精确映射存在多个donor row写入同一base row，运行时SET_ROWS不安全"
        )

    controls = metadata_control_ids(
        donor_tokens, donor_types, args.donor_config, args.donor_tokenizer
    )
    control_report = []
    for donor_id in sorted(controls):
        base_id = int(mapping[donor_id])
        control_report.append(
            {
                "donor_id": donor_id,
                "base_id": base_id if base_id >= 0 else None,
                "mapped": base_id >= 0,
                "token": token_display(donor_tokens[donor_id]),
                "token_sha256": hashlib.sha256(donor_tokens[donor_id]).hexdigest(),
                "donor_token_type": donor_types[donor_id],
                "base_token_type": base_types[base_id] if base_id >= 0 else None,
                "metadata_sources": sorted(controls[donor_id]),
            }
        )

    mapping.tofile(map_path)
    donor_output = tensor_record(donor, "output.weight")
    donor_embedding = tensor_record(donor, "token_embd.weight")
    donor_norm = tensor_record(donor, "output_norm.weight")
    report = {
        "format": FORMAT,
        "method": "exact-tokenizer.ggml.tokens-raw-bytes",
        "base": {
            "path": os.fspath(args.base.resolve()),
            "bytes": args.base.stat().st_size,
            "vocab_size": len(base_tokens),
            "eos_token_id": scalar_int(base, "tokenizer.ggml.eos_token_id"),
            "padding_token_id": scalar_int(base, "tokenizer.ggml.padding_token_id"),
            "duplicate_token_byte_groups": sum(
                1 for ids in base_ids_by_bytes.values() if len(ids) > 1
            ),
        },
        "donor": {
            "path": os.fspath(args.donor.resolve()),
            "bytes": args.donor.stat().st_size,
            "vocab_size": len(donor_tokens),
            "eos_token_id": scalar_int(donor, "tokenizer.ggml.eos_token_id"),
            "padding_token_id": scalar_int(donor, "tokenizer.ggml.padding_token_id"),
            "duplicate_token_byte_groups": sum(
                1 for ids in donor_ids_by_bytes.values() if len(ids) > 1
            ),
            "output_norm": donor_norm,
            "output": donor_output,
            "token_embedding": donor_embedding,
            "output_tied_to_embedding": donor_output["sha256"] == donor_embedding["sha256"],
        },
        "mapping": {
            "file": map_path.name,
            "dtype": "I32_LE",
            "length": len(mapping),
            "bytes": map_path.stat().st_size,
            "sha256": sha256_file(map_path),
            "mapped": int(mapped_ids.size),
            "unmapped": int(len(mapping) - mapped_ids.size),
            "donor_coverage": float(mapped_ids.size / len(mapping)),
            "unique_base_ids_covered": len(mapped_base_counts),
            "base_vocab_coverage": float(len(mapped_base_counts) / len(base_tokens)),
            "ambiguous_base_token_matches": ambiguous_base,
            "runtime_target_collisions": runtime_collisions,
        },
        "token_type_counts": {
            "base": {str(key): value for key, value in sorted(Counter(base_types).items())},
            "donor": {str(key): value for key, value in sorted(Counter(donor_types).items())},
        },
        "control_tokens": control_report,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"mapped={mapped_ids.size}/{len(mapping)} "
        f"({mapped_ids.size / len(mapping):.4%}), "
        f"base_coverage={len(mapped_base_counts)}/{len(base_tokens)} "
        f"({len(mapped_base_counts) / len(base_tokens):.4%})"
    )
    print(f"map={map_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MappingError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error
