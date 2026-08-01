"""审计 ColorLM v6 与 Qwen3.6 的 L39 MoE 路由/专家配对关系。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GGUF_PY = ROOT / "llama.cpp" / "gguf-py"
sys.path.insert(0, os.fspath(GGUF_PY))

from gguf import GGUFReader  # noqa: E402


LAYER = 39
PAIR_SUFFIXES = (
    "post_attention_norm.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
    "ffn_gate_inp_shexp.weight",
)
TOKEN_IDENTITY_FIELDS = (
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
)


def parse_args() -> argparse.Namespace:
    models = ROOT / "fast16" / "models"
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="审计 Qwen3.6 L39 闭合 MoE 配对")
    parser.add_argument(
        "--pre-router-base",
        type=Path,
        default=models / "ColorLM-v5-GLM-SynapticGraft.gguf",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=models / "ColorLM-v6-Q3Router-Fused-A1.gguf",
    )
    parser.add_argument(
        "--donor",
        type=Path,
        default=models / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    )
    parser.add_argument("--output", type=Path, default=here / "pair_audit.json")
    return parser.parse_args()


def canonical_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, tuple):
        return [canonical_value(item) for item in value]
    if isinstance(value, list):
        return [canonical_value(item) for item in value]
    return value


def value_digest(value: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(value, (list, tuple, np.ndarray)):
        digest.update(b"array-v1\0")
        for item in value:
            encoded = json.dumps(
                canonical_value(item), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        encoded = json.dumps(
            canonical_value(value), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest.update(b"scalar-v1\0")
        digest.update(encoded)
    return digest.hexdigest()


def tensor_digest(tensor) -> str:
    view = np.asarray(tensor.data).view(np.uint8).reshape(-1)
    digest = hashlib.sha256()
    chunk = 16 * 1024 * 1024
    for start in range(0, view.size, chunk):
        digest.update(memoryview(view[start : start + chunk]))
    return digest.hexdigest()


def tokenizer_manifest(reader: GGUFReader) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(name for name in reader.fields if name.startswith("tokenizer.")):
        value = reader.fields[key].contents()
        try:
            count = len(value)
        except TypeError:
            count = None
        result[key] = {"count": count, "sha256": value_digest(value)}
    return result


def tensor_map(reader: GGUFReader) -> dict[str, Any]:
    return {tensor.name: tensor for tensor in reader.tensors}


def router_metrics(left, right) -> dict[str, float]:
    a = np.asarray(left.data, dtype=np.float32).reshape(-1)
    b = np.asarray(right.data, dtype=np.float32).reshape(-1)
    delta = a - b
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "cosine": float(np.dot(a, b) / denom) if denom else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))),
        "max_abs": float(np.max(np.abs(delta))),
        "exact": bool(np.array_equal(a, b)),
    }


def model_summary(path: Path, reader: GGUFReader) -> dict[str, Any]:
    architecture = str(reader.fields["general.architecture"].contents())
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "architecture": architecture,
        "name": str(reader.fields["general.name"].contents()),
        "tensor_count": len(reader.tensors),
    }


def main() -> int:
    args = parse_args()
    for path in (args.pre_router_base, args.base, args.donor):
        if not path.is_file():
            raise FileNotFoundError(path)

    old = GGUFReader(os.fspath(args.pre_router_base), "r")
    base = GGUFReader(os.fspath(args.base), "r")
    donor = GGUFReader(os.fspath(args.donor), "r")
    old_map = tensor_map(old)
    base_map = tensor_map(base)
    donor_map = tensor_map(donor)

    tokenizer_base = tokenizer_manifest(base)
    tokenizer_donor = tokenizer_manifest(donor)
    tokenizer_keys_equal = set(tokenizer_base) == set(tokenizer_donor)
    tokenizer_mismatches = sorted(
        key
        for key in set(tokenizer_base) | set(tokenizer_donor)
        if tokenizer_base.get(key) != tokenizer_donor.get(key)
    )
    tokenizer_identity_mismatches = [
        key
        for key in TOKEN_IDENTITY_FIELDS
        if tokenizer_base.get(key) != tokenizer_donor.get(key)
    ]

    tensors = []
    replacement_bytes = 0
    replaced_base_bytes = 0
    for suffix in PAIR_SUFFIXES:
        name = f"blk.{LAYER}.{suffix}"
        if name not in old_map or name not in base_map or name not in donor_map:
            raise RuntimeError(f"缺少闭合 MoE 张量: {name}")
        old_tensor = old_map[name]
        base_tensor = base_map[name]
        donor_tensor = donor_map[name]
        if tuple(base_tensor.shape) != tuple(donor_tensor.shape):
            raise RuntimeError(
                f"张量形状不兼容: {name}: {base_tensor.shape} vs {donor_tensor.shape}"
            )
        old_hash = tensor_digest(old_tensor)
        base_hash = tensor_digest(base_tensor)
        donor_hash = tensor_digest(donor_tensor)
        tensors.append(
            {
                "name": name,
                "shape": [int(value) for value in base_tensor.shape],
                "pre_router_base": {
                    "type": int(old_tensor.tensor_type),
                    "bytes": int(old_tensor.data.nbytes),
                    "sha256": old_hash,
                },
                "v6": {
                    "type": int(base_tensor.tensor_type),
                    "bytes": int(base_tensor.data.nbytes),
                    "sha256": base_hash,
                },
                "qwen36": {
                    "type": int(donor_tensor.tensor_type),
                    "bytes": int(donor_tensor.data.nbytes),
                    "sha256": donor_hash,
                },
                "pre_router_base_equals_v6": old_hash == base_hash,
                "v6_equals_qwen36": base_hash == donor_hash,
            }
        )
        replaced_base_bytes += int(base_tensor.data.nbytes)
        replacement_bytes += int(donor_tensor.data.nbytes)

    router_name = f"blk.{LAYER}.ffn_gate_inp.weight"
    report = {
        "format": "colormlm-qwen36-expert-pair-audit-v1",
        "models": {
            "pre_router_base": model_summary(args.pre_router_base, old),
            "base": model_summary(args.base, base),
            "donor": model_summary(args.donor, donor),
        },
        "tokenizer": {
            "key_sets_equal": tokenizer_keys_equal,
            "all_fields_equal": tokenizer_keys_equal and not tokenizer_mismatches,
            "token_identity_fields": list(TOKEN_IDENTITY_FIELDS),
            "token_identity_equal": not tokenizer_identity_mismatches,
            "token_identity_mismatches": tokenizer_identity_mismatches,
            "operational_metadata_mismatches": [
                key
                for key in tokenizer_mismatches
                if key not in TOKEN_IDENTITY_FIELDS
            ],
            "field_count_base": len(tokenizer_base),
            "field_count_donor": len(tokenizer_donor),
            "mismatches": tokenizer_mismatches,
            "base": tokenizer_base,
            "donor": tokenizer_donor,
        },
        "layer": LAYER,
        "closed_moe_tensor_count": len(PAIR_SUFFIXES),
        "closed_moe_tensors": tensors,
        "router": {
            "pre_router_base_vs_v6": router_metrics(
                old_map[router_name], base_map[router_name]
            ),
            "v6_vs_qwen36": router_metrics(
                base_map[router_name], donor_map[router_name]
            ),
            "pre_router_base_vs_qwen36": router_metrics(
                old_map[router_name], donor_map[router_name]
            ),
        },
        "bytes": {
            "replaced_base_payload": replaced_base_bytes,
            "qwen36_pair_payload": replacement_bytes,
            "estimated_model_delta": replacement_bytes - replaced_base_bytes,
            "estimated_output": args.base.stat().st_size
            + replacement_bytes
            - replaced_base_bytes,
        },
    }

    non_router = [
        item
        for item in tensors
        if item["name"] != router_name
    ]
    report["conclusion"] = {
        "v6_only_changed_router_inside_l39_moe": all(
            item["pre_router_base_equals_v6"] for item in non_router
        )
        and not next(
            item for item in tensors if item["name"] == router_name
        )["pre_router_base_equals_v6"],
        "v6_router_is_not_exact_qwen36": not report["router"]["v6_vs_qwen36"][
            "exact"
        ],
        "closed_pair_is_shape_compatible": True,
        "prototype_allowed": bool(
            report["tokenizer"]["token_identity_equal"]
            and report["router"]["v6_vs_qwen36"]["cosine"] > 0.95
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["conclusion"], ensure_ascii=False, indent=2))
    print(json.dumps(report["router"], ensure_ascii=False, indent=2))
    print(json.dumps(report["bytes"], ensure_ascii=False, indent=2))
    print(f"审计报告: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
