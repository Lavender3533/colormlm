"""Build the pre-registered v40 norm-matched sequence-policy package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ALIGNMENT = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as package:
        return {name: package[name].copy() for name in package.files}


def main() -> int:
    parser = argparse.ArgumentParser(description="构建v40范数匹配策略头")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--old-weights", type=Path, required=True)
    parser.add_argument("--native-weights", type=Path, required=True)
    parser.add_argument("--scaled-weights", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract["transform"]["parameter_scan_allowed"] is not False:
        raise ValueError("v40合同必须禁止盲测后扫描")
    if sha256_file(args.old_weights) != contract["old_policy_weights_sha256"]:
        raise ValueError("旧策略头SHA-256不匹配")
    if sha256_file(args.native_weights) != contract["native_policy_weights_sha256"]:
        raise ValueError("v39原生头SHA-256不匹配")

    old = load(args.old_weights)
    native = load(args.native_weights)
    if not np.array_equal(old["token_ids"], native["token_ids"]):
        raise ValueError("候选token ID不一致")
    old_norm = float(np.linalg.norm(old["weight"].astype(np.float64)))
    native_norm = float(np.linalg.norm(native["weight"].astype(np.float64)))
    scale = old_norm / native_norm
    expected = float(contract["transform"]["expected_scale"])
    if not np.isclose(scale, expected, rtol=1e-7, atol=1e-9):
        raise ValueError(f"范数匹配比例漂移: {scale} != {expected}")

    token_ids = np.ascontiguousarray(native["token_ids"], dtype="<i8")
    weight = np.ascontiguousarray(native["weight"].astype(np.float64) * scale, dtype="<f4")
    bias = np.ascontiguousarray(native["bias"].astype(np.float64) * scale, dtype="<f4")
    correction_min = float(native["correction_min"][0]) * scale
    correction_max = float(native["correction_max"][0]) * scale

    if args.scaled_weights.exists() or args.runtime.exists():
        raise FileExistsError("v40输出已存在，拒绝覆盖研究证据")
    args.scaled_weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.scaled_weights,
        token_ids=token_ids.astype("<i4"),
        weight=weight,
        bias=bias,
        correction_min=np.asarray([correction_min], dtype="<f4"),
        correction_max=np.asarray([correction_max], dtype="<f4"),
    )

    sources = [
        ("policy.weight", "F32", [2048, len(token_ids)], weight.tobytes(order="C")),
        ("policy.bias", "F32", [len(token_ids)], bias.tobytes(order="C")),
        ("policy.base_ids", "I64", [len(token_ids)], token_ids.tobytes(order="C")),
    ]
    blob = bytearray()
    tensors = []
    for name, dtype, shape, payload in sources:
        offset = align(len(blob))
        blob.extend(b"\x00" * (offset - len(blob)))
        blob.extend(payload)
        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "ggml_shape": shape,
                "offset": offset,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )

    args.runtime.mkdir(parents=True)
    weights_path = args.runtime / "weights.bin"
    manifest_path = args.runtime / "policy.json"
    weights_path.write_bytes(blob)
    manifest = {
        "format": "colorlm-sequence-policy-runtime-v1",
        "formal": True,
        "research_status": "pre-registered-blind-candidate",
        "architecture": "ColorLMV4",
        "hidden_size": 2048,
        "base_vocab_size": 248320,
        "candidate_token_count": int(len(token_ids)),
        "normalization": "per-sample-l2",
        "correction_min": correction_min,
        "correction_max": correction_max,
        "source_contract_sha256": sha256_file(args.contract),
        "source_native_weights_sha256": sha256_file(args.native_weights),
        "source_old_weights_sha256": sha256_file(args.old_weights),
        "transform": "native-weight-norm-matched-to-old-v29",
        "scale": scale,
        "runtime_layout": "single-aligned-tensor-blob-v1",
        "alignment": ALIGNMENT,
        "tensor_count": len(tensors),
        "tensors": tensors,
        "weights": {
            "file": weights_path.name,
            "bytes": len(blob),
            "sha256": sha256_file(weights_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scale": scale,
                "scaled_weight_l2": float(np.linalg.norm(weight.astype(np.float64))),
                "scaled_weights": args.scaled_weights.as_posix(),
                "scaled_weights_sha256": sha256_file(args.scaled_weights),
                "runtime": args.runtime.as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "weights_bytes": len(blob),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
