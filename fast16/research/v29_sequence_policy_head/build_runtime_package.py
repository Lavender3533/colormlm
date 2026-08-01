"""Build the strict aligned runtime package for the accepted v29 policy head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ALIGNMENT = 64


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def main() -> int:
    parser = argparse.ArgumentParser(description="构建v29序列策略头运行包")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if not report.get("gate_passed"):
        raise ValueError("v29离线门未通过，拒绝构建运行包")
    if report["contract_sha256"] != sha256_file(args.contract):
        raise ValueError("v29报告与冻结契约不一致")
    if report["weights_sha256"] != sha256_file(args.weights):
        raise ValueError("v29报告与NPZ权重不一致")

    with np.load(args.weights, allow_pickle=False) as package:
        token_ids = np.ascontiguousarray(package["token_ids"], dtype="<i8")
        weight = np.ascontiguousarray(package["weight"], dtype="<f4")
        bias = np.ascontiguousarray(package["bias"], dtype="<f4")
        correction_min = float(package["correction_min"][0])
        correction_max = float(package["correction_max"][0])
    candidate_tokens = int(report["candidate_token_count"])
    if token_ids.shape != (candidate_tokens,):
        raise ValueError("token_ids形状错误")
    if weight.shape != (candidate_tokens, 2048) or bias.shape != (candidate_tokens,):
        raise ValueError("v29线性头形状错误")
    if token_ids.tolist() != report["candidate_token_ids"]:
        raise ValueError("候选token ID与报告不一致")
    if len(set(int(value) for value in token_ids)) != candidate_tokens:
        raise ValueError("候选token ID重复")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "policy.json"
    weights_path = args.output / "weights.bin"
    if manifest_path.exists() or weights_path.exists():
        raise FileExistsError("v29运行包已存在，拒绝覆盖")

    sources = [
        ("policy.weight", "F32", [2048, candidate_tokens], weight.tobytes(order="C")),
        ("policy.bias", "F32", [candidate_tokens], bias.tobytes(order="C")),
        ("policy.base_ids", "I64", [candidate_tokens], token_ids.tobytes(order="C")),
    ]
    records = []
    blob = bytearray()
    for name, dtype, shape, payload in sources:
        offset = align(len(blob))
        blob.extend(b"\x00" * (offset - len(blob)))
        blob.extend(payload)
        records.append(
            {
                "name": name,
                "dtype": dtype,
                "ggml_shape": shape,
                "offset": offset,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    weights_path.write_bytes(blob)
    manifest = {
        "format": "colorlm-sequence-policy-runtime-v1",
        "formal": True,
        "architecture": "ColorLMV4",
        "hidden_size": 2048,
        "base_vocab_size": 248320,
        "candidate_token_count": candidate_tokens,
        "normalization": contract["fit"]["hidden_normalization"],
        "correction_min": correction_min,
        "correction_max": correction_max,
        "source_contract_sha256": sha256_file(args.contract),
        "source_report_sha256": sha256_file(args.report),
        "runtime_layout": "single-aligned-tensor-blob-v1",
        "alignment": ALIGNMENT,
        "tensor_count": len(records),
        "tensors": records,
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
                "manifest": manifest_path.as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "weights": weights_path.as_posix(),
                "weights_sha256": sha256_file(weights_path),
                "weights_bytes": len(blob),
                "candidate_tokens": candidate_tokens,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
