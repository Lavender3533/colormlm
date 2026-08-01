"""把通过独立门的v30低秩双线性权重封装为严格对齐运行包。"""

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
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="构建v30动态词汇策略头运行包")
    parser.add_argument("--weights", type=Path, default=here / "probe_weights.npz")
    parser.add_argument("--gate-report", type=Path, default=here / "grounded_gate_report.json")
    parser.add_argument("--contract", type=Path, default=here / "grounded_gate_contract.json")
    parser.add_argument("--output", type=Path, default=here / "runtime-v1")
    args = parser.parse_args()

    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if not gate.get("gate_passed") or gate.get("status") != "passed_for_runtime_prototype":
        raise ValueError("v30独立门未通过，拒绝构建运行包")
    if gate["contract_sha256"] != sha256_file(args.contract):
        raise ValueError("v30门报告与冻结契约不一致")
    if gate["weights_sha256"] != sha256_file(args.weights):
        raise ValueError("v30门报告与冻结权重不一致")
    with np.load(args.weights, allow_pickle=False) as package:
        left = np.ascontiguousarray(package["left"], dtype="<f4")
        right = np.ascontiguousarray(package["right"], dtype="<f4")
        static_ids = np.ascontiguousarray(package["static_token_ids"], dtype="<i8")
    rank = int(contract["candidate"]["rank"])
    capacity = int(contract["candidate"]["maximum_dynamic_tokens"])
    if left.shape != (rank, 2048) or right.shape != (rank, 2048):
        raise ValueError("v30低秩权重形状错误")
    if static_ids.ndim != 1 or len(static_ids) == 0 or len(static_ids) >= capacity:
        raise ValueError("v30静态协议token数量错误")
    if len(set(int(value) for value in static_ids)) != len(static_ids):
        raise ValueError("v30静态协议token重复")
    if np.any(static_ids < 0) or np.any(static_ids >= 248320):
        raise ValueError("v30静态协议token越界")

    args.output.mkdir(parents=True, exist_ok=False)
    weights_path = args.output / "weights.bin"
    manifest_path = args.output / "policy.json"
    sources = [
        ("policy.left", "F32", [2048, rank], left.tobytes(order="C")),
        ("policy.right", "F32", [2048, rank], right.tobytes(order="C")),
        ("policy.static_ids", "I64", [len(static_ids)], static_ids.tobytes(order="C")),
    ]
    blob = bytearray()
    records = []
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
        "format": "colorlm-sequence-policy-runtime-v2",
        "formal": True,
        "architecture": "ColorLMV4",
        "mode": "dynamic-low-rank-bilinear",
        "hidden_size": 2048,
        "embedding_size": 2048,
        "base_vocab_size": 248320,
        "rank": rank,
        "candidate_capacity": capacity,
        "static_token_count": len(static_ids),
        "hidden_normalization": "per-sample-l2",
        "embedding_normalization": "per-token-l2",
        "correction_min": float(contract["candidate"]["correction_min"]),
        "correction_max": float(contract["candidate"]["correction_max"]),
        "source_contract_sha256": sha256_file(args.contract),
        "source_gate_report_sha256": sha256_file(args.gate_report),
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
                "rank": rank,
                "candidate_capacity": capacity,
                "static_token_count": len(static_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
