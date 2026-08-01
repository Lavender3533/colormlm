"""把通过F32离线门的v43策略头封装成64字节对齐运行包。"""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if not report.get("gate_passed"):
        raise ValueError("v43最终F32离线门未通过")
    if report.get("evaluation_precision") != "final-deployment-f32-parameters-with-f64-accumulation":
        raise ValueError("v43报告不是最终F32参数回放")
    if report["contract_sha256"] != sha256_file(args.contract):
        raise ValueError("v43报告与冻结合同不一致")
    if report["weights_sha256"] != sha256_file(args.weights):
        raise ValueError("v43报告与NPZ权重不一致")

    with np.load(args.weights, allow_pickle=False) as package:
        candidate_ids = np.ascontiguousarray(package["candidate_ids"], dtype="<i8")
        hidden_mean = np.ascontiguousarray(package["hidden_mean"], dtype="<f4")
        pca = np.ascontiguousarray(package["pca_components"], dtype="<f4")
        classifier = np.ascontiguousarray(package["classifier"], dtype="<f4")
        strength = np.ascontiguousarray(package["correction_strength"], dtype="<f4")
    candidate_count = int(candidate_ids.size)
    rank = int(report["pca_rank"])
    if not 4 <= candidate_count <= 16:
        raise ValueError("v43候选数必须在[4,16]")
    if candidate_ids.shape != (candidate_count,) or candidate_ids.tolist() != report["candidate_token_ids"]:
        raise ValueError("v43候选ID形状或顺序错误")
    if len(set(map(int, candidate_ids))) != candidate_count or np.any(np.diff(candidate_ids) <= 0):
        raise ValueError("v43候选ID必须严格递增且唯一")
    if hidden_mean.shape != (2048,) or pca.shape != (rank, 2048):
        raise ValueError("v43 PCA张量形状错误")
    if classifier.shape != (rank + 1, candidate_count + 1) or strength.shape != (1,):
        raise ValueError("v43分类器或强度形状错误")
    if rank != int(contract["fit"]["pca_rank"]) or float(strength[0]) != float(contract["fit"]["correction_strength"]):
        raise ValueError("v43 rank或强度偏离冻结合同")
    if not all(np.isfinite(value).all() for value in (hidden_mean, pca, classifier, strength)):
        raise ValueError("v43运行权重包含非有限值")

    # classifier[0]是截距；其余8行转置后符合ggml [rank, class]行布局。
    classifier_weight = np.ascontiguousarray(classifier[1:, :].T, dtype="<f4")
    classifier_bias = np.ascontiguousarray(classifier[0, :], dtype="<f4")
    sources = [
        ("policy.base_ids", "I64", [candidate_count], candidate_ids.tobytes(order="C")),
        ("policy.hidden_mean", "F32", [2048], hidden_mean.tobytes(order="C")),
        ("policy.pca_components", "F32", [2048, rank], pca.tobytes(order="C")),
        ("policy.classifier", "F32", [rank, candidate_count + 1], classifier_weight.tobytes(order="C")),
        ("policy.classifier_bias", "F32", [candidate_count + 1], classifier_bias.tobytes(order="C")),
        ("policy.correction_strength", "F32", [1], strength.tobytes(order="C")),
    ]

    args.output.mkdir(parents=True, exist_ok=False)
    weights_path = args.output / "weights.bin"
    manifest_path = args.output / "policy.json"
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
        "format": "colorlm-sequence-policy-runtime-v3",
        "formal": True,
        "architecture": "ColorLMV4",
        "mode": "pca-rank8-multiclass-noop-sparse",
        "hidden_size": 2048,
        "base_vocab_size": 248320,
        "candidate_token_count": candidate_count,
        "class_count": candidate_count + 1,
        "pca_rank": rank,
        "no_op_class": 0,
        "no_op_rule": "exact-no-op-iff-class-0-is-argmax",
        "hidden_normalization": "per-sample-l2",
        "correction_strength": float(strength[0]),
        "correction_min": 0.0,
        "correction_max": float(strength[0]),
        "source_contract_sha256": sha256_file(args.contract),
        "source_report_sha256": sha256_file(args.report),
        "source_weights_sha256": sha256_file(args.weights),
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
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "weights": str(weights_path),
                "weights_sha256": sha256_file(weights_path),
                "weights_bytes": len(blob),
                "candidate_tokens": candidate_count,
                "class_count": candidate_count + 1,
                "rank": rank,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
