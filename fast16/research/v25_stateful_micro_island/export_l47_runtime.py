"""把通过分站门的L47 rank-32线性候选导出为严格运行包。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ALIGNMENT = 64


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def aligned(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def main() -> int:
    parser = argparse.ArgumentParser(description="导出v25 L47微层运行包")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = np.load(args.artifact, allow_pickle=False)
    basis = np.asarray(source["basis"], dtype=np.float32)
    projection = np.asarray(source["output_projection"], dtype=np.float32)
    x_mean = np.asarray(source["x_mean"], dtype=np.float32)
    y_mean = np.asarray(source["y_mean"], dtype=np.float32)
    rho = float(np.asarray(source["rho"], dtype=np.float32)[0])
    if basis.ndim != 2 or basis.shape[1] != 2048:
        raise RuntimeError("L47运行候选输入投影shape无效")
    rank = int(basis.shape[0])
    if not 1 <= rank <= 256 or projection.shape != (2 * rank, 2048):
        raise RuntimeError("L47运行候选rank或双特征投影shape无效")
    if x_mean.shape != (2048,) or y_mean.shape != (2048,) or rho != 0.0:
        raise RuntimeError("L47运行候选均值或rho不符合冻结契约")

    # rho=0时当前状态等于当前q；把重复的[q,z]两半精确折叠成单个输出矩阵。
    folded = projection[:rank] + projection[rank:]
    bias = y_mean - (x_mean @ basis.T) @ folded
    tensors = [
        ("input.weight", "F16", basis.astype("<f2"), [2048, rank]),
        ("output.weight", "F16", folded.T.astype("<f2"), [rank, 2048]),
        ("output.bias", "F32", bias.astype("<f4"), [2048]),
    ]
    blob = bytearray()
    records = []
    for name, dtype, array, ggml_shape in tensors:
        offset = aligned(len(blob))
        blob.extend(b"\0" * (offset - len(blob)))
        payload = array.tobytes(order="C")
        blob.extend(payload)
        records.append(
            {
                "name": name,
                "dtype": dtype,
                "ggml_shape": ggml_shape,
                "offset": offset,
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "weights.bin"
    weights_path.write_bytes(blob)
    manifest = {
        "format": "colorlm-neural-micro-stage-runtime-v1",
        "candidate": True,
        "runtime_layout": "single-aligned-tensor-blob-v1",
        "alignment": ALIGNMENT,
        "source_layer": 47,
        "hidden_size": 2048,
        "rank": rank,
        "equation": "delta=output.weight*(input.weight*h)+output.bias",
        "tensor_count": len(records),
        "tensors": records,
        "weights": {
            "file": weights_path.name,
            "bytes": len(blob),
            "sha256": sha256(bytes(blob)),
        },
        "source_artifact": args.artifact.as_posix(),
    }
    manifest_path = args.output_dir / "micro_stage.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256(manifest_path.read_bytes()),
                "weights": str(weights_path.resolve()),
                "weights_sha256": manifest["weights"]["sha256"],
                "bytes": len(blob),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
