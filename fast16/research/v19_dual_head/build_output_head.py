"""Extract the donor terminal norm/head and exact sparse token indices."""

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
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


DONOR_DEFAULT = (
    ROOT
    / "fast16"
    / "models"
    / "donor"
    / "qwen3-coder-next-iq3s"
    / "Qwen3-Coder-Next-UD-IQ3_S.gguf"
)
REPORT_DEFAULT = Path(__file__).resolve().parent / "token_map_report.json"
MAP_DEFAULT = Path(__file__).resolve().parent / "donor_to_base.i32"
FORMAT = "colorlm-neural-output-head-runtime-v2"
ALIGNMENT = 64


class BuildError(RuntimeError):
    """An output-head package contract violation."""


def sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取v19供体末端Norm、Q6_K输出头和精确映射索引")
    parser.add_argument("--donor", type=Path, default=DONOR_DEFAULT)
    parser.add_argument("--token-map-report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--token-map", type=Path, default=MAP_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runtime-head-v2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise BuildError(f"输出目录已存在，拒绝覆盖: {args.output_dir}")
    report = json.loads(args.token_map_report.read_text(encoding="utf-8"))
    mapping_info = report.get("mapping", {})
    donor_info = report.get("donor", {})
    base_info = report.get("base", {})
    if (
        report.get("format") != "colorlm-donor-token-map-v1"
        or mapping_info.get("sha256") != sha256_file(args.token_map)
        or int(mapping_info.get("length", -1)) <= 0
        or mapping_info.get("runtime_target_collisions") != []
        or mapping_info.get("ambiguous_base_token_matches") != []
    ):
        raise BuildError("token map报告未通过运行契约")
    mapping = np.fromfile(args.token_map, dtype="<i4")
    if mapping.size != int(mapping_info["length"]):
        raise BuildError("token map长度不匹配")
    donor_ids = np.flatnonzero(mapping >= 0).astype("<i4")
    base_ids = np.ascontiguousarray(mapping[donor_ids], dtype="<i8")
    if donor_ids.size != int(mapping_info["mapped"]):
        raise BuildError("token map映射计数不匹配")
    if np.unique(base_ids).size != base_ids.size:
        raise BuildError("base目标ID重复，SET_ROWS不安全")
    if np.any(base_ids < 0) or np.any(base_ids >= int(base_info["vocab_size"])):
        raise BuildError("base目标ID越界")

    donor = GGUFReader(os.fspath(args.donor), "r")
    tensors = {tensor.name: tensor for tensor in donor.tensors}
    if set(("output_norm.weight", "output.weight")) - set(tensors):
        raise BuildError("donor GGUF缺少末端输出张量")
    output_norm = tensors["output_norm.weight"]
    output = tensors["output.weight"]
    if (
        [int(value) for value in output_norm.shape] != [2048]
        or str(output_norm.tensor_type) != "0"
        or int(output_norm.n_bytes) != 8192
        or [int(value) for value in output.shape]
        != [2048, int(donor_info["vocab_size"])]
        or str(output.tensor_type) != "14"
        or int(output.n_bytes) != 255252480
    ):
        raise BuildError("donor末端输出张量布局不符合已审计契约")

    # Q6_K stores each vocabulary row as eight complete 256-element blocks.
    # Copy whole raw rows so the mapped-only projection is bit-identical to
    # gathering the same rows from the original full donor head.
    row_bytes = int(output.n_bytes) // int(donor_info["vocab_size"])
    if output.data.shape != (int(donor_info["vocab_size"]), row_bytes):
        raise BuildError("donor Q6_K输出头物理行布局不符合预期")
    mapped_output = np.ascontiguousarray(output.data[donor_ids])
    if mapped_output.shape != (int(donor_ids.size), row_bytes):
        raise BuildError("mapped-only Q6_K输出头裁剪失败")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    weights_path = args.output_dir / "weights.bin"
    records: list[dict[str, Any]] = []
    offset = 0
    with weights_path.open("wb") as stream:
        payloads = [
            (
                "output_norm.weight",
                "F32",
                [2048],
                memoryview(output_norm.data).cast("B"),
            ),
            (
                "output_mapped.weight",
                "Q6_K",
                [2048, int(donor_ids.size)],
                memoryview(mapped_output).cast("B"),
            ),
            (
                "mapping.base_ids",
                "I64",
                [int(base_ids.size)],
                memoryview(base_ids).cast("B"),
            ),
        ]
        for name, dtype, shape, payload in payloads:
            aligned = align(offset)
            if aligned > offset:
                stream.write(b"\x00" * (aligned - offset))
            offset = aligned
            stream.write(payload)
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
            offset += len(payload)

    manifest = {
        "format": FORMAT,
        "name": "ColorLM-v19-Coder-Next-Terminal-Output-Head",
        "formal": True,
        "runtime_layout": "single-aligned-tensor-blob-v1",
        "alignment": ALIGNMENT,
        "source": {
            "architecture": "Qwen3NextForCausalLM",
            "gguf": os.fspath(args.donor.resolve()),
            "vocab_size": int(donor_info["vocab_size"]),
            "hidden_size": 2048,
            "output_tied_to_embedding": bool(donor_info["output_tied_to_embedding"]),
        },
        "target": {
            "architecture": "ColorLMV4",
            "vocab_size": int(base_info["vocab_size"]),
        },
        "mapping": {
            "method": report["method"],
            "report": os.fspath(args.token_map_report.resolve()),
            "report_sha256": sha256_file(args.token_map_report),
            "source_map_sha256": sha256_file(args.token_map),
            "mapped_tokens": int(donor_ids.size),
            "donor_coverage": float(mapping_info["donor_coverage"]),
            "base_vocab_coverage": float(mapping_info["base_vocab_coverage"]),
            "target_collisions": 0,
            "projection_layout": "mapped-only-q6-k-raw-rows",
            "source_row_bytes": row_bytes,
            "source_donor_ids_sha256": sha256_bytes(memoryview(donor_ids).cast("B")),
        },
        "weights": {
            "file": weights_path.name,
            "bytes": weights_path.stat().st_size,
            "sha256": sha256_file(weights_path),
        },
        "tensor_count": len(records),
        "tensors": records,
    }
    manifest_path = args.output_dir / "head.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"weights={weights_path} ({weights_path.stat().st_size} bytes)")
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={sha256_file(manifest_path)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error
