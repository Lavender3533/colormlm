"""把已验证的 K3 F16 宏胶囊编译为桥 F16、专家量化的 v3 运行包。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(ROOT / "llama.cpp" / "gguf-py"))

from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402


BRIDGE_KEYS = ("b_in", "norm", "b_out")
EXPERT_KEYS = ("gate", "up", "down")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 K3 latent macro capsule v2 的三张专家矩阵量化为 Q4_0"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quantization",
        choices=("Q4_0", "Q5_0", "Q8_0", "MXFP4"),
        default="Q4_0",
        help="专家矩阵量化类型；桥和路由不量化",
    )
    parser.add_argument(
        "--hybrid-decode",
        action="store_true",
        help="prefill保留F16专家，仅在单token decode使用原生MXFP4专家",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checked_source_file(source: Path, record: dict, key: str) -> Path:
    path = source / str(record["file"])
    expected_bytes = int(record["bytes"])
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise RuntimeError(f"{key} 源文件尺寸不匹配: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise RuntimeError(f"{key} 源文件 SHA-256 不匹配: {path}")
    return path


def install_file(source: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def quantize_expert(
    source: Path,
    shape: list[int],
    target: Path,
    qtype: GGMLQuantizationType,
) -> int:
    if len(shape) != 2 or shape[-1] % 32:
        raise RuntimeError(f"Q4_0 逻辑形状非法: {shape}")
    matrix = np.memmap(source, mode="r", dtype="<f2", shape=tuple(shape))
    encoded = quantize(matrix, qtype)
    block_bytes = {"Q4_0": 18, "Q5_0": 22, "Q8_0": 34}[qtype.name]
    expected = shape[0] * (shape[1] // 32) * block_bytes
    if encoded.nbytes != expected:
        raise RuntimeError(f"Q4_0 编码尺寸异常: {encoded.nbytes} vs {expected}")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded.tobytes(order="C"))
    temporary.replace(target)
    return expected


def plan_component(plan: dict, weight: str, component: str) -> dict:
    matches = [
        record
        for record in plan["tensors"]
        if record["weight"] == weight and record["component"] == component
    ]
    if len(matches) != 1:
        raise RuntimeError(f"原生 MXFP4 组件数量异常: {weight}.{component}")
    return matches[0]


def raw_component(raw: np.memmap, record: dict) -> np.ndarray:
    first = int(record["capsule_offset_start"])
    stop = int(record["capsule_offset_end"])
    shape = tuple(int(value) for value in record["shape"])
    view = raw[first:stop]
    if view.size != int(np.prod(shape, dtype=np.int64)):
        raise RuntimeError(f"原生 MXFP4 组件长度异常: {record['name']}")
    return view.reshape(shape)


def compile_native_mxfp4(
    expert_dir: Path,
    role: str,
    shape: list[int],
    target: Path,
    source_f16: Path,
) -> tuple[int, dict[str, float | int | bool]]:
    plan = json.loads((expert_dir / "source-plan.json").read_text(encoding="utf-8"))
    raw_path = expert_dir / "expert.mxfp4.bin"
    if sha256_file(raw_path) != str(plan.get("raw_sha256", sha256_file(raw_path))):
        raise RuntimeError("原始 MXFP4 SHA-256 不匹配")
    weight = {"gate": "w1", "up": "w3", "down": "w2"}[role]
    packed_record = plan_component(plan, weight, "weight_packed")
    scale_record = plan_component(plan, weight, "weight_scale")
    raw = np.memmap(raw_path, mode="r", dtype=np.uint8)
    packed = raw_component(raw, packed_record)
    scales = raw_component(raw, scale_record)
    rows, width = shape
    groups = width // 32
    if packed.shape != (rows, width // 2) or scales.shape != (rows, groups):
        raise RuntimeError(f"{role} 原生 MXFP4 shape 不匹配")

    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for first in range(0, rows, 128):
            stop = min(first + 128, rows)
            source_blocks = np.asarray(packed[first:stop]).reshape(-1, groups, 16)
            nibbles = np.empty((stop - first, groups, 32), dtype=np.uint8)
            nibbles[..., 0::2] = source_blocks & np.uint8(0x0F)
            nibbles[..., 1::2] = source_blocks >> np.uint8(4)
            ggml_qs = nibbles[..., :16] | (nibbles[..., 16:] << np.uint8(4))
            blocks = np.empty((stop - first, groups, 17), dtype=np.uint8)
            blocks[..., 0] = scales[first:stop]
            blocks[..., 1:] = ggml_qs
            stream.write(blocks.tobytes(order="C"))
    temporary.replace(target)
    expected = rows * groups * 17
    if target.stat().st_size != expected:
        raise RuntimeError(f"{role} 原生 MXFP4 文件尺寸异常")

    selected = np.asarray(sorted({0, rows // 2, rows - 1}), dtype=np.int64)
    encoded = np.memmap(target, mode="r", dtype=np.uint8, shape=(rows, groups * 17))
    restored = dequantize(np.ascontiguousarray(encoded[selected]), GGMLQuantizationType.MXFP4)
    restored = restored.reshape(len(selected), width)
    reference = np.asarray(
        np.memmap(source_f16, mode="r", dtype="<f2", shape=(rows, width))[selected],
        dtype=np.float32,
    )
    difference = np.abs(restored - reference)
    audit = {
        "sample_rows": int(len(selected)),
        "sample_values": int(difference.size),
        "exact_to_f16": bool(np.array_equal(restored, reference)),
        "max_abs_diff": float(difference.max(initial=0.0)),
    }
    return expected, audit


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    source_manifest_path = source / "capsule.json"
    if not source_manifest_path.is_file():
        raise RuntimeError(f"缺少源清单: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("format")
        != "colorlm-kimi-k3-latent-macro-capsule-v2"
        or source_manifest.get("runtime_layout")
        != "six-headerless-f16-le-row-major-v1"
    ):
        raise RuntimeError("只接受带 router 的 K3 v2 F16 宏胶囊")
    if args.hybrid_decode and args.quantization != "MXFP4":
        raise RuntimeError("阶段自适应精度目前只接受原生MXFP4 decode专家")

    runtime_files = source_manifest.get("runtime_files")
    if not isinstance(runtime_files, dict) or set(runtime_files) != set(BRIDGE_KEYS + EXPERT_KEYS):
        raise RuntimeError("源胶囊必须恰好包含六张运行张量")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"目标目录非空，拒绝覆盖: {output}")
    output.mkdir(parents=True, exist_ok=True)
    qtype = GGMLQuantizationType[args.quantization]
    suffix = args.quantization.lower()
    native_expert_dir: Path | None = None
    if args.quantization == "MXFP4":
        native_expert_dir = Path(str(source_manifest["expert_capsule"]["path"]))
        if not native_expert_dir.is_absolute():
            native_expert_dir = ROOT / native_expert_dir
        native_expert_dir = native_expert_dir.resolve()
        if not (native_expert_dir / "source-plan.json").is_file():
            raise RuntimeError(f"缺少原生 MXFP4 source-plan: {native_expert_dir}")

    compiled_files: dict[str, dict] = {}
    for key in BRIDGE_KEYS:
        record = runtime_files[key]
        path = checked_source_file(source, record, key)
        target = output / str(record["file"])
        install_file(path, target)
        compiled_files[key] = {
            **record,
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
        print(f"F16 保留 {key}: {target.stat().st_size / 1024**2:.2f} MiB", flush=True)

    for key in EXPERT_KEYS:
        record = runtime_files[key]
        path = checked_source_file(source, record, key)
        shape = [int(value) for value in record["shape"]]
        if args.hybrid_decode:
            f16_target = output / f"{key}.f16"
            install_file(path, f16_target)
            compiled_files[key] = {
                **record,
                "file": f16_target.name,
                "bytes": f16_target.stat().st_size,
                "sha256": sha256_file(f16_target),
            }
            record_key = f"{key}_decode"
            filename = f"{record_key}.{suffix}"
        else:
            record_key = key
            filename = f"{key}.{suffix}"
        target = output / filename
        native_audit = None
        if qtype == GGMLQuantizationType.MXFP4:
            assert native_expert_dir is not None
            size, native_audit = compile_native_mxfp4(
                native_expert_dir, key, shape, target, path
            )
        else:
            size = quantize_expert(path, shape, target, qtype)
        compiled_files[record_key] = {
            "file": filename,
            "shape": shape,
            "ggml_shape": list(reversed(shape)),
            "dtype": args.quantization,
            "layout": f"ggml-{suffix}-row-major-pytorch-out-in",
            "bytes": size,
            "sha256": sha256_file(target),
            "source_file": str(path),
            "source_sha256": str(record["sha256"]),
        }
        if native_audit is not None:
            compiled_files[record_key]["native_repack_audit"] = native_audit
        print(f"{args.quantization} 编译 {key}: {size / 1024**2:.2f} MiB", flush=True)

    router_record = source_manifest.get("runtime_router")
    if not isinstance(router_record, dict):
        raise RuntimeError("源胶囊缺少 runtime_router")
    router_source = checked_source_file(source, router_record, "router")
    router_target = output / str(router_record["file"])
    install_file(router_source, router_target)

    manifest = dict(source_manifest)
    output_format = (
        "colorlm-kimi-k3-latent-macro-capsule-v4"
        if args.hybrid_decode
        else "colorlm-kimi-k3-latent-macro-capsule-v3"
    )
    output_layout = (
        "six-f16-three-mxfp4-decode-headerless-row-major-v1"
        if args.hybrid_decode
        else "three-f16-three-quantized-headerless-row-major-v1"
    )
    manifest.update(
        {
            "format": output_format,
            "runtime_layout": output_layout,
            "expert_quantization": "F16" if args.hybrid_decode else args.quantization,
            "decode_expert_quantization": args.quantization if args.hybrid_decode else None,
            "precision_policy": "prefill_f16_decode_mxfp4" if args.hybrid_decode else "uniform",
            "runtime_total_bytes": sum(int(item["bytes"]) for item in compiled_files.values()),
            "runtime_files": compiled_files,
            "runtime_router": {
                **router_record,
                "bytes": router_target.stat().st_size,
                "sha256": sha256_file(router_target),
            },
            "compiled_from": {
                "manifest": str(source_manifest_path),
                "manifest_sha256": sha256_file(source_manifest_path),
                "expert_quantization": args.quantization,
                "bridge_quantization": "F16",
            },
        }
    )
    manifest_path = output / "capsule.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    before = int(source_manifest["runtime_total_bytes"]) + int(router_record["bytes"])
    after = int(manifest["runtime_total_bytes"]) + router_target.stat().st_size
    print(f"{output_format.rsplit('-', 1)[-1]} 清单: {manifest_path}")
    print(
        f"运行权重: {before / 1024**2:.2f} -> {after / 1024**2:.2f} MiB "
        f"({(1.0 - after / before) * 100:.2f}% reduction)"
    )
    print(f"manifest_sha256={sha256_file(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
