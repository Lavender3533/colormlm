"""Create a new Neural Island package with independently fitted in/out bridges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ISLAND_FORMAT = "colorlm-neural-island-runtime-v1"
BLOCK_FORMAT = "colorlm-neural-block-runtime-v1"
BRIDGE_FORMAT = "colorlm-deep-activation-bridge-v1"
STABILITY_BRIDGE_FORMAT = "colorlm-activation-bridge-stability-v1"
INPUT_NAME = "colorlm.neural_block.transport_in.weight"
OUTPUT_NAME = "colorlm.neural_block.transport_out.weight"


class RepackError(RuntimeError):
    """A package contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RepackError(f"JSON读取失败: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RepackError(f"JSON根必须是对象: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def resolve_package(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def package_reference(path: Path) -> str:
    try:
        return os.fspath(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return os.fspath(path.resolve())


def load_weight(path: Path) -> np.ndarray:
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if matrix.shape != (2048, 2048) or not np.isfinite(matrix).all():
        raise RepackError(f"运行桥必须是有限值2048x2048矩阵: {path}: {matrix.shape}")
    return matrix


def encoded_f16(matrix: np.ndarray) -> bytes:
    return np.ascontiguousarray(matrix, dtype="<f2").tobytes(order="C")


def find_tensor(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    records = [item for item in manifest.get("tensors", []) if item.get("name") == name]
    if len(records) != 1:
        raise RepackError(f"运行包内{name}数量不是1")
    record = records[0]
    if (
        record.get("dtype") != "F16"
        or record.get("ggml_shape") != [2048, 2048]
        or int(record.get("bytes", -1)) != 2048 * 2048 * 2
        or int(record.get("offset", -1)) % 64 != 0
    ):
        raise RepackError(f"运行包内{name}布局不受支持")
    return record


def transport_contract(
    method: str,
    bridge_report: Path,
    report_sha256: str,
    input_path: Path,
    output_path: Path,
    input_sha256: str,
    output_sha256: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "bridge_report": os.fspath(bridge_report.resolve()),
        "bridge_report_sha256": report_sha256,
        "shape": [2048, 2048],
        "runtime_dtype": "F16",
        "input": {
            "source_file": input_path.name,
            "source_sha256": input_sha256,
            "contract": "donor_column = W_in @ colorlm_column",
        },
        "output": {
            "source_file": output_path.name,
            "source_sha256": output_sha256,
            "contract": "colorlm_delta_column = W_out @ donor_delta_column",
        },
    }


def patch_block(
    source_dir: Path,
    destination_dir: Path,
    input_payload: bytes,
    output_payload: bytes,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest_path = source_dir / "block.json"
    manifest = read_json(source_manifest_path)
    if manifest.get("format") != BLOCK_FORMAT or manifest.get("formal") is not True:
        raise RepackError(f"源神经块格式错误: {source_manifest_path}")
    weights_info = manifest.get("weights", {})
    source_weights = source_dir / str(weights_info.get("file", "weights.bin"))
    if not source_weights.is_file() or source_weights.stat().st_size != int(
        weights_info.get("bytes", -1)
    ):
        raise RepackError(f"源神经块权重缺失或尺寸错误: {source_weights}")
    input_record = find_tensor(manifest, INPUT_NAME)
    output_record = find_tensor(manifest, OUTPUT_NAME)
    if len(input_payload) != int(input_record["bytes"]) or len(output_payload) != int(
        output_record["bytes"]
    ):
        raise RepackError("F16桥payload尺寸错误")

    destination_dir.mkdir(parents=True, exist_ok=False)
    destination_weights = destination_dir / source_weights.name
    shutil.copyfile(source_weights, destination_weights)
    with destination_weights.open("r+b") as stream:
        stream.seek(int(input_record["offset"]))
        stream.write(input_payload)
        stream.seek(int(output_record["offset"]))
        stream.write(output_payload)
        stream.flush()

    input_record["sha256"] = hashlib.sha256(input_payload).hexdigest()
    output_record["sha256"] = hashlib.sha256(output_payload).hexdigest()
    weights_hash = sha256_file(destination_weights)
    manifest["name"] = str(manifest.get("name", "Neural-Block")).replace(
        "ColorLM-v16", "ColorLM-v18"
    ).replace("ColorLM-v17", "ColorLM-v18")
    manifest["weights"]["sha256"] = weights_hash
    manifest["interface"]["transport"] = contract
    manifest["transport"] = contract
    manifest_path = destination_dir / "block.json"
    write_json(manifest_path, manifest)
    record = {
        "source_layer": int(manifest.get("source", {}).get("layer", 47)),
        "layer_type": manifest["attention"]["type"],
        "package": package_reference(destination_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "weights_bytes": destination_weights.stat().st_size,
        "weights_sha256": weights_hash,
        "tensor_count": int(manifest["tensor_count"]),
    }
    return manifest, record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把通过门槛的深层激活桥装入v18神经岛副本")
    parser.add_argument("--source-island", type=Path, required=True)
    parser.add_argument("--bridge-report", type=Path, required=True)
    parser.add_argument("--input-weight", type=Path, required=True)
    parser.add_argument("--output-weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-rejected",
        action="store_true",
        help="只用于研究负对照；允许安装decision=reject的桥。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise RepackError(f"输出目录已存在，拒绝覆盖: {args.output}")
    island = read_json(args.source_island)
    if (
        island.get("format") != ISLAND_FORMAT
        or island.get("formal") is not True
        or island.get("source_layers") != [44, 45, 46, 47]
        or island.get("execution")
        != "contiguous-donor-coordinates-single-entry-single-exit"
    ):
        raise RepackError("源神经岛契约不受支持")
    report = read_json(args.bridge_report)
    report_format = report.get("format")
    if report_format not in (BRIDGE_FORMAT, STABILITY_BRIDGE_FORMAT):
        raise RepackError("激活桥报告格式错误")
    if report_format == BRIDGE_FORMAT:
        decision = report.get("promotion", {}).get("decision")
        accepted_decision = "candidate"
        method = str(report.get("method", {}).get("name", "activation_procrustes"))
    else:
        decision = report.get("decision")
        accepted_decision = "stable_candidate"
        method = str(report.get("method", {}).get("bridge_strategy", "activation_procrustes"))
    if decision != accepted_decision and not args.allow_rejected:
        raise RepackError(f"激活桥未通过晋级门，拒绝安装: decision={decision}")

    input_weight = load_weight(args.input_weight)
    output_weight = load_weight(args.output_weight)
    cycle = output_weight @ input_weight
    cycle_rmse = float(
        np.linalg.norm(cycle - np.eye(2048, dtype=np.float32)) / math.sqrt(cycle.size)
    )
    if cycle_rmse > 5e-5:
        raise RepackError(f"入口/出口桥不互逆: cycle_rmse={cycle_rmse}")
    input_hash = sha256_file(args.input_weight)
    output_hash = sha256_file(args.output_weight)
    expected_runtime = report.get("runtime", {})
    if (
        expected_runtime.get("input_weight", {}).get("sha256") != input_hash
        or expected_runtime.get("output_weight", {}).get("sha256") != output_hash
    ):
        raise RepackError("桥矩阵与拟合报告SHA-256不一致")
    report_hash = sha256_file(args.bridge_report)
    contract = transport_contract(
        method,
        args.bridge_report,
        report_hash,
        args.input_weight,
        args.output_weight,
        input_hash,
        output_hash,
    )
    input_payload = encoded_f16(input_weight)
    output_payload = encoded_f16(output_weight)

    args.output.mkdir(parents=True, exist_ok=False)
    layer_records: list[dict[str, Any]] = []
    for layer in island["layers"]:
        source_layer = int(layer["source_layer"])
        source_dir = resolve_package(str(layer["package"]))
        destination_dir = args.output / f"layer{source_layer}"
        _, record = patch_block(
            source_dir,
            destination_dir,
            input_payload,
            output_payload,
            contract,
        )
        if record["source_layer"] != source_layer:
            raise RepackError(f"源层号漂移: {record['source_layer']} vs {source_layer}")
        layer_records.append(record)
        print(
            f"L{source_layer}: {record['weights_bytes'] / 1024**3:.3f} GiB, "
            f"sha256={record['weights_sha256'][:12]}...",
            flush=True,
        )

    output_island = dict(island)
    output_island["name"] = "ColorLM-v18-Coder-Neural-Island-Activation-Bridge"
    output_island["layers"] = layer_records
    output_island["total_weight_bytes"] = sum(
        int(record["weights_bytes"]) for record in layer_records
    )
    output_island.pop("transport_sha256", None)
    output_island["transport"] = contract
    output_island["activation_bridge"] = {
        "report": os.fspath(args.bridge_report.resolve()),
        "report_sha256": report_hash,
        "decision": decision,
        "cycle_rmse_f32": cycle_rmse,
    }
    output_manifest = args.output / "island.json"
    write_json(output_manifest, output_island)
    print(f"v18神经岛: {output_manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepackError as error:
        raise SystemExit(f"error: {error}") from error
