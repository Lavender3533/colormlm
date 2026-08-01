"""Build the runtime contract for the contiguous Qwen3-Coder-Next L44-L47 island."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISLAND = ROOT / "fast16" / "research" / "v17_coder_island"
DEFAULT_L47 = (
    ROOT
    / "fast16"
    / "research"
    / "neural_blocks"
    / "qwen3_coder_next_l47"
    / "q4_0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成v17连续Coder神经岛运行契约")
    parser.add_argument("--island", type=Path, default=DEFAULT_ISLAND)
    parser.add_argument(
        "--runtime",
        type=Path,
        help="运行包目录；默认使用 <island>/runtime",
    )
    parser.add_argument("--layer47", type=Path, default=DEFAULT_L47)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path) -> str:
    return os.fspath(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def load_block(directory: Path, expected_layer: int) -> tuple[dict, dict]:
    manifest_path = directory / "block.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"缺少神经块manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    source_layer = int(source.get("layer", 47 if expected_layer == 47 else -1))
    if source_layer != expected_layer:
        raise RuntimeError(f"神经块层号不匹配: {source_layer} vs {expected_layer}")
    expected_type = "full_attention" if expected_layer == 47 else "gated_deltanet"
    if manifest["attention"]["type"] != expected_type:
        raise RuntimeError(f"L{expected_layer}层类型不匹配")
    weights = directory / manifest["weights"]["file"]
    if not weights.is_file() or weights.stat().st_size != int(manifest["weights"]["bytes"]):
        raise RuntimeError(f"神经块权重尺寸不匹配: {weights}")
    record = {
        "source_layer": expected_layer,
        "layer_type": expected_type,
        "package": relative_to_root(directory),
        "manifest_sha256": sha256_file(manifest_path),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": manifest["weights"]["sha256"],
        "tensor_count": int(manifest["tensor_count"]),
    }
    return manifest, record


def main() -> int:
    args = parse_args()
    runtime = args.runtime if args.runtime is not None else args.island / "runtime"
    packages = {layer: runtime / f"layer{layer}" for layer in range(44, 47)}
    packages[47] = args.layer47

    manifests: list[dict] = []
    records: list[dict] = []
    for layer in range(44, 48):
        manifest, record = load_block(packages[layer], layer)
        manifests.append(manifest)
        records.append(record)

    transport_hashes = {
        manifest["transport"]["source_to_target_sha256"] for manifest in manifests
    }
    target_sites = {int(manifest["target"]["site"]) for manifest in manifests}
    if len(transport_hashes) != 1 or target_sites != {35}:
        raise RuntimeError("四层岛的坐标桥或目标站点不一致")

    contract = {
        "format": "colorlm-neural-island-runtime-v1",
        "name": "ColorLM-v17-Coder-Neural-Island-L44-L47",
        "formal": True,
        "target_site": 35,
        "source_architecture": "Qwen3NextForCausalLM",
        "source_layers": [44, 45, 46, 47],
        "execution": "contiguous-donor-coordinates-single-entry-single-exit",
        "state": {
            "recurrent_layers": [44, 45, 46],
            "attention_layers": [47],
            "memory": "island-private-bounded-state-required",
        },
        "transport_sha256": next(iter(transport_hashes)),
        "total_weight_bytes": sum(record["weights_bytes"] for record in records),
        "layers": records,
    }
    output = runtime / "island.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"神经岛: {output}")
    print(f"运行权重: {contract['total_weight_bytes'] / 1024**3:.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
