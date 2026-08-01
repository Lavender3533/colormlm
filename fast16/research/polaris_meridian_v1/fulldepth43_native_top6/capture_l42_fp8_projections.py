"""导出 L42 标准 packed-FP8 attention 投影的冻结 exact fixture。

该工具只读本机已校验的 DeepSeek-V4 Range cache，不下载权重。输出目录
必须尚不存在；二进制仅用于本机 GPU exact 门，不进入 Git 权重资产。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from fast16.research.polaris_meridian_v1.l42_real_reference.l42_reference import (
    DEFAULT_ASSET_ROOT,
    FROZEN_OUTPUT_SHA256,
    LAYER,
    REPO,
    REVISION,
    _InlineForward,
    _sha256,
    validate_assets,
)


FORMAT = "polaris-l42-packed-fp8-projection-fixtures-v1"
CATALOG_NAME = "fulldepth43_native_top6_catalog.json"
PROJECTION_ORDER = (
    "layers.42.attn.wq_a",
    "layers.42.attn.wkv",
    "layers.42.attn.wq_b",
    "layers.42.attn.indexer.wq_b",
    "layers.42.attn.wo_b",
)
FROZEN_PROJECTION_SHA256 = {
    "layers.42.attn.wq_a": (
        "47156935b19ca5483f0e92d2284eaa6a9417686978dc4b41ca893ee162f37577",
        "76469fd163f5db49de956eff9b29087afa4caa97d566be80bab9d9119facb0b8",
    ),
    "layers.42.attn.wkv": (
        "47156935b19ca5483f0e92d2284eaa6a9417686978dc4b41ca893ee162f37577",
        "3cc7f8f4264c6448dd32f9044c0d001107f06d57209a91a80fa56bdda59dd541",
    ),
    "layers.42.attn.wq_b": (
        "4ceb243521589b40b930c63b03da362163dfdc7fe12c0b76397100ec4b4c58e1",
        "284391a5a45d6a5367060ecd444a21770e69fa7949455bea6823317f4fb43c04",
    ),
    "layers.42.attn.indexer.wq_b": (
        "4ceb243521589b40b930c63b03da362163dfdc7fe12c0b76397100ec4b4c58e1",
        "d9adda7639665267be4fac36e2a74755bb5d730a4a2a8734695198fc4f331501",
    ),
    "layers.42.attn.wo_b": (
        "94b3f7fd24ee36b8553ed513d1986ef49162c053bd6dbf62f98b9579e20ea3f0",
        "84ce63ca9233b07bea99741f9982accac17bc65025b0098b7017acd7dab6db10",
    ),
}


def _f32_payload(array: np.ndarray) -> tuple[np.ndarray, bytes, str]:
    payload = np.ascontiguousarray(array, dtype="<f4")
    encoded = payload.tobytes()
    return payload, encoded, hashlib.sha256(encoded).hexdigest()


class _ProjectionCapture(_InlineForward):
    def __init__(self, asset_root: Path, output_dir: Path) -> None:
        bundle = validate_assets(asset_root, verify_hashes=True)
        super().__init__(bundle, capture_dir=None)
        self.asset_root = asset_root.resolve()
        self.output_dir = output_dir.resolve()
        self.records: dict[str, dict[str, Any]] = {}
        self.inputs: dict[str, np.ndarray] = {}

    @staticmethod
    def _file_stem(prefix: str) -> str:
        if not prefix.startswith("layers.42.attn."):
            raise ValueError(f"非 L42 attention projection: {prefix}")
        return prefix.removeprefix("layers.42.attn.").replace(".", "-")

    def _record(self, prefix: str, activation: np.ndarray, output: np.ndarray) -> None:
        if prefix in self.records:
            raise AssertionError(f"projection 被重复采集: {prefix}")
        input_array, input_bytes, input_sha = _f32_payload(activation)
        output_array, output_bytes, output_sha = _f32_payload(output)
        expected = FROZEN_PROJECTION_SHA256[prefix]
        if (input_sha, output_sha) != expected:
            raise AssertionError(
                f"{prefix} fixture SHA 漂移: expected={expected}, actual={(input_sha, output_sha)}"
            )
        stem = self._file_stem(prefix)
        input_path = self.output_dir / f"{stem}.input.f32le.bin"
        output_path = self.output_dir / f"{stem}.output.bf16-f32le.bin"
        input_path.write_bytes(input_bytes)
        output_path.write_bytes(output_bytes)
        weight_name = prefix + ".weight"
        scale_name = prefix + ".scale"
        weight_entry = self.bundle.entries[weight_name]
        scale_entry = self.bundle.entries[scale_name]
        self.inputs[prefix] = input_array.copy()
        self.records[prefix] = {
            "projection": prefix,
            "n": int(output_array.shape[-1]),
            "k": int(input_array.shape[-1]),
            "activation_contract": "cpu_e4m3fn_quant_dequant_f32",
            "output_rounding": "bf16_rne_then_f32_le",
            "input": {
                "file": input_path.name,
                "shape": list(input_array.shape),
                "bytes": len(input_bytes),
                "sha256": input_sha,
            },
            "output": {
                "file": output_path.name,
                "shape": list(output_array.shape),
                "bytes": len(output_bytes),
                "sha256": output_sha,
            },
            "weight": {
                "tensor": weight_name,
                "shape": list(self.bundle.specs[weight_name][1]),
                "bytes": int(weight_entry["bytes"]),
                "sha256": weight_entry["sha256"],
            },
            "scale": {
                "tensor": scale_name,
                "shape": list(self.bundle.specs[scale_name][1]),
                "bytes": int(scale_entry["bytes"]),
                "sha256": scale_entry["sha256"],
            },
        }

    def _linear_fp8(self, array: np.ndarray, prefix: str) -> np.ndarray:
        activation = self._activation_quant(array)
        output = super()._linear_fp8(array, prefix)
        if prefix in FROZEN_PROJECTION_SHA256 and prefix != "layers.42.attn.indexer.wq_b":
            self._record(prefix, activation, output)
        return output

    def run_capture(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=False)
        layer_report = super().run()

        # L42 单token参考的 sparse_attention 不需要 indexer，但 FullDepth43
        # ratio-4 路径会执行它。indexer 与 wq_b 共享同一个已量化 qr 输入。
        qr = self.inputs["layers.42.attn.wq_b"]
        weight = self._weight_fp8("layers.42.attn.indexer.wq_b")
        indexer_output = self._bf16_numpy(qr.reshape(-1, qr.shape[-1]) @ weight.T)
        indexer_output = indexer_output.reshape(*qr.shape[:-1], indexer_output.shape[-1])
        del weight
        self._record("layers.42.attn.indexer.wq_b", qr, indexer_output)

        if tuple(self.records) != (
            "layers.42.attn.wq_a",
            "layers.42.attn.wq_b",
            "layers.42.attn.wkv",
            "layers.42.attn.wo_b",
            "layers.42.attn.indexer.wq_b",
        ):
            raise AssertionError("L42 projection 实际采集顺序/覆盖漂移")
        if layer_report["layer_output"]["f32_le_sha256"] != FROZEN_OUTPUT_SHA256["layer_output"]:
            raise AssertionError("采集运行的完整 L42 输出漂移")

        catalog = self.asset_root / CATALOG_NAME
        if not catalog.is_file():
            raise FileNotFoundError(f"缺少 FullDepth43 catalog: {catalog}")
        manifest = {
            "format": FORMAT,
            "repo": REPO,
            "revision": REVISION,
            "layer": LAYER,
            "catalog_sha256": _sha256(catalog),
            "projection_count": len(PROJECTION_ORDER),
            "projections": [self.records[name] for name in PROJECTION_ORDER],
            "layer_output_sha256": layer_report["layer_output"]["f32_le_sha256"],
            "asset_integrity": {
                "hashes_checked": self.bundle.hashes_checked,
                "payload_files": len(self.bundle.entries),
                "payload_bytes": self.bundle.payload_bytes,
                "manifest_sha256": {
                    key: _sha256(path) for key, path in sorted(self.bundle.manifest_paths.items())
                },
            },
            "claim_limit": (
                "这些fixture只冻结L42标准packed-FP8投影；不证明wo_a grouped语义、"
                "完整attention、43层GPU token或端到端速度。"
            ),
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest


def capture_l42_fp8_projections(
    output_dir: str | Path,
    *,
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
) -> dict[str, Any]:
    return _ProjectionCapture(Path(asset_root), Path(output_dir)).run_capture()


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    args = parser.parse_args()
    report = capture_l42_fp8_projections(args.output_dir, asset_root=args.asset_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
