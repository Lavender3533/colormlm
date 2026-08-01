"""把六条已逐位闭合的 attention 投影回放进真实 L42 完整前向。

标准 packed-FP8 与 grouped ``wo_a`` 的输出必须先由 RX 5700 XT exact
套件证明命中同一冻结 SHA。本脚本随后在每个真实调用点重新计算输入 SHA，
只在输入身份完全一致时替换投影输出，并验证完整 L42 最终输出不变。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fast16.research.polaris_meridian_v1.l42_real_reference.l42_reference import (
    DEFAULT_ASSET_ROOT,
    FROZEN_OUTPUT_SHA256,
    REPO,
    REVISION,
    _InlineForward,
    validate_assets,
)


STANDARD_MANIFEST_SHA256 = "3b4610c57ec4c3abecde70167c4ba59e5a43c345efb07322a70efbea1e15898e"
WO_A_CAPTURE_MANIFEST_SHA256 = "f91fc9fd40a95ee04984376b7a80544db8decc88bc5f139b9a6fd382fa4bb43d"
L42_OUTPUT_SHA256 = "853b8b947a3f7a275cf748d7e97a311ebb22323cd0c2f3e5e973f27b04388895"
STANDARD_ORDER = (
    "layers.42.attn.wq_a",
    "layers.42.attn.wkv",
    "layers.42.attn.wq_b",
    "layers.42.attn.indexer.wq_b",
    "layers.42.attn.wo_b",
)
EXECUTED_ORDER = (
    "layers.42.attn.wq_a",
    "layers.42.attn.wq_b",
    "layers.42.attn.wkv",
    "layers.42.attn.wo_a",
    "layers.42.attn.wo_b",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected_sha256:
        raise ValueError(f"manifest SHA 漂移: {path} expected={expected_sha256} actual={actual}")
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise TypeError(f"manifest 顶层不是 object: {path}")
    return document


def _safe_payload(root: Path, filename: str, expected_bytes: int, expected_sha256: str) -> np.ndarray:
    relative = Path(filename)
    if len(relative.parts) != 1 or relative.suffix != ".bin":
        raise ValueError(f"fixture 文件名越界: {filename}")
    resolved = (root / relative).resolve(strict=True)
    if resolved.parent != root:
        raise ValueError(f"fixture 文件逃逸目录: {resolved}")
    payload = resolved.read_bytes()
    if len(payload) != expected_bytes or _sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"fixture bytes/SHA 漂移: {resolved}")
    values = np.frombuffer(payload, dtype="<f4").copy()
    if not np.isfinite(values).all():
        raise ValueError(f"fixture 含非有限值: {resolved}")
    return values


@dataclass(frozen=True)
class ProjectionReplay:
    name: str
    input: np.ndarray
    output: np.ndarray
    input_sha256: str
    output_sha256: str


def _standard_replays(root: Path) -> dict[str, ProjectionReplay]:
    document = _load_json(root / "manifest.json", STANDARD_MANIFEST_SHA256)
    if (
        document.get("format") != "polaris-l42-packed-fp8-projection-fixtures-v1"
        or document.get("repo") != REPO
        or document.get("revision") != REVISION
        or document.get("layer") != 42
        or document.get("projection_count") != 5
        or document.get("layer_output_sha256") != L42_OUTPUT_SHA256
    ):
        raise ValueError("标准 FP8 fixture manifest 身份漂移")
    entries = document.get("projections")
    if not isinstance(entries, list) or len(entries) != len(STANDARD_ORDER):
        raise ValueError("标准 FP8 fixture 数量漂移")
    indexed = {entry.get("projection"): entry for entry in entries if isinstance(entry, dict)}
    if tuple(indexed) != STANDARD_ORDER:
        raise ValueError(f"标准 FP8 fixture 顺序/名称漂移: {tuple(indexed)}")

    result: dict[str, ProjectionReplay] = {}
    for name in STANDARD_ORDER:
        entry = indexed[name]
        input_spec = entry.get("input")
        output_spec = entry.get("output")
        if not isinstance(input_spec, dict) or not isinstance(output_spec, dict):
            raise ValueError(f"{name} 缺少 input/output 合同")
        input_values = _safe_payload(
            root,
            str(input_spec.get("file")),
            int(input_spec.get("bytes", -1)),
            str(input_spec.get("sha256")),
        )
        output_values = _safe_payload(
            root,
            str(output_spec.get("file")),
            int(output_spec.get("bytes", -1)),
            str(output_spec.get("sha256")),
        )
        expected_input = int(entry.get("k", -1))
        expected_output = int(entry.get("n", -1))
        if input_values.size != expected_input or output_values.size != expected_output:
            raise ValueError(f"{name} fixture 元素数漂移")
        result[name] = ProjectionReplay(
            name=name,
            input=input_values,
            output=output_values,
            input_sha256=str(input_spec["sha256"]),
            output_sha256=str(output_spec["sha256"]),
        )
    return result


def _wo_a_replay(root: Path) -> ProjectionReplay:
    document = _load_json(root / "capture_manifest.json", WO_A_CAPTURE_MANIFEST_SHA256)
    if (
        document.get("format") != "polaris-l42-real-vulkan-input-capture-v1"
        or document.get("repo") != REPO
        or document.get("revision") != REVISION
        or document.get("layer") != 42
        or len(document.get("inputs", [])) != 5
    ):
        raise ValueError("wo_a capture manifest 身份漂移")
    indexed = {
        entry.get("name"): entry
        for entry in document["inputs"]
        if isinstance(entry, dict)
    }
    input_spec = indexed.get("wo_a_grouped_input_bf16")
    output_spec = indexed.get("wo_a_grouped_output_bf16")
    if not isinstance(input_spec, dict) or not isinstance(output_spec, dict):
        raise ValueError("wo_a capture 缺少 grouped input/output")
    input_values = _safe_payload(
        root,
        str(input_spec.get("file")),
        int(input_spec.get("bytes", -1)),
        str(input_spec.get("f32_le_sha256")),
    )
    output_values = _safe_payload(
        root,
        str(output_spec.get("file")),
        int(output_spec.get("bytes", -1)),
        str(output_spec.get("f32_le_sha256")),
    )
    if input_values.size != 8 * 4096 or output_values.size != 8192:
        raise ValueError("wo_a grouped fixture 元素数漂移")
    return ProjectionReplay(
        name="layers.42.attn.wo_a",
        input=input_values,
        output=output_values,
        input_sha256=str(input_spec["f32_le_sha256"]),
        output_sha256=str(output_spec["f32_le_sha256"]),
    )


class _AttentionReplay(_InlineForward):
    def __init__(self, asset_root: Path, records: dict[str, ProjectionReplay]) -> None:
        super().__init__(validate_assets(asset_root, verify_hashes=True))
        self.records = records
        self.executed: list[str] = []

    def _replace(self, prefix: str, activation: np.ndarray) -> np.ndarray:
        record = self.records[prefix]
        flat = np.ascontiguousarray(activation, dtype="<f4").reshape(-1)
        actual_sha256 = _sha256_bytes(flat.tobytes())
        if actual_sha256 != record.input_sha256 or not np.array_equal(
            flat.view(np.uint32), record.input.view(np.uint32)
        ):
            raise AssertionError(
                f"{prefix} 回放输入漂移: expected={record.input_sha256} actual={actual_sha256}"
            )
        self.executed.append(prefix)
        leading_shape = (
            activation.shape[:2]
            if prefix == "layers.42.attn.wo_a"
            else activation.shape[:-1]
        )
        return record.output.copy().reshape(*leading_shape, record.output.size)

    def _linear_fp8(self, array: np.ndarray, prefix: str) -> np.ndarray:
        if prefix in self.records and prefix != "layers.42.attn.wo_a":
            return self._replace(prefix, self._activation_quant(array))
        return super()._linear_fp8(array, prefix)

    def _grouped_wo_a(self, array: np.ndarray, prefix: str) -> np.ndarray:
        if prefix != "layers.42.attn.wo_a":
            return super()._grouped_wo_a(array, prefix)
        return self._replace(prefix, array)


def verify_l42_attention_replay(
    standard_fixture_dir: str | Path,
    wo_a_fixture_dir: str | Path,
    *,
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
) -> dict[str, Any]:
    standard_root = Path(standard_fixture_dir).resolve(strict=True)
    wo_a_root = Path(wo_a_fixture_dir).resolve(strict=True)
    records = _standard_replays(standard_root)
    records["layers.42.attn.wo_a"] = _wo_a_replay(wo_a_root)
    runner = _AttentionReplay(Path(asset_root), records)
    layer_report = runner.run()
    output_sha256 = layer_report["layer_output"]["f32_le_sha256"]
    if tuple(runner.executed) != EXECUTED_ORDER:
        raise AssertionError(f"L42 attention 回放执行顺序漂移: {tuple(runner.executed)}")
    if output_sha256 != L42_OUTPUT_SHA256 or output_sha256 != FROZEN_OUTPUT_SHA256["layer_output"]:
        raise AssertionError(f"L42 完整输出漂移: {output_sha256}")
    return {
        "format": "polaris-l42-six-attention-projection-replay-v1",
        "status": "complete",
        "repo": REPO,
        "revision": REVISION,
        "executed_projection_count": len(runner.executed),
        "executed_projections": runner.executed,
        "standalone_verified_projection": "layers.42.attn.indexer.wq_b",
        "projection_output_sha256": {
            name: records[name].output_sha256 for name in (*STANDARD_ORDER, "layers.42.attn.wo_a")
        },
        "layer_output_sha256": output_sha256,
        "claim_limit": (
            "五条实际 L42 position0 attention 投影回放后完整层输出精确不变；"
            "indexer.wq_b 由同一 GPU suite 独立逐位闭合，但该 position0 参考不调用 indexer。"
        ),
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-fixture-dir", type=Path, required=True)
    parser.add_argument("--wo-a-fixture-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    args = parser.parse_args()
    report = verify_l42_attention_replay(
        args.standard_fixture_dir,
        args.wo_a_fixture_dir,
        asset_root=args.asset_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
