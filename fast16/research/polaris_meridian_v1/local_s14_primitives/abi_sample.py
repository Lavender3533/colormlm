"""可选的真实 L42/E0 ABI 样本校验；不参与默认运行时或 CI。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .mxfp4 import FP4_E2M1_VALUES, decode_mxfp4, decode_ue8m0, fp4_linear


REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
EXPECTED_EXPERT_REGRESSION = {
    "l2": 74.81494140625,
    "mean": -0.02070457488298416,
    "maxabs": 4.613373756408691,
    "f32_le_sha256": "118710bbf6d18bae928a4b0fbc71b9b245a9b9f19504a24bd2bf5d483f32d989",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _product(shape: list[int]) -> int:
    result = 1
    for value in shape:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"非法 shape: {shape}")
        result *= value
    return result


def inspect_external_abi_sample(
    manifest_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """校验外部 ABI manifest，并从每个 expert weight/scale 对解码 32 个值。

    safetensors header 的 ``I8`` 在此合同中只是物理字节容器：每个字节的低 nibble
    是偶数 K，高 nibble 是奇数 K。函数不会因 dtype 名不是 F4 而拒绝它。
    """

    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8", errors="strict") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "polaris-deepseek-abi-sample-v1":
        raise ValueError("ABI 样本 format 不受支持")
    if manifest.get("repo") != REPO or manifest.get("revision") != REVISION:
        raise ValueError("ABI 样本来源 repo/revision 漂移")
    if manifest.get("layer") != 42 or manifest.get("expert_id") != 0:
        raise ValueError("当前 sanity 只接受冻结的 L42/E0 样本")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 8:
        raise ValueError("ABI 样本必须恰好含 8 条 Range")
    if sum(entry.get("bytes", -1) for entry in entries) != manifest.get("payload_bytes"):
        raise ValueError("payload_bytes 与 entry 字节和不一致")

    by_name: dict[str, dict[str, Any]] = {}
    checked_hashes = 0
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str) or name in by_name:
            raise ValueError("entry name 缺失或重复")
        path = Path(entry.get("path", ""))
        size = path.stat().st_size
        element_bytes = {"I8": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4}.get(entry.get("dtype"))
        if element_bytes is None:
            raise ValueError(f"{name} 含未知物理 dtype")
        if size != entry.get("bytes") or _product(entry.get("shape", [])) * element_bytes != size:
            raise ValueError(f"{name} 的文件长度/shape 与 manifest 不一致")
        if entry.get("end") - entry.get("start") + 1 != size:
            raise ValueError(f"{name} 的 Range 边界与文件长度不一致")
        if verify_hashes:
            actual = _sha256(path)
            if actual != entry.get("sha256_tofu"):
                raise ValueError(f"{name} 的 TOFU SHA-256 不匹配")
            checked_hashes += 1
        by_name[name] = entry

    pairs = []
    for component in ("w1", "w2", "w3"):
        prefix = f"layers.42.ffn.experts.0.{component}"
        weight_entry = by_name.get(f"{prefix}.weight")
        scale_entry = by_name.get(f"{prefix}.scale")
        if weight_entry is None or scale_entry is None:
            raise ValueError(f"缺少 {component} weight/scale")
        if weight_entry.get("dtype") != "I8" or scale_entry.get("dtype") != "F8_E8M0":
            raise ValueError(f"{component} 的物理 dtype 合同不匹配")
        rows, packed_k = weight_entry["shape"]
        scale_rows, scale_groups = scale_entry["shape"]
        logical_k = packed_k * 2
        if scale_rows != rows or scale_groups * 32 != logical_k:
            raise ValueError(f"{component} 的 packed/scale shape 不匹配")

        weight_path = Path(weight_entry["path"])
        scale_path = Path(scale_entry["path"])
        with weight_path.open("rb") as handle:
            packed_slice = handle.read(16)
        with scale_path.open("rb") as handle:
            scale_slice = handle.read(1)
        if len(packed_slice) != 16 or len(scale_slice) != 1:
            raise ValueError(f"{component} 的 sanity slice 被截断")
        packed_tensor = torch.tensor(list(packed_slice), dtype=torch.uint8).reshape(1, 16)
        scale_tensor = torch.tensor(list(scale_slice), dtype=torch.uint8).reshape(1, 1)
        decoded = decode_mxfp4(packed_tensor, scale_tensor)
        if not bool(torch.isfinite(decoded).all().item()):
            raise ValueError(f"{component} 的首个 MXFP4 group 解码为非有限值")
        scale_value = float(decode_ue8m0(scale_tensor).item())
        pairs.append(
            {
                "component": component,
                "packed_header_dtype": "I8_as_two_E2M1_nibbles",
                "logical_shape": [rows, logical_k],
                "scale_shape": [scale_rows, scale_groups],
                "scale_code": scale_slice[0],
                "scale_value": scale_value,
                "first_8_values": [float(value) for value in decoded[0, :8]],
                "packed_slice_sha256": hashlib.sha256(packed_slice).hexdigest(),
            }
        )

    return {
        "format": "polaris-local-s14-external-abi-sanity-v1",
        "repo": REPO,
        "revision": REVISION,
        "layer": 42,
        "expert_id": 0,
        "manifest_payload_bytes": manifest["payload_bytes"],
        "range_entries": len(entries),
        "tofu_hashes_checked": checked_hashes,
        "nibble_order": "low_nibble_even_k_then_high_nibble_odd_k",
        "expert_slices": pairs,
        "router_entries_integrity_checked": True,
        "evidence_status": "real_bytes_format_sanity_not_model_capability",
        "weights_copied_into_repo": False,
        "claim_limit": "该报告只验证固定样本的字节、布局和小切片解码，不证明 S14 forward、速度或质量。",
    }


def run_external_expert_regression(
    manifest_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """在显式 opt-in 时运行 L42/E0 三矩阵 SwiGLU 真实字节回归。

    NumPy 完整展开路径用于复现冻结的 little-endian F32 指纹；同一输入还会经过本模块
    的 PyTorch 分块 ``fp4_linear``，两者以数值容差对齐。该函数不会被默认 CI 调用。
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - 仅可选外部回归路径
        raise RuntimeError("真实 expert 回归需要 NumPy") from exc

    sanity = inspect_external_abi_sample(manifest_path, verify_hashes=verify_hashes)
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8", errors="strict") as handle:
        manifest = json.load(handle)
    entries = {entry["name"]: entry for entry in manifest["entries"]}
    lut = np.asarray(FP4_E2M1_VALUES, dtype=np.float32)

    def entry_pair(component: str) -> tuple[dict[str, Any], dict[str, Any]]:
        prefix = f"layers.42.ffn.experts.0.{component}"
        return entries[f"{prefix}.weight"], entries[f"{prefix}.scale"]

    def numpy_weight(component: str):
        weight_entry, scale_entry = entry_pair(component)
        packed = np.memmap(
            weight_entry["path"], dtype=np.uint8, mode="r", shape=tuple(weight_entry["shape"])
        )
        scales = np.memmap(
            scale_entry["path"], dtype=np.uint8, mode="r", shape=tuple(scale_entry["shape"])
        )
        decoded = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
        decoded[:, 0::2] = lut[packed & np.uint8(0x0F)]
        decoded[:, 1::2] = lut[packed >> np.uint8(4)]
        decoded = decoded.reshape(scales.shape[0], scales.shape[1], 32)
        decoded *= np.exp2(scales.astype(np.float32) - np.float32(127.0))[:, :, None]
        return decoded.reshape(packed.shape[0], packed.shape[1] * 2)

    x_np = np.sin(np.arange(4096, dtype=np.float32) * np.float32(0.013)).astype(np.float32)
    gate_np = numpy_weight("w1") @ x_np
    up_np = numpy_weight("w3") @ x_np
    up_np = np.clip(up_np, np.float32(-10.0), np.float32(10.0))
    gate_np = np.minimum(gate_np, np.float32(10.0))
    hidden_np = (gate_np / (np.float32(1.0) + np.exp(-gate_np))) * up_np
    output_np = numpy_weight("w2") @ hidden_np.astype(np.float32)
    output_np = np.asarray(output_np, dtype=np.float32)
    numpy_stats = {
        "l2": float(np.linalg.norm(output_np)),
        "mean": float(output_np.mean()),
        "maxabs": float(np.max(np.abs(output_np))),
        "f32_le_sha256": hashlib.sha256(np.asarray(output_np, dtype="<f4").tobytes()).hexdigest(),
    }
    if numpy_stats != EXPECTED_EXPERT_REGRESSION:
        raise AssertionError(
            f"真实 L42/E0 NumPy 指纹漂移: expected={EXPECTED_EXPERT_REGRESSION}, actual={numpy_stats}"
        )

    def torch_pair(component: str) -> tuple[torch.Tensor, torch.Tensor]:
        weight_entry, scale_entry = entry_pair(component)
        weight_bytes = bytearray(Path(weight_entry["path"]).read_bytes())
        scale_bytes = bytearray(Path(scale_entry["path"]).read_bytes())
        weight = torch.frombuffer(weight_bytes, dtype=torch.uint8).reshape(weight_entry["shape"])
        scale = torch.frombuffer(scale_bytes, dtype=torch.uint8).reshape(scale_entry["shape"])
        return weight, scale

    x_torch = torch.from_numpy(x_np.copy())
    w1, s1 = torch_pair("w1")
    w3, s3 = torch_pair("w3")
    gate_torch = fp4_linear(x_torch, w1, s1, output_chunk_size=128).clamp(max=10.0)
    up_torch = fp4_linear(x_torch, w3, s3, output_chunk_size=128).clamp(-10.0, 10.0)
    hidden_torch = torch.nn.functional.silu(gate_torch) * up_torch
    w2, s2 = torch_pair("w2")
    output_torch = fp4_linear(hidden_torch, w2, s2, output_chunk_size=128)
    delta = output_torch.numpy() - output_np
    parity = {
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "rtol": 1e-5,
        "atol": 2e-5,
        "allclose": bool(np.allclose(output_torch.numpy(), output_np, rtol=1e-5, atol=2e-5)),
    }
    if not parity["allclose"]:
        raise AssertionError(f"PyTorch 分块 FP4 linear 未对齐真实样本基准: {parity}")

    return {
        **sanity,
        "format": "polaris-local-s14-external-expert-regression-v1",
        "fixture": "x=sin(arange(4096)*0.013), swiglu_limit=10.0",
        "numpy_frozen_baseline": numpy_stats,
        "pytorch_chunked_linear_parity": parity,
        "evidence_status": "real_bytes_numerical_regression_not_model_capability",
        "claim_limit": "该回归只覆盖 L42/E0 单 expert 的三矩阵数值，不包含 router、共享 expert、attention 或 S14 token。",
    }
