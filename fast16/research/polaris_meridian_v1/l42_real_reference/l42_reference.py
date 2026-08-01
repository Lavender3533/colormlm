"""DeepSeek-V4 L42 真实单层、单 token inline forward 参考。

该实现只读已有 Range cache，不含网络下载或缺失资产降级路径。
它不是 S14/43 层首 token，也不代表模型质量或 GPU 速度。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fast16.research.polaris_meridian_v1.local_s14_primitives.attention import sparse_attention
from fast16.research.polaris_meridian_v1.local_s14_primitives.hc import hc_post, hc_pre


REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
LAYER = 42
DEFAULT_ASSET_ROOT = Path("D:/models/Polaris-S14")
ROUTE_IDS = [126, 12, 205, 149, 227, 174]
ROUTE_WEIGHTS = [
    0.2747795581817627,
    0.2491425722837448,
    0.24045628309249878,
    0.24615329504013062,
    0.25386279821395874,
    0.23560553789138794,
]
FROZEN_OUTPUT_SHA256 = {
    "attention_branch": "84ce63ca9233b07bea99741f9982accac17bc65025b0098b7017acd7dab6db10",
    "post_attention_state": "1a04e9b1634409af115c39d12ba72aa031a836fe7ea825778417ace3eeef79f1",
    "ffn_input": "7e2d3167e3782eca8d762c3cc92d53bb9d64a65c7b18d37d16797ff39f611ad4",
    "moe_branch": "7ecbb213a3e025fa76cd91a21f295e9234437b301848431a7342ce73c5d62841",
    "layer_output": "853b8b947a3f7a275cf748d7e97a311ebb22323cd0c2f3e5e973f27b04388895",
}

_FP4_LUT = np.asarray(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)

_FP8_VALIDATION_LOCK = threading.Lock()
_FP8_VALIDATION_CACHE: set[tuple[tuple[str, int, int], tuple[str, int, int]]] = set()


def _fp8_lut() -> np.ndarray:
    raw = np.arange(256, dtype=np.uint8)
    sign = (raw >> 7) != 0
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    value = np.where(
        exponent == 0,
        mantissa.astype(np.float32) / 8 * 2**-6,
        np.ldexp(1 + mantissa.astype(np.float32) / 8, exponent.astype(np.int32) - 7),
    )
    value = np.where(sign, -value, value).astype(np.float32)
    value[(exponent == 15) & (mantissa == 7)] = np.nan
    return value


_FP8_LUT = _fp8_lut()


def _base_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    return {
        "layers.42.attn.attn_sink": ("F32", (64,)),
        "layers.42.attn.compressor.ape": ("F32", (4, 1024)),
        "layers.42.attn.compressor.norm.weight": ("BF16", (512,)),
        "layers.42.attn.compressor.wgate.weight": ("BF16", (1024, 4096)),
        "layers.42.attn.compressor.wkv.weight": ("BF16", (1024, 4096)),
        "layers.42.attn.indexer.compressor.ape": ("F32", (4, 256)),
        "layers.42.attn.indexer.compressor.norm.weight": ("BF16", (128,)),
        "layers.42.attn.indexer.compressor.wgate.weight": ("BF16", (256, 4096)),
        "layers.42.attn.indexer.compressor.wkv.weight": ("BF16", (256, 4096)),
        "layers.42.attn.indexer.weights_proj.weight": ("BF16", (64, 4096)),
        "layers.42.attn.indexer.wq_b.scale": ("F8_E8M0", (64, 8)),
        "layers.42.attn.indexer.wq_b.weight": ("F8_E4M3", (8192, 1024)),
        "layers.42.attn.kv_norm.weight": ("BF16", (512,)),
        "layers.42.attn.q_norm.weight": ("BF16", (1024,)),
        "layers.42.attn.wkv.scale": ("F8_E8M0", (4, 32)),
        "layers.42.attn.wkv.weight": ("F8_E4M3", (512, 4096)),
        "layers.42.attn.wo_a.scale": ("F8_E8M0", (64, 32)),
        "layers.42.attn.wo_a.weight": ("F8_E4M3", (8192, 4096)),
        "layers.42.attn.wo_b.scale": ("F8_E8M0", (32, 64)),
        "layers.42.attn.wo_b.weight": ("F8_E4M3", (4096, 8192)),
        "layers.42.attn.wq_a.scale": ("F8_E8M0", (8, 32)),
        "layers.42.attn.wq_a.weight": ("F8_E4M3", (1024, 4096)),
        "layers.42.attn.wq_b.scale": ("F8_E8M0", (256, 8)),
        "layers.42.attn.wq_b.weight": ("F8_E4M3", (32768, 1024)),
        "layers.42.attn_norm.weight": ("BF16", (4096,)),
        "layers.42.ffn.gate.bias": ("F32", (256,)),
        "layers.42.ffn.gate.weight": ("BF16", (256, 4096)),
        "layers.42.ffn_norm.weight": ("BF16", (4096,)),
        "layers.42.hc_attn_base": ("F32", (24,)),
        "layers.42.hc_attn_fn": ("F32", (24, 16384)),
        "layers.42.hc_attn_scale": ("F32", (3,)),
        "layers.42.hc_ffn_base": ("F32", (24,)),
        "layers.42.hc_ffn_fn": ("F32", (24, 16384)),
        "layers.42.hc_ffn_scale": ("F32", (3,)),
    }


def _shared_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    prefix = "layers.42.ffn.shared_experts"
    return {
        f"{prefix}.w1.scale": ("F8_E8M0", (16, 32)),
        f"{prefix}.w1.weight": ("F8_E4M3", (2048, 4096)),
        f"{prefix}.w2.scale": ("F8_E8M0", (32, 16)),
        f"{prefix}.w2.weight": ("F8_E4M3", (4096, 2048)),
        f"{prefix}.w3.scale": ("F8_E8M0", (16, 32)),
        f"{prefix}.w3.weight": ("F8_E4M3", (2048, 4096)),
    }


def _route_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    specs: dict[str, tuple[str, tuple[int, ...]]] = {}
    for expert_id in ROUTE_IDS:
        prefix = f"layers.42.ffn.experts.{expert_id}"
        specs.update(
            {
                f"{prefix}.w1.scale": ("F8_E8M0", (2048, 128)),
                f"{prefix}.w1.weight": ("I8", (2048, 2048)),
                f"{prefix}.w2.scale": ("F8_E8M0", (4096, 64)),
                f"{prefix}.w2.weight": ("I8", (4096, 1024)),
                f"{prefix}.w3.scale": ("F8_E8M0", (2048, 128)),
                f"{prefix}.w3.weight": ("I8", (2048, 2048)),
            }
        )
    return specs


def _json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"缺少真实资产文件: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _element_bytes(dtype: str) -> int:
    try:
        return {"F32": 4, "BF16": 2, "F8_E8M0": 1, "F8_E4M3": 1, "I8": 1}[dtype]
    except KeyError as exc:
        raise ValueError(f"未知物理 dtype: {dtype}") from exc


def _expected_bytes(spec: tuple[str, tuple[int, ...]]) -> int:
    dtype, shape = spec
    return math.prod(shape) * _element_bytes(dtype)


def _inside_cache(path: Path, cache_root: Path) -> bool:
    try:
        return os.path.commonpath((str(path.resolve()), str(cache_root.resolve()))) == str(cache_root.resolve())
    except ValueError:
        return False


def _validate_manifest_header(
    document: Mapping[str, Any], *, expected_format: str, expected_entries: int | None, layer: bool
) -> None:
    if document.get("format") != expected_format:
        raise ValueError(f"manifest format 不受支持: {document.get('format')!r}")
    if document.get("revision") != REVISION:
        raise ValueError("manifest revision 漂移")
    if layer and document.get("layer") != LAYER:
        raise ValueError("manifest layer 不是冻结的 L42")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError("manifest entries 必须是数组")
    if expected_entries is not None and len(entries) != expected_entries:
        raise ValueError(f"manifest entry 数应为 {expected_entries}，实际为 {len(entries)}")
    if document.get("entry_count") != len(entries):
        raise ValueError("manifest entry_count 与 entries 不一致")
    if document.get("bytes") != sum(entry.get("bytes", -1) for entry in entries):
        raise ValueError("manifest bytes 与 entry 字节和不一致")


def _index_entries(entries: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = entry.get("tensor")
        if not isinstance(name, str) or not name or name in result:
            raise ValueError(f"{label} 含缺失或重复 tensor")
        result[name] = entry
    return result


@dataclass(frozen=True)
class AssetBundle:
    root: Path
    entries: dict[str, dict[str, Any]]
    specs: dict[str, tuple[str, tuple[int, ...]]]
    hashes_checked: int
    payload_bytes: int
    manifest_paths: dict[str, Path]


def validate_assets(
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    *,
    manifest_paths: Mapping[str, str | Path] | None = None,
    verify_hashes: bool = True,
) -> AssetBundle:
    """校验并索引 L42 完整单层所需的真实 Range 资产。

    ``manifest_paths`` 仅用于自检的临时 manifest 注入；所有 payload 路径
    仍必须位于 ``asset_root/range_cache`` 内。
    """

    root = Path(asset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"真实资产根目录不存在: {root}")
    selected = {
        "base": root / "l42_base_cache_manifest.json",
        "route": root / "l42_real_layer_route_manifest.json",
        "s14": root / "s14_base_cache_manifest.json",
    }
    for key, value in (manifest_paths or {}).items():
        if key not in selected:
            raise ValueError(f"未知 manifest 别名: {key}")
        selected[key] = Path(value)

    base = _json(selected["base"])
    route = _json(selected["route"])
    s14 = _json(selected["s14"])
    _validate_manifest_header(
        base, expected_format="polaris-l42-base-cache-snapshot-v1", expected_entries=34, layer=True
    )
    _validate_manifest_header(
        route, expected_format="polaris-l42-real-layer-route-cache-v1", expected_entries=36, layer=True
    )
    _validate_manifest_header(
        s14, expected_format="polaris-s14-base-cache-snapshot-v1", expected_entries=None, layer=False
    )
    if route.get("expert_ids") != ROUTE_IDS:
        raise ValueError("真实 route expert_ids 缺失或漂移")
    if route.get("route_weights") != ROUTE_WEIGHTS:
        raise ValueError("真实 route_weights 漂移")
    if not math.isclose(sum(ROUTE_WEIGHTS), 1.5, rel_tol=0, abs_tol=1e-7):
        raise AssertionError("内部冻结 route 权重和异常")

    base_index = _index_entries(base["entries"], "base manifest")
    route_index = _index_entries(route["entries"], "route manifest")
    s14_index = _index_entries(s14["entries"], "S14 manifest")
    base_specs = _base_specs()
    shared_specs = _shared_specs()
    route_specs = _route_specs()
    if set(base_index) != set(base_specs):
        missing = sorted(set(base_specs) - set(base_index))
        extra = sorted(set(base_index) - set(base_specs))
        raise ValueError(f"base tensor 集不完整: missing={missing}, extra={extra}")
    if set(route_index) != set(route_specs):
        missing = sorted(set(route_specs) - set(route_index))
        extra = sorted(set(route_index) - set(route_specs))
        raise ValueError(f"route tensor 集不完整: missing={missing}, extra={extra}")
    missing_shared = sorted(set(shared_specs) - set(s14_index))
    if missing_shared:
        raise ValueError(f"缺少共享专家 tensor: {missing_shared}")

    entries = {**base_index, **route_index, **{name: s14_index[name] for name in shared_specs}}
    specs = {**base_specs, **route_specs, **shared_specs}
    if set(entries) != set(specs):
        raise AssertionError("内部 tensor 索引与物理合同不一致")

    cache_root = root / "range_cache"
    hashes_checked = 0
    payload_bytes = 0
    for name in sorted(specs):
        entry = entries[name]
        expected_size = _expected_bytes(specs[name])
        if entry.get("bytes") != expected_size:
            raise ValueError(f"{name} 的 bytes 与冻结 shape/dtype 不一致")
        if ".experts." in name:
            expected_expert = int(name.split(".experts.", 1)[1].split(".", 1)[0])
            if entry.get("expert_id") != expected_expert:
                raise ValueError(f"{name} 的 expert_id 漂移")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{name} 缺少 payload 路径")
        path = Path(raw_path)
        if path.suffix.lower() != ".bin" or not _inside_cache(path, cache_root):
            raise ValueError(f"{name} 的 payload 越出冻结 range_cache 边界")
        if not path.is_file():
            raise FileNotFoundError(f"缺少真实 tensor payload: {name} -> {path}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"{name} 的实际文件长度不符")
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"{name} 缺少合法 SHA-256")
        if verify_hashes:
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                raise ValueError(f"{name} 的 SHA-256 漂移")
            hashes_checked += 1
        payload_bytes += expected_size

    return AssetBundle(
        root=root,
        entries=entries,
        specs=specs,
        hashes_checked=hashes_checked,
        payload_bytes=payload_bytes,
        manifest_paths={key: path.resolve() for key, path in selected.items()},
    )


class _InlineForward:
    def __init__(self, bundle: AssetBundle, capture_dir: Path | None = None):
        self.bundle = bundle
        self.capture_dir = capture_dir
        self.captures: list[dict[str, Any]] = []

    def _entry(self, name: str) -> dict[str, Any]:
        try:
            return self.bundle.entries[name]
        except KeyError as exc:
            raise ValueError(f"缺少已校验 tensor: {name}") from exc

    def _path(self, name: str) -> Path:
        return Path(self._entry(name)["path"])

    def _load_tensor(self, name: str) -> torch.Tensor:
        dtype_name, shape = self.bundle.specs[name]
        if dtype_name not in {"F32", "BF16"}:
            raise TypeError(f"{name} 不是普通浮点 tensor")
        payload = bytearray(self._path(name).read_bytes())
        dtype = torch.float32 if dtype_name == "F32" else torch.bfloat16
        return torch.frombuffer(payload, dtype=dtype).reshape(shape).clone()

    @staticmethod
    def _bf16_numpy(array: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        return torch.from_numpy(contiguous).to(torch.bfloat16).float().numpy()

    def _weight_fp8(self, prefix: str, *, bf16: bool = False) -> np.ndarray:
        weight_name = prefix + ".weight"
        scale_name = prefix + ".scale"
        weight_shape = self.bundle.specs[weight_name][1]
        scale_shape = self.bundle.specs[scale_name][1]
        weight_path = self._path(weight_name).resolve(strict=True)
        scale_path = self._path(scale_name).resolve(strict=True)
        packed = np.memmap(weight_path, dtype=np.uint8, mode="r", shape=weight_shape)
        scales = np.memmap(scale_path, dtype=np.uint8, mode="r", shape=scale_shape)
        weight_stat = weight_path.stat()
        scale_stat = scale_path.stat()
        validation_key = (
            (str(weight_path), weight_stat.st_size, weight_stat.st_mtime_ns),
            (str(scale_path), scale_stat.st_size, scale_stat.st_mtime_ns),
        )
        with _FP8_VALIDATION_LOCK:
            already_validated = validation_key in _FP8_VALIDATION_CACHE
        if not already_validated:
            if np.any((packed == 127) | (packed == 255)) or np.any(scales == 255):
                raise ValueError(f"{prefix} 含 FP8/UE8M0 NaN 编码")
            # Range cache 同样以绝对路径、size、mtime_ns作为进程内proof key。
            # tensor文件若改变，下一次会得到新key并重新扫描；未改变的只读页
            # 不必在每个token重复遍历几十MB寻找NaN code。
            with _FP8_VALIDATION_LOCK:
                _FP8_VALIDATION_CACHE.add(validation_key)
        scale_values = np.exp2(scales.astype(np.float32) - 127)
        weight = _FP8_LUT[packed]
        # S14 的 UE8M0 scale 每个值覆盖一个 128x128 权重块。真实投影
        # 都完整按块对齐；直接把解码后的权重 view 成块并原位广播，避免
        # ``repeat`` 额外物化一张与 F32 权重同尺寸的 scale 矩阵。
        expected_shape = (scales.shape[0] * 128, scales.shape[1] * 128)
        if packed.shape == expected_shape:
            weight = weight.reshape(scales.shape[0], 128, scales.shape[1], 128)
            weight *= scale_values[:, None, :, None]
            weight = weight.reshape(packed.shape)
        else:
            # 保留非整块 fixture/未来张量的旧语义；生产 S14 不走此分支。
            expanded = np.repeat(np.repeat(scale_values, 128, 0), 128, 1)
            weight *= expanded[: packed.shape[0], : packed.shape[1]]
        return self._bf16_numpy(weight) if bf16 else weight

    def _weight_fp4(self, prefix: str) -> np.ndarray:
        weight_name = prefix + ".weight"
        scale_name = prefix + ".scale"
        weight_shape = self.bundle.specs[weight_name][1]
        scale_shape = self.bundle.specs[scale_name][1]
        packed = np.memmap(self._path(weight_name), dtype=np.uint8, mode="r", shape=weight_shape)
        scales = np.memmap(self._path(scale_name), dtype=np.uint8, mode="r", shape=scale_shape)
        if np.any(scales == 255):
            raise ValueError(f"{prefix} 含 UE8M0 NaN 编码")
        weight = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
        weight[:, 0::2] = _FP4_LUT[packed & 15]
        weight[:, 1::2] = _FP4_LUT[packed >> 4]
        weight = weight.reshape(scales.shape[0], scales.shape[1], 32)
        weight *= np.exp2(scales.astype(np.float32) - 127)[:, :, None]
        return weight.reshape(packed.shape[0], packed.shape[1] * 2)

    @staticmethod
    def _activation_quant(array: np.ndarray, group_size: int = 128) -> np.ndarray:
        shape = array.shape
        flat = np.asarray(array, dtype=np.float32).reshape(-1, shape[-1])
        if flat.shape[1] % group_size:
            raise ValueError("激活最后一维必须能被量化 group_size 整除")
        output = np.empty_like(flat)
        for start in range(0, flat.shape[1], group_size):
            block = flat[:, start : start + group_size]
            amax = np.maximum(np.abs(block).max(1, keepdims=True), np.float32(1e-4))
            scale = np.exp2(np.ceil(np.log2(amax / np.float32(448))).astype(np.int32)).astype(np.float32)
            normalized = np.ascontiguousarray(np.clip(block / scale, -448, 448))
            quantized = torch.from_numpy(normalized).to(torch.float8_e4m3fn).float().numpy()
            output[:, start : start + group_size] = quantized * scale
        return output.reshape(shape)

    def _linear_fp8(self, array: np.ndarray, prefix: str) -> np.ndarray:
        activation = self._activation_quant(array)
        weight = self._weight_fp8(prefix)
        output = activation.reshape(-1, activation.shape[-1]) @ weight.T
        del weight
        # ``weight`` is a plain NumPy array, so CPython releases it
        # immediately through reference counting.  A full cyclic-GC scan here
        # does not reclaim the matrix and used to run after every projection
        # (hundreds of times per FullDepth token).  Keep cyclic collection at
        # the outer token boundary instead of taxing every matvec.
        return self._bf16_numpy(output.reshape(*array.shape[:-1], output.shape[-1]))

    def _linear_fp4(self, array: np.ndarray, prefix: str) -> np.ndarray:
        activation = self._activation_quant(array)
        weight = self._weight_fp4(prefix)
        output = activation.reshape(-1, activation.shape[-1]) @ weight.T
        del weight
        return self._bf16_numpy(output.reshape(*array.shape[:-1], output.shape[-1]))

    @staticmethod
    def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        value = x.float()
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        return (normalized * weight.float()).to(dtype)

    @staticmethod
    def _silu(array: np.ndarray) -> np.ndarray:
        sigmoid = np.empty_like(array)
        positive = array >= 0
        sigmoid[positive] = 1 / (1 + np.exp(-array[positive]))
        exp_value = np.exp(array[~positive])
        sigmoid[~positive] = exp_value / (1 + exp_value)
        return array * sigmoid

    @staticmethod
    def _summary(tensor: torch.Tensor) -> dict[str, Any]:
        array = tensor.float().contiguous().numpy().astype("<f4", copy=False)
        return {
            "shape": list(tensor.shape),
            "l2": float(np.linalg.norm(array)),
            "mean": float(array.mean()),
            "maxabs": float(np.abs(array).max()),
            "f32_le_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }

    def _capture_kernel_input(self, name: str, array: np.ndarray) -> None:
        if self.capture_dir is None:
            return
        payload = np.ascontiguousarray(array, dtype="<f4")
        path = self.capture_dir / f"{name}.f32le.bin"
        encoded = payload.tobytes()
        path.write_bytes(encoded)
        self.captures.append(
            {
                "name": name,
                "file": path.name,
                "shape": list(payload.shape),
                "bytes": len(encoded),
                "f32_le_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )

    def _write_capture_manifest(self, ffn_input: torch.Tensor) -> None:
        if self.capture_dir is None:
            return
        source_ffn = ffn_input.float().contiguous().numpy().astype("<f4", copy=False)
        document = {
            "format": "polaris-l42-real-vulkan-input-capture-v1",
            "repo": REPO,
            "revision": REVISION,
            "layer": LAYER,
            "expert_id": ROUTE_IDS[0],
            "source_f32_le_sha256": {
                "ffn_input": hashlib.sha256(source_ffn.tobytes()).hexdigest(),
            },
            "asset_integrity": {
                "hashes_checked": self.bundle.hashes_checked,
                "payload_bytes": self.bundle.payload_bytes,
                "payload_files": len(self.bundle.entries),
                "manifest_sha256": {
                    key: _sha256(path) for key, path in sorted(self.bundle.manifest_paths.items())
                },
            },
            "inputs": self.captures,
            "semantics": (
                "Inputs captured from the hash-verified real L42 inline reference after the "
                "official UE8M0/E4M3FN activation quantization step."
            ),
        }
        if document["source_f32_le_sha256"]["ffn_input"] != FROZEN_OUTPUT_SHA256["ffn_input"]:
            raise AssertionError("capture 的真实 L42 ffn_input 指纹漂移")
        (self.capture_dir / "capture_manifest.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def run(self) -> dict[str, Any]:
        if self.capture_dir is not None:
            self.capture_dir.mkdir(parents=True, exist_ok=False)
        base = torch.sin(torch.arange(4096, dtype=torch.float32) * 0.013).to(torch.bfloat16)
        state = base.view(1, 1, 1, 4096).repeat(1, 1, 4, 1)

        branch, post, comb = hc_pre(
            state,
            self._load_tensor("layers.42.hc_attn_fn"),
            self._load_tensor("layers.42.hc_attn_scale"),
            self._load_tensor("layers.42.hc_attn_base"),
        )
        branch = self._rms_norm(branch, self._load_tensor("layers.42.attn_norm.weight"))
        branch_np = branch.float().numpy()
        self._capture_kernel_input("wq_a", self._activation_quant(branch_np))
        q_low = torch.from_numpy(self._linear_fp8(branch_np, "layers.42.attn.wq_a")).to(torch.bfloat16)
        q_low = self._rms_norm(q_low, self._load_tensor("layers.42.attn.q_norm.weight"))
        q = torch.from_numpy(self._linear_fp8(q_low.float().numpy(), "layers.42.attn.wq_b"))
        q = q.to(torch.bfloat16).reshape(1, 1, 64, 512)
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)

        kv = torch.from_numpy(self._linear_fp8(branch_np, "layers.42.attn.wkv")).to(torch.bfloat16)
        kv = self._rms_norm(kv, self._load_tensor("layers.42.attn.kv_norm.weight"))
        kv[..., :-64] = torch.from_numpy(self._activation_quant(kv[..., :-64].float().numpy(), 64)).to(
            torch.bfloat16
        )
        attention = sparse_attention(
            q,
            kv,
            self._load_tensor("layers.42.attn.attn_sink"),
            torch.tensor([[[0]]], dtype=torch.int32),
            softmax_scale=512**-0.5,
            output_dtype=torch.bfloat16,
        )
        grouped = attention.reshape(1, 1, 8, 4096).float().numpy()
        wo_a = self._weight_fp8("layers.42.attn.wo_a", bf16=True).reshape(8, 1024, 4096)
        low_output = self._bf16_numpy(
            np.stack([grouped[0, 0, group] @ wo_a[group].T for group in range(8)]).reshape(1, 1, 8192)
        )
        del wo_a
        gc.collect()
        attention_branch = torch.from_numpy(self._linear_fp8(low_output, "layers.42.attn.wo_b")).to(
            torch.bfloat16
        )
        post_attention_state = hc_post(attention_branch, state, post, comb)

        ffn_input, post_ffn, comb_ffn = hc_pre(
            post_attention_state,
            self._load_tensor("layers.42.hc_ffn_fn"),
            self._load_tensor("layers.42.hc_ffn_scale"),
            self._load_tensor("layers.42.hc_ffn_base"),
        )
        ffn_input = self._rms_norm(ffn_input, self._load_tensor("layers.42.ffn_norm.weight"))
        logits = F.linear(ffn_input.float(), self._load_tensor("layers.42.ffn.gate.weight").float())
        original_scores = F.softplus(logits).sqrt()
        route_ids = (original_scores + self._load_tensor("layers.42.ffn.gate.bias").float()).topk(6, -1).indices
        route_weights = original_scores.gather(-1, route_ids)
        route_weights = route_weights / route_weights.sum(-1, keepdim=True) * 1.5
        actual_ids = [int(value) for value in route_ids.flatten()]
        actual_weights = [float(value) for value in route_weights.flatten()]
        if actual_ids != ROUTE_IDS:
            raise AssertionError(f"原生 L42 route 漂移: expected={ROUTE_IDS}, actual={actual_ids}")
        if actual_weights != ROUTE_WEIGHTS:
            raise AssertionError(f"原生 L42 route 权重漂移: expected={ROUTE_WEIGHTS}, actual={actual_weights}")

        ffn_np = ffn_input.float().numpy()
        self._capture_kernel_input("expert_126_w1_w3", self._activation_quant(ffn_np))
        moe = np.zeros((1, 1, 4096), dtype=np.float32)
        for expert_id, route_weight in zip(actual_ids, actual_weights, strict=True):
            prefix = f"layers.42.ffn.experts.{expert_id}"
            gate = self._linear_fp4(ffn_np, prefix + ".w1").astype(np.float32)
            up = self._linear_fp4(ffn_np, prefix + ".w3").astype(np.float32)
            hidden = self._silu(np.minimum(gate, 10)) * np.clip(up, -10, 10)
            if expert_id == ROUTE_IDS[0]:
                self._capture_kernel_input("expert_126_w2", self._activation_quant(hidden))
            expert_output = self._linear_fp4(hidden, prefix + ".w2")
            moe += np.float32(route_weight) * expert_output

        shared_prefix = "layers.42.ffn.shared_experts"
        shared_gate = self._linear_fp8(ffn_np, shared_prefix + ".w1").astype(np.float32)
        shared_up = self._linear_fp8(ffn_np, shared_prefix + ".w3").astype(np.float32)
        shared_hidden = self._silu(np.minimum(shared_gate, 10)) * np.clip(shared_up, -10, 10)
        moe += self._linear_fp8(shared_hidden, shared_prefix + ".w2")
        moe_branch = torch.from_numpy(self._bf16_numpy(moe)).to(torch.bfloat16)
        layer_output = hc_post(moe_branch, post_attention_state, post_ffn, comb_ffn)
        self._write_capture_manifest(ffn_input)

        report = {
            "format": "polaris-l42-real-single-token-inline-reference-v1",
            "repo": REPO,
            "revision": REVISION,
            "layer": LAYER,
            "input": "four identical BF16 copies of sin(arange(4096)*0.013)",
            "start_pos": 0,
            "expert_ids": actual_ids,
            "route_weights": actual_weights,
            "weight_sum": float(route_weights.sum()),
            "attention_branch": self._summary(attention_branch),
            "post_attention_state": self._summary(post_attention_state),
            "ffn_input": self._summary(ffn_input),
            "moe_branch": self._summary(moe_branch),
            "layer_output": self._summary(layer_output),
            "integrity": {
                "hashes_checked": self.bundle.hashes_checked,
                "payload_bytes": self.bundle.payload_bytes,
                "payload_files": len(self.bundle.entries),
            },
            "semantics": (
                "real L42 HC+FP8 sparse attention+native route+top6 FP4 MoE+FP8 shared+HC; "
                "UE8M0 activation quant included"
            ),
            "claim_limit": "真实 L42 单层单 token 参考；不是 S14/43 层首 token，不代表质量或 GPU 速度",
        }
        return report


def run_reference(
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    *,
    verify_hashes: bool = True,
    capture_dir: str | Path | None = None,
) -> dict[str, Any]:
    """校验真实资产后执行完整 L42 单层单 token 参考前向。"""

    bundle = validate_assets(asset_root, verify_hashes=verify_hashes)
    selected_capture_dir = Path(capture_dir).resolve() if capture_dir is not None else None
    return _InlineForward(bundle, selected_capture_dir).run()


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        help="可选：导出真实 L42 Vulkan kernel 输入；目标目录必须尚不存在",
    )
    args = parser.parse_args()
    report = run_reference(args.asset_root, verify_hashes=True, capture_dir=args.capture_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
