"""F8_E4M3FN + F8_E8M0 的显式位级参考语义和分块 linear。"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .mxfp4 import decode_ue8m0


FP8_GROUP_SIZE = 128


def _tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    return value


def _fp8_as_uint8(encoded: torch.Tensor, name: str) -> torch.Tensor:
    encoded = _tensor(encoded, name)
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if encoded.dtype == torch.uint8:
        return encoded
    if fp8_dtype is not None and encoded.dtype == fp8_dtype:
        return encoded.contiguous().view(torch.uint8)
    raise TypeError(f"{name} 必须是 uint8 或 torch.float8_e4m3fn")


def _output_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("输出 dtype 必须是 float16、bfloat16、float32 或 float64")


def _decode_scale(scales: torch.Tensor, name: str) -> torch.Tensor:
    scales = _tensor(scales, name)
    ue8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scales.dtype == torch.uint8 or (ue8m0_dtype is not None and scales.dtype == ue8m0_dtype):
        return decode_ue8m0(scales)
    if not scales.is_floating_point():
        raise TypeError(f"{name} 必须是 UE8M0 原始字节或浮点张量")
    if not bool(torch.isfinite(scales).all().item()):
        raise ValueError(f"{name} 包含非有限值")
    return scales.float()


def decode_fp8_e4m3fn(
    encoded: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    reject_nan: bool = True,
) -> torch.Tensor:
    """不依赖 float8 算子的 F8_E4M3FN 位级解码。

    格式为 1 位符号、4 位指数、3 位尾数，bias=7。指数 0 表示零/次正规数；
    ``exp=15,mantissa=7``（``0x7F``/``0xFF``）是 NaN，其余均为有限值，
    最大有限值为 ±448。
    """

    _output_dtype(dtype)
    raw = _fp8_as_uint8(encoded, "encoded")
    raw_i = raw.to(torch.int32)
    sign = (raw_i & 0x80) != 0
    exponent = (raw_i >> 3) & 0x0F
    mantissa = raw_i & 0x07
    invalid = (exponent == 0x0F) & (mantissa == 0x07)
    if reject_nan and bool(invalid.any().item()):
        raise ValueError("F8_E4M3FN 包含 NaN 编码 0x7F/0xFF")

    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    mantissa_f = mantissa.to(compute_dtype)
    subnormal = (mantissa_f / 8.0) * (2.0**-6)
    normal = torch.ldexp(1.0 + mantissa_f / 8.0, exponent - 7)
    magnitude = torch.where(exponent == 0, subnormal, normal)
    decoded = torch.where(sign, -magnitude, magnitude)
    if bool(invalid.any().item()):
        decoded = decoded.masked_fill(invalid, math.nan)
    return decoded.to(dtype)


def decode_scaled_fp8_e4m3fn(
    encoded: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = FP8_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """解码最后一维分组的 E4M3FN，并乘以 UE8M0 scale。"""

    raw = _fp8_as_uint8(encoded, "encoded")
    scales = _tensor(scales, "scales")
    if raw.ndim < 1 or scales.ndim != raw.ndim:
        raise ValueError("encoded 与 scales 必须具有相同维数且至少为一维")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError("group_size 必须是正整数")
    groups = math.ceil(raw.shape[-1] / group_size)
    expected = (*raw.shape[:-1], groups)
    if tuple(scales.shape) != expected:
        raise ValueError(f"scales 应为 {expected}，实际为 {tuple(scales.shape)}")
    if scales.device != raw.device:
        raise ValueError("encoded 与 scales 必须位于同一设备")

    values = decode_fp8_e4m3fn(raw, dtype=dtype)
    scale_values = _decode_scale(scales, "scales").to(dtype)
    expanded_scales = scale_values.repeat_interleave(group_size, dim=-1)[..., : raw.shape[-1]]
    return values * expanded_scales


def fp8_linear(
    x: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    group_size: int = FP8_GROUP_SIZE,
    bias: Optional[torch.Tensor] = None,
    output_chunk_size: int = FP8_GROUP_SIZE,
    accumulation_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """分块复现官方 ``fp8_gemm`` 的 scale-corrected FP32 累加。

    ``x=[...,K]`` 和 ``weight=[N,K]`` 均为 E4M3FN 原始字节（或对应 PyTorch
    dtype）。activation scale 为 ``[...,ceil(K/128)]``；weight scale 为
    ``[ceil(N/128),ceil(K/128)]``，即一个 scale 共享一个 128×128 权重 tile。
    """

    raw_x = _fp8_as_uint8(x, "x")
    raw_weight = _fp8_as_uint8(weight, "weight")
    activation_scale = _tensor(activation_scale, "activation_scale")
    weight_scale = _tensor(weight_scale, "weight_scale")
    _output_dtype(output_dtype)
    if raw_x.ndim < 1:
        raise ValueError("x 至少需要一个维度")
    if raw_weight.ndim != 2:
        raise ValueError("weight 必须是二维 [N,K]")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError("group_size 必须是正整数")
    if not isinstance(output_chunk_size, int) or isinstance(output_chunk_size, bool) or output_chunk_size <= 0:
        raise ValueError("output_chunk_size 必须是正整数")
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise TypeError("accumulation_dtype 只支持 float32 或 float64")
    if raw_x.device != raw_weight.device or raw_x.device != activation_scale.device or raw_x.device != weight_scale.device:
        raise ValueError("x、weight 与 scale 必须位于同一设备")

    n, weight_k = raw_weight.shape
    k = raw_x.shape[-1]
    if weight_k != k or k == 0:
        raise ValueError(f"K 不匹配或为空: x.K={k}, weight.K={weight_k}")
    k_groups = math.ceil(k / group_size)
    expected_a = (*raw_x.shape[:-1], k_groups)
    expected_b = (math.ceil(n / group_size), k_groups)
    if tuple(activation_scale.shape) != expected_a:
        raise ValueError(
            f"activation_scale 应为 {expected_a}，实际为 {tuple(activation_scale.shape)}"
        )
    if tuple(weight_scale.shape) != expected_b:
        raise ValueError(f"weight_scale 应为 {expected_b}，实际为 {tuple(weight_scale.shape)}")
    if bias is not None:
        bias = _tensor(bias, "bias")
        if tuple(bias.shape) != (n,) or not bias.is_floating_point():
            raise ValueError(f"bias 必须是形状 {(n,)} 的浮点张量")
        if bias.device != raw_x.device:
            raise ValueError("bias 必须与 x 位于同一设备")

    scale_a = _decode_scale(activation_scale, "activation_scale").reshape(-1, k_groups)
    scale_b = _decode_scale(weight_scale, "weight_scale")
    x_rows = raw_x.reshape(-1, k)
    result = torch.empty((x_rows.shape[0], n), dtype=accumulation_dtype, device=raw_x.device)

    for n_start in range(0, n, output_chunk_size):
        n_end = min(n_start + output_chunk_size, n)
        acc = torch.zeros((x_rows.shape[0], n_end - n_start), dtype=accumulation_dtype, device=raw_x.device)
        weight_scale_rows = torch.arange(n_start, n_end, device=raw_x.device) // group_size
        for group in range(k_groups):
            k_start = group * group_size
            k_end = min(k_start + group_size, k)
            activation_block = decode_fp8_e4m3fn(
                x_rows[:, k_start:k_end], dtype=accumulation_dtype
            )
            activation_block *= scale_a[:, group].to(accumulation_dtype).unsqueeze(-1)
            weight_block = decode_fp8_e4m3fn(
                raw_weight[n_start:n_end, k_start:k_end], dtype=accumulation_dtype
            )
            weight_block *= scale_b[weight_scale_rows, group].to(accumulation_dtype).unsqueeze(-1)
            acc.addmm_(activation_block, weight_block.transpose(0, 1))
        if bias is not None:
            acc += bias[n_start:n_end].to(accumulation_dtype)
        result[:, n_start:n_end] = acc

    return result.reshape(*raw_x.shape[:-1], n).to(output_dtype)


def fp8_weight_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    group_size: int = FP8_GROUP_SIZE,
    bias: Optional[torch.Tensor] = None,
    output_chunk_size: int = FP8_GROUP_SIZE,
    accumulation_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """以普通浮点 activation 乘分块解码的 E4M3FN 权重。

    该入口用于隔离验证 checkpoint 的 FP8 weight/scale ABI；官方端到端 ``linear``
    还会先量化 activation，需使用同时接收 ``activation_scale`` 的 :func:`fp8_linear`。
    """

    x = _tensor(x, "x")
    raw_weight = _fp8_as_uint8(weight, "weight")
    weight_scale = _tensor(weight_scale, "weight_scale")
    _output_dtype(output_dtype)
    if x.ndim < 1 or not x.is_floating_point():
        raise TypeError("x 必须是至少一维的普通浮点张量")
    if raw_weight.ndim != 2:
        raise ValueError("weight 必须是二维 [N,K]")
    if x.device != raw_weight.device or x.device != weight_scale.device:
        raise ValueError("x、weight 与 weight_scale 必须位于同一设备")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError("group_size 必须是正整数")
    if not isinstance(output_chunk_size, int) or isinstance(output_chunk_size, bool) or output_chunk_size <= 0:
        raise ValueError("output_chunk_size 必须是正整数")
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise TypeError("accumulation_dtype 只支持 float32 或 float64")

    n, weight_k = raw_weight.shape
    k = x.shape[-1]
    if weight_k != k or k == 0:
        raise ValueError(f"K 不匹配或为空: x.K={k}, weight.K={weight_k}")
    k_groups = math.ceil(k / group_size)
    expected_scale = (math.ceil(n / group_size), k_groups)
    if tuple(weight_scale.shape) != expected_scale:
        raise ValueError(f"weight_scale 应为 {expected_scale}，实际为 {tuple(weight_scale.shape)}")
    if bias is not None:
        bias = _tensor(bias, "bias")
        if tuple(bias.shape) != (n,) or not bias.is_floating_point():
            raise ValueError(f"bias 必须是形状 {(n,)} 的浮点张量")
        if bias.device != x.device:
            raise ValueError("bias 必须与 x 位于同一设备")

    scale_b = _decode_scale(weight_scale, "weight_scale")
    x_rows = x.reshape(-1, k).to(accumulation_dtype)
    result = torch.empty((x_rows.shape[0], n), dtype=accumulation_dtype, device=x.device)
    for n_start in range(0, n, output_chunk_size):
        n_end = min(n_start + output_chunk_size, n)
        acc = torch.zeros((x_rows.shape[0], n_end - n_start), dtype=accumulation_dtype, device=x.device)
        scale_rows = torch.arange(n_start, n_end, device=x.device) // group_size
        for group in range(k_groups):
            k_start = group * group_size
            k_end = min(k_start + group_size, k)
            weight_block = decode_fp8_e4m3fn(
                raw_weight[n_start:n_end, k_start:k_end], dtype=accumulation_dtype
            )
            weight_block *= scale_b[scale_rows, group].to(accumulation_dtype).unsqueeze(-1)
            acc.addmm_(x_rows[:, k_start:k_end], weight_block.transpose(0, 1))
        if bias is not None:
            acc += bias[n_start:n_end].to(accumulation_dtype)
        result[:, n_start:n_end] = acc
    return result.reshape(*x.shape[:-1], n).to(output_dtype)
