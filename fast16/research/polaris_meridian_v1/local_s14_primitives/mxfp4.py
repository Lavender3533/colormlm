"""官方 MXFP4/UE8M0 布局的纯 PyTorch 参考实现。

DeepSeek 官方 ``Linear`` 将 FP4 权重保存为 ``[N, K // 2]``，沿 K 维每字节
打包两个 E2M1 值；scale 为 ``[N, K // 32]``。本模块只做参考语义，不声称快于
TileLang/CUDA/Vulkan 内核。
"""

from __future__ import annotations

import math
from typing import Optional

import torch


FP4_GROUP_SIZE = 32
FP8_ACTIVATION_GROUP_SIZE = 128
FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    return value


def _packed_as_uint8(packed: torch.Tensor) -> torch.Tensor:
    packed = _require_tensor(packed, "packed")
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if packed.dtype == torch.uint8:
        return packed
    if fp4_dtype is not None and packed.dtype == fp4_dtype:
        return packed.contiguous().view(torch.uint8)
    raise TypeError("packed 必须是 uint8 或 torch.float4_e2m1fn_x2")


def _scale_as_uint8(scales: torch.Tensor) -> torch.Tensor:
    scales = _require_tensor(scales, "scales")
    ue8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scales.dtype == torch.uint8:
        return scales
    if ue8m0_dtype is not None and scales.dtype == ue8m0_dtype:
        return scales.contiguous().view(torch.uint8)
    raise TypeError("UE8M0 scale 必须是 uint8 或 torch.float8_e8m0fnu")


def _require_output_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("输出 dtype 必须是 float16、bfloat16、float32 或 float64")


def unpack_mxfp4_e2m1(
    packed: torch.Tensor,
    *,
    logical_k: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """按 ``低半字节、再高半字节`` 顺序解包 E2M1。

    ``logical_k`` 只允许裁掉最后一个填充半字节；DeepSeek 官方权重的 K 为偶数，
    正常情况下无需传入该参数。
    """

    _require_output_dtype(dtype)
    raw = _packed_as_uint8(packed)
    if raw.ndim < 1:
        raise ValueError("packed 至少需要一个维度")
    capacity = raw.shape[-1] * 2
    if logical_k is None:
        logical_k = capacity
    if not isinstance(logical_k, int) or isinstance(logical_k, bool):
        raise TypeError("logical_k 必须是整数")
    if logical_k < max(0, capacity - 1) or logical_k > capacity:
        raise ValueError("logical_k 只能等于打包容量，或只裁掉最后一个填充半字节")

    table = torch.tensor(FP4_E2M1_VALUES, dtype=dtype, device=raw.device)
    result = torch.empty((*raw.shape[:-1], capacity), dtype=dtype, device=raw.device)
    result[..., 0::2] = table[(raw & 0x0F).long()]
    result[..., 1::2] = table[(raw >> 4).long()]
    return result[..., :logical_k]


def decode_ue8m0(
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    reject_nan: bool = True,
) -> torch.Tensor:
    """将 UE8M0 原始字节解码为 ``2 ** (code - 127)``。

    OCP UE8M0 的 ``0xFF`` 是 NaN 编码。权重 scale 中出现它通常代表损坏或布局
    错误，因此默认硬拒绝；``reject_nan=False`` 时显式返回 NaN。
    """

    _require_output_dtype(dtype)
    raw = _scale_as_uint8(scales)
    invalid = raw == 0xFF
    if reject_nan and bool(invalid.any().item()):
        raise ValueError("UE8M0 scale 包含 0xFF NaN 编码")

    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    exponent = raw.to(torch.int32) - 127
    decoded = torch.ldexp(torch.ones(raw.shape, dtype=compute_dtype, device=raw.device), exponent)
    if bool(invalid.any().item()):
        decoded = decoded.masked_fill(invalid, math.nan)
    return decoded.to(dtype)


def decode_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = FP4_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """解码带 UE8M0 分组 scale 的 MXFP4 张量。

    前导维必须完全相同，最后一维满足 ``packed_bytes * 2 == groups * group_size``。
    该严格检查对应固定 revision 的官方 ``[N,K//2]``/``[N,K//32]`` 合同。
    """

    raw = _packed_as_uint8(packed)
    scales = _require_tensor(scales, "scales")
    if raw.ndim < 1 or scales.ndim != raw.ndim:
        raise ValueError("packed 与 scales 必须具有相同的维数且至少为一维")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError("group_size 必须是正整数")
    logical_k = raw.shape[-1] * 2
    expected_groups, remainder = divmod(logical_k, group_size)
    if remainder:
        raise ValueError("打包后的 K 必须能被 group_size 整除")
    expected_shape = (*raw.shape[:-1], expected_groups)
    if tuple(scales.shape) != expected_shape:
        raise ValueError(
            f"MXFP4 packed/scale 形状不匹配: packed={tuple(raw.shape)}, "
            f"scales={tuple(scales.shape)}, expected_scales={expected_shape}"
        )
    values = unpack_mxfp4_e2m1(raw, dtype=dtype)
    scale_values = decode_ue8m0(scales, dtype=dtype)
    grouped = values.reshape(*raw.shape[:-1], expected_groups, group_size)
    return (grouped * scale_values.unsqueeze(-1)).reshape(*raw.shape[:-1], logical_k)


def _decode_linear_scale(scales: torch.Tensor, name: str) -> torch.Tensor:
    scales = _require_tensor(scales, name)
    if scales.dtype == torch.uint8 or scales.dtype == getattr(torch, "float8_e8m0fnu", None):
        return decode_ue8m0(scales)
    if not scales.is_floating_point():
        raise TypeError(f"{name} 必须是 UE8M0 原始字节或浮点张量")
    if not bool(torch.isfinite(scales).all().item()):
        raise ValueError(f"{name} 包含非有限值")
    return scales.float()


def fp4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    activation_scale: Optional[torch.Tensor] = None,
    activation_group_size: int = FP8_ACTIVATION_GROUP_SIZE,
    bias: Optional[torch.Tensor] = None,
    output_chunk_size: int = 128,
    accumulation_dtype: torch.dtype = torch.float32,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """分块计算 ``x @ dequant(weight).T`` 的官方 FP4 参考语义。

    ``x`` 的最后一维是 K，``packed_weight`` 为 ``[N,K//2]``，``weight_scale``
    为 ``[N,K//32]``。若给出 ``activation_scale``，则 ``x`` 被视为已量化的
    FP8 数值，按 ``activation_group_size``（官方为 128）逐组恢复 scale 后参与
    FP32 累加。实现只展开一个 ``output_chunk_size × 32`` 权重块。
    """

    x = _require_tensor(x, "x")
    raw_weight = _packed_as_uint8(packed_weight)
    weight_scale = _require_tensor(weight_scale, "weight_scale")
    if x.ndim < 1 or not x.is_floating_point():
        raise TypeError("x 必须是至少一维的浮点张量")
    if raw_weight.ndim != 2:
        raise ValueError("packed_weight 必须是二维 [N,K//2]")
    if x.device != raw_weight.device or x.device != weight_scale.device:
        raise ValueError("x、packed_weight 与 weight_scale 必须位于同一设备")
    if not isinstance(output_chunk_size, int) or isinstance(output_chunk_size, bool) or output_chunk_size <= 0:
        raise ValueError("output_chunk_size 必须是正整数")
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise TypeError("accumulation_dtype 只支持 float32 或 float64")

    n, packed_k = raw_weight.shape
    k = x.shape[-1]
    if packed_k * 2 != k:
        raise ValueError(f"K 不匹配: x.K={k}, packed_weight 容量={packed_k * 2}")
    if k == 0 or k % FP4_GROUP_SIZE:
        raise ValueError(f"K 必须是非零且能被 {FP4_GROUP_SIZE} 整除")
    weight_groups = k // FP4_GROUP_SIZE
    if tuple(weight_scale.shape) != (n, weight_groups):
        raise ValueError(
            f"weight_scale 应为 {(n, weight_groups)}，实际为 {tuple(weight_scale.shape)}"
        )

    if not isinstance(activation_group_size, int) or isinstance(activation_group_size, bool) or activation_group_size <= 0:
        raise ValueError("activation_group_size 必须是正整数")
    if activation_group_size % FP4_GROUP_SIZE or k % activation_group_size:
        raise ValueError("activation_group_size 必须是 32 的倍数且能整除 K")

    activation_scales = None
    if activation_scale is not None:
        activation_scale = _require_tensor(activation_scale, "activation_scale")
        expected = (*x.shape[:-1], k // activation_group_size)
        if tuple(activation_scale.shape) != expected:
            raise ValueError(
                f"activation_scale 应为 {expected}，实际为 {tuple(activation_scale.shape)}"
            )
        if activation_scale.device != x.device:
            raise ValueError("activation_scale 必须与 x 位于同一设备")
        activation_scales = _decode_linear_scale(activation_scale, "activation_scale")

    if bias is not None:
        bias = _require_tensor(bias, "bias")
        if tuple(bias.shape) != (n,) or not bias.is_floating_point():
            raise ValueError(f"bias 必须是形状 {(n,)} 的浮点张量")
        if bias.device != x.device:
            raise ValueError("bias 必须与 x 位于同一设备")

    if output_dtype is None:
        output_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else torch.float32
    _require_output_dtype(output_dtype)

    scale_w = _decode_linear_scale(weight_scale, "weight_scale").to(accumulation_dtype)
    x_rows = x.reshape(-1, k).to(accumulation_dtype)
    scale_a_rows = None
    if activation_scales is not None:
        scale_a_rows = activation_scales.reshape(-1, k // activation_group_size).to(accumulation_dtype)
    out = torch.empty((x_rows.shape[0], n), dtype=accumulation_dtype, device=x.device)

    packed_bytes_per_group = FP4_GROUP_SIZE // 2
    for n_start in range(0, n, output_chunk_size):
        n_end = min(n_start + output_chunk_size, n)
        acc = torch.zeros((x_rows.shape[0], n_end - n_start), dtype=accumulation_dtype, device=x.device)
        for group in range(weight_groups):
            k_start = group * FP4_GROUP_SIZE
            k_end = k_start + FP4_GROUP_SIZE
            byte_start = group * packed_bytes_per_group
            byte_end = byte_start + packed_bytes_per_group
            weight_block = unpack_mxfp4_e2m1(
                raw_weight[n_start:n_end, byte_start:byte_end], dtype=accumulation_dtype
            )
            weight_block = weight_block * scale_w[n_start:n_end, group].unsqueeze(-1)
            activation_block = x_rows[:, k_start:k_end]
            if scale_a_rows is not None:
                activation_group = k_start // activation_group_size
                activation_block = activation_block * scale_a_rows[:, activation_group].unsqueeze(-1)
            acc.addmm_(activation_block, weight_block.transpose(0, 1))
        if bias is not None:
            acc += bias[n_start:n_end].to(accumulation_dtype)
        out[:, n_start:n_end] = acc

    return out.reshape(*x.shape[:-1], n).to(output_dtype)
