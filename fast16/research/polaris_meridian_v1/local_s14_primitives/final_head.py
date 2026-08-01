"""DeepSeek-V4 固定 revision 的原生最终 HC → RMSNorm → head 参考路径。

官方 checkpoint 中 final head 权重是 BF16，但 ``inference/model.py`` 会把参数保存为
FP32，并对输入执行 ``x.float()`` 后做 linear。因此本实现接收 BF16 checkpoint 权重、
按词表行分块提升为 FP32，返回 FP32 logits；不会把 BF16 matmul 冒充官方语义。不同
分块大小可能触发不同 GEMM kernel，数学公式一致但不承诺末位 bitwise 相同。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


OFFICIAL_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
OFFICIAL_HIDDEN_SIZE = 4096
OFFICIAL_HC_MULT = 4
OFFICIAL_VOCAB_SIZE = 129280
OFFICIAL_NORM_EPS = 1e-6
OFFICIAL_HC_EPS = 1e-6


def _floating_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} 必须是浮点张量")
    return value


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是数值")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} 必须是有限正数")
    return result


def hc_head_reduce(
    x: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    *,
    norm_eps: float = OFFICIAL_NORM_EPS,
    hc_eps: float = OFFICIAL_HC_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把四路 mHC hidden 收束为单路，并返回 FP32 四路混合系数。

    ``x`` 必须为 ``[B,S,HC,D]``；官方主模型为 HC=4、D=4096。函数允许较小
    D 用于确定性测试，但严格要求所有 checkpoint 张量形状遵守同一公式。
    """

    x = _floating_tensor(x, "x")
    hc_head_fn = _floating_tensor(hc_head_fn, "hc_head_fn")
    hc_head_scale = _floating_tensor(hc_head_scale, "hc_head_scale")
    hc_head_base = _floating_tensor(hc_head_base, "hc_head_base")
    norm_eps = _positive_finite(norm_eps, "norm_eps")
    hc_eps = _positive_finite(hc_eps, "hc_eps")
    if x.ndim != 4:
        raise ValueError("x 必须是 [B,S,HC,D]")
    hc_mult = x.shape[-2]
    hidden = x.shape[-1]
    if hc_mult <= 0 or hidden <= 0:
        raise ValueError("x 的 HC 与 hidden 维必须大于零")
    if tuple(hc_head_fn.shape) != (hc_mult, hc_mult * hidden):
        raise ValueError(f"hc_head_fn 应为 {(hc_mult, hc_mult * hidden)}")
    if tuple(hc_head_scale.shape) != (1,):
        raise ValueError("hc_head_scale 应为 [1]")
    if tuple(hc_head_base.shape) != (hc_mult,):
        raise ValueError(f"hc_head_base 应为 [{hc_mult}]")
    if len({x.device, hc_head_fn.device, hc_head_scale.device, hc_head_base.device}) != 1:
        raise ValueError("HC head 的所有输入必须位于同一设备")

    shape = x.shape
    original_dtype = x.dtype
    flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat, hc_head_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_head_scale.float() + hc_head_base.float()) + hc_eps
    reduced = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2)
    return reduced.to(original_dtype), pre


def official_rms_norm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    norm_eps: float = OFFICIAL_NORM_EPS,
) -> torch.Tensor:
    """官方 RMSNorm：FP32 方差与乘法，最后恢复输入 dtype。"""

    x = _floating_tensor(x, "x")
    norm_weight = _floating_tensor(norm_weight, "norm_weight")
    norm_eps = _positive_finite(norm_eps, "norm_eps")
    if x.ndim < 1 or x.shape[-1] <= 0:
        raise ValueError("x 必须具有非空 hidden 末维")
    if tuple(norm_weight.shape) != (x.shape[-1],):
        raise ValueError(f"norm_weight 应为 [{x.shape[-1]}]")
    if x.device != norm_weight.device:
        raise ValueError("x 与 norm_weight 必须位于同一设备")
    original_dtype = x.dtype
    x_f32 = x.float()
    variance = x_f32.square().mean(dim=-1, keepdim=True)
    normalized = x_f32 * torch.rsqrt(variance + norm_eps)
    return (norm_weight.float() * normalized).to(original_dtype)


def bf16_checkpoint_head_logits(
    x: torch.Tensor,
    head_weight: torch.Tensor,
    *,
    full_logits: bool = False,
    output_chunk_size: int = 4096,
) -> torch.Tensor:
    """用 BF16 checkpoint head 复现官方 FP32 ``ParallelHead`` linear。

    单卡输入 ``head_weight`` 为 ``[V,D]``；张量并行时可对单个词表 shard 调用后，
    由上层按 rank 顺序 all-gather。默认和官方 ``full_logits=False`` 一样只投影最后 token。
    """

    x = _floating_tensor(x, "x")
    head_weight = _floating_tensor(head_weight, "head_weight")
    if head_weight.dtype != torch.bfloat16:
        raise TypeError("head_weight 必须保持 checkpoint 的 torch.bfloat16 dtype")
    if x.ndim != 3:
        raise ValueError("x 必须是 [B,S,D]")
    if head_weight.ndim != 2 or head_weight.shape[1] != x.shape[-1]:
        raise ValueError(f"head_weight 必须是 [V,{x.shape[-1]}]")
    if x.device != head_weight.device:
        raise ValueError("x 与 head_weight 必须位于同一设备")
    if isinstance(output_chunk_size, bool) or not isinstance(output_chunk_size, int):
        raise TypeError("output_chunk_size 必须是整数")
    if output_chunk_size <= 0:
        raise ValueError("output_chunk_size 必须大于零")

    projected = x if full_logits else x[:, -1]
    projected_f32 = projected.float()
    chunks = [
        F.linear(projected_f32, head_weight[start : start + output_chunk_size].float())
        for start in range(0, head_weight.shape[0], output_chunk_size)
    ]
    if not chunks:
        raise ValueError("head_weight 的词表维不能为空")
    return torch.cat(chunks, dim=-1)


def native_final_logits(
    x: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    norm_weight: torch.Tensor,
    head_weight: torch.Tensor,
    *,
    norm_eps: float = OFFICIAL_NORM_EPS,
    hc_eps: float = OFFICIAL_HC_EPS,
    full_logits: bool = False,
    output_chunk_size: int = 4096,
    require_bf16_hidden: bool = True,
    enforce_official_shape: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """执行固定最终路径，返回 ``(FP32 logits, normalized, HC pre)``。"""

    if require_bf16_hidden and x.dtype != torch.bfloat16:
        raise TypeError("原生最终路径要求 x 为 torch.bfloat16")
    if enforce_official_shape:
        if x.ndim != 4 or tuple(x.shape[-2:]) != (OFFICIAL_HC_MULT, OFFICIAL_HIDDEN_SIZE):
            raise ValueError("原生最终路径要求 x 末两维为 [4,4096]")
        if tuple(head_weight.shape) != (OFFICIAL_VOCAB_SIZE, OFFICIAL_HIDDEN_SIZE):
            raise ValueError("单卡原生最终路径要求 head_weight 为 [129280,4096]")
    reduced, pre = hc_head_reduce(
        x,
        hc_head_fn,
        hc_head_scale,
        hc_head_base,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
    )
    normalized = official_rms_norm(reduced, norm_weight, norm_eps=norm_eps)
    logits = bf16_checkpoint_head_logits(
        normalized,
        head_weight,
        full_logits=full_logits,
        output_chunk_size=output_chunk_size,
    )
    return logits, normalized, pre
