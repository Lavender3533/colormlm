"""DeepSeek-V4 Hyper-Connection 拆分与 Sinkhorn 参考语义。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _floating_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} 必须是浮点张量")
    return value


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 HC logits 拆成 pre/post/comb，并复现官方 Sinkhorn 顺序。

    所有计算固定为 FP32。``comb`` 先做稳定 row-softmax 并加 ``eps``，再归一化
    列；后续 ``sinkhorn_iters - 1`` 轮依次归一化行和列。
    """

    mixes = _floating_tensor(mixes, "mixes")
    hc_scale = _floating_tensor(hc_scale, "hc_scale")
    hc_base = _floating_tensor(hc_base, "hc_base")
    if not isinstance(hc_mult, int) or isinstance(hc_mult, bool) or hc_mult <= 0:
        raise ValueError("hc_mult 必须是正整数")
    if not isinstance(sinkhorn_iters, int) or isinstance(sinkhorn_iters, bool) or sinkhorn_iters < 1:
        raise ValueError("sinkhorn_iters 必须是至少 1 的整数")
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps 必须是有限正数")
    mix_hc = (2 + hc_mult) * hc_mult
    if mixes.ndim < 1 or mixes.shape[-1] != mix_hc:
        raise ValueError(f"mixes 最后一维必须是 {mix_hc}")
    if tuple(hc_scale.shape) != (3,):
        raise ValueError("hc_scale 形状必须是 [3]")
    if tuple(hc_base.shape) != (mix_hc,):
        raise ValueError(f"hc_base 形状必须是 [{mix_hc}]")
    if mixes.device != hc_scale.device or mixes.device != hc_base.device:
        raise ValueError("mixes、hc_scale 与 hc_base 必须位于同一设备")
    if not bool(torch.isfinite(mixes).all().item()):
        raise ValueError("mixes 包含非有限值")
    if not bool(torch.isfinite(hc_scale).all().item()) or not bool(torch.isfinite(hc_base).all().item()):
        raise ValueError("hc_scale/hc_base 包含非有限值")

    mixes_f32 = mixes.float()
    scale_f32 = hc_scale.float()
    base_f32 = hc_base.float()
    pre_logits, post_logits, comb_logits = torch.split(
        mixes_f32, (hc_mult, hc_mult, hc_mult * hc_mult), dim=-1
    )
    pre = torch.sigmoid(pre_logits * scale_f32[0] + base_f32[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        post_logits * scale_f32[1] + base_f32[hc_mult : 2 * hc_mult]
    )
    comb = comb_logits.reshape(*mixes.shape[:-1], hc_mult, hc_mult)
    comb = comb * scale_f32[2] + base_f32[2 * hc_mult :].reshape(hc_mult, hc_mult)

    row_max = comb.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """复现官方 block 的 HC reduce：四流 hidden -> 单流 branch 输入。"""

    x = _floating_tensor(x, "x")
    hc_fn = _floating_tensor(hc_fn, "hc_fn")
    if x.ndim != 4:
        raise ValueError("x 必须是 [B,S,HC,D]")
    if not math.isfinite(float(norm_eps)) or norm_eps <= 0:
        raise ValueError("norm_eps 必须是有限正数")
    hc_mult = x.shape[-2]
    hidden = x.shape[-1]
    mix_hc = (2 + hc_mult) * hc_mult
    if tuple(hc_fn.shape) != (mix_hc, hc_mult * hidden):
        raise ValueError(f"hc_fn 应为 {(mix_hc, hc_mult * hidden)}")
    if hc_fn.device != x.device:
        raise ValueError("x 与 hc_fn 必须位于同一设备")
    shape = x.shape
    original_dtype = x.dtype
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn.float()) * rsqrt
    pre, post, comb = hc_split_sinkhorn(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=hc_eps,
    )
    reduced = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
    return reduced.to(original_dtype), post, comb


def hc_post(
    branch: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """复现官方 ``sum(dim=2)`` HC expand/merge。"""

    branch = _floating_tensor(branch, "branch")
    residual = _floating_tensor(residual, "residual")
    post = _floating_tensor(post, "post")
    comb = _floating_tensor(comb, "comb")
    if branch.ndim != 3 or residual.ndim != 4:
        raise ValueError("branch/residual 必须分别是 [B,S,D] 和 [B,S,HC,D]")
    b, s, hidden = branch.shape
    if residual.shape[:2] != (b, s) or residual.shape[-1] != hidden:
        raise ValueError("branch 与 residual 的 batch/seq/hidden 不匹配")
    hc_mult = residual.shape[-2]
    if tuple(post.shape) != (b, s, hc_mult):
        raise ValueError(f"post 应为 {(b, s, hc_mult)}")
    if tuple(comb.shape) != (b, s, hc_mult, hc_mult):
        raise ValueError(f"comb 应为 {(b, s, hc_mult, hc_mult)}")
    if len({branch.device, residual.device, post.device, comb.device}) != 1:
        raise ValueError("HC post 的所有输入必须位于同一设备")
    merged = post.unsqueeze(-1) * branch.unsqueeze(-2)
    merged = merged + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return merged.type_as(branch)
