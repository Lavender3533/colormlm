"""DeepSeek-V4 共享 KV 稀疏注意力的纯 PyTorch 参考语义。"""

from __future__ import annotations

import math
from typing import Optional

import torch


def _tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    return value


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    query_chunk_size: int = 16,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """计算带 attention-sink 分母项的共享 KV 稀疏注意力。

    形状与官方 kernel 一致：``q=[B,M,H,D]``、``kv=[B,N,D]``、
    ``topk_indices=[B,M,K]``。``-1`` 是唯一的 padding sentinel；其余负数或
    ``>=N`` 的索引会被硬拒绝。sink 只有 logit，没有 value，因此只增加 softmax
    分母。query 分块限制参考实现的临时 gather 内存。
    """

    q = _tensor(q, "q")
    kv = _tensor(kv, "kv")
    attn_sink = _tensor(attn_sink, "attn_sink")
    topk_indices = _tensor(topk_indices, "topk_indices")
    if q.ndim != 4 or kv.ndim != 3:
        raise ValueError("q 必须为 [B,M,H,D]，kv 必须为 [B,N,D]")
    if not q.is_floating_point() or not kv.is_floating_point() or not attn_sink.is_floating_point():
        raise TypeError("q、kv 与 attn_sink 必须是浮点张量")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_indices 必须是 int32 或 int64")
    b, m, h, d = q.shape
    if kv.shape[0] != b or kv.shape[2] != d:
        raise ValueError("kv 的 batch/head_dim 必须与 q 一致")
    n = kv.shape[1]
    if tuple(attn_sink.shape) != (h,):
        raise ValueError(f"attn_sink 形状必须是 {(h,)}")
    if topk_indices.ndim != 3 or tuple(topk_indices.shape[:2]) != (b, m):
        raise ValueError(f"topk_indices 前两维必须是 {(b, m)}")
    if topk_indices.shape[-1] == 0:
        raise ValueError("topk_indices 的 K 不能为 0")
    if q.device != kv.device or q.device != attn_sink.device or q.device != topk_indices.device:
        raise ValueError("所有输入必须位于同一设备")
    if not isinstance(query_chunk_size, int) or isinstance(query_chunk_size, bool) or query_chunk_size <= 0:
        raise ValueError("query_chunk_size 必须是正整数")
    if softmax_scale is None:
        softmax_scale = d ** -0.5
    if not isinstance(softmax_scale, (int, float)) or isinstance(softmax_scale, bool) or not math.isfinite(float(softmax_scale)):
        raise ValueError("softmax_scale 必须是有限数")
    if not bool(torch.isfinite(attn_sink).all().item()):
        raise ValueError("attn_sink 包含非有限值")

    invalid = (topk_indices < -1) | (topk_indices >= n)
    if bool(invalid.any().item()):
        bad = int(topk_indices[invalid][0].item())
        raise IndexError(f"topk_indices 含越界索引 {bad}；只允许 -1 或 [0,{n})")
    if output_dtype is None:
        output_dtype = q.dtype
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("output_dtype 必须是常规浮点 dtype")

    q_f32 = q.float()
    kv_f32 = kv.float()
    sink_f32 = attn_sink.float().reshape(1, 1, h, 1)
    result = torch.empty((b, m, h, d), dtype=torch.float32, device=q.device)
    batch = torch.arange(b, device=q.device).reshape(b, 1, 1)

    for start in range(0, m, query_chunk_size):
        end = min(start + query_chunk_size, m)
        indices = topk_indices[:, start:end].long()
        padding = indices == -1
        safe_indices = indices.clamp_min(0)
        gathered = kv_f32[batch, safe_indices]
        scores = torch.einsum("bmhd,bmkd->bmhk", q_f32[:, start:end], gathered)
        scores = scores * float(softmax_scale)
        scores = scores.masked_fill(padding.unsqueeze(2), -torch.inf)
        sink = sink_f32.expand(b, end - start, h, 1)
        probabilities = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
        result[:, start:end] = torch.einsum("bmhk,bmkd->bmhd", probabilities, gathered)

    return result.to(output_dtype)
