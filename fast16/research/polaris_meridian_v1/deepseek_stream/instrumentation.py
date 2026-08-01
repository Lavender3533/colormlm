"""对官方 DeepSeek inference Transformer 注入 L39-L42 只读采集探针。"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

try:
    from .capture_io import HC_MULT, HIDDEN, LAYERS, TOP_K
except ImportError:
    from capture_io import HC_MULT, HIDDEN, LAYERS, TOP_K


@dataclass
class _Patch:
    owner: Any
    name: str
    had_instance_value: bool
    instance_value: Any


class OfficialDeepSeekStepProbe:
    """采集官方 `inference/model.py` 的真实张量；不创建或替代任何模型计算。"""

    def __init__(self, model: Any):
        self.model = model
        self.layers = _find_layers(model)
        if len(self.layers) < max(LAYERS) + 1:
            raise ValueError(f"模型层数不足: {len(self.layers)}")
        self._patches: list[_Patch] = []
        self._hooks: list[Any] = []
        self._active = False
        self._step: dict[int, dict[str, Any]] = {}

    def attach(self) -> "OfficialDeepSeekStepProbe":
        if self._active:
            raise RuntimeError("probe 已 attach")
        for layer_id in LAYERS:
            block = self.layers[layer_id]
            if not all(hasattr(block, name) for name in ("hc_pre", "hc_attn_fn", "hc_ffn_fn", "ffn")):
                raise TypeError(f"L{layer_id} 不是官方 mHC Block 接口")
            gate = getattr(block.ffn, "gate", None)
            if gate is None or not hasattr(gate, "forward"):
                raise TypeError(f"L{layer_id} 缺少官方 MoE gate")
            self._patch_hc_pre(layer_id, block)
            self._patch_gate(layer_id, gate)
            self._hooks.append(block.register_forward_hook(self._forward_hook(layer_id)))
        self._active = True
        self.begin_step()
        return self

    def begin_step(self) -> None:
        self._step = {layer: {} for layer in LAYERS}

    def materialize(self) -> dict[int, dict[str, bytes]]:
        """在一次完整 forward 后把少量探针张量搬到 host，并返回 CNOB payload。"""

        if not self._active:
            raise RuntimeError("probe 尚未 attach")
        result: dict[int, dict[str, bytes]] = {}
        for layer in LAYERS:
            state = self._step[layer]
            missing = {"streams", "attn_post", "attn_comb", "ffn_post", "ffn_comb", "router_ids", "router_weights"} - set(state)
            if missing:
                raise RuntimeError(f"L{layer} probe 数据不完整: {sorted(missing)}")
            streams = _last_token(state["streams"], expected_rank=4)
            if tuple(streams.shape) != (HC_MULT, HIDDEN):
                raise ValueError(f"L{layer} mHC streams shape 错误: {tuple(streams.shape)}")
            mean = streams.mean(dim=0)
            attn_mix = _post_comb(state["attn_post"], state["attn_comb"])
            ffn_mix = _post_comb(state["ffn_post"], state["ffn_comb"])
            router_ids = state["router_ids"].reshape(-1, state["router_ids"].shape[-1])[-1]
            router_weights = state["router_weights"].reshape(-1, state["router_weights"].shape[-1])[-1]
            if int(router_ids.numel()) != TOP_K or int(router_weights.numel()) != TOP_K:
                raise ValueError(f"L{layer} router top-k 不是 {TOP_K}")
            result[layer] = {
                "hidden_mean_bf16": _tensor_bytes(mean, "bf16"),
                "mhc_streams_bf16": _tensor_bytes(streams, "bf16"),
                "mhc_attention_post_comb": _tensor_bytes(attn_mix, "f32"),
                "mhc_ffn_post_comb": _tensor_bytes(ffn_mix, "f32"),
                "router_topk_ids": _tensor_bytes(router_ids, "i32"),
                "router_topk_weights": _tensor_bytes(router_weights, "f32"),
            }
        return result

    def close(self) -> None:
        for hook in reversed(self._hooks):
            hook.remove()
        self._hooks.clear()
        for patch in reversed(self._patches):
            if patch.had_instance_value:
                setattr(patch.owner, patch.name, patch.instance_value)
            else:
                try:
                    delattr(patch.owner, patch.name)
                except AttributeError:
                    pass
        self._patches.clear()
        self._active = False

    def __enter__(self) -> "OfficialDeepSeekStepProbe":
        return self.attach()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _save_patch(self, owner: Any, name: str) -> None:
        namespace = getattr(owner, "__dict__", {})
        self._patches.append(_Patch(owner, name, name in namespace, namespace.get(name)))

    def _patch_hc_pre(self, layer_id: int, block: Any) -> None:
        original = block.hc_pre
        self._save_patch(block, "hc_pre")

        def wrapped(instance: Any, x: Any, hc_fn: Any, hc_scale: Any, hc_base: Any):
            y, post, comb = original(x, hc_fn, hc_scale, hc_base)
            if _same_parameter(hc_fn, instance.hc_attn_fn):
                branch = "attn"
            elif _same_parameter(hc_fn, instance.hc_ffn_fn):
                branch = "ffn"
            else:
                raise RuntimeError(f"L{layer_id} hc_pre 收到未知 hc_fn")
            self._step[layer_id][f"{branch}_post"] = post.detach()
            self._step[layer_id][f"{branch}_comb"] = comb.detach()
            return y, post, comb

        block.hc_pre = types.MethodType(wrapped, block)

    def _patch_gate(self, layer_id: int, gate: Any) -> None:
        original = gate.forward
        self._save_patch(gate, "forward")

        def wrapped(instance: Any, x: Any, input_ids: Any = None):
            weights, indices = original(x, input_ids)
            self._step[layer_id]["router_weights"] = weights.detach()
            self._step[layer_id]["router_ids"] = indices.detach()
            return weights, indices

        gate.forward = types.MethodType(wrapped, gate)

    def _forward_hook(self, layer_id: int):
        def hook(module: Any, args: Any, output: Any) -> None:
            if not hasattr(output, "detach"):
                raise TypeError(f"L{layer_id} forward output 不是 tensor")
            self._step[layer_id]["streams"] = output.detach()

        return hook


def _find_layers(model: Any) -> Any:
    candidates = [model, getattr(model, "model", None), getattr(model, "transformer", None)]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
    raise TypeError("找不到官方 Transformer.layers")


def _same_parameter(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        return int(left.data_ptr()) == int(right.data_ptr())
    except (AttributeError, RuntimeError):
        return False


def _last_token(value: Any, *, expected_rank: int) -> Any:
    if int(value.dim()) != expected_rank:
        raise ValueError(f"tensor rank={value.dim()}，期望 {expected_rank}")
    if int(value.shape[0]) != 1:
        raise ValueError("第一阶段只允许 batch=1，避免 record/token 对齐歧义")
    return value[0, -1]


def _post_comb(post: Any, comb: Any) -> Any:
    post_value = _last_token(post, expected_rank=3).reshape(-1)
    comb_value = _last_token(comb, expected_rank=4).reshape(-1)
    if int(post_value.numel()) != HC_MULT or int(comb_value.numel()) != HC_MULT * HC_MULT:
        raise ValueError("mHC post/comb shape 错误")
    import torch

    return torch.cat([post_value, comb_value], dim=0)


def _tensor_bytes(value: Any, encoding: str) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("CNOB v2 当前只支持 little-endian host")
    import torch

    tensor = value.detach().contiguous().cpu()
    if encoding == "bf16":
        tensor = tensor.to(torch.bfloat16).contiguous().view(torch.uint16)
    elif encoding == "f32":
        tensor = tensor.to(torch.float32).contiguous()
    elif encoding == "i32":
        tensor = tensor.to(torch.int32).contiguous()
    else:
        raise ValueError(f"未知 tensor encoding={encoding}")
    return tensor.numpy().tobytes(order="C")
