"""Structured knowledge-donor plan for compiling ColorLM v4 tensors."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .architecture import ColorLMV4Config, colorlm_v4_budget


@dataclass(frozen=True)
class TensorTransfer:
    source: str | None
    target: str
    operation: str
    layer: int | None = None


@dataclass(frozen=True)
class LayerPlan:
    index: int
    temporal_core: str
    experts: int
    active_experts: int
    transfers: tuple[TensorTransfer, ...]


@dataclass(frozen=True)
class DonorPlan:
    format: str
    version: int
    donor_architecture: str
    final_architecture: str
    runtime_requires_donor: bool
    layers: tuple[LayerPlan, ...]
    global_transfers: tuple[TensorTransfer, ...]
    maximum_model_bytes: int
    allocated_model_bytes: int

    def validate(self) -> None:
        if self.runtime_requires_donor:
            raise ValueError("最终运行时不能依赖供体")
        if self.final_architecture != "colorlm-state-moe-v4":
            raise ValueError("最终架构标识错误")
        if len(self.layers) != 40:
            raise ValueError("v4 必须包含 40 个状态层")
        delta_count = sum(layer.temporal_core == "color_delta" for layer in self.layers)
        kernel_count = sum(layer.temporal_core == "color_kernel" for layer in self.layers)
        if (delta_count, kernel_count) != (30, 10):
            raise ValueError("v4 必须包含 30 个 ColorDelta 和 10 个 ColorKernel")
        if self.allocated_model_bytes > self.maximum_model_bytes:
            raise ValueError("供体编译计划超过模型大小上限")
        for layer in self.layers:
            if layer.experts != 256 or layer.active_experts != 8:
                raise ValueError(f"第 {layer.index} 层专家配置错误")
            for transfer in layer.transfers:
                if ".attn" in transfer.target or "transformer" in transfer.target.lower():
                    raise ValueError(f"目标仍包含旧注意力张量: {transfer.target}")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n"

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def _transfer(
    source: str | None,
    target: str,
    operation: str,
    layer: int,
) -> TensorTransfer:
    return TensorTransfer(source, target, operation, layer)


def _common_layer_transfers(layer: int) -> list[TensorTransfer]:
    source = f"blk.{layer}"
    target = f"layers.{layer}"
    return [
        _transfer(f"{source}.attn_norm.weight", f"{target}.state_norm", "copy", layer),
        _transfer(
            f"{source}.post_attention_norm.weight",
            f"{target}.expert_norm",
            "copy",
            layer,
        ),
        _transfer(
            f"{source}.ffn_gate_inp.weight",
            f"{target}.router.weight",
            "copy_f16",
            layer,
        ),
        _transfer(
            f"{source}.ffn_gate_exps.weight",
            f"{target}.experts.gate.colorq3",
            "colorq3_vector_quantize",
            layer,
        ),
        _transfer(
            f"{source}.ffn_up_exps.weight",
            f"{target}.experts.up.colorq3",
            "colorq3_vector_quantize",
            layer,
        ),
        _transfer(
            f"{source}.ffn_down_exps.weight",
            f"{target}.experts.down.colorq3",
            "colorq3_vector_quantize",
            layer,
        ),
        _transfer(
            f"{source}.ffn_gate_inp_shexp.weight",
            f"{target}.shared.gate",
            "copy_f16",
            layer,
        ),
        _transfer(
            f"{source}.ffn_gate_shexp.weight",
            f"{target}.shared.ffn_gate",
            "quantize_q5",
            layer,
        ),
        _transfer(
            f"{source}.ffn_up_shexp.weight",
            f"{target}.shared.ffn_up",
            "quantize_q5",
            layer,
        ),
        _transfer(
            f"{source}.ffn_down_shexp.weight",
            f"{target}.shared.ffn_down",
            "quantize_q5",
            layer,
        ),
        _transfer(None, f"{target}.fast_state.zero", "initialize_zero", layer),
    ]


def _delta_transfers(layer: int) -> list[TensorTransfer]:
    source = f"blk.{layer}"
    target = f"layers.{layer}.delta"
    names = [
        ("attn_qkv.weight", "qkv", "copy_q5"),
        ("attn_gate.weight", "output_gate", "copy_q5"),
        ("ssm_conv1d.weight", "conv", "copy_f16"),
        ("ssm_dt.bias", "time_bias", "copy_f16"),
        ("ssm_a", "decay", "copy_f16"),
        ("ssm_beta.weight", "beta", "copy_f16"),
        ("ssm_alpha.weight", "alpha", "copy_f16"),
        ("ssm_norm.weight", "norm", "copy_f16"),
        ("ssm_out.weight", "output", "copy_q5"),
    ]
    return [
        _transfer(f"{source}.{old}", f"{target}.{new}", operation, layer)
        for old, new, operation in names
    ]


def _kernel_transfers(layer: int, donor_hash: str) -> list[TensorTransfer]:
    source = f"blk.{layer}"
    target = f"layers.{layer}.kernel"
    return [
        _transfer(
            f"{source}.attn_q.weight",
            f"{target}.query_and_gate",
            "split_interleaved_query_gate",
            layer,
        ),
        _transfer(f"{source}.attn_k.weight", f"{target}.key", "copy_q5", layer),
        _transfer(f"{source}.attn_v.weight", f"{target}.value", "copy_q5", layer),
        _transfer(f"{source}.attn_output.weight", f"{target}.output", "copy_q5", layer),
        _transfer(f"{source}.attn_q_norm.weight", f"{target}.query_norm", "copy_f16", layer),
        _transfer(f"{source}.attn_k_norm.weight", f"{target}.key_norm", "copy_f16", layer),
        _transfer(
            None,
            f"{target}.positive_features",
            f"orthogonal_features_sha256:{donor_hash}:layer:{layer}",
            layer,
        ),
        _transfer(None, f"{target}.state.zero", "initialize_zero", layer),
        _transfer(None, f"{target}.normalizer.zero", "initialize_zero", layer),
    ]


def build_qwen35_donor_plan(
    *,
    donor_hash: str = "pending",
    config: ColorLMV4Config | None = None,
) -> DonorPlan:
    """Build a deterministic migration plan; no donor tensor is loaded here."""

    config = config or ColorLMV4Config()
    config.validate()
    layers: list[LayerPlan] = []
    for index in range(config.layers):
        is_kernel = (index + 1) % 4 == 0
        temporal_core = "color_kernel" if is_kernel else "color_delta"
        transfers = _common_layer_transfers(index)
        if is_kernel:
            transfers.extend(_kernel_transfers(index, donor_hash))
        else:
            transfers.extend(_delta_transfers(index))
        layers.append(
            LayerPlan(
                index=index,
                temporal_core=temporal_core,
                experts=config.experts,
                active_experts=config.active_experts,
                transfers=tuple(transfers),
            )
        )

    budget = colorlm_v4_budget()
    plan = DonorPlan(
        format="CLM-Donor-Plan",
        version=1,
        donor_architecture="qwen35moe",
        final_architecture="colorlm-state-moe-v4",
        runtime_requires_donor=False,
        layers=tuple(layers),
        global_transfers=(
            TensorTransfer("token_embd.weight", "embedding.weight", "quantize_q5"),
            TensorTransfer("output_norm.weight", "head.norm", "copy_f16"),
            TensorTransfer("output.weight", "head.weight", "quantize_q5"),
        ),
        maximum_model_bytes=config.max_model_bytes,
        allocated_model_bytes=budget.total_bytes,
    )
    plan.validate()
    return plan
