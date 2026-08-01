"""Stage-level roofline and shortest dependency schedule for one S14 token."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import math
from pathlib import Path
from typing import Any

from .anchors import StageAnchors, load_real_anchors


GIB = 1024**3
DEFAULT_VRAM_BYTES = 8 * GIB
DEFAULT_RAM_BYTES = 32 * GIB
DEFAULT_SSD_BYTES = 118 * GIB
DEFAULT_PCIE_BYTES_PER_SECOND = 22.03e9


class CommandBufferMode(str, Enum):
    PER_MEASURED_GPU_DISPATCH = "per_measured_gpu_dispatch"
    ROUTE_SPLIT_PER_LAYER = "route_split_per_layer"
    RESIDENT_PER_LAYER = "resident_per_layer"
    WHOLE_TOKEN_PERSISTENT = "whole_token_persistent"


@dataclass(frozen=True)
class SimulationConfig:
    target_tps: float
    expert_cache_hit_rate: float
    pcie_bytes_per_second: float = DEFAULT_PCIE_BYTES_PER_SECOND
    vram_bytes: int = DEFAULT_VRAM_BYTES
    batch_size: int = 1
    batch_efficiency: float = 1.0
    kernel_fusion_speedup: float = 1.0
    attention_remainder_ms_per_layer: float = 0.0
    hc_ms_per_layer: float = 0.0
    router_ms_per_layer: float = 0.0
    norm_head_ms_per_token: float = 0.0
    command_buffer_mode: CommandBufferMode = CommandBufferMode.ROUTE_SPLIT_PER_LAYER
    submit_overhead_us: float = 0.0

    @property
    def measured_kernel_speedup(self) -> float:
        return self.batch_size * self.batch_efficiency * self.kernel_fusion_speedup

    def validate(self) -> None:
        if self.target_tps <= 0 or self.pcie_bytes_per_second <= 0:
            raise ValueError("target_tps 和 PCIe 带宽必须为正")
        if not 0 <= self.expert_cache_hit_rate <= 1:
            raise ValueError("expert_cache_hit_rate 必须在 [0,1]")
        if self.batch_size <= 0 or not 0 < self.batch_efficiency <= 1:
            raise ValueError("batch_size/efficiency 非法")
        if self.kernel_fusion_speedup <= 0:
            raise ValueError("kernel_fusion_speedup 必须为正")
        values = (
            self.attention_remainder_ms_per_layer,
            self.hc_ms_per_layer,
            self.router_ms_per_layer,
            self.norm_head_ms_per_token,
            self.submit_overhead_us,
        )
        if any(value < 0 for value in values):
            raise ValueError("阶段时间不能为负")


@dataclass(frozen=True)
class StageEvent:
    name: str
    category: str
    resource: str
    layer: int | None
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["duration_ms"] = self.duration_ms
        return value


@dataclass(frozen=True)
class TokenSimulation:
    config: SimulationConfig
    stage_work_ms: dict[str, float]
    events: tuple[StageEvent, ...]
    gpu_copy_compute_critical_ms: float
    host_dispatch_work_ms: float
    sequential_latency_ms: float
    ideal_pipeline_period_ms: float
    ideal_pipeline_throughput_tps: float
    target_period_ms: float
    target_met_under_ideal_pipeline: bool
    shared_copy_overlap_credit_ms: float
    expert_miss_bytes: float
    command_buffers_per_token: int

    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "config": {
                **asdict(self.config),
                "command_buffer_mode": self.config.command_buffer_mode.value,
                "measured_kernel_speedup": self.config.measured_kernel_speedup,
            },
            "stage_work_ms": self.stage_work_ms,
            "gpu_copy_compute_critical_ms": self.gpu_copy_compute_critical_ms,
            "host_dispatch_work_ms": self.host_dispatch_work_ms,
            "sequential_latency_ms": self.sequential_latency_ms,
            "ideal_pipeline_period_ms": self.ideal_pipeline_period_ms,
            "ideal_pipeline_throughput_tps": self.ideal_pipeline_throughput_tps,
            "target_period_ms": self.target_period_ms,
            "target_met_under_ideal_pipeline": self.target_met_under_ideal_pipeline,
            "shared_copy_overlap_credit_ms": self.shared_copy_overlap_credit_ms,
            "expert_miss_bytes": self.expert_miss_bytes,
            "command_buffers_per_token": self.command_buffers_per_token,
        }
        if include_events:
            value["events"] = [event.to_dict() for event in self.events]
        return value


def command_buffers_per_token(anchors: StageAnchors, mode: CommandBufferMode) -> int:
    if mode is CommandBufferMode.PER_MEASURED_GPU_DISPATCH:
        return anchors.layer_count * (anchors.top6_dispatches + anchors.wq_a_dispatches)
    if mode is CommandBufferMode.ROUTE_SPLIT_PER_LAYER:
        return anchors.layer_count * 2 + 1
    if mode is CommandBufferMode.RESIDENT_PER_LAYER:
        return anchors.layer_count + 1
    if mode is CommandBufferMode.WHOLE_TOKEN_PERSISTENT:
        return 1
    raise AssertionError(mode)


def simulate_token(anchors: StageAnchors, config: SimulationConfig) -> TokenSimulation:
    """Schedule exact dependencies; SSD DMA may overlap only the shared expert."""

    config.validate()
    speedup = config.measured_kernel_speedup
    routed_ms = anchors.six_routed_envelope_ms_per_layer / speedup
    shared_ms = anchors.shared_and_batch_residual_ms_per_layer / speedup
    attention_ms = anchors.wq_a_linear_ms / speedup + config.attention_remainder_ms_per_layer
    hc_quarter_ms = config.hc_ms_per_layer / 4.0
    miss_bytes_per_layer = anchors.routed_bytes_per_layer * (1.0 - config.expert_cache_hit_rate)
    miss_ms = miss_bytes_per_layer / config.pcie_bytes_per_second * 1000.0
    events: list[StageEvent] = []
    totals = {
        "attention": 0.0,
        "hc": 0.0,
        "router": 0.0,
        "six_routed": 0.0,
        "shared": 0.0,
        "norm_head": config.norm_head_ms_per_token,
        "ssd_miss": 0.0,
        "dispatch_overhead": 0.0,
    }

    def add(name: str, category: str, resource: str, layer: int | None, start: float, duration: float) -> float:
        end = start + duration
        events.append(StageEvent(name, category, resource, layer, start, end))
        totals[category] += duration
        return end

    cursor = 0.0
    overlap_credit = 0.0
    for layer in anchors.layers:
        cursor = add("hc_attn_pre", "hc", "gpu_compute", layer, cursor, hc_quarter_ms)
        cursor = add("attention", "attention", "gpu_compute", layer, cursor, attention_ms)
        cursor = add("hc_attn_post", "hc", "gpu_compute", layer, cursor, hc_quarter_ms)
        cursor = add("hc_ffn_pre", "hc", "gpu_compute", layer, cursor, hc_quarter_ms)
        cursor = add("native_top6_router", "router", "gpu_compute", layer, cursor, config.router_ms_per_layer)

        branch_start = cursor
        copy_end = add("expert_page_ssd_miss", "ssd_miss", "copy_engine", layer, branch_start, miss_ms)
        shared_end = add("shared_expert", "shared", "gpu_compute", layer, branch_start, shared_ms)
        overlap_credit += min(miss_ms, shared_ms)
        cursor = max(copy_end, shared_end)
        cursor = add("six_routed_experts", "six_routed", "gpu_compute", layer, cursor, routed_ms)
        cursor = add("hc_ffn_post", "hc", "gpu_compute", layer, cursor, hc_quarter_ms)

    cursor = add("final_hc_norm_head", "norm_head", "gpu_compute", None, cursor, config.norm_head_ms_per_token)
    command_buffers = command_buffers_per_token(anchors, config.command_buffer_mode)
    host_ms = command_buffers * config.submit_overhead_us / 1000.0
    totals["dispatch_overhead"] = host_ms
    sequential = cursor + host_ms
    # Optimistic steady state: a host submission worker runs ahead of the GPU.
    pipeline_period = max(cursor, host_ms)
    target_period = 1000.0 / config.target_tps
    return TokenSimulation(
        config=config,
        stage_work_ms=totals,
        events=tuple(events),
        gpu_copy_compute_critical_ms=cursor,
        host_dispatch_work_ms=host_ms,
        sequential_latency_ms=sequential,
        ideal_pipeline_period_ms=pipeline_period,
        ideal_pipeline_throughput_tps=1000.0 / pipeline_period,
        target_period_ms=target_period,
        target_met_under_ideal_pipeline=pipeline_period <= target_period + 1e-12,
        shared_copy_overlap_credit_ms=overlap_credit,
        expert_miss_bytes=miss_bytes_per_layer * anchors.layer_count,
        command_buffers_per_token=command_buffers,
    )


def solve_required_expert_hit_rate(anchors: StageAnchors, config: SimulationConfig) -> float | None:
    """Return the minimum hit rate for the supplied stage scenario."""

    if simulate_token(anchors, replace(config, expert_cache_hit_rate=1.0)).target_met_under_ideal_pipeline is False:
        return None
    if simulate_token(anchors, replace(config, expert_cache_hit_rate=0.0)).target_met_under_ideal_pipeline:
        return 0.0
    low = 0.0
    high = 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if simulate_token(anchors, replace(config, expert_cache_hit_rate=middle)).target_met_under_ideal_pipeline:
            high = middle
        else:
            low = middle
    return high


def _minimum_kernel_speedup(anchors: StageAnchors, target_tps: float) -> float:
    base = SimulationConfig(target_tps=target_tps, expert_cache_hit_rate=1.0)
    if simulate_token(anchors, base).target_met_under_ideal_pipeline:
        return 1.0
    low = 1.0
    high = 2.0
    while not simulate_token(anchors, replace(base, kernel_fusion_speedup=high)).target_met_under_ideal_pipeline:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if simulate_token(anchors, replace(base, kernel_fusion_speedup=middle)).target_met_under_ideal_pipeline:
            high = middle
        else:
            low = middle
    return high


def _stage_budgets(anchors: StageAnchors, target_tps: float) -> dict[str, Any]:
    period = 1000.0 / target_tps
    attention = anchors.s14_wq_a_floor_ms
    routed = anchors.layer_count * anchors.six_routed_envelope_ms_per_layer
    shared = anchors.layer_count * anchors.shared_and_batch_residual_ms_per_layer
    known = attention + routed + shared
    joint_pool = period - known
    floors = {
        "attention": attention,
        "hc": 0.0,
        "router": 0.0,
        "six_routed": routed,
        "shared": shared,
        "norm_head": 0.0,
        "ssd_miss": 0.0,
        "dispatch_overhead": 0.0,
    }
    rows: dict[str, Any] = {}
    for stage, floor in floors.items():
        other_floors = known - floor
        rows[stage] = {
            "current_measured_or_zero_floor_ms": floor,
            "exclusive_sequential_max_ms_if_all_other_unmeasured_stages_are_zero": period - other_floors,
            "measurement_status": (
                "measured necessary sub-operation/envelope"
                if stage in {"attention", "six_routed", "shared"}
                else "unmeasured; zero is a roofline placeholder, not an estimate"
            ),
        }
    rows["ssd_miss"]["exclusive_pipeline_max_ms_with_shared_overlap"] = period - attention - routed
    return {
        "target_tps": target_tps,
        "token_period_ms": period,
        "current_known_gpu_anchor_ms": known,
        "joint_unmeasured_pool_ms_at_100_percent_expert_hit": joint_pool,
        "stages": rows,
    }


def _target_report(anchors: StageAnchors, target_tps: float) -> dict[str, Any]:
    floor_config = SimulationConfig(target_tps=target_tps, expert_cache_hit_rate=1.0)
    floor = simulate_token(anchors, floor_config)
    required_hit = solve_required_expert_hit_rate(anchors, floor_config)
    minimum_speedup = _minimum_kernel_speedup(anchors, target_tps)
    rescue_config = replace(floor_config, kernel_fusion_speedup=minimum_speedup * (1.0 + 1e-12))
    rescue_hit = solve_required_expert_hit_rate(anchors, rescue_config)
    period = 1000.0 / target_tps
    remaining_at_hit = period - floor.gpu_copy_compute_critical_ms

    command_modes: dict[str, Any] = {}
    for mode in CommandBufferMode:
        count = command_buffers_per_token(anchors, mode)
        command_modes[mode.value] = {
            "command_buffers_per_token": count,
            "maximum_average_submit_overhead_us_at_current_kernels_100_percent_hit": (
                max(0.0, remaining_at_hit) * 1000.0 / count if remaining_at_hit >= 0 else None
            ),
        }

    batch_rows = []
    for batch_size in (1, 2, 4, 8):
        required_efficiency = minimum_speedup / batch_size
        batch_rows.append(
            {
                "batch_size": batch_size,
                "minimum_fraction_of_ideal_linear_batch_speedup": required_efficiency,
                "possible_under_ideal_batching": required_efficiency <= 1.0,
            }
        )
    return {
        "target_tps": target_tps,
        "hard_stage_budget": _stage_budgets(anchors, target_tps),
        "current_kernel_100_percent_hit_floor": floor.to_dict(include_events=False),
        "required_expert_cache_hit_rate_current_kernels": required_hit,
        "minimum_measured_kernel_speedup_at_100_percent_hit": minimum_speedup,
        "required_hit_rate_at_zero_unknown_budget_and_minimum_speedup": rescue_hit,
        "minimum_ideal_batch_size": math.ceil(minimum_speedup - 1e-12),
        "batch_requirements": batch_rows,
        "command_buffer_requirements": command_modes,
    }


def _cold_counterexample(anchors: StageAnchors, required_hit_20: float | None) -> dict[str, Any]:
    observed = anchors.two_token_previous_only_hit_rate
    observed_sim = simulate_token(
        anchors,
        SimulationConfig(target_tps=20.0, expert_cache_hit_rate=observed),
    )
    if required_hit_20 is None:
        required_pages = additional_pages = None
        predictive_coverage = None
    else:
        required_pages = math.ceil(required_hit_20 * anchors.two_token_current_pages - 1e-12)
        additional_pages = max(0, required_pages - anchors.two_token_previous_overlap_pages)
        cold_pages = anchors.two_token_current_pages - anchors.two_token_previous_overlap_pages
        predictive_coverage = additional_pages / cold_pages
    fixed_s14_bytes = anchors.s14_fixed_weight_bytes
    vram_page_capacity = max(0, DEFAULT_VRAM_BYTES - fixed_s14_bytes) // anchors.expert_page_bytes
    return {
        "source_report_sha256": anchors.two_token_report_sha256,
        "observed_scope": "exactly token0->token1; cache retains only immediately previous token routed pages",
        "previous_token_intersection_pages": anchors.two_token_previous_overlap_pages,
        "current_token_pages": anchors.two_token_current_pages,
        "observed_previous_only_hit_rate": observed,
        "intersection_by_layer": {str(layer): count for layer, count in anchors.two_token_intersection_by_layer},
        "token1_expert_miss_bytes": anchors.two_token_expert_miss_bytes,
        "token1_total_new_bytes": anchors.two_token_new_bytes,
        "token1_extra_nonexpert_bytes": anchors.two_token_new_bytes - anchors.two_token_expert_miss_bytes,
        "20_tps_current_kernel_pipeline_at_observed_hit": observed_sim.to_dict(include_events=False),
        "20_tps_required_total_hit_pages_out_of_84": required_pages,
        "20_tps_additional_successful_prefetch_or_history_hits_beyond_previous_token": additional_pages,
        "20_tps_required_coverage_of_the_76_observed_cold_pages": predictive_coverage,
        "minimum_layer_expert_working_set_pages_previous_set_plus_successful_prefetch": (
            anchors.two_token_current_pages + additional_pages if additional_pages is not None else None
        ),
        "s14_fixed_weight_bytes_excluding_full_embedding": fixed_s14_bytes,
        "theoretical_vram_expert_page_capacity_after_s14_fixed_weights": vram_page_capacity,
        "capacity_caveat": "未扣 KV/activation/workspace/runtime；容量不等于命中率，prefetch 假阳性还会额外占页。",
        "non_generalization": "8/84 只是两个真实 token 的冷缓存反例，不是长序列稳态命中率。",
    }


def shortest_kernel_order() -> list[dict[str, Any]]:
    return [
        {"order": 1, "stage": "HC-attention pre + attention norm", "dependency": "previous layer output"},
        {"order": 2, "stage": "attention", "dependency": "wq_a -> q_norm/wq_b and wkv -> sparse attention -> wo_a/wo_b"},
        {"order": 3, "stage": "HC-attention post -> HC-FFN pre -> FFN norm", "dependency": "attention branch"},
        {"order": 4, "stage": "native top-6 router", "dependency": "FFN input; exact expert IDs become known here"},
        {"order": 5, "stage": "async routed-page lookup/SSD->GPU copy || shared expert", "dependency": "router; only these two branches may overlap exactly"},
        {"order": 6, "stage": "six routed experts", "dependency": "all six pages resident; current measured order is 6x[w1,w3,SwiGLU,w2,weighted accumulate]"},
        {"order": 7, "stage": "routed/shared accumulate -> HC-FFN post", "dependency": "both MoE branches"},
        {"order": 8, "stage": "next retained layer", "dependency": "current layer output; no exact future-layer route prefetch before this"},
        {"order": 9, "stage": "final HC -> norm -> full BF16 head/top1", "dependency": "L42 output"},
    ]


def build_roofline_report(
    asset_root: str | Path = "D:/models/Polaris-S14",
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    anchors = load_real_anchors(asset_root, repo_root)
    # Local import avoids a module cycle: full_depth reuses the hardware constants
    # and the measured StageAnchors type defined by this package.
    from .full_depth import build_full_depth_report

    target20 = _target_report(anchors, 20.0)
    target50 = _target_report(anchors, 50.0)
    required_hit_20 = target20["required_expert_cache_hit_rate_current_kernels"]
    cpu_tps = 1000.0 / anchors.cpu_warm_token_ms
    return {
        "format": "polaris-draft-and-fulldepth-stage-roofline-v2",
        "profile": "S14/top6 draft GPU token only; not final quality authority",
        "quality_authority": "FullDepth43/native-top6 exact causal verifier",
        "anchors": anchors.to_dict(),
        "cpu_observation": {
            "warm_cache_ms_per_token": anchors.cpu_warm_token_ms,
            "measured_tps": cpu_tps,
            "speedup_ratio_to_20_tps": 20.0 / cpu_tps,
            "speedup_ratio_to_50_tps": 50.0 / cpu_tps,
            "use_limit": "只是 CPU 真实基线与需求倍数；不参与 GPU 阶段时间计算，不是单核外推。",
        },
        "hardware_roof": {
            "pcie_bytes_per_second": DEFAULT_PCIE_BYTES_PER_SECOND,
            "pcie_semantics": "optimistic SSD/RAM-to-device transfer roof; actual SSD path can only be slower",
            "vram_bytes": DEFAULT_VRAM_BYTES,
            "ram_bytes": DEFAULT_RAM_BYTES,
            "ssd_bytes": DEFAULT_SSD_BYTES,
            "s14_fixed_weight_bytes": anchors.s14_fixed_weight_bytes,
            "s14_full_expert_bank_bytes": anchors.s14_full_expert_bank_bytes,
            "s14_profile_bytes": anchors.s14_profile_bytes,
            "s14_profile_fits_ram": anchors.s14_profile_bytes <= DEFAULT_RAM_BYTES,
            "s14_profile_fits_ssd": anchors.s14_profile_bytes <= DEFAULT_SSD_BYTES,
            "ram_expert_pages_after_fixed_mirror": (
                max(0, DEFAULT_RAM_BYTES - anchors.s14_fixed_weight_bytes) // anchors.expert_page_bytes
            ),
            "ssd_expert_pages_after_fixed_mirror": (
                max(0, DEFAULT_SSD_BYTES - anchors.s14_fixed_weight_bytes) // anchors.expert_page_bytes
            ),
            "capacity_caveat": "容量未扣 KV/activation/workspace/runtime，且容量覆盖不等于路由命中率。",
        },
        "targets": {"20_tps": target20, "50_tps": target50},
        "two_token_cold_cache_counterexample": _cold_counterexample(anchors, required_hit_20),
        "shortest_exact_kernel_order": shortest_kernel_order(),
        "full_depth43_exact_verifier": build_full_depth_report(asset_root, anchors),
        "hard_conclusions": [
            "当前 RX5700XT MoE+WQ_A 必要子操作锚点为 20.3476056 ms/token；尚未包含完整 attention、HC、router、norm/head 和 host 开销。",
            "20 tok/s 只在乐观 shared/DMA 重叠、其他未测阶段为 0 时存在条件可行性；所需专家命中率由报告前沿给出。",
            "50 tok/s 在当前单 token 锚点下即使 100% 专家命中也超出 20 ms；必须先有 batch/kernel fusion 加速，然后才有非零的未测阶段预算。",
            "真实 token0->token1 仅复用 8/84 专家页；仅保留上一 token top-6 不足以满足 20 tok/s 的理论命中门，需要更大历史工作集或成功的预测预取。",
        ],
        "claim_limit": "阶段 roofline/调度条件，不是完整 GPU token 实测，不证明 20/50 tok/s，不证明已达到 DeepSeek 质量。",
    }
