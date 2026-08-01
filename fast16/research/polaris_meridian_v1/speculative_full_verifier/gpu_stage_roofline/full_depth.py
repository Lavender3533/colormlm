"""FullDepth43 exact verifier 的阶段硬预算与 causal-block 前沿。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from ..assets import AssetAudit, LAYERS, TOP_K, audit_assets
from ..cost_model import committed_tokens
from .anchors import AnchorContractError, StageAnchors
from .model import (
    DEFAULT_PCIE_BYTES_PER_SECOND,
    DEFAULT_RAM_BYTES,
    DEFAULT_SSD_BYTES,
    DEFAULT_VRAM_BYTES,
    GIB,
)


_WQ_A_RE = re.compile(r"^layers\.(\d+)\.attn\.wq_a\.(weight|scale)$")
_ROUTED_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
_SHARED_RE = re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.(w[123])\.(weight|scale)$")
_ROUTER_RE = re.compile(r"^layers\.(\d+)\.ffn\.gate\.(weight|bias)$")


@dataclass(frozen=True)
class FullDepthShapeAudit:
    header_count: int
    wq_a_tensor_count: int
    routed_tensor_count: int
    shared_tensor_count: int
    router_tensor_count: int
    wq_a_signatures: tuple[tuple[str, str, tuple[int, ...], int], ...]
    routed_signatures: tuple[tuple[str, str, str, tuple[int, ...], int], ...]
    shared_signatures: tuple[tuple[str, str, str, tuple[int, ...], int], ...]
    router_signatures: tuple[tuple[str, str, tuple[int, ...], int], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorContractError(f"无法读取 FullDepth header {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnchorContractError(f"FullDepth header 顶层不是 object: {path}")
    return value


def _tensor_signature(metadata: dict[str, Any]) -> tuple[str, tuple[int, ...], int]:
    offsets = metadata.get("data_offsets")
    shape = metadata.get("shape")
    dtype = metadata.get("dtype")
    if (
        not isinstance(dtype, str)
        or not isinstance(shape, list)
        or not all(isinstance(value, int) for value in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) for value in offsets)
    ):
        raise AnchorContractError("FullDepth tensor dtype/shape/offset 非法")
    return dtype, tuple(shape), offsets[1] - offsets[0]


def audit_full_depth_projection_shapes(asset_root: str | Path) -> FullDepthShapeAudit:
    """证明 L42 计时锚点所覆盖的张量形状在 43 层一致。

    这里只证明同构投影有相同 packed shape，不把 L42 时间升级为 43 层实测。
    """

    header_paths = sorted((Path(asset_root) / "headers").glob("*.safetensors.header.json"))
    if len(header_paths) != 45:
        raise AnchorContractError("FullDepth shape audit 必须恰好看到 45 个 base header")

    wq_a: dict[tuple[int, str], tuple[str, tuple[int, ...], int]] = {}
    routed: dict[tuple[int, int, str, str], tuple[str, tuple[int, ...], int]] = {}
    shared: dict[tuple[int, str, str], tuple[str, tuple[int, ...], int]] = {}
    router: dict[tuple[int, str], tuple[str, tuple[int, ...], int]] = {}
    for path in header_paths:
        tensors = _read_json(path).get("tensors")
        if not isinstance(tensors, dict):
            raise AnchorContractError(f"header 缺少 tensors: {path.name}")
        for name, metadata in tensors.items():
            if not isinstance(metadata, dict):
                raise AnchorContractError(f"tensor metadata 非法: {name}")
            match = _WQ_A_RE.match(name)
            if match:
                wq_a[(int(match.group(1)), match.group(2))] = _tensor_signature(metadata)
                continue
            match = _ROUTED_RE.match(name)
            if match:
                routed[(int(match.group(1)), int(match.group(2)), match.group(3), match.group(4))] = _tensor_signature(metadata)
                continue
            match = _SHARED_RE.match(name)
            if match:
                shared[(int(match.group(1)), match.group(2), match.group(3))] = _tensor_signature(metadata)
                continue
            match = _ROUTER_RE.match(name)
            if match:
                router[(int(match.group(1)), match.group(2))] = _tensor_signature(metadata)

    expected_wq_keys = {(layer, item) for layer in range(LAYERS) for item in ("weight", "scale")}
    expected_routed_keys = {
        (layer, expert, weight, item)
        for layer in range(LAYERS)
        for expert in range(256)
        for weight in ("w1", "w2", "w3")
        for item in ("weight", "scale")
    }
    expected_shared_keys = {
        (layer, weight, item)
        for layer in range(LAYERS)
        for weight in ("w1", "w2", "w3")
        for item in ("weight", "scale")
    }
    # Base headers have a gate weight on all 43 layers and a gate bias only on
    # L3..L42.  This is an exact structural distinction, not missing metadata.
    expected_router_keys = (
        {(layer, "weight") for layer in range(LAYERS)}
        | {(layer, "bias") for layer in range(3, LAYERS)}
    )
    if set(wq_a) != expected_wq_keys:
        raise AnchorContractError("43 层 wq_a shape coverage 不完整")
    if set(routed) != expected_routed_keys:
        raise AnchorContractError("43x256 routed expert shape coverage 不完整")
    if set(shared) != expected_shared_keys:
        raise AnchorContractError("43 层 shared expert shape coverage 不完整")
    if set(router) != expected_router_keys:
        raise AnchorContractError("FullDepth native router shape coverage 漂移")

    def grouped_signatures(
        rows: dict[tuple[Any, ...], tuple[str, tuple[int, ...], int]],
        suffix_width: int,
    ) -> tuple[tuple[Any, ...], ...]:
        grouped: dict[tuple[str, ...], set[tuple[str, tuple[int, ...], int]]] = {}
        for key, signature in rows.items():
            suffix = tuple(str(value) for value in key[-suffix_width:])
            grouped.setdefault(suffix, set()).add(signature)
        result: list[tuple[Any, ...]] = []
        for suffix, signatures in sorted(grouped.items()):
            if len(signatures) != 1:
                raise AnchorContractError(f"FullDepth 同构 shape 漂移: {suffix} -> {signatures}")
            dtype, shape, size = next(iter(signatures))
            result.append((*suffix, dtype, shape, size))
        return tuple(result)

    return FullDepthShapeAudit(
        header_count=len(header_paths),
        wq_a_tensor_count=len(wq_a),
        routed_tensor_count=len(routed),
        shared_tensor_count=len(shared),
        router_tensor_count=len(router),
        wq_a_signatures=grouped_signatures(wq_a, 1),
        routed_signatures=grouped_signatures(routed, 2),
        shared_signatures=grouped_signatures(shared, 2),
        router_signatures=grouped_signatures(router, 1),
    )


def _projected_stage_work(stage_anchors: StageAnchors, block_size: int) -> dict[str, float]:
    positions_layers = block_size * LAYERS
    return {
        "attention": positions_layers * stage_anchors.wq_a_linear_ms,
        "hc": 0.0,
        "router": 0.0,
        "six_routed": positions_layers * stage_anchors.six_routed_envelope_ms_per_layer,
        "shared": positions_layers * stage_anchors.shared_and_batch_residual_ms_per_layer,
        "norm_head": 0.0,
        "ssd_miss": 0.0,
        "dispatch_overhead": 0.0,
    }


def _unique_expert_page_cases(block_size: int) -> dict[str, int]:
    return {
        "perfect_within_block_route_reuse": LAYERS * TOP_K,
        "no_within_block_route_reuse": LAYERS * min(256, TOP_K * block_size),
    }


def _required_hit_rate(allowed_bytes: float, fixed_bytes: int, expert_bytes: int) -> float | None:
    if allowed_bytes + 1e-9 < fixed_bytes:
        return None
    expert_allowance = allowed_bytes - fixed_bytes
    if expert_allowance >= expert_bytes:
        return 0.0
    return max(0.0, 1.0 - expert_allowance / expert_bytes)


def _io_policy_rows(
    audit: AssetAudit,
    *,
    block_size: int,
    committed: int,
    target_tps: float,
) -> dict[str, Any]:
    allowed_bytes = committed * DEFAULT_PCIE_BYTES_PER_SECOND / target_tps
    cases = _unique_expert_page_cases(block_size)
    policies = {
        "stream_fixed_each_block": {
            "fixed_transfer_bytes": audit.fixed_decode_scan_bytes,
            "theoretical_expert_cache_capacity_pages": DEFAULT_VRAM_BYTES // audit.expert_page_bytes,
            "capacity_semantics": "固定权重也需流式工作缓冲；此处仍把全部 8 GiB 算作专家缓存，是偏乐观上限。",
        },
        "resident_fixed_after_warmup": {
            "fixed_transfer_bytes": 0,
            "theoretical_expert_cache_capacity_pages": (
                max(0, DEFAULT_VRAM_BYTES - audit.fixed_decode_scan_bytes) // audit.expert_page_bytes
            ),
            "capacity_semantics": "固定 43 层非路由+BF16 head 常驻；剩余页数尚未扣 runtime/KV/workspace。",
        },
    }
    result: dict[str, Any] = {}
    for policy_name, policy in policies.items():
        fixed = int(policy["fixed_transfer_bytes"])
        capacity = int(policy["theoretical_expert_cache_capacity_pages"])
        rows: dict[str, Any] = {}
        for case, pages in cases.items():
            expert_bytes = pages * audit.expert_page_bytes
            hit_rate = _required_hit_rate(allowed_bytes, fixed, expert_bytes)
            required_pages = None if hit_rate is None else math.ceil(hit_rate * pages - 1e-12)
            transfer_bytes = (
                None
                if required_pages is None
                else fixed + (pages - required_pages) * audit.expert_page_bytes
            )
            rows[case] = {
                "unique_expert_pages_after_block_dedup": pages,
                "unique_expert_bytes": expert_bytes,
                "cold_fixed_plus_expert_transfer_ms": (
                    fixed + expert_bytes
                ) / DEFAULT_PCIE_BYTES_PER_SECOND * 1000.0,
                "required_device_expert_cache_hit_rate": hit_rate,
                "required_device_expert_cache_pages": required_pages,
                "transfer_ms_after_required_whole_page_hits": (
                    None
                    if transfer_bytes is None
                    else transfer_bytes / DEFAULT_PCIE_BYTES_PER_SECOND * 1000.0
                ),
                "capacity_feasible_before_runtime": (
                    required_pages is not None and required_pages <= capacity
                ),
            }
        result[policy_name] = {
            **policy,
            "pcie_block_allowance_bytes": allowed_bytes,
            "pcie_block_allowance_ms": committed * 1000.0 / target_tps,
            "cases": rows,
        }

    hybrid_cases: dict[str, Any] = {}
    for case, pages in cases.items():
        unique_bytes = audit.fixed_decode_scan_bytes + pages * audit.expert_page_bytes
        minimum_resident = max(0.0, unique_bytes - allowed_bytes)
        hybrid_cases[case] = {
            "unique_fixed_plus_expert_bytes": unique_bytes,
            "minimum_useful_bytes_resident_before_block": minimum_resident,
            "fits_8gib_before_runtime": minimum_resident <= DEFAULT_VRAM_BYTES,
            "minimum_bytes_forced_across_pcie_even_with_ideal_8gib_contents": max(
                0, unique_bytes - DEFAULT_VRAM_BYTES
            ),
        }
    result["optimistic_any_weight_residency_lower_bound"] = {
        "pcie_block_allowance_bytes": allowed_bytes,
        "cases": hybrid_cases,
        "semantics": "允许 8 GiB 在 fixed/expert 间任意最优分配；只用于证明不可能，忽略 runtime。",
    }
    return result


def _command_buffer_rows(stage_anchors: StageAnchors, block_size: int, block_budget_ms: float) -> dict[str, Any]:
    measured_dispatches = stage_anchors.top6_dispatches + stage_anchors.wq_a_dispatches
    counts = {
        "per_projected_measured_dispatch": LAYERS * block_size * measured_dispatches,
        "causal_block_batched_measured_dispatch": LAYERS * measured_dispatches,
        "route_split_per_layer": LAYERS * 2 + 1,
        "resident_gpu_driven_per_layer": LAYERS + 1,
        "whole_block_persistent": 1,
    }
    return {
        name: {
            "command_buffers_per_block": count,
            "exclusive_max_average_submit_overhead_us_if_all_other_work_zero": (
                block_budget_ms * 1000.0 / count
            ),
        }
        for name, count in counts.items()
    }


def _hard_stage_budget(stage_work: dict[str, float], block_budget_ms: float) -> dict[str, Any]:
    known = sum(stage_work.values())
    rows: dict[str, Any] = {}
    for stage, projection in stage_work.items():
        other = known - projection
        rows[stage] = {
            "current_L42_homogeneous_projection_or_zero_ms": projection,
            "exclusive_max_ms_if_other_projected_stages_stay_serial_and_unknowns_are_zero": (
                block_budget_ms - other
            ),
            "evidence": (
                "L42 GPU timing projected only after 43-layer packed-shape audit"
                if stage in {"attention", "six_routed", "shared"}
                else (
                    "policy-dependent PCIe byte lower bound is reported separately under io_requirements"
                    if stage == "ssd_miss"
                    else (
                        "command-buffer count/budget is reported separately; zero is not an estimate"
                        if stage == "dispatch_overhead"
                        else "unmeasured zero placeholder; not an estimate"
                    )
                )
            ),
        }
    return {
        "block_wall_budget_ms": block_budget_ms,
        "projected_known_stage_work_ms": known,
        "stages": rows,
    }


def _target_frontier(
    audit: AssetAudit,
    stage_anchors: StageAnchors,
    *,
    block_size: int,
    target_tps: float,
) -> dict[str, Any]:
    stage_work = _projected_stage_work(stage_anchors, block_size)
    projected_known = sum(stage_work.values())
    frontier: list[dict[str, Any]] = []
    for accepted in range(block_size + 1):
        committed = committed_tokens(block_size, accepted)
        budget_ms = committed * 1000.0 / target_tps
        minimum_speedup = projected_known / budget_ms
        frontier.append(
            {
                "accepted_prefix_length": accepted,
                "outcome": "full_match" if accepted == block_size else "mismatch_then_native_fallback",
                "committed_tokens": committed,
                "all_draft_positions_execute_all_43_layers": True,
                "hard_stage_budget": _hard_stage_budget(stage_work, budget_ms),
                "minimum_uniform_speedup_of_projected_known_gpu_kernels": minimum_speedup,
                "minimum_fraction_of_ideal_K_way_batch_speedup": minimum_speedup / block_size,
                "possible_from_ideal_K_way_batching_alone_for_projected_known_kernels": (
                    minimum_speedup <= block_size + 1e-12
                ),
                "io_requirements": _io_policy_rows(
                    audit,
                    block_size=block_size,
                    committed=committed,
                    target_tps=target_tps,
                ),
                "command_buffer_requirements": _command_buffer_rows(
                    stage_anchors, block_size, budget_ms
                ),
            }
        )
    return {
        "target_tps": target_tps,
        "block_size_K": block_size,
        "acceptance_is_not_assumed": True,
        "frontier": frontier,
    }


def full_depth_shortest_kernel_order() -> list[dict[str, Any]]:
    return [
        {"order": 1, "stage": "consume K draft proposals", "rule": "draft is untrusted; no token is committed"},
        {"order": 2, "stage": "FullDepth L0..L42 causal block", "rule": "every K position traverses every one of 43 layers; no layer skip"},
        {"order": 3, "stage": "per-layer HC-attn pre -> causal attention -> HC-attn post", "rule": "strict triangular causal mask over the K positions"},
        {"order": 4, "stage": "HC-FFN pre -> native top-6 router", "rule": "router runs in FullDepth for each position"},
        {"order": 5, "stage": "deduplicate (layer, expert) pages; cache lookup/PCIe copy || shared expert", "rule": "only exact page duplicates are reused; only copy/shared overlap is credited"},
        {"order": 6, "stage": "six routed experts -> routed/shared accumulate -> HC-FFN post", "rule": "all selected expert pages must be resident before routed compute"},
        {"order": 7, "stage": "FullDepth final HC -> norm -> BF16 head/top-1 for K positions", "rule": "predictions come only from L42 output"},
        {"order": 8, "stage": "longest-consistent-prefix compare", "rule": "accept only the exact longest prefix"},
        {"order": 9, "stage": "commit", "rule": "mismatch commits prefix plus one FullDepth native fallback; full match commits K; drafts never bypass verifier"},
    ]


def build_full_depth_report(
    asset_root: str | Path,
    stage_anchors: StageAnchors,
    *,
    block_sizes: tuple[int, ...] = (1, 4, 8),
    targets: tuple[float, ...] = (20.0, 50.0),
) -> dict[str, Any]:
    if any(block_size <= 0 for block_size in block_sizes):
        raise ValueError("FullDepth block size 必须为正")
    audit = audit_assets(asset_root)
    shape_audit = audit_full_depth_projection_shapes(asset_root)
    if audit.expert_page_bytes != stage_anchors.expert_page_bytes:
        raise AnchorContractError("FullDepth header 与 S14 route catalog 专家页字节不一致")

    stage_per_position = _projected_stage_work(stage_anchors, 1)
    projected_known_per_position = sum(stage_per_position.values())
    fixed = audit.fixed_decode_scan_bytes
    resident_pages = max(0, DEFAULT_VRAM_BYTES - fixed) // audit.expert_page_bytes
    block_reports = {
        f"K{block_size}": {
            "block_size_K": block_size,
            "unique_expert_page_bounds_after_block_dedup": _unique_expert_page_cases(block_size),
            "targets": {
                f"{target:g}_tps": _target_frontier(
                    audit,
                    stage_anchors,
                    block_size=block_size,
                    target_tps=target,
                )
                for target in targets
            },
        }
        for block_size in block_sizes
    }
    return {
        "format": "polaris-fulldepth43-stage-roofline-v1",
        "quality_authority": "FullDepth43/native-top6 exact causal verifier",
        "draft_only_profiles": ["S14", "v38", "v47"],
        "commit_contract": {
            "all_final_tokens_require_full_depth": True,
            "layers_executed_per_verified_position": LAYERS,
            "layer_skipping_allowed": False,
            "acceptance_rule": "longest exact consistent prefix only",
            "mismatch_rule": "commit accepted prefix plus one FullDepth native fallback",
        },
        "asset_audit": audit.to_dict(),
        "shape_audit_for_L42_projection": shape_audit.to_dict(),
        "hardware": {
            "vram_bytes": DEFAULT_VRAM_BYTES,
            "ram_bytes": DEFAULT_RAM_BYTES,
            "ssd_bytes": DEFAULT_SSD_BYTES,
            "pcie_bytes_per_second": DEFAULT_PCIE_BYTES_PER_SECOND,
            "fixed_non_routed_plus_head_bytes": fixed,
            "fixed_non_routed_plus_head_gib": fixed / GIB,
            "vram_expert_pages_after_fixed_residency": resident_pages,
            "full_base_shard_bytes": audit.full_base_shard_bytes,
            "full_base_fits_118gib_ssd": audit.full_base_shard_bytes <= DEFAULT_SSD_BYTES,
            "ssd_shortfall_bytes": max(0, audit.full_base_shard_bytes - DEFAULT_SSD_BYTES),
            "capacity_caveat": "VRAM 页数未扣 KV/activation/workspace/runtime；118 GiB 不能完整容纳 45 个 base shard。",
        },
        "projected_known_gpu_work_per_verified_position": {
            "stages_ms": stage_per_position,
            "total_ms": projected_known_per_position,
            "resident_known_projection_tps": 1000.0 / projected_known_per_position,
            "provenance": (
                "RX5700XT 的一个真实 L42 input 计时，乘以 43；45 headers 证明 wq_a、"
                "routed/shared packed shape 同构，但这仍不是 FullDepth43 token 实测。"
            ),
            "missing_work": "完整 attention、HC、router、norm/head、BF16/requant 边界和 host/runtime 开销。",
        },
        "feasibility_semantics": (
            "compute、PCIe、capacity、command-buffer 行分别是必要条件；即使全部分别满足也不构成充分条件。"
            "实际依赖只明确允许 expert copy 与 shared expert 重叠，完整延迟只会更差。"
        ),
        "blocks": block_reports,
        "shortest_exact_kernel_order": full_depth_shortest_kernel_order(),
        "hard_conclusions": [
            (
                f"FullDepth43 的已知 L42 同构投影为 {projected_known_per_position:.7f} ms/verified-position，"
                "尚未计完整 attention/HC/router/head；20/50 tok/s 对全接受块分别至少要求 "
                f"{projected_known_per_position / 50.0:.7f}x/{projected_known_per_position / 20.0:.7f}x "
                "projected-kernel 吞吐提升。"
            ),
            (
                f"非路由+BF16 head 为 {fixed} B，22.03 GB/s 下单块冷扫描下界 "
                f"{fixed / DEFAULT_PCIE_BYTES_PER_SECOND * 1000.0:.6f} ms；"
                "因此 stream-fixed 的 K=1/K=4 在 20 tok/s、K<=8 在 50 tok/s 均仅凭固定扫描即可否决。"
            ),
            (
                f"固定权重常驻 8 GiB 后理论上只剩 {resident_pages} 个专家页，而一个 native-top6 "
                f"FullDepth position 至少涉及 {LAYERS * TOP_K} 个逐层专家页；容量覆盖率不是命中率。"
            ),
            "K=4/8 的条件前沿逐项列出每个接受前缀；没有把接受率、页复用、缓存命中或线性 batch 收益设为事实。",
            "S14/v38/v47 只可提出草稿；最终提交 token 一律来自完整 43 层 native-top6 verifier，不能跳层。",
        ],
        "claim_limit": "阶段预算与必要条件，不是 FullDepth43 实测，不证明 20/50 tok/s，也不证明已达到 DeepSeek 质量。",
    }
