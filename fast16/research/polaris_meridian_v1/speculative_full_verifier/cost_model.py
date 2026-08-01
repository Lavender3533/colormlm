"""PCIe lower-bound cost model for one-pass causal-block verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from .assets import AssetAudit, LAYERS, TOP_K, audit_assets


GIB = 1024**3


@dataclass(frozen=True)
class HardwareBudget:
    vram_bytes: int = 8 * GIB
    ram_bytes: int = 32 * GIB
    ssd_bytes: int = 118 * GIB
    pcie_bytes_per_second: float = 22.03e9

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def committed_tokens(block_size: int, accepted_prefix_length: int) -> int:
    """Number of output tokens from one verification round.

    A mismatch commits the accepted draft prefix plus one native target fallback.
    A full match commits the draft only; this conservative contract does not add
    a bonus token.
    """

    if block_size <= 0:
        raise ValueError("block_size 必须为正整数")
    if not 0 <= accepted_prefix_length <= block_size:
        raise ValueError("accepted_prefix_length 越界")
    if accepted_prefix_length == block_size:
        return block_size
    return accepted_prefix_length + 1


def expert_page_bounds(block_size: int, experts_per_layer: int = 256) -> tuple[int, int]:
    """Return unique page bounds after exact within-block de-duplication."""

    if block_size <= 0:
        raise ValueError("block_size 必须为正整数")
    minimum = LAYERS * TOP_K
    maximum = LAYERS * min(experts_per_layer, TOP_K * block_size)
    return minimum, maximum


def _required_hit_rate(
    *,
    target_tps: float,
    committed: int,
    fixed_scan_bytes: int,
    unique_expert_bytes: int,
    bandwidth: float,
) -> float | None:
    """Solve fixed + (1-h)*expert <= committed*bw/target for h."""

    allowed_total = committed * bandwidth / target_tps
    expert_allowance = allowed_total - fixed_scan_bytes
    if expert_allowance < 0:
        return None
    if unique_expert_bytes == 0 or expert_allowance >= unique_expert_bytes:
        return 0.0
    required = 1.0 - expert_allowance / unique_expert_bytes
    if required > 1.0:
        return None
    return max(0.0, required)


def estimate_block(audit: AssetAudit, hardware: HardwareBudget, block_size: int) -> dict[str, Any]:
    min_pages, max_pages = expert_page_bounds(block_size)
    fixed = audit.fixed_decode_scan_bytes
    min_expert = min_pages * audit.expert_page_bytes
    max_expert = max_pages * audit.expert_page_bytes

    def tps(total_bytes: int) -> float:
        return block_size * hardware.pcie_bytes_per_second / total_bytes

    return {
        "block_size": block_size,
        "causal_full_depth_passes": 1,
        "non_routed_scan_bytes_per_block": audit.full_non_routed_bytes,
        "head_scan_bytes_per_block": audit.head_bytes,
        "fixed_scan_bytes_per_block": fixed,
        "non_routed_amortized_bytes_per_draft_position": audit.full_non_routed_bytes / block_size,
        "head_amortized_bytes_per_draft_position": audit.head_bytes / block_size,
        "unique_expert_pages": {
            "perfect_cross_token_reuse": min_pages,
            "no_cross_token_reuse": max_pages,
        },
        "expert_bytes_after_block_dedup": {
            "perfect_cross_token_reuse": min_expert,
            "no_cross_token_reuse": max_expert,
        },
        "cold_pcie_latency_lower_bound_seconds": {
            "perfect_cross_token_reuse": (fixed + min_expert) / hardware.pcie_bytes_per_second,
            "no_cross_token_reuse": (fixed + max_expert) / hardware.pcie_bytes_per_second,
        },
        "full_accept_throughput_ceiling_tps": {
            "stream_each_block_all_expert_pages_device_resident": tps(fixed),
            "stream_each_block_cold_perfect_cross_token_reuse": tps(fixed + min_expert),
            "stream_each_block_cold_no_cross_token_reuse": tps(fixed + max_expert),
            "resident_after_warmup_cold_perfect_cross_token_reuse": block_size * hardware.pcie_bytes_per_second / min_expert,
            "resident_after_warmup_cold_no_cross_token_reuse": block_size * hardware.pcie_bytes_per_second / max_expert,
        },
    }


def throughput_requirement(
    audit: AssetAudit,
    hardware: HardwareBudget,
    block_size: int,
    target_tps: float,
    *,
    fixed_scan_policy: str = "stream_each_block",
) -> dict[str, Any]:
    """Report conditions; never insert an assumed acceptance or hit rate."""

    min_pages, max_pages = expert_page_bounds(block_size)
    expert_cases = {
        "perfect_cross_token_reuse": min_pages * audit.expert_page_bytes,
        "no_cross_token_reuse": max_pages * audit.expert_page_bytes,
    }
    if fixed_scan_policy == "stream_each_block":
        fixed = audit.fixed_decode_scan_bytes
    elif fixed_scan_policy == "resident_after_warmup":
        fixed = 0
    else:
        raise ValueError(f"未知 fixed_scan_policy: {fixed_scan_policy}")
    minimum_committed = max(1, math.ceil(target_tps * fixed / hardware.pcie_bytes_per_second))
    possible = minimum_committed <= block_size
    minimum_accepted = minimum_committed - 1 if possible else None

    result: dict[str, Any] = {
        "target_tps": target_tps,
        "block_size": block_size,
        "fixed_scan_policy": fixed_scan_policy,
        "possible_under_pcie_bound": possible,
        "minimum_committed_tokens_with_perfect_expert_cache": minimum_committed if possible else None,
        "minimum_accepted_prefix_with_perfect_expert_cache": minimum_accepted,
        "minimum_acceptance_note": (
            "prefix 之后的 1 个 FullDepth43 fallback 也是已提交 token"
            if possible
            else "即使专家页 100% 设备驻留且草稿全接受，固定扫描仍超过 PCIe 预算"
        ),
        "frontier": [],
        "minimum_accepted_prefix_with_zero_device_expert_cache": {},
    }
    if not possible:
        return result

    for case, expert_bytes in expert_cases.items():
        cold_committed = math.ceil(target_tps * (fixed + expert_bytes) / hardware.pcie_bytes_per_second)
        result["minimum_accepted_prefix_with_zero_device_expert_cache"][case] = (
            cold_committed - 1 if cold_committed <= block_size else None
        )

    # B-1 mismatch and B full match commit the same number of tokens.  Keep both
    # because their quality/control-flow meanings differ.
    for accepted in range(minimum_accepted, block_size + 1):
        committed = committed_tokens(block_size, accepted)
        required_hits: dict[str, float | None] = {}
        feasible = False
        for case, expert_bytes in expert_cases.items():
            hit = _required_hit_rate(
                target_tps=target_tps,
                committed=committed,
                fixed_scan_bytes=fixed,
                unique_expert_bytes=expert_bytes,
                bandwidth=hardware.pcie_bytes_per_second,
            )
            required_hits[case] = hit
            feasible = feasible or hit is not None
        if feasible:
            result["frontier"].append(
                {
                    "accepted_prefix_length": accepted,
                    "outcome": "full_match" if accepted == block_size else "mismatch_then_native_fallback",
                    "committed_tokens": committed,
                    "required_device_expert_cache_hit_rate": required_hits,
                }
            )
    return result


def _capacity_report(audit: AssetAudit, hardware: HardwareBudget) -> dict[str, Any]:
    fixed = audit.fixed_decode_scan_bytes
    page = audit.expert_page_bytes
    vram_remaining = max(0, hardware.vram_bytes - fixed)
    ram_remaining_after_fixed_mirror = max(0, hardware.ram_bytes - fixed)
    ssd_remaining_after_non_expert = max(0, hardware.ssd_bytes - audit.non_expert_packed_storage_bytes)
    ssd_expert_pages = ssd_remaining_after_non_expert // page
    return {
        "full_base_shard_bytes": audit.full_base_shard_bytes,
        "full_base_shard_gib": audit.full_base_shard_bytes / GIB,
        "full_base_tensor_payload_bytes": audit.full_base_payload_bytes,
        "shard_container_overhead_bytes": audit.shard_container_overhead_bytes,
        "ssd_shortfall_bytes": max(0, audit.full_base_shard_bytes - hardware.ssd_bytes),
        "ssd_shortfall_gib": max(0, audit.full_base_shard_bytes - hardware.ssd_bytes) / GIB,
        "full_base_fits_ssd": audit.full_base_shard_bytes <= hardware.ssd_bytes,
        "vram_after_non_routed_plus_head_bytes": vram_remaining,
        "vram_expert_pages_after_non_routed_plus_head": vram_remaining // page,
        "ram_expert_pages_if_dedicated": hardware.ram_bytes // page,
        "ram_expert_pages_after_fixed_mirror": ram_remaining_after_fixed_mirror // page,
        "ssd_expert_pages_if_dedicated": hardware.ssd_bytes // page,
        "ssd_expert_pages_after_non_expert_payload_and_shard_overhead": ssd_expert_pages,
        "ssd_expert_page_coverage_after_non_expert_payload_and_shard_overhead": ssd_expert_pages / audit.expert_page_count,
        "capacity_caveat": "VRAM 页数未扣 KV cache、activation、workspace 和 runtime，只是物理上限。容量覆盖率不是路由命中率。",
    }


def build_analysis_report(
    asset_root: str | Path,
    *,
    hardware: HardwareBudget | None = None,
    block_sizes: tuple[int, ...] = (1, 2, 4, 8, 16),
    targets: tuple[float, ...] = (20.0, 50.0),
) -> dict[str, Any]:
    hardware = hardware or HardwareBudget()
    audit = audit_assets(asset_root)
    blocks = [estimate_block(audit, hardware, block_size) for block_size in block_sizes]
    streamed_requirements = [
        throughput_requirement(audit, hardware, block_size, target, fixed_scan_policy="stream_each_block")
        for block_size in block_sizes
        for target in targets
    ]
    resident_requirements = [
        throughput_requirement(audit, hardware, block_size, target, fixed_scan_policy="resident_after_warmup")
        for block_size in block_sizes
        for target in targets
    ]
    max_block = max(block_sizes)
    max_cached_tps = max_block * hardware.pcie_bytes_per_second / audit.fixed_decode_scan_bytes
    return {
        "format": "polaris-speculative-full-verifier-analysis-v1",
        "profiles": {
            "draft": "S14/top6 + DeepSeek tokenizer",
            "verifier": "FullDepth43/native-top6 greedy causal block",
        },
        "asset_audit": audit.to_dict(),
        "hardware": hardware.to_dict(),
        "capacity": _capacity_report(audit, hardware),
        "block_estimates": blocks,
        "throughput_requirements": {
            "stream_each_block": streamed_requirements,
            "resident_after_warmup": resident_requirements,
        },
        "assumptions": [
            "每个 causal block 将 43 层非路由权重与 BF16 head 各扫描一次，因而可在 block 位置之间摊销。",
            "每层每 token 的 native router 返回 6 个不同专家；块内相同 (layer, expert) 页只读一次。",
            "device expert cache hit 表示页在验证前已驻留 VRAM，不占用本轮 PCIe；RAM/SSD 命中仍需 PCIe 传至 GPU。",
            "在首个不一致位置提交 FullDepth43 原生 fallback；草稿全一致时不额外提交 bonus token。",
            "只计 PCIe 字节下界，忽略计算、SSD 读、KV/activation、调度和内核开销；所以 token/s 是乐观上界，不是实测。",
            "resident_after_warmup 是额外乐观分支：按 budget 字节可将非路由+head 固定在 8GiB，但未计 runtime 后仅剩 60 个专家页。",
        ],
        "hard_conclusions": [
            {
                "id": "metadata_cross_check",
                "conclusion": f"45 个 base header 共 {audit.tensor_count} 个张量；budget 与 header/catalog 字节逐项一致。",
            },
            {
                "id": "ssd_capacity",
                "conclusion": (
                    f"FullDepth43 的 45 个 base shard 为 {audit.full_base_shard_bytes / GIB:.6f} GiB，"
                    f"超出 118 GiB SSD {max(0, audit.full_base_shard_bytes - hardware.ssd_bytes) / GIB:.6f} GiB；"
                    "不能在该 SSD 上无外部后备地完整自包含。"
                ),
            },
            {
                "id": "vram_upper_bound",
                "conclusion": (
                    f"8 GiB 中放入非路由+head 后理论上仅余 "
                    f"{_capacity_report(audit, hardware)['vram_expert_pages_after_non_routed_plus_head']} 个专家页，"
                    "且尚未扣 runtime 内存。"
                ),
            },
            {
                "id": "50_tps_pcie_bound",
                "conclusion": (
                    f"在 stream_each_block 分支，最大 block={max_block}、100% 设备专家命中、全接受且忽略所有计算开销时，"
                    f"PCIe 上界仍只有 {max_cached_tps:.6f} tok/s；50 tok/s 在 block<=16 下不可达。"
                ),
            },
            {
                "id": "quality_not_measured",
                "conclusion": "未提供真实 FullDepth43 route/acceptance trace，因而不假设接受率、命中率或 DeepSeek 质量。",
            },
        ],
    }
