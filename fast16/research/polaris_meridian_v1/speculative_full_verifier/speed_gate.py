"""Speculative block 的时延成本模型与不夸大加速门。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .cost_model import committed_tokens
from .runtime_controller import BATCHED_CAUSAL_MODE, SUPPORTED_BLOCK_SIZES


EVIDENCE_KINDS = {"measured_model", "static_projection", "control_plane_microbench"}


@dataclass(frozen=True)
class RoundTiming:
    block_size: int
    accepted_prefix_length: int
    draft_seconds: float
    verifier_seconds: float
    verifier_mode: str
    verifier_forward_calls: int

    def validate(self) -> None:
        if self.block_size not in SUPPORTED_BLOCK_SIZES:
            raise ValueError(f"block_size 必须是 {SUPPORTED_BLOCK_SIZES} 之一")
        if not 0 <= self.accepted_prefix_length <= self.block_size:
            raise ValueError("accepted_prefix_length 越界")
        if self.draft_seconds < 0 or self.verifier_seconds < 0:
            raise ValueError("时延不得为负")
        if self.verifier_forward_calls <= 0:
            raise ValueError("verifier_forward_calls 必须为正数")


def evaluate_speed_gate(
    rounds: Sequence[RoundTiming],
    *,
    baseline_target_seconds_per_token: float,
    evidence_kind: str,
    same_process_adjacent_baseline: bool,
    minimum_speedup: float = 1.08,
) -> dict[str, Any]:
    """Compare committed-token latency against a native target baseline.

    Only adjacent, same-process real model timings may pass.  Static projections
    and control-plane microbenchmarks remain useful evidence, but never turn
    into an observed token/s claim.
    """

    if not rounds:
        raise ValueError("至少需要一轮 timing")
    for row in rounds:
        row.validate()
    if baseline_target_seconds_per_token <= 0:
        raise ValueError("baseline_target_seconds_per_token 必须为正数")
    if evidence_kind not in EVIDENCE_KINDS:
        raise ValueError(f"未知 evidence_kind: {evidence_kind}")
    if minimum_speedup <= 1:
        raise ValueError("minimum_speedup 必须大于 1")

    committed = sum(
        committed_tokens(row.block_size, row.accepted_prefix_length) for row in rounds
    )
    baseline_seconds = committed * baseline_target_seconds_per_token
    candidate_seconds = sum(row.draft_seconds + row.verifier_seconds for row in rounds)
    speedup = None if candidate_seconds == 0 else baseline_seconds / candidate_seconds
    batched_one_pass = all(
        row.verifier_mode == BATCHED_CAUSAL_MODE and row.verifier_forward_calls == 1
        for row in rounds
    )
    model_timing = evidence_kind == "measured_model"
    comparable = model_timing and same_process_adjacent_baseline
    speed_threshold = speedup is not None and speedup >= minimum_speedup
    passed = batched_one_pass and comparable and speed_threshold

    failures: list[str] = []
    if not batched_one_pass:
        failures.append(
            "FullDepth43 尚未做到每块一次 batched_causal forward；串行 K 次不具备加速资格"
        )
    if not model_timing:
        failures.append("证据不是真实模型时延，只能做静态/控制面判断")
    elif not same_process_adjacent_baseline:
        failures.append("缺少同进程相邻 native baseline")
    if not speed_threshold:
        failures.append(f"有效加速低于 {minimum_speedup:.3f}x")

    if passed:
        status = "pass_measured_speed_gate"
    elif not batched_one_pass:
        status = "stop_serial_verifier"
    elif not comparable:
        status = "hold_for_measured_model_ab"
    else:
        status = "stop_below_speed_threshold"
    return {
        "format": "polaris-speculative-block-speed-gate-v1",
        "status": status,
        "passed": passed,
        "evidence_kind": evidence_kind,
        "same_process_adjacent_baseline": same_process_adjacent_baseline,
        "rounds": [asdict(row) for row in rounds],
        "totals": {
            "round_count": len(rounds),
            "committed_tokens": committed,
            "baseline_seconds": baseline_seconds,
            "candidate_seconds": candidate_seconds,
            "effective_candidate_tps": (
                None if candidate_seconds == 0 else committed / candidate_seconds
            ),
            "speedup_vs_native": speedup,
        },
        "requirements": {
            "supported_block_sizes": list(SUPPORTED_BLOCK_SIZES),
            "one_batched_target_forward_per_round": batched_one_pass,
            "minimum_speedup": minimum_speedup,
            "speed_threshold_met": speed_threshold,
        },
        "failures": failures,
        "claim_limit": (
            "通过只表示同进程相邻短测中的有效 token/s 加速；"
            "不表示长上下文、质量或 20/50 token/s 已达成。"
        ),
    }


def current_serial_static_gate(
    *, baseline_target_seconds_per_token: float, block_size: int
) -> dict[str, Any]:
    """Expose why the current K-times serial reference cannot be promoted."""

    timing = RoundTiming(
        block_size=block_size,
        accepted_prefix_length=block_size,
        draft_seconds=0.0,
        verifier_seconds=block_size * baseline_target_seconds_per_token,
        verifier_mode="serial_reference",
        verifier_forward_calls=block_size,
    )
    report = evaluate_speed_gate(
        [timing],
        baseline_target_seconds_per_token=baseline_target_seconds_per_token,
        evidence_kind="static_projection",
        same_process_adjacent_baseline=False,
    )
    report["static_assumptions"] = [
        "乐观地把草稿时间设为 0。",
        "乐观地假设整块全接受。",
        "当前串行 verifier 仍需 K 次 target token forward。",
    ]
    return report
