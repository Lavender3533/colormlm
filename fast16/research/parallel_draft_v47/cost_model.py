"""v47草稿头精确解析成本、串行/单次block对比与8%停止门。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from draft_core import load_contract


def architecture_cost(contract: dict[str, Any], candidate_count: int | None = None) -> dict[str, Any]:
    d = int(contract["hidden_size"])
    rank = int(contract["future_head"]["rank"])
    positions = len(contract["future_head"]["positions"])
    candidate_limit = int(contract["shortlist"]["candidate_limit"])
    candidates = candidate_limit if candidate_count is None else int(candidate_count)
    dtype_bytes = int(contract["cost"]["runtime_weight_dtype_bytes"])

    # 标量口径：乘/加/除各1 FLOP，tanh/sqrt单列special op；矩阵dot不使用2N近似。
    normalize_flops = d + (d - 1) + d
    input_projection_flops = rank * (d + (d - 1) + 1)  # dot + bias
    recurrent_projection_flops = rank * (rank + (rank - 1) + 1)
    score_flops_per_position = candidates * (rank + rank - 1)
    key_generation_scale_flops = candidates * rank
    common_flops = (
        normalize_flops
        + input_projection_flops
        + (positions - 1) * recurrent_projection_flops
        + positions * score_flops_per_position
        + key_generation_scale_flops
    )
    base_step_flops = 2 * int(contract["cost"]["base_active_parameters_per_token"])

    cascaded_parameters = d * rank + (positions - 1) * rank * rank + positions * rank
    serial_parameters = d * rank + rank * rank + positions * rank
    cascaded_weight_bytes = cascaded_parameters * dtype_bytes
    serial_weight_bytes = serial_parameters * dtype_bytes
    # 串行头在后两位显式加入上一个已选token的hash key，各rank次加法。
    serial_flops = common_flops + (positions - 1) * rank
    cascaded_flops = common_flops
    ids_bytes = candidates * 4
    keys_f32_bytes = candidates * rank * 4
    scores_f32_bytes = positions * candidates * 4
    normalized_hidden_f32_bytes = d * 4
    cascaded_states_f32_bytes = positions * rank * 4
    serial_state_f32_bytes = rank * 4

    return {
        "format": "colorlm-v47-parallel-draft-cost-v1",
        "assumption_boundary": {
            "candidate_count": candidates,
            "candidate_count_source": "冻结limit上界" if candidate_count is None else "回放观测值",
            "runtime_weight_dtype_bytes": dtype_bytes,
            "base_step_flops": base_step_flops,
            "verification_cost_factor_upper": float(contract["cost"]["verification_cost_factor_upper"]),
            "not_included": [
                "v38真实批验证内核效率",
                "Python/运行时调度开销",
                "缓存命中与设备间传输",
                "端到端墙钟"
            ]
        },
        "first_token": {
            "source": "v38原生logits top-1",
            "trainable_parameters": 0,
            "extra_vocabulary_projection_flops": 0,
            "full_vocabulary_projection": False
        },
        "cascaded_block_head": {
            "trainable_parameters": cascaded_parameters,
            "runtime_weight_bytes": cascaded_weight_bytes,
            "runtime_weight_mib": cascaded_weight_bytes / 1024**2,
            "flops_per_cycle": cascaded_flops,
            "flops_to_base_step_ratio": cascaded_flops / base_step_flops,
            "special_ops_per_cycle": {"sqrt": 1, "tanh": positions * rank},
            "token_key_hash_elements": candidates * rank,
            "integer_token_key_ops_not_counted_as_flops": {
                "dimension_salt_multiplies": rank,
                "initial_xor_and_splitmix64_ops": 13 * candidates * rank,
                "sign_shifts": candidates * rank,
                "sign_compares": candidates * rank,
                "conditional_selects": candidates * rank,
                "total": 16 * candidates * rank + rank
            },
            "token_key_float_scale_flops_included": key_generation_scale_flops,
            "proposal_dependent_scoring_stages": 1,
            "latent_dependency_depth": positions,
            "reference_schedule_bandwidth_bytes": {
                "head_weight_read": cascaded_weight_bytes,
                "hidden_read_f16": d * dtype_bytes,
                "candidate_id_read": ids_bytes,
                "score_write_f32": scores_f32_bytes,
                "total_external_or_persistent_excluding_scratch": cascaded_weight_bytes + d * dtype_bytes + ids_bytes + scores_f32_bytes
            },
            "reference_python_peak_scratch_bytes": (
                normalized_hidden_f32_bytes + cascaded_states_f32_bytes + keys_f32_bytes + scores_f32_bytes + ids_bytes
            ),
            "streaming_key_peak_scratch_bytes": (
                normalized_hidden_f32_bytes + cascaded_states_f32_bytes + rank * 4 + scores_f32_bytes + ids_bytes
            )
        },
        "serial_future_head": {
            "definition": "第2位后每一位读取上一个已选token hash key，再用共享rank递归矩阵更新",
            "trainable_parameters": serial_parameters,
            "runtime_weight_bytes": serial_weight_bytes,
            "runtime_weight_mib": serial_weight_bytes / 1024**2,
            "flops_per_cycle": serial_flops,
            "flops_to_base_step_ratio": serial_flops / base_step_flops,
            "special_ops_per_cycle": {"sqrt": 1, "tanh": positions * rank},
            "token_key_hash_elements": candidates * rank,
            "integer_token_key_ops_not_counted_as_flops": {
                "dimension_salt_multiplies": rank,
                "initial_xor_and_splitmix64_ops": 13 * candidates * rank,
                "sign_shifts": candidates * rank,
                "sign_compares": candidates * rank,
                "conditional_selects": candidates * rank,
                "total": 16 * candidates * rank + rank
            },
            "token_key_float_scale_flops_included": key_generation_scale_flops,
            "proposal_dependent_scoring_stages": positions,
            "latent_dependency_depth": positions,
            "reference_schedule_bandwidth_bytes": {
                "physical_head_weights": serial_weight_bytes,
                "head_weight_read_with_shared_recurrence_reread": (
                    (d * rank + (positions - 1) * rank * rank + positions * rank) * dtype_bytes
                ),
                "hidden_read_f16": d * dtype_bytes,
                "candidate_id_read": positions * ids_bytes,
                "score_write_f32": scores_f32_bytes
            },
            "reference_python_peak_scratch_bytes": (
                normalized_hidden_f32_bytes + serial_state_f32_bytes + keys_f32_bytes + candidates * 4 + ids_bytes
            )
        },
        "comparison": {
            "cascaded_minus_serial_parameters": cascaded_parameters - serial_parameters,
            "cascaded_minus_serial_flops": cascaded_flops - serial_flops,
            "interpretation": (
                "串行头因共享递归矩阵而略省参数，但有3个proposal-dependent阶段并受早期错误污染；"
                "cascaded block一次从anchor hidden产生三层latent，逐层监督且不把已选token喂回核心。"
            )
        }
    }


def apply_stop_gate(
    cost: dict[str, Any], acceptance: dict[str, Any] | None, contract: dict[str, Any]
) -> dict[str, Any]:
    if acceptance is None:
        return {
            "status": "stop_evidence_missing",
            "passed": False,
            "reason": "缺少通过完整覆盖门的v38自由滚动接受报告，不能计算加速下界。",
        }
    if acceptance.get("format") != "colorlm-v47-parallel-draft-acceptance-report-v1":
        raise ValueError("acceptance报告format不匹配")
    if not acceptance.get("simulation_executed") or not acceptance.get("coverage", {}).get("gate_passed"):
        return {
            "status": "stop_coverage_or_replay_failed",
            "passed": False,
            "reason": "完整轨迹覆盖或自由滚动回放未完成。",
        }
    evaluation = acceptance["free_roll"]["evaluation"]
    samples = int(evaluation["anchors"])
    mean_acceptance = float(evaluation["mean_accepted_draft_tokens"])
    confidence = float(contract["cost"]["confidence"])
    max_value = float(contract["acceptance_gate"]["maximum_draft_tokens"])
    # Hoeffding单侧分布无关下界，A∈[0,4]；避免小样本均值被直接当成稳定接受长度。
    penalty = max_value * math.sqrt(math.log(1.0 / (1.0 - confidence)) / (2.0 * max(samples, 1)))
    lower_acceptance = max(0.0, mean_acceptance - penalty)
    verify = float(contract["cost"]["verification_cost_factor_upper"])
    draft_ratio = float(cost["cascaded_block_head"]["flops_to_base_step_ratio"])
    conservative_speedup = lower_acceptance / (verify + draft_ratio)
    conventional_diagnostic = (lower_acceptance + 1.0) / (verify + draft_ratio)
    minimum_samples = int(contract["acceptance_gate"]["minimum_evaluation_anchors"])
    threshold = float(contract["cost"]["minimum_analytical_speedup_lower_bound"])
    passed = samples >= minimum_samples and conservative_speedup >= threshold
    return {
        "status": "pass_offline_cost_gate" if passed else "stop_below_8_percent_or_insufficient_samples",
        "passed": passed,
        "evaluation_anchors": samples,
        "minimum_evaluation_anchors": minimum_samples,
        "observed_mean_accepted_draft_tokens": mean_acceptance,
        "hoeffding_one_sided_confidence": confidence,
        "hoeffding_penalty_tokens": penalty,
        "mean_acceptance_lower_bound": lower_acceptance,
        "hard_gate_advance_definition": "只计已接受草稿token，不把尚未证明可原子提交的额外validator token计入",
        "analytical_speedup_lower_bound": conservative_speedup,
        "minimum_required_speedup": threshold,
        "conventional_plus_one_diagnostic_only": conventional_diagnostic,
        "decision": "允许讨论C++临时分支验证设计" if passed else "停止，不改C++",
        "claim_limit": "这是解析FLOPs与验证因子上界下的保守门，不是端到端测速。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("拒绝覆盖已有成本报告")
    contract = load_contract(args.contract)
    if args.candidate_count is not None and not 1 <= args.candidate_count <= int(contract["shortlist"]["candidate_limit"]):
        raise ValueError("candidate-count超出冻结limit")
    cost = architecture_cost(contract, args.candidate_count)
    acceptance = None if args.acceptance is None else json.loads(args.acceptance.read_text(encoding="utf-8"))
    cost["stop_gate"] = apply_stop_gate(cost, acceptance, contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cost, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stop_gate": cost["stop_gate"]}, ensure_ascii=False, indent=2))
    return 0 if cost["stop_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
