//! S14 / FullDepth43 router 后处理纯 CPU 合成数值门。
//!
//! `cargo run --release --offline -p ssd_inference --example s14_route_postprocess_numeric`

use anyhow::{ensure, Result};
use serde_json::{json, Value};
use ssd_inference::s14_route_postprocess::{
    postprocess_s14_route, sqrt_softplus_f32, S14RouteBias, S14RoutePostprocessError,
    S14RoutePostprocessKind, S14_ROUTE_SCALE, S14_ROUTE_SUM_ABS_TOLERANCE,
};

fn main() -> Result<()> {
    let report = run_numeric_gates()?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_numeric_gates() -> Result<Value> {
    let mut passed = Vec::new();

    // Values captured from torch 2.9.1 CPU:
    // F.softplus(torch.tensor(inputs, dtype=float32)).sqrt().
    let python_inputs = [-100.0f32, -20.0, -1.0, -0.0, 0.0, 1.0, 20.0, 21.0, 100.0];
    let python_expected_bits = [
        0x1b6b_26a9u32,
        0x383e_6bce,
        0x3f0f_485c,
        0x3f55_224d,
        0x3f55_224d,
        0x3f92_af5a,
        0x408f_1bbd,
        0x4092_a476,
        0x4120_0000,
    ];
    let mut max_python_ulp = 0u32;
    for (&input, &expected_bits) in python_inputs.iter().zip(python_expected_bits.iter()) {
        let actual = sqrt_softplus_f32(input);
        ensure!(
            actual.is_finite(),
            "sqrt-softplus produced non-finite value"
        );
        max_python_ulp = max_python_ulp.max(ulp_distance_positive(actual.to_bits(), expected_bits));
    }
    ensure!(
        max_python_ulp <= 3,
        "sqrt-softplus Python drift: {max_python_ulp} ULP"
    );
    passed.push("python_sqrt_softplus_f32_max_3ulp");

    let equal_logits = vec![0.0f32; 256];
    let tied = postprocess_s14_route(
        3,
        &equal_logits,
        S14RoutePostprocessKind::ScoreTop6 { bias: None },
    )?;
    ensure!(tied.expert_ids == [0, 1, 2, 3, 4, 5]);
    verify_weight_contract(&tied.weights, tied.weight_sum_f64)?;
    passed.push("deterministic_lowest_expert_id_tie_break");

    let expected_f32_ids = [12u16, 4, 200, 8, 1, 99];
    let mut f32_bias = vec![0.0f32; 256];
    for (rank, &expert) in expected_f32_ids.iter().enumerate() {
        f32_bias[expert as usize] = (6 - rank) as f32 * 10.0;
    }
    let f32_route = postprocess_s14_route(
        42,
        &equal_logits,
        S14RoutePostprocessKind::ScoreTop6 {
            bias: Some(S14RouteBias::F32(&f32_bias)),
        },
    )?;
    ensure!(f32_route.expert_ids == expected_f32_ids);
    verify_equal_weights(&f32_route.weights)?;
    verify_weight_contract(&f32_route.weights, f32_route.weight_sum_f64)?;
    passed.push("f32_bias_selection_unbiased_weight_normalization");

    let expected_bf16_ids = [250u16, 111, 72, 33, 9, 2];
    let mut bf16_bias = vec![0u16; 256];
    for (rank, &expert) in expected_bf16_ids.iter().enumerate() {
        let value = (6 - rank) as f32;
        bf16_bias[expert as usize] = (value.to_bits() >> 16) as u16;
    }
    let bf16_route = postprocess_s14_route(
        7,
        &equal_logits,
        S14RoutePostprocessKind::ScoreTop6 {
            bias: Some(S14RouteBias::Bf16Bits(&bf16_bias)),
        },
    )?;
    ensure!(bf16_route.expert_ids == expected_bf16_ids);
    verify_equal_weights(&bf16_route.weights)?;
    verify_weight_contract(&bf16_route.weights, bf16_route.weight_sum_f64)?;
    passed.push("bf16_bias_selection");

    let physical_ids = [5u16, 2, 9, 7, 1, 3];
    let hash_route = postprocess_s14_route(
        1,
        &equal_logits,
        S14RoutePostprocessKind::Tid2EidPhysical {
            expert_ids: &physical_ids,
        },
    )?;
    ensure!(hash_route.expert_ids == physical_ids);
    verify_equal_weights(&hash_route.weights)?;
    passed.push("l0_l2_tid2eid_order_preserved");

    ensure!(matches!(
        postprocess_s14_route(
            1,
            &equal_logits,
            S14RoutePostprocessKind::ScoreTop6 { bias: None }
        ),
        Err(S14RoutePostprocessError::KindMismatch { .. })
    ));
    ensure!(matches!(
        postprocess_s14_route(
            3,
            &equal_logits,
            S14RoutePostprocessKind::Tid2EidPhysical {
                expert_ids: &physical_ids
            }
        ),
        Err(S14RoutePostprocessError::KindMismatch { .. })
    ));
    passed.push("layer_router_kind_fail_closed");

    ensure!(matches!(
        postprocess_s14_route(
            43,
            &equal_logits,
            S14RoutePostprocessKind::ScoreTop6 { bias: None }
        ),
        Err(S14RoutePostprocessError::LayerOutOfRange { .. })
    ));
    ensure!(matches!(
        postprocess_s14_route(
            3,
            &equal_logits[..255],
            S14RoutePostprocessKind::ScoreTop6 { bias: None }
        ),
        Err(S14RoutePostprocessError::LogitsShape { .. })
    ));
    let short_bias = vec![0.0f32; 255];
    ensure!(matches!(
        postprocess_s14_route(
            3,
            &equal_logits,
            S14RoutePostprocessKind::ScoreTop6 {
                bias: Some(S14RouteBias::F32(&short_bias))
            }
        ),
        Err(S14RoutePostprocessError::BiasShape { .. })
    ));
    passed.push("layer_and_shape_fail_closed");

    let mut nan_logits = equal_logits.clone();
    nan_logits[17] = f32::NAN;
    ensure!(matches!(
        postprocess_s14_route(
            3,
            &nan_logits,
            S14RoutePostprocessKind::ScoreTop6 { bias: None }
        ),
        Err(S14RoutePostprocessError::NonFiniteLogit { expert: 17 })
    ));
    let mut inf_bias = vec![0.0f32; 256];
    inf_bias[23] = f32::INFINITY;
    ensure!(matches!(
        postprocess_s14_route(
            3,
            &equal_logits,
            S14RoutePostprocessKind::ScoreTop6 {
                bias: Some(S14RouteBias::F32(&inf_bias))
            }
        ),
        Err(S14RoutePostprocessError::NonFiniteBias { expert: 23 })
    ));
    let mut bf16_inf_bias = vec![0u16; 256];
    bf16_inf_bias[31] = 0x7f80;
    ensure!(matches!(
        postprocess_s14_route(
            3,
            &equal_logits,
            S14RoutePostprocessKind::ScoreTop6 {
                bias: Some(S14RouteBias::Bf16Bits(&bf16_inf_bias))
            }
        ),
        Err(S14RoutePostprocessError::NonFiniteBias { expert: 31 })
    ));
    passed.push("nan_inf_fail_closed");

    let zero_scores = vec![-1000.0f32; 256];
    ensure!(matches!(
        postprocess_s14_route(
            3,
            &zero_scores,
            S14RoutePostprocessKind::ScoreTop6 { bias: None }
        ),
        Err(S14RoutePostprocessError::InvalidSelectedScoreSum)
    ));
    passed.push("zero_denominator_fail_closed");

    let duplicate_ids = [1u16, 2, 3, 4, 5, 5];
    let short_ids = [1u16, 2, 3, 4, 5];
    ensure!(matches!(
        postprocess_s14_route(
            0,
            &equal_logits,
            S14RoutePostprocessKind::Tid2EidPhysical {
                expert_ids: &short_ids
            }
        ),
        Err(S14RoutePostprocessError::Tid2EidShape { .. })
    ));
    ensure!(matches!(
        postprocess_s14_route(
            0,
            &equal_logits,
            S14RoutePostprocessKind::Tid2EidPhysical {
                expert_ids: &duplicate_ids
            }
        ),
        Err(S14RoutePostprocessError::DuplicateExpert { expert: 5 })
    ));
    let out_of_range_ids = [1u16, 2, 3, 4, 5, 256];
    ensure!(matches!(
        postprocess_s14_route(
            0,
            &equal_logits,
            S14RoutePostprocessKind::Tid2EidPhysical {
                expert_ids: &out_of_range_ids
            }
        ),
        Err(S14RoutePostprocessError::ExpertOutOfRange { expert: 256 })
    ));
    passed.push("tid2eid_duplicate_and_range_fail_closed");

    Ok(json!({
        "format": "polaris-s14-route-postprocess-numeric-v1",
        "status": "pass",
        "gates_passed": passed,
        "gate_count": passed.len(),
        "python_reference": {
            "formula": "torch.nn.functional.softplus(logits).sqrt()",
            "torch_version": "2.9.1+cpu",
            "max_ulp": max_python_ulp
        },
        "contracts": {
            "experts": 256,
            "top_k": 6,
            "route_scale": S14_ROUTE_SCALE,
            "route_sum_abs_tolerance": S14_ROUTE_SUM_ABS_TOLERANCE,
            "score_tie_break": "ranking_score_desc_then_physical_expert_id_asc",
            "bias_affects": "selection_only_not_route_weight",
            "layers_0_2": "tid2eid_physical_only",
            "layers_3_42": "sqrtsoftplus_plus_optional_bias_top6"
        },
        "observations": {
            "tie_ids": tied.expert_ids,
            "f32_bias_ids": f32_route.expert_ids,
            "bf16_bias_ids": bf16_route.expert_ids,
            "tid2eid_ids": hash_route.expert_ids,
            "equal_score_weights": tied.weights,
            "weight_sum_f64": tied.weight_sum_f64
        },
        "claim_limit": "Synthetic CPU postprocess gate only; no real router tensor, GPU kernel, whole-token speed, or model-quality claim."
    }))
}

fn verify_equal_weights(weights: &[f32; 6]) -> Result<()> {
    for &weight in weights {
        ensure!((weight - 0.25).abs() <= f32::EPSILON);
    }
    Ok(())
}

fn verify_weight_contract(weights: &[f32; 6], reported_sum: f64) -> Result<()> {
    ensure!(weights
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0));
    let observed = weights.iter().map(|&value| value as f64).sum::<f64>();
    ensure!(observed.to_bits() == reported_sum.to_bits());
    ensure!((observed - 1.5).abs() <= S14_ROUTE_SUM_ABS_TOLERANCE);
    Ok(())
}

fn ulp_distance_positive(actual_bits: u32, expected_bits: u32) -> u32 {
    actual_bits.abs_diff(expected_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_numeric_gates_pass() {
        run_numeric_gates().unwrap();
    }
}
