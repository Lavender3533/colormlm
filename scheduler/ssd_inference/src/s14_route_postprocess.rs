//! DeepSeek-V4 / Polaris S14 原生 router 后处理。
//!
//! 固定 revision 的 Python 参考语义是：
//!
//! ```text
//! scores = sqrt(softplus(logits))
//! ids = topk(scores + bias, 6)       // L3..L42
//! weights = scores[ids] / sum * 1.5  // bias 只参与选择
//! ```
//!
//! L0..L2 的 ID 必须来自 token 对应的 `tid2eid` 物理行，不得误用
//! score top-6。本模块用显式 `S14RoutePostprocessKind` 将两条路径分开。

use polaris_s14_runner::{router_kind_for_layer, RouterKind, EXPERTS_PER_TOKEN, N_ROUTED_EXPERTS};
use std::{cmp::Ordering, error::Error, fmt};

pub const S14_ROUTE_SCALE: f32 = 1.5;
pub const S14_SOFTPLUS_THRESHOLD: f32 = 20.0;
pub const S14_ROUTE_SUM_ABS_TOLERANCE: f64 =
    EXPERTS_PER_TOKEN as f64 * f32::EPSILON as f64 * S14_ROUTE_SCALE as f64;

/// Optional native router bias. BF16 values are already-decoded 16-bit bit
/// patterns; callers reading safetensors bytes must decode them as little-endian.
#[derive(Debug, Clone, Copy)]
pub enum S14RouteBias<'a> {
    F32(&'a [f32]),
    Bf16Bits(&'a [u16]),
}

impl S14RouteBias<'_> {
    fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::Bf16Bits(values) => values.len(),
        }
    }

    fn value(self, index: usize) -> f32 {
        match self {
            Self::F32(values) => values[index],
            Self::Bf16Bits(values) => f32::from_bits((values[index] as u32) << 16),
        }
    }
}

/// Explicit route-selection contract. `ScoreTop6` is legal only for L3..L42;
/// `Tid2EidPhysical` is legal only for L0..L2 and preserves the table row order.
#[derive(Debug, Clone, Copy)]
pub enum S14RoutePostprocessKind<'a> {
    ScoreTop6 { bias: Option<S14RouteBias<'a>> },
    Tid2EidPhysical { expert_ids: &'a [u16] },
}

impl S14RoutePostprocessKind<'_> {
    fn router_kind(self) -> RouterKind {
        match self {
            Self::ScoreTop6 { .. } => RouterKind::Score,
            Self::Tid2EidPhysical { .. } => RouterKind::Hash,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct S14RoutePostprocessOutput {
    pub layer: u8,
    pub kind: RouterKind,
    pub expert_ids: [u16; EXPERTS_PER_TOKEN],
    pub weights: [f32; EXPERTS_PER_TOKEN],
    /// Unbiased `sqrt(softplus(logit))`; these values, not ranking scores, are
    /// normalized into route weights.
    pub selected_scores: [f32; EXPERTS_PER_TOKEN],
    /// Scores used solely for selection (`selected_scores + optional bias`).
    pub selected_ranking_scores: [f32; EXPERTS_PER_TOKEN],
    pub weight_sum_f64: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum S14RoutePostprocessError {
    LayerOutOfRange {
        layer: u8,
    },
    KindMismatch {
        layer: u8,
        expected: RouterKind,
        actual: RouterKind,
    },
    LogitsShape {
        expected: usize,
        actual: usize,
    },
    BiasShape {
        expected: usize,
        actual: usize,
    },
    Tid2EidShape {
        expected: usize,
        actual: usize,
    },
    NonFiniteLogit {
        expert: usize,
    },
    NonFiniteBias {
        expert: usize,
    },
    NonFiniteDerivedScore {
        expert: usize,
    },
    DuplicateExpert {
        expert: u16,
    },
    ExpertOutOfRange {
        expert: u16,
    },
    InvalidSelectedScoreSum,
    NonFiniteWeight {
        slot: usize,
    },
    WeightSumDrift {
        actual: f64,
        tolerance: f64,
    },
}

impl fmt::Display for S14RoutePostprocessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayerOutOfRange { layer } => {
                write!(formatter, "router layer {layer} 超出官方 0..42")
            }
            Self::KindMismatch {
                layer,
                expected,
                actual,
            } => write!(
                formatter,
                "router layer {layer} 要求 {expected:?}，不允许 {actual:?} 后处理路径"
            ),
            Self::LogitsShape { expected, actual } => {
                write!(formatter, "router logits 长度为 {actual}，期望 {expected}")
            }
            Self::BiasShape { expected, actual } => {
                write!(formatter, "router bias 长度为 {actual}，期望 {expected}")
            }
            Self::Tid2EidShape { expected, actual } => {
                write!(formatter, "tid2eid 物理行长度为 {actual}，期望 {expected}")
            }
            Self::NonFiniteLogit { expert } => {
                write!(formatter, "router logit[{expert}] 为 NaN/Inf")
            }
            Self::NonFiniteBias { expert } => write!(formatter, "router bias[{expert}] 为 NaN/Inf"),
            Self::NonFiniteDerivedScore { expert } => write!(
                formatter,
                "router expert {expert} 的 softplus/ranking score 为 NaN/Inf"
            ),
            Self::DuplicateExpert { expert } => {
                write!(formatter, "tid2eid 物理行重复 expert {expert}")
            }
            Self::ExpertOutOfRange { expert } => write!(
                formatter,
                "tid2eid expert {expert} 超出 0..{}",
                N_ROUTED_EXPERTS - 1
            ),
            Self::InvalidSelectedScoreSum => {
                write!(formatter, "selected route score 之和不是有限正数")
            }
            Self::NonFiniteWeight { slot } => {
                write!(formatter, "route weight[{slot}] 为 NaN/Inf")
            }
            Self::WeightSumDrift { actual, tolerance } => write!(
                formatter,
                "route weight sum={actual:.9} 未在 1.5±{tolerance:.3e} 内"
            ),
        }
    }
}

impl Error for S14RoutePostprocessError {}

/// Match PyTorch `F.softplus(x, beta=1, threshold=20).sqrt()` for finite F32
/// inputs. The threshold branch prevents positive overflow; the remaining
/// branch evaluates `log1p(exp(x))`, which is exactly the frozen Python formula.
pub fn sqrt_softplus_f32(logit: f32) -> f32 {
    let softplus = if logit > S14_SOFTPLUS_THRESHOLD {
        logit
    } else {
        logit.exp().ln_1p()
    };
    softplus.sqrt()
}

/// Convert native router logits into exactly six physical experts and weights.
///
/// Ties on ranking score are resolved by the lowest physical expert ID. This is
/// an explicit deterministic contract because `torch.topk` does not promise a
/// stable tie order across devices/backends.
pub fn postprocess_s14_route(
    layer: u8,
    logits: &[f32],
    selection: S14RoutePostprocessKind<'_>,
) -> Result<S14RoutePostprocessOutput, S14RoutePostprocessError> {
    let expected_kind = router_kind_for_layer(layer)
        .map_err(|_| S14RoutePostprocessError::LayerOutOfRange { layer })?;
    let actual_kind = selection.router_kind();
    if actual_kind != expected_kind {
        return Err(S14RoutePostprocessError::KindMismatch {
            layer,
            expected: expected_kind,
            actual: actual_kind,
        });
    }

    let expected_experts = N_ROUTED_EXPERTS as usize;
    if logits.len() != expected_experts {
        return Err(S14RoutePostprocessError::LogitsShape {
            expected: expected_experts,
            actual: logits.len(),
        });
    }

    let mut scores = Vec::with_capacity(expected_experts);
    for (expert, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            return Err(S14RoutePostprocessError::NonFiniteLogit { expert });
        }
        let score = sqrt_softplus_f32(logit);
        if !score.is_finite() {
            return Err(S14RoutePostprocessError::NonFiniteDerivedScore { expert });
        }
        scores.push(score);
    }

    let (expert_ids, ranking_scores) = match selection {
        S14RoutePostprocessKind::ScoreTop6 { bias } => {
            if let Some(bias) = bias {
                if bias.len() != expected_experts {
                    return Err(S14RoutePostprocessError::BiasShape {
                        expected: expected_experts,
                        actual: bias.len(),
                    });
                }
            }

            let mut ranked = Vec::with_capacity(expected_experts);
            for expert in 0..expected_experts {
                let bias_value = bias.map_or(0.0, |values| values.value(expert));
                if !bias_value.is_finite() {
                    return Err(S14RoutePostprocessError::NonFiniteBias { expert });
                }
                let ranking_score = scores[expert] + bias_value;
                if !ranking_score.is_finite() {
                    return Err(S14RoutePostprocessError::NonFiniteDerivedScore { expert });
                }
                ranked.push((expert as u16, ranking_score));
            }
            ranked.sort_unstable_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });

            let mut ids = [0u16; EXPERTS_PER_TOKEN];
            let mut selected_ranking = [0.0f32; EXPERTS_PER_TOKEN];
            for slot in 0..EXPERTS_PER_TOKEN {
                ids[slot] = ranked[slot].0;
                selected_ranking[slot] = ranked[slot].1;
            }
            (ids, selected_ranking)
        }
        S14RoutePostprocessKind::Tid2EidPhysical { expert_ids } => {
            if expert_ids.len() != EXPERTS_PER_TOKEN {
                return Err(S14RoutePostprocessError::Tid2EidShape {
                    expected: EXPERTS_PER_TOKEN,
                    actual: expert_ids.len(),
                });
            }
            let mut ids = [0u16; EXPERTS_PER_TOKEN];
            let mut selected_ranking = [0.0f32; EXPERTS_PER_TOKEN];
            for (slot, &expert) in expert_ids.iter().enumerate() {
                if expert >= N_ROUTED_EXPERTS {
                    return Err(S14RoutePostprocessError::ExpertOutOfRange { expert });
                }
                if ids[..slot].contains(&expert) {
                    return Err(S14RoutePostprocessError::DuplicateExpert { expert });
                }
                ids[slot] = expert;
                selected_ranking[slot] = scores[expert as usize];
            }
            (ids, selected_ranking)
        }
    };

    let mut selected_scores = [0.0f32; EXPERTS_PER_TOKEN];
    for (slot, &expert) in expert_ids.iter().enumerate() {
        selected_scores[slot] = scores[expert as usize];
    }
    let selected_sum: f32 = selected_scores.iter().copied().sum();
    if !selected_sum.is_finite() || selected_sum <= 0.0 {
        return Err(S14RoutePostprocessError::InvalidSelectedScoreSum);
    }

    let mut weights = [0.0f32; EXPERTS_PER_TOKEN];
    for slot in 0..EXPERTS_PER_TOKEN {
        weights[slot] = selected_scores[slot] / selected_sum * S14_ROUTE_SCALE;
        if !weights[slot].is_finite() {
            return Err(S14RoutePostprocessError::NonFiniteWeight { slot });
        }
    }
    let weight_sum_f64 = weights.iter().map(|&value| value as f64).sum::<f64>();
    if (weight_sum_f64 - S14_ROUTE_SCALE as f64).abs() > S14_ROUTE_SUM_ABS_TOLERANCE {
        return Err(S14RoutePostprocessError::WeightSumDrift {
            actual: weight_sum_f64,
            tolerance: S14_ROUTE_SUM_ABS_TOLERANCE,
        });
    }

    Ok(S14RoutePostprocessOutput {
        layer,
        kind: actual_kind,
        expert_ids,
        weights,
        selected_scores,
        selected_ranking_scores: ranking_scores,
        weight_sum_f64,
    })
}
