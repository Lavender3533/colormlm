//! FullDepth43/native-top6 的层内 causal-block 专家联合计划。
//!
//! 该模块只负责把同一层 K=1/4/8 个 position 的原生 top-6 路由
//! 重排为“每个唯一专家加载一次，再分发给所有命中 token”的无损计划。
//! 它不改变专家集合、route slot 或权重，也不声称已经完成 GPU 批量执行。

use crate::{
    GraphProfile, RouteDecision, RouterKind, EXACT_CASCADE_BLOCK_SIZES, EXPERT_PAGE_BYTES,
    FULL_DEPTH_LAYERS,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalBatchPlanError(String);

impl fmt::Display for CausalBatchPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CausalBatchPlanError {}

fn plan_error(message: impl Into<String>) -> CausalBatchPlanError {
    CausalBatchPlanError(message.into())
}

/// 一个专家对 causal block 中一个 token 的原始分发位置。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertTokenDispatch {
    pub token_offset: usize,
    pub route_slot: usize,
    pub route_weight: f32,
}

/// 同一层一个唯一专家的联合工作项。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertBatchWork {
    pub expert_id: u16,
    pub dispatches: Vec<ExpertTokenDispatch>,
}

/// 一层 K-token causal block 的无损专家加载/分发计划。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerCausalBatchPlan {
    pub layer: u8,
    pub router_kind: RouterKind,
    pub block_size: usize,
    pub assignments: usize,
    pub unique_experts: usize,
    pub avoided_expert_loads: usize,
    pub serial_expert_bytes: u64,
    pub union_expert_bytes: u64,
    pub expert_byte_reduction_ratio: f64,
    pub experts: Vec<ExpertBatchWork>,
}

impl LayerCausalBatchPlan {
    /// 校验联合计划能逐项重建原 top-6；因此调度器不得静默丢专家或改权重。
    pub fn validate_against(&self, routes: &[RouteDecision]) -> Result<(), CausalBatchPlanError> {
        validate_block_size(routes.len())?;
        if self.block_size != routes.len() || self.assignments != routes.len() * 6 {
            return Err(plan_error("causal batch 维度/assignment 数漂移"));
        }
        let expected = build_layer_causal_batch_plan(routes)?;
        if self != &expected {
            return Err(plan_error("causal batch 不能无损重建原 top-6 路由"));
        }
        Ok(())
    }
}

/// 完整 43 层的 K-token 联合计划。在线执行器也可以逐层调用单层构建函数，
/// 避免为了计划未来层而违反 causal 依赖。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullDepthCausalBatchPlan {
    pub block_size: usize,
    pub layers: Vec<LayerCausalBatchPlan>,
    pub serial_expert_bytes: u64,
    pub union_expert_bytes: u64,
    pub avoided_expert_loads: usize,
    pub expert_byte_reduction_ratio: f64,
}

fn validate_block_size(block_size: usize) -> Result<(), CausalBatchPlanError> {
    if !EXACT_CASCADE_BLOCK_SIZES.contains(&block_size) {
        return Err(plan_error("causal batch 只允许 K=1/4/8"));
    }
    Ok(())
}

fn checked_expert_bytes(count: usize) -> Result<u64, CausalBatchPlanError> {
    u64::try_from(count)
        .ok()
        .and_then(|value| value.checked_mul(EXPERT_PAGE_BYTES))
        .ok_or_else(|| plan_error("专家字节统计溢出"))
}

/// 为同一层 K 个 position 构建联合专家工作项。
pub fn build_layer_causal_batch_plan(
    routes: &[RouteDecision],
) -> Result<LayerCausalBatchPlan, CausalBatchPlanError> {
    validate_block_size(routes.len())?;
    let first = routes
        .first()
        .ok_or_else(|| plan_error("causal batch 不得为空"))?;
    let layer = first.layer;
    let router_kind = first.kind;
    let mut by_expert: BTreeMap<u16, Vec<ExpertTokenDispatch>> = BTreeMap::new();

    for (token_offset, route) in routes.iter().enumerate() {
        route
            .validate_for(GraphProfile::FullDepth43NativeTop6)
            .map_err(|error| plan_error(error.to_string()))?;
        if route.layer != layer || route.kind != router_kind {
            return Err(plan_error(
                "同一 causal layer batch 的 layer/router kind 漂移",
            ));
        }
        for (route_slot, (&expert_id, &route_weight)) in route
            .expert_ids
            .iter()
            .zip(route.weights.iter())
            .enumerate()
        {
            by_expert
                .entry(expert_id)
                .or_default()
                .push(ExpertTokenDispatch {
                    token_offset,
                    route_slot,
                    route_weight,
                });
        }
    }

    let assignments = routes.len() * 6;
    let unique_experts = by_expert.len();
    let serial_expert_bytes = checked_expert_bytes(assignments)?;
    let union_expert_bytes = checked_expert_bytes(unique_experts)?;
    let avoided_expert_loads = assignments - unique_experts;
    let expert_byte_reduction_ratio = if serial_expert_bytes == 0 {
        0.0
    } else {
        1.0 - union_expert_bytes as f64 / serial_expert_bytes as f64
    };
    let experts = by_expert
        .into_iter()
        .map(|(expert_id, dispatches)| ExpertBatchWork {
            expert_id,
            dispatches,
        })
        .collect();

    Ok(LayerCausalBatchPlan {
        layer,
        router_kind,
        block_size: routes.len(),
        assignments,
        unique_experts,
        avoided_expert_loads,
        serial_expert_bytes,
        union_expert_bytes,
        expert_byte_reduction_ratio,
        experts,
    })
}

/// 从 position-major 的 43×top-6 路由构建完整计划。
pub fn build_full_depth_causal_batch_plan(
    routes_by_position: &[Vec<RouteDecision>],
) -> Result<FullDepthCausalBatchPlan, CausalBatchPlanError> {
    validate_block_size(routes_by_position.len())?;
    for (token_offset, routes) in routes_by_position.iter().enumerate() {
        if routes.len() != FULL_DEPTH_LAYERS.len() {
            return Err(plan_error(format!(
                "position {token_offset} 必须精确包含 43 层路由"
            )));
        }
        for (route, &expected_layer) in routes.iter().zip(FULL_DEPTH_LAYERS.iter()) {
            if route.layer != expected_layer {
                return Err(plan_error(format!(
                    "position {token_offset} 的 FullDepth 层顺序漂移"
                )));
            }
        }
    }

    let mut layers = Vec::with_capacity(FULL_DEPTH_LAYERS.len());
    for layer_offset in 0..FULL_DEPTH_LAYERS.len() {
        let routes: Vec<RouteDecision> = routes_by_position
            .iter()
            .map(|position| position[layer_offset].clone())
            .collect();
        layers.push(build_layer_causal_batch_plan(&routes)?);
    }
    let serial_expert_bytes = layers.iter().try_fold(0u64, |total, layer| {
        total
            .checked_add(layer.serial_expert_bytes)
            .ok_or_else(|| plan_error("FullDepth serial 字节统计溢出"))
    })?;
    let union_expert_bytes = layers.iter().try_fold(0u64, |total, layer| {
        total
            .checked_add(layer.union_expert_bytes)
            .ok_or_else(|| plan_error("FullDepth union 字节统计溢出"))
    })?;
    let avoided_expert_loads = layers.iter().map(|layer| layer.avoided_expert_loads).sum();
    let expert_byte_reduction_ratio = 1.0 - union_expert_bytes as f64 / serial_expert_bytes as f64;

    Ok(FullDepthCausalBatchPlan {
        block_size: routes_by_position.len(),
        layers,
        serial_expert_bytes,
        union_expert_bytes,
        avoided_expert_loads,
        expert_byte_reduction_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router_kind_for_layer;

    fn route(layer: u8, experts: [u16; 6]) -> RouteDecision {
        RouteDecision {
            layer,
            kind: router_kind_for_layer(layer).unwrap(),
            expert_ids: experts.to_vec(),
            weights: vec![0.25; 6],
        }
    }

    #[test]
    fn k4_unions_experts_without_losing_slots_or_weights() {
        let routes = vec![
            route(12, [1, 2, 3, 4, 5, 6]),
            route(12, [1, 2, 7, 8, 9, 10]),
            route(12, [1, 3, 7, 11, 12, 13]),
            route(12, [2, 3, 8, 11, 14, 15]),
        ];
        let plan = build_layer_causal_batch_plan(&routes).unwrap();
        assert_eq!(plan.assignments, 24);
        assert_eq!(plan.unique_experts, 15);
        assert_eq!(plan.avoided_expert_loads, 9);
        assert_eq!(plan.union_expert_bytes, 15 * EXPERT_PAGE_BYTES);
        assert_eq!(plan.serial_expert_bytes, 24 * EXPERT_PAGE_BYTES);
        assert!((plan.expert_byte_reduction_ratio - 0.375).abs() < 1e-12);
        let expert1 = plan
            .experts
            .iter()
            .find(|work| work.expert_id == 1)
            .unwrap();
        assert_eq!(expert1.dispatches.len(), 3);
        assert_eq!(expert1.dispatches[2].token_offset, 2);
        assert_eq!(expert1.dispatches[2].route_slot, 0);
        plan.validate_against(&routes).unwrap();
    }

    #[test]
    fn k1_is_exactly_the_serial_plan() {
        let routes = vec![route(3, [9, 8, 7, 6, 5, 4])];
        let plan = build_layer_causal_batch_plan(&routes).unwrap();
        assert_eq!(plan.unique_experts, 6);
        assert_eq!(plan.avoided_expert_loads, 0);
        assert_eq!(plan.expert_byte_reduction_ratio, 0.0);
    }

    #[test]
    fn rejects_non_causal_block_size_and_mixed_layers() {
        let invalid = vec![route(0, [1, 2, 3, 4, 5, 6]); 2];
        assert!(build_layer_causal_batch_plan(&invalid).is_err());
        let mixed = vec![
            route(0, [1, 2, 3, 4, 5, 6]),
            route(0, [7, 8, 9, 10, 11, 12]),
            route(0, [13, 14, 15, 16, 17, 18]),
            route(1, [19, 20, 21, 22, 23, 24]),
        ];
        assert!(build_layer_causal_batch_plan(&mixed).is_err());
    }

    #[test]
    fn full_depth_plan_requires_all_43_layers_in_order() {
        let mut position = Vec::new();
        for &layer in FULL_DEPTH_LAYERS.iter() {
            position.push(route(layer, [1, 2, 3, 4, 5, 6]));
        }
        let plan = build_full_depth_causal_batch_plan(&[
            position.clone(),
            position.clone(),
            position.clone(),
            position,
        ])
        .unwrap();
        assert_eq!(plan.layers.len(), 43);
        assert_eq!(plan.avoided_expert_loads, 43 * 18);
        assert!((plan.expert_byte_reduction_ratio - 0.75).abs() < 1e-12);

        let mut bad = plan
            .layers
            .iter()
            .map(|_| route(0, [1, 2, 3, 4, 5, 6]))
            .collect::<Vec<_>>();
        bad[1].layer = 9;
        assert!(build_full_depth_causal_batch_plan(&[bad]).is_err());
    }
}
