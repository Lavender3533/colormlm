//! 已物化 causal-block 的层内 routed-expert 批量执行基础件。
//!
//! 与 [`crate::causal_batch`] 的计划器不同，本模块真的执行 gather -> 单次专家
//! batch kernel -> route-slot scatter。每个唯一专家在一层内只会请求加载一次、
//! 调用一次 batch kernel。输出先按原始 `(token, route_slot)` 暂存，最后仍以
//! route slot 0..5 的顺序加权求和，避免因专家排序改变浮点累加顺序。
//!
//! 这个入口只接受已经由严格 causal attention 物化的 K=4/8 hidden。未来未知的
//! 自回归 token 没有这种物化证明，必须被拒绝；本模块不会把 K 次串行 forward
//! 包装成 batch。

use crate::{
    build_layer_causal_batch_plan, LayerCausalBatchPlan, RouteDecision, EXPERTS_PER_TOKEN,
    EXPERT_PAGE_BYTES,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

const EXECUTABLE_BLOCK_SIZES: [usize; 2] = [4, 8];
const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBatchExecutionError(String);

impl fmt::Display for ExpertBatchExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ExpertBatchExecutionError {}

fn execution_error(message: impl Into<String>) -> ExpertBatchExecutionError {
    ExpertBatchExecutionError(message.into())
}

/// 已知 token 的来源。二者都先固定完整 K-token 输入，再由 target 做 causal 验证。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterializedTokenSource {
    ForcedPrefill,
    SpeculativeDraft,
}

/// 层内专家 batch 的因果就绪证明。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum LayerBatchReadiness {
    /// K 个 token 已知，并且当前层 attention 已用严格下三角 mask 一次物化全部
    /// post-attention hidden。只有这个变体允许进入专家 batch。
    CausalAttentionMaterialized {
        layer: u8,
        start_position: u64,
        input_token_ids: Vec<u32>,
        source: MaterializedTokenSource,
    },
    /// 未来 token 尚未由 forced-prefill 或 draft 固定。显式保留这个负变体，
    /// 让调用方不能把“准备以后串行生成”误报为可 batch。
    UnresolvedAutoregressiveFuture {
        known_prefix_position: u64,
        requested_tokens: usize,
    },
}

/// 一层 K=4/8 的全部只读输入。
#[derive(Debug, Clone, Copy)]
pub struct MaterializedLayerBatch<'a> {
    pub layer: u8,
    /// position-major `[K, hidden_size]` F32 reference view。生产后端可以把同一
    /// 契约映射到 BF16/device buffer；这里的 F32 是可移植 correctness 基础件。
    pub hidden_rows: &'a [f32],
    pub hidden_size: usize,
    pub routes: &'a [RouteDecision],
    pub readiness: &'a LayerBatchReadiness,
}

/// 一个已验证专家页最少必须暴露的矩阵形状。
pub trait ExpertPageShape {
    fn hidden_size(&self) -> usize;
    fn intermediate_size(&self) -> usize;
}

impl<T: ExpertPageShape> ExpertPageShape for Arc<T> {
    fn hidden_size(&self) -> usize {
        self.as_ref().hidden_size()
    }

    fn intermediate_size(&self) -> usize {
        self.as_ref().intermediate_size()
    }
}

/// 生产实现应在这里接 Range/cache/Vulkan ready lease。执行器保证每个唯一
/// `(layer, expert_id)` 只调用一次。
pub trait ExpertPageProvider<Page: ExpertPageShape> {
    fn load_verified_expert(&mut self, layer: u8, expert_id: u16) -> Result<Page, String>;
}

/// 一次调用必须完成同一专家的全部命中行，而不是内部回调 K 次 token forward。
pub trait SwiGluExpertBatchKernel<Page: ExpertPageShape> {
    fn run_swiglu_batch(
        &mut self,
        page: &Page,
        input_rows: &[f32],
        rows: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, String>;
}

/// 只包含可直接复算的成本计数，不把理论节省冒充实测 token/s。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertBatchTelemetry {
    pub format: String,
    pub execution_mode: String,
    pub layer: u8,
    pub block_size: usize,
    pub hidden_size: usize,
    pub assignments: usize,
    pub unique_experts: usize,
    pub expert_load_calls: usize,
    pub batch_kernel_calls: usize,
    pub serial_kernel_call_equivalent: usize,
    pub logical_batched_matmul_calls: usize,
    pub serial_logical_matmul_call_equivalent: usize,
    pub total_batch_rows: usize,
    pub min_rows_per_expert: usize,
    pub max_rows_per_expert: usize,
    pub avoided_expert_loads: usize,
    pub serial_expert_bytes: u64,
    pub union_expert_bytes: u64,
    pub avoided_expert_bytes: u64,
    pub expert_byte_reduction_ratio: f64,
    pub gather_bytes: u64,
    pub route_slot_staging_bytes: u64,
    pub output_bytes: u64,
    pub dense_mac_equivalent: u64,
    pub causal_materialization_required: bool,
    pub wall_clock_measured: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertBatchExecution {
    /// position-major `[K, hidden_size]`，已按原 route slot 顺序完成权重求和。
    pub routed_output: Vec<f32>,
    pub telemetry: ExpertBatchTelemetry,
}

fn checked_product(values: &[usize], label: &str) -> Result<usize, ExpertBatchExecutionError> {
    values.iter().try_fold(1usize, |total, value| {
        total
            .checked_mul(*value)
            .ok_or_else(|| execution_error(format!("{label} 维度溢出")))
    })
}

fn checked_bytes(floats: usize, label: &str) -> Result<u64, ExpertBatchExecutionError> {
    u64::try_from(floats)
        .ok()
        .and_then(|value| value.checked_mul(F32_BYTES))
        .ok_or_else(|| execution_error(format!("{label} 字节统计溢出")))
}

fn validate_materialized_request(
    request: &MaterializedLayerBatch<'_>,
) -> Result<LayerCausalBatchPlan, ExpertBatchExecutionError> {
    let block_size = request.routes.len();
    if !EXECUTABLE_BLOCK_SIZES.contains(&block_size) {
        return Err(execution_error(
            "真实专家 batch 只允许 K=4/8；K=1 属于串行路径，其他 K 未冻结",
        ));
    }
    if request.hidden_size == 0 {
        return Err(execution_error("hidden_size 必须大于零"));
    }
    let expected_hidden = checked_product(&[block_size, request.hidden_size], "hidden")?;
    if request.hidden_rows.len() != expected_hidden {
        return Err(execution_error(format!(
            "hidden 必须精确为 [K={}, H={}]，实际元素数 {}",
            block_size,
            request.hidden_size,
            request.hidden_rows.len()
        )));
    }
    if request.hidden_rows.iter().any(|value| !value.is_finite()) {
        return Err(execution_error("hidden 含 NaN/Inf"));
    }

    match request.readiness {
        LayerBatchReadiness::CausalAttentionMaterialized {
            layer,
            start_position,
            input_token_ids,
            ..
        } => {
            if *layer != request.layer {
                return Err(execution_error("因果物化证明的 layer 与执行 layer 漂移"));
            }
            if input_token_ids.len() != block_size {
                return Err(execution_error("因果物化证明必须包含精确 K 个已知 token"));
            }
            let block_size_u64 = u64::try_from(block_size)
                .map_err(|_| execution_error("block_size 无法转换为 position"))?;
            start_position
                .checked_add(block_size_u64)
                .ok_or_else(|| execution_error("causal block position 溢出"))?;
        }
        LayerBatchReadiness::UnresolvedAutoregressiveFuture { .. } => {
            return Err(execution_error(
                "未来自回归 token 尚未知；禁止串行 K 次 forward 伪装为专家 batch",
            ));
        }
    }

    let plan = build_layer_causal_batch_plan(request.routes)
        .map_err(|error| execution_error(error.to_string()))?;
    if plan.layer != request.layer {
        return Err(execution_error("路由计划 layer 与 hidden 执行 layer 漂移"));
    }
    Ok(plan)
}

/// 执行一次已物化层内 K=4/8 专家 batch。
///
/// 成功返回前不会暴露半成品输出。每个专家的所有 route assignment 会被 gather
/// 成 `[rows_for_expert, hidden]`，只发起一次 provider load 和一次 kernel 调用。
pub fn execute_materialized_layer_expert_batch<Page, Provider, Kernel>(
    request: &MaterializedLayerBatch<'_>,
    provider: &mut Provider,
    kernel: &mut Kernel,
) -> Result<ExpertBatchExecution, ExpertBatchExecutionError>
where
    Page: ExpertPageShape,
    Provider: ExpertPageProvider<Page>,
    Kernel: SwiGluExpertBatchKernel<Page>,
{
    let plan = validate_materialized_request(request)?;
    let block_size = plan.block_size;
    let hidden_size = request.hidden_size;
    let slot_elements = checked_product(
        &[block_size, EXPERTS_PER_TOKEN, hidden_size],
        "route slot staging",
    )?;
    let mut route_slot_outputs = vec![0.0f32; slot_elements];
    let mut min_rows_per_expert = usize::MAX;
    let mut max_rows_per_expert = 0usize;
    let mut dense_mac_equivalent = 0u64;

    for work in &plan.experts {
        let page = provider
            .load_verified_expert(plan.layer, work.expert_id)
            .map_err(|error| {
                execution_error(format!(
                    "L{} E{} 加载失败: {error}",
                    plan.layer, work.expert_id
                ))
            })?;
        if page.hidden_size() != hidden_size || page.intermediate_size() == 0 {
            return Err(execution_error(format!(
                "L{} E{} 专家 shape 漂移: hidden={} intermediate={}",
                plan.layer,
                work.expert_id,
                page.hidden_size(),
                page.intermediate_size()
            )));
        }

        let rows = work.dispatches.len();
        min_rows_per_expert = min_rows_per_expert.min(rows);
        max_rows_per_expert = max_rows_per_expert.max(rows);
        let gathered_elements = checked_product(&[rows, hidden_size], "expert gather")?;
        let mut gathered = Vec::with_capacity(gathered_elements);
        for dispatch in &work.dispatches {
            let start = dispatch
                .token_offset
                .checked_mul(hidden_size)
                .ok_or_else(|| execution_error("hidden row offset 溢出"))?;
            gathered.extend_from_slice(&request.hidden_rows[start..start + hidden_size]);
        }

        let outputs = kernel
            .run_swiglu_batch(&page, &gathered, rows, hidden_size)
            .map_err(|error| {
                execution_error(format!(
                    "L{} E{} batch kernel 失败: {error}",
                    plan.layer, work.expert_id
                ))
            })?;
        if outputs.len() != gathered_elements {
            return Err(execution_error(format!(
                "L{} E{} batch kernel 输出应为 [{rows},{hidden_size}]，实际元素数 {}",
                plan.layer,
                work.expert_id,
                outputs.len()
            )));
        }
        if outputs.iter().any(|value| !value.is_finite()) {
            return Err(execution_error(format!(
                "L{} E{} batch kernel 输出含 NaN/Inf",
                plan.layer, work.expert_id
            )));
        }

        for (batch_row, dispatch) in work.dispatches.iter().enumerate() {
            let source = batch_row * hidden_size;
            let destination =
                (dispatch.token_offset * EXPERTS_PER_TOKEN + dispatch.route_slot) * hidden_size;
            route_slot_outputs[destination..destination + hidden_size]
                .copy_from_slice(&outputs[source..source + hidden_size]);
        }

        let expert_macs = u64::try_from(rows)
            .ok()
            .and_then(|rows| rows.checked_mul(hidden_size as u64))
            .and_then(|value| value.checked_mul(page.intermediate_size() as u64))
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| execution_error("dense MAC telemetry 溢出"))?;
        dense_mac_equivalent = dense_mac_equivalent
            .checked_add(expert_macs)
            .ok_or_else(|| execution_error("dense MAC telemetry 累加溢出"))?;
    }

    let output_elements = checked_product(&[block_size, hidden_size], "routed output")?;
    let mut routed_output = vec![0.0f32; output_elements];
    for token_offset in 0..block_size {
        for route_slot in 0..EXPERTS_PER_TOKEN {
            let route_weight = request.routes[token_offset].weights[route_slot];
            let source = (token_offset * EXPERTS_PER_TOKEN + route_slot) * hidden_size;
            let destination = token_offset * hidden_size;
            for hidden_offset in 0..hidden_size {
                routed_output[destination + hidden_offset] +=
                    route_weight * route_slot_outputs[source + hidden_offset];
            }
        }
    }

    let assignments = plan.assignments;
    let unique_experts = plan.unique_experts;
    let avoided_expert_bytes = plan
        .serial_expert_bytes
        .checked_sub(plan.union_expert_bytes)
        .ok_or_else(|| execution_error("专家节省字节统计下溢"))?;
    let gather_elements = checked_product(&[assignments, hidden_size], "gather telemetry")?;
    let telemetry = ExpertBatchTelemetry {
        format: "polaris-layer-expert-batch-telemetry-v1".into(),
        execution_mode: "materialized_unique_expert_batch".into(),
        layer: plan.layer,
        block_size,
        hidden_size,
        assignments,
        unique_experts,
        expert_load_calls: unique_experts,
        batch_kernel_calls: unique_experts,
        serial_kernel_call_equivalent: assignments,
        logical_batched_matmul_calls: unique_experts * 3,
        serial_logical_matmul_call_equivalent: assignments * 3,
        total_batch_rows: assignments,
        min_rows_per_expert,
        max_rows_per_expert,
        avoided_expert_loads: plan.avoided_expert_loads,
        serial_expert_bytes: plan.serial_expert_bytes,
        union_expert_bytes: plan.union_expert_bytes,
        avoided_expert_bytes,
        expert_byte_reduction_ratio: plan.expert_byte_reduction_ratio,
        gather_bytes: checked_bytes(gather_elements, "gather telemetry")?,
        route_slot_staging_bytes: checked_bytes(slot_elements, "route slot telemetry")?,
        output_bytes: checked_bytes(output_elements, "output telemetry")?,
        dense_mac_equivalent,
        causal_materialization_required: true,
        wall_clock_measured: false,
    };
    debug_assert_eq!(
        telemetry.union_expert_bytes,
        unique_experts as u64 * EXPERT_PAGE_BYTES
    );
    Ok(ExpertBatchExecution {
        routed_output,
        telemetry,
    })
}

/// 小矩阵 CPU correctness 页；权重均为 row-major F32。
#[derive(Debug, Clone, PartialEq)]
pub struct DenseSwiGluExpert {
    hidden_size: usize,
    intermediate_size: usize,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

impl DenseSwiGluExpert {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
    ) -> Result<Self, ExpertBatchExecutionError> {
        if hidden_size == 0 || intermediate_size == 0 {
            return Err(execution_error("CPU reference 专家维度必须大于零"));
        }
        let in_projection = checked_product(&[intermediate_size, hidden_size], "gate/up")?;
        let down_projection = checked_product(&[hidden_size, intermediate_size], "down")?;
        if gate.len() != in_projection || up.len() != in_projection || down.len() != down_projection
        {
            return Err(execution_error("CPU reference 专家权重 shape 漂移"));
        }
        if gate
            .iter()
            .chain(up.iter())
            .chain(down.iter())
            .any(|value| !value.is_finite())
        {
            return Err(execution_error("CPU reference 专家权重含 NaN/Inf"));
        }
        Ok(Self {
            hidden_size,
            intermediate_size,
            gate,
            up,
            down,
        })
    }
}

impl ExpertPageShape for DenseSwiGluExpert {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
}

/// 内存页提供器，仅用于 CPU reference/后端 bring-up；每次 load 记录精确调用数。
#[derive(Debug, Default)]
pub struct InMemoryDenseExpertProvider {
    pages: BTreeMap<(u8, u16), Arc<DenseSwiGluExpert>>,
    load_calls: BTreeMap<(u8, u16), usize>,
}

impl InMemoryDenseExpertProvider {
    pub fn insert(&mut self, layer: u8, expert_id: u16, page: DenseSwiGluExpert) {
        self.pages.insert((layer, expert_id), Arc::new(page));
    }

    pub fn page(&self, layer: u8, expert_id: u16) -> Option<&DenseSwiGluExpert> {
        self.pages.get(&(layer, expert_id)).map(Arc::as_ref)
    }

    pub fn load_calls(&self, layer: u8, expert_id: u16) -> usize {
        self.load_calls
            .get(&(layer, expert_id))
            .copied()
            .unwrap_or(0)
    }

    pub fn total_load_calls(&self) -> usize {
        self.load_calls.values().sum()
    }
}

impl ExpertPageProvider<Arc<DenseSwiGluExpert>> for InMemoryDenseExpertProvider {
    fn load_verified_expert(
        &mut self,
        layer: u8,
        expert_id: u16,
    ) -> Result<Arc<DenseSwiGluExpert>, String> {
        let page = self
            .pages
            .get(&(layer, expert_id))
            .cloned()
            .ok_or_else(|| format!("缺少 L{layer} E{expert_id} CPU reference 页"))?;
        *self.load_calls.entry((layer, expert_id)).or_default() += 1;
        Ok(page)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuBatchKernelCall {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub rows: usize,
}

/// 纯 Rust F32 CPU SwiGLU batch reference。一次函数调用处理一个专家的全部行。
#[derive(Debug, Default)]
pub struct CpuDenseSwiGluBatchKernel {
    pub calls: Vec<CpuBatchKernelCall>,
}

impl SwiGluExpertBatchKernel<Arc<DenseSwiGluExpert>> for CpuDenseSwiGluBatchKernel {
    fn run_swiglu_batch(
        &mut self,
        page: &Arc<DenseSwiGluExpert>,
        input_rows: &[f32],
        rows: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, String> {
        if hidden_size != page.hidden_size || input_rows.len() != rows * hidden_size {
            return Err("CPU batch 输入 shape 漂移".into());
        }
        self.calls.push(CpuBatchKernelCall {
            hidden_size,
            intermediate_size: page.intermediate_size,
            rows,
        });
        let mut output = vec![0.0f32; rows * hidden_size];
        let mut activated = vec![0.0f32; page.intermediate_size];
        for row in 0..rows {
            let input = &input_rows[row * hidden_size..(row + 1) * hidden_size];
            for (intermediate, activated_value) in activated.iter_mut().enumerate() {
                let weight_offset = intermediate * hidden_size;
                let mut gate = 0.0f32;
                let mut up = 0.0f32;
                for (hidden, &input_value) in input.iter().enumerate() {
                    gate += input_value * page.gate[weight_offset + hidden];
                    up += input_value * page.up[weight_offset + hidden];
                }
                let silu = gate / (1.0 + (-gate).exp());
                *activated_value = silu * up;
            }
            let output_row = &mut output[row * hidden_size..(row + 1) * hidden_size];
            for (hidden, output_value) in output_row.iter_mut().enumerate() {
                let weight_offset = hidden * page.intermediate_size;
                let mut value = 0.0f32;
                for (intermediate, &activated_value) in activated.iter().enumerate() {
                    value += activated_value * page.down[weight_offset + intermediate];
                }
                *output_value = value;
            }
        }
        Ok(output)
    }
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

    fn expert(expert_id: u16, hidden: usize, intermediate: usize) -> DenseSwiGluExpert {
        let seed = expert_id as f32 + 1.0;
        let gate = (0..intermediate * hidden)
            .map(|index| (seed + index as f32) * 0.013 - 0.11)
            .collect();
        let up = (0..intermediate * hidden)
            .map(|index| (seed * 0.7 - index as f32) * 0.017 + 0.03)
            .collect();
        let down = (0..hidden * intermediate)
            .map(|index| (seed + index as f32 * 0.5) * 0.019 - 0.07)
            .collect();
        DenseSwiGluExpert::new(hidden, intermediate, gate, up, down).unwrap()
    }

    fn provider_for_routes(
        layer: u8,
        routes: &[RouteDecision],
        hidden: usize,
        intermediate: usize,
    ) -> InMemoryDenseExpertProvider {
        let mut provider = InMemoryDenseExpertProvider::default();
        for route in routes {
            for &expert_id in &route.expert_ids {
                if provider.page(layer, expert_id).is_none() {
                    provider.insert(layer, expert_id, expert(expert_id, hidden, intermediate));
                }
            }
        }
        provider
    }

    fn direct_expert(page: &DenseSwiGluExpert, input: &[f32]) -> Vec<f32> {
        let mut activated = vec![0.0f32; page.intermediate_size];
        for (intermediate, activated_value) in activated.iter_mut().enumerate() {
            let offset = intermediate * page.hidden_size;
            let mut gate = 0.0f32;
            let mut up = 0.0f32;
            for (hidden, &input_value) in input.iter().enumerate() {
                gate += input_value * page.gate[offset + hidden];
                up += input_value * page.up[offset + hidden];
            }
            *activated_value = gate / (1.0 + (-gate).exp()) * up;
        }
        let mut output = vec![0.0f32; page.hidden_size];
        for (hidden, output_value) in output.iter_mut().enumerate() {
            for (intermediate, &activated_value) in activated.iter().enumerate() {
                *output_value +=
                    activated_value * page.down[hidden * page.intermediate_size + intermediate];
            }
        }
        output
    }

    fn serial_reference(
        provider: &InMemoryDenseExpertProvider,
        routes: &[RouteDecision],
        hidden_rows: &[f32],
        hidden: usize,
    ) -> Vec<f32> {
        let mut result = vec![0.0f32; routes.len() * hidden];
        for (token_offset, route) in routes.iter().enumerate() {
            let input = &hidden_rows[token_offset * hidden..(token_offset + 1) * hidden];
            for route_slot in 0..EXPERTS_PER_TOKEN {
                let page = provider
                    .page(route.layer, route.expert_ids[route_slot])
                    .unwrap();
                let expert_output = direct_expert(page, input);
                for hidden_offset in 0..hidden {
                    result[token_offset * hidden + hidden_offset] +=
                        route.weights[route_slot] * expert_output[hidden_offset];
                }
            }
        }
        result
    }

    fn readiness(layer: u8, block_size: usize) -> LayerBatchReadiness {
        LayerBatchReadiness::CausalAttentionMaterialized {
            layer,
            start_position: 17,
            input_token_ids: (0..block_size as u32).map(|value| 100 + value).collect(),
            source: MaterializedTokenSource::SpeculativeDraft,
        }
    }

    #[test]
    fn k4_batches_each_unique_expert_once_and_matches_serial_reference_bitwise() {
        let layer = 12;
        let hidden = 3;
        let routes = vec![
            route(layer, [1, 2, 3, 4, 5, 6]),
            route(layer, [1, 2, 7, 8, 9, 10]),
            route(layer, [1, 3, 7, 11, 12, 13]),
            route(layer, [2, 3, 8, 11, 14, 15]),
        ];
        let hidden_rows = vec![
            0.2, -0.4, 0.8, -0.1, 0.7, 0.3, 1.1, -0.2, 0.5, -0.6, 0.9, 0.4,
        ];
        let proof = readiness(layer, 4);
        let mut provider = provider_for_routes(layer, &routes, hidden, 2);
        let expected = serial_reference(&provider, &routes, &hidden_rows, hidden);
        let mut kernel = CpuDenseSwiGluBatchKernel::default();
        let execution = execute_materialized_layer_expert_batch(
            &MaterializedLayerBatch {
                layer,
                hidden_rows: &hidden_rows,
                hidden_size: hidden,
                routes: &routes,
                readiness: &proof,
            },
            &mut provider,
            &mut kernel,
        )
        .unwrap();

        assert_eq!(execution.routed_output, expected);
        assert_eq!(provider.total_load_calls(), 15);
        assert!((1..=15).all(|expert_id| provider.load_calls(layer, expert_id) == 1));
        assert_eq!(kernel.calls.len(), 15);
        assert_eq!(kernel.calls.iter().map(|call| call.rows).sum::<usize>(), 24);
        assert_eq!(execution.telemetry.assignments, 24);
        assert_eq!(execution.telemetry.unique_experts, 15);
        assert_eq!(execution.telemetry.avoided_expert_loads, 9);
        assert_eq!(execution.telemetry.batch_kernel_calls, 15);
        assert_eq!(execution.telemetry.serial_kernel_call_equivalent, 24);
        assert_eq!(execution.telemetry.logical_batched_matmul_calls, 45);
        assert_eq!(
            execution.telemetry.serial_logical_matmul_call_equivalent,
            72
        );
        assert_eq!(execution.telemetry.max_rows_per_expert, 3);
        assert_eq!(execution.telemetry.min_rows_per_expert, 1);
        assert_eq!(execution.telemetry.dense_mac_equivalent, 24 * 3 * 2 * 3);
        assert!(!execution.telemetry.wall_clock_measured);
    }

    #[test]
    fn k8_runs_six_real_eight_row_expert_batches() {
        let layer = 28;
        let hidden = 4;
        let routes = vec![route(layer, [9, 8, 7, 6, 5, 4]); 8];
        let hidden_rows: Vec<f32> = (0..8 * hidden)
            .map(|index| index as f32 * 0.03 - 0.2)
            .collect();
        let proof = readiness(layer, 8);
        let mut provider = provider_for_routes(layer, &routes, hidden, 3);
        let expected = serial_reference(&provider, &routes, &hidden_rows, hidden);
        let mut kernel = CpuDenseSwiGluBatchKernel::default();
        let execution = execute_materialized_layer_expert_batch(
            &MaterializedLayerBatch {
                layer,
                hidden_rows: &hidden_rows,
                hidden_size: hidden,
                routes: &routes,
                readiness: &proof,
            },
            &mut provider,
            &mut kernel,
        )
        .unwrap();

        assert_eq!(execution.routed_output, expected);
        assert_eq!(provider.total_load_calls(), 6);
        assert_eq!(kernel.calls.len(), 6);
        assert!(kernel.calls.iter().all(|call| call.rows == 8));
        assert_eq!(execution.telemetry.unique_experts, 6);
        assert_eq!(execution.telemetry.assignments, 48);
        assert_eq!(execution.telemetry.avoided_expert_loads, 42);
        assert!((execution.telemetry.expert_byte_reduction_ratio - 0.875).abs() < 1e-12);
    }

    #[test]
    fn unresolved_future_tokens_are_rejected_before_load_or_kernel() {
        let layer = 7;
        let routes = vec![route(layer, [1, 2, 3, 4, 5, 6]); 4];
        let hidden_rows = vec![0.0f32; 4 * 2];
        let proof = LayerBatchReadiness::UnresolvedAutoregressiveFuture {
            known_prefix_position: 20,
            requested_tokens: 4,
        };
        let mut provider = provider_for_routes(layer, &routes, 2, 2);
        let mut kernel = CpuDenseSwiGluBatchKernel::default();
        let error = execute_materialized_layer_expert_batch(
            &MaterializedLayerBatch {
                layer,
                hidden_rows: &hidden_rows,
                hidden_size: 2,
                routes: &routes,
                readiness: &proof,
            },
            &mut provider,
            &mut kernel,
        )
        .unwrap_err();

        assert!(error.to_string().contains("禁止串行 K 次"));
        assert_eq!(provider.total_load_calls(), 0);
        assert!(kernel.calls.is_empty());
    }

    #[test]
    fn k1_and_incomplete_materialization_proofs_fail_closed() {
        let layer = 3;
        let route = route(layer, [1, 2, 3, 4, 5, 6]);
        let one_route = vec![route.clone()];
        let hidden_rows = vec![0.1f32, 0.2];
        let k1_proof = readiness(layer, 1);
        let mut provider = provider_for_routes(layer, &one_route, 2, 2);
        let mut kernel = CpuDenseSwiGluBatchKernel::default();
        let k1_error = execute_materialized_layer_expert_batch(
            &MaterializedLayerBatch {
                layer,
                hidden_rows: &hidden_rows,
                hidden_size: 2,
                routes: &one_route,
                readiness: &k1_proof,
            },
            &mut provider,
            &mut kernel,
        )
        .unwrap_err();
        assert!(k1_error.to_string().contains("只允许 K=4/8"));

        let routes = vec![route; 4];
        let bad_proof = LayerBatchReadiness::CausalAttentionMaterialized {
            layer,
            start_position: 0,
            input_token_ids: vec![1, 2, 3],
            source: MaterializedTokenSource::ForcedPrefill,
        };
        let error = execute_materialized_layer_expert_batch(
            &MaterializedLayerBatch {
                layer,
                hidden_rows: &[0.0f32; 8],
                hidden_size: 2,
                routes: &routes,
                readiness: &bad_proof,
            },
            &mut provider,
            &mut kernel,
        )
        .unwrap_err();
        assert!(error.to_string().contains("精确 K 个已知 token"));
        assert_eq!(provider.total_load_calls(), 0);
        assert!(kernel.calls.is_empty());
    }

    struct ShortOutputKernel;

    impl SwiGluExpertBatchKernel<Arc<DenseSwiGluExpert>> for ShortOutputKernel {
        fn run_swiglu_batch(
            &mut self,
            _page: &Arc<DenseSwiGluExpert>,
            _input_rows: &[f32],
            rows: usize,
            hidden_size: usize,
        ) -> Result<Vec<f32>, String> {
            Ok(vec![0.0; rows * hidden_size - 1])
        }
    }

    #[test]
    fn malformed_kernel_output_is_never_partially_committed() {
        let layer = 6;
        let routes = vec![route(layer, [1, 2, 3, 4, 5, 6]); 4];
        let proof = readiness(layer, 4);
        let mut provider = provider_for_routes(layer, &routes, 2, 2);
        let mut kernel = ShortOutputKernel;
        let error = execute_materialized_layer_expert_batch(
            &MaterializedLayerBatch {
                layer,
                hidden_rows: &[0.2f32; 8],
                hidden_size: 2,
                routes: &routes,
                readiness: &proof,
            },
            &mut provider,
            &mut kernel,
        )
        .unwrap_err();
        assert!(error.to_string().contains("batch kernel 输出应为"));
    }
}
