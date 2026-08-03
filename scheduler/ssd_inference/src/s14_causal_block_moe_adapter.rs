//! S14 K=4/8 causal-block union materialize/upload/grouped-MoE production adapter。
//!
//! 本适配器只负责 MoE 段：它消费 attention/router 已真实产生的 K×top-6 route，
//! 以 FullDepth catalog 重建完整 Range identity，执行 proof/SHA/mmap，整层一次上传，
//! 再把全部唯一专家录进单个 grouped compute command buffer。真实 grouped shader 由
//! `S14CausalBlockGroupedMoeRecorder` 注入；未注入 attention route 或 recorder 返回的
//! 覆盖回执不完整时一律 drain/abort，绝不生成数值回执。

use crate::{
    s14_causal_block_grouped_graph::{
        S14CausalBlockGroupedGraph, S14CausalBlockGroupedGraphTelemetry,
        S14CausalBlockGroupedMoeRecorder,
    },
    s14_causal_block_layer::{
        S14CausalBlockAttentionRouterOutput, S14CausalBlockGroupedMoeOutput,
        S14CausalBlockHiddenBinding, S14CausalBlockLayerInput, S14CausalBlockLayerRangePlan,
        S14CausalBlockUnionBankBinding, S14CausalBlockUnionMaterializeReceipt,
    },
    s14_causal_block_resources::S14CausalBlockUnionBankPlan,
    s14_causal_block_union_materializer::{
        build_causal_block_union_identity_plan, S14CausalBlockUnionMaterializer,
    },
    s14_dynamic_page_cache_readiness::DynamicPageFetchMode,
    s14_dynamic_routed_page_plan::FullDepthExpertCatalog,
    VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{GraphProfile, LayerCausalBatchPlan, RouteDecision, FULL_DEPTH_LAYERS};
use std::{fmt, path::Path, path::PathBuf, sync::Arc};

/// Vulkan backend 与 MoE production owner 之间的最窄对象安全边界。
///
/// attention/HQKV 实现必须在返回 `S14CausalBlockAttentionRouterOutput` 前调用
/// `capture_attention_router_output`；否则 materialize 会 fail-closed。`destroy` 必须在
/// backend/runtime 销毁 VulkanContext 前调用。
pub trait S14CausalBlockVulkanMoeAdapter: fmt::Debug {
    fn begin_block(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        base_position: u32,
        block_size: usize,
    ) -> std::result::Result<(), String>;

    fn capture_attention_router_output(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
        output: &S14CausalBlockAttentionRouterOutput,
    ) -> std::result::Result<(), String>;

    fn materialize_union_ranges(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> std::result::Result<S14CausalBlockUnionMaterializeReceipt, String>;

    fn run_grouped_moe(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> std::result::Result<S14CausalBlockGroupedMoeOutput, String>;

    fn seal_and_drain(&mut self, completed_layers: usize) -> std::result::Result<(), String>;

    fn drain_and_abort(&mut self, completed_layers: usize) -> std::result::Result<(), String>;

    /// terminal/head/checkpoint 已由 orchestrator 验收，允许释放 sealed block 闩锁。
    fn finish_validated_block(&mut self) -> std::result::Result<(), String>;

    /// 显式销毁 graph/timeline/staging；成功后才允许释放 recorder owner。
    fn destroy(&mut self) -> std::result::Result<(), String>;
}

#[derive(Debug)]
enum AdapterLayerPhase {
    AwaitingAttention,
    RoutesReady {
        layer: u8,
        routes: Vec<RouteDecision>,
        post_attention_hidden: S14CausalBlockHiddenBinding,
    },
    UnionUploaded {
        layer: u8,
        routes: Vec<RouteDecision>,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        range_plan: S14CausalBlockLayerRangePlan,
    },
}

#[derive(Debug)]
struct AdapterActiveBlock {
    base_position: u32,
    block_size: usize,
    bank: S14CausalBlockUnionBankBinding,
    next_layer: usize,
    layer_phase: AdapterLayerPhase,
}

#[derive(Debug)]
enum AdapterPhase {
    Idle,
    Active(AdapterActiveBlock),
    LayersSealed,
    Poisoned {
        completed_layers: usize,
        drained: bool,
    },
    Destroyed,
}

/// MoE 段的完整 owner。字段顺序不承担销毁语义；`destroy_inner` 明确先 drain/destroy
/// graph，再 drop recorder，保证 pipeline/descriptor owner 不早于已提交 command 消失。
pub struct S14CausalBlockProductionMoeAdapter<R: S14CausalBlockGroupedMoeRecorder> {
    graph: Option<S14CausalBlockGroupedGraph>,
    recorder: Option<R>,
    materializer: S14CausalBlockUnionMaterializer,
    catalog: FullDepthExpertCatalog,
    cache_root: PathBuf,
    phase: AdapterPhase,
}

impl<R: S14CausalBlockGroupedMoeRecorder> fmt::Debug for S14CausalBlockProductionMoeAdapter<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockProductionMoeAdapter")
            .field("cache_root", &self.cache_root)
            .field("phase", &self.phase)
            .field("graph_present", &self.graph.is_some())
            .field("recorder_present", &self.recorder.is_some())
            .finish_non_exhaustive()
    }
}

impl<R: S14CausalBlockGroupedMoeRecorder> S14CausalBlockProductionMoeAdapter<R> {
    pub fn new(
        ctx: Arc<VulkanContext>,
        catalog: FullDepthExpertCatalog,
        cache_root: &Path,
        fetch_mode: DynamicPageFetchMode,
        recorder: R,
    ) -> Result<Self> {
        let max_union_bytes = S14CausalBlockUnionBankPlan::build(8)
            .map_err(|error| anyhow::anyhow!(error))?
            .allocated_bank_bytes;
        let materializer = S14CausalBlockUnionMaterializer::new(cache_root, fetch_mode)?;
        let graph = S14CausalBlockGroupedGraph::new(ctx, max_union_bytes)?;
        Ok(Self {
            graph: Some(graph),
            recorder: Some(recorder),
            materializer,
            catalog,
            cache_root: cache_root.to_path_buf(),
            phase: AdapterPhase::Idle,
        })
    }

    pub fn telemetry(&self) -> Option<S14CausalBlockGroupedGraphTelemetry> {
        self.graph
            .as_ref()
            .map(S14CausalBlockGroupedGraph::telemetry)
    }

    fn begin_block_inner(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        base_position: u32,
        block_size: usize,
    ) -> Result<()> {
        if !matches!(self.phase, AdapterPhase::Idle) {
            bail!("causal-block MoE adapter 已有未释放 block");
        }
        let plan = S14CausalBlockUnionBankPlan::build(block_size)
            .map_err(|error| anyhow::anyhow!(error))?;
        if bank.buffer == vk::Buffer::null()
            || bank.bank_index >= 2
            || bank.allocated_bank_bytes < plan.allocated_bank_bytes
        {
            bail!("causal-block MoE adapter union bank/K 容量非法");
        }
        base_position
            .checked_add(block_size as u32)
            .context("causal-block MoE block position overflow")?;
        self.recorder
            .as_mut()
            .context("causal-block grouped recorder 已销毁")?
            .begin_block(base_position, block_size)?;
        if let Err(error) =
            self.graph_mut()?
                .begin_block(base_position, block_size, bank.bank_index)
        {
            let cleanup = self
                .recorder
                .as_mut()
                .context("causal-block grouped recorder 已销毁")?
                .finish_block_after_drain(true);
            return match cleanup {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(anyhow::anyhow!(
                    "{error:#}; recorder begin rollback 失败: {cleanup_error:#}"
                )),
            };
        }
        self.phase = AdapterPhase::Active(AdapterActiveBlock {
            base_position,
            block_size,
            bank,
            next_layer: 0,
            layer_phase: AdapterLayerPhase::AwaitingAttention,
        });
        Ok(())
    }

    fn capture_attention_router_output_inner(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
        output: &S14CausalBlockAttentionRouterOutput,
    ) -> Result<()> {
        let active = self.active()?;
        let expected_layer = expected_layer(active.next_layer)?;
        if input.base_position != active.base_position
            || input.layer != expected_layer
            || input.input_token_ids.len() != active.block_size
            || output.forward_calls != 1
            || output.routes.len() != active.block_size
            || output.post_attention_hidden.buffer == vk::Buffer::null()
            || output.post_attention_hidden.block_size != active.block_size
            || output.post_attention_hidden.bytes != input.input_hidden.bytes
            || output.post_attention_hidden.generation
                != input
                    .input_hidden
                    .generation
                    .checked_add(1)
                    .context("causal-block attention hidden generation overflow")?
            || (output.post_attention_hidden.buffer == input.input_hidden.buffer
                && output.post_attention_hidden.offset == input.input_hidden.offset)
            || !matches!(active.layer_phase, AdapterLayerPhase::AwaitingAttention)
        {
            bail!("causal-block MoE attention route/hidden 与 active layer 漂移");
        }
        for route in &output.routes {
            route
                .validate_for(GraphProfile::FullDepth43NativeTop6)
                .context("causal-block MoE online route 非法")?;
            if route.layer != expected_layer {
                bail!("causal-block MoE route layer 漂移");
            }
        }
        let active = self.active_mut()?;
        active.layer_phase = AdapterLayerPhase::RoutesReady {
            layer: expected_layer,
            routes: output.routes.clone(),
            post_attention_hidden: output.post_attention_hidden,
        };
        Ok(())
    }

    fn materialize_union_ranges_inner(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockUnionMaterializeReceipt> {
        let active = self.active()?;
        let (layer, routes, post_attention_hidden) = match &active.layer_phase {
            AdapterLayerPhase::RoutesReady {
                layer,
                routes,
                post_attention_hidden,
            } => (*layer, routes.clone(), *post_attention_hidden),
            _ => bail!("causal-block MoE materialize 前缺少同层真实 attention route"),
        };
        if bank != active.bank
            || range_plan.layer != layer
            || range_plan.block_size != active.block_size
        {
            bail!("causal-block MoE materialize layer/bank/K 漂移");
        }
        let base_position = active.base_position;
        let identity = build_causal_block_union_identity_plan(
            &self.catalog,
            &self.cache_root,
            base_position,
            &routes,
            range_plan,
        )?;
        let materialized = self.materializer.materialize(&identity)?;
        if materialized.layer != layer
            || materialized.unique_experts != range_plan.unique_experts
            || materialized.union_expert_bytes != range_plan.union_expert_bytes
            || materialized.telemetry.physical_ranges != range_plan.physical_ranges
            || materialized.telemetry.proof_assets != range_plan.physical_ranges
            || materialized.telemetry.gpu_upload_copy_regions != 1
        {
            bail!("causal-block MoE proof/SHA materialize 回执漂移");
        }
        self.graph_mut()?.upload_union_layer(bank, &materialized)?;
        let active = self.active_mut()?;
        active.layer_phase = AdapterLayerPhase::UnionUploaded {
            layer,
            routes,
            post_attention_hidden,
            range_plan: range_plan.clone(),
        };
        Ok(S14CausalBlockUnionMaterializeReceipt {
            layer,
            bank_index: bank.bank_index,
            unique_experts: range_plan.unique_experts,
            physical_ranges: range_plan.physical_ranges,
            uploaded_bytes: materialized.union_expert_bytes,
            materialize_calls: 1,
        })
    }

    fn run_grouped_moe_inner(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockGroupedMoeOutput> {
        let active = self.active()?;
        let (expected_layer, captured_routes, captured_hidden, captured_range_plan) =
            match &active.layer_phase {
                AdapterLayerPhase::UnionUploaded {
                    layer,
                    routes,
                    post_attention_hidden,
                    range_plan,
                } => (
                    *layer,
                    routes.clone(),
                    *post_attention_hidden,
                    range_plan.clone(),
                ),
                _ => bail!("causal-block grouped MoE 前缺少同层已提交 union upload"),
            };
        if bank != active.bank
            || post_attention_hidden != captured_hidden
            || routes != captured_routes
            || range_plan != &captured_range_plan
            || batch_plan.layer != expected_layer
            || batch_plan.block_size != active.block_size
        {
            bail!("causal-block grouped MoE 输入与已物化 route/range identity 漂移");
        }
        batch_plan
            .validate_against(routes)
            .context("causal-block grouped MoE 不能无损重建 K×top-6")?;
        let graph = self
            .graph
            .as_mut()
            .context("causal-block grouped graph 已销毁")?;
        let recorder = self
            .recorder
            .as_mut()
            .context("causal-block grouped recorder 已销毁")?;
        let submitted = graph.record_and_submit_grouped_moe(
            post_attention_hidden,
            routes,
            batch_plan,
            range_plan,
            recorder,
        )?;
        if submitted.layer != expected_layer
            || submitted.staged_bytes != range_plan.union_expert_bytes
            || submitted.gpu_upload_copy_regions != 1
            || submitted.grouped_submit_calls != 1
            || submitted.grouped_expert_work_items != batch_plan.unique_experts
            || submitted.lane_assignments != batch_plan.assignments
            || submitted.serial_token_forward_calls != 0
        {
            bail!("causal-block grouped graph submit/coverage 强回执漂移");
        }
        let output = S14CausalBlockGroupedMoeOutput {
            output_hidden: submitted.output_hidden,
            grouped_submit_calls: submitted.grouped_submit_calls,
            serial_token_forward_calls: submitted.serial_token_forward_calls,
            unique_experts: submitted.grouped_expert_work_items,
        };
        let active = self.active_mut()?;
        active.next_layer = active
            .next_layer
            .checked_add(1)
            .context("causal-block MoE layer counter overflow")?;
        active.layer_phase = AdapterLayerPhase::AwaitingAttention;
        Ok(output)
    }

    fn seal_and_drain_inner(&mut self, completed_layers: usize) -> Result<()> {
        let active = self.active()?;
        if completed_layers != FULL_DEPTH_LAYERS.len()
            || active.next_layer != completed_layers
            || !matches!(active.layer_phase, AdapterLayerPhase::AwaitingAttention)
        {
            bail!("causal-block MoE adapter 禁止 seal 不完整/悬空 layer");
        }
        self.graph_mut()?.seal_and_drain()?;
        if let Err(error) = self
            .recorder
            .as_mut()
            .context("causal-block grouped recorder 已销毁")?
            .finish_block_after_drain(false)
        {
            self.phase = AdapterPhase::Poisoned {
                completed_layers,
                drained: true,
            };
            return Err(error.context("causal-block grouped recorder drain 后数值门失败"));
        }
        self.phase = AdapterPhase::LayersSealed;
        Ok(())
    }

    fn finish_validated_block_inner(&mut self) -> Result<()> {
        if !matches!(self.phase, AdapterPhase::LayersSealed) {
            bail!("causal-block MoE adapter 没有已验收的 sealed block");
        }
        self.phase = AdapterPhase::Idle;
        Ok(())
    }

    fn drain_and_abort_inner(&mut self, completed_layers: usize) -> Result<()> {
        let (expected_completed, needs_drain) = match self.phase {
            AdapterPhase::Active(ref active) => (active.next_layer, true),
            AdapterPhase::LayersSealed => (FULL_DEPTH_LAYERS.len(), false),
            AdapterPhase::Poisoned {
                completed_layers,
                drained,
            } => (completed_layers, !drained),
            AdapterPhase::Idle => return Ok(()),
            AdapterPhase::Destroyed => bail!("causal-block MoE adapter 已销毁"),
        };
        if needs_drain {
            self.graph_mut()?.drain_and_abort()?;
            let recorder_result = self
                .recorder
                .as_mut()
                .context("causal-block grouped recorder 已销毁")?
                .finish_block_after_drain(true);
            self.phase = AdapterPhase::Idle;
            recorder_result.context("causal-block grouped recorder abort cleanup 失败")?;
        } else {
            self.phase = AdapterPhase::Idle;
        }
        if completed_layers != expected_completed {
            bail!(
                "causal-block MoE abort completed_layers 漂移: reported={completed_layers} expected={expected_completed}"
            );
        }
        Ok(())
    }

    fn destroy_inner(&mut self) -> Result<()> {
        if matches!(self.phase, AdapterPhase::Destroyed) {
            return Ok(());
        }
        let completed_layers = match self.phase {
            AdapterPhase::Active(ref active) => Some(active.next_layer),
            AdapterPhase::Poisoned {
                completed_layers, ..
            } => Some(completed_layers),
            AdapterPhase::LayersSealed => Some(FULL_DEPTH_LAYERS.len()),
            AdapterPhase::Idle => None,
            AdapterPhase::Destroyed => None,
        };
        if let Some(completed_layers) = completed_layers {
            self.drain_and_abort_inner(completed_layers)?;
        }
        if let Some(graph) = self.graph.take() {
            graph.destroy()?;
        }
        // graph 已 drain 且 command/timeline 已销毁，此时才允许销毁 recorder owner。
        if let Some(recorder) = self.recorder.as_mut() {
            recorder.destroy()?;
        }
        drop(self.recorder.take());
        self.phase = AdapterPhase::Destroyed;
        Ok(())
    }

    fn active(&self) -> Result<&AdapterActiveBlock> {
        match &self.phase {
            AdapterPhase::Active(active) => Ok(active),
            AdapterPhase::Poisoned { .. } => bail!("causal-block MoE adapter 已 poisoned"),
            _ => bail!("causal-block MoE adapter 当前没有 active block"),
        }
    }

    fn active_mut(&mut self) -> Result<&mut AdapterActiveBlock> {
        match &mut self.phase {
            AdapterPhase::Active(active) => Ok(active),
            AdapterPhase::Poisoned { .. } => bail!("causal-block MoE adapter 已 poisoned"),
            _ => bail!("causal-block MoE adapter 当前没有 active block"),
        }
    }

    fn graph_mut(&mut self) -> Result<&mut S14CausalBlockGroupedGraph> {
        self.graph
            .as_mut()
            .context("causal-block grouped graph 已销毁")
    }

    fn poison(&mut self, error: anyhow::Error) -> anyhow::Error {
        let (completed_layers, already_drained) = match &self.phase {
            AdapterPhase::Active(active) => (active.next_layer, false),
            AdapterPhase::Poisoned {
                completed_layers,
                drained,
            } => (*completed_layers, *drained),
            _ => return error,
        };
        if already_drained {
            return anyhow::anyhow!(
                "{error:#}; causal-block MoE graph 已处于 poisoned/drained，completed_layers={completed_layers}"
            );
        }
        let drain = self.graph_mut().and_then(|graph| graph.drain_and_abort());
        match drain {
            Ok(_) => {
                let recorder_cleanup = self
                    .recorder
                    .as_mut()
                    .context("causal-block grouped recorder 已销毁")
                    .and_then(|recorder| recorder.finish_block_after_drain(true));
                self.phase = AdapterPhase::Poisoned {
                    completed_layers,
                    drained: true,
                };
                match recorder_cleanup {
                    Ok(()) => anyhow::anyhow!(
                        "{error:#}; causal-block MoE graph 已立即 timeline drain/abort，completed_layers={completed_layers}"
                    ),
                    Err(cleanup_error) => anyhow::anyhow!(
                        "{error:#}; causal-block MoE graph 已 timeline drain，但 recorder cleanup/数值门失败: {cleanup_error:#}"
                    ),
                }
            }
            Err(drain_error) => {
                self.phase = AdapterPhase::Poisoned {
                    completed_layers,
                    drained: false,
                };
                anyhow::anyhow!(
                    "{error:#}; causal-block MoE graph drain/abort 失败并保持 poisoned: {drain_error:#}"
                )
            }
        }
    }

    fn fail_closed<T>(&mut self, result: Result<T>) -> Result<T> {
        result.map_err(|error| self.poison(error))
    }
}

impl<R: S14CausalBlockGroupedMoeRecorder> S14CausalBlockVulkanMoeAdapter
    for S14CausalBlockProductionMoeAdapter<R>
{
    fn begin_block(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        base_position: u32,
        block_size: usize,
    ) -> std::result::Result<(), String> {
        self.begin_block_inner(bank, base_position, block_size)
            .map_err(|error| format!("{error:#}"))
    }

    fn capture_attention_router_output(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
        output: &S14CausalBlockAttentionRouterOutput,
    ) -> std::result::Result<(), String> {
        let result = self.capture_attention_router_output_inner(input, output);
        self.fail_closed(result)
            .map_err(|error| format!("{error:#}"))
    }

    fn materialize_union_ranges(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> std::result::Result<S14CausalBlockUnionMaterializeReceipt, String> {
        let result = self.materialize_union_ranges_inner(bank, range_plan);
        self.fail_closed(result)
            .map_err(|error| format!("{error:#}"))
    }

    fn run_grouped_moe(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> std::result::Result<S14CausalBlockGroupedMoeOutput, String> {
        let result =
            self.run_grouped_moe_inner(bank, post_attention_hidden, routes, batch_plan, range_plan);
        self.fail_closed(result)
            .map_err(|error| format!("{error:#}"))
    }

    fn seal_and_drain(&mut self, completed_layers: usize) -> std::result::Result<(), String> {
        let result = self.seal_and_drain_inner(completed_layers);
        self.fail_closed(result)
            .map_err(|error| format!("{error:#}"))
    }

    fn drain_and_abort(&mut self, completed_layers: usize) -> std::result::Result<(), String> {
        self.drain_and_abort_inner(completed_layers)
            .map_err(|error| format!("{error:#}"))
    }

    fn finish_validated_block(&mut self) -> std::result::Result<(), String> {
        self.finish_validated_block_inner()
            .map_err(|error| format!("{error:#}"))
    }

    fn destroy(&mut self) -> std::result::Result<(), String> {
        self.destroy_inner().map_err(|error| format!("{error:#}"))
    }
}

fn expected_layer(next_layer: usize) -> Result<u8> {
    FULL_DEPTH_LAYERS
        .get(next_layer)
        .copied()
        .context("causal-block MoE layer index 越出43层")
}
