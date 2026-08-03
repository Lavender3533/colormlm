//! 纯 S14 StarFold 的 FullDepth43/K4 production 生命周期适配器。
//!
//! 本 owner 不生成 route、hidden 或 checkpoint fixture。每一层必须先由外部 production
//! stage 在真实 attention/router command 中产出四行在线 top-6、F32 MoE 输入以及同层
//! prefix-checkpoint 写回，再由 `S14StarfoldB4RoutedLayerOwner` 流式执行 W1/W3/W2，最后
//! 由同一 production stage 把 24 条 routed-down 分支归并成下一层 device hidden。
//! 43 层全部完成后才允许 seal checkpoint chain；token/head/最长前缀提交仍在本 owner
//! 之外，避免把结构回执冒充已发布文本。

use crate::{
    s14_causal_block_hc_qkv_adapter::{
        S14CausalBlockHcQkvLayerRecordingReceipt, S14CausalBlockProductionHcQkvAdapter,
        S14CausalBlockVulkanHcQkvAdapter, S14Position0CommittedGenerationProvenance,
    },
    s14_causal_block_hc_qkv_recorder::{
        S14CausalBlockHiddenBank, S14CausalBlockProductionHcQkvLayerRecorder,
        S14CausalBlockStarfoldPrefillPrefixProduct, S14CausalBlockStarfoldTerminalBlockOwners,
    },
    s14_causal_block_layer::{
        S14CausalBlockAttentionRouterOutput, S14CausalBlockGroupedMoeOutput,
        S14CausalBlockHiddenBinding, S14CausalBlockLayerInput,
        S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE,
    },
    s14_causal_block_prefix_producer::S14CausalBlockPrefixStateProducer,
    s14_causal_block_production_bundle::S14CausalBlockProductionHcQkvResourceProvider,
    s14_dynamic_routed_page_plan::FullDepthExpertCatalog,
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_starfold_b4_owner::{
        S14StarfoldB4RoutedLayerOwner, S14StarfoldB4RoutedLayerReceipt,
        S14StarfoldRoutedDownBinding,
    },
    s14_starfold_cache::{STARFOLD_B4_LANES, STARFOLD_TOP_K},
    s14_starfold_hc_bridge::{S14StarfoldHcBridgeOwner, S14StarfoldHcPrepareReceipt},
    s14_starfold_mxfp4_tile::S14StarfoldMxfp4ExternalSlice,
    s14_starfold_runtime::{
        S14StarfoldCommitBoundary, S14StarfoldRuntime, S14_STARFOLD_DEFAULT_MICROTILE_BYTES,
    },
    s14_starfold_shared_reduce_owner::S14StarfoldSharedReduceOwner,
    s14_starwave_draft::S14StarwaveDraftProposal,
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    DecoderStateV1, GraphProfile, MaterializedTokenSource, RouteDecision, FULL_DEPTH_LAYERS,
    VOCAB_SIZE,
};
use std::{fmt, path::Path, sync::Arc};

const K4: usize = STARFOLD_B4_LANES;
const F32_BYTES: u64 = 4;
const BF16_BYTES: u64 = 2;
const HIDDEN: u64 = 4096;

/// 外部 attention/router/checkpoint stage 的单层输入。token 来自调用方当前真实 K4
/// block；adapter 从不替换 BOS、固定轨迹或 host fixture。
#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldK4AttentionInput<'a> {
    pub base_position: u32,
    pub layer: u8,
    pub input_token_ids: &'a [u32],
    pub input_hidden: S14CausalBlockHiddenBinding,
    pub source: &'a MaterializedTokenSource,
}

/// 一次真实 K-lane attention/router command 的输出。
#[derive(Clone, Debug)]
pub struct S14StarfoldK4AttentionCheckpointOutput {
    pub routes: Vec<RouteDecision>,
    pub post_attention_hidden: S14CausalBlockHiddenBinding,
    /// `[4,4096]` F32；由真实 post-attention/HC stage 产生，直接喂给 StarFold W1/W3。
    pub moe_input_f32: S14StarfoldMxfp4ExternalSlice,
    /// 同层真实 prefix arena 录制回执；只接受现有 K4 producer 的强类型回执。
    pub checkpoint: S14CausalBlockHcQkvLayerRecordingReceipt,
    pub attention_router_submit_calls: u32,
    pub checkpoint_recording_calls: u32,
    pub serial_token_forward_calls: u32,
}

/// StarFold routed-down 归并、shared expert/HC-post 后的下一层 hidden 回执。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4HiddenCommitReceipt {
    pub base_position: u32,
    pub layer: u8,
    pub next_hidden: S14CausalBlockHiddenBinding,
    pub routed_reduce_dispatch_calls: u32,
    pub hc_post_dispatch_calls: u32,
    pub queue_submit_calls: u32,
    pub serial_token_forward_calls: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4BeginReceipt {
    pub base_position: u32,
    pub block_size: usize,
    pub begin_calls: u32,
    pub serial_token_forward_calls: u32,
}

/// checkpoint arena 在 43 层完成后的 seal 回执。这里只证明 K4 candidates 已闭合，
/// 不表示 final head、最长前缀或 session commit 已发生。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4CheckpointSealReceipt {
    pub base_position: u32,
    pub block_size: usize,
    pub completed_layers: usize,
    pub checkpoint_count: usize,
    pub prefix_program_seal_calls: u32,
    pub checkpoint_commit_calls: u32,
    pub serial_token_forward_calls: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4AbortReceipt {
    pub completed_layers: usize,
    pub drain_calls: u32,
    pub aborted: bool,
}

/// 真实 HC/QKV、prefix checkpoint 与 routed-down reduction owner 的最窄接线面。
/// 实现方必须持有其 Vulkan buffers/pipelines/command resources，直到对应 queue/fence
/// 完成。该 trait 故意没有默认实现，也没有 host hidden/route 注入入口。
pub trait S14StarfoldK4ProductionStage: fmt::Debug {
    fn begin_k4_block(
        &mut self,
        base_position: u32,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
    ) -> std::result::Result<S14StarfoldK4BeginReceipt, String>;

    /// 普通 draft begin 必须继续拒绝 base0。只有携带已验证 committed-generation
    /// provenance 的专用入口可以把 position0 交给下游 HC/QKV typed begin。
    fn begin_position0_committed_generation_k4_block(
        &mut self,
        _provenance: S14Position0CommittedGenerationProvenance,
        _input_token_ids: &[u32],
        _initial_hidden: S14CausalBlockHiddenBinding,
    ) -> std::result::Result<S14StarfoldK4BeginReceipt, String> {
        Err("S14 StarFold stage 未实现 position0 committed-generation begin".into())
    }

    /// 普通 stage 默认不开放 ForcedPrefill；只有绑定同源 provider/prefix producer 的
    /// concrete stage 才能实现该入口。
    fn begin_teacher_forced_prefill_k4_block(
        &mut self,
        _base_position: u32,
        _input_token_ids: &[u32],
        _initial_hidden: S14CausalBlockHiddenBinding,
    ) -> std::result::Result<S14StarfoldK4BeginReceipt, String> {
        Err("S14 StarFold stage 未实现 ForcedPrefill begin".into())
    }

    fn produce_attention_route_and_checkpoint(
        &mut self,
        input: S14StarfoldK4AttentionInput<'_>,
    ) -> std::result::Result<S14StarfoldK4AttentionCheckpointOutput, String>;

    fn commit_routed_down_to_next_hidden(
        &mut self,
        input: S14StarfoldK4AttentionInput<'_>,
        routes: &[RouteDecision],
        down: S14StarfoldRoutedDownBinding,
        expert: &S14StarfoldB4RoutedLayerReceipt,
    ) -> std::result::Result<S14StarfoldK4HiddenCommitReceipt, String>;

    fn seal_checkpoint_chain(
        &mut self,
        completed_layers: usize,
        final_hidden: S14CausalBlockHiddenBinding,
    ) -> std::result::Result<S14StarfoldK4CheckpointSealReceipt, String>;

    /// terminal/head/checkpoint consumer 已接管 sealed block 后才调用。
    fn finish_validated_block(&mut self) -> std::result::Result<(), String>;

    /// 失败回滚必须以 stage 内部已经物理完成的层数为权威，禁止调用方把提交前的旧计数
    /// 重新灌入。回执中的 `completed_layers` 是实际被 drain 的层数。
    fn drain_and_abort_k4_block(
        &mut self,
    ) -> std::result::Result<S14StarfoldK4AbortReceipt, String>;

    fn destroy(&mut self) -> std::result::Result<(), String>;
}

#[derive(Clone, Debug)]
pub struct S14StarfoldK4LayerReceipt {
    pub layer: u8,
    pub base_position: u32,
    pub routes: Vec<RouteDecision>,
    pub checkpoint: S14CausalBlockHcQkvLayerRecordingReceipt,
    pub expert: S14StarfoldB4RoutedLayerReceipt,
    pub hidden_commit: S14StarfoldK4HiddenCommitReceipt,
}

#[derive(Clone, Debug)]
pub struct S14StarfoldK4FullDepthReceipt {
    pub base_position: u32,
    pub block_size: usize,
    pub physical_input_token_ids: [u32; STARFOLD_B4_LANES],
    pub completed_layers: usize,
    pub routes_by_position: Vec<Vec<RouteDecision>>,
    pub layers: Vec<S14StarfoldK4LayerReceipt>,
    pub final_hidden: S14CausalBlockHiddenBinding,
    pub checkpoint_seal: S14StarfoldK4CheckpointSealReceipt,
    /// ForcedPrefill 与 SpeculativeDraft 的物理 seal 不可互换。
    pub source: MaterializedTokenSource,
    pub packed_uploads: u32,
    pub packed_upload_bytes: u64,
    pub lane_dispatches: u32,
    pub serial_token_forward_calls: u32,
    pub terminal_ready: bool,
    pub token_committed: bool,
}

#[derive(Debug)]
struct ActiveK4Block {
    base_position: u32,
    input_token_ids: Vec<u32>,
    current_hidden: S14CausalBlockHiddenBinding,
    source: MaterializedTokenSource,
    next_layer_index: usize,
    routes_by_position: Vec<Vec<RouteDecision>>,
    layers: Vec<S14StarfoldK4LayerReceipt>,
}

#[derive(Debug)]
enum AdapterPhase {
    Idle,
    Active(ActiveK4Block),
    LayersSealed { base_position: u32 },
    Poisoned { completed_layers: usize },
    Destroyed,
}

/// 唯一 production owner：独占 StarFold runtime、B4 routed owner 与外部 HC/checkpoint
/// stage。所有层均按 `FULL_DEPTH_LAYERS` 的网络深度顺序执行；不存在按 token 调用旧
/// whole-token forward 的路径。
pub struct S14StarfoldK4ProductionAdapter<S: S14StarfoldK4ProductionStage> {
    runtime: Option<S14StarfoldRuntime>,
    b4_owner: Option<S14StarfoldB4RoutedLayerOwner>,
    stage: Option<S>,
    catalog: Arc<FullDepthExpertCatalog>,
    phase: AdapterPhase,
    validated_blocks: u64,
    last_finished_base_position: Option<u32>,
    committed_rebinds: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct S14StarfoldK4AdapterOwnerCounts {
    pub runtime_owners: usize,
    pub microtile_window_owners: usize,
    pub b4_owners: usize,
    pub stage_owners: usize,
    pub catalog_owners: usize,
}

impl<S: S14StarfoldK4ProductionStage> fmt::Debug for S14StarfoldK4ProductionAdapter<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14StarfoldK4ProductionAdapter")
            .field("phase", &self.phase)
            .field("runtime_present", &self.runtime.is_some())
            .field("b4_owner_present", &self.b4_owner.is_some())
            .field("stage_present", &self.stage.is_some())
            .field("validated_blocks", &self.validated_blocks)
            .field(
                "last_finished_base_position",
                &self.last_finished_base_position,
            )
            .field("committed_rebinds", &self.committed_rebinds)
            .finish_non_exhaustive()
    }
}

impl<S: S14StarfoldK4ProductionStage> S14StarfoldK4ProductionAdapter<S> {
    pub fn new(
        context: Arc<VulkanContext>,
        cache_root: &Path,
        microtile_bytes: u32,
        catalog: FullDepthExpertCatalog,
        stage: S,
    ) -> Result<Self> {
        let runtime = S14StarfoldRuntime::new(Arc::clone(&context), cache_root, microtile_bytes)
            .context("构造 S14 StarFold K4 runtime")?;
        let b4_owner = match S14StarfoldB4RoutedLayerOwner::new(context) {
            Ok(owner) => owner,
            Err(error) => {
                runtime.destroy()?;
                return Err(error).context("构造 S14 StarFold K4 B4 owner");
            }
        };
        Ok(Self {
            runtime: Some(runtime),
            b4_owner: Some(b4_owner),
            stage: Some(stage),
            catalog: Arc::new(catalog),
            phase: AdapterPhase::Idle,
            validated_blocks: 0,
            last_finished_base_position: None,
            committed_rebinds: 0,
        })
    }

    /// 接管 `S14Runtime` 已创建的唯一 StarFold runtime/windows。production builder
    /// 必须使用本入口，禁止再次调用 `S14StarfoldRuntime::new`。
    pub(crate) fn from_owned_runtime(
        context: Arc<VulkanContext>,
        runtime: S14StarfoldRuntime,
        catalog: Arc<FullDepthExpertCatalog>,
        mut stage: S,
    ) -> Result<Self> {
        let b4_owner = match S14StarfoldB4RoutedLayerOwner::new(context) {
            Ok(owner) => owner,
            Err(error) => {
                let stage_cleanup = stage.destroy();
                let runtime_cleanup = runtime.destroy();
                return Err(anyhow!(
                    "构造 S14 StarFold K4 B4 owner: {error:#}; stage cleanup={stage_cleanup:?}; runtime cleanup={runtime_cleanup:?}"
                ));
            }
        };
        Ok(Self {
            runtime: Some(runtime),
            b4_owner: Some(b4_owner),
            stage: Some(stage),
            catalog,
            phase: AdapterPhase::Idle,
            validated_blocks: 0,
            last_finished_base_position: None,
            committed_rebinds: 0,
        })
    }

    pub fn one_mib(
        context: Arc<VulkanContext>,
        cache_root: &Path,
        catalog: FullDepthExpertCatalog,
        stage: S,
    ) -> Result<Self> {
        Self::new(
            context,
            cache_root,
            S14_STARFOLD_DEFAULT_MICROTILE_BYTES,
            catalog,
            stage,
        )
    }

    pub fn begin_block(
        &mut self,
        base_position: u32,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
    ) -> Result<()> {
        self.begin_block_with_source(
            base_position,
            input_token_ids,
            initial_hidden,
            MaterializedTokenSource::SpeculativeDraft,
        )
    }

    pub fn begin_block_with_source(
        &mut self,
        base_position: u32,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
        source: MaterializedTokenSource,
    ) -> Result<()> {
        if source != MaterializedTokenSource::SpeculativeDraft {
            bail!("生成 block 只接受 SpeculativeDraft；ForcedPrefill 必须走显式 prefill 入口");
        }
        self.begin_block_validated(base_position, input_token_ids, initial_hidden, source, None)
    }

    /// 纯 K4 prompt prefill 入口。base0 只在真实初始 DecoderState identity 闭合时开放；
    /// 后续 prefill block 也必须连续消费上一 teacher-forced committed state。
    pub fn begin_teacher_forced_prefill_block(
        &mut self,
        authoritative: &DecoderStateV1,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
    ) -> Result<()> {
        validate_teacher_forced_prefill_base(authoritative, input_token_ids)?;
        self.begin_block_validated(
            authoritative.position,
            input_token_ids,
            initial_hidden,
            MaterializedTokenSource::ForcedPrefill,
            None,
        )
    }

    fn begin_block_validated(
        &mut self,
        base_position: u32,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
        source: MaterializedTokenSource,
        position0_provenance: Option<S14Position0CommittedGenerationProvenance>,
    ) -> Result<()> {
        if !matches!(self.phase, AdapterPhase::Idle) {
            bail!("S14 StarFold K4 adapter 已有未释放 block");
        }
        let is_position0_draft =
            source == MaterializedTokenSource::SpeculativeDraft && base_position == 0;
        if input_token_ids.len() != K4
            || (is_position0_draft && position0_provenance.is_none())
            || (position0_provenance.is_some() && !is_position0_draft)
        {
            bail!("S14 StarFold source/base/K/origin 合同非法：普通 draft 要求 nonzero base；base0 draft 必须走 committed-origin 强入口");
        }
        validate_hidden(initial_hidden, K4)?;
        let begin_result = match (source, position0_provenance) {
            (MaterializedTokenSource::SpeculativeDraft, Some(provenance)) => self
                .stage_mut()?
                .begin_position0_committed_generation_k4_block(
                    provenance,
                    input_token_ids,
                    initial_hidden,
                ),
            (MaterializedTokenSource::SpeculativeDraft, None) => {
                self.stage_mut()?
                    .begin_k4_block(base_position, input_token_ids, initial_hidden)
            }
            (MaterializedTokenSource::ForcedPrefill, None) => {
                self.stage_mut()?.begin_teacher_forced_prefill_k4_block(
                    base_position,
                    input_token_ids,
                    initial_hidden,
                )
            }
            (MaterializedTokenSource::ForcedPrefill, Some(_)) => {
                unreachable!("validated source/provenance combination")
            }
        };
        let begin = match begin_result
            .map_err(anyhow::Error::msg)
            .context("启动 S14 StarFold K4 production stage")
        {
            Ok(begin) => begin,
            Err(error) => return self.fail_after_stage_begin(0, error),
        };
        if begin.base_position != base_position
            || begin.block_size != K4
            || begin.begin_calls != 1
            || begin.serial_token_forward_calls != 0
        {
            return self.fail_after_stage_begin(0, anyhow!("S14 StarFold K4 begin 强回执漂移"));
        }
        self.phase = AdapterPhase::Active(ActiveK4Block {
            base_position,
            input_token_ids: input_token_ids.to_vec(),
            current_hidden: initial_hidden,
            source,
            next_layer_index: 0,
            routes_by_position: (0..K4)
                .map(|_| Vec::with_capacity(FULL_DEPTH_LAYERS.len()))
                .collect(),
            layers: Vec::with_capacity(FULL_DEPTH_LAYERS.len()),
        });
        Ok(())
    }

    /// 执行下一层真实 production 路径：attention/route/checkpoint → route plan → StarFold
    /// B4 → routed reduction/HC-post。任何一步失败都会立即请求 stage drain/abort。
    pub fn step_layer(&mut self) -> Result<S14StarfoldK4LayerReceipt> {
        let (base_position, layer, input_token_ids, input_hidden, source, completed_layers) = {
            let active = self.active()?;
            let layer = *FULL_DEPTH_LAYERS
                .get(active.next_layer_index)
                .context("S14 StarFold K4 已完成43层")?;
            (
                active.base_position,
                layer,
                active.input_token_ids.clone(),
                active.current_hidden,
                active.source.clone(),
                active.next_layer_index,
            )
        };
        let input = S14StarfoldK4AttentionInput {
            base_position,
            layer,
            input_token_ids: &input_token_ids,
            input_hidden,
            source: &source,
        };
        let result = self.step_layer_inner(input);
        match result {
            Ok(receipt) => Ok(receipt),
            Err(error) => Err(self.abort_after_error(completed_layers, error)),
        }
    }

    fn step_layer_inner(
        &mut self,
        input: S14StarfoldK4AttentionInput<'_>,
    ) -> Result<S14StarfoldK4LayerReceipt> {
        let attention = self
            .stage_mut()?
            .produce_attention_route_and_checkpoint(input)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "生产 S14 StarFold L{} attention/route/checkpoint",
                    input.layer
                )
            })?;
        validate_attention_output(input, &attention)?;

        let layer_plan = self.runtime()?.plan_k4_k8_layer(
            self.catalog.as_ref(),
            u64::from(input.base_position),
            &attention.routes,
        )?;
        if layer_plan.layer != u16::from(input.layer)
            || layer_plan.base_position != u64::from(input.base_position)
            || layer_plan.block_size != K4
            || layer_plan.b4_blocks.len() != 1
            || layer_plan.commit_boundary
                != S14StarfoldCommitBoundary::ExistingLongestReliablePrefix
        {
            bail!("S14 StarFold K4 layer plan identity/commit boundary 漂移");
        }
        let b4_plan = layer_plan
            .b4_blocks
            .first()
            .context("S14 StarFold K4 layer plan 缺少唯一 B4 block")?
            .clone();
        let (expert, down) = {
            let runtime = self
                .runtime
                .as_mut()
                .context("S14 StarFold runtime 已销毁")?;
            let owner = self
                .b4_owner
                .as_mut()
                .context("S14 StarFold B4 owner 已销毁")?;
            owner.execute(runtime, &b4_plan, attention.moe_input_f32)?
        };
        validate_expert_receipt(input, &attention.routes, &expert, down)?;

        let hidden_commit = self
            .stage_mut()?
            .commit_routed_down_to_next_hidden(input, &attention.routes, down, &expert)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("提交 S14 StarFold L{} 下一层 hidden", input.layer))?;
        validate_hidden_commit(input, attention.post_attention_hidden, hidden_commit)?;

        let receipt = S14StarfoldK4LayerReceipt {
            layer: input.layer,
            base_position: input.base_position,
            routes: attention.routes,
            checkpoint: attention.checkpoint,
            expert,
            hidden_commit,
        };
        let active = self.active_mut()?;
        if active.next_layer_index >= FULL_DEPTH_LAYERS.len()
            || FULL_DEPTH_LAYERS[active.next_layer_index] != input.layer
            || active.current_hidden != input.input_hidden
        {
            bail!("S14 StarFold K4 active layer 在物理执行期间发生漂移");
        }
        for (position, route) in active
            .routes_by_position
            .iter_mut()
            .zip(receipt.routes.iter().cloned())
        {
            position.push(route);
        }
        active.current_hidden = receipt.hidden_commit.next_hidden;
        active.next_layer_index += 1;
        active.layers.push(receipt.clone());
        Ok(receipt)
    }

    pub fn seal_layers(&mut self) -> Result<S14StarfoldK4FullDepthReceipt> {
        let (base_position, completed_layers, final_hidden, source) = {
            let active = self.active()?;
            (
                active.base_position,
                active.next_layer_index,
                active.current_hidden,
                active.source,
            )
        };
        if completed_layers != FULL_DEPTH_LAYERS.len() {
            bail!("S14 StarFold K4 禁止 seal 不完整 FullDepth43");
        }
        if self.active()?.routes_by_position.iter().any(|routes| {
            routes.len() != FULL_DEPTH_LAYERS.len()
                || routes
                    .iter()
                    .zip(FULL_DEPTH_LAYERS)
                    .any(|(route, layer)| route.layer != layer)
        }) {
            return Err(self.abort_after_error(
                completed_layers,
                anyhow!("S14 StarFold K4 position-major route chain 不完整"),
            ));
        }
        let seal = match self
            .stage_mut()?
            .seal_checkpoint_chain(completed_layers, final_hidden)
            .map_err(anyhow::Error::msg)
            .context("seal S14 StarFold K4 checkpoint chain")
        {
            Ok(seal) => seal,
            Err(error) => return Err(self.abort_after_error(completed_layers, error)),
        };
        if seal.base_position != base_position
            || seal.block_size != K4
            || seal.completed_layers != FULL_DEPTH_LAYERS.len()
            || seal.checkpoint_count != K4
            || seal.prefix_program_seal_calls != 1
            || seal.checkpoint_commit_calls != 0
            || seal.serial_token_forward_calls != 0
        {
            return Err(self.abort_after_error(
                completed_layers,
                anyhow!("S14 StarFold K4 checkpoint seal 强回执漂移"),
            ));
        }

        // stage 已经 seal；所有仍可能失败的 host-side 聚合必须在 phase 切换前完成，
        // 并在失败时立即走同一个 stage-authoritative abort，不能把清理拖到 Drop。
        let host_receipt = (|| -> Result<(u32, u64, u32, [u32; STARFOLD_B4_LANES])> {
            let active = self.active()?;
            let (packed_uploads, packed_upload_bytes, lane_dispatches) =
                aggregate_physical_counts(&active.layers)?;
            let physical_input_token_ids = active
                .input_token_ids
                .as_slice()
                .try_into()
                .map_err(|_| anyhow!("S14 StarFold sealed K4 input token 数量漂移"))?;
            Ok((
                packed_uploads,
                packed_upload_bytes,
                lane_dispatches,
                physical_input_token_ids,
            ))
        })();
        let (packed_uploads, packed_upload_bytes, lane_dispatches, physical_input_token_ids) =
            match host_receipt {
                Ok(receipt) => receipt,
                Err(error) => return Err(self.abort_after_error(completed_layers, error)),
            };

        let active = match std::mem::replace(
            &mut self.phase,
            AdapterPhase::LayersSealed { base_position },
        ) {
            AdapterPhase::Active(active) => active,
            other => {
                self.phase = other;
                return Err(self.abort_after_error(
                    completed_layers,
                    anyhow!("S14 StarFold K4 seal phase 漂移"),
                ));
            }
        };
        Ok(S14StarfoldK4FullDepthReceipt {
            base_position,
            block_size: K4,
            physical_input_token_ids,
            completed_layers,
            routes_by_position: active.routes_by_position,
            layers: active.layers,
            final_hidden,
            checkpoint_seal: seal,
            source,
            packed_uploads,
            packed_upload_bytes,
            lane_dispatches,
            serial_token_forward_calls: 0,
            terminal_ready: true,
            token_committed: false,
        })
    }

    /// 便利入口；循环的是 43 层网络深度，不是四次 token forward。
    pub fn execute_full_depth(
        &mut self,
        base_position: u32,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
    ) -> Result<S14StarfoldK4FullDepthReceipt> {
        self.execute_full_depth_with_source(
            base_position,
            input_token_ids,
            initial_hidden,
            MaterializedTokenSource::SpeculativeDraft,
        )
    }

    pub fn execute_full_depth_with_source(
        &mut self,
        base_position: u32,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
        source: MaterializedTokenSource,
    ) -> Result<S14StarfoldK4FullDepthReceipt> {
        self.begin_block_with_source(base_position, input_token_ids, initial_hidden, source)?;
        while self.active()?.next_layer_index < FULL_DEPTH_LAYERS.len() {
            self.step_layer()?;
        }
        self.seal_layers()
    }

    pub fn execute_teacher_forced_prefill_full_depth(
        &mut self,
        authoritative: &DecoderStateV1,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
    ) -> Result<S14StarfoldK4FullDepthReceipt> {
        self.begin_teacher_forced_prefill_block(authoritative, input_token_ids, initial_hidden)?;
        while self.active()?.next_layer_index < FULL_DEPTH_LAYERS.len() {
            self.step_layer()?;
        }
        self.seal_layers()
    }

    /// 单-token prompt 的显式 generation base0 入口。普通 SpeculativeDraft API 仍拒绝
    /// base0；这里只接受 StarWave 已绑定真实 host/device committed-origin 的 proposal。
    pub fn execute_generation_from_position0(
        &mut self,
        authoritative: &DecoderStateV1,
        proposal: &S14StarwaveDraftProposal,
        initial_hidden: S14CausalBlockHiddenBinding,
    ) -> Result<S14StarfoldK4FullDepthReceipt> {
        proposal
            .validate_for(authoritative)
            .map_err(|error| anyhow!(error.to_string()))
            .context("校验 StarWave position0 committed-origin proposal")?;
        if authoritative.position != 0
            || proposal.position0_committed_origin().is_none()
            || proposal.authoritative_position() != 0
        {
            bail!("S14 StarFold generation base0 缺少 StarWave committed-origin 强证明");
        }
        let provenance = S14Position0CommittedGenerationProvenance::validate(
            authoritative,
            proposal.position0_committed_origin(),
        )
        .map_err(anyhow::Error::msg)
        .context("构造 StarFold position0 committed-generation typed provenance")?;
        self.begin_block_validated(
            0,
            proposal.input_token_ids(),
            initial_hidden,
            MaterializedTokenSource::SpeculativeDraft,
            Some(provenance),
        )?;
        while self.active()?.next_layer_index < FULL_DEPTH_LAYERS.len() {
            self.step_layer()?;
        }
        self.seal_layers()
    }

    /// terminal/head/最长前缀 consumer 已验收并接管 sealed future 后释放 block 闩锁。
    pub fn finish_validated_block(&mut self) -> Result<()> {
        let base_position = match self.phase {
            AdapterPhase::LayersSealed { base_position } => base_position,
            _ => bail!("S14 StarFold K4 当前没有可释放的 sealed block"),
        };
        self.stage_mut()?
            .finish_validated_block()
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!("释放 S14 StarFold K4 sealed block position={base_position}")
            })?;
        self.validated_blocks = self
            .validated_blocks
            .checked_add(1)
            .context("S14 StarFold validated block counter overflow")?;
        self.last_finished_base_position = Some(base_position);
        self.phase = AdapterPhase::Idle;
        Ok(())
    }

    pub(crate) fn validated_blocks(&self) -> u64 {
        self.validated_blocks
    }

    pub(crate) fn last_finished_base_position(&self) -> Option<u32> {
        self.last_finished_base_position
    }

    pub(crate) fn committed_rebinds(&self) -> u64 {
        self.committed_rebinds
    }

    pub(crate) fn owner_counts(&self) -> S14StarfoldK4AdapterOwnerCounts {
        let runtime_owners = usize::from(self.runtime.is_some());
        let microtile_window_owners = self
            .runtime
            .as_ref()
            .filter(|runtime| runtime.physical_allocation_bytes() != 0)
            .map_or(0, |runtime| runtime.contract().window_count);
        S14StarfoldK4AdapterOwnerCounts {
            runtime_owners,
            microtile_window_owners,
            b4_owners: usize::from(self.b4_owner.is_some()),
            stage_owners: usize::from(self.stage.is_some()),
            catalog_owners: usize::from(self.runtime.is_some()),
        }
    }

    /// 在一个请求的最后一个 block 已完成原子提交并回到 `Idle` 后，拆除请求级
    /// stage/B4 owner，但把唯一 StarFold 双窗口 runtime 原样归还给 resident root。
    /// 这不是 teardown：microtile windows、proof store 与 transfer executor 不会重建。
    pub(crate) fn into_resident_runtime(mut self) -> Result<S14StarfoldRuntime> {
        if !matches!(self.phase, AdapterPhase::Idle) {
            bail!("只有 Idle 的 S14 StarFold adapter 才能归还 resident runtime");
        }
        if let Some(owner) = self.b4_owner.as_mut() {
            owner
                .try_destroy()
                .context("归还 resident runtime 前销毁请求级 B4 owner")?;
        }
        self.b4_owner.take();
        if let Some(stage) = self.stage.as_mut() {
            stage
                .destroy()
                .map_err(anyhow::Error::msg)
                .context("归还 resident runtime 前销毁请求级 production stage")?;
        }
        self.stage.take();
        let runtime = self
            .runtime
            .take()
            .context("归还 resident runtime 时唯一 StarFold runtime 已缺失")?;
        self.phase = AdapterPhase::Destroyed;
        Ok(runtime)
    }

    pub fn destroy(&mut self) -> Result<()> {
        if matches!(self.phase, AdapterPhase::Destroyed) {
            return Ok(());
        }
        let completed = match self.phase {
            AdapterPhase::Active(ref active) => Some(active.next_layer_index),
            AdapterPhase::Poisoned { completed_layers } => Some(completed_layers),
            AdapterPhase::LayersSealed { .. } => Some(FULL_DEPTH_LAYERS.len()),
            AdapterPhase::Idle | AdapterPhase::Destroyed => None,
        };
        if completed.is_some() {
            let _ = self.stage_mut()?.drain_and_abort_k4_block();
        }
        if let Some(owner) = self.b4_owner.as_mut() {
            owner.try_destroy()?;
        }
        self.b4_owner.take();
        if let Some(runtime) = self.runtime.take() {
            runtime.destroy()?;
        }
        if let Some(stage) = self.stage.as_mut() {
            stage.destroy().map_err(anyhow::Error::msg)?;
        }
        self.stage.take();
        self.phase = AdapterPhase::Destroyed;
        Ok(())
    }

    fn active(&self) -> Result<&ActiveK4Block> {
        match &self.phase {
            AdapterPhase::Active(active) => Ok(active),
            AdapterPhase::Poisoned { .. } => bail!("S14 StarFold K4 adapter 已 poisoned"),
            _ => bail!("S14 StarFold K4 adapter 当前没有 active block"),
        }
    }

    fn active_mut(&mut self) -> Result<&mut ActiveK4Block> {
        match &mut self.phase {
            AdapterPhase::Active(active) => Ok(active),
            AdapterPhase::Poisoned { .. } => bail!("S14 StarFold K4 adapter 已 poisoned"),
            _ => bail!("S14 StarFold K4 adapter 当前没有 active block"),
        }
    }

    fn runtime(&self) -> Result<&S14StarfoldRuntime> {
        self.runtime.as_ref().context("S14 StarFold runtime 已销毁")
    }

    fn stage_mut(&mut self) -> Result<&mut S> {
        self.stage
            .as_mut()
            .context("S14 StarFold K4 production stage 已销毁")
    }

    fn fail_after_stage_begin<T>(
        &mut self,
        completed_layers: usize,
        error: anyhow::Error,
    ) -> Result<T> {
        Err(self.abort_after_error(completed_layers, error))
    }

    fn abort_after_error(
        &mut self,
        completed_layers: usize,
        error: anyhow::Error,
    ) -> anyhow::Error {
        let abort = self
            .stage
            .as_mut()
            .context("S14 StarFold K4 production stage 已销毁")
            .and_then(|stage| stage.drain_and_abort_k4_block().map_err(anyhow::Error::msg));
        let latest_allowed_completed_layers = completed_layers
            .checked_add(1)
            .unwrap_or(usize::MAX)
            .min(FULL_DEPTH_LAYERS.len());
        match abort {
            Ok(receipt)
                if (completed_layers..=latest_allowed_completed_layers)
                    .contains(&receipt.completed_layers)
                    && receipt.drain_calls == 1
                    && receipt.aborted =>
            {
                self.phase = AdapterPhase::Idle;
                anyhow!(
                    "{error:#}; S14 StarFold K4 已按 stage 权威层数 drain/abort, outer_before={} stage_completed={}",
                    completed_layers,
                    receipt.completed_layers
                )
            }
            Ok(receipt) => {
                self.phase = AdapterPhase::Poisoned { completed_layers };
                anyhow!("{error:#}; S14 StarFold K4 abort 回执漂移: {receipt:?}")
            }
            Err(abort_error) => {
                self.phase = AdapterPhase::Poisoned { completed_layers };
                anyhow!("{error:#}; S14 StarFold K4 drain/abort 失败: {abort_error:#}")
            }
        }
    }
}

impl<P>
    S14StarfoldK4ProductionAdapter<
        S14StarfoldConcreteK4Stage<
            S14CausalBlockProductionHcQkvAdapter<S14CausalBlockProductionHcQkvLayerRecorder<P>>,
        >,
    >
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    pub(crate) fn starfold_teacher_forced_prefill_prefix_product(
        &self,
    ) -> Result<S14CausalBlockStarfoldPrefillPrefixProduct> {
        if !matches!(self.phase, AdapterPhase::LayersSealed { .. }) {
            bail!("StarFold ForcedPrefill prefix 要求 FullDepth43 adapter 已 seal");
        }
        self.stage
            .as_ref()
            .context("S14 StarFold production stage 已销毁")?
            .starfold_teacher_forced_prefill_prefix_product()
    }

    pub(crate) fn starfold_terminal_block_owners(
        &self,
        final_hidden: S14CausalBlockHiddenBinding,
    ) -> Result<S14CausalBlockStarfoldTerminalBlockOwners> {
        if !matches!(self.phase, AdapterPhase::LayersSealed { .. }) {
            bail!("StarFold terminal owner 要求 FullDepth43 adapter 已 seal");
        }
        self.stage
            .as_ref()
            .context("S14 StarFold production stage 已销毁")?
            .starfold_terminal_block_owners(final_hidden)
    }

    pub(crate) fn rebind_committed_block_state(
        &mut self,
        base_position: u32,
        provider: P,
        hidden_banks: [S14CausalBlockHiddenBank; 2],
        mut prefix_producer: S14CausalBlockPrefixStateProducer,
    ) -> Result<()> {
        let next_rebind = match self.committed_rebinds.checked_add(1) {
            Some(value) => value,
            None => {
                let cleanup = prefix_producer.destroy();
                bail!("S14 StarFold committed rebind counter overflow; cleanup={cleanup:?}");
            }
        };
        if !matches!(self.phase, AdapterPhase::Idle)
            || self.validated_blocks == 0
            || self.validated_blocks != next_rebind
            || self.runtime.is_none()
            || self.b4_owner.is_none()
            || self.stage.is_none()
        {
            let cleanup = prefix_producer.destroy();
            bail!(
                "S14 StarFold adapter 未完成上一 block 或固定 execution owners 缺失; cleanup={cleanup:?}"
            );
        }
        let result = self
            .stage
            .as_mut()
            .context("S14 StarFold production stage 已销毁")?
            .rebind_committed_block_state(base_position, provider, hidden_banks, prefix_producer);
        if let Err(error) = result {
            self.phase = AdapterPhase::Poisoned {
                completed_layers: 0,
            };
            return Err(error.context("S14 StarFold committed-state rebind 失败并 poisoned"));
        }
        self.committed_rebinds = next_rebind;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct S14StarfoldConcretePendingLayer {
    layer: u8,
    base_position: u32,
    post_attention_hidden: S14CausalBlockHiddenBinding,
    hc: S14StarfoldHcPrepareReceipt,
}

/// 现有 production HC/QKV/prefix owner 与 StarFold expert/HC bridge 之间的真实 stage。
/// 它不持有 terminal/head；43 层 seal 后由原有 checkpoint/terminal owner 接管。
pub struct S14StarfoldConcreteK4Stage<H: S14CausalBlockVulkanHcQkvAdapter> {
    hc_qkv: H,
    shared_reduce: S14StarfoldSharedReduceOwner,
    hc_bridge: S14StarfoldHcBridgeOwner,
    position0_generation_provenance: Option<S14Position0CommittedGenerationProvenance>,
    base_position: Option<u32>,
    completed_layers: usize,
    pending: Option<S14StarfoldConcretePendingLayer>,
    destroyed: bool,
}

impl<H: S14CausalBlockVulkanHcQkvAdapter> fmt::Debug for S14StarfoldConcreteK4Stage<H> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14StarfoldConcreteK4Stage")
            .field("hc_qkv", &self.hc_qkv)
            .field(
                "position0_generation_provenance",
                &self.position0_generation_provenance,
            )
            .field("base_position", &self.base_position)
            .field("completed_layers", &self.completed_layers)
            .field("pending", &self.pending)
            .field("destroyed", &self.destroyed)
            .finish_non_exhaustive()
    }
}

impl<H: S14CausalBlockVulkanHcQkvAdapter> S14StarfoldConcreteK4Stage<H> {
    pub fn new(
        context: Arc<VulkanContext>,
        static_arena: Arc<S14Position0PagedWeightArena>,
        hidden_banks: [S14CausalBlockHiddenBank; 2],
        mut hc_qkv: H,
        position0_generation_provenance: Option<S14Position0CommittedGenerationProvenance>,
    ) -> Result<Self> {
        let shared_reduce = match S14StarfoldSharedReduceOwner::new(
            Arc::clone(&context),
            Arc::clone(&static_arena),
        ) {
            Ok(owner) => owner,
            Err(error) => {
                let cleanup = hc_qkv.destroy();
                return Err(anyhow!(
                    "构造 S14 StarFold shared reduce owner: {error:#}; HC/QKV cleanup={cleanup:?}"
                ));
            }
        };
        let hc_bridge = match S14StarfoldHcBridgeOwner::new(context, static_arena, hidden_banks) {
            Ok(owner) => owner,
            Err(error) => {
                let mut shared_reduce = shared_reduce;
                let reduce_cleanup = shared_reduce.try_destroy();
                let hc_cleanup = hc_qkv.destroy();
                return Err(anyhow!(
                    "构造 S14 StarFold HC bridge owner: {error:#}; reduce cleanup={reduce_cleanup:?}; HC/QKV cleanup={hc_cleanup:?}"
                ));
            }
        };
        Ok(Self {
            hc_qkv,
            shared_reduce,
            hc_bridge,
            position0_generation_provenance,
            base_position: None,
            completed_layers: 0,
            pending: None,
            destroyed: false,
        })
    }

    fn ensure_active(&self, input: S14StarfoldK4AttentionInput<'_>) -> Result<()> {
        if self.destroyed
            || self.base_position != Some(input.base_position)
            || self.completed_layers >= FULL_DEPTH_LAYERS.len()
            || FULL_DEPTH_LAYERS[self.completed_layers] != input.layer
        {
            bail!("S14 StarFold concrete stage active block/layer 漂移");
        }
        Ok(())
    }
}

impl<P>
    S14StarfoldConcreteK4Stage<
        S14CausalBlockProductionHcQkvAdapter<S14CausalBlockProductionHcQkvLayerRecorder<P>>,
    >
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    fn starfold_teacher_forced_prefill_prefix_product(
        &self,
    ) -> Result<S14CausalBlockStarfoldPrefillPrefixProduct> {
        if self.destroyed
            || self.base_position.is_none()
            || self.completed_layers != FULL_DEPTH_LAYERS.len()
            || self.pending.is_some()
        {
            bail!("StarFold concrete stage 尚未闭合 ForcedPrefill FullDepth43 boundary");
        }
        self.hc_qkv
            .starfold_teacher_forced_prefill_prefix_product()
            .map_err(anyhow::Error::msg)
            .context("从同源 HC/QKV recorder 获取 StarFold ForcedPrefill prefix")
    }

    fn starfold_terminal_block_owners(
        &self,
        final_hidden: S14CausalBlockHiddenBinding,
    ) -> Result<S14CausalBlockStarfoldTerminalBlockOwners> {
        if self.destroyed
            || self.base_position.is_none()
            || self.completed_layers != FULL_DEPTH_LAYERS.len()
            || self.pending.is_some()
        {
            bail!("StarFold concrete stage 尚未闭合 FullDepth43 terminal boundary");
        }
        self.hc_qkv
            .starfold_terminal_block_owners(final_hidden)
            .map_err(anyhow::Error::msg)
            .context("从同源 HC/QKV recorder 获取 StarFold terminal owner")
    }

    pub(crate) fn rebind_committed_block_state(
        &mut self,
        base_position: u32,
        provider: P,
        hidden_banks: [S14CausalBlockHiddenBank; 2],
        mut prefix_producer: S14CausalBlockPrefixStateProducer,
    ) -> Result<()> {
        let preflight = (|| -> Result<()> {
            if self.destroyed
                || self.base_position.is_some()
                || self.completed_layers != 0
                || self.pending.is_some()
                || self.position0_generation_provenance.is_some()
            {
                bail!("S14 StarFold concrete stage 未 finish/drain，禁止 committed-state rebind");
            }
            self.shared_reduce
                .wait()
                .context("S14 StarFold rebind 前等待 shared-reduce drain")?;
            self.hc_bridge.validate_hidden_bank_rebind(&hidden_banks)?;
            Ok(())
        })();
        if let Err(error) = preflight {
            let cleanup = prefix_producer.destroy();
            return Err(anyhow!("{error:#}; 新 prefix producer cleanup={cleanup:?}"));
        }
        let recorder_banks = hidden_banks.clone();
        self.hc_qkv
            .rebind_idle_recorder(|recorder| {
                recorder
                    .rebind_idle_block_state(
                        base_position,
                        provider,
                        recorder_banks,
                        prefix_producer,
                    )
                    .map_err(|error| format!("{error:#}"))
            })
            .map_err(anyhow::Error::msg)
            .context("原地 rebind S14 StarFold HC/QKV recorder block state")?;
        self.hc_bridge
            .rebind_hidden_banks(hidden_banks)
            .context("原地 rebind S14 StarFold HC bridge hidden owners")?;
        Ok(())
    }
}

impl<H: S14CausalBlockVulkanHcQkvAdapter> S14StarfoldK4ProductionStage
    for S14StarfoldConcreteK4Stage<H>
{
    fn begin_k4_block(
        &mut self,
        base_position: u32,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
    ) -> std::result::Result<S14StarfoldK4BeginReceipt, String> {
        if self.destroyed
            || self.base_position.is_some()
            || self.pending.is_some()
            || input_token_ids.len() != K4
            || base_position == 0
            || self.position0_generation_provenance.is_some()
        {
            return Err(
                "S14 StarFold concrete stage 普通 draft begin 禁止 base0/provenance 漂移".into(),
            );
        }
        validate_hidden(initial_hidden, K4).map_err(|error| format!("{error:#}"))?;
        self.hc_qkv.begin_block(base_position, K4)?;
        self.base_position = Some(base_position);
        self.completed_layers = 0;
        Ok(S14StarfoldK4BeginReceipt {
            base_position,
            block_size: K4,
            begin_calls: 1,
            serial_token_forward_calls: 0,
        })
    }

    fn begin_position0_committed_generation_k4_block(
        &mut self,
        provenance: S14Position0CommittedGenerationProvenance,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
    ) -> std::result::Result<S14StarfoldK4BeginReceipt, String> {
        if self.destroyed
            || self.base_position.is_some()
            || self.pending.is_some()
            || input_token_ids.len() != K4
            || self.position0_generation_provenance != Some(provenance)
        {
            return Err(
                "S14 StarFold position0 committed-generation stage provenance/phase/K 漂移".into(),
            );
        }
        validate_hidden(initial_hidden, K4).map_err(|error| format!("{error:#}"))?;
        self.hc_qkv
            .begin_position0_committed_generation_block(provenance, K4)?;
        self.position0_generation_provenance = None;
        self.base_position = Some(0);
        self.completed_layers = 0;
        Ok(S14StarfoldK4BeginReceipt {
            base_position: 0,
            block_size: K4,
            begin_calls: 1,
            serial_token_forward_calls: 0,
        })
    }

    fn begin_teacher_forced_prefill_k4_block(
        &mut self,
        base_position: u32,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
    ) -> std::result::Result<S14StarfoldK4BeginReceipt, String> {
        if self.destroyed
            || self.base_position.is_some()
            || self.pending.is_some()
            || input_token_ids.len() != K4
            || self.position0_generation_provenance.is_some()
        {
            return Err("S14 StarFold ForcedPrefill stage begin phase/K 漂移".into());
        }
        validate_hidden(initial_hidden, K4).map_err(|error| format!("{error:#}"))?;
        self.hc_qkv
            .begin_teacher_forced_prefill_block(base_position, K4)?;
        self.base_position = Some(base_position);
        self.completed_layers = 0;
        Ok(S14StarfoldK4BeginReceipt {
            base_position,
            block_size: K4,
            begin_calls: 1,
            serial_token_forward_calls: 0,
        })
    }

    fn produce_attention_route_and_checkpoint(
        &mut self,
        input: S14StarfoldK4AttentionInput<'_>,
    ) -> std::result::Result<S14StarfoldK4AttentionCheckpointOutput, String> {
        self.ensure_active(input)
            .map_err(|error| format!("{error:#}"))?;
        if self.pending.is_some() {
            return Err("S14 StarFold concrete stage 上一层尚未归约".into());
        }
        let layer_input = S14CausalBlockLayerInput {
            base_position: input.base_position,
            layer: input.layer,
            input_token_ids: input.input_token_ids,
            input_hidden: input.input_hidden,
            source: input.source.clone(),
        };
        let recorded = self
            .hc_qkv
            .run_k_lane_hc_qkv_attention_router(&layer_input)?;
        recorded.receipt.validate(&layer_input, &recorded.output)?;
        let hc = self
            .hc_bridge
            .prepare_layer(
                input.layer,
                input.base_position,
                recorded.output.post_attention_hidden,
            )
            .map_err(|error| format!("{error:#}"))?;
        let pending = S14StarfoldConcretePendingLayer {
            layer: input.layer,
            base_position: input.base_position,
            post_attention_hidden: recorded.output.post_attention_hidden,
            hc,
        };
        self.pending = Some(pending);
        Ok(S14StarfoldK4AttentionCheckpointOutput {
            routes: recorded.output.routes,
            post_attention_hidden: recorded.output.post_attention_hidden,
            moe_input_f32: hc.moe_input_f32,
            checkpoint: recorded.receipt,
            attention_router_submit_calls: 1,
            checkpoint_recording_calls: 1,
            serial_token_forward_calls: 0,
        })
    }

    fn commit_routed_down_to_next_hidden(
        &mut self,
        input: S14StarfoldK4AttentionInput<'_>,
        routes: &[RouteDecision],
        down: S14StarfoldRoutedDownBinding,
        expert: &S14StarfoldB4RoutedLayerReceipt,
    ) -> std::result::Result<S14StarfoldK4HiddenCommitReceipt, String> {
        self.ensure_active(input)
            .map_err(|error| format!("{error:#}"))?;
        let pending = self
            .pending
            .ok_or_else(|| "S14 StarFold concrete stage 缺少 HC-pre output".to_owned())?;
        if pending.layer != input.layer
            || pending.base_position != input.base_position
            || routes.len() != K4
            || expert.layer != u16::from(input.layer)
        {
            return Err("S14 StarFold concrete stage reduce 输入 identity 漂移".into());
        }
        let shared = self
            .shared_reduce
            .submit_after_routed_w2(
                expert,
                pending.hc.moe_input_f32,
                down,
                pending.hc.reduced_output_f32,
            )
            .map_err(|error| format!("{error:#}"))?;
        self.shared_reduce
            .wait()
            .map_err(|error| format!("{error:#}"))?;
        let finalized = self
            .hc_bridge
            .finalize_layer(
                input.layer,
                input.base_position,
                pending.hc.reduced_output_f32,
            )
            .map_err(|error| format!("{error:#}"))?;
        let grouped = S14CausalBlockGroupedMoeOutput {
            output_hidden: finalized.next_hidden,
            grouped_submit_calls: 1,
            serial_token_forward_calls: 0,
            unique_experts: expert.unique_experts as usize,
        };
        self.hc_qkv
            .capture_grouped_moe_output(pending.post_attention_hidden, &grouped)?;
        self.pending = None;
        self.completed_layers = self
            .completed_layers
            .checked_add(1)
            .ok_or_else(|| "S14 StarFold concrete stage layer counter overflow".to_owned())?;
        Ok(S14StarfoldK4HiddenCommitReceipt {
            base_position: input.base_position,
            layer: input.layer,
            next_hidden: finalized.next_hidden,
            routed_reduce_dispatch_calls: shared.exact_reduce_dispatch_calls,
            hc_post_dispatch_calls: finalized.hc_post_dispatch_calls,
            queue_submit_calls: shared.queue_submit_calls + finalized.queue_submit_calls,
            serial_token_forward_calls: 0,
        })
    }

    fn seal_checkpoint_chain(
        &mut self,
        completed_layers: usize,
        final_hidden: S14CausalBlockHiddenBinding,
    ) -> std::result::Result<S14StarfoldK4CheckpointSealReceipt, String> {
        let base_position = self
            .base_position
            .ok_or_else(|| "S14 StarFold concrete stage 没有 active block".to_owned())?;
        if completed_layers != FULL_DEPTH_LAYERS.len()
            || self.completed_layers != completed_layers
            || self.pending.is_some()
            || final_hidden.block_size != K4
        {
            return Err("S14 StarFold concrete stage 禁止 seal 不完整 FullDepth43".into());
        }
        self.shared_reduce
            .wait()
            .map_err(|error| format!("{error:#}"))?;
        self.hc_qkv.seal_and_drain(completed_layers)?;
        Ok(S14StarfoldK4CheckpointSealReceipt {
            base_position,
            block_size: K4,
            completed_layers,
            checkpoint_count: K4,
            prefix_program_seal_calls: 1,
            checkpoint_commit_calls: 0,
            serial_token_forward_calls: 0,
        })
    }

    fn finish_validated_block(&mut self) -> std::result::Result<(), String> {
        if self.base_position.is_none()
            || self.completed_layers != FULL_DEPTH_LAYERS.len()
            || self.pending.is_some()
        {
            return Err("S14 StarFold concrete stage 没有 sealed block".into());
        }
        self.hc_qkv.finish_validated_block()?;
        self.base_position = None;
        self.completed_layers = 0;
        Ok(())
    }

    fn drain_and_abort_k4_block(
        &mut self,
    ) -> std::result::Result<S14StarfoldK4AbortReceipt, String> {
        let completed_layers = self.completed_layers;
        self.hc_bridge.abort_prepared_layer();
        self.pending = None;
        let shared = self
            .shared_reduce
            .wait()
            .map_err(|error| format!("{error:#}"));
        let hc = self.hc_qkv.drain_and_abort(completed_layers);
        shared?;
        hc?;
        self.base_position = None;
        self.completed_layers = 0;
        Ok(S14StarfoldK4AbortReceipt {
            completed_layers,
            drain_calls: 1,
            aborted: true,
        })
    }

    fn destroy(&mut self) -> std::result::Result<(), String> {
        if self.destroyed {
            return Ok(());
        }
        if self.base_position.is_some() {
            let _ = self.drain_and_abort_k4_block();
        }
        self.shared_reduce
            .wait()
            .map_err(|error| format!("{error:#}"))?;
        self.shared_reduce
            .try_destroy()
            .map_err(|error| format!("{error:#}"))?;
        self.hc_bridge
            .destroy()
            .map_err(|error| format!("{error:#}"))?;
        self.hc_qkv.destroy()?;
        self.destroyed = true;
        Ok(())
    }
}

impl<S: S14StarfoldK4ProductionStage> Drop for S14StarfoldK4ProductionAdapter<S> {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

fn validate_attention_output(
    input: S14StarfoldK4AttentionInput<'_>,
    output: &S14StarfoldK4AttentionCheckpointOutput,
) -> Result<()> {
    if output.attention_router_submit_calls != 1
        || output.checkpoint_recording_calls != 1
        || output.serial_token_forward_calls != 0
        || output.routes.len() != K4
    {
        bail!("S14 StarFold K4 attention/router/checkpoint 调用计数漂移");
    }
    for route in &output.routes {
        route
            .validate_for(GraphProfile::FullDepth43NativeTop6)
            .map_err(anyhow::Error::new)
            .context("S14 StarFold K4 在线 top-6 route 非法")?;
        if route.layer != input.layer {
            bail!("S14 StarFold K4 route layer 漂移");
        }
    }
    validate_moe_input(output.moe_input_f32)?;
    validate_checkpoint_receipt(input, output, output.checkpoint)
}

fn validate_checkpoint_receipt(
    input: S14StarfoldK4AttentionInput<'_>,
    output: &S14StarfoldK4AttentionCheckpointOutput,
    receipt: S14CausalBlockHcQkvLayerRecordingReceipt,
) -> Result<()> {
    let layer_input = S14CausalBlockLayerInput {
        base_position: input.base_position,
        layer: input.layer,
        input_token_ids: input.input_token_ids,
        input_hidden: input.input_hidden,
        source: input.source.clone(),
    };
    let attention = S14CausalBlockAttentionRouterOutput {
        post_attention_hidden: output.post_attention_hidden,
        routes: output.routes.clone(),
        forward_calls: 1,
    };
    receipt
        .validate(&layer_input, &attention)
        .map_err(anyhow::Error::msg)
        .context("S14 StarFold K4 HC/QKV+prefix layer 回执不完整")
}

fn validate_expert_receipt(
    input: S14StarfoldK4AttentionInput<'_>,
    routes: &[RouteDecision],
    receipt: &S14StarfoldB4RoutedLayerReceipt,
    down: S14StarfoldRoutedDownBinding,
) -> Result<()> {
    let unique_experts = routes
        .iter()
        .flat_map(|route| route.expert_ids.iter().copied())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if receipt.layer != u16::from(input.layer)
        || receipt.base_position != u64::from(input.base_position)
        || receipt.unique_experts as usize != unique_experts
        || receipt.packed_uploads == 0
        || receipt.packed_upload_bytes == 0
        || receipt.lane_dispatches == 0
        || receipt.serial_token_forward_calls != 0
        || down.positions as usize != K4
        || down.branches_per_position as usize != STARFOLD_TOP_K
    {
        bail!("S14 StarFold K4 B4 physical receipt/route coverage 漂移");
    }
    validate_down_binding(down)
}

fn validate_hidden_commit(
    input: S14StarfoldK4AttentionInput<'_>,
    post_attention_hidden: S14CausalBlockHiddenBinding,
    receipt: S14StarfoldK4HiddenCommitReceipt,
) -> Result<()> {
    validate_hidden(post_attention_hidden, K4)?;
    validate_hidden(receipt.next_hidden, K4)?;
    let expected_post_attention_generation = input
        .input_hidden
        .generation
        .checked_add(1)
        .context("S14 StarFold K4 post-attention hidden generation overflow")?;
    let expected_generation = input
        .input_hidden
        .generation
        .checked_add(2)
        .context("S14 StarFold K4 hidden generation overflow")?;
    let next_reuses_input_bank = receipt.next_hidden.buffer == input.input_hidden.buffer
        && receipt.next_hidden.offset == input.input_hidden.offset
        && receipt.next_hidden.bytes == input.input_hidden.bytes
        && receipt.next_hidden.block_size == input.input_hidden.block_size;
    let next_end = receipt
        .next_hidden
        .offset
        .checked_add(receipt.next_hidden.bytes)
        .context("S14 StarFold K4 next hidden range overflow")?;
    let post_attention_end = post_attention_hidden
        .offset
        .checked_add(post_attention_hidden.bytes)
        .context("S14 StarFold K4 post-attention hidden range overflow")?;
    let next_overlaps_post_attention = receipt.next_hidden.buffer == post_attention_hidden.buffer
        && receipt.next_hidden.offset < post_attention_end
        && post_attention_hidden.offset < next_end;
    if receipt.base_position != input.base_position
        || receipt.layer != input.layer
        || post_attention_hidden.generation != expected_post_attention_generation
        || receipt.next_hidden.generation != expected_generation
        || receipt.routed_reduce_dispatch_calls != 1
        || receipt.hc_post_dispatch_calls != K4 as u32
        || receipt.queue_submit_calls != 2
        || receipt.serial_token_forward_calls != 0
        // 双 bank 的真实物理链是 A(g) -> B(g+1) -> A(g+2)。下一层必须回到输入
        // bank，且不得覆盖仍代表 attention 输出的 opposite bank。
        || !next_reuses_input_bank
        || next_overlaps_post_attention
    {
        bail!("S14 StarFold K4 下一层 hidden commit 强回执漂移");
    }
    Ok(())
}

fn validate_teacher_forced_prefill_base(
    authoritative: &DecoderStateV1,
    input_token_ids: &[u32],
) -> Result<()> {
    authoritative
        .validate()
        .map_err(|error| anyhow!("ForcedPrefill authoritative DecoderState 非法: {error}"))?;
    if input_token_ids.len() != K4
        || input_token_ids[0] != authoritative.input_token_id
        || input_token_ids.iter().any(|&token| token >= VOCAB_SIZE)
        || authoritative.native.position != authoritative.position
        || authoritative.commit_epoch != u64::from(authoritative.position)
        || usize::from(authoritative.active_fixed_bank) != (authoritative.position as usize & 1)
    {
        bail!("ForcedPrefill K4 输入没有连续消费真实 authoritative state");
    }
    if authoritative.position == 0
        && (authoritative.commit_epoch != 0
            || authoritative.active_fixed_bank != 0
            || !authoritative.committed_tokens.is_empty())
    {
        bail!("ForcedPrefill base0 只接受 position0/epoch0/bank0 空 ledger");
    }
    Ok(())
}

fn validate_hidden(binding: S14CausalBlockHiddenBinding, block_size: usize) -> Result<()> {
    let expected = (block_size as u64)
        .checked_mul(S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64)
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .context("S14 StarFold K4 hidden bytes overflow")?;
    if binding.buffer == vk::Buffer::null()
        || binding.offset % 4 != 0
        || binding.bytes != expected
        || binding.block_size != block_size
    {
        bail!("S14 StarFold K4 hidden 不是精确 device [4,4,4096] BF16");
    }
    Ok(())
}

fn validate_moe_input(binding: S14StarfoldMxfp4ExternalSlice) -> Result<()> {
    let expected = K4 as u64 * HIDDEN * F32_BYTES;
    validate_external_slice(binding, expected, "MoE input [4,4096] F32")
}

fn validate_down_binding(binding: S14StarfoldRoutedDownBinding) -> Result<()> {
    let expected = (K4 * STARFOLD_TOP_K) as u64 * HIDDEN * F32_BYTES;
    validate_external_slice(binding.branches, expected, "routed down [4,top6,4096] F32")
}

fn validate_external_slice(
    binding: S14StarfoldMxfp4ExternalSlice,
    expected_bytes: u64,
    label: &str,
) -> Result<()> {
    if binding.buffer == vk::Buffer::null()
        || binding.offset % 4 != 0
        || binding.logical_bytes != expected_bytes
        || binding
            .offset
            .checked_add(binding.logical_bytes)
            .is_none_or(|end| end > binding.capacity_bytes)
    {
        bail!("S14 StarFold {label} binding 越界或 ABI 漂移");
    }
    Ok(())
}

fn aggregate_physical_counts(layers: &[S14StarfoldK4LayerReceipt]) -> Result<(u32, u64, u32)> {
    if layers.len() != FULL_DEPTH_LAYERS.len() {
        bail!("S14 StarFold K4 physical receipt 不是完整43层");
    }
    layers.iter().try_fold((0u32, 0u64, 0u32), |totals, layer| {
        Ok((
            totals
                .0
                .checked_add(layer.expert.packed_uploads)
                .context("S14 StarFold K4 packed upload count overflow")?,
            totals
                .1
                .checked_add(layer.expert.packed_upload_bytes)
                .context("S14 StarFold K4 packed upload bytes overflow")?,
            totals
                .2
                .checked_add(layer.expert.lane_dispatches)
                .context("S14 StarFold K4 lane dispatch count overflow")?,
        ))
    })
}
