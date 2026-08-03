//! S14 K=4/8 block-major 的 production 单层与43层结构接线。
//!
//! 一次调用严格执行三段：K-lane causal attention/router、该层 union Range
//! placement/materialize、一次 grouped MoE。43层后只接受一次 batched final head
//! 与 K 份真实 checkpoint；最长一致前缀只计算不提交，绝不修改 `S14Session`。

use crate::{
    s14_causal_block_resources::{
        prepare_causal_block_launch, S14CausalBlockLayerUnionPlacement,
        S14CausalBlockResourceError, S14CausalBlockUnionBankPlan, S14CausalBlockUnionBanks,
        S14_CAUSAL_BLOCK_PHYSICAL_RANGES_PER_EXPERT, S14_CAUSAL_BLOCK_UNION_BANKS,
    },
    s14_dynamic_routed_page_plan::{
        ExpertRangeIdentity, FullDepthExpertCatalog, OnlineTop6, RoutedProjection, RoutedRangePart,
    },
    s14_head_chunk_argmax::{
        S14HeadArgmaxResult, S14HeadChunkArgmaxRecordingReceipt, S14HeadChunkArgmaxShape,
    },
    s14_runtime::{S14Runtime, S14Session},
    s14_whole_token_device::WholeTokenDeviceBlockCommitReceipt,
    GpuBuffer,
};
use ash::vk;
use polaris_s14_runner::{
    build_layer_causal_batch_plan, BatchedWholeTokenOutput, DecoderStateV1, GraphProfile,
    LayerCausalBatchPlan, LongestPrefixDecision, MaterializedTokenSource, RouteDecision,
    WholeTokenBlockCommit, WholeTokenBlockRollback, WholeTokenFutureBlock, EXPERTS_PER_TOKEN,
    FULL_DEPTH_LAYERS, VOCAB_SIZE,
};
use std::{collections::BTreeMap, fmt};

pub const S14_CAUSAL_BLOCK_HC_STREAMS: usize = 4;
pub const S14_CAUSAL_BLOCK_STREAM_WIDTH: usize = 4096;
pub const S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE: usize =
    S14_CAUSAL_BLOCK_HC_STREAMS * S14_CAUSAL_BLOCK_STREAM_WIDTH;

/// Backend-owned、全程 device-resident 的 `[K,4,4096]` BF16 HC stream binding。
/// orchestrator 只传递强身份，不映射或回读 hidden。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockHiddenBinding {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub bytes: u64,
    pub block_size: usize,
    /// 每完成一个 attention/router 或 grouped-MoE 阶段必须精确递增1。
    pub generation: u64,
}

impl S14CausalBlockHiddenBinding {
    fn validate(self, block_size: usize) -> Result<(), S14CausalBlockLayerError> {
        let expected_bytes = u64::try_from(
            block_size
                .checked_mul(S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE)
                .ok_or_else(|| layer_error("causal-block HC stream shape overflow"))?,
        )
        .map_err(|_| layer_error("causal-block HC stream bytes overflow"))?
        .checked_mul(2)
        .ok_or_else(|| layer_error("causal-block HC stream BF16 bytes overflow"))?;
        if self.buffer == vk::Buffer::null()
            || self.offset % 4 != 0
            || self.bytes != expected_bytes
            || self.block_size != block_size
        {
            return Err(layer_error(
                "causal-block hidden binding 不是精确 device [K,4,4096] BF16",
            ));
        }
        Ok(())
    }

    fn next_generation(self) -> Result<u64, S14CausalBlockLayerError> {
        self.generation
            .checked_add(1)
            .ok_or_else(|| layer_error("causal-block hidden generation overflow"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockUnionBankBinding {
    pub bank_index: usize,
    pub buffer: vk::Buffer,
    pub allocated_bank_bytes: u64,
}

impl S14CausalBlockUnionBankBinding {
    fn from_runtime_bank(bank_index: usize, buffer: &GpuBuffer) -> Self {
        Self {
            bank_index,
            buffer: buffer.handle(),
            allocated_bank_bytes: buffer.size(),
        }
    }

    fn validate(&self, plan: &S14CausalBlockUnionBankPlan) -> Result<(), S14CausalBlockLayerError> {
        if self.bank_index >= S14_CAUSAL_BLOCK_UNION_BANKS
            || self.buffer == vk::Buffer::null()
            || self.allocated_bank_bytes < plan.allocated_bank_bytes
        {
            return Err(layer_error("causal-block union bank binding/capacity 非法"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14CausalBlockPhysicalRange {
    pub expert_id: u16,
    pub projection: RoutedProjection,
    pub part: RoutedRangePart,
    pub tensor: String,
    pub range_key: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14CausalBlockLayerRangePlan {
    pub layer: u8,
    pub block_size: usize,
    pub unique_experts: usize,
    pub physical_ranges: usize,
    pub union_expert_bytes: u64,
    pub ranges: Vec<S14CausalBlockPhysicalRange>,
}

#[derive(Clone, Debug)]
pub struct S14CausalBlockAttentionRouterOutput {
    pub post_attention_hidden: S14CausalBlockHiddenBinding,
    pub routes: Vec<RouteDecision>,
    /// 必须精确为1；K次串行 attention/router 会在任何 Range 上传前被拒绝。
    pub forward_calls: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRangeEvidenceReceipt {
    pub proof_assets: usize,
    pub explicit_fetch_lane_plans: usize,
    pub mmap_requests: u64,
    pub mmap_hits: u64,
    pub mmap_misses: u64,
    pub sha256_bytes: u64,
    pub staging_range_copies: usize,
    pub gpu_upload_copy_regions: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14CausalBlockUnionMaterializeReceipt {
    pub layer: u8,
    pub bank_index: usize,
    pub unique_experts: usize,
    pub physical_ranges: usize,
    pub uploaded_bytes: u64,
    pub materialize_calls: u32,
    pub range_evidence: S14CausalBlockRangeEvidenceReceipt,
}

#[derive(Clone, Debug)]
pub struct S14CausalBlockGroupedMoeOutput {
    pub output_hidden: S14CausalBlockHiddenBinding,
    /// 整层全部唯一专家必须收进一次 grouped submit。
    pub grouped_submit_calls: u32,
    /// 必须为0；禁止在 grouped 实现内部调用 K 次 token forward。
    pub serial_token_forward_calls: u32,
    pub unique_experts: usize,
}

#[derive(Clone, Debug)]
pub struct S14CausalBlockLayerReceipt {
    pub layer: u8,
    pub block_size: usize,
    pub bank_index: usize,
    pub unique_experts: usize,
    pub physical_ranges: usize,
    pub union_expert_bytes: u64,
    pub attention_router_forward_calls: u32,
    pub union_range_materialize_calls: u32,
    pub range_evidence: S14CausalBlockRangeEvidenceReceipt,
    pub grouped_moe_submit_calls: u32,
    pub serial_token_forward_calls: u32,
    pub routes: Vec<RouteDecision>,
    pub output_hidden: S14CausalBlockHiddenBinding,
    /// head/checkpoint/最长前缀提交尚未接入；调用方不得发布本层输出。
    pub head_commit_ready: bool,
}

pub struct S14CausalBlockLayerInput<'a> {
    pub base_position: u32,
    pub layer: u8,
    pub input_token_ids: &'a [u32],
    pub input_hidden: S14CausalBlockHiddenBinding,
    pub source: MaterializedTokenSource,
}

/// Production backend 的最窄单层 ABI。每个方法由 orchestrator 精确调用一次。
pub trait S14CausalBlockLayerBackend {
    fn run_k_lane_attention_router(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> Result<S14CausalBlockAttentionRouterOutput, String>;

    fn materialize_union_ranges(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockUnionMaterializeReceipt, String>;

    fn run_grouped_moe(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockGroupedMoeOutput, String>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockBeginReceipt {
    pub begin_calls: u32,
    pub base_position: u32,
    pub block_size: usize,
    pub bank_index: usize,
    pub active: bool,
    pub serial_token_forward_calls: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockSealReceipt {
    pub seal_calls: u32,
    pub completed_layers: usize,
    pub drained: bool,
    pub active: bool,
    pub head_submit_calls: u32,
    pub checkpoint_commit_calls: u32,
    pub serial_token_forward_calls: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockAbortReceipt {
    pub abort_calls: u32,
    pub completed_layers: usize,
    pub drained: bool,
    pub active: bool,
    pub head_submit_calls: u32,
    pub checkpoint_commit_calls: u32,
}

/// 43层 block-major backend 生命周期。成功路径只允许 seal 已完成的层图；失败路径必须
/// drain 后 abort。两个路径都禁止录制 final head 或提交 checkpoint。
pub trait S14CausalBlockFullDepthBackend: S14CausalBlockLayerBackend {
    fn begin_full_depth_block(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        base_position: u32,
        block_size: usize,
    ) -> Result<S14CausalBlockBeginReceipt, String>;

    fn seal_full_depth_layers(
        &mut self,
        completed_layers: usize,
    ) -> Result<S14CausalBlockSealReceipt, String>;

    fn drain_and_abort_full_depth_block(
        &mut self,
        completed_layers: usize,
    ) -> Result<S14CausalBlockAbortReceipt, String>;
}

#[derive(Debug)]
pub struct S14CausalBlockFinalOutput {
    pub output: BatchedWholeTokenOutput,
    /// 一次 K-row、32 chunk 的 production head 录制回执。
    pub head_recording: S14HeadChunkArgmaxRecordingReceipt,
    /// 与 GPU accumulator 一次解码得到的 K 行 top-1；必须逐行等于 output prediction。
    pub head_results: Vec<S14HeadArgmaxResult>,
    /// 与 K 份 host checkpoint 同源的 device prefix checkpoint / 无损 delta journal。
    /// 所有权必须从 backend 移入 sealed future，不能只留下可复制的 Vulkan handle。
    pub device_future: S14CausalBlockOwnedDeviceFuture,
    /// K 行 hidden 必须在一次 batched final-head submit 中完成。
    pub batched_head_submit_calls: u32,
    /// K 份完整 KV/HC/compressor/indexer candidate 必须一次导出。
    pub checkpoint_export_calls: u32,
    /// 必须为0；禁止用 K 次单 token head/step 冒充 block forward。
    pub serial_token_forward_calls: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14CausalBlockDeviceCheckpointStorage {
    PrefixCheckpoints,
    LosslessDeltaJournal,
}

/// Device future 的强回执。它只描述 backing owner 持有的资源，不能独立延长资源寿命。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockDeviceFutureReceipt {
    pub base_position: u32,
    pub block_size: usize,
    pub checkpoint_count: usize,
    pub storage: S14CausalBlockDeviceCheckpointStorage,
    pub checkpoint_arena: vk::Buffer,
    pub checkpoint_arena_offset: u64,
    pub checkpoint_arena_bytes: u64,
    /// 每个 prefix snapshot 的固定步长及其中 authoritative state 的精确字节数。
    pub checkpoint_stride_bytes: u64,
    pub checkpoint_state_bytes: u64,
    pub final_hidden: S14CausalBlockHiddenBinding,
    /// 导出与 terminal/head 已完成的 timeline；publish recorder 必须等待此值。
    pub ready_timeline: vk::Semaphore,
    pub ready_timeline_value: u64,
}

impl S14CausalBlockDeviceFutureReceipt {
    pub(crate) fn validate(
        self,
        base_position: u32,
        block_size: usize,
        final_hidden: S14CausalBlockHiddenBinding,
    ) -> Result<(), S14CausalBlockLayerError> {
        final_hidden.validate(block_size)?;
        let arena_required = self
            .checkpoint_stride_bytes
            .checked_mul(block_size.saturating_sub(1) as u64)
            .and_then(|prefix| prefix.checked_add(self.checkpoint_state_bytes));
        if self.base_position != base_position
            || self.block_size != block_size
            || self.checkpoint_count != block_size
            || self.checkpoint_arena == vk::Buffer::null()
            || self.checkpoint_arena_offset % 4 != 0
            || self.checkpoint_stride_bytes < self.checkpoint_state_bytes
            || self.checkpoint_state_bytes == 0
            || arena_required.is_none_or(|required| required > self.checkpoint_arena_bytes)
            || self.final_hidden != final_hidden
            || self.ready_timeline == vk::Semaphore::null()
            || self.ready_timeline_value == 0
        {
            return Err(layer_error(
                "device future owner/receipt 与 K-lane checkpoints 或 terminal hidden 不闭合",
            ));
        }
        Ok(())
    }
}

/// 真实 backend 对 future backing resources 的独占所有权。实现者的 `Drop` 必须在
/// rollback/拒绝/错误路径释放或归还 checkpoint arena；trait 故意不提供 Clone/commit。
pub trait S14CausalBlockDeviceFutureOwner: fmt::Debug {
    fn receipt(&self) -> S14CausalBlockDeviceFutureReceipt;
}

/// 不可克隆的 type-erased device owner。只有本对象存活时 receipt 中的 handle 才有效。
pub struct S14CausalBlockOwnedDeviceFuture {
    owner: Box<dyn S14CausalBlockDeviceFutureOwner>,
}

impl S14CausalBlockOwnedDeviceFuture {
    pub fn new(owner: impl S14CausalBlockDeviceFutureOwner + 'static) -> Self {
        Self {
            owner: Box::new(owner),
        }
    }

    pub fn receipt(&self) -> S14CausalBlockDeviceFutureReceipt {
        self.owner.receipt()
    }
}

impl fmt::Debug for S14CausalBlockOwnedDeviceFuture {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockOwnedDeviceFuture")
            .field("receipt", &self.receipt())
            .finish_non_exhaustive()
    }
}

/// 只有真实 backend 已拥有 K 份完整 KV/HC/compressor/indexer candidate，并能对
/// K 行 terminal hidden 执行一次 batched final head 时才实现此接口。orchestrator
/// 不生成 token、head 或状态，只校验 backend 的一次强回执并交给 runner 复核 checkpoint chain。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockPostSealReceipt {
    pub hook_calls: u32,
    pub completed_layers: usize,
    pub base_position: u32,
    pub block_size: usize,
    pub published_terminal_sources: u32,
    pub serial_token_forward_calls: u32,
}

pub trait S14CausalBlockCheckpointBackend: S14CausalBlockFullDepthBackend {
    /// `seal_full_depth_layers` 已完成、terminal/head 尚未开始时的唯一 producer 窗口。
    /// production 总装层在这里发布同一 block 的真实 terminal source；默认零调用仅供
    /// 不需要 out-of-band source 的测试/专用 backend。禁止在 hook 内循环执行 token step。
    fn publish_terminal_source_after_full_depth_seal(
        &mut self,
        completed_layers: usize,
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
        _routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<S14CausalBlockPostSealReceipt, String> {
        Ok(S14CausalBlockPostSealReceipt {
            hook_calls: 0,
            completed_layers,
            base_position,
            block_size: final_hidden.block_size,
            published_terminal_sources: 0,
            serial_token_forward_calls: 0,
        })
    }

    fn run_batched_final_head_and_export_checkpoints(
        &mut self,
        completed_layers: usize,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<S14CausalBlockFinalOutput, String>;

    /// Orchestrator 已完成 head/checkpoint/route/device owner 的全部强校验，并将 owned
    /// future 接入 sealed future。production backend 用它解除“export 尚未验收”闩锁；
    /// 默认实现仅供不复用真实 arena 的测试 backend。
    fn acknowledge_export_validated(&mut self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14CausalBlockLayerSummary {
    pub layer: u8,
    pub unique_experts: usize,
    pub physical_ranges: usize,
    pub union_expert_bytes: u64,
    pub range_evidence: S14CausalBlockRangeEvidenceReceipt,
}

#[derive(Clone, Debug)]
pub struct S14CausalBlockFullDepthReceipt {
    pub base_position: u32,
    pub block_size: usize,
    pub bank_index: usize,
    pub completed_layers: usize,
    pub layer_summaries: Vec<S14CausalBlockLayerSummary>,
    /// position-major `[K][43]` 原生在线 route；绑定 future checkpoint 时必须逐项相等。
    pub routes_by_position: Vec<Vec<RouteDecision>>,
    pub attention_router_forward_calls: u32,
    pub union_range_materialize_calls: u32,
    pub grouped_moe_submit_calls: u32,
    pub serial_token_forward_calls: u32,
    /// terminal `[K,4,4096]` BF16 HC streams；batched final head 的唯一输入。
    pub final_hidden: S14CausalBlockHiddenBinding,
    pub lifecycle_sealed: bool,
    /// full-depth hidden 已闭合，但 final head 尚未进入该 graph。
    pub head_ready: bool,
    /// 没有最长一致前缀决策，session/checkpoint 绝不能发布。
    pub checkpoint_commit_ready: bool,
}

/// 已通过43层、完整 checkpoint chain 与最长一致前缀计算，但尚未发布的 future block。
/// 本类型不暴露 commit 入口；drop 或 `rollback` 都不会改变 authoritative state。
#[derive(Debug)]
pub struct S14CausalBlockSealedFuture {
    pub layers: S14CausalBlockFullDepthReceipt,
    decision: LongestPrefixDecision,
    future: WholeTokenFutureBlock,
    device_future: S14CausalBlockOwnedDeviceFuture,
}

/// 同一个 sealed future 选出的 host checkpoint 与仍被 owner 保活的 device prefix。
/// 构造器私有，调用方不能把另一个 future 的 host/device 两侧拼接起来。
#[derive(Debug)]
pub struct S14CausalBlockSelectedPrefix<'a> {
    accepted_tokens: usize,
    checkpoint_index: usize,
    checkpoint: DecoderStateV1,
    device_future: &'a S14CausalBlockOwnedDeviceFuture,
}

#[derive(Debug)]
pub struct S14CausalBlockPublishReceipt {
    pub host: WholeTokenBlockCommit,
    pub device: WholeTokenDeviceBlockCommitReceipt,
}

impl S14CausalBlockSelectedPrefix<'_> {
    pub fn accepted_tokens(&self) -> usize {
        self.accepted_tokens
    }

    pub fn checkpoint_index(&self) -> usize {
        self.checkpoint_index
    }

    pub fn checkpoint(&self) -> &DecoderStateV1 {
        &self.checkpoint
    }

    pub fn device_receipt(&self) -> S14CausalBlockDeviceFutureReceipt {
        self.device_future.receipt()
    }
}

impl S14CausalBlockSealedFuture {
    pub fn decision(&self) -> &LongestPrefixDecision {
        &self.decision
    }

    pub fn device_receipt(&self) -> S14CausalBlockDeviceFutureReceipt {
        self.device_future.receipt()
    }

    pub fn selected_prefix(
        &self,
    ) -> Result<S14CausalBlockSelectedPrefix<'_>, S14CausalBlockLayerError> {
        let accepted_tokens = self.decision.committed_token_ids.len();
        let (checkpoint_index, checkpoint) = self
            .future
            .selected_checkpoint()
            .map_err(|error| layer_error(format!("选择 host prefix checkpoint 失败: {error}")))?;
        if !(1..=self.layers.block_size).contains(&accepted_tokens)
            || checkpoint_index + 1 != accepted_tokens
        {
            return Err(layer_error(
                "最长前缀 accepted count 与 host/device checkpoint index 漂移",
            ));
        }
        Ok(S14CausalBlockSelectedPrefix {
            accepted_tokens,
            checkpoint_index,
            checkpoint,
            device_future: &self.device_future,
        })
    }

    pub fn rollback(
        self,
        authoritative: &DecoderStateV1,
    ) -> Result<WholeTokenBlockRollback, S14CausalBlockLayerError> {
        let Self {
            future,
            device_future,
            ..
        } = self;
        let rollback = future
            .rollback(authoritative)
            .map_err(|error| layer_error(format!("future block rollback 失败: {error}")))?;
        // device owner 在 host rollback 已验证后离开作用域；其 Drop 负责归还未发布资源。
        drop(device_future);
        Ok(rollback)
    }
}

/// 严格按层主序执行43层。这里的循环是网络深度，不是按token循环调用 `step()`；每层
/// backend 一次接收全部K行，并由单层强合同拒绝任何串行token forward。
pub fn execute_causal_block_full_depth<B: S14CausalBlockFullDepthBackend>(
    union_plan: &S14CausalBlockUnionBankPlan,
    bank: S14CausalBlockUnionBankBinding,
    catalog: &FullDepthExpertCatalog,
    base_position: u32,
    input_token_ids: &[u32],
    initial_hidden: S14CausalBlockHiddenBinding,
    source: MaterializedTokenSource,
    backend: &mut B,
) -> Result<S14CausalBlockFullDepthReceipt, S14CausalBlockLayerError> {
    validate_layer_input(
        union_plan,
        bank,
        &S14CausalBlockLayerInput {
            base_position,
            layer: FULL_DEPTH_LAYERS[0],
            input_token_ids,
            input_hidden: initial_hidden,
            source,
        },
    )?;
    let begin = backend
        .begin_full_depth_block(bank, base_position, union_plan.block_size)
        .map_err(|error| layer_error(format!("begin full-depth causal block 失败: {error}")))?;
    if begin.begin_calls != 1
        || begin.base_position != base_position
        || begin.block_size != union_plan.block_size
        || begin.bank_index != bank.bank_index
        || !begin.active
        || begin.serial_token_forward_calls != 0
    {
        return Err(abort_full_depth_after_error(
            backend,
            0,
            layer_error("full-depth causal block begin 强回执漂移"),
        ));
    }

    let mut current_hidden = initial_hidden;
    let mut layer_summaries = Vec::with_capacity(FULL_DEPTH_LAYERS.len());
    let mut routes_by_position = (0..union_plan.block_size)
        .map(|_| Vec::with_capacity(FULL_DEPTH_LAYERS.len()))
        .collect::<Vec<_>>();
    let mut attention_calls = 0u32;
    let mut materialize_calls = 0u32;
    let mut grouped_calls = 0u32;
    for &layer in &FULL_DEPTH_LAYERS {
        let layer_result = execute_causal_block_layer(
            union_plan,
            bank,
            catalog,
            S14CausalBlockLayerInput {
                base_position,
                layer,
                input_token_ids,
                input_hidden: current_hidden,
                source,
            },
            backend,
        );
        let mut receipt = match layer_result {
            Ok(receipt) => receipt,
            Err(error) => {
                return Err(abort_full_depth_after_error(
                    backend,
                    layer_summaries.len(),
                    error,
                ));
            }
        };
        if receipt.layer != layer
            || receipt.block_size != union_plan.block_size
            || receipt.bank_index != bank.bank_index
            || receipt.head_commit_ready
            || receipt.serial_token_forward_calls != 0
            || receipt.routes.len() != union_plan.block_size
        {
            return Err(abort_full_depth_after_error(
                backend,
                layer_summaries.len(),
                layer_error("full-depth layer receipt/order 漂移"),
            ));
        }
        // 单层强合同已把三项精确限制为1，循环上界固定43，因此这里不会溢出。
        attention_calls += receipt.attention_router_forward_calls;
        materialize_calls += receipt.union_range_materialize_calls;
        grouped_calls += receipt.grouped_moe_submit_calls;
        for (position_routes, route) in routes_by_position.iter_mut().zip(receipt.routes.drain(..))
        {
            position_routes.push(route);
        }
        current_hidden = receipt.output_hidden;
        layer_summaries.push(S14CausalBlockLayerSummary {
            layer,
            unique_experts: receipt.unique_experts,
            physical_ranges: receipt.physical_ranges,
            union_expert_bytes: receipt.union_expert_bytes,
            range_evidence: receipt.range_evidence,
        });
    }

    let seal = match backend.seal_full_depth_layers(layer_summaries.len()) {
        Ok(receipt) => receipt,
        Err(error) => {
            return Err(abort_full_depth_after_error(
                backend,
                layer_summaries.len(),
                layer_error(format!("seal full-depth layers 失败: {error}")),
            ));
        }
    };
    if seal.seal_calls != 1
        || seal.completed_layers != FULL_DEPTH_LAYERS.len()
        || !seal.drained
        || seal.active
        || seal.head_submit_calls != 0
        || seal.checkpoint_commit_calls != 0
        || seal.serial_token_forward_calls != 0
    {
        return Err(abort_full_depth_after_error(
            backend,
            layer_summaries.len(),
            layer_error("full-depth seal/drain/head/commit 强回执漂移"),
        ));
    }
    let expected_calls = FULL_DEPTH_LAYERS.len() as u32;
    if attention_calls != expected_calls
        || materialize_calls != expected_calls
        || grouped_calls != expected_calls
        || routes_by_position
            .iter()
            .any(|routes| routes.len() != FULL_DEPTH_LAYERS.len())
    {
        return Err(abort_full_depth_after_error(
            backend,
            layer_summaries.len(),
            layer_error("full-depth 43层三段调用计数漂移"),
        ));
    }

    Ok(S14CausalBlockFullDepthReceipt {
        base_position,
        block_size: union_plan.block_size,
        bank_index: bank.bank_index,
        completed_layers: layer_summaries.len(),
        layer_summaries,
        routes_by_position,
        attention_router_forward_calls: attention_calls,
        union_range_materialize_calls: materialize_calls,
        grouped_moe_submit_calls: grouped_calls,
        serial_token_forward_calls: 0,
        final_hidden: current_hidden,
        lifecycle_sealed: true,
        head_ready: false,
        checkpoint_commit_ready: false,
    })
}

/// 在43层seal之后，把 backend 已真实生成的 K 份完整 candidate checkpoint 绑定到
/// `WholeTokenFutureBlock`。本函数只计算最长一致前缀，不调用 commit。
pub fn execute_causal_block_full_depth_with_checkpoints<B: S14CausalBlockCheckpointBackend>(
    union_plan: &S14CausalBlockUnionBankPlan,
    bank: S14CausalBlockUnionBankBinding,
    catalog: &FullDepthExpertCatalog,
    authoritative: &DecoderStateV1,
    draft_token_ids: &[u32],
    input_token_ids: &[u32],
    initial_hidden: S14CausalBlockHiddenBinding,
    source: MaterializedTokenSource,
    backend: &mut B,
) -> Result<S14CausalBlockSealedFuture, S14CausalBlockLayerError> {
    authoritative
        .validate()
        .map_err(|error| layer_error(format!("authoritative DecoderState 非法: {error}")))?;
    if draft_token_ids.len() != union_plan.block_size
        || input_token_ids.len() != union_plan.block_size
        || input_token_ids.first().copied() != Some(authoritative.input_token_id)
        || input_token_ids[1..] != draft_token_ids[..union_plan.block_size - 1]
    {
        return Err(layer_error(
            "future block draft/input token 与 authoritative state 不闭合",
        ));
    }
    let authoritative_before = authoritative.clone();
    let mut layers = execute_causal_block_full_depth(
        union_plan,
        bank,
        catalog,
        authoritative.position,
        input_token_ids,
        initial_hidden,
        source,
        backend,
    )?;
    if authoritative != &authoritative_before {
        return Err(abort_full_depth_after_error(
            backend,
            layers.completed_layers,
            layer_error("43层seal后 authoritative state 被修改"),
        ));
    }

    let post_seal = match backend.publish_terminal_source_after_full_depth_seal(
        layers.completed_layers,
        layers.base_position,
        layers.final_hidden,
        &layers.routes_by_position,
    ) {
        Ok(receipt) => receipt,
        Err(error) => {
            return Err(abort_full_depth_after_error(
                backend,
                layers.completed_layers,
                layer_error(format!("post-seal terminal source 发布失败: {error}")),
            ));
        }
    };
    if post_seal.completed_layers != layers.completed_layers
        || post_seal.base_position != layers.base_position
        || post_seal.block_size != layers.block_size
        || post_seal.hook_calls > 1
        || post_seal.published_terminal_sources != post_seal.hook_calls
        || post_seal.serial_token_forward_calls != 0
    {
        return Err(abort_full_depth_after_error(
            backend,
            layers.completed_layers,
            layer_error("post-seal terminal producer identity/调用计数漂移或退化为串行token"),
        ));
    }

    let final_output = match backend.run_batched_final_head_and_export_checkpoints(
        layers.completed_layers,
        layers.final_hidden,
        &layers.routes_by_position,
    ) {
        Ok(output) => output,
        Err(error) => {
            return Err(abort_full_depth_after_error(
                backend,
                layers.completed_layers,
                layer_error(format!("batched final head/checkpoint 导出失败: {error}")),
            ));
        }
    };
    if final_output.batched_head_submit_calls != 1
        || final_output.checkpoint_export_calls != 1
        || final_output.serial_token_forward_calls != 0
    {
        return Err(abort_full_depth_after_error(
            backend,
            layers.completed_layers,
            layer_error("batched final head/checkpoint 强回执漂移"),
        ));
    }
    if let Err(error) = validate_batched_head_output(&final_output, layers.block_size) {
        return Err(abort_full_depth_after_error(
            backend,
            layers.completed_layers,
            error,
        ));
    }
    if let Err(error) = final_output.device_future.receipt().validate(
        layers.base_position,
        layers.block_size,
        layers.final_hidden,
    ) {
        return Err(abort_full_depth_after_error(
            backend,
            layers.completed_layers,
            error,
        ));
    }
    let S14CausalBlockFinalOutput {
        output,
        device_future,
        ..
    } = final_output;
    layers.head_ready = true;
    if output.positions.len() != layers.block_size
        || output
            .positions
            .iter()
            .zip(&layers.routes_by_position)
            .any(|(position, expected_routes)| position.routes != *expected_routes)
    {
        return Err(abort_full_depth_after_error(
            backend,
            layers.completed_layers,
            layer_error("future checkpoint routes 与43层在线route逐项漂移"),
        ));
    }
    let future = match WholeTokenFutureBlock::from_batched_output(
        authoritative,
        draft_token_ids.to_vec(),
        output,
    ) {
        Ok(future) => future,
        Err(error) => {
            return Err(abort_full_depth_after_error(
                backend,
                layers.completed_layers,
                layer_error(format!("future checkpoint chain 非法: {error}")),
            ));
        }
    };
    if authoritative != &authoritative_before {
        return Err(abort_full_depth_after_error(
            backend,
            layers.completed_layers,
            layer_error("future block构造阶段修改了 authoritative state"),
        ));
    }
    let decision = future.decision();
    if let Err(error) = backend.acknowledge_export_validated() {
        return Err(abort_full_depth_after_error(
            backend,
            layers.completed_layers,
            layer_error(format!("backend future export 验收回执失败: {error}")),
        ));
    }
    Ok(S14CausalBlockSealedFuture {
        layers,
        decision,
        future,
        device_future,
    })
}

fn validate_batched_head_output(
    final_output: &S14CausalBlockFinalOutput,
    block_size: usize,
) -> Result<(), S14CausalBlockLayerError> {
    let batch =
        u32::try_from(block_size).map_err(|_| layer_error("K-lane head batch 无法转换为u32"))?;
    let expected_shape = S14HeadChunkArgmaxShape::production_batched(batch)
        .map_err(|error| layer_error(format!("K-lane production head shape 非法: {error}")))?;
    if final_output.output.forward_calls != 1
        || final_output.output.positions.len() != block_size
        || final_output.head_recording.shape != expected_shape
        || final_output.head_recording.submitted_chunks != expected_shape.chunk_count()
        || final_output.head_recording.expected_next_token != expected_shape.vocab
        || final_output.head_results.len() != block_size
        || final_output
            .head_results
            .iter()
            .zip(&final_output.output.positions)
            .any(|(head, position)| {
                head.token_id != position.predicted_token_id
                    || head.token_id >= VOCAB_SIZE
                    || !head.logit.is_finite()
            })
    {
        return Err(layer_error(
            "K-lane batched lm-head 未证明一次32-chunk权重扫描或 prediction 漂移",
        ));
    }
    Ok(())
}

fn abort_full_depth_after_error<B: S14CausalBlockFullDepthBackend>(
    backend: &mut B,
    completed_layers: usize,
    original: S14CausalBlockLayerError,
) -> S14CausalBlockLayerError {
    match backend.drain_and_abort_full_depth_block(completed_layers) {
        Ok(receipt)
            if receipt.abort_calls == 1
                && receipt.completed_layers == completed_layers
                && receipt.drained
                && !receipt.active
                && receipt.head_submit_calls == 0
                && receipt.checkpoint_commit_calls == 0 =>
        {
            layer_error(format!(
                "{original}; full-depth candidate 已drain/abort，completed_layers={completed_layers}"
            ))
        }
        Ok(receipt) => layer_error(format!(
            "{original}; full-depth abort 强回执漂移: {receipt:?}"
        )),
        Err(abort_error) => layer_error(format!(
            "{original}; full-depth drain/abort 失败: {abort_error}"
        )),
    }
}

pub fn execute_causal_block_layer<B: S14CausalBlockLayerBackend>(
    union_plan: &S14CausalBlockUnionBankPlan,
    bank: S14CausalBlockUnionBankBinding,
    catalog: &FullDepthExpertCatalog,
    input: S14CausalBlockLayerInput<'_>,
    backend: &mut B,
) -> Result<S14CausalBlockLayerReceipt, S14CausalBlockLayerError> {
    validate_layer_input(union_plan, bank, &input)?;
    let block_size = input.input_token_ids.len();
    let attention = backend
        .run_k_lane_attention_router(&input)
        .map_err(|error| layer_error(format!("K-lane attention/router 失败: {error}")))?;
    let expected_attention_generation = input.input_hidden.next_generation()?;
    if attention.forward_calls != 1 || attention.routes.len() != block_size {
        return Err(layer_error(
            "K-lane attention/router 必须一次返回 device [K,4,4096] BF16 binding 与 K 份 route",
        ));
    }
    attention.post_attention_hidden.validate(block_size)?;
    if attention.post_attention_hidden.generation != expected_attention_generation
        || (attention.post_attention_hidden.buffer == input.input_hidden.buffer
            && attention.post_attention_hidden.offset == input.input_hidden.offset)
    {
        return Err(layer_error("K-lane attention hidden generation 漂移"));
    }
    for route in &attention.routes {
        route
            .validate_for(GraphProfile::FullDepth43NativeTop6)
            .map_err(|error| layer_error(format!("online route 非法: {error}")))?;
        if route.layer != input.layer {
            return Err(layer_error("attention/router 输出 layer 漂移"));
        }
    }

    let batch_plan = build_layer_causal_batch_plan(&attention.routes)
        .map_err(|error| layer_error(format!("构造 layer union plan 失败: {error}")))?;
    let placement = union_plan
        .place_layer_plan(&batch_plan)
        .map_err(map_resource_error)?;
    let range_plan = build_layer_union_range_plan(
        catalog,
        input.base_position,
        &attention.routes,
        &batch_plan,
        &placement,
    )?;

    let materialized = backend
        .materialize_union_ranges(bank, &range_plan)
        .map_err(|error| layer_error(format!("union Range materialize 失败: {error}")))?;
    if materialized.layer != input.layer
        || materialized.bank_index != bank.bank_index
        || materialized.unique_experts != range_plan.unique_experts
        || materialized.physical_ranges != range_plan.physical_ranges
        || materialized.uploaded_bytes != range_plan.union_expert_bytes
        || materialized.materialize_calls != 1
        || materialized.range_evidence.proof_assets != range_plan.physical_ranges
        || materialized.range_evidence.mmap_requests != range_plan.physical_ranges as u64
        || materialized
            .range_evidence
            .mmap_hits
            .checked_add(materialized.range_evidence.mmap_misses)
            != Some(materialized.range_evidence.mmap_requests)
        || (materialized.range_evidence.mmap_misses != 0
            && materialized.range_evidence.sha256_bytes == 0)
        || materialized.range_evidence.staging_range_copies != range_plan.physical_ranges
        || materialized.range_evidence.gpu_upload_copy_regions != 1
        || materialized.range_evidence.explicit_fetch_lane_plans > block_size
    {
        return Err(layer_error("union Range materialize 强回执漂移"));
    }

    let grouped = backend
        .run_grouped_moe(
            bank,
            attention.post_attention_hidden,
            &attention.routes,
            &batch_plan,
            &range_plan,
        )
        .map_err(|error| layer_error(format!("grouped MoE 失败: {error}")))?;
    let expected_grouped_generation = attention.post_attention_hidden.next_generation()?;
    if grouped.grouped_submit_calls != 1
        || grouped.serial_token_forward_calls != 0
        || grouped.unique_experts != batch_plan.unique_experts
    {
        return Err(layer_error(
            "grouped MoE 必须一次submit、零串行token forward并返回 device HC binding",
        ));
    }
    grouped.output_hidden.validate(block_size)?;
    if grouped.output_hidden.generation != expected_grouped_generation
        || (grouped.output_hidden.buffer == attention.post_attention_hidden.buffer
            && grouped.output_hidden.offset == attention.post_attention_hidden.offset)
    {
        return Err(layer_error("grouped MoE hidden generation 漂移"));
    }

    Ok(S14CausalBlockLayerReceipt {
        layer: input.layer,
        block_size,
        bank_index: bank.bank_index,
        unique_experts: batch_plan.unique_experts,
        physical_ranges: range_plan.physical_ranges,
        union_expert_bytes: range_plan.union_expert_bytes,
        attention_router_forward_calls: attention.forward_calls,
        union_range_materialize_calls: materialized.materialize_calls,
        range_evidence: materialized.range_evidence,
        grouped_moe_submit_calls: grouped.grouped_submit_calls,
        serial_token_forward_calls: grouped.serial_token_forward_calls,
        routes: attention.routes,
        output_hidden: grouped.output_hidden,
        head_commit_ready: false,
    })
}

fn validate_layer_input(
    union_plan: &S14CausalBlockUnionBankPlan,
    bank: S14CausalBlockUnionBankBinding,
    input: &S14CausalBlockLayerInput<'_>,
) -> Result<(), S14CausalBlockLayerError> {
    bank.validate(union_plan)?;
    if !FULL_DEPTH_LAYERS.contains(&input.layer)
        || input.input_token_ids.len() != union_plan.block_size
        || input
            .input_token_ids
            .iter()
            .any(|&token| token >= VOCAB_SIZE)
    {
        return Err(layer_error(
            "causal-block layer input K/layer/token/hidden 非法",
        ));
    }
    input.input_hidden.validate(union_plan.block_size)?;
    let end = input
        .base_position
        .checked_add(union_plan.block_size as u32)
        .ok_or_else(|| layer_error("causal-block layer position overflow"))?;
    if end > 127 {
        return Err(layer_error("causal-block layer 越出当前 position<127 边界"));
    }
    Ok(())
}

fn build_layer_union_range_plan(
    catalog: &FullDepthExpertCatalog,
    base_position: u32,
    routes: &[RouteDecision],
    batch_plan: &LayerCausalBatchPlan,
    placement: &S14CausalBlockLayerUnionPlacement,
) -> Result<S14CausalBlockLayerRangePlan, S14CausalBlockLayerError> {
    let mut identities: BTreeMap<
        (u16, &'static str),
        (RoutedProjection, ExpertRangeIdentity, ExpertRangeIdentity),
    > = BTreeMap::new();
    for (offset, route) in routes.iter().enumerate() {
        let expert_ids: [u16; EXPERTS_PER_TOKEN] = route
            .expert_ids
            .clone()
            .try_into()
            .map_err(|_| layer_error("route expert IDs 不是精确 top-6"))?;
        let route_weights: [f32; EXPERTS_PER_TOKEN] = route
            .weights
            .clone()
            .try_into()
            .map_err(|_| layer_error("route weights 不是精确 top-6"))?;
        let position = u64::from(base_position)
            .checked_add(offset as u64)
            .ok_or_else(|| layer_error("union Range position overflow"))?;
        let route_plan = catalog
            .plan(OnlineTop6 {
                layer: route.layer,
                position,
                expert_ids,
                route_weights,
            })
            .map_err(|error| layer_error(format!("catalog route plan 失败: {error:#}")))?;
        for page in route_plan.pages {
            let key = (page.expert_id, page.projection.tensor_stem());
            let value = (page.projection, page.weight.clone(), page.scale.clone());
            if let Some(existing) = identities.get(&key) {
                if existing != &value {
                    return Err(layer_error("同一 union expert Range identity 漂移"));
                }
            } else {
                identities.insert(key, value);
            }
        }
    }

    let mut ranges = Vec::with_capacity(placement.physical_ranges);
    let mut bytes = 0u64;
    for expert in &batch_plan.experts {
        for stem in ["w1", "w2", "w3"] {
            let (projection, weight, scale) = identities
                .get(&(expert.expert_id, stem))
                .ok_or_else(|| layer_error("union expert 缺少完整 w1/w2/w3 Range"))?;
            for (part, range) in [
                (RoutedRangePart::Weight, weight),
                (RoutedRangePart::Scale, scale),
            ] {
                bytes = bytes
                    .checked_add(range.bytes)
                    .ok_or_else(|| layer_error("union Range bytes overflow"))?;
                ranges.push(S14CausalBlockPhysicalRange {
                    expert_id: expert.expert_id,
                    projection: *projection,
                    part,
                    tensor: range.tensor.clone(),
                    range_key: range.range_key.clone(),
                    bytes: range.bytes,
                });
            }
        }
    }
    if ranges.len() != placement.physical_ranges
        || ranges.len() != batch_plan.unique_experts * S14_CAUSAL_BLOCK_PHYSICAL_RANGES_PER_EXPERT
        || bytes != placement.used_bytes
        || bytes != batch_plan.union_expert_bytes
    {
        return Err(layer_error(
            "union Range count/bytes 与 bank placement 漂移",
        ));
    }
    Ok(S14CausalBlockLayerRangePlan {
        layer: batch_plan.layer,
        block_size: batch_plan.block_size,
        unique_experts: batch_plan.unique_experts,
        physical_ranges: ranges.len(),
        union_expert_bytes: bytes,
        ranges,
    })
}

impl S14Runtime {
    /// 仅闭合一个未发布层；不会运行 final head，也不会修改 session/checkpoint。
    pub fn execute_causal_block_layer_contract<B: S14CausalBlockLayerBackend>(
        &self,
        session: &S14Session,
        draft_token_ids: &[u32],
        layer: u8,
        input_hidden: S14CausalBlockHiddenBinding,
        source: MaterializedTokenSource,
        backend: &mut B,
    ) -> Result<S14CausalBlockLayerReceipt, S14CausalBlockLayerError> {
        if !self.owns_session(session) {
            return Err(layer_error("S14 session 不属于当前 runtime"));
        }
        let launch = prepare_causal_block_launch(
            session.position(),
            session.input_token_id(),
            draft_token_ids,
        )
        .map_err(map_resource_error)?;
        let banks: &S14CausalBlockUnionBanks = self
            .causal_block_union_banks()
            .ok_or_else(|| layer_error("S14 runtime union banks 已销毁"))?;
        let union_plan = banks.plan(launch.block_size).map_err(map_resource_error)?;
        if union_plan != &launch.union_banks {
            return Err(layer_error("runtime/launch union plan 漂移"));
        }
        let bank_index = session.commit_epoch() as usize % S14_CAUSAL_BLOCK_UNION_BANKS;
        let bank = banks.bank(bank_index).map_err(map_resource_error)?;
        let binding = S14CausalBlockUnionBankBinding::from_runtime_bank(bank_index, bank);
        let mut input_token_ids = Vec::with_capacity(launch.block_size);
        input_token_ids.push(launch.input_token_id);
        input_token_ids.extend_from_slice(&launch.draft_token_ids[..launch.block_size - 1]);
        execute_causal_block_layer(
            union_plan,
            binding,
            self.expert_catalog(),
            S14CausalBlockLayerInput {
                base_position: launch.base_position,
                layer,
                input_token_ids: &input_token_ids,
                input_hidden,
                source,
            },
            backend,
        )
    }

    /// 使用 runtime 常驻的真实 union bank 与 expert catalog 闭合43层 block-major 图。
    /// 本入口故意只借用 session；没有 final head/最长前缀决策，因此不能修改任何提交态。
    pub fn execute_causal_block_full_depth_contract<B: S14CausalBlockFullDepthBackend>(
        &self,
        session: &S14Session,
        draft_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
        source: MaterializedTokenSource,
        backend: &mut B,
    ) -> Result<S14CausalBlockFullDepthReceipt, S14CausalBlockLayerError> {
        if !self.owns_session(session) {
            return Err(layer_error("S14 session 不属于当前 runtime"));
        }
        let launch = prepare_causal_block_launch(
            session.position(),
            session.input_token_id(),
            draft_token_ids,
        )
        .map_err(map_resource_error)?;
        let banks = self
            .causal_block_union_banks()
            .ok_or_else(|| layer_error("S14 runtime union banks 已销毁"))?;
        let union_plan = banks.plan(launch.block_size).map_err(map_resource_error)?;
        if union_plan != &launch.union_banks {
            return Err(layer_error("runtime/launch union plan 漂移"));
        }
        let bank_index = session.commit_epoch() as usize % S14_CAUSAL_BLOCK_UNION_BANKS;
        let bank = banks.bank(bank_index).map_err(map_resource_error)?;
        let binding = S14CausalBlockUnionBankBinding::from_runtime_bank(bank_index, bank);
        let mut input_token_ids = Vec::with_capacity(launch.block_size);
        input_token_ids.push(launch.input_token_id);
        input_token_ids.extend_from_slice(&launch.draft_token_ids[..launch.block_size - 1]);
        execute_causal_block_full_depth(
            union_plan,
            binding,
            self.expert_catalog(),
            launch.base_position,
            &input_token_ids,
            initial_hidden,
            source,
            backend,
        )
    }

    /// 与 `execute_causal_block_full_depth_contract` 相同地借用 runtime 资源，并把真实
    /// K-lane candidate checkpoints 封进未发布 `WholeTokenFutureBlock`。session 始终只读。
    pub fn execute_causal_block_checkpoint_contract<B: S14CausalBlockCheckpointBackend>(
        &self,
        session: &S14Session,
        draft_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
        source: MaterializedTokenSource,
        backend: &mut B,
    ) -> Result<S14CausalBlockSealedFuture, S14CausalBlockLayerError> {
        let authoritative = self
            .authoritative_state(session)
            .ok_or_else(|| layer_error("S14 session 不属于当前 runtime"))?;
        let launch = prepare_causal_block_launch(
            authoritative.position,
            authoritative.input_token_id,
            draft_token_ids,
        )
        .map_err(map_resource_error)?;
        let banks = self
            .causal_block_union_banks()
            .ok_or_else(|| layer_error("S14 runtime union banks 已销毁"))?;
        let union_plan = banks.plan(launch.block_size).map_err(map_resource_error)?;
        if union_plan != &launch.union_banks {
            return Err(layer_error("runtime/launch union plan 漂移"));
        }
        let bank_index = authoritative.commit_epoch as usize % S14_CAUSAL_BLOCK_UNION_BANKS;
        let bank = banks.bank(bank_index).map_err(map_resource_error)?;
        let binding = S14CausalBlockUnionBankBinding::from_runtime_bank(bank_index, bank);
        let mut input_token_ids = Vec::with_capacity(launch.block_size);
        input_token_ids.push(launch.input_token_id);
        input_token_ids.extend_from_slice(&launch.draft_token_ids[..launch.block_size - 1]);
        execute_causal_block_full_depth_with_checkpoints(
            union_plan,
            binding,
            self.expert_catalog(),
            authoritative,
            draft_token_ids,
            &input_token_ids,
            initial_hidden,
            source,
            backend,
        )
    }

    /// 将已经 sealed 的同源 host/device prefix 原子发布到 session。
    /// GPU timeline wait 与 checkpoint copy 先写第三块 scratch；host commit 仍可能失败时
    /// active device bank 保持不变。host 成功后 device publish 只做不可失败的 owner swap。
    pub fn publish_causal_block_longest_prefix(
        &self,
        session: &mut S14Session,
        sealed: S14CausalBlockSealedFuture,
    ) -> Result<S14CausalBlockPublishReceipt, S14CausalBlockLayerError> {
        let (ctx, host, device) = self
            .session_host_device_mut(session)
            .ok_or_else(|| layer_error("S14 session/device state 不属于当前 runtime"))?;
        let selected = sealed.selected_prefix()?;
        let prepared = device
            .prepare_block_prefix_commit(ctx, &selected)
            .map_err(|error| layer_error(format!("prepare device prefix publish 失败: {error}")))?;
        drop(selected);
        let S14CausalBlockSealedFuture {
            future,
            device_future,
            ..
        } = sealed;
        let host_receipt = match future.commit_longest_prefix(host) {
            Ok(receipt) => receipt,
            Err(error) => {
                let rollback = device.rollback_prepared_block_commit(prepared);
                drop(device_future);
                return Err(match rollback {
                    Ok(()) => layer_error(format!(
                        "host longest-prefix commit 失败，device scratch 已rollback: {error}"
                    )),
                    Err(rollback_error) => layer_error(format!(
                        "host longest-prefix commit 失败且device rollback失败: {error}; {rollback_error}"
                    )),
                });
            }
        };
        let device_receipt = device.publish_prepared_block_commit(prepared);
        drop(device_future);
        assert_eq!(host_receipt.committed_epoch, device_receipt.epoch);
        assert_eq!(host_receipt.committed_position, device_receipt.position);
        assert_eq!(
            host_receipt.checkpoint_index,
            device_receipt.checkpoint_index
        );
        assert_eq!(
            host_receipt.decision.committed_token_ids.len(),
            device_receipt.accepted_tokens
        );
        assert_eq!(
            usize::from(host.active_fixed_bank),
            device_receipt.active_bank
        );
        Ok(S14CausalBlockPublishReceipt {
            host: host_receipt,
            device: device_receipt,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14CausalBlockLayerError(String);

impl fmt::Display for S14CausalBlockLayerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for S14CausalBlockLayerError {}

fn layer_error(message: impl Into<String>) -> S14CausalBlockLayerError {
    S14CausalBlockLayerError(message.into())
}

fn map_resource_error(error: S14CausalBlockResourceError) -> S14CausalBlockLayerError {
    layer_error(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s14_dynamic_routed_packing::{
        S14_DYNAMIC_ROUTED_SCALE_BYTES, S14_DYNAMIC_ROUTED_WEIGHT_BYTES,
    };
    use ash::vk::Handle;
    use polaris_s14_runner::{
        router_kind_for_layer, BatchedWholeTokenPosition, Position0CompressorInput,
        WholeTokenCandidate, BATCHED_CAUSAL_WHOLE_TOKEN_MODE, MODEL_REPO, MODEL_REVISION,
    };
    use serde_json::json;
    use std::collections::BTreeSet;

    const TEST_LAYER: u8 = 7;
    const TEST_FILE: &str = "model-test.safetensors";
    const TEST_HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn hidden_binding(block_size: usize, generation: u64) -> S14CausalBlockHiddenBinding {
        let buffer = if generation % 2 == 0 { 12 } else { 11 };
        S14CausalBlockHiddenBinding {
            buffer: vk::Buffer::from_raw(buffer),
            offset: 0,
            bytes: (block_size * S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE * 2) as u64,
            block_size,
            generation,
        }
    }

    #[derive(Debug)]
    struct FakeDeviceFutureOwner {
        receipt: S14CausalBlockDeviceFutureReceipt,
    }

    impl S14CausalBlockDeviceFutureOwner for FakeDeviceFutureOwner {
        fn receipt(&self) -> S14CausalBlockDeviceFutureReceipt {
            self.receipt
        }
    }

    fn device_future(
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
    ) -> S14CausalBlockOwnedDeviceFuture {
        owned_device_future(S14CausalBlockDeviceFutureReceipt {
            base_position,
            block_size: final_hidden.block_size,
            checkpoint_count: final_hidden.block_size,
            storage: S14CausalBlockDeviceCheckpointStorage::PrefixCheckpoints,
            checkpoint_arena: vk::Buffer::from_raw(99),
            checkpoint_arena_offset: 0,
            checkpoint_arena_bytes: final_hidden.bytes * final_hidden.block_size as u64,
            checkpoint_stride_bytes: final_hidden.bytes,
            checkpoint_state_bytes: final_hidden.bytes,
            final_hidden,
            ready_timeline: vk::Semaphore::from_raw(77),
            ready_timeline_value: 1,
        })
    }

    fn owned_device_future(
        receipt: S14CausalBlockDeviceFutureReceipt,
    ) -> S14CausalBlockOwnedDeviceFuture {
        S14CausalBlockOwnedDeviceFuture::new(FakeDeviceFutureOwner { receipt })
    }

    #[derive(Default)]
    struct FakeBackend {
        routes: Vec<RouteDecision>,
        attention_calls: u32,
        materialize_calls: u32,
        grouped_calls: u32,
        reported_serial_token_forward_calls: u32,
        serial_forward_layer: Option<u8>,
        begin_calls: u32,
        seal_calls: u32,
        abort_calls: u32,
        active: bool,
        base_position: u32,
        last_abort_completed_layers: Option<usize>,
        future_output: Option<BatchedWholeTokenOutput>,
        device_receipt_override: Option<S14CausalBlockDeviceFutureReceipt>,
        head_batch_override: Option<u32>,
        post_seal_calls: u32,
        export_calls: u32,
    }

    impl S14CausalBlockLayerBackend for FakeBackend {
        fn run_k_lane_attention_router(
            &mut self,
            input: &S14CausalBlockLayerInput<'_>,
        ) -> Result<S14CausalBlockAttentionRouterOutput, String> {
            self.attention_calls += 1;
            let mut routes = self.routes.clone();
            for route in &mut routes {
                route.layer = input.layer;
                route.kind = router_kind_for_layer(input.layer).unwrap();
            }
            Ok(S14CausalBlockAttentionRouterOutput {
                post_attention_hidden: hidden_binding(
                    input.input_token_ids.len(),
                    input.input_hidden.generation + 1,
                ),
                routes,
                forward_calls: 1,
            })
        }

        fn materialize_union_ranges(
            &mut self,
            bank: S14CausalBlockUnionBankBinding,
            range_plan: &S14CausalBlockLayerRangePlan,
        ) -> Result<S14CausalBlockUnionMaterializeReceipt, String> {
            self.materialize_calls += 1;
            Ok(S14CausalBlockUnionMaterializeReceipt {
                layer: range_plan.layer,
                bank_index: bank.bank_index,
                unique_experts: range_plan.unique_experts,
                physical_ranges: range_plan.physical_ranges,
                uploaded_bytes: range_plan.union_expert_bytes,
                materialize_calls: 1,
                range_evidence: S14CausalBlockRangeEvidenceReceipt {
                    proof_assets: range_plan.physical_ranges,
                    explicit_fetch_lane_plans: 0,
                    mmap_requests: range_plan.physical_ranges as u64,
                    mmap_hits: range_plan.physical_ranges as u64,
                    mmap_misses: 0,
                    sha256_bytes: 0,
                    staging_range_copies: range_plan.physical_ranges,
                    gpu_upload_copy_regions: 1,
                },
            })
        }

        fn run_grouped_moe(
            &mut self,
            _bank: S14CausalBlockUnionBankBinding,
            post_attention_hidden: S14CausalBlockHiddenBinding,
            _routes: &[RouteDecision],
            batch_plan: &LayerCausalBatchPlan,
            _range_plan: &S14CausalBlockLayerRangePlan,
        ) -> Result<S14CausalBlockGroupedMoeOutput, String> {
            self.grouped_calls += 1;
            Ok(S14CausalBlockGroupedMoeOutput {
                output_hidden: hidden_binding(
                    batch_plan.block_size,
                    post_attention_hidden.generation + 1,
                ),
                grouped_submit_calls: 1,
                serial_token_forward_calls: if self.serial_forward_layer == Some(batch_plan.layer) {
                    4
                } else {
                    self.reported_serial_token_forward_calls
                },
                unique_experts: batch_plan.unique_experts,
            })
        }
    }

    impl S14CausalBlockFullDepthBackend for FakeBackend {
        fn begin_full_depth_block(
            &mut self,
            bank: S14CausalBlockUnionBankBinding,
            base_position: u32,
            block_size: usize,
        ) -> Result<S14CausalBlockBeginReceipt, String> {
            if self.active {
                return Err("已有active block".into());
            }
            self.begin_calls += 1;
            self.active = true;
            self.base_position = base_position;
            Ok(S14CausalBlockBeginReceipt {
                begin_calls: 1,
                base_position,
                block_size,
                bank_index: bank.bank_index,
                active: true,
                serial_token_forward_calls: 0,
            })
        }

        fn seal_full_depth_layers(
            &mut self,
            completed_layers: usize,
        ) -> Result<S14CausalBlockSealReceipt, String> {
            if !self.active {
                return Err("没有active block可seal".into());
            }
            self.seal_calls += 1;
            self.active = false;
            Ok(S14CausalBlockSealReceipt {
                seal_calls: 1,
                completed_layers,
                drained: true,
                active: false,
                head_submit_calls: 0,
                checkpoint_commit_calls: 0,
                serial_token_forward_calls: 0,
            })
        }

        fn drain_and_abort_full_depth_block(
            &mut self,
            completed_layers: usize,
        ) -> Result<S14CausalBlockAbortReceipt, String> {
            self.abort_calls += 1;
            self.active = false;
            self.last_abort_completed_layers = Some(completed_layers);
            Ok(S14CausalBlockAbortReceipt {
                abort_calls: 1,
                completed_layers,
                drained: true,
                active: false,
                head_submit_calls: 0,
                checkpoint_commit_calls: 0,
            })
        }
    }

    impl S14CausalBlockCheckpointBackend for FakeBackend {
        fn publish_terminal_source_after_full_depth_seal(
            &mut self,
            completed_layers: usize,
            base_position: u32,
            final_hidden: S14CausalBlockHiddenBinding,
            routes_by_position: &[Vec<RouteDecision>],
        ) -> Result<S14CausalBlockPostSealReceipt, String> {
            if self.active
                || completed_layers != FULL_DEPTH_LAYERS.len()
                || base_position != self.base_position
                || routes_by_position.len() != final_hidden.block_size
            {
                return Err("fake post-seal identity 漂移".into());
            }
            self.post_seal_calls += 1;
            Ok(S14CausalBlockPostSealReceipt {
                hook_calls: 1,
                completed_layers,
                base_position,
                block_size: final_hidden.block_size,
                published_terminal_sources: 1,
                serial_token_forward_calls: 0,
            })
        }

        fn run_batched_final_head_and_export_checkpoints(
            &mut self,
            _completed_layers: usize,
            final_hidden: S14CausalBlockHiddenBinding,
            _routes_by_position: &[Vec<RouteDecision>],
        ) -> Result<S14CausalBlockFinalOutput, String> {
            if self.post_seal_calls != 1 {
                return Err("terminal export 未位于唯一 post-seal producer 之后".into());
            }
            self.export_calls += 1;
            final_hidden
                .validate(self.routes.len())
                .map_err(|error| error.to_string())?;
            let output = self
                .future_output
                .take()
                .ok_or_else(|| "缺少future checkpoint output".to_owned())?;
            let device_future = match self.device_receipt_override.take() {
                Some(receipt) => owned_device_future(receipt),
                None => device_future(self.base_position, final_hidden),
            };
            let head_shape = S14HeadChunkArgmaxShape::production_batched(
                self.head_batch_override
                    .unwrap_or(output.positions.len() as u32),
            )
            .map_err(|error| error.to_string())?;
            let head_results = output
                .positions
                .iter()
                .map(|position| S14HeadArgmaxResult {
                    token_id: position.predicted_token_id,
                    logit: 1.0,
                })
                .collect();
            Ok(S14CausalBlockFinalOutput {
                output,
                head_recording: S14HeadChunkArgmaxRecordingReceipt {
                    shape: head_shape,
                    submitted_chunks: head_shape.chunk_count(),
                    expected_next_token: head_shape.vocab,
                },
                head_results,
                device_future,
                batched_head_submit_calls: 1,
                checkpoint_export_calls: 1,
                serial_token_forward_calls: self.reported_serial_token_forward_calls,
            })
        }
    }

    fn routes(block_size: usize) -> Vec<RouteDecision> {
        let patterns = [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 7, 8, 9, 10],
            [1, 3, 7, 11, 12, 13],
            [2, 3, 8, 11, 14, 15],
        ];
        (0..block_size)
            .map(|offset| patterns[offset % patterns.len()])
            .map(|expert_ids| RouteDecision {
                layer: TEST_LAYER,
                kind: router_kind_for_layer(TEST_LAYER).unwrap(),
                expert_ids: expert_ids.to_vec(),
                weights: vec![0.25; EXPERTS_PER_TOKEN],
            })
            .collect()
    }

    fn catalog(routes: &[RouteDecision]) -> FullDepthExpertCatalog {
        let expert_ids: BTreeSet<u16> = routes
            .iter()
            .flat_map(|route| route.expert_ids.iter().copied())
            .collect();
        let mut ordinal = 0u64;
        let file_bytes = 100_000_000_000u64;
        let mut layers = serde_json::Map::new();
        for &layer in &FULL_DEPTH_LAYERS {
            let mut experts = serde_json::Map::new();
            for &expert_id in &expert_ids {
                let mut ranges = Vec::with_capacity(6);
                for projection in ["w1", "w2", "w3"] {
                    for (part, dtype, shape, bytes) in [
                        (
                            "weight",
                            "I8",
                            vec![2048u64, 2048],
                            S14_DYNAMIC_ROUTED_WEIGHT_BYTES,
                        ),
                        (
                            "scale",
                            "F8_E8M0",
                            vec![2048u64, 128],
                            S14_DYNAMIC_ROUTED_SCALE_BYTES,
                        ),
                    ] {
                        let start = ordinal * S14_DYNAMIC_ROUTED_WEIGHT_BYTES;
                        let end = start + bytes - 1;
                        ranges.push(json!({
                            "tensor": format!(
                                "layers.{layer}.ffn.experts.{expert_id}.{projection}.{part}"
                            ),
                            "kind": "routed_expert",
                            "layer": layer,
                            "file": TEST_FILE,
                            "file_bytes": file_bytes,
                            "header_tensor_table_sha256": TEST_HASH,
                            "start": start,
                            "end": end,
                            "bytes": bytes,
                            "dtype": dtype,
                            "shape": shape,
                            "range_key": format!("{TEST_FILE}:{start}-{end}"),
                            "expert_id": expert_id,
                        }));
                        ordinal += 1;
                    }
                }
                experts.insert(expert_id.to_string(), json!(ranges));
            }
            layers.insert(layer.to_string(), json!({ "experts": experts }));
        }
        FullDepthExpertCatalog::from_json_str(
            &json!({
                "format": crate::s14_dynamic_routed_page_plan::FULL_DEPTH_EXPERT_CATALOG_FORMAT,
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "profile": {
                    "id": "fulldepth43_native_top6",
                    "repo": MODEL_REPO,
                    "revision": MODEL_REVISION,
                    "layers": FULL_DEPTH_LAYERS.to_vec(),
                    "top_k": EXPERTS_PER_TOKEN,
                },
                "headers": {
                    "files": {
                        TEST_FILE: {
                            "file_bytes": file_bytes,
                            "tensor_table_sha256": TEST_HASH,
                        }
                    }
                },
                "layers": layers
            })
            .to_string(),
        )
        .unwrap()
    }

    fn stage_fixture_layer(candidate: &mut WholeTokenCandidate, layer: u8, position: u32) {
        let window_kv = vec![0x3f80 + layer as u16 + position as u16; 512];
        let bias = position as f32 * 100.0;
        let ratio = candidate.staged_native_mut().kv[layer as usize].compress_ratio;
        let compressor = match ratio {
            0 => Position0CompressorInput::None,
            4 if position % 4 == 3 => Position0CompressorInput::Ratio4Boundary {
                main_kv: &vec![layer as f32 + 1.0 + bias; 1024],
                main_score: &vec![layer as f32 + 2.0 + bias; 1024],
                indexer_kv: &vec![layer as f32 + 3.0 + bias; 256],
                indexer_score: &vec![layer as f32 + 4.0 + bias; 256],
                main_compressed_kv_bf16: &vec![0x4100 + layer as u16 + position as u16; 512],
                indexer_compressed_kv_bf16: &vec![0x4200 + layer as u16 + position as u16; 128],
            },
            4 => Position0CompressorInput::Ratio4 {
                main_kv: &vec![layer as f32 + 1.0 + bias; 1024],
                main_score: &vec![layer as f32 + 2.0 + bias; 1024],
                indexer_kv: &vec![layer as f32 + 3.0 + bias; 256],
                indexer_score: &vec![layer as f32 + 4.0 + bias; 256],
            },
            128 => Position0CompressorInput::Ratio128 {
                main_kv: &vec![layer as f32 + 1.0 + bias; 512],
                main_score: &vec![layer as f32 + 2.0 + bias; 512],
            },
            _ => unreachable!(),
        };
        candidate
            .stage_layer_state(layer, &window_kv, compressor)
            .unwrap();
        candidate.complete_layer(layer).unwrap();
    }

    fn fixture_routes_by_position(base_routes: &[RouteDecision]) -> Vec<Vec<RouteDecision>> {
        base_routes
            .iter()
            .map(|base_route| {
                FULL_DEPTH_LAYERS
                    .iter()
                    .map(|&layer| {
                        let mut route = base_route.clone();
                        route.layer = layer;
                        route.kind = router_kind_for_layer(layer).unwrap();
                        route
                    })
                    .collect()
            })
            .collect()
    }

    fn fixture_future_output(
        authoritative: &DecoderStateV1,
        draft_token_ids: &[u32],
        predicted_token_ids: &[u32],
        base_routes: &[RouteDecision],
    ) -> BatchedWholeTokenOutput {
        let routes_by_position = fixture_routes_by_position(base_routes);
        let mut private = authoritative.clone();
        let positions = draft_token_ids
            .iter()
            .zip(predicted_token_ids)
            .zip(routes_by_position)
            .map(|((&teacher_force, &predicted_token_id), routes)| {
                let position = private.position;
                let mut candidate = private
                    .begin_token(private.commit_epoch, position, private.input_token_id)
                    .unwrap();
                for &layer in &FULL_DEPTH_LAYERS {
                    stage_fixture_layer(&mut candidate, layer, position);
                }
                candidate.stage_hc_state(&vec![0x3f00; 4 * 4096]).unwrap();
                candidate.complete_final(predicted_token_id).unwrap();
                candidate
                    .commit_with_next_input(&mut private, Some(teacher_force))
                    .unwrap();
                BatchedWholeTokenPosition {
                    predicted_token_id,
                    routes,
                    checkpoint: private.clone(),
                }
            })
            .collect();
        BatchedWholeTokenOutput {
            mode: BATCHED_CAUSAL_WHOLE_TOKEN_MODE.into(),
            forward_calls: 1,
            positions,
        }
    }

    #[test]
    fn k4_layer_is_one_attention_one_union_one_grouped_submit_and_rejects_serial_forward() {
        let routes = routes(4);
        let catalog = catalog(&routes);
        let union_plan = S14CausalBlockUnionBankPlan::build(4).unwrap();
        let bank = S14CausalBlockUnionBankBinding {
            bank_index: 0,
            buffer: vk::Buffer::from_raw(1),
            allocated_bank_bytes: union_plan.allocated_bank_bytes,
        };
        let token_ids = [0, 5, 223, 939];
        let hidden = hidden_binding(4, 0);
        let input = || S14CausalBlockLayerInput {
            base_position: 0,
            layer: TEST_LAYER,
            input_token_ids: &token_ids,
            input_hidden: hidden,
            source: MaterializedTokenSource::SpeculativeDraft,
        };

        let mut backend = FakeBackend {
            routes: routes.clone(),
            ..FakeBackend::default()
        };
        let receipt =
            execute_causal_block_layer(&union_plan, bank, &catalog, input(), &mut backend).unwrap();
        assert_eq!(
            (
                backend.attention_calls,
                backend.materialize_calls,
                backend.grouped_calls
            ),
            (1, 1, 1)
        );
        assert_eq!(receipt.attention_router_forward_calls, 1);
        assert_eq!(receipt.union_range_materialize_calls, 1);
        assert_eq!(receipt.grouped_moe_submit_calls, 1);
        assert_eq!(receipt.serial_token_forward_calls, 0);
        assert_eq!(receipt.output_hidden, hidden_binding(4, 2));
        assert_eq!(receipt.physical_ranges, receipt.unique_experts * 6);
        assert_eq!(
            receipt.union_expert_bytes,
            15 * polaris_s14_runner::EXPERT_PAGE_BYTES
        );
        assert!(!receipt.head_commit_ready);

        let mut serial_backend = FakeBackend {
            routes,
            reported_serial_token_forward_calls: 4,
            ..FakeBackend::default()
        };
        let error =
            execute_causal_block_layer(&union_plan, bank, &catalog, input(), &mut serial_backend)
                .unwrap_err();
        assert!(error.to_string().contains("零串行token forward"));
        assert_eq!(
            (
                serial_backend.attention_calls,
                serial_backend.materialize_calls,
                serial_backend.grouped_calls,
            ),
            (1, 1, 1)
        );
    }

    #[test]
    fn k4_k8_full_depth_run_43_block_major_layers_and_abort_is_fail_closed() {
        for block_size in [4, 8] {
            let routes = routes(block_size);
            let catalog = catalog(&routes);
            let union_plan = S14CausalBlockUnionBankPlan::build(block_size).unwrap();
            let bank = S14CausalBlockUnionBankBinding {
                bank_index: 1,
                buffer: vk::Buffer::from_raw(2),
                allocated_bank_bytes: union_plan.allocated_bank_bytes,
            };
            let token_ids = (0..block_size)
                .map(|token| token as u32)
                .collect::<Vec<_>>();
            let hidden = hidden_binding(block_size, 0);
            let mut backend = FakeBackend {
                routes,
                ..FakeBackend::default()
            };
            let receipt = execute_causal_block_full_depth(
                &union_plan,
                bank,
                &catalog,
                0,
                &token_ids,
                hidden,
                MaterializedTokenSource::SpeculativeDraft,
                &mut backend,
            )
            .unwrap();
            assert_eq!(receipt.block_size, block_size);
            assert_eq!(receipt.completed_layers, FULL_DEPTH_LAYERS.len());
            assert_eq!(
                receipt
                    .layer_summaries
                    .iter()
                    .map(|layer| layer.layer)
                    .collect::<Vec<_>>(),
                FULL_DEPTH_LAYERS
            );
            assert_eq!(receipt.attention_router_forward_calls, 43);
            assert_eq!(receipt.union_range_materialize_calls, 43);
            assert_eq!(receipt.grouped_moe_submit_calls, 43);
            assert_eq!(receipt.serial_token_forward_calls, 0);
            assert_eq!(receipt.final_hidden, hidden_binding(block_size, 86));
            assert!(receipt.lifecycle_sealed);
            assert!(!receipt.head_ready);
            assert!(!receipt.checkpoint_commit_ready);
            assert_eq!(
                (
                    backend.attention_calls,
                    backend.materialize_calls,
                    backend.grouped_calls,
                    backend.begin_calls,
                    backend.seal_calls,
                    backend.abort_calls,
                ),
                (43, 43, 43, 1, 1, 0)
            );
            assert!(!backend.active);
        }

        let routes = routes(4);
        let catalog = catalog(&routes);
        let union_plan = S14CausalBlockUnionBankPlan::build(4).unwrap();
        let bank = S14CausalBlockUnionBankBinding {
            bank_index: 1,
            buffer: vk::Buffer::from_raw(2),
            allocated_bank_bytes: union_plan.allocated_bank_bytes,
        };
        let token_ids = [0, 5, 223, 939];
        let hidden = hidden_binding(4, 0);
        let mut serial_backend = FakeBackend {
            routes,
            serial_forward_layer: Some(7),
            ..FakeBackend::default()
        };
        let error = execute_causal_block_full_depth(
            &union_plan,
            bank,
            &catalog,
            0,
            &token_ids,
            hidden,
            MaterializedTokenSource::SpeculativeDraft,
            &mut serial_backend,
        )
        .unwrap_err();
        assert!(error.to_string().contains("零串行token forward"));
        assert!(error.to_string().contains("已drain/abort"));
        assert_eq!(serial_backend.last_abort_completed_layers, Some(7));
        assert_eq!(
            (
                serial_backend.attention_calls,
                serial_backend.materialize_calls,
                serial_backend.grouped_calls,
                serial_backend.begin_calls,
                serial_backend.seal_calls,
                serial_backend.abort_calls,
            ),
            (8, 8, 8, 1, 0, 1)
        );
        assert!(!serial_backend.active);
    }

    #[test]
    fn k4_k8_future_checkpoints_seal_without_commit_and_invalid_chain_aborts() {
        let routes4 = routes(4);
        let catalog4 = catalog(&routes4);
        let union_plan = S14CausalBlockUnionBankPlan::build(4).unwrap();
        let bank = S14CausalBlockUnionBankBinding {
            bank_index: 0,
            buffer: vk::Buffer::from_raw(3),
            allocated_bank_bytes: union_plan.allocated_bank_bytes,
        };
        let authoritative = DecoderStateV1::new(32, 0).unwrap();
        let authoritative_before = authoritative.clone();
        let draft = vec![5, 223, 939, 21];
        let predicted = vec![5, 222, 939, 21];
        let input_tokens = vec![0, 5, 223, 939];
        let hidden = hidden_binding(4, 0);
        let mut backend = FakeBackend {
            routes: routes4.clone(),
            future_output: Some(fixture_future_output(
                &authoritative,
                &draft,
                &predicted,
                &routes4,
            )),
            ..FakeBackend::default()
        };
        let sealed = execute_causal_block_full_depth_with_checkpoints(
            &union_plan,
            bank,
            &catalog4,
            &authoritative,
            &draft,
            &input_tokens,
            hidden,
            MaterializedTokenSource::SpeculativeDraft,
            &mut backend,
        )
        .unwrap();
        assert_eq!(authoritative, authoritative_before);
        assert_eq!(sealed.layers.completed_layers, 43);
        assert!(sealed.layers.head_ready);
        assert!(!sealed.layers.checkpoint_commit_ready);
        assert_eq!(sealed.decision().accepted_prefix, [5]);
        assert_eq!(sealed.decision().fallback_token_id, Some(222));
        assert_eq!(sealed.decision().rejected_draft_suffix, [223, 939, 21]);
        let selected = sealed.selected_prefix().unwrap();
        assert_eq!(selected.accepted_tokens(), 2);
        assert_eq!(selected.checkpoint_index(), 1);
        assert_eq!(selected.checkpoint().position, 2);
        assert_eq!(selected.checkpoint().commit_epoch, 2);
        assert_eq!(selected.checkpoint().active_fixed_bank, 0);
        drop(selected);
        assert_eq!(
            sealed.device_receipt(),
            S14CausalBlockDeviceFutureReceipt {
                base_position: 0,
                block_size: 4,
                checkpoint_count: 4,
                storage: S14CausalBlockDeviceCheckpointStorage::PrefixCheckpoints,
                checkpoint_arena: vk::Buffer::from_raw(99),
                checkpoint_arena_offset: 0,
                checkpoint_arena_bytes: hidden_binding(4, 86).bytes * 4,
                checkpoint_stride_bytes: hidden_binding(4, 86).bytes,
                checkpoint_state_bytes: hidden_binding(4, 86).bytes,
                final_hidden: hidden_binding(4, 86),
                ready_timeline: vk::Semaphore::from_raw(77),
                ready_timeline_value: 1,
            }
        );
        assert_eq!(backend.export_calls, 1);
        assert_eq!(backend.post_seal_calls, 1);
        assert_eq!(backend.abort_calls, 0);
        let rollback = sealed.rollback(&authoritative).unwrap();
        assert_eq!(rollback.position, authoritative_before.position);
        assert_eq!(rollback.commit_epoch, authoritative_before.commit_epoch);
        assert_eq!(authoritative, authoritative_before);

        let routes8 = routes(8);
        let catalog8 = catalog(&routes8);
        let union_plan8 = S14CausalBlockUnionBankPlan::build(8).unwrap();
        let bank8 = S14CausalBlockUnionBankBinding {
            allocated_bank_bytes: union_plan8.allocated_bank_bytes,
            ..bank
        };
        let draft8 = vec![5, 223, 939, 21, 695, 553, 1266, 16179];
        let mut input_tokens8 = vec![0];
        input_tokens8.extend_from_slice(&draft8[..7]);
        let hidden8 = hidden_binding(8, 0);
        let mut backend8 = FakeBackend {
            routes: routes8.clone(),
            future_output: Some(fixture_future_output(
                &authoritative,
                &draft8,
                &draft8,
                &routes8,
            )),
            ..FakeBackend::default()
        };
        let sealed8 = execute_causal_block_full_depth_with_checkpoints(
            &union_plan8,
            bank8,
            &catalog8,
            &authoritative,
            &draft8,
            &input_tokens8,
            hidden8,
            MaterializedTokenSource::SpeculativeDraft,
            &mut backend8,
        )
        .unwrap();
        assert_eq!(sealed8.layers.block_size, 8);
        assert!(sealed8.layers.head_ready);
        assert_eq!(sealed8.decision().accepted_prefix, draft8);
        assert_eq!(sealed8.decision().fallback_token_id, None);
        let selected8 = sealed8.selected_prefix().unwrap();
        assert_eq!(selected8.accepted_tokens(), 8);
        assert_eq!(selected8.checkpoint_index(), 7);
        assert_eq!(selected8.checkpoint().active_fixed_bank, 0);
        drop(selected8);
        assert_eq!(authoritative, authoritative_before);
        sealed8.rollback(&authoritative).unwrap();

        let mut invalid_head_backend = FakeBackend {
            routes: routes4.clone(),
            future_output: Some(fixture_future_output(
                &authoritative,
                &draft,
                &predicted,
                &routes4,
            )),
            head_batch_override: Some(1),
            ..FakeBackend::default()
        };
        let error = execute_causal_block_full_depth_with_checkpoints(
            &union_plan,
            bank,
            &catalog4,
            &authoritative,
            &draft,
            &input_tokens,
            hidden,
            MaterializedTokenSource::SpeculativeDraft,
            &mut invalid_head_backend,
        )
        .unwrap_err();
        assert!(error.to_string().contains("batched lm-head"));
        assert!(error.to_string().contains("已drain/abort"));
        assert_eq!(invalid_head_backend.export_calls, 1);
        assert_eq!(invalid_head_backend.abort_calls, 1);
        assert_eq!(authoritative, authoritative_before);

        let invalid_device_receipt = S14CausalBlockDeviceFutureReceipt {
            base_position: authoritative.position,
            block_size: 4,
            checkpoint_count: 3,
            storage: S14CausalBlockDeviceCheckpointStorage::LosslessDeltaJournal,
            checkpoint_arena: vk::Buffer::from_raw(100),
            checkpoint_arena_offset: 0,
            checkpoint_arena_bytes: 4096,
            checkpoint_stride_bytes: 1024,
            checkpoint_state_bytes: 1024,
            final_hidden: hidden_binding(4, 86),
            ready_timeline: vk::Semaphore::from_raw(78),
            ready_timeline_value: 2,
        };
        let mut invalid_device_backend = FakeBackend {
            routes: routes4.clone(),
            future_output: Some(fixture_future_output(
                &authoritative,
                &draft,
                &predicted,
                &routes4,
            )),
            device_receipt_override: Some(invalid_device_receipt),
            ..FakeBackend::default()
        };
        let error = execute_causal_block_full_depth_with_checkpoints(
            &union_plan,
            bank,
            &catalog4,
            &authoritative,
            &draft,
            &input_tokens,
            hidden,
            MaterializedTokenSource::SpeculativeDraft,
            &mut invalid_device_backend,
        )
        .unwrap_err();
        assert!(error.to_string().contains("device future owner/receipt"));
        assert!(error.to_string().contains("已drain/abort"));
        assert_eq!(invalid_device_backend.export_calls, 1);
        assert_eq!(invalid_device_backend.abort_calls, 1);
        assert_eq!(authoritative, authoritative_before);

        let mut invalid_output =
            fixture_future_output(&authoritative, &draft, &predicted, &routes4);
        invalid_output.positions[2].checkpoint.input_token_id = 17;
        let mut invalid_backend = FakeBackend {
            routes: routes4,
            future_output: Some(invalid_output),
            ..FakeBackend::default()
        };
        let error = execute_causal_block_full_depth_with_checkpoints(
            &union_plan,
            bank,
            &catalog4,
            &authoritative,
            &draft,
            &input_tokens,
            hidden,
            MaterializedTokenSource::SpeculativeDraft,
            &mut invalid_backend,
        )
        .unwrap_err();
        assert!(error.to_string().contains("checkpoint chain 非法"));
        assert!(error.to_string().contains("已drain/abort"));
        assert_eq!(invalid_backend.export_calls, 1);
        assert_eq!(invalid_backend.abort_calls, 1);
        assert_eq!(invalid_backend.last_abort_completed_layers, Some(43));
        assert_eq!(authoritative, authoritative_before);
    }
}
