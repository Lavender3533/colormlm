//! 43 层 K-lane production producer 与 batched terminal recorder 的一次性交接层。
//!
//! 本模块只传递真实 owner 保活的 device buffer/timeline；不会从裸 Vulkan handle 构造
//! `StorageBufferSlice`，也不会合成 candidate checkpoint、normalized head row 或 host output。
//! host candidate batch 由 producer 保持未完成，只有 GPU head 回读 K 个 token 后才一次性
//! 完成并导出 host checkpoint，避免 predicted token 的前置循环依赖。producer 未发布、
//! 身份漂移或资源缺失时一律消费并拒绝当前 source，避免陈旧 source 被下一 block 误用。
//! backend abort 会清掉尚未消费的 source，形成显式 rollback。

use crate::{
    compute::StorageBufferSlice,
    s14_causal_block_layer::{
        S14CausalBlockFinalOutput, S14CausalBlockHiddenBinding,
        S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE,
    },
    s14_causal_block_terminal::{
        S14CausalBlockBatchedTerminalRecorder, S14CausalBlockCheckpointSource,
        S14CausalBlockTerminalGpuExport, S14CausalBlockTerminalInputBinding,
    },
    s14_causal_block_vulkan_backend::S14CausalBlockVulkanTerminalRecorder,
    s14_head_chunk_argmax::{S14HeadArgmaxResult, S14HeadChunkArgmaxShape},
    s14_position0_hybrid_upload::S14Position0HeadChunkReceipt,
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    BatchedWholeTokenOutput, RouteDecision, BATCHED_CAUSAL_WHOLE_TOKEN_MODE, FULL_DEPTH_LAYERS,
};
use std::{
    fmt,
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14CausalBlockTerminalResource {
    FinalHidden,
    NormalizedHeadRows,
    CandidateCheckpoint(usize),
    HeadBank(usize),
}

/// 43 层 K-lane recorder 对 terminal 输入资源的真实 owner 边界。
///
/// 实现者必须拥有（或以等价强所有权保活）返回的 `GpuBuffer` 和 producer timeline，直到
/// 本 owner 被 drop。provider 用 `Arc` 持有它，terminal submit/timeline wait 返回前不会释放。
pub trait S14CausalBlockTerminalResourceOwner: fmt::Debug {
    fn buffer(&self, resource: S14CausalBlockTerminalResource) -> Option<&GpuBuffer>;
    fn producer_timeline(&self) -> vk::Semaphore;

    /// 32个逻辑 chunk 只能流经同一 paged arena 的两个真实 bank。
    fn paged_weight_arena(&self) -> Option<&Arc<S14Position0PagedWeightArena>> {
        None
    }

    /// 只录制当前 chunk 的 staging→`bank=chunk%2` copy，不读取 payload、不 submit。
    unsafe fn record_next_head_chunk_copy(
        &self,
        _ctx: &VulkanContext,
        _command: vk::CommandBuffer,
        _chunk: u32,
    ) -> Result<S14Position0HeadChunkReceipt, String> {
        Err("terminal resource owner 未实现 paged head copy recorder".into())
    }

    /// timeline 已确认 staging bank 可复用后，必须通过 verified store 完成
    /// proof/SHA/mmap，并把当前 chunk payload 写入录制 copy 所引用的 staging bank。
    fn stage_recorded_head_chunk(
        &self,
        _receipt: S14Position0HeadChunkReceipt,
        _timeline_bank: usize,
    ) -> Result<S14Position0HeadChunkReceipt, String> {
        Err("terminal resource owner 未实现 verified paged head staging".into())
    }

    /// 仅在 transfer/compute 已排空后回滚 uploader 游标。
    fn abort_head_stream_after_drain(&self) {}

    /// terminal recorder 已等待 producer timeline 且完成 GPU batched head 后调用。
    /// concrete owner 必须在这里校验同一 timeline 覆盖的 HC/norm sticky status；
    /// 校验成功前不得消费 host snapshot finalizer。
    fn validate_after_producer_timeline(&self, expected_value: u64) -> Result<(), String>;
}

/// producer 已完成43层状态写入、但尚未绑定最终 token 的 K 份 host candidate。
///
/// 该对象必须独占 candidate，且只能在 terminal 已回读 GPU head 后被消费。实现者用
/// `head_results` 完成每份 candidate 的 final token/host checkpoint，并返回同一次 batched
/// forward 的完整 output；adapter 会再次校验 token、route、position 与 checkpoint 同源。
pub trait S14CausalBlockHostCandidateFinalizer: fmt::Debug {
    fn block_size(&self) -> usize;
    fn base_position(&self) -> u32;

    fn complete_after_gpu_head(
        self: Box<Self>,
        head_results: &[S14HeadArgmaxResult],
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<BatchedWholeTokenOutput>;
}

/// producer seal 43 层后发布的一次性真实 source。offset 只描述 owner 内部布局；buffer
/// handle 始终从 `resources` 借出，禁止调用方用裸 handle 绕过生命周期。
#[derive(Debug)]
pub struct S14CausalBlockTerminalProductionSource {
    pub completed_layers: usize,
    pub base_position: u32,
    pub final_hidden: S14CausalBlockHiddenBinding,
    pub normalized_head_rows_offset: u64,
    pub checkpoint_offsets: Vec<u64>,
    pub head_chunk_count: usize,
    pub producer_timeline_value: u64,
    pub routes_by_position: Vec<Vec<RouteDecision>>,
    pub host_candidates: Box<dyn S14CausalBlockHostCandidateFinalizer>,
    pub resources: Arc<dyn S14CausalBlockTerminalResourceOwner>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14CausalBlockTerminalProviderTelemetry {
    pub published: u64,
    pub take_attempts: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub rollbacks: u64,
    pub pending: bool,
}

#[derive(Debug, Default)]
struct ProviderState {
    pending: Option<S14CausalBlockTerminalProductionSource>,
    telemetry: S14CausalBlockTerminalProviderTelemetry,
}

type SharedProviderState = Arc<Mutex<ProviderState>>;

/// 交给43层 K-lane recorder 的发布端。未消费 source 不允许被覆盖。
#[derive(Clone)]
pub struct S14CausalBlockTerminalProductionPublisher {
    shared: SharedProviderState,
}

impl fmt::Debug for S14CausalBlockTerminalProductionPublisher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockTerminalProductionPublisher")
            .field("telemetry", &self.telemetry().ok())
            .finish()
    }
}

impl S14CausalBlockTerminalProductionPublisher {
    pub fn publish(&self, source: S14CausalBlockTerminalProductionSource) -> Result<(), String> {
        if let Err(error) = validate_source_envelope(&source) {
            self.note_rejection()?;
            return Err(error.to_string());
        }
        let mut state = lock_provider(&self.shared)?;
        if state.pending.is_some() {
            state.telemetry.rejected = state.telemetry.rejected.saturating_add(1);
            return Err("K-lane terminal production source 尚未消费，禁止覆盖".into());
        }
        state.pending = Some(source);
        state.telemetry.published = state.telemetry.published.saturating_add(1);
        state.telemetry.pending = true;
        Ok(())
    }

    /// producer/backend abort 时丢弃尚未提交给 terminal 的 candidate source。
    pub fn rollback_pending(&self) -> Result<bool, String> {
        rollback_pending(&self.shared)
    }

    pub fn telemetry(&self) -> Result<S14CausalBlockTerminalProviderTelemetry, String> {
        telemetry(&self.shared)
    }

    fn note_rejection(&self) -> Result<(), String> {
        let mut state = lock_provider(&self.shared)?;
        state.telemetry.rejected = state.telemetry.rejected.saturating_add(1);
        Ok(())
    }
}

/// terminal adapter 独占的消费端。每次 backend terminal 调用至多消费一份 source。
pub struct S14CausalBlockTerminalProductionProvider {
    shared: SharedProviderState,
}

impl fmt::Debug for S14CausalBlockTerminalProductionProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockTerminalProductionProvider")
            .field("telemetry", &self.telemetry().ok())
            .finish()
    }
}

impl S14CausalBlockTerminalProductionProvider {
    pub fn telemetry(&self) -> Result<S14CausalBlockTerminalProviderTelemetry, String> {
        telemetry(&self.shared)
    }

    fn take_matching(
        &self,
        completed_layers: usize,
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<S14CausalBlockTerminalProductionSource, String> {
        let source = {
            let mut state = lock_provider(&self.shared)?;
            state.telemetry.take_attempts = state.telemetry.take_attempts.saturating_add(1);
            let Some(source) = state.pending.take() else {
                state.telemetry.rejected = state.telemetry.rejected.saturating_add(1);
                state.telemetry.pending = false;
                return Err(
                    "43层 K-lane recorder 未发布真实 candidate-state/normalized-head source".into(),
                );
            };
            state.telemetry.pending = false;
            source
        };

        if let Err(error) = validate_expected_source(
            &source,
            completed_layers,
            base_position,
            final_hidden,
            routes_by_position,
        ) {
            let mut state = lock_provider(&self.shared)?;
            state.telemetry.rejected = state.telemetry.rejected.saturating_add(1);
            return Err(error.to_string());
        }
        let mut state = lock_provider(&self.shared)?;
        state.telemetry.accepted = state.telemetry.accepted.saturating_add(1);
        Ok(source)
    }

    fn rollback_pending(&self) -> Result<bool, String> {
        rollback_pending(&self.shared)
    }
}

/// 创建单 producer / 单 consumer 的一次性交接通道。provider 应直接移入 terminal adapter；
/// publisher 应移入43层 K-lane recorder，在完整 seal 后发布。
pub fn s14_causal_block_terminal_production_channel() -> (
    S14CausalBlockTerminalProductionPublisher,
    S14CausalBlockTerminalProductionProvider,
) {
    let shared = Arc::new(Mutex::new(ProviderState::default()));
    (
        S14CausalBlockTerminalProductionPublisher {
            shared: Arc::clone(&shared),
        },
        S14CausalBlockTerminalProductionProvider { shared },
    )
}

/// backend terminal trait 的 production 实现。recorder 持有持久 Vulkan command/workspace；
/// provider 只在本次调用作用域借出真实资源，先完成 GPU export，再绑定同源 host output。
pub struct S14CausalBlockTerminalProductionAdapter {
    recorder: S14CausalBlockBatchedTerminalRecorder,
    provider: S14CausalBlockTerminalProductionProvider,
}

impl fmt::Debug for S14CausalBlockTerminalProductionAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockTerminalProductionAdapter")
            .field("recorder", &self.recorder)
            .field("provider", &self.provider)
            .finish()
    }
}

impl S14CausalBlockTerminalProductionAdapter {
    pub fn new(
        recorder: S14CausalBlockBatchedTerminalRecorder,
        provider: S14CausalBlockTerminalProductionProvider,
    ) -> Self {
        Self { recorder, provider }
    }

    pub fn telemetry(&self) -> Result<S14CausalBlockTerminalProviderTelemetry, String> {
        self.provider.telemetry()
    }

    fn record_source(
        &mut self,
        source: &S14CausalBlockTerminalProductionSource,
    ) -> Result<S14CausalBlockTerminalGpuExport> {
        let normalized_head_rows = StorageBufferSlice {
            buffer: required_buffer(
                source.resources.as_ref(),
                S14CausalBlockTerminalResource::NormalizedHeadRows,
            )?,
            offset: source.normalized_head_rows_offset,
        };
        let checkpoints = source
            .checkpoint_offsets
            .iter()
            .enumerate()
            .map(|(lane, &offset)| {
                Ok(S14CausalBlockCheckpointSource {
                    state: StorageBufferSlice {
                        buffer: required_buffer(
                            source.resources.as_ref(),
                            S14CausalBlockTerminalResource::CandidateCheckpoint(lane),
                        )?,
                        offset,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let input = S14CausalBlockTerminalInputBinding {
            base_position: source.base_position,
            final_hidden: source.final_hidden,
            normalized_head_rows,
            checkpoints: &checkpoints,
            head_stream: source.resources.as_ref(),
            producer_timeline: source.resources.producer_timeline(),
            producer_timeline_value: source.producer_timeline_value,
        };
        let export = self.recorder.record_gpu_export(input)?;
        source
            .resources
            .validate_after_producer_timeline(source.producer_timeline_value)
            .map_err(anyhow::Error::msg)
            .context("validate production terminal HC/norm after producer timeline")?;
        Ok(export)
    }
}

impl S14CausalBlockVulkanTerminalRecorder for S14CausalBlockTerminalProductionAdapter {
    fn record_batched_terminal_head_and_checkpoints(
        &mut self,
        completed_layers: usize,
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<S14CausalBlockFinalOutput, String> {
        let source = self.provider.take_matching(
            completed_layers,
            base_position,
            final_hidden,
            routes_by_position,
        )?;
        let export = self
            .record_source(&source)
            .map_err(|error| format!("K-lane terminal GPU export 失败: {error:#}"))?;
        let S14CausalBlockTerminalProductionSource {
            base_position,
            final_hidden,
            routes_by_position,
            host_candidates,
            ..
        } = source;
        let host_output = complete_host_candidates_after_gpu_head(
            host_candidates,
            base_position,
            final_hidden.block_size,
            &routes_by_position,
            &export.head_results,
        )
        .map_err(|error| format!("K-lane terminal host candidate 完成失败: {error:#}"))?;
        export
            .bind_host_output(host_output)
            .map_err(|error| format!("K-lane terminal host output 绑定失败: {error:#}"))
    }

    fn drain_and_abort_batched_terminal(&mut self, _completed_layers: usize) -> Result<(), String> {
        self.provider.rollback_pending()?;
        Ok(())
    }
}

fn validate_expected_source(
    source: &S14CausalBlockTerminalProductionSource,
    completed_layers: usize,
    base_position: u32,
    final_hidden: S14CausalBlockHiddenBinding,
    routes_by_position: &[Vec<RouteDecision>],
) -> Result<()> {
    validate_source_envelope(source)?;
    if source.completed_layers != completed_layers
        || source.base_position != base_position
        || source.final_hidden != final_hidden
        || source.routes_by_position.as_slice() != routes_by_position
    {
        bail!("K-lane terminal production source 与 sealed 43层 block 身份漂移");
    }
    Ok(())
}

fn validate_source_envelope(source: &S14CausalBlockTerminalProductionSource) -> Result<()> {
    let block_size = source.final_hidden.block_size;
    let shape = S14HeadChunkArgmaxShape::production_batched(block_size as u32)?;
    let expected_hidden_bytes = (block_size as u64)
        .checked_mul(S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64)
        .and_then(|elements| elements.checked_mul(2))
        .context("K-lane terminal final hidden bytes overflow")?;
    if source.completed_layers != FULL_DEPTH_LAYERS.len()
        || !matches!(block_size, 4 | 8)
        || source.final_hidden.buffer == vk::Buffer::null()
        || source.final_hidden.bytes != expected_hidden_bytes
        || source.checkpoint_offsets.len() != block_size
        || source.head_chunk_count != shape.chunk_count() as usize
        || source.routes_by_position.len() != block_size
        || source
            .routes_by_position
            .iter()
            .any(|routes| routes.len() != FULL_DEPTH_LAYERS.len())
        || source.resources.producer_timeline() == vk::Semaphore::null()
        || source.producer_timeline_value == 0
    {
        bail!("K-lane terminal production source K/43层/resource/timeline 不完整");
    }
    source
        .base_position
        .checked_add(block_size as u32)
        .context("K-lane terminal source position overflow")?;

    let final_hidden_buffer = required_buffer(
        source.resources.as_ref(),
        S14CausalBlockTerminalResource::FinalHidden,
    )?;
    if final_hidden_buffer.handle() != source.final_hidden.buffer {
        bail!("K-lane terminal final hidden handle 未与 resource owner 强绑定");
    }
    validate_owned_range(
        final_hidden_buffer,
        source.final_hidden.offset,
        source.final_hidden.bytes,
        "final hidden",
    )?;
    validate_owned_range(
        required_buffer(
            source.resources.as_ref(),
            S14CausalBlockTerminalResource::NormalizedHeadRows,
        )?,
        source.normalized_head_rows_offset,
        shape.normalized_input_bytes()?,
        "normalized head rows",
    )?;
    for (lane, &offset) in source.checkpoint_offsets.iter().enumerate() {
        validate_owned_range(
            required_buffer(
                source.resources.as_ref(),
                S14CausalBlockTerminalResource::CandidateCheckpoint(lane),
            )?,
            offset,
            1,
            "candidate checkpoint",
        )?;
    }
    let arena = source
        .resources
        .paged_weight_arena()
        .context("K-lane terminal resource owner 缺少 paged arena")?;
    for bank in 0..2 {
        let expected = arena.head_chunk(bank)?;
        let observed = required_buffer(
            source.resources.as_ref(),
            S14CausalBlockTerminalResource::HeadBank(bank),
        )?;
        if expected.handle() != observed.handle() {
            bail!("K-lane terminal head bank 未与 paged arena 强绑定");
        }
        validate_owned_range(observed, 0, shape.max_chunk_weight_bytes()?, "head bank")?;
    }

    if source.host_candidates.block_size() != block_size
        || source.host_candidates.base_position() != source.base_position
    {
        bail!("K-lane terminal host candidate K/base position 与 device source 不同源");
    }
    Ok(())
}

fn complete_host_candidates_after_gpu_head(
    host_candidates: Box<dyn S14CausalBlockHostCandidateFinalizer>,
    base_position: u32,
    block_size: usize,
    routes_by_position: &[Vec<RouteDecision>],
    head_results: &[S14HeadArgmaxResult],
) -> Result<BatchedWholeTokenOutput> {
    if !matches!(block_size, 4 | 8)
        || host_candidates.block_size() != block_size
        || host_candidates.base_position() != base_position
        || routes_by_position.len() != block_size
        || head_results.len() != block_size
    {
        bail!("K-lane terminal GPU head/host candidate K/base position 不闭合");
    }
    let output = host_candidates.complete_after_gpu_head(head_results, routes_by_position)?;
    validate_completed_host_output(
        &output,
        base_position,
        block_size,
        routes_by_position,
        head_results,
    )?;
    Ok(output)
}

fn validate_completed_host_output(
    output: &BatchedWholeTokenOutput,
    base_position: u32,
    block_size: usize,
    routes_by_position: &[Vec<RouteDecision>],
    head_results: &[S14HeadArgmaxResult],
) -> Result<()> {
    if output.mode != BATCHED_CAUSAL_WHOLE_TOKEN_MODE
        || output.forward_calls != 1
        || output.positions.len() != block_size
        || routes_by_position.len() != block_size
        || head_results.len() != block_size
    {
        bail!("K-lane terminal host output 不是同一 production batched forward");
    }
    for (lane, ((position, routes), head)) in output
        .positions
        .iter()
        .zip(routes_by_position)
        .zip(head_results)
        .enumerate()
    {
        let expected_position = base_position
            .checked_add(lane as u32 + 1)
            .context("K-lane terminal checkpoint position overflow")?;
        position
            .checkpoint
            .validate()
            .map_err(|error| anyhow::anyhow!("candidate checkpoint {lane} 非法: {error}"))?;
        let record = position
            .checkpoint
            .committed_tokens
            .last()
            .context("K-lane terminal candidate checkpoint 缺少 token ledger")?;
        let expected_record_position = expected_position
            .checked_sub(1)
            .context("K-lane terminal token ledger position underflow")?;
        if &position.routes != routes
            || position.predicted_token_id != head.token_id
            || position.checkpoint.position != expected_position
            || record.position != expected_record_position
            || record.predicted_token_id != position.predicted_token_id
        {
            bail!("K-lane terminal host checkpoint/routes/GPU prediction 不同源");
        }
    }
    Ok(())
}

fn required_buffer(
    owner: &dyn S14CausalBlockTerminalResourceOwner,
    resource: S14CausalBlockTerminalResource,
) -> Result<&GpuBuffer> {
    owner
        .buffer(resource)
        .with_context(|| format!("K-lane terminal resource owner 缺少 {resource:?}"))
}

fn validate_owned_range(buffer: &GpuBuffer, offset: u64, bytes: u64, name: &str) -> Result<()> {
    if buffer.handle() == vk::Buffer::null()
        || bytes == 0
        || offset % 4 != 0
        || offset
            .checked_add(bytes)
            .is_none_or(|end| end > buffer.size())
    {
        bail!("K-lane terminal {name} owner range 越界/未对齐");
    }
    Ok(())
}

fn lock_provider(
    shared: &SharedProviderState,
) -> Result<std::sync::MutexGuard<'_, ProviderState>, String> {
    shared
        .lock()
        .map_err(|_| "K-lane terminal production provider poisoned".to_owned())
}

fn rollback_pending(shared: &SharedProviderState) -> Result<bool, String> {
    let mut state = lock_provider(shared)?;
    let rolled_back = state.pending.take().is_some();
    if rolled_back {
        state.telemetry.rollbacks = state.telemetry.rollbacks.saturating_add(1);
    }
    state.telemetry.pending = false;
    Ok(rolled_back)
}

fn telemetry(
    shared: &SharedProviderState,
) -> Result<S14CausalBlockTerminalProviderTelemetry, String> {
    Ok(lock_provider(shared)?.telemetry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct RecordingHostCandidateFinalizer {
        seen_gpu_tokens: Arc<Mutex<Vec<u32>>>,
    }

    impl S14CausalBlockHostCandidateFinalizer for RecordingHostCandidateFinalizer {
        fn block_size(&self) -> usize {
            4
        }

        fn base_position(&self) -> u32 {
            7
        }

        fn complete_after_gpu_head(
            self: Box<Self>,
            head_results: &[S14HeadArgmaxResult],
            routes_by_position: &[Vec<RouteDecision>],
        ) -> Result<BatchedWholeTokenOutput> {
            assert_eq!(routes_by_position.len(), 4);
            *self.seen_gpu_tokens.lock().unwrap() =
                head_results.iter().map(|head| head.token_id).collect();
            bail!("synthetic stop after GPU head")
        }
    }

    #[test]
    fn production_provider_missing_source_fails_closed_without_reuse() {
        let (publisher, provider) = s14_causal_block_terminal_production_channel();
        let hidden = S14CausalBlockHiddenBinding {
            buffer: vk::Buffer::null(),
            offset: 0,
            bytes: 0,
            block_size: 4,
            generation: 0,
        };
        let routes = vec![Vec::new(); 4];

        let first = provider.take_matching(FULL_DEPTH_LAYERS.len(), 7, hidden, &routes);
        let second = provider.take_matching(FULL_DEPTH_LAYERS.len(), 7, hidden, &routes);

        assert!(first.is_err());
        assert!(second.is_err());
        assert!(!publisher.rollback_pending().unwrap());
        assert_eq!(
            publisher.telemetry().unwrap(),
            S14CausalBlockTerminalProviderTelemetry {
                take_attempts: 2,
                rejected: 2,
                ..S14CausalBlockTerminalProviderTelemetry::default()
            }
        );
    }

    #[test]
    fn host_candidates_receive_gpu_head_before_checkpoint_completion() {
        let seen_gpu_tokens = Arc::new(Mutex::new(Vec::new()));
        let finalizer: Box<dyn S14CausalBlockHostCandidateFinalizer> =
            Box::new(RecordingHostCandidateFinalizer {
                seen_gpu_tokens: Arc::clone(&seen_gpu_tokens),
            });
        let routes = vec![Vec::new(); 4];
        let head_results = [
            S14HeadArgmaxResult {
                token_id: 5,
                logit: 1.0,
            },
            S14HeadArgmaxResult {
                token_id: 223,
                logit: 2.0,
            },
            S14HeadArgmaxResult {
                token_id: 939,
                logit: 3.0,
            },
            S14HeadArgmaxResult {
                token_id: 21,
                logit: 4.0,
            },
        ];

        let error =
            complete_host_candidates_after_gpu_head(finalizer, 7, 4, &routes, &head_results)
                .unwrap_err();

        assert!(error.to_string().contains("synthetic stop after GPU head"));
        assert_eq!(*seen_gpu_tokens.lock().unwrap(), vec![5, 223, 939, 21]);
    }
}
