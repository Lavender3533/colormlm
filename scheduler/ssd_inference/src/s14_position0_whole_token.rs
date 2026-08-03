//! FullDepth43 position0 的整 token 原子编排器。
//!
//! 本模块只负责所有权、顺序和证据合同，不提供任何默认数值实现。真实后端必须把
//! embedding、L0..L42 与 final head 写入同一个 inactive candidate bank；允许用多次
//! transfer/compute submit 滚动执行43层，但 token 末尾只能有一次 host wait。任何能力
//! 缺失、payload 未验证、回执漂移或 GPU submit 失败都会 fail-closed；固定 token 和历史
//! fixture 不得进入生产路径。

use crate::{GpuBuffer, VulkanContext};
use ash::vk;
use polaris_s14_runner::{
    BufferSlice, DecoderStateV1, NativeState, Position0Asset, Position0CompressorInput,
    Position0Final, Position0Layer, Position0ManifestError, Position0WholeTokenManifest,
    TokenRecord, WholeTokenCandidate, WholeTokenError, FULL_DEPTH_LAYERS,
};
use std::{fmt, ops::Range};

use crate::s14_whole_token_device::{
    WholeTokenDeviceCommitReceipt, WholeTokenDeviceState, WholeTokenPreparedCommit,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position0BackendCapabilities {
    pub embedding: bool,
    pub all_layers: bool,
    pub final_head: bool,
    pub payload_sha256: bool,
    pub route_receipts: bool,
    pub position0_state_outputs: bool,
}

impl Position0BackendCapabilities {
    fn missing(self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        for (enabled, name) in [
            (self.embedding, "embedding"),
            (self.all_layers, "all_layers"),
            (self.final_head, "final_head"),
            (self.payload_sha256, "payload_sha256"),
            (self.route_receipts, "route_receipts"),
            (self.position0_state_outputs, "position0_state_outputs"),
        ] {
            if !enabled {
                missing.push(name);
            }
        }
        missing
    }
}

/// 一次正在分段提交的真实 GPU candidate。所有 buffer 都只在回调期间有效；这里刻意
/// 不暴露 active bank，也不携带可被后续层误复用的 bootstrap command。
pub struct Position0GpuCandidate<'a> {
    pub ctx: &'a VulkanContext,
    pub candidate_state: &'a GpuBuffer,
    pub sticky_status: &'a GpuBuffer,
    pub committed_host_state: &'a DecoderStateV1,
    pub base_epoch: u64,
    pub candidate_bank: usize,
}

/// 首段 embedding compute 的 bootstrap。`prologue_command` 已由
/// `WholeTokenDeviceState::begin_candidate` 录入 committed→inactive 修复与 sticky 清零；
/// 后端必须把 embedding 追加到该 command，结束并作为本 token 的首个 compute segment 提交。
pub struct Position0GpuBootstrap<'a> {
    pub candidate: Position0GpuCandidate<'a>,
    pub prologue_command: vk::CommandBuffer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position0EmbeddingReceipt {
    pub input_token_id: u32,
    pub payload_verified: bool,
}

#[derive(Debug, Clone)]
pub enum Position0CompressorOutput {
    None,
    Ratio4 {
        main_kv: Vec<f32>,
        main_score: Vec<f32>,
        indexer_kv: Vec<f32>,
        indexer_score: Vec<f32>,
    },
    Ratio4Boundary {
        main_kv: Vec<f32>,
        main_score: Vec<f32>,
        indexer_kv: Vec<f32>,
        indexer_score: Vec<f32>,
        main_compressed_kv_bf16: Vec<u16>,
        indexer_compressed_kv_bf16: Vec<u16>,
    },
    Ratio128 {
        main_kv: Vec<f32>,
        main_score: Vec<f32>,
    },
}

impl Position0CompressorOutput {
    fn ratio(&self) -> u16 {
        match self {
            Self::None => 0,
            Self::Ratio4 { .. } | Self::Ratio4Boundary { .. } => 4,
            Self::Ratio128 { .. } => 128,
        }
    }

    fn as_input(&self) -> Position0CompressorInput<'_> {
        match self {
            Self::None => Position0CompressorInput::None,
            Self::Ratio4 {
                main_kv,
                main_score,
                indexer_kv,
                indexer_score,
            } => Position0CompressorInput::Ratio4 {
                main_kv,
                main_score,
                indexer_kv,
                indexer_score,
            },
            Self::Ratio4Boundary {
                main_kv,
                main_score,
                indexer_kv,
                indexer_score,
                main_compressed_kv_bf16,
                indexer_compressed_kv_bf16,
            } => Position0CompressorInput::Ratio4Boundary {
                main_kv,
                main_score,
                indexer_kv,
                indexer_score,
                main_compressed_kv_bf16,
                indexer_compressed_kv_bf16,
            },
            Self::Ratio128 {
                main_kv,
                main_score,
            } => Position0CompressorInput::Ratio128 {
                main_kv,
                main_score,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Position0LayerReceipt {
    pub layer: u8,
    pub payloads_verified: bool,
    pub route_source: String,
    pub expert_ids: Vec<u16>,
    pub route_weights: Vec<f32>,
    pub ffn_activation_f32_le_sha256: String,
    pub moe_output_bf16_sha256: String,
    pub moe_branch_f32_le_sha256: String,
    pub layer_output_f32_le_sha256: String,
    pub window_kv_bf16: Vec<u16>,
    pub compressor_output: Position0CompressorOutput,
    /// whole-token wait 后由真实后端回报的 GPU state 写区间；必须与 host layout 推导值逐项一致。
    pub state_ranges_written: Vec<Range<u64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position0FinalReceipt {
    pub predicted_token_id: u32,
    pub payloads_verified: bool,
    pub normalized_f32_le_sha256: String,
    pub gpu_head_sha256: String,
    pub hc_streams_bf16: Vec<u16>,
    pub state_ranges_written: Vec<Range<u64>>,
}

#[derive(Debug, Clone)]
pub struct Position0GraphReceipt {
    pub embedding: Position0EmbeddingReceipt,
    pub layers: Vec<Position0LayerReceipt>,
    pub final_head: Position0FinalReceipt,
}

/// 外部 timeline 唯一 host wait 的身份回执。`final_compute_value` 必须是本 token 最后一个
/// compute signal，而不是上一 token 或 transfer-only ticket。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position0BackendCompletion {
    pub base_epoch: u64,
    pub candidate_bank: usize,
    pub final_compute_value: u64,
    pub token_host_waits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Position0BackendError {
    Unavailable(String),
    Execution(String),
}

impl fmt::Display for Position0BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Position0BackendError {}

/// 生产后端的强类型边界。这里刻意没有默认方法或默认实现。
pub trait Position0LayerBackend {
    fn capabilities(&self) -> Position0BackendCapabilities;

    /// 追加 embedding，并提交含 candidate prologue 的首个 compute segment。返回成功后，
    /// orchestrator 会把 device candidate 标记为 in-flight。
    fn submit_embedding(
        &mut self,
        bootstrap: &Position0GpuBootstrap<'_>,
        embedding: &Position0Asset,
    ) -> Result<(), Position0BackendError>;

    /// 可以先提交下一层 transfer，再提交等待该 transfer 的 compute。不得逐层 host wait。
    fn submit_layer(
        &mut self,
        candidate: &Position0GpuCandidate<'_>,
        layer: &Position0Layer,
    ) -> Result<(), Position0BackendError>;

    fn submit_final(
        &mut self,
        candidate: &Position0GpuCandidate<'_>,
        final_section: &Position0Final,
    ) -> Result<(), Position0BackendError>;

    /// 整 token 成功路径的唯一 host wait。实现通常等待 `S14DualQueueTimeline` 的最后一个
    /// compute ticket；返回成功时，所有 descriptor/resource lease 与 candidate 写入均已完成。
    fn wait_candidate(&mut self) -> Result<Position0BackendCompletion, Position0BackendError>;

    /// 只允许在 `wait_candidate` 成功后读取本次 GPU graph 的动态 route、状态和哈希。
    fn finish_receipts(
        &mut self,
        manifest: &Position0WholeTokenManifest,
    ) -> Result<Position0GraphReceipt, Position0BackendError>;

    /// 错误路径的唯一 drain 点。即使首段尚未真正入队也必须可调用；返回成功表示所有可能
    /// 引用 candidate/prologue 的 GPU 工作均已完成，可以安全 reset command pool 并丢弃 bank。
    fn abort_candidate(&mut self) -> Result<(), Position0BackendError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position0WholeTokenOutcome {
    pub token: TokenRecord,
    pub committed_epoch: u64,
    pub active_bank: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Position0WholeTokenError {
    Manifest(String),
    HostState(String),
    Device(String),
    BackendUnavailable(String),
    Backend {
        stage: String,
        detail: String,
    },
    PayloadUnverified(String),
    EmbeddingToken {
        expected: u32,
        actual: u32,
    },
    LayerReceipt {
        expected: u8,
        actual: u8,
    },
    RouteReceipt(u8),
    NumericReceipt(u8),
    CompressorRatio {
        layer: u8,
        expected: u16,
        actual: u16,
    },
    MissingDirtyRange(u8),
    FinalStateRange,
    FinalToken {
        expected: u32,
        actual: u32,
    },
    FinalNumericReceipt,
    AtomicInvariant(String),
    Rollback {
        cause: String,
        rollback: String,
    },
}

impl fmt::Display for Position0WholeTokenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Position0WholeTokenError {}

impl From<Position0ManifestError> for Position0WholeTokenError {
    fn from(value: Position0ManifestError) -> Self {
        Self::Manifest(value.to_string())
    }
}

impl From<WholeTokenError> for Position0WholeTokenError {
    fn from(value: WholeTokenError) -> Self {
        Self::HostState(value.to_string())
    }
}

/// 执行 BOS position0。只有真实后端、manifest、host candidate 与 GPU candidate
/// 全部闭合后才返回 token；本函数不含任何 fixture 或固定 token 路径。
pub fn run_bos_position0(
    ctx: &VulkanContext,
    manifest: &Position0WholeTokenManifest,
    host_state: &mut DecoderStateV1,
    device_state: &mut WholeTokenDeviceState,
    backend: &mut dyn Position0LayerBackend,
) -> Result<Position0WholeTokenOutcome, Position0WholeTokenError> {
    manifest.validate()?;
    ensure_backend_capabilities(backend.capabilities())?;
    let mut execution = LiveCandidateExecution {
        ctx,
        device: device_state,
        backend,
        bootstrap_command: None,
        backend_started: false,
        backend_drained: false,
        prepared_commit: None,
    };
    run_bos_transaction(manifest, host_state, &mut execution)
}

fn ensure_backend_capabilities(
    capabilities: Position0BackendCapabilities,
) -> Result<(), Position0WholeTokenError> {
    let missing = capabilities.missing();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(Position0WholeTokenError::BackendUnavailable(
            missing.join(","),
        ))
    }
}

trait CandidateExecution {
    fn epoch(&self) -> u64;
    fn active_bank(&self) -> usize;
    fn state_bytes(&self) -> u64;
    fn begin(&mut self, expected_epoch: u64) -> Result<(), Position0WholeTokenError>;
    fn record_embedding(
        &mut self,
        host: &DecoderStateV1,
        asset: &Position0Asset,
    ) -> Result<(), Position0WholeTokenError>;
    fn record_layer(
        &mut self,
        host: &DecoderStateV1,
        entry: &Position0Layer,
    ) -> Result<(), Position0WholeTokenError>;
    fn mark_dirty(&mut self, range: Range<u64>) -> Result<(), Position0WholeTokenError>;
    fn record_final(
        &mut self,
        host: &DecoderStateV1,
        final_section: &Position0Final,
    ) -> Result<(), Position0WholeTokenError>;
    fn wait(&mut self) -> Result<(), Position0WholeTokenError>;
    fn finish_receipts(
        &mut self,
        manifest: &Position0WholeTokenManifest,
    ) -> Result<Position0GraphReceipt, Position0WholeTokenError>;
    fn prepare_commit(&mut self, expected_epoch: u64) -> Result<(), Position0WholeTokenError>;
    fn publish_commit(&mut self) -> WholeTokenDeviceCommitReceipt;
    fn rollback(&mut self) -> Result<(), Position0WholeTokenError>;
}

struct LiveCandidateExecution<'a> {
    ctx: &'a VulkanContext,
    device: &'a mut WholeTokenDeviceState,
    backend: &'a mut dyn Position0LayerBackend,
    bootstrap_command: Option<vk::CommandBuffer>,
    backend_started: bool,
    backend_drained: bool,
    prepared_commit: Option<WholeTokenPreparedCommit>,
}

impl CandidateExecution for LiveCandidateExecution<'_> {
    fn epoch(&self) -> u64 {
        self.device.epoch()
    }

    fn active_bank(&self) -> usize {
        self.device.active_bank()
    }

    fn state_bytes(&self) -> u64 {
        self.device.state_bytes()
    }

    fn begin(&mut self, expected_epoch: u64) -> Result<(), Position0WholeTokenError> {
        let command = self
            .device
            .begin_candidate(self.ctx, expected_epoch)
            .map_err(|error| Position0WholeTokenError::Device(error.to_string()))?;
        self.bootstrap_command = Some(command);
        self.backend_started = false;
        self.backend_drained = false;
        Ok(())
    }

    fn record_embedding(
        &mut self,
        host: &DecoderStateV1,
        asset: &Position0Asset,
    ) -> Result<(), Position0WholeTokenError> {
        let command = self.bootstrap_command.ok_or_else(|| {
            Position0WholeTokenError::AtomicInvariant("GPU candidate 尚未开始".into())
        })?;
        let base_epoch = self.device.epoch();
        let candidate_bank = 1 - self.device.active_bank();
        // 在调用前即标记 started：后端可能在 queue_submit 成功后才发现后续错误；此时
        // rollback 仍必须进入 abort_candidate，而不能假设没有 in-flight 工作。
        self.backend_started = true;
        {
            let bootstrap = Position0GpuBootstrap {
                candidate: Position0GpuCandidate {
                    ctx: self.ctx,
                    candidate_state: self
                        .device
                        .candidate_buffer()
                        .map_err(|error| Position0WholeTokenError::Device(error.to_string()))?,
                    sticky_status: self
                        .device
                        .sticky_status_buffer()
                        .map_err(|error| Position0WholeTokenError::Device(error.to_string()))?,
                    committed_host_state: host,
                    base_epoch,
                    candidate_bank,
                },
                prologue_command: command,
            };
            self.backend
                .submit_embedding(&bootstrap, asset)
                .map_err(|error| map_backend_error("embedding submit", error))?;
        }
        self.device
            .mark_candidate_in_flight()
            .map_err(|error| Position0WholeTokenError::Device(error.to_string()))?;
        self.bootstrap_command = None;
        Ok(())
    }

    fn record_layer(
        &mut self,
        host: &DecoderStateV1,
        entry: &Position0Layer,
    ) -> Result<(), Position0WholeTokenError> {
        if self.bootstrap_command.is_some() || !self.backend_started {
            return Err(Position0WholeTokenError::AtomicInvariant(
                "layer submit 发生在 embedding bootstrap 之前".into(),
            ));
        }
        let candidate = Position0GpuCandidate {
            ctx: self.ctx,
            candidate_state: self
                .device
                .candidate_buffer()
                .map_err(|error| Position0WholeTokenError::Device(error.to_string()))?,
            sticky_status: self
                .device
                .sticky_status_buffer()
                .map_err(|error| Position0WholeTokenError::Device(error.to_string()))?,
            committed_host_state: host,
            base_epoch: self.device.epoch(),
            candidate_bank: 1 - self.device.active_bank(),
        };
        self.backend
            .submit_layer(&candidate, entry)
            .map_err(|error| map_backend_error(&format!("L{} submit", entry.layer), error))
    }

    fn mark_dirty(&mut self, range: Range<u64>) -> Result<(), Position0WholeTokenError> {
        self.device
            .mark_candidate_dirty(range.start, range.end.saturating_sub(range.start))
            .map_err(|error| Position0WholeTokenError::Device(error.to_string()))
    }

    fn record_final(
        &mut self,
        host: &DecoderStateV1,
        final_section: &Position0Final,
    ) -> Result<(), Position0WholeTokenError> {
        if self.bootstrap_command.is_some() || !self.backend_started {
            return Err(Position0WholeTokenError::AtomicInvariant(
                "final submit 发生在 embedding bootstrap 之前".into(),
            ));
        }
        let candidate = Position0GpuCandidate {
            ctx: self.ctx,
            candidate_state: self
                .device
                .candidate_buffer()
                .map_err(|error| Position0WholeTokenError::Device(error.to_string()))?,
            sticky_status: self
                .device
                .sticky_status_buffer()
                .map_err(|error| Position0WholeTokenError::Device(error.to_string()))?,
            committed_host_state: host,
            base_epoch: self.device.epoch(),
            candidate_bank: 1 - self.device.active_bank(),
        };
        self.backend
            .submit_final(&candidate, final_section)
            .map_err(|error| map_backend_error("final submit", error))
    }

    fn wait(&mut self) -> Result<(), Position0WholeTokenError> {
        if self.bootstrap_command.is_some() || !self.backend_started {
            return Err(Position0WholeTokenError::AtomicInvariant(
                "whole-token wait 发生在 bootstrap submit 之前".into(),
            ));
        }
        let completion = self
            .backend
            .wait_candidate()
            .map_err(|error| map_backend_error("whole-token wait", error))?;
        self.backend_drained = true;
        validate_backend_completion(
            completion,
            self.device.epoch(),
            1 - self.device.active_bank(),
        )?;
        self.device
            .finish_external_candidate(completion.base_epoch, completion.candidate_bank)
            .map_err(|error| Position0WholeTokenError::Device(error.to_string()))
    }

    fn finish_receipts(
        &mut self,
        manifest: &Position0WholeTokenManifest,
    ) -> Result<Position0GraphReceipt, Position0WholeTokenError> {
        self.backend
            .finish_receipts(manifest)
            .map_err(|error| map_backend_error("post-fence receipts", error))
    }

    fn prepare_commit(&mut self, expected_epoch: u64) -> Result<(), Position0WholeTokenError> {
        let prepared = self
            .device
            .prepare_candidate_commit(expected_epoch)
            .map_err(|error| Position0WholeTokenError::Device(error.to_string()))?;
        self.prepared_commit = Some(prepared);
        Ok(())
    }

    fn publish_commit(&mut self) -> WholeTokenDeviceCommitReceipt {
        self.device.publish_prepared_commit(
            self.prepared_commit
                .take()
                .expect("publish_commit 必须紧跟成功的 prepare_commit"),
        )
    }

    fn rollback(&mut self) -> Result<(), Position0WholeTokenError> {
        self.prepared_commit = None;
        if self.backend_started && !self.backend_drained {
            self.backend
                .abort_candidate()
                .map_err(|error| map_backend_error("whole-token abort", error))?;
            self.backend_drained = true;
        }
        let result = if self.backend_started {
            self.device.rollback_external_candidate(self.ctx)
        } else {
            self.device.rollback_candidate(self.ctx)
        };
        self.bootstrap_command = None;
        self.backend_started = false;
        result.map_err(|error| Position0WholeTokenError::Device(error.to_string()))
    }
}

fn map_backend_error(stage: &str, error: Position0BackendError) -> Position0WholeTokenError {
    match error {
        Position0BackendError::Unavailable(detail) => {
            Position0WholeTokenError::BackendUnavailable(format!("{stage}: {detail}"))
        }
        Position0BackendError::Execution(detail) => Position0WholeTokenError::Backend {
            stage: stage.into(),
            detail,
        },
    }
}

fn validate_backend_completion(
    completion: Position0BackendCompletion,
    expected_epoch: u64,
    expected_bank: usize,
) -> Result<(), Position0WholeTokenError> {
    if completion.base_epoch != expected_epoch
        || completion.candidate_bank != expected_bank
        || completion.final_compute_value == 0
        || completion.token_host_waits != 1
    {
        return Err(Position0WholeTokenError::AtomicInvariant(
            "whole-token completion ticket 的 epoch/bank/value/wait-count 漂移".into(),
        ));
    }
    Ok(())
}

fn run_bos_transaction(
    manifest: &Position0WholeTokenManifest,
    host_state: &mut DecoderStateV1,
    execution: &mut dyn CandidateExecution,
) -> Result<Position0WholeTokenOutcome, Position0WholeTokenError> {
    manifest.validate()?;
    host_state.validate()?;
    if manifest.position != 0
        || manifest.input_token_id != 0
        || host_state.position != 0
        || host_state.input_token_id != manifest.input_token_id
    {
        return Err(Position0WholeTokenError::AtomicInvariant(
            "BOS/position0 host 与 manifest 不闭合".into(),
        ));
    }
    if host_state.commit_epoch != execution.epoch()
        || host_state.active_fixed_bank as usize != execution.active_bank()
        || host_state.native_arena.len() as u64 != execution.state_bytes()
    {
        return Err(Position0WholeTokenError::AtomicInvariant(
            "host/device epoch、bank 或 state bytes 不闭合".into(),
        ));
    }

    let snapshot = host_state.clone();
    let base_epoch = host_state.commit_epoch;
    let base_bank = execution.active_bank();
    let host_candidate = host_state.begin_token(base_epoch, 0, manifest.input_token_id)?;
    execution.begin(base_epoch)?;

    let attempt = execute_candidate(
        manifest,
        &snapshot,
        host_candidate,
        execution,
        host_state,
        base_epoch,
        base_bank,
    );
    match attempt {
        Ok(outcome) => Ok(outcome),
        Err(cause) => {
            *host_state = snapshot;
            match execution.rollback() {
                Ok(()) => Err(cause),
                Err(rollback) => Err(Position0WholeTokenError::Rollback {
                    cause: cause.to_string(),
                    rollback: rollback.to_string(),
                }),
            }
        }
    }
}

fn execute_candidate(
    manifest: &Position0WholeTokenManifest,
    committed_host: &DecoderStateV1,
    mut host_candidate: WholeTokenCandidate,
    execution: &mut dyn CandidateExecution,
    host_state: &mut DecoderStateV1,
    base_epoch: u64,
    base_bank: usize,
) -> Result<Position0WholeTokenOutcome, Position0WholeTokenError> {
    execution.record_embedding(committed_host, &manifest.embedding_row)?;
    for (&expected_layer, entry) in FULL_DEPTH_LAYERS.iter().zip(&manifest.layers) {
        execution.record_layer(committed_host, entry)?;
        for range in expected_layer_state_ranges(&committed_host.native, expected_layer)? {
            execution.mark_dirty(range)?;
        }
    }
    execution.record_final(committed_host, &manifest.final_section)?;
    let final_state_ranges = expected_final_state_ranges(&committed_host.native)?;
    for range in &final_state_ranges {
        execution.mark_dirty(range.clone())?;
    }

    // 唯一 whole-token host wait。此前可以有任意次 transfer/compute submit；所有动态
    // route、SHA、KV、HC 与 argmax 回执必须在
    // 此后读取；record 阶段绝不允许回显 manifest 伪装成 GPU 结果。
    execution.wait()?;
    let receipts = execution.finish_receipts(manifest)?;
    let embedding = receipts.embedding;
    if !embedding.payload_verified {
        return Err(Position0WholeTokenError::PayloadUnverified(
            "embedding".into(),
        ));
    }
    if embedding.input_token_id != manifest.input_token_id {
        return Err(Position0WholeTokenError::EmbeddingToken {
            expected: manifest.input_token_id,
            actual: embedding.input_token_id,
        });
    }

    if receipts.layers.len() != FULL_DEPTH_LAYERS.len() {
        return Err(Position0WholeTokenError::AtomicInvariant(
            "post-fence layer receipt 不是 43 层".into(),
        ));
    }
    for ((&expected_layer, entry), receipt) in FULL_DEPTH_LAYERS
        .iter()
        .zip(&manifest.layers)
        .zip(&receipts.layers)
    {
        let expected_ranges = expected_layer_state_ranges(&committed_host.native, expected_layer)?;
        validate_layer_receipt(expected_layer, entry, receipt, &expected_ranges)?;
        host_candidate.stage_position0_layer_state(
            expected_layer,
            &receipt.window_kv_bf16,
            receipt.compressor_output.as_input(),
        )?;
        host_candidate.complete_layer(expected_layer)?;
    }

    let final_receipt = receipts.final_head;
    if !final_receipt.payloads_verified {
        return Err(Position0WholeTokenError::PayloadUnverified("final".into()));
    }
    if final_receipt.normalized_f32_le_sha256 != manifest.final_section.normalized_f32_le_sha256
        || final_receipt.gpu_head_sha256 != manifest.final_section.gpu_head_sha256
    {
        return Err(Position0WholeTokenError::FinalNumericReceipt);
    }
    if final_receipt.predicted_token_id != manifest.expected_output_token_id {
        return Err(Position0WholeTokenError::FinalToken {
            expected: manifest.expected_output_token_id,
            actual: final_receipt.predicted_token_id,
        });
    }
    if !same_ranges(&final_receipt.state_ranges_written, &final_state_ranges)? {
        return Err(Position0WholeTokenError::FinalStateRange);
    }
    host_candidate.stage_position0_hc_state(&final_receipt.hc_streams_bf16)?;
    host_candidate.complete_final(final_receipt.predicted_token_id)?;

    let expected_epoch = base_epoch
        .checked_add(1)
        .ok_or_else(|| Position0WholeTokenError::AtomicInvariant("epoch overflow".into()))?;
    let expected_bank = 1 - base_bank;
    let mut next_host = host_state.clone();
    let token = host_candidate.commit(&mut next_host)?;
    if next_host.commit_epoch != expected_epoch
        || next_host.active_fixed_bank as usize != expected_bank
        || token.predicted_token_id != manifest.expected_output_token_id
    {
        return Err(Position0WholeTokenError::AtomicInvariant(
            "host 提交后的 epoch/bank/token 漂移".into(),
        ));
    }
    execution.prepare_commit(base_epoch)?;
    let _device_receipt = execution.publish_commit();
    // publish_commit 由 prepare 签发的一次性 ticket 驱动，发布阶段无可恢复错误；
    // 到这里才替换真实 host state，之后不再执行任何 fallible 操作。
    *host_state = next_host;
    Ok(Position0WholeTokenOutcome {
        token,
        committed_epoch: expected_epoch,
        active_bank: expected_bank,
    })
}

fn validate_layer_receipt(
    expected_layer: u8,
    entry: &Position0Layer,
    receipt: &Position0LayerReceipt,
    expected_state_ranges: &[Range<u64>],
) -> Result<(), Position0WholeTokenError> {
    if receipt.layer != expected_layer || entry.layer != expected_layer {
        return Err(Position0WholeTokenError::LayerReceipt {
            expected: expected_layer,
            actual: receipt.layer,
        });
    }
    if !receipt.payloads_verified {
        return Err(Position0WholeTokenError::PayloadUnverified(format!(
            "L{expected_layer}"
        )));
    }
    if receipt.route_source != entry.route_source
        || receipt.expert_ids != entry.expert_ids
        || !same_f32_bits(&receipt.route_weights, &entry.route_weights)
    {
        return Err(Position0WholeTokenError::RouteReceipt(expected_layer));
    }
    if receipt.ffn_activation_f32_le_sha256 != entry.capture.input_sha256
        || receipt.moe_output_bf16_sha256 != entry.capture.k1_moe_output_sha256
        || receipt.moe_branch_f32_le_sha256 != entry.reference.moe_branch_f32_le_sha256
        || receipt.layer_output_f32_le_sha256 != entry.reference.layer_output_f32_le_sha256
    {
        return Err(Position0WholeTokenError::NumericReceipt(expected_layer));
    }
    let actual_ratio = receipt.compressor_output.ratio();
    if actual_ratio != entry.compress_ratio {
        return Err(Position0WholeTokenError::CompressorRatio {
            layer: expected_layer,
            expected: entry.compress_ratio,
            actual: actual_ratio,
        });
    }
    if !same_ranges(&receipt.state_ranges_written, expected_state_ranges)? {
        return Err(Position0WholeTokenError::MissingDirtyRange(expected_layer));
    }
    Ok(())
}

fn same_f32_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

/// position0 每层真实写集必须与 `Position0StateTxn::stage_layer` 完全相同。
/// ratio4 的 indexer `kv_cache` 在 position0 尚未形成压缩 token，因此刻意不在写集中；
/// 这里只写 window KV 与 main/indexer compressor remainder。
fn expected_layer_state_ranges(
    state: &NativeState,
    layer: u8,
) -> Result<Vec<Range<u64>>, Position0WholeTokenError> {
    state
        .validate()
        .map_err(|error| Position0WholeTokenError::HostState(error.to_string()))?;
    let kv = state
        .kv
        .iter()
        .find(|entry| entry.layer == layer)
        .ok_or_else(|| {
            Position0WholeTokenError::AtomicInvariant(format!("L{layer} KV layout 缺失"))
        })?;
    let mut ranges = vec![row_range(&kv.cache, 0, &format!("L{layer} window KV"))?];

    match kv.compress_ratio {
        0 => {}
        4 => {
            let compressor = state
                .compressors
                .iter()
                .find(|entry| entry.layer == layer && entry.compress_ratio == 4)
                .ok_or_else(|| {
                    Position0WholeTokenError::AtomicInvariant(format!(
                        "L{layer} ratio4 compressor layout 缺失"
                    ))
                })?;
            let indexer = state
                .indexers
                .iter()
                .find(|entry| entry.layer == layer)
                .ok_or_else(|| {
                    Position0WholeTokenError::AtomicInvariant(format!(
                        "L{layer} ratio4 indexer layout 缺失"
                    ))
                })?;
            ranges.push(row_range(
                &compressor.kv_state,
                4,
                &format!("L{layer} main compressor KV remainder"),
            )?);
            ranges.push(row_range(
                &compressor.score_state,
                4,
                &format!("L{layer} main compressor score remainder"),
            )?);
            ranges.push(row_range(
                &indexer.compressor_kv_state,
                4,
                &format!("L{layer} indexer compressor KV remainder"),
            )?);
            ranges.push(row_range(
                &indexer.compressor_score_state,
                4,
                &format!("L{layer} indexer compressor score remainder"),
            )?);
        }
        128 => {
            let compressor = state
                .compressors
                .iter()
                .find(|entry| entry.layer == layer && entry.compress_ratio == 128)
                .ok_or_else(|| {
                    Position0WholeTokenError::AtomicInvariant(format!(
                        "L{layer} ratio128 compressor layout 缺失"
                    ))
                })?;
            ranges.push(row_range(
                &compressor.kv_state,
                0,
                &format!("L{layer} main compressor KV remainder"),
            )?);
            ranges.push(row_range(
                &compressor.score_state,
                0,
                &format!("L{layer} main compressor score remainder"),
            )?);
        }
        ratio => {
            return Err(Position0WholeTokenError::AtomicInvariant(format!(
                "L{layer} 未知 compress ratio {ratio}"
            )));
        }
    }
    Ok(ranges)
}

fn expected_final_state_ranges(
    state: &NativeState,
) -> Result<Vec<Range<u64>>, Position0WholeTokenError> {
    state
        .validate()
        .map_err(|error| Position0WholeTokenError::HostState(error.to_string()))?;
    Ok(vec![whole_slice_range(
        &state.hc.streams,
        "final HC streams",
    )?])
}

fn row_range(
    slice: &BufferSlice,
    row: u64,
    label: &str,
) -> Result<Range<u64>, Position0WholeTokenError> {
    if slice.shape.len() != 3 || slice.shape[0] != 1 {
        return Err(Position0WholeTokenError::AtomicInvariant(format!(
            "{label} 不是 [1,rows,width]"
        )));
    }
    let rows = u64::from(slice.shape[1]);
    if rows == 0 || row >= rows || slice.bytes % rows != 0 {
        return Err(Position0WholeTokenError::AtomicInvariant(format!(
            "{label} row/bytes layout 漂移"
        )));
    }
    let row_bytes = slice.bytes / rows;
    let relative = row.checked_mul(row_bytes).ok_or_else(|| {
        Position0WholeTokenError::AtomicInvariant(format!("{label} offset overflow"))
    })?;
    checked_slice_range(slice, relative, row_bytes, label)
}

fn whole_slice_range(
    slice: &BufferSlice,
    label: &str,
) -> Result<Range<u64>, Position0WholeTokenError> {
    checked_slice_range(slice, 0, slice.bytes, label)
}

fn checked_slice_range(
    slice: &BufferSlice,
    relative: u64,
    bytes: u64,
    label: &str,
) -> Result<Range<u64>, Position0WholeTokenError> {
    let relative_end = relative.checked_add(bytes).ok_or_else(|| {
        Position0WholeTokenError::AtomicInvariant(format!("{label} range overflow"))
    })?;
    if bytes == 0
        || relative_end > slice.bytes
        || relative % 4 != 0
        || bytes % 4 != 0
        || slice.offset % 4 != 0
    {
        return Err(Position0WholeTokenError::AtomicInvariant(format!(
            "{label} range 非法"
        )));
    }
    let start = slice.offset.checked_add(relative).ok_or_else(|| {
        Position0WholeTokenError::AtomicInvariant(format!("{label} start overflow"))
    })?;
    let end = start.checked_add(bytes).ok_or_else(|| {
        Position0WholeTokenError::AtomicInvariant(format!("{label} end overflow"))
    })?;
    Ok(start..end)
}

fn same_ranges(
    actual: &[Range<u64>],
    expected: &[Range<u64>],
) -> Result<bool, Position0WholeTokenError> {
    fn canonical(ranges: &[Range<u64>]) -> Result<Vec<(u64, u64)>, Position0WholeTokenError> {
        let mut result = Vec::with_capacity(ranges.len());
        for range in ranges {
            if range.start >= range.end || range.start % 4 != 0 || range.end % 4 != 0 {
                return Err(Position0WholeTokenError::AtomicInvariant(
                    "state receipt range 必须非空且 4-byte 对齐".into(),
                ));
            }
            result.push((range.start, range.end));
        }
        result.sort_unstable();
        if result.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(Position0WholeTokenError::AtomicInvariant(
                "state receipt range 重叠".into(),
            ));
        }
        Ok(result)
    }

    Ok(canonical(actual)? == canonical(expected)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Failure {
        Layer(u8),
        Final,
        WrongFinal,
        UnverifiedLayer(u8),
        WrongRoute(u8),
        WrongLayer(u8),
        ShaDrift(u8),
        LayerRangeMissing(u8),
        LayerRangeExtra(u8),
        LayerRangeArbitrary(u8),
        FinalRangeMissing,
        FinalRangeExtra,
        FinalRangeArbitrary,
        Submit,
        Finish,
        Prepare,
    }

    struct FixtureExecution {
        epoch: u64,
        bank: usize,
        state_bytes: u64,
        phase: u8,
        failure: Option<Failure>,
        events: Vec<String>,
        host_waits: usize,
        dirty: Vec<Range<u64>>,
        native: NativeState,
    }

    impl FixtureExecution {
        fn new(state: &DecoderStateV1, failure: Option<Failure>) -> Self {
            Self {
                epoch: state.commit_epoch,
                bank: state.active_fixed_bank as usize,
                state_bytes: state.native_arena.len() as u64,
                phase: 0,
                failure,
                events: Vec::new(),
                host_waits: 0,
                dirty: Vec::new(),
                native: state.native.clone(),
            }
        }
    }

    impl CandidateExecution for FixtureExecution {
        fn epoch(&self) -> u64 {
            self.epoch
        }

        fn active_bank(&self) -> usize {
            self.bank
        }

        fn state_bytes(&self) -> u64 {
            self.state_bytes
        }

        fn begin(&mut self, expected_epoch: u64) -> Result<(), Position0WholeTokenError> {
            assert_eq!(expected_epoch, self.epoch);
            self.phase = 1;
            Ok(())
        }

        fn record_embedding(
            &mut self,
            _host: &DecoderStateV1,
            _asset: &Position0Asset,
        ) -> Result<(), Position0WholeTokenError> {
            if self.phase != 1 {
                return Err(Position0WholeTokenError::AtomicInvariant(
                    "embedding 必须在 recording 阶段".into(),
                ));
            }
            self.events.push("submit:embedding".into());
            Ok(())
        }

        fn record_layer(
            &mut self,
            _host: &DecoderStateV1,
            entry: &Position0Layer,
        ) -> Result<(), Position0WholeTokenError> {
            if self.phase != 1 {
                return Err(Position0WholeTokenError::AtomicInvariant(
                    "layer 必须在 recording 阶段".into(),
                ));
            }
            self.events.push(format!("submit:L{}", entry.layer));
            if self.failure == Some(Failure::Layer(entry.layer)) {
                return Err(Position0WholeTokenError::Backend {
                    stage: format!("L{}", entry.layer),
                    detail: "injected".into(),
                });
            }
            Ok(())
        }

        fn mark_dirty(&mut self, range: Range<u64>) -> Result<(), Position0WholeTokenError> {
            assert!(range.start < range.end);
            self.dirty.push(range);
            Ok(())
        }

        fn record_final(
            &mut self,
            _host: &DecoderStateV1,
            _final_section: &Position0Final,
        ) -> Result<(), Position0WholeTokenError> {
            if self.phase != 1 {
                return Err(Position0WholeTokenError::AtomicInvariant(
                    "final 必须在 recording 阶段".into(),
                ));
            }
            self.events.push("submit:final".into());
            if self.failure == Some(Failure::Final) {
                return Err(Position0WholeTokenError::Backend {
                    stage: "final".into(),
                    detail: "injected".into(),
                });
            }
            Ok(())
        }

        fn wait(&mut self) -> Result<(), Position0WholeTokenError> {
            self.events.push("wait".into());
            self.host_waits += 1;
            if self.failure == Some(Failure::Submit) {
                self.phase = 3;
                return Err(Position0WholeTokenError::Device("wait injected".into()));
            }
            self.phase = 2;
            Ok(())
        }

        fn finish_receipts(
            &mut self,
            manifest: &Position0WholeTokenManifest,
        ) -> Result<Position0GraphReceipt, Position0WholeTokenError> {
            self.events.push("finish".into());
            if self.phase != 2 {
                return Err(Position0WholeTokenError::Backend {
                    stage: "post-fence receipts".into(),
                    detail: "finish_receipts 只能在 whole-token wait 后执行".into(),
                });
            }
            if self.failure == Some(Failure::Finish) {
                return Err(Position0WholeTokenError::Backend {
                    stage: "post-fence receipts".into(),
                    detail: "injected".into(),
                });
            }
            let layers = manifest
                .layers
                .iter()
                .map(|entry| fixture_layer_receipt(&self.native, entry, self.failure))
                .collect::<Result<Vec<_>, _>>()?;
            let final_head =
                fixture_final_receipt(&self.native, &manifest.final_section, self.failure)?;
            Ok(Position0GraphReceipt {
                embedding: Position0EmbeddingReceipt {
                    input_token_id: 0,
                    payload_verified: true,
                },
                layers,
                final_head,
            })
        }

        fn prepare_commit(&mut self, expected_epoch: u64) -> Result<(), Position0WholeTokenError> {
            self.events.push("prepare".into());
            assert_eq!(expected_epoch, self.epoch);
            if self.phase != 2 {
                return Err(Position0WholeTokenError::Device(
                    "prepare 必须在 ready 阶段".into(),
                ));
            }
            if self.failure == Some(Failure::Prepare) {
                return Err(Position0WholeTokenError::Device("prepare injected".into()));
            }
            assert!(!self.dirty.is_empty());
            self.phase = 3;
            Ok(())
        }

        fn publish_commit(&mut self) -> WholeTokenDeviceCommitReceipt {
            self.events.push("publish".into());
            assert_eq!(self.phase, 3);
            self.epoch += 1;
            self.bank = 1 - self.bank;
            self.phase = 0;
            WholeTokenDeviceCommitReceipt {
                epoch: self.epoch,
                active_bank: self.bank,
            }
        }

        fn rollback(&mut self) -> Result<(), Position0WholeTokenError> {
            if self.phase == 1 {
                self.events.push("abort-wait".into());
                self.host_waits += 1;
            }
            self.events.push("rollback".into());
            self.phase = 0;
            self.dirty.clear();
            Ok(())
        }
    }

    fn fixture_layer_receipt(
        native: &NativeState,
        entry: &Position0Layer,
        failure: Option<Failure>,
    ) -> Result<Position0LayerReceipt, Position0WholeTokenError> {
        let compressor_output = match entry.compress_ratio {
            0 => Position0CompressorOutput::None,
            4 => Position0CompressorOutput::Ratio4 {
                main_kv: vec![1.0; 1024],
                main_score: vec![2.0; 1024],
                indexer_kv: vec![3.0; 256],
                indexer_score: vec![4.0; 256],
            },
            128 => Position0CompressorOutput::Ratio128 {
                main_kv: vec![1.0; 512],
                main_score: vec![2.0; 512],
            },
            ratio => panic!("unexpected ratio {ratio}"),
        };
        let mut expert_ids = entry.expert_ids.clone();
        if failure == Some(Failure::WrongRoute(entry.layer)) {
            expert_ids[0] = (expert_ids[0] + 1) % 256;
        }
        let mut ffn_sha = entry.capture.input_sha256.clone();
        if failure == Some(Failure::ShaDrift(entry.layer)) {
            ffn_sha = "0".repeat(64);
        }
        let mut ranges = expected_layer_state_ranges(native, entry.layer)?;
        match failure {
            Some(Failure::LayerRangeMissing(layer)) if layer == entry.layer => {
                ranges.pop();
            }
            Some(Failure::LayerRangeExtra(layer)) if layer == entry.layer => {
                ranges.push(native.arena_bytes..native.arena_bytes + 4);
            }
            Some(Failure::LayerRangeArbitrary(layer)) if layer == entry.layer => {
                ranges = vec![4..8];
            }
            _ => {}
        }
        Ok(Position0LayerReceipt {
            layer: if failure == Some(Failure::WrongLayer(entry.layer)) {
                entry.layer.saturating_add(1)
            } else {
                entry.layer
            },
            payloads_verified: failure != Some(Failure::UnverifiedLayer(entry.layer)),
            route_source: entry.route_source.clone(),
            expert_ids,
            route_weights: entry.route_weights.clone(),
            ffn_activation_f32_le_sha256: ffn_sha,
            moe_output_bf16_sha256: entry.capture.k1_moe_output_sha256.clone(),
            moe_branch_f32_le_sha256: entry.reference.moe_branch_f32_le_sha256.clone(),
            layer_output_f32_le_sha256: entry.reference.layer_output_f32_le_sha256.clone(),
            window_kv_bf16: vec![0; 512],
            compressor_output,
            state_ranges_written: ranges,
        })
    }

    fn fixture_final_receipt(
        native: &NativeState,
        final_section: &Position0Final,
        failure: Option<Failure>,
    ) -> Result<Position0FinalReceipt, Position0WholeTokenError> {
        let mut ranges = expected_final_state_ranges(native)?;
        match failure {
            Some(Failure::FinalRangeMissing) => {
                ranges.pop();
            }
            Some(Failure::FinalRangeExtra) => {
                ranges.push(native.arena_bytes..native.arena_bytes + 4);
            }
            Some(Failure::FinalRangeArbitrary) => ranges = vec![4..8],
            _ => {}
        }
        Ok(Position0FinalReceipt {
            predicted_token_id: if failure == Some(Failure::WrongFinal) {
                final_section.expected_output_token_id + 1
            } else {
                final_section.expected_output_token_id
            },
            payloads_verified: true,
            normalized_f32_le_sha256: final_section.normalized_f32_le_sha256.clone(),
            gpu_head_sha256: final_section.gpu_head_sha256.clone(),
            hc_streams_bf16: vec![0x3f00; 4 * 4096],
            state_ranges_written: ranges,
        })
    }

    fn manifest() -> Position0WholeTokenManifest {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
        );
        Position0WholeTokenManifest::load(&path).unwrap()
    }

    #[test]
    fn success_orders_all_stages_and_commits_both_banks() {
        let manifest = manifest();
        let mut state = DecoderStateV1::new(16, 0).unwrap();
        let mut execution = FixtureExecution::new(&state, None);
        let outcome = run_bos_transaction(&manifest, &mut state, &mut execution).unwrap();
        assert_eq!(
            outcome.token.predicted_token_id,
            manifest.expected_output_token_id
        );
        assert_eq!(state.position, 1);
        assert_eq!(state.commit_epoch, 1);
        assert_eq!(state.active_fixed_bank, 1);
        assert_eq!(execution.epoch, 1);
        assert_eq!(execution.bank, 1);
        let mut expected = vec!["submit:embedding".to_string()];
        expected.extend((0..43).map(|layer| format!("submit:L{layer}")));
        expected.extend([
            "submit:final".into(),
            "wait".into(),
            "finish".into(),
            "prepare".into(),
            "publish".into(),
        ]);
        assert_eq!(execution.events, expected);
        assert_eq!(execution.host_waits, 1);
    }

    #[test]
    fn l0_l42_final_and_wrong_final_fail_without_committing_either_side() {
        let manifest = manifest();
        for failure in [
            Failure::Layer(0),
            Failure::Layer(42),
            Failure::Final,
            Failure::WrongFinal,
        ] {
            let mut state = DecoderStateV1::new(16, 0).unwrap();
            let snapshot = state.clone();
            let mut execution = FixtureExecution::new(&state, Some(failure));
            assert!(run_bos_transaction(&manifest, &mut state, &mut execution).is_err());
            assert_eq!(state, snapshot);
            assert_eq!(execution.epoch, 0);
            assert_eq!(execution.bank, 0);
            assert_eq!(
                execution.events.last().map(String::as_str),
                Some("rollback")
            );
            assert_eq!(
                execution.host_waits, 1,
                "每条失败路径只能在 token 末尾 wait 或 abort-drain 一次"
            );
        }
    }

    #[test]
    fn segmented_layer_failure_aborts_once_before_rollback_and_never_publishes() {
        let manifest = manifest();
        let mut state = DecoderStateV1::new(16, 0).unwrap();
        let snapshot = state.clone();
        let mut execution = FixtureExecution::new(&state, Some(Failure::Layer(12)));
        assert!(run_bos_transaction(&manifest, &mut state, &mut execution).is_err());
        assert_eq!(state, snapshot);
        assert_eq!(execution.epoch, 0);
        assert_eq!(execution.bank, 0);
        assert_eq!(execution.host_waits, 1);
        assert_eq!(
            execution.events,
            (std::iter::once("submit:embedding".to_string())
                .chain((0..=12).map(|layer| format!("submit:L{layer}")))
                .chain(["abort-wait".into(), "rollback".into()]))
            .collect::<Vec<_>>()
        );
        assert!(!execution.events.iter().any(|event| event == "publish"));
    }

    #[test]
    fn post_wait_receipt_failure_does_not_wait_twice_or_publish() {
        let manifest = manifest();
        let mut state = DecoderStateV1::new(16, 0).unwrap();
        let snapshot = state.clone();
        let mut execution = FixtureExecution::new(&state, Some(Failure::ShaDrift(28)));
        assert!(run_bos_transaction(&manifest, &mut state, &mut execution).is_err());
        assert_eq!(state, snapshot);
        assert_eq!(execution.epoch, 0);
        assert_eq!(execution.bank, 0);
        assert_eq!(execution.host_waits, 1);
        assert_eq!(
            execution
                .events
                .iter()
                .filter(|event| event.as_str() == "wait")
                .count(),
            1
        );
        assert!(!execution.events.iter().any(|event| event == "abort-wait"));
        assert!(!execution.events.iter().any(|event| event == "publish"));
    }

    #[test]
    fn receipt_drift_and_unverified_payload_fail_closed() {
        let manifest = manifest();
        for failure in [
            Failure::UnverifiedLayer(12),
            Failure::WrongRoute(28),
            Failure::WrongLayer(42),
        ] {
            let mut state = DecoderStateV1::new(16, 0).unwrap();
            let snapshot = state.clone();
            let mut execution = FixtureExecution::new(&state, Some(failure));
            assert!(run_bos_transaction(&manifest, &mut state, &mut execution).is_err());
            assert_eq!(state, snapshot);
            assert_eq!(execution.epoch, 0);
            assert_eq!(execution.bank, 0);
        }
    }

    #[test]
    fn gpu_submit_finish_or_prepare_failure_restores_both_sides() {
        let manifest = manifest();
        for failure in [Failure::Submit, Failure::Finish, Failure::Prepare] {
            let mut state = DecoderStateV1::new(16, 0).unwrap();
            let snapshot = state.clone();
            let mut execution = FixtureExecution::new(&state, Some(failure));
            assert!(run_bos_transaction(&manifest, &mut state, &mut execution).is_err());
            assert_eq!(state, snapshot);
            assert_eq!(execution.epoch, 0);
            assert_eq!(execution.bank, 0);
            assert_eq!(
                execution.events.last().map(String::as_str),
                Some("rollback")
            );
        }
    }

    #[test]
    fn exact_position0_write_sets_match_host_transaction_layout() {
        let state = DecoderStateV1::new(16, 0).unwrap();
        for &layer in &FULL_DEPTH_LAYERS {
            let ranges = expected_layer_state_ranges(&state.native, layer).unwrap();
            let ratio = state
                .native
                .kv
                .iter()
                .find(|entry| entry.layer == layer)
                .unwrap()
                .compress_ratio;
            assert_eq!(
                ranges.len(),
                match ratio {
                    0 => 1,
                    4 => 5,
                    128 => 3,
                    _ => unreachable!(),
                }
            );
            assert!(ranges.iter().all(|range| {
                range.start < range.end
                    && range.start % 4 == 0
                    && range.end % 4 == 0
                    && range.end <= state.native.arena_bytes
            }));
            if ratio == 4 {
                let indexer = state
                    .native
                    .indexers
                    .iter()
                    .find(|entry| entry.layer == layer)
                    .unwrap();
                let cache = whole_slice_range(&indexer.kv_cache, "test indexer cache").unwrap();
                assert!(ranges
                    .iter()
                    .all(|range| range.end <= cache.start || range.start >= cache.end));
            }
        }
        assert_eq!(
            expected_final_state_ranges(&state.native).unwrap(),
            vec![whole_slice_range(&state.native.hc.streams, "test HC").unwrap()]
        );
    }

    #[test]
    fn finish_receipts_before_token_wait_is_rejected() {
        let manifest = manifest();
        let state = DecoderStateV1::new(16, 0).unwrap();
        let mut execution = FixtureExecution::new(&state, None);
        execution.begin(0).unwrap();
        assert!(matches!(
            execution.finish_receipts(&manifest),
            Err(Position0WholeTokenError::Backend { .. })
        ));
        assert_eq!(execution.epoch, 0);
        assert_eq!(execution.bank, 0);
    }

    #[test]
    fn post_wait_sha_drift_fails_closed() {
        let manifest = manifest();
        let mut state = DecoderStateV1::new(16, 0).unwrap();
        let snapshot = state.clone();
        let mut execution = FixtureExecution::new(&state, Some(Failure::ShaDrift(28)));
        assert_eq!(
            run_bos_transaction(&manifest, &mut state, &mut execution),
            Err(Position0WholeTokenError::NumericReceipt(28))
        );
        assert_eq!(state, snapshot);
        assert_eq!(execution.epoch, 0);
        assert_eq!(execution.bank, 0);
    }

    #[test]
    fn layer_state_range_missing_extra_or_arbitrary_four_bytes_is_rejected() {
        let manifest = manifest();
        for failure in [
            Failure::LayerRangeMissing(12),
            Failure::LayerRangeExtra(12),
            Failure::LayerRangeArbitrary(12),
        ] {
            let mut state = DecoderStateV1::new(16, 0).unwrap();
            let snapshot = state.clone();
            let mut execution = FixtureExecution::new(&state, Some(failure));
            assert_eq!(
                run_bos_transaction(&manifest, &mut state, &mut execution),
                Err(Position0WholeTokenError::MissingDirtyRange(12))
            );
            assert_eq!(state, snapshot);
            assert_eq!(execution.epoch, 0);
            assert_eq!(execution.bank, 0);
        }
    }

    #[test]
    fn final_state_range_missing_extra_or_arbitrary_four_bytes_is_rejected() {
        let manifest = manifest();
        for failure in [
            Failure::FinalRangeMissing,
            Failure::FinalRangeExtra,
            Failure::FinalRangeArbitrary,
        ] {
            let mut state = DecoderStateV1::new(16, 0).unwrap();
            let snapshot = state.clone();
            let mut execution = FixtureExecution::new(&state, Some(failure));
            assert_eq!(
                run_bos_transaction(&manifest, &mut state, &mut execution),
                Err(Position0WholeTokenError::FinalStateRange)
            );
            assert_eq!(state, snapshot);
            assert_eq!(execution.epoch, 0);
            assert_eq!(execution.bank, 0);
        }
    }

    #[test]
    fn missing_backend_capability_is_explicit() {
        let capabilities = Position0BackendCapabilities {
            embedding: true,
            all_layers: false,
            final_head: true,
            payload_sha256: false,
            route_receipts: true,
            position0_state_outputs: true,
        };
        assert_eq!(
            ensure_backend_capabilities(capabilities),
            Err(Position0WholeTokenError::BackendUnavailable(
                "all_layers,payload_sha256".into()
            ))
        );
    }

    #[test]
    fn completion_ticket_requires_current_candidate_and_exactly_one_host_wait() {
        let valid = Position0BackendCompletion {
            base_epoch: 7,
            candidate_bank: 1,
            final_compute_value: 44,
            token_host_waits: 1,
        };
        assert_eq!(validate_backend_completion(valid, 7, 1), Ok(()));
        for invalid in [
            Position0BackendCompletion {
                base_epoch: 6,
                ..valid
            },
            Position0BackendCompletion {
                candidate_bank: 0,
                ..valid
            },
            Position0BackendCompletion {
                final_compute_value: 0,
                ..valid
            },
            Position0BackendCompletion {
                token_host_waits: 0,
                ..valid
            },
            Position0BackendCompletion {
                token_host_waits: 2,
                ..valid
            },
        ] {
            assert!(matches!(
                validate_backend_completion(invalid, 7, 1),
                Err(Position0WholeTokenError::AtomicInvariant(_))
            ));
        }
    }
}
