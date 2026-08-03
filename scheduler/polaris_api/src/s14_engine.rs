use crate::{
    EngineChatRequest, EngineDelta, EngineDone, EngineError, EngineErrorKind, EngineEvent,
    EngineEventSender, FinishReason, ResidentChatBackend,
};
use sha2::{Digest, Sha256};
use ssd_inference::s14_runtime::{S14Runtime, S14Session};
use std::{fs, path::Path};
use tokenizers::Tokenizer;

pub const S14_N8_EVIDENCE_SHA256: &str =
    "8096d5a8798c840fc7d7725aa3281c3d95790d834452625b52739f1e69621dc8";
pub const OFFICIAL_CHAT_ENCODING_REVISION: &str = "7872f01b1d1fe23eabc4c98b48bffcef5a386062";
pub const OFFICIAL_CHAT_ENCODING_SHA256: &str =
    "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6";
pub const S14_TOKENIZER_SHA256: &str =
    "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf";
pub const DEFAULT_S14_TOKENIZER_PATH: &str = r"D:\models\Polaris-S14\tokenizer.json";
pub const DEFAULT_S14_N8_EVIDENCE_PATH: &str =
    r"D:\project\大模型ssd化\.tmp-polaris-tests\n8-production-20260803-proxy.stdout.log";

const S14_VOCAB_SIZE: usize = 129_280;
const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
const USER_TOKEN: &str = "<｜User｜>";
const ASSISTANT_TOKEN: &str = "<｜Assistant｜>";
const THINKING_START_TOKEN: &str = "<think>";
const THINKING_END_TOKEN: &str = "</think>";
const DSML_TOKEN: &str = "｜DSML｜";
const LATEST_REMINDER_TOKEN: &str = "<｜latest_reminder｜>";
const EOS_TOKEN_ID: u32 = 1;
const OFFICIAL_ENCODING_SOURCE: &str =
    include_str!("../../../fast16/research/polaris_meridian_v1/s14_chat_encoding/encoding_dsv4.py");

/// 聊天模板/tokenizer 与 S14 runtime 之间的窄边界。
///
/// `decode_completion` 必须返回整个 completion token 前缀的追加式 UTF-8 文本。
pub trait S14ChatCodec: 'static {
    fn encode_chat(&mut self, request: &EngineChatRequest) -> Result<Vec<u32>, EngineError>;
    fn decode_completion(&mut self, token_ids: &[u32]) -> Result<String, EngineError>;
    fn is_eos(&self, token_id: u32) -> bool;
}

/// 官方 DeepSeek-V4 encoding 的最小 Rust production profile。
///
/// 当前协议层尚未表达官方 reasoning/tool-call 结构，因此只冻结无工具的
/// `thinking_mode="thinking"` + `reasoning_effort="low"` forced-prefill。工具、name、
/// tool role 或保留标记注入均 fail-closed，不会降级为猜测模板。
pub struct DeepSeekV4ChatCodec {
    tokenizer: Tokenizer,
}

impl DeepSeekV4ChatCodec {
    pub fn load_production(path: impl AsRef<Path>) -> Result<Self, EngineError> {
        verify_official_encoding_source()?;
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|error| {
            EngineError::runtime_unavailable(format!(
                "读取 S14 tokenizer 失败 {}: {error}",
                path.display()
            ))
        })?;
        let fingerprint = sha256_hex(&bytes);
        if fingerprint != S14_TOKENIZER_SHA256 {
            return Err(EngineError::runtime_unavailable(format!(
                "S14 tokenizer SHA-256 漂移: 期望 {S14_TOKENIZER_SHA256}，实际 {fingerprint}"
            )));
        }
        let tokenizer = Tokenizer::from_file(path).map_err(|error| {
            EngineError::runtime_unavailable(format!(
                "加载 S14 tokenizer 失败 {}: {error}",
                path.display()
            ))
        })?;
        if tokenizer.get_vocab_size(true) != S14_VOCAB_SIZE {
            return Err(EngineError::runtime_unavailable(format!(
                "S14 tokenizer vocab 漂移: 期望 {S14_VOCAB_SIZE}，实际 {}",
                tokenizer.get_vocab_size(true)
            )));
        }
        for (token, expected_id) in [
            (BOS_TOKEN, 0),
            (EOS_TOKEN, 1),
            (USER_TOKEN, 128_803),
            (ASSISTANT_TOKEN, 128_804),
            (THINKING_START_TOKEN, 128_821),
            (THINKING_END_TOKEN, 128_822),
            (DSML_TOKEN, 128_825),
            (LATEST_REMINDER_TOKEN, 128_828),
        ] {
            let actual_id = tokenizer.token_to_id(token);
            let encoded = tokenizer
                .encode(token, false)
                .map_err(|error| codec_unavailable(format!("协议 token 编码失败: {error}")))?;
            if actual_id != Some(expected_id) || encoded.get_ids() != [expected_id] {
                return Err(codec_unavailable(format!(
                    "S14 tokenizer 协议 token 漂移: {token:?} 期望 {expected_id}，实际 {actual_id:?}/{:?}",
                    encoded.get_ids()
                )));
            }
        }
        Ok(Self { tokenizer })
    }
}

impl S14ChatCodec for DeepSeekV4ChatCodec {
    fn encode_chat(&mut self, request: &EngineChatRequest) -> Result<Vec<u32>, EngineError> {
        let prompt = render_official_forced_prefill(request)?;
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| codec_invalid(format!("S14 tokenizer 编码失败: {error}")))?;
        let ids = encoded.get_ids().to_vec();
        if ids.first() != Some(&0) || ids.iter().any(|&id| id as usize >= S14_VOCAB_SIZE) {
            return Err(codec_invalid(
                "S14 forced-prefill token 序列未以 BOS 0 开始或含越界 token",
            ));
        }
        Ok(ids)
    }

    fn decode_completion(&mut self, token_ids: &[u32]) -> Result<String, EngineError> {
        let visible = token_ids.strip_suffix(&[EOS_TOKEN_ID]).unwrap_or(token_ids);
        self.tokenizer
            .decode(visible, false)
            .map_err(|error| codec_invalid(format!("S14 tokenizer 解码失败: {error}")))
    }

    fn is_eos(&self, token_id: u32) -> bool {
        token_id == EOS_TOKEN_ID
    }
}

/// 2026-08-03 production N=8 数值真门的冻结回执。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14N8Evidence {
    pub max_position_exclusive: u32,
    pub output_tokens: u32,
    pub online_top6_routes: u32,
    pub dynamic_physical_ranges: u32,
    pub cpu_compute_fallbacks: u32,
    pub final_commit_epoch: u32,
    pub log_sha256: String,
}

/// 只有完整 N=8 数值真门与冻结日志哈希逐项一致时才可构造该值。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedS14NumericalGate {
    max_position_exclusive: u32,
    evidence: String,
}

impl VerifiedS14NumericalGate {
    /// 只用于在 loopback 上产生下一份冻结数值证据。调用者必须
    /// 将服务限制为单请求并在请求后关闭；这不放松 runtime 的
    /// Range proof/SHA、Vulkan 计算或 host/device commit 门。
    pub fn live_n12_evidence_probe() -> Self {
        Self {
            max_position_exclusive: 12,
            evidence: "live-n12-evidence-probe-loopback-only".to_owned(),
        }
    }

    /// 只用于一次 loopback 第二轮 16-token 验收。调用方必须把服务
    /// 限制为单请求并在请求后关闭；26 是冻结第二轮 prompt 的精确
    /// `(prompt.len() - 1) + max_tokens`，不是可配置的任意放宽。
    pub fn live_n26_second_turn_evidence_probe() -> Self {
        Self {
            max_position_exclusive: 26,
            evidence: "live-n26-second-turn-evidence-probe-loopback-only".to_owned(),
        }
    }

    pub fn from_n8_evidence_file(path: impl AsRef<Path>) -> Result<Self, EngineError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|error| {
            EngineError::runtime_unavailable(format!(
                "读取 S14 N=8 数值证据失败 {}: {error}",
                path.display()
            ))
        })?;
        let fingerprint = sha256_hex(&bytes);
        if fingerprint != S14_N8_EVIDENCE_SHA256 {
            return Err(EngineError::runtime_unavailable(format!(
                "S14 N=8 数值证据 SHA-256 漂移: 期望 {S14_N8_EVIDENCE_SHA256}，实际 {fingerprint}"
            )));
        }
        let text = std::str::from_utf8(&bytes).map_err(|error| {
            EngineError::runtime_unavailable(format!("S14 N=8 数值证据不是 UTF-8: {error}"))
        })?;
        verify_n8_evidence_text(text)?;
        Self::from_n8_evidence(S14N8Evidence {
            max_position_exclusive: 8,
            output_tokens: 8,
            online_top6_routes: 344,
            dynamic_physical_ranges: 12_384,
            cpu_compute_fallbacks: 0,
            final_commit_epoch: 8,
            log_sha256: fingerprint,
        })
        .map(|mut gate| {
            gate.evidence = format!("{}#sha256={}", path.display(), gate.evidence);
            gate
        })
    }

    pub fn from_n8_evidence(evidence: S14N8Evidence) -> Result<Self, EngineError> {
        let hash = evidence.log_sha256.trim().to_ascii_lowercase();
        let valid = evidence.max_position_exclusive == 8
            && evidence.output_tokens == 8
            && evidence.online_top6_routes == 344
            && evidence.dynamic_physical_ranges == 12_384
            && evidence.cpu_compute_fallbacks == 0
            && evidence.final_commit_epoch == 8
            && hash == S14_N8_EVIDENCE_SHA256;
        if !valid {
            return Err(EngineError::runtime_unavailable(
                "S14 N=8 数值门证据无效：必须精确匹配 8 tokens、344 routes、12384 ranges、0 fallback、commit_epoch=8 与冻结日志 SHA-256",
            ));
        }
        Ok(Self {
            max_position_exclusive: evidence.max_position_exclusive,
            evidence: format!("n8-production-20260803-proxy:{hash}"),
        })
    }

    pub fn max_position_exclusive(&self) -> u32 {
        self.max_position_exclusive
    }

    pub fn evidence(&self) -> &str {
        &self.evidence
    }
}

#[derive(Clone, Debug)]
pub struct S14RuntimeChatConfig {
    pub max_seq_len: u32,
    pub default_max_tokens: u32,
    pub numerical_gate: VerifiedS14NumericalGate,
}

/// 由 resident worker 独占的 production runtime + codec。
pub struct S14RuntimeChatBackend<C> {
    runtime: S14Runtime,
    codec: C,
    config: S14RuntimeChatConfig,
}

impl<C: S14ChatCodec> S14RuntimeChatBackend<C> {
    pub fn new(
        runtime: S14Runtime,
        codec: C,
        config: S14RuntimeChatConfig,
    ) -> Result<Self, EngineError> {
        if config.max_seq_len == 0 || config.default_max_tokens == 0 {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "S14 chat max_seq_len/default_max_tokens 必须大于 0",
            ));
        }
        Ok(Self {
            runtime,
            codec,
            config,
        })
    }

    fn run_request(
        &mut self,
        request: EngineChatRequest,
        events: &EngineEventSender,
    ) -> Result<(), EngineError> {
        if request.tools.is_some() || request.tool_choice.is_some() {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "S14 官方工具调用编码尚未接入，拒绝伪造 tool_calls",
            ));
        }
        if request.temperature.is_some_and(|value| value != 0.0) {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "当前 S14 runtime 仅有 greedy argmax；temperature 只能省略或设为 0",
            ));
        }
        let prompt = self.codec.encode_chat(&request)?;
        if prompt.is_empty() {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "S14 chat codec 生成了空 prompt",
            ));
        }
        let max_tokens = request.max_tokens.unwrap_or(self.config.default_max_tokens);
        let prompt_prefill_positions = prompt.len().saturating_sub(1);
        let required_positions = u32::try_from(prompt_prefill_positions)
            .ok()
            .and_then(|prefill| prefill.checked_add(max_tokens))
            .ok_or_else(|| {
                EngineError::new(
                    EngineErrorKind::InvalidRequest,
                    "prompt + max_tokens 长度溢出",
                )
            })?;
        if required_positions > self.config.max_seq_len
            || required_positions > self.config.numerical_gate.max_position_exclusive()
        {
            return Err(EngineError::unsupported_position(
                self.config.numerical_gate.max_position_exclusive(),
                format!(
                    "请求最多需要 {required_positions} 个 position，但当前权威数值门只覆盖 [0,{})",
                    self.config.numerical_gate.max_position_exclusive()
                ),
            ));
        }

        let mut session = self
            .runtime
            .new_session(prompt[0], self.config.max_seq_len)
            .map_err(|error| runtime_error(0, error))?;
        let result = self.run_session(&mut session, &prompt, max_tokens, &request.stop, events);
        let cleanup = session.destroy().map_err(|error| {
            let mut mapped = runtime_error(0, error);
            mapped.message = format!("S14 session 清理失败: {}", mapped.message);
            mapped
        });
        match (result, cleanup) {
            (Err(mut error), Err(cleanup)) => {
                error.message = format!("{}；同时 {}", error.message, cleanup.message);
                Err(error)
            }
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    fn run_session(
        &mut self,
        session: &mut S14Session,
        prompt: &[u32],
        max_tokens: u32,
        stops: &[String],
        events: &EngineEventSender,
    ) -> Result<(), EngineError> {
        for &next_prompt_token in prompt.iter().skip(1) {
            if events.is_closed() {
                return Ok(());
            }
            let position = session.position();
            let output = self
                .runtime
                .step_with_next_input(session, Some(next_prompt_token))
                .map_err(|error| runtime_error(position, error))?;
            eprintln!(
                "s14_api_step phase=prefill position={} input_token={} output_token={} commit_epoch={} routes={} physical_ranges={} elapsed_ms={:.3}",
                output.position,
                output.input_token_id,
                output.predicted_token_id,
                output.commit_epoch,
                output.online_top6_routes,
                output.dynamic_physical_ranges,
                output.elapsed_ms,
            );
        }

        let mut completion_ids = Vec::with_capacity(max_tokens as usize);
        let mut emitted = String::new();
        for index in 0..max_tokens {
            if events.is_closed() {
                return Ok(());
            }
            let position = session.position();
            let output = self
                .runtime
                .step(session)
                .map_err(|error| runtime_error(position, error))?;
            eprintln!(
                "s14_api_step phase=generation position={} input_token={} output_token={} commit_epoch={} routes={} physical_ranges={} elapsed_ms={:.3}",
                output.position,
                output.input_token_id,
                output.predicted_token_id,
                output.commit_epoch,
                output.online_top6_routes,
                output.dynamic_physical_ranges,
                output.elapsed_ms,
            );
            completion_ids.push(output.predicted_token_id);
            let decoded = self.codec.decode_completion(&completion_ids)?;
            if !decoded.starts_with(&emitted) {
                return Err(EngineError::new(
                    EngineErrorKind::Internal,
                    "S14 codec 输出不是追加式前缀；拒绝发布不可回滚文本",
                ));
            }

            let stop_at = stops.iter().filter_map(|stop| decoded.find(stop)).min();
            let eos = self.codec.is_eos(output.predicted_token_id);
            let length = index + 1 == max_tokens;
            let finish = if stop_at.is_some() || eos {
                Some(FinishReason::Stop)
            } else if length {
                Some(FinishReason::Length)
            } else {
                None
            };
            let visible_len = match (stop_at, finish) {
                (Some(offset), _) => offset,
                (None, Some(_)) => decoded.len(),
                (None, None) => stable_visible_len(&decoded, stops),
            };
            if visible_len < emitted.len() || !decoded.is_char_boundary(visible_len) {
                return Err(EngineError::new(
                    EngineErrorKind::Internal,
                    "S14 stop/codec 边界要求撤回已发布文本；拒绝继续",
                ));
            }
            if visible_len > emitted.len() {
                let delta = decoded[emitted.len()..visible_len].to_owned();
                emitted.push_str(&delta);
                if events
                    .blocking_send(Ok(EngineEvent::Delta(EngineDelta {
                        text: delta,
                        token_id: Some(output.predicted_token_id),
                    })))
                    .is_err()
                {
                    return Ok(());
                }
            }
            if let Some(finish_reason) = finish {
                let _ = events.blocking_send(Ok(EngineEvent::Done(EngineDone {
                    finish_reason,
                    prompt_tokens: Some(prompt.len() as u64),
                    completion_tokens: Some(completion_ids.len() as u64),
                })));
                return Ok(());
            }
        }
        Err(EngineError::new(
            EngineErrorKind::Internal,
            "S14 generation 循环异常退出且没有 Done",
        ))
    }
}

impl<C: S14ChatCodec> ResidentChatBackend for S14RuntimeChatBackend<C> {
    fn run_chat(
        &mut self,
        request: EngineChatRequest,
        events: &EngineEventSender,
    ) -> Result<(), EngineError> {
        self.run_request(request, events)
    }
}

/// Resident K=4 decoder 与 ChatEngine 之间的可恢复 checkpoint 身份。
/// SHA 由 production decoder 的 durable checkpoint 回执提供，API 层只做链式验收。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14ResidentK4Checkpoint {
    pub position: u32,
    pub commit_epoch: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct S14ResidentK4CommittedBlock {
    pub consumed: S14ResidentK4Checkpoint,
    pub committed: S14ResidentK4Checkpoint,
    pub token_ids: Vec<u32>,
    pub wall_ms: f64,
}

/// 每个 HTTP 请求独占的 K=4 continuation。实现必须消费上一块的
/// 真实 committed checkpoint，不能从 position0 重启或用 fixture 替换。
pub trait S14ResidentK4Request: Send {
    fn checkpoint(&self) -> &S14ResidentK4Checkpoint;

    fn execute_next_block(
        &mut self,
        remaining_tokens: u32,
    ) -> Result<S14ResidentK4CommittedBlock, EngineError>;

    fn close(self) -> Result<(), EngineError>;
}

/// 长驻 decoder 保有 Vulkan、paged arena、union banks 与 verified mapped store；
/// `begin_request` 只创建请求状态，不得新启动第二模型实例。
pub trait S14ResidentK4Decoder: Send + 'static {
    type Request: S14ResidentK4Request;

    fn begin_request(
        &mut self,
        prompt_token_ids: &[u32],
        max_seq_len: u32,
    ) -> Result<Self::Request, EngineError>;
}

/// ChatEngine 的 resident K=4 接线。此类型不提供 mock/fallback production
/// 实现；只有注入真实 `S14ResidentK4Decoder` 后才能构造。
pub struct S14ResidentK4ChatBackend<C, D> {
    codec: C,
    decoder: D,
    max_seq_len: u32,
    default_max_tokens: u32,
}

impl<C: S14ChatCodec, D: S14ResidentK4Decoder> S14ResidentK4ChatBackend<C, D> {
    pub fn new(
        codec: C,
        decoder: D,
        max_seq_len: u32,
        default_max_tokens: u32,
    ) -> Result<Self, EngineError> {
        if max_seq_len == 0 || default_max_tokens == 0 {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "resident K4 max_seq_len/default_max_tokens 必须大于0",
            ));
        }
        Ok(Self {
            codec,
            decoder,
            max_seq_len,
            default_max_tokens,
        })
    }

    fn run_request(
        &mut self,
        request: EngineChatRequest,
        events: &EngineEventSender,
    ) -> Result<(), EngineError> {
        if request.tools.is_some()
            || request.tool_choice.is_some()
            || request.temperature.is_some_and(|value| value != 0.0)
        {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "resident K4 当前只接受无tools、greedy temperature=0 请求",
            ));
        }
        let prompt = self.codec.encode_chat(&request)?;
        let max_tokens = request.max_tokens.unwrap_or(self.default_max_tokens);
        if prompt.is_empty()
            || max_tokens == 0
            || u32::try_from(prompt.len())
                .ok()
                .and_then(|value| value.checked_add(max_tokens))
                .is_none_or(|required| required > self.max_seq_len)
        {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "resident K4 prompt/max_tokens 越界",
            ));
        }

        let mut resident = self.decoder.begin_request(&prompt, self.max_seq_len)?;
        let result = self.run_blocks(&mut resident, prompt.len(), max_tokens, &request.stop, events);
        let cleanup = resident.close();
        match (result, cleanup) {
            (Err(mut error), Err(cleanup)) => {
                error.message = format!("{}; 同时 resident K4 close 失败: {}", error.message, cleanup.message);
                Err(error)
            }
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    fn run_blocks<R: S14ResidentK4Request>(
        &mut self,
        resident: &mut R,
        prompt_tokens: usize,
        max_tokens: u32,
        stops: &[String],
        events: &EngineEventSender,
    ) -> Result<(), EngineError> {
        let mut completion_ids = Vec::with_capacity(max_tokens as usize);
        let mut emitted = String::new();
        while completion_ids.len() < max_tokens as usize {
            let expected = resident.checkpoint().clone();
            let remaining = max_tokens - completion_ids.len() as u32;
            let block = resident.execute_next_block(remaining)?;
            if block.consumed != expected
                || block.committed.position <= block.consumed.position
                || block.committed.commit_epoch <= block.consumed.commit_epoch
                || block.token_ids.is_empty()
                || block.token_ids.len() > 4
                || block.token_ids.len() > remaining as usize
                || block.committed.position - block.consumed.position
                    != block.token_ids.len() as u32
                || block.committed.sha256.len() != 64
                || block.committed != *resident.checkpoint()
            {
                return Err(EngineError::new(
                    EngineErrorKind::Internal,
                    "resident K4 block checkpoint/commit/token 链漂移",
                ));
            }

            for token_id in block.token_ids {
                completion_ids.push(token_id);
                let decoded = self.codec.decode_completion(&completion_ids)?;
                if !decoded.starts_with(&emitted) {
                    return Err(EngineError::new(
                        EngineErrorKind::Internal,
                        "resident K4 codec 输出不是追加式前缀",
                    ));
                }
                let stop_at = stops.iter().filter_map(|stop| decoded.find(stop)).min();
                let eos = self.codec.is_eos(token_id);
                let length = completion_ids.len() == max_tokens as usize;
                let finish = if stop_at.is_some() || eos {
                    Some(FinishReason::Stop)
                } else if length {
                    Some(FinishReason::Length)
                } else {
                    None
                };
                let visible_len = match (stop_at, finish) {
                    (Some(offset), _) => offset,
                    (None, Some(_)) => decoded.len(),
                    (None, None) => stable_visible_len(&decoded, stops),
                };
                if visible_len > emitted.len() {
                    let delta = decoded[emitted.len()..visible_len].to_owned();
                    emitted.push_str(&delta);
                    if events
                        .blocking_send(Ok(EngineEvent::Delta(EngineDelta {
                            text: delta,
                            token_id: Some(token_id),
                        })))
                        .is_err()
                    {
                        return Ok(());
                    }
                }
                if let Some(finish_reason) = finish {
                    let _ = events.blocking_send(Ok(EngineEvent::Done(EngineDone {
                        finish_reason,
                        prompt_tokens: Some(prompt_tokens as u64),
                        completion_tokens: Some(completion_ids.len() as u64),
                    })));
                    return Ok(());
                }
            }
        }
        Err(EngineError::new(
            EngineErrorKind::Internal,
            "resident K4 生成循环异常退出",
        ))
    }
}

impl<C, D> ResidentChatBackend for S14ResidentK4ChatBackend<C, D>
where
    C: S14ChatCodec + Send + 'static,
    D: S14ResidentK4Decoder,
{
    fn run_chat(
        &mut self,
        request: EngineChatRequest,
        events: &EngineEventSender,
    ) -> Result<(), EngineError> {
        self.run_request(request, events)
    }
}

fn render_official_forced_prefill(request: &EngineChatRequest) -> Result<String, EngineError> {
    if request.tools.is_some() || request.tool_choice.is_some() {
        return Err(codec_invalid(
            "官方 DSML tool-call 结构尚未进入 EngineEvent，当前 profile 拒绝 tools",
        ));
    }
    let Some(last_user_index) = request
        .messages
        .iter()
        .rposition(|message| matches!(message.role.as_str(), "user" | "developer"))
    else {
        return Err(codec_invalid(
            "官方 forced-prefill 至少需要一条 user/developer 消息",
        ));
    };
    if !matches!(
        request.messages.last().map(|message| message.role.as_str()),
        Some("user" | "developer")
    ) {
        return Err(codec_invalid(
            "官方 forced-prefill 必须停在最后一条 user/developer 消息之后",
        ));
    }
    for (index, message) in request.messages.iter().enumerate() {
        if message.name.is_some() {
            return Err(codec_invalid(format!(
                "messages[{index}].name 尚无官方 DeepSeek-V4 编码合同"
            )));
        }
        if !matches!(
            message.role.as_str(),
            "system" | "developer" | "user" | "assistant"
        ) {
            return Err(codec_invalid(format!(
                "messages[{index}].role={} 无法无损映射到当前官方 profile",
                message.role
            )));
        }
        if [
            BOS_TOKEN,
            EOS_TOKEN,
            USER_TOKEN,
            ASSISTANT_TOKEN,
            LATEST_REMINDER_TOKEN,
            THINKING_START_TOKEN,
            THINKING_END_TOKEN,
            DSML_TOKEN,
            "<tool_result>",
            "</tool_result>",
        ]
        .iter()
        .any(|marker| message.content.contains(marker))
        {
            return Err(codec_invalid(format!(
                "messages[{index}].content 含 DeepSeek 保留协议标记"
            )));
        }
    }

    // 与官方 `_drop_thinking_messages` 一致：最后 user/developer 之前的 developer 被丢弃，
    // system/user/assistant 保留；当前 EngineChatMessage 不携带 reasoning_content。
    let messages: Vec<_> = request
        .messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            if index >= last_user_index
                || matches!(message.role.as_str(), "system" | "user" | "assistant")
            {
                Some(message)
            } else {
                None
            }
        })
        .collect();
    let mut prompt = String::from(BOS_TOKEN);
    for (index, message) in messages.iter().enumerate() {
        match message.role.as_str() {
            "system" => prompt.push_str(&message.content),
            "developer" => {
                if message.content.is_empty() {
                    return Err(codec_invalid("developer content 不能为空"));
                }
                prompt.push_str(USER_TOKEN);
                prompt.push_str(&message.content);
            }
            "user" => {
                prompt.push_str(USER_TOKEN);
                prompt.push_str(&message.content);
            }
            "assistant" => {
                prompt.push_str(&message.content);
                prompt.push_str(EOS_TOKEN);
            }
            _ => unreachable!("前置角色校验"),
        }

        let next_is_assistant = messages
            .get(index + 1)
            .is_some_and(|next| next.role == "assistant");
        if matches!(message.role.as_str(), "user" | "developer")
            && (index + 1 == messages.len() || next_is_assistant)
        {
            prompt.push_str(ASSISTANT_TOKEN);
            prompt.push_str(THINKING_START_TOKEN);
        }
    }
    if !prompt.starts_with(BOS_TOKEN)
        || !prompt.ends_with(&format!("{ASSISTANT_TOKEN}{THINKING_START_TOKEN}"))
        || prompt.matches(BOS_TOKEN).count() != 1
    {
        return Err(codec_invalid(
            "官方 DeepSeek forced-prefill 未闭合到 assistant thinking 边界",
        ));
    }
    Ok(prompt)
}

fn verify_official_encoding_source() -> Result<(), EngineError> {
    let normalized = OFFICIAL_ENCODING_SOURCE.replace("\r\n", "\n");
    let canonical = format!("{}\n", normalized.trim_end_matches('\n'));
    let actual = sha256_hex(canonical.as_bytes());
    if actual != OFFICIAL_CHAT_ENCODING_SHA256 {
        return Err(codec_unavailable(format!(
            "官方 DeepSeek chat encoding source 漂移: revision={OFFICIAL_CHAT_ENCODING_REVISION} 期望 {OFFICIAL_CHAT_ENCODING_SHA256}，实际 {actual}"
        )));
    }
    Ok(())
}

fn verify_n8_evidence_text(text: &str) -> Result<(), EngineError> {
    let trace_lines: Vec<_> = text
        .lines()
        .filter(|line| line.starts_with("trace=online_top6_dynamic_ranges "))
        .collect();
    if trace_lines.len() != 344
        || trace_lines.iter().any(|line| {
            !line.contains(" physical_ranges=36 ")
                || !line.contains(" rust_proof_sha_verified=true ")
                || !line.ends_with("cpu_compute_fallbacks=0")
        })
    {
        return Err(EngineError::runtime_unavailable(
            "S14 N=8 证据逐层 route/Range/proof/fallback 合同不闭合",
        ));
    }
    let expected_outputs = [5u32, 223, 939, 21, 695, 553, 1266, 16179];
    for (position, output) in expected_outputs.iter().enumerate() {
        let position_marker = format!(" position={position} ");
        let output_marker = format!(" output_token={output} ");
        let matches = text
            .lines()
            .filter(|line| {
                line.starts_with("status=pass ")
                    && line.contains(&position_marker)
                    && line.contains(" layers=43 ")
                    && line.contains(" online_top6_routes=43 ")
                    && line.contains(" dynamic_physical_ranges=1548 ")
                    && line.contains(" cpu_compute_fallbacks=0 ")
                    && line.contains(&output_marker)
            })
            .count();
        if matches != 1 {
            return Err(EngineError::runtime_unavailable(format!(
                "S14 N=8 证据 position{position}/output{output} 回执缺失或重复"
            )));
        }
    }
    let summary = text
        .lines()
        .filter(|line| {
            line.starts_with("status=pass mode=polaris_s14_production_paged_continuous_n8 ")
        })
        .collect::<Vec<_>>();
    let valid_summary = summary.len() == 1
        && [
            " positions=8 ",
            " output_tokens=[5, 223, 939, 21, 695, 553, 1266, 16179] ",
            " online_top6_routes=344 ",
            " dynamic_physical_ranges=12384 ",
            " rust_proof_sha_verified=true ",
            " cpu_compute_fallbacks=0 ",
            " final_commit_epoch=8 ",
            " ratio4_boundaries_committed=2 ",
            " ratio128_boundary_pending=true ",
            " position4_numeric_backend=compressed_indexer_exact ",
        ]
        .iter()
        .all(|field| summary[0].contains(field));
    if !valid_summary {
        return Err(EngineError::runtime_unavailable(
            "S14 N=8 最终数值门摘要不闭合",
        ));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn codec_unavailable(message: impl Into<String>) -> EngineError {
    EngineError::runtime_unavailable(message)
}

fn codec_invalid(message: impl Into<String>) -> EngineError {
    EngineError::new(EngineErrorKind::InvalidRequest, message)
}

fn stable_visible_len(text: &str, stops: &[String]) -> usize {
    let mut visible_len = text.find('\u{fffd}').unwrap_or(text.len());
    let mut held_suffix = 0usize;
    for stop in stops {
        for (prefix_len, _) in stop.char_indices().skip(1) {
            if prefix_len < stop.len() && text.ends_with(&stop[..prefix_len]) {
                held_suffix = held_suffix.max(prefix_len);
            }
        }
    }
    visible_len = visible_len.min(text.len().saturating_sub(held_suffix));
    visible_len
}

fn runtime_error(position: u32, error: impl std::fmt::Display) -> EngineError {
    EngineError {
        kind: EngineErrorKind::Internal,
        message: format!("S14 runtime position{position} 失败: {error}"),
        failed_position: Some(position),
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineChatMessage, POLARIS_MODEL_ID};

    #[test]
    fn old_n4_coverage_cannot_construct_the_n8_gate() {
        let error = VerifiedS14NumericalGate::from_n8_evidence(S14N8Evidence {
            max_position_exclusive: 4,
            output_tokens: 4,
            online_top6_routes: 172,
            dynamic_physical_ranges: 6_192,
            cpu_compute_fallbacks: 0,
            final_commit_epoch: 4,
            log_sha256: "old-n4".to_owned(),
        })
        .expect_err("旧 N=4 证据必须拒绝");
        assert_eq!(error.kind, EngineErrorKind::RuntimeUnavailable);
    }

    #[test]
    fn exact_production_n8_evidence_constructs_the_gate() {
        let gate = VerifiedS14NumericalGate::from_n8_evidence(S14N8Evidence {
            max_position_exclusive: 8,
            output_tokens: 8,
            online_top6_routes: 344,
            dynamic_physical_ranges: 12_384,
            cpu_compute_fallbacks: 0,
            final_commit_epoch: 8,
            log_sha256: S14_N8_EVIDENCE_SHA256.to_owned(),
        })
        .expect("冻结 N=8 证据应通过");
        assert_eq!(gate.max_position_exclusive(), 8);
        assert!(gate.evidence().contains(S14_N8_EVIDENCE_SHA256));
    }

    #[test]
    fn stop_prefix_is_withheld_until_disambiguated() {
        assert_eq!(stable_visible_len("hello<ST", &["<STOP>".to_owned()]), 5);
        assert_eq!(stable_visible_len("hello!", &["<STOP>".to_owned()]), 6);
    }

    #[test]
    fn frozen_official_source_hash_is_current() {
        verify_official_encoding_source().expect("官方 encoding 源码 SHA 应闭合");
    }

    #[test]
    fn simple_user_prompt_matches_frozen_official_forced_prefill() {
        let request = EngineChatRequest {
            model: POLARIS_MODEL_ID.to_owned(),
            messages: vec![EngineChatMessage {
                role: "user".to_owned(),
                content: "你好".to_owned(),
                name: None,
            }],
            max_tokens: Some(3),
            temperature: Some(0.0),
            stop: Vec::new(),
            tools: None,
            tool_choice: None,
        };
        let prompt = render_official_forced_prefill(&request).unwrap();
        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>你好<｜Assistant｜><think>"
        );
        assert_eq!(
            sha256_hex(prompt.as_bytes()),
            "5081d1f13730de5364bdcd23071f160aeb4792baf043f2848c05c13ac50abfcc"
        );
    }

    #[test]
    fn production_tokenizer_matches_first_preview_forced_prefill() {
        let mut codec = DeepSeekV4ChatCodec::load_production(DEFAULT_S14_TOKENIZER_PATH)
            .expect("正式 tokenizer SHA/协议 token 应闭合");
        let request = EngineChatRequest {
            model: POLARIS_MODEL_ID.to_owned(),
            messages: vec![EngineChatMessage {
                role: "user".to_owned(),
                content: "你好".to_owned(),
                name: None,
            }],
            max_tokens: Some(3),
            temperature: Some(0.0),
            stop: Vec::new(),
            tools: None,
            tool_choice: None,
        };
        assert_eq!(
            codec.encode_chat(&request).unwrap(),
            [0, 128_803, 30_594, 128_804, 128_821]
        );
    }

    #[test]
    fn production_n8_log_content_and_hash_construct_gate() {
        let gate = VerifiedS14NumericalGate::from_n8_evidence_file(DEFAULT_S14_N8_EVIDENCE_PATH)
            .expect("正式 N=8 日志内容/SHA 应闭合");
        assert_eq!(gate.max_position_exclusive(), 8);
        assert!(gate.evidence().contains(S14_N8_EVIDENCE_SHA256));
    }
}
