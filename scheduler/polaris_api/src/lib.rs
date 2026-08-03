//! Polaris S14 的最小 HTTP 协议适配层。
//!
//! 本 crate 只负责 OpenAI/Ollama 协议与模型事件之间的映射，不代理任何旧模型，
//! 也不自行生成 token。生产侧应实现 [`ChatEngine`]，并把持久 S14 runtime 的真实
//! token/结束/失败事件送入返回的 channel。

mod resident;
mod s14_engine;

pub use resident::{ResidentChatBackend, ResidentChatEngine};
pub use s14_engine::{
    DeepSeekV4ChatCodec, S14ChatCodec, S14N8Evidence, S14RuntimeChatBackend, S14RuntimeChatConfig,
    VerifiedS14NumericalGate, DEFAULT_S14_N8_EVIDENCE_PATH, DEFAULT_S14_TOKENIZER_PATH,
    OFFICIAL_CHAT_ENCODING_REVISION, S14_N8_EVIDENCE_SHA256,
};

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, StatusCode},
    response::sse::{Event, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::{
    convert::Infallible,
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub const POLARIS_MODEL_ID: &str = "Polaris-S14";

/// 引擎产生的事件流。`Done` 是唯一成功结束标志；channel 提前关闭视为失败。
pub type EngineEventReceiver = mpsc::Receiver<Result<EngineEvent, EngineError>>;
pub type EngineEventSender = mpsc::Sender<Result<EngineEvent, EngineError>>;
pub type EngineStartFuture<'a> =
    Pin<Box<dyn Future<Output = Result<EngineEventReceiver, EngineError>> + Send + 'a>>;

#[derive(Debug, Clone, PartialEq)]
pub struct EngineChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineChatRequest {
    pub model: String,
    pub messages: Vec<EngineChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stop: Vec<String>,
    /// 原样保留给后续 S14 官方 DSML 编码器；协议层不解释或伪造工具调用。
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineDelta {
    /// 必须来自引擎真实解码结果；适配层不合成文本。
    pub text: String,
    pub token_id: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

impl FinishReason {
    fn openai_name(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
        }
    }

    fn ollama_name(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineDone {
    pub finish_reason: FinishReason,
    /// 未知时保持 `None`；协议层不会用 0 冒充真实计数。
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    Delta(EngineDelta),
    Done(EngineDone),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineErrorKind {
    InvalidRequest,
    ModelNotFound,
    RuntimeUnavailable,
    QueueFull,
    UnsupportedPosition,
    Internal,
    StreamIncomplete,
}

impl EngineErrorKind {
    fn code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::ModelNotFound => "model_not_found",
            Self::RuntimeUnavailable => "runtime_unavailable",
            Self::QueueFull => "queue_full",
            Self::UnsupportedPosition => "unsupported_position",
            Self::Internal => "engine_internal_error",
            Self::StreamIncomplete => "engine_stream_incomplete",
        }
    }

    fn status(self) -> StatusCode {
        match self {
            Self::InvalidRequest => StatusCode::BAD_REQUEST,
            Self::ModelNotFound => StatusCode::NOT_FOUND,
            Self::RuntimeUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::QueueFull => StatusCode::TOO_MANY_REQUESTS,
            Self::UnsupportedPosition => StatusCode::NOT_IMPLEMENTED,
            Self::Internal | Self::StreamIncomplete => StatusCode::BAD_GATEWAY,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError {
    pub kind: EngineErrorKind,
    pub message: String,
    pub failed_position: Option<u32>,
    pub retryable: bool,
}

impl EngineError {
    pub fn new(kind: EngineErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            failed_position: None,
            retryable: false,
        }
    }

    pub fn runtime_unavailable(message: impl Into<String>) -> Self {
        Self::new(EngineErrorKind::RuntimeUnavailable, message)
    }

    pub fn unsupported_position(position: u32, message: impl Into<String>) -> Self {
        Self {
            kind: EngineErrorKind::UnsupportedPosition,
            message: message.into(),
            failed_position: Some(position),
            retryable: false,
        }
    }

    fn incomplete() -> Self {
        Self::new(
            EngineErrorKind::StreamIncomplete,
            "模型事件流在 Done 之前关闭；本次生成未完成",
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineHealth {
    pub ready: bool,
    pub detail: String,
}

/// 持久模型 runtime 的插入边界。
///
/// 实现方必须逐事件发送真实模型结果，并且只有模型确实完成时才发送 `Done`。
/// 超出调用方冻结数值门的 position（当前 N=8 门之外）应发送带 `failed_position` 的
/// `EngineError`，不能伪装成 `Done`。
pub trait ChatEngine: Send + Sync + 'static {
    fn start_chat(&self, request: EngineChatRequest) -> EngineStartFuture<'_>;
    fn health(&self) -> EngineHealth;
}

/// 未接入持久 S14 runtime 时的安全默认引擎。它从不返回 token。
pub struct UnavailableEngine {
    reason: String,
}

impl UnavailableEngine {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl ChatEngine for UnavailableEngine {
    fn start_chat(&self, _request: EngineChatRequest) -> EngineStartFuture<'_> {
        let error = EngineError::runtime_unavailable(self.reason.clone());
        Box::pin(async move { Err(error) })
    }

    fn health(&self) -> EngineHealth {
        EngineHealth {
            ready: false,
            detail: self.reason.clone(),
        }
    }
}

#[derive(Clone)]
struct AppState {
    engine: Arc<dyn ChatEngine>,
}

#[derive(Debug, Deserialize)]
struct ProtocolMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    messages: Vec<ProtocolMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaOptions {
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    num_predict: Option<u32>,
    #[serde(default)]
    stop: Option<Value>,
}

fn default_ollama_stream() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct OllamaChatRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    messages: Vec<ProtocolMessage>,
    #[serde(default = "default_ollama_stream")]
    stream: bool,
    #[serde(default)]
    options: OllamaOptions,
    #[serde(default)]
    tools: Option<Value>,
}

/// 构建可嵌入测试或常驻进程的路由。
pub fn router(engine: Arc<dyn ChatEngine>) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/models", get(openai_models))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/api/tags", get(ollama_tags))
        .route("/api/chat", post(ollama_chat))
        .with_state(AppState { engine })
}

fn readiness_error(health: EngineHealth) -> Option<EngineError> {
    (!health.ready).then(|| {
        EngineError::runtime_unavailable(format!("S14 health 未 ready：{}", health.detail))
    })
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    engine: Arc<dyn ChatEngine>,
) -> std::io::Result<()> {
    axum::serve(listener, router(engine)).await
}

async fn health(State(state): State<AppState>) -> Response {
    let health = state.engine.health();
    let status = if health.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(json!({
            "model": POLARIS_MODEL_ID,
            "ready": health.ready,
            "detail": health.detail,
        })),
    )
        .into_response()
}

async fn openai_models(State(state): State<AppState>) -> Response {
    if let Some(error) = readiness_error(state.engine.health()) {
        return openai_error_response(error);
    }
    Json(json!({
        "object": "list",
        "data": [{
            "id": POLARIS_MODEL_ID,
            "object": "model",
            "created": unix_seconds(),
            "owned_by": "local",
        }]
    }))
    .into_response()
}

async fn ollama_tags(State(state): State<AppState>) -> Response {
    if let Some(error) = readiness_error(state.engine.health()) {
        return ollama_error_response(error);
    }
    Json(json!({
        "models": [{
            "name": POLARIS_MODEL_ID,
            "model": POLARIS_MODEL_ID,
            "modified_at": utc_now_rfc3339(),
        }]
    }))
    .into_response()
}

async fn openai_chat(
    State(state): State<AppState>,
    Json(request): Json<OpenAiChatRequest>,
) -> Response {
    let stream = request.stream;
    let engine_request = match openai_engine_request(request) {
        Ok(request) => request,
        Err(error) => return openai_error_response(error),
    };
    if let Some(error) = readiness_error(state.engine.health()) {
        return openai_error_response(error);
    }
    let model = engine_request.model.clone();
    let request_id = next_request_id("chatcmpl-polaris");
    let receiver = match state.engine.start_chat(engine_request).await {
        Ok(receiver) => receiver,
        Err(error) => return openai_error_response(error),
    };

    if stream {
        openai_stream_response(receiver, request_id, model)
    } else {
        match collect_completion(receiver).await {
            Ok((content, done)) => {
                let mut body = json!({
                    "id": request_id,
                    "object": "chat.completion",
                    "created": unix_seconds(),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": done.finish_reason.openai_name(),
                    }],
                });
                insert_openai_usage(&mut body, &done);
                Json(body).into_response()
            }
            Err(error) => openai_error_response(error),
        }
    }
}

async fn ollama_chat(
    State(state): State<AppState>,
    Json(request): Json<OllamaChatRequest>,
) -> Response {
    let stream = request.stream;
    let engine_request = match ollama_engine_request(request) {
        Ok(request) => request,
        Err(error) => return ollama_error_response(error),
    };
    if let Some(error) = readiness_error(state.engine.health()) {
        return ollama_error_response(error);
    }
    let model = engine_request.model.clone();
    let receiver = match state.engine.start_chat(engine_request).await {
        Ok(receiver) => receiver,
        Err(error) => return ollama_error_response(error),
    };

    if stream {
        ollama_stream_response(receiver, model)
    } else {
        match collect_completion(receiver).await {
            Ok((content, done)) => {
                let mut body = json!({
                    "model": model,
                    "created_at": utc_now_rfc3339(),
                    "message": {"role": "assistant", "content": content},
                    "done": true,
                    "done_reason": done.finish_reason.ollama_name(),
                });
                insert_ollama_usage(&mut body, &done);
                Json(body).into_response()
            }
            Err(error) => ollama_error_response(error),
        }
    }
}

fn openai_engine_request(request: OpenAiChatRequest) -> Result<EngineChatRequest, EngineError> {
    build_engine_request(
        request.model,
        request.messages,
        request.max_completion_tokens.or(request.max_tokens),
        request.temperature,
        request.stop,
        request.tools,
        request.tool_choice,
    )
}

fn ollama_engine_request(request: OllamaChatRequest) -> Result<EngineChatRequest, EngineError> {
    build_engine_request(
        request.model,
        request.messages,
        request.options.num_predict,
        request.options.temperature,
        request.options.stop,
        request.tools,
        None,
    )
}

fn build_engine_request(
    model: String,
    messages: Vec<ProtocolMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    stop: Option<Value>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
) -> Result<EngineChatRequest, EngineError> {
    if model != POLARIS_MODEL_ID {
        return Err(EngineError::new(
            EngineErrorKind::ModelNotFound,
            format!("未知模型 `{model}`；本服务只暴露 `{POLARIS_MODEL_ID}`"),
        ));
    }
    if messages.is_empty() {
        return Err(EngineError::new(
            EngineErrorKind::InvalidRequest,
            "messages 不能为空",
        ));
    }
    if max_tokens == Some(0) {
        return Err(EngineError::new(
            EngineErrorKind::InvalidRequest,
            "max_tokens/num_predict 必须大于 0",
        ));
    }
    if let Some(temperature) = temperature {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "temperature 必须是有限的非负数",
            ));
        }
    }

    let messages = messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| {
            if !matches!(
                message.role.as_str(),
                "system" | "developer" | "user" | "assistant" | "tool"
            ) {
                return Err(EngineError::new(
                    EngineErrorKind::InvalidRequest,
                    format!("messages[{index}].role 不受支持: `{}`", message.role),
                ));
            }
            Ok(EngineChatMessage {
                role: message.role,
                content: content_to_string(&message.content, index)?,
                name: message.name,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (tools, tool_choice) = normalize_tool_controls(tools, tool_choice);

    Ok(EngineChatRequest {
        model,
        messages,
        max_tokens,
        temperature,
        stop: parse_stop(stop)?,
        tools,
        tool_choice,
    })
}

/// Open WebUI 会在普通聊天请求中序列化空 `tools`，有时还会同时发送
/// `tool_choice: "auto"`。空工具集不代表请求了工具调用；将其规范化为物理无工具路径。
/// 非空工具定义仍原样保留，并由 S14 backend 在 DSML tool-call ABI 完成前 fail-closed。
fn normalize_tool_controls(
    tools: Option<Value>,
    tool_choice: Option<Value>,
) -> (Option<Value>, Option<Value>) {
    let tools = tools.filter(|value| match value {
        Value::Null => false,
        Value::Array(items) => !items.is_empty(),
        Value::Object(entries) => !entries.is_empty(),
        _ => true,
    });

    if tools.is_none() {
        let tool_choice = tool_choice.filter(|value| {
            !value.is_null()
                && !value
                    .as_str()
                    .is_some_and(|choice| matches!(choice, "auto" | "none"))
        });
        return (None, tool_choice);
    }

    if tool_choice
        .as_ref()
        .and_then(Value::as_str)
        .is_some_and(|choice| choice == "none")
    {
        return (None, None);
    }

    (tools, tool_choice.filter(|value| !value.is_null()))
}

fn content_to_string(value: &Value, index: usize) -> Result<String, EngineError> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_owned());
    }
    if value.is_null() {
        return Ok(String::new());
    }
    if let Some(parts) = value.as_array() {
        let mut text = Vec::new();
        for (part_index, part) in parts.iter().enumerate() {
            if let Some(value) = part.as_str() {
                text.push(value.to_owned());
                continue;
            }
            let kind = part.get("type").and_then(Value::as_str).unwrap_or("text");
            if !matches!(kind, "text" | "input_text") {
                return Err(EngineError::new(
                    EngineErrorKind::InvalidRequest,
                    format!(
                        "messages[{index}].content[{part_index}] 类型 `{kind}` 尚未接入 S14 runtime"
                    ),
                ));
            }
            let value = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                EngineError::new(
                    EngineErrorKind::InvalidRequest,
                    format!("messages[{index}].content[{part_index}].text 缺失"),
                )
            })?;
            text.push(value.to_owned());
        }
        return Ok(text.join("\n"));
    }
    Err(EngineError::new(
        EngineErrorKind::InvalidRequest,
        format!("messages[{index}].content 必须是字符串或文本数组"),
    ))
}

fn parse_stop(stop: Option<Value>) -> Result<Vec<String>, EngineError> {
    let Some(stop) = stop else {
        return Ok(Vec::new());
    };
    if stop.is_null() {
        return Ok(Vec::new());
    }
    if let Some(stop) = stop.as_str() {
        if stop.is_empty() {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "stop 不能为空字符串",
            ));
        }
        return Ok(vec![stop.to_owned()]);
    }
    if let Some(items) = stop.as_array() {
        let parsed: Vec<String> = items
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value.as_str().map(str::to_owned).ok_or_else(|| {
                    EngineError::new(
                        EngineErrorKind::InvalidRequest,
                        format!("stop[{index}] 必须是字符串"),
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        if parsed.iter().any(String::is_empty) {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "stop 不能包含空字符串",
            ));
        }
        return Ok(parsed);
    }
    Err(EngineError::new(
        EngineErrorKind::InvalidRequest,
        "stop 必须是字符串或字符串数组",
    ))
}

async fn collect_completion(
    mut receiver: EngineEventReceiver,
) -> Result<(String, EngineDone), EngineError> {
    let mut content = String::new();
    while let Some(event) = receiver.recv().await {
        match event? {
            EngineEvent::Delta(delta) => content.push_str(&delta.text),
            EngineEvent::Done(done) => return Ok((content, done)),
        }
    }
    Err(EngineError::incomplete())
}

fn openai_stream_response(
    mut receiver: EngineEventReceiver,
    request_id: String,
    model: String,
) -> Response {
    let (sender, output) = mpsc::channel::<Result<Event, Infallible>>(16);
    tokio::spawn(async move {
        let telemetry = stream_telemetry_enabled();
        let mut sequence = 0u64;
        while let Some(event) = receiver.recv().await {
            let event = match event {
                Ok(EngineEvent::Delta(delta)) => {
                    trace_stream_delta(telemetry, "openai", &request_id, sequence, &delta);
                    sequence += 1;
                    json!({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": unix_seconds(),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": delta.text},
                            "finish_reason": null,
                        }],
                    })
                }
                Ok(EngineEvent::Done(done)) => {
                    trace_stream_done(telemetry, "openai", &request_id, sequence, &done);
                    let final_chunk = json!({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": unix_seconds(),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": done.finish_reason.openai_name(),
                        }],
                    });
                    if sender
                        .send(Ok(Event::default().data(final_chunk.to_string())))
                        .await
                        .is_ok()
                    {
                        let _ = sender.send(Ok(Event::default().data("[DONE]"))).await;
                    }
                    return;
                }
                Err(error) => {
                    trace_stream_error(telemetry, "openai", &request_id, sequence, &error);
                    let _ = sender
                        .send(Ok(
                            Event::default().data(openai_stream_error_json(&error).to_string())
                        ))
                        .await;
                    return;
                }
            };
            if sender
                .send(Ok(Event::default().data(event.to_string())))
                .await
                .is_err()
            {
                return;
            }
        }
        let error = EngineError::incomplete();
        trace_stream_error(telemetry, "openai", &request_id, sequence, &error);
        let _ = sender
            .send(Ok(
                Event::default().data(openai_stream_error_json(&error).to_string())
            ))
            .await;
    });

    Sse::new(ReceiverStream::new(output))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn ollama_stream_response(mut receiver: EngineEventReceiver, model: String) -> Response {
    let (sender, output) = mpsc::channel::<Result<Bytes, Infallible>>(16);
    let request_id = next_request_id("ollama-polaris");
    tokio::spawn(async move {
        let telemetry = stream_telemetry_enabled();
        let mut sequence = 0u64;
        while let Some(event) = receiver.recv().await {
            let line = match event {
                Ok(EngineEvent::Delta(delta)) => {
                    trace_stream_delta(telemetry, "ollama", &request_id, sequence, &delta);
                    sequence += 1;
                    json!({
                        "model": model,
                        "created_at": utc_now_rfc3339(),
                        "message": {"role": "assistant", "content": delta.text},
                        "done": false,
                    })
                }
                Ok(EngineEvent::Done(done)) => {
                    trace_stream_done(telemetry, "ollama", &request_id, sequence, &done);
                    let mut value = json!({
                        "model": model,
                        "created_at": utc_now_rfc3339(),
                        "message": {"role": "assistant", "content": ""},
                        "done": true,
                        "done_reason": done.finish_reason.ollama_name(),
                    });
                    insert_ollama_usage(&mut value, &done);
                    let _ = sender.send(Ok(Bytes::from(format!("{value}\n")))).await;
                    return;
                }
                Err(error) => {
                    trace_stream_error(telemetry, "ollama", &request_id, sequence, &error);
                    let mut value = error_json(&error);
                    if let Some(object) = value.as_object_mut() {
                        object.insert("done".to_owned(), Value::Bool(false));
                    }
                    let _ = sender.send(Ok(Bytes::from(format!("{value}\n")))).await;
                    return;
                }
            };
            if sender
                .send(Ok(Bytes::from(format!("{line}\n"))))
                .await
                .is_err()
            {
                return;
            }
        }
        let error = EngineError::incomplete();
        trace_stream_error(telemetry, "ollama", &request_id, sequence, &error);
        let mut value = error_json(&error);
        if let Some(object) = value.as_object_mut() {
            object.insert("done".to_owned(), Value::Bool(false));
        }
        let _ = sender.send(Ok(Bytes::from(format!("{value}\n")))).await;
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(ReceiverStream::new(output)))
        .expect("静态 Ollama streaming response 必须可构造")
}

/// 只记录流事件的形状和计数，绝不记录 prompt、正文或 stop 文本。
/// 生产排障时显式设置 `POLARIS_STREAM_TELEMETRY=1`；默认完全静默。
fn stream_telemetry_enabled() -> bool {
    std::env::var("POLARIS_STREAM_TELEMETRY")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

fn trace_stream_delta(
    enabled: bool,
    protocol: &str,
    request_id: &str,
    sequence: u64,
    delta: &EngineDelta,
) {
    if enabled {
        eprintln!(
            "polaris_stream request_id={request_id} protocol={protocol} frame=delta sequence={sequence} utf8_bytes={} chars={} token_id_present={}",
            delta.text.len(),
            delta.text.chars().count(),
            delta.token_id.is_some(),
        );
    }
}

fn trace_stream_done(
    enabled: bool,
    protocol: &str,
    request_id: &str,
    sequence: u64,
    done: &EngineDone,
) {
    if enabled {
        eprintln!(
            "polaris_stream request_id={request_id} protocol={protocol} frame=done sequence={sequence} finish_reason={} prompt_tokens_known={} completion_tokens_known={}",
            done.finish_reason.openai_name(),
            done.prompt_tokens.is_some(),
            done.completion_tokens.is_some(),
        );
    }
}

fn trace_stream_error(
    enabled: bool,
    protocol: &str,
    request_id: &str,
    sequence: u64,
    error: &EngineError,
) {
    if enabled {
        eprintln!(
            "polaris_stream request_id={request_id} protocol={protocol} frame=error sequence={sequence} code={} failed_position={:?} retryable={}",
            error.kind.code(),
            error.failed_position,
            error.retryable,
        );
    }
}

fn openai_error_response(error: EngineError) -> Response {
    (
        error.kind.status(),
        Json(json!({
            "error": {
                "message": error.message,
                "type": "engine_error",
                "code": error.kind.code(),
                "failed_position": error.failed_position,
                "retryable": error.retryable,
            }
        })),
    )
        .into_response()
}

fn ollama_error_response(error: EngineError) -> Response {
    (error.kind.status(), Json(error_json(&error))).into_response()
}

fn error_json(error: &EngineError) -> Value {
    json!({
        "error": error.message,
        "error_code": error.kind.code(),
        "failed_position": error.failed_position,
        "retryable": error.retryable,
    })
}

fn openai_stream_error_json(error: &EngineError) -> Value {
    json!({
        "error": {
            "message": error.message,
            "type": "engine_error",
            "code": error.kind.code(),
            "failed_position": error.failed_position,
            "retryable": error.retryable,
        }
    })
}

fn insert_openai_usage(body: &mut Value, done: &EngineDone) {
    if let (Some(prompt), Some(completion)) = (done.prompt_tokens, done.completion_tokens) {
        body.as_object_mut().expect("response body 是对象").insert(
            "usage".to_owned(),
            json!({
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt.saturating_add(completion),
            }),
        );
    }
}

fn insert_ollama_usage(body: &mut Value, done: &EngineDone) {
    let object: &mut Map<String, Value> = body.as_object_mut().expect("response body 是对象");
    if let Some(prompt) = done.prompt_tokens {
        object.insert("prompt_eval_count".to_owned(), Value::from(prompt));
    }
    if let Some(completion) = done.completion_tokens {
        object.insert("eval_count".to_owned(), Value::from(completion));
    }
}

fn next_request_id(prefix: &str) -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    format!(
        "{prefix}-{}-{}",
        unix_seconds(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// 不额外引入时间库的 UTC RFC3339 秒精度格式化。
fn utc_now_rfc3339() -> String {
    let seconds = unix_seconds() as i64;
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}
