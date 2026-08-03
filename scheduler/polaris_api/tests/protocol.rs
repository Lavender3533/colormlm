use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    Router,
};
use polaris_api::{
    router, ChatEngine, EngineChatRequest, EngineDelta, EngineDone, EngineError, EngineEvent,
    EngineEventReceiver, EngineHealth, EngineStartFuture, FinishReason, UnavailableEngine,
    POLARIS_MODEL_ID,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tower::ServiceExt;

struct ScriptEngine {
    events: Vec<Result<EngineEvent, EngineError>>,
    start_error: Option<EngineError>,
    requests: Mutex<Vec<EngineChatRequest>>,
}

impl ScriptEngine {
    fn events(events: Vec<Result<EngineEvent, EngineError>>) -> Arc<Self> {
        Arc::new(Self {
            events,
            start_error: None,
            requests: Mutex::new(Vec::new()),
        })
    }

    fn start_error(error: EngineError) -> Arc<Self> {
        Arc::new(Self {
            events: Vec::new(),
            start_error: Some(error),
            requests: Mutex::new(Vec::new()),
        })
    }
}

impl ChatEngine for ScriptEngine {
    fn start_chat(&self, request: EngineChatRequest) -> EngineStartFuture<'_> {
        self.requests.lock().expect("request lock").push(request);
        let start_error = self.start_error.clone();
        let events = self.events.clone();
        Box::pin(async move {
            if let Some(error) = start_error {
                return Err(error);
            }
            let (sender, receiver): (_, EngineEventReceiver) = mpsc::channel(16);
            tokio::spawn(async move {
                for event in events {
                    if sender.send(event).await.is_err() {
                        break;
                    }
                }
            });
            Ok(receiver)
        })
    }

    fn health(&self) -> EngineHealth {
        EngineHealth {
            ready: self.start_error.is_none(),
            detail: "test engine".to_owned(),
        }
    }
}

fn done(prompt_tokens: Option<u64>, completion_tokens: Option<u64>) -> EngineEvent {
    EngineEvent::Done(EngineDone {
        finish_reason: FinishReason::Stop,
        prompt_tokens,
        completion_tokens,
    })
}

fn delta(text: &str, token_id: u32) -> EngineEvent {
    EngineEvent::Delta(EngineDelta {
        text: text.to_owned(),
        token_id: Some(token_id),
    })
}

async fn post(app: Router, path: &str, payload: Value) -> (StatusCode, String) {
    let response = app
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (
        status,
        String::from_utf8(body.to_vec()).expect("UTF-8 body"),
    )
}

async fn get(app: Router, path: &str) -> (StatusCode, String) {
    let response = app
        .oneshot(Request::get(path).body(Body::empty()).expect("request"))
        .await
        .expect("response");
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (
        status,
        String::from_utf8(body.to_vec()).expect("UTF-8 body"),
    )
}

#[tokio::test]
async fn openai_non_stream_forwards_real_engine_content_and_usage() {
    let engine = ScriptEngine::events(vec![
        Ok(delta("真实", 5)),
        Ok(delta("输出", 223)),
        Ok(done(Some(11), Some(2))),
    ]);
    let app = router(engine.clone());
    let (status, body) = post(
        app,
        "/v1/chat/completions",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [
                {"role": "system", "content": "系统"},
                {"role": "user", "content": "你好"}
            ],
            "stream": false
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let body: Value = serde_json::from_str(&body).expect("JSON");
    assert_eq!(body["choices"][0]["message"]["content"], "真实输出");
    assert_eq!(body["usage"]["prompt_tokens"], 11);
    assert_eq!(body["usage"]["completion_tokens"], 2);
    assert_eq!(body["usage"]["total_tokens"], 13);
    let requests = engine.requests.lock().expect("requests");
    assert_eq!(requests[0].messages.len(), 2);
    assert_eq!(requests[0].messages[0].content, "系统");
}

#[tokio::test]
async fn open_webui_openai_stream_keeps_delta_before_finish_frame() {
    let engine = ScriptEngine::events(vec![Ok(delta("好的", 223)), Ok(done(Some(7), Some(1)))]);
    let (status, body) = post(
        router(engine),
        "/v1/chat/completions",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [{"role": "user", "content": "你好"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let frames = body
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), 3, "delta、finish 与 [DONE] 必须各自成帧");
    let delta_frame: Value = serde_json::from_str(frames[0]).expect("delta JSON");
    let finish_frame: Value = serde_json::from_str(frames[1]).expect("finish JSON");
    assert_eq!(delta_frame["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(delta_frame["choices"][0]["delta"]["content"], "好的");
    assert_eq!(delta_frame["choices"][0]["finish_reason"], Value::Null);
    assert_eq!(finish_frame["choices"][0]["delta"], json!({}));
    assert_eq!(finish_frame["choices"][0]["finish_reason"], "stop");
    assert_eq!(delta_frame["id"], finish_frame["id"]);
    assert_eq!(frames[2], "[DONE]");
}

#[tokio::test]
async fn unknown_usage_is_omitted_instead_of_faked_as_zero() {
    let engine = ScriptEngine::events(vec![Ok(delta("好", 5)), Ok(done(None, None))]);
    let (status, body) = post(
        router(engine),
        "/v1/chat/completions",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [{"role": "user", "content": "你好"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body: Value = serde_json::from_str(&body).expect("JSON");
    assert!(body.get("usage").is_none());
}

#[tokio::test]
async fn open_webui_empty_tools_are_normalized_to_plain_chat() {
    let engine = ScriptEngine::events(vec![Ok(done(None, None))]);
    let (status, _) = post(
        router(engine.clone()),
        "/v1/chat/completions",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [{"role": "user", "content": "你好"}],
            "tools": [],
            "tool_choice": "auto"
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let requests = engine.requests.lock().expect("requests");
    assert!(requests[0].tools.is_none());
    assert!(requests[0].tool_choice.is_none());
}

#[tokio::test]
async fn openai_stream_exposes_position_failure_without_done_marker() {
    let engine = ScriptEngine::events(vec![
        Ok(delta("已提交片段", 5)),
        Err(EngineError::unsupported_position(
            2,
            "position2 数值后端尚未闭合",
        )),
    ]);
    let (status, body) = post(
        router(engine),
        "/v1/chat/completions",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [{"role": "user", "content": "继续"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("已提交片段"));
    assert!(body.contains("unsupported_position"));
    assert!(body.contains("failed_position\":2"));
    assert!(!body.contains("[DONE]"));
    assert!(!body.contains("finish_reason\":\"stop\""));
}

#[tokio::test]
async fn ollama_non_stream_uses_ollama_shape() {
    let engine = ScriptEngine::events(vec![Ok(delta("好的", 223)), Ok(done(Some(7), Some(1)))]);
    let (status, body) = post(
        router(engine),
        "/api/chat",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [{"role": "user", "content": "你好"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body: Value = serde_json::from_str(&body).expect("JSON");
    assert_eq!(body["message"]["content"], "好的");
    assert_eq!(body["done"], true);
    assert_eq!(body["prompt_eval_count"], 7);
    assert_eq!(body["eval_count"], 1);
}

#[tokio::test]
async fn open_webui_ollama_stream_keeps_delta_before_done_frame() {
    let engine = ScriptEngine::events(vec![Ok(delta("好的", 223)), Ok(done(Some(7), Some(1)))]);
    let (status, body) = post(
        router(engine),
        "/api/chat",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [{"role": "user", "content": "你好"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let frames = body
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("NDJSON frame"))
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), 2, "delta 与 done 必须各自成帧");
    assert_eq!(frames[0]["message"]["role"], "assistant");
    assert_eq!(frames[0]["message"]["content"], "好的");
    assert_eq!(frames[0]["done"], false);
    assert_eq!(frames[1]["message"]["content"], "");
    assert_eq!(frames[1]["done"], true);
    assert_eq!(frames[1]["done_reason"], "stop");
    assert_eq!(frames[1]["prompt_eval_count"], 7);
    assert_eq!(frames[1]["eval_count"], 1);
}

#[tokio::test]
async fn ollama_stream_failure_is_done_false_and_never_done_true() {
    let engine = ScriptEngine::events(vec![
        Ok(delta("片段", 5)),
        Err(EngineError::unsupported_position(
            2,
            "position2 fail-closed",
        )),
    ]);
    let (status, body) = post(
        router(engine),
        "/api/chat",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [{"role": "user", "content": "继续"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("unsupported_position"));
    assert!(body.contains("\"done\":false"));
    assert!(!body.contains("\"done\":true"));
}

#[tokio::test]
async fn immediate_runtime_unavailable_returns_503_without_content() {
    let engine = ScriptEngine::start_error(EngineError::runtime_unavailable("S14 未接入"));
    let (status, body) = post(
        router(engine),
        "/v1/chat/completions",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [{"role": "user", "content": "你好"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body.contains("runtime_unavailable"));
    assert!(!body.contains("choices"));
}

#[tokio::test]
async fn production_default_engine_never_emits_placeholder_tokens() {
    let engine = Arc::new(UnavailableEngine::new("持久 runtime 未接入"));
    let (status, body) = post(
        router(engine),
        "/api/chat",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [{"role": "user", "content": "你好"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body.contains("runtime_unavailable"));
    assert!(!body.contains("message"));
}

#[tokio::test]
async fn open_webui_discovery_routes_are_closed_while_s14_is_unavailable() {
    let engine = Arc::new(UnavailableEngine::new("持久 runtime 未接入"));
    let app = router(engine);
    let (openai_status, openai_body) = get(app.clone(), "/v1/models").await;
    let (ollama_status, ollama_body) = get(app, "/api/tags").await;

    assert_eq!(openai_status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(ollama_status, StatusCode::SERVICE_UNAVAILABLE);
    let openai: Value = serde_json::from_str(&openai_body).expect("OpenAI models JSON");
    let ollama: Value = serde_json::from_str(&ollama_body).expect("Ollama tags JSON");
    assert_eq!(openai["error"]["code"], "runtime_unavailable");
    assert_eq!(ollama["error_code"], "runtime_unavailable");
    assert!(!openai_body.contains(POLARIS_MODEL_ID));
    assert!(!ollama_body.contains(POLARIS_MODEL_ID));
}

#[tokio::test]
async fn open_webui_discovery_exposes_only_polaris_s14_after_health_ready() {
    let engine = ScriptEngine::events(vec![Ok(done(None, None))]);
    let app = router(engine);
    let (openai_status, openai_body) = get(app.clone(), "/v1/models").await;
    let (ollama_status, ollama_body) = get(app, "/api/tags").await;

    assert_eq!(openai_status, StatusCode::OK);
    assert_eq!(ollama_status, StatusCode::OK);
    let openai: Value = serde_json::from_str(&openai_body).expect("OpenAI models JSON");
    let ollama: Value = serde_json::from_str(&ollama_body).expect("Ollama tags JSON");
    assert_eq!(openai["data"].as_array().expect("models").len(), 1);
    assert_eq!(openai["data"][0]["id"], POLARIS_MODEL_ID);
    assert_eq!(ollama["models"].as_array().expect("models").len(), 1);
    assert_eq!(ollama["models"][0]["model"], POLARIS_MODEL_ID);
}

#[tokio::test]
async fn invalid_temperature_and_empty_stop_fail_before_engine_start() {
    let engine = ScriptEngine::events(vec![Ok(done(None, None))]);
    let (temperature_status, temperature_body) = post(
        router(engine.clone()),
        "/v1/chat/completions",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": -0.1
        }),
    )
    .await;
    let (stop_status, stop_body) = post(
        router(engine),
        "/api/chat",
        json!({
            "model": POLARIS_MODEL_ID,
            "messages": [{"role": "user", "content": "你好"}],
            "stream": false,
            "options": {"stop": [""]}
        }),
    )
    .await;

    assert_eq!(temperature_status, StatusCode::BAD_REQUEST);
    assert!(temperature_body.contains("invalid_request"));
    assert_eq!(stop_status, StatusCode::BAD_REQUEST);
    assert!(stop_body.contains("invalid_request"));
}
