//! Persistent UTF-8 JSONL bridge from the Rust route-first runner to the
//! Python `RouteFirstSession` range cache.
//!
//! The bridge transports verified local page handles only.  The official
//! native executor remains the sole routing authority and supplies the exact
//! current-token top-6 after attention.

use crate::{
    BaseLoadTicket, GraphProfile, ProviderError, RangeArtifact, ReadyBaseLease, ReadyRoutedLease,
    RouteDecision, RouteFirstProvider, RoutedLoadTicket, TransferObservation, MODEL_REPO,
    MODEL_REVISION, SELECTED_LAYERS,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

pub const RANGE_JSONL_PROTOCOL: &str = "polaris-s14-range-jsonl-v1";

#[derive(Debug, Clone)]
pub struct RangeBridgeConfig {
    pub response_timeout: Duration,
    /// This must match the worker's explicit runtime authorization.  It does
    /// not inherit or elevate the catalog's always-false safety marker.
    pub download_authorized: bool,
}

impl Default for RangeBridgeConfig {
    fn default() -> Self {
        Self {
            response_timeout: Duration::from_secs(30),
            download_authorized: false,
        }
    }
}

#[derive(Debug, Deserialize)]
struct WireResponse {
    protocol: String,
    request_id: u64,
    op: String,
    status: String,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default)]
    revision: Option<String>,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    selected_layers: Option<Vec<u8>>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    download_authorized: Option<bool>,
    #[serde(default)]
    layer: Option<u8>,
    #[serde(default)]
    expert_ids: Vec<u16>,
    #[serde(default)]
    artifacts: Vec<RangeArtifact>,
    #[serde(default)]
    final_artifacts: Vec<RangeArtifact>,
    #[serde(default)]
    observation: TransferObservation,
}

pub struct SubprocessRangeProvider {
    child: Child,
    input: BufWriter<ChildStdin>,
    responses: Receiver<Result<String, String>>,
    config: RangeBridgeConfig,
    next_request_id: u64,
    next_ticket_id: u64,
    pending_base: Option<BaseLoadTicket>,
    pending_routed: Option<RoutedLoadTicket>,
    live_layer: Option<u8>,
    routed_ready: bool,
    active_token_id: Option<u32>,
    final_artifacts: Vec<RangeArtifact>,
    poisoned: bool,
}

impl SubprocessRangeProvider {
    pub fn spawn(mut command: Command, config: RangeBridgeConfig) -> Result<Self, ProviderError> {
        command.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|error| ProviderError(format!("启动 Range worker 失败: {error}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| ProviderError("Range worker stdin 未建立".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ProviderError("Range worker stdout 未建立".into()))?;
        let (sender, responses) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        let _ = sender.send(Err("Range worker stdout EOF".into()));
                        break;
                    }
                    Ok(_) => {
                        while line.ends_with(['\r', '\n']) {
                            line.pop();
                        }
                        if sender.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(format!("读取 Range worker 失败: {error}")));
                        break;
                    }
                }
            }
        });
        let mut provider = Self {
            child,
            input: BufWriter::new(stdin),
            responses,
            config,
            next_request_id: 1,
            next_ticket_id: 1,
            pending_base: None,
            pending_routed: None,
            live_layer: None,
            routed_ready: false,
            active_token_id: None,
            final_artifacts: Vec::new(),
            poisoned: false,
        };
        let request_id = provider.allocate_request_id();
        let response = provider.roundtrip(json!({
            "protocol": RANGE_JSONL_PROTOCOL,
            "request_id": request_id,
            "op": "hello",
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "profile": "s14_top6",
            "selected_layers": SELECTED_LAYERS,
            "top_k": 6,
            "download_authorized": provider.config.download_authorized,
        }))?;
        provider.validate_hello(response)?;
        Ok(provider)
    }

    fn allocate_request_id(&mut self) -> u64 {
        let value = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);
        value
    }

    fn allocate_ticket_id(&mut self) -> u64 {
        let value = self.next_ticket_id;
        self.next_ticket_id = self.next_ticket_id.saturating_add(1);
        value
    }

    fn terminate(&mut self) {
        self.poisoned = true;
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn send(&mut self, value: &Value) -> Result<(), ProviderError> {
        if self.poisoned {
            return Err(ProviderError("Range bridge 已 poisoned".into()));
        }
        serde_json::to_writer(&mut self.input, value)
            .map_err(|error| ProviderError(format!("编码 Range JSONL 失败: {error}")))?;
        self.input
            .write_all(b"\n")
            .and_then(|_| self.input.flush())
            .map_err(|error| ProviderError(format!("写入 Range worker 失败: {error}")))
    }

    fn receive(&mut self, request_id: u64, op: &str) -> Result<WireResponse, ProviderError> {
        let line = match self.responses.recv_timeout(self.config.response_timeout) {
            Ok(Ok(line)) => line,
            Ok(Err(error)) => {
                self.terminate();
                return Err(ProviderError(error));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.terminate();
                return Err(ProviderError(format!(
                    "Range worker op={op} 超时（{} ms）",
                    self.config.response_timeout.as_millis()
                )));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.terminate();
                return Err(ProviderError("Range worker reader 已退出".into()));
            }
        };
        let response: WireResponse = match serde_json::from_str(&line) {
            Ok(response) => response,
            Err(error) => {
                self.terminate();
                return Err(ProviderError(format!("Range worker 返回非 JSONL: {error}")));
            }
        };
        if response.protocol != RANGE_JSONL_PROTOCOL
            || response.request_id != request_id
            || response.op != op
        {
            self.terminate();
            return Err(ProviderError("Range worker response 身份/顺序漂移".into()));
        }
        if response.status != "ok" {
            let message = response
                .error
                .unwrap_or_else(|| "Range worker 未提供错误详情".into());
            self.terminate();
            return Err(ProviderError(format!("Range worker 拒绝 {op}: {message}")));
        }
        Ok(response)
    }

    fn roundtrip(&mut self, request: Value) -> Result<WireResponse, ProviderError> {
        let request_id = request["request_id"]
            .as_u64()
            .ok_or_else(|| ProviderError("bridge request_id 缺失".into()))?;
        let op = request["op"]
            .as_str()
            .ok_or_else(|| ProviderError("bridge op 缺失".into()))?
            .to_string();
        self.send(&request)?;
        self.receive(request_id, &op)
    }

    fn validate_hello(&mut self, response: WireResponse) -> Result<(), ProviderError> {
        let exact = response.repo.as_deref() == Some(MODEL_REPO)
            && response.revision.as_deref() == Some(MODEL_REVISION)
            && response.profile.as_deref() == Some("s14_top6")
            && response.selected_layers.as_deref() == Some(SELECTED_LAYERS.as_slice())
            && response.top_k == Some(6)
            && response.download_authorized == Some(self.config.download_authorized);
        if !exact {
            self.terminate();
            return Err(ProviderError(
                "Range worker 未确认冻结 revision/S14/top-6/download_authorized".into(),
            ));
        }
        Ok(())
    }

    fn validate_artifacts(
        &mut self,
        artifacts: &[RangeArtifact],
        routed_ids: Option<&[u16]>,
    ) -> Result<(), ProviderError> {
        if artifacts.is_empty() {
            self.terminate();
            return Err(ProviderError("Range worker 返回空页面集合".into()));
        }
        let mut tensors = BTreeSet::new();
        for artifact in artifacts {
            let sha_ok = artifact.observed_sha256.len() == 64
                && artifact
                    .observed_sha256
                    .bytes()
                    .all(|value| value.is_ascii_hexdigit() && !value.is_ascii_uppercase());
            let path = Path::new(&artifact.path);
            let file_ok = path.is_absolute()
                && path.is_file()
                && std::fs::metadata(path)
                    .map(|metadata| metadata.len() == artifact.bytes)
                    .unwrap_or(false);
            let kind_ok = match routed_ids {
                None => {
                    matches!(
                        artifact.kind.as_str(),
                        "boundary" | "embedding_row" | "non_expert" | "router"
                    ) && artifact.expert_id.is_none()
                }
                Some(ids) => match artifact.kind.as_str() {
                    "shared" => artifact.expert_id.is_none(),
                    "routed_expert" => artifact
                        .expert_id
                        .map(|expert| ids.contains(&expert))
                        .unwrap_or(false),
                    _ => false,
                },
            };
            if artifact.tensor.is_empty()
                || !tensors.insert(artifact.tensor.clone())
                || artifact.bytes == 0
                || !sha_ok
                || !file_ok
                || !kind_ok
            {
                self.terminate();
                return Err(ProviderError(format!(
                    "Range artifact 契约失败: {}",
                    artifact.tensor
                )));
            }
        }
        Ok(())
    }

    fn validate_final_artifacts(
        &mut self,
        artifacts: &[RangeArtifact],
    ) -> Result<(), ProviderError> {
        self.validate_artifacts(artifacts, None)?;
        let actual: BTreeSet<&str> = artifacts.iter().map(|item| item.tensor.as_str()).collect();
        let expected: BTreeSet<&str> = [
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
            "head.weight",
            "norm.weight",
        ]
        .into_iter()
        .collect();
        if actual != expected || artifacts.iter().any(|item| item.kind != "boundary") {
            self.terminate();
            return Err(ProviderError("final artifacts 集合/类型不完整".into()));
        }
        Ok(())
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

impl RouteFirstProvider for SubprocessRangeProvider {
    fn begin_base_load(
        &mut self,
        layer: u8,
        input_token_id: u32,
    ) -> Result<BaseLoadTicket, ProviderError> {
        if self.pending_base.is_some() || self.pending_routed.is_some() || self.live_layer.is_some()
        {
            return Err(ProviderError(
                "Range bridge 同时只允许一个 live layer".into(),
            ));
        }
        if layer == SELECTED_LAYERS[0] {
            self.active_token_id = Some(input_token_id);
        }
        let token_id = self
            .active_token_id
            .filter(|active| *active == input_token_id)
            .ok_or_else(|| ProviderError("S14 token 必须从 L0 开始".into()))?;
        let request_id = self.allocate_request_id();
        self.send(&json!({
            "protocol": RANGE_JSONL_PROTOCOL,
            "request_id": request_id,
            "op": "prepare_base",
            "layer": layer,
            "token_id": token_id,
            "download_authorized": self.config.download_authorized,
        }))?;
        let ticket = BaseLoadTicket {
            layer,
            ticket_id: self.allocate_ticket_id(),
        };
        self.pending_base = Some(ticket);
        Ok(ticket)
    }

    fn wait_base_ready(&mut self, ticket: BaseLoadTicket) -> Result<ReadyBaseLease, ProviderError> {
        if self.pending_base != Some(ticket) {
            return Err(ProviderError("base ticket 非当前 pending request".into()));
        }
        let response = self.receive(self.next_request_id - 1, "prepare_base")?;
        if response.layer != Some(ticket.layer) {
            self.terminate();
            return Err(ProviderError("base response layer 漂移".into()));
        }
        self.validate_artifacts(&response.artifacts, None)?;
        self.pending_base = None;
        self.live_layer = Some(ticket.layer);
        self.routed_ready = false;
        Ok(ReadyBaseLease {
            layer: ticket.layer,
            lease_id: ticket.ticket_id,
            artifacts: response.artifacts,
            observation: response.observation,
        })
    }

    fn begin_routed_load(
        &mut self,
        base: &ReadyBaseLease,
        route: &RouteDecision,
    ) -> Result<RoutedLoadTicket, ProviderError> {
        if self.live_layer != Some(base.layer) || self.pending_routed.is_some() {
            return Err(ProviderError("routed request 没有对应 live base".into()));
        }
        route
            .validate_for(GraphProfile::S14Top6)
            .map_err(|error| ProviderError(error.to_string()))?;
        let token_id = self
            .active_token_id
            .ok_or_else(|| ProviderError("active token 缺失".into()))?;
        let request_id = self.allocate_request_id();
        self.send(&json!({
            "protocol": RANGE_JSONL_PROTOCOL,
            "request_id": request_id,
            "op": "prepare_routed",
            "layer": base.layer,
            "token_id": token_id,
            "expert_ids": route.expert_ids,
            "download_authorized": self.config.download_authorized,
        }))?;
        let ticket = RoutedLoadTicket {
            layer: base.layer,
            ticket_id: self.allocate_ticket_id(),
            expert_ids: route.expert_ids.clone(),
        };
        self.pending_routed = Some(ticket.clone());
        Ok(ticket)
    }

    fn wait_routed_ready(
        &mut self,
        ticket: RoutedLoadTicket,
    ) -> Result<ReadyRoutedLease, ProviderError> {
        if self.pending_routed.as_ref() != Some(&ticket) {
            return Err(ProviderError("routed ticket 非当前 pending request".into()));
        }
        let response = self.receive(self.next_request_id - 1, "prepare_routed")?;
        if response.layer != Some(ticket.layer) || response.expert_ids != ticket.expert_ids {
            self.terminate();
            return Err(ProviderError("routed response layer/top-6 漂移".into()));
        }
        self.validate_artifacts(&response.artifacts, Some(&ticket.expert_ids))?;
        self.pending_routed = None;
        self.routed_ready = true;
        Ok(ReadyRoutedLease {
            layer: ticket.layer,
            lease_id: ticket.ticket_id,
            expert_ids: ticket.expert_ids,
            artifacts: response.artifacts,
            observation: response.observation,
        })
    }

    fn release_layer(&mut self, layer: u8) -> Result<(), ProviderError> {
        if self.live_layer != Some(layer)
            || self.pending_base.is_some()
            || self.pending_routed.is_some()
        {
            return Err(ProviderError("release 不是当前 ready layer".into()));
        }
        let token_id = self
            .active_token_id
            .ok_or_else(|| ProviderError("active token 缺失".into()))?;
        let request_id = self.allocate_request_id();
        let op = if self.routed_ready {
            "release_layer"
        } else {
            "abort_layer"
        };
        let response = self.roundtrip(json!({
            "protocol": RANGE_JSONL_PROTOCOL,
            "request_id": request_id,
            "op": op,
            "layer": layer,
            "token_id": token_id,
            "download_authorized": self.config.download_authorized,
        }))?;
        if response.layer != Some(layer) {
            self.terminate();
            return Err(ProviderError("release response layer 漂移".into()));
        }
        if layer == *SELECTED_LAYERS.last().unwrap() {
            self.validate_final_artifacts(&response.final_artifacts)?;
            self.final_artifacts = response.final_artifacts;
            self.active_token_id = None;
        }
        self.live_layer = None;
        self.routed_ready = false;
        Ok(())
    }

    fn take_final_artifacts(&mut self) -> Result<Vec<RangeArtifact>, ProviderError> {
        if self.live_layer.is_some() || self.pending_base.is_some() || self.pending_routed.is_some()
        {
            return Err(ProviderError(
                "仍有 live layer，拒绝发布 final artifacts".into(),
            ));
        }
        if self.final_artifacts.is_empty() {
            return Err(ProviderError("最后一层未返回 final artifacts".into()));
        }
        self.active_token_id = None;
        Ok(std::mem::take(&mut self.final_artifacts))
    }
}

impl Drop for SubprocessRangeProvider {
    fn drop(&mut self) {
        if !self.poisoned {
            let request_id = self.allocate_request_id();
            let _ = self.send(&json!({
                "protocol": RANGE_JSONL_PROTOCOL,
                "request_id": request_id,
                "op": "shutdown",
                "download_authorized": self.config.download_authorized,
            }));
        }
        self.terminate();
    }
}
