//! Persistent fail-closed JSONL bridge from `NativeS14Executor` to Python.
//!
//! JSONL is control-plane only. Hidden/recursive state and logits live in one
//! bridge-owned binary arena whose exact path, offsets, sizes, dtypes and
//! shapes are carried by every request. No tensor values are embedded in JSON.

use crate::{
    BufferSlice, DType, GraphProfile, NativeS14Executor, NativeState, RangeArtifact,
    ReadyBaseLease, ReadyRoutedLease, RouteDecision, RouterKind, HC_STREAMS, HIDDEN_SIZE,
    MODEL_REPO, MODEL_REVISION, SELECTED_LAYERS, VOCAB_SIZE,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

pub const EXECUTOR_JSONL_PROTOCOL: &str = "polaris-s14-executor-jsonl-v1";
const MAX_JSONL_BYTES: usize = 1 << 20;
const BF16_SENTINEL: u16 = 0x7fc1;
const F32_SENTINEL: u32 = 0x7fc0_0001;
const FILE_ALIGNMENT: u64 = 4096;
const DEFAULT_MAX_ARENA_BYTES: u64 = 512 << 20;
static ARENA_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct ExecutorBridgeConfig {
    pub response_timeout: Duration,
    /// Existing directory used for a bridge-owned binary arena. `None` uses
    /// the OS temporary directory. The arena is removed on drop.
    pub workspace_dir: Option<PathBuf>,
    /// Hard bound covering recursive state plus logits scratch.
    pub max_arena_bytes: u64,
}

impl Default for ExecutorBridgeConfig {
    fn default() -> Self {
        Self {
            response_timeout: Duration::from_secs(30),
            workspace_dir: None,
            max_arena_bytes: DEFAULT_MAX_ARENA_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryTensorView {
    pub path: String,
    pub offset: u64,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<u32>,
}

#[derive(Debug, Clone)]
struct ExecutorShape {
    mode: &'static str,
    hidden_shape: Vec<u32>,
    logits_shape: Vec<u32>,
    production: bool,
}

impl ExecutorShape {
    fn production() -> Self {
        Self {
            mode: "production",
            hidden_shape: vec![HC_STREAMS, HIDDEN_SIZE],
            logits_shape: vec![VOCAB_SIZE],
            production: true,
        }
    }

    #[cfg(test)]
    fn fixture(hidden_streams: u32, hidden_size: u32, logits: u32) -> Self {
        Self {
            mode: "fixture_test",
            hidden_shape: vec![hidden_streams, hidden_size],
            logits_shape: vec![logits],
            production: false,
        }
    }

    fn hidden_bytes(&self) -> Result<u64, String> {
        tensor_bytes(&self.hidden_shape, 2)
    }

    fn logits_bytes(&self) -> Result<u64, String> {
        tensor_bytes(&self.logits_shape, 4)
    }
}

#[derive(Debug)]
struct ArenaLayout {
    path: PathBuf,
    path_json: String,
    state_bytes: u64,
    file_bytes: u64,
    state: BinaryTensorView,
    hidden: BinaryTensorView,
    logits: BinaryTensorView,
}

struct ArenaFileGuard {
    path: PathBuf,
    armed: bool,
}

impl ArenaFileGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ArenaFileGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
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
    mode: Option<String>,
    #[serde(default)]
    hidden_shape: Option<Vec<u32>>,
    #[serde(default)]
    hidden_dtype: Option<String>,
    #[serde(default)]
    logits_shape: Option<Vec<u32>>,
    #[serde(default)]
    logits_dtype: Option<String>,
    #[serde(default)]
    position: Option<u32>,
    #[serde(default)]
    layer: Option<u8>,
    #[serde(default)]
    state_epoch: Option<u64>,
    #[serde(default)]
    state_view: Option<BinaryTensorView>,
    #[serde(default)]
    hidden_written: Option<BinaryTensorView>,
    #[serde(default)]
    logits_written: Option<BinaryTensorView>,
    #[serde(default)]
    router_kind: Option<RouterKind>,
    #[serde(default)]
    expert_ids: Vec<u16>,
    #[serde(default)]
    route_weights: Vec<f32>,
}

pub struct SubprocessNativeExecutor {
    child: Child,
    requests: Sender<Vec<u8>>,
    write_acks: Receiver<Result<(), String>>,
    responses: Receiver<Result<String, String>>,
    config: ExecutorBridgeConfig,
    shape: ExecutorShape,
    next_request_id: u64,
    state_epoch: u64,
    active_position: Option<u32>,
    active_token_id: Option<u32>,
    next_layer_index: usize,
    pending_route: Option<RouteDecision>,
    arena: Option<ArenaLayout>,
    poisoned: bool,
}

impl SubprocessNativeExecutor {
    /// Spawn a production-shaped bridge. This constructor has no override for
    /// 4×4096 BF16 hidden state or 129,280 F32 logits.
    pub fn spawn(command: Command, config: ExecutorBridgeConfig) -> Result<Self, String> {
        Self::spawn_with_shape(command, config, ExecutorShape::production())
    }

    #[cfg(test)]
    fn spawn_fixture(
        command: Command,
        config: ExecutorBridgeConfig,
        hidden_streams: u32,
        hidden_size: u32,
        logits: u32,
    ) -> Result<Self, String> {
        Self::spawn_with_shape(
            command,
            config,
            ExecutorShape::fixture(hidden_streams, hidden_size, logits),
        )
    }

    fn spawn_with_shape(
        mut command: Command,
        config: ExecutorBridgeConfig,
        shape: ExecutorShape,
    ) -> Result<Self, String> {
        if config.response_timeout.is_zero() {
            return Err("Executor bridge response_timeout 必须为正数".into());
        }
        if config.max_arena_bytes == 0 {
            return Err("Executor bridge max_arena_bytes 必须为正数".into());
        }
        shape.hidden_bytes()?;
        shape.logits_bytes()?;
        command.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|error| format!("启动 Native executor worker 失败: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or("Native executor worker stdin 未建立")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("Native executor worker stdout 未建立")?;
        let (requests, request_receiver) = mpsc::channel::<Vec<u8>>();
        let (write_ack_sender, write_acks) = mpsc::channel();
        std::thread::spawn(move || {
            let mut writer = BufWriter::new(stdin);
            while let Ok(encoded) = request_receiver.recv() {
                let result = writer
                    .write_all(&encoded)
                    .and_then(|_| writer.write_all(b"\n"))
                    .and_then(|_| writer.flush())
                    .map_err(|error| format!("写入 Native executor worker 失败: {error}"));
                let failed = result.is_err();
                if write_ack_sender.send(result).is_err() || failed {
                    break;
                }
            }
        });
        let (sender, responses) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                match read_bounded_jsonl(&mut reader) {
                    Ok(Some(line)) => {
                        if sender.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Ok(None) => {
                        let _ = sender.send(Err("Native executor worker stdout EOF".into()));
                        break;
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        break;
                    }
                }
            }
        });
        let mut executor = Self {
            child,
            requests,
            write_acks,
            responses,
            config,
            shape,
            next_request_id: 1,
            state_epoch: 0,
            active_position: None,
            active_token_id: None,
            next_layer_index: 0,
            pending_route: None,
            arena: None,
            poisoned: false,
        };
        let request_id = executor.allocate_request_id()?;
        let response = executor.roundtrip(json!({
            "protocol": EXECUTOR_JSONL_PROTOCOL,
            "request_id": request_id,
            "op": "hello",
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "profile": "s14_top6",
            "mode": executor.shape.mode,
            "hidden_shape": executor.shape.hidden_shape,
            "hidden_dtype": "bf16_le",
            "logits_shape": executor.shape.logits_shape,
            "logits_dtype": "f32_le",
            "tensor_transport": "binary_file_views_v1",
            "max_jsonl_bytes": MAX_JSONL_BYTES,
            "token_output_allowed": false,
        }))?;
        executor.validate_hello(response)?;
        Ok(executor)
    }

    fn allocate_request_id(&mut self) -> Result<u64, String> {
        let value = self.next_request_id;
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or("Native executor request_id exhausted")?;
        Ok(value)
    }

    fn stop_child(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn terminate(&mut self) {
        self.poisoned = true;
        self.stop_child();
    }

    fn send(&mut self, value: &Value) -> Result<(), String> {
        if self.poisoned {
            return Err("Native executor bridge 已 poisoned".into());
        }
        let encoded = serde_json::to_vec(value)
            .map_err(|error| format!("编码 Executor JSONL 失败: {error}"))?;
        if encoded.len() > MAX_JSONL_BYTES {
            return Err(format!(
                "Executor JSONL request 超过 {} bytes；tensor 必须走 binary arena",
                MAX_JSONL_BYTES
            ));
        }
        self.requests
            .send(encoded)
            .map_err(|_| "Native executor writer 已退出".to_string())?;
        match self.write_acks.recv_timeout(self.config.response_timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(format!(
                "Native executor JSONL 写入超时（{} ms）",
                self.config.response_timeout.as_millis()
            )),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err("Native executor writer ack channel 已退出".into())
            }
        }
    }

    fn receive(&mut self, request_id: u64, op: &str) -> Result<WireResponse, String> {
        let line = match self.responses.recv_timeout(self.config.response_timeout) {
            Ok(Ok(line)) => line,
            Ok(Err(error)) => {
                self.terminate();
                return Err(error);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.terminate();
                return Err(format!(
                    "Native executor op={op} 超时（{} ms），bridge/state 已 poisoned",
                    self.config.response_timeout.as_millis()
                ));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.terminate();
                return Err("Native executor reader 已退出，bridge/state 已 poisoned".into());
            }
        };
        let raw: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                self.terminate();
                return Err(format!("Native executor 返回非 JSONL: {error}"));
            }
        };
        for forbidden in [
            "hidden_values",
            "state_values",
            "logits",
            "token",
            "token_id",
            "tokens",
        ] {
            if raw.get(forbidden).is_some() {
                self.terminate();
                return Err(format!(
                    "Native executor response 禁止字段 {forbidden}；tensor 必须走 binary arena，worker 不得返回 token"
                ));
            }
        }
        let response: WireResponse = match serde_json::from_value(raw) {
            Ok(response) => response,
            Err(error) => {
                self.terminate();
                return Err(format!("Native executor response 契约解析失败: {error}"));
            }
        };
        if response.protocol != EXECUTOR_JSONL_PROTOCOL
            || response.request_id != request_id
            || response.op != op
        {
            self.terminate();
            return Err("Native executor response 身份/顺序漂移".into());
        }
        if response.status != "ok" {
            let error = response
                .error
                .unwrap_or_else(|| "worker 未提供错误详情".into());
            self.terminate();
            return Err(format!("Native executor worker 拒绝 {op}: {error}"));
        }
        Ok(response)
    }

    fn roundtrip(&mut self, request: Value) -> Result<WireResponse, String> {
        let request_id = request["request_id"]
            .as_u64()
            .ok_or("Executor bridge request_id 缺失")?;
        let op = request["op"]
            .as_str()
            .ok_or("Executor bridge op 缺失")?
            .to_string();
        if let Err(error) = self.send(&request) {
            self.terminate();
            return Err(error);
        }
        self.receive(request_id, &op)
    }

    fn validate_hello(&mut self, response: WireResponse) -> Result<(), String> {
        let exact = response.repo.as_deref() == Some(MODEL_REPO)
            && response.revision.as_deref() == Some(MODEL_REVISION)
            && response.profile.as_deref() == Some("s14_top6")
            && response.mode.as_deref() == Some(self.shape.mode)
            && response.hidden_shape.as_deref() == Some(self.shape.hidden_shape.as_slice())
            && response.hidden_dtype.as_deref() == Some("bf16_le")
            && response.logits_shape.as_deref() == Some(self.shape.logits_shape.as_slice())
            && response.logits_dtype.as_deref() == Some("f32_le");
        if !exact {
            self.terminate();
            return Err(
                "Native executor worker 未确认冻结 revision/profile/hidden/logits shape".into(),
            );
        }
        Ok(())
    }

    fn ensure_state_contract(&self, state: &NativeState) -> Result<(), String> {
        if self.poisoned || state.poisoned {
            return Err("Native executor bridge/state 已 poisoned".into());
        }
        state
            .validate_for(GraphProfile::S14Top6)
            .map_err(|error| format!("Native executor state layout: {error}"))?;
        validate_native_state_slices(state)?;
        if self.shape.production {
            let hidden = &state.hc.streams;
            if hidden.shape != [1, 1, HC_STREAMS, HIDDEN_SIZE]
                || hidden.bytes != self.shape.hidden_bytes()?
            {
                return Err("production hidden 必须严格为 4×4096 BF16".into());
            }
        }
        Ok(())
    }

    fn ensure_arena(&mut self, state: &NativeState) -> Result<(), String> {
        self.ensure_state_contract(state)?;
        if let Some(arena) = &self.arena {
            if self.shape.production && arena.state_bytes != state.arena_bytes {
                return Err("Native executor recursive arena_bytes 漂移".into());
            }
            validate_file_length(&arena.path, arena.file_bytes)?;
            return Ok(());
        }
        let directory = self
            .config
            .workspace_dir
            .clone()
            .unwrap_or_else(std::env::temp_dir);
        let directory = directory
            .canonicalize()
            .map_err(|error| format!("Executor arena 目录不可用: {error}"))?;
        if !directory.is_dir() {
            return Err("Executor arena workspace_dir 不是目录".into());
        }
        let hidden_bytes = self.shape.hidden_bytes()?;
        let state_bytes = if self.shape.production {
            state.arena_bytes
        } else {
            hidden_bytes
        };
        let hidden_offset = if self.shape.production {
            state.hc.streams.offset
        } else {
            0
        };
        let logits_offset = align_up(state_bytes, FILE_ALIGNMENT)?;
        let file_bytes = logits_offset
            .checked_add(self.shape.logits_bytes()?)
            .ok_or("Executor arena 字节溢出")?;
        if file_bytes > self.config.max_arena_bytes {
            return Err(format!(
                "Executor arena 超限: requested={file_bytes} limit={}",
                self.config.max_arena_bytes
            ));
        }
        let (path, file) = create_unique_arena(&directory)?;
        let mut guard = ArenaFileGuard {
            path: path.clone(),
            armed: true,
        };
        file.set_len(file_bytes)
            .map_err(|error| format!("预分配 Executor arena 失败: {error}"))?;
        file.sync_data()
            .map_err(|error| format!("同步 Executor arena 失败: {error}"))?;
        drop(file);
        let path = path
            .canonicalize()
            .map_err(|error| format!("解析 Executor arena 路径失败: {error}"))?;
        let path_json = path
            .to_str()
            .ok_or("Executor arena 路径不是 UTF-8")?
            .to_string();
        let hidden = BinaryTensorView {
            path: path_json.clone(),
            offset: hidden_offset,
            bytes: hidden_bytes,
            dtype: "bf16_le".into(),
            shape: self.shape.hidden_shape.clone(),
        };
        let state_shape: u32 = state_bytes
            .try_into()
            .map_err(|_| "Executor state arena 超过 u32 shape 表达范围")?;
        let state_view = BinaryTensorView {
            path: path_json.clone(),
            offset: 0,
            bytes: state_bytes,
            dtype: "u8_opaque_state".into(),
            shape: vec![state_shape],
        };
        let logits = BinaryTensorView {
            path: path_json.clone(),
            offset: logits_offset,
            bytes: self.shape.logits_bytes()?,
            dtype: "f32_le".into(),
            shape: self.shape.logits_shape.clone(),
        };
        validate_view_bounds(&state_view, file_bytes)?;
        validate_view_bounds(&hidden, file_bytes)?;
        validate_view_bounds(&logits, file_bytes)?;
        guard.disarm();
        self.arena = Some(ArenaLayout {
            path,
            path_json,
            state_bytes,
            file_bytes,
            state: state_view,
            hidden,
            logits,
        });
        Ok(())
    }

    fn state_wire(&self, state: &NativeState) -> Result<Value, String> {
        let arena = self.arena.as_ref().ok_or("Executor arena 尚未建立")?;
        Ok(json!({
            "arena_path": arena.path_json,
            "state_bytes": arena.state_bytes,
            "file_bytes": arena.file_bytes,
            "position": state.position,
            "max_seq_len": state.max_seq_len,
            "profile": "s14_top6",
            "poisoned": state.poisoned,
            "state_epoch": self.state_epoch,
            "arena": arena.state,
            "hidden": arena.hidden,
            "layout": state,
        }))
    }

    fn validate_state_response(
        &mut self,
        response: &WireResponse,
        op: &str,
        position: u32,
        layer: Option<u8>,
        expected_epoch: u64,
    ) -> Result<(), String> {
        if response.position != Some(position)
            || response.layer != layer
            || response.state_epoch != Some(expected_epoch)
        {
            self.terminate();
            return Err(format!(
                "{op} response position/layer/epoch 漂移；state 已 poisoned"
            ));
        }
        let arena = self.arena.as_ref().ok_or("Executor arena 缺失")?;
        if response.state_view.as_ref() != Some(&arena.state)
            || response.hidden_written.as_ref() != Some(&arena.hidden)
        {
            self.terminate();
            return Err(format!(
                "{op} state/hidden descriptor 漂移；state 已 poisoned"
            ));
        }
        let arena = self.arena.as_ref().unwrap();
        if let Err(error) = validate_file_length(&arena.path, arena.file_bytes) {
            self.terminate();
            return Err(error);
        }
        Ok(())
    }

    fn validate_artifacts(&self, artifacts: &[RangeArtifact], label: &str) -> Result<(), String> {
        if artifacts.is_empty() {
            return Err(format!("{label} artifacts 不能为空"));
        }
        let mut tensors = BTreeSet::new();
        for artifact in artifacts {
            let digest_ok = artifact.observed_sha256.len() == 64
                && artifact
                    .observed_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
            let path = Path::new(&artifact.path);
            let file_ok = path.is_absolute()
                && path.is_file()
                && std::fs::metadata(path)
                    .map(|metadata| metadata.len() == artifact.bytes)
                    .unwrap_or(false);
            if artifact.tensor.is_empty()
                || !tensors.insert(artifact.tensor.as_str())
                || artifact.bytes == 0
                || !digest_ok
                || !file_ok
            {
                return Err(format!("{label} artifact 契约失败: {}", artifact.tensor));
            }
        }
        Ok(())
    }

    fn embed_impl(
        &mut self,
        token_id: u32,
        base: &ReadyBaseLease,
        state: &mut NativeState,
    ) -> Result<(), String> {
        self.ensure_arena(state)?;
        if base.layer != SELECTED_LAYERS[0]
            || self.active_position.is_some()
            || self.next_layer_index != 0
            || self.pending_route.is_some()
        {
            return Err("embed_row 只能在新 position 的 L0 base Ready 后执行".into());
        }
        self.validate_artifacts(&base.artifacts, "embed/base")?;
        let hidden = self.arena.as_ref().unwrap().hidden.clone();
        write_repeated_u16(&hidden, BF16_SENTINEL)?;
        let request_id = self.allocate_request_id()?;
        let expected_epoch = self
            .state_epoch
            .checked_add(1)
            .ok_or("Native executor state_epoch exhausted")?;
        let response = self.roundtrip(json!({
            "protocol": EXECUTOR_JSONL_PROTOCOL,
            "request_id": request_id,
            "op": "embed_row",
            "token_id": token_id,
            "position": state.position,
            "state_epoch": self.state_epoch,
            "base": base,
            "state": self.state_wire(state)?,
        }))?;
        self.validate_state_response(&response, "embed_row", state.position, None, expected_epoch)?;
        validate_bf16_region(&hidden, Some(BF16_SENTINEL))?;
        self.state_epoch = expected_epoch;
        self.active_position = Some(state.position);
        self.active_token_id = Some(token_id);
        Ok(())
    }

    fn attention_impl(
        &mut self,
        layer: u8,
        input_token_id: u32,
        base: &ReadyBaseLease,
        state: &mut NativeState,
    ) -> Result<RouteDecision, String> {
        self.ensure_arena(state)?;
        let expected_layer = SELECTED_LAYERS
            .get(self.next_layer_index)
            .copied()
            .ok_or("attention_then_route 已无待执行层")?;
        if self.active_position != Some(state.position)
            || self.active_token_id != Some(input_token_id)
            || layer != expected_layer
            || base.layer != layer
            || self.pending_route.is_some()
        {
            return Err("attention_then_route position/layer/base/order 不匹配".into());
        }
        self.validate_artifacts(&base.artifacts, "attention/base")?;
        let hidden = self.arena.as_ref().unwrap().hidden.clone();
        let before = read_region(&hidden)?;
        let request_id = self.allocate_request_id()?;
        let expected_epoch = self
            .state_epoch
            .checked_add(1)
            .ok_or("Native executor state_epoch exhausted")?;
        let response = self.roundtrip(json!({
            "protocol": EXECUTOR_JSONL_PROTOCOL,
            "request_id": request_id,
            "op": "attention_then_route",
            "layer": layer,
            "input_token_id": input_token_id,
            "position": state.position,
            "state_epoch": self.state_epoch,
            "base": base,
            "state": self.state_wire(state)?,
        }))?;
        self.validate_state_response(
            &response,
            "attention_then_route",
            state.position,
            Some(layer),
            expected_epoch,
        )?;
        let after = validate_bf16_region(&hidden, None)?;
        if after == before {
            return Err("attention_then_route 未提交 hidden/state 变更".into());
        }
        let route = RouteDecision {
            layer,
            kind: response
                .router_kind
                .ok_or("attention_then_route 缺少 router_kind")?,
            expert_ids: response.expert_ids,
            weights: response.route_weights,
        };
        route
            .validate_for(GraphProfile::S14Top6)
            .map_err(|error| error.to_string())?;
        self.state_epoch = expected_epoch;
        self.pending_route = Some(route.clone());
        Ok(route)
    }

    fn moe_impl(
        &mut self,
        layer: u8,
        route: &RouteDecision,
        base: &ReadyBaseLease,
        routed: &ReadyRoutedLease,
        state: &mut NativeState,
    ) -> Result<(), String> {
        self.ensure_arena(state)?;
        if self.active_position != Some(state.position)
            || self.pending_route.as_ref() != Some(route)
            || base.layer != layer
            || routed.layer != layer
            || routed.expert_ids != route.expert_ids
        {
            return Err("routed/shared MoE position/layer/route/lease 不匹配".into());
        }
        self.validate_artifacts(&base.artifacts, "moe/base")?;
        self.validate_artifacts(&routed.artifacts, "moe/routed")?;
        let hidden = self.arena.as_ref().unwrap().hidden.clone();
        let before = read_region(&hidden)?;
        let request_id = self.allocate_request_id()?;
        let expected_epoch = self
            .state_epoch
            .checked_add(1)
            .ok_or("Native executor state_epoch exhausted")?;
        let response = self.roundtrip(json!({
            "protocol": EXECUTOR_JSONL_PROTOCOL,
            "request_id": request_id,
            "op": "routed_and_shared_moe_then_hc_post",
            "layer": layer,
            "position": state.position,
            "state_epoch": self.state_epoch,
            "route": route,
            "base": base,
            "routed": routed,
            "state": self.state_wire(state)?,
        }))?;
        self.validate_state_response(
            &response,
            "routed_and_shared_moe_then_hc_post",
            state.position,
            Some(layer),
            expected_epoch,
        )?;
        let after = validate_bf16_region(&hidden, None)?;
        if after == before {
            return Err("routed/shared MoE 未提交 hidden/state 变更".into());
        }
        self.state_epoch = expected_epoch;
        self.pending_route = None;
        self.next_layer_index += 1;
        Ok(())
    }

    fn head_impl(
        &mut self,
        final_artifacts: &[RangeArtifact],
        state: &NativeState,
    ) -> Result<Vec<f32>, String> {
        self.ensure_arena(state)?;
        if self.active_position != Some(state.position)
            || self.next_layer_index != SELECTED_LAYERS.len()
            || self.pending_route.is_some()
        {
            return Err("hc_head 只能在当前 position 全部 S14 层提交后执行".into());
        }
        self.validate_artifacts(final_artifacts, "head/final")?;
        let logits = self.arena.as_ref().unwrap().logits.clone();
        write_repeated_u32(&logits, F32_SENTINEL)?;
        let request_id = self.allocate_request_id()?;
        let response = self.roundtrip(json!({
            "protocol": EXECUTOR_JSONL_PROTOCOL,
            "request_id": request_id,
            "op": "hc_head_norm_full_logits",
            "position": state.position,
            "state_epoch": self.state_epoch,
            "final_artifacts": final_artifacts,
            "state": self.state_wire(state)?,
            "logits_out": logits,
        }))?;
        if response.position != Some(state.position)
            || response.layer.is_some()
            || response.state_epoch != Some(self.state_epoch)
            || response.logits_written.as_ref() != Some(&logits)
        {
            self.terminate();
            return Err("hc_head logits descriptor/position/epoch 漂移".into());
        }
        let arena = self.arena.as_ref().unwrap();
        validate_file_length(&arena.path, arena.file_bytes)?;
        let values = read_f32_region(&logits)?;
        self.active_position = None;
        self.active_token_id = None;
        self.next_layer_index = 0;
        Ok(values)
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn arena_path(&self) -> Option<&Path> {
        self.arena.as_ref().map(|arena| arena.path.as_path())
    }
}

impl NativeS14Executor for SubprocessNativeExecutor {
    fn embed_row(
        &mut self,
        token_id: u32,
        base: &ReadyBaseLease,
        state: &mut NativeState,
    ) -> Result<(), String> {
        let result = self.embed_impl(token_id, base, state);
        if result.is_err() {
            state.poisoned = true;
            self.terminate();
        }
        result
    }

    fn attention_then_route(
        &mut self,
        layer: u8,
        input_token_id: u32,
        base: &ReadyBaseLease,
        state: &mut NativeState,
    ) -> Result<RouteDecision, String> {
        let result = self.attention_impl(layer, input_token_id, base, state);
        if result.is_err() {
            state.poisoned = true;
            self.terminate();
        }
        result
    }

    fn routed_and_shared_moe_then_hc_post(
        &mut self,
        layer: u8,
        route: &RouteDecision,
        base: &ReadyBaseLease,
        routed: &ReadyRoutedLease,
        state: &mut NativeState,
    ) -> Result<(), String> {
        let result = self.moe_impl(layer, route, base, routed, state);
        if result.is_err() {
            state.poisoned = true;
            self.terminate();
        }
        result
    }

    fn hc_head_norm_full_logits(
        &mut self,
        final_artifacts: &[RangeArtifact],
        state: &NativeState,
    ) -> Result<Vec<f32>, String> {
        let result = self.head_impl(final_artifacts, state);
        if result.is_err() {
            self.terminate();
        }
        result
    }
}

impl Drop for SubprocessNativeExecutor {
    fn drop(&mut self) {
        if !self.poisoned {
            if let Ok(request_id) = self.allocate_request_id() {
                let request = json!({
                    "protocol": EXECUTOR_JSONL_PROTOCOL,
                    "request_id": request_id,
                    "op": "shutdown",
                });
                if let Ok(encoded) = serde_json::to_vec(&request) {
                    let _ = self.requests.send(encoded);
                }
            }
        }
        self.stop_child();
        if let Some(arena) = self.arena.take() {
            let _ = std::fs::remove_file(arena.path);
        }
    }
}

fn tensor_bytes(shape: &[u32], element_bytes: u64) -> Result<u64, String> {
    if shape.is_empty() || shape.contains(&0) {
        return Err("binary tensor shape 必须非空且各维为正数".into());
    }
    shape.iter().try_fold(element_bytes, |bytes, dimension| {
        bytes
            .checked_mul(*dimension as u64)
            .ok_or_else(|| "binary tensor bytes 溢出".into())
    })
}

fn validate_native_state_slices(state: &NativeState) -> Result<(), String> {
    if state.arena_bytes == 0 || !state.arena_bytes.is_multiple_of(256) {
        return Err("NativeState arena_bytes 必须是非零 256-byte 对齐".into());
    }
    let arena_id = state.hc.streams.arena_id;
    if arena_id == 0 {
        return Err("NativeState arena_id 不能为 0".into());
    }
    let mut slices: Vec<&BufferSlice> = vec![&state.hc.streams];
    slices.extend(state.kv.iter().map(|entry| &entry.cache));
    for entry in &state.compressors {
        slices.push(&entry.kv_state);
        slices.push(&entry.score_state);
    }
    for entry in &state.indexers {
        slices.push(&entry.kv_cache);
        slices.push(&entry.compressor_kv_state);
        slices.push(&entry.compressor_score_state);
    }
    let mut intervals = Vec::with_capacity(slices.len());
    for slice in slices {
        let element_bytes = match slice.dtype {
            DType::Bf16 => 2,
            DType::F32 => 4,
        };
        let expected_bytes = tensor_bytes(&slice.shape, element_bytes)?;
        let end = slice
            .offset
            .checked_add(slice.bytes)
            .ok_or("NativeState BufferSlice end 溢出")?;
        if slice.arena_id != arena_id
            || slice.offset % 256 != 0
            || slice.bytes != expected_bytes
            || end > state.arena_bytes
        {
            return Err(format!(
                "NativeState BufferSlice 契约失败: arena={} offset={} bytes={} shape={:?}",
                slice.arena_id, slice.offset, slice.bytes, slice.shape
            ));
        }
        intervals.push((slice.offset, end));
    }
    intervals.sort_unstable();
    if intervals.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err("NativeState BufferSlice 发生重叠".into());
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|candidate| candidate & !mask)
        .ok_or_else(|| "Executor arena alignment 溢出".into())
}

fn create_unique_arena(directory: &Path) -> Result<(PathBuf, File), String> {
    for _ in 0..128 {
        let sequence = ARENA_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(
            "polaris-s14-executor-{}-{sequence}.bin",
            std::process::id()
        ));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("创建 Executor arena 失败: {error}")),
        }
    }
    Err("无法分配唯一 Executor arena 文件".into())
}

fn validate_view_bounds(view: &BinaryTensorView, file_bytes: u64) -> Result<(), String> {
    let end = view
        .offset
        .checked_add(view.bytes)
        .ok_or("binary tensor view 字节溢出")?;
    if view.bytes == 0 || end > file_bytes || !Path::new(&view.path).is_absolute() {
        return Err("binary tensor view 越界/路径非法".into());
    }
    Ok(())
}

fn validate_file_length(path: &Path, expected: u64) -> Result<(), String> {
    let actual = std::fs::metadata(path)
        .map_err(|error| format!("Executor arena 不可读: {error}"))?
        .len();
    if actual != expected {
        return Err(format!(
            "Executor arena 长度漂移: expected={expected} actual={actual}"
        ));
    }
    Ok(())
}

fn read_region(view: &BinaryTensorView) -> Result<Vec<u8>, String> {
    let length: usize = view
        .bytes
        .try_into()
        .map_err(|_| "binary tensor 太大，无法映射到 usize")?;
    let mut file =
        File::open(&view.path).map_err(|error| format!("打开 binary tensor 失败: {error}"))?;
    file.seek(SeekFrom::Start(view.offset))
        .map_err(|error| format!("定位 binary tensor 失败: {error}"))?;
    let mut bytes = vec![0u8; length];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("读取 binary tensor 失败: {error}"))?;
    Ok(bytes)
}

fn write_repeated_u16(view: &BinaryTensorView, value: u16) -> Result<(), String> {
    if view.dtype != "bf16_le" || !view.bytes.is_multiple_of(2) {
        return Err("u16 binary tensor descriptor 不兼容".into());
    }
    let count: usize = (view.bytes / 2)
        .try_into()
        .map_err(|_| "u16 binary tensor 太大")?;
    let pattern = value.to_le_bytes();
    let mut bytes = Vec::with_capacity(count * 2);
    for _ in 0..count {
        bytes.extend_from_slice(&pattern);
    }
    write_region(view, &bytes)
}

fn write_repeated_u32(view: &BinaryTensorView, value: u32) -> Result<(), String> {
    if view.dtype != "f32_le" || !view.bytes.is_multiple_of(4) {
        return Err("u32 binary tensor descriptor 不兼容".into());
    }
    let count: usize = (view.bytes / 4)
        .try_into()
        .map_err(|_| "u32 binary tensor 太大")?;
    let pattern = value.to_le_bytes();
    let mut bytes = Vec::with_capacity(count * 4);
    for _ in 0..count {
        bytes.extend_from_slice(&pattern);
    }
    write_region(view, &bytes)
}

fn write_region(view: &BinaryTensorView, bytes: &[u8]) -> Result<(), String> {
    if bytes.len() as u64 != view.bytes {
        return Err("binary tensor 写入长度与 descriptor 不一致".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .open(&view.path)
        .map_err(|error| format!("打开 binary tensor 写入失败: {error}"))?;
    file.seek(SeekFrom::Start(view.offset))
        .map_err(|error| format!("定位 binary tensor 写入失败: {error}"))?;
    file.write_all(bytes)
        .and_then(|_| file.flush())
        .map_err(|error| format!("写入 binary tensor 失败: {error}"))?;
    file.sync_data()
        .map_err(|error| format!("同步 binary tensor 失败: {error}"))
}

fn validate_bf16_region(
    view: &BinaryTensorView,
    forbidden: Option<u16>,
) -> Result<Vec<u8>, String> {
    let bytes = read_region(view)?;
    if bytes.len() % 2 != 0 {
        return Err("BF16 hidden 字节数不是 2 的倍数".into());
    }
    for pair in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        if forbidden == Some(bits) || bits & 0x7f80 == 0x7f80 {
            return Err("BF16 hidden 含未写 sentinel/NaN/Inf".into());
        }
    }
    Ok(bytes)
}

fn read_f32_region(view: &BinaryTensorView) -> Result<Vec<f32>, String> {
    let bytes = read_region(view)?;
    if bytes.len() % 4 != 0 {
        return Err("F32 logits 字节数不是 4 的倍数".into());
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let value = f32::from_bits(bits);
        if bits == F32_SENTINEL || !value.is_finite() {
            return Err("F32 logits 含未写 sentinel/NaN/Inf".into());
        }
        values.push(value);
    }
    Ok(values)
}

fn read_bounded_jsonl(reader: &mut BufReader<ChildStdout>) -> Result<Option<String>, String> {
    let mut bytes = Vec::new();
    loop {
        let buffer = reader
            .fill_buf()
            .map_err(|error| format!("读取 Native executor worker 失败: {error}"))?;
        if buffer.is_empty() {
            if bytes.is_empty() {
                return Ok(None);
            }
            break;
        }
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let take = newline.map_or(buffer.len(), |index| index + 1);
        if bytes.len() + take > MAX_JSONL_BYTES {
            return Err(format!(
                "Native executor JSONL 超过 {} bytes；拒绝 tensor JSON",
                MAX_JSONL_BYTES
            ));
        }
        bytes.extend_from_slice(&buffer[..take]);
        reader.consume(take);
        if newline.is_some() {
            break;
        }
    }
    while bytes.ends_with(b"\n") || bytes.ends_with(b"\r") {
        bytes.pop();
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|error| format!("Native executor JSONL 不是 UTF-8: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BaseLoadTicket, CapabilityManifest, LocalS14Runner, ProviderError, RouteFirstProvider,
        RoutedLoadTicket, RunnerMode, TransferObservation,
    };
    use std::time::Instant;

    fn fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("executor_jsonl_fixture.py")
    }

    fn artifact(tensor: &str, kind: &str, expert_id: Option<u16>) -> RangeArtifact {
        let path = fixture_path().canonicalize().unwrap();
        RangeArtifact {
            tensor: tensor.into(),
            kind: kind.into(),
            expert_id,
            bytes: path.metadata().unwrap().len(),
            path: path.to_string_lossy().into_owned(),
            cache_hit: true,
            observed_sha256: "0".repeat(64),
            authoritative: false,
        }
    }

    fn base_lease(layer: u8) -> ReadyBaseLease {
        ReadyBaseLease {
            layer,
            lease_id: layer as u64 + 1,
            artifacts: vec![artifact(
                &format!("fixture.base.{layer}"),
                "non_expert",
                None,
            )],
            observation: TransferObservation::default(),
        }
    }

    fn spawn_fixture(mode: &str, timeout_ms: u64) -> SubprocessNativeExecutor {
        let python = std::env::var("PYTHON").unwrap_or_else(|_| "python".into());
        let mut command = Command::new(python);
        command.arg("-X").arg("utf8").arg(fixture_path()).arg(mode);
        SubprocessNativeExecutor::spawn_fixture(
            command,
            ExecutorBridgeConfig {
                response_timeout: Duration::from_millis(timeout_ms),
                workspace_dir: None,
                max_arena_bytes: 1 << 20,
            },
            4,
            8,
            32,
        )
        .unwrap()
    }

    #[derive(Default)]
    struct FixtureProvider {
        live: Option<u8>,
        pending_route: Option<RouteDecision>,
        released: usize,
    }

    impl RouteFirstProvider for FixtureProvider {
        fn begin_base_load(
            &mut self,
            layer: u8,
            _input_token_id: u32,
        ) -> Result<BaseLoadTicket, ProviderError> {
            if self.live.replace(layer).is_some() {
                return Err(ProviderError(
                    "fixture provider duplicate live layer".into(),
                ));
            }
            Ok(BaseLoadTicket {
                layer,
                ticket_id: layer as u64 + 1,
            })
        }

        fn wait_base_ready(
            &mut self,
            ticket: BaseLoadTicket,
        ) -> Result<ReadyBaseLease, ProviderError> {
            if self.live != Some(ticket.layer) {
                return Err(ProviderError("fixture base ticket drift".into()));
            }
            Ok(base_lease(ticket.layer))
        }

        fn begin_routed_load(
            &mut self,
            base: &ReadyBaseLease,
            route: &RouteDecision,
        ) -> Result<RoutedLoadTicket, ProviderError> {
            if self.live != Some(base.layer) || route.layer != base.layer {
                return Err(ProviderError("fixture route/base drift".into()));
            }
            self.pending_route = Some(route.clone());
            Ok(RoutedLoadTicket {
                layer: base.layer,
                ticket_id: 10_000 + base.layer as u64,
                expert_ids: route.expert_ids.clone(),
            })
        }

        fn wait_routed_ready(
            &mut self,
            ticket: RoutedLoadTicket,
        ) -> Result<ReadyRoutedLease, ProviderError> {
            let route = self
                .pending_route
                .as_ref()
                .ok_or_else(|| ProviderError("fixture pending route missing".into()))?;
            if route.layer != ticket.layer || route.expert_ids != ticket.expert_ids {
                return Err(ProviderError("fixture routed ticket drift".into()));
            }
            Ok(ReadyRoutedLease {
                layer: ticket.layer,
                lease_id: ticket.ticket_id,
                expert_ids: ticket.expert_ids.clone(),
                artifacts: ticket
                    .expert_ids
                    .iter()
                    .map(|expert| {
                        artifact(
                            &format!("fixture.expert.{}.{}", ticket.layer, expert),
                            "routed_expert",
                            Some(*expert),
                        )
                    })
                    .collect(),
                observation: TransferObservation::default(),
            })
        }

        fn release_layer(&mut self, layer: u8) -> Result<(), ProviderError> {
            if self.live.take() != Some(layer) {
                return Err(ProviderError("fixture release layer drift".into()));
            }
            self.pending_route = None;
            self.released += 1;
            Ok(())
        }

        fn take_final_artifacts(&mut self) -> Result<Vec<RangeArtifact>, ProviderError> {
            if self.live.is_some() || self.released != SELECTED_LAYERS.len() {
                return Err(ProviderError("fixture final requested too early".into()));
            }
            Ok(vec![artifact("fixture.final", "boundary", None)])
        }
    }

    #[test]
    fn small_persistent_fixture_covers_runner_but_produces_no_token() {
        let provider = FixtureProvider::default();
        let executor = spawn_fixture("normal", 1_000);
        let state = NativeState::decode_layout(2).unwrap();
        let mut runner = LocalS14Runner::new(
            &CapabilityManifest::synthetic_test_pass(),
            RunnerMode::SyntheticTest,
            provider,
            executor,
            state,
        )
        .unwrap();

        let error = runner.step(17).unwrap_err().to_string();
        assert!(error.contains("要求完整 129280 logits，实际 32"), "{error}");
        assert!(runner.state().poisoned);
        assert_eq!(runner.counters().committed_tokens, 0);
        let (provider, executor, state) = runner.into_parts();
        assert_eq!(provider.released, SELECTED_LAYERS.len());
        assert!(!executor.is_poisoned());
        assert!(state.poisoned);
        let arena_path = executor.arena_path().unwrap().to_path_buf();
        assert!(arena_path.is_file());
        drop(executor);
        assert!(!arena_path.exists());
    }

    #[test]
    fn timeout_is_bounded_and_poisons_executor_and_state() {
        let mut executor = spawn_fixture("timeout", 200);
        let mut state = NativeState::decode_layout(2).unwrap();
        let base = base_lease(0);
        let started = Instant::now();
        let error = executor.embed_row(17, &base, &mut state).unwrap_err();
        assert!(error.contains("超时"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(executor.is_poisoned());
        assert!(state.poisoned);
    }

    #[test]
    fn worker_error_and_descriptor_drift_are_fail_closed() {
        for (mode, expected) in [
            ("reject", "synthetic executor rejection"),
            ("bad_descriptor", "state/hidden descriptor 漂移"),
            ("bad_position", "position/layer/epoch 漂移"),
            ("malformed", "返回非 JSONL"),
            ("exit", "stdout EOF"),
            ("oversize", "JSONL 超过"),
            ("tensor_json", "禁止字段 hidden_values"),
        ] {
            let mut executor = spawn_fixture(mode, 1_000);
            let mut state = NativeState::decode_layout(2).unwrap();
            let error = executor
                .embed_row(17, &base_lease(0), &mut state)
                .unwrap_err();
            assert!(error.contains(expected), "mode={mode}: {error}");
            assert!(executor.is_poisoned(), "mode={mode}");
            assert!(state.poisoned, "mode={mode}");
        }
    }

    #[test]
    fn public_shape_gate_is_exact_production_contract() {
        let shape = ExecutorShape::production();
        assert!(shape.production);
        assert_eq!(shape.hidden_shape, [4, 4096]);
        assert_eq!(shape.hidden_bytes().unwrap(), 4 * 4096 * 2);
        assert_eq!(shape.logits_shape, [129_280]);
        assert_eq!(shape.logits_bytes().unwrap(), 129_280 * 4);

        let state = NativeState::decode_layout(2).unwrap();
        validate_native_state_slices(&state).unwrap();
        let mut overlapping = state.clone();
        overlapping.kv[0].cache.offset = overlapping.hc.streams.offset;
        assert!(validate_native_state_slices(&overlapping)
            .unwrap_err()
            .contains("发生重叠"));
    }

    #[test]
    fn oversized_request_never_crosses_jsonl_control_plane() {
        let mut executor = spawn_fixture("normal", 1_000);
        let mut state = NativeState::decode_layout(2).unwrap();
        let mut base = base_lease(0);
        base.artifacts[0].tensor = "x".repeat(MAX_JSONL_BYTES);
        let error = executor.embed_row(17, &base, &mut state).unwrap_err();
        assert!(error.contains("request 超过"), "{error}");
        assert!(executor.is_poisoned());
        assert!(state.poisoned);
    }

    #[test]
    fn public_constructor_cannot_negotiate_fixture_shape() {
        let python = std::env::var("PYTHON").unwrap_or_else(|_| "python".into());
        let mut command = Command::new(python);
        command
            .arg("-X")
            .arg("utf8")
            .arg(fixture_path())
            .arg("normal");
        let error = SubprocessNativeExecutor::spawn(
            command,
            ExecutorBridgeConfig {
                response_timeout: Duration::from_secs(1),
                workspace_dir: None,
                max_arena_bytes: DEFAULT_MAX_ARENA_BYTES,
            },
        )
        .err()
        .unwrap();
        assert!(error.contains("fixture refuses production mode"), "{error}");
    }
}
