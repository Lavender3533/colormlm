//! Exact Cascade 的 fail-closed FullDepth43/native-top6 数据合同。
//!
//! 本模块只定义生产 gate、causal-block 请求/响应和原子提交状态机，不实现
//! 模型算子。当前 capability manifest 会在构造生产会话前硬拒绝，因此合同
//! 测试中的合成响应不可能经由生产入口返回 token。

#[cfg(test)]
use crate::{router_kind_for_layer, EvidenceKind};
use crate::{
    CapabilityManifest, GateError, GraphProfile, NativeState, RouteDecision, RouterKind,
    StateLayoutError, COMPRESS_RATIOS, FULL_DEPTH_LAYERS, MODEL_REPO, MODEL_REVISION, N_LAYERS,
    VOCAB_SIZE,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

pub const EXACT_CASCADE_PROFILE_ID: &str = "FullDepth43/native-top6";
pub const EXACT_CASCADE_BLOCK_SIZES: [usize; 3] = [1, 4, 8];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateMutationStatus {
    Advanced,
    NotApplicable,
}

/// 一次 token position 的单层原生执行记录。布尔字段必须全部为 true；
/// compressor/indexer 则严格跟随官方 compression ratio 配置。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeLayerRecord {
    pub layer: u8,
    pub router_kind: RouterKind,
    pub expert_ids: Vec<u16>,
    pub route_weights: Vec<f32>,
    pub attention_executed: bool,
    pub shared_expert_executed: bool,
    pub mhc_state_advanced: bool,
    pub kv_state_advanced: bool,
    pub compressor_state: StateMutationStatus,
    pub indexer_state: StateMutationStatus,
}

impl NativeLayerRecord {
    fn validate(&self, expected_layer: u8) -> Result<(), ExactCascadeError> {
        if self.layer != expected_layer {
            return Err(contract_error(format!(
                "causal position 缺层或乱序: 期望 L{expected_layer}，实际 L{}",
                self.layer
            )));
        }
        let route = RouteDecision {
            layer: self.layer,
            kind: self.router_kind,
            expert_ids: self.expert_ids.clone(),
            weights: self.route_weights.clone(),
        };
        route
            .validate_for(GraphProfile::FullDepth43NativeTop6)
            .map_err(|error| contract_error(error.to_string()))?;
        if !self.attention_executed
            || !self.shared_expert_executed
            || !self.mhc_state_advanced
            || !self.kv_state_advanced
        {
            return Err(contract_error(format!(
                "L{} 必须执行 attention、top6+shared、mHC 与 KV 状态推进",
                self.layer
            )));
        }
        let ratio = COMPRESS_RATIOS[self.layer as usize];
        let expected_compressor = if ratio == 0 {
            StateMutationStatus::NotApplicable
        } else {
            StateMutationStatus::Advanced
        };
        let expected_indexer = if ratio == 4 {
            StateMutationStatus::Advanced
        } else {
            StateMutationStatus::NotApplicable
        };
        if self.compressor_state != expected_compressor || self.indexer_state != expected_indexer {
            return Err(contract_error(format!(
                "L{} compressor/indexer 状态与官方 ratio={} 不一致",
                self.layer, ratio
            )));
        }
        Ok(())
    }
}

/// 后端 checkpoint 必须能在未提交分支失败时恢复；ID 只是 ABI 句柄，
/// 只有 capability 中的实测原子 checkpoint 证据通过后才可用于生产。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeStateCheckpoint {
    pub checkpoint_id: String,
    pub parent_checkpoint_id: String,
    pub position_after: u32,
    pub committed_token_id: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalPositionResponse {
    pub token_offset: usize,
    pub predicted_token_id: u32,
    pub layers: Vec<NativeLayerRecord>,
    pub state_after_prediction: NativeStateCheckpoint,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactCascadeRequest {
    pub request_id: u64,
    pub repo: String,
    pub revision: String,
    pub profile: GraphProfile,
    pub causal: bool,
    pub depth: u8,
    pub top_k: usize,
    pub block_size: usize,
    pub base_position: u32,
    pub base_checkpoint_id: String,
    pub context_token_ids: Vec<u32>,
    pub draft_token_ids: Vec<u32>,
}

impl ExactCascadeRequest {
    pub fn validate(&self) -> Result<(), ExactCascadeError> {
        if self.repo != MODEL_REPO || self.revision != MODEL_REVISION {
            return Err(contract_error("拒绝非冻结 repo/revision"));
        }
        if self.profile != GraphProfile::FullDepth43NativeTop6
            || !self.causal
            || self.depth != N_LAYERS
            || self.top_k != 6
        {
            return Err(contract_error(
                "请求必须是 FullDepth43/native-top6 causal block",
            ));
        }
        if !EXACT_CASCADE_BLOCK_SIZES.contains(&self.block_size)
            || self.draft_token_ids.len() != self.block_size
        {
            return Err(contract_error("block_size 只允许 K=1/4/8 且草稿必须等长"));
        }
        validate_token_ids(&self.context_token_ids)?;
        validate_token_ids(&self.draft_token_ids)?;
        if self.base_checkpoint_id.trim().is_empty() {
            return Err(contract_error("base checkpoint ID 不能为空"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactCascadeResponse {
    pub request_id: u64,
    pub repo: String,
    pub revision: String,
    pub profile: GraphProfile,
    pub causal: bool,
    pub depth: u8,
    pub top_k: usize,
    pub block_size: usize,
    pub base_position: u32,
    pub base_checkpoint_id: String,
    pub positions: Vec<CausalPositionResponse>,
}

impl ExactCascadeResponse {
    pub fn validate_against(&self, request: &ExactCascadeRequest) -> Result<(), ExactCascadeError> {
        request.validate()?;
        if self.request_id != request.request_id
            || self.repo != request.repo
            || self.revision != request.revision
            || self.profile != request.profile
            || self.causal != request.causal
            || self.depth != request.depth
            || self.top_k != request.top_k
            || self.block_size != request.block_size
            || self.base_position != request.base_position
            || self.base_checkpoint_id != request.base_checkpoint_id
        {
            return Err(contract_error("causal-block 响应身份与请求不一致"));
        }
        if self.positions.len() != request.block_size {
            return Err(contract_error("响应 position 数必须与 K 一一对齐"));
        }

        let mut parent = request.base_checkpoint_id.as_str();
        let mut checkpoint_ids = BTreeSet::new();
        for (offset, position) in self.positions.iter().enumerate() {
            if position.token_offset != offset || position.predicted_token_id >= VOCAB_SIZE {
                return Err(contract_error("响应 token offset/ID 非法"));
            }
            if position.layers.len() != FULL_DEPTH_LAYERS.len() {
                return Err(contract_error(format!(
                    "position {offset} 必须恰好包含 43 层"
                )));
            }
            for (record, &layer) in position.layers.iter().zip(FULL_DEPTH_LAYERS.iter()) {
                record.validate(layer)?;
            }

            let checkpoint = &position.state_after_prediction;
            let expected_position = request
                .base_position
                .checked_add(offset as u32 + 1)
                .ok_or_else(|| contract_error("checkpoint position 溢出"))?;
            if checkpoint.checkpoint_id.trim().is_empty()
                || checkpoint.parent_checkpoint_id != parent
                || checkpoint.position_after != expected_position
                || checkpoint.committed_token_id != position.predicted_token_id
                || !checkpoint_ids.insert(checkpoint.checkpoint_id.clone())
            {
                return Err(contract_error(format!(
                    "position {offset} checkpoint 链或 token/position 非法"
                )));
            }
            parent = checkpoint.checkpoint_id.as_str();
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExactCascadePhase {
    Ready,
    Drafting,
    AwaitingVerification,
    Verifying,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactCascadeCommit {
    pub accepted_prefix: Vec<u32>,
    pub fallback_token_id: Option<u32>,
    pub rejected_draft_suffix: Vec<u32>,
    pub committed_token_ids: Vec<u32>,
    pub mismatch_index: Option<usize>,
    pub committed_checkpoint_id: String,
}

#[derive(Debug, Clone)]
struct PendingRound {
    request_id: u64,
    block_size: usize,
    draft_token_ids: Vec<u32>,
    request: Option<ExactCascadeRequest>,
    base_context_len: usize,
    base_state: NativeState,
    base_checkpoint_id: String,
}

/// 生产构造函数始终先过 FullDepth43 capability gate。会话在完整响应校验
/// 成功前不修改 context、NativeState 或 committed checkpoint。
pub struct ExactCascadeSession {
    context_token_ids: Vec<u32>,
    native_state: NativeState,
    committed_checkpoint_id: String,
    phase: ExactCascadePhase,
    pending: Option<PendingRound>,
    next_request_id: u64,
}

impl ExactCascadeSession {
    pub fn new(
        manifest: &CapabilityManifest,
        context_token_ids: Vec<u32>,
        native_state: NativeState,
        committed_checkpoint_id: String,
    ) -> Result<Self, ExactCascadeError> {
        manifest.gate_production()?;
        Self::new_gated(
            manifest,
            context_token_ids,
            native_state,
            committed_checkpoint_id,
        )
    }

    #[cfg(test)]
    fn new_synthetic(
        manifest: &CapabilityManifest,
        context_token_ids: Vec<u32>,
        native_state: NativeState,
        committed_checkpoint_id: String,
    ) -> Result<Self, ExactCascadeError> {
        manifest.validate_identity()?;
        if manifest.evidence_kind != EvidenceKind::SyntheticTest || manifest.native_forward_ready {
            return Err(ExactCascadeError::Gate(GateError::Identity(
                "synthetic Exact Cascade 只允许 cfg(test) evidence".into(),
            )));
        }
        Self::new_gated(
            manifest,
            context_token_ids,
            native_state,
            committed_checkpoint_id,
        )
    }

    fn new_gated(
        manifest: &CapabilityManifest,
        context_token_ids: Vec<u32>,
        native_state: NativeState,
        committed_checkpoint_id: String,
    ) -> Result<Self, ExactCascadeError> {
        if manifest.profile != GraphProfile::FullDepth43NativeTop6 {
            return Err(contract_error(
                "Exact Cascade 生产入口只接受 FullDepth43/native-top6",
            ));
        }
        native_state.validate_for(manifest.profile)?;
        validate_token_ids(&context_token_ids)?;
        if context_token_ids.len() as u64 != native_state.position as u64 {
            return Err(contract_error("context 长度与递归状态 position 不一致"));
        }
        if committed_checkpoint_id.trim().is_empty() {
            return Err(contract_error("committed checkpoint ID 不能为空"));
        }
        Ok(Self {
            context_token_ids,
            native_state,
            committed_checkpoint_id,
            phase: ExactCascadePhase::Ready,
            pending: None,
            next_request_id: 1,
        })
    }

    pub fn begin_round(&mut self, block_size: usize) -> Result<(), ExactCascadeError> {
        if self.phase != ExactCascadePhase::Ready {
            return Err(contract_error("只能从 ready 开始 Exact Cascade round"));
        }
        if !EXACT_CASCADE_BLOCK_SIZES.contains(&block_size) {
            return Err(contract_error("Exact Cascade 只允许 K=1/4/8"));
        }
        let next_position = self
            .native_state
            .position
            .checked_add(block_size as u32)
            .ok_or_else(|| contract_error("block position 溢出"))?;
        if next_position > self.native_state.max_seq_len {
            return Err(contract_error("causal block 超出 max_seq_len"));
        }
        let following_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or_else(|| contract_error("request ID 溢出"))?;
        self.pending = Some(PendingRound {
            request_id: self.next_request_id,
            block_size,
            draft_token_ids: Vec::new(),
            request: None,
            base_context_len: self.context_token_ids.len(),
            base_state: self.native_state.clone(),
            base_checkpoint_id: self.committed_checkpoint_id.clone(),
        });
        self.next_request_id = following_request_id;
        self.phase = ExactCascadePhase::Drafting;
        Ok(())
    }

    pub fn submit_draft(&mut self, draft_token_ids: Vec<u32>) -> Result<(), ExactCascadeError> {
        if self.phase != ExactCascadePhase::Drafting {
            return Err(contract_error("当前 phase 不接受草稿"));
        }
        let block_size = self
            .pending
            .as_ref()
            .expect("drafting 必有 pending")
            .block_size;
        if draft_token_ids.len() != block_size {
            self.rollback_pending();
            return Err(contract_error("草稿 token 数必须恰好等于 K"));
        }
        if let Err(error) = validate_token_ids(&draft_token_ids) {
            self.rollback_pending();
            return Err(error);
        }
        self.pending
            .as_mut()
            .expect("drafting 必有 pending")
            .draft_token_ids = draft_token_ids;
        self.phase = ExactCascadePhase::AwaitingVerification;
        Ok(())
    }

    pub fn make_request(&mut self) -> Result<ExactCascadeRequest, ExactCascadeError> {
        if self.phase != ExactCascadePhase::AwaitingVerification {
            return Err(contract_error("当前 phase 不能发起 FullDepth 验证"));
        }
        let pending = self.pending.as_mut().expect("awaiting 必有 pending");
        let request = ExactCascadeRequest {
            request_id: pending.request_id,
            repo: MODEL_REPO.into(),
            revision: MODEL_REVISION.into(),
            profile: GraphProfile::FullDepth43NativeTop6,
            causal: true,
            depth: N_LAYERS,
            top_k: 6,
            block_size: pending.block_size,
            base_position: pending.base_state.position,
            base_checkpoint_id: pending.base_checkpoint_id.clone(),
            context_token_ids: self.context_token_ids.clone(),
            draft_token_ids: pending.draft_token_ids.clone(),
        };
        request.validate()?;
        pending.request = Some(request.clone());
        self.phase = ExactCascadePhase::Verifying;
        Ok(request)
    }

    pub fn finish_response(
        &mut self,
        response: ExactCascadeResponse,
    ) -> Result<ExactCascadeCommit, ExactCascadeError> {
        if self.phase != ExactCascadePhase::Verifying {
            return Err(contract_error("当前没有待完成的 FullDepth 验证"));
        }
        let result = self.validate_and_plan_commit(&response);
        match result {
            Ok((commit, new_state)) => {
                self.context_token_ids
                    .extend(commit.committed_token_ids.iter().copied());
                self.native_state = new_state;
                self.committed_checkpoint_id = commit.committed_checkpoint_id.clone();
                self.pending = None;
                self.phase = ExactCascadePhase::Ready;
                Ok(commit)
            }
            Err(error) => {
                self.rollback_pending();
                Err(error)
            }
        }
    }

    fn validate_and_plan_commit(
        &self,
        response: &ExactCascadeResponse,
    ) -> Result<(ExactCascadeCommit, NativeState), ExactCascadeError> {
        let pending = self.pending.as_ref().expect("verifying 必有 pending");
        let request = pending.request.as_ref().expect("verifying 必有 request");
        response.validate_against(request)?;

        let predictions: Vec<u32> = response
            .positions
            .iter()
            .map(|position| position.predicted_token_id)
            .collect();
        let mismatch_index = request
            .draft_token_ids
            .iter()
            .zip(predictions.iter())
            .position(|(draft, native)| draft != native);
        let (accepted_prefix, fallback_token_id, rejected_draft_suffix, committed_token_ids) =
            match mismatch_index {
                Some(index) => {
                    let accepted = request.draft_token_ids[..index].to_vec();
                    let fallback = predictions[index];
                    let mut committed = accepted.clone();
                    committed.push(fallback);
                    (
                        accepted,
                        Some(fallback),
                        request.draft_token_ids[index..].to_vec(),
                        committed,
                    )
                }
                None => (
                    request.draft_token_ids.clone(),
                    None,
                    Vec::new(),
                    request.draft_token_ids.clone(),
                ),
            };
        let checkpoint_index = mismatch_index.unwrap_or(request.block_size - 1);
        let checkpoint = &response.positions[checkpoint_index].state_after_prediction;
        let mut new_state = pending.base_state.clone();
        new_state.position = checkpoint.position_after;
        new_state.poisoned = false;
        new_state.validate_for(GraphProfile::FullDepth43NativeTop6)?;
        if pending.base_context_len + committed_token_ids.len() != new_state.position as usize {
            return Err(contract_error(
                "提交 token 数、context 长度与 checkpoint position 不闭合",
            ));
        }
        Ok((
            ExactCascadeCommit {
                accepted_prefix,
                fallback_token_id,
                rejected_draft_suffix,
                committed_token_ids,
                mismatch_index,
                committed_checkpoint_id: checkpoint.checkpoint_id.clone(),
            },
            new_state,
        ))
    }

    pub fn fail_verification(&mut self, message: impl Into<String>) -> ExactCascadeError {
        let error = ExactCascadeError::Backend(message.into());
        self.rollback_pending();
        error
    }

    pub fn abort_round(&mut self) {
        self.rollback_pending();
    }

    fn rollback_pending(&mut self) {
        if let Some(pending) = self.pending.take() {
            self.context_token_ids.truncate(pending.base_context_len);
            self.native_state = pending.base_state;
            self.committed_checkpoint_id = pending.base_checkpoint_id;
        }
        self.phase = ExactCascadePhase::Ready;
    }

    pub fn context_token_ids(&self) -> &[u32] {
        &self.context_token_ids
    }

    pub fn native_state(&self) -> &NativeState {
        &self.native_state
    }

    pub fn committed_checkpoint_id(&self) -> &str {
        &self.committed_checkpoint_id
    }

    pub fn phase(&self) -> ExactCascadePhase {
        self.phase
    }
}

fn validate_token_ids(token_ids: &[u32]) -> Result<(), ExactCascadeError> {
    if token_ids.iter().any(|&token| token >= VOCAB_SIZE) {
        return Err(contract_error("token ID 越出冻结 vocab"));
    }
    Ok(())
}

fn contract_error(message: impl Into<String>) -> ExactCascadeError {
    ExactCascadeError::Contract(message.into())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactCascadeError {
    Gate(GateError),
    State(StateLayoutError),
    Contract(String),
    Backend(String),
}

impl fmt::Display for ExactCascadeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gate(error) => write!(f, "{error}"),
            Self::State(error) => write!(f, "state: {error}"),
            Self::Contract(error) => write!(f, "Exact Cascade contract: {error}"),
            Self::Backend(error) => write!(f, "FullDepth backend: {error}"),
        }
    }
}

impl std::error::Error for ExactCascadeError {}

impl From<GateError> for ExactCascadeError {
    fn from(value: GateError) -> Self {
        Self::Gate(value)
    }
}

impl From<StateLayoutError> for ExactCascadeError {
    fn from(value: StateLayoutError) -> Self {
        Self::State(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(context: Vec<u32>) -> ExactCascadeSession {
        let profile = GraphProfile::FullDepth43NativeTop6;
        let manifest = CapabilityManifest::synthetic_test_pass_for(profile);
        let mut state = NativeState::decode_layout_for(profile, 64).unwrap();
        state.position = context.len() as u32;
        ExactCascadeSession::new_synthetic(&manifest, context, state, "root".into()).unwrap()
    }

    fn response(request: &ExactCascadeRequest, predictions: &[u32]) -> ExactCascadeResponse {
        assert_eq!(predictions.len(), request.block_size);
        let mut parent = request.base_checkpoint_id.clone();
        let positions = predictions
            .iter()
            .enumerate()
            .map(|(offset, &predicted_token_id)| {
                let checkpoint_id = format!("round{}-cp{offset}", request.request_id);
                let layers = FULL_DEPTH_LAYERS
                    .iter()
                    .map(|&layer| {
                        let ratio = COMPRESS_RATIOS[layer as usize];
                        NativeLayerRecord {
                            layer,
                            router_kind: router_kind_for_layer(layer).unwrap(),
                            expert_ids: vec![0, 1, 2, 3, 4, 5],
                            route_weights: vec![0.25; 6],
                            attention_executed: true,
                            shared_expert_executed: true,
                            mhc_state_advanced: true,
                            kv_state_advanced: true,
                            compressor_state: if ratio == 0 {
                                StateMutationStatus::NotApplicable
                            } else {
                                StateMutationStatus::Advanced
                            },
                            indexer_state: if ratio == 4 {
                                StateMutationStatus::Advanced
                            } else {
                                StateMutationStatus::NotApplicable
                            },
                        }
                    })
                    .collect();
                let position = CausalPositionResponse {
                    token_offset: offset,
                    predicted_token_id,
                    layers,
                    state_after_prediction: NativeStateCheckpoint {
                        checkpoint_id: checkpoint_id.clone(),
                        parent_checkpoint_id: parent.clone(),
                        position_after: request.base_position + offset as u32 + 1,
                        committed_token_id: predicted_token_id,
                    },
                };
                parent = checkpoint_id;
                position
            })
            .collect();
        ExactCascadeResponse {
            request_id: request.request_id,
            repo: request.repo.clone(),
            revision: request.revision.clone(),
            profile: request.profile,
            causal: request.causal,
            depth: request.depth,
            top_k: request.top_k,
            block_size: request.block_size,
            base_position: request.base_position,
            base_checkpoint_id: request.base_checkpoint_id.clone(),
            positions,
        }
    }

    fn request_for(session: &mut ExactCascadeSession, draft: &[u32]) -> ExactCascadeRequest {
        session.begin_round(draft.len()).unwrap();
        session.submit_draft(draft.to_vec()).unwrap();
        session.make_request().unwrap()
    }

    #[test]
    fn k1_request_response_is_full_depth_native_top6() {
        let mut session = session(vec![90]);
        let request = request_for(&mut session, &[7]);
        assert_eq!(request.block_size, 1);
        assert_eq!(request.profile, GraphProfile::FullDepth43NativeTop6);
        assert_eq!(request.depth, 43);
        assert_eq!(request.top_k, 6);
        let commit = session.finish_response(response(&request, &[7])).unwrap();
        assert_eq!(commit.accepted_prefix, [7]);
        assert_eq!(commit.fallback_token_id, None);
        assert_eq!(session.context_token_ids(), [90, 7]);
        assert_eq!(session.native_state().position, 2);
    }

    #[test]
    fn k4_commits_only_longest_prefix_and_native_fallback() {
        let mut session = session(vec![90]);
        let request = request_for(&mut session, &[1, 2, 3, 4]);
        let commit = session
            .finish_response(response(&request, &[1, 2, 9, 777]))
            .unwrap();
        assert_eq!(commit.accepted_prefix, [1, 2]);
        assert_eq!(commit.fallback_token_id, Some(9));
        assert_eq!(commit.rejected_draft_suffix, [3, 4]);
        assert_eq!(commit.committed_token_ids, [1, 2, 9]);
        assert_eq!(commit.mismatch_index, Some(2));
        assert_eq!(commit.committed_checkpoint_id, "round1-cp2");
        assert_eq!(session.context_token_ids(), [90, 1, 2, 9]);
        assert_eq!(session.native_state().position, 4);
    }

    #[test]
    fn k8_full_match_commits_exactly_eight_without_bonus() {
        let draft = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut session = session(Vec::new());
        let request = request_for(&mut session, &draft);
        let commit = session.finish_response(response(&request, &draft)).unwrap();
        assert_eq!(commit.accepted_prefix, draft);
        assert_eq!(commit.fallback_token_id, None);
        assert_eq!(commit.committed_token_ids.len(), 8);
        assert_eq!(session.native_state().position, 8);
    }

    #[test]
    fn skipped_layer_or_top1_reduction_is_rejected_and_rolled_back() {
        for mutate in 0..2 {
            let mut session = session(vec![90]);
            let before_state = session.native_state().clone();
            let before_context = session.context_token_ids().to_vec();
            let before_checkpoint = session.committed_checkpoint_id().to_string();
            let request = request_for(&mut session, &[1, 2, 3, 4]);
            let mut invalid = response(&request, &[1, 2, 3, 4]);
            if mutate == 0 {
                invalid.positions[2].layers.pop();
            } else {
                invalid.positions[2].layers[7].expert_ids = vec![0];
                invalid.positions[2].layers[7].route_weights = vec![1.5];
            }
            assert!(session.finish_response(invalid).is_err());
            assert_eq!(session.context_token_ids(), before_context);
            assert_eq!(session.native_state(), &before_state);
            assert_eq!(session.committed_checkpoint_id(), before_checkpoint);
            assert_eq!(session.phase(), ExactCascadePhase::Ready);
        }
    }

    #[test]
    fn k8_backend_failure_rolls_back_all_pending_state() {
        let mut session = session(vec![90]);
        let before_state = session.native_state().clone();
        let _request = request_for(&mut session, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let error = session.fail_verification("injected operator failure");
        assert!(matches!(error, ExactCascadeError::Backend(_)));
        assert_eq!(session.context_token_ids(), [90]);
        assert_eq!(session.native_state(), &before_state);
        assert_eq!(session.committed_checkpoint_id(), "root");
        assert_eq!(session.phase(), ExactCascadePhase::Ready);
    }

    #[test]
    fn current_capability_hard_rejects_before_any_token_can_be_emitted() {
        let manifest: CapabilityManifest =
            serde_json::from_str(crate::CURRENT_FULL_DEPTH_CAPABILITIES_JSON).unwrap();
        let state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 64).unwrap();
        let result = ExactCascadeSession::new(&manifest, Vec::new(), state, "root".into());
        assert!(matches!(
            result,
            Err(ExactCascadeError::Gate(GateError::Unavailable(_)))
        ));
    }

    #[test]
    fn independent_wire_contract_matches_rust_constants() {
        let wire: serde_json::Value =
            serde_json::from_str(crate::EXACT_CASCADE_CONTRACT_JSON).unwrap();
        assert_eq!(wire["verifier_profile"], EXACT_CASCADE_PROFILE_ID);
        assert_eq!(wire["allowed_block_sizes"], serde_json::json!([1, 4, 8]));
        assert_eq!(wire["depth"], 43);
        assert_eq!(wire["experts_per_token_per_layer"], 6);
        assert_eq!(wire["production"]["token_on_gate_failure"], "forbidden");
    }
}
