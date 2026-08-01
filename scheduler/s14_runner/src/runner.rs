use crate::{
    router_kind_for_layer, CapabilityManifest, EvidenceKind, GateError, GraphProfile, NativeState,
    RouterKind, RuntimeCounters, StateLayoutError, TransferObservation, N_LAYERS, N_ROUTED_EXPERTS,
    VOCAB_SIZE,
};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RangeArtifact {
    pub tensor: String,
    pub kind: String,
    pub expert_id: Option<u16>,
    pub path: String,
    pub bytes: u64,
    pub cache_hit: bool,
    pub observed_sha256: String,
    pub authoritative: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunnerMode {
    Production,
    SyntheticTest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseLoadTicket {
    pub layer: u8,
    pub ticket_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadyBaseLease {
    pub layer: u8,
    pub lease_id: u64,
    /// Python Range 状态机已完成 SHA 校验的只读页面。executor 只能消费这些句柄，
    /// 不能据此改变官方 router 的决策。
    pub artifacts: Vec<RangeArtifact>,
    pub observation: TransferObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutedLoadTicket {
    pub layer: u8,
    pub ticket_id: u64,
    pub expert_ids: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadyRoutedLease {
    pub layer: u8,
    pub lease_id: u64,
    pub expert_ids: Vec<u16>,
    pub artifacts: Vec<RangeArtifact>,
    pub observation: TransferObservation,
}

/// Provider 必须分两阶段发布：base 先 Ready，官方 router 真正运行后，才允许
/// 请求 routed expert。`wait_*_ready` 只能在 Vulkan fence 完成后返回。
pub trait RouteFirstProvider {
    fn begin_base_load(
        &mut self,
        layer: u8,
        input_token_id: u32,
    ) -> Result<BaseLoadTicket, ProviderError>;
    fn wait_base_ready(&mut self, ticket: BaseLoadTicket) -> Result<ReadyBaseLease, ProviderError>;
    fn begin_routed_load(
        &mut self,
        base: &ReadyBaseLease,
        route: &RouteDecision,
    ) -> Result<RoutedLoadTicket, ProviderError>;
    fn wait_routed_ready(
        &mut self,
        ticket: RoutedLoadTicket,
    ) -> Result<ReadyRoutedLease, ProviderError>;
    /// 成功、算子失败和传输失败都必须释放当前逻辑层租约。
    fn release_layer(&mut self, layer: u8) -> Result<(), ProviderError>;
    /// 最后一层释放后返回已经校验的 HC/norm/lm-head 页面；其他时刻必须为空。
    fn take_final_artifacts(&mut self) -> Result<Vec<RangeArtifact>, ProviderError>;
}

/// 官方 block 被拆在 router 边界：attention+HC-pre/FFN-norm 先运行并产出
/// route，provider 再装 Top-6，随后执行 routed+shared MoE 和 HC-post。
pub trait NativeS14Executor {
    fn embed_row(
        &mut self,
        token_id: u32,
        base: &ReadyBaseLease,
        state: &mut NativeState,
    ) -> Result<(), String>;

    fn attention_then_route(
        &mut self,
        layer: u8,
        input_token_id: u32,
        base: &ReadyBaseLease,
        state: &mut NativeState,
    ) -> Result<RouteDecision, String>;

    fn routed_and_shared_moe_then_hc_post(
        &mut self,
        layer: u8,
        route: &RouteDecision,
        base: &ReadyBaseLease,
        routed: &ReadyRoutedLease,
        state: &mut NativeState,
    ) -> Result<(), String>;

    /// 必须返回完整 129,280 维 logits。runner 自己执行稳定的最低 ID tie-break
    /// argmax；executor 无权直接返回 token ID。
    fn hc_head_norm_full_logits(
        &mut self,
        final_artifacts: &[RangeArtifact],
        state: &NativeState,
    ) -> Result<Vec<f32>, String>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub layer: u8,
    pub kind: RouterKind,
    pub expert_ids: Vec<u16>,
    pub weights: Vec<f32>,
}

impl RouteDecision {
    pub fn validate_for(&self, profile: GraphProfile) -> Result<(), RunnerError> {
        if profile.layers().binary_search(&self.layer).is_err() {
            return Err(RunnerError::Contract(format!(
                "route layer {} 不属于预注册 profile {:?}",
                self.layer, profile
            )));
        }
        if self.expert_ids.len() != profile.experts_per_token()
            || self.weights.len() != profile.experts_per_token()
        {
            return Err(RunnerError::Contract(format!(
                "profile {:?} 要求 top-{}，实际 ids={} weights={}",
                profile,
                profile.experts_per_token(),
                self.expert_ids.len(),
                self.weights.len()
            )));
        }
        let expected_kind = router_kind_for_layer(self.layer)
            .map_err(|error| RunnerError::Contract(error.to_string()))?;
        if self.kind != expected_kind {
            return Err(RunnerError::Contract(format!(
                "L{} router kind 应为 {:?}",
                self.layer, expected_kind
            )));
        }
        for (index, (&expert, &weight)) in
            self.expert_ids.iter().zip(self.weights.iter()).enumerate()
        {
            if expert >= N_ROUTED_EXPERTS {
                return Err(RunnerError::Contract(format!(
                    "route[{index}] expert {expert} 越界"
                )));
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err(RunnerError::Contract(format!(
                    "route[{index}] weight 非有限非负"
                )));
            }
            if self.expert_ids[..index].contains(&expert) {
                return Err(RunnerError::Contract(format!("route expert {expert} 重复")));
            }
        }
        let sum: f32 = self.weights.iter().sum();
        if (sum - 1.5).abs() > 1e-4 {
            return Err(RunnerError::Contract(format!(
                "官方 normalized weights*route_scale 应和为 1.5，实际 {sum}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerPhase {
    Idle,
    LoadingBase,
    BaseReady,
    Routing,
    LoadingExperts,
    ExpertsReady,
    Executing,
    Releasing,
    Released,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerLifecycle {
    pub layer: u8,
    pub phase: LayerPhase,
}

impl LayerLifecycle {
    pub fn new(layer: u8) -> Self {
        Self {
            layer,
            phase: LayerPhase::Idle,
        }
    }

    pub fn transition(&mut self, next: LayerPhase) -> Result<(), RunnerError> {
        use LayerPhase::*;
        let valid = matches!(
            (self.phase, next),
            (Idle, LoadingBase)
                | (LoadingBase, BaseReady)
                | (BaseReady, Routing)
                | (Routing, LoadingExperts)
                | (LoadingExperts, ExpertsReady)
                | (ExpertsReady, Executing)
                | (Executing, Releasing)
                | (Releasing, Released)
        );
        if !valid {
            return Err(RunnerError::Contract(format!(
                "L{} 非法 lifecycle {:?}->{:?}",
                self.layer, self.phase, next
            )));
        }
        self.phase = next;
        Ok(())
    }

    fn fail(&mut self) {
        self.phase = LayerPhase::Failed;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerEventKind {
    Identity,
    LoadingBase,
    BaseReady,
    Routing,
    LoadingExperts,
    ExpertsReady,
    Executing,
    Releasing,
    Released,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerEvent {
    pub layer: u8,
    pub kind: LayerEventKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GreedyToken {
    pub token_id: u32,
    pub max_logit: f32,
    pub tie_break: String,
    pub origin: RunnerMode,
}

pub struct LocalS14Runner<P, E> {
    provider: P,
    executor: E,
    state: NativeState,
    mode: RunnerMode,
    profile: GraphProfile,
    events: Vec<LayerEvent>,
    counters: RuntimeCounters,
}

impl<P: RouteFirstProvider, E: NativeS14Executor> LocalS14Runner<P, E> {
    pub fn new(
        manifest: &CapabilityManifest,
        mode: RunnerMode,
        provider: P,
        executor: E,
        state: NativeState,
    ) -> Result<Self, RunnerError> {
        match mode {
            RunnerMode::Production => manifest.gate_production()?,
            RunnerMode::SyntheticTest => {
                manifest.validate_identity()?;
                if manifest.evidence_kind != EvidenceKind::SyntheticTest
                    || manifest.native_forward_ready
                {
                    return Err(RunnerError::Gate(GateError::Identity(
                        "synthetic runner 必须显式 test evidence 且 native_forward_ready=false"
                            .into(),
                    )));
                }
            }
        }
        state.validate_for(manifest.profile)?;
        Ok(Self {
            provider,
            executor,
            state,
            mode,
            profile: manifest.profile,
            events: Vec::new(),
            counters: RuntimeCounters::start_now(),
        })
    }

    pub fn step(&mut self, input_token_id: u32) -> Result<GreedyToken, RunnerError> {
        if self.state.poisoned {
            return Err(RunnerError::Poisoned);
        }
        if input_token_id >= VOCAB_SIZE {
            return Err(RunnerError::Contract(format!(
                "input token {input_token_id} 超出 vocab"
            )));
        }
        if self.state.position >= self.state.max_seq_len {
            return Err(RunnerError::Contract("state 已达 max_seq_len".into()));
        }

        let result = self.step_inner(input_token_id);
        if result.is_err() {
            self.state.poisoned = true;
        }
        result
    }

    fn step_inner(&mut self, input_token_id: u32) -> Result<GreedyToken, RunnerError> {
        for layer in 0..N_LAYERS {
            if self.profile.layers().binary_search(&layer).is_ok() {
                self.execute_selected_layer(layer, input_token_id)?;
            } else {
                // Frozen identity: provider, executor and recursive state are untouched.
                self.events.push(LayerEvent {
                    layer,
                    kind: LayerEventKind::Identity,
                });
            }
        }
        let final_artifacts = self.provider.take_final_artifacts()?;
        let logits = self
            .executor
            .hc_head_norm_full_logits(&final_artifacts, &self.state)
            .map_err(RunnerError::Executor)?;
        let (token_id, max_logit) = greedy_full_vocab_argmax(&logits)?;
        self.state.position += 1;
        self.counters.commit_now();
        Ok(GreedyToken {
            token_id,
            max_logit,
            tie_break: "lowest_token_id".into(),
            origin: self.mode,
        })
    }

    fn execute_selected_layer(
        &mut self,
        layer: u8,
        input_token_id: u32,
    ) -> Result<(), RunnerError> {
        let mut lifecycle = LayerLifecycle::new(layer);
        let mut layer_started = false;
        let execution = (|| {
            let ticket = self.provider.begin_base_load(layer, input_token_id)?;
            layer_started = true;
            if ticket.layer != layer {
                return Err(RunnerError::Contract("base ticket layer 漂移".into()));
            }
            transition_event(&mut lifecycle, LayerPhase::LoadingBase, &mut self.events)?;

            let base = self.provider.wait_base_ready(ticket)?;
            if base.layer != layer {
                return Err(RunnerError::Contract("base Ready lease layer 漂移".into()));
            }
            self.counters.observe_transfer(base.observation);
            transition_event(&mut lifecycle, LayerPhase::BaseReady, &mut self.events)?;
            if layer == self.profile.layers()[0] {
                self.executor
                    .embed_row(input_token_id, &base, &mut self.state)
                    .map_err(RunnerError::Executor)?;
            }
            transition_event(&mut lifecycle, LayerPhase::Routing, &mut self.events)?;

            let route = self
                .executor
                .attention_then_route(layer, input_token_id, &base, &mut self.state)
                .map_err(RunnerError::Executor)?;
            route.validate_for(self.profile)?;
            if route.layer != layer {
                return Err(RunnerError::Contract("route layer 漂移".into()));
            }

            let routed_ticket = self.provider.begin_routed_load(&base, &route)?;
            if routed_ticket.layer != layer || routed_ticket.expert_ids != route.expert_ids {
                return Err(RunnerError::Contract(
                    "provider routed ticket 与官方 route 不一致".into(),
                ));
            }
            transition_event(&mut lifecycle, LayerPhase::LoadingExperts, &mut self.events)?;
            let routed = self.provider.wait_routed_ready(routed_ticket)?;
            if routed.layer != layer || routed.expert_ids != route.expert_ids {
                return Err(RunnerError::Contract(
                    "provider Ready experts 与官方 route 不一致".into(),
                ));
            }
            self.counters.observe_transfer(routed.observation);
            transition_event(&mut lifecycle, LayerPhase::ExpertsReady, &mut self.events)?;
            transition_event(&mut lifecycle, LayerPhase::Executing, &mut self.events)?;
            self.executor
                .routed_and_shared_moe_then_hc_post(layer, &route, &base, &routed, &mut self.state)
                .map_err(RunnerError::Executor)?;
            Ok(())
        })();

        if layer_started {
            if execution.is_ok() {
                transition_event(&mut lifecycle, LayerPhase::Releasing, &mut self.events)?;
            } else {
                lifecycle.fail();
                self.events.push(LayerEvent {
                    layer,
                    kind: LayerEventKind::Failed,
                });
            }
            let cleanup = self.provider.release_layer(layer);
            if execution.is_ok() {
                cleanup?;
                // After a successful release, record the two terminal phases.
                transition_event(&mut lifecycle, LayerPhase::Released, &mut self.events)?;
            } else if let Err(error) = cleanup {
                return Err(RunnerError::Provider(ProviderError(format!(
                    "执行失败且释放失败: {error}"
                ))));
            }
        }
        execution
    }

    pub fn state(&self) -> &NativeState {
        &self.state
    }

    pub fn events(&self) -> &[LayerEvent] {
        &self.events
    }

    pub fn counters(&self) -> crate::CounterReport {
        self.counters.report()
    }

    pub fn into_parts(self) -> (P, E, NativeState) {
        (self.provider, self.executor, self.state)
    }
}

fn transition_event(
    lifecycle: &mut LayerLifecycle,
    phase: LayerPhase,
    events: &mut Vec<LayerEvent>,
) -> Result<(), RunnerError> {
    lifecycle.transition(phase)?;
    let kind = match phase {
        LayerPhase::LoadingBase => LayerEventKind::LoadingBase,
        LayerPhase::BaseReady => LayerEventKind::BaseReady,
        LayerPhase::Routing => LayerEventKind::Routing,
        LayerPhase::LoadingExperts => LayerEventKind::LoadingExperts,
        LayerPhase::ExpertsReady => LayerEventKind::ExpertsReady,
        LayerPhase::Executing => LayerEventKind::Executing,
        LayerPhase::Releasing => LayerEventKind::Releasing,
        LayerPhase::Released => LayerEventKind::Released,
        LayerPhase::Failed => LayerEventKind::Failed,
        LayerPhase::Idle => return Ok(()),
    };
    events.push(LayerEvent {
        layer: lifecycle.layer,
        kind,
    });
    Ok(())
}

fn greedy_full_vocab_argmax(logits: &[f32]) -> Result<(u32, f32), RunnerError> {
    if logits.len() != VOCAB_SIZE as usize {
        return Err(RunnerError::Contract(format!(
            "greedy head 要求完整 {} logits，实际 {}",
            VOCAB_SIZE,
            logits.len()
        )));
    }
    let mut best_id = 0usize;
    let mut best = logits[0];
    if best.is_nan() {
        return Err(RunnerError::Contract("logits[0] 是 NaN".into()));
    }
    for (token_id, &value) in logits.iter().enumerate().skip(1) {
        if value.is_nan() {
            return Err(RunnerError::Contract(format!("logits[{token_id}] 是 NaN")));
        }
        if value > best {
            best = value;
            best_id = token_id;
        }
    }
    Ok((best_id as u32, best))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError(pub String);

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProviderError {}

#[derive(Debug)]
pub enum RunnerError {
    Gate(GateError),
    State(StateLayoutError),
    Provider(ProviderError),
    Executor(String),
    Contract(String),
    Poisoned,
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gate(error) => write!(f, "{error}"),
            Self::State(error) => write!(f, "state: {error}"),
            Self::Provider(error) => write!(f, "provider: {error}"),
            Self::Executor(error) => write!(f, "executor: {error}"),
            Self::Contract(error) => write!(f, "contract: {error}"),
            Self::Poisoned => f.write_str("recursive S14 state 已 poisoned，拒绝继续或返回 token"),
        }
    }
}

impl std::error::Error for RunnerError {}

impl From<GateError> for RunnerError {
    fn from(value: GateError) -> Self {
        Self::Gate(value)
    }
}

impl From<StateLayoutError> for RunnerError {
    fn from(value: StateLayoutError) -> Self {
        Self::State(value)
    }
}

impl From<ProviderError> for RunnerError {
    fn from(value: ProviderError) -> Self {
        Self::Provider(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SELECTED_LAYERS;
    use std::collections::BTreeSet;

    fn artifact(tensor: &str, kind: &str) -> RangeArtifact {
        RangeArtifact {
            tensor: tensor.into(),
            kind: kind.into(),
            expert_id: None,
            path: format!("fixture/{tensor}"),
            bytes: 2,
            cache_hit: true,
            observed_sha256: "0".repeat(64),
            authoritative: false,
        }
    }

    #[derive(Default)]
    struct MockProvider {
        live: Option<u8>,
        route_seen: BTreeSet<u8>,
        log: Vec<String>,
        releases: usize,
    }

    impl RouteFirstProvider for MockProvider {
        fn begin_base_load(
            &mut self,
            layer: u8,
            _input_token_id: u32,
        ) -> Result<BaseLoadTicket, ProviderError> {
            if self.live.replace(layer).is_some() {
                return Err(ProviderError("more than one live layer".into()));
            }
            self.log.push(format!("L{layer}:base_loading"));
            Ok(BaseLoadTicket {
                layer,
                ticket_id: layer as u64,
            })
        }

        fn wait_base_ready(
            &mut self,
            ticket: BaseLoadTicket,
        ) -> Result<ReadyBaseLease, ProviderError> {
            self.log.push(format!("L{}:base_ready", ticket.layer));
            Ok(ReadyBaseLease {
                layer: ticket.layer,
                lease_id: ticket.ticket_id,
                artifacts: if ticket.layer == 0 {
                    vec![artifact("embed.weight[123:124]", "embedding_row")]
                } else {
                    Vec::new()
                },
                observation: TransferObservation::default(),
            })
        }

        fn begin_routed_load(
            &mut self,
            base: &ReadyBaseLease,
            route: &RouteDecision,
        ) -> Result<RoutedLoadTicket, ProviderError> {
            assert_eq!(self.live, Some(base.layer));
            let profile = if SELECTED_LAYERS.binary_search(&route.layer).is_ok() {
                GraphProfile::S14Top6
            } else {
                GraphProfile::FullDepth43NativeTop6
            };
            route.validate_for(profile).unwrap();
            self.route_seen.insert(base.layer);
            self.log.push(format!("L{}:experts_loading", base.layer));
            Ok(RoutedLoadTicket {
                layer: base.layer,
                ticket_id: 1000 + base.layer as u64,
                expert_ids: route.expert_ids.clone(),
            })
        }

        fn wait_routed_ready(
            &mut self,
            ticket: RoutedLoadTicket,
        ) -> Result<ReadyRoutedLease, ProviderError> {
            assert!(self.route_seen.contains(&ticket.layer));
            self.log.push(format!("L{}:experts_ready", ticket.layer));
            Ok(ReadyRoutedLease {
                layer: ticket.layer,
                lease_id: ticket.ticket_id,
                expert_ids: ticket.expert_ids,
                artifacts: Vec::new(),
                observation: TransferObservation {
                    expert_cache_hits: 6,
                    ..TransferObservation::default()
                },
            })
        }

        fn release_layer(&mut self, layer: u8) -> Result<(), ProviderError> {
            if self.live.take() != Some(layer) {
                return Err(ProviderError("release non-live layer".into()));
            }
            self.releases += 1;
            self.log.push(format!("L{layer}:released"));
            Ok(())
        }

        fn take_final_artifacts(&mut self) -> Result<Vec<RangeArtifact>, ProviderError> {
            Ok(vec![artifact("head.weight", "boundary")])
        }
    }

    struct MockExecutor {
        executed: Vec<u8>,
        fail_layer: Option<u8>,
    }

    impl NativeS14Executor for MockExecutor {
        fn embed_row(
            &mut self,
            _token_id: u32,
            _base: &ReadyBaseLease,
            _state: &mut NativeState,
        ) -> Result<(), String> {
            if !_base
                .artifacts
                .iter()
                .any(|item| item.kind == "embedding_row")
            {
                return Err("missing verified embedding row".into());
            }
            Ok(())
        }

        fn attention_then_route(
            &mut self,
            layer: u8,
            _input_token_id: u32,
            _base: &ReadyBaseLease,
            _state: &mut NativeState,
        ) -> Result<RouteDecision, String> {
            if self.fail_layer == Some(layer) {
                return Err("injected attention failure".into());
            }
            Ok(RouteDecision {
                layer,
                kind: router_kind_for_layer(layer).unwrap(),
                expert_ids: vec![0, 1, 2, 3, 4, 5],
                weights: vec![0.25; 6],
            })
        }

        fn routed_and_shared_moe_then_hc_post(
            &mut self,
            layer: u8,
            _route: &RouteDecision,
            _base: &ReadyBaseLease,
            _routed: &ReadyRoutedLease,
            _state: &mut NativeState,
        ) -> Result<(), String> {
            self.executed.push(layer);
            Ok(())
        }

        fn hc_head_norm_full_logits(
            &mut self,
            _final_artifacts: &[RangeArtifact],
            _state: &NativeState,
        ) -> Result<Vec<f32>, String> {
            if !_final_artifacts
                .iter()
                .any(|item| item.tensor == "head.weight")
            {
                return Err("missing verified final head".into());
            }
            let mut logits = vec![-1.0; VOCAB_SIZE as usize];
            logits[7] = 2.0;
            logits[9] = 2.0;
            Ok(logits)
        }
    }

    fn synthetic_runner(fail_layer: Option<u8>) -> LocalS14Runner<MockProvider, MockExecutor> {
        LocalS14Runner::new(
            &CapabilityManifest::synthetic_test_pass(),
            RunnerMode::SyntheticTest,
            MockProvider::default(),
            MockExecutor {
                executed: Vec::new(),
                fail_layer,
            },
            NativeState::decode_layout(4096).unwrap(),
        )
        .unwrap()
    }

    fn synthetic_full_depth_runner() -> LocalS14Runner<MockProvider, MockExecutor> {
        let profile = GraphProfile::FullDepth43NativeTop6;
        LocalS14Runner::new(
            &CapabilityManifest::synthetic_test_pass_for(profile),
            RunnerMode::SyntheticTest,
            MockProvider::default(),
            MockExecutor {
                executed: Vec::new(),
                fail_layer: None,
            },
            NativeState::decode_layout_for(profile, 4096).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn executes_route_first_identity_graph_and_lowest_id_greedy() {
        let mut runner = synthetic_runner(None);
        let token = runner.step(123).unwrap();
        assert_eq!(token.token_id, 7, "tie must choose lowest token id");
        assert_eq!(token.origin, RunnerMode::SyntheticTest);
        assert_eq!(runner.state().position, 1);
        assert_eq!(
            runner
                .events()
                .iter()
                .filter(|event| event.kind == LayerEventKind::Identity)
                .count(),
            29
        );
        let (provider, executor, _) = runner.into_parts();
        assert_eq!(executor.executed, SELECTED_LAYERS);
        assert_eq!(provider.releases, 14);
        assert_eq!(provider.live, None);
        for layer in SELECTED_LAYERS {
            let route_pos = provider
                .log
                .iter()
                .position(|event| event == &format!("L{layer}:experts_loading"))
                .unwrap();
            let base_pos = provider
                .log
                .iter()
                .position(|event| event == &format!("L{layer}:base_ready"))
                .unwrap();
            assert!(route_pos > base_pos);
        }
    }

    #[test]
    fn failure_releases_layer_poisons_state_and_commits_no_token() {
        let mut runner = synthetic_runner(Some(7));
        assert!(runner.step(123).is_err());
        assert!(runner.state().poisoned);
        assert_eq!(runner.counters().committed_tokens, 0);
        assert!(matches!(runner.step(123), Err(RunnerError::Poisoned)));
        let (provider, _, _) = runner.into_parts();
        assert_eq!(provider.live, None);
        assert_eq!(provider.releases, 5); // L0,L1,L2,L6 success; L7 failure cleanup.
    }

    #[test]
    fn lifecycle_rejects_skipping_ready_barrier() {
        let mut lifecycle = LayerLifecycle::new(0);
        lifecycle.transition(LayerPhase::LoadingBase).unwrap();
        assert!(lifecycle.transition(LayerPhase::Routing).is_err());
    }

    #[test]
    fn fulldepth_native_top6_executes_every_official_block_without_identity() {
        let mut runner = synthetic_full_depth_runner();
        let token = runner.step(1).unwrap();
        assert_eq!(token.origin, RunnerMode::SyntheticTest);
        assert_eq!(
            runner
                .events()
                .iter()
                .filter(|event| event.kind == LayerEventKind::Identity)
                .count(),
            0
        );
        let (provider, executor, _) = runner.into_parts();
        assert_eq!(provider.releases, 43);
        assert_eq!(executor.executed.len(), 43);
        assert_eq!(provider.route_seen.len(), 43);
    }
}
