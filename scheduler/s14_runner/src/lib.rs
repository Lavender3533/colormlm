//! DeepSeek-V4-Flash-0731 原生稀深 S14 的后端无关执行契约。
//!
//! 这个 crate 不实现任何 DeepSeek 算子，也不会生成占位 token。生产 runner
//! 必须先通过完整 capability gate；当前 Vulkan 审计矩阵会被明确拒绝。

mod abi;
mod capability;
mod contract;
mod memory;
mod metrics;
mod range_bridge;
mod runner;
mod state;

pub use abi::{
    validate_expert_abi_manifest_json, AbiEntry, AbiError, ExpertAbiManifest, ValidatedExpertAbi,
    ABI_SAMPLE_BYTES,
};
pub use capability::{
    CapabilityEntry, CapabilityManifest, CapabilityStatus, EvidenceKind, GateError,
    REQUIRED_CAPABILITIES,
};
pub use contract::{
    is_selected_layer, router_kind_for_layer, ContractError, GraphProfile, RouterKind, S14Contract,
    COMPRESS_RATIOS, EXPERTS_PER_TOKEN, FULL_DEPTH_LAYERS, HC_STREAMS, HIDDEN_SIZE, MODEL_REPO,
    MODEL_REVISION, N_LAYERS, N_ROUTED_EXPERTS, SELECTED_LAYERS, VOCAB_SIZE,
};
pub use memory::{BudgetKind, MemoryLedger, MemoryLine, EXPERT_PAGE_BYTES};
pub use metrics::{CounterReport, RuntimeCounters, TransferObservation};
pub use range_bridge::{RangeBridgeConfig, SubprocessRangeProvider, RANGE_JSONL_PROTOCOL};
pub use runner::{
    BaseLoadTicket, GreedyToken, LayerEvent, LayerEventKind, LayerLifecycle, LayerPhase,
    LocalS14Runner, NativeS14Executor, ProviderError, RangeArtifact, ReadyBaseLease,
    ReadyRoutedLease, RouteDecision, RouteFirstProvider, RoutedLoadTicket, RunnerError, RunnerMode,
};
pub use state::{
    BufferSlice, CompressorState, DType, HcState, IndexerState, KvState, NativeState,
    StateLayoutError,
};

/// 供 Rust 和 Python 共同读取的冻结互操作契约。
pub const INTEROP_CONTRACT_JSON: &str = include_str!("../contracts/s14_contract.json");

/// 基线 Vulkan 能力审计。它故意保持 `native_forward_ready=false`。
pub const CURRENT_VULKAN_CAPABILITIES_JSON: &str =
    include_str!("../contracts/current_vulkan_capabilities.json");

/// Pre-registered FullDepth/top-1 fallback audit; also hard-refused.
pub const CURRENT_FULL_DEPTH_CAPABILITIES_JSON: &str =
    include_str!("../contracts/current_fulldepth_top1_capabilities.json");
