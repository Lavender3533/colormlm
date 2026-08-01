"""Polaris S14 草稿 + FullDepth43 原生验证的离线合同原型。

该包只读 tokenizer 和权重元数据；不加载或运行模型权重。
"""

from .assets import AssetAudit, AssetContractError, audit_assets
from .cache_replay import ExpertPageCache, ReplayReport, RouteBlock, replay_blocks
from .cost_model import HardwareBudget, build_analysis_report
from .runtime_controller import (
    AtomicSpeculativeController,
    NativeBatchedVerifierBackend,
    SerialSnapshotDraftBackend,
    SerialSnapshotVerifierBackend,
    SpeculativeRoundResult,
)
from .speed_gate import RoundTiming, evaluate_speed_gate
from .verifier import (
    DeepSeekTokenizer,
    DraftStep,
    FullDepthVerification,
    SessionState,
    SpeculativeSession,
    VerificationResult,
)

__all__ = [
    "AssetAudit",
    "AssetContractError",
    "AtomicSpeculativeController",
    "DeepSeekTokenizer",
    "DraftStep",
    "ExpertPageCache",
    "FullDepthVerification",
    "HardwareBudget",
    "NativeBatchedVerifierBackend",
    "ReplayReport",
    "RouteBlock",
    "SessionState",
    "SerialSnapshotDraftBackend",
    "SerialSnapshotVerifierBackend",
    "SpeculativeRoundResult",
    "SpeculativeSession",
    "VerificationResult",
    "audit_assets",
    "build_analysis_report",
    "evaluate_speed_gate",
    "replay_blocks",
    "RoundTiming",
]
