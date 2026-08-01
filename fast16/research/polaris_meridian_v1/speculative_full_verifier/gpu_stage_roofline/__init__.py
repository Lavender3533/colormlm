"""Polaris S14 GPU 阶段级 roofline 与流水调度原型。"""

from .anchors import AnchorContractError, StageAnchors, load_real_anchors
from .model import (
    CommandBufferMode,
    SimulationConfig,
    build_roofline_report,
    simulate_token,
    solve_required_expert_hit_rate,
)

__all__ = [
    "AnchorContractError",
    "CommandBufferMode",
    "SimulationConfig",
    "StageAnchors",
    "build_roofline_report",
    "load_real_anchors",
    "simulate_token",
    "solve_required_expert_hit_rate",
]
