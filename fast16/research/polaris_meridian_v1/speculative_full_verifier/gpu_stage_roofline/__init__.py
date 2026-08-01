"""Polaris S14 GPU 阶段级 roofline 与流水调度原型。"""

from .anchors import AnchorContractError, StageAnchors, load_real_anchors
from .full_depth import (
    FullDepthShapeAudit,
    audit_full_depth_projection_shapes,
    build_full_depth_report,
)
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
    "FullDepthShapeAudit",
    "SimulationConfig",
    "StageAnchors",
    "build_roofline_report",
    "build_full_depth_report",
    "audit_full_depth_projection_shapes",
    "load_real_anchors",
    "simulate_token",
    "solve_required_expert_hit_rate",
]
