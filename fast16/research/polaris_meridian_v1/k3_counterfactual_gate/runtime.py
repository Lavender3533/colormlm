"""FullDepth 连续隐藏态到 K3 胶囊的最小 fail-closed 运行时。

这个模块不将 router prior 当作能力标签。只有预先冻结的
counterfactual NLL + leave-one-task-out 证据通过后，线性门才可资格
触发 portal 和胶囊的懒加载。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch
import torch.nn.functional as F


GATE_FORMAT = "polaris-k3-counterfactual-gate-v1"
PORTAL_FORMAT = "polaris-fulldepth-to-colorlm-portal-v1"
SUPPORTED_CAPSULE_FORMATS = {
    "colorlm-kimi-k3-latent-macro-capsule-v1",
    "colorlm-kimi-k3-latent-macro-capsule-v2",
    "colorlm-kimi-k3-latent-macro-capsule-v3",
    "colorlm-kimi-k3-latent-macro-capsule-v4",
}


class ContractError(ValueError):
    """运行时或证据合同不成立。"""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} 必须是 JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_single_token(hidden: torch.Tensor, width: int, label: str) -> torch.Tensor:
    if not isinstance(hidden, torch.Tensor) or hidden.shape[-1:] != (width,):
        raise ContractError(f"{label} 必须以 width={width} 结尾")
    if hidden.numel() != width:
        raise ContractError(f"{label} 只接受当前单 token 连续隐藏态")
    if not bool(torch.isfinite(hidden).all().item()):
        raise ContractError(f"{label} 含 NaN/Inf")
    return hidden.reshape(1, width)


@dataclass(frozen=True)
class GateDecision:
    selected: bool
    score: float | None
    reason: str


@dataclass(frozen=True)
class LinearCapabilityGate:
    """由反事实 NLL 拟合、在 FullDepth 原生连续态上打分的门。"""

    weights: torch.Tensor
    bias: float
    threshold: float
    approved: bool
    counterfactual_nll_passed: bool
    leave_one_task_out_passed: bool
    frozen_contract: bool
    source: str = ""

    def __post_init__(self) -> None:
        if self.weights.ndim != 1 or self.weights.numel() == 0:
            raise ContractError("gate weights 必须是非空一维向量")
        if not bool(torch.isfinite(self.weights).all().item()):
            raise ContractError("gate weights 含 NaN/Inf")
        if not all(math.isfinite(value) for value in (self.bias, self.threshold)):
            raise ContractError("gate bias/threshold 必须有限")

    @property
    def input_width(self) -> int:
        return int(self.weights.numel())

    @property
    def eligible(self) -> bool:
        return bool(
            self.approved
            and self.counterfactual_nll_passed
            and self.leave_one_task_out_passed
            and self.frozen_contract
        )

    @classmethod
    def rejected(cls, input_width: int, *, source: str = "no evidence") -> "LinearCapabilityGate":
        if input_width <= 0:
            raise ContractError("input_width 必须为正数")
        return cls(
            weights=torch.zeros(input_width, dtype=torch.float32),
            bias=0.0,
            threshold=0.0,
            approved=False,
            counterfactual_nll_passed=False,
            leave_one_task_out_passed=False,
            frozen_contract=False,
            source=source,
        )

    @classmethod
    def from_manifest(cls, path: Path) -> "LinearCapabilityGate":
        manifest = _read_json(path)
        if manifest.get("format") != GATE_FORMAT:
            raise ContractError("gate manifest format 不支持")
        evidence = manifest.get("evidence")
        weight = manifest.get("weights")
        if not isinstance(evidence, dict) or not isinstance(weight, dict):
            raise ContractError("gate manifest 缺少 evidence/weights")
        weight_path = path.parent / str(weight.get("file", ""))
        if not weight_path.is_file() or _sha256(weight_path) != weight.get("sha256"):
            raise ContractError("gate weight 缺失或 SHA-256 不匹配")
        array = np.load(weight_path, allow_pickle=False)
        expected = (int(manifest.get("input_width", 0)),)
        if array.dtype != np.dtype("float32") or array.shape != expected:
            raise ContractError(f"gate weight 形状/类型异常: {array.shape}/{array.dtype}")
        return cls(
            weights=torch.from_numpy(np.array(array, copy=True)),
            bias=float(manifest.get("bias", math.nan)),
            threshold=float(manifest.get("threshold", math.nan)),
            approved=manifest.get("approved") is True,
            counterfactual_nll_passed=evidence.get("counterfactual_nll_passed") is True,
            leave_one_task_out_passed=evidence.get("leave_one_task_out_passed") is True,
            frozen_contract=evidence.get("frozen_contract") is True,
            source=str(path),
        )

    def decide(self, full_hidden: torch.Tensor) -> GateDecision:
        if not self.eligible:
            return GateDecision(False, None, "counterfactual_gate_not_approved")
        row = _require_single_token(full_hidden, self.input_width, "FullDepth hidden")
        score = float((row.float() @ self.weights.float().reshape(-1, 1)).item() + self.bias)
        return GateDecision(score > self.threshold, score, "selected" if score > self.threshold else "below_threshold")


class Portal(Protocol):
    full_width: int
    bus_width: int

    def project_in(self, hidden: torch.Tensor) -> torch.Tensor: ...

    def project_out(self, delta: torch.Tensor) -> torch.Tensor: ...


class Capsule(Protocol):
    input_width: int

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class DensePortal:
    """显式 4096↔2048→4096 portal；必须由独立证据 manifest 授权。"""

    full_to_bus: torch.Tensor
    bus_to_full: torch.Tensor

    def __post_init__(self) -> None:
        if self.full_to_bus.ndim != 2 or self.bus_to_full.ndim != 2:
            raise ContractError("portal matrix 必须是二维")
        bus_width, full_width = self.full_to_bus.shape
        if self.bus_to_full.shape != (full_width, bus_width):
            raise ContractError("portal 输入/输出矩阵形状不对称")
        if not bool(torch.isfinite(self.full_to_bus).all().item()) or not bool(
            torch.isfinite(self.bus_to_full).all().item()
        ):
            raise ContractError("portal matrix 含 NaN/Inf")

    @property
    def full_width(self) -> int:
        return int(self.full_to_bus.shape[1])

    @property
    def bus_width(self) -> int:
        return int(self.full_to_bus.shape[0])

    @classmethod
    def from_manifest(cls, path: Path) -> "DensePortal":
        manifest = _read_json(path)
        evidence = manifest.get("evidence")
        if manifest.get("format") != PORTAL_FORMAT or manifest.get("approved") is not True:
            raise ContractError("FullDepth portal 未批准")
        if not isinstance(evidence, dict) or not all(
            evidence.get(key) is True
            for key in ("held_out_hidden_passed", "counterfactual_nll_passed")
        ):
            raise ContractError("FullDepth portal 缺少 held-out/NLL 证据")

        def load_matrix(key: str) -> torch.Tensor:
            spec = manifest.get("matrices", {}).get(key)
            if not isinstance(spec, dict):
                raise ContractError(f"portal 缺少 {key}")
            matrix_path = path.parent / str(spec.get("file", ""))
            if not matrix_path.is_file() or _sha256(matrix_path) != spec.get("sha256"):
                raise ContractError(f"portal {key} 缺失或 SHA-256 不匹配")
            array = np.load(matrix_path, allow_pickle=False)
            shape = tuple(int(value) for value in spec.get("shape", []))
            if array.dtype != np.dtype("float32") or array.shape != shape:
                raise ContractError(f"portal {key} 形状/类型异常")
            return torch.from_numpy(np.array(array, copy=True))

        return cls(load_matrix("full_to_bus"), load_matrix("bus_to_full"))

    def project_in(self, hidden: torch.Tensor) -> torch.Tensor:
        row = _require_single_token(hidden, self.full_width, "FullDepth hidden")
        return F.linear(row.float(), self.full_to_bus.float()).reshape(*hidden.shape[:-1], self.bus_width)

    def project_out(self, delta: torch.Tensor) -> torch.Tensor:
        row = _require_single_token(delta, self.bus_width, "K3 delta")
        return F.linear(row.float(), self.bus_to_full.float()).reshape(*delta.shape[:-1], self.full_width)


class F16K3Capsule:
    """真实 K3 latent macro 胶囊的纯 CPU F16-资产/F32-计算参考路径。"""

    REQUIRED = ("b_in", "gate", "up", "down", "norm", "b_out")

    def __init__(self, weights: Mapping[str, torch.Tensor], *, eps: float, source: str):
        missing = set(self.REQUIRED) - set(weights)
        if missing:
            raise ContractError(f"K3 capsule 缺少矩阵: {sorted(missing)}")
        self.weights = {name: value.float() for name, value in weights.items()}
        self.eps = float(eps)
        self.source = source
        self.input_width = int(self.weights["b_in"].shape[1])
        latent = int(self.weights["b_in"].shape[0])
        intermediate = int(self.weights["gate"].shape[0])
        expected = {
            "gate": (intermediate, latent),
            "up": (intermediate, latent),
            "down": (latent, intermediate),
            "norm": (latent,),
            "b_out": (self.input_width, latent),
        }
        for name, shape in expected.items():
            if tuple(self.weights[name].shape) != shape:
                raise ContractError(f"K3 capsule {name} 形状异常")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ContractError("K3 capsule RMS eps 必须为正有限数")

    @classmethod
    def from_runtime_dir(cls, runtime_dir: Path) -> "F16K3Capsule":
        manifest_path = runtime_dir / "capsule.json"
        manifest = _read_json(manifest_path)
        if manifest.get("format") not in SUPPORTED_CAPSULE_FORMATS:
            raise ContractError("K3 runtime capsule format 不支持")
        specs = manifest.get("runtime_files")
        if not isinstance(specs, dict):
            raise ContractError("K3 runtime capsule 缺少 runtime_files")
        weights: dict[str, torch.Tensor] = {}
        for name in cls.REQUIRED:
            spec = specs.get(name)
            if not isinstance(spec, dict) or spec.get("dtype") != "float16-le":
                raise ContractError(f"K3 runtime {name} 必须是 float16-le")
            shape = tuple(int(value) for value in spec.get("shape", []))
            path = runtime_dir / str(spec.get("file", ""))
            expected_bytes = math.prod(shape) * 2
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ContractError(f"K3 runtime {name} 缺失或字节数异常")
            if _sha256(path) != spec.get("sha256"):
                raise ContractError(f"K3 runtime {name} SHA-256 不匹配")
            mapped = np.memmap(path, dtype="<f2", mode="r", shape=shape)
            weights[name] = torch.from_numpy(np.array(mapped, dtype=np.float32, copy=True))
        return cls(
            weights,
            eps=float(manifest.get("rms_norm_eps", 1.0e-5)),
            source=str(manifest_path),
        )

    @staticmethod
    def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate_branch = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
        up_branch = 25.0 * torch.tanh(up / 25.0)
        return gate_branch * up_branch

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        row = _require_single_token(hidden, self.input_width, "ColorLM portal hidden").float()
        latent = F.linear(row, self.weights["b_in"])
        activation = self._situ(
            F.linear(latent, self.weights["gate"]),
            F.linear(latent, self.weights["up"]),
        )
        latent_output = F.linear(activation, self.weights["down"])
        normalized = latent_output * torch.rsqrt(
            latent_output.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = normalized * self.weights["norm"]
        return F.linear(normalized, self.weights["b_out"]).reshape(*hidden.shape)


@dataclass(frozen=True)
class BusResult:
    hidden: torch.Tensor
    decision: GateDecision
    exact_bypass: bool
    portal_loaded: bool
    capsule_loaded: bool


class FullDepthK3Bus:
    """在当前 token 的 FullDepth 连续 hidden 上执行可归因残差。"""

    def __init__(
        self,
        gate: LinearCapabilityGate,
        *,
        portal_authorized: bool = False,
        portal_loader: Callable[[], Portal] | None = None,
        capsule_loader: Callable[[], Capsule] | None = None,
    ) -> None:
        self.gate = gate
        self.portal_authorized = bool(portal_authorized)
        self.portal_loader = portal_loader
        self.capsule_loader = capsule_loader

    @staticmethod
    def _bypass(hidden: torch.Tensor, reason: str, score: float | None = None) -> BusResult:
        return BusResult(
            hidden=hidden,
            decision=GateDecision(False, score, reason),
            exact_bypass=True,
            portal_loaded=False,
            capsule_loaded=False,
        )

    def apply(self, full_hidden: torch.Tensor, *, alpha: float = 0.0) -> BusResult:
        if not math.isfinite(alpha) or alpha < 0:
            raise ContractError("alpha 必须是非负有限数")
        # 硬旁路必须在 hidden 形状检查、portal 和胶囊懒加载之前。
        if alpha == 0.0:
            return self._bypass(full_hidden, "alpha_zero_physical_bypass")
        if not self.gate.eligible:
            return self._bypass(full_hidden, "counterfactual_gate_not_approved")

        decision = self.gate.decide(full_hidden)
        if not decision.selected:
            return self._bypass(full_hidden, decision.reason, decision.score)
        if not self.portal_authorized or self.portal_loader is None:
            return self._bypass(full_hidden, "full_to_colorlm_portal_not_approved", decision.score)
        if self.capsule_loader is None:
            return self._bypass(full_hidden, "capsule_loader_missing", decision.score)

        portal = self.portal_loader()
        if portal.full_width != self.gate.input_width:
            raise ContractError("portal FullDepth width 与 gate 不一致")
        bus_hidden = portal.project_in(full_hidden)
        capsule = self.capsule_loader()
        if capsule.input_width != portal.bus_width:
            raise ContractError("K3 capsule width 与 portal 不一致")
        delta = portal.project_out(capsule(bus_hidden))
        output = full_hidden + delta.to(device=full_hidden.device, dtype=full_hidden.dtype) * alpha
        if not bool(torch.isfinite(output).all().item()):
            raise ContractError("K3 residual 产生 NaN/Inf")
        return BusResult(
            hidden=output,
            decision=decision,
            exact_bypass=False,
            portal_loaded=True,
            capsule_loaded=True,
        )
