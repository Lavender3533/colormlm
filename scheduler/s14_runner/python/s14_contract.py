"""Rust/Python 共享的 Polaris local S14/FullDepth 互操作合同。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
CONTRACTS = HERE.parent / "contracts"
REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
PROFILES = {
    "s14_top6": {
        "layers": [0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42],
        "topk": 6,
        "capability": "s14_identity_skip_state_parity",
    },
    "full_depth_top1": {
        "layers": list(range(43)),
        "topk": 1,
        "capability": "fulldepth_top1_route_reduction_parity",
    },
}


class ContractError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} 必须是 JSON object")
    return value


def validate_shared_contract(root: Mapping[str, Any]) -> None:
    contract = root.get("contract", {})
    if contract.get("format") != "polaris-local-s14-interop-v1":
        raise ContractError("互操作 format 错误")
    if contract.get("repo") != REPO or contract.get("revision") != REVISION:
        raise ContractError("互操作 donor 身份错误")
    if contract.get("selected_layers") != PROFILES["s14_top6"]["layers"]:
        raise ContractError("默认 S14 层集合漂移")
    if contract.get("hc_streams") != 4 or contract.get("hidden_size") != 4096:
        raise ContractError("原生四路 HC/hidden ABI 漂移")
    if root.get("expert_abi", {}).get("router_sidecar_is_routing_authority") is not False:
        raise ContractError("ABI sidecar 不得成为路由权威")


def validate_capability_manifest(manifest: Mapping[str, Any]) -> list[str]:
    if manifest.get("format") != "polaris-local-s14-capabilities-v1":
        raise ContractError("capability format 错误")
    if manifest.get("repo") != REPO or manifest.get("revision") != REVISION:
        raise ContractError("capability donor 身份错误")
    profile_id = manifest.get("profile")
    if profile_id not in PROFILES:
        raise ContractError("只允许两个预注册 profile")
    profile = PROFILES[profile_id]
    if manifest.get("selected_layers") != profile["layers"]:
        raise ContractError("profile 层集合漂移")
    if manifest.get("experts_per_token") != profile["topk"]:
        raise ContractError("profile top-k 漂移")
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ContractError("capabilities 必须是 object")
    missing = [
        name
        for name, value in capabilities.items()
        if not isinstance(value, dict)
        or value.get("status") != "passed"
        or not str(value.get("evidence", "")).strip()
    ]
    profile_capability = str(profile["capability"])
    if capabilities.get(profile_capability, {}).get("status") != "passed":
        if profile_capability not in missing:
            missing.append(profile_capability)
    if manifest.get("evidence_kind") != "measured_runtime":
        missing.append("evidence_kind:measured_runtime")
    if manifest.get("native_forward_ready") is not True:
        missing.append("native_forward_ready")
    return missing


def validate_abi_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("format") != "polaris-deepseek-abi-sample-v1":
        raise ContractError("ABI format 错误")
    if manifest.get("repo") != REPO or manifest.get("revision") != REVISION:
        raise ContractError("ABI donor 身份错误")
    if manifest.get("purpose") != "format_and_kernel_abi_only_not_capability":
        raise ContractError("ABI purpose 越权")
    entries = manifest.get("entries", [])
    expert_bytes = sum(int(row["bytes"]) for row in entries if row.get("kind") == "expert_tensor")
    sidecar_bytes = sum(int(row["bytes"]) for row in entries if row.get("kind") in {"router_row", "router_bias"})
    if expert_bytes != 13_369_344 or sidecar_bytes != 8_196:
        raise ContractError("ABI expert/sidecar 字节不闭合")
    if expert_bytes + sidecar_bytes != int(manifest.get("payload_bytes", -1)):
        raise ContractError("ABI payload_bytes 不闭合")
    return {
        "expert_payload_bytes": expert_bytes,
        "router_sidecar_bytes": sidecar_bytes,
        "routing_authority": False,
    }
