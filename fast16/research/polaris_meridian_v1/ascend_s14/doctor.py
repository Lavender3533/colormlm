"""只读检查 GitCode Ascend 910B4 环境和 S14 原生语义缺口。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from adapter import REPO, REVISION, REQUIRED_NATIVE_CAPABILITIES, configure_utf8, read_json, validate_matrix


def discover_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def probe_npu() -> dict[str, Any]:
    """显式调用时才导入 torch_npu；不分配测试 tensor。"""

    try:
        import torch
        import torch_npu

        available = bool(torch_npu.npu.is_available())
        name = str(torch_npu.npu.get_device_name(0)) if available else None
        free_bytes = None
        total_bytes = None
        mem_get_info = getattr(torch_npu.npu, "mem_get_info", None)
        if available and callable(mem_get_info):
            free_bytes, total_bytes = (int(value) for value in mem_get_info(0))
        return {
            "probed": True,
            "available": available,
            "device_name": name,
            "hbm_free_bytes": free_bytes,
            "hbm_total_bytes": total_bytes,
            "torch_version": str(torch.__version__),
            "torch_npu_version": str(getattr(torch_npu, "__version__", "unknown")),
            "dtype_symbols": {
                "fp8_e4m3": hasattr(torch, "float8_e4m3fn"),
                "fp4_e2m1_x2": hasattr(torch, "float4_e2m1fn_x2"),
                "ue8m0": hasattr(torch, "float8_e8m0fnu"),
            },
            "operator_support_proven": False,
        }
    except Exception as exc:
        return {
            "probed": True,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "operator_support_proven": False,
        }


def make_report(matrix: dict[str, Any], do_probe: bool, required_device: str | None) -> dict[str, Any]:
    validate_matrix(matrix)
    machine = platform.machine().lower()
    modules = {name: discover_module(name) for name in ("torch", "torch_npu", "safetensors")}
    device = probe_npu() if do_probe else {
        "probed": False,
        "available": None,
        "reason": "使用 --probe-npu 才会导入 torch_npu；默认 doctor 不触碰设备",
        "operator_support_proven": False,
    }
    name = str(device.get("device_name") or "")
    device_match = None if not do_probe else bool(name and (required_device is None or required_device.lower() in name.lower()))
    architecture_ok = machine in {"aarch64", "arm64"}
    capability_rows = []
    statuses = matrix["required_for_native_forward"]
    for capability in REQUIRED_NATIVE_CAPABILITIES:
        capability_rows.append(
            {
                "capability": capability,
                "status": statuses.get(capability, "missing"),
                "required": True,
            }
        )
    native_ready = bool(
        do_probe
        and device.get("available")
        and device_match
        and architecture_ok
        and matrix.get("native_forward_ready") is True
        and all(row["status"] == "passed" for row in capability_rows)
    )
    return {
        "format": "polaris-ascend-s14-doctor-v1",
        "source": {"repo": REPO, "revision": REVISION},
        "platform": {
            "python": sys.version.split()[0],
            "system": platform.platform(),
            "machine": platform.machine(),
            "aarch64_expected": True,
            "architecture_ok": architecture_ok,
        },
        "modules": modules,
        "device": device,
        "required_device": required_device,
        "device_match": device_match,
        "hbm_gate": {
            "minimum_total_bytes": 32 * 1024**3,
            "observed_total_bytes": device.get("hbm_total_bytes"),
            "pass": None
            if device.get("hbm_total_bytes") is None
            else int(device["hbm_total_bytes"]) >= 32 * 1024**3,
        },
        "semantic_gaps": capability_rows,
        "native_forward_ready": native_ready,
        "ok": native_ready,
        "claim_limit": "dtype 名称、torch_npu 可导入和 NPU 可见都不等于 FP4/FP8/mHC/CSA-HCA 原生语义兼容。",
    }


def main() -> int:
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).with_name("support_matrix.json"),
    )
    parser.add_argument("--probe-npu", action="store_true")
    parser.add_argument("--require-device", default="Ascend910B4")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = make_report(read_json(args.matrix), args.probe_npu, args.require_device)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    # 未实现 native 语义是预期状态；doctor 报告成功生成即返回 0，ready 单独看字段。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
