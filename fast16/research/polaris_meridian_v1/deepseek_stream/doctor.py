"""检查 Ascend 910B4 容器能否进入 DeepSeek 流式采集第一阶段。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

try:
    from .planner import DEFAULT_SNAPSHOT, make_plan, read_json, verify_metadata
except ImportError:
    from planner import DEFAULT_SNAPSHOT, make_plan, read_json, verify_metadata


REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"


def _configure_utf8() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _memory() -> dict[str, int | None]:
    values: dict[str, int | None] = {"total_bytes": None, "available_bytes": None}
    try:
        fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            fields[key] = int(raw.strip().split()[0]) * 1024
        values["total_bytes"] = fields.get("MemTotal")
        values["available_bytes"] = fields.get("MemAvailable")
    except (OSError, ValueError, UnicodeError):
        pass
    return values


def _disk(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _network_probe(base_url: str, expected: dict[str, Any], timeout: float) -> dict[str, Any]:
    files = ["config.json", "model.safetensors.index.json"]
    checks = []
    for name in files:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/{name}", method="HEAD")
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            size = response.headers.get("Content-Length")
            checks.append(
                {
                    "file": name,
                    "status": int(getattr(response, "status", 200)),
                    "content_length": None if size is None else int(size),
                    "expected_bytes": expected[name]["bytes"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    return {"ok": all(row["status"] < 400 for row in checks), "checks": checks}


def _device_probe(preference: str, matrix_size: int) -> dict[str, Any]:
    import torch

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch_npu_version = None
    selected = "cpu"
    npu_error = None
    if preference in {"auto", "npu"}:
        try:
            import torch_npu

            torch_npu_version = str(getattr(torch_npu, "__version__", "unknown"))
            if not bool(torch_npu.npu.is_available()):
                raise RuntimeError("torch_npu.npu.is_available() 为 false")
            device = torch.device("npu:0")
            selected = str(device)
        except Exception as exc:
            npu_error = f"{type(exc).__name__}: {exc}"
            if preference == "npu":
                raise
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    torch.manual_seed(47)
    left = torch.randn((matrix_size, matrix_size), dtype=torch.float16, device=device)
    right = torch.randn((matrix_size, matrix_size), dtype=torch.float16, device=device)
    if selected.startswith("npu"):
        import torch_npu

        torch_npu.npu.synchronize()
    started = time.perf_counter()
    result = left @ right
    if selected.startswith("npu"):
        import torch_npu

        torch_npu.npu.synchronize()
    finite = bool(torch.isfinite(result).all().cpu())
    name = None
    if selected.startswith("npu"):
        import torch_npu

        name = str(torch_npu.npu.get_device_name(0))
    return {
        "ok": finite,
        "requested": preference,
        "selected": selected,
        "fallback_used": preference == "auto" and selected == "cpu",
        "fallback_reason": npu_error,
        "device_name": name,
        "torch_version": str(torch.__version__),
        "torch_npu_version": torch_npu_version,
        "operator": "float16_matmul",
        "matrix_size": matrix_size,
        "elapsed_seconds": time.perf_counter() - started,
        "finite": finite,
        "dtype_symbols": {
            "float8_e4m3fn": hasattr(torch, "float8_e4m3fn"),
            "float4_e2m1fn_x2": hasattr(torch, "float4_e2m1fn_x2"),
            "float8_e8m0fnu": hasattr(torch, "float8_e8m0fnu"),
        },
    }


def main() -> int:
    _configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--device", choices=("auto", "npu", "cpu"), default="auto")
    parser.add_argument("--matrix-size", type=int, default=128)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--network-mib-s", type=float, default=50.0)
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument(
        "--metadata-base-url",
        default=f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/{REVISION}",
    )
    parser.add_argument("--network-timeout", type=float, default=15.0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        snapshot = read_json(args.snapshot)
        if snapshot.get("revision") != REVISION:
            raise ValueError("snapshot revision 错误")
        metadata = verify_metadata(snapshot, args.config, args.index)
        workspace = args.workspace.resolve()
        disk = _disk(workspace)
        memory = _memory()
        available_ram_gib = (memory["available_bytes"] or 32 * 1024**3) / 1024**3
        available_disk_gib = disk["free_bytes"] / 1024**3
        device = _device_probe(args.device, args.matrix_size)
        network: dict[str, Any]
        if args.skip_network:
            network = {"ok": None, "skipped": True}
        else:
            try:
                network = _network_probe(
                    args.metadata_base_url,
                    snapshot["metadata_artifacts"],
                    args.network_timeout,
                )
            except Exception as exc:
                network = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        plan = make_plan(
            snapshot,
            task_count=8,
            observed_tokens_per_task=64,
            network_mib_s=args.network_mib_s,
            available_disk_gib=max(0.001, available_disk_gib),
            available_ram_gib=max(0.001, available_ram_gib),
            hbm_gib=32.0 if device["selected"].startswith("npu") else 0.001,
            window_seconds=7200,
            metadata_verification=metadata,
        )
        phase1_ready = bool(device["ok"] and metadata["ok"])
        native_blockers = [
            "尚未提供官方 DeepSeek-V4 Ascend runtime adapter",
            "官方 inference/kernel.py 为 CUDA/TileLang 路径，不能因 torch_npu matmul 成功就视为 FP4/FP8 kernel 已支持",
            "45 个 base-forward 权重文件或等价远端 tensor source 尚未校验",
            "单卡 32 GiB HBM 不能常驻 156.023 GB base-forward 文件",
            "流式逐层 NPU 执行器尚未实现",
        ]
        result = {
            "format": "polaris-deepseek-stream-doctor-v1",
            "ok": phase1_ready,
            "phase1_contract_ready": phase1_ready,
            "native_forward_ready": False,
            "platform": {
                "python": sys.version.split()[0],
                "machine": platform.machine(),
                "system": platform.platform(),
                "workspace": str(workspace),
            },
            "snapshot": {"repo": snapshot["repo"], "revision": snapshot["revision"]},
            "metadata": metadata,
            "device": device,
            "storage": disk,
            "memory": memory,
            "network": network,
            "budget": plan["two_hour_budget"],
            "memory_budget": plan["memory"],
            "python_modules": {
                "torch": importlib.util.find_spec("torch") is not None,
                "torch_npu": importlib.util.find_spec("torch_npu") is not None,
                "safetensors": importlib.util.find_spec("safetensors") is not None,
            },
            "native_forward_blockers": native_blockers,
            "claim_limit": "doctor 只证明容器、NPU 基础算子和采集契约；native_forward_ready 固定为 false。",
        }
        encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            if args.output.exists() and not args.force:
                raise FileExistsError(f"拒绝覆盖 {args.output}；使用 --force")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8", newline="\n")
        print(encoded, end="")
        return 0 if phase1_ready else 1
    except Exception as exc:
        result = {
            "format": "polaris-deepseek-stream-doctor-v1",
            "ok": False,
            "phase1_contract_ready": False,
            "native_forward_ready": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
