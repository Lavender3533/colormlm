"""检查 ColorLM Ascend 910B4 环境并真实执行矩阵乘 NPU 算子。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .training_device import select_training_device, synchronize
except ImportError:
    from training_device import select_training_device, synchronize


def _read_text(path: str) -> str | None:
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
        return value or None
    except (OSError, UnicodeError):
        return None


def _limit_info() -> dict[str, str | None]:
    return {
        "cgroup_v2_memory_max": _read_text("/sys/fs/cgroup/memory.max"),
        "cgroup_v2_cpu_max": _read_text("/sys/fs/cgroup/cpu.max"),
        "cgroup_v1_memory_limit": _read_text("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        "cgroup_v1_cpu_quota": _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        "cgroup_v1_cpu_period": _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    }


def _disk(path: str) -> dict[str, int] | None:
    try:
        usage = shutil.disk_usage(path)
        return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    except OSError:
        return None


def run_operator(device: Any, matrix_size: int) -> dict[str, Any]:
    import torch

    if matrix_size < 8:
        raise ValueError("matrix-size 必须至少为 8")
    torch.manual_seed(47)
    left = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
    right = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
    synchronize(device)
    started = time.perf_counter()
    product = left @ right
    synchronize(device)
    elapsed = time.perf_counter() - started
    checksum = float(product.float().mean().cpu())
    finite = bool(torch.isfinite(product).all().cpu())
    return {
        "operator": "torch.matmul",
        "dtype": "float16",
        "shape": [matrix_size, matrix_size],
        "elapsed_seconds": elapsed,
        "checksum_mean": checksum,
        "finite": finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--matrix-size", type=int, default=4096)
    args = parser.parse_args()
    try:
        device, selection = select_training_device(args.device, allow_cpu_fallback=False)
        operation = run_operator(device, args.matrix_size)
        result = {
            "format": "colorlm-ascend910b-doctor-v1",
            "ok": operation["finite"],
            "platform": {
                "python": sys.version.split()[0],
                "machine": platform.machine(),
                "system": platform.platform(),
                "pid": os.getpid(),
            },
            "selection": selection.to_json(),
            "storage": {"root": _disk("/"), "dev_shm": _disk("/dev/shm")},
            "limits": _limit_info(),
            "operation": operation,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "format": "colorlm-ascend910b-doctor-v1",
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
