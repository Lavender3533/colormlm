"""Ascend/CPU 训练设备选择、共享内存安全配置和最小训练自检。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal


DevicePreference = Literal["auto", "npu", "cpu"]


def _configure_utf8() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    selected: str
    fallback_used: bool
    fallback_reason: str | None
    torch_version: str
    torch_npu_version: str | None
    device_name: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def configure_process(cpu_threads: int = 8) -> None:
    """设置适合 64 MiB /dev/shm 容器的保守训练进程参数。"""

    _configure_utf8()
    threads = max(1, min(int(cpu_threads), os.cpu_count() or 1))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def safe_dataloader_kwargs() -> dict[str, Any]:
    """返回不会启用 worker 进程或占用 /dev/shm 队列的 DataLoader 参数。"""

    return {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }


def _import_torch() -> Any:
    import torch

    return torch


def _try_npu(torch: Any) -> tuple[Any | None, str | None, str | None, str | None]:
    """探测并实际执行一个极小 NPU 算子，失败时返回原因。"""

    try:
        import torch_npu
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        return None, None, None, f"torch_npu 导入失败: {type(exc).__name__}: {exc}"

    version = str(getattr(torch_npu, "__version__", "unknown"))
    try:
        if not bool(torch_npu.npu.is_available()):
            return None, version, None, "torch_npu.npu.is_available() 为 false"
        device = torch.device("npu:0")
        probe = torch.ones((8, 8), dtype=torch.float32, device=device)
        probe = probe + probe
        torch_npu.npu.synchronize()
        if float(probe.mean().cpu()) != 2.0:
            return None, version, None, "NPU 探针结果错误"
        name = str(torch_npu.npu.get_device_name(0))
        return device, version, name, None
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        return None, version, None, f"NPU 探针失败: {type(exc).__name__}: {exc}"


def select_training_device(
    preference: DevicePreference = "auto",
    *,
    allow_cpu_fallback: bool = True,
    cpu_threads: int = 8,
) -> tuple[Any, DeviceSelection]:
    """选择训练设备；auto/npu 探测失败时可安全回退到 CPU。"""

    if preference not in {"auto", "npu", "cpu"}:
        raise ValueError(f"未知设备偏好: {preference}")
    configure_process(cpu_threads)
    torch = _import_torch()
    torch.set_num_threads(max(1, min(int(cpu_threads), os.cpu_count() or 1)))

    if preference == "cpu":
        device = torch.device("cpu")
        return device, DeviceSelection(
            requested=preference,
            selected=str(device),
            fallback_used=False,
            fallback_reason=None,
            torch_version=str(torch.__version__),
            torch_npu_version=None,
            device_name=None,
        )

    device, npu_version, name, error = _try_npu(torch)
    if device is not None:
        return device, DeviceSelection(
            requested=preference,
            selected=str(device),
            fallback_used=False,
            fallback_reason=None,
            torch_version=str(torch.__version__),
            torch_npu_version=npu_version,
            device_name=name,
        )
    if not allow_cpu_fallback:
        raise RuntimeError(error or "NPU 不可用")
    cpu = torch.device("cpu")
    return cpu, DeviceSelection(
        requested=preference,
        selected=str(cpu),
        fallback_used=True,
        fallback_reason=error,
        torch_version=str(torch.__version__),
        torch_npu_version=npu_version,
        device_name=None,
    )


def seed_everything(seed: int, device: Any | None = None) -> None:
    random.seed(seed)
    torch = _import_torch()
    torch.manual_seed(seed)
    if device is not None and str(device).startswith("npu"):
        try:
            import torch_npu

            torch_npu.npu.manual_seed_all(seed)
        except (AttributeError, RuntimeError):
            pass


def move_to_device(value: Any, device: Any) -> Any:
    """递归移动 batch；保持字符串和其他元数据不变。"""

    torch = _import_torch()
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def synchronize(device: Any) -> None:
    if not str(device).startswith("npu"):
        return
    import torch_npu

    torch_npu.npu.synchronize()


def run_training_selftest(device: Any, *, seed: int = 47, steps: int = 12) -> dict[str, Any]:
    """在所选设备执行一个小型真实反向传播，并用单进程 DataLoader 取数。"""

    torch = _import_torch()
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(seed, device)
    features = torch.randn(96, 16, generator=torch.Generator().manual_seed(seed))
    labels = ((features[:, 0] + features[:, 1] * 0.5) > 0).long()
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=24,
        shuffle=False,
        **safe_dataloader_kwargs(),
    )
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 2)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    started = time.perf_counter()
    iterator = iter(loader)
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch_features, batch_labels = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch_features), batch_labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    synchronize(device)
    return {
        "ok": all(value == value for value in losses),
        "device": str(device),
        "steps": steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "elapsed_seconds": time.perf_counter() - started,
        "dataloader": safe_dataloader_kwargs(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "npu", "cpu"), default="auto")
    parser.add_argument("--require-device", action="store_true", help="禁止回退 CPU")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    try:
        device, selection = select_training_device(
            args.device,
            allow_cpu_fallback=not args.require_device,
        )
        result = {
            "format": "colorlm-ascend910b-training-device-selftest-v1",
            "selection": selection.to_json(),
            "training": run_training_selftest(device, steps=args.steps),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["training"]["ok"] else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "format": "colorlm-ascend910b-training-device-selftest-v1",
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
