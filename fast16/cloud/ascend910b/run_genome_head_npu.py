"""用 Ascend/CPU 设备运行原版 v47 Parallel Genome Head 训练器。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .training_device import seed_everything, select_training_device
except ImportError:
    from training_device import seed_everything, select_training_device


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
TRAINER_PATH = PROJECT_ROOT / "fast16/research/v47_dual_tempo_bus/fit_parallel_genome_head.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trainer() -> Any:
    spec = importlib.util.spec_from_file_location("_colorlm_v47_genome_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载训练器: {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TorchDeviceFacade:
    """只把原训练器的 NumPy batch 移到目标设备，其他 torch API 原样转发。"""

    def __init__(self, torch_module: Any, device: Any) -> None:
        self._torch = torch_module
        self._device = device

    def from_numpy(self, array: Any) -> Any:
        return self._torch.from_numpy(array).to(device=self._device, non_blocking=False)

    def _factory(self, name: str, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("device", self._device)
        return getattr(self._torch, name)(*args, **kwargs)

    def arange(self, *args: Any, **kwargs: Any) -> Any:
        return self._factory("arange", *args, **kwargs)

    def empty(self, *args: Any, **kwargs: Any) -> Any:
        return self._factory("empty", *args, **kwargs)

    def full(self, *args: Any, **kwargs: Any) -> Any:
        return self._factory("full", *args, **kwargs)

    def ones(self, *args: Any, **kwargs: Any) -> Any:
        return self._factory("ones", *args, **kwargs)

    def rand(self, *args: Any, **kwargs: Any) -> Any:
        return self._factory("rand", *args, **kwargs)

    def randn(self, *args: Any, **kwargs: Any) -> Any:
        return self._factory("randn", *args, **kwargs)

    def tensor(self, *args: Any, **kwargs: Any) -> Any:
        return self._factory("tensor", *args, **kwargs)

    def zeros(self, *args: Any, **kwargs: Any) -> Any:
        return self._factory("zeros", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)


def _install_device_adapter(trainer: Any, device: Any) -> None:
    original_class = trainer.ParallelGenomeHead

    class DeviceParallelGenomeHead(original_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.to(device)

    DeviceParallelGenomeHead.__name__ = original_class.__name__
    DeviceParallelGenomeHead.__qualname__ = original_class.__qualname__
    trainer.ParallelGenomeHead = DeviceParallelGenomeHead
    trainer.torch = _TorchDeviceFacade(trainer.torch, device)


def _argument_value(arguments: list[str], option: str) -> Path | None:
    try:
        index = arguments.index(option)
    except ValueError:
        return None
    if index + 1 >= len(arguments):
        return None
    return Path(arguments[index + 1])


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=("auto", "npu", "cpu"), default="auto")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--runtime-report", type=Path)
    wrapper_args, trainer_args = parser.parse_known_args()

    device, selection = select_training_device(
        wrapper_args.device,
        allow_cpu_fallback=not wrapper_args.require_device,
    )
    seed_everything(47, device)
    trainer = _load_trainer()
    _install_device_adapter(trainer, device)

    report_path = _argument_value(trainer_args, "--report")
    runtime_report = wrapper_args.runtime_report
    if runtime_report is None and report_path is not None:
        runtime_report = report_path.with_suffix(report_path.suffix + ".runtime.json")
    if runtime_report is not None and runtime_report.exists():
        raise FileExistsError(f"拒绝覆盖设备运行报告: {runtime_report}")

    previous_argv = sys.argv
    started = time.perf_counter()
    try:
        sys.argv = [str(TRAINER_PATH), *trainer_args]
        exit_code = int(trainer.main())
    finally:
        sys.argv = previous_argv

    runtime = {
        "format": "colorlm-v47-parallel-genome-head-device-run-v1",
        "ok": exit_code == 0,
        "selection": selection.to_json(),
        "trainer": str(TRAINER_PATH),
        "trainer_sha256": _sha256(TRAINER_PATH),
        "elapsed_seconds": time.perf_counter() - started,
        "trainer_report": None if report_path is None else str(report_path.resolve()),
        "trainer_report_sha256": (
            _sha256(report_path) if report_path is not None and report_path.exists() else None
        ),
        "data_loading": {
            "strategy": "direct_numpy_batch",
            "num_workers": 0,
            "pin_memory": False,
            "uses_dev_shm_worker_queue": False,
        },
    }
    if runtime_report is not None:
        runtime_report.parent.mkdir(parents=True, exist_ok=True)
        runtime_report.write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(runtime, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
