"""FullDepth43 Vulkan candidate 的可撤销端到端阶段剖析器。

本模块不修改 executor 或 shader。运行时通过上下文管理器临时包装已冻结边界，
分别记录 inclusive/exclusive 墙钟；离开上下文后恢复所有原方法。离线入口只从
既有 candidate 报告提取能够直接证明的 worker/GPU/层墙钟，未测部分保留为
``unattributed_residual``，不会把局部 kernel 时间冒充整 token 速度。
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import math
import os
import threading
import time
from collections import OrderedDict, defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PROFILE_FORMAT = "polaris-fulldepth43-candidate-profile-v1"


@dataclass(frozen=True)
class MethodTarget:
    owner: Any
    attribute: str
    label: str
    layer_argument: int | None = None
    set_sticky_layer: bool = False
    clear_sticky_layer: bool = False
    worker_evidence: bool = False


@dataclass
class _Frame:
    label: str
    layer: int | None
    started: float
    child_seconds: float = 0.0


@dataclass(frozen=True)
class _Observation:
    label: str
    layer: int | None
    inclusive_seconds: float
    exclusive_seconds: float


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summarize(values: Sequence[float]) -> dict[str, Any]:
    frozen = tuple(float(value) for value in values)
    return {
        "calls": len(frozen),
        "total_seconds": sum(frozen),
        "mean_seconds": 0.0 if not frozen else sum(frozen) / len(frozen),
        "p50_seconds": _percentile(frozen, 0.50),
        "p95_seconds": _percentile(frozen, 0.95),
        "max_seconds": max(frozen, default=0.0),
    }


def default_targets() -> tuple[MethodTarget, ...]:
    """延迟导入真实运行时，避免只做离线报告时加载全部执行器。"""

    from fast16.research.polaris_meridian_v1.fulldepth43_native_top6 import (
        executor as full,
    )
    from fast16.research.polaris_meridian_v1.fulldepth43_native_top6 import (
        vulkan_writeback,
    )
    from fast16.research.polaris_meridian_v1.l42_real_reference import (
        l42_reference,
    )
    from fast16.research.polaris_meridian_v1.s14_first_real_token import (
        executor as s14,
    )

    return (
        MethodTarget(full.FullDepthTokenWorker, "__call__", "token_total"),
        MethodTarget(
            full.FullDepthRangeSession,
            "prepare_embedding_row",
            "range_embedding",
        ),
        MethodTarget(
            full.FullDepthRangeSession,
            "prepare_layer",
            "range_prepare_layer",
            layer_argument=1,
            set_sticky_layer=True,
        ),
        MethodTarget(
            full.FullDepthRangeSession,
            "fetch_routed",
            "range_fetch_routed",
            layer_argument=1,
        ),
        MethodTarget(
            full.FullDepthRangeSession,
            "finish_layer",
            "range_finish_layer",
            layer_argument=1,
            clear_sticky_layer=True,
        ),
        MethodTarget(full.FullDepthRangeSession, "prepare_final", "range_prepare_final"),
        MethodTarget(s14.TensorStore, "add_ranges", "range_proof_index"),
        MethodTarget(s14.NativeLayerReference, "_attention", "attention"),
        MethodTarget(
            s14.NativeLayerReference,
            "_advance_compressor_state",
            "compressor_indexer",
        ),
        MethodTarget(s14.NativeLayerReference, "_route", "router"),
        MethodTarget(s14.NativeLayerReference, "_load_i64", "load_i64"),
        MethodTarget(l42_reference._InlineForward, "_load_tensor", "load_tensor"),
        MethodTarget(l42_reference._InlineForward, "_weight_fp8", "materialize_fp8"),
        MethodTarget(l42_reference._InlineForward, "_weight_fp4", "materialize_fp4"),
        MethodTarget(l42_reference._InlineForward, "_linear_fp8", "linear_fp8"),
        MethodTarget(l42_reference._InlineForward, "_linear_fp4", "linear_fp4"),
        MethodTarget(s14, "hc_pre", "hc_pre"),
        MethodTarget(s14, "hc_post", "hc_post"),
        MethodTarget(full, "_write_vulkan_bridge_capture", "capture_io"),
        MethodTarget(
            vulkan_writeback.PersistentVulkanWriteback,
            "execute",
            "python_rust_vulkan",
            worker_evidence=True,
        ),
        MethodTarget(s14.FinalHeadReference, "forward", "final_head"),
    )


class CandidateProfiler(AbstractContextManager["CandidateProfiler"]):
    """线程安全、可撤销的 inclusive/exclusive phase profiler。"""

    def __init__(
        self,
        targets: Sequence[MethodTarget] | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.targets = tuple(default_targets() if targets is None else targets)
        self.clock = clock
        self._local = threading.local()
        self._lock = threading.Lock()
        self._observations: list[_Observation] = []
        self._worker_rows: list[dict[str, float]] = []
        self._patches: list[tuple[Any, str, Any, Any]] = []
        self._active = False

    def _stack(self) -> list[_Frame]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _sticky_layer(self) -> int | None:
        value = getattr(self._local, "sticky_layer", None)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _resolve_layer(self, target: MethodTarget, args: tuple[Any, ...]) -> int | None:
        if target.layer_argument is not None and len(args) > target.layer_argument:
            value = args[target.layer_argument]
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        if args:
            value = getattr(args[0], "layer", None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return self._sticky_layer()

    def _capture_worker(self, result: Any, execute_wall: float) -> None:
        if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[1], Mapping):
            return
        evidence = result[1]
        worker_ms = evidence.get("worker_wall_ms")
        gpu_ms = evidence.get("gpu_kernel_ms")
        if not isinstance(worker_ms, (int, float)) or not isinstance(gpu_ms, (int, float)):
            return
        if worker_ms < 0 or gpu_ms < 0 or gpu_ms > worker_ms + 1e-6:
            return
        self._worker_rows.append(
            {
                "python_execute_seconds": execute_wall,
                "worker_reported_seconds": float(worker_ms) / 1000.0,
                "gpu_kernel_seconds": float(gpu_ms) / 1000.0,
            }
        )

    def _wrapper(self, target: MethodTarget, original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            old_sticky = self._sticky_layer()
            layer = self._resolve_layer(target, args)
            if target.set_sticky_layer:
                self._local.sticky_layer = layer
            frame = _Frame(target.label, layer, self.clock())
            stack = self._stack()
            stack.append(frame)
            result: Any = None
            succeeded = False
            try:
                result = original(*args, **kwargs)
                succeeded = True
                return result
            finally:
                elapsed = max(0.0, self.clock() - frame.started)
                popped = stack.pop()
                if popped is not frame:
                    raise RuntimeError("candidate profiler 调用栈损坏")
                exclusive = max(0.0, elapsed - frame.child_seconds)
                if stack:
                    stack[-1].child_seconds += elapsed
                with self._lock:
                    self._observations.append(
                        _Observation(target.label, layer, elapsed, exclusive)
                    )
                    if target.worker_evidence and succeeded:
                        self._capture_worker(result, elapsed)
                if target.set_sticky_layer and not succeeded:
                    self._local.sticky_layer = old_sticky
                if target.clear_sticky_layer and succeeded:
                    self._local.sticky_layer = None

        return measured

    def __enter__(self) -> "CandidateProfiler":
        if self._active:
            raise RuntimeError("candidate profiler 不允许重复进入")
        self._active = True
        try:
            for target in self.targets:
                original = getattr(target.owner, target.attribute)
                replacement = self._wrapper(target, original)
                setattr(target.owner, target.attribute, replacement)
                self._patches.append((target.owner, target.attribute, original, replacement))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        restore_error: RuntimeError | None = None
        for owner, attribute, original, replacement in reversed(self._patches):
            if getattr(owner, attribute) is not replacement:
                restore_error = RuntimeError(f"剖析期间 {attribute} 被其他代码替换")
                continue
            setattr(owner, attribute, original)
        self._patches.clear()
        self._active = False
        if restore_error is not None and exc is None:
            raise restore_error

    def snapshot(self) -> dict[str, Any]:
        by_label_inclusive: dict[str, list[float]] = defaultdict(list)
        by_label_exclusive: dict[str, list[float]] = defaultdict(list)
        by_layer: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for row in self._observations:
            by_label_inclusive[row.label].append(row.inclusive_seconds)
            by_label_exclusive[row.label].append(row.exclusive_seconds)
            if row.layer is not None:
                by_layer[row.layer][row.label] += row.inclusive_seconds
        labels = sorted(by_label_inclusive)
        phases = {
            label: {
                "inclusive": _summarize(by_label_inclusive[label]),
                "exclusive": _summarize(by_label_exclusive[label]),
            }
            for label in labels
        }
        python_execute = sum(row["python_execute_seconds"] for row in self._worker_rows)
        worker_reported = sum(row["worker_reported_seconds"] for row in self._worker_rows)
        gpu_kernel = sum(row["gpu_kernel_seconds"] for row in self._worker_rows)
        token_wall = sum(by_label_inclusive.get("token_total", ()))
        exclusive_partition = sum(
            sum(values) for values in by_label_exclusive.values()
        )
        return {
            "format": PROFILE_FORMAT,
            "source": "runtime_instrumentation",
            "gc_status": gc_collection_status(),
            "token_wall_seconds": token_wall,
            "phases": phases,
            "per_layer_inclusive_seconds": {
                str(layer): dict(sorted(values.items()))
                for layer, values in sorted(by_layer.items())
            },
            "vulkan_boundary": {
                "calls": len(self._worker_rows),
                "python_to_rust_execute_seconds": python_execute,
                "worker_reported_seconds": worker_reported,
                "python_ipc_and_validation_seconds": max(0.0, python_execute - worker_reported),
                "gpu_kernel_seconds": gpu_kernel,
                "worker_non_kernel_seconds": max(0.0, worker_reported - gpu_kernel),
            },
            "exclusive_partition": {
                "seconds": exclusive_partition,
                "equals_single_profiled_token_only_when_token_total_calls_is_one": len(
                    by_label_inclusive.get("token_total", ())
                )
                == 1,
            },
            "measurement_note": (
                "inclusive 阶段会重叠；exclusive 只减去被本剖析器包装的嵌套子调用。"
                "memmap materialize 包含页缺失与解量化，不能解释为纯磁盘 I/O。"
            ),
        }


def gc_collection_status() -> dict[str, Any]:
    from fast16.research.polaris_meridian_v1.l42_real_reference.l42_reference import (
        _InlineForward,
    )

    sources = {
        "_linear_fp8": inspect.getsource(_InlineForward._linear_fp8),
        "_linear_fp4": inspect.getsource(_InlineForward._linear_fp4),
    }
    explicit = sum(source.count("gc.collect(") for source in sources.values())
    return {
        "gc_removed": explicit == 0,
        "explicit_gc_collect_calls_in_linear_source": explicit,
        "contract": "线性层依赖 CPython/NumPy 引用计数释放临时权重，不在热路径强制全代 GC",
    }


def offline_profile(
    report: Mapping[str, Any],
    *,
    measurement_gc_removed: bool | None = None,
) -> dict[str, Any]:
    """从旧 candidate 报告提取可证实下界，不臆造未记录的子阶段。"""

    total = float(report["execution_seconds"])
    tokens = report.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != 1:
        raise ValueError("离线剖析当前只接受单 token candidate 报告")
    layers = tokens[0].get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("candidate 报告缺少 layer 明细")
    layer_rows: list[dict[str, Any]] = []
    layer_wall = 0.0
    worker_wall = 0.0
    gpu_kernel = 0.0
    for item in layers:
        elapsed = float(item["elapsed_seconds"])
        evidence = item.get("vulkan_writeback")
        worker = 0.0
        gpu = 0.0
        if isinstance(evidence, Mapping):
            worker = float(evidence.get("worker_wall_ms", 0.0)) / 1000.0
            gpu = float(evidence.get("gpu_kernel_ms", 0.0)) / 1000.0
        if worker < 0 or gpu < 0 or gpu > worker + 1e-6:
            raise ValueError("worker/GPU 时间非法")
        layer_wall += elapsed
        worker_wall += worker
        gpu_kernel += gpu
        layer_rows.append(
            {
                "layer": int(item["layer"]),
                "layer_wall_seconds": elapsed,
                "worker_reported_seconds": worker,
                "gpu_kernel_seconds": gpu,
                "layer_outside_worker_seconds": max(0.0, elapsed - worker),
            }
        )
    if layer_wall > total + 1e-6:
        raise ValueError("layer 墙钟总和超过 token 总墙钟")
    return {
        "format": PROFILE_FORMAT,
        "source": "existing_candidate_report",
        "gc_status": {
            "current_code": gc_collection_status(),
            "measurement_code_gc_removed": measurement_gc_removed,
            "measurement_note": (
                "旧 candidate 若在热路径仍含 gc.collect，必须显式标 false；"
                "不能把当前源码状态倒填到旧墙钟。"
            ),
        },
        "end_to_end": {
            "token_wall_seconds": total,
            "tokens_per_second": 1.0 / total,
            "layer_wall_seconds": layer_wall,
            "outside_layer_loop_seconds": max(0.0, total - layer_wall),
        },
        "vulkan_boundary": {
            "worker_reported_seconds": worker_wall,
            "gpu_kernel_seconds": gpu_kernel,
            "worker_non_kernel_seconds": max(0.0, worker_wall - gpu_kernel),
            "end_to_end_outside_worker_seconds": max(0.0, total - worker_wall),
        },
        "layers": layer_rows,
        "unattributed_residual": {
            "seconds": max(0.0, total - worker_wall),
            "may_include": [
                "attention",
                "HC",
                "router",
                "compressor/indexer",
                "Range/file I/O",
                "capture I/O",
                "Python/Rust validation",
                "final head",
                "orchestration",
            ],
            "claim_limit": "旧报告没有子阶段计时，禁止继续硬分摊",
        },
    }


class MaterializedFp8Cache(AbstractContextManager["MaterializedFp8Cache"]):
    """跨 token 复用只读 FP8 解量化权重的有界 host-RAM LRU。

    默认不缓存 FP4 routed expert，避免与 Vulkan worker 的 payload LRU 重复占用。
    本基础件必须包住多个 token 才可能命中；单 token 不作速度声明。
    """

    def __init__(self, max_bytes: int = 8 * 1024**3) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes 必须为正整数")
        self.max_bytes = max_bytes
        self._values: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._resident_bytes = 0
        self._lock = threading.Lock()
        self._inline_class: Any = None
        self._original: Any = None
        self._replacement: Any = None
        self.hits = 0
        self.misses = 0
        self.insertions = 0
        self.evictions = 0
        self.oversize_skips = 0

    @staticmethod
    def _key(owner: Any, prefix: str, bf16: bool) -> tuple[Any, ...]:
        paths = []
        for suffix in (".weight", ".scale"):
            path = owner._path(prefix + suffix).resolve(strict=True)
            stat = path.stat()
            paths.append((str(path), stat.st_size, stat.st_mtime_ns))
        return ("fp8", prefix, bool(bf16), *paths)

    def _get_or_compute(
        self,
        owner: Any,
        prefix: str,
        bf16: bool,
        compute: Callable[[], np.ndarray],
    ) -> np.ndarray:
        key = self._key(owner, prefix, bf16)
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                self.hits += 1
                self._values.move_to_end(key)
                return cached
            self.misses += 1
        value = np.asarray(compute())
        value.setflags(write=False)
        size = int(value.nbytes)
        if size > self.max_bytes:
            with self._lock:
                self.oversize_skips += 1
            return value
        with self._lock:
            raced = self._values.get(key)
            if raced is not None:
                self.hits += 1
                self._values.move_to_end(key)
                return raced
            while self._values and self._resident_bytes + size > self.max_bytes:
                _, evicted = self._values.popitem(last=False)
                self._resident_bytes -= int(evicted.nbytes)
                self.evictions += 1
            self._values[key] = value
            self._resident_bytes += size
            self.insertions += 1
        return value

    def __enter__(self) -> "MaterializedFp8Cache":
        from fast16.research.polaris_meridian_v1.l42_real_reference.l42_reference import (
            _InlineForward,
        )

        if self._replacement is not None:
            raise RuntimeError("materialized FP8 cache 不允许重复进入")
        original = _InlineForward._weight_fp8

        @functools.wraps(original)
        def cached(owner: Any, prefix: str, *, bf16: bool = False) -> np.ndarray:
            return self._get_or_compute(
                owner,
                prefix,
                bf16,
                lambda: original(owner, prefix, bf16=bf16),
            )

        self._inline_class = _InlineForward
        self._original = original
        self._replacement = cached
        _InlineForward._weight_fp8 = cached
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._replacement is None:
            return
        if self._inline_class._weight_fp8 is not self._replacement:
            raise RuntimeError("缓存期间 _weight_fp8 被其他代码替换")
        self._inline_class._weight_fp8 = self._original
        self._inline_class = None
        self._original = None
        self._replacement = None

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._resident_bytes = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            lookups = self.hits + self.misses
            return {
                "max_bytes": self.max_bytes,
                "resident_bytes": self._resident_bytes,
                "entries": len(self._values),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": 0.0 if lookups == 0 else self.hits / lookups,
                "insertions": self.insertions,
                "evictions": self.evictions,
                "oversize_skips": self.oversize_skips,
                "scope": "FP8 only; FP4 routed experts intentionally excluded",
            }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, destination)


def _main() -> int:
    parser = argparse.ArgumentParser(description="离线提取 FullDepth candidate 可证实时分布")
    parser.add_argument("--offline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--measurement-explicit-gc-present",
        action="store_true",
        help="标记被读取的旧运行在线性热路径含显式 gc.collect",
    )
    args = parser.parse_args()
    source = json.loads(args.offline_report.read_text(encoding="utf-8", errors="strict"))
    value = offline_profile(
        source,
        measurement_gc_removed=(
            False if args.measurement_explicit_gc_present else None
        ),
    )
    write_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
