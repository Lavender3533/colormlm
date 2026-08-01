from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.candidate_profile import (
    CandidateProfiler,
    MaterializedFp8Cache,
    MethodTarget,
    offline_profile,
)


class _Nested:
    layer = 7

    def inner(self) -> str:
        time.sleep(0.001)
        return "ok"

    def outer(self) -> str:
        time.sleep(0.001)
        return self.inner()


class _Worker:
    def execute(self) -> tuple[None, dict[str, float]]:
        time.sleep(0.003)
        return None, {"worker_wall_ms": 2.0, "gpu_kernel_ms": 0.5}


def test_runtime_profiler_tracks_nested_exclusive_and_worker_boundary() -> None:
    original_inner = _Nested.inner
    original_outer = _Nested.outer
    original_execute = _Worker.execute
    profiler = CandidateProfiler(
        (
            MethodTarget(_Nested, "outer", "outer"),
            MethodTarget(_Nested, "inner", "inner"),
            MethodTarget(_Worker, "execute", "ipc", worker_evidence=True),
        )
    )
    with profiler:
        assert _Nested().outer() == "ok"
        _Worker().execute()
    assert _Nested.inner is original_inner
    assert _Nested.outer is original_outer
    assert _Worker.execute is original_execute
    report = profiler.snapshot()
    assert report["phases"]["outer"]["inclusive"]["calls"] == 1
    assert (
        report["phases"]["outer"]["inclusive"]["total_seconds"]
        > report["phases"]["outer"]["exclusive"]["total_seconds"]
    )
    assert report["per_layer_inclusive_seconds"]["7"]["inner"] > 0
    assert report["vulkan_boundary"]["calls"] == 1
    assert report["vulkan_boundary"]["worker_reported_seconds"] == 0.002
    assert report["vulkan_boundary"]["gpu_kernel_seconds"] == 0.0005


def test_offline_profile_keeps_unmeasured_time_unattributed() -> None:
    report = offline_profile(
        {
            "execution_seconds": 10.0,
            "tokens": [
                {
                    "layers": [
                        {
                            "layer": 0,
                            "elapsed_seconds": 4.0,
                            "vulkan_writeback": {
                                "worker_wall_ms": 1000.0,
                                "gpu_kernel_ms": 250.0,
                            },
                        },
                        {
                            "layer": 1,
                            "elapsed_seconds": 5.0,
                            "vulkan_writeback": {
                                "worker_wall_ms": 2000.0,
                                "gpu_kernel_ms": 500.0,
                            },
                        },
                    ]
                }
            ],
        },
        measurement_gc_removed=False,
    )
    assert report["end_to_end"]["layer_wall_seconds"] == 9.0
    assert report["end_to_end"]["outside_layer_loop_seconds"] == 1.0
    assert report["vulkan_boundary"]["worker_reported_seconds"] == 3.0
    assert report["vulkan_boundary"]["gpu_kernel_seconds"] == 0.75
    assert report["unattributed_residual"]["seconds"] == 7.0
    assert report["gc_status"]["measurement_code_gc_removed"] is False


class _FakeInline:
    calls = 0

    def __init__(self, root: Path) -> None:
        weight = root / "x.weight"
        scale = root / "x.scale"
        weight.write_bytes(b"1234")
        scale.write_bytes(b"56")
        self.bundle = SimpleNamespace(
            entries={
                "x.weight": {"path": str(weight)},
                "x.scale": {"path": str(scale)},
            }
        )

    def _path(self, name: str) -> Path:
        return Path(self.bundle.entries[name]["path"])

    def _weight_fp8(self, prefix: str, *, bf16: bool = False) -> np.ndarray:
        del prefix, bf16
        type(self).calls += 1
        return np.arange(8, dtype=np.float32)


def test_materialized_fp8_cache_reuses_read_only_array(tmp_path: Path) -> None:
    from fast16.research.polaris_meridian_v1.l42_real_reference import l42_reference

    original_class = l42_reference._InlineForward
    l42_reference._InlineForward = _FakeInline
    try:
        _FakeInline.calls = 0
        cache = MaterializedFp8Cache(max_bytes=1024)
        helper = _FakeInline(tmp_path)
        with cache:
            first = helper._weight_fp8("x")
            second = helper._weight_fp8("x")
        assert first is second
        assert not first.flags.writeable
        assert _FakeInline.calls == 1
        assert cache.stats()["hits"] == 1
        assert cache.stats()["resident_bytes"] == first.nbytes
    finally:
        l42_reference._InlineForward = original_class
