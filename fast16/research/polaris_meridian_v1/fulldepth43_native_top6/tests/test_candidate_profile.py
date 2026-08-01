from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6 import (
    run_candidate_profile as candidate_runner,
)
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


def test_materialized_fp8_cache_resists_sequential_scan_pollution(tmp_path: Path) -> None:
    cache = MaterializedFp8Cache(max_bytes=32)
    owners = []
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir()
        owners.append(_FakeInline(root))

    first = cache._get_or_compute(
        owners[0], "x", False, lambda: np.arange(8, dtype=np.float32)
    )
    skipped = cache._get_or_compute(
        owners[1], "x", False, lambda: np.arange(8, dtype=np.float32) + 10
    )
    hit = cache._get_or_compute(
        owners[0], "x", False, lambda: np.full(8, -1, dtype=np.float32)
    )

    assert hit is first
    assert skipped is not first
    stats = cache.stats()
    assert stats["resident_bytes"] == 32
    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["capacity_skips"] == 1
    assert stats["evictions"] == 0


@pytest.mark.parametrize(
    ("token_count", "execution_seconds", "token_walls"),
    (
        (1, 4.0, [3.5]),
        (3, 9.0, [3.5, 2.5, 2.0]),
    ),
)
def test_profiled_candidate_runs_continuous_tokens_and_keeps_single_token_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token_count: int,
    execution_seconds: float,
    token_walls: list[float],
) -> None:
    worker = tmp_path / "writeback.exe"
    attention_worker = tmp_path / "attention.exe"
    final_head_worker = tmp_path / "head.exe"
    worker.write_bytes(b"worker")
    attention_worker.write_bytes(b"attention")
    final_head_worker.write_bytes(b"head")
    attention_scratch = tmp_path / "attention-scratch"
    scratch = tmp_path / "head-scratch"
    observed_configs = []

    class FakeProfiler:
        def __init__(self) -> None:
            self._observations = [
                SimpleNamespace(label="token_total", inclusive_seconds=value)
                for value in token_walls
            ]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def snapshot(self):
            return {
                "token_wall_seconds": sum(token_walls),
                "gc_status": {"gc_removed": True},
                "vulkan_boundary": {"calls": token_count * 43},
            }

    def fake_execute(config):
        observed_configs.append(config)
        return {
            "status": "complete",
            "execution_seconds": execution_seconds,
            "tokens": [
                {
                    "position": position,
                    "layers": [{"layer": layer} for layer in range(43)],
                }
                for position in range(token_count)
            ],
            "committed_tokens": [
                {
                    "position": position,
                    "input_token_id": 100 + position,
                    "output_token_id": 200 + position,
                }
                for position in range(token_count)
            ],
            "vulkan_writeback_fallbacks": [],
        }

    monkeypatch.setattr(candidate_runner, "CandidateProfiler", FakeProfiler)
    monkeypatch.setattr(candidate_runner, "execute", fake_execute)
    output_root = tmp_path / f"run-{token_count}"
    summary = candidate_runner.run_profiled_candidate(
        worker=worker,
        output_root=output_root,
        token_count=token_count,
        vulkan_attention_worker=attention_worker,
        vulkan_attention_scratch=attention_scratch,
        vulkan_attention_shared_batch=True,
        vulkan_final_head_worker=final_head_worker,
        vulkan_final_head_scratch=scratch,
    )

    assert len(observed_configs) == 1
    config = observed_configs[0]
    assert config.token_count == token_count
    assert config.vulkan_attention_worker == attention_worker.resolve()
    assert config.vulkan_attention_scratch == attention_scratch.resolve()
    assert config.vulkan_attention_shared_batch is True
    assert config.vulkan_final_head_worker == final_head_worker.resolve()
    assert config.vulkan_final_head_scratch == scratch.resolve()
    assert summary["committed_token_count"] == token_count
    assert summary["per_token_wall_seconds"] == token_walls
    assert [row["wall_seconds"] for row in summary["tokens"]] == token_walls
    assert summary["output_token_id"] == 199 + token_count
    assert summary["tokens_per_second"] == token_count / execution_seconds
    assert summary["effective_tokens_per_second"] == token_count / execution_seconds
    runtime = json.loads(
        (output_root / "runtime_profile.json").read_text(encoding="utf-8")
    )
    assert runtime["per_token_wall_seconds"] == token_walls


def test_profiled_candidate_rejects_token_count_outside_contract(tmp_path: Path) -> None:
    for value in (True, 0, 17):
        with pytest.raises(ValueError, match="1..16"):
            candidate_runner.run_profiled_candidate(
                worker=tmp_path / "missing.exe",
                output_root=tmp_path / f"invalid-{value}",
                token_count=value,
            )


def test_profiled_candidate_surfaces_model_error_before_timing_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = tmp_path / "worker.exe"
    worker.write_bytes(b"worker")

    class FakeProfiler:
        def __init__(self) -> None:
            self._observations = []

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def snapshot(self):
            return {"gc_status": {"gc_removed": True}, "vulkan_boundary": {}}

    monkeypatch.setattr(candidate_runner, "CandidateProfiler", FakeProfiler)
    monkeypatch.setattr(
        candidate_runner,
        "execute",
        lambda _config: {
            "status": "blocked",
            "error": {"stage": "attention", "message": "frozen failure"},
        },
    )
    with pytest.raises(candidate_runner.FullDepthError, match="frozen failure"):
        candidate_runner.run_profiled_candidate(
            worker=worker,
            output_root=tmp_path / "blocked-run",
            token_count=2,
        )


def test_profiled_candidate_cli_forwards_continuous_head_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(candidate_runner, "run_profiled_candidate", fake_run)
    worker = tmp_path / "worker.exe"
    output_root = tmp_path / "output"
    final_worker = tmp_path / "head.exe"
    attention_worker = tmp_path / "attention.exe"
    attention_scratch = tmp_path / "attention-scratch"
    scratch = tmp_path / "scratch"
    assert candidate_runner.main(
        [
            "--worker",
            str(worker),
            "--output-root",
            str(output_root),
            "--token-count",
            "4",
            "--vulkan-attention-worker",
            str(attention_worker),
            "--vulkan-attention-scratch",
            str(attention_scratch),
            "--vulkan-attention-shared-batch",
            "--vulkan-final-head-worker",
            str(final_worker),
            "--vulkan-final-head-scratch",
            str(scratch),
        ]
    ) == 0
    assert captured["token_count"] == 4
    assert captured["vulkan_attention_worker"] == attention_worker
    assert captured["vulkan_attention_scratch"] == attention_scratch
    assert captured["vulkan_attention_shared_batch"] is True
    assert captured["vulkan_final_head_worker"] == final_worker
    assert captured["vulkan_final_head_scratch"] == scratch
