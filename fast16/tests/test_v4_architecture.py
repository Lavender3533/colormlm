"""Behavioral invariants for the ColorLM v4 state and size budget."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from fast16.v4 import (
    ColorLMV4Config,
    DirectMLStateCell,
    FastWeightState,
    build_qwen35_donor_plan,
    colorlm_v4_budget,
    export_state_cell,
)
from fast16.v4.transplant import rewrite_metadata_prefix


class ColorLMV4ArchitectureTests(unittest.TestCase):
    def test_model_budget_stays_within_15_gib(self) -> None:
        config = ColorLMV4Config()
        config.validate()
        budget = colorlm_v4_budget()
        budget.validate(config.max_model_bytes)
        self.assertLessEqual(budget.total_bytes, 15 * 1024**3)

    def test_donor_plan_produces_only_colorlm_temporal_cores(self) -> None:
        plan = build_qwen35_donor_plan(donor_hash="ab" * 32)
        plan.validate()
        kinds = [layer.temporal_core for layer in plan.layers]
        self.assertEqual(kinds.count("color_delta"), 30)
        self.assertEqual(kinds.count("color_kernel"), 10)
        targets = [
            transfer.target
            for layer in plan.layers
            for transfer in layer.transfers
        ]
        self.assertFalse(any(".attn" in target for target in targets))
        self.assertFalse(plan.runtime_requires_donor)

    def test_transplant_rewrites_metadata_without_moving_tensor_offsets(self) -> None:
        original = b"general.architecture=qwen35moe;qwen35moe.block=40;Qwen3.5-35B-A3B"
        rewritten, architecture_count, name_count = rewrite_metadata_prefix(original)
        self.assertEqual(len(rewritten), len(original))
        self.assertEqual(architecture_count, 2)
        self.assertEqual(name_count, 1)
        self.assertIn(b"colorlmv4", rewritten)
        self.assertIn(b"ColorLM-v4-SMoE", rewritten)

    def test_fast_state_learns_without_text_matching(self) -> None:
        state = FastWeightState(rank=64)
        rng = np.random.default_rng(7)
        key = rng.normal(size=64).astype(np.float32)
        value = rng.normal(size=64).astype(np.float32)
        before = np.linalg.norm(value - state.read(key))
        started = time.perf_counter()
        for _ in range(8):
            state.update(key, value, learning_rate=0.5)
        elapsed = time.perf_counter() - started
        after = np.linalg.norm(value - state.read(key))
        self.assertLess(after, before * 0.01)
        self.assertLess(elapsed, 1.0)

    def test_fast_state_round_trip(self) -> None:
        state = FastWeightState(rank=16)
        key = np.arange(1, 17, dtype=np.float32)
        value = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
        state.update(key, value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.clms"
            state.save(str(path))
            loaded = FastWeightState.load(str(path))
        np.testing.assert_array_equal(loaded.matrix, state.matrix)

    def test_state_cell_executes_neural_graph_on_directml(self) -> None:
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as directory:
            graph = export_state_cell(Path(directory) / "cell.onnx")
            runtime = DirectMLStateCell(graph, profile=True)
            hidden = rng.normal(size=(1, 64)).astype(np.float32)
            state = np.zeros((1, 16, 16), dtype=np.float32)
            kernel_state = np.zeros((1, 8, 64), dtype=np.float32)
            kernel_norm = np.zeros((1, 8), dtype=np.float32)
            next_hidden, next_state, next_kernel, next_norm, expert_ids = runtime.step(
                hidden, state, kernel_state, kernel_norm
            )
            providers = runtime.finish_profile()
        self.assertEqual(next_hidden.shape, (1, 64))
        self.assertEqual(next_state.shape, (1, 16, 16))
        self.assertEqual(next_kernel.shape, (1, 8, 64))
        self.assertEqual(next_norm.shape, (1, 8))
        self.assertEqual(expert_ids.shape, (1, 2))
        self.assertTrue(np.isfinite(next_hidden).all())
        self.assertGreater(np.linalg.norm(next_state), 0.0)
        self.assertGreater(np.linalg.norm(next_kernel), 0.0)
        self.assertTrue((next_norm > 0.0).all())
        self.assertGreater(providers.get("DmlExecutionProvider", 0), 0)


if __name__ == "__main__":
    unittest.main()
