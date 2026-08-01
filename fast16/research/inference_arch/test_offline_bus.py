#!/usr/bin/env python3
"""offline_bus 的纯 CPU 自测。"""

from __future__ import annotations

import unittest

import numpy as np

from offline_bus import (
    apply_gate,
    fit_gate,
    make_synthetic_capture,
    replay_alphas,
    sparsemax,
    validate_complementarity,
    validate_single_donor,
)


class OfflineBusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.capture = make_synthetic_capture()

    def test_sparsemax_simplex_and_exact_no_op(self) -> None:
        scores = np.asarray([[4.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
        weights = sparsemax(scores)
        np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)
        np.testing.assert_array_equal(weights[0], np.asarray([1.0, 0.0, 0.0]))
        self.assertTrue(np.all(weights >= 0.0))

    def test_alpha_zero_is_exact(self) -> None:
        report = replay_alphas(self.capture, 0, (0.0, 0.01, 0.03))
        self.assertTrue(report["alpha_zero_exact"])
        self.assertEqual(report["points"][0]["mean_delta"], 0.0)

    def test_gate_is_numeric_and_normalized(self) -> None:
        model = fit_gate(self.capture, (0, 1), epochs=300)
        logits, gate = apply_gate(self.capture, model)
        self.assertEqual(logits.shape, self.capture.base_logits.shape)
        np.testing.assert_allclose(np.sum(gate, axis=1), 1.0, atol=1e-10)
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_second_donor_is_conditionally_complementary(self) -> None:
        report = validate_complementarity(self.capture, epochs=350)
        self.assertGreater(report["conflict_count"], 0)
        self.assertGreater(report["conflict_accuracy"]["resolvable_count"], 0)
        self.assertGreater(report["conflict_accuracy"]["fused_resolvable_accuracy"], 0.8)
        self.assertGreater(report["no_op_rate"], 0.05)
        self.assertLess(report["mean_delta"], 0.0)
        self.assertTrue(report["conditionally_complementary"])

    def test_single_donor_noop_gate_uses_loto(self) -> None:
        report = validate_single_donor(self.capture, donor_index=0, epochs=300)
        self.assertEqual(report["task_count"], 6)
        self.assertGreater(report["exact_no_op_rate"], 0.0)
        self.assertEqual(len(report["folds"]), 6)
        self.assertIn("weights", report["full_model"])


if __name__ == "__main__":
    unittest.main()
