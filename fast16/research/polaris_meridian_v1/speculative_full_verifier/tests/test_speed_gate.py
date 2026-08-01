from __future__ import annotations

import unittest

from ..runtime_controller import BATCHED_CAUSAL_MODE
from ..speed_gate import RoundTiming, current_serial_static_gate, evaluate_speed_gate


class SpeedGateTest(unittest.TestCase):
    def test_serial_k8_is_rejected_even_with_zero_draft_cost_and_full_acceptance(self):
        report = current_serial_static_gate(
            baseline_target_seconds_per_token=10.0, block_size=8
        )
        self.assertEqual(report["status"], "stop_serial_verifier")
        self.assertFalse(report["passed"])
        self.assertEqual(report["totals"]["speedup_vs_native"], 1.0)

    def test_static_one_pass_projection_never_becomes_measured_claim(self):
        report = evaluate_speed_gate(
            [RoundTiming(8, 8, 1.0, 10.0, BATCHED_CAUSAL_MODE, 1)],
            baseline_target_seconds_per_token=10.0,
            evidence_kind="static_projection",
            same_process_adjacent_baseline=False,
        )
        self.assertGreater(report["totals"]["speedup_vs_native"], 7.0)
        self.assertEqual(report["status"], "hold_for_measured_model_ab")
        self.assertFalse(report["passed"])

    def test_adjacent_measured_batched_run_can_pass(self):
        report = evaluate_speed_gate(
            [
                RoundTiming(4, 3, 0.2, 2.0, BATCHED_CAUSAL_MODE, 1),
                RoundTiming(4, 4, 0.2, 2.0, BATCHED_CAUSAL_MODE, 1),
            ],
            baseline_target_seconds_per_token=1.0,
            evidence_kind="measured_model",
            same_process_adjacent_baseline=True,
        )
        self.assertEqual(report["totals"]["committed_tokens"], 8)
        self.assertTrue(report["passed"])

    def test_apparent_speedup_without_adjacent_baseline_is_held(self):
        report = evaluate_speed_gate(
            [RoundTiming(4, 4, 0.1, 1.0, BATCHED_CAUSAL_MODE, 1)],
            baseline_target_seconds_per_token=10.0,
            evidence_kind="measured_model",
            same_process_adjacent_baseline=False,
        )
        self.assertEqual(report["status"], "hold_for_measured_model_ab")
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
