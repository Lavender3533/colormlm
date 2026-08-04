#!/usr/bin/env python3
"""Blind Composite-Axis Gate 的秒级合同测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from axis_core import CompositeAxis, read_json, schema_from_snapshot
from evaluate_frame import evaluate
from generate_world import generate
from synthesize_frame import synthesize


class BlindCompositeAxisTests(unittest.TestCase):
    def test_discovery_does_not_expose_hidden_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = generate(Path(directory), seed=17, discovery_count=64, holdout_count=128, intervention_count=16)
            discovery_text = paths["discovery"].read_text(encoding="utf-8")
            discovery = json.loads(discovery_text)
            self.assertNotIn("hidden_axis", discovery_text)
            self.assertNotIn("workload", discovery_text)
            self.assertNotIn("runtime_class", discovery_text)
            self.assertFalse(discovery["contract"]["contains_hidden_expression"])

    def test_frame_is_composite_and_fits_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = generate(root, seed=29, discovery_count=128, holdout_count=128, intervention_count=16)
            frame = synthesize(paths["discovery"], root / "frame.json")
            axis = CompositeAxis.from_snapshot(frame["axis"])
            self.assertGreaterEqual(len(axis.required_organs), 2)
            self.assertEqual(frame["training"]["accuracy"], 1.0)
            self.assertFalse(frame["process_contract"]["holdout_argument_exists"])

    def test_independent_holdout_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = generate(root, seed=43, discovery_count=256, holdout_count=512, intervention_count=32)
            synthesize(paths["discovery"], root / "frame.json")
            receipt = evaluate(
                discovery_path=paths["discovery"],
                frame_path=root / "frame.json",
                manifest_path=paths["manifest"],
                holdout_path=paths["holdout"],
                output_path=root / "evaluation.json",
            )
            self.assertTrue(receipt["passed"])
            self.assertGreaterEqual(receipt["metrics"]["holdout_accuracy"], 0.98)
            permutation = receipt["metrics"]["permuted_label_metrics"]
            self.assertLessEqual(permutation["mean_information_accuracy"], 0.62)
            self.assertLessEqual(
                permutation["target_recovery_rate"],
                permutation["recovery_rate_limit_3sigma"],
            )

    def test_tampered_holdout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = generate(root, seed=71, discovery_count=64, holdout_count=128, intervention_count=16)
            synthesize(paths["discovery"], root / "frame.json")
            holdout = read_json(paths["holdout"])
            episodes = holdout["episodes"]
            assert isinstance(episodes, list)
            episodes[0]["outcome"] = not episodes[0]["outcome"]
            paths["holdout"].write_text(json.dumps(holdout), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "holdout SHA"):
                evaluate(
                    discovery_path=paths["discovery"],
                    frame_path=root / "frame.json",
                    manifest_path=paths["manifest"],
                    holdout_path=paths["holdout"],
                    output_path=root / "evaluation.json",
                )

    def test_schema_contains_only_raw_organs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = generate(Path(directory), seed=101, discovery_count=64, holdout_count=128, intervention_count=16)
            discovery = read_json(paths["discovery"])
            schema = schema_from_snapshot(discovery["schema"])
            self.assertEqual(len(schema), 4)
            self.assertTrue(all(len(features) == 2 for features in schema.values()))


if __name__ == "__main__":
    unittest.main()
