from __future__ import annotations

import unittest

from fast16.research.polaris_meridian_v1.speculative_full_verifier.route_speed_probe import (
    RouteSpeedProbeError,
    analyze_report,
)


def _layer(layer: int, experts: list[int], weights: list[float]) -> dict:
    return {"layer": layer, "expert_ids": experts, "route_weights": weights}


class RouteSpeedProbeTests(unittest.TestCase):
    def test_reports_overlap_and_mass_adaptive_bytes(self) -> None:
        report = {
            "status": "complete",
            "preflight": {
                "dynamic_routed_experts": {
                    "native_top6_cold_bytes_per_token": 600,
                }
            },
            "tokens": [
                {
                    "position": 0,
                    "layers": [
                        _layer(0, [0, 1, 2, 3, 4, 5], [9, 1, 0, 0, 0, 0]),
                    ],
                },
                {
                    "position": 1,
                    "layers": [
                        _layer(0, [0, 6, 7, 8, 9, 10], [4, 3, 2, 1, 0, 0]),
                    ],
                },
            ],
        }
        result = analyze_report(report, thresholds=(0.8, 0.9))

        self.assertEqual(result["evidence_status"], "complete_trace")
        self.assertEqual(result["route_rows"], 2)
        self.assertEqual(result["adjacent_route_reuse"]["comparisons"], 1)
        self.assertEqual(result["adjacent_route_reuse"]["mean_overlap"], 1)
        self.assertEqual(result["adjacent_route_reuse"]["mean_union"], 11)
        self.assertAlmostEqual(
            result["adjacent_route_reuse"]["expert_io_fraction_vs_serial"],
            11 / 12,
        )
        self.assertEqual(
            result["adjacent_route_reuse"]["per_pair"][0]["left_position"], 0
        )
        self.assertEqual(
            result["adjacent_route_reuse"]["steady_state_excluding_bos_pair"][
                "comparisons"
            ],
            0,
        )
        eighty, ninety = result["mass_adaptive_candidates"]
        self.assertEqual(eighty["mean_selected_experts"], 2)
        self.assertEqual(eighty["projected_dynamic_bytes_per_token"], 200)
        self.assertEqual(ninety["mean_selected_experts"], 2)
        self.assertEqual(ninety["projected_dynamic_bytes_per_token"], 200)

    def test_running_source_remains_partial(self) -> None:
        result = analyze_report(
            {
                "status": "running",
                "tokens": [
                    {
                        "position": 0,
                        "layers": [
                            _layer(0, [0, 1, 2, 3, 4, 5], [1, 1, 1, 1, 1, 1])
                        ],
                    }
                ],
            }
        )
        self.assertEqual(result["evidence_status"], "partial_live_trace")
        self.assertIn("incomplete_reason", result)
        self.assertIsNone(result["adjacent_route_reuse"]["mean_overlap"])

    def test_k4_window_reports_union_fraction(self) -> None:
        result = analyze_report(
            {
                "status": "complete",
                "tokens": [
                    {
                        "position": position,
                        "layers": [
                            _layer(0, [0, 1, 2, 3, 4, 5], [1, 1, 1, 1, 1, 1])
                        ],
                    }
                    for position in range(4)
                ],
            }
        )
        windows = result["causal_block_route_reuse"]
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["block_size"], 4)
        self.assertEqual(windows[0]["mean_unique_experts_per_layer"], 6)
        self.assertEqual(windows[0]["expert_io_fraction_vs_serial"], 0.25)

    def test_rejects_non_native_or_invalid_routes(self) -> None:
        with self.assertRaises(RouteSpeedProbeError):
            analyze_report(
                {
                    "status": "complete",
                    "tokens": [
                        {
                            "position": 0,
                            "layers": [_layer(0, [0, 1], [1.0, 1.0])],
                        }
                    ],
                }
            )
        with self.assertRaises(RouteSpeedProbeError):
            analyze_report(
                {
                    "status": "complete",
                    "tokens": [
                        {
                            "position": 0,
                            "layers": [
                                _layer(0, [0, 1, 2, 3, 4, 5], [1, 1, 1, 1, 1, -1])
                            ],
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
