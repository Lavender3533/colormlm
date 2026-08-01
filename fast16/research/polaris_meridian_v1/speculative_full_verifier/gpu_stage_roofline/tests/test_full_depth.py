from __future__ import annotations

from pathlib import Path
import unittest

from ..anchors import load_real_anchors
from ..full_depth import audit_full_depth_projection_shapes, build_full_depth_report


ASSET_ROOT = Path("D:/models/Polaris-S14")
HAS_ASSETS = (ASSET_ROOT / "fulldepth_kadaptive_budget.json").is_file()


@unittest.skipUnless(HAS_ASSETS, "Polaris-S14 FullDepth 元数据未安装")
class FullDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        anchors = load_real_anchors(ASSET_ROOT)
        cls.report = build_full_depth_report(ASSET_ROOT, anchors)

    def test_all_45_headers_cover_homogeneous_projected_shapes(self) -> None:
        shapes = audit_full_depth_projection_shapes(ASSET_ROOT)
        self.assertEqual(shapes.header_count, 45)
        self.assertEqual(shapes.wq_a_tensor_count, 43 * 2)
        self.assertEqual(shapes.routed_tensor_count, 43 * 256 * 6)
        self.assertEqual(shapes.shared_tensor_count, 43 * 6)
        self.assertEqual(shapes.router_tensor_count, 43 + 40)

    def test_full_depth_is_the_only_commit_authority(self) -> None:
        contract = self.report["commit_contract"]
        self.assertTrue(contract["all_final_tokens_require_full_depth"])
        self.assertEqual(contract["layers_executed_per_verified_position"], 43)
        self.assertFalse(contract["layer_skipping_allowed"])
        self.assertEqual(self.report["draft_only_profiles"], ["S14", "v38", "v47"])

    def test_projected_known_work_is_not_labeled_full_measurement(self) -> None:
        projected = self.report["projected_known_gpu_work_per_verified_position"]
        self.assertAlmostEqual(projected["stages_ms"]["attention"], 43 * 0.0838004)
        self.assertAlmostEqual(projected["total_ms"], 62.4962172)
        self.assertIn("不是 FullDepth43 token 实测", projected["provenance"])
        self.assertIn("完整 attention", projected["missing_work"])

    def test_K1_K4_K8_each_have_unassumed_20_and_50_frontiers(self) -> None:
        for block_size in (1, 4, 8):
            block = self.report["blocks"][f"K{block_size}"]
            self.assertEqual(block["block_size_K"], block_size)
            for target in (20, 50):
                row = block["targets"][f"{target}_tps"]
                self.assertTrue(row["acceptance_is_not_assumed"])
                self.assertEqual(len(row["frontier"]), block_size + 1)
                self.assertTrue(all(item["all_draft_positions_execute_all_43_layers"] for item in row["frontier"]))

    def test_streamed_fixed_scan_hard_impossibilities(self) -> None:
        for block_size, target in ((1, 20), (1, 50), (4, 20), (4, 50), (8, 50)):
            full_match = self.report["blocks"][f"K{block_size}"]["targets"][f"{target}_tps"]["frontier"][-1]
            cases = full_match["io_requirements"]["stream_fixed_each_block"]["cases"]
            self.assertTrue(all(case["required_device_expert_cache_hit_rate"] is None for case in cases.values()))

    def test_K8_20_streamed_full_match_exposes_route_reuse_cache_range(self) -> None:
        full_match = self.report["blocks"]["K8"]["targets"]["20_tps"]["frontier"][-1]
        cases = full_match["io_requirements"]["stream_fixed_each_block"]["cases"]
        self.assertAlmostEqual(
            cases["perfect_within_block_route_reuse"]["required_device_expert_cache_hit_rate"],
            0.7027294009919405,
        )
        self.assertAlmostEqual(
            cases["no_within_block_route_reuse"]["required_device_expert_cache_hit_rate"],
            0.9628411751239926,
        )

    def test_K1_is_capacity_impossible_even_with_optimal_weight_residency(self) -> None:
        for target in (20, 50):
            full_match = self.report["blocks"]["K1"]["targets"][f"{target}_tps"]["frontier"][-1]
            cases = full_match["io_requirements"]["optimistic_any_weight_residency_lower_bound"]["cases"]
            self.assertTrue(all(not case["fits_8gib_before_runtime"] for case in cases.values()))

    def test_full_accept_compute_speedup_requirements_do_not_assume_batch(self) -> None:
        for block_size in (1, 4, 8):
            target20 = self.report["blocks"][f"K{block_size}"]["targets"]["20_tps"]["frontier"][-1]
            target50 = self.report["blocks"][f"K{block_size}"]["targets"]["50_tps"]["frontier"][-1]
            self.assertAlmostEqual(target20["minimum_uniform_speedup_of_projected_known_gpu_kernels"], 1.249924344)
            self.assertAlmostEqual(target50["minimum_uniform_speedup_of_projected_known_gpu_kernels"], 3.12481086)
        self.assertFalse(
            self.report["blocks"]["K1"]["targets"]["20_tps"]["frontier"][-1][
                "possible_from_ideal_K_way_batching_alone_for_projected_known_kernels"
            ]
        )


if __name__ == "__main__":
    unittest.main()
