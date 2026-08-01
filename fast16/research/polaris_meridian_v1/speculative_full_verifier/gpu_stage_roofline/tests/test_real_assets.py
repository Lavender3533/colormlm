from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ..anchors import load_real_anchors
from ..cli import main
from ..model import build_roofline_report


ASSET_ROOT = Path("D:/models/Polaris-S14")
HAS_ASSETS = (ASSET_ROOT / "s14_two_real_tokens_report.json").is_file()


@unittest.skipUnless(HAS_ASSETS, "Polaris-S14 元数据未安装")
class RealAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.anchors = load_real_anchors(ASSET_ROOT)
        cls.report = build_roofline_report(ASSET_ROOT)

    def test_real_byte_and_timing_anchors(self) -> None:
        self.assertEqual(self.anchors.layer_count, 14)
        self.assertEqual(self.anchors.expert_page_bytes, 13_369_344)
        self.assertEqual(self.anchors.routed_bytes_per_layer, 80_216_064)
        self.assertAlmostEqual(self.anchors.s14_current_moe_anchor_ms, 19.1744)
        self.assertAlmostEqual(self.anchors.s14_wq_a_floor_ms, 1.1732056)
        self.assertAlmostEqual(self.anchors.s14_current_known_gpu_anchor_ms, 20.3476056)

    def test_two_token_counterexample_is_exact_and_non_generalized(self) -> None:
        cold = self.report["two_token_cold_cache_counterexample"]
        self.assertEqual(cold["previous_token_intersection_pages"], 8)
        self.assertEqual(cold["current_token_pages"], 84)
        self.assertAlmostEqual(cold["observed_previous_only_hit_rate"], 8 / 84)
        self.assertEqual(cold["token1_expert_miss_bytes"], 1_016_070_144)
        self.assertEqual(cold["token1_total_new_bytes"], 1_016_078_336)
        self.assertEqual(cold["20_tps_required_total_hit_pages_out_of_84"], 26)
        self.assertEqual(cold["20_tps_additional_successful_prefetch_or_history_hits_beyond_previous_token"], 18)
        self.assertIn("不是长序列稳态", cold["non_generalization"])

    def test_target_frontiers(self) -> None:
        target20 = self.report["targets"]["20_tps"]
        target50 = self.report["targets"]["50_tps"]
        self.assertAlmostEqual(target20["required_expert_cache_hit_rate_current_kernels"], 0.300948426283132)
        self.assertIsNone(target50["required_expert_cache_hit_rate_current_kernels"])
        self.assertGreater(target50["minimum_measured_kernel_speedup_at_100_percent_hit"], 1.0)
        self.assertEqual(target50["minimum_ideal_batch_size"], 2)

    def test_capacity_dimensions_are_present(self) -> None:
        hardware = self.report["hardware_roof"]
        self.assertEqual(hardware["vram_bytes"], 8 * 1024**3)
        self.assertEqual(hardware["ram_bytes"], 32 * 1024**3)
        self.assertEqual(hardware["ssd_bytes"], 118 * 1024**3)
        self.assertFalse(hardware["s14_profile_fits_ram"])
        self.assertTrue(hardware["s14_profile_fits_ssd"])

    def test_cli_writes_utf8_without_bom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            self.assertEqual(main(["report", "--asset-root", str(ASSET_ROOT), "--output", str(output)]), 0)
            raw = output.read_bytes()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
            parsed = json.loads(raw.decode("utf-8"))
            self.assertEqual(parsed["format"], "polaris-s14-stage-roofline-v1")


if __name__ == "__main__":
    unittest.main()
