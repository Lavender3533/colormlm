from __future__ import annotations

from pathlib import Path
import unittest

from ..assets import audit_assets
from ..cost_model import build_analysis_report
from ..verifier import DeepSeekTokenizer


ASSET_ROOT = Path("D:/models/Polaris-S14")


@unittest.skipUnless((ASSET_ROOT / "fulldepth_kadaptive_budget.json").is_file(), "Polaris-S14 metadata not installed")
class RealAssetIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = audit_assets(ASSET_ROOT)

    def test_all_real_byte_sources_cross_check(self):
        self.assertEqual(self.audit.header_count, 45)
        self.assertEqual(self.audit.tensor_count, 67612)
        self.assertEqual(self.audit.full_base_shard_bytes, 156023192948)
        self.assertEqual(self.audit.full_base_payload_bytes, 156015698140)
        self.assertEqual(self.audit.shard_container_overhead_bytes, 7494808)
        self.assertEqual(self.audit.full_non_routed_bytes, 6727565512)
        self.assertEqual(self.audit.head_bytes, 1059061760)
        self.assertEqual(self.audit.expert_page_count, 43 * 256)
        self.assertEqual(self.audit.expert_page_bytes, 13369344)
        self.assertEqual(self.audit.route_catalog_range_count, 22013)
        self.assertEqual(self.audit.route_catalog_range_bytes, 52228836324)
        self.assertFalse(self.audit.weights_downloaded)
        self.assertTrue(self.audit.budget_vram_fit_nonrouted_head_one_layer_top6)

    def test_deepseek_tokenizer_produces_exact_n_token_draft(self):
        tokenizer = DeepSeekTokenizer(ASSET_ROOT)
        token_ids = tokenizer.encode_exact_draft("北极星 Polaris 全深度原生验证。" * 8, 16)
        self.assertEqual(len(token_ids), 16)
        self.assertTrue(all(0 <= token < 129280 for token in token_ids))
        self.assertEqual(tokenizer.fingerprint, self.audit.tokenizer_sha256)

    def test_report_refuses_capacity_and_quality_overclaim(self):
        report = build_analysis_report(ASSET_ROOT)
        self.assertFalse(report["capacity"]["full_base_fits_ssd"])
        conclusion_ids = {row["id"] for row in report["hard_conclusions"]}
        self.assertIn("50_tps_pcie_bound", conclusion_ids)
        self.assertIn("quality_not_measured", conclusion_ids)


if __name__ == "__main__":
    unittest.main()
