from __future__ import annotations

import unittest

from ..assets import AssetAudit
from ..cost_model import (
    HardwareBudget,
    committed_tokens,
    estimate_block,
    expert_page_bounds,
    throughput_requirement,
)


def frozen_audit() -> AssetAudit:
    return AssetAudit(
        asset_root="fixture",
        budget_sha256="b",
        route_catalog_sha256="c",
        tokenizer_sha256="t",
        header_count=45,
        header_set_sha256="h",
        tensor_count=67612,
        full_base_shard_bytes=156023192948,
        full_base_payload_bytes=156015698140,
        shard_container_overhead_bytes=7494808,
        full_non_routed_bytes=6727565512,
        head_bytes=1059061760,
        embedding_bytes=1059061760,
        other_boundary_bytes=270356,
        expert_page_bytes=13369344,
        expert_page_count=11008,
        expert_bank_bytes=147169738752,
        route_catalog_range_count=22010,
        route_catalog_range_bytes=52228574160,
        route_catalog_selected_layers=(0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42),
        weights_downloaded=False,
        budget_vram_fit_nonrouted_head_one_layer_top6=True,
    )


class CostModelTest(unittest.TestCase):
    def setUp(self):
        self.audit = frozen_audit()
        self.hardware = HardwareBudget()

    def test_expert_dedup_bounds(self):
        self.assertEqual(expert_page_bounds(1), (258, 258))
        self.assertEqual(expert_page_bounds(16), (258, 4128))
        estimate = estimate_block(self.audit, self.hardware, 16)
        self.assertEqual(estimate["expert_bytes_after_block_dedup"]["perfect_cross_token_reuse"], 3449290752)
        self.assertEqual(estimate["expert_bytes_after_block_dedup"]["no_cross_token_reuse"], 55188652032)
        self.assertEqual(estimate["non_routed_amortized_bytes_per_draft_position"], 420472844.5)

    def test_fallback_counts_as_one_native_committed_token(self):
        self.assertEqual(committed_tokens(8, 0), 1)
        self.assertEqual(committed_tokens(8, 7), 8)
        self.assertEqual(committed_tokens(8, 8), 8)

    def test_20_tps_requires_block_8_and_seven_accepted_at_minimum(self):
        for block in (1, 2, 4):
            self.assertFalse(throughput_requirement(self.audit, self.hardware, block, 20)["possible_under_pcie_bound"])
        requirement = throughput_requirement(self.audit, self.hardware, 8, 20)
        self.assertTrue(requirement["possible_under_pcie_bound"])
        self.assertEqual(requirement["minimum_accepted_prefix_with_perfect_expert_cache"], 7)
        first = requirement["frontier"][0]
        self.assertAlmostEqual(
            first["required_device_expert_cache_hit_rate"]["perfect_cross_token_reuse"],
            0.7027294009919405,
        )
        self.assertAlmostEqual(
            first["required_device_expert_cache_hit_rate"]["no_cross_token_reuse"],
            0.9628411751239926,
        )

    def test_block_16_full_accept_20_tps_cache_tradeoff(self):
        requirement = throughput_requirement(self.audit, self.hardware, 16, 20)
        full_match = requirement["frontier"][-1]
        self.assertEqual(full_match["accepted_prefix_length"], 16)
        self.assertEqual(full_match["required_device_expert_cache_hit_rate"]["perfect_cross_token_reuse"], 0.0)
        self.assertAlmostEqual(
            full_match["required_device_expert_cache_hit_rate"]["no_cross_token_reuse"],
            0.8217500814787793,
        )

    def test_50_tps_impossible_through_block_16_even_with_perfect_cache(self):
        for block in (1, 2, 4, 8, 16):
            requirement = throughput_requirement(self.audit, self.hardware, block, 50)
            self.assertFalse(requirement["possible_under_pcie_bound"])
            self.assertEqual(requirement["frontier"], [])

    def test_resident_fixed_branch_exposes_50_tps_conditions(self):
        requirement = throughput_requirement(
            self.audit,
            self.hardware,
            8,
            50,
            fixed_scan_policy="resident_after_warmup",
        )
        self.assertTrue(requirement["possible_under_pcie_bound"])
        self.assertEqual(requirement["minimum_accepted_prefix_with_zero_device_expert_cache"]["perfect_cross_token_reuse"], 7)
        full = requirement["frontier"][-1]
        self.assertEqual(full["required_device_expert_cache_hit_rate"]["perfect_cross_token_reuse"], 0.0)
        self.assertAlmostEqual(full["required_device_expert_cache_hit_rate"]["no_cross_token_reuse"], 0.8722635951334264)


if __name__ == "__main__":
    unittest.main()
