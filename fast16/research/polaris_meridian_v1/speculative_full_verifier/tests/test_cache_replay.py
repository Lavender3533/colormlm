from __future__ import annotations

import unittest

from ..cache_replay import ExpertPageCache, ReplayContractError, RouteBlock, replay_blocks


PAGE_BYTES = 13369344


def rows(block_size: int, *, diverse: bool = False):
    result = []
    for token in range(block_size):
        for layer in range(43):
            start = token * 6 if diverse else 0
            result.append({"token_offset": token, "layer": layer, "expert_ids": list(range(start, start + 6))})
    return result


class CacheReplayTest(unittest.TestCase):
    def test_block_dedup_is_separate_from_warm_cache_hits(self):
        first = RouteBlock.from_rows("first", 4, rows(4), accepted_prefix_length=3)
        second = RouteBlock.from_rows("second", 4, rows(4), accepted_prefix_length=4)
        report = replay_blocks((first, second), ExpertPageCache(258), PAGE_BYTES)

        self.assertEqual(report.blocks[0].selected_page_references, 1032)
        self.assertEqual(report.blocks[0].unique_pages_after_block_dedup, 258)
        self.assertEqual(report.blocks[0].block_deduplicated_references, 774)
        self.assertEqual(report.blocks[0].cache_hits, 0)
        self.assertEqual(report.blocks[0].cache_misses, 258)
        self.assertEqual(report.blocks[1].cache_hits, 258)
        self.assertEqual(report.blocks[1].cache_misses, 0)
        self.assertEqual(report.pcie_expert_bytes, 258 * PAGE_BYTES)

    def test_no_cross_token_reuse_reaches_upper_bound(self):
        block = RouteBlock.from_rows("diverse", 2, rows(2, diverse=True))
        report = replay_blocks((block,), ExpertPageCache(0), PAGE_BYTES)
        self.assertEqual(report.unique_page_requests, 516)
        self.assertEqual(report.block_deduplicated_references, 0)
        self.assertEqual(report.cache_misses, 516)

    def test_small_global_lru_exposes_scan_thrashing(self):
        block_a = RouteBlock.from_rows("a", 1, rows(1))
        block_b = RouteBlock.from_rows("b", 1, rows(1))
        report = replay_blocks((block_a, block_b), ExpertPageCache(60), PAGE_BYTES)
        self.assertEqual(report.blocks[1].cache_hits, 0)

    def test_incomplete_or_duplicate_routes_are_rejected(self):
        incomplete = rows(1)[:-1]
        with self.assertRaises(ReplayContractError):
            RouteBlock.from_rows("bad", 1, incomplete)
        duplicate = rows(1)
        duplicate.append(duplicate[0])
        with self.assertRaises(ReplayContractError):
            RouteBlock.from_rows("bad", 1, duplicate)


if __name__ == "__main__":
    unittest.main()
