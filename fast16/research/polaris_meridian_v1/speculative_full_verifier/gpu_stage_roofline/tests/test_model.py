from __future__ import annotations

from dataclasses import replace
import unittest

from ..anchors import StageAnchors
from ..model import (
    CommandBufferMode,
    SimulationConfig,
    command_buffers_per_token,
    simulate_token,
    solve_required_expert_hit_rate,
)


def fake_anchors() -> StageAnchors:
    return StageAnchors(
        model_revision="test",
        layers=(42,),
        cpu_warm_token_ms=1.0,
        cpu_report_sha256="cpu",
        two_token_report_sha256="two",
        route_catalog_sha256="catalog",
        gpu_evidence_sha256="gpu",
        top6_evidence_sha256="top6",
        device="test gpu",
        single_expert_chain_ms=1.0,
        single_expert_dispatches=5,
        single_expert_iterations=100,
        top6_routed_plus_shared_ms=7.0,
        top6_dispatches=35,
        top6_iterations=1,
        wq_a_linear_ms=1.0,
        wq_a_dispatches=1,
        wq_a_iterations=100,
        expert_page_bytes=1_000,
        routed_bytes_per_layer=6_000,
        shared_bytes_per_layer=2_000,
        attention_weight_bytes_s14=10,
        hc_weight_bytes_s14=20,
        router_weight_bytes_s14=30,
        shared_weight_bytes_s14=40,
        final_norm_head_bytes=50,
        two_token_previous_overlap_pages=1,
        two_token_current_pages=6,
        two_token_new_bytes=5_008,
        two_token_expert_miss_bytes=5_000,
        two_token_intersection_by_layer=((42, 1),),
    )


class ScheduleTests(unittest.TestCase):
    def test_copy_overlaps_only_shared_then_gates_routed(self) -> None:
        anchors = fake_anchors()
        result = simulate_token(
            anchors,
            SimulationConfig(
                target_tps=100.0,
                expert_cache_hit_rate=0.0,
                pcie_bytes_per_second=1_000_000.0,
            ),
        )
        events = {event.name: event for event in result.events}
        self.assertEqual(events["expert_page_ssd_miss"].start_ms, events["shared_expert"].start_ms)
        self.assertEqual(events["six_routed_experts"].start_ms, 7.0)
        self.assertEqual(result.gpu_copy_compute_critical_ms, 13.0)
        self.assertEqual(result.shared_copy_overlap_credit_ms, 1.0)
        self.assertEqual(result.stage_work_ms["ssd_miss"], 6.0)

    def test_required_hit_rate_is_solved_not_assumed(self) -> None:
        anchors = fake_anchors()
        config = SimulationConfig(
            target_tps=100.0,
            expert_cache_hit_rate=0.0,
            pcie_bytes_per_second=1_000_000.0,
        )
        self.assertAlmostEqual(solve_required_expert_hit_rate(anchors, config), 0.5)
        just_below = simulate_token(anchors, replace(config, expert_cache_hit_rate=0.499))
        at_boundary = simulate_token(anchors, replace(config, expert_cache_hit_rate=0.5))
        self.assertFalse(just_below.target_met_under_ideal_pipeline)
        self.assertTrue(at_boundary.target_met_under_ideal_pipeline)

    def test_command_buffer_modes_are_explicit(self) -> None:
        anchors = fake_anchors()
        self.assertEqual(command_buffers_per_token(anchors, CommandBufferMode.PER_MEASURED_GPU_DISPATCH), 36)
        self.assertEqual(command_buffers_per_token(anchors, CommandBufferMode.ROUTE_SPLIT_PER_LAYER), 3)
        self.assertEqual(command_buffers_per_token(anchors, CommandBufferMode.RESIDENT_PER_LAYER), 2)
        self.assertEqual(command_buffers_per_token(anchors, CommandBufferMode.WHOLE_TOKEN_PERSISTENT), 1)

    def test_unknown_stage_costs_reduce_the_roofline(self) -> None:
        anchors = fake_anchors()
        base = SimulationConfig(target_tps=100.0, expert_cache_hit_rate=1.0)
        self.assertTrue(simulate_token(anchors, base).target_met_under_ideal_pipeline)
        costly = replace(base, hc_ms_per_layer=3.0, router_ms_per_layer=1.0)
        self.assertFalse(simulate_token(anchors, costly).target_met_under_ideal_pipeline)

    def test_input_validation_is_fail_closed(self) -> None:
        anchors = fake_anchors()
        with self.assertRaises(ValueError):
            simulate_token(anchors, SimulationConfig(target_tps=20.0, expert_cache_hit_rate=1.01))
        with self.assertRaises(ValueError):
            simulate_token(
                anchors,
                SimulationConfig(target_tps=20.0, expert_cache_hit_rate=0.0, submit_overhead_us=-1.0),
            )


if __name__ == "__main__":
    unittest.main()
