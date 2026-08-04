#!/usr/bin/env python3
"""星云 Reframe 的秒级定向门。"""

from __future__ import annotations

import unittest

from evidence_constellation import ConstellationField, make_claim
from nebula_reframe import NebulaReframer, install_frame_for_evidence, make_fragment


def architecture_fragments():
    # 每个器官只报告一个局部切面；没有任何输入含“按 workload 分区”的完整方案。
    return (
        make_fragment(
            "request.code", "code_organ",
            context={"workload": "request", "deployment": "shared_host"},
            assertions={"runtime": "php"},
        ),
        make_fragment(
            "request.env", "environment_organ",
            context={"workload": "request", "deployment": "managed"},
            assertions={"runtime": "php"},
        ),
        make_fragment(
            "batch.code", "code_organ",
            context={"workload": "batch", "deployment": "managed"},
            assertions={"runtime": "jvm"},
        ),
        make_fragment(
            "batch.ops", "operations_organ",
            context={"workload": "batch", "deployment": "shared_host"},
            assertions={"runtime": "jvm"},
        ),
        make_fragment(
            "interaction.ui", "ui_organ",
            context={"workload": "interaction", "deployment": "browser"},
            assertions={"runtime": "javascript"},
        ),
        make_fragment(
            "interaction.product", "product_organ",
            context={"workload": "interaction", "deployment": "edge"},
            assertions={"runtime": "javascript"},
        ),
    )


class NebulaReframeTests(unittest.TestCase):
    def test_discovers_new_problem_partition_without_candidate_list(self) -> None:
        decision = NebulaReframer().discover(architecture_fragments())
        self.assertIsNotNone(decision.selected)
        frame = decision.selected
        assert frame is not None
        self.assertEqual(frame.partition_keys, ("workload",))
        self.assertEqual(decision.flat_conflicts, 4)
        self.assertEqual(frame.residual_conflicts, 0)
        self.assertEqual(frame.singleton_groups, 0)
        self.assertTrue(frame.emergent)
        self.assertGreater(frame.gain_over_flat, 0.0)

    def test_coordinate_names_do_not_control_the_answer(self) -> None:
        renamed = tuple(
            make_fragment(
                fragment.fragment_id,
                fragment.organ_id,
                context={
                    "axis_z": fragment.context_map["workload"],
                    "axis_a": fragment.context_map["deployment"],
                },
                assertions=fragment.assertion_map,
            )
            for fragment in architecture_fragments()
        )
        decision = NebulaReframer().discover(renamed)
        self.assertIsNotNone(decision.selected)
        assert decision.selected is not None
        self.assertEqual(decision.selected.partition_keys, ("axis_z",))

    def test_relations_can_select_a_different_coordinate(self) -> None:
        fragments = (
            make_fragment("a", "o1", context={"workload": "w1", "deployment": "d1"}, assertions={"runtime": "x"}),
            make_fragment("b", "o2", context={"workload": "w2", "deployment": "d1"}, assertions={"runtime": "x"}),
            make_fragment("c", "o3", context={"workload": "w1", "deployment": "d2"}, assertions={"runtime": "y"}),
            make_fragment("d", "o4", context={"workload": "w2", "deployment": "d2"}, assertions={"runtime": "y"}),
        )
        decision = NebulaReframer().discover(fragments)
        self.assertIsNotNone(decision.selected)
        assert decision.selected is not None
        self.assertEqual(decision.selected.partition_keys, ("deployment",))

    def test_rejects_singleton_memorization(self) -> None:
        fragments = tuple(
            make_fragment(
                f"f{index}", f"o{index}",
                context={"identity": f"only_{index}"},
                assertions={"runtime": value},
            )
            for index, value in enumerate(("a", "b", "c"))
        )
        decision = NebulaReframer().discover(fragments)
        self.assertIsNone(decision.selected)
        self.assertIn("复杂度", decision.reason)

    def test_rejects_reframe_when_there_is_no_conflict(self) -> None:
        fragments = (
            make_fragment("a", "o1", context={"zone": "left"}, assertions={"runtime": "same"}),
            make_fragment("b", "o2", context={"zone": "right"}, assertions={"runtime": "same"}),
        )
        decision = NebulaReframer().discover(fragments)
        self.assertIsNone(decision.selected)
        self.assertIn("无需", decision.reason)

    def test_frame_must_pass_existing_evidence_phase(self) -> None:
        decision = NebulaReframer().discover(architecture_fragments())
        assert decision.selected is not None
        field = ConstellationField("nebula.evidence.bridge")
        field.add_anchor(make_claim("anchor.goal", "goal", "deliver_site", "user", anchor=True))
        bindings = install_frame_for_evidence(
            field, decision.selected, goal_claim_id="anchor.goal"
        )
        field.seal_proposals()

        before = field.partial_commit()
        self.assertEqual(before.detail["newly_committed_claims"], [])
        self.assertEqual(len(field.committed), 1)

        for index, binding in enumerate(bindings):
            field.attach_evidence(
                receipt_id=f"runtime.{index}",
                requirement_id=binding.requirement_id,
                observed=binding.expected,
                source_group=f"runtime_probe.{index}",
            )
        after = field.partial_commit()
        self.assertEqual(len(after.detail["newly_committed_claims"]), 3)
        self.assertEqual(len(field.committed), 4)


if __name__ == "__main__":
    unittest.main()
