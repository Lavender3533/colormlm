#!/usr/bin/env python3
"""证据耦合星座内核的秒级定向门。"""

from __future__ import annotations

import unittest

from evidence_constellation import (
    ConstellationField,
    EvidenceRequirement,
    Proposal,
    make_claim,
)


class ConstellationFieldTests(unittest.TestCase):
    def test_anchor_cannot_be_overwritten(self) -> None:
        field = ConstellationField("anchor")
        field.add_anchor(make_claim("a", "goal", "keep", "user", anchor=True))
        bad = make_claim("bad", "goal", "replace", "model")
        with self.assertRaisesRegex(ValueError, "锚点"):
            field.propose(
                "branch.0",
                Proposal("p", "model", (bad,), 1.0, 1.0, 1.0, 1.0),
            )

    def test_multi_claim_proposal_is_atomic_on_preflight_failure(self) -> None:
        field = ConstellationField("atomic_proposal")
        field.add_anchor(make_claim("a", "goal", "keep", "user", anchor=True))
        good = make_claim("good", "new_slot", "candidate", "model")
        bad = make_claim("bad", "goal", "replace", "model")
        before = field.snapshot()
        with self.assertRaisesRegex(ValueError, "锚点"):
            field.propose(
                "branch.0",
                Proposal("p", "model", (good, bad), 1.0, 1.0, 1.0, 1.0),
            )
        self.assertNotIn("good", field.claims)
        self.assertNotIn("p", field.proposals)
        self.assertEqual(field.branches["branch.0"].assignments, before["branches"]["branch.0"]["assignments"])

    def test_conflict_branches_and_disjoint_patch_merges(self) -> None:
        field = ConstellationField("branch")
        left = make_claim("left", "choice", "A", "organ")
        right = make_claim("right", "choice", "B", "organ")
        shared = make_claim("shared", "shared", 1, "organ")
        field.propose(
            "branch.0", Proposal("p1", "organ", (left,), 1.0, 1.0, 1.0, 1.0)
        )
        receipt = field.propose(
            "branch.0", Proposal("p2", "organ", (right,), 1.0, 1.0, 1.0, 1.0)
        )
        field.propose(
            "branch.0", Proposal("p3", "organ", (shared,), 1.0, 1.0, 1.0, 1.0)
        )
        self.assertEqual(receipt.action, "branch")
        self.assertEqual(len(field.branches), 2)
        self.assertEqual(field.branches["branch.0"].assignments["shared"], "shared")

    def test_unsealed_round_cannot_commit(self) -> None:
        field = ConstellationField("seal")
        claim = make_claim("x", "x", 1, "organ")
        field.propose(
            "branch.0", Proposal("p", "organ", (claim,), 1.0, 1.0, 1.0, 1.0)
        )
        with self.assertRaisesRegex(RuntimeError, "seal"):
            field.partial_commit()

    def test_model_proposal_without_evidence_stays_pending(self) -> None:
        field = ConstellationField("no_vote")
        claim = make_claim(
            "x",
            "answer",
            1,
            "model",
            requirements=(EvidenceRequirement("probe", True),),
        )
        field.propose(
            "branch.0", Proposal("p", "model", (claim,), 10.0, 10.0, 10.0, 0.0)
        )
        field.seal_proposals()
        field.partial_commit()
        self.assertEqual(field.assess_claim("x").status, "pending")
        self.assertNotIn("answer", field.committed)

    def test_latest_receipt_from_same_source_supersedes_old(self) -> None:
        field = ConstellationField("evidence")
        field.add_anchor(make_claim("a", "goal", 1, "user", anchor=True))
        claim = make_claim(
            "x",
            "answer",
            True,
            "organ",
            requirements=(EvidenceRequirement("probe", True),),
            depends_on=("a",),
        )
        field.propose(
            "branch.0", Proposal("p", "organ", (claim,), 1.0, 1.0, 1.0, 1.0)
        )
        field.seal_proposals()
        field.attach_evidence(
            receipt_id="r1",
            requirement_id="probe",
            observed=True,
            source_group="tool",
        )
        field.partial_commit()
        self.assertIn("answer", field.committed)
        receipt = field.attach_evidence(
            receipt_id="r2",
            requirement_id="probe",
            observed=False,
            source_group="tool",
        )
        self.assertNotIn("answer", field.committed)
        self.assertIn("x", receipt.detail["rolled_back_claims"])
        self.assertEqual(field.assess_claim("x").status, "contradicted")


if __name__ == "__main__":
    unittest.main()
