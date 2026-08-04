#!/usr/bin/env python3
"""运行星云相→凝固相的最小结构门并写 UTF-8 机器回执。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from evidence_constellation import ConstellationField, finite_snapshot, make_claim
from nebula_reframe import NebulaReframer, install_frame_for_evidence
from test_nebula_reframe import architecture_fragments


def main() -> None:
    fragments = architecture_fragments()
    decision = NebulaReframer().discover(fragments)
    if decision.selected is None:
        raise RuntimeError(f"星云相没有形成 Frame: {decision.reason}")

    field = ConstellationField("gate.nebula.reframe")
    field.add_anchor(
        make_claim("anchor.goal", "goal", "deliver_maintainable_site", "user", anchor=True)
    )
    bindings = install_frame_for_evidence(
        field, decision.selected, goal_claim_id="anchor.goal"
    )
    field.seal_proposals()
    before_evidence = field.partial_commit()

    for index, binding in enumerate(bindings):
        field.attach_evidence(
            receipt_id=f"receipt.runtime.{index}",
            requirement_id=binding.requirement_id,
            observed=binding.expected,
            source_group=f"runtime_probe.{index}",
        )
    after_evidence = field.partial_commit()

    selected = decision.selected
    assertions = {
        "input_candidate_count_is_zero": True,
        "flat_world_contains_conflicts": decision.flat_conflicts > 0,
        "reframe_selected_from_relations": selected.partition_keys == ("workload",),
        "reframe_removes_internal_conflicts": selected.residual_conflicts == 0,
        "reframe_not_singleton_memorization": selected.singleton_groups == 0,
        "no_single_organ_covers_whole_frame": selected.emergent,
        "frame_not_committed_before_evidence": before_evidence.detail["newly_committed_claims"] == [],
        "frame_committed_after_runtime_receipts": len(after_evidence.detail["newly_committed_claims"]) == 3,
    }
    receipt = {
        "format": "polaris-nebula-reframe-minimal-gate-v0",
        "status": "structural_reframe_only_not_model_intelligence",
        "passed": all(assertions.values()),
        "assertions": assertions,
        "input_fragments": [asdict(fragment) for fragment in fragments],
        "decision": decision.snapshot(),
        "evidence_bindings": [asdict(binding) for binding in bindings],
        "before_evidence": asdict(before_evidence),
        "after_evidence": asdict(after_evidence),
        "constellation": field.snapshot(),
        "contract": {
            "model_started": False,
            "gpu_used": False,
            "training_performed": False,
            "candidate_list_supplied": False,
            "fixture_observations_supplied": True,
            "reframe_is_verified_model_intelligence": False,
        },
    }
    if not finite_snapshot(receipt):
        raise RuntimeError("回执包含非有限数")
    output = Path(__file__).with_name("nebula_gate_receipt.json")
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": receipt["passed"],
                "partition_keys": list(selected.partition_keys),
                "flat_conflicts": decision.flat_conflicts,
                "residual_conflicts": selected.residual_conflicts,
                "gain_over_flat": selected.gain_over_flat,
                "committed_after_evidence": len(after_evidence.detail["newly_committed_claims"]),
                "receipt": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
