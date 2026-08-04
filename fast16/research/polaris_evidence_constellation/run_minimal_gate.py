#!/usr/bin/env python3
"""运行证据耦合星座动力学的三个冻结场景。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from evidence_constellation import (
    ConstellationField,
    EvidenceRequirement,
    Proposal,
    finite_snapshot,
    make_claim,
)


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _proposal(
    proposal_id: str,
    organ_id: str,
    claims: tuple,
    *,
    alignment: float = 1.0,
    novelty: float = 0.5,
    gain: float = 1.0,
    cost: float = 1.0,
) -> Proposal:
    return Proposal(
        proposal_id=proposal_id,
        organ_id=organ_id,
        claims=claims,
        goal_alignment=alignment,
        novelty=novelty,
        expected_gain=gain,
        physical_cost=cost,
    )


def _evidence(
    field: ConstellationField,
    requirement_id: str,
    observed: Any,
    source_group: str,
    suffix: str,
) -> None:
    field.attach_evidence(
        receipt_id=f"receipt.{suffix}",
        requirement_id=requirement_id,
        observed=observed,
        source_group=source_group,
    )


def run_website_branching() -> dict[str, Any]:
    field = ConstellationField("scenario.website.java_php")
    field.add_anchor(
        make_claim(
            "anchor.website_goal",
            "goal",
            "build_responsive_web_application",
            "user",
            anchor=True,
        )
    )

    layout = make_claim(
        "claim.shared_layout",
        "design.layout",
        "responsive_dashboard",
        "design_organ",
        requirements=(EvidenceRequirement("user.layout_accepted", True),),
        depends_on=("anchor.website_goal",),
    )
    api_contract = make_claim(
        "claim.shared_api_contract",
        "design.api_contract",
        "items_v1",
        "design_organ",
        requirements=(EvidenceRequirement("schema.items_v1_valid", True),),
        depends_on=("anchor.website_goal",),
    )
    field.propose(
        "branch.0",
        _proposal("proposal.shared_design", "design_organ", (layout, api_contract), cost=0.2),
    )

    java = make_claim(
        "claim.backend_java",
        "backend.stack",
        "Java/Spring",
        "code_island",
        requirements=(
            EvidenceRequirement("env.pom_xml", True),
            EvidenceRequirement("env.jvm_runtime", True),
        ),
        depends_on=("claim.shared_api_contract",),
    )
    php = make_claim(
        "claim.backend_php",
        "backend.stack",
        "PHP/Laravel",
        "code_island",
        requirements=(
            EvidenceRequirement("env.composer_json", True),
            EvidenceRequirement("env.php_runtime", True),
            EvidenceRequirement("env.composer_install", True),
        ),
        depends_on=("claim.shared_api_contract",),
    )
    field.propose(
        "branch.0",
        _proposal("proposal.backend_java", "code_island", (java,), cost=2.0),
    )
    branch_receipt = field.propose(
        "branch.0",
        _proposal("proposal.backend_php", "code_island", (php,), cost=1.6),
    )
    field.seal_proposals()

    _evidence(field, "user.layout_accepted", True, "user_contract", "layout")
    _evidence(field, "schema.items_v1_valid", True, "schema_validator", "schema")
    common_commit = field.partial_commit()
    common_committed = {
        "design.layout",
        "design.api_contract",
    }.issubset(field.committed) and "backend.stack" not in field.committed

    _evidence(field, "env.pom_xml", True, "filesystem", "pom")
    _evidence(field, "env.jvm_runtime", False, "runtime_probe", "jvm_missing")
    _evidence(field, "env.composer_json", True, "filesystem", "composer_json")
    _evidence(field, "env.php_runtime", True, "runtime_probe", "php_runtime")
    _evidence(field, "env.composer_install", True, "composer_tool", "composer_ok")
    php_commit = field.partial_commit()
    php_committed = field.committed.get("backend.stack") == "claim.backend_php"

    rollback = field.attach_evidence(
        receipt_id="receipt.composer_failed",
        requirement_id="env.composer_install",
        observed=False,
        source_group="composer_tool",
    )
    blocked_after_failure = field.partial_commit()
    local_rollback = (
        "claim.backend_php" in rollback.detail["rolled_back_claims"]
        and "backend.stack" not in field.committed
        and field.committed.get("design.layout") == "claim.shared_layout"
        and field.committed.get("design.api_contract") == "claim.shared_api_contract"
        and blocked_after_failure.detail["blocked"]
    )

    _evidence(field, "env.jvm_runtime", True, "runtime_probe", "jvm_provisioned")
    java_commit = field.partial_commit()
    java_committed = field.committed.get("backend.stack") == "claim.backend_java"

    assertions = {
        "java_php_conflict_created_two_branches": (
            branch_receipt.action == "branch" and len(field.branches) == 2
        ),
        "common_design_committed_before_backend_choice": common_committed,
        "php_committed_only_after_complete_evidence": php_committed,
        "composer_failure_rolled_back_only_php": local_rollback,
        "java_branch_revived_after_new_evidence": java_committed,
        "same_source_receipt_superseded_not_double_counted": (
            field.assess_claim("claim.backend_php").status == "contradicted"
        ),
    }
    return {
        "scenario": field.field_id,
        "passed": all(assertions.values()),
        "assertions": assertions,
        "checkpoints": {
            "branch_birth": branch_receipt.detail,
            "common_commit": common_commit.detail,
            "php_commit": php_commit.detail,
            "rollback": rollback.detail,
            "java_commit": java_commit.detail,
        },
        "snapshot": field.snapshot(),
    }


def run_visual_synthesis() -> dict[str, Any]:
    field = ConstellationField("scenario.visual.responsive_synthesis")
    field.add_anchor(
        make_claim(
            "anchor.screenshot",
            "input.screenshot",
            "sha256:fixture-responsive-dashboard",
            "user",
            anchor=True,
        )
    )

    data_contract = make_claim(
        "claim.visual_data_contract",
        "ui.data_contract",
        "50_rows_with_warning_state",
        "vision_island",
        requirements=(EvidenceRequirement("vision.fields_measured", True),),
        depends_on=("anchor.screenshot",),
    )
    interaction_contract = make_claim(
        "claim.interaction_contract",
        "ui.interactions",
        "filter_detail_alert",
        "frontend_organ",
        requirements=(EvidenceRequirement("browser.interactions_required", True),),
        depends_on=("anchor.screenshot",),
    )
    field.propose(
        "branch.0",
        _proposal(
            "proposal.visual_common",
            "vision_island",
            (data_contract,),
            cost=1.0,
        ),
    )
    field.propose(
        "branch.0",
        _proposal(
            "proposal.interaction_common",
            "frontend_organ",
            (interaction_contract,),
            cost=0.8,
        ),
    )

    table = make_claim(
        "claim.layout_all_table",
        "ui.layout_strategy",
        "table_on_all_viewports",
        "frontend_organ",
        requirements=(
            EvidenceRequirement("browser.desktop_table_ok", True),
            EvidenceRequirement("browser.mobile_table_ok", True),
        ),
        depends_on=("claim.visual_data_contract",),
    )
    cards = make_claim(
        "claim.layout_all_cards",
        "ui.layout_strategy",
        "cards_on_all_viewports",
        "frontend_organ",
        requirements=(
            EvidenceRequirement("browser.desktop_cards_ok", True),
            EvidenceRequirement("browser.mobile_cards_ok", True),
        ),
        depends_on=("claim.visual_data_contract",),
    )
    hybrid = make_claim(
        "claim.layout_responsive_hybrid",
        "ui.layout_strategy",
        "desktop_table_mobile_cards",
        "composition_organ",
        requirements=(
            EvidenceRequirement("browser.desktop_table_ok", True),
            EvidenceRequirement("browser.mobile_cards_ok", True),
        ),
        depends_on=("claim.visual_data_contract", "claim.interaction_contract"),
    )
    field.propose(
        "branch.0",
        _proposal("proposal.layout_table", "frontend_organ", (table,), cost=1.0),
    )
    cards_branch = field.propose(
        "branch.0",
        _proposal("proposal.layout_cards", "frontend_organ", (cards,), cost=0.9),
    )
    hybrid_branch = field.propose(
        "branch.0",
        _proposal("proposal.layout_hybrid", "composition_organ", (hybrid,), cost=1.2),
    )
    field.seal_proposals()

    _evidence(field, "vision.fields_measured", True, "vision_measurement", "visual_fields")
    _evidence(
        field,
        "browser.interactions_required",
        True,
        "browser_trace",
        "interactions",
    )
    early_commit = field.partial_commit()
    common_before_layout = (
        field.committed.get("ui.data_contract") == "claim.visual_data_contract"
        and field.committed.get("ui.interactions") == "claim.interaction_contract"
        and "ui.layout_strategy" not in field.committed
    )

    _evidence(field, "browser.desktop_table_ok", True, "browser_desktop", "desktop_table")
    _evidence(field, "browser.mobile_table_ok", False, "browser_mobile", "mobile_table")
    _evidence(field, "browser.desktop_cards_ok", False, "browser_desktop", "desktop_cards")
    _evidence(field, "browser.mobile_cards_ok", True, "browser_mobile", "mobile_cards")
    final_commit = field.partial_commit()
    assessments = {
        claim_id: field.assess_claim(claim_id).status
        for claim_id in (
            "claim.layout_all_table",
            "claim.layout_all_cards",
            "claim.layout_responsive_hybrid",
        )
    }
    assertions = {
        "three_layout_worlds_existed_without_probability_averaging": (
            len(field.branches) == 3
            and cards_branch.action == "branch"
            and hybrid_branch.action == "branch"
        ),
        "shared_visual_objects_committed_before_layout": common_before_layout,
        "desktop_and_mobile_receipts_rejected_global_layouts": (
            assessments["claim.layout_all_table"] == "contradicted"
            and assessments["claim.layout_all_cards"] == "contradicted"
        ),
        "synthesis_was_a_third_branch_not_an_average": (
            assessments["claim.layout_responsive_hybrid"] == "certified"
            and field.committed.get("ui.layout_strategy")
            == "claim.layout_responsive_hybrid"
        ),
        "only_hybrid_branch_remained_viable": len(field.viable_branch_ids()) == 1,
    }
    return {
        "scenario": field.field_id,
        "passed": all(assertions.values()),
        "assertions": assertions,
        "checkpoints": {
            "early_commit": early_commit.detail,
            "final_commit": final_commit.detail,
            "layout_assessments": assessments,
        },
        "snapshot": field.snapshot(),
    }


def run_fact_conflict() -> dict[str, Any]:
    field = ConstellationField("scenario.fact.conflicting_models")
    field.add_anchor(
        make_claim(
            "anchor.api_question",
            "query",
            "Does API X exist?",
            "user",
            anchor=True,
        )
    )
    exists = make_claim(
        "claim.api_exists",
        "answer.api_x",
        True,
        "large_model_a",
        requirements=(EvidenceRequirement("probe.api_x_exists", True),),
        depends_on=("anchor.api_question",),
    )
    missing = make_claim(
        "claim.api_missing",
        "answer.api_x",
        False,
        "large_model_b",
        requirements=(EvidenceRequirement("probe.api_x_exists", False),),
        depends_on=("anchor.api_question",),
    )
    proposal_a = _proposal(
        "proposal.model_a",
        "large_model_a",
        (exists,),
        alignment=1.0,
        novelty=0.8,
        gain=2.0,
        cost=4.0,
    )
    proposal_b = _proposal(
        "proposal.model_b",
        "large_model_b",
        (missing,),
        alignment=1.0,
        novelty=0.7,
        gain=1.8,
        cost=2.0,
    )
    field.propose("branch.0", proposal_a)
    conflict = field.propose("branch.0", proposal_b)
    # 同一神经器官再说一次仍只是提议，不会生成证据。
    field.propose(
        "branch.0",
        _proposal(
            "proposal.model_a_repeat",
            "large_model_a",
            (exists,),
            alignment=1.0,
            novelty=0.1,
            gain=0.1,
            cost=4.0,
        ),
    )
    field.seal_proposals()
    before_probe = field.partial_commit()
    no_neural_vote_commit = "answer.api_x" not in field.committed

    field.attach_evidence(
        receipt_id="receipt.runtime_probe",
        requirement_id="probe.api_x_exists",
        observed=True,
        source_group="runtime_probe",
        reliability=1.0,
        match_strength=1.0,
    )
    after_probe = field.partial_commit()
    assertions = {
        "conflicting_model_outputs_became_branches": conflict.action == "branch",
        "higher_or_repeated_neural_priority_did_not_certify_truth": no_neural_vote_commit,
        "runtime_evidence_certified_matching_claim": (
            field.assess_claim("claim.api_exists").status == "certified"
        ),
        "runtime_evidence_contradicted_opposite_claim": (
            field.assess_claim("claim.api_missing").status == "contradicted"
        ),
        "answer_committed_only_after_external_receipt": (
            field.committed.get("answer.api_x") == "claim.api_exists"
        ),
    }
    return {
        "scenario": field.field_id,
        "passed": all(assertions.values()),
        "assertions": assertions,
        "checkpoints": {
            "before_probe": before_probe.detail,
            "after_probe": after_probe.detail,
            "proposal_priorities": {
                proposal_a.proposal_id: proposal_a.exploration_priority,
                proposal_b.proposal_id: proposal_b.exploration_priority,
            },
        },
        "snapshot": field.snapshot(),
    }


def run_all(output_path: Path) -> dict[str, Any]:
    scenarios = [
        run_website_branching(),
        run_visual_synthesis(),
        run_fact_conflict(),
    ]
    assertions = [
        value
        for scenario in scenarios
        for value in scenario["assertions"].values()
    ]
    report = {
        "format": "polaris-evidence-constellation-minimal-gate-v0",
        "status": "structural_semantics_only_not_model_intelligence",
        "passed": all(assertions),
        "scenario_count": len(scenarios),
        "assertion_count": len(assertions),
        "assertions_passed": sum(bool(value) for value in assertions),
        "contract": {
            "model_started": False,
            "training_performed": False,
            "neural_confidence_is_evidence": False,
            "conflicts_are_averaged": False,
            "partial_commit_is_surviving_branch_intersection": True,
            "same_source_receipt_supersedes_previous": True,
        },
        "scenarios": scenarios,
    }
    if not finite_snapshot(report):
        raise ValueError("回执包含 NaN/Inf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    _configure_utf8()
    output = Path(__file__).with_name("minimal_gate_receipt.json")
    report = run_all(output)
    compact = {
        "passed": report["passed"],
        "scenario_count": report["scenario_count"],
        "assertions": f'{report["assertions_passed"]}/{report["assertion_count"]}',
        "scenarios": {
            scenario["scenario"]: scenario["passed"]
            for scenario in report["scenarios"]
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"回执: {output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
