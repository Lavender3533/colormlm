#!/usr/bin/env python3
"""Polaris Cognitive Substrate 的最小事务式思维图实验。"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


@dataclass
class ThoughtNode:
    node_id: str
    kind: str
    value: Any
    source: str
    epoch: int
    status: str = "active"
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Requirement:
    fact_id: str
    expected: Any


@dataclass
class Hypothesis:
    hypothesis_id: str
    value: str
    requirements: tuple[Requirement, ...]
    status: str = "pending"
    evidence: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewriteReceipt:
    epoch: int
    event: str
    changed_nodes: tuple[str, ...]
    preserved_nodes: tuple[str, ...]
    evaluated_hypotheses: tuple[str, ...]
    decision: str | None
    hypothesis_status: dict[str, str]


@dataclass
class ThoughtGraph:
    nodes: dict[str, ThoughtNode] = field(default_factory=dict)
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    dependency_index: dict[str, set[str]] = field(default_factory=dict)
    receipts: list[RewriteReceipt] = field(default_factory=list)
    epoch: int = 0

    def add_node(
        self,
        node_id: str,
        kind: str,
        value: Any,
        source: str,
        *,
        depends_on: tuple[str, ...] = (),
    ) -> None:
        self.nodes[node_id] = ThoughtNode(
            node_id=node_id,
            kind=kind,
            value=value,
            source=source,
            epoch=self.epoch,
            depends_on=depends_on,
        )

    def add_hypothesis(
        self,
        hypothesis_id: str,
        value: str,
        requirements: tuple[Requirement, ...],
    ) -> None:
        self.hypotheses[hypothesis_id] = Hypothesis(
            hypothesis_id=hypothesis_id,
            value=value,
            requirements=requirements,
        )
        for requirement in requirements:
            self.dependency_index.setdefault(requirement.fact_id, set()).add(hypothesis_id)

    def _evaluate_hypotheses(self, hypothesis_ids: set[str]) -> tuple[str, ...]:
        evaluated: list[str] = []
        for hypothesis_id in sorted(hypothesis_ids):
            hypothesis = self.hypotheses[hypothesis_id]
            evaluated.append(hypothesis_id)
            evidence: list[str] = []
            contradictions: list[str] = []
            missing = False
            for requirement in hypothesis.requirements:
                fact = self.nodes.get(requirement.fact_id)
                if fact is None or fact.status != "active":
                    missing = True
                    continue
                evidence.append(requirement.fact_id)
                if fact.value != requirement.expected:
                    contradictions.append(requirement.fact_id)
            if contradictions:
                status = "contradicted"
            elif missing:
                status = "pending"
            else:
                status = "supported"
            hypothesis.status = status
            hypothesis.evidence = tuple(evidence)
            hypothesis.contradictions = tuple(contradictions)
        return tuple(evaluated)

    def _rollback_invalid_decision(self) -> list[str]:
        decision = self.nodes.get("decision.stack")
        if decision is None or decision.status != "active":
            return []
        hypothesis = self.hypotheses[str(decision.value)]
        if hypothesis.status == "supported":
            return []
        decision.status = "rolled_back"
        decision.epoch = self.epoch
        return [decision.node_id]

    def _late_commit(self) -> list[str]:
        active_decision = self.nodes.get("decision.stack")
        if active_decision is not None and active_decision.status == "active":
            return []
        supported = [item for item in self.hypotheses.values() if item.status == "supported"]
        unresolved = [item for item in self.hypotheses.values() if item.status == "pending"]
        if len(supported) != 1 or unresolved:
            return []
        winner = supported[0]
        self.add_node(
            "decision.stack",
            "commit",
            winner.hypothesis_id,
            "polaris.late_commit",
            depends_on=(winner.hypothesis_id, *winner.evidence),
        )
        return ["decision.stack"]

    def apply_evidence(self, fact_id: str, value: Any, source: str) -> RewriteReceipt:
        self.epoch += 1
        before = {
            node_id: (node.value, node.status, node.epoch)
            for node_id, node in self.nodes.items()
        }
        self.add_node(fact_id, "evidence", value, source)
        evaluated = self._evaluate_hypotheses(self.dependency_index.get(fact_id, set()))
        changed = [fact_id]
        changed.extend(self._rollback_invalid_decision())
        changed.extend(self._late_commit())

        preserved = []
        for node_id, snapshot in before.items():
            current = self.nodes[node_id]
            if (current.value, current.status, current.epoch) == snapshot:
                preserved.append(node_id)
        decision_node = self.nodes.get("decision.stack")
        decision = (
            str(decision_node.value)
            if decision_node is not None and decision_node.status == "active"
            else None
        )
        receipt = RewriteReceipt(
            epoch=self.epoch,
            event=f"evidence:{fact_id}={value!r}",
            changed_nodes=tuple(dict.fromkeys(changed)),
            preserved_nodes=tuple(sorted(preserved)),
            evaluated_hypotheses=evaluated,
            decision=decision,
            hypothesis_status={
                item.hypothesis_id: item.status for item in self.hypotheses.values()
            },
        )
        self.receipts.append(receipt)
        return receipt


def run_minimal_gate(output_path: Path) -> dict[str, Any]:
    graph = ThoughtGraph()
    graph.add_node("goal.website", "goal", "build_web_application", "user")
    graph.add_node("design.layout", "design_ir", "dashboard_two_column", "visual_island")
    goal_epoch = graph.nodes["goal.website"].epoch
    design_epoch = graph.nodes["design.layout"].epoch

    graph.add_hypothesis(
        "stack.php",
        "PHP",
        (
            Requirement("fact.composer_json", True),
            Requirement("fact.php_runtime", True),
            Requirement("fact.composer_install", True),
        ),
    )
    graph.add_hypothesis(
        "stack.java",
        "Java",
        (
            Requirement("fact.pom_xml", True),
            Requirement("fact.jvm_runtime", True),
        ),
    )

    graph.apply_evidence("fact.composer_json", True, "filesystem")
    graph.apply_evidence("fact.pom_xml", True, "filesystem")
    no_premature_commit = graph.nodes.get("decision.stack") is None

    graph.apply_evidence("fact.php_runtime", True, "runtime_probe")
    graph.apply_evidence("fact.jvm_runtime", False, "runtime_probe")
    graph.apply_evidence("fact.composer_install", True, "tool_receipt")
    first_decision = graph.nodes["decision.stack"].value

    rollback_receipt = graph.apply_evidence(
        "fact.composer_install", False, "tool_receipt:install_failed"
    )
    decision_after_failure = rollback_receipt.decision
    local_rewrite = rollback_receipt.evaluated_hypotheses == ("stack.php",)
    unrelated_state_preserved = (
        graph.nodes["goal.website"].epoch == goal_epoch
        and graph.nodes["design.layout"].epoch == design_epoch
        and graph.nodes["goal.website"].status == "active"
        and graph.nodes["design.layout"].status == "active"
        and "goal.website" in rollback_receipt.preserved_nodes
        and "design.layout" in rollback_receipt.preserved_nodes
    )

    graph.apply_evidence("fact.jvm_runtime", True, "environment:provisioned")
    final_decision = graph.nodes["decision.stack"].value

    assertions = {
        "both_hypotheses_retained_before_evidence": no_premature_commit,
        "php_committed_after_support": first_decision == "stack.php",
        "failed_php_was_rolled_back": decision_after_failure is None,
        "unrelated_goal_and_design_preserved": unrelated_state_preserved,
        "java_committed_after_new_evidence": final_decision == "stack.java",
        "composer_failure_only_recomputed_php": local_rewrite,
    }
    report = {
        "format": "polaris-cognitive-substrate-minimal-gate-v0",
        "status": "structural_contract_only_not_intelligence_evidence",
        "passed": all(assertions.values()),
        "assertions": assertions,
        "final_decision": final_decision,
        "nodes": {key: asdict(value) for key, value in graph.nodes.items()},
        "hypotheses": {key: asdict(value) for key, value in graph.hypotheses.items()},
        "receipts": [asdict(item) for item in graph.receipts],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    _configure_utf8()
    output = Path(__file__).with_name("minimal_gate_receipt.json")
    report = run_minimal_gate(output)
    print(json.dumps({"passed": report["passed"], **report["assertions"]}, ensure_ascii=False, indent=2))
    print(f"回执: {output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
