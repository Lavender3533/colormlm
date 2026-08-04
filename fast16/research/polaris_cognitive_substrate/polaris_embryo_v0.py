#!/usr/bin/env python3
"""用现成 v47 规划资产和 v38 产物组装 Polaris Embryo 的第一条短纵切面。

默认路径只做冻结证据重放和确定性重新验收：不下载、不训练、不编译、不启动 S14，
也不启动任何模型进程。候选未通过时只产生能力岛出生请求，不在本短门里执行长回退。
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from genesis_contracts import EmbryoState, OrganSpec, Proposal


MODEL_V38 = "ColorLM-v38-Qwen36-Shared-Sequence-Policy"
TASK_ID = "pf47-train-01"


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def _read_task(path: Path, task_id: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("id") == task_id]
    if len(selected) != 1:
        raise ValueError(f"找不到唯一任务 {task_id}")
    return selected[0]


def _verify_v47_release(release_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = release_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    results: dict[str, Any] = {}
    for name, expected in manifest.get("files", {}).items():
        path = release_dir / name
        if not path.is_file():
            results[name] = {"exists": False, "verified": False, "expected": expected}
            continue
        actual = _sha256(path)
        results[name] = {
            "exists": True,
            "verified": actual == expected,
            "expected": expected,
            "actual": actual,
            "bytes": path.stat().st_size,
        }
    required = {"genome_head.npz", "genome_head_ontology.json", "negative_contract.json"}
    if not required.issubset(results):
        raise RuntimeError(f"v47 发布包缺少核心声明: {sorted(required - set(results))}")
    if not all(item.get("verified") for item in results.values()):
        bad = [name for name, item in results.items() if not item.get("verified")]
        raise RuntimeError(f"v47 发布资产 SHA 验证失败: {bad}")
    return manifest, {
        "manifest_sha256": _sha256(manifest_path),
        "verified_files": results,
        "core_parameter_count": manifest.get("parameter_count"),
        "claim_limit": manifest.get("claim_limit"),
    }


def _load_design_ir(v47_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(v47_dir))
    try:
        from run_frontend_ir_ab import parse_ir

        return parse_ir(
            (v47_dir / "frontend_ir_ab_train01_retry1/design_ir_raw.txt").read_text(
                encoding="utf-8"
            )
        )
    finally:
        sys.path.pop(0)


def _evaluate(frontend_dir: Path, task: dict[str, Any], html_path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(frontend_dir))
    try:
        from validate_gates import evaluate_one

        return evaluate_one(task, html_path)
    finally:
        sys.path.pop(0)


def run_embryo(output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[3]
    v47_dir = root / "fast16/research/v47_dual_tempo_bus"
    frontend_dir = root / "fast16/research/parallel_frontend_v47"
    release_dir = root / "fast16/release/polaris-v47-genome-head-preview"
    evidence_dir = v47_dir / "frontend_ir_ab_train01_retry1"
    task_path = frontend_dir / "data/train.jsonl"
    baseline_path = frontend_dir / "fixtures/ordinary_three_cards.html"
    candidate_path = evidence_dir / "candidate.html"
    browser_path = evidence_dir / "browser_visual_report.json"
    launcher_path = root / "fast16/run-colormlm-v38-qwen36-sequence-policy.ps1"

    task = _read_task(task_path, TASK_ID)
    v47_manifest, v47_verification = _verify_v47_release(release_dir)
    design_ir = _load_design_ir(v47_dir)
    baseline_gate = _evaluate(frontend_dir, task, baseline_path)
    candidate_gate = _evaluate(frontend_dir, task, candidate_path)
    browser_receipt = _read_json(browser_path)

    state = EmbryoState(task_id="polaris-embryo-v38-v47-reuse-001")
    goal = state.add_packet(
        "goal.frontend.001",
        "goal.frontend",
        {"task_id": TASK_ID, "prompt": task["prompt"]},
        "user_fixture",
    )
    v47_organ = OrganSpec(
        organ_id="organ.polaris.v47.parallel_genome_head.preview",
        architecture="terminal_hidden[2048]->parallel_design_genome[20_fields]",
        input_ports=("hidden.terminal.2048", "copy.slots", "goal.frontend"),
        output_port="design.genome.v47",
        manifest_sha256=v47_verification["manifest_sha256"],
        lifecycle="conditional_research_preview_no_spawn",
    )
    v47_organ_packet = state.add_packet(
        "organ.v47.genome.001",
        "organ.lifecycle.v47_genome_head",
        {
            "version": v47_manifest["version"],
            "parameter_count": v47_manifest["parameter_count"],
            "organ": asdict(v47_organ),
            "state": "eligible_for_compiler_ab_only",
            "activation_requires": [
                "fresh genome output lineage",
                "frozen validation static pass",
                "375/768/1024/1440 browser action trace",
                "blind pass",
            ],
            "hard_bypass": True,
            **v47_verification,
        },
        "polaris-v47-genome-head-preview",
        parents=(goal.packet_id,),
    )
    design_ir_path = evidence_dir / "design_ir_raw.txt"
    design = state.add_packet(
        "design.ir.v47.001",
        "design.ir.frontend.v47_compat",
        {
            "ir": design_ir,
            "raw_path": str(design_ir_path.relative_to(root)),
            "raw_sha256": _sha256(design_ir_path),
            "provenance": "historical.v38.planning_fixture",
            "v47_compatibility_only": True,
        },
        "historical.v38.planning_fixture",
        parents=(goal.packet_id,),
    )

    v38_organ = OrganSpec(
        organ_id="organ.v38.shared_sequence_policy",
        architecture="Qwen3.6 shared backbone + ColorLM Neural Bus",
        input_ports=("goal.frontend", "design.ir.frontend.v47_compat"),
        output_port="candidate.artifact.html",
        manifest_sha256=_sha256(launcher_path),
        lifecycle="existing_local_runtime_replayed",
    )
    organ_packet = state.add_packet(
        "organ.v38.001",
        "organ.transformer.v38",
        {"model": MODEL_V38, "organ": asdict(v38_organ)},
        "existing_project_runtime",
        parents=(design.packet_id,),
    )
    artifact_v1 = state.add_packet(
        "artifact.html.v1",
        "artifact.html",
        {
            "path": str(baseline_path.relative_to(root)),
            "sha256": _sha256(baseline_path),
            "bytes": baseline_path.stat().st_size,
        },
        "frozen_baseline",
        parents=(goal.packet_id,),
        version=1,
    )
    initial_validation = state.add_packet(
        "validation.html.v1",
        "validation.frontend.static",
        {
            "passed": bool(baseline_gate["passed"]),
            "final_score": baseline_gate["audit"]["summary"]["final_score"],
            "critical": baseline_gate["critical"],
        },
        "parallel_frontend_v47.validate_gates",
        parents=(artifact_v1.packet_id,),
    )
    compile_report_path = evidence_dir / "compile_report.json"
    compile_report = _read_json(compile_report_path)
    candidate = state.add_packet(
        "candidate.html.v2",
        "candidate.artifact.html",
        {
            "path": str(candidate_path.relative_to(root)),
            "sha256": _sha256(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "historical_generator": MODEL_V38,
            "source": "historical.v38.expression+deterministic_tail",
            "compile_report_sha256": _sha256(compile_report_path),
            "pure_model_claim_allowed": bool(compile_report.get("pure_model_claim_allowed", False)),
        },
        v38_organ.organ_id,
        parents=(goal.packet_id, design.packet_id, organ_packet.packet_id, initial_validation.packet_id),
        status="candidate",
    )
    static_validation = state.add_packet(
        "validation.html.v2.static",
        "validation.frontend.static",
        {
            "passed": bool(candidate_gate["passed"]),
            "subject_packet": candidate.packet_id,
            "subject_sha256": candidate.payload["sha256"],
            "final_score": candidate_gate["audit"]["summary"]["final_score"],
            "template_penalty": candidate_gate["audit"]["summary"]["template_penalty"],
            "critical": candidate_gate["critical"],
            "score_checks": candidate_gate["score_checks"],
            "evidence_mode": "deterministic_recompute_on_historical_fixture",
            "general_claim_allowed": False,
        },
        "parallel_frontend_v47.validate_gates",
        parents=(candidate.packet_id,),
    )
    browser_passed = bool(
        browser_receipt.get("decision", {}).get("functional_browser_check_passed")
        and not browser_receipt.get("console_errors_or_warnings")
    )
    browser_validation = state.add_packet(
        "validation.html.v2.browser",
        "validation.frontend.browser_receipt",
        {
            "passed": browser_passed,
            "subject_packet": candidate.packet_id,
            "subject_sha256": candidate.payload["sha256"],
            "report_path": str(browser_path.relative_to(root)),
            "receipt_sha256": _sha256(browser_path),
            "desktop": browser_receipt.get("desktop"),
            "mobile": browser_receipt.get("mobile"),
            "evidence_mode": "historical_replay",
            "general_claim_allowed": False,
        },
        "frozen_browser_action_trace",
        parents=(candidate.packet_id,),
    )
    accepted = bool(static_validation.payload["passed"] and browser_validation.payload["passed"])
    helix_validation = state.add_packet(
        "validation.html.v2.helix",
        "validation.frontend.helix",
        {
            "passed": accepted,
            "subject_packet": candidate.packet_id,
            "subject_sha256": candidate.payload["sha256"],
            "scope": "single_train_hybrid_fixture_commit_only",
            "capability_promotion_allowed": False,
            "missing_gates": [
                "768px browser",
                "1024px browser",
                "contrast",
                "full focus trace",
                "frozen validation",
                "blind",
            ],
        },
        "polaris.helix",
        parents=(static_validation.packet_id, browser_validation.packet_id),
    )
    fallback = state.add_packet(
        "genesis.decision.001",
        "organ.genesis.decision",
        {
            "accepted_without_new_island": accepted,
            "ephemeral_island_spawned": False,
            "on_failure": "emit_spawn_request_then_stop_short_gate",
        },
        "polaris.genesis",
        parents=(helix_validation.packet_id,),
    )
    if not accepted:
        raise RuntimeError("v38/v47 候选未通过短门；已停止，未启动长能力岛")

    proposal = Proposal(
        proposal_id="proposal.frontend.001",
        organ_id=v38_organ.organ_id,
        input_packets=(goal.packet_id, design.packet_id),
        output_packet=candidate.packet_id,
        status="validated",
    )
    artifact_v2 = state.add_packet(
        "artifact.html.v2",
        "artifact.html",
        candidate.payload,
        "polaris.commit",
        parents=(candidate.packet_id, helix_validation.packet_id),
        version=2,
    )
    commit = state.build_commit(
        commit_id="commit.frontend.001",
        proposal=proposal,
        validation_packet=helix_validation.packet_id,
        superseded_packet=artifact_v1.packet_id,
        committed_packet=artifact_v2.packet_id,
        preserved_packets=(goal.packet_id, v47_organ_packet.packet_id, design.packet_id),
    )
    state.add_packet(
        "commit.frontend.001",
        "commit.receipt",
        asdict(commit),
        "polaris.commit",
        parents=(artifact_v2.packet_id, helix_validation.packet_id),
    )
    state.supersede(artifact_v1.packet_id, artifact_v2.packet_id)
    candidate.status = "committed"

    assertions = {
        "v47_release_assets_sha_verified": all(
            item["verified"] for item in v47_verification["verified_files"].values()
        ),
        "v47_head_registered_as_conditional_only": v47_organ_packet.payload["hard_bypass"]
        and v47_organ_packet.payload["state"] == "eligible_for_compiler_ab_only",
        "historical_design_ir_lineage_is_honest": design.parents == (goal.packet_id,)
        and design.payload["provenance"] == "historical.v38.planning_fixture",
        "v38_existing_artifact_is_consumed": candidate.payload["historical_generator"] == MODEL_V38,
        "helix_static_gate_recomputed": bool(static_validation.payload["passed"]),
        "frozen_browser_receipt_consumed": bool(browser_validation.payload["passed"]),
        "failed_baseline_superseded": not initial_validation.payload["passed"]
        and artifact_v1.status == "superseded",
        "candidate_committed_without_island": candidate.status == "committed"
        and not fallback.payload["ephemeral_island_spawned"],
        "fixture_commit_is_not_capability_promotion": not helix_validation.payload[
            "capability_promotion_allowed"
        ]
        and v47_organ_packet.packet_id not in candidate.parents,
        "no_s14_dependency": "s14" not in json.dumps(
            {key: asdict(value) for key, value in state.packets.items()},
            ensure_ascii=False,
        ).lower(),
    }
    report = {
        "format": "polaris-embryo-v38-v47-reuse-v0",
        "status": "pass" if all(assertions.values()) else "fail",
        "mode": "frozen_reuse_replay",
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "definition": (
            "v47 合同约束历史 Design IR 的结构，v38 Transformer 历史产物负责表达；"
            "Helix 复验后只提交夹具；v47 Genome Head 保持条件器官，失败才请求临时能力岛。"
        ),
        "assertions": assertions,
        "proposal": asdict(proposal),
        "commit": asdict(commit),
        "packets": {key: asdict(value) for key, value in state.packets.items()},
        "events": state.events,
        "limitations": [
            "本门重放现有 v38/v47 冻结产物，不等于任意新任务的在线生成。",
            "浏览器动作使用既有冻结回执，本次只重新执行确定性静态门。",
            "v47 Genome Head 仍是研究预览且不是本候选父血缘；本门不宣称能力晋级。",
            "能力岛失败回退只形成出生决策，刻意不在两分钟短门内执行。",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    _configure_utf8()
    output = Path(__file__).with_name("embryo_v0") / "reuse_receipt.json"
    try:
        report = run_embryo(output)
    except Exception as error:  # noqa: BLE001 - 顶层输出精确失败
        print(f"embryo_error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "mode": report["mode"],
                "elapsed_seconds": report["elapsed_seconds"],
                **report["assertions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"回执: {output.resolve()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
