"""K3 前端专家社区聚合器的无模型、无下载合成自检。"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import community


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def make_fixture(directory: Path) -> tuple[Path, Path, Path]:
    tasks: list[dict] = []
    traces: list[dict] = []
    categories = ["html_semantic_tag", "css_layout_or_responsive"]
    target_by_layer = {20: [101, 111], 21: [202], 22: [303]}

    for task_index in range(4):
        task_id = f"synthetic-frontend-{task_index:02d}"
        prompt = f"合成前端任务 {task_index}；只用于社区算法自检。"
        critical_tokens = []
        for decision_index, category in enumerate(categories):
            decision_id = f"decision-{decision_index:02d}"
            prefix = f"{task_id}/prefix/{decision_index}"
            critical_tokens.append(
                {
                    "decision_id": decision_id,
                    "prefix_utf8": prefix,
                    "prefix_sha256": community.sha256_bytes(prefix.encode("utf-8")),
                    "token_id": 1000 + task_index * 10 + decision_index,
                    "token_text": "<main>" if decision_index == 0 else "grid-template-columns",
                    "category": category,
                    "selection_reason": "在查看合成 router/NLL 前固定的前端关键决策",
                    "selected_before_trace": True,
                }
            )
        task = {
            "format": community.TASK_FORMAT,
            "task_id": task_id,
            "split": "train",
            "source_task_sha256": "b" * 64,
            "prompt_utf8": prompt,
            "prompt_sha256": community.sha256_bytes(prompt.encode("utf-8")),
            "critical_tokens": critical_tokens,
            "frozen_before_trace": True,
        }
        tasks.append(task)

        for decision in critical_tokens:
            for layer, target_experts in target_by_layer.items():
                distractor = 700 + task_index
                selected = list(target_experts) + [distractor]
                filler = 0
                while len(selected) < community.TOP_K:
                    if filler not in selected:
                        selected.append(filler)
                    filler += 1
                weights = [0.15 if expert in target_experts else 0.55 / (community.TOP_K - len(target_experts)) for expert in selected]
                native_nll = 1.0
                ablations = [
                    {
                        "expert_id": expert,
                        "ablated_nll": native_nll + 0.12 + 0.01 * (layer - 20),
                        "delta_nll": 0.12 + 0.01 * (layer - 20),
                    }
                    for expert in target_experts
                ]
                ablations.append(
                    {
                        "expert_id": distractor,
                        "ablated_nll": 0.97,
                        "delta_nll": -0.03,
                    }
                )
                traces.append(
                    {
                        "format": community.TRACE_FORMAT,
                        "repo": community.REPO,
                        "revision": community.REVISION,
                        "synthetic": True,
                        "native_forward_completed": False,
                        "task_id": task_id,
                        "split": "train",
                        "prompt_sha256": task["prompt_sha256"],
                        "token": {
                            "position": 20 + int(decision["decision_id"].split("-")[-1]),
                            "phase": "teacher_force",
                            "prefix_sha256": decision["prefix_sha256"],
                            "target_token_id": decision["token_id"],
                            "target_token_text": decision["token_text"],
                            "target_logprob": -native_nll,
                            "target_nll": native_nll,
                        },
                        "decision": {
                            "decision_id": decision["decision_id"],
                            "category": decision["category"],
                            "selection_reason": decision["selection_reason"],
                            "selected_before_trace": True,
                        },
                        "layer": layer,
                        "router": {
                            "topk_expert_ids": selected,
                            "topk_weights": weights,
                        },
                        "counterfactual": {
                            "mode": "leave_one_selected_expert_out",
                            "native_nll": native_nll,
                            "ablations": ablations,
                        },
                    }
                )

    catalog_entries = []
    for index, (layer, experts) in enumerate(target_by_layer.items()):
        for expert in experts:
            start = (index * 10 + expert) * 20_000_000
            catalog_entries.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "source_shard": f"model-{layer + 1:05d}-of-000096.safetensors",
                    "source_shard_lfs_sha256": "a" * 64,
                    "header_tensors_sha256": "c" * 64,
                    "range": {
                        "start": start,
                        "end_inclusive": start + community.EXPERT_PAGE_BYTES - 1,
                        "bytes": community.EXPERT_PAGE_BYTES,
                    },
                }
            )
    catalog = {
        "format": community.CATALOG_FORMAT,
        "repo": community.REPO,
        "revision": community.REVISION,
        "entries": catalog_entries,
    }
    tasks_path = directory / "tasks.synthetic.jsonl"
    trace_path = directory / "trace.synthetic.jsonl"
    catalog_path = directory / "ranges.synthetic.json"
    write_jsonl(tasks_path, tasks)
    write_jsonl(trace_path, traces)
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return tasks_path, trace_path, catalog_path


def expect_failure(function, name: str) -> bool:
    try:
        function()
    except (ValueError, RuntimeError, TypeError):
        return True
    raise AssertionError(f"负向自检未拒绝: {name}")


def run_selftest() -> dict:
    with tempfile.TemporaryDirectory(prefix="polaris-k3-community-") as raw:
        directory = Path(raw)
        tasks_path, trace_path, catalog_path = make_fixture(directory)
        output_path = directory / "range-candidates.json"
        args = argparse.Namespace(
            trace=trace_path,
            tasks=tasks_path,
            output=output_path,
            header_catalog=catalog_path,
            source_contract=Path(__file__).with_name("source_contract.json"),
            workspace_root=Path(__file__).resolve().parents[4],
            min_task_coverage=3,
            min_benefit_tasks=3,
            min_positive_task_fraction=0.67,
            min_mean_delta_nll=0.01,
            min_edge_tasks=3,
            max_layer_gap=2,
            allow_synthetic=True,
            force=False,
        )
        result = community.run(args)
        task_rows = community.read_jsonl(tasks_path)
        task_map, decisions = community.load_task_contract(tasks_path, None, True)
        trace_rows = community.read_jsonl(trace_path)

        wrong_revision = copy.deepcopy(trace_rows)
        wrong_revision[0]["revision"] = "master"
        duplicate_router = copy.deepcopy(trace_rows)
        duplicate_router[0]["router"]["topk_expert_ids"][1] = duplicate_router[0]["router"]["topk_expert_ids"][0]
        wrong_delta = copy.deepcopy(trace_rows)
        wrong_delta[0]["counterfactual"]["ablations"][0]["delta_nll"] = 9.0
        wrong_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        wrong_catalog["revision"] = "master"
        wrong_catalog_path = directory / "wrong-catalog.json"
        wrong_catalog_path.write_text(json.dumps(wrong_catalog), encoding="utf-8")

        selected = {(item["layer"], item["expert"]) for item in result["range_candidates"]}
        expected = {(20, 101), (20, 111), (21, 202), (22, 303)}
        checks = {
            "fixture_tasks_created": len(task_rows) == 4,
            "fixture_trace_records": len(trace_rows) == 24,
            "one_cross_layer_community": len(result["communities"]) == 1,
            "only_expected_experts_selected": selected == expected,
            "all_ranges_resolved_from_pinned_catalog": result["exact_range_count"] == 4 and result["blocked_range_count"] == 0,
            "dry_run_only": result["dry_run"] is True and result["downloads_performed"] is False and result["download_authorized"] is False,
            "revision_drift_rejected": expect_failure(
                lambda: community.validate_trace(wrong_revision, task_map, decisions, True),
                "revision_drift",
            ),
            "duplicate_router_expert_rejected": expect_failure(
                lambda: community.validate_trace(duplicate_router, task_map, decisions, True),
                "duplicate_router",
            ),
            "counterfactual_delta_mismatch_rejected": expect_failure(
                lambda: community.validate_trace(wrong_delta, task_map, decisions, True),
                "wrong_delta",
            ),
            "unpinned_catalog_rejected": expect_failure(
                lambda: community.load_range_catalog(wrong_catalog_path),
                "wrong_catalog_revision",
            ),
        }
        return {
            "format": "polaris-k3-frontend-community-selftest-v1",
            "ok": all(checks.values()),
            "evidence_status": "synthetic_algorithm_fixture_not_k3_capability_evidence",
            "checks": checks,
            "fixture_summary": {
                "tasks": len(task_rows),
                "trace_records": len(trace_rows),
                "community_count": len(result["communities"]),
                "selected_nodes": sorted(f"L{layer:02d}/E{expert:03d}" for layer, expert in selected),
                "exact_range_count": result["exact_range_count"],
            },
            "claim_limit": "合成 fixture 只验证社区聚合、严格输入门和 Range dry-run；没有运行 Kimi K3。",
        }


def main() -> int:
    community.configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("SELFTEST_REPORT.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = run_selftest()
    community.write_json(args.output, report, args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
