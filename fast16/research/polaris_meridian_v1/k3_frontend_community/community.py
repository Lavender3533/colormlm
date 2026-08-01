"""从 Kimi K3 原生 router/NLL trace 聚合前端专家社区，只生成 Range dry-run。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = "moonshotai/Kimi-K3"
REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
TRACE_FORMAT = "polaris-k3-frontend-router-trace-v1"
TASK_FORMAT = "polaris-k3-frontend-community-task-v1"
CATALOG_FORMAT = "polaris-k3-pinned-expert-range-catalog-v1"
OUTPUT_FORMAT = "polaris-k3-frontend-range-candidates-v1"
EXPERT_PAGE_BYTES = 17_547_264
TOP_K = 16
EXPERTS = 896


def configure_utf8() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} 顶层必须是 object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} JSON 错误: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} 必须是 object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} 没有记录")
    return rows


def write_json(path: Path, value: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"拒绝覆盖 {path}；使用 --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    partial.replace(path)


def require_sha256(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} 不是小写 SHA-256")
    return text


def validate_source_contract(path: Path, workspace_root: Path, verify_files: bool) -> dict[str, Any]:
    contract = read_json(path)
    if contract.get("format") != "polaris-k3-frontend-community-source-contract-v1":
        raise ValueError("source contract format 错误")
    donor = contract.get("donor", {})
    if donor.get("repo") != REPO or donor.get("revision") != REVISION:
        raise ValueError("source contract donor 漂移")
    if int(donor.get("experts_per_token", -1)) != TOP_K or int(donor.get("experts_per_layer", -1)) != EXPERTS:
        raise ValueError("source contract K3 MoE ABI 错误")
    if int(donor.get("single_expert_page_bytes", -1)) != EXPERT_PAGE_BYTES:
        raise ValueError("source contract expert page 字节错误")
    if verify_files:
        for item in contract.get("frozen_frontend_tasks", []):
            source = workspace_root / str(item["path"])
            if not source.is_file() or sha256_file(source) != item["sha256"]:
                raise RuntimeError(f"冻结前端题缺失或 SHA 漂移: {source}")
        scoring = contract["scoring_contract"]
        source = workspace_root / str(scoring["path"])
        if not source.is_file() or sha256_file(source) != scoring["sha256"]:
            raise RuntimeError(f"前端评分合同缺失或 SHA 漂移: {source}")
    return contract


def load_frozen_source_tasks(contract: dict[str, Any], workspace_root: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for source in contract["frozen_frontend_tasks"]:
        path = workspace_root / source["path"]
        for row in read_jsonl(path):
            task_id = str(row.get("id", ""))
            if not task_id or task_id in tasks:
                raise ValueError(f"冻结前端题 ID 为空或重复: {task_id!r}")
            tasks[task_id] = {
                "split": row.get("split"),
                "prompt_utf8": row.get("prompt"),
                "prompt_sha256": sha256_bytes(str(row.get("prompt", "")).encode("utf-8")),
                "source_task_sha256": canonical_sha256(row),
            }
    return tasks


def load_task_contract(
    path: Path,
    source_tasks: dict[str, dict[str, Any]] | None,
    allow_synthetic: bool,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    tasks: dict[str, dict[str, Any]] = {}
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, row in enumerate(read_jsonl(path), start=1):
        if row.get("format") != TASK_FORMAT:
            raise ValueError(f"tasks 第 {line_number} 行 format 错误")
        task_id = str(row.get("task_id", ""))
        if not task_id or task_id in tasks:
            raise ValueError(f"tasks 第 {line_number} 行 task_id 为空或重复")
        if row.get("frozen_before_trace") is not True:
            raise ValueError(f"{task_id} 没有在 trace 前冻结")
        prompt = str(row.get("prompt_utf8", ""))
        if sha256_bytes(prompt.encode("utf-8")) != require_sha256(row.get("prompt_sha256"), "prompt_sha256"):
            raise ValueError(f"{task_id} prompt_sha256 不匹配")
        if source_tasks is not None and task_id not in source_tasks:
            raise ValueError(f"{task_id} 不在现有 24 道冻结前端题中")
        if source_tasks is not None:
            expected = source_tasks[task_id]
            if row.get("split") != expected["split"] or prompt != expected["prompt_utf8"]:
                raise ValueError(f"{task_id} prompt/split 与冻结源不一致")
            if row.get("source_task_sha256") != expected["source_task_sha256"]:
                raise ValueError(f"{task_id} source_task_sha256 不匹配")
        elif not allow_synthetic:
            raise RuntimeError("真实任务必须绑定现有冻结前端题")
        critical = row.get("critical_tokens")
        if not isinstance(critical, list) or not critical:
            raise ValueError(f"{task_id} critical_tokens 为空")
        local_ids: set[str] = set()
        for decision in critical:
            decision_id = str(decision.get("decision_id", ""))
            if not decision_id or decision_id in local_ids:
                raise ValueError(f"{task_id} decision_id 为空或重复")
            local_ids.add(decision_id)
            if decision.get("selected_before_trace") is not True:
                raise ValueError(f"{task_id}/{decision_id} 不是预冻结 token")
            prefix = str(decision.get("prefix_utf8", ""))
            if sha256_bytes(prefix.encode("utf-8")) != require_sha256(decision.get("prefix_sha256"), "prefix_sha256"):
                raise ValueError(f"{task_id}/{decision_id} prefix_sha256 不匹配")
            if int(decision.get("token_id", -1)) < 0 or not isinstance(decision.get("token_text"), str):
                raise ValueError(f"{task_id}/{decision_id} token 不完整")
            if not str(decision.get("selection_reason", "")):
                raise ValueError(f"{task_id}/{decision_id} 缺少选择理由")
            decisions[(task_id, decision_id)] = decision
        tasks[task_id] = row
    return tasks, decisions


def validate_trace(
    rows: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    decisions: dict[tuple[str, str], dict[str, Any]],
    allow_synthetic: bool,
) -> bool:
    seen: set[tuple[str, str, int]] = set()
    synthetic_values: set[bool] = set()
    for line_number, row in enumerate(rows, start=1):
        prefix = f"trace 第 {line_number} 行"
        if row.get("format") != TRACE_FORMAT or row.get("repo") != REPO or row.get("revision") != REVISION:
            raise ValueError(f"{prefix} format/repo/revision 错误")
        synthetic = row.get("synthetic") is True
        synthetic_values.add(synthetic)
        if synthetic and not allow_synthetic:
            raise RuntimeError(f"{prefix} 是 synthetic；真实 dry-run 拒绝")
        if not synthetic and row.get("native_forward_completed") is not True:
            raise RuntimeError(f"{prefix} 未证明 K3 原生 forward")
        task_id = str(row.get("task_id", ""))
        if task_id not in tasks:
            raise ValueError(f"{prefix} task_id 未冻结")
        task = tasks[task_id]
        if row.get("split") != task.get("split") or row.get("prompt_sha256") != task.get("prompt_sha256"):
            raise ValueError(f"{prefix} task split/prompt SHA 漂移")
        decision = row.get("decision")
        if not isinstance(decision, dict):
            raise TypeError(f"{prefix} decision 缺失")
        decision_id = str(decision.get("decision_id", ""))
        expected = decisions.get((task_id, decision_id))
        if expected is None:
            raise ValueError(f"{prefix} decision 未预冻结")
        if decision.get("selected_before_trace") is not True:
            raise ValueError(f"{prefix} decision 是事后选择")
        token = row.get("token")
        if not isinstance(token, dict):
            raise TypeError(f"{prefix} token 缺失")
        require_sha256(token.get("prefix_sha256"), "token.prefix_sha256")
        checks = {
            "prefix_sha256": expected["prefix_sha256"],
            "target_token_id": expected["token_id"],
            "target_token_text": expected["token_text"],
        }
        for field, expected_value in checks.items():
            if token.get(field) != expected_value:
                raise ValueError(f"{prefix} {field} 与预冻结 token 不一致")
        if decision.get("category") != expected.get("category"):
            raise ValueError(f"{prefix} category 漂移")
        layer = int(row.get("layer", -1))
        if not 1 <= layer <= 92:
            raise ValueError(f"{prefix} layer 越界")
        key = (task_id, decision_id, layer)
        if key in seen:
            raise ValueError(f"{prefix} task/decision/layer 重复")
        seen.add(key)
        router = row.get("router", {})
        ids = router.get("topk_expert_ids")
        weights = router.get("topk_weights")
        if not isinstance(ids, list) or len(ids) != TOP_K or len(set(ids)) != TOP_K:
            raise ValueError(f"{prefix} 必须有 {TOP_K} 个唯一 expert")
        if any(not isinstance(value, int) or not 0 <= value < EXPERTS for value in ids):
            raise ValueError(f"{prefix} expert ID 越界")
        if not isinstance(weights, list) or len(weights) != TOP_K:
            raise ValueError(f"{prefix} router weight 数量错误")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError(f"{prefix} router weight 非法")
        if sum(float(value) for value in weights) <= 0:
            raise ValueError(f"{prefix} router weight 全零")
        native_nll = float(row.get("counterfactual", {}).get("native_nll", math.nan))
        if not math.isfinite(native_nll) or native_nll < 0:
            raise ValueError(f"{prefix} native_nll 非法")
        if not math.isclose(native_nll, float(token.get("target_nll", math.nan)), rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"{prefix} native_nll 与 token.target_nll 不一致")
        if row.get("counterfactual", {}).get("mode") != "leave_one_selected_expert_out":
            raise ValueError(f"{prefix} 反事实模式错误")
        ablated_ids: set[int] = set()
        for ablation in row.get("counterfactual", {}).get("ablations", []):
            expert = int(ablation.get("expert_id", -1))
            if expert not in ids or expert in ablated_ids:
                raise ValueError(f"{prefix} ablation expert 未选中或重复")
            ablated_ids.add(expert)
            ablated_nll = float(ablation.get("ablated_nll", math.nan))
            delta = float(ablation.get("delta_nll", math.nan))
            if not math.isfinite(ablated_nll) or not math.isfinite(delta) or ablated_nll < 0:
                raise ValueError(f"{prefix} ablation NLL 非法")
            if not math.isclose(ablated_nll - native_nll, delta, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{prefix} delta_nll 必须等于 ablated-native")
    if len(synthetic_values) != 1:
        raise ValueError("禁止混合 synthetic 与 real trace")
    return next(iter(synthetic_values))


def node_name(node: tuple[int, int]) -> str:
    return f"L{node[0]:02d}/E{node[1]:03d}"


def aggregate(
    rows: list[dict[str, Any]],
    min_task_coverage: int,
    min_benefit_tasks: int,
    min_positive_task_fraction: float,
    min_mean_delta_nll: float,
    min_edge_tasks: int,
    max_layer_gap: int,
) -> dict[str, Any]:
    node_route_tasks: dict[tuple[int, int], set[str]] = defaultdict(set)
    node_route_hits: dict[tuple[int, int], int] = defaultdict(int)
    node_route_mass: dict[tuple[int, int], float] = defaultdict(float)
    node_task_benefits: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    contexts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    categories: dict[tuple[int, int], set[str]] = defaultdict(set)

    for row in rows:
        layer = int(row["layer"])
        task = str(row["task_id"])
        decision = str(row["decision"]["decision_id"])
        category = str(row["decision"]["category"])
        ids = row["router"]["topk_expert_ids"]
        weights = row["router"]["topk_weights"]
        weight_by_expert = {int(expert): float(weight) for expert, weight in zip(ids, weights, strict=True)}
        benefit_by_expert = {
            int(item["expert_id"]): float(item["delta_nll"])
            for item in row["counterfactual"]["ablations"]
        }
        for expert, weight in weight_by_expert.items():
            node = (layer, expert)
            node_route_tasks[node].add(task)
            node_route_hits[node] += 1
            node_route_mass[node] += weight
            categories[node].add(category)
        for expert, benefit in benefit_by_expert.items():
            node_task_benefits[(layer, expert)][task].append(benefit)
        contexts[(task, decision)].append(
            {
                "layer": layer,
                "weights": weight_by_expert,
                "benefits": benefit_by_expert,
            }
        )

    node_reports: dict[tuple[int, int], dict[str, Any]] = {}
    admitted: set[tuple[int, int]] = set()
    for node in sorted(node_route_hits):
        task_means = {
            task: sum(values) / len(values)
            for task, values in node_task_benefits.get(node, {}).items()
        }
        benefits = list(task_means.values())
        positive_tasks = sum(value > 0 for value in benefits)
        positive_fraction = positive_tasks / len(benefits) if benefits else 0.0
        mean_benefit = sum(benefits) / len(benefits) if benefits else 0.0
        checks = {
            "route_task_coverage": len(node_route_tasks[node]) >= min_task_coverage,
            "counterfactual_task_coverage": len(benefits) >= min_benefit_tasks,
            "positive_task_fraction": positive_fraction >= min_positive_task_fraction,
            "mean_delta_nll": mean_benefit >= min_mean_delta_nll,
        }
        if all(checks.values()):
            admitted.add(node)
        node_reports[node] = {
            "node": node_name(node),
            "layer": node[0],
            "expert": node[1],
            "route_hits": node_route_hits[node],
            "route_task_coverage": len(node_route_tasks[node]),
            "mean_route_weight": node_route_mass[node] / node_route_hits[node],
            "counterfactual_task_coverage": len(benefits),
            "positive_benefit_tasks": positive_tasks,
            "positive_task_fraction": positive_fraction,
            "mean_delta_nll": mean_benefit,
            "categories": sorted(categories[node]),
            "checks": checks,
            "admitted": all(checks.values()),
        }

    edge_stats: dict[tuple[tuple[int, int], tuple[int, int]], dict[str, Any]] = {}
    for (task, _decision), records in contexts.items():
        positive: list[tuple[tuple[int, int], float, float]] = []
        for record in records:
            layer = record["layer"]
            for expert, benefit in record["benefits"].items():
                node = (layer, expert)
                if node in admitted and benefit > 0:
                    positive.append((node, record["weights"][expert], benefit))
        for left_index, (left, left_weight, left_benefit) in enumerate(positive):
            for right, right_weight, right_benefit in positive[left_index + 1 :]:
                gap = abs(left[0] - right[0])
                if gap != 0 and gap > max_layer_gap:
                    continue
                pair = tuple(sorted((left, right)))
                stats = edge_stats.setdefault(
                    pair,
                    {
                        "tasks": set(),
                        "cooccurrence_hits": 0,
                        "continuous_layer_hits": 0,
                        "strength_sum": 0.0,
                    },
                )
                stats["tasks"].add(task)
                if gap == 0:
                    stats["cooccurrence_hits"] += 1
                else:
                    stats["continuous_layer_hits"] += 1
                stats["strength_sum"] += math.sqrt(left_weight * right_weight) * min(left_benefit, right_benefit)

    admitted_edges: list[dict[str, Any]] = []
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for pair, stats in sorted(edge_stats.items()):
        task_coverage = len(stats["tasks"])
        if task_coverage < min_edge_tasks:
            continue
        left, right = pair
        adjacency[left].add(right)
        adjacency[right].add(left)
        hits = stats["cooccurrence_hits"] + stats["continuous_layer_hits"]
        admitted_edges.append(
            {
                "left": node_name(left),
                "right": node_name(right),
                "task_coverage": task_coverage,
                "cooccurrence_hits": stats["cooccurrence_hits"],
                "continuous_layer_hits": stats["continuous_layer_hits"],
                "mean_strength": stats["strength_sum"] / hits,
            }
        )

    components: list[list[tuple[int, int]]] = []
    remaining = set(admitted)
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[tuple[int, int]] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency.get(node, set()) - component, reverse=True))
        remaining -= component
        components.append(sorted(component))

    communities: list[dict[str, Any]] = []
    node_to_community: dict[tuple[int, int], str] = {}
    for component in components:
        layers = sorted({node[0] for node in component})
        if len(component) < 2 or len(layers) < 2:
            continue
        tasks = set().union(*(node_route_tasks[node] for node in component))
        mean_benefit = sum(node_reports[node]["mean_delta_nll"] for node in component) / len(component)
        internal = [
            edge
            for edge in admitted_edges
            if any(edge["left"] == node_name(node) for node in component)
            and any(edge["right"] == node_name(node) for node in component)
        ]
        continuity = sum(edge["continuous_layer_hits"] for edge in internal)
        score = len(tasks) + 4.0 * mean_benefit + 0.05 * continuity
        communities.append(
            {
                "nodes": [node_name(node) for node in component],
                "layers": layers,
                "task_coverage": len(tasks),
                "mean_delta_nll": mean_benefit,
                "continuous_layer_hits": continuity,
                "score": score,
            }
        )
    communities.sort(key=lambda item: (-item["score"], item["nodes"]))
    for index, community in enumerate(communities, start=1):
        community_id = f"community-{index:03d}"
        community["community_id"] = community_id
        for name in community["nodes"]:
            layer_text, expert_text = name.split("/")
            node_to_community[(int(layer_text[1:]), int(expert_text[1:]))] = community_id

    return {
        "node_reports": [node_reports[node] for node in sorted(node_reports)],
        "admitted_edges": admitted_edges,
        "communities": communities,
        "node_to_community": node_to_community,
    }


def load_range_catalog(path: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    if path is None:
        return {}
    catalog = read_json(path)
    if catalog.get("format") != CATALOG_FORMAT or catalog.get("repo") != REPO or catalog.get("revision") != REVISION:
        raise ValueError("Range catalog format/repo/revision 错误")
    entries: dict[tuple[int, int], dict[str, Any]] = {}
    for item in catalog.get("entries", []):
        layer = int(item.get("layer", -1))
        expert = int(item.get("expert", -1))
        if not 1 <= layer <= 92 or not 0 <= expert < EXPERTS:
            raise ValueError("Range catalog layer/expert 越界")
        key = (layer, expert)
        if key in entries:
            raise ValueError("Range catalog layer/expert 重复")
        byte_range = item.get("range", {})
        start = int(byte_range.get("start", -1))
        end = int(byte_range.get("end_inclusive", -1))
        size = int(byte_range.get("bytes", -1))
        if start < 0 or end - start + 1 != size or size != EXPERT_PAGE_BYTES:
            raise ValueError(f"Range catalog {node_name(key)} 字节不一致")
        require_sha256(item.get("source_shard_lfs_sha256"), "source_shard_lfs_sha256")
        require_sha256(item.get("header_tensors_sha256"), "header_tensors_sha256")
        entries[key] = item
    return entries


def tensor_names(layer: int, expert: int) -> list[str]:
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}"
    return [
        f"{prefix}.{weight}.{component}"
        for weight in ("w1", "w2", "w3")
        for component in ("weight_packed", "weight_scale")
    ]


def build_output(
    trace_path: Path,
    task_path: Path,
    synthetic: bool,
    aggregation: dict[str, Any],
    ranges: dict[tuple[int, int], dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for node, community_id in sorted(aggregation["node_to_community"].items()):
        report = next(item for item in aggregation["node_reports"] if item["layer"] == node[0] and item["expert"] == node[1])
        catalog = ranges.get(node)
        item = {
            "community_id": community_id,
            "layer": node[0],
            "expert": node[1],
            "selection_basis": "native_trace_cooccurrence_counterfactual_nll_and_continuous_layer_dependency",
            "evidence": {
                "route_task_coverage": report["route_task_coverage"],
                "counterfactual_task_coverage": report["counterfactual_task_coverage"],
                "positive_task_fraction": report["positive_task_fraction"],
                "mean_delta_nll": report["mean_delta_nll"],
                "categories": report["categories"],
            },
            "expected_page_bytes": EXPERT_PAGE_BYTES,
            "tensor_names": tensor_names(node[0], node[1]),
            "range_status": "exact_pinned_header_candidate" if catalog else "blocked_missing_pinned_header",
            "source_shard": None if catalog is None else catalog["source_shard"],
            "source_shard_lfs_sha256": None if catalog is None else catalog["source_shard_lfs_sha256"],
            "header_tensors_sha256": None if catalog is None else catalog["header_tensors_sha256"],
            "http_range": None if catalog is None else catalog["range"],
            "download_authorized": False,
        }
        candidates.append(item)
    return {
        "format": OUTPUT_FORMAT,
        "status": "dry_run_range_candidates_only",
        "dry_run": True,
        "downloads_performed": False,
        "download_authorized": False,
        "source": {"repo": REPO, "revision": REVISION},
        "trace": {"path": str(trace_path.resolve()), "sha256": sha256_file(trace_path), "synthetic": synthetic},
        "tasks": {"path": str(task_path.resolve()), "sha256": sha256_file(task_path)},
        "thresholds": thresholds,
        "communities": aggregation["communities"],
        "range_candidates": candidates,
        "candidate_count": len(candidates),
        "exact_range_count": sum(item["http_range"] is not None for item in candidates),
        "blocked_range_count": sum(item["http_range"] is None for item in candidates),
        "rejected_node_count": sum(not item["admitted"] for item in aggregation["node_reports"]),
        "claim_limit": "候选来自预冻结 token 的 K3 原生 trace；Range dry-run 不下载权重，也不证明跨模型能力移植。",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = args.workspace_root.resolve()
    source_contract = validate_source_contract(
        args.source_contract,
        workspace_root,
        verify_files=not args.allow_synthetic,
    )
    source_tasks = None if args.allow_synthetic else load_frozen_source_tasks(source_contract, workspace_root)
    tasks, decisions = load_task_contract(args.tasks, source_tasks, args.allow_synthetic)
    traces = read_jsonl(args.trace)
    synthetic = validate_trace(traces, tasks, decisions, args.allow_synthetic)
    if synthetic != bool(args.allow_synthetic):
        raise RuntimeError("--allow-synthetic 只允许合成测试，不能用于真实 trace")
    thresholds = {
        "minimum_route_task_coverage": args.min_task_coverage,
        "minimum_counterfactual_task_coverage": args.min_benefit_tasks,
        "minimum_positive_task_fraction": args.min_positive_task_fraction,
        "minimum_mean_delta_nll": args.min_mean_delta_nll,
        "minimum_edge_task_coverage": args.min_edge_tasks,
        "maximum_continuous_layer_gap": args.max_layer_gap,
    }
    aggregation = aggregate(
        traces,
        args.min_task_coverage,
        args.min_benefit_tasks,
        args.min_positive_task_fraction,
        args.min_mean_delta_nll,
        args.min_edge_tasks,
        args.max_layer_gap,
    )
    ranges = load_range_catalog(args.header_catalog)
    result = build_output(args.trace, args.tasks, synthetic, aggregation, ranges, thresholds)
    write_json(args.output, result, args.force)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="预先冻结并解析 K3 token ID 的任务 JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--header-catalog", type=Path)
    parser.add_argument("--source-contract", type=Path, default=here / "source_contract.json")
    parser.add_argument("--workspace-root", type=Path, default=here.parents[3])
    parser.add_argument("--min-task-coverage", type=int, default=3)
    parser.add_argument("--min-benefit-tasks", type=int, default=3)
    parser.add_argument("--min-positive-task-fraction", type=float, default=0.67)
    parser.add_argument("--min-mean-delta-nll", type=float, default=0.01)
    parser.add_argument("--min-edge-tasks", type=int, default=3)
    parser.add_argument("--max-layer-gap", type=int, default=2)
    parser.add_argument("--allow-synthetic", action="store_true", help="仅 selftest fixture 使用")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.min_task_coverage < 1 or args.min_benefit_tasks < 1 or args.min_edge_tasks < 1:
        parser.error("coverage 阈值必须为正整数")
    if not 0 < args.min_positive_task_fraction <= 1:
        parser.error("positive fraction 必须在 (0,1]")
    if args.min_mean_delta_nll < 0 or args.max_layer_gap < 1:
        parser.error("NLL/gap 阈值非法")
    return args


def main(argv: list[str] | None = None) -> int:
    configure_utf8()
    args = parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_count": result["candidate_count"],
                "exact_range_count": result["exact_range_count"],
                "blocked_range_count": result["blocked_range_count"],
                "downloads_performed": result["downloads_performed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
