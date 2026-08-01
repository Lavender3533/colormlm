"""独立验证 v44 数据的分割、确定性标签、泄漏和 judge 变异拒绝率。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} 不是JSON对象")
        rows.append(value)
    return rows


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("colorlm_v44_dataset_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载生成器: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def semantic_json(content: Any) -> Any | None:
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def judge_action(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if not isinstance(actual, dict) or actual.get("type") != expected.get("type"):
        return False
    if expected["type"] == "tool":
        return actual.get("name") == expected["name"] and actual.get("arguments") == expected["arguments"]
    if expected["type"] == "finish":
        expected_json = semantic_json(expected.get("content"))
        actual_json = semantic_json(actual.get("content"))
        return expected_json is not None and actual_json == expected_json
    return False


def action_matches_declared_schema(task: dict[str, Any], action: dict[str, Any]) -> bool:
    if action["type"] == "finish":
        return isinstance(semantic_json(action.get("content")), dict)
    schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in task["tools"]
    }
    parameters = schemas.get(action.get("name"))
    if parameters is None or not isinstance(action.get("arguments"), dict):
        return False
    arguments = action["arguments"]
    required = set(parameters.get("required", []))
    properties = parameters.get("properties", {})
    if not required.issubset(arguments) or (
        parameters.get("additionalProperties") is False and set(arguments) - set(properties)
    ):
        return False
    type_map = {"string": str, "integer": int, "boolean": bool, "number": (int, float)}
    for name, value in arguments.items():
        schema = properties[name]
        expected_type = type_map.get(schema.get("type"))
        if expected_type is not None and (not isinstance(value, expected_type) or schema.get("type") == "integer" and isinstance(value, bool)):
            return False
        if "enum" in schema and value not in schema["enum"]:
            return False
    return True


def wrong_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "-wrong"
    if value is None:
        return "wrong"
    return {"wrong": True}


def mutations(expected: dict[str, Any]) -> list[dict[str, Any]]:
    if expected["type"] == "tool":
        arguments = expected["arguments"]
        first = next(iter(arguments))
        missing = copy.deepcopy(expected)
        missing["arguments"].pop(first)
        wrong = copy.deepcopy(expected)
        wrong["arguments"][first] = wrong_value(wrong["arguments"][first])
        extra = copy.deepcopy(expected)
        extra["arguments"]["unexpected"] = "value"
        wrong_name = copy.deepcopy(expected)
        wrong_name["name"] += "_wrong"
        return [wrong_name, missing, wrong, extra]
    value = semantic_json(expected["content"])
    assert isinstance(value, dict)
    extra_value = dict(value)
    extra_value["unexpected"] = True
    return [
        {"type": "tool", "name": "wrong_tool", "arguments": {}},
        {"type": "finish", "content": "{}"},
        {"type": "finish", "content": json.dumps(extra_value, ensure_ascii=False)},
        {"type": "finish", "content": "not-json"},
    ]


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        if "tool_call_id" in message:
            message["tool_call_id"] = "CALL"
        for call in message.get("tool_calls") or []:
            call["id"] = "CALL"
    return normalized


def input_fingerprint(task: dict[str, Any]) -> str:
    value = {
        "messages": normalize_messages(task["messages"]),
        "tools": task["tools"],
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_difference_count(left: dict[str, Any], right: dict[str, Any]) -> int:
    keys = set(left) | set(right)
    return sum(left.get(key) != right.get(key) for key in keys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE / "dataset_selfcheck.json")
    args = parser.parse_args()
    generator_path = args.dataset / "generate_v44_dataset.py"
    tasks_path = args.dataset / "trajectory_tasks_v1.jsonl"
    oracle_path = args.dataset / "trajectory_oracle_v1.jsonl"
    contract_path = args.dataset / "dataset_contract.json"
    manifest_path = args.dataset / "dataset_manifest.json"
    generator = load_generator(generator_path)
    tasks = read_jsonl(tasks_path)
    oracles = read_jsonl(oracle_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    text_paths = [generator_path, tasks_path, oracle_path, contract_path, manifest_path, Path(__file__)]
    utf8_rows = []
    for path in text_paths:
        payload = path.read_bytes()
        try:
            payload.decode("utf-8")
            valid = True
        except UnicodeDecodeError:
            valid = False
        bom = payload.startswith(b"\xef\xbb\xbf")
        utf8_rows.append({"file": path.name, "valid_utf8": valid, "has_bom": bom})
        if not valid or bom:
            errors.append(f"UTF-8或BOM错误: {path.name}")

    if manifest["generator_sha256"] != sha256_file(generator_path):
        errors.append("manifest生成器SHA-256不一致")
    for key, path in (("tasks", tasks_path), ("oracle", oracle_path), ("contract", contract_path)):
        if manifest[f"{key}_sha256"] != sha256_file(path):
            errors.append(f"manifest {key} SHA-256不一致")

    rebuilt_tasks, rebuilt_oracles = generator.build_rows()
    reproducible = {
        "tasks": hashlib.sha256(generator.jsonl_bytes(rebuilt_tasks)).hexdigest() == sha256_file(tasks_path),
        "oracle": hashlib.sha256(generator.jsonl_bytes(rebuilt_oracles)).hexdigest() == sha256_file(oracle_path),
        "contract": hashlib.sha256(generator.json_bytes(generator.dataset_contract())).hexdigest() == sha256_file(contract_path),
    }
    if not all(reproducible.values()):
        errors.extend(f"不可复现: {name}" for name, passed in reproducible.items() if not passed)

    task_by_id = {row["id"]: row for row in tasks}
    oracle_by_id = {row["case_id"]: row for row in oracles}
    if len(task_by_id) != len(tasks) or len(oracle_by_id) != len(oracles):
        errors.append("task或oracle ID重复")
    if set(task_by_id) != set(oracle_by_id):
        errors.append("task/oracle ID不是双射")

    expected_tasks = int(contract["scope"]["task_count"])
    expected_groups = int(contract["scope"]["group_count"])
    if len(tasks) != expected_tasks or len({row["group_id"] for row in tasks}) != expected_groups:
        errors.append("任务数或组数不符合合同")
    split_counts = Counter(row["split"] for row in tasks)
    if dict(split_counts) != contract["split_contract"]["expected_task_counts"]:
        errors.append(f"split数量错误: {dict(split_counts)!r}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    capability_clusters: dict[str, set[str]] = defaultdict(set)
    split_capability_clusters: dict[tuple[str, str], set[str]] = defaultdict(set)
    fingerprints: dict[str, set[str]] = defaultdict(set)
    fixtures: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        group_splits[task["group_id"]].add(task["split"])
        cluster_splits[task["template_cluster_id"]].add(task["split"])
        capability_clusters[task["capability"]].add(task["template_cluster_id"])
        split_capability_clusters[(task["split"], task["capability"])].add(task["template_cluster_id"])
        fingerprints[task["split"]].add(input_fingerprint(task))
        fixtures[task["split"]].add(task["leakage_keys"]["fixture_id"])
        if task["leakage_keys"]["value_pool"] != task["split"]:
            errors.append(f"value_pool与split不一致: {task['id']}")
    group_leakage = sum(len(values) > 1 for values in group_splits.values())
    cluster_leakage = sum(len(values) > 1 for values in cluster_splits.values())
    fingerprint_overlap = sum(
        len(fingerprints[left] & fingerprints[right])
        for index, left in enumerate(("train", "validation", "blind"))
        for right in ("train", "validation", "blind")[index + 1 :]
    )
    fixture_overlap = sum(
        len(fixtures[left] & fixtures[right])
        for index, left in enumerate(("train", "validation", "blind"))
        for right in ("train", "validation", "blind")[index + 1 :]
    )
    if group_leakage or cluster_leakage or fingerprint_overlap or fixture_overlap:
        errors.append("检测到group/template/input/fixture跨split泄漏")

    expected_cluster_distribution = {"train": 2, "validation": 1, "blind": 1}
    for capability in generator.CLUSTERS:
        if len(capability_clusters[capability]) != 4:
            errors.append(f"{capability}模板簇数量不是4")
        for split, expected in expected_cluster_distribution.items():
            actual = len(split_capability_clusters[(split, capability)])
            if actual != expected:
                errors.append(f"{capability}/{split}模板簇={actual}，预期{expected}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        groups[task["group_id"]].append(task)
    pair_checks = {
        "groups": len(groups),
        "two_variants": 0,
        "same_split_cluster_tools": 0,
        "one_observation_message_difference": 0,
        "one_state_field_difference": 0,
        "different_expected_action": 0,
        "coarse_state_label_flips": 0,
    }
    for group_id, pair in groups.items():
        pair.sort(key=lambda row: row["variant"])
        if len(pair) != 2 or [row["variant"] for row in pair] != ["a", "b"]:
            errors.append(f"反事实组不是a/b两条: {group_id}")
            continue
        pair_checks["two_variants"] += 1
        left, right = pair
        same_metadata = (
            left["split"] == right["split"]
            and left["template_cluster_id"] == right["template_cluster_id"]
            and left["tools"] == right["tools"]
        )
        if same_metadata:
            pair_checks["same_split_cluster_tools"] += 1
        else:
            errors.append(f"反事实对split/cluster/tools不一致: {group_id}")
        message_differences = sum(a != b for a, b in zip(left["messages"], right["messages"]))
        if len(left["messages"]) == len(right["messages"]) and message_differences == 1:
            pair_checks["one_observation_message_difference"] += 1
        else:
            errors.append(f"反事实对不是单条观测差异: {group_id}")
        left_oracle, right_oracle = oracle_by_id[left["id"]], oracle_by_id[right["id"]]
        state_difference = state_difference_count(
            left_oracle["state_machine"]["state"], right_oracle["state_machine"]["state"]
        )
        if state_difference == 1:
            pair_checks["one_state_field_difference"] += 1
        else:
            errors.append(f"反事实状态差异字段数={state_difference}: {group_id}")
        if left_oracle["expected_action"] != right_oracle["expected_action"]:
            pair_checks["different_expected_action"] += 1
        else:
            errors.append(f"反事实对没有翻转动作: {group_id}")
        pair_checks["coarse_state_label_flips"] += int(left["label"] != right["label"])

    derived_labels = 0
    oracle_commitments = 0
    judge_accepts = 0
    schema_valid_actions = 0
    critical_semantics_verified = 0
    judge_false_rejects = 0
    mutation_count = 0
    mutation_false_accepts = 0
    for case_id, task in task_by_id.items():
        oracle = oracle_by_id[case_id]
        derived = generator.derive_expected_action(
            oracle["capability"],
            oracle["template_key"],
            oracle["state_machine"]["state"],
        )
        if derived == oracle["expected_action"] and generator.action_target(derived) == task["target"]:
            derived_labels += 1
        else:
            errors.append(f"确定性规则不能重建标签: {case_id}")
        commitment = hashlib.sha256((generator.canonical_sorted(oracle) + "\n").encode("utf-8")).hexdigest()
        if commitment == task["oracle_commitment_sha256"]:
            oracle_commitments += 1
        else:
            errors.append(f"oracle commitment错误: {case_id}")
        if judge_action(oracle["expected_action"], oracle["expected_action"]):
            judge_accepts += 1
        else:
            judge_false_rejects += 1
        if action_matches_declared_schema(task, oracle["expected_action"]):
            schema_valid_actions += 1
        else:
            errors.append(f"expected action不符合声明schema: {case_id}")
        if oracle.get("critical_semantics") == generator.critical_semantics(oracle["expected_action"]):
            critical_semantics_verified += 1
        else:
            errors.append(f"关键动作语义不能重建: {case_id}")
        for mutation in mutations(oracle["expected_action"]):
            mutation_count += 1
            mutation_false_accepts += int(judge_action(oracle["expected_action"], mutation))
    if judge_false_rejects or mutation_false_accepts:
        errors.append("judge出现false reject或mutation false accept")

    report = {
        "format": "colorlm-v44-critical-action-dataset-selfcheck-v1",
        "passed": not errors,
        "errors": errors,
        "inputs": {
            "generator_sha256": sha256_file(generator_path),
            "tasks_sha256": sha256_file(tasks_path),
            "oracle_sha256": sha256_file(oracle_path),
            "contract_sha256": sha256_file(contract_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "counts": {
            "tasks": len(tasks),
            "groups": len(groups),
            "template_clusters": len(cluster_splits),
            "splits": dict(sorted(split_counts.items())),
            "capabilities": dict(sorted(Counter(row["capability"] for row in tasks).items())),
            "labels": dict(sorted(Counter(row["label"] for row in tasks).items())),
        },
        "template_clusters": {
            capability: {
                split: sorted(split_capability_clusters[(split, capability)])
                for split in ("train", "validation", "blind")
            }
            for capability in sorted(generator.CLUSTERS)
        },
        "leakage": {
            "group_split_overlap": group_leakage,
            "template_cluster_split_overlap": cluster_leakage,
            "normalized_input_fingerprint_overlap": fingerprint_overlap,
            "fixture_id_split_overlap": fixture_overlap,
        },
        "counterfactual_pairs": pair_checks,
        "deterministic_oracle": {
            "labels_rederived": derived_labels,
            "oracle_commitments_verified": oracle_commitments,
            "actions_matching_declared_tool_schema": schema_valid_actions,
            "critical_semantics_rederived": critical_semantics_verified,
        },
        "judge": {
            "expected_actions_accepted": judge_accepts,
            "false_rejects": judge_false_rejects,
            "mutations_tested": mutation_count,
            "mutation_false_accepts": mutation_false_accepts,
        },
        "reproducibility": reproducible,
        "encoding": utf8_rows,
        "scientific_limits": contract["scientific_limits"],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
