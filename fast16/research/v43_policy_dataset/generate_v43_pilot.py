"""生成 v43 成对工作流状态先导集，并冻结数据与训练合同。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
DATASET = ROOT / "trajectory_tasks_v1.jsonl"
ORACLE = ROOT / "trajectory_oracle_v1.jsonl"
CONTRACT = ROOT / "policy_contract.json"
MANIFEST = ROOT / "dataset_manifest.json"
SOURCE_POLICY = PROJECT / "fast16/research/v29_sequence_policy_head/runtime-v1/policy.json"
SOURCE_WEIGHTS = PROJECT / "fast16/research/v29_sequence_policy_head/runtime-v1/weights.bin"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_frozen(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"冻结文件已存在且内容不同，拒绝覆盖: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS = {
    "read_file": tool("read_file", "读取文本文件", {"path": {"type": "string"}}, ["path"]),
    "list_files": tool(
        "list_files",
        "列出匹配文件",
        {"path": {"type": "string"}, "glob": {"type": "string"}},
        ["path", "glob"],
    ),
    "run_command": tool(
        "run_command",
        "在指定目录运行命令",
        {"command": {"type": "string"}, "cwd": {"type": "string"}},
        ["command", "cwd"],
    ),
    "deploy_config": tool(
        "deploy_config",
        "部署构建产物",
        {
            "artifact": {"type": "string"},
            "environment": {"type": "string", "enum": ["staging", "production"]},
            "account_id": {"type": "string"},
        },
        ["artifact", "environment", "account_id"],
    ),
    "activate_config": tool(
        "activate_config",
        "激活环境配置",
        {"environment": {"type": "string"}},
        ["environment"],
    ),
    "resolve_context": tool(
        "resolve_context",
        "读取当前部署上下文",
        {"workspace": {"type": "string"}},
        ["workspace"],
    ),
    "inspect_ui": tool("inspect_ui", "读取当前窗口可访问性树", {}, []),
    "click_ui": tool(
        "click_ui",
        "点击可访问性控件",
        {"control_id": {"type": "string"}},
        ["control_id"],
    ),
    "ask_user_question": tool(
        "ask_user_question",
        "询问缺失且无法安全推断的信息",
        {"question": {"type": "string"}, "field": {"type": "string"}},
        ["question", "field"],
    ),
}


def tool_call(name: str, arguments: dict[str, Any]) -> str:
    return "<tool_call>\n" + canonical({"name": name, "arguments": arguments}) + "\n</tool_call>"


def assistant_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": canonical(arguments)},
            }
        ],
    }


def split_for(group_index: int) -> str:
    if group_index < 6:
        return "train"
    if group_index < 8:
        return "validation"
    return "test"


def add_case(
    tasks: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    *,
    capability: str,
    group_index: int,
    variant: str,
    family: str,
    label: str,
    messages: list[dict[str, Any]],
    tools: list[str],
    target: str,
    expected_action: dict[str, Any],
) -> None:
    split = split_for(group_index)
    group_id = f"{capability}-{group_index:02d}"
    case_id = f"{group_id}-{variant}"
    task = {
        "id": case_id,
        "split": split,
        "family": family,
        "label": label,
        "capability": capability,
        "group_id": group_id,
        "counterfactual_pair_id": group_id,
        "variant": variant,
        "template_cluster_id": f"{capability}-{split}-cluster-{group_index:02d}",
        "messages": messages,
        "tools": [TOOLS[name] for name in tools],
        "target": target,
    }
    tasks.append(task)
    oracle.append(
        {
            "format": "colorlm-v43-trajectory-oracle-v1",
            "case_id": case_id,
            "group_id": group_id,
            "split": split,
            "capability": capability,
            "state_label": label,
            "canonical_target": target,
            "expected_action": expected_action,
            "label_source": "deterministic-counterfactual-state",
        }
    )


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    projects = ["atlas", "birch", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "iris", "juniper"]
    files = ["src/api.py", "src/cache.ts", "lib/auth.rs", "cmd/main.go", "app/router.py", "pkg/store.go", "ui/form.ts", "core/time.rs", "jobs/sync.py", "web/state.ts"]

    for index, project in enumerate(projects):
        call_id = f"repair-{index}"
        base = [
            {"role": "user", "content": f"把 {project}.zip 部署到生产环境。参数错误时按确定性错误提示修正；信息不足时不要猜。"},
            assistant_call(call_id, "deploy_config", {"artifact": f"{project}.zip", "environment": "prod", "account_id": f"acct-{index:03d}"}),
        ]
        retry = {"artifact": f"{project}.zip", "environment": "production", "account_id": f"acct-{index:03d}"}
        add_case(tasks, oracle, capability="tool_argument_repair", group_index=index, variant="a", family="continue", label="continue_tool", messages=base + [{"role": "tool", "tool_call_id": call_id, "content": "ERROR: environment='prod' 无效；唯一合法修复值为 production。"}], tools=["deploy_config", "ask_user_question"], target=tool_call("deploy_config", retry), expected_action={"type": "tool", "name": "deploy_config", "arguments": retry})
        ask = {"question": "请提供部署所需的 account_id。", "field": "account_id"}
        add_case(tasks, oracle, capability="tool_argument_repair", group_index=index, variant="b", family="clarify", label="ask_user", messages=base + [{"role": "tool", "tool_call_id": call_id, "content": "ERROR: account_id 不存在且没有默认值；必须由用户提供。"}], tools=["deploy_config", "ask_user_question"], target=tool_call("ask_user_question", ask), expected_action={"type": "tool", "name": "ask_user_question", "arguments": ask})

        read_id = f"read-{index}"
        read_base = [
            {"role": "user", "content": f"读取 reports/{project}.json。若结果分片就继续读 next_path；完整时只返回 checksum JSON。"},
            assistant_call(read_id, "read_file", {"path": f"reports/{project}.json"}),
        ]
        next_path = f"reports/{project}.part2.json"
        read_next = {"path": next_path}
        add_case(tasks, oracle, capability="continue_or_finish", group_index=index, variant="a", family="continue", label="continue_tool", messages=read_base + [{"role": "tool", "tool_call_id": read_id, "content": canonical({"complete": False, "next_path": next_path})}], tools=["read_file"], target=tool_call("read_file", read_next), expected_action={"type": "tool", "name": "read_file", "arguments": read_next})
        finish = canonical({"checksum": f"sha256:{index:064x}"})
        add_case(tasks, oracle, capability="continue_or_finish", group_index=index, variant="b", family="finish", label="finish", messages=read_base + [{"role": "tool", "tool_call_id": read_id, "content": canonical({"complete": True, "checksum": f"sha256:{index:064x}"})}], tools=["read_file"], target=finish, expected_action={"type": "finish", "content": finish})

        ctx_id = f"context-{index}"
        ctx_base = [
            {"role": "user", "content": f"在工作区 {project} 部署 build-{index}.tgz；先读取上下文，环境无法唯一确定时询问。"},
            assistant_call(ctx_id, "resolve_context", {"workspace": project}),
        ]
        ask_env = {"question": "请指定部署环境：staging 或 production。", "field": "environment"}
        add_case(tasks, oracle, capability="missing_parameter_clarification", group_index=index, variant="a", family="clarify", label="ask_user", messages=ctx_base + [{"role": "tool", "tool_call_id": ctx_id, "content": canonical({"account_id": f"acct-{index:03d}", "environment": None})}], tools=["resolve_context", "deploy_config", "ask_user_question"], target=tool_call("ask_user_question", ask_env), expected_action={"type": "tool", "name": "ask_user_question", "arguments": ask_env})
        deploy = {"artifact": f"build-{index}.tgz", "environment": "staging", "account_id": f"acct-{index:03d}"}
        add_case(tasks, oracle, capability="missing_parameter_clarification", group_index=index, variant="b", family="continue", label="continue_tool", messages=ctx_base + [{"role": "tool", "tool_call_id": ctx_id, "content": canonical({"account_id": f"acct-{index:03d}", "environment": "staging"})}], tools=["resolve_context", "deploy_config", "ask_user_question"], target=tool_call("deploy_config", deploy), expected_action={"type": "tool", "name": "deploy_config", "arguments": deploy})

        validate_id = f"validate-{index}"
        plan_base = [
            {"role": "user", "content": f"验证 {project}/deploy.yaml，必要时激活 staging；完成且健康后只返回状态 JSON。"},
            assistant_call(validate_id, "run_command", {"command": "validate-config deploy.yaml", "cwd": project}),
        ]
        activate = {"environment": "staging"}
        add_case(tasks, oracle, capability="multi_step_planning", group_index=index, variant="a", family="continue", label="continue_tool", messages=plan_base + [{"role": "tool", "tool_call_id": validate_id, "content": canonical({"valid": True, "activated": False, "health": "unknown"})}], tools=["run_command", "activate_config"], target=tool_call("activate_config", activate), expected_action={"type": "tool", "name": "activate_config", "arguments": activate})
        plan_finish = canonical({"status": "healthy", "environment": "staging"})
        add_case(tasks, oracle, capability="multi_step_planning", group_index=index, variant="b", family="finish", label="finish", messages=plan_base + [{"role": "tool", "tool_call_id": validate_id, "content": canonical({"valid": True, "activated": True, "health": "ok"})}], tools=["run_command", "activate_config"], target=plan_finish, expected_action={"type": "finish", "content": plan_finish})

        inspect_id = f"ui-{index}"
        ui_base = [
            {"role": "user", "content": f"在设置窗口保存 {project} 项目；控件不可用时说明缺少哪个必填项，不要盲点。"},
            assistant_call(inspect_id, "inspect_ui", {}),
        ]
        click = {"control_id": f"save-button-{index}"}
        add_case(tasks, oracle, capability="computer_operation", group_index=index, variant="a", family="continue", label="computer_action", messages=ui_base + [{"role": "tool", "tool_call_id": inspect_id, "content": canonical({"project_name": project, "save": {"id": f"save-button-{index}", "enabled": True}})}], tools=["inspect_ui", "click_ui", "ask_user_question"], target=tool_call("click_ui", click), expected_action={"type": "tool", "name": "click_ui", "arguments": click})
        ask_name = {"question": "请提供项目名称后再保存。", "field": "project_name"}
        add_case(tasks, oracle, capability="computer_operation", group_index=index, variant="b", family="clarify", label="ask_user", messages=ui_base + [{"role": "tool", "tool_call_id": inspect_id, "content": canonical({"project_name": "", "save": {"id": f"save-button-{index}", "enabled": False, "reason": "project_name required"}})}], tools=["inspect_ui", "click_ui", "ask_user_question"], target=tool_call("ask_user_question", ask_name), expected_action={"type": "tool", "name": "ask_user_question", "arguments": ask_name})

        test_id = f"test-{index}"
        cwd = f"repos/{project}"
        debug_base = [
            {"role": "user", "content": f"运行 {project} 的测试；失败时先读报错文件，全部通过时只返回测试状态 JSON。"},
            assistant_call(test_id, "run_command", {"command": "pytest -q", "cwd": cwd}),
        ]
        read_error = {"path": files[index]}
        add_case(tasks, oracle, capability="code_debugging", group_index=index, variant="a", family="continue", label="continue_tool", messages=debug_base + [{"role": "tool", "tool_call_id": test_id, "content": f"FAILED {files[index]}:{20 + index} - AssertionError: boundary mismatch"}], tools=["run_command", "read_file"], target=tool_call("read_file", read_error), expected_action={"type": "tool", "name": "read_file", "arguments": read_error})
        test_finish = canonical({"tests": "passed", "count": 20 + index})
        add_case(tasks, oracle, capability="code_debugging", group_index=index, variant="b", family="finish", label="finish", messages=debug_base + [{"role": "tool", "tool_call_id": test_id, "content": f"{20 + index} passed in 0.{index + 1}s"}], tools=["run_command", "read_file"], target=test_finish, expected_action={"type": "finish", "content": test_finish})

    return tasks, oracle


def validate(tasks: list[dict[str, Any]], oracle: list[dict[str, Any]]) -> dict[str, Any]:
    if len(tasks) != 120 or len(oracle) != 120:
        raise RuntimeError("v43先导集必须恰有120条轨迹")
    ids = [row["id"] for row in tasks]
    if len(set(ids)) != len(ids):
        raise RuntimeError("任务ID重复")
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        counts[split] = sum(row["split"] == split for row in tasks)
    if counts != {"train": 72, "validation": 24, "test": 24}:
        raise RuntimeError(f"split数量错误: {counts}")
    groups: dict[str, set[str]] = {}
    clusters: dict[str, set[str]] = {}
    for row in tasks:
        groups.setdefault(row["group_id"], set()).add(row["split"])
        clusters.setdefault(row["template_cluster_id"], set()).add(row["split"])
    if any(len(value) != 1 for value in groups.values()):
        raise RuntimeError("反事实组跨split泄漏")
    if any(len(value) != 1 for value in clusters.values()):
        raise RuntimeError("模板簇跨split泄漏")
    for group_id in groups:
        pair = [row for row in tasks if row["group_id"] == group_id]
        if len(pair) != 2 or pair[0]["label"] == pair[1]["label"]:
            raise RuntimeError(f"反事实组没有形成动作翻转: {group_id}")
    return {
        "task_count": len(tasks),
        "group_count": len(groups),
        "split_counts": counts,
        "capability_counts": {
            capability: sum(row["capability"] == capability for row in tasks)
            for capability in sorted({row["capability"] for row in tasks})
        },
        "group_leakage": 0,
        "template_cluster_leakage": 0,
    }


def main() -> int:
    tasks, oracle = build()
    stats = validate(tasks, oracle)
    task_text = "".join(canonical(row) + "\n" for row in tasks)
    oracle_text = "".join(canonical(row) + "\n" for row in oracle)
    write_frozen(DATASET, task_text)
    write_frozen(ORACLE, oracle_text)

    contract = {
        "format": "colorlm-v43-paired-policy-contract-v1",
        "date": "2026-08-01",
        "role": "preregistered-before-v43-model-capture",
        "base_model": "fast16/models/ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf",
        "source_tasks": "fast16/research/v43_policy_dataset/trajectory_tasks_v1.jsonl",
        "source_tasks_sha256": sha256_file(DATASET),
        "source_oracle": "fast16/research/v43_policy_dataset/trajectory_oracle_v1.jsonl",
        "source_oracle_sha256": sha256_file(ORACLE),
        "source_candidate_policy": "fast16/research/v29_sequence_policy_head/runtime-v1/policy.json",
        "source_candidate_policy_sha256": sha256_file(SOURCE_POLICY),
        "source_candidate_weights_sha256": sha256_file(SOURCE_WEIGHTS),
        "teacher": {"maximum_target_tokens_per_task": 6, "maximum_prefix_tokens": 4096},
        "fit": {
            "family": "train-only-pca-multiclass-ridge-with-explicit-no-op",
            "hidden_normalization": "per-sample-l2",
            "candidate_source": "fixed-v29-allowlist-intersect-train-tokens",
            "minimum_distinct_train_groups_per_candidate": 3,
            "pca_rank": 8,
            "ridge_lambda": 0.1,
            "correction_strength": 12.0,
            "no_op_rule": "exact-no-op-iff-class-0-is-argmax",
            "parameter_scan_allowed_after_capture": False,
            "development_source": "consumed-v39-20-task-capture-only",
        },
        "offline_gate": {
            "validation_mean_target_nll_delta_max": -0.01,
            "test_mean_target_nll_delta_max": -0.01,
            "minimum_task_wins_minus_losses": 2,
            "maximum_task_mean_nll_regression": 0.03,
            "minimum_exact_no_op_rate": 0.10,
            "maximum_exact_no_op_rate": 0.90,
        },
        "runtime_gate": {
            "test_tasks": 24,
            "minimum_net_fixes": 3,
            "maximum_regressions": 1,
            "no_tools_physical_bypass": True,
            "maximum_speed_regression": 0.05,
        },
        "failure_action": "stop v43 policy candidate; retain frozen dataset and do not scan against validation/test",
    }
    write_frozen(CONTRACT, pretty(contract))
    manifest = {
        "format": "colorlm-v43-paired-policy-dataset-manifest-v1",
        "generator": Path(__file__).name,
        "generator_sha256": sha256_file(Path(__file__)),
        "dataset": DATASET.name,
        "dataset_sha256": sha256_file(DATASET),
        "oracle": ORACLE.name,
        "oracle_sha256": sha256_file(ORACLE),
        "contract": CONTRACT.name,
        "contract_sha256": sha256_file(CONTRACT),
        **stats,
    }
    write_frozen(MANIFEST, pretty(manifest))
    print(pretty(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
