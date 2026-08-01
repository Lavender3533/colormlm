"""生成 v44 跨模板关键动作数据集；标签只由确定性状态规则产生。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
FORMAT_TASK = "colorlm-v44-critical-action-task-v1"
FORMAT_ORACLE = "colorlm-v44-deterministic-state-oracle-v1"
DATASET_VERSION = "v44.0"
GROUPS_PER_CLUSTER = 5

STEMS = {
    "train": ["atlas", "boreal", "coral", "drift", "ember"],
    "validation": ["juniper", "kestrel", "lumen", "marble", "nimbus"],
    "blind": ["quartz", "raven", "solstice", "topaz", "umbra"],
}

CLUSTERS: dict[str, list[tuple[str, str]]] = {
    "tool_argument_repair": [
        ("invalid_enum", "train"),
        ("wrong_scalar_type", "train"),
        ("missing_path_resolution", "validation"),
        ("cross_field_dependency", "blind"),
    ],
    "continue_or_finish": [
        ("paginated_read", "train"),
        ("test_result", "train"),
        ("artifact_verification", "validation"),
        ("asynchronous_job", "blind"),
    ],
    "missing_parameter_clarification": [
        ("deployment_environment", "train"),
        ("overwrite_confirmation", "train"),
        ("schedule_timezone", "validation"),
        ("repository_scope", "blind"),
    ],
    "multi_step_planning": [
        ("validate_then_activate", "train"),
        ("backup_then_migrate", "train"),
        ("build_then_deploy", "validation"),
        ("health_then_rollback", "blind"),
    ],
    "computer_operation": [
        ("save_dialog", "train"),
        ("overwrite_dialog", "train"),
        ("filter_menu", "validation"),
        ("submit_form", "blind"),
    ],
    "code_debugging": [
        ("compiler_diagnostic", "train"),
        ("unit_test_failure", "train"),
        ("lint_recovery", "validation"),
        ("post_patch_regression", "blind"),
    ],
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def canonical_sorted(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_seed(case_id: str) -> int:
    return int.from_bytes(hashlib.sha256(case_id.encode("utf-8")).digest()[:4], "little")


def function_tool(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> dict[str, Any]:
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


def tool_action(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool", "name": name, "arguments": arguments}


def finish_action(value: dict[str, Any]) -> dict[str, Any]:
    return {"type": "finish", "content": canonical(value)}


def action_target(action: dict[str, Any]) -> str:
    if action["type"] == "tool":
        return "<tool_call>\n" + canonical({"name": action["name"], "arguments": action["arguments"]}) + "\n</tool_call>"
    if action["type"] == "finish":
        return action["content"]
    raise ValueError(f"未知动作类型: {action!r}")


def state_label(action: dict[str, Any]) -> str:
    if action["type"] == "finish":
        return "finish"
    if action["name"] == "ask_user_question":
        return "ask_user"
    return "continue_tool"


def critical_semantics(action: dict[str, Any]) -> list[dict[str, Any]]:
    if action["type"] == "tool":
        return [
            {"role": "action_prefix", "value": "tool"},
            {"role": "tool_name", "value": action["name"]},
            *(
                {"role": "argument", "key": key, "value": value}
                for key, value in action["arguments"].items()
            ),
        ]
    content = json.loads(action["content"])
    return [
        {"role": "finish_field", "key": key, "value": value}
        for key, value in content.items()
    ]


def call_message(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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


def result_message(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def ask_tool() -> dict[str, Any]:
    return function_tool(
        "ask_user_question",
        "询问缺失且无法从当前状态唯一推出的信息",
        {"field": {"type": "string"}, "question": {"type": "string"}},
        ["field", "question"],
    )


def derive_expected_action(capability: str, template_key: str, state: dict[str, Any]) -> dict[str, Any]:
    """唯一标签函数；生成器和独立自检均从这里重新推导答案。"""

    if capability == "tool_argument_repair":
        if template_key == "invalid_enum":
            if state["correct_environment"] is not None:
                return tool_action(
                    "deploy_config",
                    {
                        "artifact": state["artifact"],
                        "environment": state["correct_environment"],
                        "account_id": state["account_id"],
                    },
                )
            return tool_action(
                "ask_user_question",
                {"field": "environment", "question": "请选择 staging 或 production 部署环境。"},
            )
        if template_key == "wrong_scalar_type":
            if state["timeout_seconds"] is not None:
                return tool_action(
                    "run_job", {"job": state["job"], "timeout_seconds": state["timeout_seconds"]}
                )
            return tool_action(
                "ask_user_question",
                {"field": "timeout_seconds", "question": "任务超时秒数应设为多少？"},
            )
        if template_key == "missing_path_resolution":
            if len(state["path_candidates"]) == 1:
                return tool_action(
                    "archive_file",
                    {"source_path": state["path_candidates"][0], "destination": state["destination"]},
                )
            return tool_action(
                "ask_user_question",
                {"field": "source_path", "question": "请选择要归档的源文件路径。"},
            )
        if template_key == "cross_field_dependency":
            if state["correct_region"] is not None:
                return tool_action(
                    "publish_release",
                    {
                        "project_id": state["project_id"],
                        "account": state["account"],
                        "region": state["correct_region"],
                    },
                )
            return tool_action(
                "ask_user_question",
                {"field": "region", "question": "请选择该账户允许的发布区域。"},
            )

    if capability == "continue_or_finish":
        if template_key == "paginated_read":
            if state["complete"]:
                return finish_action({"checksum": state["checksum"]})
            return tool_action("read_file", {"path": state["next_path"]})
        if template_key == "test_result":
            if state["passed"]:
                return finish_action({"status": "tests_passed", "suite": state["suite"]})
            return tool_action("read_file", {"path": state["failure_path"]})
        if template_key == "artifact_verification":
            if state["verified"]:
                return finish_action({"status": "verified", "artifact": state["artifact"]})
            return tool_action("verify_artifact", {"artifact": state["artifact"]})
        if template_key == "asynchronous_job":
            if state["status"] == "succeeded":
                return finish_action({"status": "complete", "job_id": state["job_id"]})
            return tool_action("get_job_status", {"job_id": state["job_id"]})

    if capability == "missing_parameter_clarification":
        if template_key == "deployment_environment":
            if state["environment"] is None:
                return tool_action(
                    "ask_user_question",
                    {"field": "environment", "question": "部署到 staging 还是 production？"},
                )
            return tool_action(
                "deploy_service", {"service": state["service"], "environment": state["environment"]}
            )
        if template_key == "overwrite_confirmation":
            if state["overwrite"] is None:
                return tool_action(
                    "ask_user_question",
                    {"field": "overwrite", "question": "目标已存在，是否允许覆盖？"},
                )
            return tool_action(
                "write_file",
                {"path": state["path"], "content": state["content"], "overwrite": state["overwrite"]},
            )
        if template_key == "schedule_timezone":
            if state["timezone"] is None:
                return tool_action(
                    "ask_user_question",
                    {"field": "timezone", "question": "该计划使用哪个时区？"},
                )
            return tool_action(
                "schedule_job",
                {"job": state["job"], "time": state["time"], "timezone": state["timezone"]},
            )
        if template_key == "repository_scope":
            if state["repository"] is None:
                return tool_action(
                    "ask_user_question",
                    {"field": "repository", "question": "应在哪个仓库中搜索该符号？"},
                )
            return tool_action(
                "search_symbol", {"repository": state["repository"], "symbol": state["symbol"]}
            )

    if capability == "multi_step_planning":
        if template_key == "validate_then_activate":
            return (
                tool_action("activate_config", {"environment": state["environment"]})
                if state["validated"]
                else tool_action("validate_config", {"path": state["path"]})
            )
        if template_key == "backup_then_migrate":
            return (
                tool_action("migrate_database", {"database": state["database"], "target": state["target"]})
                if state["backup_complete"]
                else tool_action("create_backup", {"database": state["database"]})
            )
        if template_key == "build_then_deploy":
            return (
                tool_action("deploy_artifact", {"artifact": state["artifact"], "environment": state["environment"]})
                if state["build_passed"]
                else tool_action("read_file", {"path": state["failure_path"]})
            )
        if template_key == "health_then_rollback":
            return (
                finish_action({"status": "healthy", "deployment_id": state["deployment_id"]})
                if state["healthy"]
                else tool_action("rollback_deployment", {"deployment_id": state["deployment_id"]})
            )

    if capability == "computer_operation":
        if template_key == "save_dialog":
            return (
                tool_action("ui_click", {"target": "Save"})
                if state["filename"] == state["target_filename"]
                else tool_action("ui_type", {"target": "Filename", "text": state["target_filename"]})
            )
        if template_key == "overwrite_dialog":
            return tool_action("ui_click", {"target": "Replace" if state["overwrite_allowed"] else "Cancel"})
        if template_key == "filter_menu":
            return (
                tool_action("ui_click", {"target": "Apply"})
                if state["error_selected"] and not state["warning_selected"]
                else tool_action("ui_click", {"target": "Error"})
            )
        if template_key == "submit_form":
            return (
                tool_action("ui_click", {"target": "Submit"})
                if state["email"] == state["target_email"]
                else tool_action("ui_type", {"target": "Email", "text": state["target_email"]})
            )

    if capability == "code_debugging":
        if template_key == "compiler_diagnostic":
            return (
                finish_action({"status": "compile_passed", "project": state["project"]})
                if state["passed"]
                else tool_action("read_file", {"path": state["diagnostic_path"]})
            )
        if template_key == "unit_test_failure":
            return (
                finish_action({"status": "tests_passed", "project": state["project"]})
                if state["passed"]
                else tool_action("read_file", {"path": state["failure_path"]})
            )
        if template_key == "lint_recovery":
            return (
                tool_action("run_command", {"command": state["fix_command"], "cwd": state["cwd"]})
                if state["autofixable"]
                else tool_action("read_file", {"path": state["failure_path"]})
            )
        if template_key == "post_patch_regression":
            return (
                finish_action({"status": "verified", "patch_id": state["patch_id"]})
                if not state["regression"]
                else tool_action("revert_patch", {"patch_id": state["patch_id"]})
            )
    raise ValueError(f"没有状态规则: capability={capability}, template={template_key}")


def common_task_payload(
    capability: str,
    key: str,
    split: str,
    stem: str,
    index: int,
    variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """返回 messages、tools、state。成对任务只允许最后一个tool-result事实不同。"""

    flag = variant == "b"
    call_id = f"v44-{capability[:3]}-{key[:3]}-{index}"

    if capability == "tool_argument_repair" and key == "invalid_enum":
        state = {
            "artifact": f"{stem}.zip",
            "account_id": f"acct-{index:03d}",
            "correct_environment": "production" if not flag else None,
        }
        result = (
            "ERROR: environment=prod无效；唯一合法修复值为production。"
            if not flag else "ERROR: environment=prod无效；staging和production都合法，必须由用户选择。"
        )
        tools = [
            function_tool("deploy_config", "部署产物", {"artifact": {"type": "string"}, "environment": {"type": "string", "enum": ["staging", "production"]}, "account_id": {"type": "string"}}, ["artifact", "environment", "account_id"]),
            ask_tool(),
        ]
        messages = [
            {"role": "user", "content": f"部署{stem}.zip；按工具的确定性错误修正，信息不足则询问。"},
            call_message(call_id, "deploy_config", {"artifact": f"{stem}.zip", "environment": "prod", "account_id": f"acct-{index:03d}"}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "tool_argument_repair" and key == "wrong_scalar_type":
        state = {"job": f"{stem}-index", "timeout_seconds": 45 + index if not flag else None}
        result = (
            f"ERROR: timeout_seconds必须为整数；本任务唯一修复值为{45 + index}。"
            if not flag else "ERROR: timeout_seconds缺失且没有安全默认值，必须询问用户。"
        )
        tools = [
            function_tool("run_job", "运行后台任务", {"job": {"type": "string"}, "timeout_seconds": {"type": "integer"}}, ["job", "timeout_seconds"]),
            ask_tool(),
        ]
        messages = [
            {"role": "user", "content": f"运行{stem}-index任务；参数错误时只按工具给出的唯一事实继续。"},
            call_message(call_id, "run_job", {"job": f"{stem}-index", "timeout_seconds": "fast"}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "tool_argument_repair" and key == "missing_path_resolution":
        candidates = [f"build/{stem}.tar"] if not flag else [f"build/{stem}.tar", f"dist/{stem}.tar"]
        state = {"path_candidates": candidates, "destination": f"archive/{stem}.tar"}
        result = "ERROR: source_path不存在；候选=" + canonical(candidates)
        tools = [
            function_tool("archive_file", "归档文件", {"source_path": {"type": "string"}, "destination": {"type": "string"}}, ["source_path", "destination"]),
            ask_tool(),
        ]
        messages = [
            {"role": "user", "content": f"归档{stem}构建产物；只有唯一候选时才能自动修正路径。"},
            call_message(call_id, "archive_file", {"source_path": f"out/{stem}.tar", "destination": f"archive/{stem}.tar"}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "tool_argument_repair" and key == "cross_field_dependency":
        state = {"project_id": f"proj-{stem}", "account": f"team-{index}", "correct_region": "ap-northeast-1" if not flag else None}
        result = (
            "ERROR: account与region不兼容；唯一合法region为ap-northeast-1。"
            if not flag else "ERROR: account允许多个region，无法唯一修复，必须由用户选择。"
        )
        tools = [
            function_tool("publish_release", "发布版本", {"project_id": {"type": "string"}, "account": {"type": "string"}, "region": {"type": "string"}}, ["project_id", "account", "region"]),
            ask_tool(),
        ]
        messages = [
            {"role": "user", "content": f"发布{stem}版本；只修复错误明确给出的字段。"},
            call_message(call_id, "publish_release", {"project_id": f"proj-{stem}", "account": f"team-{index}", "region": "global"}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "continue_or_finish" and key == "paginated_read":
        state = {"complete": flag, "next_path": f"reports/{stem}.part2.json", "checksum": f"sha256:{index:064x}"}
        result = canonical({"complete": flag, **({"checksum": state["checksum"]} if flag else {"next_path": state["next_path"]})})
        tools = [function_tool("read_file", "读取文件", {"path": {"type": "string"}}, ["path"])]
        messages = [
            {"role": "user", "content": f"读取reports/{stem}.json；分片则继续，完整则只返回checksum JSON。"},
            call_message(call_id, "read_file", {"path": f"reports/{stem}.json"}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "continue_or_finish" and key == "test_result":
        state = {"passed": flag, "suite": f"{stem}-api", "failure_path": f"src/{stem}_handler.py"}
        result = canonical({"passed": flag, **({} if flag else {"first_failure": state["failure_path"]})})
        tools = [
            function_tool("run_command", "运行命令", {"command": {"type": "string"}, "cwd": {"type": "string"}}, ["command", "cwd"]),
            function_tool("read_file", "读取文件", {"path": {"type": "string"}}, ["path"]),
        ]
        messages = [
            {"role": "user", "content": f"运行{stem}测试；失败读取首个源文件，通过则结束。"},
            call_message(call_id, "run_command", {"command": "pytest -q", "cwd": stem}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "continue_or_finish" and key == "artifact_verification":
        state = {"verified": flag, "artifact": f"dist/{stem}.bin"}
        result = canonical({"uploaded": True, "verified": flag, "artifact": state["artifact"]})
        tools = [
            function_tool("upload_artifact", "上传产物", {"artifact": {"type": "string"}}, ["artifact"]),
            function_tool("verify_artifact", "验证产物", {"artifact": {"type": "string"}}, ["artifact"]),
        ]
        messages = [
            {"role": "user", "content": f"上传并验证{stem}产物；验证完成后才结束。"},
            call_message(call_id, "upload_artifact", {"artifact": state["artifact"]}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "continue_or_finish" and key == "asynchronous_job":
        state = {"status": "succeeded" if flag else "running", "job_id": f"job-{stem}-{index}"}
        result = canonical(state)
        tools = [function_tool("get_job_status", "查询任务状态", {"job_id": {"type": "string"}}, ["job_id"])]
        messages = [
            {"role": "user", "content": f"等待{stem}后台任务完成；运行中继续查询，成功后结束。"},
            call_message(call_id, "get_job_status", {"job_id": state["job_id"]}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "missing_parameter_clarification" and key == "deployment_environment":
        state = {"service": f"{stem}-api", "environment": "staging" if flag else None}
        result = canonical({"service": state["service"], **({"environment": state["environment"]} if flag else {})})
        tools = [
            function_tool("read_file", "读取请求", {"path": {"type": "string"}}, ["path"]),
            function_tool("deploy_service", "部署服务", {"service": {"type": "string"}, "environment": {"type": "string"}}, ["service", "environment"]),
            ask_tool(),
        ]
        messages = [
            {"role": "user", "content": f"读取{stem}部署请求；缺环境必须询问，否则继续部署。"},
            call_message(call_id, "read_file", {"path": f"requests/{stem}.json"}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "missing_parameter_clarification" and key == "overwrite_confirmation":
        state = {"path": f"output/{stem}.txt", "content": f"ready-{index}", "overwrite": True if flag else None}
        result = canonical({"exists": True, **({"overwrite": True} if flag else {})})
        tools = [
            function_tool("file_info", "读取文件状态", {"path": {"type": "string"}}, ["path"]),
            function_tool("write_file", "写文件", {"path": {"type": "string"}, "content": {"type": "string"}, "overwrite": {"type": "boolean"}}, ["path", "content", "overwrite"]),
            ask_tool(),
        ]
        messages = [
            {"role": "user", "content": f"写入output/{stem}.txt；目标存在时没有覆盖许可就先问。"},
            call_message(call_id, "file_info", {"path": state["path"]}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "missing_parameter_clarification" and key == "schedule_timezone":
        state = {"job": f"{stem}-backup", "time": f"0{index}:30", "timezone": "Asia/Tokyo" if flag else None}
        result = canonical({"job": state["job"], "time": state["time"], **({"timezone": state["timezone"]} if flag else {})})
        tools = [
            function_tool("read_file", "读取计划", {"path": {"type": "string"}}, ["path"]),
            function_tool("schedule_job", "设置计划", {"job": {"type": "string"}, "time": {"type": "string"}, "timezone": {"type": "string"}}, ["job", "time", "timezone"]),
            ask_tool(),
        ]
        messages = [
            {"role": "user", "content": f"按{stem}计划调度任务；时区缺失时不能猜。"},
            call_message(call_id, "read_file", {"path": f"schedules/{stem}.json"}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "missing_parameter_clarification" and key == "repository_scope":
        state = {"repository": f"org/{stem}" if flag else None, "symbol": f"Route{index}"}
        result = canonical({"symbol": state["symbol"], **({"repository": state["repository"]} if flag else {})})
        tools = [
            function_tool("read_file", "读取搜索单", {"path": {"type": "string"}}, ["path"]),
            function_tool("search_symbol", "搜索符号", {"repository": {"type": "string"}, "symbol": {"type": "string"}}, ["repository", "symbol"]),
            ask_tool(),
        ]
        messages = [
            {"role": "user", "content": f"执行{stem}符号搜索；仓库范围缺失时先澄清。"},
            call_message(call_id, "read_file", {"path": f"search/{stem}.json"}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "multi_step_planning" and key == "validate_then_activate":
        state = {"validated": flag, "path": f"config/{stem}.yaml", "environment": "staging"}
        result = canonical({"path": state["path"], "validated": flag})
        tools = [
            function_tool("inspect_config", "读取配置状态", {"path": {"type": "string"}}, ["path"]),
            function_tool("validate_config", "验证配置", {"path": {"type": "string"}}, ["path"]),
            function_tool("activate_config", "激活配置", {"environment": {"type": "string"}}, ["environment"]),
        ]
        messages = [
            {"role": "user", "content": f"先验证config/{stem}.yaml，再激活staging。只执行当前下一步。"},
            call_message(call_id, "inspect_config", {"path": state["path"]}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "multi_step_planning" and key == "backup_then_migrate":
        state = {"backup_complete": flag, "database": f"{stem}_db", "target": f"v{index + 2}"}
        result = canonical({"database": state["database"], "backup_complete": flag})
        tools = [
            function_tool("database_state", "读取数据库状态", {"database": {"type": "string"}}, ["database"]),
            function_tool("create_backup", "创建备份", {"database": {"type": "string"}}, ["database"]),
            function_tool("migrate_database", "迁移数据库", {"database": {"type": "string"}, "target": {"type": "string"}}, ["database", "target"]),
        ]
        messages = [
            {"role": "user", "content": f"迁移{stem}_db到{state['target']}，备份是硬前置条件。"},
            call_message(call_id, "database_state", {"database": state["database"]}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "multi_step_planning" and key == "build_then_deploy":
        state = {"build_passed": flag, "artifact": f"dist/{stem}.zip", "environment": "canary", "failure_path": f"src/{stem}.ts"}
        result = canonical({"passed": flag, **({"artifact": state["artifact"]} if flag else {"failure_path": state["failure_path"]})})
        tools = [
            function_tool("run_command", "运行构建", {"command": {"type": "string"}, "cwd": {"type": "string"}}, ["command", "cwd"]),
            function_tool("read_file", "读取文件", {"path": {"type": "string"}}, ["path"]),
            function_tool("deploy_artifact", "部署产物", {"artifact": {"type": "string"}, "environment": {"type": "string"}}, ["artifact", "environment"]),
        ]
        messages = [
            {"role": "user", "content": f"构建{stem}；失败先检查源文件，成功才部署canary。"},
            call_message(call_id, "run_command", {"command": "npm run build", "cwd": stem}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "multi_step_planning" and key == "health_then_rollback":
        state = {"healthy": flag, "deployment_id": f"dep-{stem}-{index}"}
        result = canonical({"healthy": flag, "deployment_id": state["deployment_id"]})
        tools = [
            function_tool("health_check", "检查健康", {"deployment_id": {"type": "string"}}, ["deployment_id"]),
            function_tool("rollback_deployment", "回滚部署", {"deployment_id": {"type": "string"}}, ["deployment_id"]),
        ]
        messages = [
            {"role": "user", "content": f"检查{stem}部署；健康就结束，不健康立即回滚。"},
            call_message(call_id, "health_check", {"deployment_id": state["deployment_id"]}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "computer_operation" and key == "save_dialog":
        state = {"filename": f"{stem}.txt" if flag else "", "target_filename": f"{stem}.txt"}
        result = canonical({"dialog": "Save As", "filename": state["filename"], "focused": "Filename", "save_enabled": True})
        tools = [
            function_tool("ui_state", "读取界面状态", {}, []),
            function_tool("ui_type", "输入文本", {"target": {"type": "string"}, "text": {"type": "string"}}, ["target", "text"]),
            function_tool("ui_click", "点击控件", {"target": {"type": "string"}}, ["target"]),
        ]
        messages = [
            {"role": "user", "content": f"在保存对话框中把文件保存为{stem}.txt；每次只做下一动作。"},
            call_message(call_id, "ui_state", {}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "computer_operation" and key == "overwrite_dialog":
        state = {"overwrite_allowed": flag, "file": f"{stem}.json"}
        result = canonical({"dialog": "Confirm Replace", "file": state["file"], "user_policy": "replace" if flag else "preserve"})
        tools = [
            function_tool("ui_state", "读取界面状态", {}, []),
            function_tool("ui_click", "点击控件", {"target": {"type": "string"}}, ["target"]),
        ]
        messages = [
            {"role": "user", "content": f"处理{stem}.json覆盖确认；遵循界面状态中的用户策略。"},
            call_message(call_id, "ui_state", {}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "computer_operation" and key == "filter_menu":
        state = {"error_selected": flag, "warning_selected": False, "view": f"{stem}-logs"}
        result = canonical({"menu": "Filter", "Error": flag, "Warning": False, "Apply": "enabled"})
        tools = [
            function_tool("ui_state", "读取界面状态", {}, []),
            function_tool("ui_click", "点击控件", {"target": {"type": "string"}}, ["target"]),
        ]
        messages = [
            {"role": "user", "content": f"在{stem}日志筛选中只显示Error；每次只做下一动作。"},
            call_message(call_id, "ui_state", {}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "computer_operation" and key == "submit_form":
        state = {"email": f"ops+{stem}@example.test" if flag else "", "target_email": f"ops+{stem}@example.test"}
        result = canonical({"form": "Release", "email": state["email"], "submit_enabled": True})
        tools = [
            function_tool("ui_state", "读取界面状态", {}, []),
            function_tool("ui_type", "输入文本", {"target": {"type": "string"}, "text": {"type": "string"}}, ["target", "text"]),
            function_tool("ui_click", "点击控件", {"target": {"type": "string"}}, ["target"]),
        ]
        messages = [
            {"role": "user", "content": f"提交{stem}发布表单，Email必须是ops+{stem}@example.test。"},
            call_message(call_id, "ui_state", {}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "code_debugging" and key == "compiler_diagnostic":
        state = {"passed": flag, "project": stem, "diagnostic_path": f"src/{stem}_router.rs"}
        result = canonical({"passed": flag, **({} if flag else {"diagnostic_path": state["diagnostic_path"], "code": "E0308"})})
        tools = [
            function_tool("run_command", "运行命令", {"command": {"type": "string"}, "cwd": {"type": "string"}}, ["command", "cwd"]),
            function_tool("read_file", "读取文件", {"path": {"type": "string"}}, ["path"]),
        ]
        messages = [
            {"role": "user", "content": f"检查{stem}编译；失败先读取诊断文件，通过则结束。"},
            call_message(call_id, "run_command", {"command": "cargo check", "cwd": stem}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "code_debugging" and key == "unit_test_failure":
        state = {"passed": flag, "project": stem, "failure_path": f"src/{stem}_service.py"}
        result = canonical({"passed": flag, **({} if flag else {"failure_path": state["failure_path"], "test": f"test_{stem}_boundary"})})
        tools = [
            function_tool("run_command", "运行命令", {"command": {"type": "string"}, "cwd": {"type": "string"}}, ["command", "cwd"]),
            function_tool("read_file", "读取文件", {"path": {"type": "string"}}, ["path"]),
        ]
        messages = [
            {"role": "user", "content": f"运行{stem}单测；失败读取首个归因源文件，通过则结束。"},
            call_message(call_id, "run_command", {"command": "pytest -q", "cwd": stem}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "code_debugging" and key == "lint_recovery":
        state = {"autofixable": flag, "fix_command": "npm run lint -- --fix", "cwd": stem, "failure_path": f"src/{stem}Panel.tsx"}
        result = canonical({"passed": False, "autofixable": flag, "failure_path": state["failure_path"]})
        tools = [
            function_tool("run_command", "运行命令", {"command": {"type": "string"}, "cwd": {"type": "string"}}, ["command", "cwd"]),
            function_tool("read_file", "读取文件", {"path": {"type": "string"}}, ["path"]),
        ]
        messages = [
            {"role": "user", "content": f"修复{stem} lint；可自动修复则执行固定命令，否则先读源文件。"},
            call_message(call_id, "run_command", {"command": "npm run lint", "cwd": stem}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    if capability == "code_debugging" and key == "post_patch_regression":
        state = {"regression": not flag, "patch_id": f"patch-{stem}-{index}"}
        result = canonical({"patch_id": state["patch_id"], "regression": state["regression"], "hidden_tests_passed": flag})
        tools = [
            function_tool("run_command", "运行回归测试", {"command": {"type": "string"}, "cwd": {"type": "string"}}, ["command", "cwd"]),
            function_tool("revert_patch", "回滚补丁", {"patch_id": {"type": "string"}}, ["patch_id"]),
        ]
        messages = [
            {"role": "user", "content": f"验证{stem}补丁；隐藏回归则回滚，全部通过才结束。"},
            call_message(call_id, "run_command", {"command": "python -m pytest -q", "cwd": stem}),
            result_message(call_id, result),
        ]
        return messages, tools, state

    raise ValueError(f"没有模板生成器: capability={capability}, key={key}")


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    for capability, clusters in CLUSTERS.items():
        for key, split in clusters:
            cluster_id = f"{capability}.{key}.v1"
            for index, stem in enumerate(STEMS[split]):
                group_id = f"{capability}-{key}-{index:02d}"
                for variant in ("a", "b"):
                    case_id = f"{group_id}-{variant}"
                    messages, tools, state = common_task_payload(
                        capability, key, split, stem, index, variant
                    )
                    expected = derive_expected_action(capability, key, state)
                    label = state_label(expected)
                    target = action_target(expected)
                    oracle = {
                        "format": FORMAT_ORACLE,
                        "dataset_version": DATASET_VERSION,
                        "case_id": case_id,
                        "group_id": group_id,
                        "split": split,
                        "capability": capability,
                        "template_cluster_id": cluster_id,
                        "template_key": key,
                        "variant": variant,
                        "state_machine": {
                            "machine_id": f"{capability}.{key}.deterministic-v1",
                            "current_state": "critical_decision",
                            "state": state,
                            "on_expected": "terminal" if label == "finish" else "await_tool_result",
                        },
                        "state_label": label,
                        "expected_action": expected,
                        "canonical_target": target,
                        "critical_semantics": critical_semantics(expected),
                        "supervision": {
                            "mode": "complete-critical-action-spans",
                            "first_n_token_window": None,
                            "token_spans_must_be_derived_from_full_canonical_target": True,
                        },
                        "label_source": "deterministic-state-machine",
                        "judge": {
                            "type": "exact-semantic-action-v1",
                            "accept_json_key_reordering": True,
                            "allow_extra_text": False,
                            "allow_additional_arguments": False,
                        },
                    }
                    oracle_commitment = sha256_bytes((canonical_sorted(oracle) + "\n").encode("utf-8"))
                    task = {
                        "format": FORMAT_TASK,
                        "dataset_version": DATASET_VERSION,
                        "id": case_id,
                        "split": split,
                        "family": "finish" if label == "finish" else "clarify" if label == "ask_user" else "continue",
                        "label": label,
                        "capability": capability,
                        "group_id": group_id,
                        "counterfactual_pair_id": group_id,
                        "variant": variant,
                        "template_cluster_id": cluster_id,
                        "generator_seed": deterministic_seed(case_id),
                        "max_output_tokens": 96,
                        "leakage_keys": {
                            "fixture_id": f"{cluster_id}.{stem}.{index}",
                            "value_pool": split,
                            "template_cluster": cluster_id,
                        },
                        "messages": messages,
                        "tools": tools,
                        "target": target,
                        "oracle_commitment_sha256": oracle_commitment,
                    }
                    tasks.append(task)
                    oracles.append(oracle)
    return tasks, oracles


def dataset_contract() -> dict[str, Any]:
    return {
        "format": "colorlm-v44-critical-action-dataset-contract-v1",
        "date": "2026-08-01",
        "status": "frozen-before-model-capture",
        "dataset_version": DATASET_VERSION,
        "scope": {
            "capabilities": list(CLUSTERS),
            "task_count": 240,
            "group_count": 120,
            "counterfactual_variants_per_group": 2,
            "groups_per_template_cluster": GROUPS_PER_CLUSTER,
            "template_clusters_per_capability": 4,
        },
        "split_contract": {
            "unit": "template_cluster_id",
            "train_clusters_per_capability": 2,
            "validation_clusters_per_capability": 1,
            "blind_clusters_per_capability": 1,
            "expected_task_counts": {"train": 120, "validation": 60, "blind": 60},
            "group_or_cluster_overlap_allowed": False,
            "target_token_overlap_allowed": True,
            "common_tool_protocol_names_allowed": True,
        },
        "label_contract": {
            "source": "derive_expected_action(capability, template_key, state)",
            "teacher_or_donor_labels_allowed": False,
            "post_hoc_human_labels_allowed": False,
            "online_text_keyword_routing_allowed": False,
            "counterfactual_pair_must_change_expected_action": True,
            "pair_must_differ_in_exactly_one_observation_message": True,
        },
        "critical_action_supervision": {
            "source": "full canonical target",
            "first_n_token_truncation_allowed": False,
            "semantic_roles": ["action_prefix", "tool_name", "argument", "finish_field"],
            "token_spans_are_added_only_after_byte_exact_tokenizer_mapping": True,
            "maximum_generation_tokens": 96,
        },
        "judge_contract": {
            "type": "exact-semantic-action-v1",
            "expected_action_must_pass": True,
            "minimum_rejected_mutations_per_case": 4,
            "mutation_types": ["wrong_action_type_or_name", "missing_field", "wrong_value", "extra_field"],
        },
        "integrity_gate": {
            "utf8_without_bom": True,
            "byte_reproducible_generation": True,
            "task_oracle_id_bijection": True,
            "template_cluster_leakage": 0,
            "group_leakage": 0,
            "normalized_input_fingerprint_leakage": 0,
            "judge_false_accepts": 0,
            "judge_false_rejects": 0,
        },
        "scientific_limits": [
            "这是确定性合成关键动作集，不是实际Claude Code仓库任务成功率。",
            "computer_operation使用结构化可访问性状态，不含截图视觉理解。",
            "code_debugging只监督下一关键动作，不执行真实补丁和隐藏单元测试。",
            "blind表示模板簇隔离但目标仍在本地文件中，不是密码学隐藏盲测。",
            "每个validation/blind能力目前各只有一个模板簇，适合短门，不足以支持前沿模型声明。",
            "精确语义judge假设唯一动作；真实环境中的多条等价计划需要另建终态模拟器。",
        ],
    }


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((canonical(row) + "\n").encode("utf-8") for row in rows)


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def build_manifest(
    output: Path,
    tasks: list[dict[str, Any]],
    oracles: list[dict[str, Any]],
) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in tasks)
    capability_counts = Counter(row["capability"] for row in tasks)
    cluster_counts = Counter(row["capability"] for row in tasks for _ in [row["template_cluster_id"]])
    unique_clusters = {row["template_cluster_id"] for row in tasks}
    return {
        "format": "colorlm-v44-critical-action-dataset-manifest-v1",
        "dataset_version": DATASET_VERSION,
        "generator": Path(__file__).name,
        "generator_sha256": sha256_file(Path(__file__)),
        "tasks": "trajectory_tasks_v1.jsonl",
        "tasks_sha256": sha256_file(output / "trajectory_tasks_v1.jsonl"),
        "oracle": "trajectory_oracle_v1.jsonl",
        "oracle_sha256": sha256_file(output / "trajectory_oracle_v1.jsonl"),
        "contract": "dataset_contract.json",
        "contract_sha256": sha256_file(output / "dataset_contract.json"),
        "task_count": len(tasks),
        "group_count": len({row["group_id"] for row in tasks}),
        "template_cluster_count": len(unique_clusters),
        "split_counts": dict(sorted(split_counts.items())),
        "capability_counts": dict(sorted(capability_counts.items())),
        "template_clusters_per_capability": {
            capability: len({row["template_cluster_id"] for row in tasks if row["capability"] == capability})
            for capability in sorted(CLUSTERS)
        },
        "label_counts": dict(sorted(Counter(row["label"] for row in tasks).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destinations = [
        args.output / "trajectory_tasks_v1.jsonl",
        args.output / "trajectory_oracle_v1.jsonl",
        args.output / "dataset_contract.json",
        args.output / "dataset_manifest.json",
    ]
    if not args.force and any(path.exists() for path in destinations):
        raise FileExistsError("目标数据文件已存在；使用--force仅用于确定性重建")
    tasks, oracles = build_rows()
    (args.output / "trajectory_tasks_v1.jsonl").write_bytes(jsonl_bytes(tasks))
    (args.output / "trajectory_oracle_v1.jsonl").write_bytes(jsonl_bytes(oracles))
    (args.output / "dataset_contract.json").write_bytes(json_bytes(dataset_contract()))
    manifest = build_manifest(args.output, tasks, oracles)
    (args.output / "dataset_manifest.json").write_bytes(json_bytes(manifest))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
