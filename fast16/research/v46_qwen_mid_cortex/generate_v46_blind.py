"""生成一次性 v46 新模板 blind；不复用 v44 的六个 blind 模板簇。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "v46_blind_tasks_v1.jsonl"
MANIFEST = HERE / "v46_blind_tasks_v1.manifest.json"


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


def target_call(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))
    return f"<tool_call>\n{payload}\n</tool_call>"


def prior_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
        ],
    }


def add(rows: list[dict[str, Any]], capability: str, case: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], target: str) -> None:
    rows.append(
        {
            "format": "colorlm-v46-one-shot-blind-task-v1",
            "id": f"{capability}-{case}",
            "split": "blind-v46",
            "capability": capability,
            "template_cluster_id": f"v46.{capability}.{case.split('-')[0]}",
            "messages": messages,
            "tools": tools,
            "target": target,
            "max_output_tokens": 96,
        }
    )


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    string = {"type": "string"}

    deploy = tool("deploy_package", "部署已定位的包", {"path": string, "environment": {"type": "string", "enum": ["staging", "production"]}}, ["path", "environment"])
    ask = tool("ask_user_question", "询问无法唯一推出的字段", {"field": string, "question": string}, ["field", "question"])
    repair_cases = [
        ("unique-a", "ERROR: bundle.zip不在根目录；manifest唯一记录为dist/bundle.zip。", target_call("deploy_package", {"path": "dist/bundle.zip", "environment": "staging"})),
        ("unique-b", "ERROR: app.tar不在根目录；manifest唯一记录为release/app.tar。", target_call("deploy_package", {"path": "release/app.tar", "environment": "production"})),
        ("ambiguous-a", "ERROR: site.zip存在两份：dist/site.zip与backup/site.zip；无默认顺序。", target_call("ask_user_question", {"field": "path", "question": "请选择 dist/site.zip 或 backup/site.zip。"})),
        ("ambiguous-b", "ERROR: api.bin存在两份：build/api.bin与cache/api.bin；必须由用户选择。", target_call("ask_user_question", {"field": "path", "question": "请选择 build/api.bin 或 cache/api.bin。"})),
    ]
    for index, (case, result, target) in enumerate(repair_cases):
        path = ["bundle.zip", "app.tar", "site.zip", "api.bin"][index]
        environment = ["staging", "production", "staging", "production"][index]
        add(rows, "tool_argument_repair", case, [{"role": "user", "content": "按manifest结果修正路径；不唯一时询问，不要猜。"}, prior_call(f"repair-{index}", "deploy_package", {"path": path, "environment": environment}), {"role": "tool", "tool_call_id": f"repair-{index}", "content": result}], [deploy, ask], target)

    commit = tool("commit_lease", "提交已就绪容量租约", {"lease_id": string}, ["lease_id"])
    renew = tool("renew_lease", "续期过期容量租约", {"lease_id": string}, ["lease_id"])
    lease_cases = [
        ("ready-a", "READY lease_id=lease-701 expires=18:00Z", target_call("commit_lease", {"lease_id": "lease-701"})),
        ("ready-b", "READY lease_id=lease-884 expires=19:30Z", target_call("commit_lease", {"lease_id": "lease-884"})),
        ("expired", "EXPIRED lease_id=lease-119 renewable=true", target_call("renew_lease", {"lease_id": "lease-119"})),
        ("committed", "COMMITTED lease_id=lease-552 receipt=rcpt-552", json.dumps({"status": "committed", "lease_id": "lease-552", "receipt": "rcpt-552"}, ensure_ascii=False, separators=(",", ":"))),
    ]
    for index, (case, result, target) in enumerate(lease_cases):
        add(rows, "continue_or_finish", case, [{"role": "user", "content": "根据容量租约的最新确定状态继续，已提交则结束。"}, prior_call(f"lease-{index}", "reserve_capacity", {"request_id": f"req-{index}"}), {"role": "tool", "tool_call_id": f"lease-{index}", "content": result}], [commit, renew], target)

    backup = tool("start_backup", "启动指定租户与区域的备份", {"tenant": string, "region": string}, ["tenant", "region"])
    clarify_cases = [
        ("complete-a", "给tenant-orchid在ap-northeast-1启动备份。", target_call("start_backup", {"tenant": "tenant-orchid", "region": "ap-northeast-1"})),
        ("complete-b", "给tenant-lotus在eu-west-1启动备份。", target_call("start_backup", {"tenant": "tenant-lotus", "region": "eu-west-1"})),
        ("missing-region", "给tenant-maple启动备份；区域由我决定。", target_call("ask_user_question", {"field": "region", "question": "请提供 tenant-maple 的备份区域。"})),
        ("missing-tenant", "在us-east-2启动备份；租户不能从上下文推出。", target_call("ask_user_question", {"field": "tenant", "question": "请提供要备份的租户。"})),
    ]
    for case, prompt, target in clarify_cases:
        add(rows, "missing_parameter_clarification", case, [{"role": "user", "content": prompt}], [backup, ask], target)

    scan = tool("run_security_scan", "执行发布安全扫描", {"artifact": string}, ["artifact"])
    sign = tool("sign_artifact", "签名已通过扫描的产物", {"artifact": string, "scan_id": string}, ["artifact", "scan_id"])
    publish = tool("publish_artifact", "发布已签名产物", {"artifact": string, "signature": string}, ["artifact", "signature"])
    verify = tool("verify_publication", "核验发布回执", {"publication_id": string}, ["publication_id"])
    plan_cases = [
        ("need-scan", "产物nova.tgz已构建，未扫描、未签名、未发布。", target_call("run_security_scan", {"artifact": "nova.tgz"})),
        ("need-sign", "产物ember.tgz的扫描scan-203已通过，未签名。", target_call("sign_artifact", {"artifact": "ember.tgz", "scan_id": "scan-203"})),
        ("need-publish", "产物cedar.tgz已签名，signature=sig-778，未发布。", target_call("publish_artifact", {"artifact": "cedar.tgz", "signature": "sig-778"})),
        ("need-verify", "产物iris.tgz已发布，publication_id=pub-912，尚未核验。", target_call("verify_publication", {"publication_id": "pub-912"})),
    ]
    for case, state, target in plan_cases:
        add(rows, "multi_step_planning", case, [{"role": "user", "content": "只执行发布链中的下一个未完成步骤。\n当前状态：" + state}], [scan, sign, publish, verify], target)

    click = tool("ui_click", "点击可访问性树中的元素", {"element_id": string}, ["element_id"])
    select = tool("ui_select", "选择下拉选项", {"element_id": string, "value": string}, ["element_id", "value"])
    focus = tool("ui_focus", "聚焦指定窗口", {"window_id": string}, ["window_id"])
    close = tool("ui_close_dialog", "关闭阻塞对话框", {"dialog_id": string}, ["dialog_id"])
    ui_cases = [
        ("focus", "可访问性状态：目标窗口window-editor已打开但未聚焦；先聚焦它。", target_call("ui_focus", {"window_id": "window-editor"})),
        ("close", "可访问性状态：模态对话框dialog-update遮挡目标；先关闭它。", target_call("ui_close_dialog", {"dialog_id": "dialog-update"})),
        ("select", "可访问性状态：下拉框select-theme已聚焦；选择dark。", target_call("ui_select", {"element_id": "select-theme", "value": "dark"})),
        ("click", "可访问性状态：菜单已打开，menu-export可用；点击它。", target_call("ui_click", {"element_id": "menu-export"})),
    ]
    for case, prompt, target in ui_cases:
        add(rows, "computer_operation", case, [{"role": "user", "content": prompt}], [click, select, focus, close], target)

    read = tool("read_file", "读取定位编译环境的文件", {"path": string}, ["path"])
    tests = tool("run_tests", "运行相关短测试", {"target": string}, ["target"])
    restore = tool("restore_file", "恢复已知正确的备份", {"path": string, "backup": string}, ["path", "backup"])
    edit = tool("edit_file", "应用确定的单行替换", {"path": string, "old": string, "new": string}, ["path", "old", "new"])
    debug_cases = [
        ("inspect-config", "build失败：compiler 1.78与项目声明的1.80不一致；尚未读取toolchain.toml。", target_call("read_file", {"path": "toolchain.toml"})),
        ("inspect-ci", "本地为1.80，CI报compiler 1.76；尚未读取.ci/build.yml。", target_call("read_file", {"path": ".ci/build.yml"})),
        ("apply-known-fix", "已读取toolchain.toml，唯一错误行昨channel = \"1.78\"，项目锁定1.80。", target_call("edit_file", {"path": "toolchain.toml", "old": "channel = \"1.78\"", "new": "channel = \"1.80\""})),
        ("verify-fix", "toolchain.toml已修正为1.80，尚未运行相关compiler短测试。", target_call("run_tests", {"target": "compiler-smoke"})),
    ]
    for case, result, target in debug_cases:
        add(rows, "code_debugging", case, [{"role": "user", "content": "根据现有证据只做下一个最小调试动作。"}, prior_call(f"debug-{case}", "run_build", {"target": "app"}), {"role": "tool", "tool_call_id": f"debug-{case}", "content": result}], [read, tests, restore, edit], target)
    return rows


def main() -> int:
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("拒绝覆盖已冻结v46 blind")
    rows = build_rows()
    counts = Counter(row["capability"] for row in rows)
    if len(rows) != 24 or set(counts.values()) != {4} or len({row["id"] for row in rows}) != 24:
        raise RuntimeError(f"blind数量或能力分布错误: {counts}")
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    OUTPUT.write_text(payload, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    manifest = {
        "format": "colorlm-v46-one-shot-blind-manifest-v1",
        "tasks": OUTPUT.name,
        "tasks_sha256": digest,
        "task_count": len(rows),
        "capability_counts": dict(sorted(counts.items())),
        "template_clusters": sorted({row["template_cluster_id"] for row in rows}),
        "old_v44_blind_reused": False,
        "one_shot": True,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
