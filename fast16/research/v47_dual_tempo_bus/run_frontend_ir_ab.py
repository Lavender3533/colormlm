"""对一个开发题做 v38 原始生成 vs Design IR 两阶段生成的最小 A/B。"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 300.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"请求失败 {url}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{url} 返回值不是 JSON 对象")
    return value


def chat(endpoint: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, dict[str, Any], float]:
    started = time.perf_counter()
    response = request_json(
        endpoint.rstrip("/") + "/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "temperature": 0.15,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "stream": False,
        },
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"chat completion 缺少 choices: {response}")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("chat completion 没有非空 content")
    return content, response.get("usage", {}), time.perf_counter() - started


def parse_ir(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.I | re.S)
    candidate = fenced.group(1) if fenced else stripped[stripped.find("{") : stripped.rfind("}") + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError(f"Design IR 不是合法 JSON: {error}") from error
    required = {
        "identity", "audience", "content_hierarchy", "layout_intent", "visual_tokens",
        "components", "interactions", "responsive_rules", "accessibility", "asset_policy",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"Design IR 字段不严格匹配: {sorted(value) if isinstance(value, dict) else type(value)}")
    if not isinstance(value["content_hierarchy"], list) or len(value["content_hierarchy"]) < 3:
        raise ValueError("Design IR content_hierarchy 少于 3 项")
    if not isinstance(value["interactions"], list) or len(value["interactions"]) < 2:
        raise ValueError("Design IR interactions 少于 2 项")
    for field in ("content_hierarchy", "components", "interactions", "responsive_rules", "accessibility"):
        if not isinstance(value[field], list) or not all(isinstance(item, str) and item for item in value[field]):
            raise ValueError(f"Design IR {field} 必须是非空字符串数组")
    if set(value["layout_intent"]) != {"composition", "anti_template_rule"}:
        raise ValueError("Design IR layout_intent 字段错误")
    if set(value["visual_tokens"]) != {"palette", "typography", "spacing", "surface"}:
        raise ValueError("Design IR visual_tokens 字段错误")
    policy = value.get("asset_policy")
    if not isinstance(policy, dict) or policy.get("allow_remote_random") is not False:
        raise ValueError("Design IR 必须禁止随机远程资产")
    return value


def extract_html(text: str) -> str:
    start = text.lower().find("<!doctype html")
    if start < 0:
        start = text.lower().find("<html")
    end = text.lower().rfind("</html>")
    if start < 0 or end < start:
        fenced = re.search(r"```(?:html)?\s*(.*?)\s*```", text, re.I | re.S)
        if fenced:
            text = fenced.group(1)
            start = text.lower().find("<!doctype html")
            if start < 0:
                start = text.lower().find("<html")
            end = text.lower().rfind("</html>")
    if start < 0 or end < start:
        raise ValueError("响应中找不到完整 HTML 文档")
    return text[start : end + len("</html>")].strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8138")
    parser.add_argument("--model", default="ColorLM-v38-Qwen36-Shared-Sequence-Policy")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-html", type=Path, help="使用冻结静态基线，跳过一次直接生成")
    parser.add_argument("--ir-max-tokens", type=int, default=700)
    parser.add_argument("--html-max-tokens", type=int, default=2200)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝混入旧结果: {args.output_dir}")
    tasks = read_jsonl(args.tasks)
    selected = [row for row in tasks if row.get("id") == args.task_id]
    if len(selected) != 1:
        raise ValueError(f"找不到唯一任务 {args.task_id}")
    task = selected[0]
    models = request_json(args.endpoint.rstrip("/") + "/v1/models", timeout=10)
    aliases = [str(item.get("id")) for item in models.get("data", []) if isinstance(item, dict)]
    if args.model not in aliases:
        raise RuntimeError(f"服务模型不匹配: expected={args.model}, actual={aliases}")

    system_html = (
        "你是资深前端产品设计师与工程师。严格满足任务中的每一个功能、响应式和无障碍要求。"
        "输出一个完整、自包含、无需构建的单文件 HTML。只输出 HTML，不解释，不使用 Markdown 围栏；"
        "禁止随机远程图片，交互必须真的用 JavaScript 接线。源码必须紧凑，约1200 token，"
        "不要注释、不要重复样式、不要省略闭合标签；功能完整优先于装饰堆砌。"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.baseline_html is not None:
        baseline_html = extract_html(args.baseline_html.read_text(encoding="utf-8"))
        baseline_usage: dict[str, Any] = {}
        baseline_seconds = 0.0
        baseline_source = str(args.baseline_html.resolve())
    else:
        baseline_text, baseline_usage, baseline_seconds = chat(
            args.endpoint,
            args.model,
            [{"role": "system", "content": system_html}, {"role": "user", "content": task["prompt"]}],
            args.html_max_tokens,
        )
        (args.output_dir / "baseline_raw.txt").write_text(baseline_text, encoding="utf-8")
        baseline_html = extract_html(baseline_text)
        baseline_source = "direct_v38_generation"

    ir_system = (
        "你是前端设计规划器。不要写 HTML，只输出合法 JSON 对象，且必须恰有这些字段："
        "identity字符串、audience字符串、content_hierarchy字符串数组、"
        "layout_intent对象(composition,anti_template_rule)、"
        "visual_tokens对象(palette数组,typography,spacing,surface)、components数组、interactions数组、"
        "responsive_rules数组、accessibility数组、asset_policy对象(allow_remote_random必须false,fallback字符串)。"
        "所有数组元素都只能是短字符串，绝不能放对象。content_hierarchy恰好4项、components最多6项、"
        "interactions恰好3项、responsive_rules恰好3项、accessibility恰好4项、palette恰好4项。"
        "每个字符串尽量不超过20个汉字，整个JSON不超过1200个字符。规划必须具体、有品牌身份、"
        "避开渐变Hero加三张卡片模板，并覆盖用户要求的真实交互。使用简体中文，只输出JSON。"
    )
    ir_text, ir_usage, ir_seconds = chat(
        args.endpoint,
        args.model,
        [{"role": "system", "content": ir_system}, {"role": "user", "content": task["prompt"]}],
        args.ir_max_tokens,
    )
    (args.output_dir / "design_ir_raw.txt").write_text(ir_text, encoding="utf-8")
    design_ir = parse_ir(ir_text)
    candidate_prompt = (
        task["prompt"]
        + "\n\n下面是已经冻结的 Design IR。实现时必须逐项落实，不能把它缩回通用落地页：\n"
        + json.dumps(design_ir, ensure_ascii=False, indent=2)
    )
    candidate_text, candidate_usage, candidate_seconds = chat(
        args.endpoint,
        args.model,
        [{"role": "system", "content": system_html}, {"role": "user", "content": candidate_prompt}],
        args.html_max_tokens,
    )
    (args.output_dir / "candidate_raw.txt").write_text(candidate_text, encoding="utf-8")
    candidate_html = extract_html(candidate_text)

    (args.output_dir / "baseline.html").write_text(baseline_html, encoding="utf-8")
    (args.output_dir / "candidate.html").write_text(candidate_html, encoding="utf-8")
    (args.output_dir / "design_ir.json").write_text(
        json.dumps(design_ir, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "format": "colorlm-v47-frontend-ir-ab-generation-v1",
        "status": "single_train_task_development_only",
        "task_id": args.task_id,
        "model": args.model,
        "endpoint": args.endpoint,
        "baseline": {"source": baseline_source, "seconds": baseline_seconds, "usage": baseline_usage, "bytes": len(baseline_html.encode("utf-8"))},
        "ir": {"seconds": ir_seconds, "usage": ir_usage},
        "candidate": {"seconds": candidate_seconds, "usage": candidate_usage, "bytes": len(candidate_html.encode("utf-8"))},
        "warning": "This generation artifact is not a blind gate and does not prove an embedded capability island.",
    }
    (args.output_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
