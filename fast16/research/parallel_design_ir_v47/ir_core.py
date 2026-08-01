#!/usr/bin/env python3
"""Design Genome v1 的零依赖读取、规范化与严格契约校验。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


IR_VERSION = "dg1"
PREFERRED_MIN_BYTES = 150
PREFERRED_MAX_BYTES = 600
HARD_MAX_BYTES = 1200
LAYOUTS = {"dashboard", "editorial", "timeline", "split", "docs", "settings", "schedule", "story"}
MOBILE = {"stack", "cards", "drawer", "accordion"}
MODES = {"light", "dark", "paper"}
PALETTES = {"cyan", "indigo", "coral", "green", "amber", "violet", "blue"}
DENSITIES = {"compact", "normal", "airy"}
SHAPES = {"square", "soft", "round"}
ACTIONS = {"filter", "reset", "sort", "open", "confirm", "count", "compare", "validate", "copy", "tab", "toggle", "favorite", "year", "accordion", "navigate", "inspect"}
TRANSFORMS = {"table>cards", "drawer>full", "grid>stack", "aside>drawer", "schedule>accordion", "chart>scroll", "code>scroll"}
COMPONENT_REGISTRY = {
    "hero": {"plain", "editorial", "story"},
    "filters": {"status", "catalog", "project", "date-stage", "year", "generic"},
    "table": {"ops", "devices", "data", "generic"},
    "drawer": {"detail", "navigation"},
    "dialog": {"confirm", "preview", "danger", "success"},
    "products": {"magazine", "catalog"},
    "timeline": {"case-study", "process"},
    "compare": {"before-after", "range"},
    "form": {"booking", "search", "generic"},
    "sidebar": {"docs", "settings", "generic"},
    "code": {"request", "response", "generic"},
    "tabs": {"response", "language", "generic"},
    "toggles": {"privacy", "generic"},
    "schedule": {"stages", "generic"},
    "chart": {"year-series", "spark", "generic"},
    "note": {"source", "method", "generic"},
    "metrics": {"ops", "story", "generic"},
    "bag": {"counter"},
}
COMPONENT_ROLE_NAMES = ("primary", "controls", "content", "detail", "support")
COMPONENT_ROLE_GENES = (
    {("hero", "plain"), ("hero", "editorial"), ("hero", "story"), ("sidebar", "docs"), ("sidebar", "settings"), ("metrics", "ops"), ("metrics", "generic")},
    {("filters", "status"), ("filters", "catalog"), ("filters", "project"), ("filters", "date-stage"), ("filters", "year"), ("filters", "generic"), ("toggles", "privacy"), ("toggles", "generic"), ("tabs", "language"), ("form", "search")},
    {("table", "ops"), ("table", "devices"), ("table", "data"), ("table", "generic"), ("products", "magazine"), ("products", "catalog"), ("timeline", "case-study"), ("timeline", "process"), ("form", "booking"), ("form", "generic"), ("code", "request"), ("code", "response"), ("code", "generic"), ("schedule", "stages"), ("schedule", "generic"), ("chart", "year-series"), ("chart", "spark"), ("chart", "generic")},
    {("drawer", "detail"), ("drawer", "navigation"), ("dialog", "confirm"), ("dialog", "preview"), ("dialog", "danger"), ("dialog", "success"), ("compare", "before-after"), ("compare", "range"), ("tabs", "response"), ("tabs", "generic"), ("table", "data"), ("table", "generic")},
    {("metrics", "ops"), ("metrics", "story"), ("metrics", "generic"), ("note", "source"), ("note", "method"), ("note", "generic"), ("bag", "counter"), ("dialog", "confirm"), ("dialog", "success"), ("dialog", "danger"), ("drawer", "navigation"), ("drawer", "detail"), ("code", "response"), ("code", "generic")},
)
ACTION_ROLE_NAMES = ("data", "view", "commit", "state")
ACTION_ROLE_VALUES = (
    {"filter", "sort", "year", "navigate", "none"},
    {"open", "compare", "tab", "accordion", "inspect", "none"},
    {"confirm", "count", "validate", "copy", "favorite", "none"},
    {"reset", "toggle", "sort", "announce", "save", "none"},
)
RESPONSIVE_ROLE_NAMES = ("main", "overlay")
RESPONSIVE_ROLE_VALUES = (
    {"table>cards", "grid>stack", "schedule>accordion", "chart>scroll", "code>scroll"},
    {"drawer>full", "aside>drawer", "none"},
)
LAYOUT_ROLE_FAMILIES = {
    "dashboard": ({"metrics", "hero"}, {"filters"}, {"table"}, {"drawer"}, {"dialog", "note"}),
    "editorial": ({"hero"}, {"filters"}, {"products"}, {"dialog"}, {"bag", "note"}),
    "timeline": ({"hero"}, {"filters"}, {"timeline"}, {"compare"}, {"metrics", "note"}),
    "split": ({"hero"}, {"filters", "form"}, {"form"}, {"dialog"}, {"metrics", "note"}),
    "docs": ({"sidebar"}, {"tabs", "form"}, {"code"}, {"tabs", "drawer"}, {"drawer", "note", "code"}),
    "settings": ({"sidebar"}, {"toggles"}, {"table"}, {"dialog"}, {"note", "metrics"}),
    "schedule": ({"hero"}, {"filters"}, {"schedule"}, {"dialog"}, {"metrics", "note"}),
    "story": ({"hero"}, {"filters"}, {"chart"}, {"table"}, {"note", "metrics"}),
}
ACTION_COMPONENT = {
    "filter": {"filters"}, "reset": {"filters", "form"}, "sort": {"table"}, "open": {"drawer", "dialog"},
    "confirm": {"dialog"}, "count": {"bag"}, "compare": {"compare"},
    "validate": {"form"}, "copy": {"code"}, "tab": {"tabs"},
    "toggle": {"toggles"}, "favorite": {"schedule"}, "year": {"chart"},
    "accordion": {"sidebar", "schedule"}, "navigate": {"sidebar"},
    "inspect": {"chart", "table"},
}
TRANSFORM_COMPONENT = {
    "table>cards": {"table"}, "drawer>full": {"drawer", "dialog"},
    "grid>stack": {"hero", "products", "metrics", "timeline"},
    "aside>drawer": {"sidebar", "drawer"}, "schedule>accordion": {"schedule"},
    "chart>scroll": {"chart"}, "code>scroll": {"code"},
}


class IRError(ValueError):
    """Genome 或 copy slots 读取/契约错误。"""


def read_utf8_no_bom(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise IRError(f"{path}: 禁止 UTF-8 BOM")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise IRError(f"{path}: 非法 UTF-8: {error}") from error


def canonical_text(ir: dict[str, Any]) -> str:
    return json.dumps(ir, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def utf8_size(ir: dict[str, Any]) -> int:
    return len(canonical_text(ir).encode("utf-8"))


def _expect(errors: list[str], condition: bool, path: str, message: str) -> None:
    if not condition:
        errors.append(f"{path}: {message}")


def validate_ir(ir: Any, *, enforce_target_size: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(ir, dict):
        return ["$: 必须是 JSON 对象"]
    required = {"v", "q", "y", "l", "c", "x", "r", "a", "z"}
    allowed = required | {"e"}
    _expect(errors, required <= set(ir), "$", f"缺少字段 {sorted(required - set(ir))}")
    _expect(errors, set(ir) <= allowed, "$", f"未知字段 {sorted(set(ir) - allowed)}")
    _expect(errors, ir.get("v") == IR_VERSION, "$.v", f"必须为 {IR_VERSION}")

    q = ir.get("q")
    _expect(errors, isinstance(q, list) and len(q) == 2 and all(isinstance(x, int) and 0 <= x <= 63 for x in q), "$.q", "必须是 title/lede 两个 slot id")
    y = ir.get("y")
    _expect(errors, isinstance(y, list) and len(y) == 4, "$.y", "必须是四元视觉基因")
    if isinstance(y, list) and len(y) == 4:
        _expect(errors, y[0] in MODES, "$.y[0]", "未知 mode")
        _expect(errors, y[1] in PALETTES, "$.y[1]", "未知 palette")
        _expect(errors, y[2] in DENSITIES, "$.y[2]", "未知 density")
        _expect(errors, y[3] in SHAPES, "$.y[3]", "未知 shape")
    layout = ir.get("l")
    _expect(errors, isinstance(layout, list) and len(layout) == 3, "$.l", "必须是 layout/mobile/breakpoint-profile 三元组")
    if isinstance(layout, list) and len(layout) == 3:
        _expect(errors, layout[0] in LAYOUTS, "$.l[0]", "未知 layout")
        _expect(errors, layout[1] in MOBILE, "$.l[1]", "未知 mobile grammar")
        _expect(errors, isinstance(layout[2], int) and 0 <= layout[2] <= 2, "$.l[2]", "断点 profile 必须为 0..2")

    genes = ir.get("c")
    families: list[str] = []
    if isinstance(genes, list):
        _expect(errors, len(genes) == 5, "$.c", "生成契约固定为 primary/controls/content/detail/support 五槽")
        for index, gene in enumerate(genes):
            at = f"$.c[{index}]"
            _expect(errors, isinstance(gene, list) and len(gene) == 2 and all(isinstance(x, str) for x in gene), at, "必须为 [family,variant]")
            if isinstance(gene, list) and len(gene) == 2:
                family, variant = gene
                families.append(family)
                _expect(errors, family in COMPONENT_REGISTRY, f"{at}[0]", "未知 family")
                if family in COMPONENT_REGISTRY:
                    _expect(errors, variant in COMPONENT_REGISTRY[family], f"{at}[1]", "该 family 不支持此 variant")
                if index < len(COMPONENT_ROLE_GENES):
                    _expect(errors, (family, variant) in COMPONENT_ROLE_GENES[index], at, f"不属于 {COMPONENT_ROLE_NAMES[index]} 角色槽")
        _expect(errors, len(genes) == len(set(tuple(x) for x in genes if isinstance(x, list))), "$.c", "组件基因不得重复")
    else:
        errors.append("$.c: 必须是数组")

    actions = ir.get("x")
    _expect(errors, isinstance(actions, list) and len(actions) == 4, "$.x", "必须按 data/view/commit/state 固定四槽")
    if isinstance(actions, list) and len(actions) == 4:
        for index, action in enumerate(actions):
            _expect(errors, action in ACTION_ROLE_VALUES[index], f"$.x[{index}]", f"不属于 {ACTION_ROLE_NAMES[index]} 动作槽")
    transforms = ir.get("r")
    _expect(errors, isinstance(transforms, list) and len(transforms) == 2, "$.r", "必须按 main/overlay 固定两槽")
    if isinstance(transforms, list) and len(transforms) == 2:
        for index, transform in enumerate(transforms):
            _expect(errors, transform in RESPONSIVE_ROLE_VALUES[index], f"$.r[{index}]", f"不属于 {RESPONSIVE_ROLE_NAMES[index]} 响应式槽")
    if isinstance(layout, list) and len(layout) == 3 and layout[0] in LAYOUT_ROLE_FAMILIES and len(families) == 5:
        for index, allowed_families in enumerate(LAYOUT_ROLE_FAMILIES[layout[0]]):
            _expect(errors, families[index] in allowed_families, f"$.c[{index}]", f"{layout[0]} 的 {COMPONENT_ROLE_NAMES[index]} 槽不允许 {families[index]}")
    if isinstance(actions, list):
        for action in actions:
            if action in ACTION_COMPONENT and action != "none":
                _expect(errors, bool(ACTION_COMPONENT[action] & set(families)), "$.x", f"动作 {action} 没有对应组件")
    if isinstance(transforms, list):
        for transform in transforms:
            if transform in TRANSFORM_COMPONENT and transform != "none":
                _expect(errors, bool(TRANSFORM_COMPONENT[transform] & set(families)), "$.r", f"变换 {transform} 没有对应组件")
    _expect(errors, isinstance(ir.get("a"), int) and 15 <= ir["a"] <= 255, "$.a", "无障碍位图必须为 15..255")
    _expect(errors, ir.get("z") in {"inline", "none"}, "$.z", "资产策略必须为 inline/none")
    if "e" in ir:
        ext = ir["e"]
        _expect(errors, isinstance(ext, dict), "$.e", "必须是对象")
        if isinstance(ext, dict):
            for key, value in ext.items():
                _expect(errors, bool(re.fullmatch(r"[a-z][a-z0-9-]*:[a-z][a-z0-9-]*", key)), f"$.e.{key}", "扩展键须带命名空间")
                _expect(errors, isinstance(value, (bool, int, float)) and not isinstance(value, str), f"$.e.{key}", "扩展值只允许标量，禁止自由文案")
    if enforce_target_size:
        size = utf8_size(ir)
        _expect(errors, PREFERRED_MIN_BYTES <= size <= HARD_MAX_BYTES, "$", f"规范序列为 {size} 字节，要求 {PREFERRED_MIN_BYTES}..{HARD_MAX_BYTES}")
    return errors


def validate_slots(payload: Any, prompt: str | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["slots$: 必须是对象"]
    _expect(errors, set(payload) == {"schema_version", "task_id", "prompt_sha256", "slots"}, "slots$", "字段不严格匹配")
    _expect(errors, payload.get("schema_version") == "design-copy-slots-v1", "slots$.schema_version", "版本错误")
    _expect(errors, isinstance(payload.get("task_id"), str) and bool(payload["task_id"]), "slots$.task_id", "不能为空")
    _expect(errors, isinstance(payload.get("prompt_sha256"), str) and bool(re.fullmatch(r"[0-9a-f]{64}", payload["prompt_sha256"])), "slots$.prompt_sha256", "SHA-256 格式错误")
    slots = payload.get("slots")
    ids: list[int] = []
    if isinstance(slots, list):
        _expect(errors, 2 <= len(slots) <= 64, "slots$.slots", "数量必须为 2..64")
        for index, slot in enumerate(slots):
            at = f"slots$.slots[{index}]"
            _expect(errors, isinstance(slot, dict) and set(slot) == {"id", "kind", "text"}, at, "字段必须精确为 id/kind/text")
            if not isinstance(slot, dict):
                continue
            _expect(errors, isinstance(slot.get("id"), int) and 0 <= slot["id"] <= 63, f"{at}.id", "范围错误")
            if isinstance(slot.get("id"), int):
                ids.append(slot["id"])
            _expect(errors, slot.get("kind") in {"title", "lede", "label", "value"}, f"{at}.kind", "枚举错误")
            _expect(errors, isinstance(slot.get("text"), str) and 1 <= len(slot["text"]) <= 160, f"{at}.text", "长度错误")
            if prompt is not None and isinstance(slot.get("text"), str):
                _expect(errors, slot["text"] in prompt, f"{at}.text", "必须是 prompt 的连续原文片段")
        _expect(errors, len(ids) == len(set(ids)), "slots$.slots", "slot id 必须唯一")
    else:
        errors.append("slots$.slots: 必须是数组")
    if prompt is not None:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        _expect(errors, payload.get("prompt_sha256") == digest, "slots$.prompt_sha256", "与 prompt 不一致")
    return errors


def load_ir(path: Path, *, enforce_target_size: bool = False) -> dict[str, Any]:
    try:
        ir = json.loads(read_utf8_no_bom(path))
    except json.JSONDecodeError as error:
        raise IRError(f"{path}: JSON 解析失败: {error}") from error
    errors = validate_ir(ir, enforce_target_size=enforce_target_size)
    if errors:
        raise IRError("\n".join(errors))
    return ir


def load_slots(path: Path, prompt: str | None = None) -> dict[int, str]:
    try:
        payload = json.loads(read_utf8_no_bom(path))
    except json.JSONDecodeError as error:
        raise IRError(f"{path}: JSON 解析失败: {error}") from error
    errors = validate_slots(payload, prompt)
    if errors:
        raise IRError("\n".join(errors))
    return {item["id"]: item["text"] for item in payload["slots"]}


def write_utf8_no_bom(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
