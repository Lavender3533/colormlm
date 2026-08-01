#!/usr/bin/env python3
"""ColorLM v47 可复现的单文件 HTML 静态评分器。

只使用 Python 标准库；不联网、不执行 JavaScript、不启动浏览器。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


SCORER_VERSION = "parallel-frontend-static-v1.0.0"
CONTRACT_VERSION = "parallel-frontend-contract-v1"
DIMENSION_MAX = {
    "structure": 18,
    "responsive": 16,
    "interaction": 16,
    "visual_complexity": 20,
    "dependency_safety": 10,
    "accessibility": 20,
}
TEMPLATE_PENALTY_MAX = 20
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
OPTIONAL_END_TAGS = {"li", "dt", "dd", "p", "rt", "rp", "optgroup", "option", "thead", "tbody", "tfoot", "tr", "td", "th"}
INTERACTIVE_TAGS = {"a", "button", "input", "select", "textarea", "details", "summary"}
LANDMARK_TAGS = {"header", "nav", "main", "aside", "footer"}
GENERIC_CARD_RE = re.compile(r"(?:^|[-_])(card|feature|service|pricing|testimonial|benefit)(?:$|[-_])", re.I)
URL_RE = re.compile(r"(?P<url>(?:https?:)?//[^\s\"')>]+)", re.I)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def round2(value: float) -> float:
    return round(value + 1e-9, 2)


@dataclass
class Element:
    tag: str
    attrs: dict[str, str]
    line: int


@dataclass
class ParsedHTML:
    elements: list[Element] = field(default_factory=list)
    starts: Counter[str] = field(default_factory=Counter)
    ids: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    text_chunks: list[str] = field(default_factory=list)
    style_blocks: list[str] = field(default_factory=list)
    script_blocks: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    stack: list[str] = field(default_factory=list)
    mismatched_end_tags: list[str] = field(default_factory=list)
    unclosed_tags: list[str] = field(default_factory=list)


class Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.data = ParsedHTML()
        self._capture: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        normalized = {k.lower(): (v or "") for k, v in attrs}
        self.data.elements.append(Element(tag, normalized, self.getpos()[0]))
        self.data.starts[tag] += 1
        if "id" in normalized:
            self.data.ids.append(normalized["id"])
        if "class" in normalized:
            self.data.classes.extend(normalized["class"].split())
        if tag not in VOID_TAGS:
            if tag in OPTIONAL_END_TAGS and self.data.stack and self.data.stack[-1] == tag:
                self.data.stack.pop()
            self.data.stack.append(tag)
        if tag in {"style", "script"}:
            self._capture = tag
            self._buffer = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in VOID_TAGS and self.data.stack and self.data.stack[-1] == tag.lower():
            self.data.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._capture == tag:
            block = "".join(self._buffer)
            (self.data.style_blocks if tag == "style" else self.data.script_blocks).append(block)
            self._capture = None
            self._buffer = []
        if tag in VOID_TAGS:
            return
        if tag in self.data.stack:
            while self.data.stack:
                popped = self.data.stack.pop()
                if popped == tag:
                    break
                if popped not in OPTIONAL_END_TAGS:
                    self.data.mismatched_end_tags.append(f"{popped}->{tag}")
        elif tag not in OPTIONAL_END_TAGS:
            self.data.mismatched_end_tags.append(f"orphan:{tag}")

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buffer.append(data)
        elif data.strip():
            self.data.text_chunks.append(data.strip())

    def handle_comment(self, data: str) -> None:
        self.data.comments.append(data)

    def close(self) -> None:
        super().close()
        self.data.unclosed_tags = [tag for tag in self.data.stack if tag not in OPTIONAL_END_TAGS]


def add(checks: list[dict[str, Any]], check_id: str, points: float, max_points: float, passed: bool, evidence: Any) -> float:
    earned = points if passed else 0.0
    checks.append({
        "id": check_id,
        "earned": round2(earned),
        "max": round2(max_points),
        "passed": bool(passed),
        "evidence": evidence,
    })
    return earned


def partial(checks: list[dict[str, Any]], check_id: str, earned: float, max_points: float, evidence: Any) -> float:
    earned = clamp(earned, 0, max_points)
    checks.append({
        "id": check_id,
        "earned": round2(earned),
        "max": round2(max_points),
        "passed": earned >= max_points,
        "evidence": evidence,
    })
    return earned


def elements(parsed: ParsedHTML, tag: str) -> list[Element]:
    return [item for item in parsed.elements if item.tag == tag]


def has_attr_value(parsed: ParsedHTML, tag: str, key: str, pattern: str) -> bool:
    rx = re.compile(pattern, re.I)
    return any(rx.search(item.attrs.get(key, "")) for item in elements(parsed, tag))


def parse_document(text: str) -> ParsedHTML:
    parser = Collector()
    parser.feed(text)
    parser.close()
    return parser.data


def extract_css(text: str, parsed: ParsedHTML) -> str:
    inline = [item.attrs.get("style", "") for item in parsed.elements if item.attrs.get("style")]
    return "\n".join(parsed.style_blocks + inline)


def external_dependencies(parsed: ParsedHTML, css: str) -> list[dict[str, Any]]:
    deps: list[dict[str, Any]] = []
    for item in parsed.elements:
        for attr in ("src", "href", "poster"):
            value = item.attrs.get(attr, "").strip()
            if not value or value.startswith(("#", "data:", "mailto:", "tel:", "javascript:")):
                continue
            if value.startswith(("http://", "https://", "//")):
                parsed_url = urlparse("https:" + value if value.startswith("//") else value)
                deps.append({
                    "tag": item.tag,
                    "attribute": attr,
                    "url": value,
                    "host": parsed_url.netloc.lower(),
                    "active": (
                        attr in {"src", "poster"}
                        or item.tag in {"script", "link", "iframe", "img", "video", "audio", "source"}
                    ),
                    "integrity": bool(item.attrs.get("integrity")),
                    "crossorigin": bool(item.attrs.get("crossorigin")),
                })
    for match in URL_RE.finditer(css):
        value = match.group("url")
        parsed_url = urlparse("https:" + value if value.startswith("//") else value)
        deps.append({"tag": "css", "attribute": "url", "url": value, "host": parsed_url.netloc.lower(), "active": True, "integrity": False, "crossorigin": False})
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for dep in deps:
        key = (dep["tag"], dep["attribute"], dep["url"])
        if key not in seen:
            unique.append(dep)
            seen.add(key)
    return unique


def score_structure(text: str, parsed: ParsedHTML) -> tuple[float, list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    findings: list[str] = []
    score = 0.0
    score += add(checks, "doctype", 2, 2, bool(re.search(r"<!doctype\s+html", text, re.I)), "HTML5 doctype")
    htmls, heads, bodies = elements(parsed, "html"), elements(parsed, "head"), elements(parsed, "body")
    skeleton_ok = len(htmls) == len(heads) == len(bodies) == 1
    score += add(checks, "single_document_skeleton", 3, 3, skeleton_ok, {"html": len(htmls), "head": len(heads), "body": len(bodies)})
    score += add(checks, "title", 1.5, 1.5, parsed.starts["title"] == 1, parsed.starts["title"])
    score += add(checks, "charset_utf8", 1.5, 1.5, has_attr_value(parsed, "meta", "charset", r"^utf-8$") or has_attr_value(parsed, "meta", "content", r"charset\s*=\s*utf-8"), "meta charset=utf-8")
    score += add(checks, "one_h1", 2, 2, parsed.starts["h1"] == 1, parsed.starts["h1"])
    landmarks = sorted(tag for tag in LANDMARK_TAGS if parsed.starts[tag])
    score += partial(checks, "semantic_landmarks", min(3, len(landmarks)), 3, landmarks)
    duplicate_ids = sorted(k for k, v in Counter(parsed.ids).items() if k and v > 1)
    score += add(checks, "unique_ids", 2, 2, not duplicate_ids, duplicate_ids)
    balance_errors = parsed.mismatched_end_tags + parsed.unclosed_tags
    score += add(checks, "balanced_tags", 3, 3, not balance_errors, balance_errors[:20])
    if not skeleton_ok:
        findings.append("文档骨架不是唯一的 html/head/body。")
    if duplicate_ids:
        findings.append(f"存在重复 id：{', '.join(duplicate_ids[:5])}。")
    if balance_errors:
        findings.append(f"静态栈检查发现 {len(balance_errors)} 个标签闭合异常。")
    return round2(score), checks, findings


def score_responsive(text: str, css: str, parsed: ParsedHTML) -> tuple[float, list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    findings: list[str] = []
    score = 0.0
    viewport = any(item.attrs.get("name", "").lower() == "viewport" and "width=device-width" in item.attrs.get("content", "").lower() for item in elements(parsed, "meta"))
    score += add(checks, "viewport", 3, 3, viewport, "width=device-width")
    media_queries = re.findall(r"@media\s*\(([^)]+)\)", css, re.I)
    distinct_breakpoints = sorted(set(re.findall(r"\d+(?:\.\d+)?(?:px|rem|em)", " ".join(media_queries), re.I)))
    score += partial(checks, "media_queries", min(4, len(media_queries) * 2), 4, {"count": len(media_queries), "breakpoints": distinct_breakpoints})
    fluid_tokens = re.findall(r"(?:\d+(?:\.\d+)?(?:%|vw|vh|vmin|vmax)|\b(?:clamp|min|max|minmax|auto-fit|auto-fill)\s*\()", css, re.I)
    score += partial(checks, "fluid_layout", min(3, len(set(x.lower() for x in fluid_tokens)) * 0.75), 3, sorted(set(fluid_tokens))[:20])
    layout_modes = [name for name, rx in (("grid", r"display\s*:\s*grid"), ("flex", r"display\s*:\s*flex"), ("wrap", r"flex-wrap\s*:\s*wrap")) if re.search(rx, css, re.I)]
    score += partial(checks, "adaptive_layout_modes", min(2, len(layout_modes)), 2, layout_modes)
    responsive_media = bool(re.search(r"(?:img|video|canvas|svg)[^{]*\{[^}]*max-width\s*:\s*100%", css, re.I | re.S)) or not any(parsed.starts[x] for x in ("img", "video", "canvas"))
    score += add(checks, "responsive_media", 1.5, 1.5, responsive_media, "媒体 max-width 或无位图媒体")
    # max-width: 1400px 是正常容器上限，不应被误判为固定超宽。
    oversized = re.findall(r"(?<!max-)(?:width|min-width)\s*:\s*(\d{4,})px", css, re.I)
    horizontal_risk = bool(oversized) or bool(re.search(r"white-space\s*:\s*nowrap", css, re.I) and not re.search(r"overflow-x\s*:\s*(?:auto|scroll|hidden)", css, re.I))
    score += add(checks, "no_obvious_horizontal_overflow", 2.5, 2.5, not horizontal_risk, {"oversized_px": oversized[:10]})
    if not viewport:
        findings.append("缺少移动端 viewport。")
    if not media_queries:
        findings.append("未发现媒体查询，无法证明断点适配。")
    if horizontal_risk:
        findings.append("发现固定超宽或 nowrap 导致的横向溢出风险。")
    return round2(score), checks, findings


def score_interaction(text: str, css: str, parsed: ParsedHTML) -> tuple[float, list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    findings: list[str] = []
    score = 0.0
    interactive = [item for item in parsed.elements if item.tag in INTERACTIVE_TAGS or item.attrs.get("role") in {"button", "link", "tab", "switch"}]
    meaningful_links = [item for item in elements(parsed, "a") if item.attrs.get("href", "").strip() not in {"", "#"} and not item.attrs.get("href", "").lower().startswith("javascript:")]
    buttons = elements(parsed, "button")
    score += partial(checks, "meaningful_controls", min(3, len(interactive) * 0.75), 3, {"interactive": len(interactive), "meaningful_links": len(meaningful_links), "buttons": len(buttons)})
    scripts = "\n".join(parsed.script_blocks)
    wiring_tokens = re.findall(r"(?:addEventListener|onclick\s*=|onchange\s*=|onsubmit\s*=|querySelector|classList\.|\.showModal\s*\(|\.toggle\s*\()", text, re.I)
    score += partial(checks, "event_wiring", min(4, len(set(x.lower() for x in wiring_tokens)) * 1.25), 4, sorted(set(wiring_tokens))[:20])
    states = [name for name, rx in (("hover", r":hover"), ("focus", r":focus(?:-visible)?"), ("active", r":active"), ("checked", r":checked"), ("expanded", r"aria-expanded")) if re.search(rx, text, re.I)]
    score += partial(checks, "interaction_states", min(3, len(states)), 3, states)
    transition_ok = bool(re.search(r"transition(?:-\w+)?\s*:\s*[^;]*(?:1[5-9]\d|2\d\d|300)ms", css, re.I)) or bool(re.search(r"transition(?:-\w+)?\s*:\s*[^;]*(?:0\.[12]|0\.3)s", css, re.I))
    score += add(checks, "bounded_transition", 1.5, 1.5, transition_ok, "150–300ms transition")
    placeholder_links = [item.line for item in elements(parsed, "a") if item.attrs.get("href", "").strip() in {"", "#"}]
    inert_buttons = len(buttons) > 0 and not scripts.strip() and not any(item.attrs.get("type", "").lower() == "submit" for item in buttons)
    score += add(checks, "no_obvious_dead_controls", 2.5, 2.5, not placeholder_links and not inert_buttons, {"placeholder_link_lines": placeholder_links[:20], "inert_button_risk": inert_buttons})
    forms = elements(parsed, "form")
    form_ok = not forms or (bool(re.search(r"(?:submit|onsubmit|action\s*=)", text, re.I)) and bool(elements(parsed, "input") or elements(parsed, "textarea") or elements(parsed, "select")))
    score += add(checks, "form_path", 2, 2, form_ok, {"forms": len(forms)})
    if not interactive:
        findings.append("页面没有可识别的原生或 ARIA 交互控件。")
    if placeholder_links:
        findings.append(f"发现 {len(placeholder_links)} 个空链接或 href=#。")
    if inert_buttons:
        findings.append("按钮存在，但静态证据中没有脚本接线或提交路径。")
    return round2(score), checks, findings


def score_visual(text: str, css: str, parsed: ParsedHTML) -> tuple[float, list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    findings: list[str] = []
    score = 0.0
    css_rules = len(re.findall(r"[^@{}][^{}]*\{[^{}]*\}", css, re.S))
    css_properties = len(re.findall(r"(?:^|[;{])\s*[-\w]+\s*:", css, re.M))
    score += partial(checks, "css_depth", min(4, css_rules / 8 + css_properties / 80), 4, {"rules": css_rules, "properties": css_properties})
    sections = parsed.starts["section"] + parsed.starts["article"]
    landmarks = sum(parsed.starts[x] for x in LANDMARK_TAGS)
    score += partial(checks, "content_composition", min(4, sections * 0.8 + landmarks * 0.4), 4, {"sections_articles": sections, "landmarks": landmarks})
    techniques = [name for name, rx in (
        ("grid", r"display\s*:\s*grid"), ("flex", r"display\s*:\s*flex"),
        ("gradient", r"(?:linear|radial|conic)-gradient"), ("shadow", r"box-shadow\s*:"),
        ("filter", r"(?:backdrop-)?filter\s*:"), ("clip", r"clip-path\s*:"),
        ("animation", r"@keyframes|animation\s*:"), ("transform", r"transform\s*:"),
        ("custom-properties", r"--[-\w]+\s*:"), ("pseudo-elements", r"::(?:before|after)"),
    ) if re.search(rx, css, re.I)]
    score += partial(checks, "visual_techniques", min(5, len(techniques) * 0.75), 5, techniques)
    colors = set(x.lower() for x in re.findall(r"#[0-9a-f]{3,8}\b|rgba?\([^)]*\)|hsla?\([^)]*\)", css, re.I))
    fonts = set(re.findall(r"font-family\s*:\s*([^;}]+)", css, re.I))
    score += partial(checks, "design_tokens", min(3, len(colors) * 0.25 + len(fonts) + (1 if re.search(r"--[-\w]+\s*:", css) else 0)), 3, {"colors": len(colors), "font_stacks": len(fonts)})
    media_count = sum(parsed.starts[x] for x in ("img", "svg", "video", "canvas", "picture"))
    score += partial(checks, "visual_assets", min(2, media_count * 0.5 + (0.5 if re.search(r"background-image\s*:", css, re.I) else 0)), 2, {"media": media_count})
    typography = [name for name, rx in (("clamp", r"font-size\s*:\s*clamp\("), ("weight", r"font-weight\s*:"), ("line-height", r"line-height\s*:"), ("letter-spacing", r"letter-spacing\s*:")) if re.search(rx, css, re.I)]
    score += partial(checks, "typography_system", min(2, len(typography) * 0.5), 2, typography)
    if score < 8:
        findings.append("视觉结构和 CSS 技法较少，接近低复杂度模板。")
    return round2(score), checks, findings


def score_dependencies(parsed: ParsedHTML, css: str) -> tuple[float, list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    findings: list[str] = []
    deps = external_dependencies(parsed, css)
    active = [d for d in deps if d["active"]]
    insecure = [d for d in deps if d["url"].startswith("http://")]
    versioned_code = [d for d in active if d["tag"] in {"script", "link", "iframe"}]
    unpinned = [d for d in versioned_code if not re.search(r"(?:@|/)(?:v?\d+\.\d+\.\d+|\d+\.\d+\.\d+)(?:/|$)", d["url"])]
    third_party_scripts = [d for d in deps if d["tag"] == "script"]
    remote_assets = [d for d in active if d["tag"] in {"img", "video", "audio", "source", "css"} or d["attribute"] == "poster"]
    score = 10.0
    score -= min(3.0, len(insecure) * 1.5)
    score -= min(3.0, len(third_party_scripts) * 1.0)
    score -= min(2.0, len(unpinned) * 0.5)
    score -= min(2.0, len(remote_assets) * 0.5)
    score -= min(2.0, max(0, len(set(d["host"] for d in deps)) - 2) * 0.5)
    score = clamp(score, 0, 10)
    partial(checks, "external_dependency_risk", score, 10, {
        "total": len(deps), "active": len(active), "http": len(insecure),
        "third_party_scripts": len(third_party_scripts), "unpinned_active": len(unpinned),
        "remote_assets": len(remote_assets),
        "hosts": sorted(set(d["host"] for d in deps)),
    })
    if insecure:
        findings.append("存在明文 HTTP 外部依赖。")
    if third_party_scripts:
        findings.append(f"存在 {len(third_party_scripts)} 个第三方脚本，离线与供应链风险上升。")
    if unpinned:
        findings.append(f"存在 {len(unpinned)} 个未显式锁版本的主动依赖。")
    if remote_assets:
        findings.append(f"存在 {len(remote_assets)} 个运行时远程视觉/媒体依赖，离线渲染可能缺失。")
    return round2(score), checks, findings, deps


def score_accessibility(text: str, css: str, parsed: ParsedHTML) -> tuple[float, list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    findings: list[str] = []
    score = 0.0
    html_lang = bool(elements(parsed, "html") and elements(parsed, "html")[0].attrs.get("lang", "").strip())
    score += add(checks, "document_language", 2, 2, html_lang, elements(parsed, "html")[0].attrs.get("lang", "") if elements(parsed, "html") else "")
    imgs = elements(parsed, "img")
    missing_alt = [x.line for x in imgs if "alt" not in x.attrs]
    score += add(checks, "image_alternatives", 3, 3, not missing_alt, {"images": len(imgs), "missing_alt_lines": missing_alt})
    controls = elements(parsed, "input") + elements(parsed, "select") + elements(parsed, "textarea")
    label_fors = {x.attrs.get("for", "") for x in elements(parsed, "label")}
    unlabeled = [x.line for x in controls if x.attrs.get("type", "").lower() != "hidden" and not (x.attrs.get("aria-label") or x.attrs.get("aria-labelledby") or (x.attrs.get("id") and x.attrs["id"] in label_fors))]
    score += add(checks, "form_labels", 3, 3, not unlabeled, {"controls": len(controls), "unlabeled_lines": unlabeled})
    icon_buttons = [x for x in elements(parsed, "button") if not x.attrs.get("aria-label") and not x.attrs.get("aria-labelledby")]
    # 只有无文本按钮才算缺名；HTMLParser 未保留父子，使用保守的源代码模式。
    empty_button_count = len(re.findall(r"<button\b(?![^>]*aria-label)[^>]*>\s*(?:<svg\b[\s\S]*?</svg>|<i\b[\s\S]*?</i>)?\s*</button>", text, re.I))
    score += add(checks, "control_names", 2, 2, empty_button_count == 0, {"unnamed_icon_buttons": empty_button_count, "buttons_checked": len(icon_buttons)})
    headings = [int(x.tag[1]) for x in parsed.elements if re.fullmatch(r"h[1-6]", x.tag)]
    jumps = [(a, b) for a, b in zip(headings, headings[1:]) if b > a + 1]
    score += add(checks, "heading_order", 2, 2, bool(headings) and not jumps, {"levels": headings, "jumps": jumps})
    score += partial(checks, "landmarks", min(2, len([x for x in LANDMARK_TAGS if parsed.starts[x]]) * 0.5), 2, sorted(x for x in LANDMARK_TAGS if parsed.starts[x]))
    focus_visible = bool(re.search(r":focus(?:-visible)?", css, re.I)) and not bool(re.search(r":focus[^{}]*\{[^{}]*outline\s*:\s*(?:none|0)\s*;?[^{}]*\}", css, re.I | re.S) and not re.search(r":focus-visible", css, re.I))
    score += add(checks, "visible_focus", 2.5, 2.5, focus_visible, ":focus/:focus-visible")
    has_motion = bool(re.search(r"@keyframes|animation\s*:|transition\s*:", css, re.I))
    reduced = bool(re.search(r"prefers-reduced-motion\s*:\s*reduce", css, re.I))
    score += add(checks, "reduced_motion", 1.5, 1.5, not has_motion or reduced, {"motion": has_motion, "reduced_motion_rule": reduced})
    skip_link = bool(re.search(r"<a\b[^>]*href=[\"']#(?:main|content|main-content)[\"']", text, re.I))
    score += add(checks, "skip_navigation", 1, 1, skip_link or parsed.starts["nav"] == 0, {"skip_link": skip_link, "nav": parsed.starts["nav"]})
    clickable_divs = [x.line for x in parsed.elements if x.tag in {"div", "span"} and any(k.startswith("on") for k in x.attrs) and x.attrs.get("role") not in {"button", "link"}]
    score += add(checks, "semantic_click_targets", 1, 1, not clickable_divs, clickable_divs)
    if missing_alt:
        findings.append(f"有 {len(missing_alt)} 张图片缺少 alt 属性。")
    if unlabeled:
        findings.append(f"有 {len(unlabeled)} 个表单控件缺少可识别标签。")
    if not focus_visible:
        findings.append("未发现可靠的键盘焦点可见样式。")
    if has_motion and not reduced:
        findings.append("存在动画/过渡，但未尊重 prefers-reduced-motion。")
    return round2(score), checks, findings


def template_penalty(text: str, css: str, parsed: ParsedHTML) -> tuple[float, list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []
    penalty = 0.0
    class_counts = Counter(c.lower() for c in parsed.classes)
    generic_repeats = {name: count for name, count in class_counts.items() if GENERIC_CARD_RE.search(name)}
    generic_cards = max(generic_repeats.values(), default=0)
    articles = parsed.starts["article"]
    repeated_three = generic_cards == 3 or articles == 3
    distinctive_techniques = sum(bool(re.search(rx, css, re.I)) for rx in (
        r"display\s*:\s*grid", r"(?:linear|radial|conic)-gradient", r"box-shadow\s*:",
        r"(?:backdrop-)?filter\s*:", r"clip-path\s*:", r"@keyframes|animation\s*:",
        r"transform\s*:", r"--[-\w]+\s*:", r"::(?:before|after)",
    ))
    common_copy = len(re.findall(r"\b(?:feature|features|learn more|get started|our services|why choose us)\b", text, re.I))
    ordinary_three = repeated_three and (distinctive_techniques < 7 or common_copy >= 3)
    p = 6.0 if ordinary_three else 0.0
    penalty += p
    checks.append({"id": "exact_three_card_signature", "penalty": p, "max_penalty": 6, "triggered": bool(p), "evidence": {"max_generic_class_repeat": generic_cards, "generic_class_repeats": generic_repeats, "articles": articles, "distinctive_techniques": distinctive_techniques}})
    if p:
        reasons.append("命中恰好三张通用卡片/文章的模板签名。")
    sections = parsed.starts["section"] + parsed.starts["article"]
    low_composition = sections <= 3 and sum(parsed.starts[x] for x in LANDMARK_TAGS) <= 2
    p = 3.0 if low_composition else 0.0
    penalty += p
    checks.append({"id": "low_composition", "penalty": p, "max_penalty": 3, "triggered": bool(p), "evidence": {"sections_articles": sections}})
    if p:
        reasons.append("页面内容层级单薄。")
    media_count = sum(parsed.starts[x] for x in ("img", "svg", "video", "canvas", "picture"))
    p = 3.0 if media_count == 0 and not re.search(r"(?:gradient|clip-path|::before|::after)", css, re.I) else 0.0
    penalty += p
    checks.append({"id": "no_distinct_visual_asset", "penalty": p, "max_penalty": 3, "triggered": bool(p), "evidence": {"media": media_count}})
    if p:
        reasons.append("没有媒体、矢量或明显的程序化视觉资产。")
    p = 2.0 if common_copy >= 3 else 0.0
    penalty += p
    checks.append({"id": "generic_landing_copy", "penalty": p, "max_penalty": 2, "triggered": bool(p), "evidence": {"matches": common_copy}})
    if p:
        reasons.append("通用落地页套话密度较高。")
    css_rules = len(re.findall(r"[^@{}][^{}]*\{[^{}]*\}", css, re.S))
    p = 3.0 if css_rules < 12 else 0.0
    penalty += p
    checks.append({"id": "shallow_style_system", "penalty": p, "max_penalty": 3, "triggered": bool(p), "evidence": {"css_rules": css_rules}})
    if p:
        reasons.append("CSS 规则量过低，缺少完整设计系统证据。")
    no_real_interaction = not re.search(r"addEventListener|onsubmit|onclick\s*=|<details\b", text, re.I)
    p = 3.0 if ordinary_three and no_real_interaction else 0.0
    penalty += p
    checks.append({"id": "three_cards_without_behavior", "penalty": p, "max_penalty": 3, "triggered": bool(p), "evidence": {"ordinary_three_cards": ordinary_three, "behavior": not no_real_interaction}})
    if p:
        reasons.append("三卡片结构没有行为层。")
    return round2(clamp(penalty, 0, TEMPLATE_PENALTY_MAX)), checks, reasons


def audit_bytes(data: bytes, display_name: str, source: str | None = None) -> dict[str, Any]:
    encoding_ok = True
    bom = data.startswith(b"\xef\xbb\xbf")
    try:
        text = data.decode("utf-8-sig" if bom else "utf-8")
    except UnicodeDecodeError:
        encoding_ok = False
        text = data.decode("utf-8", errors="replace")
    parsed = parse_document(text)
    css = extract_css(text, parsed)
    dim: dict[str, dict[str, Any]] = {}
    all_findings: list[dict[str, str]] = []
    scorers = {
        "structure": score_structure(text, parsed),
        "responsive": score_responsive(text, css, parsed),
        "interaction": score_interaction(text, css, parsed),
        "visual_complexity": score_visual(text, css, parsed),
    }
    for name, (score, checks, findings) in scorers.items():
        dim[name] = {"score": score, "max": DIMENSION_MAX[name], "checks": checks}
        all_findings.extend({"dimension": name, "severity": "warning", "message": message} for message in findings)
    dep_score, dep_checks, dep_findings, deps = score_dependencies(parsed, css)
    dim["dependency_safety"] = {"score": dep_score, "max": DIMENSION_MAX["dependency_safety"], "checks": dep_checks}
    all_findings.extend({"dimension": "dependency_safety", "severity": "warning", "message": message} for message in dep_findings)
    a11y_score, a11y_checks, a11y_findings = score_accessibility(text, css, parsed)
    dim["accessibility"] = {"score": a11y_score, "max": DIMENSION_MAX["accessibility"], "checks": a11y_checks}
    all_findings.extend({"dimension": "accessibility", "severity": "warning", "message": message} for message in a11y_findings)
    penalty, penalty_checks, penalty_reasons = template_penalty(text, css, parsed)
    gross = round2(sum(item["score"] for item in dim.values()))
    final = round2(clamp(gross - penalty, 0, 100))
    if not encoding_ok:
        final = 0.0
        all_findings.insert(0, {"dimension": "structure", "severity": "error", "message": "文件不是合法 UTF-8。"})
    if bom:
        all_findings.append({"dimension": "structure", "severity": "warning", "message": "文件带 UTF-8 BOM；本项目契约要求无 BOM。"})
    return {
        "schema_version": "parallel-frontend-audit-item-v1",
        "scorer_version": SCORER_VERSION,
        "file": display_name,
        "source": source,
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "encoding": {"valid_utf8": encoding_ok, "bom": bom},
        "summary": {
            "gross_score": gross,
            "template_penalty": penalty,
            "final_score": final,
            "tier": "advanced" if final >= 75 and penalty <= 4 else "competent" if final >= 55 and penalty <= 10 else "template_or_incomplete",
        },
        "dimensions": dim,
        "template_penalty": {"score": penalty, "max": TEMPLATE_PENALTY_MAX, "checks": penalty_checks, "reasons": penalty_reasons},
        "inventory": {
            "elements": sum(parsed.starts.values()),
            "tags": dict(sorted(parsed.starts.items())),
            "css_bytes": len(css.encode("utf-8")),
            "script_bytes": sum(len(x.encode("utf-8")) for x in parsed.script_blocks),
            "external_dependencies": deps,
        },
        "findings": all_findings,
        "limitations": [
            "静态评分不执行 JavaScript，不能证明控件在真实浏览器中可用。",
            "静态评分不栅格化页面，不能证明断点无溢出、颜色达到 WCAG 对比度或视觉审美优良。",
            "复杂度得分只衡量可观察的结构/样式多样性，不等价于设计质量。",
        ],
    }


def audit_file(path: Path, display_name: str | None = None, include_source: bool = True) -> dict[str, Any]:
    data = path.read_bytes()
    return audit_bytes(data, display_name or path.name, str(path.resolve()) if include_source else None)


def build_report(paths: Iterable[Path], source_root: str | None = None) -> dict[str, Any]:
    items = [audit_file(path) for path in paths]
    ranking = sorted(((item["file"], item["summary"]["final_score"]) for item in items), key=lambda x: (-x[1], x[0]))
    return {
        "schema_version": "parallel-frontend-audit-report-v1",
        "contract_version": CONTRACT_VERSION,
        "scorer_version": SCORER_VERSION,
        "source_root": source_root,
        "sample_count": len(items),
        "ranking": [{"rank": i + 1, "file": name, "final_score": score} for i, (name, score) in enumerate(ranking)],
        "items": items,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="HTML 文件；目录会扫描其直属 *.html")
    parser.add_argument("--output", "-o", type=Path, help="将汇总报告写为 UTF-8 JSON")
    parser.add_argument("--source-root", help="写入报告的来源说明")
    parser.add_argument("--compact", action="store_true", help="标准输出使用紧凑 JSON")
    return parser.parse_args(argv)


def expand_inputs(inputs: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for item in inputs:
        if item.is_dir():
            result.extend(sorted(item.glob("*.html"), key=lambda p: p.name))
        elif item.is_file():
            result.append(item)
        else:
            raise FileNotFoundError(item)
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in result:
        resolved = item.resolve()
        if resolved not in seen:
            unique.append(item)
            seen.add(resolved)
    if not unique:
        raise ValueError("没有找到 HTML 输入")
    return unique


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = expand_inputs(args.inputs)
        report = build_report(paths, args.source_root)
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=None if args.compact else 2))
        return 0
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
