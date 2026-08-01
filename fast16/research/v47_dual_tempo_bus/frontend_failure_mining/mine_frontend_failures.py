#!/usr/bin/env python3
"""把本地 HTML 样本转成不含原文和远程 URL 的确定性前端失败指纹。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
FRONTEND = PROJECT / "fast16/research/parallel_frontend_v47"
SCORER = FRONTEND / "score_html.py"
SCORING_CONTRACT = FRONTEND / "scoring_contract.json"
GENERATOR_VERSION = "colorlm-v47-frontend-failure-miner-v1.0.0"
REPORT_FORMAT = "colorlm-v47-frontend-failure-report-v1"
CONTRACT_FORMAT = "colorlm-v47-frontend-negative-contract-v1"
VIEWPORTS = [375, 768, 1024, 1440]

sys.path.insert(0, str(FRONTEND))
from score_html import (  # noqa: E402
    CONTRACT_VERSION,
    GENERIC_CARD_RE,
    LANDMARK_TAGS,
    SCORER_VERSION,
    audit_bytes,
    elements,
    extract_css,
    external_dependencies,
    parse_document,
)


EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0000231A-\U000023F3"
    "]"
)
PLACEHOLDER_REMOTE_RE = re.compile(
    r"(?:placehold|placeholder|picsum|loremflickr|dummyimage|unsplash|pexels|pixabay|pravatar|randomuser)",
    re.I,
)
WIRING_RE = re.compile(
    r"(?:addEventListener|onclick\s*=|onchange\s*=|onsubmit\s*=|querySelector|classList\.|\.showModal\s*\(|\.toggle\s*\()",
    re.I,
)
MEDIA_QUERY_RE = re.compile(r"@media\s*\(([^)]+)\)", re.I)
BREAKPOINT_RE = re.compile(r"\d+(?:\.\d+)?(?:px|rem|em)", re.I)

CATEGORY_ORDER = [
    "default_three_cards",
    "emoji_ui_icons",
    "dead_controls",
    "remote_placeholder_assets",
    "visible_focus",
    "reduced_motion",
    "responsive_layout",
    "semantic_html",
    "form_labels",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def check_from(audit: dict[str, Any], dimension: str, check_id: str) -> dict[str, Any]:
    checks = audit["dimensions"][dimension]["checks"]
    matches = [item for item in checks if item["id"] == check_id]
    if len(matches) != 1:
        raise ValueError(f"评分器中找不到唯一检查项：{dimension}.{check_id}")
    return matches[0]


def penalty_from(audit: dict[str, Any], check_id: str) -> dict[str, Any]:
    checks = audit["template_penalty"]["checks"]
    matches = [item for item in checks if item["id"] == check_id]
    if len(matches) != 1:
        raise ValueError(f"评分器中找不到唯一模板检查项：{check_id}")
    return matches[0]


class EmojiContextCollector(HTMLParser):
    """只统计 emoji，不保留文本片段。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict[str, str]]] = []
        self.total = 0
        self.ui_candidates = 0
        self.codepoint_hashes: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.stack.append((tag.lower(), {k.lower(): (v or "") for k, v in attrs}))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag, attrs

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        matches = EMOJI_RE.findall(data)
        if not matches:
            return
        self.total += len(matches)
        self.codepoint_hashes.update(sha256_text(char) for char in matches)
        ancestors = self.stack[-4:]
        explicit_ui = any(
            tag in {"a", "button", "summary"}
            or attrs.get("role", "") in {"button", "link", "tab", "switch"}
            or re.search(r"(?:^|[-_])(icon|emoji|badge)(?:$|[-_])", " ".join((attrs.get("class", ""), attrs.get("id", ""))), re.I)
            for tag, attrs in ancestors
        )
        remainder = EMOJI_RE.sub("", data).replace("\ufe0f", "").replace("\u200d", "").strip()
        if explicit_ui or not remainder:
            self.ui_candidates += len(matches)


def decode_utf8(data: bytes, name: str) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} 不是合法 UTF-8") from error


def feature(
    failed: bool,
    evidence: dict[str, Any],
    *,
    severity: str = "hard",
) -> dict[str, Any]:
    return {"failed": bool(failed), "severity": severity, "evidence": evidence}


def mine_sample(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    text = decode_utf8(data, path.name)
    parsed = parse_document(text)
    css = extract_css(text, parsed)
    audit = audit_bytes(data, path.name, source=None)

    template_check = penalty_from(audit, "exact_three_card_signature")
    class_counts = Counter(class_name.lower() for class_name in parsed.classes)
    generic_card_repeat = max(
        (count for class_name, count in class_counts.items() if GENERIC_CARD_RE.search(class_name)),
        default=0,
    )
    grid_three_columns = bool(re.search(r"repeat\(\s*3\s*,", css, re.I))
    three_card_failed = bool(template_check["triggered"])

    emoji_parser = EmojiContextCollector()
    emoji_parser.feed(text)
    emoji_parser.close()

    links = elements(parsed, "a")
    placeholder_links = [
        item for item in links
        if item.attrs.get("href", "").strip().lower() in {"", "#", "javascript:void(0)", "javascript:void(0);"}
    ]
    buttons = elements(parsed, "button")
    scripts = "\n".join(parsed.script_blocks)
    submit_buttons = [item for item in buttons if item.attrs.get("type", "").lower() == "submit"]
    inert_button_risk = bool(buttons and not scripts.strip() and not submit_buttons)
    form_path = check_from(audit, "interaction", "form_path")
    dead_control_failed = bool(placeholder_links or inert_button_risk or not form_path["passed"])

    deps = external_dependencies(parsed, css)
    remote_assets = [
        item for item in deps
        if item["active"]
        and (item["tag"] in {"img", "video", "audio", "source", "css"} or item["attribute"] == "poster")
    ]
    remote_asset_hashes = sorted(
        {
            sha256_text(f"{item['tag']}\0{item['attribute']}\0{item['url']}")
            for item in remote_assets
        }
    )
    remote_host_hashes = sorted({sha256_text(item["host"]) for item in remote_assets})
    placeholder_asset_count = sum(
        bool(PLACEHOLDER_REMOTE_RE.search(item["url"]) or PLACEHOLDER_REMOTE_RE.search(item["host"]))
        for item in remote_assets
    )

    focus_check = check_from(audit, "accessibility", "visible_focus")
    motion_check = check_from(audit, "accessibility", "reduced_motion")
    viewport_check = check_from(audit, "responsive", "viewport")
    media_check = check_from(audit, "responsive", "media_queries")
    overflow_check = check_from(audit, "responsive", "no_obvious_horizontal_overflow")
    media_queries = MEDIA_QUERY_RE.findall(css)
    breakpoints = sorted(set(BREAKPOINT_RE.findall(" ".join(media_queries))), key=str.lower)
    responsive_failed = not (
        viewport_check["passed"] and media_check["earned"] > 0 and overflow_check["passed"]
    )

    h1_check = check_from(audit, "structure", "one_h1")
    heading_check = check_from(audit, "accessibility", "heading_order")
    semantic_click_check = check_from(audit, "accessibility", "semantic_click_targets")
    landmarks = sorted(tag for tag in LANDMARK_TAGS if parsed.starts[tag])
    core_landmarks = {"main", "nav", "header", "footer"}
    missing_core_landmarks = sorted(core_landmarks - set(landmarks))
    semantic_failed = not (
        h1_check["passed"]
        and heading_check["passed"]
        and semantic_click_check["passed"]
        and "main" in landmarks
        and len(landmarks) >= 3
    )

    labels_check = check_from(audit, "accessibility", "form_labels")
    label_evidence = labels_check["evidence"]
    form_label_failed = not labels_check["passed"]

    failures = {
        "default_three_cards": feature(
            three_card_failed,
            {
                "contract_triggered": bool(template_check["triggered"]),
                "generic_card_class_max_repeat": generic_card_repeat,
                "article_count": parsed.starts["article"],
                "grid_three_columns_declared": grid_three_columns,
                "penalty": template_check["penalty"],
            },
        ),
        "emoji_ui_icons": feature(
            emoji_parser.ui_candidates > 0,
            {
                "emoji_total": emoji_parser.total,
                "ui_icon_candidate_count": emoji_parser.ui_candidates,
                "codepoint_sha256": sorted(emoji_parser.codepoint_hashes),
            },
        ),
        "dead_controls": feature(
            dead_control_failed,
            {
                "link_count": len(links),
                "placeholder_link_count": len(placeholder_links),
                "button_count": len(buttons),
                "submit_button_count": len(submit_buttons),
                "event_wiring_signal_count": len(WIRING_RE.findall(text)),
                "inert_button_risk": inert_button_risk,
                "form_path_passed": bool(form_path["passed"]),
            },
        ),
        "remote_placeholder_assets": feature(
            bool(remote_assets),
            {
                "remote_asset_count": len(remote_assets),
                "placeholder_like_count": placeholder_asset_count,
                "asset_identifier_sha256": remote_asset_hashes,
                "host_sha256": remote_host_hashes,
                "raw_urls_retained": False,
            },
        ),
        "visible_focus": feature(
            not focus_check["passed"],
            {"visible_focus_rule_passed": bool(focus_check["passed"])},
        ),
        "reduced_motion": feature(
            not motion_check["passed"],
            {
                "motion_declared": bool(motion_check["evidence"]["motion"]),
                "reduced_motion_rule": bool(motion_check["evidence"]["reduced_motion_rule"]),
            },
        ),
        "responsive_layout": feature(
            responsive_failed,
            {
                "viewport_meta_passed": bool(viewport_check["passed"]),
                "media_query_count": len(media_queries),
                "declared_breakpoints": breakpoints,
                "obvious_overflow_check_passed": bool(overflow_check["passed"]),
                "required_runtime_viewports": VIEWPORTS,
            },
        ),
        "semantic_html": feature(
            semantic_failed,
            {
                "landmarks_present": landmarks,
                "missing_core_landmarks": missing_core_landmarks,
                "h1_count": parsed.starts["h1"],
                "heading_order_passed": bool(heading_check["passed"]),
                "semantic_click_targets_passed": bool(semantic_click_check["passed"]),
            },
        ),
        "form_labels": feature(
            form_label_failed,
            {
                "control_count": int(label_evidence["controls"]),
                "unlabeled_control_count": len(label_evidence["unlabeled_lines"]),
                "line_numbers_retained": False,
            },
        ),
    }
    if list(failures) != CATEGORY_ORDER:
        raise AssertionError("失败特征顺序与冻结类别不一致")

    content_sha = sha256_bytes(data)
    fingerprint_payload = {
        key: {"failed": value["failed"], "evidence": value["evidence"]}
        for key, value in failures.items()
    }
    return {
        "sample_id": f"desktop-{content_sha[:16]}",
        "source_label": path.name,
        "source_sha256": content_sha,
        "source_bytes": len(data),
        "failure_fingerprint_sha256": sha256_bytes(canonical_bytes(fingerprint_payload)),
        "scoring_snapshot": {
            "gross_score": audit["summary"]["gross_score"],
            "template_penalty": audit["summary"]["template_penalty"],
            "final_score": audit["summary"]["final_score"],
            "tier": audit["summary"]["tier"],
            "dimension_scores": {
                name: value["score"] for name, value in sorted(audit["dimensions"].items())
            },
        },
        "failures": failures,
        "privacy": {
            "source_html_retained": False,
            "source_text_retained": False,
            "remote_urls_retained": False,
        },
    }


def constraint_definitions() -> list[dict[str, Any]]:
    definitions = [
        ("default_three_cards", "forbid_default_three_cards", True, "不得退化为默认三卡片或三项卖点模板；信息架构必须由任务决定。", "hard"),
        ("emoji_ui_icons", "forbid_emoji_icons", True, "不得用 emoji 充当 UI 图标；图标使用统一内联 SVG，并提供可访问名称。", "hard"),
        ("dead_controls", "forbid_dead_controls", True, "不得出现空链接、假按钮或无提交路径的表单；每个控件必须产生可验证反馈。", "hard"),
        ("remote_placeholder_assets", "forbid_remote_assets", True, "不得依赖远程图片、占位图或第三方视觉资源；页面应可离线复现。", "hard"),
        ("visible_focus", "require_visible_focus", True, "所有键盘可达控件必须具有清晰可见的 focus-visible 状态。", "hard"),
        ("reduced_motion", "require_reduced_motion", True, "使用动画或过渡时必须支持 prefers-reduced-motion: reduce。", "hard"),
        ("responsive_layout", "viewports", VIEWPORTS, "必须在 375/768/1024/1440px 检查布局、交互与横向溢出，窄屏保持信息等价。", "hard"),
        ("semantic_html", "require_semantic_html", True, "必须使用合理的 header/nav/main/footer、连续标题层级和原生交互元素。", "hard"),
        ("form_labels", "require_form_labels", True, "每个非隐藏表单控件必须通过 label、aria-label 或 aria-labelledby 获得名称。", "hard"),
    ]
    return [
        {
            "id": feature_id,
            "severity": severity,
            "dataset_key": dataset_key,
            "dataset_value": value,
            "prompt_requirement": prompt,
            "teacher_filter": {"reject_when": f"failures.{feature_id}.failed == true"},
        }
        for feature_id, dataset_key, value, prompt, severity in definitions
    ]


def build_artifacts(input_dir: Path, expected_samples: int = 6) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = sorted(input_dir.glob("*.html"), key=lambda item: item.name.casefold())
    if len(paths) != expected_samples:
        raise ValueError(f"需要恰好 {expected_samples} 个 HTML，实际找到 {len(paths)} 个")
    samples = [mine_sample(path) for path in paths]
    if len({item["source_sha256"] for item in samples}) != len(samples):
        raise ValueError("存在字节完全重复的 HTML；拒绝重复计入证据")

    constraints = constraint_definitions()
    support = []
    for constraint in constraints:
        failed_samples = [
            item["sample_id"]
            for item in samples
            if item["failures"][constraint["id"]]["failed"]
        ]
        support.append(
            {
                "id": constraint["id"],
                "failure_sample_count": len(failed_samples),
                "failure_rate": round(len(failed_samples) / len(samples), 6),
                "sample_ids": failed_samples,
            }
        )

    projection = {
        constraint["dataset_key"]: constraint["dataset_value"]
        for constraint in constraints
    }
    contract_core: dict[str, Any] = {
        "format": CONTRACT_FORMAT,
        "generator_version": GENERATOR_VERSION,
        "status": "frozen_negative_contract",
        "scope": "前端教师筛选与 Parallel Genome 数据 anti_pattern_contract 投影",
        "claim_limit": "静态负样本只能证明可观察源码信号；真实交互、视觉与四档视口仍需浏览器验证。",
        "constraints": constraints,
        "dataset_projection": {
            "target_field": "anti_pattern_contract",
            "merge_strategy": "strict_deep_merge_reject_conflicts",
            "value": projection,
        },
        "teacher_screening": {
            "policy": "all_hard_constraints_must_pass",
            "required_failure_state": {item: False for item in CATEGORY_ORDER},
            "runtime_checks_not_proven_by_static_mining": [
                "四档视口的实际横向溢出",
                "点击、键盘、ESC 与焦点回退",
                "计算后正文对比度至少 4.5:1",
                "控制台无错误",
            ],
        },
        "evidence_support": support,
    }
    contract_core["constraint_set_sha256"] = sha256_bytes(canonical_bytes(contract_core))

    report = {
        "format": REPORT_FORMAT,
        "generator_version": GENERATOR_VERSION,
        "source_contract": {
            "contract_version": CONTRACT_VERSION,
            "scorer_version": SCORER_VERSION,
            "scorer_path": "fast16/research/parallel_frontend_v47/score_html.py",
            "scorer_sha256": sha256_file(SCORER),
            "scoring_contract_path": "fast16/research/parallel_frontend_v47/scoring_contract.json",
            "scoring_contract_sha256": sha256_file(SCORING_CONTRACT),
        },
        "sample_count": len(samples),
        "samples": samples,
        "aggregate_support": support,
        "negative_contract_sha256": sha256_bytes(pretty_bytes(contract_core)),
        "privacy": {
            "whole_html_copied": False,
            "source_text_copied": False,
            "remote_urls_copied": False,
            "remote_identifiers": "SHA-256 only",
            "absolute_source_paths_copied": False,
        },
        "limitations": [
            "静态检测不执行 JavaScript。",
            "静态检测不渲染 375/768/1024/1440px 截图。",
            "emoji 上下文检测是保守启发式；最终拒绝由教师筛选和浏览器复核共同决定。",
        ],
    }
    return report, contract_core


def assert_private_payload(payload: bytes) -> None:
    lowered = payload.lower()
    forbidden = [b"<html", b"<!doctype", b"http://", b"https://", b"www."]
    hits = [token.decode("ascii") for token in forbidden if token in lowered]
    if hits:
        raise AssertionError(f"产物泄露 HTML 或远程 URL 标记：{hits}")
    if payload.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("产物不得带 UTF-8 BOM")


def write_artifacts(output_dir: Path, report: dict[str, Any], contract: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "report.json": pretty_bytes(report),
        "negative_contract.json": pretty_bytes(contract),
    }
    for name, payload in payloads.items():
        assert_private_payload(payload)
        (output_dir / name).write_bytes(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--expected-samples", type=int, default=6)
    parser.add_argument("--check", action="store_true", help="不写文件，只检查现有产物是否与重新生成结果一致")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report, contract = build_artifacts(args.input_dir, args.expected_samples)
    expected = {
        "report.json": pretty_bytes(report),
        "negative_contract.json": pretty_bytes(contract),
    }
    if args.check:
        mismatches = [
            name for name, payload in expected.items()
            if not (args.output_dir / name).is_file() or (args.output_dir / name).read_bytes() != payload
        ]
        if mismatches:
            raise SystemExit(f"现有产物与确定性重建不一致：{', '.join(mismatches)}")
        print(json.dumps({"ok": True, "mode": "check", "sample_count": len(report["samples"])}, ensure_ascii=False))
        return 0
    write_artifacts(args.output_dir, report, contract)
    print(
        json.dumps(
            {
                "ok": True,
                "sample_count": len(report["samples"]),
                "report_sha256": sha256_file(args.output_dir / "report.json"),
                "contract_sha256": sha256_file(args.output_dir / "negative_contract.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
