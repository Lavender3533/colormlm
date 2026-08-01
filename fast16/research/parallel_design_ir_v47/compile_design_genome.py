#!/usr/bin/env python3
"""把闭集 Design Genome + copy slots 确定性展开为自包含 HTML/CSS/JS。"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

from ir_core import IRError, canonical_text, load_ir, load_slots, utf8_size, write_utf8_no_bom


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "component_catalog.json"
PALETTES = {
    "cyan": ("#0e7490", "#22d3ee"), "indigo": ("#4338ca", "#818cf8"),
    "coral": ("#c2410c", "#fb7185"), "green": ("#047857", "#34d399"),
    "amber": ("#b45309", "#fbbf24"), "violet": ("#7e22ce", "#c084fc"),
    "blue": ("#1d4ed8", "#60a5fa"),
}
BREAKPOINTS = {0: (375, 768, 1200), 1: (390, 820, 1280), 2: (480, 960, 1440)}
RADIUS = {"square": "0", "soft": ".75rem", "round": "1.5rem"}
GAP = {"compact": ".7rem", "normal": "1rem", "airy": "1.5rem"}


def esc(value: Any, quote: bool = True) -> str:
    return html.escape(str(value), quote=quote)


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "value"


def load_catalog() -> dict[str, Any]:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != "design-genome-component-catalog-v1":
        raise IRError("组件目录版本错误")
    return data["components"]


def section_title(cid: str, label: str) -> str:
    return f'<h2 id="{cid}-title">{esc(label)}</h2>'


def render_hero(cid: str, spec: dict[str, Any], title: str) -> str:
    return (
        f'<section class="hero panel" id="{cid}" aria-labelledby="{cid}-title">'
        f'<div><span class="eyebrow">{esc(spec["label"])}</span><h2 id="{cid}-title">{esc(title)}</h2>'
        f'<p>{esc(spec["text"])}</p></div><svg viewBox="0 0 240 140" role="img" aria-labelledby="{cid}-visual-title">'
        f'<title id="{cid}-visual-title">{esc(spec["label"])}抽象构图</title><path d="M12 112Q68 18 122 78T228 28"/>'
        '<circle cx="55" cy="72" r="17"/><circle cx="176" cy="52" r="27"/></svg></section>'
    )


def render_filters(cid: str, spec: dict[str, Any]) -> str:
    controls: list[str] = []
    for index, field in enumerate(spec["fields"]):
        control_id = f"{cid}-{index}"
        options = "".join(f'<option value="{slug(value)}">{esc(value)}</option>' for value in field["options"])
        controls.append(f'<label for="{control_id}">{esc(field["label"])}</label><select id="{control_id}" data-filter-control>{options}</select>')
    return (
        f'<section class="filters panel" id="{cid}" data-component="filters" aria-labelledby="{cid}-title">'
        f'{section_title(cid, spec["label"])}<div class="control-row">{"".join(controls)}'
        '<button type="button" data-reset>重置筛选</button></div></section>'
    )


def render_table(cid: str, spec: dict[str, Any], overlay_id: str | None) -> str:
    headers = "".join(
        f'<th scope="col"><button type="button" data-sort="{index}" aria-label="按{esc(col)}排序">{esc(col)} <span aria-hidden="true">↕</span></button></th>'
        for index, col in enumerate(spec["cols"])
    )
    if overlay_id:
        headers += '<th scope="col">操作</th>'
    rows: list[str] = []
    cards: list[str] = []
    for row_index, row in enumerate(spec["rows"]):
        cells = "".join(f'<td>{esc(value)}</td>' for value in row)
        action = f'<td><button type="button" data-open="{overlay_id}">查看详情</button></td>' if overlay_id else ""
        text = " ".join(str(x) for x in row)
        rows.append(f'<tr data-filter-item data-row="{row_index}" data-search="{esc(text.lower())}">{cells}{action}</tr>')
        pairs = "".join(f'<span><b>{esc(col)}</b>{esc(value)}</span>' for col, value in zip(spec["cols"], row))
        card_action = f'<button type="button" data-open="{overlay_id}">查看详情</button>' if overlay_id else ""
        cards.append(f'<article class="mobile-card" data-filter-item data-search="{esc(text.lower())}">{pairs}{card_action}</article>')
    return (
        f'<section class="table-section panel" id="{cid}" data-component="table" aria-labelledby="{cid}-title">'
        f'{section_title(cid, spec["label"])}<div class="table-scroll" role="region" aria-label="{esc(spec["label"])}，可排序" tabindex="0">'
        f'<table><caption>{esc(spec["label"])}</caption><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        f'<div class="mobile-list" aria-label="{esc(spec["label"])}卡片列表">{"".join(cards)}</div></section>'
    )


def render_overlay(cid: str, spec: dict[str, Any], kind: str, next_dialog: str | None) -> str:
    fields = spec.get("fields", [])
    details = "".join(f'<li>{esc(item)}</li>' for item in fields)
    body = f'<ul class="detail-list">{details}</ul>' if details else f'<p>{esc(spec.get("text", "请复核信息后继续。"))}</p>'
    open_next = f'<button type="button" class="danger" data-open="{next_dialog}">确认告警</button>' if kind == "drawer" and next_dialog else ""
    return (
        f'<section class="overlay {kind}" id="{cid}" role="dialog" aria-modal="true" aria-labelledby="{cid}-title" hidden tabindex="-1">'
        f'<div class="overlay-card"><header>{section_title(cid, spec["label"])}<button type="button" data-close aria-label="关闭{esc(spec["label"])}">×</button></header>'
        f'<div class="overlay-body">{body}</div><footer>{open_next}<button type="button" data-close>{"取消" if kind == "dialog" else "关闭"}</button>'
        f'{"<button type=\"button\" data-confirm>确认</button>" if kind == "dialog" else ""}</footer></div></section>'
    )


def render_products(cid: str, spec: dict[str, Any], dialog_id: str | None) -> str:
    items: list[str] = []
    for index, (name, price, meta) in enumerate(spec["items"]):
        preview = f'<button type="button" data-open="{dialog_id}">快速预览</button>' if dialog_id else ""
        items.append(
            f'<article class="product" data-filter-item data-search="{esc((name + meta).lower())}"><div class="product-art" aria-hidden="true">0{index + 1}</div>'
            f'<h3>{esc(name)}</h3><p>{esc(meta)}</p><strong class="price">{esc(price)}</strong><div>{preview}<button type="button" data-add-bag>加入购物袋</button></div></article>'
        )
    return f'<section class="panel" id="{cid}" aria-labelledby="{cid}-title">{section_title(cid, spec["label"])}<div class="product-grid">{"".join(items)}</div></section>'


def render_timeline(cid: str, spec: dict[str, Any]) -> str:
    items = "".join(f'<li><span>{index:02d}</span><div><h3>{esc(name)}</h3><p>{esc(text)}</p></div></li>' for index, (name, text) in enumerate(spec["items"], 1))
    return f'<section class="timeline panel" id="{cid}" aria-labelledby="{cid}-title">{section_title(cid, spec["label"])}<ol>{items}</ol></section>'


def render_compare(cid: str, spec: dict[str, Any]) -> str:
    return (
        f'<section class="compare panel" id="{cid}" aria-labelledby="{cid}-title">{section_title(cid, spec["label"])}'
        f'<div class="compare-stage"><div class="before">原方案</div><div class="after" data-after>改进方案</div></div>'
        f'<label for="{cid}-range">前后对比位置 <output id="{cid}-out">{spec["value"]}%</output></label>'
        f'<input id="{cid}-range" type="range" min="{spec["min"]}" max="{spec["max"]}" value="{spec["value"]}" data-compare aria-describedby="{cid}-out"></section>'
    )


def render_form(cid: str, spec: dict[str, Any]) -> str:
    fields: list[str] = []
    for field in spec["fields"]:
        fid = f"{cid}-{field['id']}"
        if field["type"] == "select":
            options = "".join(f'<option>{esc(value)}</option>' for value in field["options"])
            control = f'<select id="{fid}" required aria-describedby="{fid}-error">{options}</select>'
        else:
            extra = ' min="1" max="8"' if field["type"] == "number" else ""
            control = f'<input id="{fid}" type="{esc(field["type"])}" required{extra} aria-describedby="{fid}-error">'
        fields.append(f'<div><label for="{fid}">{esc(field["label"])}</label>{control}<span class="field-error" id="{fid}-error">必填，请检查此字段。</span></div>')
    return (
        f'<section class="booking panel" id="{cid}" aria-labelledby="{cid}-title">{section_title(cid, spec["label"])}'
        f'<form data-validate>{"".join(fields)}<aside class="fee"><h3>费用摘要</h3><p>材料与场地：<strong>¥120</strong></p></aside>'
        '<button type="submit">提交预约</button><p role="status" aria-live="polite" data-form-status></p></form></section>'
    )


def render_sidebar(cid: str, spec: dict[str, Any]) -> str:
    links = "".join(f'<li><a href="#main" data-nav>{esc(item)}</a></li>' for item in spec["items"])
    return f'<aside class="sidebar panel" id="{cid}" aria-labelledby="{cid}-title"><details open><summary id="{cid}-title">{esc(spec["label"])}</summary><nav aria-label="{esc(spec["label"])}"><ul>{links}</ul></nav></details></aside>'


def render_code(cid: str, spec: dict[str, Any]) -> str:
    return (
        f'<section class="code-panel panel" id="{cid}" aria-labelledby="{cid}-title">{section_title(cid, spec["label"])}'
        f'<button type="button" data-copy="{cid}-code">复制代码</button><pre tabindex="0"><code id="{cid}-code">{esc(spec["text"])}</code></pre>'
        '<p role="status" aria-live="polite" data-copy-status></p></section>'
    )


def render_tabs(cid: str, spec: dict[str, Any]) -> str:
    buttons: list[str] = []
    panels: list[str] = []
    for index, item in enumerate(spec["items"]):
        selected = "true" if index == 0 else "false"
        hidden = "" if index == 0 else " hidden"
        buttons.append(f'<button type="button" role="tab" aria-selected="{selected}" aria-controls="{cid}-panel-{index}" id="{cid}-tab-{index}">{esc(item)}</button>')
        panels.append(f'<div role="tabpanel" id="{cid}-panel-{index}" aria-labelledby="{cid}-tab-{index}"{hidden}><p>{esc(item)}：结构化响应内容。</p></div>')
    return f'<section class="tabs panel" id="{cid}" aria-labelledby="{cid}-title">{section_title(cid, spec["label"])}<div role="tablist">{"".join(buttons)}</div>{"".join(panels)}</section>'


def render_toggles(cid: str, spec: dict[str, Any]) -> str:
    rows = "".join(
        f'<li><label for="{cid}-{index}"><span>{esc(item)}</span><input id="{cid}-{index}" type="checkbox" role="switch" aria-checked="false"></label></li>'
        for index, item in enumerate(spec["items"])
    )
    return f'<section class="toggles panel" id="{cid}" aria-labelledby="{cid}-title">{section_title(cid, spec["label"])}<ul>{rows}</ul><p role="status" aria-live="polite" data-save-status>已保存</p></section>'


def render_schedule(cid: str, spec: dict[str, Any]) -> str:
    groups: list[str] = []
    for index, (time, stage, artist, state) in enumerate(spec["items"]):
        groups.append(
            f'<details class="schedule-item" data-filter-item data-search="{esc((stage + state).lower())}" open><summary><time>{esc(time)}</time> {esc(stage)}</summary>'
            f'<div><h3>{esc(artist)}</h3><p class="state">{esc(state)}</p><button type="button" data-favorite aria-pressed="{str(index == 0).lower()}">收藏</button></div></details>'
        )
    return f'<section class="schedule panel" id="{cid}" aria-labelledby="{cid}-title">{section_title(cid, spec["label"])}<p class="conflict" role="status">时间冲突提示会同时显示文字与图标。</p>{"".join(groups)}</section>'


def render_chart(cid: str, spec: dict[str, Any]) -> str:
    bars: list[str] = []
    labels: list[str] = []
    for index, value in enumerate(spec["values"]):
        x = 35 + index * 70
        height = value * 3
        y = 210 - height
        bars.append(f'<rect data-bar data-inspect tabindex="0" role="button" aria-label="查看 {2020 + index} 年 {value}%" x="{x}" y="{y}" width="38" height="{height}" rx="5"><title>{2020 + index}：{value}%</title></rect>')
        labels.append(f'<text x="{x + 19}" y="232" text-anchor="middle">{2020 + index}</text>')
    return (
        f'<section class="chart panel" id="{cid}" aria-labelledby="{cid}-title">{section_title(cid, spec["label"])}'
        f'<svg viewBox="0 0 400 250" role="img" aria-labelledby="{cid}-chart-title {cid}-chart-desc"><title id="{cid}-chart-title">{esc(spec["label"])}</title>'
        f'<desc id="{cid}-chart-desc">{esc(spec["caption"])}</desc><g>{"".join(bars)}</g><g class="axis">{"".join(labels)}</g></svg></section>'
    )


def render_note(cid: str, spec: dict[str, Any]) -> str:
    return f'<aside class="note panel" id="{cid}" aria-labelledby="{cid}-title">{section_title(cid, spec["label"])}<p>{esc(spec["text"])}</p></aside>'


def render_metrics(cid: str, spec: dict[str, Any]) -> str:
    items = "".join(f'<div><span>{esc(label)}</span><strong>{esc(value)}</strong></div>' for label, value in spec["items"])
    return f'<section class="metrics panel" id="{cid}" aria-labelledby="{cid}-title">{section_title(cid, spec["label"])}<div class="metric-grid">{items}</div></section>'


def render_bag(cid: str, spec: dict[str, Any]) -> str:
    return f'<section class="bag panel" id="{cid}" aria-labelledby="{cid}-title"><h2 id="{cid}-title">{esc(spec["label"])}</h2><p>购物袋 <output data-bag-count aria-live="polite">{spec["value"]}</output> 件</p></section>'


def css_for(ir: dict[str, Any]) -> str:
    mode, palette, density, shape = ir["y"]
    accent, accent_soft = PALETTES[palette]
    if mode == "dark":
        bg, surface, ink, muted, line = "#0b1220", "#162033", "#f8fafc", "#a7b3c6", "#334155"
    elif mode == "paper":
        bg, surface, ink, muted, line = "#f4efe3", "#fffaf0", "#241f19", "#6b6258", "#c9bca8"
    else:
        bg, surface, ink, muted, line = "#f3f6fb", "#ffffff", "#162033", "#5d6878", "#cbd5e1"
    small, middle, large = BREAKPOINTS[ir["l"][2]]
    layout = ir["l"][0]
    layout_rule = {
        "dashboard": "grid-template-columns:minmax(0,2fr) minmax(16rem,.7fr)",
        "editorial": "grid-template-columns:minmax(0,1.25fr) minmax(0,.75fr)",
        "timeline": "grid-template-columns:minmax(15rem,.7fr) minmax(0,1.3fr)",
        "split": "grid-template-columns:minmax(15rem,.65fr) minmax(0,1.35fr)",
        "docs": "grid-template-columns:minmax(14rem,.45fr) minmax(0,1.55fr)",
        "settings": "grid-template-columns:minmax(13rem,.45fr) minmax(0,1.55fr)",
        "schedule": "grid-template-columns:minmax(0,.6fr) minmax(0,1.4fr)",
        "story": "grid-template-columns:minmax(0,1.4fr) minmax(15rem,.6fr)",
    }[layout]
    return f"""
:root{{--bg:{bg};--surface:{surface};--ink:{ink};--muted:{muted};--line:{line};--accent:{accent};--accent-soft:{accent_soft};--radius:{RADIUS[shape]};--gap:{GAP[density]};--shadow:0 18px 50px #0206171c}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:radial-gradient(circle at 92% 0,var(--accent-soft)22,transparent 35%),var(--bg);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,sans-serif}}
[hidden]{{display:none!important}}.skip{{position:fixed;left:-999px;top:.5rem;z-index:200}}.skip:focus{{left:.5rem;background:var(--surface);padding:.7rem 1rem}}
:focus-visible{{outline:3px solid var(--accent-soft);outline-offset:3px}}button,select,input{{font:inherit;color:inherit}}button,select,input:not([type=checkbox]){{min-height:2.65rem;border:1px solid var(--line);border-radius:calc(var(--radius)*.65);background:var(--surface);padding:.5rem .8rem}}
button{{cursor:pointer;transition:transform .2s,background-color .2s,color .2s}}button:hover{{transform:translateY(-1px);border-color:var(--accent)}}button:active{{transform:translateY(1px)}}
.site-header,.site-footer,.page-shell{{width:min({large}px,calc(100% - 2rem));margin:auto}}.site-header{{display:flex;align-items:end;justify-content:space-between;gap:1rem;padding:1.4rem 0;border-bottom:1px solid var(--line)}}
.brand{{display:flex;gap:.8rem;align-items:center}}.brand svg{{width:3rem;height:3rem;fill:none;stroke:var(--accent-soft);stroke-width:4}}h1{{font-size:clamp(1.8rem,4vw,3.8rem);line-height:1;margin:0;letter-spacing:-.04em}}h2{{font-size:clamp(1.15rem,2vw,1.65rem);margin:0 0 .8rem}}h3{{margin:.25rem 0}}
.lede{{max-width:44rem;color:var(--muted);margin:.5rem 0 0}}.page-shell{{display:grid;{layout_rule};gap:var(--gap);padding:var(--gap) 0 3rem;align-items:start}}.panel{{min-width:0;background:color-mix(in srgb,var(--surface) 94%,transparent);border:1px solid var(--line);border-radius:var(--radius);padding:clamp(.85rem,2vw,1.35rem);box-shadow:var(--shadow)}}
.hero{{grid-column:1/-1;display:grid;grid-template-columns:1.3fr .7fr;gap:1rem;align-items:center;overflow:hidden}}.hero svg{{width:100%;max-height:12rem;fill:var(--accent-soft);fill-opacity:.12;stroke:var(--accent);stroke-width:5}}.eyebrow{{color:var(--accent-soft);font-weight:800;text-transform:uppercase;letter-spacing:.12em}}
.filters{{grid-column:1/-1}}.control-row{{display:flex;align-items:end;gap:.7rem;flex-wrap:wrap}}label{{font-weight:650}}.control-row label{{align-self:center}}.table-section{{grid-column:1/-1}}.table-scroll{{overflow:auto;max-width:100%}}table{{width:100%;border-collapse:collapse;min-width:42rem}}caption{{text-align:left;color:var(--muted);padding:.4rem 0}}th,td{{text-align:left;padding:.72rem;border-bottom:1px solid var(--line)}}th button{{border:0;background:transparent;padding:.25rem;min-height:auto;font-weight:800}}tbody tr:hover{{background:var(--accent-soft)18}}.mobile-list{{display:none;gap:.7rem}}.mobile-card{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.65rem;padding:1rem;border:1px solid var(--line);border-radius:var(--radius)}}.mobile-card span{{display:grid;color:var(--muted)}}.mobile-card b{{color:var(--ink)}}
.metric-grid,.product-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));gap:.75rem}}.metric-grid>div{{border-left:4px solid var(--accent);padding:.7rem;background:var(--bg)}}.metric-grid strong{{display:block;font-size:1.7rem}}.product{{padding:1rem;border:1px solid var(--line);border-radius:var(--radius);display:grid;gap:.55rem}}.product-art{{aspect-ratio:4/3;display:grid;place-items:center;background:linear-gradient(135deg,var(--accent),var(--accent-soft));color:#fff;font-size:2.6rem;font-weight:900}}.price{{font-size:1.35rem}}.product>div:last-child{{display:flex;gap:.45rem;flex-wrap:wrap}}
.timeline ol{{list-style:none;margin:0;padding:0}}.timeline li{{display:grid;grid-template-columns:3rem 1fr;gap:1rem;padding:1rem 0;border-top:1px solid var(--line)}}.timeline li>span{{display:grid;place-items:center;width:2.5rem;height:2.5rem;border-radius:50%;background:var(--accent);color:#fff;font-weight:800}}.compare-stage{{position:relative;min-height:14rem;background:linear-gradient(130deg,var(--line),var(--surface));overflow:hidden;border-radius:var(--radius)}}.compare-stage>div{{position:absolute;inset:0;display:grid;place-items:center;font-size:1.4rem;font-weight:850}}.compare-stage .after{{background:linear-gradient(130deg,var(--accent),var(--accent-soft));color:#fff;clip-path:inset(0 48% 0 0)}}.compare input{{width:100%}}
.booking form{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.85rem}}.booking form>div{{display:grid;gap:.3rem}}.field-error{{display:none;color:#b91c1c;font-weight:700}}[aria-invalid=true]+.field-error{{display:block}}.fee{{grid-column:1/-1;border-left:4px solid var(--accent);padding:.7rem 1rem;background:var(--bg)}}
.sidebar{{position:sticky;top:1rem}}.sidebar summary{{font-size:1.2rem;font-weight:800;cursor:pointer}}.sidebar ul{{list-style:none;padding:0}}.sidebar a{{display:block;color:var(--ink);padding:.55rem;border-radius:.4rem;text-decoration:none}}.sidebar a:hover{{background:var(--accent-soft)22}}.code-panel pre{{overflow-x:auto;max-width:100%;padding:1rem;background:#07111f;color:#d7f6ff;border-radius:var(--radius)}}.tabs [role=tablist]{{display:flex;gap:.4rem;flex-wrap:wrap}}[role=tab][aria-selected=true]{{background:var(--accent);color:#fff}}[role=tabpanel]{{padding:1rem 0}}
.toggles ul{{list-style:none;padding:0}}.toggles li{{border-top:1px solid var(--line)}}.toggles label{{display:flex;justify-content:space-between;gap:1rem;padding:1rem 0}}[role=switch]{{width:1.4rem;height:1.4rem;accent-color:var(--accent)}}.schedule-item{{border-top:1px solid var(--line);padding:.7rem 0}}.schedule-item summary{{cursor:pointer;font-weight:800}}.schedule-item>div{{display:flex;align-items:center;justify-content:space-between;gap:.7rem;flex-wrap:wrap}}.conflict{{border:2px solid #d97706;padding:.7rem;border-radius:var(--radius)}}[data-favorite][aria-pressed=true]{{background:var(--accent);color:#fff}}
.chart svg{{display:block;width:100%;height:auto;max-width:45rem;margin:auto;overflow:visible}}.chart rect{{fill:var(--accent);transition:height .25s,y .25s}}.axis{{fill:var(--muted);font-size:12px}}.note{{border-left:5px solid var(--accent)}}.bag output{{display:inline-grid;place-items:center;min-width:2rem;height:2rem;border-radius:50%;background:var(--accent);color:#fff;font-weight:800}}
.overlay{{position:fixed;inset:0;z-index:100;background:#020617b8;display:grid;place-items:center;padding:1rem}}.overlay-card{{width:min(32rem,100%);max-height:90vh;overflow:auto;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:0 30px 90px #0008}}.drawer{{place-items:stretch end;padding:0}}.drawer .overlay-card{{height:100%;max-height:none;border-radius:0;width:min(28rem,100%)}}.overlay-card>header,.overlay-card>footer{{display:flex;align-items:center;justify-content:space-between;gap:.6rem;padding:1rem;border-bottom:1px solid var(--line)}}.overlay-card>footer{{border:0;border-top:1px solid var(--line);justify-content:flex-end}}.overlay-body{{padding:1rem}}.danger{{background:#b91c1c;color:#fff}}.site-footer{{padding:1.3rem 0;border-top:1px solid var(--line);color:var(--muted)}}
@media(max-width:{middle}px){{.site-header{{align-items:start;flex-direction:column}}.page-shell,.hero{{grid-template-columns:1fr}}.sidebar{{position:static}}.booking form{{grid-template-columns:1fr}}body[data-responsive~="table>cards"] .table-scroll{{display:none}}body[data-responsive~="table>cards"] .mobile-list{{display:grid}}body[data-responsive~="drawer>full"] .drawer .overlay-card{{width:100%}}body[data-responsive~="grid>stack"] .product-grid,body[data-responsive~="grid>stack"] .metric-grid{{grid-template-columns:1fr}}body[data-responsive~="chart>scroll"] .chart{{overflow-x:auto}}body[data-responsive~="chart>scroll"] .chart svg{{min-width:36rem}}body[data-responsive~="code>scroll"] pre{{overflow-x:auto;max-width:100%}}}}
@media(max-width:{small}px){{.site-header,.site-footer,.page-shell{{width:min(100% - 1rem,{large}px)}}.panel{{padding:.8rem}}.control-row>*{{width:100%}}.mobile-card{{grid-template-columns:1fr}}.product-grid{{grid-template-columns:1fr}}}}
@media(min-width:{large}px){{.page-shell{{gap:calc(var(--gap)*1.2)}}.panel{{padding:1.5rem}}}}
@media(prefers-reduced-motion:reduce){{*,*::before,*::after{{animation-duration:.01ms!important;transition-duration:.01ms!important;scroll-behavior:auto!important}}}}
""".strip()


RUNTIME_JS = r"""
(() => {
  'use strict';
  const actions = new Set(document.body.dataset.actions.split(','));
  const all = (selector, root = document) => [...root.querySelectorAll(selector)];
  const live = document.querySelector('#genome-live');
  let restoreFocus = null;
  const say = text => { live.textContent = ''; requestAnimationFrame(() => { live.textContent = text; }); };

  function applyFilters(root) {
    const terms = all('[data-filter-control]', root).filter(x => x.selectedIndex > 0).map(x => x.selectedOptions[0].textContent.toLowerCase());
    let visible = 0;
    all('[data-filter-item]').forEach(item => {
      const show = terms.every(term => (item.dataset.search || item.textContent.toLowerCase()).includes(term));
      item.hidden = !show;
      if (show && !item.closest('.mobile-list')) visible += 1;
    });
    say(`筛选完成，显示 ${visible} 项`);
  }
  all('[data-component=filters]').forEach(root => {
    root.addEventListener('change', () => actions.has('filter') && applyFilters(root));
    root.querySelector('[data-reset]').addEventListener('click', () => { if (!actions.has('reset')) return; all('select', root).forEach(x => x.selectedIndex = 0); applyFilters(root); });
  });
  all('[data-sort]').forEach(button => button.addEventListener('click', () => {
    if (!actions.has('sort')) return;
    const table = button.closest('table'); const body = table.tBodies[0]; const index = Number(button.dataset.sort);
    const direction = button.dataset.direction === 'up' ? -1 : 1; button.dataset.direction = direction === 1 ? 'up' : 'down';
    [...body.rows].sort((a,b) => a.cells[index].textContent.localeCompare(b.cells[index].textContent, 'zh-CN', {numeric:true}) * direction).forEach(row => body.append(row));
    say(`已按 ${button.textContent.trim()} 排序`);
  }));
  function openOverlay(id) {
    const overlay = document.getElementById(id); if (!overlay) return;
    restoreFocus = document.activeElement; overlay.hidden = false; document.body.style.overflow = 'hidden';
    (overlay.querySelector('[data-close]') || overlay).focus(); say(`${overlay.querySelector('h2').textContent}已打开`);
  }
  function closeOverlay(overlay) {
    overlay.hidden = true; if (!document.querySelector('.overlay:not([hidden])')) document.body.style.overflow = '';
    if (restoreFocus) restoreFocus.focus(); say('对话已关闭');
  }
  document.addEventListener('click', event => {
    const opener = event.target.closest('[data-open]'); if (opener && actions.has('open')) openOverlay(opener.dataset.open);
    const closer = event.target.closest('[data-close]'); if (closer) closeOverlay(closer.closest('.overlay'));
    if (event.target.matches('[data-confirm]') && actions.has('confirm')) { say('操作已确认'); closeOverlay(event.target.closest('.overlay')); }
    if (event.target.matches('[data-add-bag]') && actions.has('count')) { const out = document.querySelector('[data-bag-count]'); out.value = Number(out.value || out.textContent) + 1; out.textContent = out.value; say(`购物袋现有 ${out.value} 件`); }
  });
  document.addEventListener('keydown', event => {
    const overlay = document.querySelector('.overlay:not([hidden])');
    if (event.key === 'Escape' && overlay) closeOverlay(overlay);
    if (event.key === 'Tab' && overlay) { const focusable = all('button,input,select,a[href],[tabindex]:not([tabindex="-1"])', overlay).filter(x => !x.disabled); if (!focusable.length) return; const first=focusable[0], last=focusable.at(-1); if (event.shiftKey && document.activeElement===first) {event.preventDefault();last.focus();} else if (!event.shiftKey && document.activeElement===last) {event.preventDefault();first.focus();} }
  });
  all('[data-compare]').forEach(input => input.addEventListener('input', () => { if (!actions.has('compare')) return; const section=input.closest('.compare'); section.querySelector('[data-after]').style.clipPath=`inset(0 ${100-input.value}% 0 0)`; section.querySelector('output').value=`${input.value}%`; }));
  all('form[data-validate]').forEach(form => form.addEventListener('submit', event => { event.preventDefault(); if (!actions.has('validate')) return; let ok=true; all('[required]',form).forEach(field => { const invalid=!field.value; field.setAttribute('aria-invalid', String(invalid)); ok &&= !invalid; }); form.querySelector('[data-form-status]').textContent = ok ? '预约成功，已保存。' : '存在验证错误，请检查标记字段。'; say(ok ? '提交成功' : '表单验证失败'); }));
  all('[data-copy]').forEach(button => button.addEventListener('click', async () => { if (!actions.has('copy')) return; const text=document.getElementById(button.dataset.copy).textContent; try { await navigator.clipboard.writeText(text); } catch { /* 静态文件环境可能禁用剪贴板 */ } button.parentElement.querySelector('[data-copy-status]').textContent='代码已复制'; say('代码已复制'); }));
  all('[role=tab]').forEach(tab => tab.addEventListener('click', () => { if (!actions.has('tab')) return; const list=tab.closest('[role=tablist]'); all('[role=tab]',list).forEach(x => {x.setAttribute('aria-selected',String(x===tab)); document.getElementById(x.getAttribute('aria-controls')).hidden=x!==tab;}); tab.focus(); }));
  all('[role=switch]').forEach(toggle => toggle.addEventListener('change', () => { if (!actions.has('toggle')) return; toggle.setAttribute('aria-checked',String(toggle.checked)); const status=toggle.closest('.toggles').querySelector('[data-save-status]'); status.textContent='保存中…'; setTimeout(()=>{status.textContent='已保存';say('设置已保存');},180); }));
  all('[data-favorite]').forEach(button => button.addEventListener('click', () => { if (!actions.has('favorite')) return; const next=button.getAttribute('aria-pressed')!=='true'; button.setAttribute('aria-pressed',String(next)); button.textContent=next?'已收藏':'收藏'; say(button.textContent); }));
  all('[data-component=filters] select').forEach(select => select.addEventListener('change', () => { if (actions.has('year') && /年/.test(select.previousElementSibling?.textContent || '')) all('[data-bar]').forEach((bar,i)=>{const h=70+((i+select.selectedIndex)*29)%100;bar.setAttribute('height',h);bar.setAttribute('y',210-h);}); }));
  all('[data-inspect]').forEach(mark => { const inspect=()=>actions.has('inspect') && say(mark.getAttribute('aria-label')); mark.addEventListener('click',inspect); mark.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();inspect();}}); });
  const responsive = new Set(document.body.dataset.responsive.split(' '));
  if (responsive.has('schedule>accordion') && matchMedia('(max-width: 820px)').matches) all('.schedule-item').forEach((item,index)=>item.open=index===0);
})();
""".strip()


def compile_genome(ir: dict[str, Any], slots: dict[int, str]) -> tuple[str, dict[str, Any]]:
    title_id, lede_id = ir["q"]
    if title_id not in slots or lede_id not in slots:
        raise IRError("Genome 引用了不存在的 title/lede copy slot")
    title, lede = slots[title_id], slots[lede_id]
    catalog = load_catalog()
    indexed = [(f"g{index}-{family}", family, catalog[family][variant]) for index, (family, variant) in enumerate(ir["c"])]
    first_drawer = next((cid for cid, family, _ in indexed if family == "drawer"), None)
    first_dialog = next((cid for cid, family, _ in indexed if family == "dialog"), None)
    fragments: list[str] = []
    for cid, family, spec in indexed:
        if family == "hero": fragments.append(render_hero(cid, spec, title))
        elif family == "filters": fragments.append(render_filters(cid, spec))
        elif family == "table": fragments.append(render_table(cid, spec, first_drawer))
        elif family in {"drawer", "dialog"}: fragments.append(render_overlay(cid, spec, family, first_dialog if family == "drawer" else None))
        elif family == "products": fragments.append(render_products(cid, spec, first_dialog))
        elif family == "timeline": fragments.append(render_timeline(cid, spec))
        elif family == "compare": fragments.append(render_compare(cid, spec))
        elif family == "form": fragments.append(render_form(cid, spec))
        elif family == "sidebar": fragments.append(render_sidebar(cid, spec))
        elif family == "code": fragments.append(render_code(cid, spec))
        elif family == "tabs": fragments.append(render_tabs(cid, spec))
        elif family == "toggles": fragments.append(render_toggles(cid, spec))
        elif family == "schedule": fragments.append(render_schedule(cid, spec))
        elif family == "chart": fragments.append(render_chart(cid, spec))
        elif family == "note": fragments.append(render_note(cid, spec))
        elif family == "metrics": fragments.append(render_metrics(cid, spec))
        elif family == "bag": fragments.append(render_bag(cid, spec))
        else: raise IRError(f"未实现组件 family: {family}")
    brand_svg = '<svg viewBox="0 0 48 48" role="img" aria-labelledby="brand-symbol-title"><title id="brand-symbol-title">页面标识</title><path d="M7 34 24 7l17 27-17 7Z"/><circle cx="24" cy="25" r="6"/></svg>'
    page = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{css_for(ir)}</style></head>
<body data-layout="{esc(ir['l'][0])}" data-mobile="{esc(ir['l'][1])}" data-actions="{esc(','.join(ir['x']))}" data-responsive="{esc(' '.join(ir['r']))}">
<a class="skip" href="#main">跳到主要内容</a>
<header class="site-header"><div class="brand">{brand_svg}<div><h1>{esc(title)}</h1><p class="lede">{esc(lede)}</p></div></div><span aria-label="资产策略">离线自包含</span></header>
<main class="page-shell" id="main">{''.join(fragments)}</main>
<p id="genome-live" class="skip" role="status" aria-live="polite"></p>
<footer class="site-footer">确定性 Design Genome 编译结果 · 键盘可操作 · 支持减少运动</footer>
<script>{RUNTIME_JS}</script></body></html>
"""
    genome_bytes = utf8_size(ir)
    html_bytes = len(page.encode("utf-8"))
    report = {
        "schema_version": "design-genome-compile-report-v1",
        "compiler": "parallel-design-genome-compiler-v1",
        "genome_bytes": genome_bytes,
        "copy_slot_bytes": sum(len(value.encode("utf-8")) for value in slots.values()),
        "html_bytes": html_bytes,
        "expansion_ratio_html_per_genome_byte": round(html_bytes / genome_bytes, 3),
        "component_genes": len(ir["c"]),
        "actions": ir["x"],
        "responsive_transforms": ir["r"],
        "compiler_guarantees": ["document_closure", "semantic_landmarks", "focus_visible", "reduced_motion", "live_region", "escape_and_focus_restore", "responsive_breakpoints", "no_external_dependency"],
        "model_owned": ["layout_gene", "component_genes", "interaction_genes", "responsive_transforms", "visual_genes", "copy_slot_references"],
        "copy_owned": ["title", "lede"],
    }
    return page, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("genome", type=Path)
    parser.add_argument("--slots", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ir = load_ir(args.genome, enforce_target_size=True)
        slots = load_slots(args.slots)
        page, report = compile_genome(ir, slots)
        write_utf8_no_bom(args.output, page)
        if args.report:
            write_utf8_no_bom(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (IRError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
