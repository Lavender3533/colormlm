"""把 v38 已生成的完整主体确定性收尾；这是结构化编译原型，不冒充纯模型输出。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_frontend_ir_ab import extract_html


TAIL_MARKER = "\n    function openDrawer() {"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("拒绝覆盖确定性修复产物")
    source = args.source.read_text(encoding="utf-8")
    marker = source.rfind(TAIL_MARKER)
    if marker < 0:
        raise ValueError("找不到已知的重复截断函数边界，拒绝猜测修复")
    prefix = source[:marker]
    replacements = {
        "drawer.style.display = 'block';": "drawer.classList.add('open');",
        "document.getElementById('detailDrawer').style.display = 'none';": "document.getElementById('detailDrawer').classList.remove('open');",
        "document.getElementById('alertModal').style.display = 'flex';": "document.getElementById('alertModal').classList.add('active');",
        "document.getElementById('alertModal').style.display = 'none';": "document.getElementById('alertModal').classList.remove('active');",
        "card.className = 'card';": "card.className = 'node-card';",
        "document.getElementById('regionFilter').value = 'all';": "document.getElementById('regionFilter').value = 'all';\n        document.getElementById('statusFilter').value = 'all';",
    }
    counts = {}
    for old, new in replacements.items():
        counts[old] = prefix.count(old)
        if counts[old] != 1:
            raise ValueError(f"确定性替换边界变化: {old!r} count={counts[old]}")
        prefix = prefix.replace(old, new, 1)
    focus_css = "\n:focus-visible{outline:2px solid var(--accent);outline-offset:2px}th[tabindex]{cursor:pointer}\n"
    if "</style>" not in prefix:
        raise ValueError("缺少 </style>")
    prefix = prefix.replace("</style>", focus_css + "</style>", 1)
    suffix = r'''

    function applyFilters() {
        const status = document.getElementById('statusFilter').value;
        const region = document.getElementById('regionFilter').value;
        const data = generateMockData().filter(n =>
            (status === 'all' || ({'运行中':'running','警告':'warning','离线':'offline'}[n.status] === status)) &&
            (region === 'all' || ({'华东':'cn-shanghai','华北':'cn-beijing','美西':'us-west'}[n.region] === region))
        );
        renderTable(data);
        document.getElementById('announce').textContent = `已显示 ${data.length} 个节点`;
    }

    document.getElementById('statusFilter').addEventListener('change', applyFilters);
    document.getElementById('regionFilter').addEventListener('change', applyFilters);
    document.querySelectorAll('th[tabindex]').forEach(th => th.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); th.click(); }
    }));
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') { closeAlertModal(); closeDrawer(); }
    });
    document.querySelector('.app-container').insertAdjacentHTML('afterend','<div id="announce" role="status" aria-live="polite" class="sr-only"></div>');
    refreshData();
</script>
</body>
</html>
'''
    compiled = prefix.rstrip() + suffix
    html = extract_html(compiled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    report = {
        "format": "colorlm-v47-frontend-deterministic-tail-compile-v1",
        "status": "development_hybrid_not_pure_model_output",
        "source": str(args.source.resolve()),
        "source_bytes": len(source.encode("utf-8")),
        "output_bytes": len(html.encode("utf-8")),
        "truncated_duplicate_function_removed": True,
        "deterministic_repairs": [
            "drawer_and_modal_class_state",
            "mobile_card_class",
            "status_and_region_filters",
            "keyboard_sort_and_escape",
            "focus_visible_and_live_status",
            "document_closure"
        ],
        "pure_model_claim_allowed": False
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

