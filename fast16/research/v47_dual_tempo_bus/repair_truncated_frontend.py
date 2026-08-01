"""只续写被 max_tokens 截断的前端尾部，避免重算已生成的数千 token。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from run_frontend_ir_ab import chat, extract_html


def strip_fence(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:html)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8138")
    parser.add_argument("--model", default="ColorLM-v38-Qwen36-Shared-Sequence-Policy")
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1400)
    args = parser.parse_args()

    if args.output.exists() or args.report.exists():
        raise FileExistsError("拒绝覆盖续写产物")
    partial = args.partial.read_text(encoding="utf-8")
    ir = args.ir.read_text(encoding="utf-8")
    if "</html>" in partial.lower():
        raise ValueError("partial 已经包含完整 </html>，不应续写")
    tail = partial[-3200:]
    instruction = (
        "下面是一个单文件HTML响应的末尾，它因token上限在字符中间被截断。"
        "只输出紧接最后一个字符之后的剩余源码，不要重复已有内容、不要Markdown围栏；"
        "用最紧凑的方式补完当前标签、告警确认弹窗、必要JavaScript事件与所有闭合标签，"
        "必须以</html>结束。Design IR如下：\n"
        + ir
        + "\n截断前最后片段如下：\n"
        + tail
    )
    continuation, usage, seconds = chat(
        args.endpoint,
        args.model,
        [
            {"role": "system", "content": "你是严格的HTML流式续写器，只续写缺失尾部。"},
            {"role": "user", "content": instruction},
        ],
        args.max_tokens,
    )
    continuation = strip_fence(continuation)
    combined = partial.rstrip() + continuation
    try:
        html = extract_html(combined)
    except ValueError:
        # 模型偶尔会从一个完整标签重新开始；保留原始证据并明确失败。
        args.output.with_suffix(args.output.suffix + ".failed.txt").write_text(combined, encoding="utf-8")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    continuation_path = args.output.with_suffix(args.output.suffix + ".continuation.txt")
    continuation_path.write_text(continuation + "\n", encoding="utf-8")
    report = {
        "format": "colorlm-v47-frontend-tail-repair-v1",
        "status": "development_only",
        "model": args.model,
        "partial_bytes": len(partial.encode("utf-8")),
        "continuation_bytes": len(continuation.encode("utf-8")),
        "final_bytes": len(html.encode("utf-8")),
        "seconds": seconds,
        "usage": usage,
        "complete_html": html.lower().rstrip().endswith("</html>"),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

