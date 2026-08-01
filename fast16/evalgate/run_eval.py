"""能力门评测器:对本地 8092 服务运行冻结题库并自动判分。

- fast/deep 载荷与 chat_v4.py 保持一致,但默认贪心解码,消除采样噪声;
- 每题独立请求,不共享对话上下文;
- 判分基于「答案:X」末行解析 + 类型化归一化,不做语义猜测;
- compare 模式对两次结果做配对 McNemar 精确检验。
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from fractions import Fraction
from math import comb, sqrt
from pathlib import Path

BASE_URL = "http://127.0.0.1:8092"
SYSTEM_PROMPT = "你是一个精确的解题助手。仔细解题,最后一行必须是「答案:X」的格式。"

_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９ＡＢＣ／：，",
    "0123456789ABC/:,",
)
_UNIT_SUFFIX = re.compile(r"(个|人|岁|种|次|名|元|只|件|天)$")


def post_json(path: str, payload: dict, timeout: int = 600) -> dict:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def ask(
    question: str,
    mode: str,
    thinking_tokens: int,
    temperature: float,
    max_tokens: int = 512,
) -> dict:
    payload = {
        "model": "ColorLM-v4-SMoE",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": question + "\n\n最后一行必须是「答案:X」,X 是最终答案本身,不带单位。",
            },
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": True,
        "id_slot": 0,
    }
    if mode == "deep":
        payload["reasoning_format"] = "deepseek"
        payload["thinking_budget_tokens"] = thinking_tokens
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        payload["max_tokens"] = max(max_tokens, thinking_tokens + 256)
    started = time.perf_counter()
    response = post_json("/v1/chat/completions", payload)
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]
    usage = response.get("usage", {})
    return {
        "content": choice["message"].get("content") or "",
        "reasoning": choice["message"].get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": round(elapsed, 2),
    }


def extract_answer(content: str) -> str | None:
    matches = re.findall(r"答案[::]\s*(.+)", content)
    if not matches:
        return None
    return matches[-1].strip()


def normalize(raw: str, answer_type: str) -> str | None:
    text = raw.translate(_FULLWIDTH)
    text = text.replace("，", ",").replace("、", ",").replace(" ", "")
    text = text.strip("。.;；,**「」*")
    text = _UNIT_SUFFIX.sub("", text)
    if not text:
        return None
    if answer_type == "int":
        match = re.fullmatch(r"-?\d+(?:\.0+)?", text)
        if match:
            return str(int(float(text)))
        return None
    if answer_type == "fraction":
        match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
        if match:
            value = Fraction(int(match.group(1)), int(match.group(2)))
            return f"{value.numerator}/{value.denominator}"
        match = re.fullmatch(r"0?\.\d+", text)
        if match:
            return f"decimal:{float(text):.6f}"
        return None
    if answer_type == "names":
        parts = [p for p in re.split(r"[,和与及]", text) if p]
        if all(p in {"甲", "乙", "丙"} for p in parts) and parts:
            return ",".join(sorted(set(parts)))
        return None
    return text


def grade(expected: str, got: str | None, answer_type: str) -> bool:
    if got is None:
        return False
    normalized = normalize(got, answer_type)
    if normalized is None:
        return False
    if answer_type == "fraction" and normalized.startswith("decimal:"):
        value = float(normalized.split(":", 1)[1])
        target = Fraction(expected)
        return abs(value - float(target)) < 1e-4
    if answer_type == "names":
        return normalized == ",".join(sorted(expected.split(",")))
    return normalized == expected


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    z = 1.96
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def run(args: argparse.Namespace) -> int:
    bank = json.loads(args.items.read_text(encoding="utf-8"))
    items = bank["items"]
    if args.limit:
        items = items[: args.limit]
    results = []
    correct_count = 0
    for index, item in enumerate(items):
        try:
            reply = ask(
                item["question"], args.mode, args.budget, args.temp, args.max_tokens
            )
        except Exception as error:  # noqa: BLE001 - 记录后继续
            reply = {
                "content": "",
                "reasoning": "",
                "finish_reason": f"error:{error}",
                "completion_tokens": None,
                "seconds": None,
            }
        got = extract_answer(reply["content"])
        is_correct = grade(item["answer"], got, item["answer_type"])
        correct_count += is_correct
        results.append(
            {
                "id": item["id"],
                "family": item["family"],
                "expected": item["answer"],
                "got": got,
                "correct": is_correct,
                "finish_reason": reply["finish_reason"],
                "completion_tokens": reply["completion_tokens"],
                "seconds": reply["seconds"],
                "content": reply["content"],
                "reasoning": reply["reasoning"],
            }
        )
        print(
            f"[{index + 1}/{len(items)}] {item['id']} {item['family']}"
            f" expected={item['answer']} got={got}"
            f" {'OK' if is_correct else 'X'} ({reply['seconds']}s)",
            flush=True,
        )

    total = len(results)
    low, high = wilson_interval(correct_count, total)
    families: dict[str, list[bool]] = {}
    for row in results:
        families.setdefault(row["family"], []).append(row["correct"])
    summary = {
        "mode": args.mode,
        "budget": args.budget if args.mode == "deep" else None,
        "temperature": args.temp,
        "total": total,
        "correct": correct_count,
        "accuracy": round(correct_count / total, 4) if total else None,
        "wilson_95": [round(low, 4), round(high, 4)],
        "per_family": {
            family: f"{sum(marks)}/{len(marks)}"
            for family, marks in sorted(families.items())
        },
        "seconds_total": round(
            sum(r["seconds"] for r in results if r["seconds"]), 1
        ),
    }
    payload = {"summary": summary, "results": results}
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def mcnemar_exact(b: int, c: int) -> float:
    """b=仅A对, c=仅B对;双侧精确检验。"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def compare(args: argparse.Namespace) -> int:
    run_a = json.loads(args.run_a.read_text(encoding="utf-8"))
    run_b = json.loads(args.run_b.read_text(encoding="utf-8"))
    rows_a = {r["id"]: r for r in run_a["results"]}
    rows_b = {r["id"]: r for r in run_b["results"]}
    shared = sorted(set(rows_a) & set(rows_b))
    only_a = [i for i in shared if rows_a[i]["correct"] and not rows_b[i]["correct"]]
    only_b = [i for i in shared if rows_b[i]["correct"] and not rows_a[i]["correct"]]
    p_value = mcnemar_exact(len(only_a), len(only_b))
    report = {
        "run_a": {"file": str(args.run_a), **run_a["summary"]},
        "run_b": {"file": str(args.run_b), **run_b["summary"]},
        "paired_items": len(shared),
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "mcnemar_exact_p": round(p_value, 4),
        "conclusion": (
            "显著差异" if p_value < 0.05 else "在该样本量下未达显著,不得据此宣称能力变化"
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="能力门评测")
    sub = parser.add_subparsers(dest="command", required=True)

    runner = sub.add_parser("run")
    runner.add_argument("--mode", choices=["fast", "deep"], required=True)
    runner.add_argument("--budget", type=int, default=64)
    runner.add_argument(
        "--max-tokens",
        type=int,
        default=1536,
        help="生成上限; 512 会在模型写出「答案:」前截断, 实测需 >=1536",
    )
    runner.add_argument("--temp", type=float, default=0.0)
    runner.add_argument("--limit", type=int, default=0)
    runner.add_argument(
        "--items",
        type=Path,
        default=Path(__file__).resolve().parent / "frozen_v1.json",
    )
    runner.add_argument("--out", type=Path, required=True)
    runner.set_defaults(func=run)

    comparer = sub.add_parser("compare")
    comparer.add_argument("run_a", type=Path)
    comparer.add_argument("run_b", type=Path)
    comparer.set_defaults(func=compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
