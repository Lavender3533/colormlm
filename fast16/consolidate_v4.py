"""ColorLM v4 已验证答案巩固回路。

实现路线图第 3 步:只允许已验证轨迹进入巩固队列。

    难题 -> /deep 内部推演求解
         -> 写入门:判分核对外部验证答案,且 finish_reason 必须为 stop
         -> 模型自己生成的最终作答文本用 next-token 预测误差写入 ColorPlasticity
         -> /fast 模式此后直接继承这道题的正确解

注意:reasoning_format=deepseek 下思考流在 reasoning_content 里,被巩固的
content 主要是最终作答行,因此本回路当前证明的是「已验证答案巩固」;完整
推理轨迹巩固需要先解除服务 n_ubatch=512 的 /embeddings 上限再扩展。

写入的是连续低秩 logit 参数(与 /learn 同一条已验证机制),不是检索、
答案表或提示词注入;推理路径不变。练习题与冻结题库无参数重合。
全部请求贪心解码,单次运行的判定不受采样噪声支配。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

try:
    from fast16.chat_v4 import system_prompt
    from fast16.evalgate.run_eval import extract_answer, grade
    from fast16.plasticity_v4 import (
        DEFAULT_WEIGHTS,
        active_token_rows,
        learn,
        open_weights,
        render_question,
        tokenize,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fast16.chat_v4 import system_prompt
    from fast16.evalgate.run_eval import extract_answer, grade
    from fast16.plasticity_v4 import (
        DEFAULT_WEIGHTS,
        active_token_rows,
        learn,
        open_weights,
        render_question,
        tokenize,
    )

ENDPOINT = "http://127.0.0.1:8092/v1/chat/completions"
SUFFIX = "\n\n最后一行必须是「答案:X」,X 是最终答案本身,不带单位。"
RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
# 服务当前 n_ubatch=512,pooling=none 的 /embeddings 不可分批,超限必 500。
EMBEDDING_TOKEN_LIMIT = 500

# 练习题由解反向构造,参数已核对不在 evalgate/frozen_v1.json 冻结集内。
PRESETS = {
    "p11": (
        "今年,父亲的年龄是儿子的 3 倍;11 年后,父亲的年龄将是儿子的 2 倍。儿子今年多少岁?",
        "11",
    ),
    "p8": (
        "今年,父亲的年龄是儿子的 4 倍;4 年后,父亲的年龄将是儿子的 3 倍。儿子今年多少岁?",
        "8",
    ),
    "p6": (
        "今年,父亲的年龄是儿子的 5 倍;18 年后,父亲的年龄将是儿子的 2 倍。儿子今年多少岁?",
        "6",
    ),
    "r38": (
        "一个正整数除以 7 余 3,除以 11 余 5。满足条件的最小正整数是多少?",
        "38",
    ),
    "r93": (
        "一个正整数除以 8 余 5,除以 13 余 2。满足条件的最小正整数是多少?",
        "93",
    ),
}
# 泛化探针取同家族的另一道题,跨家族没有解释力。
FAMILY_ALT = {"p11": "p8", "p8": "p11", "p6": "p11", "r38": "r93", "r93": "r38"}
POLLUTION_PROBE = ("9+8等于几?", "17")


def ask(question: str, mode: str, budget: int = 64) -> dict:
    payload = {
        "model": "ColorLM-v4-SMoE",
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": question + SUFFIX},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
        "cache_prompt": True,
        "id_slot": 0,
    }
    if mode == "deep":
        payload["reasoning_format"] = "deepseek"
        payload["thinking_budget_tokens"] = budget
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        payload["max_tokens"] = max(1024, budget + 256)
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.load(response)
    message = body["choices"][0]["message"]
    return {
        "content": (message.get("content") or "").strip(),
        "reasoning": (message.get("reasoning_content") or "").strip(),
        "finish_reason": body["choices"][0].get("finish_reason"),
        "seconds": round(time.perf_counter() - started, 2),
    }


def check(reply: dict, expected: str) -> tuple[str | None, bool]:
    got = extract_answer(reply["content"])
    return got, grade(expected, got, "int")


def plastic_rows() -> int:
    _a, b, _n_embd, _n_vocab, _rank = open_weights(DEFAULT_WEIGHTS)
    count = int(active_token_rows(b).size)
    del b
    return count


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="已验证答案巩固回路")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="p11")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 fast 基线已答对也继续巩固(仅用于调试)",
    )
    args = parser.parse_args()

    question, expected = PRESETS[args.preset]
    alt = FAMILY_ALT[args.preset]
    probes = [
        ("pollution", *POLLUTION_PROBE),
        ("generalization", *PRESETS[alt]),
    ]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report_path = RUNTIME_DIR / f"consolidation-report-{args.preset}-{stamp}.json"
    report: dict = {
        "preset": args.preset,
        "question": question,
        "verified_answer": expected,
        "budget": args.budget,
        "decoding": "greedy(temperature=0)",
        "stages": {},
    }

    def save() -> None:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    code = 10
    try:
        # 阶段 1: 写入前基线 —— 目标题 + 全部探针。探针没有写入前读数,
        # 写入后的对错就无法归因,所以这三条请求不能省。
        print(f"[1/5] 写入前基线: {question}", flush=True)
        baseline = ask(question, "fast")
        base_got, base_ok = check(baseline, expected)
        report["stages"]["fast_baseline"] = {
            **baseline,
            "got": base_got,
            "correct": base_ok,
        }
        print(f"      目标题 -> {base_got} ({'正确' if base_ok else '错误'}, {baseline['seconds']}s)")
        probe_pre: dict[str, dict] = {}
        for name, probe_q, probe_a in probes:
            reply = ask(probe_q, "fast")
            got, ok = check(reply, probe_a)
            probe_pre[name] = {"question": probe_q, "expected": probe_a, "got": got, "correct": ok}
            print(f"      {name} 基线 -> {got} ({'正确' if ok else '错误'})")
        report["stages"]["probes_before"] = probe_pre
        save()
        if base_ok and not args.force:
            report["conclusion"] = "fast 基线已答对,换更难的 preset 再巩固"
            code = 3
            return code

        # 阶段 2: deep 推演求解。
        print(f"[2/5] deep 推演 (budget={args.budget})", flush=True)
        deep = ask(question, "deep", args.budget)
        deep_got, deep_ok = check(deep, expected)
        report["stages"]["deep_solve"] = {**deep, "got": deep_got, "correct": deep_ok}
        save()
        print(f"      -> {deep_got} ({'正确' if deep_ok else '错误'}, finish={deep['finish_reason']}, {deep['seconds']}s)")

        # 写入门:答案必须与外部验证一致,且生成必须自然收束。
        if not deep_ok or deep["finish_reason"] != "stop":
            report["conclusion"] = (
                "写入门拒绝:deep 作答未通过验证或被截断,未写入任何参数"
            )
            code = 2
            return code

        trajectory = deep["content"].strip()
        # /embeddings 受 n_ubatch=512 硬上限,巩固前先数全序列长度,超限
        # 时保留 deep 结果落报告,不让 learn() 半路 500。
        prompt_text = render_question(question + SUFFIX, system=system_prompt())
        full_tokens = len(tokenize(prompt_text + trajectory + "<|im_end|>"))
        report["stages"]["length_check"] = {
            "full_tokens": full_tokens,
            "limit": EMBEDDING_TOKEN_LIMIT,
        }
        if full_tokens > EMBEDDING_TOKEN_LIMIT:
            report["conclusion"] = (
                f"巩固序列 {full_tokens} token 超过服务 n_ubatch 上限,"
                "未写入;需要 --ubatch-size 2048 重启服务后重试"
            )
            code = 6
            return code

        rows_before = plastic_rows()
        print(
            f"[3/5] 巩固已验证作答 ({len(trajectory)} 字, {full_tokens} token, "
            f"B 活跃行 {rows_before})",
            flush=True,
        )
        try:
            outcome = learn(
                DEFAULT_WEIGHTS,
                question + SUFFIX,
                trajectory,
                margin=18.0,
                learning_rate=0.8,
                epochs=3,
                system=system_prompt(),
            )
        except Exception as error:
            report["stages"]["consolidate"] = {
                "error": repr(error),
                "warning": "learn() 中途失败,磁盘塑性文件可能已被修改;"
                "需手动 plastic_reload+erase 或 --learning-reset 恢复一致",
            }
            raise
        rows_after = plastic_rows()
        report["stages"]["consolidate"] = {
            "trajectory_chars": len(trajectory),
            "trained_tokens": outcome["trained_tokens"],
            "elapsed_seconds": outcome["elapsed_seconds"],
            "rows_before": rows_before,
            "rows_after": rows_after,
        }
        save()
        print(
            f"      -> {outcome['trained_tokens']} 个预测目标, "
            f"{outcome['elapsed_seconds']}s, B 活跃行 {rows_before} -> {rows_after}"
        )

        # 阶段 4: fast 复答同一道题。
        print("[4/5] fast 复答", flush=True)
        recall = ask(question, "fast")
        recall_got, recall_ok = check(recall, expected)
        report["stages"]["fast_recall"] = {
            **recall,
            "got": recall_got,
            "correct": recall_ok,
        }
        save()
        print(f"      -> {recall_got} ({'正确' if recall_ok else '错误'}, {recall['seconds']}s)")

        # 阶段 5: 写入后探针,与写入前基线逐条对比。
        print("[5/5] 写入后探针", flush=True)
        probe_post = {}
        for name, probe_q, probe_a in probes:
            try:
                reply = ask(probe_q, "fast")
                got, ok = check(reply, probe_a)
            except Exception as error:  # noqa: BLE001 - 单探针失败不弃整份报告
                got, ok = None, False
                probe_post[name] = {"error": repr(error)}
                print(f"      {name}: 请求失败 {error}")
                continue
            before = probe_pre[name]
            probe_post[name] = {
                "got": got,
                "correct": ok,
                "changed": ok != before["correct"],
            }
            print(
                f"      {name}: {before['got']}->{got} "
                f"({'正确' if ok else '错误'}{', 状态改变' if ok != before['correct'] else ''})"
            )
        report["stages"]["probes_after"] = probe_post

        answer_only = len(trajectory) <= 30
        label = "已验证答案巩固" if answer_only else "已验证作答巩固"
        if recall_ok:
            report["conclusion"] = (
                f"{label}成功:deep 自解并通过写入门后,fast 模式继承正确答案"
                + ("(注:被巩固内容仅为答案行,非完整轨迹)" if answer_only else "")
            )
            code = 0
        else:
            report["conclusion"] = f"{label}失败:写入完成但 fast 复答仍错误,如实记录"
            code = 4
        return code
    except Exception as error:  # noqa: BLE001 - 唯一一次真实运行,证据必须落盘
        report["error"] = {
            "repr": repr(error),
            "traceback": traceback.format_exc(),
        }
        code = 10
        return code
    finally:
        save()
        print(f"\n报告 -> {report_path} (exit {code})")


if __name__ == "__main__":
    raise SystemExit(main())
