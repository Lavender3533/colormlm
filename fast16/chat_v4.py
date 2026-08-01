"""UTF-8 streaming client for ColorLM v4 without text retrieval injection."""

from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.error
import urllib.request

try:
    from fast16.plasticity_v4 import DEFAULT_WEIGHTS, learn, reset_weights
except ModuleNotFoundError:
    from plasticity_v4 import DEFAULT_WEIGHTS, learn, reset_weights


ENDPOINT = "http://127.0.0.1:8092/v1/chat/completions"
SLOT_ENDPOINT = "http://127.0.0.1:8092/slots/0"


def system_prompt() -> str:
    today = dt.datetime.now().astimezone().date().isoformat()
    return (
        "你是 ColorLM v4-SMoE，本地运行的通用语言模型。"
        "当前版本正在执行 ColorLM 状态迁移，包含 35B 总参数、约 3B 动态激活参数，"
        "并使用图内共享权重递归精炼。"
        "你可以处理中文、推理、编程和一般知识，不局限于代码。"
        "请直接、准确地完成用户任务；不确定时明确指出具体不确定点。"
        f"运行时当前日期是 {today}。"
    )


def chat_system_prompt() -> str:
    """聊天专用人设;学习/巩固/评测继续用 system_prompt() 保持对照一致。"""
    today = dt.datetime.now().astimezone().date().isoformat()
    return (
        "你是 ColorLM，一个运行在用户自己电脑上的开源 AI 助手。"
        "你说话自然、直接、有温度：先给答案或结论，再给必要的解释；"
        "不堆免责声明，不逐句复读用户的话，不用公告腔。"
        "你擅长中文对话、编程和推理；写代码时给完整可运行的代码。"
        "遇到不确定的地方，直说哪里不确定。"
        f"今天是 {today}。"
    )


def stream_completion(
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
    deep: bool = False,
    thinking_tokens: int = 64,
    temperature: float = 0.7,
) -> str:
    # 聊天默认带采样(贪心聊天=机器人腔)。评测/巩固脚本各自锁 temp 0,
    # 不受此处影响;要可复现对照时用 /greedy 切回贪心。
    payload = {
        "model": "ColorLM-v4-SMoE",
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stream": True,
        "cache_prompt": True,
        "id_slot": 0,
    }
    if deep:
        payload["reasoning_format"] = "deepseek"
        payload["thinking_budget_tokens"] = thinking_tokens
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        payload["max_tokens"] = max(max_tokens, 320)
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    parts: list[str] = []
    started_answer = False
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            delta = event.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content", "")
            if text:
                if not started_answer:
                    print("CLM> ", end="", flush=True)
                    started_answer = True
                print(text, end="", flush=True)
                parts.append(text)
    if not started_answer:
        print("CLM> ", end="")
    print()
    return "".join(parts)


def slot_action(action: str, **values: float) -> dict[str, object]:
    request = urllib.request.Request(
        f"{SLOT_ENDPOINT}?action={action}",
        data=json.dumps(values).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    messages: list[dict[str, str]] = [{"role": "system", "content": chat_system_prompt()}]
    mode = "fast"
    greedy = False
    print("ColorLM v4-SMoE | 35B-A3B | Vulkan GPU")
    print("默认快速稳定模式；/deep 开启内部推演，/fast 恢复秒答，/greedy 切换贪心解码。")
    while True:
        try:
            prompt = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt in {"/exit", "/quit"}:
            break
        if prompt == "/clear":
            slot_action("erase")
            messages = [{"role": "system", "content": chat_system_prompt()}]
            print("上下文已清空。")
            continue
        if prompt == "/deep":
            mode = "deep"
            print("已切换到单轮深入模式。")
            continue
        if prompt == "/fast":
            mode = "fast"
            print("已切换到快速模式。")
            continue
        if prompt == "/greedy":
            greedy = not greedy
            print(f"贪心解码已{'开启(可复现,机器人腔)' if greedy else '关闭(自然采样)'}。")
            continue
        if prompt == "/learn":
            question = input("要学习的问题> ").strip()
            answer = input("确认的正确答案> ").strip()
            if not question or not answer:
                print("问题和答案都不能为空。")
                continue
            try:
                result = learn(
                    DEFAULT_WEIGHTS,
                    question,
                    answer,
                    margin=18.0,
                    learning_rate=0.8,
                    epochs=3,
                    system=system_prompt(),
                )
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                print(f"学习失败：{error}")
                continue
            messages = [{"role": "system", "content": chat_system_prompt()}]
            print(
                f"已写入连续神经参数：{result['trained_tokens']} 个预测目标，"
                f"耗时 {result['elapsed_seconds']} 秒。"
            )
            continue
        if prompt == "/learning-reset":
            confirm = input("这会清空全部快速学习参数，输入 YES 确认> ").strip()
            if confirm != "YES":
                print("已取消。")
                continue
            try:
                reset_weights(DEFAULT_WEIGHTS)
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                print(f"清空失败：{error}")
                continue
            messages = [{"role": "system", "content": chat_system_prompt()}]
            print("快速学习参数已清空。")
            continue
        if not prompt:
            continue
        messages.append({"role": "user", "content": prompt})
        try:
            answer = stream_completion(
                messages,
                deep=(mode == "deep"),
                temperature=0.0 if greedy else 0.7,
            )
        except (urllib.error.URLError, TimeoutError) as error:
            messages.pop()
            print(f"连接错误：{error}")
            continue
        messages.append({"role": "assistant", "content": answer})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
