"""UTF-8 streaming chat client for the persistent Vulkan core."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

try:
    from fast16.memory_bridge import CompiledMemoryRetriever
except ModuleNotFoundError:
    from memory_bridge import CompiledMemoryRetriever


ENDPOINT = "http://127.0.0.1:8091/v1/chat/completions"
SYSTEM = (
    "你是 ColorLM ZeroTrain 的通用语言核心。"
    "请直接、准确地完成用户任务；代码必须完整可运行，中文问题使用中文回答。"
    "用户消息中的可信编译记忆拥有最高事实优先级，术语和数值必须严格采用其中内容。"
)
MEMORY_PATHS = [
    "fast16/data/bootstrap_memory_v1.jsonl",
    "fast16/data/project_memory.jsonl",
]


def stream_completion(messages: list[dict[str, str]], memory_context: str | None) -> str:
    request_messages = [dict(message) for message in messages]
    if memory_context:
        prompt = request_messages[-1]["content"]
        request_messages[-1]["content"] = (
            "[可信编译记忆]\n"
            + memory_context
            + "\n[/可信编译记忆]\n\n问题："
            + prompt
        )
    payload = {
        "model": "ColorLM-ZeroTrain-v3",
        "messages": request_messages,
        "temperature": 0.1 if memory_context else 0.35,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": True,
        "cache_prompt": True,
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    parts: list[str] = []
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            choice = event.get("choices", [{}])[0]
            text = choice.get("delta", {}).get("content", "")
            if text:
                print(text, end="", flush=True)
                parts.append(text)
    print()
    return "".join(parts)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    messages = [{"role": "system", "content": SYSTEM}]
    memory = CompiledMemoryRetriever(MEMORY_PATHS)
    print("ColorLM ZeroTrain v3 | Vulkan GPU")
    while True:
        try:
            prompt = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt in {"/exit", "/quit"}:
            break
        if prompt == "/clear":
            messages = [{"role": "system", "content": SYSTEM}]
            print("上下文已清空。")
            continue
        if not prompt:
            continue
        messages.append({"role": "user", "content": prompt})
        hit = memory.search(prompt)
        memory_context = hit.value if hit is not None and hit.score >= 4.0 else None
        print("CLM> ", end="", flush=True)
        try:
            answer = stream_completion(messages, memory_context)
        except (urllib.error.URLError, TimeoutError) as error:
            messages.pop()
            print(f"连接错误：{error}")
            continue
        messages.append({"role": "assistant", "content": answer})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
