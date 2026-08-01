"""对同一v17实例运行冻结的n-gram推测解码短门。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "ngram_gate_v1.json"
EXPECTED_CONTRACT_SHA256 = "e42a27c855825187cfb31b15b71d10f8ac10b61f00c9979f5992bb5f8f81fdaa"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("服务响应不是JSON对象")
    return value


def canonical_message(message: dict[str, Any]) -> bytes:
    selected = {
        key: message.get(key)
        for key in ("role", "content", "tool_calls", "function_call")
        if key in message
    }
    # llama-server生成的tool call id不是模型token，跨进程会随机变化；
    # 等价比较只保留模型实际生成的函数名、参数和调用类型。
    if isinstance(selected.get("tool_calls"), list):
        selected["tool_calls"] = [
            {
                key: call.get(key)
                for key in ("type", "function")
                if key in call
            }
            if isinstance(call, dict)
            else call
            for call in selected["tool_calls"]
        ]
    return json.dumps(
        selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="运行v24 n-gram速度/等价短门")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8110/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("none", "ngram-mod"), required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    contract_hash = sha256_file(args.contract)
    if contract_hash != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError(
            f"冻结契约哈希漂移: actual={contract_hash}, "
            f"expected={EXPECTED_CONTRACT_SHA256}"
        )
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    tasks = contract.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 4:
        raise RuntimeError("冻结短门必须恰好包含4题")

    # 两种模式都执行同一个不计分热身，减少首次图构建对短门的污染。
    post_json(
        args.endpoint.rstrip("/") + "/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "temperature": 0,
            "seed": 18,
            "max_tokens": 8,
        },
        args.timeout,
    )

    rows: list[dict[str, Any]] = []
    for task in tasks:
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": task["system"]},
                {"role": "user", "content": task["user"]},
            ],
            "temperature": contract["temperature"],
            "seed": contract["seed"],
            "max_tokens": task["max_tokens"],
        }
        if "tools" in task:
            payload["tools"] = task["tools"]
            payload["tool_choice"] = task.get("tool_choice", "auto")
        started = time.monotonic()
        response = post_json(
            args.endpoint.rstrip("/") + "/chat/completions", payload, args.timeout
        )
        elapsed = time.monotonic() - started
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError(f"{task['id']}没有唯一choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"{task['id']}缺少message")
        message_bytes = canonical_message(message)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        timings = (
            response.get("timings") if isinstance(response.get("timings"), dict) else {}
        )
        completion_tokens = int(usage.get("completion_tokens", 0))
        rows.append(
            {
                "id": task["id"],
                "elapsed_seconds": elapsed,
                "completion_tokens": completion_tokens,
                "client_tokens_per_second": completion_tokens / elapsed if elapsed else None,
                "finish_reason": choice.get("finish_reason"),
                "message_sha256": sha256_bytes(message_bytes),
                "message": json.loads(message_bytes.decode("utf-8")),
                "usage": usage,
                "timings": timings,
            }
        )
        print(
            f"{task['id']}: {completion_tokens} tokens, {elapsed:.3f}s, "
            f"{completion_tokens / elapsed if elapsed else 0.0:.2f} token/s",
            flush=True,
        )

    total_tokens = sum(row["completion_tokens"] for row in rows)
    total_elapsed = sum(row["elapsed_seconds"] for row in rows)
    report = {
        "format": "colorlm-v24-ngram-gate-report-v1",
        "contract": str(args.contract.resolve()),
        "contract_sha256": contract_hash,
        "mode": args.mode,
        "endpoint": args.endpoint,
        "model": args.model,
        "tasks": rows,
        "summary": {
            "completion_tokens": total_tokens,
            "elapsed_seconds": total_elapsed,
            "client_tokens_per_second": total_tokens / total_elapsed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
