"""Run the frozen v28 post-tool-result state gate through an OpenAI endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def endpoint_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return endpoint + ("/chat/completions" if endpoint.endswith("/v1") else "/v1/chat/completions")


def main() -> int:
    parser = argparse.ArgumentParser(description="运行v28工具结果状态冻结门")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    encoded = args.contract.read_bytes()
    contract = json.loads(encoded.decode("utf-8"))
    catalog = contract["tool_catalog"]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(contract["items"], 1):
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "system",
                    "content": "继续完成当前任务。根据真实工具结果决定下一步；不要重复已成功的动作，不要跳过必要步骤。",
                },
                *item["messages"],
            ],
            "tools": [catalog[name] for name in item["available_tools"]],
            "tool_choice": "auto",
            "temperature": 0,
            "seed": args.seed,
            "max_tokens": item["max_output_tokens"],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        response = post_json(endpoint_url(args.endpoint), payload, args.timeout)
        choice = response["choices"][0]
        message = choice["message"]
        row = {
            "id": item["id"],
            "split": item["split"],
            "output": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage", {}),
            "timings": response.get("timings", {}),
            "wall_ms": round((time.perf_counter() - started) * 1000, 3),
            "model": args.model,
            "seed": args.seed,
        }
        rows.append(row)
        print(f"[{index:02d}/{len(contract['items']):02d}] {item['id']} {row['wall_ms']/1000:.2f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "format": "colorlm-agent-state-gate-run-v1",
        "contract": str(args.contract.resolve()),
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
        "model": args.model,
        "seed": args.seed,
        "records": len(rows),
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
