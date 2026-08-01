"""Capture exactly one first-decision hidden/logit pair for every frozen v28 item."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM = (
    "继续完成当前任务。根据真实工具结果决定下一步；不要重复已成功的动作，"
    "不要跳过必要步骤。"
)


def endpoint_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return endpoint + ("/chat/completions" if endpoint.endswith("/v1") else "/v1/chat/completions")


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def manifest_path(path: Path) -> str:
    """Keep manifests portable and avoid locale-dependent absolute paths."""
    return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def label_for(item: dict[str, Any]) -> str:
    kind = item["validator"]["type"]
    if kind == "tool_call":
        return "continue_tool"
    if kind == "exact_json":
        return "finish"
    raise ValueError(f"冻结契约中不支持的validator: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description="采集v28首决策hidden/logits")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    contract_bytes = args.contract.read_bytes()
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    contract = json.loads(contract_bytes.decode("utf-8"))
    catalog = contract["tool_catalog"]
    arm = args.capture.with_suffix(args.capture.suffix + ".arm")
    if args.capture.exists() or arm.exists() or args.index.exists():
        raise FileExistsError("采集文件、arm或索引已存在；拒绝覆盖研究证据")
    args.capture.parent.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    arm.write_text("armed\n", encoding="utf-8")
    try:
        for record, item in enumerate(contract["items"]):
            payload = {
                "model": args.model,
                "messages": [{"role": "system", "content": SYSTEM}, *item["messages"]],
                "tools": [catalog[name] for name in item["available_tools"]],
                "tool_choice": "auto",
                "temperature": 0,
                "seed": args.seed,
                "max_tokens": 1,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            started = time.perf_counter()
            response = post_json(endpoint_url(args.endpoint), payload, args.timeout)
            choice = response["choices"][0]
            message = choice["message"]
            rows.append(
                {
                    "record": record,
                    "id": item["id"],
                    "split": item["split"],
                    "label": label_for(item),
                    "first_output": message.get("content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                    "wall_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            print(
                f"[{record + 1:02d}/{len(contract['items']):02d}] "
                f"{item['id']} label={rows[-1]['label']} "
                f"{rows[-1]['wall_ms'] / 1000:.2f}s",
                flush=True,
            )
    finally:
        arm.unlink(missing_ok=True)

    if not args.capture.is_file():
        raise FileNotFoundError("服务没有生成CNOB采集文件")
    index = {
        "format": "colorlm-v28-decision-capture-index-v1",
        "contract": manifest_path(args.contract),
        "contract_sha256": contract_sha256,
        "capture": manifest_path(args.capture),
        "capture_sha256": sha256_file(args.capture),
        "model": args.model,
        "seed": args.seed,
        "records": rows,
    }
    args.index.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
