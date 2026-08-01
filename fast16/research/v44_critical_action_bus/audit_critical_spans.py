"""用基座词表审计 v43 的关键动作 token 是否超出六 token 监督窗口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V31 = ROOT / "fast16/research/v31_qwen36_expert_pair"
sys.path.insert(0, os.fspath(V31))

from run_v31_gate import port_available, start_server, stop_server  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def piece_bytes(piece: Any) -> bytes:
    if isinstance(piece, str):
        return piece.encode("utf-8")
    if isinstance(piece, list) and all(isinstance(value, int) and 0 <= value <= 255 for value in piece):
        return bytes(piece)
    raise ValueError(f"无法解析token piece: {piece!r}")


def find_unique(haystack: bytes, needle: str, role: str) -> tuple[int, int, str]:
    encoded = needle.encode("utf-8")
    start = haystack.find(encoded)
    if start < 0:
        raise ValueError(f"关键字节不在target中: role={role}, needle={needle!r}")
    if haystack.find(encoded, start + 1) >= 0:
        raise ValueError(f"关键字节在target中不唯一: role={role}, needle={needle!r}")
    return start, start + len(encoded), role


def critical_spans(target: str, expected: dict[str, Any]) -> list[tuple[int, int, str]]:
    encoded = target.encode("utf-8")
    kind = expected["type"]
    if kind == "tool":
        spans = [find_unique(encoded, canonical(expected["name"]), "tool_name")]
        for key, value in expected["arguments"].items():
            spans.append(find_unique(encoded, canonical(key) + ":" + canonical(value), f"argument:{key}"))
        return spans
    if kind == "finish":
        content = expected["content"]
        value = json.loads(content)
        if not isinstance(value, dict):
            return [find_unique(encoded, content, "finish_content")]
        return [
            find_unique(encoded, canonical(key) + ":" + canonical(item), f"finish:{key}")
            for key, item in value.items()
        ]
    raise ValueError(f"未知expected action类型: {kind}")


def summarize_positions(rows: list[dict[str, Any]], role_prefix: str) -> dict[str, Any]:
    positions = [
        position
        for row in rows
        for span in row["critical_spans"]
        if span["role"].startswith(role_prefix)
        for position in span["token_positions"]
    ]
    starts = [
        min(span["token_positions"])
        for row in rows
        for span in row["critical_spans"]
        if span["role"].startswith(role_prefix) and span["token_positions"]
    ]
    return {
        "span_count": len(starts),
        "token_count": len(positions),
        "minimum_start_index": min(starts) if starts else None,
        "median_start_index": sorted(starts)[len(starts) // 2] if starts else None,
        "maximum_start_index": max(starts) if starts else None,
        "tokens_outside_first_six": sum(position >= 6 for position in positions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8144)
    parser.add_argument("--server", type=Path, default=ROOT / "build/bin/Release/llama-server.exe")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "fast16/models/ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "fast16/research/v43_policy_dataset/trajectory_tasks_v1.jsonl",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=ROOT / "fast16/research/v43_policy_dataset/trajectory_oracle_v1.jsonl",
    )
    parser.add_argument("--output", type=Path, default=HERE / "critical-span-audit.json")
    args = parser.parse_args()

    for path in (args.server, args.model, args.tasks, args.oracle):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")

    tasks = read_jsonl(args.tasks)
    oracle = read_jsonl(args.oracle)
    oracle_by_id = {row["case_id"]: row for row in oracle}
    if len(oracle_by_id) != len(oracle) or {row["id"] for row in tasks} != set(oracle_by_id):
        raise ValueError("tasks/oracle ID不唯一或不对齐")

    alias = "ColorLM-v44-Tokenizer-Audit"
    process, load_seconds = start_server(args.server, args.model, alias, args.port)
    try:
        rows = []
        endpoint = f"http://127.0.0.1:{args.port}/tokenize"
        for task in tasks:
            target = task["target"]
            response = post_json(
                endpoint,
                {
                    "content": target,
                    "add_special": False,
                    "parse_special": True,
                    "with_pieces": True,
                },
            )
            tokens = response["tokens"]
            pieces = [piece_bytes(token["piece"]) for token in tokens]
            if b"".join(pieces) != target.encode("utf-8"):
                raise ValueError(f"token pieces无法精确还原target: {task['id']}")
            bounds = []
            cursor = 0
            for piece in pieces:
                bounds.append((cursor, cursor + len(piece)))
                cursor += len(piece)
            spans = []
            for start, end, role in critical_spans(target, oracle_by_id[task["id"]]["expected_action"]):
                positions = [
                    index
                    for index, (token_start, token_end) in enumerate(bounds)
                    if token_start < end and token_end > start
                ]
                if not positions:
                    raise ValueError(f"关键span未覆盖任何token: {task['id']} {role}")
                spans.append(
                    {
                        "role": role,
                        "byte_start": start,
                        "byte_end": end,
                        "token_positions": positions,
                        "token_ids": [int(tokens[index]["id"]) for index in positions],
                    }
                )
            rows.append(
                {
                    "id": task["id"],
                    "split": task["split"],
                    "capability": task["capability"],
                    "target_token_count": len(tokens),
                    "critical_spans": spans,
                }
            )
    finally:
        stop_server(process)

    all_critical = [position for row in rows for span in row["critical_spans"] for position in span["token_positions"]]
    tasks_outside = sum(
        any(position >= 6 for span in row["critical_spans"] for position in span["token_positions"])
        for row in rows
    )
    report = {
        "format": "colorlm-v44-critical-span-audit-v1",
        "source_tasks_sha256": sha256_file(args.tasks),
        "source_oracle_sha256": sha256_file(args.oracle),
        "model": str(args.model),
        "load_seconds": load_seconds,
        "teacher_window_tokens": 6,
        "summary": {
            "tasks": len(rows),
            "critical_token_occurrences": len(all_critical),
            "critical_tokens_outside_first_six": sum(position >= 6 for position in all_critical),
            "tasks_with_critical_tokens_outside_first_six": tasks_outside,
            "tool_name": summarize_positions(rows, "tool_name"),
            "arguments": summarize_positions(rows, "argument:"),
            "finish_fields": summarize_positions(rows, "finish:"),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
