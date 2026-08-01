"""从 v43 dev 轨迹构建只覆盖判别性工具/参数/结束字段的精简 teacher。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V31 = ROOT / "fast16/research/v31_qwen36_expert_pair"
sys.path.insert(0, os.fspath(V31))

from run_v31_gate import port_available, start_server, stop_server  # noqa: E402


TEACHER_FORMAT = "colorlm-v13-counterfactual-teacher-v1"
MANIFEST_FORMAT = "colorlm-v13-counterfactual-teacher-manifest-v1"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


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
    raise ValueError(f"无效token piece: {piece!r}")


def unique_pair(target: bytes, pair: str, role: str) -> tuple[int, bytes]:
    encoded = pair.encode("utf-8")
    start = target.find(encoded)
    if start < 0 or target.find(encoded, start + 1) >= 0:
        raise ValueError(f"{role}在target中缺失或不唯一: {pair!r}")
    return start, encoded


def json_inner_range(encoded: bytes, absolute_start: int) -> tuple[int, int]:
    if len(encoded) >= 2 and encoded[:1] == b'"' and encoded[-1:] == b'"':
        return absolute_start + 1, absolute_start + len(encoded) - 1
    return absolute_start, absolute_start + len(encoded)


def semantic_spans(target: str, expected: dict[str, Any]) -> list[tuple[int, int, str]]:
    raw = target.encode("utf-8")
    spans: list[tuple[int, int, str]] = []
    if expected["type"] == "tool":
        name_json = canonical(expected["name"]).encode("utf-8")
        marker = '"name":' + canonical(expected["name"])
        marker_start, marker_bytes = unique_pair(raw, marker, "tool_name")
        value_start = marker_start + len(marker_bytes) - len(name_json)
        start, end = json_inner_range(name_json, value_start)
        spans.append((start, end, "tool_name"))
        for key, value in expected["arguments"].items():
            key_json = canonical(key).encode("utf-8")
            value_json = canonical(value).encode("utf-8")
            pair = canonical(key) + ":" + canonical(value)
            pair_start, _ = unique_pair(raw, pair, f"argument:{key}")
            key_start, key_end = json_inner_range(key_json, pair_start)
            value_offset = pair_start + len(key_json) + 1
            value_start, value_end = json_inner_range(value_json, value_offset)
            spans.append((key_start, key_end, f"argument_key:{key}"))
            spans.append((value_start, value_end, f"argument_value:{key}"))
        return spans
    if expected["type"] == "finish":
        value = json.loads(expected["content"])
        if not isinstance(value, dict):
            return [(0, len(raw), "finish_content")]
        for key, item in value.items():
            key_json = canonical(key).encode("utf-8")
            value_json = canonical(item).encode("utf-8")
            pair = canonical(key) + ":" + canonical(item)
            pair_start, _ = unique_pair(raw, pair, f"finish:{key}")
            key_start, key_end = json_inner_range(key_json, pair_start)
            value_offset = pair_start + len(key_json) + 1
            value_start, value_end = json_inner_range(value_json, value_offset)
            spans.append((key_start, key_end, f"finish_key:{key}"))
            spans.append((value_start, value_end, f"finish_value:{key}"))
        return spans
    raise ValueError(f"未知expected action类型: {expected['type']}")


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
    parser.add_argument(
        "--source-teacher",
        type=Path,
        default=ROOT / "fast16/research/v43_policy_dataset/teacher.jsonl",
    )
    parser.add_argument("--output", type=Path, default=HERE / "critical-teacher-dev-v1.jsonl")
    parser.add_argument("--splits", default="train,validation")
    args = parser.parse_args()

    for path in (args.server, args.model, args.tasks, args.oracle, args.source_teacher):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError("关键teacher产物已存在，拒绝覆盖")
    if not port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    if not splits or not splits <= {"train", "validation"}:
        raise ValueError("只允许从train/validation构建dev teacher，严禁引入已消费test")

    tasks = read_jsonl(args.tasks)
    selected_tasks = [row for row in tasks if row["split"] in splits]
    oracle_by_id = {row["case_id"]: row for row in read_jsonl(args.oracle)}
    source_teacher = read_jsonl(args.source_teacher)
    first_prefix: dict[str, list[int]] = {}
    for row in source_teacher:
        if int(row["token_index"]) == 0:
            first_prefix[str(row["task_id"])] = [int(value) for value in row["prefix_tokens"]]
    if any(task["id"] not in first_prefix or task["id"] not in oracle_by_id for task in selected_tasks):
        raise ValueError("dev任务缺少原始prompt前缀或oracle")

    alias = "ColorLM-v44-Critical-Teacher"
    process, load_seconds = start_server(args.server, args.model, alias, args.port)
    rows: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    role_counts: defaultdict[str, int] = defaultdict(int)
    try:
        endpoint = f"http://127.0.0.1:{args.port}/tokenize"
        for task in selected_tasks:
            response = post_json(
                endpoint,
                {
                    "content": task["target"],
                    "add_special": False,
                    "parse_special": True,
                    "with_pieces": True,
                },
            )
            tokens = response["tokens"]
            ids = [int(token["id"]) for token in tokens]
            pieces = [piece_bytes(token["piece"]) for token in tokens]
            target_bytes = task["target"].encode("utf-8")
            if b"".join(pieces) != target_bytes:
                raise ValueError(f"target无法被token pieces精确还原: {task['id']}")
            bounds = []
            cursor = 0
            for piece in pieces:
                bounds.append((cursor, cursor + len(piece)))
                cursor += len(piece)
            positions: defaultdict[int, list[str]] = defaultdict(list)
            for start, end, role in semantic_spans(
                task["target"], oracle_by_id[task["id"]]["expected_action"]
            ):
                candidates = [
                    index
                    for index, (token_start, token_end) in enumerate(bounds)
                    if token_start < end and token_end > start
                ]
                if not candidates:
                    raise ValueError(f"语义span未覆盖token: {task['id']} {role}")
                # 每个语义字段只采首个判别token；它是最小、最快且可证伪的介入点。
                positions[candidates[0]].append(role)
            task_counts[task["id"]] = len(positions)
            base_prefix = first_prefix[task["id"]]
            for position in sorted(positions):
                roles = sorted(positions[position])
                for role in roles:
                    role_counts[role.split(":", 1)[0]] += 1
                piece = tokens[position]["piece"]
                rows.append(
                    {
                        "format": TEACHER_FORMAT,
                        "sample_id": f"{task['id']}:critical:{position:04d}",
                        "task_id": task["id"],
                        "token_index": position,
                        "prefix_tokens": base_prefix + ids[:position],
                        "target_token_id": ids[position],
                        "boundary_mode": "critical-semantic-span-first-token",
                        "split": task["split"],
                        "capability": task["capability"],
                        "critical_roles": roles,
                        "target_piece": piece,
                    }
                )
    finally:
        stop_server(process)

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)) or not rows:
        raise ValueError("关键teacher为空或sample_id重复")
    if any(row["split"] == "test" for row in rows):
        raise ValueError("关键teacher泄漏v43 test")
    write_jsonl(args.output, rows)
    manifest = {
        "format": MANIFEST_FORMAT,
        "purpose": "v44-critical-semantic-span-dev-only",
        "source_tasks": str(args.tasks.resolve()),
        "source_tasks_sha256": sha256_file(args.tasks),
        "source_oracle": str(args.oracle.resolve()),
        "source_oracle_sha256": sha256_file(args.oracle),
        "source_teacher": str(args.source_teacher.resolve()),
        "source_teacher_sha256": sha256_file(args.source_teacher),
        "teacher": args.output.name,
        "teacher_sha256": sha256_file(args.output),
        "sample_count": len(rows),
        "task_count": len(selected_tasks),
        "splits": sorted(splits),
        "role_counts": dict(sorted(role_counts.items())),
        "minimum_samples_per_task": min(task_counts.values()),
        "maximum_samples_per_task": max(task_counts.values()),
        "selection": "first-token-overlapping-each-semantic-field",
        "test_leakage_count": 0,
        "load_seconds": load_seconds,
        "tokenizer_endpoint": endpoint,
        "model": str(args.model),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
