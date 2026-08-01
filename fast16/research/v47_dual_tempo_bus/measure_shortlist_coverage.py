"""用现有 CNOB logits 测量 rank-shortlist 草稿头的词表候选覆盖率。"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HEADER = struct.Struct("<6I4qQ")
MAGIC = 0x424F4E43
BASE_LOGITS = 1
F32 = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stream_topk(path: Path, count: int, expected_records: int) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    with path.open("rb") as source:
        while True:
            encoded = source.read(HEADER.size)
            if not encoded:
                break
            if len(encoded) != HEADER.size:
                raise ValueError("CNOB 头被截断")
            magic, version, kind, record, dtype, reserved, *tail = HEADER.unpack(encoded)
            ne = tuple(int(value) for value in tail[:4])
            payload_bytes = int(tail[4])
            if magic != MAGIC or version != 1 or dtype != F32 or reserved != 0:
                raise ValueError("CNOB 头不符合契约")
            if kind != BASE_LOGITS:
                source.seek(payload_bytes, 1)
                continue
            if ne[1:] != (1, 1, 1) or payload_bytes != ne[0] * 4:
                raise ValueError(f"logits shape 错误: record={record}, ne={ne}")
            payload = source.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError("CNOB logits payload 被截断")
            logits = np.frombuffer(payload, dtype="<f4")
            k = min(count, len(logits))
            indices = np.argpartition(logits, len(logits) - k)[-k:]
            indices = indices[np.argsort(logits[indices])[::-1]]
            result[int(record)] = [int(value) for value in indices]
    if set(result) != set(range(expected_records)):
        raise ValueError(f"logits record 不连续: got={len(result)}, expected={expected_records}")
    return result


def append_unique(target: list[int], seen: set[int], values: list[int], limit: int) -> None:
    for value in values:
        if value in seen:
            continue
        target.append(value)
        seen.add(value)
        if len(target) >= limit:
            break


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    return {
        "sample_count": len(rows),
        "covered": sum(bool(row["covered"]) for row in rows),
        "coverage": float(np.mean([row["covered"] for row in rows])),
        "native_topk_coverage": float(np.mean([row["native_covered"] for row in rows])),
        "mean_candidate_count": float(np.mean([row["candidate_count"] for row in rows])),
        "maximum_candidate_count": max(int(row["candidate_count"]) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-top-k", type=int, default=256)
    parser.add_argument("--prompt-suffix", type=int, default=192)
    parser.add_argument("--frequent-train", type=int, default=128)
    parser.add_argument("--candidate-limit", type=int, default=512)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖报告: {args.output}")
    if min(args.native_top_k, args.prompt_suffix, args.frequent_train, args.candidate_limit, args.rank) <= 0:
        raise ValueError("所有候选/秩参数必须为正数")
    if args.native_top_k > args.candidate_limit:
        raise ValueError("native-top-k 不能超过 candidate-limit")

    started = time.perf_counter()
    teacher = read_jsonl(args.teacher)
    tasks = read_jsonl(args.tasks)
    task_by_id = {str(row["id"]): row for row in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("任务 ID 重复")
    if any(str(row["task_id"]) not in task_by_id for row in teacher):
        raise ValueError("teacher 包含未知任务")
    train_frequency = Counter(
        int(row["target_token_id"])
        for row in teacher
        if task_by_id[str(row["task_id"])]["split"] == "train"
    )
    frequent = [token for token, _ in train_frequency.most_common(args.frequent_train)]
    topk = stream_topk(args.capture, args.native_top_k, len(teacher))

    evaluated = []
    for record, row in enumerate(teacher):
        task = task_by_id[str(row["task_id"])]
        candidates: list[int] = []
        seen: set[int] = set()
        append_unique(candidates, seen, topk[record], args.candidate_limit)
        suffix = [int(value) for value in row["prefix_tokens"][-args.prompt_suffix :]][::-1]
        append_unique(candidates, seen, suffix, args.candidate_limit)
        append_unique(candidates, seen, frequent, args.candidate_limit)
        target = int(row["target_token_id"])
        evaluated.append(
            {
                "record": record,
                "task_id": row["task_id"],
                "split": task["split"],
                "capability": task.get("capability", task.get("family", "unknown")),
                "token_index": int(row["token_index"]),
                "target_token_id": target,
                "candidate_count": len(candidates),
                "native_covered": target in set(topk[record]),
                "covered": target in seen,
            }
        )

    by_split = {
        split: summarize([row for row in evaluated if row["split"] == split])
        for split in sorted({str(row["split"]) for row in evaluated})
    }
    by_capability = {
        capability: summarize([row for row in evaluated if row["capability"] == capability])
        for capability in sorted({str(row["capability"]) for row in evaluated})
    }
    candidate_mean = float(np.mean([row["candidate_count"] for row in evaluated]))
    full_projection_macs = 2048 * 248320
    shortlist_projection_macs = args.rank * candidate_mean
    report = {
        "format": "colorlm-v47-shortlist-coverage-report-v1",
        "status": "development_feasibility_only",
        "warning": "teacher-forced v43 states and consumed same-template splits; not a v47 ability or speed result",
        "capture": str(args.capture.resolve()),
        "capture_sha256": sha256_file(args.capture),
        "teacher": str(args.teacher.resolve()),
        "teacher_sha256": sha256_file(args.teacher),
        "configuration": {
            "native_top_k": args.native_top_k,
            "prompt_suffix": args.prompt_suffix,
            "frequent_train": args.frequent_train,
            "candidate_limit": args.candidate_limit,
            "rank": args.rank,
            "priority": ["native_topk", "recent_prompt_tokens", "train_frequent_targets"],
        },
        "overall": summarize(evaluated),
        "by_split": by_split,
        "by_capability": by_capability,
        "projection_cost": {
            "dense_2048x248320_macs": full_projection_macs,
            "rank_times_mean_candidates_macs": shortlist_projection_macs,
            "shortlist_to_dense_ratio": shortlist_projection_macs / full_projection_macs,
            "does_not_include": ["feature_fusion", "candidate_construction", "v38_verification"],
        },
        "gate": {
            "validation_coverage_min": 0.995,
            "test_coverage_min": 0.995,
            "passed": bool(
                by_split.get("validation", {}).get("coverage", 0.0) >= 0.995
                and by_split.get("test", {}).get("coverage", 0.0) >= 0.995
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "misses": [row for row in evaluated if not row["covered"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
