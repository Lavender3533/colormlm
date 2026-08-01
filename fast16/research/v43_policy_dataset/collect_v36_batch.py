"""一次 v36 加载内采集 v43 teacher 的 terminal hidden、base logits 与精确 NLL。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
V13 = PROJECT / "fast16/research/v13_counterfactual_router.py"
V29_FIT = PROJECT / "fast16/research/v29_sequence_policy_head/fit_sequence_policy.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def import_fit():
    spec = importlib.util.spec_from_file_location("colorlm_v29_fit", V29_FIT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载采集解析器: {V29_FIT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8143")
    parser.add_argument("--expect-model", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--nll", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-target-tokens", type=int, default=6)
    parser.add_argument("--max-prefix-tokens", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    paths = [args.teacher, args.capture, args.nll, args.index, args.manifest]
    arm = args.capture.with_suffix(args.capture.suffix + ".arm")
    generated_manifests = [
        args.teacher.with_suffix(args.teacher.suffix + ".manifest.json"),
        args.nll.with_suffix(args.nll.suffix + ".manifest.json"),
    ]
    occupied = [path for path in [*paths, *generated_manifests, arm] if path.exists()]
    if occupied:
        raise FileExistsError("采集输出已存在，拒绝混入旧记录: " + ", ".join(map(str, occupied)))
    if args.max_target_tokens <= 0 or args.max_samples <= 0:
        raise ValueError("token/sample限制必须为正数")

    tasks = read_jsonl(args.tasks)
    task_by_id = {str(row["id"]): row for row in tasks}
    if len(tasks) < 100 or len(task_by_id) != len(tasks):
        raise ValueError("v43采集要求至少100个唯一任务")
    required = {"split", "family", "label", "capability", "group_id", "target"}
    for task in tasks:
        missing = required - set(task)
        if missing:
            raise ValueError(f"任务 {task.get('id')} 缺字段: {sorted(missing)}")

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(V13),
            "prepare",
            "--endpoint",
            args.endpoint,
            "--expect-model",
            args.expect_model,
            "--tasks",
            str(args.tasks),
            "--output",
            str(args.teacher),
            "--max-target-tokens",
            str(args.max_target_tokens),
            "--max-prefix-tokens",
            str(args.max_prefix_tokens),
            "--max-samples",
            str(args.max_samples),
        ]
    )
    teacher = read_jsonl(args.teacher)
    if len(teacher) > args.max_samples:
        raise RuntimeError("teacher超过采集上限")

    arm.write_text("armed\n", encoding="utf-8")
    try:
        run(
            [
                sys.executable,
                str(V13),
                "collect",
                "--endpoint",
                args.endpoint,
                "--expect-model",
                args.expect_model,
                "--route",
                "v43-v36-base",
                "--teacher",
                str(args.teacher),
                "--output",
                str(args.nll),
                "--n-probs",
                "1",
                "--timeout",
                str(args.timeout),
            ]
        )
    finally:
        arm.unlink(missing_ok=True)

    fit = import_fit()
    capture = fit.load_capture(args.capture)
    nll_rows = read_jsonl(args.nll)
    if len(teacher) != len(nll_rows):
        raise RuntimeError(f"teacher/NLL数量不一致: {len(teacher)}/{len(nll_rows)}")
    if sorted(capture) != list(range(len(teacher))):
        raise RuntimeError("CNOB record编号不连续或数量不等于teacher")
    if any(set(capture[index]) != {fit.BASE_HIDDEN, fit.BASE_LOGITS} for index in capture):
        raise RuntimeError("每个CNOB record必须恰有terminal hidden与base logits")

    index_rows: list[dict[str, Any]] = []
    for record, (teacher_row, nll_row) in enumerate(zip(teacher, nll_rows, strict=True)):
        if teacher_row["sample_id"] != nll_row["sample_id"]:
            raise RuntimeError(f"record {record} 的teacher/NLL sample_id不一致")
        task = task_by_id[str(teacher_row["task_id"])]
        index_rows.append(
            {
                "record": record,
                "sample_id": teacher_row["sample_id"],
                "task_id": teacher_row["task_id"],
                "group_id": task["group_id"],
                "split": task["split"],
                "family": task["family"],
                "label": task["label"],
                "capability": task["capability"],
                "token_index": int(teacher_row["token_index"]),
                "target_token_id": int(teacher_row["target_token_id"]),
                "target_nll": float(nll_row["target_nll"]),
                "exact": bool(nll_row["exact"]),
            }
        )
    if not all(row["exact"] for row in index_rows):
        raise RuntimeError("v43出现非精确NLL记录")
    write_jsonl(args.index, index_rows)
    manifest = {
        "format": "colorlm-v43-v36-capture-manifest-v1",
        "endpoint": args.endpoint,
        "model_alias": args.expect_model,
        "tasks": str(args.tasks.resolve()),
        "tasks_sha256": sha256_file(args.tasks),
        "teacher": str(args.teacher.resolve()),
        "teacher_sha256": sha256_file(args.teacher),
        "capture": str(args.capture.resolve()),
        "capture_sha256": sha256_file(args.capture),
        "capture_bytes": args.capture.stat().st_size,
        "nll": str(args.nll.resolve()),
        "nll_sha256": sha256_file(args.nll),
        "index": str(args.index.resolve()),
        "index_sha256": sha256_file(args.index),
        "task_count": len(tasks),
        "sample_count": len(teacher),
        "exact_nll_count": len(index_rows),
        "tensor_records": len(teacher) * 2,
        "arm_removed": not arm.exists(),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
