"""为 v47 构建“每任务一次 terminal hidden + 完整目标序列”的采集契约。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
V13 = PROJECT / "fast16/research/v13_counterfactual_router.py"
FORMAT_TEACHER = "colorlm-v13-counterfactual-teacher-v1"
FORMAT_TEACHER_MANIFEST = "colorlm-v13-counterfactual-teacher-manifest-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def import_v13():
    spec = importlib.util.spec_from_file_location("colorlm_v13", V13)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 v13 采集器: {V13}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_split(value: Any, map_test_to_blind: bool) -> str:
    split = str(value)
    if split == "test" and map_test_to_blind:
        return "blind"
    if split not in {"train", "validation", "blind"}:
        raise ValueError(f"split 必须是 train/validation/blind，实际为 {split!r}")
    return split


def verify_isolation(rows: list[dict[str, Any]]) -> None:
    for key in ("group_id", "template_cluster_id"):
        seen: dict[str, str] = {}
        for row in rows:
            value = str(row[key])
            old = seen.setdefault(value, str(row["split"]))
            if old != row["split"]:
                raise ValueError(f"{key}={value!r} 同时出现在 {old} 与 {row['split']}，拒绝泄漏")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8147")
    parser.add_argument("--expect-model", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--capture-teacher", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-target-tokens", type=int, default=256)
    parser.add_argument("--max-prefix-tokens", type=int, default=8192)
    parser.add_argument("--map-test-to-blind", action="store_true")
    parser.add_argument("--collect", action="store_true", help="准备后立刻调用 v13 collect；服务必须已配置 CNOB 环境变量")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--nll", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    outputs = [args.capture_teacher, args.dataset, args.manifest]
    if args.collect:
        if args.capture is None or args.nll is None:
            raise ValueError("--collect 必须同时给出 --capture 与 --nll")
        outputs.extend([args.capture, args.nll])
    occupied = [path for path in outputs if path.exists()]
    if occupied:
        raise FileExistsError("拒绝覆盖已有采集产物: " + ", ".join(str(path) for path in occupied))
    if args.max_target_tokens <= 0 or args.max_prefix_tokens <= 0:
        raise ValueError("token 上限必须为正数")

    v13 = import_v13()
    snapshot = v13.endpoint_snapshot(args.endpoint, args.expect_model)
    tasks = read_jsonl(args.tasks)
    if not tasks:
        raise ValueError("任务文件为空")

    capture_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    boundary_fallbacks = 0
    for record, task in enumerate(tasks):
        task_id = str(task.get("id") or "")
        target = task.get("target")
        if not task_id or not isinstance(target, str) or not target:
            raise ValueError(f"任务第 {record + 1} 行缺少非空 id/target")
        for field in ("group_id", "template_cluster_id", "capability"):
            if not isinstance(task.get(field), str) or not task[field]:
                raise ValueError(f"任务 {task_id} 缺少 {field}")
        split = normalize_split(task.get("split"), args.map_test_to_blind)
        prompt = v13.render_task_prompt(args.endpoint, task)
        prompt_tokens = v13.tokenize(args.endpoint, prompt, add_special=True)
        combined = v13.tokenize(args.endpoint, prompt + target, add_special=True)
        if combined[: len(prompt_tokens)] == prompt_tokens:
            target_tokens = combined[len(prompt_tokens) :]
            boundary_mode = "combined"
        else:
            target_tokens = v13.tokenize(args.endpoint, target, add_special=False)
            boundary_mode = "separate_target"
            boundary_fallbacks += 1
        if not target_tokens:
            raise ValueError(f"任务 {task_id} 的 target 没有 token")
        if len(target_tokens) > args.max_target_tokens:
            raise ValueError(
                f"任务 {task_id} 有 {len(target_tokens)} 个目标 token，超过上限 {args.max_target_tokens}；"
                "禁止静默截断完整序列"
            )
        if len(prompt_tokens) > args.max_prefix_tokens:
            raise ValueError(f"任务 {task_id} 的 prompt token={len(prompt_tokens)} 超过上限")

        prefix_hash = sha256_json(prompt_tokens)
        capture_rows.append(
            {
                "format": FORMAT_TEACHER,
                "sample_id": f"{task_id}:initial",
                "task_id": task_id,
                "token_index": 0,
                "prefix_tokens": prompt_tokens,
                "target_token_id": int(target_tokens[0]),
                "boundary_mode": boundary_mode,
            }
        )
        sequence_rows.append(
            {
                "format": "colorlm-v47-sequence-record-v1",
                "task_id": task_id,
                "group_id": task["group_id"],
                "template_cluster_id": task["template_cluster_id"],
                "split": split,
                "capability": task["capability"],
                "target_mode": task.get("target_mode", "exact_token_sequence"),
                "target_text": target,
                "target_token_ids": [int(token) for token in target_tokens],
                "capture_record": record,
                "prefix_sha256": prefix_hash,
                "source_task_sha256": sha256_json(task),
                "metadata": {
                    "boundary_mode": boundary_mode,
                    "prefix_token_count": len(prompt_tokens),
                    "target_token_count": len(target_tokens),
                },
            }
        )

    if len({row["task_id"] for row in sequence_rows}) != len(sequence_rows):
        raise ValueError("task_id 重复")
    verify_isolation(sequence_rows)
    write_jsonl(args.capture_teacher, capture_rows)
    teacher_manifest = {
        "format": FORMAT_TEACHER_MANIFEST,
        "source_tasks": str(args.tasks.resolve()),
        "source_tasks_sha256": sha256_file(args.tasks),
        "teacher": args.capture_teacher.name,
        "teacher_sha256": sha256_file(args.capture_teacher),
        "sample_count": len(capture_rows),
        "task_count": len(capture_rows),
        "boundary_fallbacks": boundary_fallbacks,
        "max_target_tokens": 1,
        "max_prefix_tokens": args.max_prefix_tokens,
        "tokenizer_endpoint": snapshot,
        "purpose": "v47-one-initial-hidden-per-complete-sequence",
    }
    write_json(args.capture_teacher.with_suffix(args.capture_teacher.suffix + ".manifest.json"), teacher_manifest)
    write_jsonl(args.dataset, sequence_rows)

    manifest: dict[str, Any] = {
        "format": "colorlm-v47-sequence-capture-manifest-v1",
        "status": "prepared",
        "model_alias": args.expect_model,
        "tokenizer_endpoint": snapshot,
        "tasks": str(args.tasks.resolve()),
        "tasks_sha256": sha256_file(args.tasks),
        "capture_teacher": str(args.capture_teacher.resolve()),
        "capture_teacher_sha256": sha256_file(args.capture_teacher),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "task_count": len(sequence_rows),
        "split_counts": dict(sorted((key, sum(row["split"] == key for row in sequence_rows)) for key in {row["split"] for row in sequence_rows})),
        "boundary_fallbacks": boundary_fallbacks,
        "one_initial_hidden_per_task": True,
        "full_sequence_truncations": 0,
    }

    if args.collect:
        assert args.capture is not None and args.nll is not None
        arm = args.capture.with_suffix(args.capture.suffix + ".arm")
        arm.write_text("armed\n", encoding="utf-8")
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(V13),
                    "collect",
                    "--endpoint",
                    args.endpoint,
                    "--expect-model",
                    args.expect_model,
                    "--route",
                    "v47-initial-state",
                    "--teacher",
                    str(args.capture_teacher),
                    "--output",
                    str(args.nll),
                    "--n-probs",
                    "1",
                    "--timeout",
                    str(args.timeout),
                ],
                cwd=PROJECT,
                check=True,
            )
        finally:
            arm.unlink(missing_ok=True)
        if not args.capture.is_file():
            raise RuntimeError(
                "服务没有生成 CNOB；启动服务前必须设置 COLORLM_NEURAL_OUTPUT_CAPTURE、"
                "COLORLM_NEURAL_OUTPUT_CAPTURE_ARM 和 COLORLM_NEURAL_OUTPUT_CAPTURE_MAX_RECORDS"
            )
        manifest.update(
            {
                "status": "captured",
                "capture": str(args.capture.resolve()),
                "capture_sha256": sha256_file(args.capture),
                "capture_bytes": args.capture.stat().st_size,
                "nll": str(args.nll.resolve()),
                "nll_sha256": sha256_file(args.nll),
                "arm_removed": not arm.exists(),
            }
        )

    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

