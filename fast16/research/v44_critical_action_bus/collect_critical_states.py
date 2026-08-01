"""一次加载 v36，采集 v44 dev 关键动作 teacher 的 hidden、full logits 和精确 NLL。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V13 = ROOT / "fast16/research/v13_counterfactual_router.py"
V29_FIT = ROOT / "fast16/research/v29_sequence_policy_head/fit_sequence_policy.py"
V31 = ROOT / "fast16/research/v31_qwen36_expert_pair"
sys.path.insert(0, os.fspath(V31))

from run_v31_gate import port_available, start_server, stop_server  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def import_fit():
    spec = importlib.util.spec_from_file_location("colorlm_v29_fit", V29_FIT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载CNOB验证器: {V29_FIT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8144)
    parser.add_argument("--server", type=Path, default=ROOT / "build/bin/Release/llama-server.exe")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "fast16/models/ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf",
    )
    parser.add_argument("--teacher", type=Path, default=HERE / "critical-teacher-dev-v1.jsonl")
    parser.add_argument("--capture", type=Path, default=HERE / "critical-states-dev-v1.cnob")
    parser.add_argument("--nll", type=Path, default=HERE / "critical-nll-dev-v1.jsonl")
    parser.add_argument("--index", type=Path, default=HERE / "critical-capture-index-dev-v1.jsonl")
    parser.add_argument("--manifest", type=Path, default=HERE / "critical-capture-manifest-dev-v1.json")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    teacher_manifest = args.teacher.with_suffix(args.teacher.suffix + ".manifest.json")
    nll_manifest = args.nll.with_suffix(args.nll.suffix + ".manifest.json")
    arm = Path(str(args.capture) + ".arm")
    for path in (args.server, args.model, args.teacher, teacher_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    outputs = (args.capture, args.nll, args.index, args.manifest, nll_manifest, arm)
    occupied = [path for path in outputs if path.exists()]
    if occupied:
        raise FileExistsError("关键采集产物已存在，拒绝覆盖: " + ", ".join(map(str, occupied)))
    if not port_available(args.port):
        raise RuntimeError(f"端口已占用: {args.port}")

    contract = json.loads(teacher_manifest.read_text(encoding="utf-8"))
    teacher = read_jsonl(args.teacher)
    if (
        contract.get("purpose") != "v44-critical-semantic-span-dev-only"
        or contract.get("teacher_sha256") != sha256_file(args.teacher)
        or contract.get("sample_count") != len(teacher)
        or contract.get("test_leakage_count") != 0
        or any(row.get("split") == "test" for row in teacher)
    ):
        raise ValueError("关键teacher合同、哈希或test隔离无效")
    if len(teacher) > 500:
        raise ValueError("关键teacher超出500样本速度预算")
    for path in (args.capture, args.nll, args.index, args.manifest):
        path.parent.mkdir(parents=True, exist_ok=True)

    alias = "ColorLM-v44-Critical-Capture"
    environment = {
        "COLORLM_NEURAL_OUTPUT_CAPTURE": str(args.capture.resolve()),
        "COLORLM_NEURAL_OUTPUT_CAPTURE_ARM": str(arm.resolve()),
        "COLORLM_NEURAL_OUTPUT_CAPTURE_MAX_RECORDS": str(len(teacher)),
    }
    process, load_seconds = start_server(
        args.server,
        args.model,
        alias,
        args.port,
        extra_environment=environment,
    )
    arm.write_text("armed\n", encoding="utf-8")
    try:
        run(
            [
                sys.executable,
                str(V13),
                "collect",
                "--endpoint",
                f"http://127.0.0.1:{args.port}",
                "--expect-model",
                alias,
                "--route",
                "v44-critical-v36-base",
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
        stop_server(process)

    fit = import_fit()
    capture = fit.load_capture(args.capture)
    nll_rows = read_jsonl(args.nll)
    if len(nll_rows) != len(teacher) or sorted(capture) != list(range(len(teacher))):
        raise RuntimeError("teacher/NLL/CNOB数量或record编号不一致")
    if any(set(bucket) != {fit.BASE_HIDDEN, fit.BASE_LOGITS} for bucket in capture.values()):
        raise RuntimeError("每个record必须恰有terminal hidden和base logits")

    index_rows = []
    for record, (teacher_row, nll_row) in enumerate(zip(teacher, nll_rows, strict=True)):
        if teacher_row["sample_id"] != nll_row["sample_id"] or not nll_row.get("exact"):
            raise RuntimeError(f"record {record} teacher/NLL不对齐或非精确")
        index_rows.append(
            {
                "record": record,
                "sample_id": teacher_row["sample_id"],
                "task_id": teacher_row["task_id"],
                "split": teacher_row["split"],
                "capability": teacher_row["capability"],
                "token_index": teacher_row["token_index"],
                "target_token_id": teacher_row["target_token_id"],
                "target_piece": teacher_row["target_piece"],
                "critical_roles": teacher_row["critical_roles"],
                "target_nll": float(nll_row["target_nll"]),
                "exact": True,
            }
        )
    write_jsonl(args.index, index_rows)
    manifest = {
        "format": "colorlm-v44-critical-capture-manifest-v1",
        "model": str(args.model),
        "model_alias": alias,
        "load_seconds": load_seconds,
        "teacher": str(args.teacher.resolve()),
        "teacher_sha256": sha256_file(args.teacher),
        "capture": str(args.capture.resolve()),
        "capture_sha256": sha256_file(args.capture),
        "capture_bytes": args.capture.stat().st_size,
        "nll": str(args.nll.resolve()),
        "nll_sha256": sha256_file(args.nll),
        "index": str(args.index.resolve()),
        "index_sha256": sha256_file(args.index),
        "sample_count": len(teacher),
        "exact_nll_count": len(nll_rows),
        "tensor_records": len(teacher) * 2,
        "test_leakage_count": 0,
        "arm_removed": not arm.exists(),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
