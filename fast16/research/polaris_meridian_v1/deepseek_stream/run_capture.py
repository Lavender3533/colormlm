"""运行经显式适配器证明的 DeepSeek 原生采集；默认只 dry-run。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

try:
    from .capture_io import CaptureWriter, read_jsonl, sha256_file, validate_capture
    from .planner import DEFAULT_SNAPSHOT, make_plan, read_json, verify_metadata
except ImportError:
    from capture_io import CaptureWriter, read_jsonl, sha256_file, validate_capture
    from planner import DEFAULT_SNAPSHOT, make_plan, read_json, verify_metadata


REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
ADAPTER_API_VERSION = 1


def load_tasks(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    seen: set[str] = set()
    for line, row in enumerate(rows, start=1):
        if row.get("format") != "polaris-deepseek-capture-task-v1":
            raise ValueError(f"tasks 第 {line} 行 format 错误")
        sequence_id = str(row.get("sequence_id", ""))
        if not sequence_id or sequence_id in seen:
            raise ValueError(f"tasks 第 {line} 行 sequence_id 为空或重复")
        seen.add(sequence_id)
        if not isinstance(row.get("prompt_utf8"), str) or not row["prompt_utf8"]:
            raise ValueError(f"tasks 第 {line} 行 prompt_utf8 为空")
        if int(row.get("max_observed_tokens", 0)) <= 0:
            raise ValueError(f"tasks 第 {line} 行 max_observed_tokens 必须为正数")
    return rows


def load_adapter(spec: str) -> Callable[..., dict[str, Any]]:
    if ":" not in spec:
        raise ValueError("--adapter 必须是 module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    api = int(getattr(module, "POLARIS_DEEPSEEK_ADAPTER_API", -1))
    if api != ADAPTER_API_VERSION:
        raise RuntimeError(f"adapter API={api}，期望 {ADAPTER_API_VERSION}")
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"adapter 函数不可调用: {spec}")
    return function


def validate_attestation(value: dict[str, Any], records: int) -> None:
    required = {
        "format": "polaris-deepseek-native-adapter-attestation-v1",
        "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": REVISION,
        "native_forward_completed": True,
        "synthetic": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise RuntimeError(f"adapter attestation {key}={value.get(key)!r}，期望 {expected!r}")
    if int(value.get("records", -1)) != records:
        raise RuntimeError("adapter attestation records 与 writer 不一致")
    verification = value.get("weights_verification")
    if not isinstance(verification, dict) or verification.get("status") not in {
        "all_45_files_sha256_verified",
        "verified_remote_tensor_source",
    }:
        raise RuntimeError("adapter 未证明完整 base-forward 权重来源")
    runtime = value.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("implementation"):
        raise RuntimeError("adapter runtime 证明缺失")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", help="module:function；模块必须声明 POLARIS_DEEPSEEK_ADAPTER_API=1")
    parser.add_argument("--execute-native", action="store_true", help="显式允许调用 adapter；省略时严格 dry-run")
    parser.add_argument("--network-mib-s", type=float, default=50.0)
    parser.add_argument("--available-disk-gib", type=float, default=50.0)
    parser.add_argument("--available-ram-gib", type=float, default=1500.0)
    args = parser.parse_args()

    snapshot = read_json(args.snapshot)
    if snapshot.get("revision") != REVISION:
        raise ValueError("snapshot revision 错误")
    metadata = verify_metadata(snapshot, args.config, args.index)
    tasks = load_tasks(args.tasks)
    observed = max(int(row["max_observed_tokens"]) for row in tasks)
    plan = make_plan(
        snapshot,
        task_count=len(tasks),
        observed_tokens_per_task=observed,
        network_mib_s=args.network_mib_s,
        available_disk_gib=args.available_disk_gib,
        available_ram_gib=args.available_ram_gib,
        hbm_gib=32.0,
        window_seconds=7200,
        metadata_verification=metadata,
    )
    if not args.execute_native:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if not args.adapter:
        raise ValueError("--execute-native 必须同时提供 --adapter")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cnob = args.output_dir / "deepseek_native_states.cnob"
    sidecar = args.output_dir / "deepseek_native_tokens.jsonl"
    manifest = args.output_dir / "capture_manifest.json"
    if manifest.exists():
        raise FileExistsError(f"拒绝覆盖 {manifest}")
    adapter = load_adapter(args.adapter)
    writer = CaptureWriter(cnob, sidecar)
    try:
        context = {
            "adapter_api": ADAPTER_API_VERSION,
            "snapshot": snapshot,
            "capture_contract": read_json(Path(__file__).with_name("capture_contract.json")),
            "tasks_path": str(args.tasks.resolve()),
            "tasks_sha256": sha256_file(args.tasks),
            "claim_boundary": "adapter 只能写真实官方 forward；合成数据请使用 selftest.py。",
        }
        attestation = adapter(context=context, tasks=tasks, writer=writer)
        if not isinstance(attestation, dict):
            raise TypeError("adapter 必须返回 attestation object")
        validate_attestation(attestation, writer.records)
        writer.finish()
        validation = validate_capture(cnob, sidecar, require_real=True)
    except Exception:
        writer.abort()
        raise

    result = {
        "format": "polaris-deepseek-native-capture-manifest-v1",
        "status": "native_capture_completed",
        "source": {"repo": snapshot["repo"], "revision": snapshot["revision"]},
        "tasks": {"path": str(args.tasks.resolve()), "sha256": sha256_file(args.tasks), "count": len(tasks)},
        "adapter": args.adapter,
        "attestation": attestation,
        "validation": validation,
        "plan": plan,
        "claim_limit": "该 manifest 只证明本次 adapter 声明与文件契约；能力结论仍需独立冻结评测。",
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    manifest.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
