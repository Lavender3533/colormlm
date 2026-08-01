#!/usr/bin/env python3
"""Rust bridge 子进程测试专用 peer；不读取模型数据。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


PROTOCOL = "polaris-s14-range-jsonl-v1"
REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
LAYERS = [0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42]


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)


def artifact(kind: str, expert_id: int | None, suffix: str) -> dict:
    path = Path(__file__).resolve()
    return {
        "tensor": f"fixture.{suffix}",
        "kind": kind,
        "expert_id": expert_id,
        "path": str(path),
        "bytes": path.stat().st_size,
        "cache_hit": True,
        "observed_sha256": "0" * 64,
        "authoritative": False,
    }


def main() -> int:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    for line_index, line in enumerate(sys.stdin):
        request = json.loads(line)
        request_id = request["request_id"]
        op = request["op"]
        if line_index == 0:
            emit(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "op": "hello",
                    "status": "ok",
                    "repo": REPO,
                    "revision": REVISION,
                    "profile": "s14_top6",
                    "selected_layers": LAYERS,
                    "top_k": 6,
                    "download_authorized": mode == "auth_mismatch",
                }
            )
            continue
        if mode == "timeout":
            time.sleep(10)
            return 0
        if mode == "exit":
            return 7
        if mode == "malformed":
            print("not-json", flush=True)
            return 0
        if mode == "reject":
            emit(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "op": op,
                    "status": "error",
                    "error": "synthetic rejection",
                }
            )
            return 1
        if op == "prepare_base":
            emit(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "op": op,
                    "status": "ok",
                    "layer": request["layer"],
                    "artifacts": [artifact("non_expert", None, "base")],
                }
            )
        elif op == "prepare_routed":
            artifacts = [
                artifact("routed_expert", expert_id, f"expert.{index}")
                for index, expert_id in enumerate(request["expert_ids"])
            ]
            emit(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "op": op,
                    "status": "ok",
                    "layer": request["layer"],
                    "expert_ids": request["expert_ids"],
                    "artifacts": artifacts,
                    "observation": {
                        "disk_bytes": 0,
                        "host_to_device_bytes": 0,
                        "expert_cache_hits": 6,
                        "expert_cache_misses": 0,
                        "miss_stall_ns": 0,
                    },
                }
            )
        elif op in {"release_layer", "abort_layer"}:
            emit(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "op": op,
                    "status": "ok",
                    "layer": request["layer"],
                    "final_artifacts": [],
                }
            )
        elif op == "shutdown":
            return 0
        else:
            raise RuntimeError(f"unexpected op {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
