#!/usr/bin/env python3
"""Native executor bridge 的小尺寸离线 peer；不读取模型或生成 token。"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path


PROTOCOL = "polaris-s14-executor-jsonl-v1"
REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"


def emit(value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 1 << 20:
        raise RuntimeError("fixture response exceeded JSONL control-plane limit")
    print(encoded, flush=True)


def write_repeated(view: dict, pattern: bytes) -> None:
    path = Path(view["path"])
    offset = int(view["offset"])
    size = int(view["bytes"])
    if not path.is_absolute() or size <= 0 or size % len(pattern):
        raise RuntimeError("invalid binary tensor view")
    with path.open("r+b", buffering=0) as arena:
        arena.seek(offset)
        arena.write(pattern * (size // len(pattern)))
        arena.flush()


def write_hidden(state: dict, epoch: int) -> dict:
    hidden = state["hidden"]
    if hidden["dtype"] != "bf16_le":
        raise RuntimeError("fixture expected bf16_le hidden")
    # 有限 BF16，且每个 epoch 的位模式不同，便于 Rust 验证状态确实推进。
    bits = 0x3F80 + (epoch & 0x3F)
    write_repeated(hidden, struct.pack("<H", bits))
    return hidden


def write_logits(view: dict) -> dict:
    if view["dtype"] != "f32_le" or len(view["shape"]) != 1:
        raise RuntimeError("fixture expected one-dimensional f32_le logits")
    count = int(view["shape"][0])
    payload = b"".join(struct.pack("<f", float(index) / 16.0) for index in range(count))
    if len(payload) != int(view["bytes"]):
        raise RuntimeError("fixture logits bytes mismatch")
    path = Path(view["path"])
    with path.open("r+b", buffering=0) as arena:
        arena.seek(int(view["offset"]))
        arena.write(payload)
        arena.flush()
    return view


def main() -> int:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    epoch = 0
    hello_seen = False
    for line in sys.stdin:
        if len(line.encode("utf-8")) > 1 << 20:
            raise RuntimeError("fixture request exceeded JSONL limit")
        request = json.loads(line)
        request_id = request["request_id"]
        op = request["op"]
        if not hello_seen:
            if op != "hello" or request["protocol"] != PROTOCOL:
                raise RuntimeError("hello required")
            if request["mode"] != "fixture_test":
                emit(
                    {
                        "protocol": PROTOCOL,
                        "request_id": request_id,
                        "op": op,
                        "status": "error",
                        "error": "fixture refuses production mode",
                    }
                )
                return 2
            emit(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "op": op,
                    "status": "ok",
                    "repo": REPO,
                    "revision": REVISION,
                    "profile": "s14_top6",
                    "mode": request["mode"],
                    "hidden_shape": request["hidden_shape"],
                    "hidden_dtype": request["hidden_dtype"],
                    "logits_shape": request["logits_shape"],
                    "logits_dtype": request["logits_dtype"],
                }
            )
            hello_seen = True
            continue

        if mode == "timeout":
            time.sleep(10)
            return 0
        if mode == "exit":
            return 7
        if mode == "malformed":
            print("not-json", flush=True)
            return 0
        if mode == "oversize":
            print(
                json.dumps({"padding": "x" * ((1 << 20) + 1)}, separators=(",", ":")),
                flush=True,
            )
            return 0
        if mode == "reject":
            emit(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "op": op,
                    "status": "error",
                    "error": "synthetic executor rejection",
                }
            )
            return 3
        if op == "shutdown":
            return 0

        request_epoch = int(request["state_epoch"])
        if request_epoch != epoch:
            raise RuntimeError(f"epoch drift: expected {epoch}, actual {request_epoch}")
        position = int(request["position"])
        if op == "embed_row":
            epoch += 1
            descriptor = write_hidden(request["state"], epoch)
            if mode == "bad_descriptor":
                descriptor = {**descriptor, "offset": int(descriptor["offset"]) + 2}
            response = {
                "protocol": PROTOCOL,
                "request_id": request_id,
                "op": op,
                "status": "ok",
                "position": position + (1 if mode == "bad_position" else 0),
                "state_epoch": epoch,
                "state_view": request["state"]["arena"],
                "hidden_written": descriptor,
            }
            if mode == "tensor_json":
                response["hidden_values"] = []
            emit(response)
        elif op == "attention_then_route":
            epoch += 1
            layer = int(request["layer"])
            descriptor = write_hidden(request["state"], epoch)
            start = (layer * 17 + 5) % 256
            experts = [(start + index * 43) % 256 for index in range(6)]
            emit(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "op": op,
                    "status": "ok",
                    "position": position,
                    "layer": layer,
                    "state_epoch": epoch,
                    "state_view": request["state"]["arena"],
                    "hidden_written": descriptor,
                    "router_kind": "hash" if layer < 3 else "score",
                    "expert_ids": experts,
                    "route_weights": [0.25] * 6,
                }
            )
        elif op == "routed_and_shared_moe_then_hc_post":
            epoch += 1
            descriptor = write_hidden(request["state"], epoch)
            emit(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "op": op,
                    "status": "ok",
                    "position": position,
                    "layer": request["layer"],
                    "state_epoch": epoch,
                    "state_view": request["state"]["arena"],
                    "hidden_written": descriptor,
                }
            )
        elif op == "hc_head_norm_full_logits":
            descriptor = write_logits(request["logits_out"])
            emit(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "op": op,
                    "status": "ok",
                    "position": position,
                    "state_epoch": epoch,
                    "logits_written": descriptor,
                }
            )
        else:
            raise RuntimeError(f"unexpected op {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
