"""L42 ``wq_a`` packed-FP8 持久 worker 的严格共享 arena 客户端。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import struct
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROTOCOL = "polaris-fulldepth43-packed-fp8-projection-v1"
WORKER_ARG = "--l42-wq-a-fp8-projection-worker"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
PROFILE = "fulldepth43_native_top6"
LAYER = 42
POSITION = 0
PROJECTION = "layers.42.attn.wq_a"
PROJECTION_N = 1024
PROJECTION_K = 4096
ACTIVATION_CONTRACT = "cpu_e4m3fn_quant_dequant_f32"
OUTPUT_ROUNDING = "bf16_rne_then_f32_le"
NUMERIC_MODE = "packed_fp8_e4m3_ue8m0_exact_audit"
ARENA_TRANSPORT = "shared_binary_file"
INPUT_DTYPE = "f32_le"
INPUT_SHAPE = (1, 1, 4096)
INPUT_BYTES = 4096 * 4
OUTPUT_DTYPE = "f32_le_bf16_rounded"
OUTPUT_SHAPE = (1, 1, 1024)
OUTPUT_BYTES = 1024 * 4
ARENA_MAX_BYTES = 64 * 1024 * 1024
WEIGHT_SHA256 = "1efcea39938dfadc143c41813bc32327a9bb5369b2b612feac76d9dfb8001ce7"
SCALE_SHA256 = "dfb4085717aa527f8affa5a1640c5f806867c5ba6e0301d170f387be8b6660cf"
CATALOG_SHA256 = "ca619984d4a46ad1a3701d2b4035766ea40c3a3dbedd3a474ce1df7aad4d0049"
INPUT_SHA256 = "47156935b19ca5483f0e92d2284eaa6a9417686978dc4b41ca893ee162f37577"
OUTPUT_SHA256 = "76469fd163f5db49de956eff9b29087afa4caa97d566be80bab9d9119facb0b8"
STATIC_UPLOADED_BYTES = 4_194_304 + 256
_OUTPUT_POISON = 0xA5
_MAX_JSON_BYTES = 65_536


class PackedFp8ProjectionError(RuntimeError):
    """任何 arena/worker 合同漂移都必须终止当前 worker。"""


def _strict_json(line: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise PackedFp8ProjectionError(f"worker JSON 含重复 key: {key}")
            value[key] = item
        return value

    def invalid_constant(value: str) -> None:
        raise PackedFp8ProjectionError(f"worker JSON 含非有限常量: {value}")

    try:
        document = json.loads(
            line,
            object_pairs_hook=unique,
            parse_constant=invalid_constant,
        )
    except PackedFp8ProjectionError:
        raise
    except json.JSONDecodeError as exc:
        raise PackedFp8ProjectionError(f"worker 返回非法 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise PackedFp8ProjectionError("worker JSON 顶层必须是对象")
    return document


def _require_exact_keys(document: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(document)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise PackedFp8ProjectionError(
            f"{label} key 合同漂移: missing={missing}, extra={extra}"
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_plain_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class PackedFp8Arena:
    """L42 投影的固定形状共享二进制 arena。"""

    def __init__(
        self,
        path: Path,
        *,
        input_offset: int = 0,
        output_offset: int = INPUT_BYTES,
        create: bool = False,
        arena_bytes: int | None = None,
    ) -> None:
        raw_path = Path(path)
        if raw_path.suffix.lower() != ".bin":
            raise PackedFp8ProjectionError("arena 必须使用 .bin 文件")
        for value, label in (
            (input_offset, "input_offset"),
            (output_offset, "output_offset"),
        ):
            if not _is_plain_nonnegative_int(value) or value % 4 != 0:
                raise PackedFp8ProjectionError(f"{label} 必须是非负 4-byte 对齐整数")
        input_end = input_offset + INPUT_BYTES
        output_end = output_offset + OUTPUT_BYTES
        if input_offset < output_end and output_offset < input_end:
            raise PackedFp8ProjectionError("arena input/output 区间不得重叠")
        minimum_bytes = max(input_end, output_end)
        requested_bytes = minimum_bytes if arena_bytes is None else arena_bytes
        if (
            not _is_plain_nonnegative_int(requested_bytes)
            or requested_bytes < minimum_bytes
            or requested_bytes > ARENA_MAX_BYTES
        ):
            raise PackedFp8ProjectionError("arena 大小超出固定视图或 64 MiB 上限")

        if create:
            parent = raw_path.parent.resolve(strict=True)
            candidate = parent / raw_path.name
            try:
                with candidate.open("xb") as stream:
                    stream.truncate(requested_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise PackedFp8ProjectionError(f"无法创建 arena: {exc}") from exc
        try:
            resolved = raw_path.resolve(strict=True)
            stat = resolved.stat()
        except OSError as exc:
            raise PackedFp8ProjectionError(f"无法打开 arena: {exc}") from exc
        if not resolved.is_file() or resolved.suffix.lower() != ".bin":
            raise PackedFp8ProjectionError("arena 必须是可解析的普通 .bin 文件")
        if stat.st_size < minimum_bytes or stat.st_size > ARENA_MAX_BYTES:
            raise PackedFp8ProjectionError("arena 文件大小越界")
        if arena_bytes is not None and stat.st_size != requested_bytes:
            raise PackedFp8ProjectionError("arena 文件大小与声明不一致")

        self.path = resolved
        self.input_offset = input_offset
        self.output_offset = output_offset
        self.arena_bytes = stat.st_size

    @property
    def input_view(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "offset": self.input_offset,
            "bytes": INPUT_BYTES,
            "dtype": INPUT_DTYPE,
            "shape": list(INPUT_SHAPE),
        }

    @property
    def output_view(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "offset": self.output_offset,
            "bytes": OUTPUT_BYTES,
            "dtype": OUTPUT_DTYPE,
            "shape": list(OUTPUT_SHAPE),
        }

    def _read_range(self, offset: int, byte_count: int) -> bytes:
        try:
            with self.path.open("rb") as stream:
                stream.seek(offset)
                payload = stream.read(byte_count)
        except OSError as exc:
            raise PackedFp8ProjectionError(f"读取 arena 失败: {exc}") from exc
        if len(payload) != byte_count:
            raise PackedFp8ProjectionError("arena 短读")
        return payload

    def prepare(self, activation: torch.Tensor) -> None:
        if (
            tuple(activation.shape) != INPUT_SHAPE
            or activation.dtype != torch.float32
            or activation.device.type != "cpu"
            or not bool(torch.isfinite(activation).all().item())
        ):
            raise PackedFp8ProjectionError(
                "wq_a activation 必须是有限 CPU F32 [1,1,4096]"
            )
        values = activation.detach().contiguous().reshape(-1).tolist()
        payload = struct.pack(f"<{len(values)}f", *values)
        if len(payload) != INPUT_BYTES or _sha256_bytes(payload) != INPUT_SHA256:
            raise PackedFp8ProjectionError("wq_a frozen input SHA-256 漂移")
        poison = bytes([_OUTPUT_POISON]) * OUTPUT_BYTES
        if _sha256_bytes(poison) == OUTPUT_SHA256:
            raise PackedFp8ProjectionError("arena output poison 与预期输出冲突")
        try:
            with self.path.open("r+b", buffering=0) as stream:
                stream.seek(self.input_offset)
                stream.write(payload)
                stream.seek(self.output_offset)
                stream.write(poison)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise PackedFp8ProjectionError(f"写入 arena 失败: {exc}") from exc
        if _sha256_bytes(self._read_range(self.input_offset, INPUT_BYTES)) != INPUT_SHA256:
            raise PackedFp8ProjectionError("arena input 回读 SHA-256 漂移")
        if self._read_range(self.output_offset, OUTPUT_BYTES) != poison:
            raise PackedFp8ProjectionError("arena output poison 未完整覆盖")

    def read_verified_output(self) -> torch.Tensor:
        payload = self._read_range(self.output_offset, OUTPUT_BYTES)
        if _sha256_bytes(payload) != OUTPUT_SHA256:
            raise PackedFp8ProjectionError("wq_a BF16-rounded output SHA-256 漂移")
        values: list[float] = []
        for (bits,) in struct.iter_unpack("<I", payload):
            if bits & 0xFFFF:
                raise PackedFp8ProjectionError("wq_a output 未完整执行 BF16 舍入")
            value = struct.unpack("<f", bits.to_bytes(4, "little"))[0]
            if not math.isfinite(value):
                raise PackedFp8ProjectionError("wq_a output 含非有限值")
            values.append(value)
        if len(values) != OUTPUT_SHAPE[-1]:
            raise PackedFp8ProjectionError("wq_a output 元素数漂移")
        return torch.tensor(values, dtype=torch.float32).reshape(OUTPUT_SHAPE)


class PersistentPackedFp8Projection:
    """一次启动 worker，按 epoch 串行执行冻结 L42 ``wq_a`` 投影。"""

    _HELLO_KEYS = {
        "protocol",
        "op",
        "ready",
        "revision",
        "profile",
        "layer",
        "position",
        "projection",
        "arena_transport",
        "weight_resident",
        "weight_sha256",
        "scale_sha256",
        "catalog_sha256",
        "input_sha256",
        "output_sha256",
        "numeric_mode",
    }
    _RESPONSE_KEYS = {
        "protocol",
        "request_id",
        "ok",
        "revision",
        "profile",
        "layer",
        "position",
        "arena_epoch",
        "projection",
        "input",
        "output_written",
        "input_sha256",
        "output_sha256",
        "weight_sha256",
        "scale_sha256",
        "catalog_sha256",
        "weight_resident",
        "static_uploaded_bytes",
        "request_uploaded_bytes",
        "numeric_mode",
        "output_rounding",
    }

    def __init__(
        self,
        command: Sequence[str],
        arena: PackedFp8Arena,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not command or WORKER_ARG not in tuple(str(value) for value in command):
            raise PackedFp8ProjectionError(f"worker command 必须显式包含 {WORKER_ARG}")
        if not isinstance(arena, PackedFp8Arena) or timeout_seconds <= 0:
            raise PackedFp8ProjectionError("arena 和 timeout 必须有效")
        self.command = tuple(str(value) for value in command)
        self.arena = arena
        self.timeout_seconds = float(timeout_seconds)
        self.process: subprocess.Popen[str] | None = None
        self.poisoned = False
        self.counter = 0
        self.arena_epoch = 0
        self.hello: dict[str, Any] | None = None
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=64)
        self._threads: list[threading.Thread] = []
        self._start()

    @staticmethod
    def _projection_spec() -> dict[str, Any]:
        return {
            "name": PROJECTION,
            "n": PROJECTION_N,
            "k": PROJECTION_K,
            "activation_contract": ACTIVATION_CONTRACT,
            "output_rounding": OUTPUT_ROUNDING,
        }

    def _start(self) -> None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise PackedFp8ProjectionError(f"无法启动 packed-FP8 worker: {exc}") from exc
        self.process = process
        assert process.stdout is not None and process.stderr is not None

        def read_stdout() -> None:
            try:
                for line in process.stdout:
                    self._stdout.put(line)
            finally:
                self._stdout.put(None)

        def read_stderr() -> None:
            for line in process.stderr:
                self._stderr.append(line.rstrip("\r\n"))

        for target, name in (
            (read_stdout, "polaris-packed-fp8-stdout"),
            (read_stderr, "polaris-packed-fp8-stderr"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        try:
            hello = _strict_json(self._readline())
            self._validate_hello(hello)
        except PackedFp8ProjectionError as exc:
            self._fail(str(exc))
        self.hello = hello

    def _validate_hello(self, hello: Mapping[str, Any]) -> None:
        _require_exact_keys(hello, self._HELLO_KEYS, "packed-FP8 hello")
        if (
            hello.get("protocol") != PROTOCOL
            or hello.get("op") != "hello"
            or hello.get("ready") is not True
            or hello.get("revision") != REVISION
            or hello.get("profile") != PROFILE
            or hello.get("layer") != LAYER
            or hello.get("position") != POSITION
            or hello.get("projection") != self._projection_spec()
            or hello.get("arena_transport") != ARENA_TRANSPORT
            or hello.get("weight_resident") is not True
            or hello.get("weight_sha256") != WEIGHT_SHA256
            or hello.get("scale_sha256") != SCALE_SHA256
            or hello.get("catalog_sha256") != CATALOG_SHA256
            or hello.get("input_sha256") != INPUT_SHA256
            or hello.get("output_sha256") != OUTPUT_SHA256
            or hello.get("numeric_mode") != NUMERIC_MODE
        ):
            raise PackedFp8ProjectionError("packed-FP8 worker hello 身份合同漂移")

    def _readline(self) -> str:
        try:
            line = self._stdout.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._fail("packed-FP8 worker 响应超时")
            raise AssertionError from exc
        if line is None:
            code = None if self.process is None else self.process.poll()
            stderr = " | ".join(self._stderr)
            self._fail(f"packed-FP8 worker 提前退出 code={code}: {stderr}")
        assert line is not None
        if len(line.encode("utf-8")) > _MAX_JSON_BYTES:
            self._fail("packed-FP8 worker 响应超过 64 KiB")
        return line

    def _fail(self, message: str) -> None:
        self.poisoned = True
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if process is not None:
            self._close_streams(process)
        raise PackedFp8ProjectionError(message)

    def _close_streams(self, process: subprocess.Popen[str]) -> None:
        for thread in self._threads:
            thread.join(timeout=1)
        self._threads.clear()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass

    def execute(self, activation: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.poisoned or self.process is None or self.process.poll() is not None:
            raise PackedFp8ProjectionError("packed-FP8 worker 已 poisoned/退出")
        try:
            self.arena.prepare(activation)
        except PackedFp8ProjectionError as exc:
            self._fail(str(exc))
        self.counter += 1
        request_id = f"py-{os.getpid()}-{self.counter}"
        request = {
            "protocol": PROTOCOL,
            "op": "execute_fp8_projection",
            "request_id": request_id,
            "revision": REVISION,
            "profile": PROFILE,
            "layer": LAYER,
            "position": POSITION,
            "arena_epoch": self.arena_epoch,
            "input_sha256": INPUT_SHA256,
            "projection": self._projection_spec(),
            "input": self.arena.input_view,
            "output": self.arena.output_view,
        }
        assert self.process.stdin is not None
        try:
            line = json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
            if len(line.encode("utf-8")) > _MAX_JSON_BYTES:
                self._fail("packed-FP8 request 超过 64 KiB")
            self.process.stdin.write(line)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._fail(f"写入 packed-FP8 worker 失败: {exc}")

        try:
            response = _strict_json(self._readline())
            if response.get("ok") is not True:
                raise PackedFp8ProjectionError(
                    f"packed-FP8 worker 拒绝请求: {response.get('error')}"
                )
            _require_exact_keys(response, self._RESPONSE_KEYS, "packed-FP8 response")
            if (
                response.get("protocol") != PROTOCOL
                or response.get("request_id") != request_id
                or response.get("revision") != REVISION
                or response.get("profile") != PROFILE
                or response.get("layer") != LAYER
                or response.get("position") != POSITION
                or response.get("arena_epoch") != self.arena_epoch
                or response.get("projection") != PROJECTION
                or response.get("input") != self.arena.input_view
                or response.get("output_written") != self.arena.output_view
                or response.get("input_sha256") != INPUT_SHA256
                or response.get("output_sha256") != OUTPUT_SHA256
                or response.get("weight_sha256") != WEIGHT_SHA256
                or response.get("scale_sha256") != SCALE_SHA256
                or response.get("catalog_sha256") != CATALOG_SHA256
                or response.get("weight_resident") is not True
                or response.get("static_uploaded_bytes") != STATIC_UPLOADED_BYTES
                or response.get("request_uploaded_bytes") != INPUT_BYTES
                or response.get("numeric_mode") != NUMERIC_MODE
                or response.get("output_rounding") != OUTPUT_ROUNDING
            ):
                raise PackedFp8ProjectionError("packed-FP8 response 身份/SHA 合同漂移")
            output = self.arena.read_verified_output()
        except PackedFp8ProjectionError as exc:
            self._fail(str(exc))
        evidence = {
            "protocol": PROTOCOL,
            "request_id": request_id,
            "arena_epoch": self.arena_epoch,
            "arena": str(self.arena.path),
            "projection": PROJECTION,
            "input_sha256": INPUT_SHA256,
            "output_sha256": OUTPUT_SHA256,
            "weight_sha256": WEIGHT_SHA256,
            "scale_sha256": SCALE_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "weight_resident": True,
            "static_uploaded_bytes": STATIC_UPLOADED_BYTES,
            "request_uploaded_bytes": INPUT_BYTES,
            "numeric_mode": NUMERIC_MODE,
        }
        self.arena_epoch += 1
        return output, evidence

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        self._close_streams(process)

    def __enter__(self) -> "PersistentPackedFp8Projection":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "PackedFp8Arena",
    "PackedFp8ProjectionError",
    "PersistentPackedFp8Projection",
    "PROTOCOL",
    "WORKER_ARG",
]
