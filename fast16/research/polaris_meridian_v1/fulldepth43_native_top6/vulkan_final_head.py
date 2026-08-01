"""FullDepth43 到持久 Vulkan BF16 全词表头的严格协议。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROTOCOL = "polaris-s14-bf16-head-worker-v1"
VOCAB_SIZE = 129_280
HIDDEN_SIZE = 4_096
ALLOWED_BATCHES = (1, 4, 8)


class VulkanFinalHeadError(RuntimeError):
    """Worker/hidden/返回值不满足合同，当前 token 不得提交。"""


def _strict_json(line: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise VulkanFinalHeadError(f"head worker JSON 含重复 key: {key}")
            value[key] = item
        return value

    def invalid_constant(value: str) -> None:
        raise VulkanFinalHeadError(f"head worker JSON 含非有限常量: {value}")

    try:
        document = json.loads(
            line,
            object_pairs_hook=unique,
            parse_constant=invalid_constant,
        )
    except VulkanFinalHeadError:
        raise
    except json.JSONDecodeError as exc:
        raise VulkanFinalHeadError(f"head worker 返回非法 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise VulkanFinalHeadError("head worker JSON 顶层必须是 object")
    return document


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class PersistentVulkanFinalHead:
    """head 权重在 VRAM 常驻，UTF-8 JSONL 只传输控制和证据。"""

    def __init__(self, command: Sequence[str], *, timeout_seconds: float = 30.0) -> None:
        if not command or timeout_seconds <= 0:
            raise VulkanFinalHeadError("head worker command/timeout 必须有效")
        self.command = tuple(str(value) for value in command)
        self.timeout_seconds = float(timeout_seconds)
        self.process: subprocess.Popen[str] | None = None
        self.poisoned = False
        self.counter = 0
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=64)
        self._threads: list[threading.Thread] = []
        self.hello: dict[str, Any] | None = None
        self._start()

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
            raise VulkanFinalHeadError(f"无法启动 Vulkan final-head worker: {exc}") from exc
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
            (read_stdout, "polaris-head-stdout"),
            (read_stderr, "polaris-head-stderr"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

        hello = _strict_json(self._readline())
        head_sha = hello.get("head_sha256")
        if (
            hello.get("protocol") != PROTOCOL
            or hello.get("status") != "ready"
            or hello.get("head_bytes") != VOCAB_SIZE * HIDDEN_SIZE * 2
            or not isinstance(head_sha, str)
            or len(head_sha) != 64
            or not _finite_number(hello.get("upload_wall_ms"))
        ):
            self._fail("Vulkan final-head hello 合同漂移")
        self.hello = hello

    def _readline(self) -> str:
        try:
            line = self._stdout.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._fail("Vulkan final-head worker 响应超时")
            raise AssertionError from exc
        if line is None:
            code = None if self.process is None else self.process.poll()
            stderr = " | ".join(self._stderr)
            self._fail(f"Vulkan final-head worker 提前退出 code={code}: {stderr}")
        assert line is not None
        if len(line.encode("utf-8")) > 65_536:
            self._fail("Vulkan final-head worker 响应超过 64 KiB")
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
        raise VulkanFinalHeadError(message)

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

    @staticmethod
    def _normalize_hidden(hidden: torch.Tensor) -> tuple[torch.Tensor, int]:
        if hidden.dtype != torch.float32 or hidden.device.type != "cpu":
            raise VulkanFinalHeadError("final-head hidden 必须是 CPU F32")
        if hidden.ndim < 2 or hidden.shape[-1] != HIDDEN_SIZE:
            raise VulkanFinalHeadError("final-head hidden 末维必须是 4096")
        batch = hidden.numel() // HIDDEN_SIZE
        if batch not in ALLOWED_BATCHES or hidden.numel() != batch * HIDDEN_SIZE:
            raise VulkanFinalHeadError("final-head hidden 只允许 K=1/4/8")
        contiguous = hidden.detach().contiguous().reshape(batch, HIDDEN_SIZE)
        if not bool(torch.isfinite(contiguous).all().item()):
            raise VulkanFinalHeadError("final-head hidden 含 NaN/Inf")
        return contiguous, batch

    def execute(
        self,
        hidden: torch.Tensor,
        scratch_dir: Path,
        *,
        diagnostics: bool = False,
    ) -> tuple[list[int], dict[str, Any]]:
        if self.poisoned or self.process is None or self.process.poll() is not None:
            raise VulkanFinalHeadError("Vulkan final-head worker 已 poisoned/退出")
        normalized, batch = self._normalize_hidden(hidden)
        scratch = scratch_dir.resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        self.counter += 1
        request_id = self.counter
        input_path = scratch / f"head-input-{os.getpid()}-{request_id}.f32le.bin"
        if input_path.exists():
            self._fail("final-head hidden 临时路径冲突")
        payload = normalized.numpy().tobytes(order="C")
        expected_bytes = batch * HIDDEN_SIZE * 4
        if len(payload) != expected_bytes:
            self._fail("final-head hidden 序列化字节漂移")
        input_sha = _sha256_bytes(payload)
        temporary = input_path.with_suffix(input_path.suffix + ".tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, input_path)
            request = {
                "protocol": PROTOCOL,
                "request_id": request_id,
                "input_path": str(input_path),
                "input_bytes": expected_bytes,
                "input_sha256": input_sha,
                "batch": batch,
                "diagnostics": bool(diagnostics),
            }
            assert self.process.stdin is not None
            try:
                self.process.stdin.write(
                    json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._fail(f"写入 Vulkan final-head worker 失败: {exc}")

            response = _strict_json(self._readline())
            if response.get("status") != "ok":
                self._fail(f"Vulkan final-head worker 拒绝请求: {response.get('error')}")
            hello = self.hello or {}
            if (
                response.get("protocol") != PROTOCOL
                or response.get("request_id") != request_id
                or response.get("batch") != batch
                or response.get("input_bytes") != expected_bytes
                or response.get("input_sha256") != input_sha
                or response.get("head_sha256") != hello.get("head_sha256")
            ):
                self._fail("Vulkan final-head response 身份/SHA 漂移")
            token_ids = response.get("argmax_token_ids")
            max_logits = response.get("max_logits")
            if (
                not isinstance(token_ids, list)
                or len(token_ids) != batch
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < VOCAB_SIZE
                    for value in token_ids
                )
                or not isinstance(max_logits, list)
                or len(max_logits) != batch
                or any(not _finite_number(value) for value in max_logits)
            ):
                self._fail("Vulkan final-head argmax 输出漂移")
            for key in (
                "input_ready_wall_ms",
                "kernel_wall_ms",
                "postprocess_wall_ms",
                "worker_wall_ms",
                "equivalent_head_tokens_per_second",
            ):
                if not _finite_number(response.get(key)) or float(response[key]) < 0:
                    self._fail(f"Vulkan final-head telemetry 漂移: {key}")
            if diagnostics:
                top10 = response.get("top10")
                summary = response.get("logits")
                if not isinstance(top10, list) or len(top10) != batch or not isinstance(summary, Mapping):
                    self._fail("Vulkan final-head 诊断证据缺失")
            elif response.get("top10") is not None or response.get("logits") is not None:
                self._fail("Vulkan final-head 生产路径意外返回全logits诊断")
            evidence = {
                "protocol": PROTOCOL,
                "head_sha256": hello.get("head_sha256"),
                "input_sha256": input_sha,
                "batch": batch,
                "max_logits": [float(value) for value in max_logits],
                "top10": response.get("top10"),
                "logits": response.get("logits"),
                "input_ready_wall_ms": float(response["input_ready_wall_ms"]),
                "gpu_head_argmax_ms": float(response["kernel_wall_ms"]),
                "postprocess_wall_ms": float(response["postprocess_wall_ms"]),
                "worker_wall_ms": float(response["worker_wall_ms"]),
                "equivalent_head_tokens_per_second": float(
                    response["equivalent_head_tokens_per_second"]
                ),
                "persistent_context": True,
                "diagnostics": bool(diagnostics),
            }
            return [int(value) for value in token_ids], evidence
        finally:
            for path in (temporary, input_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

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

    def __enter__(self) -> "PersistentVulkanFinalHead":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
