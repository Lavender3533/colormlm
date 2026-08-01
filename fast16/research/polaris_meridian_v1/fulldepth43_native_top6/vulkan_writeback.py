"""FullDepth43 到持久 Vulkan worker 的严格单层回写协议。"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROTOCOL = "polaris-fulldepth43-vulkan-writeback-v1"
OUTPUT_FILE = "vulkan_moe_branch.bf16le.bin"


class VulkanWritebackError(RuntimeError):
    """Worker 或回写合同不满足；不允许继续提交当前 token。"""


def _strict_json(line: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise VulkanWritebackError(f"worker JSON 含重复 key: {key}")
            value[key] = item
        return value

    def invalid_constant(value: str) -> None:
        raise VulkanWritebackError(f"worker JSON 含非有限常量: {value}")

    try:
        document = json.loads(
            line,
            object_pairs_hook=unique,
            parse_constant=invalid_constant,
        )
    except VulkanWritebackError:
        raise
    except json.JSONDecodeError as exc:
        raise VulkanWritebackError(f"worker 返回非法 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise VulkanWritebackError("worker JSON 顶层必须是对象")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PersistentVulkanWriteback:
    """一次建立 Vulkan device/pipeline，以 UTF-8 JSONL 处理单层请求。"""

    def __init__(self, command: Sequence[str], *, timeout_seconds: float = 30.0) -> None:
        if not command or timeout_seconds <= 0:
            raise VulkanWritebackError("worker command 和 timeout 必须有效")
        self.command = tuple(str(value) for value in command)
        self.timeout_seconds = float(timeout_seconds)
        self.process: subprocess.Popen[str] | None = None
        self.poisoned = False
        self.counter = 0
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=64)
        self._threads: list[threading.Thread] = []
        self.hello: dict[str, Any] | None = None
        self._last_position: int | None = None
        self._last_layer: int | None = None
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
            raise VulkanWritebackError(f"无法启动 Vulkan worker: {exc}") from exc
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
            (read_stdout, "polaris-vulkan-stdout"),
            (read_stderr, "polaris-vulkan-stderr"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        hello = _strict_json(self._readline())
        if (
            hello.get("protocol") != PROTOCOL
            or hello.get("op") != "hello"
            or hello.get("ready") is not True
            or hello.get("persistent_context") is not True
            or hello.get("official_boundary_graph") is not True
        ):
            self._fail("Vulkan worker hello 合同漂移")
        self.hello = hello

    def _readline(self) -> str:
        try:
            line = self._stdout.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._fail("Vulkan worker 响应超时")
            raise AssertionError from exc
        if line is None:
            code = None if self.process is None else self.process.poll()
            stderr = " | ".join(self._stderr)
            self._fail(f"Vulkan worker 提前退出 code={code}: {stderr}")
        assert line is not None
        if len(line.encode("utf-8")) > 65_536:
            self._fail("Vulkan worker 响应超过 64 KiB")
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
        raise VulkanWritebackError(message)

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

    def execute(self, manifest_path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.poisoned or self.process is None or self.process.poll() is not None:
            raise VulkanWritebackError("Vulkan worker 已 poisoned/退出")
        manifest_path = manifest_path.resolve(strict=True)
        if manifest_path.name != "bridge_manifest.json":
            self._fail("Vulkan manifest 文件名漂移")
        capture_root = manifest_path.parent.resolve(strict=True)
        expected_manifest_sha = _sha256(manifest_path)
        manifest = _strict_json(manifest_path.read_text(encoding="utf-8", errors="strict"))
        manifest_position = manifest.get("position")
        manifest_layer = manifest.get("layer")
        manifest_input_token_id = manifest.get("input_token_id")
        if (
            isinstance(manifest_position, bool)
            or not isinstance(manifest_position, int)
            or manifest_position < 0
            or isinstance(manifest_layer, bool)
            or not isinstance(manifest_layer, int)
            or not 0 <= manifest_layer <= 42
            or isinstance(manifest_input_token_id, bool)
            or not isinstance(manifest_input_token_id, int)
            or manifest_input_token_id < 0
        ):
            self._fail("Vulkan manifest position/layer/token 合同漂移")
        if self._last_position is not None and self._last_layer is not None:
            expected_position = self._last_position + (1 if self._last_layer == 42 else 0)
            expected_layer = 0 if self._last_layer == 42 else self._last_layer + 1
            if (manifest_position, manifest_layer) != (expected_position, expected_layer):
                self._fail(
                    "Vulkan manifest 请求序列漂移: "
                    f"expected position={expected_position}/layer={expected_layer}, "
                    f"got position={manifest_position}/layer={manifest_layer}"
                )
        self.counter += 1
        request_id = f"py-{os.getpid()}-{self.counter}"
        request = {
            "protocol": PROTOCOL,
            "op": "execute_single_layer",
            "request_id": request_id,
            "manifest": str(manifest_path),
        }
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(
                json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._fail(f"写入 Vulkan worker 失败: {exc}")

        response = _strict_json(self._readline())
        if response.get("ok") is not True:
            self._fail(f"Vulkan worker 拒绝请求: {response.get('error')}")
        if (
            response.get("protocol") != PROTOCOL
            or response.get("request_id") != request_id
            or response.get("manifest_sha256") != expected_manifest_sha
            or response.get("position") != manifest_position
            or response.get("layer") != manifest_layer
            or response.get("input_token_id") != manifest_input_token_id
            or response.get("expansion_status") != "single_real_layer_writeback_only"
        ):
            self._fail("Vulkan worker response 身份/SHA 漂移")
        output = response.get("output")
        if not isinstance(output, Mapping) or (
            output.get("dtype") != "bf16_le"
            or output.get("shape") != [1, 1, 4096]
            or output.get("bytes") != 8192
        ):
            self._fail("Vulkan worker output dtype/shape/bytes 漂移")
        raw_path = output.get("path")
        if not isinstance(raw_path, str):
            self._fail("Vulkan worker output 缺少路径")
        output_path = Path(raw_path).resolve(strict=True)
        try:
            same_parent = os.path.samefile(output_path.parent, capture_root)
        except OSError as exc:
            self._fail(f"Vulkan output 边界检查失败: {exc}")
        if output_path.name != OUTPUT_FILE or not same_parent:
            self._fail("Vulkan output 越出 capture 边界")
        expected_output_sha = output.get("sha256")
        if not isinstance(expected_output_sha, str) or _sha256(output_path) != expected_output_sha:
            self._fail("Vulkan output SHA-256 漂移")
        payload = bytearray(output_path.read_bytes())
        if len(payload) != 8192:
            self._fail("Vulkan output 读取后字节数漂移")
        tensor = torch.frombuffer(payload, dtype=torch.bfloat16).clone().reshape(1, 1, 4096)
        if not bool(torch.isfinite(tensor).all().item()):
            self._fail("Vulkan output 含非有限值")
        evidence = {
            "protocol": PROTOCOL,
            "device": response.get("device"),
            "manifest_sha256": expected_manifest_sha,
            "output_sha256": expected_output_sha,
            "gpu_kernel_ms": response.get("gpu_kernel_ms"),
            "worker_wall_ms": response.get("wall_ms"),
            "payload_cache": response.get("payload_cache"),
            "gpu_payload_cache": response.get("gpu_payload_cache"),
            "reusable_gpu_slot": response.get("reusable_gpu_slot"),
            "boundaries": response.get("boundaries"),
            "persistent_context": True,
        }
        self._last_position = manifest_position
        self._last_layer = manifest_layer
        return tensor, evidence

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

    def __enter__(self) -> "PersistentVulkanWriteback":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def verify_exact_bf16_writeback(
    cpu_branch: torch.Tensor,
    vulkan_branch: torch.Tensor,
) -> dict[str, Any]:
    if tuple(cpu_branch.shape) != (1, 1, 4096) or cpu_branch.dtype != torch.bfloat16:
        raise VulkanWritebackError("CPU MoE reference 必须是 BF16 [1,1,4096]")
    if tuple(vulkan_branch.shape) != tuple(cpu_branch.shape) or vulkan_branch.dtype != torch.bfloat16:
        raise VulkanWritebackError("Vulkan MoE branch dtype/shape 与 CPU 不一致")
    difference = vulkan_branch.float() - cpu_branch.float()
    max_abs = float(difference.abs().max().item())
    rmse = float(torch.sqrt(difference.square().mean()).item())
    exact = bool(torch.equal(cpu_branch, vulkan_branch))
    if not exact:
        raise VulkanWritebackError(
            f"Vulkan/CPU BF16 不能逐位对齐: max_abs={max_abs:.9g}, rmse={rmse:.9g}"
        )
    return {
        "exact_bf16_equal": True,
        "element_count": cpu_branch.numel(),
        "max_abs": max_abs,
        "rmse": rmse,
    }
