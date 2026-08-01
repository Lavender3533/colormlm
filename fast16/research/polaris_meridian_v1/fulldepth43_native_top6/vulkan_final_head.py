"""FullDepth43 到持久 Vulkan BF16 全词表头的严格协议。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from fast16.research.polaris_meridian_v1.local_s14_primitives.final_head import (
    bf16_checkpoint_head_logits,
    hc_head_reduce,
    official_rms_norm,
)
from fast16.research.polaris_meridian_v1.s14_first_real_token import executor as s14
from fast16.research.polaris_meridian_v1.s14_range_pack import online_range


PROTOCOL = "polaris-s14-bf16-head-worker-v2"
VOCAB_SIZE = 129_280
HIDDEN_SIZE = 4_096
ALLOWED_BATCHES = (1, 4, 8)
HEAD_BYTES = VOCAB_SIZE * HIDDEN_SIZE * 2
POSITION_CONTRACT = "first_any_then_strict_increment"
PRODUCTION_RESPONSE = "argmax_only_no_top10_or_logits"


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

    def __init__(
        self,
        command: Sequence[str],
        *,
        expected_head_sha256: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not command or timeout_seconds <= 0:
            raise VulkanFinalHeadError("head worker command/timeout 必须有效")
        if len(expected_head_sha256) != 64:
            raise VulkanFinalHeadError("expected head SHA-256 非法")
        self.command = tuple(str(value) for value in command)
        self.expected_head_sha256 = expected_head_sha256
        self.timeout_seconds = float(timeout_seconds)
        self.process: subprocess.Popen[str] | None = None
        self.poisoned = False
        self.counter = 0
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=64)
        self._threads: list[threading.Thread] = []
        self.hello: dict[str, Any] | None = None
        self.last_position: int | None = None
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
            or hello.get("head_bytes") != HEAD_BYTES
            or not isinstance(head_sha, str)
            or head_sha != self.expected_head_sha256
            or not _finite_number(hello.get("upload_wall_ms"))
            or hello.get("position_contract") != POSITION_CONTRACT
            or hello.get("production_response") != PRODUCTION_RESPONSE
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
        position: int,
        diagnostics: bool = False,
    ) -> tuple[list[int], dict[str, Any]]:
        if self.poisoned or self.process is None or self.process.poll() is not None:
            raise VulkanFinalHeadError("Vulkan final-head worker 已 poisoned/退出")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise VulkanFinalHeadError("Vulkan final-head position 非法")
        if self.last_position is not None and position != self.last_position + 1:
            raise VulkanFinalHeadError("Vulkan final-head position 必须严格递增 1")
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
                "position": position,
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
                or response.get("position") != position
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
                "position": position,
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
            self.last_position = position
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


class FinalHiddenNormalizer:
    """CPU 只执行原生 HC 收束和 RMSNorm，不 mmap/计算全词表 head。"""

    REQUIRED = {
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "norm.weight",
        "head.weight",
    }

    def __init__(
        self,
        final_ranges: Sequence[online_range.CachedRange],
        cache_root: Path,
    ) -> None:
        self.store = s14.TensorStore(cache_root)
        self.store.add_ranges(final_ranges)
        if set(self.store.sources) != self.REQUIRED:
            raise VulkanFinalHeadError(
                f"final tensor 集漂移: {sorted(self.store.sources)}"
            )
        self.helper = s14._InlineForward(self.store.bundle())
        head = self.store.source("head.weight")
        if head.entry.get("dtype") != "BF16" or head.entry.get("shape") != [
            VOCAB_SIZE,
            HIDDEN_SIZE,
        ]:
            raise VulkanFinalHeadError("真实 final head 必须是 BF16 [129280,4096]")
        self.head_path = head.path
        self.head_sha256 = str(head.proof["observed_sha256"])
        self.range_shas = {
            item.entry["tensor"]: item.proof["observed_sha256"]
            for item in final_ranges
        }

    def validate_ranges(
        self,
        final_ranges: Sequence[online_range.CachedRange],
    ) -> None:
        observed = {
            item.entry["tensor"]: item.proof["observed_sha256"]
            for item in final_ranges
        }
        if observed != self.range_shas:
            raise VulkanFinalHeadError("跨 token final Range/SHA 漂移")

    def normalize(self, state: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if (
            state.dtype != torch.bfloat16
            or state.device.type != "cpu"
            or tuple(state.shape) != (1, 1, s14.HC_MULT, s14.HIDDEN_SIZE)
        ):
            raise VulkanFinalHeadError(
                "final normalization 输入必须是 CPU BF16 [1,1,4,4096]"
            )
        reduced, pre = hc_head_reduce(
            state,
            self.helper._load_tensor("hc_head_fn"),
            self.helper._load_tensor("hc_head_scale"),
            self.helper._load_tensor("hc_head_base"),
        )
        normalized_bf16 = official_rms_norm(
            reduced,
            self.helper._load_tensor("norm.weight"),
        )
        normalized = normalized_bf16.float().reshape(1, HIDDEN_SIZE).contiguous()
        if not bool(torch.isfinite(normalized).all().item()):
            raise VulkanFinalHeadError("真实 normalized hidden 含 NaN/Inf")
        return normalized, {
            "hc_pre": [float(value) for value in pre.flatten().tolist()],
            "normalized": s14.NativeLayerReference._summary_tensor(normalized_bf16),
            "integrity": self.store.integrity(),
            "cpu_scope": "hc_reduce_and_rmsnorm_only",
        }


def cpu_reference_token_top10(
    normalized_hidden: torch.Tensor,
    head_path: Path,
    *,
    head_chunk_size: int = 4096,
) -> dict[str, Any]:
    """显式的一次性校验路径；生产 token 不调用这个 CPU 全词表投影。"""

    normalized, batch = PersistentVulkanFinalHead._normalize_hidden(normalized_hidden)
    if batch != 1:
        raise VulkanFinalHeadError("CPU token/top10 校验只允许单个真实 hidden")
    path = head_path.resolve()
    if not path.is_file() or path.stat().st_size != HEAD_BYTES:
        raise VulkanFinalHeadError("CPU 校验 head 路径/字节漂移")
    head = torch.from_file(
        str(path),
        shared=False,
        size=VOCAB_SIZE * HIDDEN_SIZE,
        dtype=torch.bfloat16,
    ).reshape(VOCAB_SIZE, HIDDEN_SIZE)
    started = time.perf_counter()
    logits = bf16_checkpoint_head_logits(
        normalized.reshape(1, 1, HIDDEN_SIZE),
        head,
        output_chunk_size=head_chunk_size,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    token_id = int(logits.argmax(dim=-1).item())
    top_values, top_ids = logits.topk(10, dim=-1)
    top10 = [
        {"token_id": int(index), "logit": float(value)}
        for index, value in zip(
            top_ids[0].tolist(),
            top_values[0].tolist(),
            strict=True,
        )
    ]
    return {
        "token_id": token_id,
        "top10": top10,
        "elapsed_ms": elapsed_ms,
        "mode": "explicit_one_time_cpu_validation_not_production",
    }


class FullDepthVulkanFinalHead:
    """FullDepth 生产末端：CPU normalization 后只走持久 GPU head+argmax。"""

    def __init__(
        self,
        final_ranges: Sequence[online_range.CachedRange],
        cache_root: Path,
        worker_command: Sequence[str],
        scratch_dir: Path,
        *,
        timeout_seconds: float,
        validate_cpu_once: bool = False,
        head_chunk_size: int = 4096,
    ) -> None:
        self.normalizer = FinalHiddenNormalizer(final_ranges, cache_root)
        if not worker_command:
            raise VulkanFinalHeadError("Vulkan final-head worker command 为空")
        self.worker = PersistentVulkanFinalHead(
            (
                *(str(value) for value in worker_command),
                "--worker",
                str(self.normalizer.head_path),
            ),
            expected_head_sha256=self.normalizer.head_sha256,
            timeout_seconds=timeout_seconds,
        )
        self.scratch_dir = scratch_dir.resolve()
        self.validate_cpu_once = bool(validate_cpu_once)
        self.cpu_validation_completed = False
        self.head_chunk_size = head_chunk_size

    @property
    def hello(self) -> Mapping[str, Any]:
        return self.worker.hello or {}

    def validate_ranges(
        self,
        final_ranges: Sequence[online_range.CachedRange],
    ) -> None:
        self.normalizer.validate_ranges(final_ranges)

    def forward(self, state: torch.Tensor, *, position: int) -> dict[str, Any]:
        normalized, normalization = self.normalizer.normalize(state)
        run_cpu_validation = self.validate_cpu_once and not self.cpu_validation_completed
        token_ids, gpu = self.worker.execute(
            normalized,
            self.scratch_dir,
            position=position,
            diagnostics=run_cpu_validation,
        )
        token_id = token_ids[0]
        cpu_validation = None
        if run_cpu_validation:
            cpu_validation = cpu_reference_token_top10(
                normalized,
                self.normalizer.head_path,
                head_chunk_size=self.head_chunk_size,
            )
            gpu_top10_rows = gpu.get("top10")
            if not isinstance(gpu_top10_rows, list) or len(gpu_top10_rows) != 1:
                raise VulkanFinalHeadError("一次性 CPU 校验缺少 GPU top10")
            gpu_top10 = gpu_top10_rows[0]
            cpu_top10 = cpu_validation["top10"]
            if token_id != cpu_validation["token_id"] or [
                item["token_id"] for item in gpu_top10
            ] != [item["token_id"] for item in cpu_top10]:
                raise VulkanFinalHeadError("真实 normalized hidden 的 CPU/GPU token/top10 不一致")
            max_error = max(
                abs(float(gpu_item["logit"]) - float(cpu_item["logit"]))
                for gpu_item, cpu_item in zip(gpu_top10, cpu_top10, strict=True)
            )
            if max_error > 2.0e-3:
                raise VulkanFinalHeadError(
                    f"真实 normalized hidden CPU/GPU top10 logit 误差过大: {max_error}"
                )
            cpu_validation = {
                **cpu_validation,
                "gpu_top10_max_abs_error": max_error,
                "passed": True,
                "position": position,
            }
            self.cpu_validation_completed = True
        return {
            "token_id": token_id,
            "position": position,
            "hc_pre": normalization["hc_pre"],
            "normalized": normalization["normalized"],
            "logits": gpu["logits"],
            "top10": gpu["top10"][0] if run_cpu_validation else None,
            "integrity": normalization["integrity"],
            "backend": "persistent_vulkan_bf16_head_device_argmax",
            "cpu_scope": normalization["cpu_scope"],
            "gpu": gpu,
            "cpu_validation": cpu_validation,
            "production_full_logits_returned": False,
        }

    def close(self) -> None:
        self.worker.close()
