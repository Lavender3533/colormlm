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

from fast16.research.polaris_meridian_v1.s14_range_pack import online_range


PROTOCOL = "polaris-fulldepth43-vulkan-writeback-v1"
OUTPUT_FILE = "vulkan_moe_branch.bf16le.bin"
PAYLOAD_IDENTITY_CONTRACT = (
    "sha256(v1_nul || sorted(length_le64(tensor),tensor,bytes_le64,expected_sha256_ascii))"
)
PAYLOAD_VERIFICATION_SCOPE = "all_listed_payloads_before_corresponding_gpu_compute"
PAYLOAD_VERIFICATION_KEYS = (
    "verification_owner",
    "verified_count",
    "verified_bytes",
    "payload_identity_sha256",
    "payload_identity_contract",
    "verified_before_compute",
    "verification_scope",
)
BATCH_PAYLOAD_VERIFICATION_KEYS = {
    "enabled",
    "batch_entries",
    "batch_hits",
    "batch_misses",
    "batch_disk_bytes_read",
    "concurrency_limit",
    "followup_cached_loader_hits",
    "all_verified_before_compute",
}
BATCH_PAYLOAD_VERIFICATION_CONCURRENCY = 8


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_payload_verification(payloads: object) -> dict[str, object]:
    if not isinstance(payloads, list) or not payloads:
        raise VulkanWritebackError("Vulkan manifest payload verification 不能为空")
    identities: list[tuple[str, int, str]] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise VulkanWritebackError("Vulkan manifest payload 身份必须是对象")
        tensor = payload.get("tensor")
        byte_count = payload.get("bytes")
        sha256 = payload.get("sha256")
        if (
            not isinstance(tensor, str)
            or not tensor
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not _is_sha256(sha256)
        ):
            raise VulkanWritebackError("Vulkan manifest payload 验证身份非法")
        assert isinstance(sha256, str)
        identities.append((tensor, byte_count, sha256))
    identities.sort(key=lambda item: item[0])
    names = [item[0] for item in identities]
    if len(names) != len(set(names)):
        raise VulkanWritebackError("Vulkan manifest payload 含重复 tensor")
    digest = hashlib.sha256()
    digest.update(b"polaris-rust-vulkan-payload-identity-v1\0")
    verified_bytes = 0
    for tensor, byte_count, sha256 in identities:
        encoded = tensor.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
        digest.update(byte_count.to_bytes(8, "little", signed=False))
        digest.update(sha256.encode("ascii"))
        verified_bytes += byte_count
    return {
        "verification_owner": "rust_vulkan_worker",
        "verified_count": len(identities),
        "verified_bytes": verified_bytes,
        "payload_identity_sha256": digest.hexdigest(),
        "payload_identity_contract": PAYLOAD_IDENTITY_CONTRACT,
        "verified_before_compute": True,
        "verification_scope": PAYLOAD_VERIFICATION_SCOPE,
    }


def _require_payload_verification(
    response: Mapping[str, Any], payloads: object
) -> dict[str, object]:
    expected = _expected_payload_verification(payloads)
    observed = {key: response.get(key) for key in PAYLOAD_VERIFICATION_KEYS}
    if observed != expected:
        raise VulkanWritebackError("Vulkan worker GPU计算前 payload 验证回执漂移")
    assert isinstance(payloads, list)
    identities = [
        (payload["tensor"], payload["bytes"], payload["sha256"])
        for payload in payloads
    ]
    return {
        **observed,
        "python_expected_payload_identity_sha256": expected[
            "payload_identity_sha256"
        ],
        "python_deferred_identity_multiset_contract": (
            online_range.DEFERRED_IDENTITY_MULTISET_CONTRACT
        ),
        "python_deferred_identity_multiset_sum_u256": (
            online_range.deferred_identity_multiset_sum_u256(identities)
        ),
    }


def _require_batch_payload_verification(
    response: Mapping[str, Any],
    *,
    enabled: bool,
    payload_count: object,
) -> dict[str, object]:
    receipt = response.get("batch_payload_verification")
    if not isinstance(receipt, Mapping) or set(receipt) != BATCH_PAYLOAD_VERIFICATION_KEYS:
        raise VulkanWritebackError("Vulkan worker batch payload 验证回执结构漂移")
    if isinstance(payload_count, bool) or not isinstance(payload_count, int) or payload_count <= 0:
        raise VulkanWritebackError("Vulkan manifest payload_count 非法")

    integer_fields = (
        "batch_entries",
        "batch_hits",
        "batch_misses",
        "batch_disk_bytes_read",
        "concurrency_limit",
        "followup_cached_loader_hits",
    )
    if any(
        isinstance(receipt.get(key), bool)
        or not isinstance(receipt.get(key), int)
        or int(receipt[key]) < 0
        for key in integer_fields
    ):
        raise VulkanWritebackError("Vulkan worker batch payload 验证回执整数账本漂移")
    if receipt.get("enabled") is not enabled:
        raise VulkanWritebackError("Vulkan worker batch payload 验证开关漂移")
    if receipt.get("concurrency_limit") != BATCH_PAYLOAD_VERIFICATION_CONCURRENCY:
        raise VulkanWritebackError("Vulkan worker batch payload 验证并发上限漂移")

    if enabled:
        if (
            receipt.get("batch_entries") != payload_count
            or int(receipt["batch_hits"]) + int(receipt["batch_misses"]) != payload_count
            or receipt.get("followup_cached_loader_hits") != payload_count
            or receipt.get("all_verified_before_compute") is not True
            or (
                int(receipt["batch_misses"]) == 0
                and int(receipt["batch_disk_bytes_read"]) != 0
            )
            or (
                int(receipt["batch_misses"]) > 0
                and int(receipt["batch_disk_bytes_read"]) <= 0
            )
        ):
            raise VulkanWritebackError("Vulkan worker batch payload 验证账本未闭合")
    elif (
        any(int(receipt[key]) != 0 for key in integer_fields if key != "concurrency_limit")
        or receipt.get("all_verified_before_compute") is not False
    ):
        raise VulkanWritebackError("Vulkan worker 关闭 batch payload 验证时仍声明工作量")
    return dict(receipt)


class PersistentVulkanWriteback:
    """一次建立 Vulkan device/pipeline，以 UTF-8 JSONL 处理单层请求。"""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
        batch_verify_payloads: bool = False,
    ) -> None:
        if not command or timeout_seconds <= 0:
            raise VulkanWritebackError("worker command 和 timeout 必须有效")
        self.command = tuple(str(value) for value in command)
        self.timeout_seconds = float(timeout_seconds)
        self.batch_verify_payloads = bool(batch_verify_payloads)
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
            or hello.get("inline_manifest_json") is not True
            or hello.get("batch_payload_verification") is not True
            or hello.get("batch_payload_verification_concurrency_limit")
            != BATCH_PAYLOAD_VERIFICATION_CONCURRENCY
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

    def execute(
        self,
        manifest_source: Path | Mapping[str, Any],
        *,
        capture_root: Path | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.poisoned or self.process is None or self.process.poll() is not None:
            raise VulkanWritebackError("Vulkan worker 已 poisoned/退出")
        if isinstance(manifest_source, Mapping):
            if capture_root is None:
                self._fail("inline Vulkan manifest 缺少 capture_root")
            capture_root = capture_root.resolve(strict=True)
            if not capture_root.is_dir():
                self._fail("inline Vulkan capture_root 不是目录")
            try:
                manifest_json = json.dumps(
                    manifest_source,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                self._fail(f"inline Vulkan manifest 无法规范序列化: {exc}")
            manifest_bytes = manifest_json.encode("utf-8", errors="strict")
            expected_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            manifest = _strict_json(manifest_json)
            manifest_transport = "inline_json"
            request_manifest_fields: dict[str, Any] = {
                "capture_root": str(capture_root),
                "manifest_json": manifest_json,
                "manifest_sha256": expected_manifest_sha,
            }
            request_op = "execute_single_layer_inline_manifest"
        else:
            if capture_root is not None:
                self._fail("文件 Vulkan manifest 禁止额外 capture_root")
            manifest_path = Path(manifest_source).resolve(strict=True)
            if manifest_path.name != "bridge_manifest.json":
                self._fail("Vulkan manifest 文件名漂移")
            capture_root = manifest_path.parent.resolve(strict=True)
            expected_manifest_sha = _sha256(manifest_path)
            manifest = _strict_json(
                manifest_path.read_text(encoding="utf-8", errors="strict")
            )
            manifest_transport = "capture_file"
            request_manifest_fields = {"manifest": str(manifest_path)}
            request_op = "execute_single_layer"
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
            "op": request_op,
            "request_id": request_id,
            "batch_verify_payloads": self.batch_verify_payloads,
            **request_manifest_fields,
        }
        request_line = json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(request_line.encode("utf-8", errors="strict")) > 65_536:
            self._fail("Vulkan worker 请求超过 64 KiB")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(request_line + "\n")
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
            or response.get("manifest_transport") != manifest_transport
            or response.get("position") != manifest_position
            or response.get("layer") != manifest_layer
            or response.get("input_token_id") != manifest_input_token_id
            or response.get("expansion_status") != "single_real_layer_writeback_only"
        ):
            self._fail("Vulkan worker response 身份/SHA 漂移")
        try:
            payload_verification = _require_payload_verification(
                response,
                manifest.get("payloads"),
            )
        except VulkanWritebackError as error:
            self._fail(str(error))
        try:
            batch_payload_verification = _require_batch_payload_verification(
                response,
                enabled=self.batch_verify_payloads,
                payload_count=manifest.get("payload_count"),
            )
        except VulkanWritebackError as error:
            self._fail(str(error))
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
            "manifest_transport": manifest_transport,
            "output_sha256": expected_output_sha,
            "gpu_kernel_ms": response.get("gpu_kernel_ms"),
            "worker_wall_ms": response.get("wall_ms"),
            "payload_cache": response.get("payload_cache"),
            "gpu_payload_cache": response.get("gpu_payload_cache"),
            "shared_gpu_payload_cache": response.get("shared_gpu_payload_cache"),
            "reusable_gpu_slot": response.get("reusable_gpu_slot"),
            "payload_verification": payload_verification,
            "batch_payload_verification": batch_payload_verification,
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
