"""持久 worker 的隔离 causal-block 单层 replay 客户端。

本模块只消费已经落盘的 ``bridge_manifest.json``，并严格验证 worker
返回的 K=4/K=8 BF16 单层结果。它是离线正确性 replay，不是速度候选。
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.vulkan_writeback import (
    PROTOCOL as WRITEBACK_PROTOCOL,
)


REQUEST_OP = "execute_causal_block_layer_replay"
RESPONSE_MODE = "causal_block_layer_replay"
OUTPUT_FILE = "vulkan_moe_block_branches.bf16le.bin"
BLOCK_SIZES = frozenset((4, 8))
MAX_JSONL_BYTES = 65_536
OUTPUT_SHAPE = (1, 1, 4096)
OUTPUT_BYTES = 8192


class CachedLayerReplayError(RuntimeError):
    """Replay 合同失配；当前持久 worker 不再允许复用。"""


def _strict_json(line: str, *, subject: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CachedLayerReplayError(f"{subject} JSON 含重复 key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise CachedLayerReplayError(f"{subject} JSON 含非有限常量: {value}")

    try:
        document = json.loads(
            line,
            object_pairs_hook=unique,
            parse_constant=invalid_constant,
        )
    except CachedLayerReplayError:
        raise
    except json.JSONDecodeError as exc:
        raise CachedLayerReplayError(f"{subject} 返回非法 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise CachedLayerReplayError(f"{subject} JSON 顶层必须是对象")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_int(value: object, *, minimum: int = 0) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


class PersistentCachedLayerReplay:
    """用 UTF-8 JSONL 驱动持久 causal-block replay worker。"""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not command or timeout_seconds <= 0:
            raise CachedLayerReplayError("worker command 和 timeout 必须有效")
        self.command = tuple(str(value) for value in command)
        self.timeout_seconds = float(timeout_seconds)
        self.process: subprocess.Popen[str] | None = None
        self.poisoned = False
        self.closed = False
        self.counter = 0
        self.hello: dict[str, Any] | None = None
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=64)
        self._threads: list[threading.Thread] = []
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
            raise CachedLayerReplayError(f"无法启动 cached replay worker: {exc}") from exc
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
            (read_stdout, "polaris-cached-replay-stdout"),
            (read_stderr, "polaris-cached-replay-stderr"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

        try:
            hello = _strict_json(self._readline(), subject="worker hello")
        except CachedLayerReplayError as exc:
            self._fail(str(exc))
        if (
            hello.get("protocol") != WRITEBACK_PROTOCOL
            or hello.get("op") != "hello"
            or hello.get("ready") is not True
            or hello.get("causal_block_layer_replay") is not True
            or hello.get("causal_block_sizes") != [4, 8]
            or hello.get("batch_payload_verification") is not True
        ):
            self._fail("cached replay worker hello 合同漂移")
        self.hello = hello

    def _readline(self) -> str:
        try:
            line = self._stdout.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._fail("cached replay worker 响应超时")
            raise AssertionError from exc
        if line is None:
            code = None if self.process is None else self.process.poll()
            stderr = " | ".join(self._stderr)
            self._fail(f"cached replay worker 提前退出 code={code}: {stderr}")
        assert line is not None
        try:
            byte_count = len(line.encode("utf-8", errors="strict"))
        except UnicodeError as exc:
            self._fail(f"cached replay worker 响应不是严格 UTF-8: {exc}")
        if byte_count > MAX_JSONL_BYTES:
            self._fail("cached replay worker 响应超过 64 KiB")
        return line

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
        raise CachedLayerReplayError(message)

    def _load_manifests(
        self,
        manifests: Sequence[Path],
    ) -> tuple[
        tuple[Path, ...],
        tuple[Path, ...],
        int,
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, ...], ...],
        tuple[str, ...],
    ]:
        if isinstance(manifests, (str, bytes, os.PathLike)):
            self._fail("manifests 必须是 4 或 8 个路径组成的序列")
        try:
            sources = tuple(manifests)
        except TypeError:
            self._fail("manifests 必须是路径序列")
        if len(sources) not in BLOCK_SIZES:
            self._fail("manifest 数量只能是 4 或 8")

        paths: list[Path] = []
        capture_roots: list[Path] = []
        layers: list[int] = []
        positions: list[int] = []
        input_token_ids: list[int] = []
        expert_ids: list[tuple[int, ...]] = []
        sha256s: list[str] = []
        for index, source in enumerate(sources):
            try:
                supplied = Path(source)
            except TypeError:
                self._fail(f"manifest[{index}] 不是路径")
            if not supplied.is_absolute():
                self._fail(f"manifest[{index}] 必须是绝对路径")
            if supplied.name != "bridge_manifest.json":
                self._fail(f"manifest[{index}] 文件名必须是 bridge_manifest.json")
            try:
                path = supplied.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                self._fail(f"manifest[{index}] 不存在或不可访问: {exc}")
            if not path.is_file():
                self._fail(f"manifest[{index}] 不是普通文件")
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
                document = _strict_json(text, subject=f"manifest[{index}]")
                sha256 = _sha256(path)
            except (OSError, UnicodeError, CachedLayerReplayError) as exc:
                self._fail(f"manifest[{index}] 读取/解析失败: {exc}")
            layer = document.get("layer")
            position = document.get("position")
            input_token_id = document.get("input_token_id")
            manifest_expert_ids = document.get("expert_ids")
            if not _strict_int(layer) or int(layer) > 42:
                self._fail(f"manifest[{index}] layer 非法")
            if not _strict_int(position):
                self._fail(f"manifest[{index}] position 非法")
            if not _strict_int(input_token_id):
                self._fail(f"manifest[{index}] input_token_id 非法")
            if (
                not isinstance(manifest_expert_ids, list)
                or len(manifest_expert_ids) != 6
                or any(not _strict_int(value) for value in manifest_expert_ids)
                or len(set(manifest_expert_ids)) != 6
            ):
                self._fail(f"manifest[{index}] expert_ids 非法")
            paths.append(path)
            capture_roots.append(path.parent.resolve(strict=True))
            layers.append(int(layer))
            positions.append(int(position))
            input_token_ids.append(int(input_token_id))
            expert_ids.append(tuple(int(value) for value in manifest_expert_ids))
            sha256s.append(sha256)

        if len(set(layers)) != 1:
            self._fail("manifest 必须全部属于同一层")
        expected_positions = tuple(range(positions[0], positions[0] + len(positions)))
        if tuple(positions) != expected_positions:
            self._fail("manifest position 必须按输入顺序严格连续")
        return (
            tuple(paths),
            tuple(capture_roots),
            layers[0],
            tuple(positions),
            tuple(input_token_ids),
            tuple(expert_ids),
            tuple(sha256s),
        )

    def execute(
        self,
        manifests: Sequence[Path],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
        if self.poisoned:
            raise CachedLayerReplayError("cached replay worker 已 poisoned")
        if self.closed or self.process is None or self.process.poll() is not None:
            self.poisoned = True
            raise CachedLayerReplayError("cached replay worker 已关闭/退出")

        (
            paths,
            capture_roots,
            layer,
            positions,
            input_token_ids,
            expert_ids,
            manifest_sha256s,
        ) = (
            self._load_manifests(manifests)
        )
        self.counter += 1
        request_id = f"py-cached-{os.getpid()}-{self.counter}"
        request = {
            "protocol": WRITEBACK_PROTOCOL,
            "op": REQUEST_OP,
            "request_id": request_id,
            "manifests": [str(path) for path in paths],
            "batch_verify_payloads": True,
        }
        if set(request) != {
            "protocol",
            "op",
            "request_id",
            "manifests",
            "batch_verify_payloads",
        }:
            self._fail("cached replay 请求字段合同漂移")
        try:
            request_line = json.dumps(
                request,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            request_bytes = request_line.encode("utf-8", errors="strict")
        except (TypeError, ValueError, UnicodeError) as exc:
            self._fail(f"cached replay 请求无法严格序列化: {exc}")
        if len(request_bytes) > MAX_JSONL_BYTES:
            self._fail("cached replay worker 请求超过 64 KiB")

        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(request_line + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, UnicodeError) as exc:
            self._fail(f"写入 cached replay worker 失败: {exc}")

        try:
            response = _strict_json(self._readline(), subject="worker response")
        except CachedLayerReplayError as exc:
            self._fail(str(exc))
        if response.get("ok") is not True:
            self._fail(f"cached replay worker 拒绝请求: {response.get('error')}")

        response_block_size = response.get("block_size")
        response_layer = response.get("layer")
        response_positions = response.get("positions")
        if (
            response.get("protocol") != WRITEBACK_PROTOCOL
            or response.get("request_id") != request_id
            or response.get("mode") != RESPONSE_MODE
            or not _strict_int(response_block_size)
            or response_block_size != len(paths)
            or not _strict_int(response_layer)
            or response_layer != layer
            or not isinstance(response_positions, list)
            or any(not _strict_int(value) for value in response_positions)
            or tuple(response_positions) != positions
        ):
            self._fail("cached replay worker response 身份/block/layer/positions 漂移")
        if response.get("speed_eligible_verifier") is not False:
            self._fail("cached replay worker 禁止声明 speed eligible")

        outputs = response.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != len(paths):
            self._fail("cached replay worker output 数量漂移")

        expected_output_path = (capture_roots[0] / OUTPUT_FILE).resolve()
        output_views: list[tuple[int, Mapping[str, Any]]] = []
        for index, (output, position, input_token_id, manifest_sha256, experts) in enumerate(
            zip(
                outputs,
                positions,
                input_token_ids,
                manifest_sha256s,
                expert_ids,
                strict=True,
            )
        ):
            if not isinstance(output, Mapping):
                self._fail(f"cached replay output[{index}] 必须是对象")
            output_position = output.get("position")
            output_input_token_id = output.get("input_token_id")
            output_manifest_sha256 = output.get("manifest_sha256")
            output_expert_ids = output.get("expert_ids")
            view = output.get("output")
            if (
                not _strict_int(output_position)
                or output_position != position
                or not _strict_int(output_input_token_id)
                or output_input_token_id != input_token_id
                or not _valid_sha256(output_manifest_sha256)
                or output_manifest_sha256 != manifest_sha256
                or not isinstance(output_expert_ids, list)
                or tuple(output_expert_ids) != experts
                or not isinstance(view, Mapping)
            ):
                self._fail(f"cached replay output[{index}] 身份/manifest SHA 漂移")
            output_views.append((position, view))

        for index, (_, view) in enumerate(output_views):
            raw_path = view.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                self._fail(f"cached replay output[{index}] 缺少路径")
            supplied_output_path = Path(raw_path)
            if not supplied_output_path.is_absolute():
                self._fail(f"cached replay output[{index}] 路径必须是绝对路径")
            if supplied_output_path.name != OUTPUT_FILE:
                self._fail(f"cached replay output[{index}] 文件名漂移")
            try:
                supplied_parent = supplied_output_path.parent.resolve(strict=True)
                same_parent = os.path.samefile(supplied_parent, capture_roots[0])
            except (OSError, RuntimeError) as exc:
                self._fail(f"cached replay output[{index}] 边界检查失败: {exc}")
            if not same_parent:
                self._fail(f"cached replay output[{index}] 越出 capture 边界")

        try:
            combined_payload = bytearray(expected_output_path.read_bytes())
        except OSError as exc:
            self._fail(f"cached replay combined output 无法读取: {exc}")
        expected_combined_bytes = len(paths) * OUTPUT_BYTES
        if len(combined_payload) != expected_combined_bytes:
            self._fail("cached replay combined output 字节数漂移")

        tensors: list[torch.Tensor] = []
        output_sha256s: list[str] = []
        output_paths: list[str] = []
        for index, (position, view) in enumerate(output_views):
            shape = view.get("shape")
            byte_count = view.get("bytes")
            offset = view.get("offset")
            if (
                view.get("dtype") != "bf16_le"
                or not isinstance(shape, list)
                or any(not _strict_int(value) for value in shape)
                or tuple(shape) != OUTPUT_SHAPE
                or not _strict_int(byte_count)
                or byte_count != OUTPUT_BYTES
                or not _strict_int(offset)
                or offset != index * OUTPUT_BYTES
            ):
                self._fail(
                    f"cached replay output[{index}] offset/dtype/shape/bytes 漂移"
                )

            raw_path = view.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                self._fail(f"cached replay output[{index}] 缺少路径")
            supplied_output_path = Path(raw_path)
            if not supplied_output_path.is_absolute():
                self._fail(f"cached replay output[{index}] 路径必须是绝对路径")
            if supplied_output_path.name != OUTPUT_FILE:
                self._fail(f"cached replay output[{index}] 文件名漂移")
            try:
                output_path = supplied_output_path.resolve(strict=True)
                same_output = os.path.samefile(output_path, expected_output_path)
                same_parent = os.path.samefile(output_path.parent, capture_roots[0])
            except (OSError, RuntimeError) as exc:
                self._fail(f"cached replay output[{index}] 边界检查失败: {exc}")
            if (
                not output_path.is_file()
                or output_path.name != OUTPUT_FILE
                or not same_output
                or not same_parent
            ):
                self._fail(f"cached replay output[{index}] 越出 capture 边界")

            expected_sha256 = view.get("sha256")
            assert isinstance(offset, int)
            payload = combined_payload[offset : offset + OUTPUT_BYTES]
            actual_sha256 = hashlib.sha256(payload).hexdigest()
            if not _valid_sha256(expected_sha256) or actual_sha256 != expected_sha256:
                self._fail(f"cached replay output[{index}] SHA-256 漂移")
            if len(payload) != OUTPUT_BYTES:
                self._fail(f"cached replay output[{index}] 读取后字节数漂移")
            if sys.byteorder != "little":
                self._fail("当前主机无法直接解释 BF16 little-endian 输出")
            tensor = (
                torch.frombuffer(payload, dtype=torch.bfloat16)
                .clone()
                .reshape(OUTPUT_SHAPE)
            )
            if not bool(torch.isfinite(tensor).all().item()):
                self._fail(f"cached replay output[{index}] 含非有限 BF16")
            tensors.append(tensor)
            output_sha256s.append(actual_sha256)
            output_paths.append(str(output_path))

        evidence = {
            "protocol": WRITEBACK_PROTOCOL,
            "mode": RESPONSE_MODE,
            "request_id": request_id,
            "block_size": len(paths),
            "layer": layer,
            "positions": list(positions),
            "manifest_sha256s": list(manifest_sha256s),
            "output_sha256s": output_sha256s,
            "output_paths": output_paths,
            "worker_response": response,
            "speed_eligible_verifier": False,
        }
        return tuple(tensors), evidence

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
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

    def __enter__(self) -> "PersistentCachedLayerReplay":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
