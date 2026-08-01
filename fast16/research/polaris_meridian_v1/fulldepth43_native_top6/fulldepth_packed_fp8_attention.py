"""FullDepth43 通用 packed-FP8 attention Vulkan worker 客户端。

客户端只传控制信息与共享 arena 视图。weight/scale 必须已经位于本机
``D:/models/Polaris-S14/range_cache``，Rust worker 会再次校验真实路径、字节与 SHA。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROTOCOL = "polaris-fulldepth43-packed-fp8-attention-v1"
WORKER_ARG = "--fulldepth43-packed-fp8-attention-worker"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
PROFILE = "fulldepth43_native_top6"
CATALOG_SHA256 = "ca619984d4a46ad1a3701d2b4035766ea40c3a3dbedd3a474ce1df7aad4d0049"
OUTPUT_ROUNDING = "bf16_rne_then_f32_le"
ARENA_MAX_BYTES = 64 * 1024 * 1024
POSITION_MAX = 1_048_575
_MAX_JSON_BYTES = 65_536
_OUTPUT_POISON = 0xA5
_PROJECTIONS = ["wq_a", "wkv", "wq_b", "indexer.wq_b", "wo_b", "wo_a"]
_KERNELS = ["standard", "grouped_wo_a"]


class FullDepthPackedFp8Error(RuntimeError):
    """协议、路径或数值合同漂移。"""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(line: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FullDepthPackedFp8Error(f"worker JSON 含重复 key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise FullDepthPackedFp8Error(f"worker JSON 含非有限常量: {value}")

    try:
        result = json.loads(
            line,
            object_pairs_hook=unique,
            parse_constant=invalid_constant,
        )
    except FullDepthPackedFp8Error:
        raise
    except json.JSONDecodeError as exc:
        raise FullDepthPackedFp8Error(f"worker 返回非法 JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise FullDepthPackedFp8Error("worker JSON 顶层必须是对象")
    return result


def _require_keys(document: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(document)
    if observed != expected:
        raise FullDepthPackedFp8Error(
            f"{label} key 漂移: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


@dataclass(frozen=True)
class PackedFp8Asset:
    tensor: str
    path: Path
    bytes: int
    sha256: str
    dtype: str
    shape: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PackedFp8Asset":
        required = {"tensor", "path", "bytes", "sha256", "dtype", "shape"}
        _require_keys(value, required, "packed-FP8 asset")
        asset = cls(
            tensor=str(value["tensor"]),
            path=Path(value["path"]).resolve(strict=True),
            bytes=int(value["bytes"]),
            sha256=str(value["sha256"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),
        )
        if (
            not asset.path.is_file()
            or asset.path.suffix.lower() != ".bin"
            or asset.path.parent.name != "range_cache"
            or asset.bytes <= 0
            or asset.path.stat().st_size != asset.bytes
            or not _is_sha256(asset.sha256)
            or any(dimension <= 0 for dimension in asset.shape)
        ):
            raise FullDepthPackedFp8Error("packed-FP8 asset 路径/字节/SHA/shape 非法")
        return asset

    def to_request(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


def projection_spec(layer: int, suffix: str) -> dict[str, Any]:
    table: dict[str, tuple[str, int, int, int | None, int | None, str]] = {
        "wq_a": ("standard", 1024, 4096, None, None, "cpu_e4m3fn_quant_dequant_f32"),
        "wkv": ("standard", 512, 4096, None, None, "cpu_e4m3fn_quant_dequant_f32"),
        "wq_b": ("standard", 32768, 1024, None, None, "cpu_e4m3fn_quant_dequant_f32"),
        "indexer.wq_b": (
            "standard",
            8192,
            1024,
            None,
            None,
            "cpu_e4m3fn_quant_dequant_f32",
        ),
        "wo_b": ("standard", 4096, 8192, None, None, "cpu_e4m3fn_quant_dequant_f32"),
        "wo_a": (
            "grouped_wo_a",
            8192,
            4096,
            8,
            1024,
            "bf16_carrying_f32_per_group",
        ),
    }
    if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer <= 42:
        raise FullDepthPackedFp8Error("layer 必须位于 0..42")
    try:
        kernel, n, k, groups, n_per_group, activation_contract = table[suffix]
    except KeyError as exc:
        raise FullDepthPackedFp8Error(f"未批准的 attention projection: {suffix}") from exc
    return {
        "name": f"layers.{layer}.attn.{suffix}",
        "kernel": kernel,
        "n": n,
        "k": k,
        "groups": groups,
        "n_per_group": n_per_group,
        "activation_contract": activation_contract,
        "output_rounding": OUTPUT_ROUNDING,
    }


class FullDepthPackedFp8Arena:
    """允许所有已批准 attention shape 复用的单文件 arena。"""

    def __init__(self, path: Path, *, create: bool = False) -> None:
        raw = Path(path)
        if raw.suffix.lower() != ".bin":
            raise FullDepthPackedFp8Error("arena 必须是 .bin 文件")
        if create:
            parent = raw.parent.resolve(strict=True)
            try:
                with (parent / raw.name).open("xb") as stream:
                    stream.truncate(512 * 1024)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise FullDepthPackedFp8Error(f"无法创建 arena: {exc}") from exc
        self.path = raw.resolve(strict=True)
        if (
            not self.path.is_file()
            or self.path.stat().st_size < 512 * 1024
            or self.path.stat().st_size > ARENA_MAX_BYTES
        ):
            raise FullDepthPackedFp8Error("arena 文件大小非法")

    @staticmethod
    def _shape(spec: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if spec["kernel"] == "grouped_wo_a":
            input_shape = (1, 1, int(spec["groups"]), int(spec["k"]))
        else:
            input_shape = (1, 1, int(spec["k"]))
        return input_shape, (1, 1, int(spec["n"]))

    def prepare(
        self, activation: torch.Tensor, spec: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        input_view, output_views, input_sha256 = self.prepare_shared(
            activation, (spec,)
        )
        return input_view, output_views[0], input_sha256

    def prepare_shared(
        self,
        activation: torch.Tensor,
        specs: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
        if not specs:
            raise FullDepthPackedFp8Error("shared attention batch 不得为空")
        shapes = [self._shape(spec) for spec in specs]
        input_shape = shapes[0][0]
        if any(shape != input_shape for shape, _ in shapes):
            raise FullDepthPackedFp8Error("shared attention batch 输入 shape 不一致")
        grouped = any(spec["kernel"] == "grouped_wo_a" for spec in specs)
        if grouped and len(specs) != 1:
            raise FullDepthPackedFp8Error("grouped wo_a 不允许进入 shared-input batch")
        if (
            activation.device.type != "cpu"
            or activation.dtype != torch.float32
            or tuple(activation.shape) != input_shape
            or not bool(torch.isfinite(activation).all().item())
        ):
            raise FullDepthPackedFp8Error("attention activation CPU/F32/shape/有限数合同漂移")
        activation_array = (
            activation.detach().contiguous().numpy().astype("<f4", copy=False)
        )
        if grouped and bool(
            np.any(activation_array.view("<u4") & np.uint32(0xFFFF))
        ):
            raise FullDepthPackedFp8Error("grouped wo_a activation 必须是 BF16-carrying F32")
        payload = activation_array.tobytes(order="C")
        output_views: list[dict[str, Any]] = []
        next_offset = len(payload)
        for _, output_shape in shapes:
            output_bytes = math.prod(output_shape) * 4
            output_views.append(
                {
                    "path": str(self.path),
                    "offset": next_offset,
                    "bytes": output_bytes,
                    "dtype": "f32_le_bf16_rounded",
                    "shape": list(output_shape),
                }
            )
            next_offset += output_bytes
        if next_offset > self.path.stat().st_size:
            raise FullDepthPackedFp8Error("arena 容量不足")
        poison = bytes([_OUTPUT_POISON]) * (next_offset - len(payload))
        try:
            with self.path.open("r+b", buffering=0) as stream:
                stream.seek(0)
                stream.write(payload)
                stream.seek(len(payload))
                stream.write(poison)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise FullDepthPackedFp8Error(f"写入 arena 失败: {exc}") from exc
        input_view = {
            "path": str(self.path),
            "offset": 0,
            "bytes": len(payload),
            "dtype": "f32_le",
            "shape": list(input_shape),
        }
        return input_view, output_views, _sha256(payload)

    def read_output(self, view: Mapping[str, Any], expected_sha256: str) -> torch.Tensor:
        try:
            with self.path.open("rb") as stream:
                stream.seek(int(view["offset"]))
                payload = stream.read(int(view["bytes"]))
        except OSError as exc:
            raise FullDepthPackedFp8Error(f"读取 arena output 失败: {exc}") from exc
        if len(payload) != int(view["bytes"]) or _sha256(payload) != expected_sha256:
            raise FullDepthPackedFp8Error("arena output 字节/SHA 漂移")
        values = np.frombuffer(payload, dtype="<f4")
        bits = values.view("<u4")
        if bool(np.any(bits & np.uint32(0xFFFF))) or not bool(np.isfinite(values).all()):
            raise FullDepthPackedFp8Error("worker output 不是有限 BF16-carrying F32")
        return torch.from_numpy(values.copy()).reshape(tuple(view["shape"]))


class PersistentFullDepthPackedFp8Attention:
    """单个严格串行、epoch 单调递增的通用 worker 进程。"""

    _HELLO_KEYS = {
        "protocol", "op", "ready", "revision", "profile", "layers",
        "position_max", "projections", "kernels", "arena_transport",
        "catalog_sha256", "payload_root", "payload_cache_bytes", "output_rounding",
    }
    _RESPONSE_KEYS = {
        "protocol", "request_id", "ok", "revision", "profile", "layer", "position",
        "arena_epoch", "projection", "input", "output_written", "input_sha256",
        "output_sha256", "weight_sha256", "scale_sha256", "catalog_sha256",
        "payload_hash_verified", "gpu_slot_cache_hit", "gpu_slot_cache_entries",
        "gpu_slot_resident_bytes", "payload_uploaded_bytes",
        "activation_uploaded_bytes", "numeric_mode", "output_rounding",
    }
    _BATCH_RESPONSE_KEYS = {
        "protocol", "request_id", "ok", "revision", "profile", "layer",
        "position", "arena_epoch", "input", "input_sha256", "outputs",
        "catalog_sha256", "gpu_slot_cache_entries", "activation_uploaded_bytes",
    }
    _BATCH_OUTPUT_KEYS = {
        "projection", "output_written", "output_sha256", "weight_sha256",
        "scale_sha256", "payload_hash_verified", "gpu_slot_cache_hit",
        "gpu_slot_resident_bytes", "payload_uploaded_bytes", "numeric_mode",
        "output_rounding",
    }

    def __init__(
        self,
        command: Sequence[str],
        arena: FullDepthPackedFp8Arena,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not command or WORKER_ARG not in tuple(str(item) for item in command):
            raise FullDepthPackedFp8Error(f"worker command 必须包含 {WORKER_ARG}")
        self.command = tuple(str(item) for item in command)
        self.arena = arena
        self.timeout_seconds = float(timeout_seconds)
        self.process: subprocess.Popen[str] | None = None
        self.hello: dict[str, Any] | None = None
        self.epoch = 0
        self.counter = 0
        self.poisoned = False
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=64)
        self._threads: list[threading.Thread] = []
        self._start()

    def _start(self) -> None:
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
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise FullDepthPackedFp8Error(f"无法启动 worker: {exc}") from exc
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

        for target in (read_stdout, read_stderr):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        hello = _strict_json(self._readline())
        _require_keys(hello, self._HELLO_KEYS, "FullDepth43 hello")
        if (
            hello.get("protocol") != PROTOCOL
            or hello.get("op") != "hello"
            or hello.get("ready") is not True
            or hello.get("revision") != REVISION
            or hello.get("profile") != PROFILE
            or hello.get("layers") != {"min": 0, "max": 42}
            or hello.get("position_max") != POSITION_MAX
            or hello.get("projections") != _PROJECTIONS
            or hello.get("kernels") != _KERNELS
            or hello.get("arena_transport") != "shared_binary_file"
            or hello.get("catalog_sha256") != CATALOG_SHA256
            or Path(hello.get("payload_root", "")).resolve(strict=True).name
            != "range_cache"
            or hello.get("payload_cache_bytes") != 256 * 1024 * 1024
            or hello.get("output_rounding") != OUTPUT_ROUNDING
        ):
            self._fail("FullDepth43 worker hello 身份漂移")
        self.hello = hello

    def _readline(self) -> str:
        try:
            line = self._stdout.get(timeout=self.timeout_seconds)
        except queue.Empty:
            self._fail("FullDepth43 worker 响应超时")
        if line is None:
            self._fail(
                f"FullDepth43 worker 提前退出: {' | '.join(self._stderr)}"
            )
        assert line is not None
        if len(line.encode("utf-8")) > _MAX_JSON_BYTES:
            self._fail("FullDepth43 worker 响应超过 64 KiB")
        return line

    def _cleanup_process(self, process: subprocess.Popen[str] | None) -> None:
        if process is None:
            return
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        for thread in self._threads:
            thread.join(timeout=1)
        self._threads.clear()
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass

    def _fail(self, message: str) -> None:
        self.poisoned = True
        process, self.process = self.process, None
        self._cleanup_process(process)
        raise FullDepthPackedFp8Error(message)

    def execute(
        self,
        **request: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        try:
            return self._execute(**request)
        except FullDepthPackedFp8Error as error:
            if not self.poisoned:
                self._fail(str(error))
            raise
        except Exception as error:
            self._fail(f"FullDepth43 worker 客户端异常: {type(error).__name__}: {error}")

    def execute_shared_batch(
        self,
        *,
        layer: int,
        position: int,
        suffixes: Sequence[str],
        activation: torch.Tensor,
        assets: Mapping[str, tuple[PackedFp8Asset, PackedFp8Asset]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        try:
            return self._execute_shared_batch(
                layer=layer,
                position=position,
                suffixes=suffixes,
                activation=activation,
                assets=assets,
            )
        except FullDepthPackedFp8Error as error:
            if not self.poisoned:
                self._fail(str(error))
            raise
        except Exception as error:
            self._fail(
                f"FullDepth43 shared batch 客户端异常: {type(error).__name__}: {error}"
            )

    def _execute_shared_batch(
        self,
        *,
        layer: int,
        position: int,
        suffixes: Sequence[str],
        activation: torch.Tensor,
        assets: Mapping[str, tuple[PackedFp8Asset, PackedFp8Asset]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if self.poisoned or self.process is None or self.process.poll() is not None:
            raise FullDepthPackedFp8Error("FullDepth43 worker 已 poisoned/退出")
        if not isinstance(position, int) or isinstance(position, bool) or not 0 <= position <= POSITION_MAX:
            self._fail("position 超出合同")
        ordered_suffixes = tuple(suffixes)
        approved_batches = {
            ("wq_a", "wkv"),
            ("wq_b", "indexer.wq_b"),
        }
        if len(set(ordered_suffixes)) != len(ordered_suffixes):
            self._fail("shared batch 含重复 suffix")
        if ordered_suffixes not in approved_batches:
            self._fail(f"shared batch 组合未批准: {ordered_suffixes}")
        if set(assets) != set(ordered_suffixes):
            self._fail("shared batch assets 与 suffixes 身份不一致")

        specs = [projection_spec(layer, suffix) for suffix in ordered_suffixes]
        input_view, output_views, input_sha256 = self.arena.prepare_shared(
            activation, specs
        )
        projection_requests: list[dict[str, Any]] = []
        ordered_assets: list[tuple[PackedFp8Asset, PackedFp8Asset]] = []
        for suffix, spec, output_view in zip(
            ordered_suffixes, specs, output_views, strict=True
        ):
            weight, scale = assets[suffix]
            if weight.tensor != f"{spec['name']}.weight" or scale.tensor != f"{spec['name']}.scale":
                self._fail("shared batch weight/scale tensor 身份与 projection 不一致")
            ordered_assets.append((weight, scale))
            projection_requests.append(
                {
                    "projection": spec,
                    "weight": weight.to_request(),
                    "scale": scale.to_request(),
                    "output": output_view,
                }
            )

        self.counter += 1
        request_id = f"py-{os.getpid()}-{self.counter}"
        request = {
            "protocol": PROTOCOL,
            "op": "execute_fp8_attention_shared_batch",
            "request_id": request_id,
            "revision": REVISION,
            "profile": PROFILE,
            "layer": layer,
            "position": position,
            "arena_epoch": self.epoch,
            "input_sha256": input_sha256,
            "input": input_view,
            "projections": projection_requests,
        }
        assert self.process.stdin is not None
        try:
            line = json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
            if len(line.encode("utf-8")) > _MAX_JSON_BYTES:
                self._fail("FullDepth43 shared batch request 超过 64 KiB")
            self.process.stdin.write(line)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._fail(f"写入 shared batch worker 失败: {exc}")

        response = _strict_json(self._readline())
        if response.get("ok") is not True:
            self._fail(f"worker 拒绝 shared batch: {response.get('error')}")
        _require_keys(response, self._BATCH_RESPONSE_KEYS, "FullDepth43 batch response")
        response_outputs = response.get("outputs")
        if (
            response.get("protocol") != PROTOCOL
            or response.get("request_id") != request_id
            or response.get("revision") != REVISION
            or response.get("profile") != PROFILE
            or response.get("layer") != layer
            or response.get("position") != position
            or response.get("arena_epoch") != self.epoch
            or response.get("input") != input_view
            or response.get("input_sha256") != input_sha256
            or response.get("catalog_sha256") != CATALOG_SHA256
            or not isinstance(response.get("gpu_slot_cache_entries"), int)
            or response.get("gpu_slot_cache_entries", -1) < len(ordered_suffixes)
            or response.get("activation_uploaded_bytes") != input_view["bytes"]
            or not isinstance(response_outputs, list)
            or len(response_outputs) != len(ordered_suffixes)
        ):
            self._fail("FullDepth43 batch response 顶层身份/数量合同漂移")

        outputs: dict[str, torch.Tensor] = {}
        assert isinstance(response_outputs, list)
        for suffix, spec, output_view, asset_pair, item in zip(
            ordered_suffixes,
            specs,
            output_views,
            ordered_assets,
            response_outputs,
            strict=True,
        ):
            if not isinstance(item, dict):
                self._fail("FullDepth43 batch output 缺项/类型漂移")
            _require_keys(item, self._BATCH_OUTPUT_KEYS, "FullDepth43 batch output")
            weight, scale = asset_pair
            if (
                item.get("projection") != spec
                or item.get("output_written") != output_view
                or item.get("weight_sha256") != weight.sha256
                or item.get("scale_sha256") != scale.sha256
                or item.get("payload_hash_verified") is not True
                or not isinstance(item.get("gpu_slot_cache_hit"), bool)
                or not isinstance(item.get("gpu_slot_resident_bytes"), int)
                or item.get("gpu_slot_resident_bytes", -1) < 0
                or item.get("payload_uploaded_bytes")
                != (
                    0
                    if item.get("gpu_slot_cache_hit")
                    else weight.bytes + scale.bytes
                )
                or item.get("numeric_mode")
                != "packed_fp8_e4m3_ue8m0_bf16_output"
                or item.get("output_rounding") != OUTPUT_ROUNDING
                or not _is_sha256(item.get("output_sha256"))
            ):
                self._fail("FullDepth43 batch output 顺序/身份/SHA 合同漂移")
            outputs[suffix] = self.arena.read_output(
                output_view, item["output_sha256"]
            )
        self.epoch += 1
        return outputs, dict(response)

    def _execute(
        self,
        *,
        layer: int,
        position: int,
        suffix: str,
        activation: torch.Tensor,
        weight: PackedFp8Asset,
        scale: PackedFp8Asset,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.poisoned or self.process is None or self.process.poll() is not None:
            raise FullDepthPackedFp8Error("FullDepth43 worker 已 poisoned/退出")
        if not isinstance(position, int) or isinstance(position, bool) or not 0 <= position <= POSITION_MAX:
            self._fail("position 超出合同")
        spec = projection_spec(layer, suffix)
        if weight.tensor != f"{spec['name']}.weight" or scale.tensor != f"{spec['name']}.scale":
            self._fail("weight/scale tensor 身份与 projection 不一致")
        input_view, output_view, input_sha256 = self.arena.prepare(activation, spec)
        self.counter += 1
        request_id = f"py-{os.getpid()}-{self.counter}"
        request = {
            "protocol": PROTOCOL,
            "op": "execute_fp8_attention",
            "request_id": request_id,
            "revision": REVISION,
            "profile": PROFILE,
            "layer": layer,
            "position": position,
            "arena_epoch": self.epoch,
            "input_sha256": input_sha256,
            "projection": spec,
            "weight": weight.to_request(),
            "scale": scale.to_request(),
            "input": input_view,
            "output": output_view,
        }
        assert self.process.stdin is not None
        try:
            line = json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
            if len(line.encode("utf-8")) > _MAX_JSON_BYTES:
                self._fail("FullDepth43 request 超过 64 KiB")
            self.process.stdin.write(line)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._fail(f"写入 worker 失败: {exc}")
        response = _strict_json(self._readline())
        if response.get("ok") is not True:
            self._fail(f"worker 拒绝请求: {response.get('error')}")
        _require_keys(response, self._RESPONSE_KEYS, "FullDepth43 response")
        if (
            response.get("protocol") != PROTOCOL
            or response.get("request_id") != request_id
            or response.get("revision") != REVISION
            or response.get("profile") != PROFILE
            or response.get("layer") != layer
            or response.get("position") != position
            or response.get("arena_epoch") != self.epoch
            or response.get("projection") != spec
            or response.get("input") != input_view
            or response.get("output_written") != output_view
            or response.get("input_sha256") != input_sha256
            or response.get("weight_sha256") != weight.sha256
            or response.get("scale_sha256") != scale.sha256
            or response.get("catalog_sha256") != CATALOG_SHA256
            or response.get("payload_hash_verified") is not True
            or not isinstance(response.get("gpu_slot_cache_hit"), bool)
            or not isinstance(response.get("gpu_slot_cache_entries"), int)
            or response.get("gpu_slot_cache_entries", -1) < 0
            or not isinstance(response.get("gpu_slot_resident_bytes"), int)
            or response.get("gpu_slot_resident_bytes", -1) < 0
            or response.get("payload_uploaded_bytes")
            != (
                0
                if response.get("gpu_slot_cache_hit")
                else weight.bytes + scale.bytes
            )
            or response.get("activation_uploaded_bytes") != input_view["bytes"]
            or response.get("numeric_mode")
            != (
                "grouped_packed_fp8_e4m3_ue8m0_bf16_input_output"
                if spec["kernel"] == "grouped_wo_a"
                else "packed_fp8_e4m3_ue8m0_bf16_output"
            )
            or response.get("output_rounding") != OUTPUT_ROUNDING
            or not _is_sha256(response.get("output_sha256"))
        ):
            self._fail("FullDepth43 response 身份/SHA 合同漂移")
        output = self.arena.read_output(output_view, response["output_sha256"])
        evidence = dict(response)
        self.epoch += 1
        return output, evidence

    def close(self) -> None:
        process, self.process = self.process, None
        self._cleanup_process(process)

    def __enter__(self) -> "PersistentFullDepthPackedFp8Attention":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "FullDepthPackedFp8Arena",
    "FullDepthPackedFp8Error",
    "PackedFp8Asset",
    "PersistentFullDepthPackedFp8Attention",
    "PROTOCOL",
    "WORKER_ARG",
    "projection_spec",
]
