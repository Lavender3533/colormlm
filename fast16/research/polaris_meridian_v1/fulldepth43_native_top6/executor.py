"""FullDepth43/native-top6 最小真实 CPU/PyTorch reference executor。

执行入口严格跑完 0..42 层、当前 token/当前层原生 top-6、
shared expert、mHC、window KV 与 compressor remainder，最后才允许原生
HC/norm/BF16 head argmax 产生 token。静态页缺失或预算不足时 fail closed。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from fast16.research.polaris_meridian_v1.s14_first_real_token import executor as s14
from fast16.research.polaris_meridian_v1.s14_range_pack import online_range

from .catalog import build_catalog, read_json, validate_catalog, write_json
from . import checkpoint as decoder_checkpoint
from .preflight import DEFAULT_ASSET_ROOT, DEFAULT_CATALOG, run_preflight
from .profile import FULLDEPTH43_NATIVE_TOP6, ExecutionProfile
from .fulldepth_packed_fp8_attention import (
    FullDepthPackedFp8Arena,
    FullDepthPackedFp8Error,
    PackedFp8Asset,
    PersistentFullDepthPackedFp8Attention,
    WORKER_ARG as FULLDEPTH_FP8_ATTENTION_WORKER_ARG,
)
from .vulkan_writeback import (
    PAYLOAD_IDENTITY_CONTRACT,
    PAYLOAD_VERIFICATION_SCOPE,
    PersistentVulkanWriteback,
    VulkanWritebackError,
    verify_exact_bf16_writeback,
)
from .vulkan_final_head import (
    FullDepthVulkanFinalHead,
    VulkanFinalHeadError,
)


REPORT_FORMAT = "polaris-fulldepth43-native-top6-reference-v1"
VULKAN_BRIDGE_FORMAT = "polaris-fulldepth43-vulkan-bridge-capture-v1"
DEFAULT_REPORT = Path(__file__).resolve().parent / "last_run_report.json"
DEFAULT_FORCED_PREFILL = Path(__file__).resolve().parent / "first_preview_forced_prefill.json"
DEFAULT_VULKAN_FINAL_HEAD_WORKER = (
    Path(__file__).resolve().parents[4]
    / "scheduler"
    / "target"
    / "release"
    / "examples"
    / "s14_bf16_head.exe"
)


class FullDepthError(RuntimeError):
    pass


def _gpu_verification_owner(entry: Mapping[str, Any]) -> str | None:
    """只把正式GPU独占消费的页交给对应worker做唯一一次内容SHA。"""

    tensor = entry.get("tensor")
    kind = entry.get("kind")
    dtype = entry.get("dtype")
    if (
        kind == "non_expert"
        and isinstance(tensor, str)
        and ".attn." in tensor
        and dtype in {"F8_E4M3", "F8_E8M0"}
    ):
        return "vulkan_attention_worker"
    if kind in {"routed_expert", "shared"}:
        return "vulkan_moe_worker"
    return None


class SessionPhase(str, Enum):
    INIT = "init"
    AWAITING_LAYER = "awaiting_layer"
    LAYER_BASE_READY = "layer_base_ready"
    ROUTED = "routed"
    LAYER_READY = "layer_ready"
    FINAL_PENDING = "final_pending"
    COMPLETE = "complete"


@dataclass(frozen=True)
class LayerPrerequisites:
    layer: int
    token_id: int
    non_expert: tuple[online_range.CachedRange, ...]
    router: tuple[online_range.CachedRange, ...]


@dataclass(frozen=True)
class RoutedLayer:
    layer: int
    token_id: int
    expert_ids: tuple[int, ...]
    experts: Mapping[int, tuple[online_range.CachedRange, ...]]
    shared: tuple[online_range.CachedRange, ...]


@dataclass(frozen=True)
class StaticPrefetchResult:
    prerequisites: LayerPrerequisites
    fetch_seconds: float


class FullDepthRangeSession:
    """固定 43 层 route-first 状态机；路由前无 expert 读取入口。"""

    def __init__(
        self,
        catalog: Mapping[str, Any],
        cache: online_range.RangeCache,
        *,
        profile: ExecutionProfile = FULLDEPTH43_NATIVE_TOP6,
        range_attempts: int = 4,
        range_workers: int = 3,
        range_pool: ThreadPoolExecutor | None = None,
        prefetch_pool: ThreadPoolExecutor | None = None,
        owns_pools: bool = True,
    ) -> None:
        profile.validate()
        validate_catalog(catalog)
        self.profile = profile
        self.catalog = catalog
        self.cache = cache
        if range_attempts <= 0 or not 1 <= range_workers <= 8:
            raise FullDepthError("Range attempts/workers 必须分别为正数和 1..8")
        self.range_attempts = range_attempts
        self.range_workers = range_workers
        if (range_pool is None) != (prefetch_pool is None):
            raise FullDepthError("静态预取必须同时提供 Range 池与独立协调池")
        if range_pool is not None and range_pool is prefetch_pool:
            raise FullDepthError("Range 池与静态预取协调池必须是两个独立线程池")
        self.range_pool = range_pool
        self.prefetch_pool = prefetch_pool
        self.owns_pools = owns_pools
        self._prefetch_future: Future[StaticPrefetchResult] | None = None
        self._prefetch_layer: int | None = None
        self._prefetch_token_id: int | None = None
        self._prefetch_scheduled_after_layer: int | None = None
        self._prefetch_scheduled_at: float | None = None
        self._closed = False
        self.prefetch_events: list[dict[str, Any]] = []
        self.phase = SessionPhase.INIT
        self.layer_index = 0
        self.token_id: int | None = None
        self.route: tuple[int, ...] | None = None

    @property
    def current_layer(self) -> int | None:
        if self.layer_index >= len(self.profile.layers):
            return None
        return self.profile.layers[self.layer_index]

    def _fetch_one(self, entry: Mapping[str, Any]) -> online_range.CachedRange:
        for attempt in range(1, self.range_attempts + 1):
            try:
                return self.cache.fetch(entry)
            except s14._RECOVERABLE_TRANSPORT_ERRORS:
                if attempt == self.range_attempts:
                    raise
                time.sleep(min(2 ** (attempt - 1), 4))
        raise AssertionError("Range retry 不应到达")

    def _fetch_all(self, entries: Iterable[Mapping[str, Any]]) -> tuple[online_range.CachedRange, ...]:
        frozen = tuple(entries)
        if self.range_workers == 1 or len(frozen) <= 1:
            return tuple(self._fetch_one(entry) for entry in frozen)
        range_pool = getattr(self, "range_pool", None)
        if range_pool is not None:
            return tuple(range_pool.map(self._fetch_one, frozen))
        with ThreadPoolExecutor(max_workers=self.range_workers, thread_name_prefix="fd43-range") as pool:
            return tuple(pool.map(self._fetch_one, frozen))

    def _fetch_static(self, layer: int, token_id: int) -> StaticPrefetchResult:
        started = time.perf_counter()
        row = self.catalog["layers"][str(layer)]
        prerequisites = LayerPrerequisites(
            layer=layer,
            token_id=token_id,
            non_expert=self._fetch_all(row["non_expert"]),
            router=self._fetch_all(row["router"]),
        )
        return StaticPrefetchResult(
            prerequisites=prerequisites,
            fetch_seconds=time.perf_counter() - started,
        )

    def schedule_next_static(self, layer: int, token_id: int) -> bool:
        """仅在当前 routed fetch 成功后调度紧邻下一层静态页。"""

        if self._closed:
            raise FullDepthError("Range session 已关闭")
        if self.range_pool is None or self.prefetch_pool is None:
            return False
        if self.phase is not SessionPhase.LAYER_READY:
            raise FullDepthError("静态预取只允许在当前层 routed fetch 完成后调度")
        if token_id != self.token_id or self.current_layer is None:
            raise FullDepthError("静态预取 token 与当前层状态不一致")
        next_index = self.layer_index + 1
        if next_index >= len(self.profile.layers):
            return False
        expected = self.profile.layers[next_index]
        if layer != expected:
            raise FullDepthError(
                f"静态预取只能读取紧邻下一层: expected={expected}, got={layer}"
            )
        if self._prefetch_future is not None:
            raise FullDepthError("上一项静态预取尚未正式消费")
        self._prefetch_layer = layer
        self._prefetch_token_id = token_id
        self._prefetch_scheduled_after_layer = self.current_layer
        self._prefetch_scheduled_at = time.perf_counter()
        self._prefetch_future = self.prefetch_pool.submit(
            self._fetch_static,
            layer,
            token_id,
        )
        return True

    def prepare_embedding_row(self, token_id: int) -> online_range.CachedRange:
        if self.phase is not SessionPhase.INIT:
            raise FullDepthError("embedding row 只允许在 init 获取")
        result = self._fetch_one(online_range.embedding_row_entry(self.catalog, token_id))
        self.phase = SessionPhase.AWAITING_LAYER
        return result

    def prepare_layer(self, layer: int, token_id: int) -> LayerPrerequisites:
        if self.phase is not SessionPhase.AWAITING_LAYER or layer != self.current_layer:
            raise FullDepthError(f"层顺序错误: expected={self.current_layer}, got={layer}")
        future = self._prefetch_future
        if future is None:
            if self.prefetch_pool is not None and self.layer_index > 0:
                raise FullDepthError("启用静态预取后，非首层 prepare 缺少对应 Future")
            result = self._fetch_static(layer, token_id).prerequisites
        else:
            if layer != self._prefetch_layer or token_id != self._prefetch_token_id:
                raise FullDepthError("正式 prepare 与已调度静态预取的 layer/token 不一致")
            wait_started = time.perf_counter()
            try:
                prefetched = future.result()
            except BaseException as error:
                self.prefetch_events.append(
                    {
                        "layer": layer,
                        "scheduled_after_layer": self._prefetch_scheduled_after_layer,
                        "status": "failed",
                        "formal_wait_seconds": time.perf_counter() - wait_started,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                raise
            formal_wait = time.perf_counter() - wait_started
            result = prefetched.prerequisites
            scheduled_at = self._prefetch_scheduled_at
            self.prefetch_events.append(
                {
                    "layer": layer,
                    "scheduled_after_layer": self._prefetch_scheduled_after_layer,
                    "status": "consumed",
                    "fetch_seconds": prefetched.fetch_seconds,
                    "formal_wait_seconds": formal_wait,
                    "hidden_seconds": max(prefetched.fetch_seconds - formal_wait, 0.0),
                    "schedule_to_consume_seconds": (
                        None
                        if scheduled_at is None
                        else time.perf_counter() - scheduled_at
                    ),
                    "non_expert_count": len(result.non_expert),
                    "router_count": len(result.router),
                }
            )
            self._prefetch_future = None
            self._prefetch_layer = None
            self._prefetch_token_id = None
            self._prefetch_scheduled_after_layer = None
            self._prefetch_scheduled_at = None
        self.token_id = token_id
        self.route = None
        self.phase = SessionPhase.LAYER_BASE_READY
        return result

    def submit_top6(self, layer: int, token_id: int, expert_ids: Sequence[int]) -> tuple[int, ...]:
        if self.phase is not SessionPhase.LAYER_BASE_READY:
            raise FullDepthError("必须先完成当前层 attention/router")
        if layer != self.current_layer or token_id != self.token_id:
            raise FullDepthError("top-6 layer/token 与当前状态不一致")
        if (
            len(expert_ids) != self.profile.top_k
            or len(set(expert_ids)) != self.profile.top_k
            or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 256 for value in expert_ids)
        ):
            raise FullDepthError("原生 route 必须是 6 个唯一且有效的 expert ID")
        self.route = tuple(expert_ids)
        self.phase = SessionPhase.ROUTED
        return self.route

    def fetch_routed(self, layer: int, token_id: int) -> RoutedLayer:
        if self.phase is not SessionPhase.ROUTED or self.route is None:
            raise FullDepthError("路由前禁止读取 expert 页")
        if layer != self.current_layer or token_id != self.token_id:
            raise FullDepthError("routed layer/token 漂移")
        row = self.catalog["layers"][str(layer)]
        expert_entries = {
            value: tuple(row["experts"][str(value)]) for value in self.route
        }
        shared_entries = tuple(row["shared"])
        flat_entries = tuple(
            entry
            for value in self.route
            for entry in expert_entries[value]
        ) + shared_entries
        fetched = self._fetch_all(flat_entries)
        experts: dict[int, tuple[online_range.CachedRange, ...]] = {}
        cursor = 0
        for value in self.route:
            width = len(expert_entries[value])
            experts[value] = fetched[cursor : cursor + width]
            cursor += width
        shared = fetched[cursor:]
        if len(shared) != len(shared_entries):
            raise FullDepthError("routed/shared Range 扁平并发重组失败")
        result = RoutedLayer(
            layer=layer,
            token_id=token_id,
            expert_ids=self.route,
            experts=experts,
            shared=shared,
        )
        self.phase = SessionPhase.LAYER_READY
        return result

    def finish_layer(self, layer: int, token_id: int) -> None:
        if self.phase is not SessionPhase.LAYER_READY:
            raise FullDepthError("必须完成 top-6+shared 才能提交层")
        if layer != self.current_layer or token_id != self.token_id:
            raise FullDepthError("提交层的 layer/token 漂移")
        self.layer_index += 1
        self.token_id = None
        self.route = None
        self.phase = (
            SessionPhase.FINAL_PENDING
            if self.layer_index == len(self.profile.layers)
            else SessionPhase.AWAITING_LAYER
        )

    def prepare_final(self) -> tuple[online_range.CachedRange, ...]:
        if self.phase is not SessionPhase.FINAL_PENDING:
            raise FullDepthError("必须完成全部 43 层才能读取 final head")
        result = self._fetch_all(self.catalog["boundary"]["final"])
        self.phase = SessionPhase.COMPLETE
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        future = self._prefetch_future
        if future is not None:
            if not future.cancel():
                try:
                    future.result()
                except BaseException as error:
                    first_error = error
        if self.owns_pools:
            for pool in (self.prefetch_pool, self.range_pool):
                if pool is None:
                    continue
                try:
                    pool.shutdown(wait=True, cancel_futures=True)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error


class FullDepthNativeLayerReference(s14.NativeLayerReference):
    def __init__(
        self,
        layer: int,
        store: s14.TensorStore,
        *,
        profile: ExecutionProfile = FULLDEPTH43_NATIVE_TOP6,
        attention_worker: PersistentFullDepthPackedFp8Attention | None = None,
        attention_position: int | None = None,
        attention_verify_cpu: bool = False,
        attention_shared_batch: bool = False,
        attention_output_chain: bool = False,
        gpu_verifier_ownership: bool = False,
    ) -> None:
        profile.validate()
        if attention_worker is not None and (
            isinstance(attention_position, bool)
            or not isinstance(attention_position, int)
            or attention_position < 0
        ):
            raise FullDepthError("Vulkan attention 必须携带有效 position")
        self.attention_worker = attention_worker
        self.attention_position = attention_position
        self.attention_verify_cpu = attention_verify_cpu
        self.attention_shared_batch = attention_shared_batch
        self.attention_output_chain = attention_output_chain
        self.gpu_verifier_ownership = gpu_verifier_ownership
        self.attention_vulkan_evidence: list[dict[str, Any]] = []
        self._attention_batch_cache: dict[str, Any] = {}
        super().__init__(
            layer,
            store,
            profile_layers=profile.layers,
            compress_ratios={value: profile.ratio_for(value) for value in profile.layers},
        )

    def _packed_asset(self, tensor: str) -> PackedFp8Asset:
        source = self.store.source(tensor)
        if self.gpu_verifier_ownership:
            if source.content_verified or source.verification_owner != "vulkan_attention_worker":
                raise FullDepthError(
                    f"{tensor} 未按合同交给 Vulkan attention worker 做唯一内容验证"
                )
        elif not source.content_verified:
            raise FullDepthError(
                f"{tensor} 内容尚未验证，禁止进入未启用 GPU verifier ownership 的路径"
            )
        return PackedFp8Asset.from_mapping(
            {
                "tensor": tensor,
                "path": str(source.path),
                "bytes": source.entry["bytes"],
                "sha256": source.proof["observed_sha256"],
                "dtype": source.entry["dtype"],
                "shape": source.entry["shape"],
            }
        )

    def _vulkan_attention_projection(
        self,
        array: Any,
        prefix: str,
        suffix: str,
        *,
        activation_already_quantized: bool,
    ) -> Any:
        worker = self.attention_worker
        position = self.attention_position
        if worker is None or position is None:
            raise FullDepthError("Vulkan attention projection 未绑定 worker/position")
        activation_array = (
            array if activation_already_quantized else self._activation_quant(array)
        )
        activation = torch.from_numpy(activation_array).to(dtype=torch.float32).contiguous()
        started = time.perf_counter()
        try:
            output, evidence = worker.execute(
                layer=self.layer,
                position=position,
                suffix=suffix,
                activation=activation,
                weight=self._packed_asset(prefix + ".weight"),
                scale=self._packed_asset(prefix + ".scale"),
            )
        except FullDepthPackedFp8Error as error:
            raise FullDepthError(
                f"L{self.layer} {prefix} Vulkan attention 失败: {error}"
            ) from error
        self.attention_vulkan_evidence.append(
            {
                **evidence,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if self.attention_verify_cpu:
            cpu_output = (
                super()._grouped_wo_a(array, prefix)
                if suffix == "wo_a"
                else super()._linear_fp8(array, prefix)
            )
            cpu_tensor = torch.from_numpy(cpu_output).to(torch.float32)
            exact = bool(torch.equal(output, cpu_tensor))
            self.attention_vulkan_evidence[-1]["cpu_exact_bf16"] = exact
            self.attention_vulkan_evidence[-1]["cpu_output_sha256"] = _sha256_bytes(
                cpu_tensor.contiguous().numpy().astype("<f4", copy=False).tobytes()
            )
            if not exact:
                max_abs = float((output - cpu_tensor).abs().max().item())
                raise FullDepthError(
                    f"L{self.layer} {prefix} Vulkan/CPU BF16 不等价，max_abs={max_abs}"
                )
        return output.numpy()

    def _vulkan_attention_shared_batch(
        self,
        array: Any,
        suffixes: tuple[str, str],
    ) -> dict[str, Any]:
        worker = self.attention_worker
        position = self.attention_position
        if worker is None or position is None:
            raise FullDepthError("Vulkan attention shared batch 未绑定 worker/position")
        layer_prefix = f"layers.{self.layer}.attn."
        activation_array = self._activation_quant(array)
        activation = torch.from_numpy(activation_array).to(
            dtype=torch.float32
        ).contiguous()
        assets = {
            suffix: (
                self._packed_asset(f"{layer_prefix}{suffix}.weight"),
                self._packed_asset(f"{layer_prefix}{suffix}.scale"),
            )
            for suffix in suffixes
        }
        started = time.perf_counter()
        try:
            outputs, batch_evidence = worker.execute_shared_batch(
                layer=self.layer,
                position=position,
                suffixes=suffixes,
                activation=activation,
                assets=assets,
            )
        except FullDepthPackedFp8Error as error:
            raise FullDepthError(
                f"L{self.layer} {suffixes} Vulkan attention shared batch 失败: {error}"
            ) from error
        elapsed = time.perf_counter() - started
        response_outputs = batch_evidence["outputs"]
        if tuple(outputs) != suffixes or len(response_outputs) != len(suffixes):
            raise FullDepthError("Vulkan attention shared batch 输出顺序/数量漂移")
        result: dict[str, Any] = {}
        for suffix, item in zip(suffixes, response_outputs, strict=True):
            output = outputs[suffix]
            projection_evidence = {
                **item,
                "protocol": batch_evidence["protocol"],
                "request_id": batch_evidence["request_id"],
                "layer": self.layer,
                "position": position,
                "arena_epoch": batch_evidence["arena_epoch"],
                "input": batch_evidence["input"],
                "input_sha256": batch_evidence["input_sha256"],
                "catalog_sha256": batch_evidence["catalog_sha256"],
                "gpu_slot_cache_entries": batch_evidence[
                    "gpu_slot_cache_entries"
                ],
                "activation_uploaded_bytes": batch_evidence[
                    "activation_uploaded_bytes"
                ]
                // len(suffixes),
                "batch_projection_count": len(suffixes),
                "batch_elapsed_seconds": elapsed,
                "elapsed_seconds": elapsed / len(suffixes),
            }
            if self.attention_verify_cpu:
                prefix = f"{layer_prefix}{suffix}"
                cpu_output = super()._linear_fp8(array, prefix)
                cpu_tensor = torch.from_numpy(cpu_output).to(torch.float32)
                exact = bool(torch.equal(output, cpu_tensor))
                projection_evidence["cpu_exact_bf16"] = exact
                projection_evidence["cpu_output_sha256"] = _sha256_bytes(
                    cpu_tensor.contiguous().numpy().astype("<f4", copy=False).tobytes()
                )
                if not exact:
                    max_abs = float((output - cpu_tensor).abs().max().item())
                    raise FullDepthError(
                        f"L{self.layer} {prefix} Vulkan/CPU BF16 不等价，max_abs={max_abs}"
                    )
            self.attention_vulkan_evidence.append(projection_evidence)
            result[suffix] = output.numpy()
        return result

    def _vulkan_attention_output_chain(
        self,
        array: Any,
    ) -> tuple[Any, Any]:
        worker = self.attention_worker
        position = self.attention_position
        if worker is None or position is None:
            raise FullDepthError("Vulkan attention output-chain 未绑定 worker/position")
        layer_prefix = f"layers.{self.layer}.attn."
        activation = torch.from_numpy(np.asarray(array)).to(
            dtype=torch.float32
        ).contiguous()
        assets = {
            suffix: (
                self._packed_asset(f"{layer_prefix}{suffix}.weight"),
                self._packed_asset(f"{layer_prefix}{suffix}.scale"),
            )
            for suffix in ("wo_a", "wo_b")
        }
        started = time.perf_counter()
        try:
            final_output, chain_evidence = worker.execute_output_chain(
                layer=self.layer,
                position=position,
                activation=activation,
                assets=assets,
            )
        except FullDepthPackedFp8Error as error:
            raise FullDepthError(
                f"L{self.layer} Vulkan attention output-chain 失败: {error}"
            ) from error
        elapsed = time.perf_counter() - started
        slots = chain_evidence.get("slots")
        if not isinstance(slots, list) or len(slots) != 2:
            raise FullDepthError("Vulkan attention output-chain slot 数量漂移")

        cpu_low: Any = np.zeros((1, 1, 8192), dtype=np.float32)
        cpu_final: torch.Tensor | None = None
        cpu_hashes: tuple[str, str, str] | None = None
        if self.attention_verify_cpu:
            cpu_low = super()._grouped_wo_a(array, f"{layer_prefix}wo_a")
            cpu_requantized = self._activation_quant(cpu_low)
            cpu_final_array = super()._linear_fp8(cpu_low, f"{layer_prefix}wo_b")
            cpu_final = torch.from_numpy(cpu_final_array).to(torch.float32)
            cpu_hashes = tuple(
                _sha256_bytes(
                    np.asarray(value, dtype="<f4").tobytes(order="C")
                )
                for value in (cpu_low, cpu_requantized, cpu_final_array)
            )
            observed_hashes = (
                chain_evidence.get("wo_a_output_sha256"),
                chain_evidence.get("requantized_activation_sha256"),
                chain_evidence.get("output_sha256"),
            )
            if cpu_hashes != observed_hashes or not torch.equal(final_output, cpu_final):
                max_abs = float((final_output - cpu_final).abs().max().item())
                raise FullDepthError(
                    f"L{self.layer} output-chain Vulkan/CPU 边界不等价，max_abs={max_abs}"
                )

        for index, slot in enumerate(slots):
            suffix = ("wo_a", "wo_b")[index]
            projection_evidence = {
                **slot,
                "protocol": chain_evidence["protocol"],
                "request_id": chain_evidence["request_id"],
                "layer": self.layer,
                "position": position,
                "arena_epoch": chain_evidence["arena_epoch"],
                "input": chain_evidence["input"],
                "input_sha256": chain_evidence["input_sha256"],
                "catalog_sha256": chain_evidence["catalog_sha256"],
                "gpu_slot_cache_entries": slot["gpu_slot_cache_entries"],
                "output_chain": True,
                "output_chain_stage": index,
                "output_chain_elapsed_seconds": elapsed,
                "elapsed_seconds": elapsed / 2,
                "output_sha256": (
                    chain_evidence["wo_a_output_sha256"]
                    if index == 0
                    else chain_evidence["output_sha256"]
                ),
                "requantized_activation_sha256": chain_evidence[
                    "requantized_activation_sha256"
                ],
            }
            if cpu_hashes is not None:
                projection_evidence["cpu_exact_bf16"] = True
                projection_evidence["cpu_output_sha256"] = cpu_hashes[
                    0 if suffix == "wo_a" else 2
                ]
            self.attention_vulkan_evidence.append(projection_evidence)
        return cpu_low, final_output.numpy()

    def _linear_fp8(self, array: Any, prefix: str) -> Any:
        approved = {
            "wq_a",
            "wkv",
            "wq_b",
            "indexer.wq_b",
            "wo_b",
        }
        layer_prefix = f"layers.{self.layer}.attn."
        suffix = prefix[len(layer_prefix) :] if prefix.startswith(layer_prefix) else ""
        if self.attention_worker is None or suffix not in approved:
            return super()._linear_fp8(array, prefix)
        if (
            (self.attention_shared_batch or self.attention_output_chain)
            and suffix in self._attention_batch_cache
        ):
            return self._attention_batch_cache.pop(suffix)
        batch_suffixes: tuple[str, str] | None = None
        if self.attention_shared_batch and suffix == "wq_a":
            batch_suffixes = ("wq_a", "wkv")
        elif (
            self.attention_shared_batch
            and suffix == "wq_b"
            and self.compress_ratios[self.layer] == 4
        ):
            batch_suffixes = ("wq_b", "indexer.wq_b")
        if batch_suffixes is not None:
            batch_outputs = self._vulkan_attention_shared_batch(array, batch_suffixes)
            requested = batch_outputs.pop(suffix)
            self._attention_batch_cache.update(batch_outputs)
            return requested
        return self._vulkan_attention_projection(
            array,
            prefix,
            suffix,
            activation_already_quantized=False,
        )

    def _grouped_wo_a(self, array: Any, prefix: str) -> Any:
        expected = f"layers.{self.layer}.attn.wo_a"
        if self.attention_worker is None or prefix != expected:
            return super()._grouped_wo_a(array, prefix)
        if self.attention_output_chain:
            low_output, final_output = self._vulkan_attention_output_chain(array)
            if "wo_b" in self._attention_batch_cache:
                raise FullDepthError("Vulkan attention output-chain cache 重复")
            self._attention_batch_cache["wo_b"] = final_output
            return low_output
        return self._vulkan_attention_projection(
            array,
            prefix,
            "wo_a",
            activation_already_quantized=True,
        )


@dataclass(frozen=True)
class ExecutionConfig:
    asset_root: Path = DEFAULT_ASSET_ROOT
    catalog_path: Path = DEFAULT_CATALOG
    report_path: Path = DEFAULT_REPORT
    endpoint: str = "https://huggingface.co"
    allow_fetch: bool = False
    download_budget_bytes: int = 0
    token_count: int = 1
    head_chunk_size: int = 4096
    range_attempts: int = 4
    range_workers: int = 3
    range_static_prefetch: bool = False
    range_gpu_verifier_ownership: bool = False
    forced_prefill_path: Path | None = None
    vulkan_bridge_capture: Path | None = None
    vulkan_bridge_layer: int = 42
    vulkan_writeback_worker: Path | None = None
    vulkan_writeback_timeout_seconds: float = 30.0
    checkpoint_path: Path | None = None
    resume_checkpoint_path: Path | None = None
    vulkan_writeback_all_layers: bool = False
    vulkan_writeback_verify_cpu: bool = True
    vulkan_writeback_cpu_fallback: bool = True
    vulkan_writeback_fast_production: bool = False
    vulkan_writeback_batch_verify_payloads: bool = False
    vulkan_writeback_inline_manifest: bool = False
    vulkan_final_head_worker: Path | None = None
    vulkan_final_head_timeout_seconds: float = 60.0
    vulkan_final_head_scratch: Path | None = None
    vulkan_final_head_validate_cpu_once: bool = False
    vulkan_attention_worker: Path | None = None
    vulkan_attention_timeout_seconds: float = 60.0
    vulkan_attention_scratch: Path | None = None
    vulkan_attention_verify_cpu: bool = False
    vulkan_attention_shared_batch: bool = False
    vulkan_attention_output_chain: bool = False

    def validate(self) -> None:
        if not self.endpoint.startswith("https://"):
            raise FullDepthError("endpoint 必须是 HTTPS")
        if self.allow_fetch != (self.download_budget_bytes > 0):
            raise FullDepthError("下载授权与正数 budget 必须同时存在或同时缺席")
        if not 1 <= self.token_count <= s14.MAX_RUNTIME_POSITIONS:
            raise FullDepthError(
                f"token_count 必须在 1..{s14.MAX_RUNTIME_POSITIONS}"
            )
        if self.head_chunk_size <= 0:
            raise FullDepthError("head_chunk_size 必须为正整数")
        if self.range_attempts <= 0 or not 1 <= self.range_workers <= 8:
            raise FullDepthError("range_attempts/workers 必须分别为正数和 1..8")
        if self.vulkan_bridge_layer not in FULLDEPTH43_NATIVE_TOP6.layers:
            raise FullDepthError("Vulkan bridge layer 必须位于 0..42")
        if (
            self.vulkan_bridge_capture is not None
            and self.token_count != 1
            and not self.vulkan_writeback_all_layers
        ):
            raise FullDepthError("Vulkan bridge capture 当前只允许单 token")
        if self.vulkan_writeback_worker is not None:
            if self.vulkan_bridge_capture is None:
                raise FullDepthError("Vulkan writeback 必须同时指定 bridge capture")
            if not self.vulkan_writeback_worker.resolve().is_file():
                raise FullDepthError("Vulkan writeback worker 不存在")
        if self.vulkan_writeback_timeout_seconds <= 0:
            raise FullDepthError("Vulkan writeback timeout 必须为正数")
        if self.resume_checkpoint_path is not None and self.forced_prefill_path is not None:
            raise FullDepthError("resume checkpoint 已包含 forced cursor，禁止再注入 forced-prefill")
        if self.resume_checkpoint_path is not None and self.vulkan_bridge_capture is not None:
            raise FullDepthError("resume checkpoint 不能重放 position0 Vulkan capture")
        if self.checkpoint_path is not None and self.checkpoint_path.resolve() == self.report_path.resolve():
            raise FullDepthError("checkpoint manifest 不得覆盖 execution report")
        if self.vulkan_writeback_all_layers and self.vulkan_writeback_worker is None:
            raise FullDepthError("全层 Vulkan writeback 必须指定 worker")
        if self.vulkan_writeback_fast_production and self.vulkan_writeback_verify_cpu:
            raise FullDepthError("fast production 累加顺序不适用逐 BF16 CPU 审计")
        if self.vulkan_writeback_batch_verify_payloads and (
            self.vulkan_writeback_worker is None
            or not self.vulkan_writeback_all_layers
            or not self.vulkan_writeback_fast_production
        ):
            raise FullDepthError(
                "batch payload 验证要求全层 fast-production Vulkan MoE worker"
            )
        if self.vulkan_writeback_inline_manifest and (
            self.vulkan_writeback_worker is None
            or not self.vulkan_writeback_all_layers
            or not self.vulkan_writeback_fast_production
        ):
            raise FullDepthError(
                "inline manifest 要求全层 fast-production Vulkan MoE worker"
            )
        if (
            self.vulkan_final_head_worker is not None
            and not self.vulkan_final_head_worker.resolve().is_file()
        ):
            raise FullDepthError("Vulkan final-head worker 不存在")
        if self.vulkan_final_head_timeout_seconds <= 0:
            raise FullDepthError("Vulkan final-head timeout 必须为正数")
        if (
            self.vulkan_attention_worker is not None
            and not self.vulkan_attention_worker.resolve().is_file()
        ):
            raise FullDepthError("Vulkan attention worker 不存在")
        if self.vulkan_attention_timeout_seconds <= 0:
            raise FullDepthError("Vulkan attention timeout 必须为正数")
        if self.range_gpu_verifier_ownership:
            if self.allow_fetch or self.download_budget_bytes != 0:
                raise FullDepthError(
                    "GPU verifier ownership 仅允许零下载的既有 Range cache hit"
                )
            if self.vulkan_attention_worker is None:
                raise FullDepthError(
                    "GPU verifier ownership 要求全层 Vulkan attention worker"
                )
            if self.vulkan_attention_verify_cpu:
                raise FullDepthError(
                    "GPU verifier ownership 禁止 Vulkan attention CPU verify"
                )
            if (
                self.vulkan_writeback_worker is None
                or not self.vulkan_writeback_all_layers
                or not self.vulkan_writeback_fast_production
            ):
                raise FullDepthError(
                    "GPU verifier ownership 要求全层 fast-production Vulkan MoE"
                )
            if self.vulkan_writeback_verify_cpu or self.vulkan_writeback_cpu_fallback:
                raise FullDepthError(
                    "GPU verifier ownership 禁止 MoE CPU verify/fallback"
                )

    def resolved_vulkan_final_head_worker(self) -> Path:
        worker = self.vulkan_final_head_worker or DEFAULT_VULKAN_FINAL_HEAD_WORKER
        resolved = worker.resolve()
        if not resolved.is_file():
            raise FullDepthError(
                "生产路径要求已编译的 Vulkan final-head worker；"
                f"缺少 {resolved}"
            )
        return resolved

    def resolved_vulkan_final_head_scratch(self) -> Path:
        scratch = self.vulkan_final_head_scratch
        if scratch is None:
            scratch = self.asset_root / "runtime" / "vulkan_final_head"
        return scratch.resolve()

    def resolved_vulkan_attention_scratch(self) -> Path:
        scratch = self.vulkan_attention_scratch
        if scratch is None:
            scratch = self.asset_root / "runtime" / "vulkan_attention"
        return scratch.resolve()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _gpu_verifier_receipt_closure(
    report: Mapping[str, Any],
    range_telemetry: object,
) -> dict[str, Any]:
    """把 Python 延迟所有权逐项闭合到 Rust 计算前强验证回执。"""

    observed = {
        "vulkan_attention_worker": {"ranges": 0, "bytes": 0},
        "vulkan_moe_worker": {"ranges": 0, "bytes": 0},
    }
    receipt_count = 0
    invalid_receipts: list[dict[str, Any]] = []
    observed_identity_sums = {owner: 0 for owner in observed}

    def is_u256_hex(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def consume(owner: str, receipt: object, *, position: Any, layer: Any) -> None:
        nonlocal receipt_count
        receipt_count += 1
        if not isinstance(receipt, Mapping):
            invalid_receipts.append(
                {"owner": owner, "position": position, "layer": layer, "reason": "missing"}
            )
            return
        count = receipt.get("verified_count")
        byte_count = receipt.get("verified_bytes")
        identity = receipt.get("payload_identity_sha256")
        python_expected_identity = receipt.get(
            "python_expected_payload_identity_sha256"
        )
        deferred_identity_sum = receipt.get(
            "python_deferred_identity_multiset_sum_u256"
        )
        if (
            receipt.get("verification_owner") != "rust_vulkan_worker"
            or receipt.get("verified_before_compute") is not True
            or receipt.get("payload_identity_contract") != PAYLOAD_IDENTITY_CONTRACT
            or receipt.get("verification_scope") != PAYLOAD_VERIFICATION_SCOPE
            or not isinstance(identity, str)
            or len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
            or identity != python_expected_identity
            or receipt.get("python_deferred_identity_multiset_contract")
            != online_range.DEFERRED_IDENTITY_MULTISET_CONTRACT
            or not is_u256_hex(deferred_identity_sum)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            invalid_receipts.append(
                {"owner": owner, "position": position, "layer": layer, "reason": "contract"}
            )
            return
        observed[owner]["ranges"] += count
        observed[owner]["bytes"] += byte_count
        assert isinstance(deferred_identity_sum, str)
        observed_identity_sums[owner] = (
            observed_identity_sums[owner] + int(deferred_identity_sum, 16)
        ) % online_range.DEFERRED_IDENTITY_MULTISET_MODULUS

    for token in report.get("tokens", ()):
        if not isinstance(token, Mapping):
            continue
        position = token.get("position")
        for layer_row in token.get("layers", ()):
            if not isinstance(layer_row, Mapping):
                continue
            layer = layer_row.get("layer")
            for projection in layer_row.get("vulkan_attention", ()):
                consume(
                    "vulkan_attention_worker",
                    projection,
                    position=position,
                    layer=layer,
                )
            writeback = layer_row.get("vulkan_writeback")
            receipt = (
                writeback.get("payload_verification")
                if isinstance(writeback, Mapping)
                else None
            )
            consume(
                "vulkan_moe_worker",
                receipt,
                position=position,
                layer=layer,
            )

    telemetry_errors: list[str] = []
    if not isinstance(range_telemetry, Mapping):
        raw_expected: object = None
        telemetry_errors.append("range_proof_cache missing_or_not_mapping")
    else:
        raw_expected = range_telemetry.get("deferred_by_owner")
        if not isinstance(raw_expected, Mapping):
            telemetry_errors.append("deferred_by_owner missing_or_not_mapping")
        elif set(raw_expected) != set(observed):
            telemetry_errors.append("deferred_by_owner owner_set_drift")
        if (
            range_telemetry.get("deferred_identity_multiset_contract")
            != online_range.DEFERRED_IDENTITY_MULTISET_CONTRACT
        ):
            telemetry_errors.append("deferred_identity_multiset_contract drift")

    def expected_stat(owner: str, key: str) -> int:
        if not isinstance(raw_expected, Mapping):
            return -1
        row = raw_expected.get(owner)
        if not isinstance(row, Mapping):
            telemetry_errors.append(f"{owner} missing_or_not_mapping")
            return -1
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            telemetry_errors.append(f"{owner}.{key} invalid")
            return -1
        return value

    expected = {
        owner: {
            "ranges": expected_stat(owner, "ranges"),
            "bytes": expected_stat(owner, "bytes"),
        }
        for owner in observed
    }

    def expected_identity_sum(owner: str) -> int:
        if not isinstance(raw_expected, Mapping):
            return -1
        row = raw_expected.get(owner)
        if not isinstance(row, Mapping):
            return -1
        value = row.get("identity_multiset_sum_u256")
        if not is_u256_hex(value):
            telemetry_errors.append(f"{owner}.identity_multiset_sum_u256 invalid")
            return -1
        assert isinstance(value, str)
        return int(value, 16)

    expected_identity_sums = {
        owner: expected_identity_sum(owner) for owner in observed
    }
    expected_range_total = sum(row["ranges"] for row in expected.values())
    expected_byte_total = sum(row["bytes"] for row in expected.values())
    expected_global_identity_sum = (
        sum(expected_identity_sums.values())
        % online_range.DEFERRED_IDENTITY_MULTISET_MODULUS
        if all(value >= 0 for value in expected_identity_sums.values())
        else -1
    )
    if isinstance(range_telemetry, Mapping):
        top_ranges = range_telemetry.get("deferred")
        top_bytes = range_telemetry.get("bytes_deferred")
        top_identity_sum = range_telemetry.get(
            "deferred_identity_multiset_sum_u256"
        )
        if (
            isinstance(top_ranges, bool)
            or not isinstance(top_ranges, int)
            or top_ranges < 0
            or top_ranges != expected_range_total
        ):
            telemetry_errors.append("deferred top_level_total_drift")
        if (
            isinstance(top_bytes, bool)
            or not isinstance(top_bytes, int)
            or top_bytes < 0
            or top_bytes != expected_byte_total
        ):
            telemetry_errors.append("bytes_deferred top_level_total_drift")
        if (
            not is_u256_hex(top_identity_sum)
            or expected_global_identity_sum < 0
            or int(top_identity_sum, 16) != expected_global_identity_sum
        ):
            telemetry_errors.append(
                "deferred_identity_multiset_sum_u256 top_level_drift"
            )

    closed = (
        not invalid_receipts
        and not telemetry_errors
        and observed == expected
        and observed_identity_sums == expected_identity_sums
    )
    return {
        "closed": closed,
        "receipt_count": receipt_count,
        "expected_deferred": expected,
        "rust_verified_before_compute": observed,
        "invalid_receipts": invalid_receipts,
        "telemetry_errors": telemetry_errors,
        "expected_deferred_identity_multiset_sum_u256": {
            owner: (None if value < 0 else f"{value:064x}")
            for owner, value in expected_identity_sums.items()
        },
        "python_receipt_identity_multiset_sum_u256": {
            owner: f"{value:064x}"
            for owner, value in observed_identity_sums.items()
        },
        "contract": (
            "Rust receipt identities first equal Python request identities; then "
            "sum_mod_2^256(Python request deferred payload identities) and "
            "count/bytes equal the independent Python RangeCache owner ledgers"
        ),
    }


def _bridge_payload_entry(
    cached: online_range.CachedRange,
    *,
    kind: str,
    expert_id: int | None,
    gpu_verifier_ownership: bool = False,
) -> dict[str, Any]:
    entry = dict(cached.entry)
    tensor = entry.get("tensor")
    observed = cached.proof.get("observed_sha256")
    path = cached.path.resolve()
    if not isinstance(tensor, str) or not tensor.startswith("layers."):
        raise FullDepthError("Vulkan bridge 只接受已命名的真实层 payload")
    if not isinstance(observed, str) or len(observed) != 64:
        raise FullDepthError(f"Vulkan bridge payload 缺少 SHA proof: {tensor}")
    if not path.is_file() or path.stat().st_size != entry.get("bytes"):
        raise FullDepthError(f"Vulkan bridge payload 字节漂移: {tensor}")
    if gpu_verifier_ownership:
        if cached.content_verified or cached.verification_owner != "vulkan_moe_worker":
            raise FullDepthError(
                f"{tensor} 未按合同交给 Vulkan MoE worker 做唯一内容验证"
            )
    elif not cached.content_verified:
        raise FullDepthError(
            f"{tensor} 内容尚未验证，禁止进入未启用 GPU verifier ownership 的路径"
        )
    return {
        "tensor": tensor,
        "kind": kind,
        "expert_id": expert_id,
        "dtype": entry.get("dtype"),
        "shape": entry.get("shape"),
        "bytes": entry.get("bytes"),
        "path": str(path),
        "sha256": observed,
        "cache_hit": cached.cache_hit,
        "hash_authority": cached.proof.get("hash_authority"),
    }


@dataclass(frozen=True)
class VulkanBridgeCapture:
    """单层 Vulkan 回写输入；正式路径的 manifest 只保留在内存。"""

    capture_root: Path
    manifest: Mapping[str, Any]
    report: Mapping[str, Any]


def _write_vulkan_bridge_capture(
    capture_dir: Path,
    *,
    layer: int,
    position: int,
    token_id: int,
    completed_layers: Sequence[int],
    pending: s14.PendingLayer,
    routed: RoutedLayer,
    kernel: FullDepthNativeLayerReference,
    profile: ExecutionProfile,
    gpu_verifier_ownership: bool = False,
    persist_manifest: bool = True,
) -> VulkanBridgeCapture:
    capture_dir = capture_dir.resolve()
    if capture_dir.exists():
        raise FullDepthError(f"Vulkan bridge capture 目录已存在: {capture_dir}")
    if list(completed_layers) != list(range(layer)):
        raise FullDepthError("Vulkan bridge capture 前缀不是连续真实 FullDepth 层")
    if tuple(pending.route_ids) != routed.expert_ids or len(set(pending.route_ids)) != profile.top_k:
        raise FullDepthError("Vulkan bridge route 与已获取 payload 不一致")

    capture_dir.mkdir(parents=True, exist_ok=False)
    raw_ffn = pending.ffn_input.float().contiguous().numpy().astype("<f4", copy=False)
    quantized = kernel._activation_quant(raw_ffn).astype("<f4", copy=False)
    input_bytes = quantized.tobytes(order="C")
    input_file = "ffn_input_activation_quant.f32le.bin"
    temporary = capture_dir / (input_file + ".tmp")
    temporary.write_bytes(input_bytes)
    os.replace(temporary, capture_dir / input_file)

    payloads: list[dict[str, Any]] = []
    for expert_id in pending.route_ids:
        pages = routed.experts[expert_id]
        if len(pages) != 6:
            raise FullDepthError(f"Vulkan bridge E{expert_id} 不是完整 6 payload")
        payloads.extend(
            _bridge_payload_entry(
                page,
                kind="routed",
                expert_id=expert_id,
                gpu_verifier_ownership=gpu_verifier_ownership,
            )
            for page in pages
        )
    if len(routed.shared) != 6:
        raise FullDepthError("Vulkan bridge shared expert 不是完整 6 payload")
    payloads.extend(
        _bridge_payload_entry(
            page,
            kind="shared",
            expert_id=None,
            gpu_verifier_ownership=gpu_verifier_ownership,
        )
        for page in routed.shared
    )
    if len(payloads) != 42:
        raise FullDepthError("Vulkan bridge top6+shared 必须精确包含 42 payload")

    document = {
        "format": VULKAN_BRIDGE_FORMAT,
        "revision": profile.revision,
        "profile": profile.profile_id,
        "layer": layer,
        "position": position,
        "input_token_id": token_id,
        "completed_layers_before_capture": list(completed_layers),
        "route_source": pending.route_source,
        "expert_ids": pending.route_ids,
        "route_weights": pending.route_weights,
        "route_weight_sum": float(sum(pending.route_weights)),
        "source_ffn_input_f32_le_sha256": _sha256_bytes(raw_ffn.tobytes(order="C")),
        "input": {
            "name": "ffn_input_activation_quant",
            "file": input_file,
            "shape": list(quantized.shape),
            "bytes": len(input_bytes),
            "f32_le_sha256": _sha256_bytes(input_bytes),
        },
        "payload_count": len(payloads),
        "payload_bytes": sum(int(entry["bytes"]) for entry in payloads),
        "payloads": payloads,
        "reference_semantics": (
            "FullDepth43 live L{layer} post-attention/HC/RMSNorm activation, then official "
            "activation quantization as the input to the bounded Vulkan top6+shared "
            "packed chain with official BF16 boundaries, route-weight-before-w2 and "
            "E4M3FN group-128 activation requantization."
        ).format(layer=layer),
    }
    manifest_json = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    manifest_sha256 = _sha256_bytes(manifest_json.encode("utf-8", errors="strict"))
    manifest_path: Path | None = None
    if persist_manifest:
        manifest_path = capture_dir / "bridge_manifest.json"
        write_json(manifest_path, document)
        manifest_sha256 = _sha256_bytes(manifest_path.read_bytes())
    report = {
        "manifest": None if manifest_path is None else str(manifest_path),
        "manifest_transport": "capture_file" if persist_manifest else "inline_json",
        "manifest_sha256": manifest_sha256,
        "capture_root": str(capture_dir),
        "input_f32_le_sha256": document["input"]["f32_le_sha256"],
        "source_ffn_input_f32_le_sha256": document["source_ffn_input_f32_le_sha256"],
        "payload_count": len(payloads),
        "payload_bytes": document["payload_bytes"],
    }
    return VulkanBridgeCapture(
        capture_root=capture_dir,
        manifest=document,
        report=report,
    )


@dataclass
class DecoderState:
    position: int = 0
    input_token_id: int = s14.BOS_TOKEN_ID
    layer_states: Mapping[int, s14.LayerRuntimeState] = field(default_factory=dict)
    committed_tokens: list[dict[str, Any]] = field(default_factory=list)
    forced_queue: s14.ForcedTokenQueue | None = None

    def __post_init__(self) -> None:
        if self.forced_queue is not None:
            self.forced_queue.validate()
            if self.forced_queue.active and self.input_token_id != self.forced_queue.current_token_id:
                raise FullDepthError("decoder input_token_id 与 forced-prefill cursor 漂移")

    def previous_for(self, profile: ExecutionProfile) -> Mapping[int, s14.LayerRuntimeState]:
        if self.position == 0:
            if self.layer_states:
                raise FullDepthError("position0 不得携带旧 KV/compressor state")
        elif set(self.layer_states) != set(profile.layers):
            raise FullDepthError("decode token 缺少 43 层 KV/compressor state")
        return {
            layer: s14._clone_layer_state(state)
            for layer, state in self.layer_states.items()
        }

    def clone(self) -> "DecoderState":
        """深拷贝可变 tensor/ledger，用于 checkpoint-before-commit。"""

        return DecoderState(
            position=self.position,
            input_token_id=self.input_token_id,
            layer_states={
                layer: s14._clone_layer_state(layer_state)
                for layer, layer_state in self.layer_states.items()
            },
            committed_tokens=[dict(row) for row in self.committed_tokens],
            forced_queue=self.forced_queue,
        )

    def commit(
        self,
        *,
        output_token_id: int,
        next_states: Mapping[int, s14.LayerRuntimeState],
        profile: ExecutionProfile,
    ) -> None:
        if set(next_states) != set(profile.layers):
            raise FullDepthError("禁止提交跳层 state")
        if isinstance(output_token_id, bool) or not 0 <= output_token_id < s14.VOCAB_SIZE:
            raise FullDepthError("禁止提交假 token/越界 token")
        for layer in profile.layers:
            state = next_states[layer]
            if state.layer != layer or state.position != self.position:
                raise FullDepthError(f"L{layer} runtime state 的 layer/position 漂移")

        next_input_token_id = output_token_id
        next_forced_queue = self.forced_queue
        committed: dict[str, Any] = {
            "position": self.position,
            "input_token_id": self.input_token_id,
            "output_token_id": output_token_id,
        }
        if self.forced_queue is not None and self.forced_queue.active:
            next_forced_queue = s14.ForcedTokenQueue(
                token_ids=self.forced_queue.token_ids,
                cursor=self.forced_queue.cursor + 1,
                artifact_sha256=self.forced_queue.artifact_sha256,
            )
            next_forced_queue.validate()
            if next_forced_queue.active:
                next_input_token_id = next_forced_queue.current_token_id
            committed.update(
                {
                    "input_source": "forced_prefill",
                    "forced_cursor": self.forced_queue.cursor,
                    "next_input_token_id": next_input_token_id,
                }
            )

        # 所有验证、cursor 推演和状态复制完成后，才一次替换整组提交字段。
        next_values = {
            "position": self.position + 1,
            "input_token_id": next_input_token_id,
            "layer_states": dict(next_states),
            "committed_tokens": [*self.committed_tokens, committed],
            "forced_queue": next_forced_queue,
        }
        self.__dict__.update(next_values)


def _load_or_build_catalog(config: ExecutionConfig) -> dict[str, Any]:
    if config.catalog_path.is_file():
        catalog = read_json(config.catalog_path)
    else:
        catalog = build_catalog(asset_root=config.asset_root)
        write_json(config.catalog_path, catalog)
    validate_catalog(catalog)
    return catalog


@dataclass(frozen=True)
class FullDepthTokenComputation:
    """One uncommitted FullDepth43 token result.

    The worker computes the complete 43-layer native-top6 path and final
    argmax, but deliberately does not mutate ``DecoderState``.  Callers must
    explicitly commit ``next_layer_states`` after validating the whole
    result.  This is the shared boundary used by the report runner and the
    causal-block snapshot bridge.
    """

    predicted_token_id: int
    next_layer_states: Mapping[int, s14.LayerRuntimeState]
    top6_by_layer: Mapping[int, tuple[int, ...]]
    value: Mapping[str, Any]


class FullDepthTokenWorker:
    """Reusable, fail-before-commit FullDepth43 single-token worker.

    This is still the CPU/PyTorch correctness implementation.  Calling it K
    times through ``CpuCausalBlockReferenceBackend`` is *not* a batched model
    forward and carries no speed claim.
    """

    def __init__(
        self,
        config: ExecutionConfig,
        catalog: Mapping[str, Any],
        cache: online_range.RangeCache,
        *,
        profile: ExecutionProfile = FULLDEPTH43_NATIVE_TOP6,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        config.validate()
        profile.validate()
        validate_catalog(catalog)
        self.config = config
        self.catalog = catalog
        self.cache = cache
        self.profile = profile
        self.progress_callback = progress_callback
        self.stage = "idle"
        self.current_layer: int | None = None
        self.last_token_report: dict[str, Any] | None = None
        self.writeback_layers: list[int] = []
        self.writeback_fallbacks: list[dict[str, Any]] = []
        self._final_head: FullDepthVulkanFinalHead | None = None
        self._writeback: PersistentVulkanWriteback | None = None
        self._attention: PersistentFullDepthPackedFp8Attention | None = None
        self._attention_arena_path: Path | None = None
        self._range_pool: ThreadPoolExecutor | None = None
        self._range_prefetch_pool: ThreadPoolExecutor | None = None
        self._active_range_session: FullDepthRangeSession | None = None
        self._token_call_lock = threading.RLock()
        self.range_cleanup_errors: list[dict[str, str]] = []
        self.attention_layers: list[int] = []
        self.attention_projection_count = 0
        self._started = False
        self._closed = False

    @property
    def writeback_hello(self) -> Mapping[str, Any] | None:
        return None if self._writeback is None else self._writeback.hello

    @property
    def final_head_hello(self) -> Mapping[str, Any] | None:
        return None if self._final_head is None else self._final_head.hello

    @property
    def attention_hello(self) -> Mapping[str, Any] | None:
        return None if self._attention is None else self._attention.hello

    def start(self) -> None:
        with self._token_call_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._closed:
            raise FullDepthError("FullDepth token worker 已关闭")
        if self._started:
            return
        if self.config.vulkan_attention_worker is not None:
            self.stage = "vulkan_attention_worker_start"
            scratch = self.config.resolved_vulkan_attention_scratch()
            scratch.mkdir(parents=True, exist_ok=True)
            descriptor, arena_name = tempfile.mkstemp(
                prefix="fulldepth43-attention-",
                suffix=".bin",
                dir=scratch,
            )
            os.close(descriptor)
            arena_path = Path(arena_name).resolve()
            try:
                with arena_path.open("r+b") as stream:
                    stream.truncate(512 * 1024)
                arena = FullDepthPackedFp8Arena(arena_path)
                self._attention = PersistentFullDepthPackedFp8Attention(
                    (
                        str(self.config.vulkan_attention_worker.resolve()),
                        FULLDEPTH_FP8_ATTENTION_WORKER_ARG,
                    ),
                    arena,
                    timeout_seconds=self.config.vulkan_attention_timeout_seconds,
                )
            except Exception:
                arena_path.unlink(missing_ok=True)
                raise
            self._attention_arena_path = arena_path
        if self.config.vulkan_writeback_worker is not None:
            self.stage = "vulkan_writeback_worker_start"
            worker_arg = (
                "--fulldepth43-production-worker"
                if self.config.vulkan_writeback_fast_production
                else "--fulldepth43-writeback-worker"
            )
            self._writeback = PersistentVulkanWriteback(
                (
                    str(self.config.vulkan_writeback_worker.resolve()),
                    worker_arg,
                ),
                timeout_seconds=self.config.vulkan_writeback_timeout_seconds,
                batch_verify_payloads=(
                    self.config.vulkan_writeback_batch_verify_payloads
                ),
            )
            expected_mode = (
                "fast_production_128_lane"
                if self.config.vulkan_writeback_fast_production
                else "exact_audit_12_lane_mxfp4_8_lane_fp8"
            )
            if self._writeback.hello.get("numeric_mode") != expected_mode:
                self._writeback.close()
                self._writeback = None
                raise FullDepthError(
                    f"Vulkan worker numeric_mode 漂移，期望 {expected_mode}"
                )
        if self.config.range_static_prefetch:
            self._range_pool = ThreadPoolExecutor(
                max_workers=self.config.range_workers,
                thread_name_prefix="fd43-range-persistent",
            )
            self._range_prefetch_pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="fd43-static-prefetch",
            )
        self._started = True

    def close(self) -> None:
        with self._token_call_lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None

        def cleanup(action: Callable[[], None]) -> None:
            nonlocal first_error
            try:
                action()
            except BaseException as error:
                if first_error is None:
                    first_error = error

        if self._active_range_session is not None:
            cleanup(self._active_range_session.close)
            self._active_range_session = None
        if self._range_prefetch_pool is not None:
            cleanup(
                lambda: self._range_prefetch_pool.shutdown(
                    wait=True,
                    cancel_futures=True,
                )
            )
            self._range_prefetch_pool = None
        if self._range_pool is not None:
            cleanup(
                lambda: self._range_pool.shutdown(wait=True, cancel_futures=True)
            )
            self._range_pool = None
        if self._writeback is not None:
            cleanup(self._writeback.close)
        if self._final_head is not None:
            cleanup(self._final_head.close)
        if self._attention is not None:
            cleanup(self._attention.close)
        if self._attention_arena_path is not None:
            cleanup(lambda: self._attention_arena_path.unlink(missing_ok=True))
        if first_error is not None:
            raise first_error

    def _notify(self, token_report: dict[str, Any]) -> None:
        self.last_token_report = token_report
        if self.progress_callback is not None:
            self.progress_callback(token_report)

    def _validate_previous(
        self,
        position: int,
        previous: Mapping[int, s14.LayerRuntimeState],
    ) -> dict[int, s14.LayerRuntimeState]:
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise FullDepthError("FullDepth worker position 非法")
        if position == 0:
            if previous:
                raise FullDepthError("FullDepth worker position0 不得携带旧状态")
        elif set(previous) != set(self.profile.layers):
            raise FullDepthError("FullDepth worker 非零 position 缺少 43 层状态")
        cloned: dict[int, s14.LayerRuntimeState] = {}
        for layer, layer_state in previous.items():
            if layer_state.layer != layer or layer_state.position != position - 1:
                raise FullDepthError(f"FullDepth worker L{layer} 历史状态漂移")
            cloned[layer] = s14._clone_layer_state(layer_state)
        return cloned

    def __call__(
        self,
        position: int,
        input_token_id: int,
        previous: Mapping[int, s14.LayerRuntimeState],
    ) -> FullDepthTokenComputation:
        if not self._token_call_lock.acquire(blocking=False):
            raise FullDepthError("FullDepth token worker 不允许并发 token session")
        try:
            return self._call_one(position, input_token_id, previous)
        finally:
            self._token_call_lock.release()

    def _call_one(
        self,
        position: int,
        input_token_id: int,
        previous: Mapping[int, s14.LayerRuntimeState],
    ) -> FullDepthTokenComputation:
        self.start()
        if (
            isinstance(input_token_id, bool)
            or not isinstance(input_token_id, int)
            or not 0 <= input_token_id < s14.VOCAB_SIZE
        ):
            raise FullDepthError("FullDepth worker input token ID 越界")
        private_previous = self._validate_previous(position, previous)
        if self._active_range_session is not None:
            raise FullDepthError("FullDepth token worker 不允许并发 token session")
        session = FullDepthRangeSession(
            self.catalog,
            self.cache,
            profile=self.profile,
            range_attempts=self.config.range_attempts,
            range_workers=self.config.range_workers,
            range_pool=self._range_pool,
            prefetch_pool=self._range_prefetch_pool,
            owns_pools=False,
        )
        self._active_range_session = session
        try:
            result = self._compute_token(
                position,
                input_token_id,
                private_previous,
                session,
            )
        except BaseException:
            try:
                session.close()
            except BaseException as cleanup_error:
                self.range_cleanup_errors.append(
                    {
                        "type": type(cleanup_error).__name__,
                        "message": str(cleanup_error),
                    }
                )
            raise
        else:
            session.close()
            return result
        finally:
            self._active_range_session = None

    def _compute_token(
        self,
        position: int,
        input_token_id: int,
        private_previous: Mapping[int, s14.LayerRuntimeState],
        session: FullDepthRangeSession,
    ) -> FullDepthTokenComputation:
        token_report: dict[str, Any] = {
            "position": position,
            "input_token_id": input_token_id,
            "completed_layers": [],
            "layers": [],
            "final": None,
            "state_committed": False,
            "range_static_prefetch": session.prefetch_events,
        }
        self.current_layer = None
        self._notify(token_report)

        self.stage = f"position_{position}_embedding"
        embedding = session.prepare_embedding_row(input_token_id)
        state = s14._initial_state(embedding, input_token_id)
        next_states: dict[int, s14.LayerRuntimeState] = {}
        top6_by_layer: dict[int, tuple[int, ...]] = {}
        for layer in self.profile.layers:
            layer_started = time.perf_counter()
            self.current_layer = layer
            self.stage = f"position_{position}_layer_{layer}_base"
            prerequisites = session.prepare_layer(layer, input_token_id)
            store = s14.TensorStore(self.config.asset_root.resolve() / "range_cache")
            store.add_ranges((*prerequisites.non_expert, *prerequisites.router))
            kernel = FullDepthNativeLayerReference(
                layer,
                store,
                profile=self.profile,
                attention_worker=self._attention,
                attention_position=position if self._attention is not None else None,
                attention_verify_cpu=self.config.vulkan_attention_verify_cpu,
                attention_shared_batch=self.config.vulkan_attention_shared_batch,
                attention_output_chain=self.config.vulkan_attention_output_chain,
                gpu_verifier_ownership=self.config.range_gpu_verifier_ownership,
            )

            self.stage = f"position_{position}_layer_{layer}_native_route"
            pending = kernel.prepare_route(
                state,
                token_id=input_token_id,
                position=position,
                previous_runtime=private_previous.get(layer),
            )
            route = session.submit_top6(layer, input_token_id, pending.route_ids)
            top6_by_layer[layer] = route

            self.stage = f"position_{position}_layer_{layer}_top6_shared"
            routed = session.fetch_routed(layer, input_token_id)
            next_index = session.layer_index + 1
            if next_index < len(self.profile.layers):
                session.schedule_next_static(
                    self.profile.layers[next_index],
                    input_token_id,
                )
            pages = list(routed.shared)
            for expert_id in pending.route_ids:
                pages.extend(routed.experts[expert_id])
            kernel.add_routed(pages)
            writeback_requested = self._writeback is not None and (
                self.config.vulkan_writeback_all_layers
                or (position == 0 and layer == self.config.vulkan_bridge_layer)
            )
            capture_requested = (
                self.config.vulkan_bridge_capture is not None
                and (
                    (self.config.vulkan_writeback_all_layers and writeback_requested)
                    or (position == 0 and layer == self.config.vulkan_bridge_layer)
                )
            )
            bridge_capture = None
            if capture_requested:
                capture_dir = self.config.vulkan_bridge_capture
                assert capture_dir is not None
                if self.config.vulkan_writeback_all_layers:
                    capture_dir = (
                        capture_dir
                        / f"position-{position:06d}"
                        / f"layer-{layer:02d}"
                    )
                bridge_capture = _write_vulkan_bridge_capture(
                    capture_dir,
                    layer=layer,
                    position=position,
                    token_id=input_token_id,
                    completed_layers=token_report["completed_layers"],
                    pending=pending,
                    routed=routed,
                    kernel=kernel,
                    profile=self.profile,
                    gpu_verifier_ownership=self.config.range_gpu_verifier_ownership,
                    persist_manifest=not (
                        writeback_requested
                        and self.config.vulkan_writeback_inline_manifest
                    ),
                )
            cpu_moe_branch: torch.Tensor | None = None
            cpu_state: torch.Tensor | None = None
            if not writeback_requested or self.config.vulkan_writeback_verify_cpu:
                cpu_moe_branch, cpu_state = kernel.finish_layer(pending)
            writeback_evidence = None
            if writeback_requested:
                if bridge_capture is None:
                    raise FullDepthError("Vulkan writeback 缺少同层 capture")
                self.stage = f"position_{position}_layer_{layer}_vulkan_writeback"
                assert self._writeback is not None
                try:
                    if self.config.vulkan_writeback_inline_manifest:
                        vulkan_moe_branch, worker_evidence = self._writeback.execute(
                            bridge_capture.manifest,
                            capture_root=bridge_capture.capture_root,
                        )
                    else:
                        manifest_path = bridge_capture.report.get("manifest")
                        if not isinstance(manifest_path, str):
                            raise FullDepthError("文件 manifest 路径缺失")
                        vulkan_moe_branch, worker_evidence = self._writeback.execute(
                            Path(manifest_path)
                        )
                except VulkanWritebackError as error:
                    if not self.config.vulkan_writeback_cpu_fallback:
                        raise FullDepthError(
                            f"L{layer} Vulkan writeback 失败且禁止 CPU fallback: {error}"
                        ) from error
                    if cpu_moe_branch is None or cpu_state is None:
                        cpu_moe_branch, cpu_state = kernel.finish_layer(pending)
                    fallback = {
                        "position": position,
                        "layer": layer,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                    self.writeback_fallbacks.append(fallback)
                    self._writeback = None
                    moe_branch, state = cpu_moe_branch, cpu_state
                    writeback_evidence = {
                        "status": "cpu_fallback_after_vulkan_failure",
                        "failure": fallback,
                        "state_source": "cpu_reference",
                    }
                else:
                    comparison = None
                    verification_error: VulkanWritebackError | None = None
                    if self.config.vulkan_writeback_verify_cpu:
                        if cpu_moe_branch is None or cpu_state is None:
                            raise FullDepthError("Vulkan verify 模式缺少 CPU reference")
                        try:
                            comparison = verify_exact_bf16_writeback(
                                cpu_moe_branch,
                                vulkan_moe_branch,
                            )
                        except VulkanWritebackError as error:
                            verification_error = error
                    if verification_error is not None:
                        if not self.config.vulkan_writeback_cpu_fallback:
                            raise verification_error
                        assert cpu_moe_branch is not None and cpu_state is not None
                        fallback = {
                            "position": position,
                            "layer": layer,
                            "type": type(verification_error).__name__,
                            "message": str(verification_error),
                        }
                        self.writeback_fallbacks.append(fallback)
                        moe_branch, state = cpu_moe_branch, cpu_state
                        writeback_evidence = {
                            **worker_evidence,
                            "status": "cpu_fallback_after_exact_verification_failure",
                            "failure": fallback,
                            "cpu_verification_enabled": True,
                            "state_source": "cpu_reference",
                        }
                    else:
                        moe_branch = vulkan_moe_branch
                        state = s14.hc_post(
                            moe_branch,
                            pending.post_attention_state,
                            pending.post_ffn,
                            pending.comb_ffn,
                        )
                        self.writeback_layers.append(layer)
                        writeback_evidence = {
                            **worker_evidence,
                            "comparison": comparison,
                            "cpu_verification_enabled": self.config.vulkan_writeback_verify_cpu,
                            "state_source": "vulkan_moe_branch_then_cpu_hc_post",
                            "cpu_layer_output_exact": (
                                None
                                if cpu_state is None
                                else bool(torch.equal(state, cpu_state))
                            ),
                        }
                        if (
                            cpu_state is not None
                            and not writeback_evidence["cpu_layer_output_exact"]
                        ):
                            raise FullDepthError(
                                "Vulkan writeback 后 hc_post 与 CPU 层输出不等价"
                            )
            else:
                if cpu_moe_branch is None or cpu_state is None:
                    raise FullDepthError("CPU 层路径缺少 reference 输出")
                moe_branch, state = cpu_moe_branch, cpu_state
            if (
                tuple(state.shape) != (1, 1, s14.HC_MULT, s14.HIDDEN_SIZE)
                or state.dtype != torch.bfloat16
            ):
                raise FullDepthError(
                    f"L{layer} 破坏 BF16 [1,1,4,4096] mHC 状态"
                )
            session.finish_layer(layer, input_token_id)
            attention_vulkan = list(kernel.attention_vulkan_evidence)
            if attention_vulkan:
                if self.config.vulkan_attention_shared_batch:
                    # Batch A executes wq_a+wkv at the first logical wq_a call;
                    # telemetry follows the real GPU execution order.
                    expected_suffixes = ["wq_a", "wkv", "wq_b", "wo_a", "wo_b"]
                    if self.profile.ratio_for(layer) == 4:
                        expected_suffixes.insert(3, "indexer.wq_b")
                else:
                    expected_suffixes = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"]
                    if self.profile.ratio_for(layer) == 4:
                        expected_suffixes.insert(3, "indexer.wq_b")
                observed_suffixes = [
                    row["projection"]["name"].removeprefix(
                        f"layers.{layer}.attn."
                    )
                    for row in attention_vulkan
                ]
                if observed_suffixes != expected_suffixes:
                    raise FullDepthError(
                        f"L{layer} Vulkan attention 投影闭包漂移: "
                        f"expected={expected_suffixes}, observed={observed_suffixes}"
                    )
                self.attention_layers.append(layer)
                self.attention_projection_count += len(attention_vulkan)
            next_states[layer] = pending.runtime_state
            token_report["completed_layers"].append(layer)
            token_report["layers"].append(
                {
                    "layer": layer,
                    "compress_ratio": self.profile.ratio_for(layer),
                    "route_source": pending.route_source,
                    "expert_ids": pending.route_ids,
                    "route_weights": pending.route_weights,
                    "shared_and_expert_ranges": len(pages),
                    "moe_branch": kernel._summary_tensor(moe_branch),
                    "layer_output": kernel._summary_tensor(state),
                    "elapsed_seconds": time.perf_counter() - layer_started,
                    "vulkan_bridge_capture": (
                        None if bridge_capture is None else bridge_capture.report
                    ),
                    "vulkan_writeback": writeback_evidence,
                    "vulkan_attention": attention_vulkan,
                }
            )
            self._notify(token_report)

        if token_report["completed_layers"] != list(self.profile.layers):
            raise FullDepthError("禁止跳层进入 final head")
        self.current_layer = None
        self.stage = f"position_{position}_final_head"
        final_ranges = session.prepare_final()
        if self._final_head is None:
            worker_path = self.config.resolved_vulkan_final_head_worker()
            self._final_head = FullDepthVulkanFinalHead(
                final_ranges,
                self.config.asset_root.resolve() / "range_cache",
                (str(worker_path),),
                self.config.resolved_vulkan_final_head_scratch(),
                timeout_seconds=self.config.vulkan_final_head_timeout_seconds,
                validate_cpu_once=self.config.vulkan_final_head_validate_cpu_once,
                head_chunk_size=self.config.head_chunk_size,
            )
        else:
            self._final_head.validate_ranges(final_ranges)
        try:
            final = self._final_head.forward(state, position=position)
        except VulkanFinalHeadError as error:
            raise FullDepthError(
                f"position {position} Vulkan final-head 失败；禁止 CPU head fallback: {error}"
            ) from error
        output_token_id = int(final["token_id"])
        if not 0 <= output_token_id < s14.VOCAB_SIZE:
            raise FullDepthError("FullDepth final head token ID 越界")
        token_report["final"] = final
        self.stage = f"position_{position}_ready_to_commit"
        self._notify(token_report)
        return FullDepthTokenComputation(
            predicted_token_id=output_token_id,
            next_layer_states=next_states,
            top6_by_layer=top6_by_layer,
            value={"token_report": token_report, "final": final},
        )


def execute(
    config: ExecutionConfig,
    *,
    profile: ExecutionProfile = FULLDEPTH43_NATIVE_TOP6,
) -> dict[str, Any]:
    """显式执行真实 FullDepth token；任何缺口都不提交 token。"""

    config.validate()
    profile.validate()
    catalog = _load_or_build_catalog(config)
    catalog_sha256 = decoder_checkpoint.catalog_fingerprint(catalog)
    preflight = run_preflight(asset_root=config.asset_root, catalog=catalog, profile=profile)
    cold_upper = int(preflight["cold_execution_upper_bound"]["total_bytes"])
    authorized_to_fill = (
        config.allow_fetch
        and config.download_budget_bytes >= cold_upper
        and preflight["storage"]["cold_upper_bound_fits"]
    )
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "status": "running",
        "repo": profile.repo,
        "revision": profile.revision,
        "profile": profile.as_dict(),
        "download_authorized": config.allow_fetch,
        "download_budget_bytes": config.download_budget_bytes,
        "range_static_prefetch_enabled": config.range_static_prefetch,
        "range_gpu_verifier_ownership_enabled": config.range_gpu_verifier_ownership,
        "forced_prefill_path": (
            None
            if config.forced_prefill_path is None
            else str(config.forced_prefill_path.resolve())
        ),
        "forced_prefill": None,
        "checkpoint_path": (
            None if config.checkpoint_path is None else str(config.checkpoint_path.resolve())
        ),
        "resume_checkpoint_path": (
            None
            if config.resume_checkpoint_path is None
            else str(config.resume_checkpoint_path.resolve())
        ),
        "resume_checkpoint": None,
        "checkpoint": None,
        "preflight": preflight,
        "tokens": [],
        "committed_tokens": [],
        "native_token_executed": False,
        "fake_token_emitted": False,
        "error": None,
    }
    if preflight["status"] != "ready" and not authorized_to_fill:
        report["status"] = "blocked"
        report["error"] = {
            "stage": "preflight",
            "type": "FullDepthError",
            "message": "静态页缺失，且未显式授权足额 Range budget",
            "required_cold_upper_bytes": cold_upper,
        }
        report["claim_limit"] = "fail-closed before model forward; no token emitted"
        write_json(config.report_path, report)
        return report

    cache = online_range.RangeCache(
        config.asset_root.resolve() / "range_cache",
        endpoint=config.endpoint,
        allow_fetch=config.allow_fetch,
        download_budget_bytes=config.download_budget_bytes,
        timeout=300.0,
        deferred_verifier=(
            _gpu_verification_owner
            if config.range_gpu_verifier_ownership
            else None
        ),
    )
    try:
        if config.resume_checkpoint_path is not None:
            decoder, resume_evidence = decoder_checkpoint.load_decoder_checkpoint(
                config.resume_checkpoint_path,
                profile=profile,
            )
            forced_queue = decoder.forced_queue
            report["resume_checkpoint"] = resume_evidence
            if resume_evidence["provenance"].get("catalog_sha256") != catalog_sha256:
                raise decoder_checkpoint.CheckpointError(
                    "checkpoint catalog provenance 与当前 FullDepth catalog 不匹配"
                )
            report["committed_tokens"] = decoder.committed_tokens
        else:
            forced_queue = (
                None
                if config.forced_prefill_path is None
                else s14._load_forced_prefill(config.forced_prefill_path)
            )
            decoder = DecoderState(
                input_token_id=(
                    s14.BOS_TOKEN_ID if forced_queue is None else forced_queue.current_token_id
                ),
                forced_queue=forced_queue,
            )
    except Exception as error:
        report["status"] = "blocked"
        report["error"] = {
            "stage": "checkpoint_restore",
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc().splitlines(),
        }
        report["claim_limit"] = "checkpoint rejected before model forward; no state restored or token emitted"
        write_json(config.report_path, report)
        return report
    if forced_queue is not None:
        report["forced_prefill"] = {
            "artifact_sha256": forced_queue.artifact_sha256,
            "token_count": len(forced_queue.token_ids),
            "cursor": forced_queue.cursor,
        }
    def persist_worker_progress(token_report: dict[str, Any]) -> None:
        token_report.setdefault(
            "input_source",
            (
                "forced_prefill"
                if decoder.forced_queue is not None and decoder.forced_queue.active
                else "model_argmax"
            ),
        )
        if not any(row is token_report for row in report["tokens"]):
            report["tokens"].append(token_report)
        write_json(config.report_path, report)

    worker = FullDepthTokenWorker(
        config,
        catalog,
        cache,
        profile=profile,
        progress_callback=persist_worker_progress,
    )
    execution_started = time.perf_counter()
    try:
        worker.start()
        if worker.writeback_hello is not None:
            report["vulkan_writeback_worker"] = {
                "path": str(config.vulkan_writeback_worker.resolve()),
                "hello": worker.writeback_hello,
                "mode": "persistent_single_process",
                "all_layers": config.vulkan_writeback_all_layers,
                "cpu_verification": config.vulkan_writeback_verify_cpu,
                "cpu_fallback": config.vulkan_writeback_cpu_fallback,
                "fast_production": config.vulkan_writeback_fast_production,
                "batch_verify_payloads": (
                    config.vulkan_writeback_batch_verify_payloads
                ),
                "inline_manifest": config.vulkan_writeback_inline_manifest,
            }
        attention_hello = getattr(worker, "attention_hello", None)
        if attention_hello is not None:
            report["vulkan_attention_worker"] = {
                "path": str(config.vulkan_attention_worker.resolve()),
                "hello": attention_hello,
                "mode": "persistent_process_dynamic_layer_projection",
                "cpu_fallback": False,
                "activation_quantization": (
                    "worker_group128_e4m3fn_for_wo_a_to_wo_b"
                    if config.vulkan_attention_output_chain
                    else "cpu_e4m3fn_quant_dequant"
                ),
                "cpu_verification": config.vulkan_attention_verify_cpu,
                "shared_input_batch": config.vulkan_attention_shared_batch,
                "output_chain": config.vulkan_attention_output_chain,
                "range_gpu_verifier_ownership": config.range_gpu_verifier_ownership,
            }
        for _ in range(config.token_count):
            position = decoder.position
            input_token_id = decoder.input_token_id
            previous = decoder.previous_for(profile)
            computation = worker(position, input_token_id, previous)
            token_report = computation.value["token_report"]
            final_head_hello = getattr(worker, "final_head_hello", None)
            if final_head_hello is not None:
                report["vulkan_final_head_worker"] = {
                    "path": str(config.resolved_vulkan_final_head_worker()),
                    "hello": final_head_hello,
                    "mode": "persistent_gpu_head_device_argmax",
                    "cpu_scope": "hc_reduce_and_rmsnorm_only",
                    "cpu_validation_once": config.vulkan_final_head_validate_cpu_once,
                    "cpu_fallback": False,
                    "production_full_logits_returned": False,
                }
            if config.range_gpu_verifier_ownership:
                worker.stage = f"position_{position}_gpu_verifier_ownership_closure"
                closure = _gpu_verifier_receipt_closure(
                    report,
                    getattr(cache, "proof_cache_telemetry", None),
                )
                report["gpu_verifier_ownership_closure"] = closure
                if closure["closed"] is not True:
                    raise FullDepthError(
                        "Python 延迟所有权与 Rust 计算前验证回执未闭合；禁止提交token/checkpoint"
                    )
                worker.stage = f"position_{position}_ready_to_commit"
            if config.checkpoint_path is None:
                decoder.commit(
                    output_token_id=computation.predicted_token_id,
                    next_states=computation.next_layer_states,
                    profile=profile,
                )
            else:
                # 先在私有 decoder 上完成全量验证与 checkpoint
                # 原子发布；任一步失败都不修改运行中 decoder。
                candidate = decoder.clone()
                candidate.commit(
                    output_token_id=computation.predicted_token_id,
                    next_states=computation.next_layer_states,
                    profile=profile,
                )
                checkpoint_evidence = decoder_checkpoint.save_decoder_checkpoint(
                    candidate,
                    config.checkpoint_path,
                    profile=profile,
                    provenance={
                        "producer": "fulldepth43_native_top6.executor.execute",
                        "execution_report": str(config.report_path.resolve()),
                        "catalog_sha256": catalog_sha256,
                        "resumed_from_checkpoint_sha256": (
                            None
                            if report["resume_checkpoint"] is None
                            else report["resume_checkpoint"]["checkpoint_sha256"]
                        ),
                    },
                )
                decoder.__dict__.update(candidate.__dict__)
                token_report["checkpoint"] = checkpoint_evidence
                report["checkpoint"] = checkpoint_evidence
            token_report["state_committed"] = True
            report["committed_tokens"] = decoder.committed_tokens
            report["runtime"] = {
                "next_position": decoder.position,
                "next_input_token_id": decoder.input_token_id,
                "committed_layer_states": sorted(decoder.layer_states),
                "forced_prefill_cursor": (
                    None if decoder.forced_queue is None else decoder.forced_queue.cursor
                ),
                "forced_prefill_exhausted": (
                    None if decoder.forced_queue is None else not decoder.forced_queue.active
                ),
            }
            report["native_token_executed"] = True
            write_json(config.report_path, report)

        report["status"] = "complete"
        all_attention_vulkan = set(getattr(worker, "attention_layers", ())) == set(
            profile.layers
        )
        if set(worker.writeback_layers) == set(profile.layers):
            report["claim_limit"] = (
                "FullDepth43/native-top6 with all 43 MoE branches written back from Vulkan; "
                + (
                    "all approved attention FP8 projections also execute through Vulkan; "
                    if all_attention_vulkan
                    else "attention remains CPU or only partially executes through Vulkan; "
                )
                + "final BF16 head+argmax is persistent Vulkan while HC/router remain CPU, "
                "so this is not a full-GPU token, "
                "20/50 token/s, or quality claim"
            )
        elif worker.writeback_layers:
            report["claim_limit"] = (
                "FullDepth43/native-top6 correctness path with a subset of exact-BF16 Vulkan "
                "MoE writeback layers and persistent Vulkan final head; not a full-layer/full-token "
                "GPU, speed, or quality claim"
            )
        else:
            report["claim_limit"] = (
                "FullDepth43/native-top6 correctness path with "
                + (
                    "all approved attention FP8 projections on Vulkan, "
                    if all_attention_vulkan
                    else "CPU or partial Vulkan attention, "
                )
                + "CPU MoE/HC/router, and persistent Vulkan final head+device argmax; "
                "not a full-model speed or quality claim"
            )
    except Exception as error:
        report["status"] = "blocked"
        report["native_token_executed"] = bool(decoder.committed_tokens)
        report["committed_tokens"] = decoder.committed_tokens
        report["runtime"] = {
            "next_position": decoder.position,
            "next_input_token_id": decoder.input_token_id,
            "committed_layer_states": sorted(decoder.layer_states),
            "forced_prefill_cursor": (
                None if decoder.forced_queue is None else decoder.forced_queue.cursor
            ),
            "forced_prefill_exhausted": (
                None if decoder.forced_queue is None else not decoder.forced_queue.active
            ),
        }
        report["error"] = {
            "stage": worker.stage,
            "layer": worker.current_layer,
            "position": decoder.position,
            "input_token_id": decoder.input_token_id,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc().splitlines(),
        }
        report["claim_limit"] = "failure before current token commit; no fake token emitted"
    finally:
        report["vulkan_writeback_layers"] = worker.writeback_layers
        report["vulkan_writeback_fallbacks"] = worker.writeback_fallbacks
        report["vulkan_attention_layers"] = list(
            getattr(worker, "attention_layers", ())
        )
        report["vulkan_attention_projection_count"] = int(
            getattr(worker, "attention_projection_count", 0)
        )
        report["range_cleanup_errors"] = list(
            getattr(worker, "range_cleanup_errors", ())
        )
        try:
            worker.close()
        except Exception as close_error:
            report["worker_close_error"] = {
                "type": type(close_error).__name__,
                "message": str(close_error),
            }
            if report.get("status") != "blocked":
                report["status"] = "blocked"
                report["claim_limit"] = (
                    "Vulkan worker cleanup failed; execution report retained but run is not promotable"
                )
    report["range_proof_cache"] = getattr(cache, "proof_cache_telemetry", None)
    if config.range_gpu_verifier_ownership and report.get("status") == "complete":
        closure = _gpu_verifier_receipt_closure(
            report,
            report["range_proof_cache"],
        )
        report["gpu_verifier_ownership_closure"] = closure
        if closure["closed"] is not True:
            report["status"] = "blocked"
            report["error"] = {
                "stage": "gpu_verifier_ownership_closure",
                "type": "FullDepthError",
                "message": "Python 延迟所有权与 Rust 计算前验证回执未闭合",
            }
            report["claim_limit"] = (
                "token computation completed but verifier ownership audit failed; run is not promotable"
            )
    report["execution_seconds"] = time.perf_counter() - execution_started
    write_json(config.report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--endpoint", default=os.environ.get("POLARIS_HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--download-budget-bytes", type=int, default=0)
    parser.add_argument("--token-count", type=int, default=1)
    parser.add_argument(
        "--forced-prefill",
        type=Path,
        nargs="?",
        const=DEFAULT_FORCED_PREFILL,
        help="官方 token JSON；不带路径时使用 first_preview_forced_prefill.json",
    )
    parser.add_argument("--head-chunk-size", type=int, default=4096)
    parser.add_argument("--range-attempts", type=int, default=4)
    parser.add_argument("--range-workers", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--range-static-prefetch", action="store_true")
    parser.add_argument("--range-gpu-verifier-ownership", action="store_true")
    parser.add_argument("--vulkan-bridge-capture", type=Path)
    parser.add_argument("--vulkan-bridge-layer", type=int, choices=range(43), default=42)
    parser.add_argument("--vulkan-writeback-worker", type=Path)
    parser.add_argument("--vulkan-writeback-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--vulkan-writeback-all-layers", action="store_true")
    parser.add_argument("--vulkan-writeback-no-cpu-verify", action="store_true")
    parser.add_argument("--vulkan-writeback-no-cpu-fallback", action="store_true")
    parser.add_argument("--vulkan-writeback-fast-production", action="store_true")
    parser.add_argument(
        "--vulkan-writeback-batch-verify-payloads",
        action="store_true",
    )
    parser.add_argument("--vulkan-final-head-worker", type=Path)
    parser.add_argument("--vulkan-final-head-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--vulkan-final-head-scratch", type=Path)
    parser.add_argument("--vulkan-attention-worker", type=Path)
    parser.add_argument("--vulkan-attention-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--vulkan-attention-scratch", type=Path)
    parser.add_argument("--vulkan-attention-verify-cpu", action="store_true")
    parser.add_argument("--vulkan-attention-shared-batch", action="store_true")
    parser.add_argument("--vulkan-attention-output-chain", action="store_true")
    parser.add_argument(
        "--vulkan-final-head-validate-cpu-once",
        action="store_true",
        help="仅首个真实 normalized hidden 做一次 CPU token/top10 对照；生产默认关闭",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="每次 token 完整计算后原子替换的 DecoderState UTF-8 manifest",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="验证并恢复完整 43 层 KV/compressor 状态后继续生成",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = ExecutionConfig(
            asset_root=args.asset_root,
            catalog_path=args.catalog,
            report_path=args.report,
            endpoint=args.endpoint,
            allow_fetch=args.download_missing,
            download_budget_bytes=args.download_budget_bytes,
            token_count=args.token_count,
            head_chunk_size=args.head_chunk_size,
            range_attempts=args.range_attempts,
            range_workers=args.range_workers,
            range_static_prefetch=args.range_static_prefetch,
            range_gpu_verifier_ownership=args.range_gpu_verifier_ownership,
            forced_prefill_path=args.forced_prefill,
            vulkan_bridge_capture=args.vulkan_bridge_capture,
            vulkan_bridge_layer=args.vulkan_bridge_layer,
            vulkan_writeback_worker=args.vulkan_writeback_worker,
            vulkan_writeback_timeout_seconds=args.vulkan_writeback_timeout_seconds,
            vulkan_writeback_all_layers=args.vulkan_writeback_all_layers,
            vulkan_writeback_verify_cpu=not args.vulkan_writeback_no_cpu_verify,
            vulkan_writeback_cpu_fallback=not args.vulkan_writeback_no_cpu_fallback,
            vulkan_writeback_fast_production=args.vulkan_writeback_fast_production,
            vulkan_writeback_batch_verify_payloads=(
                args.vulkan_writeback_batch_verify_payloads
            ),
            vulkan_final_head_worker=args.vulkan_final_head_worker,
            vulkan_final_head_timeout_seconds=args.vulkan_final_head_timeout_seconds,
            vulkan_final_head_scratch=args.vulkan_final_head_scratch,
            vulkan_final_head_validate_cpu_once=(
                args.vulkan_final_head_validate_cpu_once
            ),
            vulkan_attention_worker=args.vulkan_attention_worker,
            vulkan_attention_timeout_seconds=args.vulkan_attention_timeout_seconds,
            vulkan_attention_scratch=args.vulkan_attention_scratch,
            vulkan_attention_verify_cpu=args.vulkan_attention_verify_cpu,
            vulkan_attention_shared_batch=args.vulkan_attention_shared_batch,
            vulkan_attention_output_chain=args.vulkan_attention_output_chain,
            checkpoint_path=args.checkpoint,
            resume_checkpoint_path=args.resume_checkpoint,
        )
        if args.command == "preflight":
            catalog = _load_or_build_catalog(config)
            report = run_preflight(asset_root=config.asset_root, catalog=catalog)
            write_json(config.report_path, report)
        else:
            report = execute(config)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(config.report_path.resolve()),
                    "native_token_executed": report.get("native_token_executed", False),
                    "committed_tokens": report.get("committed_tokens", []),
                    "error": report.get("error"),
                },
                ensure_ascii=False,
            )
        )
        return 0 if report["status"] == "complete" or args.command == "preflight" and report["status"] == "ready" else 2
    except Exception as error:
        print(
            json.dumps(
                {"status": "invalid", "type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
            )
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
