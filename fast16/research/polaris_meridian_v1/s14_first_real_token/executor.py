"""以真实 Range 页连续执行 Polaris S14 token0 与 token1。

执行器只接受固定 revision 与固定 14 层。每层先读取 non-expert/router，
算出原生 top-6 后才允许 Range 状态机提供命中专家页。它复用已经冻结的 L42
FP8/FP4/HC CPU 数值路径，并修正为官方 Expert.forward 的 route-weight-before-w2
与 w2 前 BF16 边界。

这是一条 CPU/PyTorch correctness 路径，不是速度或质量评测。
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import socket
import sys
import time
import traceback
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fast16.research.polaris_meridian_v1.l42_real_reference.l42_reference import (  # noqa: E402
    _InlineForward,
)
from fast16.research.polaris_meridian_v1.local_s14_primitives.attention import (  # noqa: E402
    sparse_attention,
)
from fast16.research.polaris_meridian_v1.local_s14_primitives.final_head import (  # noqa: E402
    native_final_logits,
)
from fast16.research.polaris_meridian_v1.local_s14_primitives.hc import (  # noqa: E402
    hc_post,
    hc_pre,
)
from fast16.research.polaris_meridian_v1.s14_range_pack import (  # noqa: E402
    online_range,
)


REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
REGISTERED_LAYERS = (0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42)
BOS_TOKEN_ID = 0
HIDDEN_SIZE = 4096
HC_MULT = 4
VOCAB_SIZE = 129280
TOP_K = 6
WINDOW_SIZE = 128
ROUTE_SCALE = 1.5
ROUTE_SUM_ABS_TOL = TOP_K * torch.finfo(torch.float32).eps * ROUTE_SCALE
SWIGLU_LIMIT = 10.0
DEFAULT_ASSET_ROOT = Path("D:/models/Polaris-S14")
DEFAULT_REPORT = Path(__file__).resolve().parent / "last_run_report.json"
HASH_ROUTE_ANCHORS = {
    0: [254, 222, 245, 200, 53, 35],
    1: [163, 137, 158, 97, 184, 8],
}
COMPRESS_RATIOS = {
    0: 0,
    1: 0,
    2: 4,
    6: 4,
    7: 128,
    14: 4,
    15: 128,
    22: 4,
    23: 128,
    30: 4,
    31: 128,
    40: 4,
    41: 128,
    42: 4,
}
_DTYPE_BYTES = {
    "F32": 4,
    "BF16": 2,
    "I64": 8,
    "F8_E8M0": 1,
    "F8_E4M3": 1,
    "I8": 1,
}


class ContractError(RuntimeError):
    """真实执行合同不满足。"""


def _precompute_position_freqs(compress_ratio: int, seqlen: int = 2) -> torch.Tensor:
    """精确移植固定 revision ``model.py::precompute_freqs_cis``。"""

    dim = 64
    original_seq_len = 65536 if compress_ratio else 0
    base = 160000.0 if compress_ratio else 10000.0
    factor = 16.0
    beta_fast = 32
    beta_slow = 1

    def find_correction_dim(num_rotations: int) -> float:
        return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range() -> tuple[int, int]:
        low = math.floor(find_correction_dim(beta_fast))
        high = math.ceil(find_correction_dim(beta_slow))
        return max(low, 0), min(high, dim - 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range()
        if low == high:
            high += 0.001
        linear = (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)
        smooth = 1 - torch.clamp(linear, 0, 1)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    positions = torch.arange(seqlen)
    return torch.polar(torch.ones((seqlen, dim // 2), dtype=torch.float32), torch.outer(positions, freqs))


def _apply_position_rope(x: torch.Tensor, freqs_cis: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """精确移植固定 revision ``model.py::apply_rotary_emb`` 的单位置分支。"""

    y = x
    complex_x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if complex_x.ndim == 3:
        freqs_cis = freqs_cis.view(1, complex_x.size(1), complex_x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, complex_x.size(1), 1, complex_x.size(-1))
    rotated = torch.view_as_real(complex_x * freqs_cis).flatten(-2)
    y.copy_(rotated)
    return y


def _window_topk_indices(position: int) -> torch.Tensor:
    """固定 revision ``get_window_topk_idxs`` 的单 token decode 分支。"""

    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ContractError("position 必须是非负整数")
    if position == 0:
        matrix = torch.tensor([0], dtype=torch.int32)
    elif position >= WINDOW_SIZE - 1:
        start = position % WINDOW_SIZE
        matrix = torch.cat((torch.arange(start + 1, WINDOW_SIZE), torch.arange(start + 1))).to(torch.int32)
    else:
        matrix = F.pad(
            torch.arange(position + 1, dtype=torch.int32),
            (0, WINDOW_SIZE - position - 1),
            value=-1,
        )
    return matrix.view(1, 1, -1).contiguous()


def _advance_window_kv(
    kv: torch.Tensor,
    *,
    position: int,
    previous: torch.Tensor | None,
) -> torch.Tensor:
    """clone 已提交窗口并在 attention 前写入当前 KV。"""

    if tuple(kv.shape) != (1, 1, 512) or kv.dtype != torch.bfloat16:
        raise ContractError("当前 KV 必须是 BF16 [1,1,512]")
    if position == 0:
        if previous is not None:
            raise ContractError("position0 不得携带旧 window KV")
        window = torch.zeros((1, WINDOW_SIZE, 512), dtype=torch.bfloat16)
    else:
        if previous is None or tuple(previous.shape) != (1, WINDOW_SIZE, 512):
            raise ContractError("decode 必须携带 [1,128,512] 的旧 window KV")
        if previous.dtype != torch.bfloat16:
            raise ContractError("旧 window KV 必须是 BF16")
        window = previous.clone()
    window[:, position % WINDOW_SIZE] = kv[:, 0]
    return window


@dataclass(frozen=True)
class ExecutionConfig:
    asset_root: Path = DEFAULT_ASSET_ROOT
    report_path: Path = DEFAULT_REPORT
    endpoint: str = "https://huggingface.co"
    allow_fetch: bool = False
    download_budget_bytes: int = 0
    stop_after_layer: int | None = None
    head_chunk_size: int = 4096
    range_attempts: int = 4
    range_workers: int = 3
    token_count: int = 1

    def validate(self) -> None:
        if self.stop_after_layer is not None and self.stop_after_layer not in REGISTERED_LAYERS:
            raise ValueError(f"stop_after_layer 必须属于 {list(REGISTERED_LAYERS)}")
        if self.download_budget_bytes < 0:
            raise ValueError("download_budget_bytes 不能为负")
        if self.allow_fetch and self.download_budget_bytes <= 0:
            raise ValueError("允许下载时必须提供正的 download_budget_bytes")
        if self.head_chunk_size <= 0:
            raise ValueError("head_chunk_size 必须为正整数")
        if self.range_attempts <= 0:
            raise ValueError("range_attempts 必须为正整数")
        if not 1 <= self.range_workers <= 3:
            raise ValueError("range_workers 必须在 1..3")
        if not 1 <= self.token_count <= 2:
            raise ValueError("当前 correctness executor 的 token_count 只允许 1 或 2")
        if self.stop_after_layer is not None and self.token_count != 1:
            raise ValueError("stop_after_layer 只能用于单 token checkpoint")
        if not self.endpoint.startswith("https://"):
            raise ValueError("endpoint 必须是 HTTPS")


@dataclass(frozen=True)
class TensorSource:
    entry: Mapping[str, Any]
    path: Path
    proof: Mapping[str, Any]
    cache_hit: bool


class TensorStore:
    """把 RangeCache 的校验结果收窄成只读 tensor 索引。"""

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root.resolve()
        self.sources: dict[str, TensorSource] = {}

    def add_ranges(self, ranges: Iterable[online_range.CachedRange]) -> None:
        for cached in ranges:
            entry = dict(cached.entry)
            name = entry.get("tensor")
            if not isinstance(name, str) or not name:
                raise ContractError("Range 缺少 tensor 名")
            if name in self.sources:
                raise ContractError(f"重复 tensor Range: {name}")
            path = cached.path.resolve()
            try:
                path.relative_to(self.cache_root)
            except ValueError as exc:
                raise ContractError(f"tensor 越出 range_cache: {name}") from exc
            dtype = entry.get("dtype")
            shape = entry.get("shape")
            if dtype not in _DTYPE_BYTES or not isinstance(shape, list) or not shape:
                raise ContractError(f"tensor dtype/shape 不受支持: {name}")
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
                raise ContractError(f"tensor shape 非法: {name}")
            expected = math.prod(shape) * _DTYPE_BYTES[str(dtype)]
            if entry.get("bytes") != expected or path.stat().st_size != expected:
                raise ContractError(f"tensor 字节数与 dtype/shape 不闭合: {name}")
            proof = dict(cached.proof)
            digest = proof.get("observed_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ContractError(f"tensor 缺少 Range SHA proof: {name}")
            self.sources[name] = TensorSource(entry, path, proof, cached.cache_hit)

    def bundle(self) -> SimpleNamespace:
        entries: dict[str, dict[str, Any]] = {}
        specs: dict[str, tuple[str, tuple[int, ...]]] = {}
        for name, source in self.sources.items():
            entries[name] = {**source.entry, "path": str(source.path)}
            specs[name] = (str(source.entry["dtype"]), tuple(source.entry["shape"]))
        return SimpleNamespace(entries=entries, specs=specs)

    def source(self, name: str) -> TensorSource:
        try:
            return self.sources[name]
        except KeyError as exc:
            raise ContractError(f"缺少已校验 tensor: {name}") from exc

    def integrity(self) -> dict[str, Any]:
        return {
            "payload_files": len(self.sources),
            "payload_bytes": sum(int(source.entry["bytes"]) for source in self.sources.values()),
            "cache_hits": sum(source.cache_hit for source in self.sources.values()),
            "sha_proofs_checked": len(self.sources),
            "hash_authorities": sorted({str(source.proof.get("hash_authority")) for source in self.sources.values()}),
        }


@dataclass
class CompressorRemainderState:
    ratio: int
    overlap: bool
    main_kv_state: torch.Tensor
    main_score_state: torch.Tensor
    indexer_kv_state: torch.Tensor | None
    indexer_score_state: torch.Tensor | None


@dataclass
class LayerRuntimeState:
    layer: int
    position: int
    window_kv: torch.Tensor
    compressor: CompressorRemainderState | None


def _clone_compressor_state(state: CompressorRemainderState | None) -> CompressorRemainderState | None:
    if state is None:
        return None
    return CompressorRemainderState(
        ratio=state.ratio,
        overlap=state.overlap,
        main_kv_state=state.main_kv_state.clone(),
        main_score_state=state.main_score_state.clone(),
        indexer_kv_state=None if state.indexer_kv_state is None else state.indexer_kv_state.clone(),
        indexer_score_state=None if state.indexer_score_state is None else state.indexer_score_state.clone(),
    )


def _clone_layer_state(state: LayerRuntimeState) -> LayerRuntimeState:
    return LayerRuntimeState(
        layer=state.layer,
        position=state.position,
        window_kv=state.window_kv.clone(),
        compressor=_clone_compressor_state(state.compressor),
    )


@dataclass(frozen=True)
class DecoderSnapshot:
    position: int = 0
    input_token_id: int = BOS_TOKEN_ID
    layer_states: Mapping[int, LayerRuntimeState] = field(default_factory=dict)
    committed_tokens: tuple[Mapping[str, int], ...] = ()


@dataclass(frozen=True)
class TokenComputation:
    output_token_id: int
    next_layer_states: Mapping[int, LayerRuntimeState]
    value: Mapping[str, Any]


@dataclass
class DecoderRuntime:
    """跨 token runtime；RangeCache 长期复用，layer state 单指针原子提交。"""

    catalog: Mapping[str, Any]
    cache: online_range.RangeCache
    snapshot: DecoderSnapshot = field(default_factory=DecoderSnapshot)
    final_head: FinalHeadReference | None = None

    @property
    def position(self) -> int:
        return self.snapshot.position

    @property
    def input_token_id(self) -> int:
        return self.snapshot.input_token_id

    @property
    def layer_states(self) -> Mapping[int, LayerRuntimeState]:
        return self.snapshot.layer_states

    def run_token(
        self,
        worker: Callable[
            [int, int, Mapping[int, LayerRuntimeState], online_range.RouteFirstSession],
            TokenComputation,
        ],
    ) -> Mapping[str, Any]:
        """以私有 state clone 运行一个 token；任何异常都保持旧 snapshot。"""

        before = self.snapshot
        working_previous = {layer: _clone_layer_state(state) for layer, state in before.layer_states.items()}
        session = online_range.RouteFirstSession(self.catalog, self.cache)
        computation = worker(before.position, before.input_token_id, working_previous, session)
        if (
            isinstance(computation.output_token_id, bool)
            or not isinstance(computation.output_token_id, int)
            or not 0 <= computation.output_token_id < VOCAB_SIZE
        ):
            raise ContractError("final argmax token ID 越界")
        if set(computation.next_layer_states) != set(REGISTERED_LAYERS):
            raise ContractError("next_layer_states 必须覆盖全部预注册 S14 层")
        for layer in REGISTERED_LAYERS:
            state = computation.next_layer_states[layer]
            if state.layer != layer or state.position != before.position:
                raise ContractError(f"L{layer} next runtime state 的 layer/position 漂移")
        committed = {
            "position": before.position,
            "input_token_id": before.input_token_id,
            "output_token_id": computation.output_token_id,
        }
        next_snapshot = DecoderSnapshot(
            position=before.position + 1,
            input_token_id=computation.output_token_id,
            layer_states=dict(computation.next_layer_states),
            committed_tokens=before.committed_tokens + (committed,),
        )
        # 所有 tensor、final logits 与合同都成功后，只有这一处替换 committed state。
        self.snapshot = next_snapshot
        return computation.value


@dataclass
class PendingLayer:
    post_attention_state: torch.Tensor
    post_ffn: torch.Tensor
    comb_ffn: torch.Tensor
    ffn_input: torch.Tensor
    route_ids: list[int]
    route_weights: list[float]
    route_source: str
    attention_branch: torch.Tensor
    runtime_state: LayerRuntimeState


class NativeLayerReference(_InlineForward):
    """catalog 驱动的单层 CPU 参考，数值核继承冻结 L42 路径。"""

    def __init__(
        self,
        layer: int,
        store: TensorStore,
        *,
        profile_layers: Sequence[int] = REGISTERED_LAYERS,
        compress_ratios: Mapping[int, int] = COMPRESS_RATIOS,
    ):
        if layer not in profile_layers:
            raise ContractError(f"未预注册层: {layer}")
        if set(compress_ratios) != set(profile_layers):
            raise ContractError("profile layer/compressor ratio 映射不闭合")
        if compress_ratios[layer] not in {0, 4, 128}:
            raise ContractError(f"L{layer} compressor ratio 不受支持")
        self.layer = layer
        self.store = store
        self.profile_layers = tuple(profile_layers)
        self.compress_ratios = dict(compress_ratios)
        super().__init__(store.bundle())

    def add_routed(self, ranges: Iterable[online_range.CachedRange]) -> None:
        self.store.add_ranges(ranges)
        self.bundle = self.store.bundle()

    def _name(self, suffix: str) -> str:
        return f"layers.{self.layer}.{suffix}"

    def _load_i64(self, name: str) -> torch.Tensor:
        dtype_name, shape = self.bundle.specs[name]
        if dtype_name != "I64":
            raise ContractError(f"{name} 物理 dtype 必须是 I64，实际为 {dtype_name}")
        payload = bytearray(self._path(name).read_bytes())
        return torch.frombuffer(payload, dtype=torch.int64).reshape(shape).clone()

    @staticmethod
    def _summary_tensor(tensor: torch.Tensor) -> dict[str, Any]:
        return _InlineForward._summary(tensor)

    def _compressor_projection(
        self,
        x: torch.Tensor,
        prefix: str,
        *,
        ratio: int,
        head_dim: int,
        overlap: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coff = 2 if overlap else 1
        expected = coff * head_dim
        wkv = self._load_tensor(prefix + ".wkv.weight")
        wgate = self._load_tensor(prefix + ".wgate.weight")
        ape = self._load_tensor(prefix + ".ape")
        if tuple(wkv.shape) != (expected, HIDDEN_SIZE) or tuple(wgate.shape) != tuple(wkv.shape):
            raise ContractError(f"{prefix} projection shape 漂移")
        if tuple(ape.shape) != (ratio, expected):
            raise ContractError(f"{prefix}.ape shape 漂移")
        projected = F.linear(x.float(), wkv.float())
        score = F.linear(x.float(), wgate.float()) + ape[0].view(1, 1, expected)
        state_rows = coff * ratio
        kv_state = torch.zeros((x.shape[0], state_rows, expected), dtype=torch.float32)
        score_state = torch.full_like(kv_state, -torch.inf)
        offset = ratio if overlap else 0
        kv_state[:, offset] = projected[:, 0]
        score_state[:, offset] = score[:, 0]
        return kv_state, score_state

    def _advance_compressor_state(
        self,
        x: torch.Tensor,
        *,
        position: int,
        previous: CompressorRemainderState | None,
    ) -> CompressorRemainderState | None:
        ratio = self.compress_ratios[self.layer]
        if ratio == 0:
            if previous is not None:
                raise ContractError(f"L{self.layer} ratio0 不得携带 compressor state")
            return None
        if position > 1:
            raise ContractError("当前连续 executor 只实现到 position1；尚未触碰 p3/p127 压缩边界")
        overlap = ratio == 4
        if position > 0:
            if previous is None:
                raise ContractError(f"L{self.layer} position{position} 缺少已提交 compressor remainder")
            if previous.ratio != ratio or previous.overlap != overlap:
                raise ContractError(f"L{self.layer} compressor ratio/overlap 漂移")
            current = _clone_compressor_state(previous)
            if current is None:
                raise AssertionError("compressor clone 不应为空")
            row = ratio + position % ratio if overlap else position % ratio

            def append(prefix: str, kv_state: torch.Tensor, score_state: torch.Tensor) -> None:
                expected = kv_state.shape[-1]
                wkv = self._load_tensor(prefix + ".wkv.weight")
                wgate = self._load_tensor(prefix + ".wgate.weight")
                ape = self._load_tensor(prefix + ".ape")
                if tuple(wkv.shape) != (expected, HIDDEN_SIZE) or tuple(wgate.shape) != tuple(wkv.shape):
                    raise ContractError(f"{prefix} projection shape 漂移")
                if tuple(ape.shape) != (ratio, expected):
                    raise ContractError(f"{prefix}.ape shape 漂移")
                projected = F.linear(x.float(), wkv.float())
                score = F.linear(x.float(), wgate.float()) + ape[position % ratio].view(1, 1, expected)
                kv_state[:, row] = projected[:, 0]
                score_state[:, row] = score[:, 0]

            append(self._name("attn.compressor"), current.main_kv_state, current.main_score_state)
            if ratio == 4:
                if current.indexer_kv_state is None or current.indexer_score_state is None:
                    raise ContractError(f"L{self.layer} ratio4 缺少独立 indexer remainder")
                append(
                    self._name("attn.indexer.compressor"),
                    current.indexer_kv_state,
                    current.indexer_score_state,
                )
            elif current.indexer_kv_state is not None or current.indexer_score_state is not None:
                raise ContractError(f"L{self.layer} ratio128 不得携带 indexer remainder")
            return current

        if previous is not None:
            raise ContractError(f"L{self.layer} position0 不得携带旧 compressor state")
        main_prefix = self._name("attn.compressor")
        main_kv, main_score = self._compressor_projection(
            x,
            main_prefix,
            ratio=ratio,
            head_dim=512,
            overlap=overlap,
        )
        index_kv: torch.Tensor | None = None
        index_score: torch.Tensor | None = None
        if ratio == 4:
            index_kv, index_score = self._compressor_projection(
                x,
                self._name("attn.indexer.compressor"),
                ratio=ratio,
                head_dim=128,
                overlap=True,
            )
        return CompressorRemainderState(
            ratio=ratio,
            overlap=overlap,
            main_kv_state=main_kv,
            main_score_state=main_score,
            indexer_kv_state=index_kv,
            indexer_score_state=index_score,
        )

    def _attention(
        self,
        state: torch.Tensor,
        *,
        position: int,
        previous_runtime: LayerRuntimeState | None,
    ) -> tuple[torch.Tensor, torch.Tensor, LayerRuntimeState]:
        prefix = f"layers.{self.layer}"
        branch, post, comb = hc_pre(
            state,
            self._load_tensor(prefix + ".hc_attn_fn"),
            self._load_tensor(prefix + ".hc_attn_scale"),
            self._load_tensor(prefix + ".hc_attn_base"),
        )
        branch = self._rms_norm(branch, self._load_tensor(prefix + ".attn_norm.weight"))
        if position == 0:
            if previous_runtime is not None:
                raise ContractError(f"L{self.layer} position0 不得携带旧 layer state")
        elif previous_runtime is None or previous_runtime.layer != self.layer or previous_runtime.position != position - 1:
            raise ContractError(f"L{self.layer} position{position} 缺少连续的前一 token state")
        compressor_state = self._advance_compressor_state(
            branch,
            position=position,
            previous=None if previous_runtime is None else previous_runtime.compressor,
        )
        branch_np = branch.float().numpy()

        q_low = torch.from_numpy(self._linear_fp8(branch_np, prefix + ".attn.wq_a")).to(torch.bfloat16)
        q_low = self._rms_norm(q_low, self._load_tensor(prefix + ".attn.q_norm.weight"))
        q = torch.from_numpy(self._linear_fp8(q_low.float().numpy(), prefix + ".attn.wq_b"))
        q = q.to(torch.bfloat16).reshape(1, 1, 64, 512)
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
        freqs_cis = _precompute_position_freqs(self.compress_ratios[self.layer], seqlen=position + 1)[
            position : position + 1
        ]
        if position:
            _apply_position_rope(q[..., -64:], freqs_cis)

        kv = torch.from_numpy(self._linear_fp8(branch_np, prefix + ".attn.wkv")).to(torch.bfloat16)
        kv = self._rms_norm(kv, self._load_tensor(prefix + ".attn.kv_norm.weight"))
        if position:
            _apply_position_rope(kv[..., -64:], freqs_cis)
        kv[..., :-64] = torch.from_numpy(
            self._activation_quant(kv[..., :-64].float().numpy(), 64)
        ).to(torch.bfloat16)
        window_kv = _advance_window_kv(
            kv,
            position=position,
            previous=None if previous_runtime is None else previous_runtime.window_kv,
        )
        topk_indices = _window_topk_indices(position)
        attention = sparse_attention(
            q,
            kv if position == 0 else window_kv,
            self._load_tensor(prefix + ".attn.attn_sink"),
            topk_indices,
            softmax_scale=512**-0.5,
            output_dtype=torch.bfloat16,
        )
        if position:
            _apply_position_rope(attention[..., -64:], freqs_cis, inverse=True)
        grouped = attention.reshape(1, 1, 8, 4096).float().numpy()
        wo_a = self._weight_fp8(prefix + ".attn.wo_a", bf16=True).reshape(8, 1024, 4096)
        low_output = self._bf16_numpy(
            np.stack([grouped[0, 0, group] @ wo_a[group].T for group in range(8)]).reshape(1, 1, 8192)
        )
        del wo_a
        attention_branch = torch.from_numpy(self._linear_fp8(low_output, prefix + ".attn.wo_b")).to(
            torch.bfloat16
        )
        post_attention_state = hc_post(attention_branch, state, post, comb)
        return attention_branch, post_attention_state, LayerRuntimeState(
            layer=self.layer,
            position=position,
            window_kv=window_kv,
            compressor=compressor_state,
        )

    def _route(self, ffn_input: torch.Tensor, *, token_id: int) -> tuple[list[int], list[float], str]:
        prefix = f"layers.{self.layer}.ffn.gate"
        flat = ffn_input.reshape(-1, HIDDEN_SIZE)
        logits = F.linear(flat.float(), self._load_tensor(prefix + ".weight").float())
        scores = F.softplus(logits).sqrt()
        if self.layer < 3:
            physical = self._load_i64(prefix + ".tid2eid")
            if isinstance(token_id, bool) or not 0 <= token_id < physical.shape[0]:
                raise ContractError(f"L{self.layer} token_id 越出 tid2eid")
            row = physical[token_id]
            if bool(((row < 0) | (row >= 256)).any().item()):
                raise ContractError(f"L{self.layer} token{token_id} tid2eid 含越界 expert")
            route_ids = [int(value) for value in row.tolist()]
            anchor = HASH_ROUTE_ANCHORS.get(self.layer) if token_id == BOS_TOKEN_ID else None
            if anchor is not None and route_ids != anchor:
                raise ContractError(f"L{self.layer} token0 hash route 漂移: {route_ids}")
            indices = row.to(torch.int64).view(1, TOP_K)
            source = "current_token_tid2eid_physical_i64"
        else:
            bias = self._load_tensor(prefix + ".bias").float()
            indices = (scores + bias).topk(TOP_K, dim=-1).indices
            route_ids = [int(value) for value in indices[0].tolist()]
            source = "sqrtsoftplus_plus_bias_top6"
        if len(set(route_ids)) != TOP_K:
            raise ContractError(f"L{self.layer} top-6 含重复 expert")
        weights = scores.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True) * ROUTE_SCALE
        route_weights = [float(value) for value in weights[0].tolist()]
        route_sum = math.fsum(route_weights)
        if not math.isclose(route_sum, ROUTE_SCALE, rel_tol=0, abs_tol=ROUTE_SUM_ABS_TOL):
            raise ContractError(
                f"L{self.layer} route 权重和超出 float32 归一化容差: "
                f"sum={route_sum:.9g}, expected={ROUTE_SCALE}, tol={ROUTE_SUM_ABS_TOL:.3g}"
            )
        return route_ids, route_weights, source

    def prepare_route(
        self,
        state: torch.Tensor,
        *,
        token_id: int = BOS_TOKEN_ID,
        position: int = 0,
        previous_runtime: LayerRuntimeState | None = None,
    ) -> PendingLayer:
        if tuple(state.shape) != (1, 1, HC_MULT, HIDDEN_SIZE) or state.dtype != torch.bfloat16:
            raise ContractError("每层输入必须是 BF16 [1,1,4,4096]")
        attention_branch, post_attention_state, runtime_state = self._attention(
            state,
            position=position,
            previous_runtime=previous_runtime,
        )
        prefix = f"layers.{self.layer}"
        ffn_input, post_ffn, comb_ffn = hc_pre(
            post_attention_state,
            self._load_tensor(prefix + ".hc_ffn_fn"),
            self._load_tensor(prefix + ".hc_ffn_scale"),
            self._load_tensor(prefix + ".hc_ffn_base"),
        )
        ffn_input = self._rms_norm(ffn_input, self._load_tensor(prefix + ".ffn_norm.weight"))
        route_ids, route_weights, route_source = self._route(ffn_input, token_id=token_id)
        return PendingLayer(
            post_attention_state=post_attention_state,
            post_ffn=post_ffn,
            comb_ffn=comb_ffn,
            ffn_input=ffn_input,
            route_ids=route_ids,
            route_weights=route_weights,
            route_source=route_source,
            attention_branch=attention_branch,
            runtime_state=runtime_state,
        )

    @staticmethod
    def _limited_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate.float().clamp(max=SWIGLU_LIMIT)) * up.float().clamp(
            min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT
        )

    def finish_layer(self, pending: PendingLayer) -> tuple[torch.Tensor, torch.Tensor]:
        ffn_np = pending.ffn_input.float().numpy()
        moe = torch.zeros((1, 1, HIDDEN_SIZE), dtype=torch.float32)
        for expert_id, route_weight in zip(pending.route_ids, pending.route_weights, strict=True):
            prefix = f"layers.{self.layer}.ffn.experts.{expert_id}"
            gate = torch.from_numpy(self._linear_fp4(ffn_np, prefix + ".w1")).to(torch.bfloat16)
            up = torch.from_numpy(self._linear_fp4(ffn_np, prefix + ".w3")).to(torch.bfloat16)
            hidden = self._limited_swiglu(gate, up)
            # 官方 Expert.forward 在 w2 的 activation quantize 之前乘 route weight，
            # 并先恢复输入 BF16；不能把权重挪到 w2 输出之后。
            weighted_hidden = (hidden * route_weight).to(torch.bfloat16)
            expert_output = torch.from_numpy(
                self._linear_fp4(weighted_hidden.float().numpy(), prefix + ".w2")
            ).to(torch.bfloat16)
            moe += expert_output.float()

        shared = f"layers.{self.layer}.ffn.shared_experts"
        shared_gate = torch.from_numpy(self._linear_fp8(ffn_np, shared + ".w1")).to(torch.bfloat16)
        shared_up = torch.from_numpy(self._linear_fp8(ffn_np, shared + ".w3")).to(torch.bfloat16)
        shared_hidden = self._limited_swiglu(shared_gate, shared_up).to(torch.bfloat16)
        shared_output = torch.from_numpy(
            self._linear_fp8(shared_hidden.float().numpy(), shared + ".w2")
        ).to(torch.bfloat16)
        moe += shared_output.float()
        moe_branch = moe.to(torch.bfloat16)
        layer_output = hc_post(
            moe_branch,
            pending.post_attention_state,
            pending.post_ffn,
            pending.comb_ffn,
        )
        return moe_branch, layer_output


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except FileNotFoundError as exc:
        raise ContractError(f"缺少 JSON 资产: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON 顶层必须是对象: {path}")
    return value


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _source_summary(cached: online_range.CachedRange) -> dict[str, Any]:
    return {
        "tensor": cached.entry["tensor"],
        "bytes": cached.entry["bytes"],
        "cache_hit": cached.cache_hit,
        "observed_sha256": cached.proof["observed_sha256"],
        "hash_authority": cached.proof["hash_authority"],
    }


def _runtime_state_report(state: LayerRuntimeState) -> dict[str, Any]:
    active_window = min(state.position + 1, WINDOW_SIZE)
    report: dict[str, Any] = {
        "position": state.position,
        "window_kv_shape": list(state.window_kv.shape),
        "window_position0": NativeLayerReference._summary_tensor(state.window_kv[:, :1]),
        "window_current": NativeLayerReference._summary_tensor(
            state.window_kv[:, state.position % WINDOW_SIZE : state.position % WINDOW_SIZE + 1]
        ),
        "active_window_rows": active_window,
    }
    compressor = state.compressor
    if compressor is None:
        report["compressor"] = None
        return report
    row = (
        compressor.ratio + state.position % compressor.ratio
        if compressor.overlap
        else state.position % compressor.ratio
    )
    comp: dict[str, Any] = {
        "ratio": compressor.ratio,
        "overlap": compressor.overlap,
        "main_kv_state_shape": list(compressor.main_kv_state.shape),
        "main_score_state_shape": list(compressor.main_score_state.shape),
        "written_row": row,
        "main_kv_written": NativeLayerReference._summary_tensor(
            compressor.main_kv_state[:, row : row + 1]
        ),
        "main_score_written": NativeLayerReference._summary_tensor(
            compressor.main_score_state[:, row : row + 1]
        ),
    }
    if compressor.indexer_kv_state is not None and compressor.indexer_score_state is not None:
        comp.update(
            {
                "indexer_kv_state_shape": list(compressor.indexer_kv_state.shape),
                "indexer_score_state_shape": list(compressor.indexer_score_state.shape),
                "indexer_written_row": row,
                "indexer_kv_written": NativeLayerReference._summary_tensor(
                    compressor.indexer_kv_state[:, row : row + 1]
                ),
                "indexer_score_written": NativeLayerReference._summary_tensor(
                    compressor.indexer_score_state[:, row : row + 1]
                ),
            }
        )
    report["compressor"] = comp
    return report


def _initial_state(embedding: online_range.CachedRange, token_id: int = BOS_TOKEN_ID) -> torch.Tensor:
    entry = embedding.entry
    if (
        entry.get("tensor") != f"embed.weight[{token_id}:{token_id + 1}]"
        or entry.get("parent_tensor") != "embed.weight"
        or entry.get("row_start") != token_id
        or entry.get("row_end_exclusive") != token_id + 1
        or entry.get("dtype") != "BF16"
    ):
        raise ContractError("embedding row tensor/dtype 漂移")
    if entry.get("shape") != [1, HIDDEN_SIZE] or entry.get("bytes") != HIDDEN_SIZE * 2:
        raise ContractError("embedding row 必须是 BF16 [1,4096]")
    payload = bytearray(embedding.path.read_bytes())
    row = torch.frombuffer(payload, dtype=torch.bfloat16).reshape(HIDDEN_SIZE).clone()
    if not bool(torch.isfinite(row).all().item()):
        raise ContractError(f"token{token_id} embedding row 含非有限值")
    return row.view(1, 1, 1, HIDDEN_SIZE).repeat(1, 1, HC_MULT, 1)


def _decode_token(asset_root: Path, token_id: int) -> str | None:
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(asset_root / "tokenizer.json")).decode([token_id])
    except Exception:
        return None


class FinalHeadReference:
    """跨 token 复用同一 BF16 head mmap；每次仍重做 HC/norm/logits。"""

    def __init__(
        self,
        final_ranges: Sequence[online_range.CachedRange],
        cache_root: Path,
        *,
        head_chunk_size: int,
    ) -> None:
        self.store = TensorStore(cache_root)
        self.store.add_ranges(final_ranges)
        self.bundle = self.store.bundle()
        self.helper = _InlineForward(self.bundle)
        required = {"hc_head_base", "hc_head_fn", "hc_head_scale", "norm.weight", "head.weight"}
        if set(self.store.sources) != required:
            raise ContractError(f"final tensor 集漂移: {sorted(self.store.sources)}")
        head_source = self.store.source("head.weight")
        if head_source.entry.get("dtype") != "BF16" or head_source.entry.get("shape") != [VOCAB_SIZE, HIDDEN_SIZE]:
            raise ContractError("真实 final head 必须是 BF16 [129280,4096]")
        self.head = torch.from_file(
            str(head_source.path),
            shared=False,
            size=VOCAB_SIZE * HIDDEN_SIZE,
            dtype=torch.bfloat16,
        ).reshape(VOCAB_SIZE, HIDDEN_SIZE)
        self.head_chunk_size = head_chunk_size
        self.range_shas = {
            item.entry["tensor"]: item.proof["observed_sha256"] for item in final_ranges
        }

    def validate_ranges(self, final_ranges: Sequence[online_range.CachedRange]) -> None:
        observed = {item.entry["tensor"]: item.proof["observed_sha256"] for item in final_ranges}
        if observed != self.range_shas:
            raise ContractError("跨 token final Range/SHA 漂移")

    def forward(self, state: torch.Tensor) -> dict[str, Any]:
        logits, normalized, pre = native_final_logits(
            state,
            self.helper._load_tensor("hc_head_fn"),
            self.helper._load_tensor("hc_head_scale"),
            self.helper._load_tensor("hc_head_base"),
            self.helper._load_tensor("norm.weight"),
            self.head,
            output_chunk_size=self.head_chunk_size,
        )
        token_id = int(logits.argmax(dim=-1).item())
        top_values, top_ids = logits.topk(10, dim=-1)
        top10 = [
            {"token_id": int(index), "logit": float(value)}
            for index, value in zip(top_ids[0].tolist(), top_values[0].tolist(), strict=True)
        ]
        return {
            "token_id": token_id,
            "hc_pre": [float(value) for value in pre.flatten().tolist()],
            "normalized": NativeLayerReference._summary_tensor(normalized),
            "logits": NativeLayerReference._summary_tensor(logits),
            "top10": top10,
            "integrity": self.store.integrity(),
        }


def _final_forward(
    state: torch.Tensor,
    final_ranges: Sequence[online_range.CachedRange],
    cache_root: Path,
    *,
    head_chunk_size: int,
) -> dict[str, Any]:
    return FinalHeadReference(
        final_ranges,
        cache_root,
        head_chunk_size=head_chunk_size,
    ).forward(state)


def _base_report(config: ExecutionConfig) -> dict[str, Any]:
    return {
        "format": "polaris-s14-contiguous-token-report-v2",
        "status": "running",
        "repo": REPO,
        "revision": REVISION,
        "token_id": BOS_TOKEN_ID,
        "registered_layers": list(REGISTERED_LAYERS),
        "endpoint": config.endpoint,
        "download_authorized": config.allow_fetch,
        "download_budget_bytes": config.download_budget_bytes,
        "stop_after_layer": config.stop_after_layer,
        "range_attempts": config.range_attempts,
        "range_workers": config.range_workers,
        "requested_token_count": config.token_count,
        "range_retries": [],
        "routed_prefetch_plans": [],
        "tokens": [],
        "committed_tokens": [],
        "completed_layers": [],
        "layers": [],
        "final": None,
        "error": None,
        "claim_limit": (
            "CPU/PyTorch 的真实权重、原生路由、命中专家 Range correctness 证据；"
            "不代表完整 43 层 DeepSeek-V4、质量或速度"
        ),
    }


_RECOVERABLE_TRANSPORT_ERRORS = (
    ConnectionError,
    TimeoutError,
    socket.timeout,
    http.client.RemoteDisconnected,
    urllib.error.URLError,
)


def _fetch_routed_with_retries(
    session: online_range.RouteFirstSession,
    *,
    layer: int,
    position: int,
    token_id: int,
    attempts: int,
    report: dict[str, Any],
    report_path: Path,
    workers: int,
) -> online_range.RoutedLayer:
    """只重试可恢复传输错误；合同、SHA、预算错误立即早停。"""

    for attempt in range(1, attempts + 1):
        try:
            _prefetch_submitted_experts(
                session,
                layer=layer,
                position=position,
                workers=workers,
                report=report,
                report_path=report_path,
            )
            return session.fetch_routed(layer, token_id)
        except _RECOVERABLE_TRANSPORT_ERRORS as exc:
            event = {
                "position": position,
                "layer": layer,
                "attempt": attempt,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            report["range_retries"].append(event)
            _write_json(report_path, report)
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 4))
    raise AssertionError("range retry 循环不应到达此处")


def _prefetch_submitted_experts(
    session: online_range.RouteFirstSession,
    *,
    layer: int,
    position: int,
    workers: int,
    report: dict[str, Any],
    report_path: Path,
) -> None:
    """对已提交 top-6 的 expert 页做可选的三路精确 Range 预取。

    并发只在整层 catalog 字节和可一次通过剩余预算时启用；否则保持
    ``RouteFirstSession.fetch_routed`` 的原始顺序与逐 miss 预算语义。
    """

    if workers == 1 or not session.cache.allow_fetch:
        return
    route = getattr(session, "_route", None)
    if session.phase is not online_range.SessionPhase.ROUTED or route is None:
        raise ContractError("并发预取只能发生在 submit_top6 之后")
    entries: list[Mapping[str, Any]] = []
    for expert_id in route:
        entries.extend(session.catalog["layers"][str(layer)]["experts"][str(expert_id)])
    unique: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        key = str(entry.get("range_key", ""))
        if not key or key in unique:
            raise ContractError(f"L{layer} routed range_key 缺失或重复: {key!r}")
        unique[key] = entry
    total_bytes = sum(int(entry["bytes"]) for entry in unique.values())
    remaining_budget = session.cache.download_budget_bytes - session.cache.downloaded_bytes
    existing = next(
        (
            item
            for item in report["routed_prefetch_plans"]
            if item["position"] == position and item["layer"] == layer
        ),
        None,
    )
    if existing is None:
        plan = {
            "position": position,
            "layer": layer,
            "expert_ids": list(route),
            "unique_payloads": len(unique),
            "catalog_bytes": total_bytes,
            "workers_requested": workers,
            "remaining_download_budget_before": remaining_budget,
            "mode": "parallel_exact_ranges" if total_bytes <= remaining_budget else "sequential_budget_exact",
        }
        report["routed_prefetch_plans"].append(plan)
        _write_json(report_path, report)
    if total_bytes > remaining_budget:
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"s14-p{position}-l{layer}-range") as pool:
        futures = [pool.submit(session.cache.fetch, entry) for entry in unique.values()]
        try:
            for future in as_completed(futures):
                future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise


class _CheckpointComplete(RuntimeError):
    """单 token 分层 checkpoint 的内部控制流。"""


def execute(config: ExecutionConfig) -> dict[str, Any]:
    """在同一 DecoderRuntime 中连续运行 token0，并可继续真实 token1。"""

    config.validate()
    report = _base_report(config)
    report_path = config.report_path.resolve()
    _write_json(report_path, report)
    stage = "bootstrap"
    current_layer: int | None = None
    cache: online_range.RangeCache | None = None
    runtime: DecoderRuntime | None = None
    started = time.perf_counter()
    try:
        root = config.asset_root.resolve()
        if not root.is_dir():
            raise ContractError(f"资产根目录不存在: {root}")
        catalog = _read_json(root / "route_first_catalog.json")
        online_range.validate_catalog(catalog)
        if catalog.get("repo") != REPO or catalog.get("revision") != REVISION:
            raise ContractError("catalog repo/revision 漂移")
        if tuple(catalog.get("selected_layers", [])) != REGISTERED_LAYERS:
            raise ContractError("catalog S14 层注册表漂移")
        cache = online_range.RangeCache(
            root / "range_cache",
            endpoint=config.endpoint,
            allow_fetch=config.allow_fetch,
            download_budget_bytes=config.download_budget_bytes,
            timeout=300,
        )
        runtime = DecoderRuntime(catalog=catalog, cache=cache)

        for _ in range(config.token_count):
            token_started = time.perf_counter()
            downloaded_before = cache.downloaded_bytes
            token_report: dict[str, Any] = {
                "position": runtime.position,
                "input_token_id": runtime.input_token_id,
                "status": "running",
                "state_committed": False,
                "embedding": None,
                "completed_layers": [],
                "layers": [],
                "final": None,
                "downloaded_bytes_before": downloaded_before,
            }
            report["tokens"].append(token_report)
            # 保留首 token 报告的顶层字段，内容始终指向当前 token。
            report["token_id"] = runtime.input_token_id
            report["embedding"] = None
            report["completed_layers"] = token_report["completed_layers"]
            report["layers"] = token_report["layers"]
            report["final"] = None
            _write_json(report_path, report)

            def worker(
                position: int,
                input_token_id: int,
                previous_states: Mapping[int, LayerRuntimeState],
                session: online_range.RouteFirstSession,
            ) -> TokenComputation:
                nonlocal stage, current_layer
                if position == 0 and previous_states:
                    raise ContractError("position0 不得继承 layer states")
                if position > 0 and set(previous_states) != set(REGISTERED_LAYERS):
                    raise ContractError("decode token 缺少完整的前一 token layer states")

                stage = f"position_{position}_embedding_row_token_{input_token_id}"
                embedding = session.prepare_embedding_row(input_token_id)
                state = _initial_state(embedding, input_token_id)
                token_report["embedding"] = {
                    **_source_summary(embedding),
                    "state": NativeLayerReference._summary_tensor(state),
                }
                report["embedding"] = token_report["embedding"]
                report["downloaded_bytes"] = cache.downloaded_bytes
                _write_json(report_path, report)

                next_layer_states: dict[int, LayerRuntimeState] = {}
                for layer in REGISTERED_LAYERS:
                    current_layer = layer
                    layer_started = time.perf_counter()
                    layer_input = NativeLayerReference._summary_tensor(state)
                    stage = f"position_{position}_layer_{layer}_base"
                    prerequisites = session.prepare_layer(layer, input_token_id)
                    store = TensorStore(root / "range_cache")
                    store.add_ranges((*prerequisites.non_expert, *prerequisites.router))
                    kernel = NativeLayerReference(layer, store)

                    stage = f"position_{position}_layer_{layer}_attention_and_router"
                    pending = kernel.prepare_route(
                        state,
                        token_id=input_token_id,
                        position=position,
                        previous_runtime=previous_states.get(layer),
                    )
                    session.submit_top6(layer, input_token_id, pending.route_ids)
                    layer_report: dict[str, Any] = {
                        "position": position,
                        "input_token_id": input_token_id,
                        "layer": layer,
                        "layer_input": layer_input,
                        "compress_ratio": COMPRESS_RATIOS[layer],
                        "route_source": pending.route_source,
                        "expert_ids": pending.route_ids,
                        "route_weights": pending.route_weights,
                        "route_weight_sum": float(sum(pending.route_weights)),
                        "attention_branch": NativeLayerReference._summary_tensor(pending.attention_branch),
                        "post_attention_state": NativeLayerReference._summary_tensor(pending.post_attention_state),
                        "ffn_input": NativeLayerReference._summary_tensor(pending.ffn_input),
                        "runtime_state": _runtime_state_report(pending.runtime_state),
                        "base_integrity": store.integrity(),
                        "status": "route_ready",
                    }
                    token_report["layers"].append(layer_report)
                    report["downloaded_bytes"] = cache.downloaded_bytes
                    _write_json(report_path, report)

                    stage = f"position_{position}_layer_{layer}_routed_ranges"
                    routed = _fetch_routed_with_retries(
                        session,
                        layer=layer,
                        position=position,
                        token_id=input_token_id,
                        attempts=config.range_attempts,
                        report=report,
                        report_path=report_path,
                        workers=config.range_workers,
                    )
                    routed_ranges = [*routed.shared]
                    for expert_id in pending.route_ids:
                        routed_ranges.extend(routed.experts[expert_id])
                    kernel.add_routed(routed_ranges)

                    stage = f"position_{position}_layer_{layer}_moe_and_hc"
                    moe_branch, state = kernel.finish_layer(pending)
                    if tuple(state.shape) != (1, 1, HC_MULT, HIDDEN_SIZE) or state.dtype != torch.bfloat16:
                        raise ContractError(f"L{layer} 输出破坏 BF16 [1,1,4,4096] 合同")
                    session.finish_layer(layer, input_token_id)
                    next_layer_states[layer] = pending.runtime_state
                    layer_report.update(
                        {
                            "status": "complete",
                            "moe_branch": NativeLayerReference._summary_tensor(moe_branch),
                            "layer_output": NativeLayerReference._summary_tensor(state),
                            "routed_integrity": {
                                "payload_files": len(routed_ranges),
                                "payload_bytes": sum(int(item.entry["bytes"]) for item in routed_ranges),
                                "cache_hits": sum(item.cache_hit for item in routed_ranges),
                                "expert_payload_files": sum(
                                    len(routed.experts[value]) for value in pending.route_ids
                                ),
                                "expert_payload_bytes": sum(
                                    int(item.entry["bytes"])
                                    for value in pending.route_ids
                                    for item in routed.experts[value]
                                ),
                            },
                            "elapsed_s": time.perf_counter() - layer_started,
                        }
                    )
                    token_report["completed_layers"].append(layer)
                    report["downloaded_bytes"] = cache.downloaded_bytes
                    report["current_state"] = NativeLayerReference._summary_tensor(state)
                    _write_json(report_path, report)

                    if config.stop_after_layer == layer:
                        token_report["status"] = "checkpoint_complete"
                        report["status"] = "checkpoint_complete"
                        report["elapsed_s"] = time.perf_counter() - started
                        report["claim_limit"] = (
                            f"真实 token{input_token_id} position{position} 已执行到 S14 L{layer}；"
                            "尚未生成 final token，不代表完整 S14、质量或速度"
                        )
                        _write_json(report_path, report)
                        raise _CheckpointComplete

                stage = f"position_{position}_final_hc_norm_bf16_head"
                current_layer = None
                final_ranges = session.prepare_final()
                head_was_ready = runtime.final_head is not None
                if runtime.final_head is None:
                    runtime.final_head = FinalHeadReference(
                        final_ranges,
                        root / "range_cache",
                        head_chunk_size=config.head_chunk_size,
                    )
                else:
                    runtime.final_head.validate_ranges(final_ranges)
                final = runtime.final_head.forward(state)
                final["head_mmap_reused"] = head_was_ready
                final["text"] = _decode_token(root, int(final["token_id"]))
                token_report["final"] = final
                report["final"] = final
                token_report["status"] = "final_ready_not_committed"
                report["downloaded_bytes"] = cache.downloaded_bytes
                _write_json(report_path, report)
                return TokenComputation(
                    output_token_id=int(final["token_id"]),
                    next_layer_states=next_layer_states,
                    value=token_report,
                )

            runtime.run_token(worker)
            token_report["status"] = "complete"
            token_report["state_committed"] = True
            token_report["elapsed_s"] = time.perf_counter() - token_started
            token_report["downloaded_bytes_after"] = cache.downloaded_bytes
            token_report["downloaded_bytes_new"] = cache.downloaded_bytes - downloaded_before
            report["committed_tokens"] = [dict(item) for item in runtime.snapshot.committed_tokens]
            report["runtime"] = {
                "next_position": runtime.position,
                "next_input_token_id": runtime.input_token_id,
                "committed_layer_states": sorted(runtime.layer_states),
            }
            report["downloaded_bytes"] = cache.downloaded_bytes
            _write_json(report_path, report)

        report["status"] = "complete"
        report["elapsed_s"] = time.perf_counter() - started
        report["claim_limit"] = (
            f"真实 embedding 经固定 S14 14 层、原生 top-6 命中专家与真实 final head 的"
            f"连续 {config.token_count} token CPU correctness 结果；"
            "不代表完整 43 层 DeepSeek-V4、质量或速度"
        )
        _write_json(report_path, report)
        return report
    except _CheckpointComplete:
        return report
    except Exception as exc:
        if cache is not None:
            report["downloaded_bytes"] = cache.downloaded_bytes
        if report["tokens"]:
            report["tokens"][-1]["status"] = "blocked"
            report["tokens"][-1]["state_committed"] = False
        report["status"] = "blocked"
        report["elapsed_s"] = time.perf_counter() - started
        report["committed_tokens"] = (
            [] if runtime is None else [dict(item) for item in runtime.snapshot.committed_tokens]
        )
        report["runtime"] = None if runtime is None else {
            "next_position": runtime.position,
            "next_input_token_id": runtime.input_token_id,
            "committed_layer_states": sorted(runtime.layer_states),
        }
        report["error"] = {
            "stage": stage,
            "layer": current_layer,
            "position": None if runtime is None else runtime.position,
            "input_token_id": None if runtime is None else runtime.input_token_id,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines(),
        }
        _write_json(report_path, report)
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("POLARIS_HF_ENDPOINT", "https://huggingface.co"),
    )
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--download-budget-bytes", type=int, default=0)
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--head-chunk-size", type=int, default=4096)
    parser.add_argument("--range-attempts", type=int, default=4)
    parser.add_argument("--range-workers", type=int, default=3)
    parser.add_argument("--token-count", type=int, choices=(1, 2), default=1)
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ExecutionConfig(
        asset_root=args.asset_root,
        report_path=args.report,
        endpoint=args.endpoint,
        allow_fetch=args.download_missing,
        download_budget_bytes=args.download_budget_bytes,
        stop_after_layer=args.stop_after_layer,
        head_chunk_size=args.head_chunk_size,
        range_attempts=args.range_attempts,
        range_workers=args.range_workers,
        token_count=args.token_count,
    )
    try:
        report = execute(config)
    except Exception as exc:
        print(json.dumps({"status": "invalid_arguments", "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        return 2
    output = {
        "status": report["status"],
        "report": str(config.report_path.resolve()),
        "completed_layers": report.get("completed_layers", []),
        "downloaded_bytes": report.get("downloaded_bytes", 0),
        "final_token_id": (report.get("final") or {}).get("token_id"),
        "committed_tokens": report.get("committed_tokens", []),
        "error": report.get("error"),
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0 if report["status"] in {"complete", "checkpoint_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(_main())
