"""以真实 Range 页执行 Polaris S14 的 BOS 首 token。

执行器只接受固定 revision、固定 14 层和 token 0。每层先读取 non-expert/router，
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
from typing import Any, Iterable, Mapping, Sequence

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
    window_kv: torch.Tensor
    compressor: CompressorRemainderState | None


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

    def __init__(self, layer: int, store: TensorStore):
        if layer not in REGISTERED_LAYERS:
            raise ContractError(f"未预注册层: {layer}")
        self.layer = layer
        self.store = store
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

    def _write_token0_compressor_state(self, x: torch.Tensor) -> CompressorRemainderState | None:
        ratio = COMPRESS_RATIOS[self.layer]
        if ratio == 0:
            return None
        overlap = ratio == 4
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

    def _attention(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, LayerRuntimeState]:
        prefix = f"layers.{self.layer}"
        branch, post, comb = hc_pre(
            state,
            self._load_tensor(prefix + ".hc_attn_fn"),
            self._load_tensor(prefix + ".hc_attn_scale"),
            self._load_tensor(prefix + ".hc_attn_base"),
        )
        branch = self._rms_norm(branch, self._load_tensor(prefix + ".attn_norm.weight"))
        compressor_state = self._write_token0_compressor_state(branch)
        branch_np = branch.float().numpy()

        q_low = torch.from_numpy(self._linear_fp8(branch_np, prefix + ".attn.wq_a")).to(torch.bfloat16)
        q_low = self._rms_norm(q_low, self._load_tensor(prefix + ".attn.q_norm.weight"))
        q = torch.from_numpy(self._linear_fp8(q_low.float().numpy(), prefix + ".attn.wq_b"))
        q = q.to(torch.bfloat16).reshape(1, 1, 64, 512)
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)

        kv = torch.from_numpy(self._linear_fp8(branch_np, prefix + ".attn.wkv")).to(torch.bfloat16)
        kv = self._rms_norm(kv, self._load_tensor(prefix + ".attn.kv_norm.weight"))
        kv[..., :-64] = torch.from_numpy(
            self._activation_quant(kv[..., :-64].float().numpy(), 64)
        ).to(torch.bfloat16)
        # token0/position0 的 RoPE 为恒等；窗口中只有位置 0。
        attention = sparse_attention(
            q,
            kv,
            self._load_tensor(prefix + ".attn.attn_sink"),
            torch.tensor([[[0]]], dtype=torch.int32),
            softmax_scale=512**-0.5,
            output_dtype=torch.bfloat16,
        )
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
        window_kv = torch.zeros((1, WINDOW_SIZE, 512), dtype=torch.bfloat16)
        window_kv[:, 0] = kv[:, 0]
        return attention_branch, post_attention_state, LayerRuntimeState(
            layer=self.layer,
            window_kv=window_kv,
            compressor=compressor_state,
        )

    def _route(self, ffn_input: torch.Tensor) -> tuple[list[int], list[float], str]:
        prefix = f"layers.{self.layer}.ffn.gate"
        flat = ffn_input.reshape(-1, HIDDEN_SIZE)
        logits = F.linear(flat.float(), self._load_tensor(prefix + ".weight").float())
        scores = F.softplus(logits).sqrt()
        if self.layer < 3:
            physical = self._load_i64(prefix + ".tid2eid")
            row = physical[BOS_TOKEN_ID]
            if bool(((row < 0) | (row >= 256)).any().item()):
                raise ContractError(f"L{self.layer} token0 tid2eid 含越界 expert")
            route_ids = [int(value) for value in row.tolist()]
            anchor = HASH_ROUTE_ANCHORS.get(self.layer)
            if anchor is not None and route_ids != anchor:
                raise ContractError(f"L{self.layer} token0 hash route 漂移: {route_ids}")
            indices = row.to(torch.int64).view(1, TOP_K)
            source = "token0_tid2eid_physical_i64"
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
        if not math.isclose(sum(route_weights), ROUTE_SCALE, rel_tol=0, abs_tol=2e-7):
            raise ContractError(f"L{self.layer} route 权重和不等于 1.5")
        return route_ids, route_weights, source

    def prepare_route(self, state: torch.Tensor) -> PendingLayer:
        if tuple(state.shape) != (1, 1, HC_MULT, HIDDEN_SIZE) or state.dtype != torch.bfloat16:
            raise ContractError("每层输入必须是 BF16 [1,1,4,4096]")
        attention_branch, post_attention_state, runtime_state = self._attention(state)
        prefix = f"layers.{self.layer}"
        ffn_input, post_ffn, comb_ffn = hc_pre(
            post_attention_state,
            self._load_tensor(prefix + ".hc_ffn_fn"),
            self._load_tensor(prefix + ".hc_ffn_scale"),
            self._load_tensor(prefix + ".hc_ffn_base"),
        )
        ffn_input = self._rms_norm(ffn_input, self._load_tensor(prefix + ".ffn_norm.weight"))
        route_ids, route_weights, route_source = self._route(ffn_input)
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
    report: dict[str, Any] = {
        "window_kv_shape": list(state.window_kv.shape),
        "window_position0": NativeLayerReference._summary_tensor(state.window_kv[:, :1]),
    }
    compressor = state.compressor
    if compressor is None:
        report["compressor"] = None
        return report
    offset = compressor.ratio if compressor.overlap else 0
    comp: dict[str, Any] = {
        "ratio": compressor.ratio,
        "overlap": compressor.overlap,
        "main_kv_state_shape": list(compressor.main_kv_state.shape),
        "main_score_state_shape": list(compressor.main_score_state.shape),
        "written_row": offset,
        "main_kv_written": NativeLayerReference._summary_tensor(
            compressor.main_kv_state[:, offset : offset + 1]
        ),
        "main_score_written": NativeLayerReference._summary_tensor(
            compressor.main_score_state[:, offset : offset + 1]
        ),
    }
    if compressor.indexer_kv_state is not None and compressor.indexer_score_state is not None:
        comp.update(
            {
                "indexer_kv_state_shape": list(compressor.indexer_kv_state.shape),
                "indexer_score_state_shape": list(compressor.indexer_score_state.shape),
                "indexer_written_row": offset,
                "indexer_kv_written": NativeLayerReference._summary_tensor(
                    compressor.indexer_kv_state[:, offset : offset + 1]
                ),
                "indexer_score_written": NativeLayerReference._summary_tensor(
                    compressor.indexer_score_state[:, offset : offset + 1]
                ),
            }
        )
    report["compressor"] = comp
    return report


def _initial_state(embedding: online_range.CachedRange) -> torch.Tensor:
    entry = embedding.entry
    if (
        entry.get("tensor") != "embed.weight[0:1]"
        or entry.get("parent_tensor") != "embed.weight"
        or entry.get("row_start") != BOS_TOKEN_ID
        or entry.get("row_end_exclusive") != BOS_TOKEN_ID + 1
        or entry.get("dtype") != "BF16"
    ):
        raise ContractError("embedding row tensor/dtype 漂移")
    if entry.get("shape") != [1, HIDDEN_SIZE] or entry.get("bytes") != HIDDEN_SIZE * 2:
        raise ContractError("embedding row 必须是 BF16 [1,4096]")
    payload = bytearray(embedding.path.read_bytes())
    row = torch.frombuffer(payload, dtype=torch.bfloat16).reshape(HIDDEN_SIZE).clone()
    if not bool(torch.isfinite(row).all().item()):
        raise ContractError("token0 embedding row 含非有限值")
    return row.view(1, 1, 1, HIDDEN_SIZE).repeat(1, 1, HC_MULT, 1)


def _decode_token(asset_root: Path, token_id: int) -> str | None:
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(asset_root / "tokenizer.json")).decode([token_id])
    except Exception:
        return None


def _final_forward(
    state: torch.Tensor,
    final_ranges: Sequence[online_range.CachedRange],
    cache_root: Path,
    *,
    head_chunk_size: int,
) -> dict[str, Any]:
    store = TensorStore(cache_root)
    store.add_ranges(final_ranges)
    bundle = store.bundle()
    helper = _InlineForward(bundle)
    required = {"hc_head_base", "hc_head_fn", "hc_head_scale", "norm.weight", "head.weight"}
    if set(store.sources) != required:
        raise ContractError(f"final tensor 集漂移: {sorted(store.sources)}")
    head_source = store.source("head.weight")
    if head_source.entry.get("dtype") != "BF16" or head_source.entry.get("shape") != [VOCAB_SIZE, HIDDEN_SIZE]:
        raise ContractError("真实 final head 必须是 BF16 [129280,4096]")
    head = torch.from_file(
        str(head_source.path),
        shared=False,
        size=VOCAB_SIZE * HIDDEN_SIZE,
        dtype=torch.bfloat16,
    ).reshape(VOCAB_SIZE, HIDDEN_SIZE)
    logits, normalized, pre = native_final_logits(
        state,
        helper._load_tensor("hc_head_fn"),
        helper._load_tensor("hc_head_scale"),
        helper._load_tensor("hc_head_base"),
        helper._load_tensor("norm.weight"),
        head,
        output_chunk_size=head_chunk_size,
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
        "integrity": store.integrity(),
    }


def _base_report(config: ExecutionConfig) -> dict[str, Any]:
    return {
        "format": "polaris-s14-first-real-token-report-v1",
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
        "range_retries": [],
        "routed_prefetch_plans": [],
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
                workers=workers,
                report=report,
                report_path=report_path,
            )
            return session.fetch_routed(layer, token_id)
        except _RECOVERABLE_TRANSPORT_ERRORS as exc:
            event = {
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
        (item for item in report["routed_prefetch_plans"] if item["layer"] == layer),
        None,
    )
    if existing is None:
        plan = {
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
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"s14-l{layer}-range") as pool:
        futures = [pool.submit(session.cache.fetch, entry) for entry in unique.values()]
        try:
            for future in as_completed(futures):
                future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise


def execute(config: ExecutionConfig) -> dict[str, Any]:
    """运行固定 S14 前缀或全部 14 层，并始终落一份机器可读报告。"""

    config.validate()
    report = _base_report(config)
    report_path = config.report_path.resolve()
    _write_json(report_path, report)
    stage = "bootstrap"
    current_layer: int | None = None
    cache: online_range.RangeCache | None = None
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
        session = online_range.RouteFirstSession(catalog, cache)

        stage = "embedding_row_token0"
        embedding = session.prepare_embedding_row(BOS_TOKEN_ID)
        state = _initial_state(embedding)
        report["embedding"] = {
            **_source_summary(embedding),
            "state": NativeLayerReference._summary_tensor(state),
        }
        _write_json(report_path, report)

        runtime_states: dict[int, LayerRuntimeState] = {}
        for layer in REGISTERED_LAYERS:
            current_layer = layer
            layer_started = time.perf_counter()
            layer_input = NativeLayerReference._summary_tensor(state)
            stage = f"layer_{layer}_base"
            prerequisites = session.prepare_layer(layer, BOS_TOKEN_ID)
            store = TensorStore(root / "range_cache")
            store.add_ranges((*prerequisites.non_expert, *prerequisites.router))
            kernel = NativeLayerReference(layer, store)

            stage = f"layer_{layer}_attention_and_router"
            pending = kernel.prepare_route(state)
            session.submit_top6(layer, BOS_TOKEN_ID, pending.route_ids)
            layer_report: dict[str, Any] = {
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
            report["layers"].append(layer_report)
            report["downloaded_bytes"] = cache.downloaded_bytes
            _write_json(report_path, report)

            stage = f"layer_{layer}_routed_ranges"
            routed = _fetch_routed_with_retries(
                session,
                layer=layer,
                token_id=BOS_TOKEN_ID,
                attempts=config.range_attempts,
                report=report,
                report_path=report_path,
                workers=config.range_workers,
            )
            routed_ranges = [*routed.shared]
            for expert_id in pending.route_ids:
                routed_ranges.extend(routed.experts[expert_id])
            kernel.add_routed(routed_ranges)

            stage = f"layer_{layer}_moe_and_hc"
            moe_branch, state = kernel.finish_layer(pending)
            if tuple(state.shape) != (1, 1, HC_MULT, HIDDEN_SIZE) or state.dtype != torch.bfloat16:
                raise ContractError(f"L{layer} 输出破坏 BF16 [1,1,4,4096] 合同")
            session.finish_layer(layer, BOS_TOKEN_ID)
            runtime_states[layer] = pending.runtime_state
            layer_report.update(
                {
                    "status": "complete",
                    "moe_branch": NativeLayerReference._summary_tensor(moe_branch),
                    "layer_output": NativeLayerReference._summary_tensor(state),
                    "routed_integrity": {
                        "payload_files": len(routed_ranges),
                        "payload_bytes": sum(int(item.entry["bytes"]) for item in routed_ranges),
                        "cache_hits": sum(item.cache_hit for item in routed_ranges),
                        "expert_payload_files": sum(len(routed.experts[value]) for value in pending.route_ids),
                        "expert_payload_bytes": sum(
                            int(item.entry["bytes"])
                            for value in pending.route_ids
                            for item in routed.experts[value]
                        ),
                    },
                    "elapsed_s": time.perf_counter() - layer_started,
                }
            )
            report["completed_layers"].append(layer)
            report["downloaded_bytes"] = cache.downloaded_bytes
            report["current_state"] = NativeLayerReference._summary_tensor(state)
            _write_json(report_path, report)

            if config.stop_after_layer == layer:
                report["status"] = "checkpoint_complete"
                report["elapsed_s"] = time.perf_counter() - started
                report["claim_limit"] = (
                    f"真实 BOS token0 已执行到 S14 L{layer}；尚未生成 final token，"
                    "不代表完整 S14、质量或速度"
                )
                _write_json(report_path, report)
                return report

        stage = "final_hc_norm_bf16_head"
        current_layer = None
        final_ranges = session.prepare_final()
        final = _final_forward(
            state,
            final_ranges,
            root / "range_cache",
            head_chunk_size=config.head_chunk_size,
        )
        final["text"] = _decode_token(root, int(final["token_id"]))
        report["final"] = final
        report["status"] = "complete"
        report["downloaded_bytes"] = cache.downloaded_bytes
        report["elapsed_s"] = time.perf_counter() - started
        report["claim_limit"] = (
            "真实 BOS embedding 经固定 S14 14 层、原生 top-6 命中专家与真实 final head 的"
            "第一个 token correctness 结果；不代表完整 43 层 DeepSeek-V4、质量或速度"
        )
        _write_json(report_path, report)
        return report
    except Exception as exc:
        if cache is not None:
            report["downloaded_bytes"] = cache.downloaded_bytes
        report["status"] = "blocked"
        report["elapsed_s"] = time.perf_counter() - started
        report["error"] = {
            "stage": stage,
            "layer": current_layer,
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
        "error": report.get("error"),
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0 if report["status"] in {"complete", "checkpoint_complete"} else 2


if __name__ == "__main__":
    raise SystemExit(_main())
