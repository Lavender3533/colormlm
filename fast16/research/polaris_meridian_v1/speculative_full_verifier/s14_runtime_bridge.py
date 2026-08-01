"""Existing S14 ``DecoderRuntime`` -> speculative snapshot runtime bridge."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from fast16.research.polaris_meridian_v1.s14_first_real_token import executor as s14

from .runtime_controller import NativeTokenStep
from .verifier import DRAFT_PROFILE, S14_LAYERS, VerifierContractError


@dataclass(frozen=True)
class S14BridgeState:
    snapshot: s14.DecoderSnapshot
    context_token_ids: tuple[int, ...]


class S14DecoderRuntimeBridge:
    """Expose a real S14 runtime as a snapshot-isolated K-token drafter.

    ``worker`` is the existing model-specific callback accepted by
    :meth:`s14.DecoderRuntime.run_token`.  The executor currently builds that
    callback beside its report writer, so wiring it into a service only needs
    to lift the callback into a reusable factory; no token or route conversion
    is performed here.

    Speculation starts only after forced prefill is exhausted.  This prevents a
    draft transaction from advancing the prompt queue as if prompt tokens were
    generated continuation tokens.
    """

    profile = DRAFT_PROFILE
    depth = len(S14_LAYERS)
    vocab_size = s14.VOCAB_SIZE

    def __init__(
        self,
        runtime: s14.DecoderRuntime,
        worker: Callable[..., s14.TokenComputation],
        *,
        context_token_ids: Sequence[int],
        tokenizer_fingerprint: str,
    ) -> None:
        context = tuple(context_token_ids)
        if not context or context[-1] != runtime.snapshot.input_token_id:
            raise VerifierContractError("S14 bridge context 必须以 runtime pending token 结尾")
        if len(tokenizer_fingerprint) != 64:
            raise VerifierContractError("S14 tokenizer SHA-256 指纹非法")
        queue = runtime.snapshot.forced_queue
        if queue is not None and queue.active:
            raise VerifierContractError("S14 speculative draft 只能在 forced-prefill 耗尽后开始")
        self.runtime = runtime
        self.worker = worker
        self._context_token_ids = context
        self.tokenizer_fingerprint = tokenizer_fingerprint

    @property
    def context_token_ids(self) -> tuple[int, ...]:
        return self._context_token_ids

    def capture_state(self) -> S14BridgeState:
        return S14BridgeState(self.runtime.snapshot, self._context_token_ids)

    def restore_state(self, state: S14BridgeState) -> None:
        if not isinstance(state, S14BridgeState):
            raise TypeError("S14 bridge state 类型错误")
        self.runtime.snapshot = state.snapshot
        self._context_token_ids = state.context_token_ids

    def replace_pending_token(self, state: S14BridgeState, token_id: int) -> S14BridgeState:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < self.vocab_size:
            raise VerifierContractError("S14 pending token ID 越界")
        if not state.context_token_ids or not state.snapshot.committed_tokens:
            raise VerifierContractError("S14 回退点缺少可替换的 pending token")
        queue = state.snapshot.forced_queue
        if queue is not None and queue.active:
            raise VerifierContractError("禁止修改未耗尽 forced-prefill 的 pending token")
        records = list(state.snapshot.committed_tokens)
        last = dict(records[-1])
        last["output_token_id"] = token_id
        if "next_input_token_id" in last:
            last["next_input_token_id"] = token_id
        records[-1] = last
        snapshot = replace(
            state.snapshot,
            input_token_id=token_id,
            committed_tokens=tuple(records),
        )
        return S14BridgeState(
            snapshot=snapshot,
            context_token_ids=state.context_token_ids[:-1] + (token_id,),
        )

    def step(self, *, forced_next_token_id: int | None = None) -> NativeTokenStep:
        if forced_next_token_id is not None:
            raise VerifierContractError("S14 草稿 bridge 不接受 teacher-force")
        queue = self.runtime.snapshot.forced_queue
        if queue is not None and queue.active:
            raise VerifierContractError("S14 speculative step 不得消费 forced-prefill token")
        value = self.runtime.run_token(self.worker)
        if not isinstance(value, Mapping):
            raise VerifierContractError("S14 worker 未返回 token report mapping")
        records = self.runtime.snapshot.committed_tokens
        if not records:
            raise VerifierContractError("S14 worker 未提交 token")
        token_id = int(records[-1]["output_token_id"])
        layers = value.get("layers")
        if not isinstance(layers, list):
            raise VerifierContractError("S14 token report 缺少 layers")
        routes: dict[int, tuple[int, ...]] = {}
        for row in layers:
            if not isinstance(row, Mapping):
                raise VerifierContractError("S14 layer report 类型错误")
            layer = int(row["layer"])
            if layer in routes:
                raise VerifierContractError(f"S14 layer report 重复 L{layer}")
            routes[layer] = tuple(int(value) for value in row["expert_ids"])
        if set(routes) != set(S14_LAYERS):
            raise VerifierContractError("S14 token report 未覆盖冻结 14 层")
        self._context_token_ids = self._context_token_ids + (token_id,)
        return NativeTokenStep(token_id, routes)
