"""FullDepth43 ``DecoderState`` -> causal-block snapshot bridge.

本桥可直接接入 ``executor.FullDepthTokenWorker``，复用真实
``DecoderState`` 和 ``LayerRuntimeState``；不转换 token ID，不简化
KV/compressor 状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6 import executor as fd43
from fast16.research.polaris_meridian_v1.s14_first_real_token import executor as s14

from .assets import LAYERS, TOP_K
from .runtime_controller import NativeTokenStep
from .verifier import VERIFIER_PROFILE, VerifierContractError


# Backward-compatible public name; there is now one authoritative result type
# shared by the CLI executor and the causal-block runtime.
FullDepthTokenComputation = fd43.FullDepthTokenComputation


@dataclass(frozen=True)
class FullDepthBridgeState:
    decoder: fd43.DecoderState
    context_token_ids: tuple[int, ...]


def _validate_decoder_context(
    decoder: fd43.DecoderState,
    context_token_ids: tuple[int, ...],
) -> None:
    """证明声明的 token 前缀与 decoder/KV 历史是同一条链。"""

    records = decoder.committed_tokens
    if decoder.position != len(records):
        raise VerifierContractError(
            "FullDepth decoder position 与 committed ledger 长度不一致"
        )
    if len(context_token_ids) != decoder.position + 1:
        raise VerifierContractError(
            "FullDepth bridge context 长度与 decoder position 不一致"
        )
    if not context_token_ids or context_token_ids[-1] != decoder.input_token_id:
        raise VerifierContractError(
            "FullDepth bridge context 必须以 decoder pending token 结尾"
        )
    for position, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise VerifierContractError("FullDepth committed ledger row 类型错误")
        if row.get("position") != position:
            raise VerifierContractError(
                f"FullDepth committed ledger position 不连续: {position}"
            )
        input_token_id = row.get("input_token_id")
        output_token_id = row.get("output_token_id")
        if (
            isinstance(input_token_id, bool)
            or not isinstance(input_token_id, int)
            or not 0 <= input_token_id < s14.VOCAB_SIZE
            or isinstance(output_token_id, bool)
            or not isinstance(output_token_id, int)
            or not 0 <= output_token_id < s14.VOCAB_SIZE
        ):
            raise VerifierContractError(
                f"FullDepth committed ledger token ID 非法: position={position}"
            )
        if input_token_id != context_token_ids[position]:
            raise VerifierContractError(
                f"FullDepth context 与 committed input 不一致: position={position}"
            )
        next_input = row.get("next_input_token_id", output_token_id)
        if next_input != context_token_ids[position + 1]:
            raise VerifierContractError(
                f"FullDepth committed pending 链断裂: position={position}"
            )
    if decoder.position:
        if set(decoder.layer_states) != set(range(LAYERS)):
            raise VerifierContractError("FullDepth decoder 非零 position 缺少 43 层状态")
        for layer in range(LAYERS):
            state = decoder.layer_states[layer]
            if state.layer != layer or state.position != decoder.position - 1:
                raise VerifierContractError(
                    f"FullDepth L{layer} KV/compressor position 与 decoder 不一致"
                )
    elif decoder.layer_states:
        raise VerifierContractError("FullDepth position0 不得携带旧层状态")


def _clone_decoder(state: fd43.DecoderState) -> fd43.DecoderState:
    return fd43.DecoderState(
        position=state.position,
        input_token_id=state.input_token_id,
        layer_states={
            layer: s14._clone_layer_state(layer_state)
            for layer, layer_state in state.layer_states.items()
        },
        committed_tokens=[dict(row) for row in state.committed_tokens],
        forced_queue=state.forced_queue,
    )


def _validate_routes(routes: Mapping[int, tuple[int, ...]]) -> None:
    if set(routes) != set(range(LAYERS)):
        raise VerifierContractError("FullDepth worker 未覆盖 43 层原生 route")
    for layer in range(LAYERS):
        experts = tuple(routes[layer])
        if len(experts) != TOP_K or len(set(experts)) != TOP_K:
            raise VerifierContractError(f"FullDepth L{layer} 不是 6 个唯一专家")
        if any(
            isinstance(expert, bool)
            or not isinstance(expert, int)
            or not 0 <= expert < 256
            for expert in experts
        ):
            raise VerifierContractError(f"FullDepth L{layer} expert ID 越界")


class FullDepthDecoderStateBridge:
    """把真实 FullDepth43 decoder 暴露为 snapshot token runtime。"""

    profile = VERIFIER_PROFILE
    depth = LAYERS
    vocab_size = s14.VOCAB_SIZE

    def __init__(
        self,
        decoder: fd43.DecoderState,
        worker: Callable[
            [int, int, Mapping[int, s14.LayerRuntimeState]],
            FullDepthTokenComputation,
        ],
        *,
        context_token_ids: Sequence[int],
        tokenizer_fingerprint: str,
    ) -> None:
        context = tuple(context_token_ids)
        _validate_decoder_context(decoder, context)
        if not isinstance(tokenizer_fingerprint, str) or not tokenizer_fingerprint:
            raise VerifierContractError("FullDepth tokenizer 指纹为空")
        queue = decoder.forced_queue
        if queue is not None and queue.active:
            raise VerifierContractError(
                "FullDepth causal block 只能在 forced-prefill 耗尽后开始"
            )
        decoder.previous_for(fd43.FULLDEPTH43_NATIVE_TOP6)
        self.decoder = decoder
        self.worker = worker
        self._context_token_ids = context
        self.tokenizer_fingerprint = tokenizer_fingerprint

    @property
    def context_token_ids(self) -> tuple[int, ...]:
        return self._context_token_ids

    def capture_state(self) -> FullDepthBridgeState:
        _validate_decoder_context(self.decoder, self._context_token_ids)
        return FullDepthBridgeState(
            decoder=_clone_decoder(self.decoder),
            context_token_ids=self._context_token_ids,
        )

    def restore_state(self, state: FullDepthBridgeState) -> None:
        if not isinstance(state, FullDepthBridgeState):
            raise TypeError("FullDepth bridge state 类型错误")
        _validate_decoder_context(state.decoder, state.context_token_ids)
        restored = _clone_decoder(state.decoder)
        self.decoder.__dict__.update(restored.__dict__)
        self._context_token_ids = state.context_token_ids

    def replace_pending_token(
        self, state: FullDepthBridgeState, token_id: int
    ) -> FullDepthBridgeState:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < self.vocab_size
        ):
            raise VerifierContractError("FullDepth pending token ID 越界")
        snapshot = _clone_decoder(state.decoder)
        queue = snapshot.forced_queue
        if queue is not None and queue.active:
            raise VerifierContractError("禁止修改未耗尽 forced-prefill")
        if not state.context_token_ids or not snapshot.committed_tokens:
            raise VerifierContractError("FullDepth 回退点缺少 pending token")
        last = dict(snapshot.committed_tokens[-1])
        last["output_token_id"] = token_id
        if "next_input_token_id" in last:
            last["next_input_token_id"] = token_id
        snapshot.committed_tokens[-1] = last
        snapshot.input_token_id = token_id
        replaced = FullDepthBridgeState(
            decoder=snapshot,
            context_token_ids=state.context_token_ids[:-1] + (token_id,),
        )
        _validate_decoder_context(replaced.decoder, replaced.context_token_ids)
        return replaced

    def step(self, *, forced_next_token_id: int | None = None) -> NativeTokenStep:
        _validate_decoder_context(self.decoder, self._context_token_ids)
        queue = self.decoder.forced_queue
        if queue is not None and queue.active:
            raise VerifierContractError(
                "FullDepth causal step 不得消费 forced-prefill token"
            )
        previous = self.decoder.previous_for(fd43.FULLDEPTH43_NATIVE_TOP6)
        position = self.decoder.position
        computation = self.worker(
            position,
            self.decoder.input_token_id,
            previous,
        )
        prediction = computation.predicted_token_id
        if (
            isinstance(prediction, bool)
            or not isinstance(prediction, int)
            or not 0 <= prediction < self.vocab_size
        ):
            raise VerifierContractError("FullDepth worker 预测 token ID 越界")
        _validate_routes(computation.top6_by_layer)
        pending = prediction if forced_next_token_id is None else forced_next_token_id
        if (
            isinstance(pending, bool)
            or not isinstance(pending, int)
            or not 0 <= pending < self.vocab_size
        ):
            raise VerifierContractError("FullDepth teacher-force token ID 越界")

        # DecoderState.commit 同时验证 43 层 state 的 layer/position，
        # 且只在全部成功后一次替换已提交字段。
        self.decoder.commit(
            output_token_id=pending,
            next_states=computation.next_layer_states,
            profile=fd43.FULLDEPTH43_NATIVE_TOP6,
        )
        self._context_token_ids += (pending,)
        _validate_decoder_context(self.decoder, self._context_token_ids)
        return NativeTokenStep(prediction, dict(computation.top6_by_layer))


def build_cpu_causal_block_reference_backend(
    decoder: fd43.DecoderState,
    worker: Callable[
        [int, int, Mapping[int, s14.LayerRuntimeState]],
        FullDepthTokenComputation,
    ],
    *,
    context_token_ids: Sequence[int],
    tokenizer_fingerprint: str,
) -> Any:
    """把真实 FullDepth worker 和 DecoderState 接入 CPU K=1/4/8 事务。

    返回值仍是串行 CPU 正确性参考；``forward_calls=K``，不是
    batched/GPU 加速路径。局部导入避免 package ``__init__`` 顺序循环。
    """

    from .cpu_causal_block import CpuCausalBlockReferenceBackend

    bridge = FullDepthDecoderStateBridge(
        decoder,
        worker,
        context_token_ids=context_token_ids,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    return CpuCausalBlockReferenceBackend(bridge)
