"""K=4/8 speculative block 的原子接受、回退与 runtime 边界。

这里的控制面只接受 DeepSeek 原生 token ID。草稿与 FullDepth43 验证器
都在私有事务中先行；只有两边都准备成功后才替换已提交状态。
任何异常都恢复两个 runtime 的轮次前 snapshot，避免被拒绝的草稿 KV
或 compressor remainder 泄漏到下一轮。

``SerialSnapshotVerifierBackend`` 是当前 FullDepth43 参考执行器的正确性
边界：它每个草稿位置调用一次 target，因此不具备加速资格。
``NativeBatchedVerifierBackend`` 则是生产边界：一次 causal-block 调用返回
K 个对齐预测，并由后端事务保留正确的 KV 前缀。
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Protocol, Sequence

from .assets import EXPERTS_PER_LAYER, LAYERS, S14_LAYERS, TOP_K
from .cache_replay import RouteBlock
from .verifier import (
    DRAFT_PROFILE,
    VERIFIER_PROFILE,
    DraftStep,
    FullDepthVerification,
    VerificationRequest,
    VerifierContractError,
)


SUPPORTED_BLOCK_SIZES = (4, 8)
BATCHED_CAUSAL_MODE = "batched_causal"
SERIAL_REFERENCE_MODE = "serial_reference"


@dataclass(frozen=True)
class NativeTokenStep:
    """One greedy prediction and the native routes used to produce it.

    ``forced_next_token_id`` on :class:`SnapshotTokenRuntime.step` changes only
    the next pending input used inside an uncommitted verification transaction;
    ``predicted_token_id`` always remains the native greedy target prediction.
    """

    predicted_token_id: int
    top6_by_layer: Mapping[int, tuple[int, ...]]


class SnapshotTokenRuntime(Protocol):
    """Minimal bridge for the existing serial S14/FullDepth runtimes.

    A concrete bridge owns model-specific KV/cache objects.  Snapshots must be
    immutable or copy-on-write: restoring the pre-round object must undo every
    model-visible state change from the speculative round.
    """

    profile: str
    depth: int
    vocab_size: int
    tokenizer_fingerprint: str

    @property
    def context_token_ids(self) -> tuple[int, ...]: ...

    def capture_state(self) -> Any: ...

    def restore_state(self, state: Any) -> None: ...

    def replace_pending_token(self, state: Any, token_id: int) -> Any: ...

    def step(self, *, forced_next_token_id: int | None = None) -> NativeTokenStep: ...


@dataclass(frozen=True)
class TransactionMetrics:
    mode: str
    forward_calls: int
    elapsed_seconds: float


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted_prefix: tuple[int, ...]
    fallback_token_id: int | None
    rejected_draft_suffix: tuple[int, ...]
    committed_token_ids: tuple[int, ...]
    mismatch_index: int | None

    @property
    def accepted_length(self) -> int:
        return len(self.accepted_prefix)


@dataclass(frozen=True)
class SpeculativeRoundResult:
    decision: AcceptanceDecision
    native_routes: RouteBlock
    block_size: int
    draft_metrics: TransactionMetrics
    verifier_metrics: TransactionMetrics

    @property
    def speed_eligible_verifier(self) -> bool:
        return (
            self.verifier_metrics.mode == BATCHED_CAUSAL_MODE
            and self.verifier_metrics.forward_calls == 1
        )


class DraftBlockTransaction(Protocol):
    steps: tuple[DraftStep, ...]
    metrics: TransactionMetrics

    def prepare_commit(self, decision: AcceptanceDecision) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class VerifierBlockTransaction(Protocol):
    verification: FullDepthVerification
    metrics: TransactionMetrics

    def prepare_commit(self, decision: AcceptanceDecision) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class DraftBlockBackend(Protocol):
    tokenizer_fingerprint: str
    vocab_size: int

    @property
    def context_token_ids(self) -> tuple[int, ...]: ...

    def begin_block(
        self, context_token_ids: Sequence[int], block_size: int
    ) -> DraftBlockTransaction: ...


class VerifierBlockBackend(Protocol):
    tokenizer_fingerprint: str
    vocab_size: int

    @property
    def context_token_ids(self) -> tuple[int, ...]: ...

    def begin_block(self, request: VerificationRequest) -> VerifierBlockTransaction: ...


class BatchedVerifierRuntime(Protocol):
    """Next FullDepth43 GPU boundary; one call must cover the whole block.

    The returned transaction must already contain K causally aligned target
    predictions and Kx43x6 native routes.  ``prepare_commit`` must select the
    KV/compressor prefix ending at the mismatch fallback (or the full match),
    while ``rollback`` must restore the exact pre-block state.
    """

    tokenizer_fingerprint: str
    vocab_size: int

    @property
    def context_token_ids(self) -> tuple[int, ...]: ...

    def begin_causal_block(self, request: VerificationRequest) -> VerifierBlockTransaction: ...


def _validate_block_size(block_size: int) -> None:
    if isinstance(block_size, bool) or block_size not in SUPPORTED_BLOCK_SIZES:
        raise VerifierContractError(f"block_size 必须是 {SUPPORTED_BLOCK_SIZES} 之一")


def _validate_route_map(
    routes: Mapping[int, tuple[int, ...]], expected_layers: Sequence[int]
) -> None:
    if set(routes) != set(expected_layers):
        raise VerifierContractError("原生 route 层集不完整")
    for layer in expected_layers:
        expert_ids = tuple(routes[layer])
        if len(expert_ids) != TOP_K or len(set(expert_ids)) != TOP_K:
            raise VerifierContractError(f"L{layer} 必须有 {TOP_K} 个不同专家")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < EXPERTS_PER_LAYER
            for value in expert_ids
        ):
            raise VerifierContractError(f"L{layer} expert ID 越界")


def decide_acceptance(
    draft_token_ids: Sequence[int], predicted_token_ids: Sequence[int]
) -> AcceptanceDecision:
    draft = tuple(draft_token_ids)
    predictions = tuple(predicted_token_ids)
    if not draft or len(predictions) != len(draft):
        raise VerifierContractError("target 预测必须与非空草稿块一一对齐")
    mismatch = next(
        (index for index, pair in enumerate(zip(draft, predictions)) if pair[0] != pair[1]),
        None,
    )
    if mismatch is None:
        return AcceptanceDecision(draft, None, (), draft, None)
    accepted = draft[:mismatch]
    fallback = predictions[mismatch]
    return AcceptanceDecision(
        accepted_prefix=accepted,
        fallback_token_id=fallback,
        rejected_draft_suffix=draft[mismatch:],
        committed_token_ids=accepted + (fallback,),
        mismatch_index=mismatch,
    )


class _SnapshotDraftTransaction:
    def __init__(
        self,
        runtime: SnapshotTokenRuntime,
        base_state: Any,
        checkpoints: Sequence[Any],
        steps: Sequence[DraftStep],
        elapsed_seconds: float,
    ) -> None:
        self._runtime = runtime
        self._base_state = base_state
        self._checkpoints = tuple(checkpoints)
        self.steps = tuple(steps)
        self.metrics = TransactionMetrics(
            mode=SERIAL_REFERENCE_MODE,
            forward_calls=len(self.steps),
            elapsed_seconds=elapsed_seconds,
        )
        self._prepared_state: Any | None = None

    def prepare_commit(self, decision: AcceptanceDecision) -> None:
        if decision.mismatch_index is None:
            self._prepared_state = self._checkpoints[-1]
            return
        checkpoint = self._checkpoints[decision.mismatch_index]
        assert decision.fallback_token_id is not None
        self._prepared_state = self._runtime.replace_pending_token(
            checkpoint, decision.fallback_token_id
        )

    def commit(self) -> None:
        if self._prepared_state is None:
            raise VerifierContractError("草稿事务未 prepare")
        self._runtime.restore_state(self._prepared_state)

    def rollback(self) -> None:
        self._runtime.restore_state(self._base_state)


class _SnapshotVerifierTransaction:
    def __init__(
        self,
        runtime: SnapshotTokenRuntime,
        base_state: Any,
        checkpoints: Sequence[Any],
        verification: FullDepthVerification,
        elapsed_seconds: float,
    ) -> None:
        self._runtime = runtime
        self._base_state = base_state
        self._checkpoints = tuple(checkpoints)
        self.verification = verification
        self.metrics = TransactionMetrics(
            mode=SERIAL_REFERENCE_MODE,
            forward_calls=len(verification.predicted_token_ids),
            elapsed_seconds=elapsed_seconds,
        )
        self._prepared_state: Any | None = None

    def prepare_commit(self, decision: AcceptanceDecision) -> None:
        if decision.mismatch_index is None:
            self._prepared_state = self._checkpoints[-1]
            return
        checkpoint = self._checkpoints[decision.mismatch_index]
        assert decision.fallback_token_id is not None
        self._prepared_state = self._runtime.replace_pending_token(
            checkpoint, decision.fallback_token_id
        )

    def commit(self) -> None:
        if self._prepared_state is None:
            raise VerifierContractError("verifier 事务未 prepare")
        self._runtime.restore_state(self._prepared_state)

    def rollback(self) -> None:
        self._runtime.restore_state(self._base_state)


class SerialSnapshotDraftBackend:
    """Generate K S14 tokens using snapshot-isolated serial steps."""

    def __init__(self, runtime: SnapshotTokenRuntime):
        if runtime.profile != DRAFT_PROFILE or runtime.depth != len(S14_LAYERS):
            raise VerifierContractError("草稿 runtime 必须是 S14/top6")
        self.runtime = runtime
        self.tokenizer_fingerprint = runtime.tokenizer_fingerprint
        self.vocab_size = runtime.vocab_size

    @property
    def context_token_ids(self) -> tuple[int, ...]:
        return self.runtime.context_token_ids

    def begin_block(
        self, context_token_ids: Sequence[int], block_size: int
    ) -> _SnapshotDraftTransaction:
        _validate_block_size(block_size)
        if tuple(context_token_ids) != self.context_token_ids:
            raise VerifierContractError("草稿 runtime context 与控制器漂移")
        base = self.runtime.capture_state()
        steps: list[DraftStep] = []
        checkpoints: list[Any] = []
        started = time.perf_counter()
        try:
            for _ in range(block_size):
                step = self.runtime.step()
                if not 0 <= step.predicted_token_id < self.vocab_size:
                    raise VerifierContractError("草稿 token ID 越界")
                _validate_route_map(step.top6_by_layer, S14_LAYERS)
                steps.append(DraftStep(step.predicted_token_id, dict(step.top6_by_layer)))
                checkpoints.append(self.runtime.capture_state())
        except Exception:
            self.runtime.restore_state(base)
            raise
        return _SnapshotDraftTransaction(
            self.runtime, base, checkpoints, steps, time.perf_counter() - started
        )


class SerialSnapshotVerifierBackend:
    """Runnable correctness bridge; K target calls mean it is not a speed path."""

    def __init__(self, runtime: SnapshotTokenRuntime):
        if runtime.profile != VERIFIER_PROFILE or runtime.depth != LAYERS:
            raise VerifierContractError("target runtime 必须是 FullDepth43/native-top6")
        self.runtime = runtime
        self.tokenizer_fingerprint = runtime.tokenizer_fingerprint
        self.vocab_size = runtime.vocab_size

    @property
    def context_token_ids(self) -> tuple[int, ...]:
        return self.runtime.context_token_ids

    def begin_block(self, request: VerificationRequest) -> _SnapshotVerifierTransaction:
        _validate_block_size(len(request.draft_token_ids))
        if request.profile != VERIFIER_PROFILE or not request.causal:
            raise VerifierContractError("拒绝非 FullDepth43 causal request")
        if request.tokenizer_fingerprint != self.tokenizer_fingerprint:
            raise VerifierContractError("target tokenizer 指纹漂移")
        if request.context_token_ids != self.context_token_ids:
            raise VerifierContractError("target runtime context 与控制器漂移")
        base = self.runtime.capture_state()
        predictions: list[int] = []
        checkpoints: list[Any] = []
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        try:
            for token_offset, draft_token_id in enumerate(request.draft_token_ids):
                step = self.runtime.step(forced_next_token_id=draft_token_id)
                if not 0 <= step.predicted_token_id < self.vocab_size:
                    raise VerifierContractError("target token ID 越界")
                _validate_route_map(step.top6_by_layer, tuple(range(LAYERS)))
                predictions.append(step.predicted_token_id)
                checkpoints.append(self.runtime.capture_state())
                rows.extend(
                    {
                        "token_offset": token_offset,
                        "layer": layer,
                        "expert_ids": list(step.top6_by_layer[layer]),
                    }
                    for layer in range(LAYERS)
                )
            routes = RouteBlock.from_rows(
                block_id="serial-reference", block_size=len(predictions), rows=rows
            )
            verification = FullDepthVerification(tuple(predictions), routes)
        except Exception:
            self.runtime.restore_state(base)
            raise
        return _SnapshotVerifierTransaction(
            self.runtime,
            base,
            checkpoints,
            verification,
            time.perf_counter() - started,
        )


class NativeBatchedVerifierBackend:
    """Adapter for the next one-pass FullDepth43 causal-block implementation."""

    def __init__(self, runtime: BatchedVerifierRuntime):
        self.runtime = runtime
        self.tokenizer_fingerprint = runtime.tokenizer_fingerprint
        self.vocab_size = runtime.vocab_size

    @property
    def context_token_ids(self) -> tuple[int, ...]:
        return self.runtime.context_token_ids

    def begin_block(self, request: VerificationRequest) -> VerifierBlockTransaction:
        _validate_block_size(len(request.draft_token_ids))
        started = time.perf_counter()
        transaction = self.runtime.begin_causal_block(request)
        elapsed = time.perf_counter() - started
        metrics = transaction.metrics
        if metrics.mode != BATCHED_CAUSAL_MODE or metrics.forward_calls != 1:
            transaction.rollback()
            raise VerifierContractError(
                "生产 verifier 必须报告一次 batched_causal forward"
            )
        # Backends may measure device time themselves; wall time is only a
        # fallback when their metric is zero.
        if metrics.elapsed_seconds < 0 or elapsed < 0:
            transaction.rollback()
            raise VerifierContractError("verifier elapsed_seconds 非法")
        return transaction


class AtomicSpeculativeController:
    """Run one K=4/8 round and atomically commit both model runtimes."""

    def __init__(
        self,
        context_token_ids: Sequence[int],
        draft_backend: DraftBlockBackend,
        verifier_backend: VerifierBlockBackend,
    ) -> None:
        context = tuple(context_token_ids)
        if draft_backend.tokenizer_fingerprint != verifier_backend.tokenizer_fingerprint:
            raise VerifierContractError("草稿/target tokenizer 指纹不同")
        if draft_backend.vocab_size != verifier_backend.vocab_size:
            raise VerifierContractError("草稿/target vocab 不同")
        if context != draft_backend.context_token_ids or context != verifier_backend.context_token_ids:
            raise VerifierContractError("初始 context 未在两个 runtime 对齐")
        self.context_token_ids = list(context)
        self.draft_backend = draft_backend
        self.verifier_backend = verifier_backend
        self.tokenizer_fingerprint = draft_backend.tokenizer_fingerprint

    def run_round(self, block_size: int) -> SpeculativeRoundResult:
        _validate_block_size(block_size)
        before = tuple(self.context_token_ids)
        draft_tx: DraftBlockTransaction | None = None
        verifier_tx: VerifierBlockTransaction | None = None
        try:
            draft_tx = self.draft_backend.begin_block(before, block_size)
            if len(draft_tx.steps) != block_size:
                raise VerifierContractError("草稿后端未返回完整 block")
            for step in draft_tx.steps:
                step.validate(self.draft_backend.vocab_size)
            draft_token_ids = tuple(step.token_id for step in draft_tx.steps)
            request = VerificationRequest(
                context_token_ids=before,
                draft_token_ids=draft_token_ids,
                tokenizer_fingerprint=self.tokenizer_fingerprint,
            )
            verifier_tx = self.verifier_backend.begin_block(request)
            response = verifier_tx.verification
            if (
                response.profile != VERIFIER_PROFILE
                or not response.causal
                or response.depth != LAYERS
                or response.top_k != TOP_K
            ):
                raise VerifierContractError("拒绝非 FullDepth43/native-top6 causal 响应")
            if response.native_routes.block_size != block_size:
                raise VerifierContractError("native route block 与草稿块不对齐")
            decision = decide_acceptance(draft_token_ids, response.predicted_token_ids)

            # Two-phase commit: validate/select both snapshots first, then swap
            # them.  Either commit may still fail, so both retain rollback data.
            verifier_tx.prepare_commit(decision)
            draft_tx.prepare_commit(decision)
            verifier_tx.commit()
            draft_tx.commit()
            expected = before + decision.committed_token_ids
            if self.verifier_backend.context_token_ids != expected:
                raise VerifierContractError("target 提交后 context 不等于接受决策")
            if self.draft_backend.context_token_ids != expected:
                raise VerifierContractError("草稿提交后 context 不等于接受决策")
            self.context_token_ids.extend(decision.committed_token_ids)
            return SpeculativeRoundResult(
                decision=decision,
                native_routes=response.native_routes,
                block_size=block_size,
                draft_metrics=draft_tx.metrics,
                verifier_metrics=verifier_tx.metrics,
            )
        except Exception:
            # Rollback is idempotent for the supplied transactions.  Preserve
            # the original controller context even if a backend commit failed.
            rollback_errors: list[Exception] = []
            for transaction in (verifier_tx, draft_tx):
                if transaction is None:
                    continue
                try:
                    transaction.rollback()
                except Exception as rollback_error:  # pragma: no cover - fatal backend bug
                    rollback_errors.append(rollback_error)
            self.context_token_ids[:] = before
            if rollback_errors:
                raise VerifierContractError(
                    "事务失败且 rollback 失败: "
                    + "; ".join(str(error) for error in rollback_errors)
                )
            raise
