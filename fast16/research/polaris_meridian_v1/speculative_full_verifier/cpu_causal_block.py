"""FullDepth43 K=1/4/8 CPU causal-block 正确性参考边界。

这个模块把 K 个严格自回归 token 收口到一次
``begin_causal_block`` API。CPU reference 内部仍逐 token 调用冻结
FullDepth runtime，因此 ``forward_calls=K`` 且不具备加速资格。

每个 token 结束后保留一份完整 runtime checkpoint（包括所有层的
window KV 与 compressor remainder）。在第 j 个草稿 token 失配时，
事务只选中 checkpoint[j]，并把它的 pending token 替换为 target
fallback；checkpoint[j+1:] 不会进入已提交状态。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from .assets import LAYERS, TOP_K
from .cache_replay import ReplayContractError, RouteBlock
from .runtime_controller import (
    AcceptanceDecision,
    NativeTokenStep,
    SnapshotTokenRuntime,
    TransactionMetrics,
    decide_acceptance,
)
from .verifier import (
    VERIFIER_PROFILE,
    FullDepthVerification,
    VerificationRequest,
    VerifierContractError,
)


CPU_CAUSAL_BLOCK_REFERENCE_MODE = "cpu_causal_block_reference"
REFERENCE_BLOCK_SIZES = (1, 4, 8)


def _validate_reference_request(
    request: VerificationRequest,
    *,
    context_token_ids: tuple[int, ...],
    tokenizer_fingerprint: str,
) -> int:
    block_size = len(request.draft_token_ids)
    if block_size not in REFERENCE_BLOCK_SIZES:
        raise VerifierContractError(
            f"CPU causal-block reference 只允许 K={REFERENCE_BLOCK_SIZES}"
        )
    if request.profile != VERIFIER_PROFILE or not request.causal:
        raise VerifierContractError("拒绝非 FullDepth43 causal request")
    if request.depth != LAYERS or request.top_k != TOP_K:
        raise VerifierContractError("拒绝非 43 层原生 top-6 request")
    if request.tokenizer_fingerprint != tokenizer_fingerprint:
        raise VerifierContractError("target tokenizer 指纹漂移")
    if request.context_token_ids != context_token_ids:
        raise VerifierContractError("target runtime context 与 causal block 请求漂移")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in request.draft_token_ids
    ):
        raise VerifierContractError("草稿 token ID 必须是 int")
    return block_size


def _route_block(steps: Sequence[NativeTokenStep]) -> RouteBlock:
    rows = [
        {
            "token_offset": token_offset,
            "layer": layer,
            "expert_ids": list(step.top6_by_layer[layer]),
        }
        for token_offset, step in enumerate(steps)
        for layer in range(LAYERS)
        if layer in step.top6_by_layer
    ]
    try:
        return RouteBlock.from_rows(
            "cpu-causal-block-reference",
            len(steps),
            rows,
        )
    except (KeyError, ReplayContractError) as error:
        raise VerifierContractError(
            "CPU causal block 必须保存 K×43×6 原生 route"
        ) from error


@dataclass(frozen=True)
class ReferenceBlockAudit:
    """参考 block 的可验证审计信息。

    ``checkpoint_count`` 与 ``route_rows`` 使调用方不用打开模型对象，
    也能拒绝“只返回 logits、没有保存状态/路由”的假 block。
    """

    block_api_calls: int
    model_forward_calls: int
    checkpoint_count: int
    route_rows: int
    state_isolation: str = "private checkpoints; commit selects one prefix"


class CpuCausalBlockTransaction:
    """可回滚的 CPU causal-block 事务。"""

    def __init__(
        self,
        backend: "CpuCausalBlockReferenceBackend",
        owner_token: object,
        base_generation: int,
        request: VerificationRequest,
        base_state: Any,
        checkpoints: Sequence[Any],
        steps: Sequence[NativeTokenStep],
        elapsed_seconds: float,
    ) -> None:
        self._backend = backend
        self._runtime = backend.runtime
        self._owner_token = owner_token
        self._base_generation = base_generation
        self._request = request
        self._base_state = base_state
        self._base_context = request.context_token_ids
        self._checkpoints = tuple(checkpoints)
        self._prepared_state: Any | None = None
        self._prepared_context: tuple[int, ...] | None = None
        self._phase = "open"
        self._committed_generation: int | None = None

        routes = _route_block(steps)
        predictions = tuple(step.predicted_token_id for step in steps)
        self.verification = FullDepthVerification(predictions, routes)
        self.metrics = TransactionMetrics(
            mode=CPU_CAUSAL_BLOCK_REFERENCE_MODE,
            forward_calls=len(steps),
            elapsed_seconds=elapsed_seconds,
        )
        self.audit = ReferenceBlockAudit(
            block_api_calls=1,
            model_forward_calls=len(steps),
            checkpoint_count=len(self._checkpoints),
            route_rows=len(steps) * LAYERS,
        )


    def _assert_open_base(self) -> None:
        if self._phase not in {"open", "prepared"}:
            raise VerifierContractError("事务已结束或已提交")
        try:
            self._backend._assert_owner(
                self._owner_token,
                self._base_generation,
                self._base_context,
            )
        except Exception:
            self._backend._invalidate_owner(
                self._owner_token,
                self._base_generation,
            )
            self._phase = "stale"
            raise

    def prepare_commit(self, decision: AcceptanceDecision) -> None:
        self._assert_open_base()
        expected = decide_acceptance(
            self._request.draft_token_ids,
            self.verification.predicted_token_ids,
        )
        if decision != expected:
            raise VerifierContractError("提交决策与 target causal block 不一致")
        if len(self._checkpoints) != len(self._request.draft_token_ids):
            raise VerifierContractError("checkpoint 数量与 causal block 不一致")

        if decision.mismatch_index is None:
            selected = self._checkpoints[-1]
        else:
            if decision.fallback_token_id is None:
                raise VerifierContractError("mismatch 缺少 target fallback")
            # checkpoint[j] 恰好包含到失配位置的 KV/compressor
            # append；更晚的 checkpoint 不被引用，因此被原子裁掉。
            selected = self._runtime.replace_pending_token(
                self._checkpoints[decision.mismatch_index],
                decision.fallback_token_id,
            )
        self._prepared_state = selected
        self._prepared_context = (
            self._request.context_token_ids + decision.committed_token_ids
        )
        # replace_pending_token 也必须是 COW；若适配器错误地修改了
        # committed runtime，在 prepare 阶段就 fail closed。
        self._assert_open_base()
        self._phase = "prepared"

    def commit(self) -> None:
        self._assert_open_base()
        if self._prepared_state is None or self._prepared_context is None:
            raise VerifierContractError("CPU causal block 事务未 prepare")
        try:
            self._runtime.restore_state(self._prepared_state)
            if self._runtime.context_token_ids != self._prepared_context:
                raise VerifierContractError(
                    "裁切后 runtime context 与接受前缀/fallback 不一致"
                )
        except Exception:
            self._runtime.restore_state(self._base_state)
            self._backend._finish_owner(
                self._owner_token,
                self._base_generation,
            )
            self._phase = "rolled_back"
            raise
        self._committed_generation = self._backend._finish_owner(
            self._owner_token,
            self._base_generation,
        )
        self._phase = "committed"

    def rollback(self) -> None:
        if self._phase in {"rolled_back", "stale"}:
            return
        if self._phase == "committed":
            assert self._prepared_context is not None
            assert self._committed_generation is not None
            self._backend._assert_compensatable(
                self._committed_generation,
                self._prepared_context,
            )
            self._runtime.restore_state(self._base_state)
            self._backend._finish_compensation(self._committed_generation)
            self._phase = "rolled_back"
            return
        self._assert_open_base()
        self._runtime.restore_state(self._base_state)
        self._backend._finish_owner(
            self._owner_token,
            self._base_generation,
        )
        self._phase = "rolled_back"


class CpuCausalBlockReferenceBackend:
    """FullDepth43 CPU 参考后端；可直接接入原子控制器。

    ``begin_causal_block`` 是唯一 block API。它内部只为正确性逐
    token 运行，并在返回事务前恢复 base snapshot，避免未经
    two-phase commit 的 KV/compressor 泄漏。
    """

    def __init__(self, runtime: SnapshotTokenRuntime) -> None:
        if runtime.profile != VERIFIER_PROFILE or runtime.depth != LAYERS:
            raise VerifierContractError(
                "CPU causal block runtime 必须是 FullDepth43/native-top6"
            )
        self.runtime = runtime
        self.tokenizer_fingerprint = runtime.tokenizer_fingerprint
        self.vocab_size = runtime.vocab_size
        self.block_api_calls = 0
        self._generation = 0
        self._active_owner: object | None = None

    def _assert_owner(
        self,
        owner_token: object,
        generation: int,
        base_context: tuple[int, ...],
    ) -> None:
        if self._active_owner is not owner_token or self._generation != generation:
            raise VerifierContractError("过期 causal-block 事务/代际漂移")
        if self.context_token_ids != base_context:
            raise VerifierContractError(
                "causal-block 事务期间 target runtime 已被外部推进"
            )

    def _invalidate_owner(self, owner_token: object, generation: int) -> None:
        if self._active_owner is owner_token and self._generation == generation:
            self._active_owner = None
            self._generation += 1

    def _finish_owner(self, owner_token: object, generation: int) -> int:
        if self._active_owner is not owner_token or self._generation != generation:
            raise VerifierContractError("事务 owner/代际漂移")
        self._active_owner = None
        self._generation += 1
        return self._generation

    def _assert_compensatable(
        self,
        committed_generation: int,
        committed_context: tuple[int, ...],
    ) -> None:
        if self._active_owner is not None or self._generation != committed_generation:
            raise VerifierContractError("过期 causal-block rollback 被拒绝")
        if self.context_token_ids != committed_context:
            raise VerifierContractError(
                "causal-block commit 后 target runtime 已被外部推进"
            )

    def _finish_compensation(self, committed_generation: int) -> None:
        if self._active_owner is not None or self._generation != committed_generation:
            raise VerifierContractError("无法完成过期补偿回滚")
        self._generation += 1

    @property
    def context_token_ids(self) -> tuple[int, ...]:
        return self.runtime.context_token_ids

    def begin_block(self, request: VerificationRequest) -> CpuCausalBlockTransaction:
        return self.begin_causal_block(request)

    def begin_causal_block(
        self, request: VerificationRequest
    ) -> CpuCausalBlockTransaction:
        if self._active_owner is not None:
            raise VerifierContractError("已有未结束 causal-block 事务")
        block_size = _validate_reference_request(
            request,
            context_token_ids=self.context_token_ids,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
        )
        if any(not 0 <= token_id < self.vocab_size for token_id in request.draft_token_ids):
            raise VerifierContractError("草稿 token ID 越出 target vocab")

        owner_token = object()
        base_generation = self._generation
        self._active_owner = owner_token
        try:
            base = self.runtime.capture_state()
        except Exception:
            self._invalidate_owner(owner_token, base_generation)
            raise
        steps: list[NativeTokenStep] = []
        checkpoints: list[Any] = []
        started = time.perf_counter()
        self.block_api_calls += 1
        try:
            for token_offset, draft_token_id in enumerate(request.draft_token_ids):
                step = self.runtime.step(forced_next_token_id=draft_token_id)
                if (
                    isinstance(step.predicted_token_id, bool)
                    or not isinstance(step.predicted_token_id, int)
                    or not 0 <= step.predicted_token_id < self.vocab_size
                ):
                    raise VerifierContractError("target 预测 token ID 越界")
                # 每个 step 都必须已经提交草稿 pending token；否则下一
                # 位置不是严格自回归前缀。
                expected_context = (
                    request.context_token_ids
                    + request.draft_token_ids[: token_offset + 1]
                )
                if self.runtime.context_token_ids != expected_context:
                    raise VerifierContractError(
                        f"token_offset={token_offset} 未保持严格自回归 context"
                    )
                steps.append(step)
                checkpoints.append(self.runtime.capture_state())
            if len(steps) != block_size:
                raise VerifierContractError("CPU causal block 未返回完整 K 个位置")
            transaction = CpuCausalBlockTransaction(
                self,
                owner_token,
                base_generation,
                request,
                base,
                checkpoints,
                steps,
                time.perf_counter() - started,
            )
        except Exception:
            self.runtime.restore_state(base)
            self._invalidate_owner(owner_token, base_generation)
            raise

        # begin 只产生私有 append delta，未经 prepare/commit 绝不可
        # 污染 target 的 committed state。
        try:
            self.runtime.restore_state(base)
            if self.runtime.context_token_ids != request.context_token_ids:
                raise VerifierContractError("block 返回前未恢复 base context")
        except Exception:
            self.runtime.restore_state(base)
            self._invalidate_owner(owner_token, base_generation)
            raise
        return transaction
