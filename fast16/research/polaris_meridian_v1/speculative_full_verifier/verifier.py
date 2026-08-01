"""Fail-closed speculative-decoding state machine.

The target contract is greedy token equality.  It is intentionally narrower
than stochastic speculative sampling: no draft token is committed unless the
native full-depth target predicted the same DeepSeek token ID at that position.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .assets import EXPERTS_PER_LAYER, LAYERS, S14_LAYERS, TOP_K
from .cache_replay import RouteBlock


DRAFT_PROFILE = "S14/top6"
VERIFIER_PROFILE = "FullDepth43/native-top6"


class VerifierContractError(ValueError):
    """Raised on an unsafe transition or malformed backend response."""


class TokenizerContract(Protocol):
    vocab_size: int
    fingerprint: str

    def encode(self, text: str) -> tuple[int, ...]: ...

    def validate_token_ids(self, token_ids: Sequence[int]) -> tuple[int, ...]: ...


class DeepSeekTokenizer:
    """Local wrapper around the fixed DeepSeek tokenizer.json.

    Loading this class reads tokenizer metadata only.  It never imports model
    code or touches safetensors payloads.
    """

    def __init__(self, asset_root: str | Path):
        root = Path(asset_root)
        tokenizer_path = root / "tokenizer.json"
        config_path = root / "config.json"
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise VerifierContractError("DeepSeek tokenizer 需要 Python tokenizers 包") from exc
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            # Tokenizer.from_file raises bindings-specific exception classes.
            raise VerifierContractError(f"无法加载 DeepSeek tokenizer: {exc}") from exc
        expected_vocab = config.get("vocab_size")
        actual_vocab = self._tokenizer.get_vocab_size(with_added_tokens=True)
        if not isinstance(expected_vocab, int) or actual_vocab != expected_vocab:
            raise VerifierContractError(f"tokenizer vocab {actual_vocab} 与 config {expected_vocab} 不符")
        self.vocab_size = actual_vocab
        self.fingerprint = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        self.path = str(tokenizer_path.resolve())

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text 必须是 str")
        return tuple(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def encode_exact_draft(self, generated_text: str, block_size: int) -> tuple[int, ...]:
        """Tokenize an S14-produced continuation and take exactly N tokens."""

        if block_size <= 0:
            raise VerifierContractError("block_size 必须为正整数")
        token_ids = self.encode(generated_text)
        if len(token_ids) < block_size:
            raise VerifierContractError(f"候选文本只有 {len(token_ids)} 个 DeepSeek token，不足 {block_size}")
        return token_ids[:block_size]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(self.validate_token_ids(token_ids)), skip_special_tokens=False)

    def validate_token_ids(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        result = tuple(token_ids)
        if any(not isinstance(token, int) or isinstance(token, bool) for token in result):
            raise VerifierContractError("token ID 必须是 int")
        if any(token < 0 or token >= self.vocab_size for token in result):
            raise VerifierContractError("token ID 越出 DeepSeek vocab")
        return result


@dataclass(frozen=True)
class DraftStep:
    token_id: int
    top6_by_s14_layer: Mapping[int, tuple[int, ...]]

    def validate(self, vocab_size: int) -> None:
        if not isinstance(self.token_id, int) or isinstance(self.token_id, bool) or not 0 <= self.token_id < vocab_size:
            raise VerifierContractError(f"草稿 token ID 非法: {self.token_id}")
        if set(self.top6_by_s14_layer) != set(S14_LAYERS):
            raise VerifierContractError("草稿必须带齐冻结 S14 的 14 层 route")
        for layer in S14_LAYERS:
            experts = tuple(self.top6_by_s14_layer[layer])
            if len(experts) != TOP_K or len(set(experts)) != TOP_K:
                raise VerifierContractError(f"S14 L{layer} 必须有 6 个不同专家")
            if any(not isinstance(expert, int) or isinstance(expert, bool) or not 0 <= expert < EXPERTS_PER_LAYER for expert in experts):
                raise VerifierContractError(f"S14 L{layer} 专家 ID 越界")


class S14Top6Backend(Protocol):
    def generate(self, context_token_ids: Sequence[int], block_size: int) -> Sequence[DraftStep]: ...


@dataclass(frozen=True)
class VerificationRequest:
    context_token_ids: tuple[int, ...]
    draft_token_ids: tuple[int, ...]
    tokenizer_fingerprint: str
    profile: str = VERIFIER_PROFILE
    causal: bool = True
    depth: int = LAYERS
    top_k: int = TOP_K


@dataclass(frozen=True)
class FullDepthVerification:
    """Target predictions aligned one-to-one with draft positions."""

    predicted_token_ids: tuple[int, ...]
    native_routes: RouteBlock
    profile: str = VERIFIER_PROFILE
    causal: bool = True
    depth: int = LAYERS
    top_k: int = TOP_K


class FullDepthBackend(Protocol):
    def verify_causal_block(self, request: VerificationRequest) -> FullDepthVerification: ...


@dataclass(frozen=True)
class VerificationResult:
    accepted_prefix: tuple[int, ...]
    fallback_token_id: int | None
    rejected_draft_suffix: tuple[int, ...]
    committed_token_ids: tuple[int, ...]
    mismatch_index: int | None
    native_routes: RouteBlock
    verifier_calls: int = 1

    @property
    def accepted_length(self) -> int:
        return len(self.accepted_prefix)


class SessionState(str, Enum):
    READY = "ready"
    DRAFTING = "drafting"
    AWAITING_VERIFICATION = "awaiting_verification"
    VERIFYING = "verifying"


class SpeculativeSession:
    """One-context state machine; every target round is exactly one block call."""

    def __init__(self, tokenizer: TokenizerContract, context_token_ids: Sequence[int]):
        self.tokenizer = tokenizer
        self.context_token_ids = list(tokenizer.validate_token_ids(context_token_ids))
        self.state = SessionState.READY
        self._block_size: int | None = None
        self._draft_steps: tuple[DraftStep, ...] = ()

    @classmethod
    def from_prompt(cls, tokenizer: TokenizerContract, prompt: str) -> "SpeculativeSession":
        return cls(tokenizer, tokenizer.encode(prompt))

    def start_round(self, block_size: int) -> tuple[int, ...]:
        if self.state is not SessionState.READY:
            raise VerifierContractError(f"只能从 ready 开始，当前 {self.state.value}")
        if block_size <= 0:
            raise VerifierContractError("block_size 必须为正整数")
        self._block_size = block_size
        self._draft_steps = ()
        self.state = SessionState.DRAFTING
        return tuple(self.context_token_ids)

    def submit_draft(self, steps: Sequence[DraftStep]) -> None:
        if self.state is not SessionState.DRAFTING:
            raise VerifierContractError("当前不接受草稿")
        draft = tuple(steps)
        if len(draft) != self._block_size:
            raise VerifierContractError(f"草稿必须恰好有 {self._block_size} 个 token")
        for step in draft:
            step.validate(self.tokenizer.vocab_size)
        self._draft_steps = draft
        self.state = SessionState.AWAITING_VERIFICATION

    def make_verification_request(self) -> VerificationRequest:
        if self.state is not SessionState.AWAITING_VERIFICATION:
            raise VerifierContractError("当前不能发起验证")
        self.state = SessionState.VERIFYING
        return VerificationRequest(
            context_token_ids=tuple(self.context_token_ids),
            draft_token_ids=tuple(step.token_id for step in self._draft_steps),
            tokenizer_fingerprint=self.tokenizer.fingerprint,
        )

    def finish_verification(self, response: FullDepthVerification) -> VerificationResult:
        if self.state is not SessionState.VERIFYING:
            raise VerifierContractError("当前没有待完成的验证")
        if (
            response.profile != VERIFIER_PROFILE
            or not response.causal
            or response.depth != LAYERS
            or response.top_k != TOP_K
        ):
            raise VerifierContractError("拒绝非 FullDepth43/native-top6 causal 响应")
        predictions = self.tokenizer.validate_token_ids(response.predicted_token_ids)
        draft = tuple(step.token_id for step in self._draft_steps)
        if len(predictions) != len(draft):
            raise VerifierContractError("验证预测必须与草稿位置一一对齐")
        if response.native_routes.block_size != len(draft):
            raise VerifierContractError("native top-6 route block 必须与草稿位置一一对齐")

        mismatch = next((index for index, pair in enumerate(zip(draft, predictions)) if pair[0] != pair[1]), None)
        if mismatch is None:
            accepted = draft
            fallback = None
            rejected: tuple[int, ...] = ()
            committed = draft
        else:
            accepted = draft[:mismatch]
            fallback = predictions[mismatch]
            rejected = draft[mismatch:]
            committed = accepted + (fallback,)

        self.context_token_ids.extend(committed)
        self._block_size = None
        self._draft_steps = ()
        self.state = SessionState.READY
        return VerificationResult(
            accepted_prefix=accepted,
            fallback_token_id=fallback,
            rejected_draft_suffix=rejected,
            committed_token_ids=committed,
            mismatch_index=mismatch,
            native_routes=response.native_routes,
        )

    def abort_round(self) -> None:
        """Drop all uncommitted work; context is left unchanged."""

        if self.state is SessionState.READY:
            return
        self._block_size = None
        self._draft_steps = ()
        self.state = SessionState.READY

    def run_round(
        self,
        draft_backend: S14Top6Backend,
        verifier_backend: FullDepthBackend,
        block_size: int,
    ) -> VerificationResult:
        """Run one draft and exactly one full-depth causal-block verification."""

        context = self.start_round(block_size)
        try:
            self.submit_draft(draft_backend.generate(context, block_size))
            request = self.make_verification_request()
            # Deliberately no retry/fallback verifier call: one causal block only.
            response = verifier_backend.verify_causal_block(request)
            return self.finish_verification(response)
        except Exception:
            self.abort_round()
            raise
