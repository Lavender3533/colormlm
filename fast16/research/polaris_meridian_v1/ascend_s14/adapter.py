"""DeepSeek-V4 S14 的逐层 Ascend adapter 接口。

默认命令只做 dry-run 或纯 Python 合成生命周期自检；本文件不下载权重，
也不把合成执行伪装成 DeepSeek 原生前向。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol


REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
S14_LAYERS = (0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42)
REQUIRED_NATIVE_CAPABILITIES = (
    "official_fixed_revision_python_graph",
    "mxfp4_expert_unpack_or_native_kernel",
    "ue8m0_scale_semantics",
    "fp8_attention_kernels",
    "mhc_four_stream_state_transition",
    "csa_hca_state_and_cache_semantics",
    "native_router_top6_and_shared_expert",
    "native_tokenizer_embedding_norm_lm_head",
    "verified_s14_range_provider",
    "ascend_numerical_parity_test",
)


class ContractError(RuntimeError):
    """冻结身份、层集合或生命周期被破坏。"""


class NativeForwardUnavailable(RuntimeError):
    """环境尚不具备原生 DeepSeek block 语义。"""


def configure_utf8() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} 必须是 JSON object")
    return value


def validate_matrix(matrix: Mapping[str, Any]) -> None:
    source = matrix.get("source", {})
    candidate = matrix.get("candidate", {})
    if source.get("repo") != REPO or source.get("revision") != REVISION:
        raise ContractError("拒绝非冻结 DeepSeek repo/revision")
    if tuple(candidate.get("layers", ())) != S14_LAYERS:
        raise ContractError("拒绝发生漂移的 S14 层集合")
    if not isinstance(matrix.get("native_forward_ready"), bool):
        raise ContractError("native_forward_ready 必须是显式 boolean")


@dataclass
class NativeState:
    """跨层状态容器；真实 payload 类型由官方 Ascend executor 决定。"""

    hidden: Any
    mhc_streams: Any
    csa_hca_state: Any
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerLease:
    """当前层张量租约。关闭后禁止 executor 再引用 payload。"""

    layer_id: int
    tensors: dict[str, Any]
    integrity: dict[str, Any]
    released: bool = False


class VerifiedTensorProvider(Protocol):
    """Range pack 未来接入点；每个 yield 必须已经完成来源与 Range 哈希校验。"""

    def iter_verified_tensors(
        self, layer_id: int
    ) -> Iterable[tuple[str, Any, Mapping[str, Any]]]: ...

    def release_tensor(self, name: str, payload: Any) -> None: ...


class LayerSource(Protocol):
    def load_layer(self, layer_id: int) -> LayerLease: ...

    def free_layer(self, lease: LayerLease) -> None: ...


class LayerExecutor(Protocol):
    native_semantics: bool

    def execute_layer(self, state: NativeState, lease: LayerLease) -> NativeState: ...


class RangePackLayerSource:
    """把外部已校验 tensor iterator 收束为“单层在生”的租约。

    本类不读取网络或磁盘；provider 由主线在真实容器中注入。
    """

    def __init__(self, provider: VerifiedTensorProvider):
        self.provider = provider
        self._live: LayerLease | None = None

    def load_layer(self, layer_id: int) -> LayerLease:
        if layer_id not in S14_LAYERS:
            raise ContractError(f"L{layer_id} 不属于冻结 S14")
        if self._live is not None:
            raise ContractError(f"L{self._live.layer_id} 尚未释放，拒绝加载 L{layer_id}")
        tensors: dict[str, Any] = {}
        integrity: dict[str, Any] = {}
        try:
            for name, payload, proof in self.provider.iter_verified_tensors(layer_id):
                if not bool(proof.get("verified")):
                    self.provider.release_tensor(name, payload)
                    raise ContractError(f"L{layer_id} tensor {name} 未通过完整性校验")
                if name in tensors:
                    self.provider.release_tensor(name, payload)
                    raise ContractError(f"L{layer_id} tensor 重复: {name}")
                tensors[name] = payload
                integrity[name] = dict(proof)
        except Exception:
            for loaded_name, loaded_payload in tuple(tensors.items()):
                self.provider.release_tensor(loaded_name, loaded_payload)
            raise
        if not tensors:
            raise ContractError(f"L{layer_id} 没有已校验 tensor")
        self._live = LayerLease(layer_id=layer_id, tensors=tensors, integrity=integrity)
        return self._live

    def free_layer(self, lease: LayerLease) -> None:
        if self._live is not lease or lease.released:
            raise ContractError("释放的不是当前在生层租约")
        for name, payload in tuple(lease.tensors.items()):
            self.provider.release_tensor(name, payload)
        lease.tensors.clear()
        lease.integrity.clear()
        lease.released = True
        self._live = None


class OfficialAscendLayerExecutor:
    """真实 executor 的硬门；只有注入已验证官方 block 才可执行。"""

    native_semantics = True

    def __init__(
        self,
        support_matrix: Mapping[str, Any],
        official_block: Callable[[int, NativeState, Mapping[str, Any]], NativeState] | None = None,
    ):
        validate_matrix(support_matrix)
        statuses = support_matrix.get("required_for_native_forward", {})
        missing = [name for name in REQUIRED_NATIVE_CAPABILITIES if statuses.get(name) != "passed"]
        if support_matrix.get("native_forward_ready") is not True:
            missing.append("native_forward_ready_manifest")
        if missing or official_block is None:
            detail = ", ".join(missing or ["official_block_callable"])
            raise NativeForwardUnavailable(f"Ascend 原生 forward 尚不可用: {detail}")
        self._official_block = official_block

    def execute_layer(self, state: NativeState, lease: LayerLease) -> NativeState:
        if lease.released:
            raise ContractError("禁止执行已释放层")
        result = self._official_block(lease.layer_id, state, lease.tensors)
        if not isinstance(result, NativeState):
            raise ContractError("official block 必须返回 NativeState")
        return result


class StreamingS14Runner:
    """严格执行 load → execute → free，跳过层保持 identity。"""

    def __init__(self, source: LayerSource, executor: LayerExecutor):
        self.source = source
        self.executor = executor
        self.events: list[dict[str, Any]] = []
        self.max_live_layers = 0
        self._live_layers = 0

    def run(self, state: NativeState) -> NativeState:
        for layer_id in S14_LAYERS:
            lease = self.source.load_layer(layer_id)
            self._live_layers += 1
            self.max_live_layers = max(self.max_live_layers, self._live_layers)
            self.events.append({"layer": layer_id, "event": "load"})
            try:
                state = self.executor.execute_layer(state, lease)
                self.events.append({"layer": layer_id, "event": "execute"})
            finally:
                self.source.free_layer(lease)
                self._live_layers -= 1
                self.events.append({"layer": layer_id, "event": "free"})
        if self._live_layers != 0:
            raise ContractError("执行结束仍有层租约存活")
        return state


class SyntheticProvider:
    """仅用于生命周期测试，payload 不是模型权重。"""

    def __init__(self):
        self.live_payloads = 0
        self.max_live_payloads = 0

    def iter_verified_tensors(
        self, layer_id: int
    ) -> Iterable[tuple[str, Any, Mapping[str, Any]]]:
        payload = {"synthetic_layer": layer_id}
        self.live_payloads += 1
        self.max_live_payloads = max(self.max_live_payloads, self.live_payloads)
        yield "synthetic.lifecycle.marker", payload, {"verified": True, "synthetic": True}

    def release_tensor(self, name: str, payload: Any) -> None:
        del name, payload
        self.live_payloads -= 1


class SyntheticExecutor:
    """纯 Python 状态变换；native_semantics=false。"""

    native_semantics = False

    def execute_layer(self, state: NativeState, lease: LayerLease) -> NativeState:
        if lease.released or "synthetic.lifecycle.marker" not in lease.tensors:
            raise ContractError("合成租约无效")
        history = list(state.metadata.get("synthetic_layers", []))
        history.append(lease.layer_id)
        return NativeState(
            hidden=float(state.hidden) + (lease.layer_id + 1) / 1000.0,
            mhc_streams=state.mhc_streams,
            csa_hca_state=state.csa_hca_state,
            position=state.position,
            metadata={"synthetic_layers": history, "native_forward": False},
        )


def make_dry_run(matrix: Mapping[str, Any]) -> dict[str, Any]:
    validate_matrix(matrix)
    return {
        "format": "polaris-ascend-s14-dry-run-v1",
        "source": {"repo": REPO, "revision": REVISION},
        "layers": list(S14_LAYERS),
        "lifecycle": ["load verified current-layer tensors", "execute one native block", "free current layer"],
        "max_live_layers": 1,
        "weight_io_performed": False,
        "device_touched": False,
        "native_forward_executed": False,
        "native_forward_ready": False,
        "missing_capabilities": [
            name
            for name in REQUIRED_NATIVE_CAPABILITIES
            if matrix.get("required_for_native_forward", {}).get(name) != "passed"
        ],
        "claim_limit": "这是接口 dry-run，不是 DeepSeek forward 或质量证据。",
    }


def run_synthetic() -> dict[str, Any]:
    provider = SyntheticProvider()
    runner = StreamingS14Runner(RangePackLayerSource(provider), SyntheticExecutor())
    initial = NativeState(0.0, "synthetic-mhc", "synthetic-csa-hca", 0)
    result = runner.run(initial)
    return {
        "format": "polaris-ascend-s14-synthetic-v1",
        "ok": result.metadata.get("synthetic_layers") == list(S14_LAYERS),
        "layers": result.metadata.get("synthetic_layers"),
        "events": runner.events,
        "max_live_layers": runner.max_live_layers,
        "max_live_payloads": provider.max_live_payloads,
        "remaining_live_payloads": provider.live_payloads,
        "native_forward_executed": False,
        "claim_limit": "合成数值只验证生命周期，不能解释为模型输出。",
    }


def write_report(report: Mapping[str, Any], output: Path | None) -> None:
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")


def main() -> int:
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "synthetic", "native"))
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).with_name("support_matrix.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    matrix = read_json(args.matrix)
    if args.mode == "dry-run":
        report = make_dry_run(matrix)
    elif args.mode == "synthetic":
        validate_matrix(matrix)
        report = run_synthetic()
    else:
        try:
            OfficialAscendLayerExecutor(matrix)
        except NativeForwardUnavailable as exc:
            write_report(
                {
                    "format": "polaris-ascend-s14-native-refusal-v1",
                    "ok": False,
                    "native_forward_ready": False,
                    "error": str(exc),
                },
                args.output,
            )
            return 2
        raise AssertionError("当前矩阵不得进入 native forward")
    write_report(report, args.output)
    return 0 if bool(report.get("ok", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
