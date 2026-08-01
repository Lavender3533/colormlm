"""FullDepth43/native-top6 最小真实 CPU/PyTorch reference executor。

执行入口严格跑完 0..42 层、当前 token/当前层原生 top-6、
shared expert、mHC、window KV 与 compressor remainder，最后才允许原生
HC/norm/BF16 head argmax 产生 token。静态页缺失或预算不足时 fail closed。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from fast16.research.polaris_meridian_v1.s14_first_real_token import executor as s14
from fast16.research.polaris_meridian_v1.s14_range_pack import online_range

from .catalog import build_catalog, read_json, validate_catalog, write_json
from .preflight import DEFAULT_ASSET_ROOT, DEFAULT_CATALOG, run_preflight
from .profile import FULLDEPTH43_NATIVE_TOP6, ExecutionProfile


REPORT_FORMAT = "polaris-fulldepth43-native-top6-reference-v1"
DEFAULT_REPORT = Path(__file__).resolve().parent / "last_run_report.json"
DEFAULT_FORCED_PREFILL = Path(__file__).resolve().parent / "first_preview_forced_prefill.json"


class FullDepthError(RuntimeError):
    pass


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
        if self.range_workers == 1 or not self.cache.allow_fetch or len(frozen) <= 1:
            return tuple(self._fetch_one(entry) for entry in frozen)
        with ThreadPoolExecutor(max_workers=self.range_workers, thread_name_prefix="fd43-range") as pool:
            return tuple(pool.map(self._fetch_one, frozen))

    def prepare_embedding_row(self, token_id: int) -> online_range.CachedRange:
        if self.phase is not SessionPhase.INIT:
            raise FullDepthError("embedding row 只允许在 init 获取")
        result = self._fetch_one(online_range.embedding_row_entry(self.catalog, token_id))
        self.phase = SessionPhase.AWAITING_LAYER
        return result

    def prepare_layer(self, layer: int, token_id: int) -> LayerPrerequisites:
        if self.phase is not SessionPhase.AWAITING_LAYER or layer != self.current_layer:
            raise FullDepthError(f"层顺序错误: expected={self.current_layer}, got={layer}")
        row = self.catalog["layers"][str(layer)]
        result = LayerPrerequisites(
            layer=layer,
            token_id=token_id,
            non_expert=self._fetch_all(row["non_expert"]),
            router=self._fetch_all(row["router"]),
        )
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


class FullDepthNativeLayerReference(s14.NativeLayerReference):
    def __init__(
        self,
        layer: int,
        store: s14.TensorStore,
        *,
        profile: ExecutionProfile = FULLDEPTH43_NATIVE_TOP6,
    ) -> None:
        profile.validate()
        super().__init__(
            layer,
            store,
            profile_layers=profile.layers,
            compress_ratios={value: profile.ratio_for(value) for value in profile.layers},
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
    forced_prefill_path: Path | None = None

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


def execute(
    config: ExecutionConfig,
    *,
    profile: ExecutionProfile = FULLDEPTH43_NATIVE_TOP6,
) -> dict[str, Any]:
    """显式执行真实 FullDepth token；任何缺口都不提交 token。"""

    config.validate()
    profile.validate()
    catalog = _load_or_build_catalog(config)
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
        "forced_prefill_path": (
            None
            if config.forced_prefill_path is None
            else str(config.forced_prefill_path.resolve())
        ),
        "forced_prefill": None,
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
    )
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
    if forced_queue is not None:
        report["forced_prefill"] = {
            "artifact_sha256": forced_queue.artifact_sha256,
            "token_count": len(forced_queue.token_ids),
            "cursor": forced_queue.cursor,
        }
    final_head: s14.FinalHeadReference | None = None
    current_layer: int | None = None
    stage = "bootstrap"
    try:
        for _ in range(config.token_count):
            position = decoder.position
            input_token_id = decoder.input_token_id
            previous = decoder.previous_for(profile)
            session = FullDepthRangeSession(
                catalog,
                cache,
                profile=profile,
                range_attempts=config.range_attempts,
                range_workers=config.range_workers,
            )
            token_report: dict[str, Any] = {
                "position": position,
                "input_token_id": input_token_id,
                "input_source": (
                    "forced_prefill"
                    if decoder.forced_queue is not None and decoder.forced_queue.active
                    else "model_argmax"
                ),
                "completed_layers": [],
                "layers": [],
                "final": None,
                "state_committed": False,
            }
            report["tokens"].append(token_report)
            write_json(config.report_path, report)

            stage = f"position_{position}_embedding"
            embedding = session.prepare_embedding_row(input_token_id)
            state = s14._initial_state(embedding, input_token_id)
            next_states: dict[int, s14.LayerRuntimeState] = {}
            for layer in profile.layers:
                current_layer = layer
                stage = f"position_{position}_layer_{layer}_base"
                prerequisites = session.prepare_layer(layer, input_token_id)
                store = s14.TensorStore(config.asset_root.resolve() / "range_cache")
                store.add_ranges((*prerequisites.non_expert, *prerequisites.router))
                kernel = FullDepthNativeLayerReference(layer, store, profile=profile)

                stage = f"position_{position}_layer_{layer}_native_route"
                pending = kernel.prepare_route(
                    state,
                    token_id=input_token_id,
                    position=position,
                    previous_runtime=previous.get(layer),
                )
                session.submit_top6(layer, input_token_id, pending.route_ids)

                stage = f"position_{position}_layer_{layer}_top6_shared"
                routed = session.fetch_routed(layer, input_token_id)
                pages = list(routed.shared)
                for expert_id in pending.route_ids:
                    pages.extend(routed.experts[expert_id])
                kernel.add_routed(pages)
                moe_branch, state = kernel.finish_layer(pending)
                if tuple(state.shape) != (1, 1, s14.HC_MULT, s14.HIDDEN_SIZE) or state.dtype != torch.bfloat16:
                    raise FullDepthError(f"L{layer} 破坏 BF16 [1,1,4,4096] mHC 状态")
                session.finish_layer(layer, input_token_id)
                next_states[layer] = pending.runtime_state
                token_report["completed_layers"].append(layer)
                token_report["layers"].append(
                    {
                        "layer": layer,
                        "compress_ratio": profile.ratio_for(layer),
                        "route_source": pending.route_source,
                        "expert_ids": pending.route_ids,
                        "route_weights": pending.route_weights,
                        "shared_and_expert_ranges": len(pages),
                        "moe_branch": kernel._summary_tensor(moe_branch),
                        "layer_output": kernel._summary_tensor(state),
                    }
                )
                write_json(config.report_path, report)

            if token_report["completed_layers"] != list(profile.layers):
                raise FullDepthError("禁止跳层进入 final head")
            current_layer = None
            stage = f"position_{position}_final_head"
            final_ranges = session.prepare_final()
            if final_head is None:
                final_head = s14.FinalHeadReference(
                    final_ranges,
                    config.asset_root.resolve() / "range_cache",
                    head_chunk_size=config.head_chunk_size,
                )
            else:
                final_head.validate_ranges(final_ranges)
            final = final_head.forward(state)
            output_token_id = int(final["token_id"])
            decoder.commit(output_token_id=output_token_id, next_states=next_states, profile=profile)
            token_report["final"] = final
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
        report["claim_limit"] = (
            "CPU/PyTorch FullDepth43/native-top6 correctness path; not a speed or quality claim"
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
            "stage": stage,
            "layer": current_layer,
            "position": decoder.position,
            "input_token_id": decoder.input_token_id,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc().splitlines(),
        }
        report["claim_limit"] = "failure before current token commit; no fake token emitted"
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
            forced_prefill_path=args.forced_prefill,
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
