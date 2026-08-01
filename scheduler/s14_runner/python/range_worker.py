#!/usr/bin/env python3
"""把 Python ``RouteFirstSession`` 暴露成持久 UTF-8 JSONL worker。

本进程只负责固定 revision S14/top-6 的 Range 生命周期与缓存证明；top-6
必须由 Rust runner 中的原生 executor 在当前 token/层计算后提交。默认严格离线，
只有命令行显式 ``--download-authorized`` 且预算大于零时才允许 cache miss 下载。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RANGE_PACK = REPO_ROOT / "fast16" / "research" / "polaris_meridian_v1" / "s14_range_pack"
sys.path.insert(0, str(RANGE_PACK))

import online_range as online  # noqa: E402
import range_pack as rp  # noqa: E402


PROTOCOL = "polaris-s14-range-jsonl-v1"
PROFILE = "s14_top6"
SELECTED_LAYERS = [0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42]


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _artifact(item: online.CachedRange) -> dict[str, Any]:
    entry = dict(item.entry)
    proof = dict(item.proof)
    return {
        "tensor": str(entry["tensor"]),
        "kind": str(entry["kind"]),
        "expert_id": entry.get("expert_id"),
        "path": str(item.path.resolve()),
        "bytes": int(entry["bytes"]),
        "cache_hit": bool(item.cache_hit),
        "observed_sha256": str(proof["observed_sha256"]),
        "authoritative": proof.get("authoritative") is True,
    }


def _artifacts(items: Iterable[online.CachedRange]) -> list[dict[str, Any]]:
    return [_artifact(item) for item in items]


def _observation(
    artifacts: list[dict[str, Any]],
    *,
    elapsed_ns: int,
    expert_groups: dict[int, tuple[online.CachedRange, ...]] | None = None,
) -> dict[str, int]:
    misses = [item for item in artifacts if not item["cache_hit"]]
    expert_hits = 0
    expert_misses = 0
    if expert_groups is not None:
        for pages in expert_groups.values():
            if all(page.cache_hit for page in pages):
                expert_hits += 1
            else:
                expert_misses += 1
    return {
        "disk_bytes": sum(int(item["bytes"]) for item in artifacts),
        "host_to_device_bytes": 0,
        "expert_cache_hits": expert_hits,
        "expert_cache_misses": expert_misses,
        "miss_stall_ns": elapsed_ns if misses else 0,
    }


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.download_authorized and args.download_budget_bytes <= 0:
            raise rp.ContractError("授权下载时 download budget 必须大于零")
        if not args.download_authorized and args.download_budget_bytes != 0:
            raise rp.ContractError("未授权下载时 download budget 必须为零")
        self.catalog = rp.read_json(args.catalog)
        online.validate_catalog(self.catalog)
        if self.catalog.get("download_authorized") is not False:
            raise rp.ContractError("catalog 不得自行授予下载权限")
        self.download_authorized = bool(args.download_authorized)
        self.cache = online.RangeCache(
            args.cache_dir,
            endpoint=args.endpoint,
            allow_fetch=self.download_authorized,
            download_budget_bytes=args.download_budget_bytes,
            cache_budget_bytes=args.cache_budget_bytes,
            require_authoritative=args.require_authoritative,
            timeout=args.http_timeout,
        )
        self.session: online.RouteFirstSession | None = None
        self.hello_complete = False

    @staticmethod
    def _request_identity(request: dict[str, Any]) -> None:
        if request.get("protocol") != PROTOCOL:
            raise rp.ContractError("JSONL protocol 不兼容")
        request_id = request.get("request_id")
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 1:
            raise rp.ContractError("request_id 必须是正整数")

    def _authorization(self, request: dict[str, Any]) -> None:
        if request.get("download_authorized") is not self.download_authorized:
            raise rp.ContractError("逐请求 download_authorized 与启动安全门不一致")

    def handle(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        self._request_identity(request)
        op = request.get("op")
        request_id = int(request["request_id"])
        if not isinstance(op, str):
            raise rp.ContractError("op 必须是 string")
        base: dict[str, Any] = {
            "protocol": PROTOCOL,
            "request_id": request_id,
            "op": op,
            "status": "ok",
        }
        if op == "hello":
            if self.hello_complete:
                raise rp.ContractError("hello 只能执行一次")
            self._authorization(request)
            exact = (
                request.get("repo") == rp.REPO
                and request.get("revision") == rp.REVISION
                and request.get("profile") == PROFILE
                and request.get("selected_layers") == SELECTED_LAYERS
                and request.get("top_k") == online.TOP_K
            )
            if not exact:
                raise rp.ContractError("拒绝非冻结 revision/S14/top-6 身份")
            self.hello_complete = True
            base.update(
                repo=rp.REPO,
                revision=rp.REVISION,
                profile=PROFILE,
                selected_layers=SELECTED_LAYERS,
                top_k=online.TOP_K,
                download_authorized=self.download_authorized,
            )
            return base, False
        if not self.hello_complete:
            raise rp.ContractError("必须先完成 hello")
        self._authorization(request)
        if op == "shutdown":
            return base, True
        if op == "prepare_base":
            layer = request.get("layer")
            token_id = request.get("token_id")
            if layer == SELECTED_LAYERS[0]:
                if self.session is not None and self.session.phase is not online.SessionPhase.COMPLETE:
                    raise rp.ContractError("上一 token 尚未完成，拒绝重建 session")
                self.session = online.RouteFirstSession(self.catalog, self.cache)
                started = time.perf_counter_ns()
                embedding = (self.session.prepare_embedding_row(int(token_id)),)
            else:
                started = time.perf_counter_ns()
                embedding = ()
            if self.session is None:
                raise rp.ContractError("token 必须从 L0 开始")
            ready = self.session.prepare_layer(layer, token_id)
            pages = tuple(embedding) + ready.non_expert + ready.router
            artifacts = _artifacts(pages)
            base.update(
                layer=layer,
                artifacts=artifacts,
                observation=_observation(
                    artifacts,
                    elapsed_ns=time.perf_counter_ns() - started,
                ),
            )
            return base, False
        if op == "prepare_routed":
            if self.session is None:
                raise rp.ContractError("prepare_routed 前没有 session")
            layer = request.get("layer")
            token_id = request.get("token_id")
            expert_ids = request.get("expert_ids")
            self.session.submit_top6(layer, token_id, expert_ids)
            started = time.perf_counter_ns()
            routed = self.session.fetch_routed(layer, token_id)
            pages = list(routed.shared)
            for expert_id in routed.expert_ids:
                pages.extend(routed.experts[expert_id])
            artifacts = _artifacts(pages)
            base.update(
                layer=layer,
                expert_ids=list(routed.expert_ids),
                artifacts=artifacts,
                observation=_observation(
                    artifacts,
                    elapsed_ns=time.perf_counter_ns() - started,
                    expert_groups=dict(routed.experts),
                ),
            )
            return base, False
        if op == "release_layer":
            if self.session is None:
                raise rp.ContractError("release 前没有 session")
            layer = request.get("layer")
            token_id = request.get("token_id")
            self.session.finish_layer(layer, token_id)
            final_artifacts: list[dict[str, Any]] = []
            if self.session.phase is online.SessionPhase.FINAL_PENDING:
                final_artifacts = _artifacts(self.session.prepare_final())
            base.update(layer=layer, final_artifacts=final_artifacts)
            return base, False
        if op == "abort_layer":
            if self.session is None:
                raise rp.ContractError("abort 前没有 session")
            layer = request.get("layer")
            token_id = request.get("token_id")
            if layer != self.session.current_layer or token_id != self.session._token_id:
                raise rp.ContractError("abort 的 layer/token 与当前状态不匹配")
            if self.session.phase not in {
                online.SessionPhase.LAYER_BASE_READY,
                online.SessionPhase.ROUTED,
                online.SessionPhase.LAYER_READY,
            }:
                raise rp.ContractError("当前 session phase 不能 abort")
            self.session = None
            base.update(layer=layer, aborted=True)
            return base, False
        raise rp.ContractError(f"未知 op：{op}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--download-authorized", action="store_true")
    parser.add_argument("--download-budget-bytes", type=int, default=0)
    parser.add_argument("--cache-budget-bytes", type=int)
    parser.add_argument("--require-authoritative", action="store_true")
    parser.add_argument("--http-timeout", type=float, default=300.0)
    return parser


def main() -> int:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        worker = Worker(build_parser().parse_args())
    except Exception as error:
        print(f"Range worker 启动失败: {error}", file=sys.stderr)
        return 2
    for raw in sys.stdin:
        request: dict[str, Any] | None = None
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise rp.ContractError("JSONL request 必须是 object")
            response, stop = worker.handle(request)
            _emit(response)
            if stop:
                return 0
        except Exception as error:
            request_id = request.get("request_id", 0) if isinstance(request, dict) else 0
            op = request.get("op", "invalid") if isinstance(request, dict) else "invalid"
            _emit(
                {
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "op": op,
                    "status": "error",
                    "error": str(error),
                }
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
