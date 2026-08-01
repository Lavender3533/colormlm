"""Replay interface for native FullDepth43 top-6 expert-page traces."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .assets import EXPERTS_PER_LAYER, LAYERS, TOP_K


class ReplayContractError(ValueError):
    """Raised when a route trace cannot prove complete native top-6 coverage."""


@dataclass(frozen=True, order=True)
class ExpertPage:
    layer: int
    expert_id: int


@dataclass(frozen=True)
class RouteBlock:
    block_id: str
    routes_by_token: tuple[tuple[tuple[int, ...], ...], ...]
    accepted_prefix_length: int | None = None

    @property
    def block_size(self) -> int:
        return len(self.routes_by_token)

    @classmethod
    def from_rows(
        cls,
        block_id: str,
        block_size: int,
        rows: Sequence[dict[str, Any]],
        *,
        accepted_prefix_length: int | None = None,
    ) -> "RouteBlock":
        if not isinstance(block_id, str) or not block_id:
            raise ReplayContractError("block_id 必须是非空字符串")
        if block_size <= 0:
            raise ReplayContractError("block_size 必须为正整数")
        if accepted_prefix_length is not None and not 0 <= accepted_prefix_length <= block_size:
            raise ReplayContractError("accepted_prefix_length 越界")
        table: dict[tuple[int, int], tuple[int, ...]] = {}
        for row in rows:
            token_offset = row.get("token_offset")
            layer = row.get("layer")
            experts_value = row.get("expert_ids")
            if not isinstance(token_offset, int) or isinstance(token_offset, bool) or not 0 <= token_offset < block_size:
                raise ReplayContractError(f"token_offset 越界: {token_offset}")
            if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < LAYERS:
                raise ReplayContractError(f"layer 越界: {layer}")
            if not isinstance(experts_value, list):
                raise ReplayContractError("expert_ids 必须是 list")
            experts = tuple(experts_value)
            if len(experts) != TOP_K or len(set(experts)) != TOP_K:
                raise ReplayContractError(f"token={token_offset}/L{layer} 必须恰有 6 个不同专家")
            if any(
                not isinstance(expert, int)
                or isinstance(expert, bool)
                or not 0 <= expert < EXPERTS_PER_LAYER
                for expert in experts
            ):
                raise ReplayContractError(f"token={token_offset}/L{layer} 专家 ID 越界")
            key = (token_offset, layer)
            if key in table:
                raise ReplayContractError(f"重复 route: token={token_offset}/L{layer}")
            table[key] = experts

        expected_rows = block_size * LAYERS
        if len(table) != expected_rows:
            missing = [
                f"{token}/{layer}"
                for token in range(block_size)
                for layer in range(LAYERS)
                if (token, layer) not in table
            ]
            raise ReplayContractError(f"必须覆盖 {expected_rows} 个 token/layer route，缺失 {missing[:5]}")
        routes = tuple(
            tuple(table[(token, layer)] for layer in range(LAYERS))
            for token in range(block_size)
        )
        return cls(
            block_id=block_id,
            routes_by_token=routes,
            accepted_prefix_length=accepted_prefix_length,
        )

    def unique_pages_in_execution_order(self) -> tuple[ExpertPage, ...]:
        """De-duplicate a causal block layer by layer, preserving first use."""

        seen: set[ExpertPage] = set()
        result: list[ExpertPage] = []
        for layer in range(LAYERS):
            for token_routes in self.routes_by_token:
                for expert_id in token_routes[layer]:
                    page = ExpertPage(layer, expert_id)
                    if page not in seen:
                        seen.add(page)
                        result.append(page)
        return tuple(result)


class ExpertPageCache:
    """Deterministic LRU device cache; streaming misses may evict old pages."""

    def __init__(self, capacity_pages: int):
        if capacity_pages < 0:
            raise ValueError("capacity_pages 不能为负")
        self.capacity_pages = capacity_pages
        self._pages: OrderedDict[ExpertPage, None] = OrderedDict()

    def access(self, page: ExpertPage) -> bool:
        if page in self._pages:
            self._pages.move_to_end(page)
            return True
        if self.capacity_pages == 0:
            return False
        self._pages[page] = None
        if len(self._pages) > self.capacity_pages:
            self._pages.popitem(last=False)
        return False

    def clear(self) -> None:
        self._pages.clear()

    def snapshot(self) -> tuple[ExpertPage, ...]:
        return tuple(self._pages)


@dataclass(frozen=True)
class BlockReplay:
    block_id: str
    block_size: int
    selected_page_references: int
    unique_pages_after_block_dedup: int
    block_deduplicated_references: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    pcie_expert_bytes: int
    accepted_prefix_length: int | None


@dataclass(frozen=True)
class ReplayReport:
    cache_capacity_pages: int
    expert_page_bytes: int
    blocks: tuple[BlockReplay, ...]
    selected_page_references: int
    unique_page_requests: int
    block_deduplicated_references: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    pcie_expert_bytes: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["blocks"] = [asdict(block) for block in self.blocks]
        return result


def replay_blocks(
    blocks: Iterable[RouteBlock],
    cache: ExpertPageCache,
    expert_page_bytes: int,
) -> ReplayReport:
    if expert_page_bytes <= 0:
        raise ValueError("expert_page_bytes 必须为正整数")
    block_reports: list[BlockReplay] = []
    selected_total = unique_total = dedup_total = hits_total = misses_total = 0
    for block in blocks:
        selected = block.block_size * LAYERS * TOP_K
        pages = block.unique_pages_in_execution_order()
        hits = 0
        misses = 0
        for page in pages:
            if cache.access(page):
                hits += 1
            else:
                misses += 1
        unique = len(pages)
        deduplicated = selected - unique
        hit_rate = hits / unique if unique else 0.0
        block_reports.append(
            BlockReplay(
                block_id=block.block_id,
                block_size=block.block_size,
                selected_page_references=selected,
                unique_pages_after_block_dedup=unique,
                block_deduplicated_references=deduplicated,
                cache_hits=hits,
                cache_misses=misses,
                cache_hit_rate=hit_rate,
                pcie_expert_bytes=misses * expert_page_bytes,
                accepted_prefix_length=block.accepted_prefix_length,
            )
        )
        selected_total += selected
        unique_total += unique
        dedup_total += deduplicated
        hits_total += hits
        misses_total += misses

    return ReplayReport(
        cache_capacity_pages=cache.capacity_pages,
        expert_page_bytes=expert_page_bytes,
        blocks=tuple(block_reports),
        selected_page_references=selected_total,
        unique_page_requests=unique_total,
        block_deduplicated_references=dedup_total,
        cache_hits=hits_total,
        cache_misses=misses_total,
        cache_hit_rate=hits_total / unique_total if unique_total else 0.0,
        pcie_expert_bytes=misses_total * expert_page_bytes,
    )


def load_route_blocks_jsonl(path: str | Path) -> tuple[RouteBlock, ...]:
    """Load strict UTF-8 JSONL; one object is one complete causal block."""

    result: list[RouteBlock] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                result.append(
                    RouteBlock.from_rows(
                        block_id=row["block_id"],
                        block_size=row["block_size"],
                        rows=row["routes"],
                        accepted_prefix_length=row.get("accepted_prefix_length"),
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError, ReplayContractError) as exc:
                raise ReplayContractError(f"{path}:{line_number}: {exc}") from exc
    if not result:
        raise ReplayContractError("路由回放文件为空")
    return tuple(result)
