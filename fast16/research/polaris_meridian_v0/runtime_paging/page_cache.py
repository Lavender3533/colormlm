"""具有 LRU、异步预取与 single-flight 的只读专家页缓存。"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from page_catalog import FORMAT, validate_catalog


@dataclass(frozen=True)
class RuntimeSpan:
    file: Path
    offset: int
    length: int


@dataclass(frozen=True)
class RuntimePage:
    key: str
    nbytes: int
    spans: tuple[RuntimeSpan, ...]


class PageSource:
    """每个 I/O worker 保有自己的文件句柄，避免共享 seek 锁串行化。"""

    def __init__(self) -> None:
        self._local = threading.local()
        self._handles_lock = threading.Lock()
        self._all_handles: list[Any] = []
        self._closed = False

    def _handle(self, path: Path):
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = {}
            self._local.handles = handles
        handle = handles.get(path)
        if handle is None or handle.closed:
            if self._closed:
                raise RuntimeError("PageSource 已关闭")
            handle = path.open("rb", buffering=0)
            handles[path] = handle
            with self._handles_lock:
                self._all_handles.append(handle)
        return handle

    def read(self, page: RuntimePage) -> bytes:
        payload = bytearray(page.nbytes)
        view = memoryview(payload)
        cursor = 0
        for span in page.spans:
            handle = self._handle(span.file)
            handle.seek(span.offset)
            destination = view[cursor : cursor + span.length]
            read = handle.readinto(destination)
            if read != span.length:
                raise EOFError(
                    f"专家页短读：{page.key} {span.file}，需要 {span.length}，实际 {read}"
                )
            cursor += span.length
        return bytes(payload)

    def close(self) -> None:
        with self._handles_lock:
            self._closed = True
            handles, self._all_handles = self._all_handles, []
        for handle in handles:
            handle.close()


class ExpertPageCache:
    """字节容量受限的 LRU 缓存。

    ``get`` 的缓存命中路径只做一次互斥锁、一次 dict 查询和一次 LRU 移动；
    miss 在 worker 线程执行。相同页的并发 miss 共用一个 Future，避免重复 SSD I/O。
    """

    def __init__(self, catalog_path: Path, capacity_bytes: int, workers: int = 4) -> None:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        validate_catalog(raw)
        self.pages: dict[str, RuntimePage] = {}
        for item in raw["entries"]:
            spans = tuple(
                RuntimeSpan(Path(span["file"]), int(span["offset"]), int(span["length"]))
                for span in item["spans"]
            )
            self.pages[item["key"]] = RuntimePage(item["key"], int(item["nbytes"]), spans)
        self.capacity_bytes = max(0, int(capacity_bytes))
        self._source = PageSource()
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="polaris-page")
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._inflight: dict[str, Future[bytes]] = {}
        self._resident_bytes = 0
        self._closed = False
        self._stats: dict[str, int] = {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "coalesced_waits": 0,
            "prefetch_requests": 0,
            "physical_loads": 0,
            "bytes_read": 0,
            "io_nanoseconds": 0,
            "evictions": 0,
            "oversize_bypass": 0,
        }

    def _load(self, page: RuntimePage) -> bytes:
        started = time.perf_counter_ns()
        payload = self._source.read(page)
        elapsed = time.perf_counter_ns() - started
        with self._lock:
            self._stats["physical_loads"] += 1
            self._stats["bytes_read"] += len(payload)
            self._stats["io_nanoseconds"] += elapsed
        return payload

    def _complete(self, key: str, future: Future[bytes]) -> None:
        try:
            payload = future.result()
        except BaseException:
            with self._lock:
                if self._inflight.get(key) is future:
                    self._inflight.pop(key, None)
            return
        with self._lock:
            if self._inflight.get(key) is future:
                self._inflight.pop(key, None)
            if key in self._cache:
                return
            if len(payload) > self.capacity_bytes or self.capacity_bytes == 0:
                self._stats["oversize_bypass"] += 1
                return
            while self._cache and self._resident_bytes + len(payload) > self.capacity_bytes:
                _, victim = self._cache.popitem(last=False)
                self._resident_bytes -= len(victim)
                self._stats["evictions"] += 1
            self._cache[key] = payload
            self._resident_bytes += len(payload)

    def _start(self, key: str, *, prefetch: bool) -> Future[bytes]:
        if key not in self.pages:
            raise KeyError(f"目录中没有专家页：{key}")
        with self._lock:
            if self._closed:
                raise RuntimeError("ExpertPageCache 已关闭")
            resident = self._cache.get(key)
            if resident is not None:
                completed: Future[bytes] = Future()
                completed.set_result(resident)
                return completed
            future = self._inflight.get(key)
            if future is not None:
                if not prefetch:
                    self._stats["coalesced_waits"] += 1
                return future
            self._stats["misses"] += 1
            future = self._executor.submit(self._load, self.pages[key])
            self._inflight[key] = future
            future.add_done_callback(lambda done, selected=key: self._complete(selected, done))
            return future

    def get(self, key: str) -> bytes:
        with self._lock:
            self._stats["requests"] += 1
            resident = self._cache.get(key)
            if resident is not None:
                self._stats["hits"] += 1
                self._cache.move_to_end(key)
                return resident
        return self._start(key, prefetch=False).result()

    def prefetch(self, keys: Iterable[str]) -> dict[str, Future[bytes]]:
        scheduled: dict[str, Future[bytes]] = {}
        for key in dict.fromkeys(keys):
            with self._lock:
                self._stats["prefetch_requests"] += 1
            scheduled[key] = self._start(key, prefetch=True)
        return scheduled

    def wait(self, futures: Iterable[Future[bytes]]) -> None:
        for future in futures:
            future.result()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = dict(self._stats)
            result.update(
                {
                    "catalog_pages": len(self.pages),
                    "resident_pages": len(self._cache),
                    "resident_bytes": self._resident_bytes,
                    "capacity_bytes": self.capacity_bytes,
                    "inflight": len(self._inflight),
                }
            )
        requests = result["requests"]
        result["hit_rate"] = result["hits"] / requests if requests else 0.0
        seconds = result["io_nanoseconds"] / 1e9
        result["aggregate_io_mib_per_second"] = (
            result["bytes_read"] / (1024**2) / seconds if seconds else 0.0
        )
        return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._source.close()

    def __enter__(self) -> "ExpertPageCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_catalog(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != FORMAT:
        raise ValueError("不是北极星专家页目录")
    validate_catalog(raw)
    return raw
