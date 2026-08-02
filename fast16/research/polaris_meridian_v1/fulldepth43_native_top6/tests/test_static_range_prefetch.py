from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping
from unittest.mock import patch

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor import (
    ExecutionConfig,
    FullDepthError,
    FullDepthRangeSession,
    FullDepthTokenWorker,
    SessionPhase,
)


TOKEN_ID = 23
ROUTE = (0, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class FakeProfile:
    layers: tuple[int, ...] = (0, 1, 2)
    top_k: int = 6

    def validate(self) -> None:
        return None


def entry(layer: int, kind: str, index: int) -> dict[str, Any]:
    return {
        "layer": layer,
        "kind": kind,
        "index": index,
        "identity": f"layer-{layer}-{kind}-{index}",
    }


def fake_catalog() -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer in FakeProfile().layers:
        layers[str(layer)] = {
            "non_expert": [entry(layer, "non_expert", 0), entry(layer, "non_expert", 1)],
            "router": [entry(layer, "router", 0)],
            "experts": {
                str(expert_id): [entry(layer, "expert", expert_id)]
                for expert_id in ROUTE
            },
            "shared": [entry(layer, "shared", 0)],
        }
    return {
        "layers": layers,
        "boundary": {
            "final": [entry(3, "final", 0)],
        },
    }


class RecordingCache:
    def __init__(self, *, expected_calls: int = 0) -> None:
        self.calls: list[Mapping[str, Any]] = []
        self._lock = threading.Lock()
        self.complete = threading.Event()
        self.expected_calls = expected_calls

    def fetch(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            self.calls.append(value)
            if self.expected_calls and len(self.calls) >= self.expected_calls:
                self.complete.set()
        return value


class PrefetchBoom(RuntimeError):
    pass


class FailingCache(RecordingCache):
    def fetch(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            self.calls.append(value)
        self.complete.set()
        raise PrefetchBoom("下一层静态页损坏")


class BlockingCache(RecordingCache):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def fetch(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            self.calls.append(value)
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("测试未释放阻塞 Range fetch")
        self.finished.set()
        return value


def shutdown(pool: ThreadPoolExecutor) -> None:
    pool.shutdown(wait=True, cancel_futures=True)


class StaticRangePrefetchTests(unittest.TestCase):
    def test_range_and_prefetch_pools_must_be_distinct(self) -> None:
        pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(shutdown, pool)
        with patch(
            "fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor.validate_catalog"
        ):
            with self.assertRaisesRegex(FullDepthError, "两个独立线程池"):
                FullDepthRangeSession(
                    fake_catalog(),
                    RecordingCache(),
                    profile=FakeProfile(),
                    range_attempts=1,
                    range_workers=1,
                    range_pool=pool,
                    prefetch_pool=pool,
                )

    def test_cleanup_failure_does_not_replace_primary_token_failure(self) -> None:
        class PrimaryBoom(RuntimeError):
            pass

        class CleanupBoom(RuntimeError):
            pass

        class FakeSession:
            def close(self) -> None:
                raise CleanupBoom("后台预取清理失败")

        with patch(
            "fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor.validate_catalog"
        ):
            worker = FullDepthTokenWorker(
                ExecutionConfig(),
                fake_catalog(),
                RecordingCache(),
                profile=FakeProfile(),
            )
        with (
            patch.object(worker, "start"),
            patch.object(worker, "_validate_previous", return_value={}),
            patch.object(
                worker,
                "_compute_token",
                side_effect=PrimaryBoom("主模型路径失败"),
            ),
            patch(
                "fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor.FullDepthRangeSession",
                return_value=FakeSession(),
            ),
        ):
            with self.assertRaisesRegex(PrimaryBoom, "主模型路径失败"):
                worker(0, TOKEN_ID, {})
        self.assertEqual(
            worker.range_cleanup_errors,
            [{"type": "CleanupBoom", "message": "后台预取清理失败"}],
        )

    def make_session(
        self,
        cache: RecordingCache,
        *,
        range_workers: int = 3,
        prefetch_workers: int = 1,
    ) -> tuple[FullDepthRangeSession, ThreadPoolExecutor, ThreadPoolExecutor]:
        range_pool = ThreadPoolExecutor(
            max_workers=range_workers,
            thread_name_prefix="test-persistent-range",
        )
        prefetch_pool = ThreadPoolExecutor(
            max_workers=prefetch_workers,
            thread_name_prefix="test-static-prefetch",
        )
        self.addCleanup(shutdown, prefetch_pool)
        self.addCleanup(shutdown, range_pool)
        with patch(
            "fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor.validate_catalog"
        ):
            session = FullDepthRangeSession(
                fake_catalog(),
                cache,
                profile=FakeProfile(),
                range_attempts=1,
                range_workers=range_workers,
                range_pool=range_pool,
                prefetch_pool=prefetch_pool,
            )
        session.phase = SessionPhase.LAYER_READY
        session.layer_index = 0
        session.token_id = TOKEN_ID
        session.route = ROUTE
        return session, range_pool, prefetch_pool

    def test_prefetch_reads_only_next_layer_non_expert_and_router(self) -> None:
        cache = RecordingCache(expected_calls=3)
        session, _, _ = self.make_session(cache)

        self.assertTrue(session.schedule_next_static(1, TOKEN_ID))
        self.assertTrue(cache.complete.wait(timeout=2.0))

        observed = {(value["layer"], value["kind"]) for value in cache.calls}
        self.assertEqual(observed, {(1, "non_expert"), (1, "router")})
        self.assertEqual(len(cache.calls), 3)
        self.assertFalse(
            {"expert", "shared", "final"}
            & {str(value["kind"]) for value in cache.calls}
        )
        session.close()

    def test_schedule_is_observationally_pure_for_route_first_state(self) -> None:
        cache = RecordingCache(expected_calls=3)
        session, _, _ = self.make_session(cache)
        before = (
            session.phase,
            session.layer_index,
            session.token_id,
            session.route,
            session.current_layer,
        )

        self.assertTrue(session.schedule_next_static(1, TOKEN_ID))
        self.assertTrue(cache.complete.wait(timeout=2.0))

        self.assertEqual(
            (
                session.phase,
                session.layer_index,
                session.token_id,
                session.route,
                session.current_layer,
            ),
            before,
        )
        session.close()

    def test_completed_prefetch_does_not_advance_before_formal_prepare(self) -> None:
        cache = RecordingCache(expected_calls=3)
        session, _, _ = self.make_session(cache)

        self.assertTrue(session.schedule_next_static(1, TOKEN_ID))
        self.assertTrue(cache.complete.wait(timeout=2.0))
        self.assertEqual(session.phase, SessionPhase.LAYER_READY)
        self.assertEqual(session.layer_index, 0)
        self.assertEqual(session.current_layer, 0)

        session.finish_layer(0, TOKEN_ID)
        self.assertEqual(session.phase, SessionPhase.AWAITING_LAYER)
        prepared = session.prepare_layer(1, TOKEN_ID)

        self.assertEqual(prepared.layer, 1)
        self.assertEqual(prepared.token_id, TOKEN_ID)
        self.assertEqual(
            tuple(value["kind"] for value in prepared.non_expert),
            ("non_expert", "non_expert"),
        )
        self.assertEqual(tuple(value["kind"] for value in prepared.router), ("router",))
        self.assertEqual(len(cache.calls), 3, "正式 prepare 不得重复 fetch 已预取静态页")
        self.assertEqual(session.phase, SessionPhase.LAYER_BASE_READY)
        self.assertEqual(session.layer_index, 1)
        session.close()

    def test_prefetch_exception_propagates_from_prepare_without_state_commit(self) -> None:
        cache = FailingCache()
        session, _, _ = self.make_session(cache)

        self.assertTrue(session.schedule_next_static(1, TOKEN_ID))
        self.assertTrue(cache.complete.wait(timeout=2.0))
        session.finish_layer(0, TOKEN_ID)
        before = (session.phase, session.layer_index, session.token_id, session.route)

        with self.assertRaisesRegex(PrefetchBoom, "下一层静态页损坏"):
            session.prepare_layer(1, TOKEN_ID)

        self.assertEqual(
            (session.phase, session.layer_index, session.token_id, session.route),
            before,
        )
        self.assertEqual(session.phase, SessionPhase.AWAITING_LAYER)
        with self.assertRaisesRegex(PrefetchBoom, "下一层静态页损坏"):
            session.close()

    def test_close_joins_running_prefetch(self) -> None:
        cache = BlockingCache()
        session, range_pool, prefetch_pool = self.make_session(cache, range_workers=1)
        self.assertTrue(session.schedule_next_static(1, TOKEN_ID))
        self.assertTrue(cache.started.wait(timeout=2.0))
        close_result: list[BaseException | None] = []

        def close_session() -> None:
            try:
                session.close()
            except BaseException as error:  # pragma: no cover - 失败内容由主线程断言
                close_result.append(error)
            else:
                close_result.append(None)

        closer = threading.Thread(target=close_session, name="test-session-close")
        closer.start()
        time.sleep(0.05)
        self.assertTrue(closer.is_alive(), "close 必须等待正在执行的 prefetch")
        cache.release.set()
        closer.join(timeout=2.0)

        self.assertFalse(closer.is_alive())
        self.assertEqual(close_result, [None])
        self.assertTrue(cache.finished.is_set())
        with self.assertRaises(RuntimeError):
            range_pool.submit(lambda: None)
        with self.assertRaises(RuntimeError):
            prefetch_pool.submit(lambda: None)

    def test_close_cancels_prefetch_that_has_not_started(self) -> None:
        cache = RecordingCache()
        session, _, prefetch_pool = self.make_session(cache, range_workers=1)
        blocker_started = threading.Event()
        release_blocker = threading.Event()

        def blocker() -> None:
            blocker_started.set()
            release_blocker.wait(timeout=5.0)

        blocker_future = prefetch_pool.submit(blocker)
        self.assertTrue(blocker_started.wait(timeout=2.0))
        self.assertTrue(session.schedule_next_static(1, TOKEN_ID))
        close_result: list[BaseException | None] = []

        def close_session() -> None:
            try:
                session.close()
            except BaseException as error:  # pragma: no cover - 失败内容由主线程断言
                close_result.append(error)
            else:
                close_result.append(None)

        closer = threading.Thread(target=close_session, name="test-cancel-close")
        closer.start()
        time.sleep(0.05)
        release_blocker.set()
        closer.join(timeout=2.0)
        blocker_future.result(timeout=2.0)

        self.assertFalse(closer.is_alive())
        self.assertEqual(close_result, [None])
        self.assertEqual(cache.calls, [], "close 必须取消尚未开始的静态预取")

    def test_persistent_range_pool_preserves_map_result_order(self) -> None:
        class DelayedCache(RecordingCache):
            def fetch(self, value: Mapping[str, Any]) -> tuple[int, str]:
                time.sleep(float(value["delay"]))
                return int(value["value"]), threading.current_thread().name

        cache = DelayedCache()
        session, _, _ = self.make_session(cache)
        first = session._fetch_all(
            (
                {"value": 3, "delay": 0.03},
                {"value": 1, "delay": 0.00},
                {"value": 2, "delay": 0.01},
            )
        )
        second = session._fetch_all(
            (
                {"value": 6, "delay": 0.02},
                {"value": 4, "delay": 0.00},
                {"value": 5, "delay": 0.01},
            )
        )

        self.assertEqual(tuple(value for value, _ in first), (3, 1, 2))
        self.assertEqual(tuple(value for value, _ in second), (6, 4, 5))
        self.assertTrue(
            all(
                thread_name.startswith("test-persistent-range")
                for _, thread_name in first + second
            ),
            "两批 fetch 都必须复用注入的持久 range pool",
        )
        session.close()


if __name__ == "__main__":
    unittest.main()
