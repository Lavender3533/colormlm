#!/usr/bin/env python3
"""离线验证真实 Python RouteFirstSession JSONL worker；不访问网络。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RANGE_PACK = REPO_ROOT / "fast16" / "research" / "polaris_meridian_v1" / "s14_range_pack"
sys.path.insert(0, str(RANGE_PACK))

import online_range as online  # noqa: E402
import range_worker  # noqa: E402
from test_online_range import FakeHttpsTransport, make_catalog  # noqa: E402


class RangeWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = make_catalog()
        self.catalog_path = self.root / "catalog.json"
        self.catalog_path.write_text(
            json.dumps(self.catalog, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        self.cache_dir = self.root / "cache"
        totals = {
            filename: row["file_bytes"]
            for filename, row in self.catalog["headers"]["files"].items()
        }
        transport = FakeHttpsTransport(totals)
        cache = online.RangeCache(
            self.cache_dir,
            endpoint="https://fixture.invalid",
            transport=transport,
            allow_fetch=True,
            download_budget_bytes=1 << 20,
            cache_budget_bytes=1 << 20,
        )
        layer = self.catalog["layers"]["0"]
        pages = (
            [online.embedding_row_entry(self.catalog, 0), online.embedding_row_entry(self.catalog, 1)]
            + list(layer["non_expert"])
            + list(layer["router"])
            + list(layer["shared"])
        )
        for expert_id in [126, 12, 205, 149, 227, 174]:
            pages.extend(layer["experts"][str(expert_id)])
        for entry in pages:
            cache.fetch(entry)

    def request(self, process: subprocess.Popen[str], **request: object) -> dict:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        process.stdin.flush()
        response = json.loads(process.stdout.readline())
        self.assertEqual(response["request_id"], request["request_id"])
        self.assertEqual(response["op"], request["op"])
        return response

    @staticmethod
    def close_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def test_offline_worker_runs_real_session_for_one_layer(self) -> None:
        process = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "range_worker.py"),
                "--catalog",
                str(self.catalog_path),
                "--cache-dir",
                str(self.cache_dir),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self.addCleanup(self.close_process, process)
        common = {
            "protocol": "polaris-s14-range-jsonl-v1",
            "download_authorized": False,
        }
        hello = self.request(
            process,
            **common,
            request_id=1,
            op="hello",
            repo="deepseek-ai/DeepSeek-V4-Flash-0731",
            revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            profile="s14_top6",
            selected_layers=[0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42],
            top_k=6,
        )
        self.assertEqual(hello["status"], "ok")
        base = self.request(
            process, **common, request_id=2, op="prepare_base", layer=0, token_id=0
        )
        self.assertEqual(base["status"], "ok")
        self.assertTrue(all(item["cache_hit"] for item in base["artifacts"]))
        routed = self.request(
            process,
            **common,
            request_id=3,
            op="prepare_routed",
            layer=0,
            token_id=0,
            expert_ids=[126, 12, 205, 149, 227, 174],
        )
        self.assertEqual(routed["expert_ids"], [126, 12, 205, 149, 227, 174])
        self.assertEqual(routed["observation"]["expert_cache_hits"], 6)
        released = self.request(
            process, **common, request_id=4, op="release_layer", layer=0, token_id=0
        )
        self.assertEqual(released["status"], "ok")
        self.request(process, **common, request_id=5, op="shutdown")
        self.assertEqual(process.wait(timeout=2), 0)

    def test_request_cannot_elevate_download_authorization(self) -> None:
        process = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "range_worker.py"),
                "--catalog",
                str(self.catalog_path),
                "--cache-dir",
                str(self.cache_dir),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self.addCleanup(self.close_process, process)
        response = self.request(
            process,
            protocol="polaris-s14-range-jsonl-v1",
            request_id=1,
            op="hello",
            repo="deepseek-ai/DeepSeek-V4-Flash-0731",
            revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            profile="s14_top6",
            selected_layers=[0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42],
            top_k=6,
            download_authorized=True,
        )
        self.assertEqual(response["status"], "error")
        self.assertNotEqual(process.wait(timeout=2), 0)

    def test_abort_discards_half_open_session_without_fetching_more_pages(self) -> None:
        worker = range_worker.Worker(
            Namespace(
                catalog=self.catalog_path,
                cache_dir=self.cache_dir,
                endpoint="https://fixture.invalid",
                download_authorized=False,
                download_budget_bytes=0,
                cache_budget_bytes=None,
                require_authoritative=False,
                http_timeout=1.0,
            )
        )
        common = {
            "protocol": "polaris-s14-range-jsonl-v1",
            "download_authorized": False,
        }
        worker.handle(
            {
                **common,
                "request_id": 1,
                "op": "hello",
                "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
                "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
                "profile": "s14_top6",
                "selected_layers": [0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42],
                "top_k": 6,
            }
        )
        worker.handle({**common, "request_id": 2, "op": "prepare_base", "layer": 0, "token_id": 0})
        response, _ = worker.handle(
            {**common, "request_id": 3, "op": "abort_layer", "layer": 0, "token_id": 0}
        )
        self.assertTrue(response["aborted"])
        response, _ = worker.handle(
            {**common, "request_id": 4, "op": "prepare_base", "layer": 0, "token_id": 1}
        )
        self.assertEqual(response["status"], "ok")


if __name__ == "__main__":
    unittest.main()
