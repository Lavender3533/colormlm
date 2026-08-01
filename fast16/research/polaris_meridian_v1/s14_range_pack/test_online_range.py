#!/usr/bin/env python3
"""完全离线 fake HTTPS/206 fixture；不会访问 DNS 或真实权重。"""

from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import online_range as online
import range_pack as rp


class FakeResponse:
    def __init__(
        self,
        data: bytes,
        *,
        status: int,
        content_range: str,
        final_url: str,
        interrupt_after: int | None = None,
    ) -> None:
        self.data = data
        self.status = status
        self.headers = {
            "Content-Range": content_range,
            "Content-Length": str(len(data)),
        }
        self.final_url = final_url
        self.interrupt_after = interrupt_after
        self.cursor = 0

    def read(self, size: int = -1) -> bytes:
        if self.interrupt_after is not None and self.cursor >= self.interrupt_after:
            raise ConnectionError("fixture interrupted")
        if self.cursor >= len(self.data):
            return b""
        wanted = len(self.data) - self.cursor if size < 0 else min(size, len(self.data) - self.cursor)
        if self.interrupt_after is not None:
            wanted = min(wanted, self.interrupt_after - self.cursor)
        if wanted <= 0:
            raise ConnectionError("fixture interrupted")
        result = self.data[self.cursor:self.cursor + wanted]
        self.cursor += len(result)
        return result

    def geturl(self) -> str:
        return self.final_url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class FakeHttpsTransport:
    def __init__(self, totals: dict[str, int]) -> None:
        self.totals = totals
        self.requests: list[tuple[str, int, int]] = []
        self.lock = threading.Lock()
        self.interrupt_once: dict[tuple[str, int, int], int] = {}
        self.status = 206
        self.bad_content_range = False
        self.final_scheme = "https"

    @staticmethod
    def payload(start: int, end: int) -> bytes:
        return bytes((offset * 17 + 11) % 251 for offset in range(start, end + 1))

    def open_range(self, url: str, start: int, end: int, timeout: float) -> FakeResponse:
        del timeout
        filename = urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[1])
        key = (filename, start, end)
        with self.lock:
            self.requests.append(key)
            interrupt_after = self.interrupt_once.pop(key, None)
        content_range = f"bytes {start}-{end}/{self.totals[filename]}"
        if self.bad_content_range:
            content_range = f"bytes {start}-{end}/{self.totals[filename] + 1}"
        final_url = url if self.final_scheme == "https" else "http://fixture.invalid/downgraded"
        return FakeResponse(
            self.payload(start, end),
            status=self.status,
            content_range=content_range,
            final_url=final_url,
            interrupt_after=interrupt_after,
        )


def make_catalog() -> dict:
    source = rp.read_json(rp.SOURCE_CONTRACT)
    contracts = rp.source_file_contracts(source)
    files: dict[str, dict] = {}
    for filename, contract in contracts.items():
        files[filename] = {
            "file_bytes": int(contract["bytes"]),
            "header_length": 88,
            "data_start": 96,
            "header_sha256": hashlib.sha256(("header:" + filename).encode()).hexdigest(),
            "tensor_table_sha256": hashlib.sha256(("table:" + filename).encode()).hexdigest(),
            "integrity": "tofu_fixed_revision_not_authoritative",
            "authoritative": False,
        }

    def entry(name: str, kind: str, filename: str, start: int, size: int, layer=None, expert_id=None) -> dict:
        row = {
            "tensor": name,
            "kind": kind,
            "layer": layer,
            "file": filename,
            "file_bytes": files[filename]["file_bytes"],
            "header_tensor_table_sha256": files[filename]["tensor_table_sha256"],
            "start": start,
            "end": start + size - 1,
            "bytes": size,
            "dtype": "U8",
            "shape": [size],
            "range_key": f"{filename}:{start}-{start + size - 1}",
        }
        if expert_id is not None:
            row["expert_id"] = expert_id
        return row

    embed_file = source["boundary_shards"]["embed.weight"]["file"]
    final_file = source["boundary_shards"]["norm.weight"]["file"]
    boundary = {
        "embedding": [entry("embed.weight", "boundary", embed_file, 96, 4)],
        "final": [
            entry("norm.weight", "boundary", final_file, 96, 4),
            entry("head.weight", "boundary", final_file, 100, 4),
        ],
    }
    layers = {}
    for layer in source["selected_layers"]:
        filename = source["layer_shards"][str(layer)]["file"]
        layers[str(layer)] = {
            "non_expert": [entry(f"layers.{layer}.attn.weight", "non_expert", filename, 96, 4, layer)],
            "router": [entry(f"layers.{layer}.ffn.gate.weight", "router", filename, 100, 4, layer)],
            "shared": [entry(f"layers.{layer}.ffn.shared_experts.w1.weight", "shared", filename, 104, 4, layer)],
            "experts": {
                str(expert_id): [entry(
                    f"layers.{layer}.ffn.experts.{expert_id}.w1.weight",
                    "routed_expert",
                    filename,
                    200 + expert_id * 4,
                    4,
                    layer,
                    expert_id,
                )]
                for expert_id in range(online.EXPERT_COUNT)
            },
        }
    catalog = {
        "format": online.CATALOG_FORMAT,
        "repo": rp.REPO,
        "revision": rp.REVISION,
        "selected_layers": source["selected_layers"],
        "top_k": 6,
        "expert_id_range": [0, 255],
        "download_authorized": False,
        "index": {**source["index"], "authoritative": True},
        "headers": {
            "set_sha256": online._header_set_identity(files),
            "integrity": "tofu_fixed_revision_not_authoritative",
            "authoritative": False,
            "files": files,
        },
        "boundary": boundary,
        "layers": layers,
    }
    entries = list(online._iter_catalog_entries({**catalog, "summary": {}}))
    catalog["summary"] = {
        "range_count": len(entries),
        "range_bytes": sum(row["bytes"] for row in entries),
        "prerequisite_range_count": len(online._flatten_prerequisites(catalog)),
        "route_policy": "current token/current layer exact top-6; no guessed expert",
    }
    catalog["integrity_policy"] = {
        "first_observed_range_hash": "TOFU/non-authoritative",
        "formal_reproduction": online.AUTHORITATIVE_LOCK_FORMAT,
        "tofu_must_not_be_promoted_without_external_lock": True,
    }
    online.validate_catalog(catalog)
    return catalog


class OnlineRangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = make_catalog()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        totals = {
            filename: row["file_bytes"]
            for filename, row in self.catalog["headers"]["files"].items()
        }
        self.transport = FakeHttpsTransport(totals)

    def cache(self, name: str, *, budget: int = 1 << 20, hashes=None, require=False) -> online.RangeCache:
        return online.RangeCache(
            Path(self.tmp.name) / name,
            endpoint="https://fixture.invalid",
            transport=self.transport,
            allow_fetch=True,
            download_budget_bytes=budget,
            cache_budget_bytes=budget,
            authoritative_hashes=hashes,
            require_authoritative=require,
            chunk_bytes=2,
        )

    def test_route_first_rejects_pre_route_wrong_layer_and_ids(self) -> None:
        session = online.RouteFirstSession(self.catalog, self.cache("state"))
        with self.assertRaisesRegex(rp.ContractError, "路由前拒绝"):
            session.fetch_routed(0, "tok")
        session.prepare_embedding()
        with self.assertRaisesRegex(rp.ContractError, "错误 layer"):
            session.prepare_layer(1, "tok")
        session.prepare_layer(0, "tok")
        before = len(self.transport.requests)
        with self.assertRaisesRegex(rp.ContractError, "路由前拒绝"):
            session.fetch_routed(0, "tok")
        self.assertEqual(len(self.transport.requests), before)
        for ids in ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 4], [0, 1, 2, 3, 4, 256]):
            with self.assertRaises(rp.ContractError):
                session.submit_top6(0, "tok", ids)
        with self.assertRaisesRegex(rp.ContractError, "layer/token"):
            session.submit_top6(0, "other", [0, 1, 2, 3, 4, 5])
        session.submit_top6(0, "tok", [9, 3, 7, 1, 5, 11])
        routed = session.fetch_routed(0, "tok")
        self.assertEqual(routed.expert_ids, (9, 3, 7, 1, 5, 11))
        self.assertEqual(set(routed.experts), {1, 3, 5, 7, 9, 11})
        starts = {start for filename, start, end in self.transport.requests if filename.endswith("00002-of-00048.safetensors")}
        self.assertIn(104, starts)  # shared 只在 route 后出现
        self.assertNotIn(200, starts)  # 未选择 E0，绝不猜 0..5

    def test_duplicate_concurrency_and_cache_hit_only_one_get(self) -> None:
        cache = self.cache("concurrent")
        entry = self.catalog["layers"]["0"]["non_expert"][0]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: cache.fetch(entry), range(8)))
        self.assertEqual(len(self.transport.requests), 1)
        self.assertEqual(sum(not result.cache_hit for result in results), 1)
        request_count = len(self.transport.requests)
        hit = cache.fetch(entry)
        self.assertTrue(hit.cache_hit)
        self.assertEqual(len(self.transport.requests), request_count)

    def test_interrupted_part_resumes_from_exact_offset(self) -> None:
        cache = self.cache("resume")
        entry = self.catalog["layers"]["0"]["router"][0]
        first_key = (entry["file"], entry["start"], entry["end"])
        self.transport.interrupt_once[first_key] = 2
        with self.assertRaisesRegex(ConnectionError, "interrupted"):
            cache.fetch(entry)
        parts = list((Path(self.tmp.name) / "resume").glob("*.bin.part"))
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].stat().st_size, 2)
        result = cache.fetch(entry)
        self.assertEqual(self.transport.requests[-1], (entry["file"], entry["start"] + 2, entry["end"]))
        self.assertFalse(result.proof["authoritative"])
        self.assertEqual(result.proof["hash_authority"], "tofu")
        self.assertFalse(parts[0].exists())

    def test_budget_rejected_before_request_or_part(self) -> None:
        entry = self.catalog["layers"]["0"]["shared"][0]
        cache = self.cache("budget", budget=entry["bytes"] - 1)
        with self.assertRaisesRegex(rp.ContractError, "budget 超限"):
            cache.fetch(entry)
        self.assertEqual(self.transport.requests, [])
        self.assertEqual(list((Path(self.tmp.name) / "budget").glob("*.part")), [])

    def test_tofu_upgrades_only_with_external_authoritative_lock(self) -> None:
        entry = self.catalog["boundary"]["embedding"][0]
        cache = self.cache("tofu")
        tofu = cache.fetch(entry)
        self.assertFalse(tofu.proof["authoritative"])
        wanted = rp.sha256_file(tofu.path)
        count = len(self.transport.requests)
        locked = self.cache("tofu", hashes={entry["range_key"]: wanted}, require=True).fetch(entry)
        self.assertTrue(locked.cache_hit)
        self.assertTrue(locked.proof["authoritative"])
        self.assertEqual(locked.proof["hash_authority"], "official_lock")
        self.assertEqual(len(self.transport.requests), count)

    def test_strict_https_206_content_range_and_path_boundary(self) -> None:
        entry = self.catalog["boundary"]["final"][0]
        self.transport.status = 200
        with self.assertRaisesRegex(rp.ContractError, "严格 Range"):
            self.cache("status").fetch(entry)
        self.transport.status = 206
        self.transport.bad_content_range = True
        with self.assertRaisesRegex(rp.ContractError, "严格 Range"):
            self.cache("range").fetch(entry)
        self.transport.bad_content_range = False
        self.transport.final_scheme = "http"
        with self.assertRaisesRegex(rp.ContractError, "HTTPS"):
            self.cache("https").fetch(entry)
        bad = dict(entry)
        bad["file"] = "../escape"
        bad["range_key"] = f"../escape:{bad['start']}-{bad['end']}"
        count = len(self.transport.requests)
        with self.assertRaisesRegex(rp.ContractError, "路径|basename"):
            self.cache("path").fetch(bad)
        self.assertEqual(len(self.transport.requests), count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
