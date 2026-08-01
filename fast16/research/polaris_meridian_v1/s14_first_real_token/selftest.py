"""S14 首 token executor 的零网络合同自检。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from fast16.research.polaris_meridian_v1.s14_first_real_token.executor import (
    COMPRESS_RATIOS,
    DEFAULT_ASSET_ROOT,
    HASH_ROUTE_ANCHORS,
    REGISTERED_LAYERS,
    TensorStore,
    _initial_state,
    _read_json,
)
from fast16.research.polaris_meridian_v1.s14_range_pack import online_range


HERE = Path(__file__).resolve().parent


def main() -> int:
    assert REGISTERED_LAYERS == (0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42)
    assert [layer for layer, ratio in COMPRESS_RATIOS.items() if ratio == 128] == [7, 15, 23, 31, 41]
    assert HASH_ROUTE_ANCHORS[0] == [254, 222, 245, 200, 53, 35]
    assert HASH_ROUTE_ANCHORS[1] == [163, 137, 158, 97, 184, 8]

    root = DEFAULT_ASSET_ROOT.resolve()
    catalog = _read_json(root / "route_first_catalog.json")
    online_range.validate_catalog(catalog)
    assert tuple(catalog["selected_layers"]) == REGISTERED_LAYERS
    cache = online_range.RangeCache(root / "range_cache", allow_fetch=False)
    session = online_range.RouteFirstSession(catalog, cache)
    embedding = session.prepare_embedding_row(0)
    state = _initial_state(embedding)
    assert state.dtype == torch.bfloat16 and tuple(state.shape) == (1, 1, 4, 4096)

    l0 = session.prepare_layer(0, 0)
    store = TensorStore(root / "range_cache")
    store.add_ranges((*l0.non_expert, *l0.router))
    assert len(store.sources) == 23
    tid = store.source("layers.0.ffn.gate.tid2eid")
    assert tid.entry["dtype"] == "I64" and tid.entry["shape"] == [129280, 6]

    two_token = _read_json(HERE / "TWO_TOKEN_REAL_REPORT.json")
    assert two_token["status"] == "complete"
    assert two_token["committed_tokens"] == [
        {"position": 0, "input_token_id": 0, "output_token_id": 108967},
        {"position": 1, "input_token_id": 108967, "output_token_id": 53},
    ]
    assert two_token["position1"]["logits_f32_le_sha256"] == (
        "46b95489427932a0d5acfacd5ee6bc9ceac495df3daed5a6a58681a0d95a141d"
    )
    assert two_token["position1"]["runtime_state_contract"] == {
        "active_window_rows": 2,
        "ratio4_main_written_row": 5,
        "ratio4_indexer_written_row": 5,
        "ratio128_main_written_row": 1,
        "compressed_blocks_emitted": 0,
    }

    for path in HERE.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".json"}:
            payload = path.read_bytes()
            assert not payload.startswith(b"\xef\xbb\xbf"), path
            payload.decode("utf-8", errors="strict")
    print(
        json.dumps(
            {
                "status": "pass",
                "network_accessed": False,
                "embedding_sha256": embedding.proof["observed_sha256"],
                "l0_base_router_files": len(store.sources),
                "tid2eid_physical_dtype": tid.entry["dtype"],
                "position1_token_id": two_token["position1"]["output_token_id"],
                "position1_logits_sha256": two_token["position1"]["logits_f32_le_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
