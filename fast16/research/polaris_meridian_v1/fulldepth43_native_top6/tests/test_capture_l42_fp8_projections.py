from __future__ import annotations

import hashlib

import numpy as np

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.capture_l42_fp8_projections import (
    FROZEN_PROJECTION_SHA256,
    PROJECTION_ORDER,
    _ProjectionCapture,
    _f32_payload,
)


def test_frozen_projection_contract_covers_all_standard_l42_paths() -> None:
    assert PROJECTION_ORDER == (
        "layers.42.attn.wq_a",
        "layers.42.attn.wkv",
        "layers.42.attn.wq_b",
        "layers.42.attn.indexer.wq_b",
        "layers.42.attn.wo_b",
    )
    assert set(PROJECTION_ORDER) == set(FROZEN_PROJECTION_SHA256)
    assert all(len(value) == 2 and all(len(digest) == 64 for digest in value) for value in FROZEN_PROJECTION_SHA256.values())


def test_f32_payload_is_canonical_little_endian() -> None:
    source = np.asarray([[1.0, -2.5]], dtype=">f4")
    array, payload, digest = _f32_payload(source)
    assert array.dtype == np.dtype("<f4")
    assert array.flags.c_contiguous
    assert payload == np.asarray([[1.0, -2.5]], dtype="<f4").tobytes()
    assert digest == hashlib.sha256(payload).hexdigest()


def test_projection_file_stem_is_bounded_to_l42_attention() -> None:
    assert _ProjectionCapture._file_stem("layers.42.attn.indexer.wq_b") == "indexer-wq_b"
