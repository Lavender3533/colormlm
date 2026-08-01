from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fast16.research.polaris_meridian_v1.l42_real_reference.l42_reference import (
    _FP8_LUT,
    _FP8_VALIDATION_CACHE,
    _FP8_VALIDATION_LOCK,
    _InlineForward,
)


def _helper(
    root: Path,
    packed: np.ndarray,
    scales: np.ndarray,
) -> _InlineForward:
    weight_path = root / "weight.bin"
    scale_path = root / "scale.bin"
    weight_path.write_bytes(np.asarray(packed, dtype=np.uint8).tobytes())
    scale_path.write_bytes(np.asarray(scales, dtype=np.uint8).tobytes())
    bundle = SimpleNamespace(
        entries={
            "x.weight": {"path": str(weight_path)},
            "x.scale": {"path": str(scale_path)},
        },
        specs={
            "x.weight": ("F8_E4M3", tuple(packed.shape)),
            "x.scale": ("F8_E8M0", tuple(scales.shape)),
        },
    )
    return _InlineForward(bundle)


def _old_repeat_reference(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    expanded = np.repeat(
        np.repeat(np.exp2(scales.astype(np.float32) - 127), 128, axis=0),
        128,
        axis=1,
    )
    return _FP8_LUT[packed] * expanded[: packed.shape[0], : packed.shape[1]]


def test_aligned_fp8_block_broadcast_is_bit_exact(tmp_path: Path) -> None:
    packed = (np.arange(256 * 128, dtype=np.uint32) % 126).astype(np.uint8)
    packed = packed.reshape(256, 128)
    scales = np.array([[126], [128]], dtype=np.uint8)
    actual = _helper(tmp_path, packed, scales)._weight_fp8("x")
    expected = _old_repeat_reference(packed, scales)
    assert np.array_equal(actual, expected)


def test_partial_fixture_keeps_repeat_fallback_semantics(tmp_path: Path) -> None:
    packed = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.uint8)
    scales = np.array([[127]], dtype=np.uint8)
    actual = _helper(tmp_path, packed, scales)._weight_fp8("x")
    expected = _old_repeat_reference(packed, scales)
    assert np.array_equal(actual, expected)


def test_fp8_validation_cache_rechecks_changed_file(tmp_path: Path) -> None:
    packed = np.zeros((128, 128), dtype=np.uint8)
    scales = np.full((1, 1), 127, dtype=np.uint8)
    helper = _helper(tmp_path, packed, scales)
    with _FP8_VALIDATION_LOCK:
        _FP8_VALIDATION_CACHE.clear()
    helper._weight_fp8("x")
    with _FP8_VALIDATION_LOCK:
        assert len(_FP8_VALIDATION_CACHE) == 1
    helper._weight_fp8("x")
    with _FP8_VALIDATION_LOCK:
        assert len(_FP8_VALIDATION_CACHE) == 1

    weight_path = Path(helper.bundle.entries["x.weight"]["path"])
    changed = packed.copy()
    changed[0, 0] = 127
    weight_path.write_bytes(changed.tobytes())
    with pytest.raises(ValueError, match="NaN"):
        helper._weight_fp8("x")
