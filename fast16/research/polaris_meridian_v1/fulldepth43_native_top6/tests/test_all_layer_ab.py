from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor import (
    FullDepthError,
)
from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.run_vulkan_all_layer_ab import (
    run_all_layer_ab,
)


def _report(*, token: int, seconds: float, writeback: bool) -> dict:
    result = {
        "status": "complete",
        "committed_tokens": [{"output_token_id": token}],
        "execution_seconds": seconds,
        "vulkan_writeback_layers": list(range(43)) if writeback else [],
        "vulkan_writeback_fallbacks": [],
        "tokens": [{"layers": []}],
    }
    if writeback:
        result["tokens"][0]["layers"] = [
            {
                "layer": layer,
                "vulkan_writeback": {
                    "comparison": {"exact_bf16_equal": True},
                },
            }
            for layer in range(43)
        ]
    return result


class AllLayerAbTests(unittest.TestCase):
    def test_requires_all_layer_parity_then_runs_unverified_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.exe"
            worker.write_bytes(b"fixture")
            reports = iter(
                (
                    _report(token=17, seconds=3.0, writeback=True),
                    _report(token=17, seconds=2.0, writeback=False),
                    _report(token=17, seconds=1.0, writeback=True),
                )
            )
            configs = []

            def fake_execute(config):
                configs.append(config)
                return next(reports)

            with patch(
                "fast16.research.polaris_meridian_v1.fulldepth43_native_top6."
                "run_vulkan_all_layer_ab.execute",
                side_effect=fake_execute,
            ):
                result = run_all_layer_ab(
                    worker=worker,
                    output_root=root / "ab",
                    asset_root=root,
                    catalog_path=root / "catalog.json",
                )

        self.assertEqual(result["exact_bf16_layers"], 43)
        self.assertEqual(result["timing"]["speedup_vs_cpu"], 2.0)
        self.assertTrue(result["passed_speed_smoke"])
        self.assertTrue(configs[0].vulkan_writeback_verify_cpu)
        self.assertFalse(configs[2].vulkan_writeback_verify_cpu)
        self.assertTrue(configs[0].vulkan_writeback_all_layers)
        self.assertFalse(configs[2].vulkan_writeback_cpu_fallback)

    def test_rejects_token_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.exe"
            worker.write_bytes(b"fixture")
            reports = iter(
                (
                    _report(token=17, seconds=3.0, writeback=True),
                    _report(token=17, seconds=2.0, writeback=False),
                    _report(token=18, seconds=1.0, writeback=True),
                )
            )
            with patch(
                "fast16.research.polaris_meridian_v1.fulldepth43_native_top6."
                "run_vulkan_all_layer_ab.execute",
                side_effect=lambda _config: next(reports),
            ):
                with self.assertRaises(FullDepthError):
                    run_all_layer_ab(
                        worker=worker,
                        output_root=root / "ab",
                        asset_root=root,
                        catalog_path=root / "catalog.json",
                    )


if __name__ == "__main__":
    unittest.main()
