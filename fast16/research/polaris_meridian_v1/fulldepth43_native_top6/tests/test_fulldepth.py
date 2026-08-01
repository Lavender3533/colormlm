from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.catalog import (
    CatalogError,
    read_json,
    validate_catalog,
)
from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor import (
    DecoderState,
    ExecutionConfig,
    FullDepthError,
    FullDepthNativeLayerReference,
    execute,
)
from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.preflight import run_preflight
from fast16.research.polaris_meridian_v1.fulldepth43_native_top6.profile import (
    FULLDEPTH43_NATIVE_TOP6,
    ProfileError,
)
from fast16.research.polaris_meridian_v1.s14_first_real_token.executor import TensorStore


ASSET_ROOT = Path("D:/models/Polaris-S14")
FULL_CATALOG = ASSET_ROOT / "fulldepth43_native_top6_catalog.json"
S14_CATALOG = ASSET_ROOT / "route_first_catalog.json"


class FullDepthContractTests(unittest.TestCase):
    def test_profile_is_exact_43_layer_native_top6(self) -> None:
        profile = FULLDEPTH43_NATIVE_TOP6
        profile.validate()
        self.assertEqual(profile.layers, tuple(range(43)))
        self.assertEqual(profile.top_k, 6)
        self.assertEqual(profile.ratio_for(42), 4)
        with self.assertRaises(ProfileError):
            replace(profile, layers=tuple(range(42))).validate()
        with self.assertRaises(ProfileError):
            replace(profile, top_k=1).validate()

    def test_s14_catalog_cannot_masquerade_as_fulldepth(self) -> None:
        if not S14_CATALOG.is_file():
            self.skipTest("external S14 catalog absent")
        with self.assertRaises(CatalogError):
            validate_catalog(read_json(S14_CATALOG))

    def test_native_layer_core_accepts_l3_only_under_full_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TensorStore(Path(directory))
            kernel = FullDepthNativeLayerReference(3, store)
            self.assertEqual(kernel.layer, 3)
            with self.assertRaises(Exception):
                FullDepthNativeLayerReference(43, store)

    def test_decoder_refuses_partial_layer_state_and_fake_token(self) -> None:
        decoder = DecoderState(position=1, input_token_id=7, layer_states={})
        with self.assertRaises(FullDepthError):
            decoder.previous_for(FULLDEPTH43_NATIVE_TOP6)
        decoder = DecoderState()
        with self.assertRaises(FullDepthError):
            decoder.commit(
                output_token_id=1,
                next_states={},
                profile=FULLDEPTH43_NATIVE_TOP6,
            )

    def test_download_gate_requires_explicit_matching_budget(self) -> None:
        ExecutionConfig().validate()
        with self.assertRaises(FullDepthError):
            ExecutionConfig(allow_fetch=True, download_budget_bytes=0).validate()
        with self.assertRaises(FullDepthError):
            ExecutionConfig(allow_fetch=False, download_budget_bytes=1).validate()

    def test_real_local_catalog_preflight_is_machine_readable(self) -> None:
        if not FULL_CATALOG.is_file():
            self.skipTest("external FullDepth catalog absent")
        catalog = read_json(FULL_CATALOG)
        validate_catalog(catalog)
        report = run_preflight(asset_root=ASSET_ROOT, catalog=catalog)
        self.assertEqual(report["profile"]["layers"], list(range(43)))
        self.assertEqual(report["catalog"]["range_count"], 67_612)
        self.assertFalse(report["native_token_executed"])
        self.assertFalse(report["fake_token_emitted"])
        self.assertEqual(
            report["cold_execution_upper_bound"]["total_bytes"],
            report["static_prerequisites"]["missing_bytes"]
            + report["dynamic_routed_experts"]["native_top6_cold_bytes_per_token"],
        )

    def test_offline_execute_fails_before_forward_when_static_pages_missing(self) -> None:
        if not FULL_CATALOG.is_file():
            self.skipTest("external FullDepth catalog absent")
        with tempfile.TemporaryDirectory() as directory:
            empty_asset_root = Path(directory) / "empty-assets"
            report = execute(
                ExecutionConfig(
                    asset_root=empty_asset_root,
                    catalog_path=FULL_CATALOG,
                    report_path=Path(directory) / "report.json",
                )
            )
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["error"]["stage"], "preflight")
        self.assertFalse(report["native_token_executed"])
        self.assertFalse(report["fake_token_emitted"])
        self.assertEqual(report["committed_tokens"], [])


if __name__ == "__main__":
    unittest.main()
