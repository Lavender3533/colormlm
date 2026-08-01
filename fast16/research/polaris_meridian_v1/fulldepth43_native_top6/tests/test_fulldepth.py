from __future__ import annotations

import inspect
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

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
from fast16.research.polaris_meridian_v1.s14_first_real_token import executor as s14


ASSET_ROOT = Path("D:/models/Polaris-S14")
FULL_CATALOG = ASSET_ROOT / "fulldepth43_native_top6_catalog.json"
S14_CATALOG = ASSET_ROOT / "route_first_catalog.json"
PREVIEW_FORCED_PREFILL = Path(__file__).resolve().parents[1] / "first_preview_forced_prefill.json"


def full_states(position: int) -> dict[int, s14.LayerRuntimeState]:
    return {
        layer: s14.LayerRuntimeState(
            layer=layer,
            position=position,
            window_kv=torch.full(
                (1, s14.WINDOW_SIZE, 512), float(position), dtype=torch.bfloat16
            ),
            compressor=None,
        )
        for layer in FULLDEPTH43_NATIVE_TOP6.layers
    }


class FullDepthContractTests(unittest.TestCase):
    def test_attention_reads_injected_full_depth_compression_map(self) -> None:
        source = inspect.getsource(s14.NativeLayerReference._attention)
        self.assertIn("ratio = self.compress_ratios[self.layer]", source)
        self.assertNotIn("ratio = COMPRESS_RATIOS[self.layer]", source)

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
        ExecutionConfig(token_count=5).validate()
        with self.assertRaises(FullDepthError):
            ExecutionConfig(allow_fetch=True, download_budget_bytes=0).validate()
        with self.assertRaises(FullDepthError):
            ExecutionConfig(allow_fetch=False, download_budget_bytes=1).validate()
        with self.assertRaises(FullDepthError):
            ExecutionConfig(vulkan_bridge_layer=43).validate()
        with self.assertRaises(FullDepthError):
            ExecutionConfig(
                token_count=2,
                vulkan_bridge_capture=Path("capture"),
            ).validate()
        with self.assertRaises(FullDepthError):
            ExecutionConfig(vulkan_writeback_worker=Path("missing.exe")).validate()
        with self.assertRaises(FullDepthError):
            ExecutionConfig(vulkan_writeback_all_layers=True).validate()
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "worker.exe"
            worker.write_bytes(b"fixture")
            ExecutionConfig(
                token_count=2,
                vulkan_bridge_capture=Path(directory) / "captures",
                vulkan_writeback_worker=worker,
                vulkan_writeback_all_layers=True,
                vulkan_writeback_verify_cpu=False,
            ).validate()

    def test_first_preview_forced_prefill_runs_five_inputs_before_argmax(self) -> None:
        queue = s14._load_forced_prefill(PREVIEW_FORCED_PREFILL)
        self.assertEqual(queue.token_ids, (0, 128803, 30594, 128804, 128821))
        decoder = DecoderState(
            input_token_id=queue.current_token_id,
            forced_queue=queue,
        )
        argmax_outputs = (101, 102, 103, 104, 105)

        for position, (forced_token, argmax_token) in enumerate(
            zip(queue.token_ids, argmax_outputs, strict=True)
        ):
            self.assertEqual(decoder.position, position)
            self.assertEqual(decoder.input_token_id, forced_token)
            previous = decoder.previous_for(FULLDEPTH43_NATIVE_TOP6)
            self.assertEqual(set(previous), set() if position == 0 else set(range(43)))
            decoder.commit(
                output_token_id=argmax_token,
                next_states=full_states(position),
                profile=FULLDEPTH43_NATIVE_TOP6,
            )
            self.assertEqual(set(decoder.layer_states), set(range(43)))

        self.assertIsNotNone(decoder.forced_queue)
        self.assertFalse(decoder.forced_queue.active)
        self.assertEqual(decoder.forced_queue.cursor, 5)
        self.assertEqual(decoder.input_token_id, argmax_outputs[-1])
        self.assertEqual(
            [item["input_token_id"] for item in decoder.committed_tokens],
            list(queue.token_ids),
        )
        self.assertTrue(
            all(item["input_source"] == "forced_prefill" for item in decoder.committed_tokens)
        )

    def test_exhausted_forced_prefill_switches_to_native_argmax_chain(self) -> None:
        queue = s14._load_forced_prefill(PREVIEW_FORCED_PREFILL)
        decoder = DecoderState(input_token_id=queue.current_token_id, forced_queue=queue)
        for position in range(len(queue.token_ids)):
            decoder.commit(
                output_token_id=900 + position,
                next_states=full_states(position),
                profile=FULLDEPTH43_NATIVE_TOP6,
            )
        self.assertEqual(decoder.input_token_id, 904)

        decoder.commit(
            output_token_id=777,
            next_states=full_states(5),
            profile=FULLDEPTH43_NATIVE_TOP6,
        )
        self.assertEqual(decoder.position, 6)
        self.assertEqual(decoder.input_token_id, 777)
        self.assertEqual(
            decoder.committed_tokens[-1],
            {"position": 5, "input_token_id": 904, "output_token_id": 777},
        )

    def test_forced_prefill_failure_rolls_back_cursor_token_and_layer_state(self) -> None:
        queue = s14._load_forced_prefill(PREVIEW_FORCED_PREFILL)
        decoder = DecoderState(input_token_id=queue.current_token_id, forced_queue=queue)
        decoder.commit(
            output_token_id=999,
            next_states=full_states(0),
            profile=FULLDEPTH43_NATIVE_TOP6,
        )
        before_position = decoder.position
        before_input = decoder.input_token_id
        before_cursor = decoder.forced_queue.cursor
        before_committed = list(decoder.committed_tokens)
        before_window = decoder.layer_states[0].window_kv.clone()

        private_previous = decoder.previous_for(FULLDEPTH43_NATIVE_TOP6)
        private_previous[0].window_kv.fill_(7)
        torch.testing.assert_close(
            decoder.layer_states[0].window_kv, before_window, rtol=0, atol=0
        )
        invalid_states = full_states(1)
        invalid_states[42].position = 0
        with self.assertRaises(FullDepthError):
            decoder.commit(
                output_token_id=1000,
                next_states=invalid_states,
                profile=FULLDEPTH43_NATIVE_TOP6,
            )
        self.assertEqual(decoder.position, before_position)
        self.assertEqual(decoder.input_token_id, before_input)
        self.assertEqual(decoder.forced_queue.cursor, before_cursor)
        self.assertEqual(decoder.committed_tokens, before_committed)
        torch.testing.assert_close(
            decoder.layer_states[0].window_kv, before_window, rtol=0, atol=0
        )

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
