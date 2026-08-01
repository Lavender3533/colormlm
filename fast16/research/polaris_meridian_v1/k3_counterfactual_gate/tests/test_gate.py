from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from fast16.research.polaris_meridian_v1.k3_counterfactual_gate.counterfactual import (
    CounterfactualBatch,
    fit_counterfactual_gate,
    save_calibration,
)
from fast16.research.polaris_meridian_v1.k3_counterfactual_gate.runtime import (
    DensePortal,
    F16K3Capsule,
    FullDepthK3Bus,
    LinearCapabilityGate,
)


def approved_gate() -> LinearCapabilityGate:
    return LinearCapabilityGate(
        weights=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        bias=0.0,
        threshold=0.0,
        approved=True,
        counterfactual_nll_passed=True,
        leave_one_task_out_passed=True,
        frozen_contract=True,
        source="synthetic-test",
    )


class DoubleCapsule:
    input_width = 2

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * 2.0


def test_alpha_zero_is_identity_and_loads_nothing() -> None:
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    calls = {"portal": 0, "capsule": 0}

    def portal_loader():
        calls["portal"] += 1
        raise AssertionError("alpha=0 不得加载 portal")

    def capsule_loader():
        calls["capsule"] += 1
        raise AssertionError("alpha=0 不得加载胶囊")

    result = FullDepthK3Bus(
        approved_gate(),
        portal_authorized=True,
        portal_loader=portal_loader,
        capsule_loader=capsule_loader,
    ).apply(hidden, alpha=0.0)
    assert result.hidden is hidden
    assert result.exact_bypass
    assert result.decision.reason == "alpha_zero_physical_bypass"
    assert calls == {"portal": 0, "capsule": 0}
    invalid_shape = torch.tensor(1.0)
    assert FullDepthK3Bus(approved_gate()).apply(invalid_shape, alpha=0.0).hidden is invalid_shape


def test_unapproved_or_below_threshold_never_loads_donor() -> None:
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    calls = 0

    def forbidden():
        nonlocal calls
        calls += 1
        raise AssertionError("no-op 不得加载 donor")

    rejected = FullDepthK3Bus(
        LinearCapabilityGate.rejected(4),
        portal_authorized=True,
        portal_loader=forbidden,
        capsule_loader=forbidden,
    ).apply(hidden, alpha=0.1)
    assert rejected.hidden is hidden and rejected.exact_bypass

    below = FullDepthK3Bus(
        LinearCapabilityGate(
            weights=torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            bias=0.0,
            threshold=0.0,
            approved=True,
            counterfactual_nll_passed=True,
            leave_one_task_out_passed=True,
            frozen_contract=True,
        ),
        portal_authorized=True,
        portal_loader=forbidden,
        capsule_loader=forbidden,
    ).apply(hidden, alpha=0.1)
    assert below.hidden is hidden and below.decision.reason == "below_threshold"
    assert calls == 0


def test_missing_portal_fails_closed_before_capsule_load() -> None:
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    calls = 0

    def forbidden():
        nonlocal calls
        calls += 1
        raise AssertionError("portal 未批准时不得加载胶囊")

    result = FullDepthK3Bus(
        approved_gate(), portal_authorized=False, capsule_loader=forbidden
    ).apply(hidden, alpha=0.1)
    assert result.hidden is hidden
    assert result.decision.reason == "full_to_colorlm_portal_not_approved"
    assert calls == 0


def test_selected_path_applies_residual_after_lazy_load() -> None:
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    portal = DensePortal(
        full_to_bus=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        bus_to_full=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
        ),
    )
    calls = {"portal": 0, "capsule": 0}

    def load_portal():
        calls["portal"] += 1
        return portal

    def load_capsule():
        calls["capsule"] += 1
        return DoubleCapsule()

    result = FullDepthK3Bus(
        approved_gate(),
        portal_authorized=True,
        portal_loader=load_portal,
        capsule_loader=load_capsule,
    ).apply(hidden, alpha=0.5)
    torch.testing.assert_close(result.hidden, torch.tensor([[[2.0, 4.0, 3.0, 4.0]]]))
    assert not result.exact_bypass and result.portal_loaded and result.capsule_loaded
    assert calls == {"portal": 1, "capsule": 1}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_f16_capsule_loader_executes_manifest_equation() -> None:
    rng = np.random.default_rng(7)
    arrays = {
        "b_in": rng.normal(0, 0.2, (3, 2)).astype("<f2"),
        "gate": rng.normal(0, 0.2, (4, 3)).astype("<f2"),
        "up": rng.normal(0, 0.2, (4, 3)).astype("<f2"),
        "down": rng.normal(0, 0.2, (3, 4)).astype("<f2"),
        "norm": np.ones((3,), dtype="<f2"),
        "b_out": rng.normal(0, 0.2, (2, 3)).astype("<f2"),
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        specs = {}
        for name, array in arrays.items():
            path = root / f"{name}.f16"
            array.tofile(path)
            specs[name] = {
                "file": path.name,
                "shape": list(array.shape),
                "dtype": "float16-le",
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        (root / "capsule.json").write_text(
            json.dumps(
                {
                    "format": "colorlm-kimi-k3-latent-macro-capsule-v1",
                    "dimensions": {"colorlm": 2, "latent": 3, "intermediate": 4},
                    "rms_norm_eps": 1.0e-5,
                    "runtime_files": specs,
                }
            ),
            encoding="utf-8",
        )
        capsule = F16K3Capsule.from_runtime_dir(root)
        hidden = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
        actual = capsule(hidden)
        weights = {
            name: torch.from_numpy(np.asarray(array, dtype=np.float32))
            for name, array in arrays.items()
        }
        latent = torch.nn.functional.linear(hidden, weights["b_in"])
        gate_value = torch.nn.functional.linear(latent, weights["gate"])
        up_value = torch.nn.functional.linear(latent, weights["up"])
        activation = (
            4.0 * torch.tanh(gate_value / 4.0) * torch.sigmoid(gate_value)
            * (25.0 * torch.tanh(up_value / 25.0))
        )
        latent_output = torch.nn.functional.linear(activation, weights["down"])
        normalized = latent_output * torch.rsqrt(
            latent_output.square().mean(dim=-1, keepdim=True) + 1.0e-5
        )
        expected = torch.nn.functional.linear(normalized * weights["norm"], weights["b_out"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == hidden.shape
        assert bool(torch.isfinite(actual).all().item())
        assert capsule.source.endswith("capsule.json")


def synthetic_batch() -> CounterfactualBatch:
    values = np.asarray([-2.0, -1.0, 1.0, 2.0] * 3, dtype=np.float32)
    hidden = np.stack((values, np.ones_like(values)), axis=1)
    no_op = np.zeros((12, 2), dtype=np.float32)
    donor = np.zeros((12, 2), dtype=np.float32)
    donor[:, 0] = np.where(values > 0, 2.0, -0.2)
    return CounterfactualBatch(
        hidden=hidden,
        no_op_logits=no_op,
        donor_logits=donor,
        target_ids=np.zeros(12, dtype=np.int64),
        task_ids=tuple(task for task in ("a", "b", "c") for _ in range(4)),
    )


def test_counterfactual_nll_and_loto_can_approve_and_reload_gate() -> None:
    result = fit_counterfactual_gate(
        synthetic_batch(), frozen_contract_sha256="a" * 64, ridge=0.1
    )
    assert result.approved
    assert result.report["evidence"]["counterfactual_nll_passed"]
    assert result.report["evidence"]["leave_one_task_out_passed"]
    assert all(row["passed"] for row in result.report["leave_one_task_out"])
    with tempfile.TemporaryDirectory() as directory:
        manifest, _ = save_calibration(result, Path(directory))
        gate = LinearCapabilityGate.from_manifest(manifest)
        assert gate.eligible
        assert gate.decide(torch.tensor([[[2.0, 1.0]]])).selected
        assert not gate.decide(torch.tensor([[[-2.0, 1.0]]])).selected


def test_missing_frozen_contract_prevents_approval() -> None:
    result = fit_counterfactual_gate(synthetic_batch(), frozen_contract_sha256=None, ridge=0.1)
    assert not result.approved
    assert not result.report["evidence"]["frozen_contract"]


def test_package_text_files_are_utf8_without_bom() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".json"}:
            payload = path.read_bytes()
            assert not payload.startswith(b"\xef\xbb\xbf"), path
            payload.decode("utf-8", errors="strict")
