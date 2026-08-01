"""ONNX export and DirectML execution for the ColorLM v4 state cell."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .cell import ColorStateCell


def export_state_cell(
    output_path: str | Path,
    *,
    seed: int = 17,
    hidden_size: int = 64,
    state_rank: int = 16,
    landmark_slots: int = 8,
    experts: int = 8,
    active_experts: int = 2,
    expert_size: int = 32,
) -> Path:
    import onnx

    torch.manual_seed(seed)
    model = ColorStateCell(
        hidden_size=hidden_size,
        state_rank=state_rank,
        landmark_slots=landmark_slots,
        experts=experts,
        active_experts=active_experts,
        expert_size=expert_size,
    ).eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hidden = torch.zeros((1, hidden_size), dtype=torch.float32)
    delta_state = torch.zeros((1, state_rank, state_rank), dtype=torch.float32)
    kernel_state = torch.zeros((1, landmark_slots, hidden_size), dtype=torch.float32)
    kernel_norm = torch.zeros((1, landmark_slots), dtype=torch.float32)
    torch.onnx.export(
        model,
        (hidden, delta_state, kernel_state, kernel_norm),
        output_path,
        input_names=["hidden", "delta_state", "kernel_state", "kernel_norm"],
        output_names=[
            "next_hidden",
            "next_delta_state",
            "next_kernel_state",
            "next_kernel_norm",
            "expert_ids",
        ],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output_path))
    return output_path


class DirectMLStateCell:
    provider = "DmlExecutionProvider"

    def __init__(self, graph_path: str | Path, *, profile: bool = False):
        import onnxruntime as ort

        if self.provider not in ort.get_available_providers():
            raise RuntimeError("当前 ONNX Runtime 不包含 DirectML provider")
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_profiling = profile
        self.session = ort.InferenceSession(
            str(graph_path),
            sess_options=options,
            providers=[(self.provider, {"device_id": 0}), "CPUExecutionProvider"],
        )

    def step(
        self,
        hidden: np.ndarray,
        delta_state: np.ndarray,
        kernel_state: np.ndarray,
        kernel_norm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        outputs = self.session.run(
            None,
            {
                "hidden": np.asarray(hidden, dtype=np.float32),
                "delta_state": np.asarray(delta_state, dtype=np.float32),
                "kernel_state": np.asarray(kernel_state, dtype=np.float32),
                "kernel_norm": np.asarray(kernel_norm, dtype=np.float32),
            },
        )
        return tuple(outputs)  # type: ignore[return-value]

    def finish_profile(self) -> dict[str, int]:
        profile_path = Path(self.session.end_profiling())
        events = json.loads(profile_path.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for event in events:
            provider = event.get("args", {}).get("provider")
            if provider:
                counts[provider] = counts.get(provider, 0) + 1
        profile_path.unlink(missing_ok=True)
        return counts
