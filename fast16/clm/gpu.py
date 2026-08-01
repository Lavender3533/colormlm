"""DirectML GPU export and execution for the CLM neural core."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import ZeroTrainModel


class ExportableCore(nn.Module):
    def __init__(self, model: ZeroTrainModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_context: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_with_memory_mask(input_ids, memory_context, memory_mask)


def export_directml_graph(
    clm_path: str | Path,
    output_path: str | Path,
    *,
    example_sequence: int = 16,
) -> Path:
    """Export the exact CLM neural forward graph to ONNX for DirectML."""

    import onnx

    model = ZeroTrainModel.from_clm(clm_path)
    wrapper = ExportableCore(model).eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_ids = torch.zeros((1, example_sequence), dtype=torch.long)
    memory_context = torch.zeros(model.token_embed.embedding_dim, dtype=torch.float32)
    memory_mask = torch.zeros((1, example_sequence, 1), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (input_ids, memory_context, memory_mask),
        output_path,
        input_names=["input_ids", "memory_context", "memory_mask"],
        output_names=["logits", "temperature"],
        dynamic_axes={
            "input_ids": {1: "sequence"},
            "memory_mask": {1: "sequence"},
            "logits": {1: "sequence"},
            "temperature": {1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output_path))
    return output_path


class DirectMLCore:
    """ONNX Runtime DirectML implementation of the CLM core backend."""

    provider = "DmlExecutionProvider"

    def __init__(self, graph_path: str | Path):
        import onnxruntime as ort

        available = ort.get_available_providers()
        if self.provider not in available:
            raise RuntimeError(f"DirectML provider is unavailable; providers={available}")
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(graph_path),
            sess_options=options,
            providers=[(self.provider, {"device_id": 0}), "CPUExecutionProvider"],
        )
        self.graph_path = Path(graph_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        prompt_length: int,
        memory_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = int(input_ids.shape[1])
        if memory_context is None:
            width = int(self.session.get_inputs()[1].shape[0])
            context = np.zeros(width, dtype=np.float32)
        else:
            context = memory_context.detach().cpu().float().numpy()
        memory_mask = np.zeros((1, sequence, 1), dtype=np.float32)
        memory_mask[:, prompt_length:, :] = 1.0
        logits, temperature = self.session.run(
            None,
            {
                "input_ids": input_ids.detach().cpu().numpy().astype(np.int64, copy=False),
                "memory_context": context,
                "memory_mask": memory_mask,
            },
        )
        return torch.from_numpy(logits), torch.from_numpy(temperature)

    def info(self) -> dict[str, object]:
        return {
            "backend": "DirectML",
            "provider": self.provider,
            "active_providers": self.session.get_providers(),
            "graph": str(self.graph_path),
            "inputs": [item.name for item in self.session.get_inputs()],
            "outputs": [item.name for item in self.session.get_outputs()],
        }
