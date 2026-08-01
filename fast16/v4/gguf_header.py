"""Read GGUF metadata and tensor ranges from a downloaded file prefix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

from .donor import DonorPlan


class HeaderTensor(NamedTuple):
    name: str
    tensor_type: GGMLQuantizationType
    shape: tuple[int, ...]
    n_elements: int
    n_bytes: int
    data_offset: int
    field: Any


class GGUFHeaderReader(GGUFReader):
    """GGUFReader variant that never maps tensor payloads.

    A prefix containing the complete metadata and tensor directory is enough,
    even when the tensor offsets point beyond the downloaded prefix.
    """

    tensors: list[HeaderTensor]

    def _build_tensors(self, start_offs: int, fields: list[Any]) -> None:
        tensors: list[HeaderTensor] = []
        names: set[str] = set()
        for field in fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            name = bytes(name_data).decode("utf-8")
            if name in names:
                raise ValueError(f"GGUF 包含重复张量: {name}")
            names.add(name)
            tensor_type = GGMLQuantizationType(raw_dtype[0])
            shape = tuple(int(item) for item in dims.tolist())
            n_elements = int(np.prod(dims))
            block_size, type_size = GGML_QUANT_SIZES[tensor_type]
            if n_elements % block_size:
                raise ValueError(f"张量 {name} 的元素数与量化块不对齐")
            tensors.append(
                HeaderTensor(
                    name=name,
                    tensor_type=tensor_type,
                    shape=shape,
                    n_elements=n_elements,
                    n_bytes=n_elements * type_size // block_size,
                    data_offset=int(start_offs + offset_tensor[0]),
                    field=field,
                )
            )
        self.tensors = tensors

    def metadata(self, key: str) -> Any:
        field = self.get_field(key)
        return None if field is None else field.contents()

    def tensor_map(self) -> dict[str, HeaderTensor]:
        return {tensor.name: tensor for tensor in self.tensors}

    def summary(self) -> dict[str, Any]:
        tensor_bytes = sum(tensor.n_bytes for tensor in self.tensors)
        return {
            "architecture": self.metadata("general.architecture"),
            "name": self.metadata("general.name"),
            "tensor_count": len(self.tensors),
            "tensor_bytes": tensor_bytes,
            "data_offset": self.data_offset,
            "last_tensor_end": max(
                (tensor.data_offset + tensor.n_bytes for tensor in self.tensors),
                default=self.data_offset,
            ),
        }


@dataclass(frozen=True)
class PlanValidation:
    required_sources: int
    found_sources: int
    missing_sources: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_sources


def validate_plan_sources(plan: DonorPlan, reader: GGUFHeaderReader) -> PlanValidation:
    available = set(reader.tensor_map())
    sources = {
        transfer.source
        for transfer in plan.global_transfers
        if transfer.source is not None
    }
    for layer in plan.layers:
        sources.update(
            transfer.source
            for transfer in layer.transfers
            if transfer.source is not None
        )
    missing = tuple(sorted(source for source in sources if source not in available))
    return PlanValidation(
        required_sources=len(sources),
        found_sources=len(sources) - len(missing),
        missing_sources=missing,
    )


def inspect_header(path: str | Path) -> GGUFHeaderReader:
    return GGUFHeaderReader(Path(path), "r")

