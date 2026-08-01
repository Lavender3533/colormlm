"""Executable invariants for the ColorLM v4 architecture.

This module defines the persistent fast-weight update and the model-size budget.
The full Vulkan graph will use the same equations in ggml.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GIB = 1024**3


@dataclass(frozen=True)
class ColorLMV4Config:
    hidden_size: int = 2048
    layers: int = 40
    macro_cells: int = 10
    state_heads: int = 32
    state_key_dim: int = 128
    state_value_dim: int = 128
    landmark_slots: int = 64
    experts: int = 256
    active_experts: int = 8
    shared_experts: int = 1
    fast_rank: int = 64
    max_model_bytes: int = 15 * GIB

    def validate(self) -> None:
        if self.layers != self.macro_cells * 4:
            raise ValueError("每个宏单元必须恰好包含四个状态层")
        if self.active_experts >= self.experts:
            raise ValueError("激活专家数必须小于专家总数")
        if self.hidden_size % self.state_heads:
            raise ValueError("hidden_size 必须能被 state_heads 整除")
        if self.fast_rank <= 0 or self.fast_rank > self.hidden_size:
            raise ValueError("fast_rank 超出有效范围")


@dataclass(frozen=True)
class ModelBudget:
    expert_bytes: int
    state_core_bytes: int
    embedding_bytes: int
    fast_state_bytes: int
    metadata_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.expert_bytes
            + self.state_core_bytes
            + self.embedding_bytes
            + self.fast_state_bytes
            + self.metadata_bytes
        )

    def validate(self, limit: int) -> None:
        if self.total_bytes > limit:
            raise ValueError(
                f"模型预算超限: {self.total_bytes / GIB:.3f} GiB > {limit / GIB:.3f} GiB"
            )


def colorlm_v4_budget() -> ModelBudget:
    """Return the hard allocation used by the v4 container builder."""

    return ModelBudget(
        expert_bytes=int(11.5 * GIB),
        state_core_bytes=int(1.6 * GIB),
        embedding_bytes=int(0.9 * GIB),
        fast_state_bytes=int(0.5 * GIB),
        metadata_bytes=int(0.2 * GIB),
    )


class FastWeightState:
    """Persistent low-rank state updated by a local delta rule.

    This is model state, not a text index. Every write and read is a dense,
    continuous operation over the same fixed-size matrix.
    """

    def __init__(self, rank: int, dtype: np.dtype = np.float32):
        if rank <= 0:
            raise ValueError("rank 必须为正数")
        self.rank = rank
        self.matrix = np.zeros((rank, rank), dtype=dtype)

    def read(self, key: np.ndarray) -> np.ndarray:
        key = self._normalized(key)
        return self.matrix @ key

    def update(
        self,
        key: np.ndarray,
        value: np.ndarray,
        *,
        learning_rate: float = 1.0,
        retention: float = 1.0,
    ) -> float:
        if not 0.0 <= learning_rate <= 1.0:
            raise ValueError("learning_rate 必须位于 [0, 1]")
        if not 0.0 <= retention <= 1.0:
            raise ValueError("retention 必须位于 [0, 1]")
        key = self._normalized(key)
        value = np.asarray(value, dtype=self.matrix.dtype)
        if value.shape != (self.rank,):
            raise ValueError(f"value 形状必须是 ({self.rank},)")
        error = value - self.matrix @ key
        self.matrix *= retention
        self.matrix += learning_rate * np.outer(error, key)
        return float(np.linalg.norm(error))

    def save(self, path: str) -> None:
        with open(path, "wb") as target:
            np.save(target, self.matrix, allow_pickle=False)

    @classmethod
    def load(cls, path: str) -> "FastWeightState":
        with open(path, "rb") as source:
            matrix = np.load(source, allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("快速状态必须是方阵")
        state = cls(int(matrix.shape[0]), dtype=matrix.dtype)
        state.matrix[...] = matrix
        return state

    def _normalized(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=self.matrix.dtype)
        if vector.shape != (self.rank,):
            raise ValueError(f"key 形状必须是 ({self.rank},)")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("key 不能是零向量")
        return vector / norm

