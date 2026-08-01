"""Reference ColorDelta + ColorMix + sparse MoE cell for ColorLM v4."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorStateCell(nn.Module):
    """One autoregressive state cell with no token KV cache.

    The small PyTorch module is an export reference. Production inference uses
    the exported graph and later a fused ggml/Vulkan implementation.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        state_rank: int = 16,
        landmark_slots: int = 8,
        experts: int = 8,
        active_experts: int = 2,
        expert_size: int = 32,
    ) -> None:
        super().__init__()
        if not 0 < active_experts < experts:
            raise ValueError("active_experts 必须位于 (0, experts)")
        self.hidden_size = hidden_size
        self.state_rank = state_rank
        self.landmark_slots = landmark_slots
        self.experts = experts
        self.active_experts = active_experts

        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.query = nn.Linear(hidden_size, state_rank, bias=False)
        self.key = nn.Linear(hidden_size, state_rank, bias=False)
        self.value = nn.Linear(hidden_size, state_rank, bias=False)
        self.state_gate = nn.Linear(hidden_size, 2)
        self.state_output = nn.Linear(state_rank, hidden_size, bias=False)

        self.landmark_write = nn.Linear(hidden_size, landmark_slots, bias=False)
        self.landmark_read = nn.Linear(hidden_size, landmark_slots, bias=False)
        self.landmark_value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.landmark_output = nn.Linear(hidden_size, hidden_size, bias=False)

        self.router = nn.Linear(hidden_size, experts, bias=False)
        self.expert_gate = nn.Parameter(torch.empty(experts, hidden_size, expert_size))
        self.expert_up = nn.Parameter(torch.empty(experts, hidden_size, expert_size))
        self.expert_down = nn.Parameter(torch.empty(experts, expert_size, hidden_size))
        self.shared_gate = nn.Linear(hidden_size, expert_size, bias=False)
        self.shared_up = nn.Linear(hidden_size, expert_size, bias=False)
        self.shared_down = nn.Linear(expert_size, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.expert_gate, gain=0.5)
        nn.init.xavier_uniform_(self.expert_up, gain=0.5)
        nn.init.xavier_uniform_(self.expert_down, gain=0.5)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.square().mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + 1e-6) * self.norm_weight

    def forward(
        self,
        hidden: torch.Tensor,
        delta_state: torch.Tensor,
        kernel_state: torch.Tensor,
        kernel_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self._norm(hidden)
        query = F.normalize(self.query(normalized), dim=-1)
        key = F.normalize(self.key(normalized), dim=-1)
        value = self.value(normalized)
        gates = torch.sigmoid(self.state_gate(normalized))

        prediction = torch.bmm(delta_state, key.unsqueeze(-1)).squeeze(-1)
        error = value - prediction
        correction = torch.bmm(error.unsqueeze(-1), key.unsqueeze(1))
        next_delta_state = (
            gates[:, 0].view(-1, 1, 1) * delta_state
            + gates[:, 1].view(-1, 1, 1) * correction
        )
        state_read = torch.bmm(next_delta_state, query.unsqueeze(-1)).squeeze(-1)
        hidden = hidden + self.state_output(state_read)

        normalized = self._norm(hidden)
        write = F.elu(self.landmark_write(normalized)) + 1.0
        kernel_value = self.landmark_value(normalized)
        next_kernel_state = (
            0.995 * kernel_state
            + write.unsqueeze(-1) * kernel_value.unsqueeze(1)
        )
        next_kernel_norm = 0.995 * kernel_norm + write
        read = F.elu(self.landmark_read(normalized)) + 1.0
        numerator = torch.bmm(read.unsqueeze(1), next_kernel_state).squeeze(1)
        denominator = (read * next_kernel_norm).sum(dim=-1, keepdim=True) + 1e-6
        kernel_read = numerator / denominator
        hidden = hidden + self.landmark_output(kernel_read)

        normalized = self._norm(hidden)
        routing_logits = self.router(normalized)
        routing_values, routing_ids = torch.topk(
            routing_logits, self.active_experts, dim=-1
        )
        routing_weights = torch.softmax(routing_values, dim=-1)

        gate = torch.einsum("bd,edh->beh", normalized, self.expert_gate)
        up = torch.einsum("bd,edh->beh", normalized, self.expert_up)
        expert_hidden = F.silu(gate) * up
        expert_outputs = torch.einsum(
            "beh,ehd->bed", expert_hidden, self.expert_down
        )
        gather_ids = routing_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        selected = torch.gather(expert_outputs, 1, gather_ids)
        routed = (selected * routing_weights.unsqueeze(-1)).sum(dim=1)

        shared = self.shared_down(
            F.silu(self.shared_gate(normalized)) * self.shared_up(normalized)
        )
        hidden = hidden + routed + shared
        return hidden, next_delta_state, next_kernel_state, next_kernel_norm, routing_ids
