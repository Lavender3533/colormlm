"""
VQ (Vector Quantization) Embedding Layer

核心思想:
  传统 Embedding: token_id → d_model 维浮点向量 (4096 bytes)
  VQ Embedding:   token_id → K 个码本索引 + 温度 T (K+1 bytes)

  码本 = 4 组，每组 256 个基向量
  token 的含义 = 4 个基向量的组合（类似 RGB 混色）
  温度 = 这个 token 有多重要
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    """
    向量量化层: 把连续向量压缩成离散码本索引

    输入: [batch, seq, d_model]  连续向量
    输出: [batch, seq, K]        K 个码本索引 (int)
          [batch, seq, d_model]  量化后的向量 (用于后续计算)
    """

    def __init__(self, d_model: int, n_codebooks: int = 4,
                 codebook_size: int = 256, codebook_dim: int = 64):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # K 个码本，每个有 codebook_size 个基向量，维度 codebook_dim
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, codebook_dim) * 0.02)
            for _ in range(n_codebooks)
        ])

        # 把 d_model 投影到每个码本的维度
        self.projectors = nn.ModuleList([
            nn.Linear(d_model, codebook_dim, bias=False)
            for _ in range(n_codebooks)
        ])

        # 量化后的向量拼接回 d_model
        self.out_proj = nn.Linear(n_codebooks * codebook_dim, d_model, bias=False)

    def forward(self, x):
        """
        x: [batch, seq, d_model]
        返回:
          codes: [batch, seq, K]        码本索引
          quantized: [batch, seq, d_model]  量化后的向量
          vq_loss: scalar               码本更新 loss
        """
        batch, seq, _ = x.shape
        codes = []
        quantized_parts = []
        vq_loss = 0.0

        for i in range(self.n_codebooks):
            # 投影到码本维度
            z = self.projectors[i](x)           # [batch, seq, codebook_dim]

            # 计算到每个码本向量的距离
            # z: [B, S, D], codebook: [K, D]
            dist = torch.cdist(z, self.codebooks[i].unsqueeze(0).expand(batch, -1, -1))
            # dist: [B, S, K]

            # 找最近的码本索引
            idx = dist.argmin(dim=-1)            # [B, S]
            codes.append(idx)

            # 取出量化后的向量（直通估计器 trick: 前向用量化值，反向传梯度给原始值）
            quant = F.embedding(idx, self.codebooks[i])  # [B, S, codebook_dim]

            # Straight-Through Estimator: 让梯度能传回去
            quant_st = z + (quant - z).detach()
            quantized_parts.append(quant_st)

            # 码本更新 loss: 让码本向量靠近实际数据点
            vq_loss += F.mse_loss(quant, z.detach()) + 0.1 * F.mse_loss(quant.detach(), z)

        codes = torch.stack(codes, dim=-1)       # [B, S, K]
        quantized = torch.cat(quantized_parts, dim=-1)  # [B, S, K*D]
        quantized = self.out_proj(quantized)     # [B, S, d_model]

        return codes, quantized, vq_loss


class TemperatureHead(nn.Module):
    """
    温度预测头: 预测每个 token 的重要性

    输入: [batch, seq, d_model]
    输出: [batch, seq, 1]  温度值 (0~1)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),  # 温度在 0~1 之间
        )

    def forward(self, x):
        return self.net(x)  # [B, S, 1]


class CodePredictor(nn.Module):
    """
    码本预测头: 从隐藏状态预测每个码本的索引

    输入: [batch, seq, d_model]
    输出: [batch, seq, K, codebook_size]  每个码本的 logits
    """

    def __init__(self, d_model: int, n_codebooks: int = 4, codebook_size: int = 256):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size

        # 每个码本一个独立的预测头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, codebook_size),
            )
            for _ in range(n_codebooks)
        ])

    def forward(self, x):
        """
        x: [B, S, d_model]
        返回: [B, S, K, codebook_size]
        """
        logits = [head(x) for head in self.heads]  # 每个: [B, S, codebook_size]
        return torch.stack(logits, dim=2)           # [B, S, K, codebook_size]