"""
ColorLM Student Model - 轻量蒸馏学生

从 Qwen 的隐藏状态中学习，用 VQ 码本压缩表示。
训练后完全不依赖 Qwen，独立运行。

架构:
  Embedding → 6层Transformer → VQ码本(4x128) + 温度头
  参数量: ~50M (vs Qwen 1543M, 压缩30倍)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vq_embedding import VectorQuantize, TemperatureHead


class StudentTransformer(nn.Module):
    """轻量 Transformer 块"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 双向注意力（没有因果掩码）
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ColorLMStudent(nn.Module):
    """
    ColorLM 学生模型

    输入: token IDs [B, S]
    输出:
      - vq_codes: [B, S, 4]       4个RGB索引
      - quantized: [B, S, d_mid]  量化后的向量（用于蒸馏）
      - temperature: [B, S, 1]    温度
      - vq_loss: scalar
    """

    def __init__(self, vocab_size=151643, d_model=384, n_heads=6, n_layers=6,
                 d_ff=1536, max_seq_len=512, n_codebooks=4, codebook_size=128,
                 codebook_dim=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size

        # Token + Position Embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer 层（双向，没有因果掩码）
        self.layers = nn.ModuleList([
            StudentTransformer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)

        # VQ 码本层
        self.vq = VectorQuantize(d_model, n_codebooks, codebook_size, codebook_dim)

        # 温度头
        self.temperature_head = TemperatureHead(d_model)

        # 蒸馏投影头: 把学生输出映射到 Qwen 的隐藏空间 (384 -> 1536)
        self.distill_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, 1536),  # Qwen hidden_size
        )

        # 参数统计
        total = sum(p.numel() for p in self.parameters())
        print(f"  Student model: {total/1e6:.1f}M params")

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, S]
        attention_mask: [B, S] bool, True=有效, False=padding

        Returns:
          quantized: [B, S, d_model]  VQ量化后的向量
          vq_codes: [B, S, 4]        码本索引
          temperature: [B, S, 1]     温度
          distill_out: [B, S, 1536]  蒸馏投影（对齐Qwen空间）
          vq_loss: scalar
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embedding
        tok_emb = self.token_embed(input_ids)
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)
        x = self.embed_dropout(tok_emb + pos_emb)

        # Transformer（双向注意力）
        # padding mask: True = 忽略
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True = padding = 忽略

        for layer in self.layers:
            x = layer(x, mask=key_padding_mask)

        x = self.final_norm(x)

        # VQ 量化
        vq_codes, quantized, vq_loss = self.vq(x)

        # 温度
        temperature = self.temperature_head(x)

        # 蒸馏投影
        distill_out = self.distill_proj(quantized)

        return quantized, vq_codes, temperature, distill_out, vq_loss

    @torch.no_grad()
    def encode(self, input_ids, attention_mask=None):
        """编码: token IDs -> VQ 码本索引 + 温度"""
        self.eval()
        _, vq_codes, temperature, _, _ = self.forward(input_ids, attention_mask)
        return vq_codes, temperature

    @torch.no_grad()
    def get_rgb(self, input_ids):
        """获取每个 token 的 RGB 颜色"""
        codes, temp = self.encode(input_ids)
        return codes, temp
