"""
ColorLM — 核心模型

架构:
  1. VQ Embedding: token → 离散码 + 温度
  2. 双向注意力 Transformer: 所有位置互相看（不是因果掩码）
  3. 并行预测: 一次性预测所有位置的码和温度
  4. 迭代修正: 高置信度先确定，低置信度后续修正

关键创新:
  - 温度驱动的迭代修正（不是均匀处理所有位置）
  - 双向上下文（每个位置都能看到完整序列）
  - 离散码本表示（每个 token 只需 4 个 int）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vq_embedding import VectorQuantize, TemperatureHead, CodePredictor


class BidirectionalAttention(nn.Module):
    """
    双向注意力: 每个 token 能看到所有其他 token
    （标准 Transformer 用因果掩码，只能看左边）
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [B, S, d_model]
        mask: [B, S] bool, True = 有效位置, False = 填充位置
        """
        B, S, _ = x.shape

        Q = self.q_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        # Q,K,V: [B, n_heads, S, d_k]

        # 注意力分数 — 没有因果掩码！所有位置互相关注
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores: [B, n_heads, S, S]

        # 填充掩码（如果有）
        if mask is not None:
            pad_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            scores = scores.masked_fill(~pad_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """一个 Transformer 块: 注意力 + FFN + 残差"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = BidirectionalAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # SwiGLU 的简化版
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ColorLM(nn.Module):
    """
    Color Language Model

    核心流程:
      1. 接收部分被遮盖的 token 序列
      2. 用双向注意力处理，所有位置互相关注
      3. 预测每个位置的 VQ 码本索引 + 温度
      4. 温度高的位置 → 高置信度 → 可以确定
         温度低的位置 → 低置信度 → 下一轮继续修正
    """

    def __init__(self, vocab_size: int = 32000, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 6, d_ff: int = 1024,
                 n_codebooks: int = 4, codebook_size: int = 256,
                 codebook_dim: int = 64, max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.mask_token_id = vocab_size  # 特殊的 MASK token

        # Token embedding (标准方式，用于输入)
        self.token_embed = nn.Embedding(vocab_size + 1, d_model)  # +1 for MASK
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # VQ 层: 把连续向量量化成离散码
        self.vq = VectorQuantize(d_model, n_codebooks, codebook_size, codebook_dim)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 预测头
        self.code_predictor = CodePredictor(d_model, n_codebooks, codebook_size)
        self.temperature_head = TemperatureHead(d_model)
        self.final_norm = nn.RMSNorm(d_model)

    def forward(self, masked_ids, target_ids=None, mask_positions=None):
        """
        masked_ids: [B, S]   部分位置被替换为 MASK 的 token IDs
        target_ids: [B, S]   原始 token IDs（训练时提供）
        mask_positions: [B, S] bool, True = 被遮盖的位置

        返回:
          code_logits: [B, S, K, codebook_size]  每个位置的码本预测 logits
          temperature: [B, S, 1]                 每个位置的温度
          vq_loss: scalar                        码本更新 loss
        """
        B, S = masked_ids.shape
        device = masked_ids.device

        # 1. Embedding
        tok_emb = self.token_embed(masked_ids)  # [B, S, d_model]
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos)            # [B, S, d_model]
        x = tok_emb + pos_emb

        # 2. VQ 量化（训练时提供 target，用于计算码本 loss）
        if target_ids is not None:
            target_emb = self.token_embed(target_ids) + pos_emb
            _, _, vq_loss = self.vq(target_emb)
        else:
            vq_loss = 0.0

        # 3. 双向 Transformer（没有因果掩码！）
        for layer in self.layers:
            x = layer(x)  # 不传 mask，因为我们用简单 padding=0 处理

        x = self.final_norm(x)

        # 4. 并行预测
        code_logits = self.code_predictor(x)   # [B, S, K, codebook_size]
        temperature = self.temperature_head(x)  # [B, S, 1]

        return code_logits, temperature, vq_loss

    @torch.no_grad()
    def generate(self, seq_len: int, n_refine_steps: int = 5,
                 tokenizer=None, prompt: str = None, prompt_ids=None,
                 device='cpu', temperature_boost: float = 1.0):
        """
        生成文本: 并行预测 + 迭代修正

        流程:
          Step 0: 所有位置设为 MASK
          Step 1: 并行预测所有位置的码和温度
          Step 2: 温度最高的 top-k 位置确定下来
          Step 3: 未确定的位置保持 MASK，重跑模型
          重复直到全部确定或达到最大步数
        """
        self.eval()

        # 初始化: 全部 MASK
        if prompt_ids is not None:
            # 有 prompt: prompt 部分已知，剩余部分 MASK
            prompt_len = prompt_ids.shape[0]
            ids = torch.full((1, seq_len), self.mask_token_id, dtype=torch.long, device=device)
            ids[0, :prompt_len] = prompt_ids
            start_pos = prompt_len
        else:
            ids = torch.full((1, seq_len), self.mask_token_id, dtype=torch.long, device=device)
            start_pos = 0

        # 已确定的位置（prompt 部分一开始就是确定的）
        determined = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        if prompt_ids is not None:
            determined[0, :prompt_len] = True

        for step in range(n_refine_steps):
            # 前向传播
            code_logits, temperature, _ = self.forward(ids)
            # code_logits: [1, S, K, codebook_size]
            # temperature: [1, S, 1]

            # 预测的码
            pred_codes = code_logits.argmax(dim=-1)  # [1, S, K]

            # 找到未确定的位置中，温度最高的
            temp = temperature.squeeze(-1).squeeze(0)  # [S]
            temp = temp.clone()
            temp[determined.squeeze(0)] = -1  # 已确定的不参与选择

            # 每轮确定 20% 的未确定位置
            n_undetermined = (~determined.squeeze(0)).sum().item()
            if n_undetermined == 0:
                break
            n_to_fix = max(1, n_undetermined // (n_refine_steps - step))

            # 选温度最高的位置
            _, top_indices = temp.topk(min(n_to_fix, n_undetermined))
            determined[0, top_indices] = True

            # 更新已确定位置的 token（用预测的码反查 token）
            # 简化版: 直接用 code_logits 选最可能的 token
            # 实际应该用码本反查，这里先用简单方式
            for pos in top_indices:
                # 用所有码本的平均 logits 来选 token
                avg_logits = code_logits[0, pos].mean(dim=0)  # [codebook_size]
                # 这里简化：直接用 embedding 的 argmax 近似
                # 实际生成时需要用 VQ decoder

            print(f"  Step {step+1}: 确定了 {top_indices.shape[0]} 个位置, "
                  f"剩余 {n_undetermined - top_indices.shape[0]} 个未确定")

        return ids, determined, temperature