"""
ColorLM + 预训练底座

架构:
  GPT-2 (冻结) 提供语言知识
  + VQ 码本 (新增) 压缩表示
  + 温度头 (新增) 预测重要性
  + 迭代修正推理 (推理策略)

优势:
  - 不需要从零训练
  - 底模已有语言/代码理解能力
  - 只训练新增的少量参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from colormlm.vq_embedding import VectorQuantize, TemperatureHead


class PretrainedColorLM(nn.Module):
    """
    基于预训练模型的 ColorLM

    冻结底模，只训练:
      1. VQ 码本层 (把 token embedding 压缩成离散码)
      2. 温度头 (预测每个 token 的重要性)
      3. 适配层 (把底模输出接到新层)
    """

    def __init__(self, model_name="gpt2", n_codebooks=4,
                 codebook_size=128, codebook_dim=32, device='cpu'):
        super().__init__()

        print(f"Loading pretrained model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # 冻结底模
        for param in self.backbone.parameters():
            param.requires_grad = False

        d_model = self.backbone.config.hidden_size  # GPT-2: 768
        vocab_size = self.tokenizer.vocab_size

        # 只有 padding token 才需要设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.d_model = d_model
        self.vocab_size = vocab_size

        # 新增层 (要训练的)
        # 1. 适配层: 把底模的输出维度映射到我们的 VQ 维度
        self.adapt = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 2),
        )

        # 2. VQ 码本
        self.vq = VectorQuantize(
            d_model=d_model // 2,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

        # 3. 温度头
        self.temperature_head = TemperatureHead(d_model // 2)

        # 4. 码本预测头 (训练用)
        self.code_heads = nn.ModuleList([
            nn.Linear(d_model // 2, codebook_size)
            for _ in range(n_codebooks)
        ])

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size

        # 统计
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        new_params = sum(p.numel() for p in self.get_trainable_params())
        print(f"Backbone (frozen): {backbone_params/1e6:.1f}M params")
        print(f"New layers (trainable): {new_params/1e6:.2f}M params")

    def get_trainable_params(self):
        """返回需要训练的参数"""
        return list(self.adapt.parameters()) + \
               list(self.vq.parameters()) + \
               list(self.temperature_head.parameters()) + \
               list(self.code_heads.parameters())

    def encode_text(self, text):
        """编码文本为 token IDs"""
        return self.tokenizer.encode(text, return_tensors='pt')

    def decode_ids(self, ids):
        """解码 token IDs 为文本"""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播

        input_ids: [B, S] token IDs
        返回:
          hidden: [B, S, d_model//2] 适配后的隐藏状态
          codes: [B, S, K] VQ 码本索引
          temperature: [B, S, 1] 温度
          code_logits: [B, S, K, codebook_size] 码本 logits
          vq_loss: scalar
        """
        # 底模前向 (冻结)
        with torch.no_grad():
            outputs = self.backbone(input_ids, attention_mask=attention_mask)
            backbone_hidden = outputs.last_hidden_state  # [B, S, 768]

        # 适配层
        hidden = self.adapt(backbone_hidden)  # [B, S, 384]

        # VQ 量化
        codes, quantized, vq_loss = self.vq(hidden)

        # 温度
        temperature = self.temperature_head(hidden)  # [B, S, 1]

        # 码本 logits
        code_logits = torch.stack([
            head(hidden) for head in self.code_heads
        ], dim=2)  # [B, S, K, codebook_size]

        return hidden, codes, temperature, code_logits, vq_loss

    @torch.no_grad()
    def iterative_predict(self, input_ids, mask_positions, n_steps=5):
        self.eval()
        ids = input_ids.clone()
        determined = set()
        history = []

        for step in range(n_steps):
            hidden, codes, temp, code_logits, _ = self.forward(ids)
            temp_sq = temp.squeeze(-1).squeeze(0)

            remaining = [p for p in mask_positions if p not in determined]
            if not remaining:
                break

            temps = [(p, temp_sq[p].item()) for p in remaining]
            temps.sort(key=lambda x: x[1], reverse=True)

            n_fix = max(1, len(temps) // 3)
            to_fix = temps[:n_fix]

            for pos, t in to_fix:
                emb_table = self.backbone.get_input_embeddings().weight
                adapted_emb = self.adapt(emb_table)
                hidden_pos = hidden[0, pos]
                sim = F.cosine_similarity(adapted_emb, hidden_pos.unsqueeze(0), dim=-1)
                pred_token = sim.argmax().item()
                ids[0, pos] = int(pred_token)
                determined.add(pos)

            history.append({
                "step": step + 1,
                "fixed": [(p, int(t)) for p, t in to_fix],
                "remaining": len(remaining) - n_fix,
                "avg_temp": sum(t for _, t in to_fix) / len(to_fix),
            })

        return ids, history