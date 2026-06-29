"""
蒸馏训练 - 学生模型

读取 Qwen 缓存的隐藏状态，训练轻量学生模型。
训练后学生独立运行，不需要 Qwen。

Loss = 蒸馏Loss(MSE) + VQ Loss + 温度Loss
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from colormlm.student_model import ColorLMStudent


def train(args):
    print("=" * 60)
    print("ColorLM Student - Distillation Training")
    print("=" * 60)

    # 加载缓存
    cache_path = Path(__file__).parent / "data" / "distill_cache.npz"
    if not cache_path.exists():
        print(f"ERROR: {cache_path} not found!")
        print("Run distill_cache.py first.")
        return

    print(f"\n[1] Loading distillation cache...")
    cache = np.load(str(cache_path))
    teacher_hidden = torch.from_numpy(cache['hidden_states']).float()  # [N, S, 1536]
    input_ids = torch.from_numpy(cache['input_ids']).long()            # [N, S]
    attention_mask = torch.from_numpy(cache['attention_mask']).long()   # [N, S]

    n_samples = teacher_hidden.shape[0]
    seq_len = teacher_hidden.shape[1]
    print(f"  Samples: {n_samples}, Seq len: {seq_len}")
    print(f"  Teacher hidden: {teacher_hidden.shape}")

    # 创建学生模型
    print(f"\n[2] Creating student model...")
    model = ColorLMStudent(
        vocab_size=151643,  # Qwen vocab size
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=seq_len,
        n_codebooks=args.n_codebooks,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        dropout=args.dropout,
    )

    # 训练
    print(f"\n[3] Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()
    best_loss = float('inf')

    for epoch in range(args.epochs):
        # 随机打乱
        indices = list(range(n_samples))
        random.shuffle(indices)

        epoch_loss = 0
        epoch_distill = 0
        epoch_vq = 0
        epoch_temp = 0
        n_batches = 0

        for i in range(0, n_samples - args.batch_size + 1, args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            if len(batch_idx) < args.batch_size:
                break

            ids = input_ids[batch_idx]           # [B, S]
            mask = attention_mask[batch_idx]      # [B, S]
            teacher = teacher_hidden[batch_idx]   # [B, S, 1536]

            # 学生前向
            quantized, vq_codes, temperature, distill_out, vq_loss = model(ids, mask)

            # --- Loss 1: 蒸馏 Loss ---
            # 学生的蒸馏投影要逼近 Qwen 的隐藏状态
            # 只在有效位置计算
            mask_3d = mask.unsqueeze(-1).float()  # [B, S, 1]
            distill_loss = F.mse_loss(
                distill_out * mask_3d,
                teacher * mask_3d,
                reduction='sum'
            ) / (mask_3d.sum() * 1536 + 1e-8)

            # --- Loss 2: VQ Loss ---
            vq_loss_scaled = vq_loss * 0.1

            # --- Loss 3: 温度 Loss ---
            # 有效位置温度适中（0.5），padding位置温度低（0.1）
            temp = temperature.squeeze(-1)  # [B, S]
            temp_target = torch.where(mask.bool(), 0.5, 0.1)
            temp_loss = F.mse_loss(temp, temp_target)

            # --- Total ---
            total_loss = distill_loss + vq_loss_scaled + temp_loss * 0.5

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_distill += distill_loss.item()
            epoch_vq += vq_loss.item()
            epoch_temp += temp_loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_distill = epoch_distill / max(n_batches, 1)
        avg_vq = epoch_vq / max(n_batches, 1)
        avg_temp = epoch_temp / max(n_batches, 1)

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Distill: {avg_distill:.4f} | "
              f"VQ: {avg_vq:.4f} | "
              f"Temp: {avg_temp:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存最佳模型
            save_path = Path(__file__).parent / "data" / "student_best.pt"
            torch.save(model.state_dict(), str(save_path))

    # 保存最终模型
    save_path = Path(__file__).parent / "data" / "student_final.pt"
    torch.save(model.state_dict(), str(save_path))
    print(f"\n  Best loss: {best_loss:.4f}")
    print(f"  Saved to: {save_path}")

    # --- 验证 ---
    print(f"\n[4] Verification...")
    model.eval()

    # 用缓存的数据验证
    with torch.no_grad():
        # 取前5个样本
        n_show = min(5, n_samples)
        ids = input_ids[:n_show]
        mask = attention_mask[:n_show]
        teacher = teacher_hidden[:n_show]

        quantized, vq_codes, temperature, distill_out, _ = model(ids, mask)

        # 蒸馏质量
        cos_sim = F.cosine_similarity(distill_out, teacher, dim=-1)  # [B, S]
        valid_mask = mask.bool()
        avg_sim = cos_sim[valid_mask].mean().item()
        print(f"  Avg cosine similarity (student vs Qwen): {avg_sim:.4f}")

        # VQ 码本多样性
        codes_flat = vq_codes[valid_mask]  # [N, 4]
        for k in range(args.n_codebooks):
            unique = codes_flat[:, k].unique().numel()
            print(f"  Codebook {k}: {unique}/{args.codebook_size} unique codes")

        # 温度分布
        temp_vals = temperature[valid_mask].squeeze(-1)
        print(f"  Temperature: mean={temp_vals.mean():.3f}, "
              f"min={temp_vals.min():.3f}, max={temp_vals.max():.3f}")

    print("\nDone! Student model trained and saved.")
    print("You can now use it independently without Qwen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_codebooks", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=128)
    parser.add_argument("--codebook_dim", type=int, default=32)
    args = parser.parse_args()
    train(args)
