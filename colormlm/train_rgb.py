"""
ColorLM v4 - RGB + Temperature + Bidirectional Training

核心创新:
  1. RGB 码本: 4组码本 x 128基向量 = 4个离散索引代替768维浮点
  2. 温度系统: 每个token附带0~1温度，表示重要性
  3. 双向预测: 所有位置同时预测（不是自回归逐token）

用法:
  python train_rgb.py --base gpt2          # 用GPT-2底座
  python train_rgb.py --base qwen          # 用Qwen2.5-1.5B底座
"""

import os
import sys
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from colormlm.vq_embedding import VectorQuantize, TemperatureHead


class ColorLMv4(nn.Module):
    def __init__(self, backbone, tokenizer, d_model, n_codebooks=4,
                 codebook_size=128, codebook_dim=32):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size

        for param in self.backbone.parameters():
            param.requires_grad = False

        mid = d_model // 2
        self.adapt = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.SiLU(),
            nn.Linear(mid, mid),
        )
        self.vq = VectorQuantize(mid, n_codebooks, codebook_size, codebook_dim)
        self.temperature_head = TemperatureHead(mid)
        self.code_heads = nn.ModuleList([
            nn.Linear(mid, codebook_size) for _ in range(n_codebooks)
        ])

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        new_params = sum(p.numel() for p in self.get_trainable_params())
        print(f"  Backbone (frozen): {backbone_params/1e6:.1f}M")
        print(f"  New layers (trainable): {new_params/1e6:.2f}M")

    def get_trainable_params(self):
        return (list(self.adapt.parameters()) +
                list(self.vq.parameters()) +
                list(self.temperature_head.parameters()) +
                list(self.code_heads.parameters()))

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            out = self.backbone(input_ids, attention_mask=attention_mask)
            backbone_hidden = out.last_hidden_state.float()  # bfloat16 -> float32

        hidden = self.adapt(backbone_hidden)
        codes, quantized, vq_loss = self.vq(hidden)
        temperature = self.temperature_head(hidden)
        code_logits = torch.stack([head(hidden) for head in self.code_heads], dim=2)

        return hidden, codes, temperature, code_logits, vq_loss


def load_training_data(filepath, max_chars=5000000):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read(max_chars)
    print(f"  Loaded {len(text)} chars from {filepath}")
    return text


def create_mlm_batch(tokenizer, text, seq_len=256, batch_size=8, mask_ratio=0.3):
    input_ids_list = []
    labels_list = []
    mask_positions_list = []

    tokens = tokenizer.encode(text, add_special_tokens=False)

    for _ in range(batch_size):
        if len(tokens) <= seq_len:
            start = 0
            end = len(tokens)
        else:
            start = random.randint(0, len(tokens) - seq_len - 1)
            end = start + seq_len

        chunk = tokens[start:end]
        if len(chunk) < seq_len:
            chunk = chunk + [tokenizer.pad_token_id or 0] * (seq_len - len(chunk))

        input_ids = list(chunk)
        labels = list(chunk)
        mask_positions = [False] * seq_len

        for i in range(seq_len):
            if input_ids[i] == (tokenizer.pad_token_id or 0):
                continue
            if random.random() < mask_ratio:
                mask_positions[i] = True
                labels[i] = input_ids[i]
                r = random.random()
                if r < 0.8:
                    input_ids[i] = tokenizer.pad_token_id or 0
                elif r < 0.9:
                    input_ids[i] = random.randint(0, tokenizer.vocab_size - 1)

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        mask_positions_list.append(mask_positions)

    return (torch.tensor(input_ids_list),
            torch.tensor(labels_list),
            torch.tensor(mask_positions_list, dtype=torch.bool))


def train(args):
    print("=" * 60)
    print("ColorLM v4 - RGB + Temperature + Bidirectional Training")
    print("=" * 60)

    print(f"\n[1] Loading base model: {args.base}...")
    from transformers import AutoModel, AutoTokenizer

    if args.base == 'qwen':
        model_name = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "models", "Qwen2.5-1.5B-Instruct")
        if not os.path.exists(model_name):
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    else:
        model_name = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    d_model = backbone.config.hidden_size
    print(f"  d_model = {d_model}")

    print(f"\n[2] Creating ColorLMv4...")
    model = ColorLMv4(backbone=backbone, tokenizer=tokenizer, d_model=d_model,
                      n_codebooks=args.n_codebooks, codebook_size=args.codebook_size,
                      codebook_dim=args.codebook_dim)

    print(f"\n[3] Loading training data...")
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train_code.txt")
    if not os.path.exists(data_path):
        print(f"  ERROR: {data_path} not found!")
        return
    text = load_training_data(data_path, max_chars=args.max_chars)

    print(f"\n[4] Training...")
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()
    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_code_correct = 0
        epoch_code_total = 0
        epoch_temp_high = []
        epoch_temp_low = []
        n_batches = 0

        for step in range(args.steps_per_epoch):
            input_ids, labels, mask_positions = create_mlm_batch(
                tokenizer, text, seq_len=args.seq_len,
                batch_size=args.batch_size, mask_ratio=args.mask_ratio)

            hidden, codes, temperature, code_logits, vq_loss = model(input_ids)

            code_loss = 0
            for k in range(args.n_codebooks):
                with torch.no_grad():
                    target_emb = model.backbone.get_input_embeddings()(labels).float()
                    target_hidden = model.adapt(target_emb)
                    target_proj = model.vq.projectors[k](target_hidden)
                    dist = torch.cdist(target_proj, model.vq.codebooks[k].unsqueeze(0).expand(target_hidden.shape[0], -1, -1))
                    target_codes = dist.argmin(dim=-1)

                logits_k = code_logits[:, :, k, :]
                loss_k = F.cross_entropy(
                    logits_k.view(-1, args.codebook_size),
                    target_codes.view(-1), reduction='none')
                mask_flat = mask_positions.view(-1).float()
                loss_k = (loss_k * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                code_loss += loss_k

                pred_codes = logits_k.argmax(dim=-1)
                correct = ((pred_codes == target_codes) & mask_positions).sum().item()
                total = mask_positions.sum().item()
                epoch_code_correct += correct
                epoch_code_total += total

            code_loss /= args.n_codebooks

            temp = temperature.squeeze(-1)
            temp_target = torch.where(mask_positions, 0.8, 0.2)
            temp_loss = F.mse_loss(temp, temp_target)

            vq_loss_scaled = vq_loss * 0.1
            total_loss = code_loss + temp_loss + vq_loss_scaled

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

            with torch.no_grad():
                if mask_positions.any():
                    epoch_temp_high.append(temp[mask_positions].mean().item())
                if (~mask_positions).any():
                    epoch_temp_low.append(temp[~mask_positions].mean().item())

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        code_acc = epoch_code_correct / max(epoch_code_total, 1) * 100
        avg_temp_high = sum(epoch_temp_high) / max(len(epoch_temp_high), 1)
        avg_temp_low = sum(epoch_temp_low) / max(len(epoch_temp_low), 1)
        temp_diff = avg_temp_high - avg_temp_low

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"CodeAcc: {code_acc:.1f}% | "
              f"Temp[MASK]: {avg_temp_high:.3f} | "
              f"Temp[known]: {avg_temp_low:.3f} | "
              f"Diff: {temp_diff:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", f"colormlm_v4_{args.base}.pt")
    torch.save({
        'adapt': model.adapt.state_dict(),
        'vq': model.vq.state_dict(),
        'temperature_head': model.temperature_head.state_dict(),
        'code_heads': model.code_heads.state_dict(),
        'args': vars(args),
    }, save_path)
    print(f"\n  Model saved to: {save_path}")
    print(f"  Best loss: {best_loss:.4f}")

    print(f"\n[5] Final Temperature Analysis:")
    model.eval()
    with torch.no_grad():
        test_texts = [
            "def fibonacci(n):",
            "class Stack:",
            "import os",
            "for i in range(10):",
            "return result",
        ]
        for text in test_texts:
            ids = tokenizer.encode(text, return_tensors='pt')
            _, _, temp, _, _ = model(ids)
            tokens = tokenizer.convert_ids_to_tokens(ids[0])
            temp_vals = temp[0].squeeze(-1).tolist()
            print(f"\n  '{text}'")
            for tok, t in zip(tokens, temp_vals):
                tok_safe = tok.encode('ascii', 'replace').decode()
                bar = '#' * int(t * 20)
                importance = "HIGH" if t > 0.6 else ("MED" if t > 0.4 else "LOW")
                print(f"    {tok_safe:15s} | {t:.3f} | {bar} | {importance}")

    print(f"\n[6] VQ Codebook Analysis:")
    with torch.no_grad():
        ids = tokenizer.encode("def quicksort(arr):", return_tensors='pt')
        _, codes, temp, _, _ = model(ids)
        tokens = tokenizer.convert_ids_to_tokens(ids[0])
        print("  Token -> [R, G, B, A]")
        for tok, code, t in zip(tokens, codes[0].tolist(), temp[0].squeeze(-1).tolist()):
            tok_safe = tok.encode('ascii', 'replace').decode()
            print(f"    {tok_safe:15s} -> {code} | temp={t:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="gpt2", choices=["gpt2", "qwen"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_chars", type=int, default=2000000)
    parser.add_argument("--n_codebooks", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=128)
    parser.add_argument("--codebook_dim", type=int, default=32)
    args = parser.parse_args()
    train(args)
