"""
离线蒸馏 - 极速版

只缓存少量样本（~100），快速完成。
学生模型后续通过自训练(MLM)继续提升。
"""

import os, sys, numpy as np
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

SEQ_LEN = 64
MAX_CHARS = 50000   # 50K chars only
MAX_SAMPLES = 100   # 最多100个样本

def main():
    print("=" * 50)
    print("Qwen Distillation Cache (fast)")
    print("=" * 50)

    data_path = Path(__file__).parent / "data" / "train_code.txt"
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read(MAX_CHARS)
    print(f"[1] Loaded {len(text)} chars")

    print(f"[2] Loading Qwen...")
    model_path = str(Path(__file__).parent.parent / "models" / "Qwen2.5-1.5B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float32)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    d = model.config.hidden_size
    print(f"  d_model={d}")

    print(f"[3] Tokenizing...")
    tokens = tok.encode(text, add_special_tokens=False)
    print(f"  {len(tokens)} tokens")

    # 取不重叠的chunk
    chunks = []
    for i in range(0, len(tokens) - SEQ_LEN, SEQ_LEN):
        chunks.append(tokens[i:i + SEQ_LEN])
        if len(chunks) >= MAX_SAMPLES:
            break
    print(f"  {len(chunks)} chunks")

    print(f"[4] Extracting hidden states...")
    all_h, all_i, all_m = [], [], []
    for idx, chunk in enumerate(chunks):
        ids = torch.tensor([chunk])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = model(ids, attention_mask=mask)
            h = out.last_hidden_state.float()
        all_h.append(h.numpy())
        all_i.append(ids.numpy())
        all_m.append(mask.numpy())
        if (idx+1) % 20 == 0:
            print(f"  {idx+1}/{len(chunks)}")

    all_h = np.concatenate(all_h, 0)
    all_i = np.concatenate(all_i, 0)
    all_m = np.concatenate(all_m, 0)

    save_path = Path(__file__).parent / "data" / "distill_cache.npz"
    np.savez_compressed(str(save_path), hidden_states=all_h, input_ids=all_i, attention_mask=all_m)
    mb = os.path.getsize(str(save_path)) / 1024 / 1024
    print(f"[5] Saved: {save_path} ({mb:.1f}MB), shape={all_h.shape}")
    print("Done!")

if __name__ == "__main__":
    main()
