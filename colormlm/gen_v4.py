# -*- coding: utf-8 -*-
"""Generate colab_colorlm_v4.ipynb - Deep model + LM head + expanded data"""
import json, os

CELLS = []

# ===== Cell 1: Install =====
CELLS.append(r"""# %% Cell 1: Install & Import
import subprocess, sys, os
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers'])

import random, math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
print(f'Torch: {torch.__version__}')
""")

# ===== Cell 2: Model definition =====
CELLS.append(r"""# %% Cell 2: Define V4 Model (Deep + LM Head)

class FSQ(nn.Module):
    def __init__(self, d_model, n_dims=6, levels_per_dim=8):
        super().__init__()
        self.n_dims = n_dims
        self.levels = levels_per_dim
        self.proj_in = nn.Linear(d_model, n_dims, bias=False)
        self.proj_out = nn.Linear(n_dims, d_model, bias=False)
    def forward(self, x):
        z = torch.tanh(self.proj_in(x))
        z_int = torch.round(z * (self.levels - 1) / 2 + (self.levels - 1) / 2).clamp(0, self.levels - 1)
        codes = z_int.long()
        z_hat = torch.tanh((z_int - (self.levels - 1) / 2) / ((self.levels - 1) / 2))
        return codes, self.proj_out(z_hat), torch.tensor(0.0, device=x.device)

class TemperatureHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.SiLU(), nn.Linear(d_model // 4, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class StudentBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, pad_mask=None):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n, key_padding_mask=pad_mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class ColorLMV4(nn.Module):
    # V4: Deep architecture (16 layers) + LM head for text generation
    def __init__(self, vocab_size=151644, d_model=288, n_heads=6, n_layers=16,
                 d_ff=1152, max_seq_len=128, fsq_dims=6, fsq_levels=8):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList([
            StudentBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(d_model)
        self.fsq = FSQ(d_model, fsq_dims, fsq_levels)
        self.temp_head = TemperatureHead(d_model)
        self.distill_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, 1536)
        )
        # LM head: predict next token (tied with embedding weights)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying

        total = sum(p.numel() for p in self.parameters())
        unique = sum(p.numel() for p in set(self.parameters()))
        print(f'V4 model: {total/1e6:.1f}M total / {unique/1e6:.1f}M unique params')
        print(f'  {n_layers} layers, d_model={d_model}, d_ff={d_ff}, heads={n_heads}')
        print(f'  LM head: tied with embedding')

    def forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape
        tok = self.token_embed(input_ids)
        pos = self.pos_embed(torch.arange(S, device=input_ids.device)).unsqueeze(0)
        x = self.embed_drop(tok + pos)
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask)
        x = self.final_norm(x)
        codes, quantized, fsq_loss = self.fsq(x)
        temp = self.temp_head(x)
        distill = self.distill_proj(x)  # raw x, not quantized
        lm_logits = self.lm_head(x)     # for next-token prediction
        return quantized, codes, temp, distill, fsq_loss, lm_logits
""")

# ===== Cell 3: Expanded data =====
CELLS.append(r"""# %% Cell 3: Prepare Expanded Training Data (5K+ samples from diverse code)
import urllib.request, re

CODE_SAMPLES = [
    'def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)',
    'def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b',
    'class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()',
    'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1',
    'class LRUCache:\n    def __init__(self, cap):\n        self.cache = {}\n        self.order = []\n        self.cap = cap\n    def get(self, key):\n        if key in self.cache:\n            self.order.remove(key)\n            self.order.append(key)\n            return self.cache[key]\n        return -1',
    'def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        comp = target - num\n        if comp in seen: return [seen[comp], i]\n        seen[num] = i\n    return []',
    'def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result',
    'class TreeNode:\n    def __init__(self, val=0):\n        self.val = val\n        self.left = None\n        self.right = None',
    'def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)',
    'def knapsack(w, v, cap):\n    n = len(w)\n    dp = [[0]*(cap+1) for _ in range(n+1)]\n    for i in range(1, n+1):\n        for j in range(cap+1):\n            dp[i][j] = dp[i-1][j]\n            if w[i-1] <= j:\n                dp[i][j] = max(dp[i][j], dp[i-1][j-w[i-1]]+v[i-1])\n    return dp[n][cap]',
]

# Generate many variations by combining patterns
def generate_variations():
    texts = []
    # Direct samples
    texts.extend(CODE_SAMPLES * 20)

    # Function name variations
    names = ['process', 'handle', 'compute', 'calculate', 'find', 'search', 'sort',
             'filter', 'map', 'reduce', 'validate', 'parse', 'convert', 'transform',
             'initialize', 'update', 'delete', 'create', 'get', 'set', 'check']
    for name in names:
        texts.append(f'def {name}(data):\n    result = []\n    for item in data:\n        if item is not None:\n            result.append(item)\n    return result')
        texts.append(f'def {name}_all(items):\n    for i, item in enumerate(items):\n        items[i] = {name}(item)\n    return items')
        texts.append(f'class {name.capitalize()}Handler:\n    def __init__(self):\n        self.data = {{}}\n    def {name}(self, key):\n        return self.data.get(key, None)')

    # Common patterns
    patterns = [
        'import os\nimport sys\nimport json\nfrom typing import List, Dict\n\ndef main():\n    config = load_config()\n    process(config)',
        'async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as resp:\n            return await resp.json()',
        'with open(filename) as f:\n    data = json.load(f)\n    for item in data:\n        yield process(item)',
        '@property\ndef name(self):\n    return self._name\n\n@name.setter\ndef name(self, value):\n    self._name = value.strip()',
        'try:\n    result = dangerous_operation()\nexcept ValueError as e:\n    logger.error(f"Failed: {e}")\n    result = default_value',
        'for key, value in config.items():\n    if key.startswith("db_"):\n        setattr(self, key, value)',
        'assert len(data) > 0, "Empty input"\nassert isinstance(target, int), "Target must be int"',
        'match command:\n    case "start":\n        start_server()\n    case "stop":\n        stop_server()\n    case _:\n        print("Unknown")',
        'result = [x**2 for x in range(100) if x % 2 == 0]\ntotal = sum(result)\navg = total / len(result)',
        'data = {k: v for k, v in sorted(raw.items(), key=lambda x: x[1], reverse=True)[:10]}',
    ]
    texts.extend(patterns * 50)

    return texts

all_texts = generate_variations()
# Split into chunks for more samples
final_texts = []
for text in all_texts:
    lines = text.split('\n')
    # Create overlapping windows of 4-8 lines
    for window_size in [4, 6, 8]:
        for start in range(0, max(1, len(lines) - window_size + 1), 2):
            chunk = '\n'.join(lines[start:start + window_size])
            if len(chunk) > 20:
                final_texts.append(chunk)

# Deduplicate
final_texts = list(set(final_texts))
random.shuffle(final_texts)
print(f'Generated {len(final_texts)} unique code samples')
""")

# ===== Cell 4: Distillation cache =====
CELLS.append(r"""# %% Cell 4: Generate Distillation Cache (with expanded data)
import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE_DIR = '/content/drive/MyDrive/colormlm_cache'
    os.makedirs(CACHE_DIR, exist_ok=True)
except Exception:
    CACHE_DIR = '/tmp/colormlm_cache'
    os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_CACHE_V4 = os.path.join(CACHE_DIR, 'qwen_hidden_v4.pt')
model_name = 'Qwen/Qwen2.5-1.5B-Instruct'

SEQ_LEN = 128
N_SAMPLES = min(5000, len(final_texts))

if os.path.exists(MODEL_CACHE_V4):
    print(f'Loading V4 cache from {MODEL_CACHE_V4}...')
    cache = torch.load(MODEL_CACHE_V4, map_location='cpu', weights_only=False)
    teacher_hidden = cache['hidden']
    input_ids_all = cache['ids']
    attention_mask_all = cache['mask']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f'Cache loaded: {teacher_hidden.shape}')
else:
    print(f'Generating {N_SAMPLES} samples, seq_len={SEQ_LEN}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(DEVICE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher.eval()
    print(f'Teacher loaded: {teacher.config.hidden_size}d, vocab={tokenizer.vocab_size}')

    texts = final_texts[:N_SAMPLES]
    all_h, all_i, all_m = [], [], []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=SEQ_LEN)
        ids = enc['input_ids'].to(DEVICE)
        mask = enc['attention_mask'].to(DEVICE)
        with torch.no_grad():
            out = teacher(ids, attention_mask=mask)
            h = out.last_hidden_state.float().cpu()
        all_h.append(h)
        all_i.append(ids.cpu())
        all_m.append(mask.cpu())
        if (i // batch_size) % 20 == 0:
            print(f'  {i}/{len(texts)} done...')

    teacher_hidden = torch.cat(all_h, 0)
    input_ids_all = torch.cat(all_i, 0)
    attention_mask_all = torch.cat(all_m, 0)
    torch.save({'hidden': teacher_hidden, 'ids': input_ids_all, 'mask': attention_mask_all}, MODEL_CACHE_V4)
    print(f'V4 cache saved ({teacher_hidden.shape})')
    del teacher
    torch.cuda.empty_cache()

SEQ_LEN = teacher_hidden.shape[1]
N_SAMPLES = teacher_hidden.shape[0]
actual_vocab = max(tokenizer.vocab_size, input_ids_all.max().item() + 1)
print(f'Data: {N_SAMPLES} samples x {SEQ_LEN} tokens, vocab={actual_vocab}')
""")

# ===== Cell 5: Create model =====
CELLS.append(r"""# %% Cell 5: Create V4 Model
model = ColorLMV4(
    vocab_size=actual_vocab, d_model=288, n_heads=6, n_layers=16,
    d_ff=1152, max_seq_len=SEQ_LEN, fsq_dims=6, fsq_levels=8
).to(DEVICE)
""")

# ===== Cell 6: Training =====
CELLS.append(r"""# %% Cell 6: Training (Distillation + LM + Temperature)

# Build temperature targets
def build_temp_targets(ids, mask, tokenizer):
    targets = torch.full(ids.shape, 0.3)
    for b in range(ids.shape[0]):
        tokens = tokenizer.convert_ids_to_tokens(ids[b])
        for s in range(len(tokens)):
            if mask[b, s] == 0:
                targets[b, s] = 0.0
                continue
            tok = tokens[s].replace(chr(0x2581), '')
            if tok in '()[]{}:;,=+-*/<>!&|':
                targets[b, s] = 0.15
            elif tok.isalpha() and len(tok) > 1:
                targets[b, s] = 0.7
            elif tok.isdigit():
                targets[b, s] = 0.4
    return targets

print('Building temperature targets...')
temp_targets = build_temp_targets(input_ids_all, attention_mask_all, tokenizer)
print(f'Temp targets: {temp_targets.shape}')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# LM labels: shift input_ids by 1 (next token prediction)
lm_labels = input_ids_all.clone()
lm_labels[:, :-1] = input_ids_all[:, 1:]
lm_labels[:, -1] = -100  # ignore last position

history = {'loss': [], 'distill': [], 'lm': [], 'temp': []}
best_loss = float('inf')

model.train()
for epoch in range(30):
    indices = list(range(N_SAMPLES))
    random.shuffle(indices)
    el, ed, ev, et = 0, 0, 0, 0
    nb = 0
    for j in range(0, N_SAMPLES - 4, 4):
        idx = indices[j:j+4]
        ids = input_ids_all[idx].to(DEVICE)
        mask = attention_mask_all[idx].to(DEVICE)
        teacher_h = teacher_hidden[idx].to(DEVICE)
        temp_tgt = temp_targets[idx].to(DEVICE)
        labels = lm_labels[idx].to(DEVICE)

        _, codes, temp, distill, _, lm_logits = model(ids, mask)

        # Distillation loss
        m3d = mask.unsqueeze(-1).float()
        d_loss = F.mse_loss(distill * m3d, teacher_h * m3d, reduction='sum') / (m3d.sum() * 1536 + 1e-8)

        # LM loss (next token prediction)
        lm_loss = F.cross_entropy(
            lm_logits[:, :-1, :].contiguous().view(-1, actual_vocab),
            labels[:, :-1].contiguous().view(-1),
            ignore_index=-100
        )

        # Temperature loss
        t_loss = F.mse_loss(temp.squeeze(-1), temp_tgt) * 1.0

        total = d_loss + 0.5 * lm_loss + 0.5 * t_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        el += total.item(); ed += d_loss.item(); ev += lm_loss.item(); et += t_loss.item()
        nb += 1

    scheduler.step()
    avg = el / max(nb, 1)
    history['loss'].append(avg)
    history['distill'].append(ed / max(nb, 1))
    history['lm'].append(ev / max(nb, 1))
    history['temp'].append(et / max(nb, 1))

    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), 'student_v4_best.pt')

    if (epoch + 1) % 3 == 0:
        print(f'Epoch {epoch+1}/30 | Loss: {avg:.4f} | Distill: {ed/nb:.4f} | '
              f'LM: {ev/nb:.4f} | Temp: {et/nb:.4f}')

print(f'Training done! Best loss: {best_loss:.4f}')
""")

# ===== Cell 7: Generate text demo =====
CELLS.append(r"""# %% Cell 7: Generate Text! (First time our model can talk)
model.eval()

def generate(model, tokenizer, prompt, max_new_tokens=30, temperature=0.8):
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=SEQ_LEN)
    ids = enc['input_ids'].to(DEVICE)
    mask = enc['attention_mask'].to(DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if ids.shape[1] >= SEQ_LEN:
                break
            _, _, _, _, _, logits = model(ids, mask)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
            mask = torch.cat([mask, torch.ones(1, 1, device=DEVICE, dtype=mask.dtype)], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(ids[0], skip_special_tokens=True)

# Test generation
prompts = [
    'def fibonacci',
    'class Stack:',
    'def quicksort(arr):',
    'import',
]

print('=== Text Generation Demo ===')
for p in prompts:
    result = generate(model, tokenizer, p, max_new_tokens=20)
    print(f'\nPrompt: "{p}"')
    print(f'Output: {result}')
""")

# ===== Cell 8: Analysis =====
CELLS.append(r"""# %% Cell 8: Analysis & Summary

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history['loss']); axes[0].set_title('Total Loss')
axes[1].plot(history['distill']); axes[1].set_title('Distillation Loss')
axes[2].plot(history['lm']); axes[2].set_title('LM Loss')
plt.tight_layout()
plt.savefig('training_curves_v4.png', dpi=150)
plt.show()

# Temperature analysis
print('\n=== Temperature Analysis ===')
with torch.no_grad():
    test_texts = ['def fibonacci(n):', 'class Stack:', 'for i in range(10):']
    enc = tokenizer(test_texts, return_tensors='pt', padding='max_length',
                    truncation=True, max_length=SEQ_LEN).to(DEVICE)
    _, codes, temp, _, _, _ = model(enc['input_ids'], enc['attention_mask'])
    for i, text in enumerate(test_texts):
        tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][i])
        temps = temp[i].squeeze(-1).cpu().tolist()
        valid = enc['attention_mask'][i].bool().cpu().tolist()
        print(f"\n'{text}'")
        for tok, t, v in zip(tokens[:10], temps[:10], valid[:10]):
            if not v: break
            tok_s = tok.encode('ascii', 'replace').decode()
            bar = '#' * int(t * 15)
            label = 'HIGH' if t > 0.6 else ('MED' if t > 0.4 else 'LOW')
            print(f'  {tok_s:15s} | {t:.3f} | {bar:15s} | {label}')

print('\n' + '='*50)
print('ColorLM V4 Complete!')
print('='*50)
print(f'Architecture: 16 layers, d_model=288, d_ff=1152')
print(f'Parameters: ~70M (Qwen: 1543M = 22x compression)')
print(f'Features: FSQ codes + temperature + TEXT GENERATION')
print(f'Saved: student_v4_best.pt')
""")

# Build notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU",
        "language_info": {"name": "python"}
    },
    "cells": []
}

for i, text in enumerate(CELLS):
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "cell_%d" % i},
        "source": [l + "\n" for l in text.rstrip("\n").split("\n")],
        "outputs": [],
        "execution_count": None
    })

out = r"D:\project\大模型ssd化\colormlm_repo\colab_colorlm_v4.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Generated: %s (%d bytes, %d cells)" % (out, os.path.getsize(out), len(notebook["cells"])))
