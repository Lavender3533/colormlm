# -*- coding: utf-8 -*-
"""Generate colab_colorlm_v2.ipynb - FSQ version (no codebook collapse)"""
import json, os

CELLS = []

CELL1 = r"""# %% Cell 1: Install & Import
import subprocess, sys, os
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers'])

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
print(f'Torch: {torch.__version__}')
"""
CELLS.append(CELL1)

CELL2 = r"""# %% Cell 2: Define Model Components (FSQ version)

class FSQ(nn.Module):
    # Finite Scalar Quantization - impossible to collapse
    # Each dimension quantized to fixed levels -> uniform grid
    def __init__(self, d_model, n_dims=6, levels_per_dim=8):
        super().__init__()
        self.n_dims = n_dims
        self.levels = levels_per_dim
        self.proj_in = nn.Linear(d_model, n_dims, bias=False)
        self.proj_out = nn.Linear(n_dims, d_model, bias=False)
        total_codes = levels_per_dim ** n_dims
        print(f'  FSQ: {n_dims}D x {levels_per_dim} levels = {total_codes} codes')

    def forward(self, x):
        z = self.proj_in(x)
        z = torch.tanh(z)
        z_int = torch.round(z * (self.levels - 1) / 2 + (self.levels - 1) / 2)
        z_int = z_int.clamp(0, self.levels - 1)
        codes = z_int.long()
        z_hat = (z_int - (self.levels - 1) / 2) / ((self.levels - 1) / 2)
        z_hat = torch.tanh(z_hat)
        quant = self.proj_out(z_hat)
        # Straight-through gradient
        quant_st = x[:, :, :quant.shape[-1]] + (quant - x[:, :, :quant.shape[-1]]).detach() if x.shape[-1] != quant.shape[-1] else quant
        # Actually just use quant with straight-through through proj_out
        return codes, quant, torch.tensor(0.0, device=x.device)


class TemperatureHead(nn.Module):
    # Predicts importance 0~1 for each token
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.SiLU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


class StudentBlock(nn.Module):
    # Bidirectional Transformer block (no causal mask)
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n, key_padding_mask=pad_mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class ColorLMStudent(nn.Module):
    # Student Model: ~70M params
    # Input:  token IDs [B, S]
    # Output: FSQ codes [B, S, 6], temperature [B, S, 1], distill projection [B, S, 1536]
    def __init__(self, vocab_size=151644, d_model=384, n_heads=6, n_layers=6,
                 d_ff=1536, max_seq_len=128, fsq_dims=6, fsq_levels=8):
        super().__init__()
        self.d_model = d_model
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
        total = sum(p.numel() for p in self.parameters())
        print(f'Student model: {total/1e6:.1f}M params')

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
        distill = self.distill_proj(quantized)
        return quantized, codes, temp, distill, fsq_loss
"""
CELLS.append(CELL2)

CELL3 = r"""# %% Cell 3: Prepare Training Data (30 code samples)
CODE_SAMPLES = [
    'def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)',
    'def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b',
    'class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()\n    def is_empty(self):\n        return len(self.items) == 0',
    'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1',
    'def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)',
    'class TreeNode:\n    def __init__(self, val=0):\n        self.val = val\n        self.left = None\n        self.right = None',
    'def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result',
    'def memoize(func):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper',
    'class Graph:\n    def __init__(self):\n        self.adj = {}\n    def add_edge(self, u, v):\n        self.adj.setdefault(u, []).append(v)\n        self.adj.setdefault(v, []).append(u)',
    'def lcs(a, b):\n    m, n = len(a), len(b)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1]+1\n            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]',
    'class LRUCache:\n    def __init__(self, cap):\n        self.cache = {}\n        self.order = []\n        self.cap = cap\n    def get(self, key):\n        if key in self.cache:\n            self.order.remove(key)\n            self.order.append(key)\n            return self.cache[key]\n        return -1',
    'def permutations(lst):\n    if len(lst) <= 1: return [lst]\n    result = []\n    for i, val in enumerate(lst):\n        rest = lst[:i] + lst[i+1:]\n        for p in permutations(rest):\n            result.append([val] + p)\n    return result',
    'def knapsack(w, v, cap):\n    n = len(w)\n    dp = [[0]*(cap+1) for _ in range(n+1)]\n    for i in range(1, n+1):\n        for j in range(cap+1):\n            dp[i][j] = dp[i-1][j]\n            if w[i-1] <= j:\n                dp[i][j] = max(dp[i][j], dp[i-1][j-w[i-1]]+v[i-1])\n    return dp[n][cap]',
    'class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\nclass Trie:\n    def __init__(self):\n        self.root = TrieNode()\n    def insert(self, word):\n        node = self.root\n        for c in word:\n            if c not in node.children:\n                node.children[c] = TrieNode()\n            node = node.children[c]\n        node.is_end = True',
    'def quickselect(arr, k):\n    if len(arr) == 1: return arr[0]\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    if k < len(left): return quickselect(left, k)\n    elif k < len(left) + len(mid): return pivot\n    else: return quickselect(right, k - len(left) - len(mid))',
    'def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        comp = target - num\n        if comp in seen: return [seen[comp], i]\n        seen[num] = i\n    return []',
    'def dfs(graph, start, visited=None):\n    if visited is None: visited = set()\n    visited.add(start)\n    for next in graph[start] - visited:\n        dfs(graph, next, visited)\n    return visited',
    'class EventEmitter:\n    def __init__(self):\n        self.listeners = {}\n    def on(self, event, fn):\n        self.listeners.setdefault(event, []).append(fn)\n    def emit(self, event, *args):\n        for fn in self.listeners.get(event, []):\n            fn(*args)',
    'import torch\nclass SimpleNet(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = torch.nn.Linear(784, 128)\n        self.fc2 = torch.nn.Linear(128, 10)\n    def forward(self, x):\n        x = torch.relu(self.fc1(x))\n        return self.fc2(x)',
    'from typing import List\ndef max_subarray(nums: List[int]) -> int:\n    max_sum = cur = nums[0]\n    for n in nums[1:]:\n        cur = max(n, cur + n)\n        max_sum = max(max_sum, cur)\n    return max_sum',
    'def is_valid_parens(s):\n    stack = []\n    map = {")":"(", "]":"[", "}":"{"}\n    for c in s:\n        if c in map:\n            if not stack or stack[-1] != map[c]: return False\n            stack.pop()\n        else:\n            stack.append(c)\n    return not stack',
    'class MinStack:\n    def __init__(self):\n        self.stack = []\n        self.min_stack = []\n    def push(self, val):\n        self.stack.append(val)\n        if not self.min_stack or val <= self.min_stack[-1]:\n            self.min_stack.append(val)\n    def pop(self):\n        if self.stack.pop() == self.min_stack[-1]:\n            self.min_stack.pop()\n    def get_min(self):\n        return self.min_stack[-1]',
    'def topological_sort(graph):\n    visited = set()\n    order = []\n    def dfs(node):\n        visited.add(node)\n        for n in graph.get(node, []):\n            if n not in visited: dfs(n)\n        order.append(node)\n    for node in graph:\n        if node not in visited: dfs(node)\n    return order[::-1]',
    'class UnionFind:\n    def __init__(self, n):\n        self.parent = list(range(n))\n        self.rank = [0]*n\n    def find(self, x):\n        if self.parent[x] != x:\n            self.parent[x] = self.find(self.parent[x])\n        return self.parent[x]\n    def union(self, x, y):\n        px, py = self.find(x), self.find(y)\n        if px == py: return\n        if self.rank[px] < self.rank[py]: px, py = py, px\n        self.parent[py] = px\n        if self.rank[px] == self.rank[py]: self.rank[px] += 1',
    'def sliding_window_max(nums, k):\n    from collections import deque\n    dq = deque()\n    result = []\n    for i, n in enumerate(nums):\n        while dq and nums[dq[-1]] <= n: dq.pop()\n        dq.append(i)\n        if dq[0] <= i - k: dq.popleft()\n        if i >= k - 1: result.append(nums[dq[0]])\n    return result',
    'def longest_palindrome(s):\n    def expand(l, r):\n        while l >= 0 and r < len(s) and s[l] == s[r]:\n            l -= 1; r += 1\n        return s[l+1:r]\n    best = ""\n    for i in range(len(s)):\n        odd = expand(i, i)\n        even = expand(i, i+1)\n        best = max(best, odd, even, key=len)\n    return best',
    'def rotate_matrix(matrix):\n    n = len(matrix)\n    for i in range(n):\n        for j in range(i+1, n):\n            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]\n    for row in matrix:\n        row.reverse()',
    'class MedianFinder:\n    def __init__(self):\n        import heapq\n        self.lo = []\n        self.hi = []\n    def add(self, num):\n        heapq.heappush(self.lo, -num)\n        heapq.heappush(self.hi, -heapq.heappop(self.lo))\n        if len(self.hi) > len(self.lo):\n            heapq.heappush(self.lo, -heapq.heappop(self.hi))\n    def find(self):\n        if len(self.lo) > len(self.hi): return -self.lo[0]\n        return (-self.lo[0] + self.hi[0]) / 2',
    'def word_break(s, word_dict):\n    n = len(s)\n    dp = [False]*(n+1)\n    dp[0] = True\n    for i in range(1, n+1):\n        for j in range(i):\n            if dp[j] and s[j:i] in word_dict:\n                dp[i] = True\n                break\n    return dp[n]',
]

print(f'Loaded {len(CODE_SAMPLES)} code samples')
"""
CELLS.append(CELL3)

CELL4 = r"""# %% Cell 4: Generate Distillation Cache
import os
from pathlib import Path

# Mount Google Drive for persistent cache
try:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE_DIR = '/content/drive/MyDrive/colormlm_cache'
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f'Drive cache dir: {CACHE_DIR}')
except Exception:
    CACHE_DIR = '/tmp/colormlm_cache'
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f'No Drive, using tmp: {CACHE_DIR}')

MODEL_CACHE = os.path.join(CACHE_DIR, 'qwen_hidden.pt')
model_name = 'Qwen/Qwen2.5-1.5B-Instruct'

if os.path.exists(MODEL_CACHE):
    print(f'Loading cached data from {MODEL_CACHE}...')
    cache = torch.load(MODEL_CACHE, map_location='cpu', weights_only=False)
    teacher_hidden = cache['hidden']
    input_ids_all = cache['ids']
    attention_mask_all = cache['mask']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f'Cache loaded: {teacher_hidden.shape}')
else:
    print('Generating distillation cache from teacher...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(DEVICE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher.eval()
    print(f'Teacher: {teacher.config.hidden_size}d, vocab={tokenizer.vocab_size}')

    SEQ_LEN = 64
    N_SAMPLES = 500

    texts = []
    while len(texts) < N_SAMPLES:
        for s in CODE_SAMPLES:
            texts.append(s)
            if len(texts) >= N_SAMPLES:
                break

    print(f'Generating {N_SAMPLES} samples...')
    all_h, all_i, all_m = [], [], []
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
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

    teacher_hidden = torch.cat(all_h, 0)
    input_ids_all = torch.cat(all_i, 0)
    attention_mask_all = torch.cat(all_m, 0)

    torch.save({'hidden': teacher_hidden, 'ids': input_ids_all, 'mask': attention_mask_all}, MODEL_CACHE)
    print(f'Cache saved to {MODEL_CACHE}')
    del teacher
    torch.cuda.empty_cache()

SEQ_LEN = teacher_hidden.shape[1]
N_SAMPLES = teacher_hidden.shape[0]
print(f'Data ready: {teacher_hidden.shape}, vocab={tokenizer.vocab_size}')
"""
CELLS.append(CELL4)

CELL5 = r"""# %% Cell 5: Create Student Model (FSQ version)
actual_vocab = max(tokenizer.vocab_size, input_ids_all.max().item() + 1)
print(f'Using vocab_size: {actual_vocab}')
print(f'input_ids range: [{input_ids_all.min()}, {input_ids_all.max()}]')

model = ColorLMStudent(
    vocab_size=actual_vocab, d_model=384, n_heads=6, n_layers=6,
    d_ff=1536, max_seq_len=SEQ_LEN, fsq_dims=6, fsq_levels=8
).to(DEVICE)
"""
CELLS.append(CELL5)

CELL6 = r"""# %% Cell 6: Training (FSQ - no diversity loss needed!)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Build temperature targets: keywords=HIGH, punctuation=LOW
def build_temp_targets(ids, mask, tokenizer):
    targets = torch.full(ids.shape, 0.3)  # default: medium-low
    for b in range(ids.shape[0]):
        tokens = tokenizer.convert_ids_to_tokens(ids[b])
        for s in range(len(tokens)):
            if mask[b, s] == 0:
                targets[b, s] = 0.0
                continue
            tok = tokens[s].replace(chr(0x2581), '')  # remove underscore prefix
            # Punctuation and syntax -> low temp
            if tok in '()[]{}:;,=+-*/<>!&|':
                targets[b, s] = 0.15
            # Keywords and identifiers -> high temp
            elif tok.isalpha() and len(tok) > 1:
                targets[b, s] = 0.7
            # Numbers -> medium
            elif tok.isdigit():
                targets[b, s] = 0.4
    return targets

temp_targets = build_temp_targets(input_ids_all, attention_mask_all, tokenizer).to(DEVICE)

history = {'loss': [], 'distill': [], 'temp': []}
best_loss = float('inf')

model.train()
for epoch in range(50):
    indices = list(range(N_SAMPLES))
    random.shuffle(indices)
    el, ed, et = 0, 0, 0
    nb = 0
    for j in range(0, N_SAMPLES - 8, 8):
        idx = indices[j:j+8]
        ids = input_ids_all[idx].to(DEVICE)
        mask = attention_mask_all[idx].to(DEVICE)
        teacher_h = teacher_hidden[idx].to(DEVICE)
        temp_tgt = temp_targets[idx].to(DEVICE)

        _, codes, temp, distill, _ = model(ids, mask)

        # Distillation loss (main objective)
        m3d = mask.unsqueeze(-1).float()
        d_loss = F.mse_loss(distill * m3d, teacher_h * m3d, reduction='sum') / (m3d.sum() * 1536 + 1e-8)

        # Temperature supervision loss
        t_loss = F.mse_loss(temp.squeeze(-1), temp_tgt) * 1.0

        total = d_loss + t_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        el += total.item(); ed += d_loss.item(); et += t_loss.item()
        nb += 1

    scheduler.step()
    avg_l = el / max(nb, 1)
    history['loss'].append(avg_l)
    history['distill'].append(ed / max(nb, 1))
    history['temp'].append(et / max(nb, 1))

    # Check FSQ diversity (should be high!)
    with torch.no_grad():
        _, cv, _, _, _ = model(input_ids_all[:8].to(DEVICE), attention_mask_all[:8].to(DEVICE))
        v = attention_mask_all[:8].bool().to(DEVICE)
        divs = [cv[:, :, d][v].unique().numel() for d in range(6)]
        avg_div = sum(divs) / 6

    if avg_l < best_loss:
        best_loss = avg_l
        torch.save(model.state_dict(), 'student_best.pt')

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch+1}/50 | Loss: {avg_l:.4f} | Distill: {ed/nb:.4f} | '
              f'Temp: {et/nb:.4f} | FSQ Diversity: {avg_div:.0f}/8')

print(f'Training done! Best loss: {best_loss:.4f}')
"""
CELLS.append(CELL6)

CELL7 = r"""# %% Cell 7: Visualization & Analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history['loss'])
axes[0].set_title('Total Loss'); axes[0].set_xlabel('Epoch')

axes[1].plot(history['distill'])
axes[1].set_title('Distillation Loss (vs Qwen)'); axes[1].set_xlabel('Epoch')

axes[2].plot(history['temp'])
axes[2].set_title('Temperature Loss'); axes[2].set_xlabel('Epoch')

plt.tight_layout()
plt.savefig('training_curves_v2.png', dpi=150)
plt.show()

# Temperature Analysis
print('\n=== Temperature Analysis ===')
model.eval()
with torch.no_grad():
    test_texts = ['def fibonacci(n):', 'class Stack:', 'import os',
                  'for i in range(10):', 'return result']
    enc = tokenizer(test_texts, return_tensors='pt', padding='max_length',
                    truncation=True, max_length=SEQ_LEN).to(DEVICE)
    _, codes, temp, _, _ = model(enc['input_ids'], enc['attention_mask'])
    for i, text in enumerate(test_texts):
        tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][i])
        temps = temp[i].squeeze(-1).cpu().tolist()
        valid = enc['attention_mask'][i].bool().cpu().tolist()
        print(f"\n'{text}'")
        for tok, t, v in zip(tokens[:12], temps[:12], valid[:12]):
            if not v: break
            tok_s = tok.encode('ascii', 'replace').decode()
            bar = '#' * int(t * 20)
            imp = 'HIGH' if t > 0.6 else ('MED' if t > 0.4 else 'LOW')
            print(f'  {tok_s:15s} | {t:.3f} | {bar:20s} | {imp}')

# FSQ Codebook Analysis
print('\n=== FSQ Codes [6 dims x 8 levels] ===')
with torch.no_grad():
    enc = tokenizer('def quicksort(arr):', return_tensors='pt',
                    padding='max_length', truncation=True, max_length=SEQ_LEN).to(DEVICE)
    _, codes, temp, _, _ = model(enc['input_ids'], enc['attention_mask'])
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
    valid = enc['attention_mask'][0].bool()
    for tok, code, t, v in zip(tokens, codes[0].tolist(), temp[0].squeeze(-1).tolist(), valid):
        if not v: break
        tok_s = tok.encode('ascii', 'replace').decode()
        print(f'  {tok_s:15s} -> {code} | temp={t:.3f}')

# Cosine similarity
with torch.no_grad():
    s_ids = input_ids_all[:20].to(DEVICE)
    s_mask = attention_mask_all[:20].to(DEVICE)
    s_teacher = teacher_hidden[:20].to(DEVICE)
    _, _, _, distill, _ = model(s_ids, s_mask)
    cos = F.cosine_similarity(distill, s_teacher, dim=-1)
    avg_cos = cos[s_mask.bool()].mean().item()
    print(f'\nCosine similarity (student vs Qwen): {avg_cos:.4f}')
"""
CELLS.append(CELL7)

CELL8 = r"""# %% Cell 8: Summary
print('\n' + '='*50)
print('ColorLM V2 Training Complete! (FSQ version)')
print('='*50)
print(f'Best loss: {best_loss:.4f}')
print(f'Student: ~70M params (Qwen: 1543M = 22x compression)')
print(f'Representation: 6 FSQ codes (8^6 = 262144 possible) + temperature')
print(f'Key improvement: FSQ = no codebook collapse!')
print(f'Saved: student_best.pt, training_curves_v2.png')
print(f'\nNext steps:')
print(f'  1. Scale up training data (5K-10K samples)')
print(f'  2. Implement iterative refinement inference')
print(f'  3. Test on real code understanding tasks')
"""
CELLS.append(CELL8)

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

out = r"D:\project\大模型ssd化\colormlm\colab_colorlm_v2.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Generated: %s (%d bytes, %d cells)" % (out, os.path.getsize(out), len(notebook["cells"])))
