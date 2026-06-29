"""
训练脚本 — ColorLM 原型

用法: cd colormlm && python train.py

验证目标:
  1. VQ 码本能学到有意义的离散编码
  2. 温度能区分重要 token 和不重要 token
  3. 并行预测 + 迭代修正能生成连贯文本
"""

import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# 确保能导入 colormlm 包
sys.path.insert(0, str(Path(__file__).parent.parent))
from colormlm.model import ColorLM

# ============================================================
# 配置
# ============================================================
CONFIG = {
    "vocab_size": 5000,        # 词表大小（小规模原型）
    "d_model": 128,            # 隐藏维度
    "n_heads": 4,              # 注意力头数
    "n_layers": 4,             # Transformer 层数
    "d_ff": 512,               # FFN 维度
    "n_codebooks": 4,          # 码本数量 (类似 RGB+Alpha)
    "codebook_size": 128,      # 每个码本的大小
    "codebook_dim": 32,        # 每个码本基向量的维度
    "max_seq_len": 128,        # 最大序列长度
    "mask_ratio": 0.3,         # 训练时遮盖 30% 的 token
    "batch_size": 4,
    "lr": 3e-4,
    "epochs": 50,
    "log_every": 5,
    "save_every": 20,
}


# ============================================================
# 数据: 用 Python 代码片段作为训练数据
# ============================================================
CODE_SAMPLES = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()\n    def is_empty(self):\n        return len(self.items) == 0",
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
    "class TreeNode:\n    def __init__(self, val=0):\n        self.val = val\n        self.left = None\n        self.right = None\n\ndef inorder(root):\n    if not root:\n        return []\n    return inorder(root.left) + [root.val] + inorder(root.right)",
    "def is_palindrome(s):\n    s = s.lower().strip()\n    s = ''.join(c for c in s if c.isalnum())\n    return s == s[::-1]",
    "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
    "def memoize(func):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper",
    "class Graph:\n    def __init__(self):\n        self.adj = {}\n    def add_edge(self, u, v):\n        self.adj.setdefault(u, []).append(v)\n        self.adj.setdefault(v, []).append(u)\n    def bfs(self, start):\n        visited = {start}\n        queue = [start]\n        order = []\n        while queue:\n            node = queue.pop(0)\n            order.append(node)\n            for neighbor in self.adj.get(node, []):\n                if neighbor not in visited:\n                    visited.add(neighbor)\n                    queue.append(neighbor)\n        return order",
    "def lcs(a, b):\n    m, n = len(a), len(b)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if a[i-1] == b[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]",
    "def permutations(lst):\n    if len(lst) <= 1:\n        return [lst]\n    result = []\n    for i, val in enumerate(lst):\n        rest = lst[:i] + lst[i+1:]\n        for p in permutations(rest):\n            result.append([val] + p)\n    return result",
    "import threading\n\nclass ThreadPool:\n    def __init__(self, n_workers):\n        self.tasks = []\n        self.workers = []\n        for _ in range(n_workers):\n            t = threading.Thread(target=self._worker, daemon=True)\n            t.start()\n            self.workers.append(t)\n    def submit(self, func):\n        self.tasks.append(func)\n    def _worker(self):\n        while True:\n            if self.tasks:\n                task = self.tasks.pop(0)\n                task()\n",
    "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n    for i in range(1, n + 1):\n        for w in range(capacity + 1):\n            dp[i][w] = dp[i-1][w]\n            if weights[i-1] <= w:\n                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])\n    return dp[n][capacity]",
    "class LRUCache:\n    def __init__(self, capacity):\n        self.cache = {}\n        self.order = []\n        self.capacity = capacity\n    def get(self, key):\n        if key in self.cache:\n            self.order.remove(key)\n            self.order.append(key)\n            return self.cache[key]\n        return -1\n    def put(self, key, value):\n        if key in self.cache:\n            self.order.remove(key)\n        elif len(self.cache) >= self.capacity:\n            oldest = self.order.pop(0)\n            del self.cache[oldest]\n        self.cache[key] = value\n        self.order.append(key)",
]


def build_vocab(samples):
    """构建简单字符级词表"""
    chars = set()
    for s in samples:
        chars.update(s)
    chars = sorted(chars)
    # 0 = padding, 1 = MASK, 2+ = 实际字符
    char2id = {'<pad>': 0, '<mask>': 1}
    for i, c in enumerate(chars):
        char2id[c] = i + 2
    id2char = {v: k for k, v in char2id.items()}
    return char2id, id2char, len(char2id)


def encode(text, char2id, max_len):
    """编码文本为 token IDs"""
    ids = [char2id.get(c, 0) for c in text[:max_len]]
    # padding
    ids = ids + [0] * (max_len - len(ids))
    return ids


def mask_tokens(ids, mask_id, mask_ratio=0.3):
    """随机遮盖 token，返回遮盖后的 ids 和遮盖位置"""
    ids = list(ids)
    mask_positions = []
    for i in range(len(ids)):
        if ids[i] == 0:  # 跳过 padding
            continue
        if random.random() < mask_ratio:
            mask_positions.append(i)
            ids[i] = mask_id
    return ids, mask_positions


# ============================================================
# 训练循环
# ============================================================
def train():
    print("=" * 60)
    print("  ColorLM 原型训练")
    print("=" * 60)

    # 构建词表
    char2id, id2char, vocab_size = build_vocab(CODE_SAMPLES)
    print(f"\n词表大小: {vocab_size} (字符级)")
    print(f"训练样本数: {len(CODE_SAMPLES)}")

    # 创建模型
    cfg = CONFIG.copy()
    cfg["vocab_size"] = vocab_size
    model = ColorLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        n_codebooks=cfg["n_codebooks"],
        codebook_size=cfg["codebook_size"],
        codebook_dim=cfg["codebook_dim"],
        max_seq_len=cfg["max_seq_len"],
    )

    # 参数量统计
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M")
    print(f"\n配置: {json.dumps(cfg, indent=2)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])

    # 训练
    print(f"\n{'='*60}")
    print("  开始训练")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0
        total_code_loss = 0
        total_temp_loss = 0
        total_vq_loss = 0
        n_batches = 0

        # 简单随机采样（小数据集够用）
        random.shuffle(CODE_SAMPLES)

        for i in range(0, len(CODE_SAMPLES), cfg["batch_size"]):
            batch_texts = CODE_SAMPLES[i:i + cfg["batch_size"]]
            if len(batch_texts) < cfg["batch_size"]:
                batch_texts += random.choices(CODE_SAMPLES, k=cfg["batch_size"] - len(batch_texts))

            # 编码
            batch_ids = [encode(t, char2id, cfg["max_seq_len"]) for t in batch_texts]
            target_ids = torch.tensor(batch_ids, dtype=torch.long)

            # 遮盖
            masked_batch = []
            mask_pos_batch = []
            for ids in batch_ids:
                masked, mask_pos = mask_tokens(ids, 1, cfg["mask_ratio"])  # 1 = <mask>
                masked_batch.append(masked)
                mask_pos_batch.append(mask_pos)

            masked_ids = torch.tensor(masked_batch, dtype=torch.long)

            # 前向传播
            code_logits, temperature, vq_loss = model(masked_ids, target_ids)

            # 码本预测 loss (只在被遮盖的位置计算)
            mask_tensor = torch.zeros_like(target_ids, dtype=torch.bool)
            for b, positions in enumerate(mask_pos_batch):
                for p in positions:
                    mask_tensor[b, p] = True

            # 展开 logits 计算 loss
            # code_logits: [B, S, K, codebook_size]
            # 我们需要 VQ 编码后的 target 码本索引
            # 简化: 用 token embedding 的 VQ 编码作为 target
            with torch.no_grad():
                pos = torch.arange(cfg["max_seq_len"]).unsqueeze(0).expand(target_ids.shape[0], -1)
                target_emb = model.token_embed(target_ids) + model.pos_embed(pos)
                target_codes, _, _ = model.vq(target_emb)
                # target_codes: [B, S, K]

            # Cross-entropy loss for each codebook
            code_loss = 0
            for k in range(cfg["n_codebooks"]):
                logits_k = code_logits[:, :, k, :]  # [B, S, codebook_size]
                targets_k = target_codes[:, :, k]    # [B, S]
                # 只在被遮盖位置计算 loss
                if mask_tensor.any():
                    masked_logits = logits_k[mask_tensor]
                    masked_targets = targets_k[mask_tensor]
                    code_loss += F.cross_entropy(masked_logits, masked_targets)
            code_loss /= cfg["n_codebooks"]

            # 温度 loss: 被遮盖的位置应该有高温度（需要修正）
            # 未被遮盖的位置应该有低温度（已经确定）
            temp_loss = 0
            if mask_tensor.any():
                temp_pred = temperature.squeeze(-1)  # [B, S]
                temp_target = mask_tensor.float()
                temp_loss = F.mse_loss(temp_pred, temp_target)

            # 总 loss
            loss = code_loss + temp_loss + 0.1 * vq_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_code_loss += code_loss.item()
            total_temp_loss += temp_loss.item()
            total_vq_loss += vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
            n_batches += 1

        scheduler.step()

        if epoch % cfg["log_every"] == 0 or epoch == 1:
            avg_loss = total_loss / max(n_batches, 1)
            avg_code = total_code_loss / max(n_batches, 1)
            avg_temp = total_temp_loss / max(n_batches, 1)
            avg_vq = total_vq_loss / max(n_batches, 1)
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Code: {avg_code:.4f} | "
                  f"Temp: {avg_temp:.4f} | "
                  f"VQ: {avg_vq:.4f} | "
                  f"LR: {lr:.6f}")

        if epoch % cfg["save_every"] == 0:
            save_path = Path(__file__).parent / "data" / f"colormlm_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": cfg,
                "char2id": char2id,
                "id2char": id2char,
            }, save_path)
            print(f"  -> 模型已保存: {save_path}")

    # 最终保存
    save_path = Path(__file__).parent / "data" / "colormlm_final.pt"
    torch.save({
        "epoch": cfg["epochs"],
        "model_state": model.state_dict(),
        "config": cfg,
        "char2id": char2id,
        "id2char": id2char,
    }, save_path)
    print(f"\n训练完成! 模型已保存: {save_path}")

    # --------------------------------------------------------
    # 验证: 温度分析
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("  温度分析 — 验证温度是否学到了重要性")
    print(f"{'='*60}\n")

    model.eval()
    test_text = "def quicksort(arr):"
    test_ids = encode(test_text, char2id, cfg["max_seq_len"])
    test_tensor = torch.tensor([test_ids], dtype=torch.long)

    with torch.no_grad():
        _, temp, _ = model(test_tensor, test_tensor)

    print(f"输入: '{test_text}'")
    print(f"{'字符':>6} {'温度':>8} {'可视化'}")
    print("-" * 40)
    for i, c in enumerate(test_text):
        t = temp[0, i, 0].item()
        bar = "█" * int(t * 20)
        print(f"  {c:>4}  {t:.3f}  {bar}")

    return model, char2id, id2char


if __name__ == "__main__":
    train()