"""
训练脚本 v2 — 同时训练 VQ 码本 + Token 预测 + 温度

改进:
  1. 加入直接的 token 预测 loss（不只是码本预测）
  2. 让温度真正学到"哪些位置需要修正"
  3. 更多训练数据（自动生成代码模式）
"""

import os, sys, json, random, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from colormlm.model import ColorLM


# ============================================================
# 配置
# ============================================================
CONFIG = {
    "vocab_size": 5000,
    "d_model": 192,
    "n_heads": 6,
    "n_layers": 6,
    "d_ff": 768,
    "n_codebooks": 4,
    "codebook_size": 128,
    "codebook_dim": 48,
    "max_seq_len": 128,
    "mask_ratio": 0.3,
    "batch_size": 8,
    "lr": 5e-4,
    "epochs": 100,
    "log_every": 5,
    "save_every": 25,
}


# ============================================================
# 数据: 代码模式生成器（扩充数据集）
# ============================================================

TEMPLATES = [
    "def {name}({params}):\n    {body}",
    "class {name}:\n    def __init__(self{params}):\n        {body}",
    "for {var} in {iterable}:\n    {body}",
    "if {cond}:\n    {body}\nelse:\n    {else_body}",
    "while {cond}:\n    {body}",
    "try:\n    {body}\nexcept {exc}:\n    {handler}",
    "with {ctx} as {var}:\n    {body}",
    "import {module}\nfrom {module} import {item}",
    "result = [{expr} for {var} in {iterable}]",
    "return {name}({args})",
]

NAMES = ["sort", "search", "merge", "split", "find", "count", "filter",
         "map", "reduce", "flatten", "reverse", "unique", "group_by",
         "zip_with", "chain", "pipe", "compose", "memoize", "cache",
         "validate", "parse", "serialize", "encode", "decode", "hash"]

VARIABLES = ["arr", "lst", "data", "items", "result", "value", "key",
             "index", "count", "total", "output", "input", "temp", "buf",
             "stack", "queue", "tree", "graph", "node", "edge", "weight"]

KEYWORDS = ["def", "class", "return", "yield", "import", "from", "if",
            "elif", "else", "for", "while", "try", "except", "with",
            "as", "in", "not", "and", "or", "True", "False", "None",
            "print", "len", "range", "enumerate", "zip", "map", "filter"]

OPERATORS = ["+", "-", "*", "/", "//", "%", "**", "==", "!=", "<", ">",
             "<=", ">=", "+=", "-=", "*=", "/=", "->", ":=", "and", "or", "not"]

EXPRESSIONS = [
    "len({v}) > 0", "{v} is not None", "{v} == {n}", "not {v}",
    "{v}[0]", "{v}[-1]", "{v}:{n}", "sorted({v})",
    "{v}.append({n})", "{v}.pop()", "{v}.copy()", "{v}.items()",
    "sum({v})", "max({v})", "min({v})", "abs({v})",
    "list(range({n}))", "dict()", "set()", "[]",
    "{{k: v for k, v in {v}}}", "tuple({v})",
    "str({v})", "int({v})", "float({v})",
    "{v}.get({n}, 0)", "{v}.update({v})",
    "map(lambda x: x, {v})", "filter(lambda x: x, {v})",
]


def generate_code_sample(max_len=100):
    """生成一个随机代码片段"""
    lines = []
    n_lines = random.randint(3, 12)

    # 通常以函数/类定义开始
    name = random.choice(NAMES)
    params = ", ".join(random.sample(VARIABLES, random.randint(0, 3)))
    template = random.choice(TEMPLATES[:5])  # 偏好函数定义模板

    body_lines = []
    for _ in range(n_lines):
        keyword = random.choice(KEYWORDS[:15])
        var = random.choice(VARIABLES)
        expr = random.choice(EXPRESSIONS).format(v=var, n=random.randint(1, 100))
        op = random.choice(OPERATORS)
        indent = "    " * random.randint(1, 2)

        line_type = random.choice(["assign", "keyword", "expr", "comment"])
        if line_type == "assign":
            body_lines.append(f"{indent}{var} = {expr}")
        elif line_type == "keyword":
            body_lines.append(f"{indent}{keyword} {var}:")
        elif line_type == "expr":
            body_lines.append(f"{indent}{expr}")
        else:
            body_lines.append(f"{indent}# {var} {op} {random.randint(0, 100)}")

    body = "\n".join(body_lines) if body_lines else "pass"
    code = template.format(name=name, params=params, body=body,
                          var=random.choice(VARIABLES),
                          iterable=random.choice(VARIABLES),
                          cond=f"len({random.choice(VARIABLES)}) > 0",
                          else_body="pass", exc="Exception",
                          handler="pass", ctx="open",
                          module=random.choice(NAMES),
                          item=random.choice(NAMES),
                          expr=random.choice(EXPRESSIONS[:10]).format(
                              v=random.choice(VARIABLES), n=random.randint(1, 100)))

    return code[:max_len]


def build_vocab(samples, min_freq=1):
    """构建词表"""
    chars = set()
    for s in samples:
        chars.update(s)
    chars = sorted(chars)
    char2id = {'<pad>': 0, '<mask>': 1}
    for i, c in enumerate(chars):
        char2id[c] = i + 2
    id2char = {v: k for k, v in char2id.items()}
    return char2id, id2char, len(char2id)


def encode(text, char2id, max_len):
    ids = [char2id.get(c, 0) for c in text[:max_len]]
    return ids + [0] * (max_len - len(ids))


def mask_tokens(ids, mask_id, mask_ratio=0.3):
    ids = list(ids)
    mask_positions = []
    for i in range(len(ids)):
        if ids[i] == 0:
            continue
        if random.random() < mask_ratio:
            mask_positions.append(i)
            ids[i] = mask_id
    return ids, mask_positions


# ============================================================
# 训练循环 v2
# ============================================================
def train():
    print("=" * 60)
    print("  ColorLM v2 训练 (VQ + Token Prediction + Temperature)")
    print("=" * 60)

    # 生成训练数据
    print("\n生成训练数据...")
    samples = [generate_code_sample() for _ in range(500)]
    char2id, id2char, vocab_size = build_vocab(samples)
    print(f"词表大小: {vocab_size} (字符级)")
    print(f"训练样本: {len(samples)}")

    # 预编码所有样本
    encoded = [encode(s, char2id, CONFIG["max_seq_len"]) for s in samples]

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

    # 添加 token 预测头
    model.token_pred_head = nn.Linear(cfg["d_model"], cfg["vocab_size"] + 1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])

    # 训练
    print(f"\n{'='*60}")
    print("  开始训练")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        random.shuffle(encoded)

        total_loss = 0
        total_tok_loss = 0
        total_vq_loss = 0
        n_batches = 0

        for i in range(0, len(encoded), cfg["batch_size"]):
            batch = encoded[i:i + cfg["batch_size"]]
            if len(batch) < cfg["batch_size"]:
                batch += random.choices(encoded, k=cfg["batch_size"] - len(batch))

            target_ids = torch.tensor(batch, dtype=torch.long)

            # 遮盖
            masked_batch, mask_pos_batch = [], []
            for ids in batch:
                masked, mask_pos = mask_tokens(ids, 1, cfg["mask_ratio"])
                masked_batch.append(masked)
                mask_pos_batch.append(mask_pos)

            masked_ids = torch.tensor(masked_batch, dtype=torch.long)

            # 前向 (获取 hidden state)
            tok_emb = model.token_embed(masked_ids)
            pos = torch.arange(cfg["max_seq_len"]).unsqueeze(0).expand(len(batch), -1)
            x = tok_emb + model.pos_embed(pos)
            for layer in model.layers:
                x = layer(x)
            x = model.final_norm(x)

            # Token 预测
            token_logits = model.token_pred_head(x)  # [B, S, vocab+1]

            # 构建 mask tensor
            mask_tensor = torch.zeros_like(target_ids, dtype=torch.bool)
            for b, positions in enumerate(mask_pos_batch):
                for p in positions:
                    mask_tensor[b, p] = True

            # Token prediction loss (只在被遮盖的位置)
            if mask_tensor.any():
                masked_logits = token_logits[mask_tensor]
                masked_targets = target_ids[mask_tensor]
                tok_loss = F.cross_entropy(masked_logits, masked_targets)
            else:
                tok_loss = torch.tensor(0.0)

            # VQ loss (通过原始 embedding)
            with torch.no_grad():
                target_emb = model.token_embed(target_ids) + model.pos_embed(pos)
            _, _, vq_loss = model.vq(target_emb)

            # 总 loss
            loss = tok_loss + 0.1 * vq_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tok_loss += tok_loss.item()
            total_vq_loss += vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
            n_batches += 1

        scheduler.step()

        if epoch % cfg["log_every"] == 0 or epoch == 1:
            avg_loss = total_loss / max(n_batches, 1)
            avg_tok = total_tok_loss / max(n_batches, 1)
            avg_vq = total_vq_loss / max(n_batches, 1)
            lr = scheduler.get_last_lr()[0]

            # 计算准确率
            model.eval()
            with torch.no_grad():
                test_ids = encoded[0]
                test_target = torch.tensor([test_ids])
                masked, mask_pos = mask_tokens(test_ids, 1, 0.5)
                test_masked = torch.tensor([masked])

                tok_emb = model.token_embed(test_masked)
                p = torch.arange(cfg["max_seq_len"]).unsqueeze(0)
                x = tok_emb + model.pos_embed(p)
                for layer in model.layers:
                    x = layer(x)
                x = model.final_norm(x)
                logits = model.token_pred_head(x)

                # 准确率
                correct = 0
                total = 0
                for pos in mask_pos:
                    pred = logits[0, pos].argmax().item()
                    if pred == test_ids[pos]:
                        correct += 1
                    total += 1
                acc = correct / max(total, 1)

            print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Tok: {avg_tok:.4f} | "
                  f"VQ: {avg_vq:.4f} | "
                  f"Acc: {acc:.1%} | "
                  f"LR: {lr:.6f}")

        if epoch % cfg["save_every"] == 0:
            save_path = Path(__file__).parent / "data" / f"colormlm_v2_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": cfg,
                "char2id": char2id,
                "id2char": id2char,
            }, save_path)
            print(f"  -> saved: {save_path.name}")

    # 最终保存
    save_path = Path(__file__).parent / "data" / "colormlm_v2_final.pt"
    torch.save({
        "epoch": cfg["epochs"],
        "model_state": model.state_dict(),
        "config": cfg,
        "char2id": char2id,
        "id2char": id2char,
    }, save_path)
    print(f"\nDone! saved: {save_path}")

    # --- 温度分析 ---
    print(f"\n{'='*60}")
    print("  Temperature Analysis")
    print(f"{'='*60}\n")

    model.eval()
    test_code = "def quicksort(arr):"
    test_ids = encode(test_code, char2id, cfg["max_seq_len"])
    test_tensor = torch.tensor([test_ids])

    with torch.no_grad():
        tok_emb = model.token_embed(test_tensor)
        p = torch.arange(cfg["max_seq_len"]).unsqueeze(0)
        x = tok_emb + model.pos_embed(p)
        for layer in model.layers:
            x = layer(x)
        x = model.final_norm(x)
        temp = model.temperature_head(x)

    print(f"Input: '{test_code}'")
    print(f"{'char':>6} {'temp':>8} {'vis'}")
    print("-" * 40)
    for i, c in enumerate(test_code):
        t = temp[0, i, 0].item()
        bar = "#" * int(t * 30)
        print(f"  {c:>4}  {t:.3f}  {bar}")

    return model, char2id, id2char


if __name__ == "__main__":
    train()