"""
训练脚本 v3 — 快速原型 (CPU 友好)

目标: 在 3-5 分钟内完成训练，验证迭代修正的效果
"""

import os, sys, random, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from colormlm.model import ColorLM

CONFIG = {
    "vocab_size": 5000,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "d_ff": 512,
    "n_codebooks": 4,
    "codebook_size": 64,
    "codebook_dim": 32,
    "max_seq_len": 64,
    "mask_ratio": 0.3,
    "batch_size": 16,
    "lr": 1e-3,
    "epochs": 60,
    "log_every": 2,
    "save_every": 30,
    "n_samples": 200,
}


# ============================================================
# 数据
# ============================================================

CODE_PATTERNS = [
    "def {f}(a,b):\n  if a>b: return a\n  return b",
    "def {f}(n):\n  r=[]\n  for i in range(n):\n    r.append(i*i)\n  return r",
    "class {C}:\n  def __init__(self):\n    self.data=[]\n  def add(self,x):\n    self.data.append(x)",
    "def {f}(arr):\n  if len(arr)<=1: return arr\n  m=len(arr)//2\n  return {f}(arr[:m])+{f}(arr[m:])",
    "for i in range(10):\n  if i%2==0:\n    print(i)",
    "x = [i**2 for i in range(20) if i%3==0]",
    "def {f}(s):\n  d={}\n  for c in s:\n    d[c]=d.get(c,0)+1\n  return d",
    "try:\n  r={f}()\nexcept:\n  r=None",
    "while True:\n  line=input()\n  if not line: break\n  print(line)",
    "with open('f.txt') as f:\n  data=f.read()\n  lines=data.split('\\n')",
]

NAMES = ["sort","find","merge","split","calc","run","init","load","save","test",
         "parse","eval","exec","main","loop","step","next","prev","push","pop"]
UPPER = ["Stack","Queue","Tree","Graph","Cache","Pool","Hub","Node","Link","Map"]


def gen_sample():
    p = random.choice(CODE_PATTERNS)
    return p.replace("{f}", random.choice(NAMES)).replace("{C}", random.choice(UPPER))


def build_vocab(samples):
    chars = set()
    for s in samples:
        chars.update(s)
    chars = sorted(chars)
    c2i = {'<pad>':0, '<mask>':1}
    for i, c in enumerate(chars):
        c2i[c] = i + 2
    i2c = {v:k for k,v in c2i.items()}
    return c2i, i2c, len(c2i)


def encode(text, c2i, maxlen):
    ids = [c2i.get(c,0) for c in text[:maxlen]]
    return ids + [0]*(maxlen - len(ids))


# ============================================================
# 训练
# ============================================================
def train():
    print("="*50)
    print("  ColorLM v3 (fast prototype)")
    print("="*50)

    samples = [gen_sample() for _ in range(CONFIG["n_samples"])]
    c2i, i2c, vsz = build_vocab(samples)
    encoded = [encode(s, c2i, CONFIG["max_seq_len"]) for s in samples]

    print(f"Vocab: {vsz} | Samples: {len(samples)}")

    cfg = CONFIG.copy()
    cfg["vocab_size"] = vsz

    model = ColorLM(
        vocab_size=vsz, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        n_codebooks=cfg["n_codebooks"], codebook_size=cfg["codebook_size"],
        codebook_dim=cfg["codebook_dim"], max_seq_len=cfg["max_seq_len"],
    )
    model.token_pred_head = nn.Linear(cfg["d_model"], vsz + 1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])

    print("\nTraining...\n")

    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        random.shuffle(encoded)
        total_loss = 0
        nb = 0

        for i in range(0, len(encoded), cfg["batch_size"]):
            batch = encoded[i:i+cfg["batch_size"]]
            if len(batch) < cfg["batch_size"]:
                batch += random.choices(encoded, k=cfg["batch_size"]-len(batch))

            target = torch.tensor(batch, dtype=torch.long)

            # Mask
            masked_batch, mask_pos_batch = [], []
            for ids in batch:
                ids = list(ids)
                mask_pos = []
                for j in range(len(ids)):
                    if ids[j] != 0 and random.random() < cfg["mask_ratio"]:
                        mask_pos.append(j)
                        ids[j] = 1
                masked_batch.append(ids)
                mask_pos_batch.append(mask_pos)

            masked = torch.tensor(masked_batch, dtype=torch.long)

            # Forward
            tok_emb = model.token_embed(masked)
            pos = torch.arange(cfg["max_seq_len"]).unsqueeze(0)
            x = tok_emb + model.pos_embed(pos)
            for layer in model.layers:
                x = layer(x)
            x = model.final_norm(x)

            logits = model.token_pred_head(x)

            # Mask tensor
            mt = torch.zeros_like(target, dtype=torch.bool)
            for b, positions in enumerate(mask_pos_batch):
                for p in positions:
                    mt[b, p] = True

            # Loss
            if mt.any():
                loss = F.cross_entropy(logits[mt], target[mt])
            else:
                loss = torch.tensor(0.0, requires_grad=True)

            # VQ loss
            with torch.no_grad():
                t_emb = model.token_embed(target) + model.pos_embed(pos)
            _, _, vq_loss = model.vq(t_emb)
            loss = loss + 0.05 * vq_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            nb += 1

        scheduler.step()

        if epoch % cfg["log_every"] == 0 or epoch == 1:
            # Accuracy check
            model.eval()
            with torch.no_grad():
                test_ids = encoded[0]
                test_target = torch.tensor([test_ids])
                masked_t, mask_pos_t = [], []
                ids_copy = list(test_ids)
                for j in range(len(ids_copy)):
                    if ids_copy[j] != 0 and random.random() < 0.4:
                        mask_pos_t.append(j)
                        ids_copy[j] = 1
                masked_t = torch.tensor([ids_copy])
                t_emb = model.token_embed(masked_t)
                t_pos = torch.arange(cfg["max_seq_len"]).unsqueeze(0)
                tx = t_emb + model.pos_embed(t_pos)
                for layer in model.layers:
                    tx = layer(tx)
                tx = model.final_norm(tx)
                t_logits = model.token_pred_head(tx)

                correct = sum(1 for p in mask_pos_t if t_logits[0,p].argmax().item() == test_ids[p])
                acc = correct / max(len(mask_pos_t), 1)

            avg = total_loss / max(nb, 1)
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d} | Loss {avg:.4f} | Acc {acc:.0%} | LR {lr:.6f}")

        if epoch % cfg["save_every"] == 0:
            sp = Path(__file__).parent / "data" / f"v3_e{epoch}.pt"
            torch.save({"model_state": model.state_dict(), "config": cfg,
                        "char2id": c2i, "id2char": i2c}, sp)
            print(f"  -> saved {sp.name}")

    # Final save
    sp = Path(__file__).parent / "data" / "v3_final.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg,
                "char2id": c2i, "id2char": i2c}, sp)
    print(f"\nDone! saved {sp.name}")

    # Temperature analysis
    print(f"\n{'='*50}")
    print("  Temperature Analysis")
    print(f"{'='*50}\n")

    model.eval()
    test = "def quicksort(arr):"
    tids = torch.tensor([encode(test, c2i, cfg["max_seq_len"])])
    with torch.no_grad():
        te = model.token_embed(tids)
        tp = torch.arange(cfg["max_seq_len"]).unsqueeze(0)
        tx = te + model.pos_embed(tp)
        for layer in model.layers:
            tx = layer(tx)
        tx = model.final_norm(tx)
        temp = model.temperature_head(tx)

    print(f"Input: '{test}'\n")
    for i, c in enumerate(test):
        t = temp[0, i, 0].item()
        bar = "#" * int(t * 25)
        print(f"  {c:>4}  {t:.3f}  {bar}")

    return model, c2i, i2c


if __name__ == "__main__":
    train()