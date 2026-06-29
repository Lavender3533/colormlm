# -*- coding: utf-8 -*-
import os, torch, time
from safetensors import safe_open

MODEL_DIR = r"D:\project\大模型ssd化\models\qwen3-coder"
shard = "model-00001-of-00016.safetensors"
inp = os.path.join(MODEL_DIR, shard)
key = "model.layers.0.mlp.experts.0.gate_proj.weight"

# Load one weight tensor
with safe_open(inp, framework="pt", device="cpu") as f:
    w = f.get_tensor(key).float()

print("Weight shape:", w.shape, "dtype:", w.dtype)
print("Range:", w.min().item(), "to", w.max().item())
print("Std:", w.std().item())
print()

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()

def mse(a, b):
    return ((a - b) ** 2).mean().item()

results = []

# Method 1: Uniform FSQ with different levels
for n in [16, 32, 64, 128, 256]:
    mn, mx = w.min().item(), w.max().item()
    normed = (w - mn) / (mx - mn)
    codes = torch.clamp((normed * n).long(), 0, n - 1)
    recon = codes.float() / float(n - 1) * (mx - mn) + mn
    cos = cosine(w, recon)
    err = mse(w, recon)
    bits = torch.tensor([n]).log2().ceil().item()
    results.append(("Uniform FSQ " + str(n) + " levels", cos, err, bits))

# Method 2: Per-channel min-max (different scale per output channel)
for bits in [4, 8]:
    n = 2 ** bits
    ch_min = w.min(dim=1, keepdim=True).values
    ch_max = w.max(dim=1, keepdim=True).values
    normed = (w - ch_min) / (ch_max - ch_min + 1e-10)
    codes = torch.clamp((normed * n).long(), 0, n - 1)
    recon = codes.float() / float(n - 1) * (ch_max - ch_min) + ch_min
    cos = cosine(w, recon)
    err = mse(w, recon)
    results.append(("Per-channel " + str(bits) + "-bit", cos, err, bits))

# Method 3: Block-wise quantization (group size 128)
for bits in [4, 8]:
    n = 2 ** bits
    gs = 128
    w_flat = w.flatten()
    pad = (gs - len(w_flat) % gs) % gs
    w_pad = torch.cat([w_flat, torch.zeros(pad)])
    blocks = w_pad.reshape(-1, gs)
    b_min = blocks.min(dim=1, keepdim=True).values
    b_max = blocks.max(dim=1, keepdim=True).values
    normed = (blocks - b_min) / (b_max - b_min + 1e-10)
    codes = torch.clamp((normed * n).long(), 0, n - 1)
    recon_blocks = codes.float() / float(n - 1) * (b_max - b_min) + b_min
    recon = recon_blocks.flatten()[:len(w_flat)].reshape(w.shape)
    cos = cosine(w, recon)
    err = mse(w, recon)
    results.append(("Block-128 " + str(bits) + "-bit", cos, err, bits))

# Method 4: Symmetric quantization (centered around 0)
for bits in [4, 8]:
    n = 2 ** bits
    abs_max = w.abs().max().item()
    scale = abs_max / (n // 2 - 1)
    codes = torch.clamp((w / scale).round().long(), -(n // 2), n // 2 - 1)
    recon = codes.float() * scale
    cos = cosine(w, recon)
    err = mse(w, recon)
    results.append(("Symmetric " + str(bits) + "-bit", cos, err, bits))

# Method 5: Residual FSQ (two-pass)
for n in [16, 32]:
    mn, mx = w.min().item(), w.max().item()
    normed = (w - mn) / (mx - mn)
    c1 = torch.clamp((normed * n).long(), 0, n - 1)
    r1 = c1.float() / float(n - 1) * (mx - mn) + mn
    residual = w - r1
    mn2, mx2 = residual.min().item(), residual.max().item()
    c2 = torch.clamp(((residual - mn2) / (mx2 - mn2 + 1e-10) * n).long(), 0, n - 1)
    recon = r1 + c2.float() / float(n - 1) * (mx2 - mn2) + mn2
    cos = cosine(w, recon)
    err = mse(w, recon)
    results.append(("Residual FSQ " + str(n) + "x" + str(n), cos, err, bits * 2))

# Print results
print("Method".ljust(25), "Cosine".ljust(10), "MSE".ljust(15), "Bits/weight")
print("-" * 70)
for name, cos, err, bits in sorted(results, key=lambda x: -x[1]):
    print(name.ljust(25), str(round(cos, 6)).ljust(10), str(err).ljust(15), bits)
