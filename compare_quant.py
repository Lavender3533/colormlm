# -*- coding: utf-8 -*-
import torch, os
from safetensors import safe_open

orig_path = r"D:\project\大模型ssd化\models\qwen3-coder\model-00001-of-00016.safetensors"
key = "model.layers.0.mlp.experts.0.gate_proj.weight"

with safe_open(orig_path, framework="pt", device="cpu") as f:
    w_orig = f.get_tensor(key).float()

# Our Block-8bit
quant_path = r"D:\project\大模型ssd化\models\qwen3-coder-bq8\model-00001-of-00016.safetensors"
with safe_open(quant_path, framework="pt", device="cpu") as f:
    codes = f.get_tensor(key + "._bq_codes")
    meta = f.get_tensor(key + "._bq_meta")
    shape_t = f.get_tensor(key + "._bq_shape")
    shape = [shape_t[0].item(), shape_t[1].item()]
    n = codes.shape[0]
    b_min = meta[:n].unsqueeze(1)
    b_max = meta[n:].unsqueeze(1)
    w_bq8 = (codes.float() / 255.0 * (b_max - b_min) + b_min).flatten()[:shape[0]*shape[1]].reshape(shape)

# Simulate Q4_K_M (4-bit block quantization)
def simulate_q4km(w, group_size=32):
    w_flat = w.flatten()
    pad = (group_size - len(w_flat) % group_size) % group_size
    w_pad = torch.cat([w_flat, torch.zeros(pad)])
    blocks = w_pad.reshape(-1, group_size)
    b_min = blocks.min(dim=1, keepdim=True).values
    b_max = blocks.max(dim=1, keepdim=True).values
    scale = (b_max - b_min) / 15
    codes = torch.clamp(((blocks - b_min) / (scale + 1e-10)).round().long(), 0, 15)
    recon = codes.float() * scale + b_min
    return recon.flatten()[:len(w_flat)].reshape(w.shape)

w_q4km = simulate_q4km(w_orig)

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

def mse(a, b):
    return ((a - b) ** 2).mean().item()

cos_bq8 = cosine(w_orig, w_bq8)
cos_q4km = cosine(w_orig, w_q4km)

print("=" * 60)
print("Block-8bit vs Q4_K_M")
print("=" * 60)
print()
print(f"Block-8bit: cosine={cos_bq8:.6f}, MSE={mse(w_orig, w_bq8):.2e}")
print(f"Q4_K_M:     cosine={cos_q4km:.6f}, MSE={mse(w_orig, w_q4km):.2e}")
print()
print("Compression:")
print(f"  Block-8bit: 1.8x (61GB -> 34GB)")
print(f"  Q4_K_M:     4x   (61GB -> 15-18GB)")
print()
print("Speed:")
print(f"  Block-8bit: ~0.07 tok/s (our engine, CPU)")
print(f"  Q4_K_M:     ~14 tok/s (ollama+Vulkan)")
