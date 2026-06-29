# -*- coding: utf-8 -*-
"""RGB+FSQ 权重压缩实验 (逐层处理, 省内存)"""
import torch
import numpy as np
from safetensors import safe_open

model_path = r"D:\project\大模型ssd化\models\Qwen2.5-1.5B-Instruct\model.safetensors"

print("="*60)
print("RGB + FSQ 权重压缩实验")
print("="*60)

# 先统计所有层的 min/max (不加载到float32)
print("\n[1] Scanning weight ranges...")
w_min, w_max = float('inf'), float('-inf')
key_shapes = {}
with safe_open(model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        t = f.get_tensor(key)
        key_shapes[key] = t.shape
        mn = t.float().min().item()
        mx = t.float().max().item()
        if mn < w_min: w_min = mn
        if mx > w_max: w_max = mx
        del t

w_range = w_max - w_min
total_params = sum(np.prod(s) for s in key_shapes.values())
print(f"    Total params: {total_params:,}")
print(f"    Original (fp16): {total_params * 2 / 1024**3:.2f} GB")
print(f"    Range: [{w_min:.4f}, {w_max:.4f}]")

# 逐层做 RGB+FSQ 实验
print("\n[2] Testing RGB+FSQ on layers...")

# 选几个代表性层
test_keys = [k for k in key_shapes.keys() if "weight" in k and "embed" not in k][:6]

results = {n: [] for n in [64, 256, 1024]}
direct_results = {b: [] for b in [3, 4, 5]}

with safe_open(model_path, framework="pt", device="cpu") as f:
    for key in test_keys:
        tensor = f.get_tensor(key).float()
        n = tensor.numel()

        # 直接量化 (baseline)
        w_norm = (tensor - w_min) / w_range
        for bits in [3, 4, 5]:
            levels = 2 ** bits
            q = torch.round(w_norm * (levels - 1)) / (levels - 1)
            q = q.clamp(0, 1)
            recon = q * w_range + w_min
            mse = torch.mean((tensor - recon) ** 2).item()
            direct_results[bits].append(mse * n)  # weighted MSE

        # RGB + FSQ
        # 权重 -> 连续 [0, 255] -> RGB 3通道
        scaled = w_norm * 255.0
        r = scaled
        g = (scaled - torch.floor(scaled)) * 255.0
        b_ch = (g - torch.floor(g)) * 255.0

        for n_colors in [64, 256, 1024]:
            levels_c = int(np.cbrt(n_colors))  # 3D cube root
            # 量化RGB
            r_q = torch.round(r / 255.0 * (levels_c - 1)) / (levels_c - 1) * 255.0
            g_q = torch.round(g / 255.0 * (levels_c - 1)) / (levels_c - 1) * 255.0
            b_q = torch.round(b_ch / 255.0 * (levels_c - 1)) / (levels_c - 1) * 255.0
            # 从 R 通道重建
            recon_norm = r_q / 255.0
            recon = recon_norm * w_range + w_min
            mse = torch.mean((tensor - recon) ** 2).item()
            results[n_colors].append(mse * n)

        # 纯 R 通道 (3个通道各存一份, 取平均)
        r_only = torch.round(r) / 255.0
        g_only = torch.round(g) / 255.0
        b_only = torch.round(b_ch) / 255.0
        # 3通道平均重建
        avg_norm = (r_only + g_only / 255.0 + b_only / (255.0 * 255.0)) / 1.0
        # 简单用 R 通道
        recon_rgb = r_only * w_range + w_min
        mse_rgb = torch.mean((tensor - recon_rgb) ** 2).item()

        del tensor

print("\n[3] Results (Weighted MSE across layers):")
print(f"    {'Method':<25} {'Storage':<12} {'Weighted MSE':<15}")
print("    " + "-"*52)

# Direct quantization
for bits in [3, 4, 5]:
    total_mse = sum(direct_results[bits])
    avg_mse = total_mse / total_params
    storage = total_params * bits / 8 / 1024**3
    print(f"    Direct {bits}-bit{'':<16} {storage:.2f} GB{'':<5} {avg_mse:.8f}")

# RGB+FSQ
for n_colors in [64, 256, 1024]:
    total_mse = sum(results[n_colors])
    avg_mse = total_mse / total_params
    bits = np.log2(n_colors)
    storage = total_params * bits / 8 / 1024**3
    print(f"    RGB+FSQ {n_colors} colors{'':<11} {storage:.2f} GB{'':<5} {avg_mse:.8f}")

print("\n" + "="*60)
print("Conclusion")
print("="*60)
print("如果 RGB+FSQ MSE < 直接量化 MSE, 说明 RGB 映射有优势")
print("如果 RGB+FSQ MSE > 直接量化 MSE, 说明直接量化更好")
