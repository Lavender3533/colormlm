# -*- coding: utf-8 -*-
"""Analyze Qwen 1.5B weight distributions to estimate FSQ compression potential"""
import torch
import numpy as np
from safetensors import safe_open

model_path = r"D:\project\大模型ssd化\models\Qwen2.5-1.5B-Instruct\model.safetensors"

print("Loading weights...")
tensors = {}
with safe_open(model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

print(f"Total keys: {len(tensors)}")

# Analyze each weight matrix
total_params = 0
total_size_bytes = 0
layer_stats = []

for key, tensor in tensors.items():
    n_params = tensor.numel()
    total_params += n_params
    total_size_bytes += n_params * 2  # float16 = 2 bytes

    # Flatten for analysis
    flat = tensor.float().view(-1).numpy()

    stats = {
        "name": key,
        "shape": list(tensor.shape),
        "params": n_params,
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }
    layer_stats.append(stats)

print(f"\nTotal parameters: {total_params:,}")
print(f"Total size (fp16): {total_size_bytes / 1024**3:.2f} GB")

# Estimate FSQ compression
# If we use 8 levels per dimension, each dimension stores log2(8) = 3 bits
# For a weight matrix of shape [M, N], we can:
# Option A: Treat each row as a vector, FSQ encode it
# Option B: Treat each scalar weight, FSQ encode it (like quantization)

# Let's see how well different quantization levels would work
print("\n" + "="*60)
print("Weight Distribution Analysis")
print("="*60)

# Sample a few important layers
important_keys = [k for k in tensors.keys() if "embed" in k or "q_proj" in k or "gate_proj" in k][:8]

for key in important_keys:
    tensor = tensors[key]
    flat = tensor.float().view(-1).numpy()
    std = flat.std()
    # Normalize to [-1, 1]
    if std > 0:
        normalized = flat / (4 * std)  # 4-sigma normalization
        normalized = np.clip(normalized, -1, 1)
    else:
        normalized = flat

    # How much of the distribution is captured by N levels?
    for levels in [4, 8, 16, 32]:
        bins = np.linspace(-1, 1, levels + 1)
        digitized = np.digitize(normalized, bins) - 1
        digitized = np.clip(digitized, 0, levels - 1)
        # Reconstruct
        reconstructed = (digitized + 0.5) / levels * 2 - 1
        reconstructed = reconstructed * (4 * std)
        # MSE
        mse = np.mean((flat - reconstructed) ** 2)
        snr = 10 * np.log10(np.var(flat) / (mse + 1e-10))

    bits_per_weight = np.log2(levels)
    size_gb = total_params * bits_per_weight / 8 / 1024**3

# Summary table
print(f"\n{'Levels':<10} {'Bits/w':<10} {'Size (GB)':<12} {'Compression':<15}")
print("-"*50)
for levels in [4, 8, 16, 32, 64, 128, 256]:
    bits = np.log2(levels)
    size = total_params * bits / 8 / 1024**3
    ratio = (total_size_bytes / 1024**3) / size
    print(f"{levels:<10} {bits:<10.1f} {size:<12.2f} {ratio:<15.1f}x")

# Check if weights are roughly Gaussian (good for FSQ)
print(f"\n{'='*60}")
print("Weight Distribution Shape (are they Gaussian?)")
print("="*60)
for key in important_keys[:4]:
    tensor = tensors[key]
    flat = tensor.float().view(-1).numpy()
    # Kurtosis (Gaussian = 3)
    kurt = float(np.mean((flat - flat.mean())**4) / flat.std()**4)
    skew = float(np.mean((flat - flat.mean())**3) / flat.std()**3)
    print(f"  {key:<45} kurtosis={kurt:.2f} skew={skew:.3f}")
