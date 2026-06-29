# -*- coding: utf-8 -*-
"""
RGB + FSQ 权重压缩实验
Step 1: 权重 → 连续 RGB 映射
Step 2: RGB → FSQ 颜色压缩
Step 3: 测量压缩率和重建误差
"""
import torch
import numpy as np
from safetensors import safe_open

model_path = r"D:\project\大模型ssd化\models\Qwen2.5-1.5B-Instruct\model.safetensors"

print("="*60)
print("RGB + FSQ 权重压缩实验")
print("="*60)

# Step 1: Load weights
print("\n[1] Loading Qwen 1.5B weights...")
tensors = {}
with safe_open(model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

total_params = sum(t.numel() for t in tensors.values())
print(f"    Total params: {total_params:,}")
print(f"    Original size (fp16): {total_params * 2 / 1024**3:.2f} GB")

# Step 2: Analyze weight ranges for RGB mapping
print("\n[2] Analyzing weight distributions...")
all_weights = []
for key, tensor in tensors.items():
    all_weights.append(tensor.float().view(-1))
all_weights = torch.cat(all_weights)
print(f"    Min: {all_weights.min():.4f}")
print(f"    Max: {all_weights.max():.4f}")
print(f"    Mean: {all_weights.mean():.4f}")
print(f"    Std: {all_weights.std():.4f}")

# Normalize to [0, 1] range
w_min = all_weights.min().item()
w_max = all_weights.max().item()
w_range = w_max - w_min

# Step 3: RGB mapping
# Map each weight to 3 components (like splitting decimal digits)
# But more practically: split the [0,1] range into 3 channels
print("\n[3] RGB Mapping Strategy:")
print("    每个权重 -> 3个分量 (R, G, B)")
print("    R = 权重的高位 (粗略信息)")
print("    G = 权重的中位 (中等细节)")
print("    B = 权重的低位 (精细细节)")

def weight_to_rgb(w, w_min, w_range):
    """Convert weight to continuous RGB representation"""
    # Normalize to [0, 1]
    normalized = (w - w_min) / w_range
    # Split into 3 components using fractal-like decomposition
    # R: the integer part when scaled to [0, 255]
    # G: the fractional part * 255
    # B: the second fractional part * 255
    scaled = normalized * 255.0
    r = scaled  # coarse
    g = (scaled - torch.floor(scaled)) * 255.0  # fine
    b = (g - torch.floor(g)) * 255.0  # ultra-fine
    return torch.stack([r, g, b], dim=-1)  # shape: [..., 3]

def rgb_to_weight(rgb, w_min, w_range):
    """Convert RGB back to weight (approximate)"""
    r = rgb[..., 0]
    reconstructed = r / 255.0 * w_range + w_min
    return reconstructed

# Test on a small layer first
print("\n[4] Testing on a small layer...")
test_key = "model.layers.0.self_attn.q_proj.weight"
test_tensor = tensors[test_key].float()
print(f"    Layer: {test_key}")
print(f"    Shape: {test_tensor.shape}")
print(f"    Params: {test_tensor.numel():,}")

# RGB mapping
rgb = weight_to_rgb(test_tensor, w_min, w_range)
print(f"    RGB shape: {rgb.shape}")
print(f"    R range: [{rgb[..., 0].min():.1f}, {rgb[..., 0].max():.1f}]")
print(f"    G range: [{rgb[..., 1].min():.1f}, {rgb[..., 2].max():.1f}]")

# Reconstruct from R channel only (coarse)
reconstructed = rgb_to_weight(rgb, w_min, w_range)
mse_coarse = torch.mean((test_tensor - reconstructed) ** 2).item()
print(f"    Reconstruction MSE (R only): {mse_coarse:.8f}")

# Step 4: FSQ compression of RGB space
print("\n[5] FSQ Compression of RGB space:")

def fsq_compress(rgb, n_colors=256):
    """Compress RGB to limited color palette using FSQ"""
    # Normalize RGB to [0, 1]
    rgb_norm = rgb / 255.0
    # Quantize each channel to sqrt(n_colors) levels
    levels = int(np.sqrt(n_colors))
    quantized = torch.round(rgb_norm * (levels - 1)) / (levels - 1)
    quantized = quantized.clamp(0, 1)
    # Color indices (for storage)
    r_idx = (quantized[..., 0] * (levels - 1)).long()
    g_idx = (quantized[..., 1] * (levels - 1)).long()
    b_idx = (quantized[..., 2] * (levels - 1)).long()
    color_idx = r_idx * levels * levels + g_idx * levels + b_idx
    return quantized * 255.0, color_idx, levels

for n_colors in [64, 256, 1024, 4096]:
    compressed_rgb, color_idx, levels = fsq_compress(rgb, n_colors)
    reconstructed = rgb_to_weight(compressed_rgb, w_min, w_range)
    mse = torch.mean((test_tensor - reconstructed) ** 2).item()
    
    # Storage: color index per weight + color table
    bits_per_idx = np.log2(n_colors)
    storage_gb = total_params * bits_per_idx / 8 / 1024**3
    color_table_kb = n_colors * 3 / 1024
    
    print(f"    {n_colors:>5} colors | {bits_per_idx:.1f} bits/w | "
          f"Storage: {storage_gb:.2f} GB | Color table: {color_table_kb:.1f} KB | "
          f"MSE: {mse:.6f}")

# Step 5: Compare with direct quantization
print("\n[6] Comparison with direct quantization:")
for bits in [2, 3, 4, 5, 6, 8]:
    levels = 2 ** bits
    # Direct scalar quantization
    w_norm = (test_tensor - w_min) / w_range
    quantized = torch.round(w_norm * (levels - 1)) / (levels - 1)
    reconstructed = quantized * w_range + w_min
    mse = torch.mean((test_tensor - reconstructed) ** 2).item()
    storage_gb = total_params * bits / 8 / 1024**3
    print(f"    {bits}-bit direct | {levels:>4} levels | "
          f"Storage: {storage_gb:.2f} GB | MSE: {mse:.6f}")

print("\n" + "="*60)
print("Conclusion")
print("="*60)
print("RGB+FSQ 和直接量化的对比将在上面的数据中显示。")
print("如果 RGB+FSQ 的 MSE 更低，说明 RGB 映射保留了更多信息。")
