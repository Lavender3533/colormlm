# -*- coding: utf-8 -*-
"""
FSQ 权重压缩工具
1. 加载模型权重 (逐层)
2. 用 FSQ 压缩每个权重矩阵
3. 保存压缩后的模型
4. 测试重建质量
"""
import torch
import numpy as np
import json, os, time
from safetensors import safe_open
from safetensors.torch import save_file

model_path = r"D:\project\大模型ssd化\models\Qwen2.5-1.5B-Instruct\model.safetensors"
output_dir = r"D:\project\大模型ssd化\models\Qwen1.5B-fsq"
os.makedirs(output_dir, exist_ok=True)

# FSQ 压缩函数
def fsq_compress_weight(weight, n_levels=16):
    """
    压缩单个权重矩阵
    weight: float tensor
    n_levels: 量化级别数 (8=3bit, 16=4bit, 32=5bit)
    返回: 量化后的权重, 缩放参数
    """
    # 4-sigma 归一化
    mean = weight.mean()
    std = weight.std()
    scale = 4 * std + 1e-8
    normalized = (weight - mean) / scale
    normalized = normalized.clamp(-1, 1)
    
    # FSQ: 映射到固定网格
    quantized = torch.round(normalized * (n_levels - 1) / 2 + (n_levels - 1) / 2)
    quantized = quantized.clamp(0, n_levels - 1)
    
    # 存储为 int8 (节省空间)
    codes = quantized.to(torch.int8)
    
    # 重建
    reconstructed = (quantized - (n_levels - 1) / 2) / ((n_levels - 1) / 2)
    reconstructed = reconstructed * scale + mean
    
    return codes, reconstructed, {"mean": mean.item(), "std": std.item(), "scale": scale.item()}

def fsq_decompress_weight(codes, meta, n_levels=16):
    """从 FSQ 码重建权重"""
    quantized = codes.float()
    reconstructed = (quantized - (n_levels - 1) / 2) / ((n_levels - 1) / 2)
    reconstructed = reconstructed * meta["scale"] + meta["mean"]
    return reconstructed

# 加载并压缩
print("="*60)
print("FSQ Weight Compression - Qwen 1.5B")
print("="*60)

N_LEVELS = 16  # 4-bit
BITS_PER_WEIGHT = np.log2(N_LEVELS)

print(f"\nConfig: {N_LEVELS} levels, {BITS_PER_WEIGHT:.1f} bits/weight")

# 读取所有 key
all_keys = []
with safe_open(model_path, framework="pt", device="cpu") as f:
    all_keys = list(f.keys())

print(f"Total layers: {len(all_keys)}")

# 逐层压缩
compressed = {}
total_orig_bytes = 0
total_comp_bytes = 0
total_mse = 0
total_params = 0

print("\nCompressing layers...")
with safe_open(model_path, framework="pt", device="cpu") as f:
    for i, key in enumerate(all_keys):
        tensor = f.get_tensor(key).float()
        n = tensor.numel()
        total_params += n
        
        if tensor.dtype == torch.float16 or tensor.dtype == torch.float32:
            # 压缩权重
            codes, reconstructed, meta = fsq_compress_weight(tensor, N_LEVELS)
            
            # 计算误差
            mse = torch.mean((tensor - reconstructed) ** 2).item()
            total_mse += mse * n
            
            # 存储: codes (int8) + meta
            compressed[key + ".codes"] = codes
            compressed[key + ".meta"] = torch.tensor([meta["mean"], meta["std"], meta["scale"]])
            
            orig_bytes = n * 2  # fp16
            comp_bytes = n * 1 + 12  # int8 + meta
            total_orig_bytes += orig_bytes
            total_comp_bytes += comp_bytes
            
            if i % 50 == 0:
                print(f"  [{i:3d}/{len(all_keys)}] {key:<50} {n:>12,} params | MSE: {mse:.8f}")
        else:
            # 非浮点类型直接保存
            compressed[key] = tensor
            total_orig_bytes += tensor.numel() * 4

print(f"\nSaving compressed model...")
save_path = os.path.join(output_dir, "model.safetensors")
save_file(compressed, save_path)

comp_size = os.path.getsize(save_path)
orig_size = os.path.getsize(model_path)

print(f"\n{'='*60}")
print(f"Compression Results ({N_LEVELS}-level FSQ)")
print(f"{'='*60}")
print(f"Original size:     {orig_size / 1024**3:.2f} GB")
print(f"Compressed size:   {comp_size / 1024**3:.2f} GB")
print(f"Compression ratio: {orig_size / comp_size:.2f}x")
print(f"Weighted MSE:      {total_mse / total_params:.8f}")
print(f"Bits per weight:   {BITS_PER_WEIGHT:.1f}")
print(f"\nSaved to: {save_path}")

# 验证: 重建并测试
print(f"\n{'='*60}")
print(f"Verification: Testing reconstruction quality")
print(f"{'='*60}")

# 随机选几层验证
import random
random.seed(42)
test_keys = [k for k in all_keys if "weight" in k]
test_samples = random.sample(test_keys, min(5, len(test_keys)))

with safe_open(model_path, framework="pt", device="cpu") as f:
    for key in test_samples:
        original = f.get_tensor(key).float()
        codes = compressed[key + ".codes"]
        meta_list = compressed[key + ".meta"]
        meta = {"mean": meta_list[0].item(), "std": meta_list[1].item(), "scale": meta_list[2].item()}
        reconstructed = fsq_decompress_weight(codes, meta, N_LEVELS)
        
        mse = torch.mean((original - reconstructed) ** 2).item()
        cos = torch.nn.functional.cosine_similarity(
            original.view(1, -1), reconstructed.view(1, -1)
        ).item()
        
        print(f"  {key:<50} MSE: {mse:.8f} | Cos: {cos:.6f}")
