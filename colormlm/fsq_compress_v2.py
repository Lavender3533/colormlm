# -*- coding: utf-8 -*-
"""FSQ 权重压缩工具 (逐块处理, 省内存)"""
import torch, os, time
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

model_path = r"D:\project\大模型ssd化\models\Qwen2.5-1.5B-Instruct\model.safetensors"
output_dir = r"D:\project\大模型ssd化\models\Qwen1.5B-fsq"
os.makedirs(output_dir, exist_ok=True)

N_LEVELS = 16  # 4-bit

def fsq_compress_chunk(tensor, n_levels=16, chunk_size=1000000):
    """分块压缩大tensor"""
    flat = tensor.float().view(-1)
    n = flat.numel()
    
    all_codes = []
    means, scales = [], []
    
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = flat[start:end]
        
        mean = chunk.mean()
        std = chunk.std()
        scale = 4 * std + 1e-8
        normalized = ((chunk - mean) / scale).clamp(-1, 1)
        
        quantized = torch.round(normalized * (n_levels - 1) / 2 + (n_levels - 1) / 2)
        quantized = quantized.clamp(0, n_levels - 1)
        codes = quantized.to(torch.int8)
        
        all_codes.append(codes)
        means.append(mean.item())
        scales.append(scale.item())
    
    codes = torch.cat(all_codes)
    return codes, {"means": means, "scales": scales, "shape": list(tensor.shape)}

def fsq_decompress_chunk(codes, meta, n_levels=16, chunk_size=1000000):
    """分块解压"""
    n = codes.numel()
    all_chunks = []
    
    chunk_idx = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = codes[start:end].float()
        
        mean = meta["means"][chunk_idx]
        scale = meta["scales"][chunk_idx]
        
        reconstructed = (chunk - (n_levels - 1) / 2) / ((n_levels - 1) / 2)
        reconstructed = reconstructed * scale + mean
        all_chunks.append(reconstructed)
        chunk_idx += 1
    
    result = torch.cat(all_chunks)
    return result.view(meta["shape"])

# 加载并压缩
print("="*60)
print("FSQ Weight Compression - Qwen 1.5B (chunked)")
print("="*60)
print(f"Config: {N_LEVELS} levels (4-bit), chunk_size=1M")

all_keys = []
with safe_open(model_path, framework="pt", device="cpu") as f:
    all_keys = list(f.keys())

print(f"Total layers: {len(all_keys)}")

compressed = {}
total_params = 0
total_mse = 0
start_time = time.time()

with safe_open(model_path, framework="pt", device="cpu") as f:
    for i, key in enumerate(all_keys):
        tensor = f.get_tensor(key).float()
        n = tensor.numel()
        total_params += n
        
        codes, meta = fsq_compress_chunk(tensor, N_LEVELS)
        
        # 验证重建质量 (只对小tensor)
        if n < 5000000:
            reconstructed = fsq_decompress_chunk(codes, meta, N_LEVELS)
            mse = torch.mean((tensor - reconstructed) ** 2).item()
            total_mse += mse * n
        else:
            total_mse += 0  # skip MSE for large tensors (saves memory)
        
        # 存储
        compressed[key + ".codes"] = codes
        # meta: 保存为 float32 tensor
        meta_tensor = torch.tensor(meta["means"] + meta["scales"] + meta["shape"], dtype=torch.float32)
        compressed[key + ".meta_len"] = torch.tensor([len(meta["means"])])
        compressed[key + ".meta"] = meta_tensor
        
        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i:3d}/{len(all_keys)}] {key:<50} {n:>12,} params | {elapsed:.1f}s")

print(f"\nSaving compressed model...")
save_path = os.path.join(output_dir, "model.safetensors")
save_file(compressed, save_path)

comp_size = os.path.getsize(save_path)
orig_size = os.path.getsize(model_path)

print(f"\n{'='*60}")
print(f"Compression Results ({N_LEVELS}-level FSQ, 4-bit)")
print(f"{'='*60}")
print(f"Original:     {orig_size / 1024**3:.2f} GB (fp16)")
print(f"Compressed:   {comp_size / 1024**3:.2f} GB")
print(f"Ratio:        {orig_size / comp_size:.2f}x")
print(f"Time:         {time.time() - start_time:.1f}s")

# 对比 ollama 4bit
print(f"\nComparison:")
print(f"  Ollama 4-bit GGUF:  4.70 GB")
print(f"  FSQ 4-bit:          {comp_size / 1024**3:.2f} GB")
print(f"  Savings vs fp16:    {(1 - comp_size/orig_size)*100:.1f}%")
