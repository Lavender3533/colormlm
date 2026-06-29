# -*- coding: utf-8 -*-
import torch, time

M, K, N = 2, 2048, 768
block_size = 128
n_blocks = K // block_size
N_experts = 8

x = torch.randn(M, K)
all_codes = [torch.randint(0, 256, (N, K), dtype=torch.uint8) for _ in range(N_experts)]
all_min = [torch.randn(n_blocks) * 0.1 for _ in range(N_experts)]
all_max = [m + torch.rand(n_blocks) * 0.2 for m in all_min]

def method_old(x, codes_list, mins, maxs):
    """Current: dequantize each expert individually"""
    results = []
    for codes, b_min, b_max in zip(codes_list, mins, maxs):
        scale = ((b_max - b_min) / 255.0).repeat_interleave(block_size)
        offset = b_min.repeat_interleave(block_size)
        w = codes.float() * scale.unsqueeze(0) + offset.unsqueeze(0)
        results.append(x @ w.T)
    return results

def method_batch_convert(x, codes_list, mins, maxs):
    """New: batch convert all codes to float first, then matmul"""
    # Stack all codes: [8, 768, 2048]
    stacked = torch.stack(codes_list)  # [8, N, K] uint8
    # Batch convert to float: one big operation
    stacked_f = stacked.float()  # [8, N, K] float32
    
    results = []
    for i in range(len(codes_list)):
        scale = ((maxs[i] - mins[i]) / 255.0).repeat_interleave(block_size)
        offset = mins[i].repeat_interleave(block_size)
        w = stacked_f[i] * scale.unsqueeze(0) + offset.unsqueeze(0)
        results.append(x @ w.T)
    return results

def method_prescale_all(x, codes_list, mins, maxs):
    """Best: pre-scale x once for all experts, then int matmul"""
    # For each expert, the scale/offset is different
    # But we can still batch the matmul
    
    results = []
    for codes, b_min, b_max in zip(codes_list, mins, maxs):
        scale = ((b_max - b_min) / 255.0).repeat_interleave(block_size)
        offset = b_min.repeat_interleave(block_size)
        xs = x * scale.unsqueeze(0)
        off = (x * offset.unsqueeze(0)).sum(dim=1)
        r = xs @ codes.float().T + off.unsqueeze(1)
        results.append(r)
    return results

# Warmup
for _ in range(3):
    method_old(x, all_codes, all_min, all_max)
    method_batch_convert(x, all_codes, all_min, all_max)
    method_prescale_all(x, all_codes, all_min, all_max)

N_iter = 100

t0 = time.time()
for _ in range(N_iter):
    method_old(x, all_codes, all_min, all_max)
t1 = time.time()
print(f"Old (individual dequant): {(t1-t0)/N_iter*1000:.2f} ms / 8 experts")

t0 = time.time()
for _ in range(N_iter):
    method_batch_convert(x, all_codes, all_min, all_max)
t1 = time.time()
print(f"Batch convert:            {(t1-t0)/N_iter*1000:.2f} ms / 8 experts")

t0 = time.time()
for _ in range(N_iter):
    method_prescale_all(x, all_codes, all_min, all_max)
t1 = time.time()
print(f"Pre-scale:                {(t1-t0)/N_iter*1000:.2f} ms / 8 experts")
