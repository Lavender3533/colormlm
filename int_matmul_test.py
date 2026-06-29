# -*- coding: utf-8 -*-
import torch, time

M, K, N = 1, 2048, 768
block_size = 128
n_blocks = K // block_size

x = torch.randn(M, K)
codes = torch.randint(0, 256, (N, K), dtype=torch.uint8)
b_min = torch.randn(n_blocks) * 0.1
b_max = b_min + torch.rand(n_blocks) * 0.2

# Expand scale/offset to full K dimension
scale_full = ((b_max - b_min) / 255.0).repeat_interleave(block_size)
offset_full = b_min.repeat_interleave(block_size)

def method1_dequant(x, codes, scale, offset):
    w = codes.float() * scale.unsqueeze(0) + offset.unsqueeze(0)
    return x @ w.T

def method2_prescale(x, codes, scale, offset):
    # Pre-scale x, pre-compute offset, then int matmul
    xs = x * scale.unsqueeze(0)
    off = (x * offset.unsqueeze(0)).sum()
    return xs @ codes.float().T + off

# Warmup
for _ in range(5):
    r1 = method1_dequant(x, codes, scale_full, offset_full)
    r2 = method2_prescale(x, codes, scale_full, offset_full)

# Benchmark
N_iter = 200
t0 = time.time()
for _ in range(N_iter):
    r1 = method1_dequant(x, codes, scale_full, offset_full)
t1 = time.time()
ms1 = (t1 - t0) / N_iter * 1000

t0 = time.time()
for _ in range(N_iter):
    r2 = method2_prescale(x, codes, scale_full, offset_full)
t1 = time.time()
ms2 = (t1 - t0) / N_iter * 1000

diff = (r1 - r2).abs().max().item()
print(f"Method 1 (dequant+matmul): {ms1:.3f} ms")
print(f"Method 2 (pre-scale+int):  {ms2:.3f} ms")
print(f"Speedup: {ms1/ms2:.2f}x")
print(f"Max diff: {diff:.6f}")
print()

# Test with larger matrix (realistic expert size)
M2, K2, N2 = 2, 2048, 768
x2 = torch.randn(M2, K2)
codes2 = torch.randint(0, 256, (N2, K2), dtype=torch.uint8)

t0 = time.time()
for _ in range(N_iter):
    r1 = method1_dequant(x2, codes2, scale_full, offset_full)
t1 = time.time()
ms1 = (t1 - t0) / N_iter * 1000

t0 = time.time()
for _ in range(N_iter):
    r2 = method2_prescale(x2, codes2, scale_full, offset_full)
t1 = time.time()
ms2 = (t1 - t0) / N_iter * 1000

print(f"Larger [{M2},{K2}] @ [{N2},{K2}]:")
print(f"Method 1: {ms1:.3f} ms")
print(f"Method 2: {ms2:.3f} ms")
print(f"Speedup: {ms1/ms2:.2f}x")
