# -*- coding: utf-8 -*-
"""FSQ 权重压缩 Qwen 1.5B"""
import torch, os, time
from safetensors import safe_open
from safetensors.torch import save_file

log = open(r"D:\project\大模型ssd化\colormlm\compress_log.txt", "w", encoding="utf-8")

model_path = r"D:\project\大模型ssd化\models\Qwen2.5-1.5B-Instruct\model.safetensors"
output_dir = r"D:\project\大模型ssd化\models\Qwen1.5B-fsq"
os.makedirs(output_dir, exist_ok=True)

N_LEVELS = 16
CHUNK = 500000

def compress(tensor):
    flat = tensor.float().view(-1)
    n = flat.numel()
    codes_list = []
    params = []
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        c = flat[start:end]
        m, s = c.mean(), c.std()
        sc = 4 * s + 1e-8
        norm = ((c - m) / sc).clamp(-1, 1)
        q = torch.round(norm * 7.5 + 7.5).clamp(0, 15).to(torch.int8)
        codes_list.append(q)
        params.append((m.item(), sc.item()))
    return torch.cat(codes_list), params, list(tensor.shape)

def decompress(codes, params, shape):
    n = codes.numel()
    chunks = []
    idx = 0
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        c = codes[start:end].float()
        m, sc = params[idx]
        r = (c - 7.5) / 7.5 * sc + m
        chunks.append(r)
        idx += 1
    return torch.cat(chunks).view(shape)

log.write("=" * 60 + "\n")
log.write("FSQ Weight Compression - Qwen 1.5B\n")
log.write("=" * 60 + "\n")
log.flush()

with safe_open(model_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    log.write("Total keys: %d\n" % len(keys))
    log.flush()

    compressed = {}
    total_params = 0
    total_mse = 0
    t0 = time.time()

    for i, key in enumerate(keys):
        t = f.get_tensor(key).float()
        n = t.numel()
        total_params += n

        codes, params, shape = compress(t)

        # Verify on small tensors
        if n < 2000000:
            recon = decompress(codes, params, shape)
            mse = torch.mean((t - recon) ** 2).item()
            total_mse += mse * n

        # Save compressed
        comp_key = key.replace(".", "|")
        compressed[comp_key + "_codes"] = codes
        # Flatten params: [m0, sc0, m1, sc1, ...] + shape
        flat_params = []
        for m, sc in params:
            flat_params.extend([m, sc])
        flat_params.extend([float(x) for x in shape])
        compressed[comp_key + "_meta"] = torch.tensor(flat_params, dtype=torch.float32)

        if i % 50 == 0:
            elapsed = time.time() - t0
            log.write("  [%d/%d] %s | %d params | %.1fs\n" % (i, len(keys), key, n, elapsed))
            log.flush()

    log.write("\nSaving...\n"); log.flush()
    save_path = os.path.join(output_dir, "model.safetensors")
    save_file(compressed, save_path)

    orig_size = os.path.getsize(model_path)
    comp_size = os.path.getsize(save_path)

    log.write("\n" + "=" * 60 + "\n")
    log.write("Results\n")
    log.write("=" * 60 + "\n")
    log.write("Original:   %.2f GB (fp16)\n" % (orig_size / 1024**3))
    log.write("Compressed: %.2f GB (FSQ 4-bit)\n" % (comp_size / 1024**3))
    log.write("Ratio:      %.2fx\n" % (orig_size / comp_size))
    if total_mse > 0:
        log.write("MSE:        %.8f\n" % (total_mse / total_params))
    log.write("Time:       %.1fs\n" % (time.time() - t0))
    log.flush()

log.close()
