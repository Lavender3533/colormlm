import torch
from safetensors import safe_open
import sys

log = open(r"D:\project\大模型ssd化\colormlm\out.txt", "w", encoding="utf-8")

model_path = r"D:\project\大模型ssd化\models\Qwen2.5-1.5B-Instruct\model.safetensors"
log.write("Loading...\n"); log.flush()
with safe_open(model_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    log.write("Keys: %d\n" % len(keys)); log.flush()
    t = f.get_tensor(keys[0])
    flat = t.float().view(-1)
    mean, std = flat.mean(), flat.std()
    scale = 4 * std + 1e-8
    normalized = ((flat - mean) / scale).clamp(-1, 1)
    codes = torch.round(normalized * 7.5 + 7.5).clamp(0, 15).to(torch.int8)
    recon = (codes.float() - 7.5) / 7.5 * scale + mean
    mse = torch.mean((flat - recon) ** 2).item()
    log.write("MSE: %.8f\n" % mse); log.flush()
    log.write("OK!\n"); log.flush()
log.close()
