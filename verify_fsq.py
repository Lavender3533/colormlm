# -*- coding: utf-8 -*-
from safetensors import safe_open
import torch

fsq = r"D:\project\大模型ssd化\models\qwen3-coder-fsq\model-00001-of-00016.safetensors"
ori = r"D:\project\大模型ssd化\models\qwen3-coder\model-00001-of-00016.safetensors"
k = "model.layers.0.mlp.experts.0.gate_proj.weight"

with safe_open(fsq, framework="pt") as f:
    c = f.get_tensor(k + "_c1")
    m = f.get_tensor(k + "_meta")
    w0, w1 = m[0].item(), m[1].item()
    r = c.float() / 15.0 * (w1 - w0) + w0
    print("FSQ codes:", c.shape, c.dtype)
    print("Meta: min=", w0, "max=", w1)

with safe_open(ori, framework="pt") as f:
    o = f.get_tensor(k)

mse = ((r - o) ** 2).mean().item()
cs = torch.nn.functional.cosine_similarity(
    r.flatten().unsqueeze(0), o.flatten().unsqueeze(0)
).item()
print("Orig mean=", o.mean().item(), "std=", o.std().item())
print("Recon mean=", r.mean().item(), "std=", r.std().item())
print("MSE:", mse)
print("Cosine:", cs)
if cs > 0.99:
    print("Quality: GOOD")
elif cs > 0.95:
    print("Quality: OK")
else:
    print("Quality: POOR")
