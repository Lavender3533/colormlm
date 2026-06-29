# -*- coding: utf-8 -*-
"""
FSQ 分组量化 + 校准
每 128 个权重共享缩放参数, 用校准数据优化
"""
import torch, time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

log = open(r"D:\project\大模型ssd化\colormlm\verify_log2.txt", "w", encoding="utf-8")

model_path = r"D:\project\大模型ssd化\models\Qwen2.5-1.5B-Instruct"

log.write("Loading Qwen 1.5B...\n"); log.flush()
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path, local_files_only=True,
    torch_dtype=torch.float32, device_map="cpu"
)
model.eval()
log.write("Model loaded.\n"); log.flush()

# Generate function
def generate(model, prompt, max_tokens=50, temperature=0.7):
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"]
    with torch.no_grad():
        for _ in range(max_tokens):
            out = model(ids)
            logits = out.logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# 校准数据
calibration_texts = [
    "def fibonacci(n):",
    "class Stack:",
    "import os",
    "for i in range(10):",
    "return result",
]
calib_inputs = [tokenizer(t, return_tensors="pt") for t in calibration_texts]

# 分组 FSQ 量化
def group_fsq_compress(tensor, n_levels=16, group_size=128):
    """分组量化: 每 group_size 个权重共享缩放参数"""
    flat = tensor.float().view(-1)
    n = flat.numel()
    
    # Pad to multiple of group_size
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    
    # Reshape to groups
    groups = flat.view(-1, group_size)
    
    # Per-group min/max
    g_min = groups.min(dim=1, keepdim=True).values
    g_max = groups.max(dim=1, keepdim=True).values
    g_scale = (g_max - g_min) / (n_levels - 1) + 1e-8
    
    # Quantize
    quantized = torch.round((groups - g_min) / g_scale).clamp(0, n_levels - 1)
    
    # Reconstruct
    reconstructed = quantized * g_scale + g_min
    reconstructed = reconstructed.view(-1)[:n]
    
    return reconstructed.view(tensor.shape), {
        "group_size": group_size,
        "n_levels": n_levels,
        "n_groups": groups.shape[0],
    }

# 原始模型测试
prompts = ["def fibonacci(n):", "class Stack:", "import"]

log.write("\n" + "="*60 + "\n")
log.write("ORIGINAL MODEL\n")
log.write("="*60 + "\n"); log.flush()
for p in prompts:
    t0 = time.time()
    result = generate(model, p, max_tokens=40)
    elapsed = time.time() - t0
    log.write("\nPrompt: %s\n" % p)
    log.write("Time: %.1fs\n" % elapsed)
    log.write("Output: %s\n" % result[:200])
    log.flush()

# 分组 FSQ 量化 (16级, group=128)
log.write("\n" + "="*60 + "\n")
log.write("GROUP FSQ (16-level, group=128)\n")
log.write("="*60 + "\n"); log.flush()

with torch.no_grad():
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype in [torch.float32, torch.float16]:
            recon, _ = group_fsq_compress(param.data, n_levels=16, group_size=128)
            param.data = recon.to(param.dtype)

for p in prompts:
    t0 = time.time()
    result = generate(model, p, max_tokens=40)
    elapsed = time.time() - t0
    log.write("\nPrompt: %s\n" % p)
    log.write("Time: %.1fs\n" % elapsed)
    log.write("Output: %s\n" % result[:200])
    log.flush()

# 重新加载模型, 试 32 级
log.write("\n" + "="*60 + "\n")
log.write("Reloading for 32-level test...\n")
log.write("="*60 + "\n"); log.flush()

del model
import gc; gc.collect()

model = AutoModelForCausalLM.from_pretrained(
    model_path, local_files_only=True,
    torch_dtype=torch.float32, device_map="cpu"
)
model.eval()

log.write("\n" + "="*60 + "\n")
log.write("GROUP FSQ (32-level, group=128)\n")
log.write("="*60 + "\n"); log.flush()

with torch.no_grad():
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype in [torch.float32, torch.float16]:
            recon, _ = group_fsq_compress(param.data, n_levels=32, group_size=128)
            param.data = recon.to(param.dtype)

for p in prompts:
    t0 = time.time()
    result = generate(model, p, max_tokens=40)
    elapsed = time.time() - t0
    log.write("\nPrompt: %s\n" % p)
    log.write("Time: %.1fs\n" % elapsed)
    log.write("Output: %s\n" % result[:200])
    log.flush()

log.write("\n" + "="*60 + "\n")
log.write("DONE!\n")
log.write("="*60 + "\n")
log.close()
