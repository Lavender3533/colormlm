# -*- coding: utf-8 -*-
"""
FSQ 压缩模型推理验证
加载 Qwen 1.5B, 压缩权重, 测试生成效果
"""
import torch, time, json
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

log = open(r"D:\project\大模型ssd化\colormlm\verify_log.txt", "w", encoding="utf-8")

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

# 测试: 原始模型生成
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

prompts = [
    "def fibonacci(n):",
    "class Stack:",
    "import",
]

log.write("\n" + "="*60 + "\n")
log.write("ORIGINAL MODEL (fp32)\n")
log.write("="*60 + "\n")
for p in prompts:
    t0 = time.time()
    result = generate(model, p, max_tokens=40)
    elapsed = time.time() - t0
    log.write("\nPrompt: %s\n" % p)
    log.write("Time: %.1fs\n" % elapsed)
    log.write("Output: %s\n" % result[:200])
    log.flush()

# FSQ 压缩所有权重
log.write("\n" + "="*60 + "\n")
log.write("COMPRESSING WEIGHTS (FSQ 16-level)...\n")
log.write("="*60 + "\n"); log.flush()

N_LEVELS = 16
compressed_params = {}

with torch.no_grad():
    for name, param in model.named_parameters():
        flat = param.data.float().view(-1)
        m, s = flat.mean(), flat.std()
        sc = 4 * s + 1e-8
        norm = ((flat - m) / sc).clamp(-1, 1)
        q = torch.round(norm * 7.5 + 7.5).clamp(0, 15)
        recon = (q - 7.5) / 7.5 * sc + m
        param.data = recon.view(param.shape).to(param.dtype)
        compressed_params[name] = (m.item(), sc.item())

log.write("All weights compressed.\n"); log.flush()

# 测试: 压缩后模型生成
log.write("\n" + "="*60 + "\n")
log.write("COMPRESSED MODEL (FSQ 4-bit)\n")
log.write("="*60 + "\n"); log.flush()

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
