"""ColorLM Demo - Quick test"""
import os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colormlm.pretrained_demo import PretrainedColorLM
import torch

model = PretrainedColorLM(model_name='gpt2', device='cpu')

# === Test 1: Temperature analysis ===
print('\n=== 温度分析 ===')
texts = [
    'def fibonacci(n):',
    'class Stack:',
    'import os',
    'return result',
]
for text in texts:
    ids = model.encode_text(text)
    hidden, codes, temp, code_logits, vq_loss = model.forward(ids)
    tokens = model.tokenizer.convert_ids_to_tokens(ids[0])
    temp_vals = temp[0].squeeze(-1).tolist()
    print(f'\nText: {text}')
    for tok, t in zip(tokens, temp_vals):
        bar = '#' * int(t * 20)
        tok_safe = tok.encode('ascii', 'replace').decode()
        print(f'  {tok_safe:15s} | {t:.3f} | {bar}')

# === Test 2: VQ Codes ===
print('\n=== VQ 码本分析 ===')
ids = model.encode_text('def quicksort(arr):')
hidden, codes, temp, code_logits, vq_loss = model.forward(ids)
tokens = model.tokenizer.convert_ids_to_tokens(ids[0])
print('Token -> VQ Codes:')
for tok, code in zip(tokens, codes[0].tolist()):
    tok_safe = tok.encode('ascii', 'replace').decode()
    print(f'  {tok_safe:15s} -> {code}')

# === Test 3: Iterative prediction ===
print('\n=== 迭代修正预测 ===')
text = 'def binary_search'
ids = model.encode_text(text)
print(f'Original: {text}')

masked_ids = ids.clone()
masked_ids[0, -3:] = model.tokenizer.eos_token_id
mask_positions = [ids.shape[1]-3, ids.shape[1]-2, ids.shape[1]-1]

result_ids, history = model.iterative_predict(masked_ids, mask_positions, n_steps=5)
decoded = model.decode_ids(result_ids[0])
print(f'Predicted: {decoded}')
print(f'Steps: {len(history)}')
for h in history:
    step = h["step"]
    remaining = h["remaining"]
    avg_t = h["avg_temp"]
    print(f'  Step {step}: remaining={remaining}, avg_temp={avg_t:.3f}')

# === Test 4: Different texts temperature comparison ===
print('\n=== 关键字 vs 标点 温度对比 ===')
code_text = 'for i in range(10):'
ids = model.encode_text(code_text)
hidden, codes, temp, code_logits, vq_loss = model.forward(ids)
tokens = model.tokenizer.convert_ids_to_tokens(ids[0])
temp_vals = temp[0].squeeze(-1).tolist()
for tok, t in zip(tokens, temp_vals):
    tok_safe = tok.encode('ascii', 'replace').decode()
    importance = "HIGH" if t > 0.6 else ("MED" if t > 0.4 else "LOW")
    print(f'  {tok_safe:15s} | {t:.3f} | {importance}')

print('\nDone!')
