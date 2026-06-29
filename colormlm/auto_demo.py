import os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrained_demo import PretrainedColorLM

print('='*55)
print('  ColorLM Auto Demo')
print('='*55)
print('\nLoading GPT-2 + VQ + Temperature...')
model = PretrainedColorLM('gpt2', n_codebooks=4, codebook_size=128, codebook_dim=32)

# Train temperature
print('\nTraining temperature head (20 epochs)...')
optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=1e-3)
train_texts = [
    "def quicksort(arr):", "class MyStack:", "for i in range(n):",
    "if x > 0: return True", "import os, sys", "result = [x**2 for x in items]",
    "while not done:", "try: val = int(s)", "with open(path) as f:",
    "return self.data[key]",
]
keywords = {'def','return','if','else','for','while','class','import','from','try','with','as','in','not','and'}
for epoch in range(20):
    for text in train_texts:
        ids = model.encode_text(text)
        tokens = model.tokenizer.convert_ids_to_tokens(ids[0])
        hidden, codes, temp, code_logits, vq_loss = model.forward(ids)
        temp_target = torch.zeros_like(temp)
        for i, tok in enumerate(tokens):
            clean = tok.strip().lower().replace('\u0120','')
            if clean in keywords: temp_target[0,i,0] = 1.0
            elif clean in {'(',')',':',',',' ','[',']','{','}'}: temp_target[0,i,0] = 0.1
            else: temp_target[0,i,0] = 0.5
        loss = F.mse_loss(temp, temp_target) + 0.05 * vq_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
print('Done!\n')

# Test inputs
test_inputs = [
    "def quicksort(arr):",
    "class MyStack:",
    "for i in range(10):",
    "if x > 0: return result",
    "import numpy as np",
]

for text in test_inputs:
    ids = model.encode_text(text)
    tokens = model.tokenizer.convert_ids_to_tokens(ids[0])
    with torch.no_grad():
        hidden, codes, temp, code_logits, _ = model.forward(ids)

    print(f'{"="*55}')
    print(f'  Input: "{text}"')
    print(f'{"="*55}')
    print(f'  {"Token":>15} {"Temp":>6} {"Codes":>20} {"Bar"}')
    print(f'  {"-"*55}')

    for i in range(ids.shape[1]):
        tok = tokens[i]
        t_val = temp[0,i,0].item()
        c_val = codes[0,i].tolist()
        bar = '#'*int(t_val*15)
        print(f'  {tok:>15} {t_val:.3f} {str(c_val):>20} {bar}')

    high = sorted([(tokens[i], temp[0,i,0].item()) for i in range(ids.shape[1])], key=lambda x:x[1], reverse=True)
    print(f'\n  Most important:  {high[0][0]} ({high[0][1]:.3f}), {high[1][0]} ({high[1][1]:.3f})')
    print(f'  Least important: {high[-1][0]} ({high[-1][1]:.3f}), {high[-2][0]} ({high[-2][1]:.3f})')
    print()

# Mask test
print(f'{"="*55}')
print(f'  Iterative Refinement Test')
print(f'{"="*55}')

text = "def quicksort(arr):"
ids = model.encode_text(text)
n = ids.shape[1]
mask_pos = list(range(1, n))
test_ids = ids.clone()
orig = [model.tokenizer.decode([ids[0,p].item()]) for p in mask_pos]
for p in mask_pos: test_ids[0,p] = model.tokenizer.pad_token_id or 0

print(f'\n  Original: {model.decode_ids(ids[0])}')
print(f'  Masked:   {model.decode_ids(test_ids[0])}')

model.eval()
result_ids = test_ids.clone()
determined = set()
for step in range(5):
    with torch.no_grad():
        h, c, t, cl, _ = model.forward(result_ids)
    temp_sq = t.squeeze(-1).squeeze(0)
    remaining = [p for p in mask_pos if p not in determined]
    if not remaining: break
    temps = sorted([(p, temp_sq[p].item()) for p in remaining], key=lambda x:x[1], reverse=True)
    n_fix = max(1, len(temps)//3)
    to_fix = temps[:n_fix]
    for pos, tv in to_fix:
        emb_table = model.backbone.get_input_embeddings().weight
        adapted = model.adapt(emb_table)
        sim = F.cosine_similarity(adapted, h[0,pos].unsqueeze(0), dim=-1)
        pred = int(sim.argmax().item())
        result_ids[0,pos] = pred
        determined.add(pos)
    preds = [(p, model.tokenizer.decode([result_ids[0,p].item()]), tv) for p, tv in to_fix]
    print(f'  Step {step+1}: ' + ', '.join(f'pos{p}="{d}"(t={tv:.2f})' for p,d,tv in preds))

print(f'\n  Final: {model.decode_ids(result_ids[0])}')
print(f'  (Note: token prediction needs MLM fine-tuning)')
print(f'\n{"="*55}')
print(f'  Demo complete!')