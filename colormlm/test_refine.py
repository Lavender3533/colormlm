import os, sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrained_demo import PretrainedColorLM

print('Loading model...')
model = PretrainedColorLM('gpt2', n_codebooks=4, codebook_size=128, codebook_dim=32)

# Quick temperature training
print('\nTraining temperature head (15 epochs)...')
optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=1e-3)
test_text = 'def quicksort(arr):'
ids = model.encode_text(test_text)
tokens = model.tokenizer.convert_ids_to_tokens(ids[0])

for epoch in range(15):
    hidden, codes, temp, code_logits, vq_loss = model.forward(ids)
    keywords = {'def', 'return', 'if', 'else', 'for', 'while', 'class'}
    temp_target = torch.zeros_like(temp)
    for i, tok in enumerate(tokens):
        clean = tok.strip().lower().replace('\u0120', '')
        if clean in keywords:
            temp_target[0, i, 0] = 1.0
        elif clean in {'(', ')', ':', ',', ' '}:
            temp_target[0, i, 0] = 0.1
        else:
            temp_target[0, i, 0] = 0.5
    loss = F.mse_loss(temp, temp_target) + 0.05 * vq_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Training done.\n')

# Test iterative refinement
print('='*50)
print('  Iterative Refinement Test')
print('='*50)

ids2 = model.encode_text(test_text)
n_tokens = ids2.shape[1]

# Mask positions 1, 2, 3 (qu, icks, ort)
mask_positions = [1, 2, 3]
original = [model.tokenizer.decode([ids2[0, p].item()]) for p in mask_positions]
print(f'\nOriginal: {model.decode_ids(ids2[0])}')
print(f'Masking positions {mask_positions}: {original}')

test_ids = ids2.clone()
for p in mask_positions:
    test_ids[0, p] = model.tokenizer.pad_token_id or 0
print(f'Masked:   {model.decode_ids(test_ids[0])}')

# Inline iterative predict (fix type issue)
model.eval()
result_ids = test_ids.clone()
determined = set()
history = []

for step in range(5):
    with torch.no_grad():
        hidden, codes, temp, code_logits, _ = model.forward(result_ids)
    temp_sq = temp.squeeze(-1).squeeze(0)

    remaining = [p for p in mask_positions if p not in determined]
    if not remaining:
        break

    temps = [(p, temp_sq[p].item()) for p in remaining]
    temps.sort(key=lambda x: x[1], reverse=True)

    n_fix = max(1, len(temps) // 3)
    to_fix = temps[:n_fix]

    for pos, t in to_fix:
        emb_table = model.backbone.get_input_embeddings().weight
        adapted_emb = model.adapt(emb_table)
        hidden_pos = hidden[0, pos]
        sim = F.cosine_similarity(adapted_emb, hidden_pos.unsqueeze(0), dim=-1)
        pred_token = int(sim.argmax().item())
        result_ids[0, pos] = pred_token
        determined.add(pos)

    history.append({
        "step": step + 1,
        "fixed": [(p, int(sim.argmax().item())) for p, _ in to_fix],
        "remaining": len(remaining) - n_fix,
        "avg_temp": sum(t for _, t in to_fix) / len(to_fix),
    })

print(f'\nRefinement steps:')
for h in history:
    for p, t in h['fixed']:
        pred = model.tokenizer.decode([t])
        print(f'  Step {h["step"]}: pos {p} -> "{pred}" (temp={h["avg_temp"]:.3f})')

print(f'\nResult: {model.decode_ids(result_ids[0])}')
print(f'Original: {model.decode_ids(ids2[0])}')

# Compare
print(f'\n{"="*50}')
print('  Speed comparison')
print(f'{"="*50}')
print(f'  Autoregressive: {len(mask_positions)} steps (one token per step)')
print(f'  Our method:     {len(history)} steps (parallel refinement)')
if len(history) > 0:
    print(f'  Speedup:        {len(mask_positions) / len(history):.1f}x')