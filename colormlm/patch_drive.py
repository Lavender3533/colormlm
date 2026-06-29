# -*- coding: utf-8 -*-
"""Patch: cache model to Google Drive so it persists across sessions"""
import json

path = r"D:\project\大模型ssd化\colormlm\colab_colorlm.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = "".join(cell["source"])

    # Replace Cell 4 model loading with Drive-cached version
    if "teacher_hidden" in src and "teacher = AutoModel" in src:
        new_src = """# %% Cell 4: Generate Distillation Cache (GPU: ~10 seconds)
import os
from pathlib import Path

# Mount Google Drive for persistent model cache
try:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE_DIR = '/content/drive/MyDrive/colormlm_cache'
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f'Drive cache dir: {CACHE_DIR}')
except:
    CACHE_DIR = '/tmp/colormlm_cache'
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f'No Drive, using tmp: {CACHE_DIR}')

# Cache paths
MODEL_CACHE = os.path.join(CACHE_DIR, 'qwen_hidden.pt')
model_name = 'Qwen/Qwen2.5-1.5B-Instruct'

if os.path.exists(MODEL_CACHE):
    print(f'Loading cached distillation data from {MODEL_CACHE}...')
    cache = torch.load(MODEL_CACHE, map_location='cpu', weights_only=False)
    teacher_hidden = cache['hidden']
    input_ids_all = cache['ids']
    attention_mask_all = cache['mask']
    print(f'Cache loaded: {teacher_hidden.shape}')
    # Load tokenizer only (for vocab info and Cell 7 analysis)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
else:
    print('No cache found, generating from teacher model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(DEVICE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher.eval()
    print(f'Teacher loaded: {teacher.config.hidden_size}d')
    print(f'Tokenizer vocab_size: {tokenizer.vocab_size}')
    print(f'Tokenizer pad_token_id: {tokenizer.pad_token_id}')

    SEQ_LEN = 64
    N_SAMPLES = 500

    texts = []
    while len(texts) < N_SAMPLES:
        for s in CODE_SAMPLES:
            texts.append(s)
            if len(texts) >= N_SAMPLES:
                break

    print(f'Generating {N_SAMPLES} samples...')
    all_h, all_i, all_m = [], [], []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=SEQ_LEN)
        ids = enc['input_ids'].to(DEVICE)
        mask = enc['attention_mask'].to(DEVICE)
        with torch.no_grad():
            out = teacher(ids, attention_mask=mask)
            h = out.last_hidden_state.float().cpu()
        all_h.append(h)
        all_i.append(ids.cpu())
        all_m.append(mask.cpu())

    teacher_hidden = torch.cat(all_h, 0)
    input_ids_all = torch.cat(all_i, 0)
    attention_mask_all = torch.cat(all_m, 0)

    # Save cache
    torch.save({'hidden': teacher_hidden, 'ids': input_ids_all, 'mask': attention_mask_all}, MODEL_CACHE)
    print(f'Cache saved to {MODEL_CACHE}')

    del teacher
    torch.cuda.empty_cache()
    print('Teacher released, GPU memory freed')

print(f'Data ready: {teacher_hidden.shape}')
SEQ_LEN = teacher_hidden.shape[1]
N_SAMPLES = teacher_hidden.shape[0]
"""
        cell["source"] = [l + "\n" for l in new_src.rstrip("\n").split("\n")]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patched! Model will be cached to Google Drive.")
