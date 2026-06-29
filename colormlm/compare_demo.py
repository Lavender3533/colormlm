# -*- coding: utf-8 -*-
"""Compare student vs teacher representations locally"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cpu'
print('Loading models...')

# --- Load Qwen teacher ---
from transformers import AutoModel, AutoTokenizer
model_path = r'D:\project\大模型ssd化\models\Qwen2.5-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
teacher = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
teacher.eval()
print(f'Teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.0f}M params')

# --- Define student model (must match training architecture) ---
class FSQ(nn.Module):
    def __init__(self, d_model, n_dims=6, levels_per_dim=8):
        super().__init__()
        self.n_dims = n_dims
        self.levels = levels_per_dim
        self.proj_in = nn.Linear(d_model, n_dims, bias=False)
        self.proj_out = nn.Linear(n_dims, d_model, bias=False)
    def forward(self, x):
        z = torch.tanh(self.proj_in(x))
        z_int = torch.round(z * (self.levels - 1) / 2 + (self.levels - 1) / 2).clamp(0, self.levels - 1)
        codes = z_int.long()
        z_hat = torch.tanh((z_int - (self.levels - 1) / 2) / ((self.levels - 1) / 2))
        return codes, self.proj_out(z_hat), torch.tensor(0.0)

class TemperatureHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model//4), nn.SiLU(), nn.Linear(d_model//4, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class StudentBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, pad_mask=None):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n, key_padding_mask=pad_mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class ColorLMStudent(nn.Module):
    def __init__(self, vocab_size=151644, d_model=384, n_heads=6, n_layers=6, d_ff=1536, max_seq_len=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList([StudentBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.fsq = FSQ(d_model, 6, 8)
        self.temp_head = TemperatureHead(d_model)
        self.distill_proj = nn.Sequential(nn.Linear(d_model, d_model*2), nn.SiLU(), nn.Linear(d_model*2, 1536))
    def forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape
        tok = self.token_embed(input_ids)
        pos = self.pos_embed(torch.arange(S, device=input_ids.device)).unsqueeze(0)
        x = self.embed_drop(tok + pos)
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask)
        x = self.final_norm(x)
        codes, quantized, fsq_loss = self.fsq(x)
        temp = self.temp_head(x)
        distill = self.distill_proj(x)  # V3: use raw x
        return quantized, codes, temp, distill, fsq_loss

# --- Load student ---
student_path = r'D:\project\大模型ssd化\colormlm\data\student_final.pt'
import os
if not os.path.exists(student_path):
    # Try Colab-style save
    student_path = 'student_best.pt'
    if not os.path.exists(student_path):
        print('ERROR: student model not found. Please provide path to student_best.pt')
        exit(1)

student = ColorLMStudent(vocab_size=max(tokenizer.vocab_size, 151644))
student.load_state_dict(torch.load(student_path, map_location='cpu', weights_only=True))
student.eval()
print(f'Student: {sum(p.numel() for p in student.parameters())/1e6:.1f}M params')

# --- Compare ---
test_texts = [
    'def fibonacci(n):',
    'class Stack:',
    'def quicksort(arr):',
    'import torch',
    'return result',
]

print('\n' + '='*60)
print('COMPARISON: Student vs Qwen Teacher')
print('='*60)

for text in test_texts:
    enc = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
    ids = enc['input_ids']
    mask = enc['attention_mask']

    with torch.no_grad():
        # Teacher
        t_out = teacher(ids, attention_mask=mask)
        t_hidden = t_out.last_hidden_state

        # Student
        _, codes, temp, s_hidden, _ = student(ids, mask)

    # Cosine similarity
    cos = F.cosine_similarity(s_hidden, t_hidden, dim=-1)
    valid = mask.bool()
    avg_cos = cos[valid].mean().item()

    # Show token-level comparison
    tokens = tokenizer.convert_ids_to_tokens(ids[0])
    temps = temp[0].squeeze(-1).tolist()
    code_list = codes[0].tolist()

    print(f'\n"{text}"')
    print(f'  Cosine similarity: {avg_cos:.4f}')
    print(f'  {"Token":<15} {"Temp":>6}  {"FSQ Codes":<25} {"Cos":>6}')
    print(f'  {"-"*15} {"-"*6}  {"-"*25} {"-"*6}')
    for j, (tok, t, code, v) in enumerate(zip(tokens[:10], temps[:10], code_list[:10], valid[0].tolist())):
        if not v: break
        tok_clean = tok.encode('ascii', 'replace').decode()
        c = cos[0, j].item()
        print(f'  {tok_clean:<15} {t:.3f}  {str(code):<25} {c:.3f}')

print('\n' + '='*60)
print('Summary')
print('='*60)
# Overall cosine similarity
all_cos = []
for text in test_texts:
    enc = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
    ids = enc['input_ids']
    mask = enc['attention_mask']
    with torch.no_grad():
        t_out = teacher(ids, attention_mask=mask)
        _, codes, temp, s_hidden, _ = student(ids, mask)
    cos = F.cosine_similarity(s_hidden, t_out.last_hidden_state, dim=-1)
    all_cos.append(cos[mask.bool()].mean().item())
print(f'Average cosine similarity: {sum(all_cos)/len(all_cos):.4f}')
print(f'Student: 70M params vs Teacher: 1543M params = 22x compression')
print(f'Note: Student cannot generate text yet (no lm_head)')
