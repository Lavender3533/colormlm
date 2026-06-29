# -*- coding: utf-8 -*-
"""Quick demo: load student model and show FSQ codes + temperature"""
import torch
import torch.nn as nn

DEVICE = 'cpu'

# --- Define student model ---
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
    def __init__(self, vocab_size=151644, d_model=384, n_heads=6, n_layers=6, d_ff=1536, max_seq_len=64):
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
        distill = self.distill_proj(x)
        return quantized, codes, temp, distill, fsq_loss

# --- Load tokenizer and student ---
from transformers import AutoTokenizer
model_path = r'D:\project\大模型ssd化\models\Qwen2.5-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

student = ColorLMStudent(vocab_size=151644)
student_path = r'D:\project\大模型ssd化\colormlm\data\student_best.pt'
student.load_state_dict(torch.load(student_path, map_location='cpu', weights_only=True))
student.eval()
total = sum(p.numel() for p in student.parameters())
print(f'Student loaded: {total/1e6:.1f}M params')

# --- Test ---
test_codes = [
    'def fibonacci(n):',
    'def quicksort(arr):',
    'class Stack:',
    'import os',
    'for i in range(10):',
    'return result',
]

print()
print('='*65)
print('ColorLM Student - Encoding Demo')
print('='*65)

for text in test_codes:
    enc = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
    ids = enc['input_ids']
    mask = enc['attention_mask']
    with torch.no_grad():
        _, codes, temp, _, _ = student(ids, mask)

    tokens = tokenizer.convert_ids_to_tokens(ids[0])
    temps = temp[0].squeeze(-1).tolist()
    code_list = codes[0].tolist()
    valid = mask[0].bool().tolist()

    print(f"\n'{text}'")
    print(f'  {"Token":<15} {"Temp":>6}  {"FSQ Code [R,G,B,A,?,?]":<30}')
    print(f'  {"-"*15} {"-"*6}  {"-"*30}')
    for tok, t, code, v in zip(tokens[:15], temps[:15], code_list[:15], valid):
        if not v: break
        tok_clean = tok.encode('ascii', 'replace').decode()
        bar = '#' * int(t * 10)
        label = 'HIGH' if t > 0.6 else ('MED' if t > 0.4 else 'LOW')
        print(f'  {tok_clean:<15} {t:.3f}{bar:<10} {str(code):<30} {label}')

print('\n' + '='*65)
print(f'Model size: {total/1e6:.1f}M params ({total*4/1024/1024:.1f}MB in fp32)')
print(f'Qwen teacher: 1543M params (3094MB) = 22x larger')
print(f'Representation: 6 FSQ codes (8^6 = 262144 combinations) + temperature')
print(f'Speed: instant on CPU (no GPU needed)')
