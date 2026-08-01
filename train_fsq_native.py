# -*- coding: utf-8 -*-
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from architecture import FSQTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

CODE_SAMPLES = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x < pivot]\n    right = [x for x in arr[1:] if x <= pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()",
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
    "def is_palindrome(s):\n    return s == s[::-1]",
    "class TreeNode:\n    def __init__(self, val=0):\n        self.val = val\n        self.left = None\n        self.right = None",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    return merge_sort(arr[:mid])",
]

class DS(Dataset):
    def __init__(self, texts, tok, ml=128):
        self.data = []
        for t in texts:
            enc = tok(t, truncation=True, max_length=ml, padding="max_length", return_tensors="pt")
            self.data.append({k: v.squeeze(0) for k, v in enc.items()})
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def train():
    print("Loading teacher...")
    MP = "./models/Qwen2.5-1.5B-Instruct"
    if not os.path.exists(MP):
        MP = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(MP, torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    teacher.eval()

    student = FSQTransformer(vocab=len(tok), d=256, nh=4, nl=6, ne=16, nlev=8, ng=16, tk=2, ms=128).to(DEVICE)
    print("Student params:", student.count_params())

    texts = []
    for i in range(500): texts.append(CODE_SAMPLES[i % len(CODE_SAMPLES)])
    ds = DS(texts, tok, 128)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    print("Dataset:", len(ds))

    opt = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    best = 999

    for ep in range(30):
        student.train()
        tl = 0; nb = 0
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            m = batch["attention_mask"].to(DEVICE)
            with torch.no_grad():
                out = teacher(ids, attention_mask=m, output_hidden_states=True)
                th = out.hidden_states[-1]
            logits, dl = student(ids, m, th)
            sl = logits[:, :-1, :].contiguous()
            lb = ids[:, 1:].contiguous()
            ll = F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1), ignore_index=tok.pad_token_id or 0)
            loss = 0.5 * dl + 0.5 * ll
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            tl += loss.item(); nb += 1
        sch.step()
        avg = tl / max(nb, 1)
        if (ep + 1) % 3 == 0: print("Epoch", ep+1, "/30 | Loss:", round(avg, 4))
        if avg < best:
            best = avg
            torch.save(student.state_dict(), "student_fsq_native_best.pt")

    print("Done! Best:", round(best, 4))

if __name__ == "__main__":
    train()