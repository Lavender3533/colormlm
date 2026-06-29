# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix: use config head_dim instead of computing from hidden_size
old = 'self.head_dim = self.hidden_size // self.num_heads'
new = 'self.head_dim = self.config.get("head_dim", self.hidden_size // self.num_heads)'
content = content.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed head_dim")
