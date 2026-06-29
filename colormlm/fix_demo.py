# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\colormlm\student_demo.py"
with open(path, encoding="utf-8") as f:
    c = f.read()
c = c.replace("max_seq_len=128):", "max_seq_len=64):")
c = c.replace("max_seq_len=128)", "max_seq_len=64)")
with open(path, "w", encoding="utf-8") as f:
    f.write(c)
print("Fixed max_seq_len to 64")
