# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\colormlm\compare_demo.py"
with open(path, encoding="utf-8") as f:
    content = f.read()
# Fix vocab size
content = content.replace(
    "student = ColorLMStudent(vocab_size=tokenizer.vocab_size)",
    "student = ColorLMStudent(vocab_size=max(tokenizer.vocab_size, 151644))"
)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed vocab size")
