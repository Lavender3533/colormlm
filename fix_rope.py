# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_rope = '''    def apply_rope(self, q, k, position):
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        t = position.float()
        freqs = torch.outer(t, freqs)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        return apply_rotary_pos_emb(q, k, cos, sin)'''

new_rope = '''    def apply_rope(self, q, k, position):
        half_dim = self.head_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        t = position.float()
        freqs = torch.outer(t, freqs)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        # cos/sin are [1, 1, S, half_dim], q/k are [B, H, S, head_dim]
        # rotate_half splits q into two halves of size half_dim
        q1 = q[..., :half_dim]
        q2 = q[..., half_dim:]
        k1 = k[..., :half_dim]
        k2 = k[..., half_dim:]
        q_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q_out.type_as(q), k_out.type_as(k)'''

content = content.replace(old_rope, new_rope)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed RoPE")
