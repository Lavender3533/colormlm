# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate_v2.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_rope = '''    def apply_rope(self, q, k, position_ids):
        # position_ids: [B, S]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ pos_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        # rotate_half
        half = q.shape[-1] // 2
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        
        q_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q_out, k_out'''

new_rope = '''    def apply_rope(self, q, k, position_ids):
        # position_ids: [B, S]
        inv_freq_expanded = self.inv_freq[None, :, None].float()  # [1, 64, 1]
        pos_expanded = position_ids[:, None, :].float()  # [B, 1, S]
        freqs = (inv_freq_expanded @ pos_expanded).transpose(1, 2)  # [B, S, 64]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, 128]
        cos = emb.cos().unsqueeze(1)  # [B, 1, S, 128]
        sin = emb.sin().unsqueeze(1)  # [B, 1, S, 128]
        
        # rotate_half: split last dim in half, negate and swap
        half = q.shape[-1] // 2
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        
        q_out = torch.cat([-q2, q1], dim=-1) * sin + q * cos
        k_out = torch.cat([-k2, k1], dim=-1) * sin + k * cos
        return q_out, k_out'''

content = content.replace(old_rope, new_rope)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed RoPE V2")
