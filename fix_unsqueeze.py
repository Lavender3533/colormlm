# -*- coding: utf-8 -*-
path = r"D:\project\大模型ssd化\fsq_generate.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix: remove unsqueeze(0) since input_ids is already [1, S]
content = content.replace(
    'hidden_states = embed_weight[input_ids].unsqueeze(0).float()',
    'hidden_states = embed_weight[input_ids].float()'
)

# Also add qk_norm support (Qwen3 uses it)
# Add q_norm and k_norm after Q/K projection
old_qk = '''        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)'''
new_qk = '''        # QK Norm (Qwen3 uses this)
        q_norm_w = self.get_weight(layer_idx, "self_attn.q_norm.weight")
        k_norm_w = self.get_weight(layer_idx, "self_attn.k_norm.weight")
        
        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_kv_heads, self.head_dim)
        
        q = self.rms_norm(q, q_norm_w)
        k = self.rms_norm(k, k_norm_w)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)'''
content = content.replace(old_qk, new_qk)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed unsqueeze + added QK norm")
