# -*- coding: utf-8 -*-
import os
BASE = r"D:\project\大模型ssd化"
outpath = os.path.join(BASE, "fsq_generate.py")

code = r'''# -*- coding: utf-8 -*-
"""FSQ Full Inference - Block-wise 8-bit Quantized Qwen3-Coder-30B"""
import os, torch, json, time, math
from safetensors import safe_open
from transformers import AutoTokenizer

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.ones(dim)
    
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.type_as(q), k_embed.type_as(k)

class FSQModel:
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        self.device = device
        self.shard_cache = {}
        
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
            index = json.load(f)
        self.weight_map = index.get("weight_map", {})
        
        self.num_layers = self.config["num_hidden_layers"]
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_experts = self.config.get("num_experts", 128)
        self.top_k = self.config.get("num_experts_per_tok", 8)
        self.vocab_size = self.config["vocab_size"]
        self.rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)
        
        print("Model loaded:")
        print("  Layers:", self.num_layers)
        print("  Hidden:", self.hidden_size)
        print("  Heads:", self.num_heads, "KV:", self.num_kv_heads)
        print("  Experts:", self.num_experts, "Active:", self.top_k)
        print("  Vocab:", self.vocab_size)
    
    def get_tensor(self, key):
        shard = self.weight_map.get(key)
        if shard is None:
            for s_name in self.shard_cache:
                if key in self.shard_cache[s_name]:
                    return self.shard_cache[s_name][key]
            for s_name in set(self.weight_map.values()):
                if s_name not in self.shard_cache:
                    self._load_shard(s_name)
                if key in self.shard_cache[s_name]:
                    return self.shard_cache[s_name][key]
            raise KeyError(key)
        
        if shard not in self.shard_cache:
            self._load_shard(shard)
        
        tensors = self.shard_cache[shard]
        
        if key + "._bq_codes" in tensors:
            return self._dequantize(key, tensors)
        
        return tensors[key]
    
    def _load_shard(self, shard_name):
        path = os.path.join(self.model_dir, shard_name)
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        self.shard_cache[shard_name] = tensors
    
    def _dequantize(self, key, tensors):
        codes = tensors[key + "._bq_codes"]
        meta = tensors[key + "._bq_meta"]
        shape_t = tensors[key + "._bq_shape"]
        shape = [shape_t[0].item(), shape_t[1].item()]
        
        n_blocks = codes.shape[0]
        b_min = meta[:n_blocks].unsqueeze(1)
        b_max = meta[n_blocks:].unsqueeze(1)
        recon = codes.float() / 255.0 * (b_max - b_min) + b_min
        return recon.flatten()[:shape[0]*shape[1]].reshape(shape)
    
    def get_weight(self, layer_idx, name):
        key = "model.layers." + str(layer_idx) + "." + name
        return self.get_tensor(key)
    
    def rms_norm(self, x, weight):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.rms_norm_eps).rsqrt()
        return (x.float() * norm).type_as(x) * weight
    
    def apply_rope(self, q, k, position):
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        t = position.float()
        freqs = torch.outer(t, freqs)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        return apply_rotary_pos_emb(q, k, cos, sin)
    
    def forward_layer(self, layer_idx, hidden_states, position):
        # RMSNorm
        normed = self.rms_norm(hidden_states, self.get_weight(layer_idx, "input_layernorm.weight"))
        
        # Attention
        q = normed @ self.get_weight(layer_idx, "self_attn.q_proj.weight").T
        k = normed @ self.get_weight(layer_idx, "self_attn.k_proj.weight").T
        v = normed @ self.get_weight(layer_idx, "self_attn.v_proj.weight").T
        
        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        pos = torch.arange(S, device=hidden_states.device)
        q, k = self.apply_rope(q, k, pos)
        
        # GQA: expand KV heads
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        
        # Scaled dot product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(S, S, device=hidden_states.device), diagonal=1).bool()
        attn.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        attn = torch.softmax(attn, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        
        # Output projection
        attn_out = attn_out @ self.get_weight(layer_idx, "self_attn.o_proj.weight").T
        
        # Residual
        hidden_states = hidden_states + attn_out
        
        # RMSNorm before MoE
        normed = self.rms_norm(hidden_states, self.get_weight(layer_idx, "post_attention_layernorm.weight"))
        
        # MoE Router
        router_logits = normed @ self.get_weight(layer_idx, "mlp.gate.weight").T
        topk_values, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = torch.softmax(topk_values, dim=-1)
        
        # MoE Expert computation
        moe_output = torch.zeros_like(normed)
        for i in range(self.top_k):
            expert_idx = topk_indices[0, 0, i].item()
            weight = routing_weights[0, 0, i].item()
            
            gate = normed @ self.get_weight(layer_idx, "mlp.experts." + str(expert_idx) + ".gate_proj.weight").T
            up = normed @ self.get_weight(layer_idx, "mlp.experts." + str(expert_idx) + ".up_proj.weight").T
            
            # SwiGLU
            gate = torch.nn.functional.silu(gate)
            expert_out = (gate * up) @ self.get_weight(layer_idx, "mlp.experts." + str(expert_idx) + ".down_proj.weight").T
            
            moe_output = moe_output + expert_out * weight
        
        hidden_states = hidden_states + moe_output
        
        return hidden_states
    
    def forward(self, input_ids):
        # Embedding
        embed_weight = self.get_tensor("model.embed_tokens.weight")
        hidden_states = embed_weight[input_ids].unsqueeze(0).float()
        
        # Process through layers
        for i in range(self.num_layers):
            hidden_states = self.forward_layer(i, hidden_states, torch.arange(hidden_states.shape[1]))
        
        # Final norm
        hidden_states = self.rms_norm(hidden_states, self.get_tensor("model.norm.weight"))
        
        # LM head
        lm_head = self.get_tensor("lm_head.weight")
        logits = hidden_states @ lm_head.T
        
        return logits
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        tokens = input_ids[0].tolist()
        
        print("Generating...", end="", flush=True)
        t0 = time.time()
        
        for step in range(max_new_tokens):
            input_t = torch.tensor([tokens], dtype=torch.long)
            logits = self.forward(input_t)
            
            next_logits = logits[0, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            
            probs = torch.softmax(sorted_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            next_token = sorted_indices[next_idx].item()
            
            tokens.append(next_token)
            
            if next_token == self.tokenizer.eos_token_id:
                break
            
            if step % 10 == 0:
                print(".", end="", flush=True)
        
        elapsed = time.time() - t0
        tokens_generated = len(tokens) - len(input_ids[0])
        print(" Done!")
        print("Generated", tokens_generated, "tokens in", round(elapsed, 1), "s")
        print("Speed:", round(tokens_generated / elapsed, 1), "tokens/s")
        
        return self.tokenizer.decode(tokens)

def main():
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\\project\\大模型ssd化\\models\\qwen3-coder-bq8"
    
    print("=" * 50)
    print("FSQ Full Inference Engine")
    print("=" * 50)
    
    model = FSQModel(model_dir)
    
    print("")
    print("Starting generation test...")
    result = model.generate("def fibonacci", max_new_tokens=50)
    print("")
    print("Output:")
    print(result)

if __name__ == "__main__":
    main()
'''

with open(outpath, 'w', encoding='utf-8') as f:
    f.write(code)
print("Written to", outpath)
