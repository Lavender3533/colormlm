# -*- coding: utf-8 -*-
"""FSQ Inference V2 - Correct Qwen3-MoE implementation"""
import os, torch, json, time, math
from safetensors import safe_open
from transformers import AutoTokenizer

class FSQModelV2:
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        self.device = device
        
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
            self.weight_map = json.load(f).get("weight_map", {})
        
        self.num_layers = self.config["num_hidden_layers"]
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_experts = self.config.get("num_experts", 128)
        self.top_k = self.config.get("num_experts_per_tok", 8)
        self.vocab_size = self.config["vocab_size"]
        self.rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)
        self.scaling = self.head_dim ** -0.5
        
        # RoPE
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.inv_freq = inv_freq
        
        print("Model V2 loaded:")
        print("  Layers:", self.num_layers, "Hidden:", self.hidden_size)
        print("  Heads:", self.num_heads, "KV:", self.num_kv_heads, "Head dim:", self.head_dim)
        print("  Experts:", self.num_experts, "Active:", self.top_k)
    
    def get_tensor(self, key):
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(key)
        path = os.path.join(self.model_dir, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if key + "._bq_codes" in keys:
                codes = f.get_tensor(key + "._bq_codes")
                meta = f.get_tensor(key + "._bq_meta")
                shape_t = f.get_tensor(key + "._bq_shape")
                shape = [shape_t[0].item(), shape_t[1].item()]
                n = codes.shape[0]
                b_min = meta[:n].unsqueeze(1)
                b_max = meta[n:].unsqueeze(1)
                recon = codes.float() / 255.0 * (b_max - b_min) + b_min
                return recon.flatten()[:shape[0]*shape[1]].reshape(shape)
            t = f.get_tensor(key)
            return t.float() if t.dtype == torch.bfloat16 else t
    
    def get_weight(self, layer_idx, name):
        return self.get_tensor("model.layers." + str(layer_idx) + "." + name)
    
    def rms_norm(self, x, weight):
        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        x_normed = x_f * torch.rsqrt(var + self.rms_norm_eps)
        return (weight * x_normed).to(x.dtype)
    
    def apply_rope(self, q, k, position_ids):
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
        return q_out, k_out
    
    def forward_layer(self, layer_idx, hidden_states, position_ids):
        # Attention
        residual = hidden_states
        h = self.rms_norm(hidden_states, self.get_weight(layer_idx, "input_layernorm.weight"))
        
        q = self.get_weight(layer_idx, "self_attn.q_proj.weight") @ h.transpose(-1, -2)
        q = q.transpose(-1, -2)
        k = h @ self.get_weight(layer_idx, "self_attn.k_proj.weight").T
        v = h @ self.get_weight(layer_idx, "self_attn.v_proj.weight").T
        
        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # QK Norm
        q_norm_w = self.get_weight(layer_idx, "self_attn.q_norm.weight")
        k_norm_w = self.get_weight(layer_idx, "self_attn.k_norm.weight")
        q = self.rms_norm(q, q_norm_w)
        k = self.rms_norm(k, k_norm_w)
        
        # RoPE
        q, k = self.apply_rope(q, k, position_ids)
        
        # GQA expand
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scaling
        causal = torch.triu(torch.ones(S, S, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        
        o_proj = self.get_weight(layer_idx, "self_attn.o_proj.weight")
        attn_out = (o_proj @ attn_out.transpose(-1, -2)).transpose(-1, -2)
        
        hidden_states = residual + attn_out
        
        # MoE
        residual = hidden_states
        h = self.rms_norm(hidden_states, self.get_weight(layer_idx, "post_attention_layernorm.weight"))
        
        # Router: softmax first, then topk, then normalize
        gate_w = self.get_weight(layer_idx, "mlp.gate.weight")
        router_logits = h @ gate_w.T  # [B, S, num_experts]
        router_logits = torch.softmax(router_logits.float(), dim=-1)
        topk_values, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_values /= topk_values.sum(dim=-1, keepdim=True)  # normalize
        
        # Expert computation
        moe_out = torch.zeros_like(h)
        for i in range(self.top_k):
            idx = topk_indices[0, 0, i].item()
            w = topk_values[0, 0, i].item()
            
            gate_w = self.get_weight(layer_idx, "mlp.experts." + str(idx) + ".gate_proj.weight")
            up_w = self.get_weight(layer_idx, "mlp.experts." + str(idx) + ".up_proj.weight")
            down_w = self.get_weight(layer_idx, "mlp.experts." + str(idx) + ".down_proj.weight")
            
            gate = h @ gate_w.T
            up = h @ up_w.T
            gate = torch.nn.functional.silu(gate)
            expert_out = (gate * up) @ down_w.T
            
            moe_out = moe_out + expert_out * w
        
        hidden_states = residual + moe_out
        return hidden_states
    
    def forward(self, input_ids):
        embed = self.get_tensor("model.embed_tokens.weight")
        hidden_states = embed[input_ids].float()
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        
        for i in range(self.num_layers):
            hidden_states = self.forward_layer(i, hidden_states, position_ids)
            if i % 10 == 0:
                print("  Layer", i, "...")
        
        hidden_states = self.rms_norm(hidden_states, self.get_tensor("model.norm.weight"))
        lm_head = self.get_tensor("lm_head.weight")
        logits = hidden_states @ lm_head.T
        return logits
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        tokens = input_ids[0].tolist()
        
        print("Generating...")
        t0 = time.time()
        
        for step in range(max_new_tokens):
            input_t = torch.tensor([tokens], dtype=torch.long)
            try:
                logits = self.forward(input_t)
            except Exception as e:
                print("Error:", e)
                import traceback
                traceback.print_exc()
                break
            
            next_logits = logits[0, -1, :] / temperature
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumsum - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            next_token = sorted_indices[next_idx].item()
            
            tokens.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                break
            if step % 5 == 0:
                print("  Step", step, "token:", self.tokenizer.decode([next_token]))
        
        elapsed = time.time() - t0
        n = len(tokens) - len(input_ids[0])
        print("Generated", n, "tokens in", round(elapsed, 1), "s (" + str(round(n/elapsed, 2)) + " tok/s)")
        return self.tokenizer.decode(tokens)

def main():
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\\project\\大模型ssd化\\models\\qwen3-coder-bq8"
    print("=" * 50)
    print("FSQ Inference V2")
    print("=" * 50)
    model = FSQModelV2(model_dir)
    result = model.generate("def fibonacci", max_new_tokens=50)
    print()
    print("Output:")
    print(result)

if __name__ == "__main__":
    main()
