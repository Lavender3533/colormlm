# -*- coding: utf-8 -*-
"""
ColorLM V5 - LRU Cache Inference
Key optimizations:
1. Expert LRU cache: convert uint8->float16 on first access, then cached
2. Pre-loaded non-expert weights
3. Open safetensors handles at startup (no repeated file open)
Memory: ~8GB (3GB non-expert + 5GB cache)
"""
import os, sys, torch, json, time
from collections import OrderedDict
from safetensors import safe_open
from transformers import AutoTokenizer

BLOCK = 128
MAX_CACHED_EXPERTS = 32  # 32 * 150MB = 4.8GB

class ExpertCache:
    """LRU cache for expert weights. Converts uint8->float16 on first access."""
    def __init__(self, max_size=MAX_CACHED_EXPERTS):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        self.misses += 1
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total > 0 else 0
        return f"Cache: {len(self.cache)}/{self.max_size}, hit rate: {rate:.1f}% ({self.hits}/{total})"


class FastModel:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.wmap = json.load(f).get("weight_map", {})

        self.L = self.cfg["num_hidden_layers"]
        self.H = self.cfg["hidden_size"]
        self.NH = self.cfg["num_attention_heads"]
        self.NK = self.cfg.get("num_key_value_heads", self.NH)
        self.HD = self.cfg.get("head_dim", self.H // self.NH)
        self.NE = self.cfg.get("num_experts", 128)
        self.TK = self.cfg.get("num_experts_per_tok", 8)
        self.eps = self.cfg.get("rms_norm_eps", 1e-6)
        self.scale = self.HD ** -0.5
        self.rep = self.NH // self.NK

        inv = 1.0 / (10000.0 ** (torch.arange(0, self.HD, 2).float() / self.HD))
        self.inv = inv

        # Open all safetensors handles at startup
        self._shards = {}

        # Expert LRU cache
        self.expert_cache = ExpertCache(MAX_CACHED_EXPERTS)

        print("Loading non-expert weights...")
        self.embed = self._ld("model.embed_tokens.weight").half()
        self.norm_w = self._ld("model.norm.weight").half()
        self.lm_w = self._ld("lm_head.weight").half()

        print("Pre-loading attention weights for all layers...")
        self.layers = []
        for i in range(self.L):
            lw = self._load_layer(i)
            self.layers.append(lw)
            if i % 8 == 0:
                print(f"  Layer {i}/{self.L}")

        self.kv_k = [None] * self.L
        self.kv_v = [None] * self.L
        print(f"Ready: {self.L}L, {self.NH}H, {self.NE}E, {self.TK}K")
        print(f"Expert cache: {MAX_CACHED_EXPERTS} slots")

    def _get_shard(self, name):
        if name not in self._shards:
            self._shards[name] = safe_open(
                os.path.join(self.model_dir, name), framework="pt", device="cpu"
            )
        return self._shards[name]

    def _ld(self, key):
        shard = self.wmap[key]
        f = self._get_shard(shard)
        ck = key + "._bq_codes"
        if ck in f.keys():
            codes = f.get_tensor(ck)
            meta = f.get_tensor(key + "._bq_meta")
            shape_t = f.get_tensor(key + "._bq_shape")
            shape = [shape_t[0].item(), shape_t[1].item()]
            n = codes.shape[0]
            mn = meta[:n].unsqueeze(1)
            mx = meta[n:].unsqueeze(1)
            sc = (mx - mn) / 255.0
            w = codes.float() * sc + mn
            return w.flatten()[:shape[0]*shape[1]].reshape(shape)
        t = f.get_tensor(key)
        return t.float() if t.dtype == torch.bfloat16 else t

    def _load_layer(self, idx):
        """Load non-expert weights for a layer"""
        prefix = f"model.layers.{idx}."
        w = {}
        for key in self.wmap:
            if key.startswith(prefix) and "experts." not in key and "._bq" not in key:
                w[key[len(prefix):]] = self._ld(key).half()
        return w

    def _get_expert(self, layer, expert, proj):
        """Get expert weight with LRU cache. Converts uint8->float16 on first access."""
        cache_key = f"L{layer}E{expert}_{proj}"
        cached = self.expert_cache.get(cache_key)
        if cached is not None:
            return cached

        # Cache miss: load from safetensors and convert
        key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
        shard = self.wmap[key]
        f = self._get_shard(shard)
        codes = f.get_tensor(key + "._bq_codes")
        meta = f.get_tensor(key + "._bq_meta")
        shape_t = f.get_tensor(key + "._bq_shape")
        shape = [shape_t[0].item(), shape_t[1].item()]
        n = codes.shape[0]
        mn = meta[:n].unsqueeze(1)
        mx = meta[n:].unsqueeze(1)
        sc = (mx - mn) / 255.0
        w = (codes.float() * sc + mn).flatten()[:shape[0]*shape[1]].reshape(shape).half()

        self.expert_cache.put(cache_key, w)
        return w

    def _rms(self, x, w):
        xf = x.float()
        return (w.float() * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).half()

    def _rope(self, q, k, pos):
        inv = self.inv[None, :, None].float()
        pf = pos[:, None, :].float()
        freqs = (inv @ pf).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        c = emb.cos().unsqueeze(1).half()
        s = emb.sin().unsqueeze(1).half()
        h = q.shape[-1] // 2
        return (q * c + torch.cat([-q[..., h:], q[..., :h]], dim=-1) * s,
                k * c + torch.cat([-k[..., h:], k[..., :h]], dim=-1) * s)

    def forward(self, hidden, pos):
        B, S, _ = hidden.shape
        for i in range(self.L):
            w = self.layers[i]
            res = hidden
            h = self._rms(hidden, w["input_layernorm.weight"])

            q = (h @ w["self_attn.q_proj.weight"].T).view(B, S, self.NH, self.HD).transpose(1, 2)
            k = (h @ w["self_attn.k_proj.weight"].T).view(B, S, self.NK, self.HD).transpose(1, 2)
            v = (h @ w["self_attn.v_proj.weight"].T).view(B, S, self.NK, self.HD).transpose(1, 2)

            q = self._rms(q, w["self_attn.q_norm.weight"])
            k = self._rms(k, w["self_attn.k_norm.weight"])
            q, k = self._rope(q, k, pos)

            if self.kv_k[i] is not None:
                k = torch.cat([self.kv_k[i], k], dim=2)
                v = torch.cat([self.kv_v[i], v], dim=2)
            self.kv_k[i] = k.detach()
            self.kv_v[i] = v.detach()

            if self.rep > 1:
                k = k.repeat_interleave(self.rep, dim=1)
                v = v.repeat_interleave(self.rep, dim=1)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            tl = k.shape[2]
            if tl > S:
                mask = torch.ones(S, tl, dtype=torch.bool, device=hidden.device)
                for j in range(S):
                    mask[j, :tl - S + j + 1] = False
            else:
                mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=hidden.device), diagonal=1)
            attn.masked_fill_(mask[None, None], float("-inf"))
            attn = torch.softmax(attn.float(), dim=-1).half()
            ao = (attn @ v).transpose(1, 2).reshape(B, S, -1)
            ao = (w["self_attn.o_proj.weight"] @ ao.transpose(-1, -2)).transpose(-1, -2)
            hidden = res + ao

            # MoE with expert LRU cache
            res = hidden
            h = self._rms(hidden, w["post_attention_layernorm.weight"])
            logits = (h @ w["mlp.gate.weight"].T).float()
            logits = torch.softmax(logits, dim=-1)
            tv, ti = torch.topk(logits, self.TK, dim=-1)
            tv /= tv.sum(dim=-1, keepdim=True)

            moe = torch.zeros_like(h)
            for j in range(self.TK):
                idx = ti[0, 0, j].item()
                wt = tv[0, 0, j].item()
                g = self._get_expert(i, idx, "gate_proj")
                u = self._get_expert(i, idx, "up_proj")
                d = self._get_expert(i, idx, "down_proj")
                gate = torch.nn.functional.silu(h @ g.T)
                up = h @ u.T
                moe = moe + (gate * up) @ d.T * wt

            hidden = res + moe
        return hidden

    def clear_kv(self):
        self.kv_k = [None] * self.L
        self.kv_v = [None] * self.L

    @torch.no_grad()
    def generate(self, prompt, max_new=64, temp=0.7, top_p=0.9):
        ids = self.tok.encode(prompt, return_tensors="pt")
        tokens = ids[0].tolist()
        self.clear_kv()

        print("Generating...")
        t0 = time.time()

        h = self.embed[ids].half()
        pos = torch.arange(h.shape[1]).unsqueeze(0)
        h = self.forward(h, pos)
        h = self._rms(h, self.norm_w)
        logits = (h @ self.lm_w.T).float()

        t_pre = time.time() - t0
        print(f"  Prefill: {t_pre:.2f}s")

        t_dec = time.time()
        for step in range(max_new):
            lg = logits[0, -1, :] / temp
            sl, si = torch.sort(lg, descending=True)
            probs = torch.softmax(sl, dim=-1)
            mask = torch.cumsum(probs, dim=-1) - probs >= top_p
            sl[mask] = float("-inf")
            probs = torch.softmax(sl, dim=-1)
            nxt = si[torch.multinomial(probs, 1).item()].item()
            tokens.append(nxt)
            if nxt == self.tok.eos_token_id:
                break

            h = self.embed[torch.tensor([[nxt]])].half()
            pos = torch.tensor([[len(tokens) - 1]])
            h = self.forward(h, pos)
            h = self._rms(h, self.norm_w)
            logits = (h @ self.lm_w.T).float()

            if step % 5 == 0 and step > 0:
                e = time.time() - t_dec
                print(f"  Step {step}: {step/e:.2f} tok/s | {self.expert_cache.stats()}")

        total = time.time() - t0
        dec = time.time() - t_dec
        n = len(tokens) - len(ids[0])
        speed = n / dec if dec > 0 else 0
        print(f"\nPrefill: {t_pre:.2f}s, Decode: {dec:.2f}s ({n} tok, {speed:.2f} tok/s)")
        print(self.expert_cache.stats())
        return self.tok.decode(tokens)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else r"D:\project\大模型ssd化\models\qwen3-coder-bq8"
    print("=" * 50)
    print("ColorLM V5 - LRU Cache Inference")
    print("=" * 50)
    m = FastModel(d)
    r = m.generate("def fibonacci", max_new=50)
    print("\n" + r)
