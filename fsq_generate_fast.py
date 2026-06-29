# -*- coding: utf-8 -*-
"""
ColorLM V5 - Direct Reader, zero-copy BF16->F16
"""
import os, sys, torch, json, time, gc, struct
import numpy as np
from collections import OrderedDict
from transformers import AutoTokenizer

MAX_CACHED = 12

def read_safetensor(path, key):
    """Read tensor from safetensors directly, produce float16"""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        info = header[key]
        dtype_str = info["dtype"]
        shape = info["shape"]
        begin, end = info["data_offsets"]
        f.seek(8 + header_len + begin)
        raw = f.read(end - begin)

    n = 1
    for s in shape:
        n *= s

    if dtype_str == "BF16":
        # Chunked conversion to avoid large intermediate allocations
        CHUNK = 8 * 1024 * 1024  # 8M elements per chunk (~16MB)
        out = torch.empty(n, dtype=torch.float16)
        raw_bytes = bytes(raw)
        t16 = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int16)
        for i in range(0, n, CHUNK):
            end = min(i + CHUNK, n)
            chunk_f32 = (t16[i:end].to(torch.int32) << 16).view(torch.float32)
            out[i:end] = chunk_f32.half()
            del chunk_f32
        del t16
        return out.reshape(shape)
    elif dtype_str == "F16":
        return torch.from_numpy(np.frombuffer(raw, dtype="<f2").copy()).reshape(shape)
    elif dtype_str == "F32":
        return torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy()).reshape(shape).half()
    elif dtype_str == "U8":
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).reshape(shape)
    elif dtype_str == "I32":
        return torch.from_numpy(np.frombuffer(raw, dtype=np.int32).copy()).reshape(shape)
    elif dtype_str == "I64":
        return torch.from_numpy(np.frombuffer(raw, dtype=np.int64).copy()).reshape(shape)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


class FastModel:
    def __init__(self, model_dir):
        self.dir = model_dir
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
        self.inv = 1.0 / (10000.0 ** (torch.arange(0, self.HD, 2).float() / self.HD))

        self.cache = OrderedDict()
        self.ch = 0
        self.cm = 0

        print("Loading model (direct file read)...")
        self._load_all()

        self.kv_k = [None] * self.L
        self.kv_v = [None] * self.L
        print(f"Ready: {self.L}L, {self.NE}E, cache={MAX_CACHED}")

    def _load_all(self):
        self.embed = None
        self.norm_w = None
        self.lm_w = None
        self.layers = [{} for _ in range(self.L)]

        shard_keys = {}
        for key, shard in self.wmap.items():
            if "experts." in key or "._bq" in key:
                continue
            shard_keys.setdefault(shard, []).append(key)

        for si, (shard_name, keys) in enumerate(sorted(shard_keys.items())):
            path = os.path.join(self.dir, shard_name)
            for key in keys:
                val = read_safetensor(path, key)
                if key == "model.embed_tokens.weight":
                    self.embed = val
                elif key == "model.norm.weight":
                    self.norm_w = val
                elif key == "lm_head.weight":
                    self.lm_w = val
                elif key.startswith("model.layers."):
                    idx = int(key.split(".")[2])
                    short = key[len(f"model.layers.{idx}."):]
                    self.layers[idx][short] = val
                gc.collect()
            loaded = sum(1 for l in self.layers if l)
            print(f"  Shard {si+1}/16: {loaded}/{self.L} layers loaded")

    def _dequant_expert(self, key):
        path = os.path.join(self.dir, self.wmap[key])
        codes = read_safetensor(path, key + "._bq_codes")
        meta = read_safetensor(path, key + "._bq_meta")
        shape_t = read_safetensor(path, key + "._bq_shape")
        shape = [shape_t[0].item(), shape_t[1].item()]
        n = codes.shape[0]
        sc = ((meta[n:] - meta[:n]) / 255.0).unsqueeze(1)
        w = (codes.float() * sc + meta[:n].unsqueeze(1)).flatten()[:shape[0]*shape[1]].reshape(shape).half()
        del codes, meta, shape_t, sc
        gc.collect()
        return w

    def _get_expert(self, layer, expert, proj):
        ck = f"L{layer}E{expert}_{proj}"
        if ck in self.cache:
            self.ch += 1
            self.cache.move_to_end(ck)
            return self.cache[ck]
        self.cm += 1
        key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
        w = self._dequant_expert(key)
        self.cache[ck] = w
        self.cache.move_to_end(ck)
        while len(self.cache) > MAX_CACHED:
            self.cache.popitem(last=False)
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
                t = self.ch + self.cm
                r = self.ch / t * 100 if t > 0 else 0
                print(f"  Step {step}: {step/e:.2f} tok/s | cache {len(self.cache)}/{MAX_CACHED} hit {r:.0f}%")

        dec = time.time() - t_dec
        n = len(tokens) - len(ids[0])
        speed = n / dec if dec > 0 else 0
        t = self.ch + self.cm
        r = self.ch / t * 100 if t > 0 else 0
        print(f"\nPrefill: {t_pre:.2f}s, Decode: {dec:.2f}s ({n} tok, {speed:.2f} tok/s)")
        print(f"Cache: {len(self.cache)}/{MAX_CACHED}, hit: {r:.1f}%")
        return self.tok.decode(tokens)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else r"D:\project\大模型ssd化\models\qwen3-coder-bq8"
    print("=" * 50)
    print("ColorLM V5 - Direct File Reader")
    print("=" * 50)
    m = FastModel(d)
    r = m.generate("def fibonacci", max_new=50)
    print("\n" + r)



