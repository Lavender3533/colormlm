# -*- coding: utf-8 -*-
import torch, torch.nn as nn, torch.nn.functional as F, math

class FSQEncoder(nn.Module):
    def __init__(self, d_model, n_groups=16, n_levels=8):
        super().__init__()
        self.n_groups = n_groups
        self.n_levels = n_levels
        self.proj = nn.Linear(d_model, n_groups)
    def forward(self, x):
        raw = self.proj(x)
        normed = torch.sigmoid(raw)
        h = torch.clamp((normed * self.n_levels).long(), 0, self.n_levels - 1)
        s = normed * (self.n_levels - 1)
        return (h.float() - s).detach() + s

class FSQCodebookFFN(nn.Module):
    def __init__(self, d_model, n_experts=16, n_levels=8, n_groups=16, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_levels = n_levels
        self.n_groups = n_groups
        self.top_k = top_k
        self.encoder = FSQEncoder(d_model, n_groups, n_levels)
        self.expert_codebooks = nn.Parameter(torch.randn(n_experts, n_levels, d_model) * 0.02)
        self.router = nn.Linear(d_model, n_experts)
        self.act = nn.GELU()
    def forward(self, x):
        B, S, D = x.shape
        rl = self.router(x)
        tw, ti = torch.topk(rl, k=self.top_k, dim=-1)
        tw = F.softmax(tw, dim=-1)
        codes = self.encoder(x)
        ci = codes.long().clamp(0, self.n_levels - 1)
        out = torch.zeros(B, S, D, device=x.device)
        for k in range(self.top_k):
            ei = ti[:, :, k]
            w = tw[:, :, k:k+1]
            ecb = self.expert_codebooks[ei]
            ce = ci.unsqueeze(-1).expand(-1, -1, -1, D)
            v = torch.gather(ecb, 2, ce)
            eo = v.mean(dim=2)
            out = out + w * eo
        return self.act(out)

class StdAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.nh = n_heads
        self.hd = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, x, mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.nh, self.hd)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        at = (q @ k.transpose(-2,-1)) / math.sqrt(self.hd)
        if mask is not None:
            at = at.masked_fill(mask, float('-inf'))
        at = F.softmax(at, dim=-1)
        o = (at @ v).transpose(1,2).reshape(B, S, D)
        return self.out(o)

class FSQLayer(nn.Module):
    def __init__(self, d, nh, ne, nl, ng, tk):
        super().__init__()
        self.attn = StdAttn(d, nh)
        self.ffn = FSQCodebookFFN(d, ne, nl, ng, tk)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.ffn(self.n2(x))
        return x

class FSQTransformer(nn.Module):
    def __init__(self, vocab=151644, d=256, nh=4, nl=6, ne=16, nlev=8, ng=16, tk=2, ms=512):
        super().__init__()
        self.d = d
        self.te = nn.Embedding(vocab, d)
        self.pe = nn.Embedding(ms, d)
        self.ed = nn.Dropout(0.1)
        self.layers = nn.ModuleList([FSQLayer(d, nh, ne, nlev, ng, tk) for _ in range(nl)])
        self.no = nn.LayerNorm(d)
        self.lm = nn.Linear(d, vocab)
        self.dp = None
    def forward(self, ids, mask=None, th=None):
        B, S = ids.shape
        x = self.ed(self.te(ids) + self.pe(torch.arange(S, device=ids.device)))
        pm = None
        if mask is not None:
            pm = (mask.unsqueeze(1).unsqueeze(2) == 0)
        hs = []
        for l in self.layers:
            x = l(x, pm)
            hs.append(x)
        x = self.no(x)
        logits = self.lm(x)
        dl = torch.tensor(0.0, device=x.device)
        if th is not None:
            sh = hs[-1]
            if self.dp is None:
                self.dp = nn.Linear(self.d, th.shape[-1]).to(x.device)
            dl = F.mse_loss(self.dp(sh), th)
        return logits, dl
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == '__main__':
    print('=== FSQ-Native Transformer ===')
    m = FSQTransformer()
    p = m.count_params()
    print('Params:', p)
    ids = torch.randint(0, 151644, (2, 64))
    logits, dl = m(ids)
    print('Output:', logits.shape)
    print('FFN compression:', round(2*256*1024*6 / (16*8*256*6)), 'x')
