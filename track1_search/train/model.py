"""Policy/value transformer for PTCG decisions (PyTorch training side).

Architecture (mirrored exactly by submission/nn_infer.py in numpy):
  token embedding = CardEmb[card] + KindEmb[kind] + Linear(scalars)
  + (global token only) CtxEmb[select.context] + STypeEmb[select.type]
  -> pre-LN transformer encoder x L (MHA + GELU MLP, residual)
  -> policy: per-option-token logit  (pointer head)
  -> value:  tanh(MLP(global token))

Weights are exported to a flat .npz consumed by the numpy inference module.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission"))
import nn_features as NF  # noqa: E402

D_MODEL = 96
N_LAYERS = 3
N_HEADS = 4
D_FF = 192


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)

    def forward(self, x, attn_mask):
        # x: [B, S, D]; attn_mask: [B, 1, 1, S] additive (-inf on padding)
        B, S, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, S, 3, N_HEADS, D // N_HEADS).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # [B, H, S, Dh]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // N_HEADS)
        att = att + attn_mask
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.proj(out)
        h = self.ln2(x)
        x = x + self.fc2(Fn.gelu(self.fc1(h), approximate="tanh"))
        return x


class TCGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_emb = nn.Embedding(NF.N_CARD, D_MODEL)
        self.kind_emb = nn.Embedding(NF.N_KIND, D_MODEL)
        self.ctx_emb = nn.Embedding(NF.N_CTX, D_MODEL)
        self.stype_emb = nn.Embedding(NF.N_STYPE, D_MODEL)
        self.scal_proj = nn.Linear(NF.F, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.pol_head = nn.Linear(D_MODEL, 1)
        self.val_fc1 = nn.Linear(D_MODEL, 64)
        self.val_fc2 = nn.Linear(64, 1)
        nn.init.normal_(self.card_emb.weight, std=0.02)

    def forward(self, kind, card, scal, mask, ctx_id, stype_id):
        """kind/card: [B,S] long; scal: [B,S,F]; mask: [B,S]; ctx/stype: [B]."""
        x = self.card_emb(card) + self.kind_emb(kind) + self.scal_proj(scal)
        g = self.ctx_emb(ctx_id) + self.stype_emb(stype_id)   # [B, D]
        x = torch.cat([x[:, :1, :] + g[:, None, :], x[:, 1:, :]], dim=1)
        x = x * mask[:, :, None]
        attn_mask = (1.0 - mask)[:, None, None, :] * -1e9
        for b in self.blocks:
            x = b(x, attn_mask)
        x = self.ln_f(x)
        pol_logits = self.pol_head(x).squeeze(-1)             # [B, S]
        pol_logits = pol_logits.masked_fill(mask < 0.5, -1e9)
        v = torch.tanh(self.val_fc2(
            Fn.gelu(self.val_fc1(x[:, 0, :]), approximate="tanh"))).squeeze(-1)
        return pol_logits, v


def export_npz(model: TCGNet, path: str):
    out = {}
    sd = model.state_dict()
    for k, v in sd.items():
        out[k] = v.detach().cpu().numpy().astype(np.float32)
    out["_meta"] = np.array([D_MODEL, N_LAYERS, N_HEADS, D_FF], dtype=np.int64)
    np.savez_compressed(path, **out)


if __name__ == "__main__":
    m = TCGNet()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n:,}")
    export_npz(m, os.path.join(os.path.dirname(__file__), "model_init.npz"))
    print("exported model_init.npz")
