"""Numpy-only inference for the policy/value transformer (2-vCPU friendly).

Exact mirror of train/model.py::TCGNet. Loads the flat .npz exported by
export_npz(). Batched forward: ~1-3 ms for a batch of 32 on 2 CPU cores.
"""
from __future__ import annotations

import numpy as np


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class NumpyNet:
    def __init__(self, path):
        z = np.load(path)
        self.w = {k: z[k] for k in z.files}
        meta = self.w["_meta"]
        self.d_model, self.n_layers, self.n_heads, self.d_ff = (int(x) for x in meta)

    def forward(self, kind, card, scal, mask, ctx_id, stype_id):
        """All args numpy. kind/card:[B,S] int, scal:[B,S,F], mask:[B,S] f32,
        ctx_id/stype_id:[B] int. Returns (pol_logits [B,S], value [B])."""
        w = self.w
        D, H = self.d_model, self.n_heads
        Dh = D // H
        x = (w["card_emb.weight"][card]
             + w["kind_emb.weight"][kind]
             + scal @ w["scal_proj.weight"].T + w["scal_proj.bias"])
        g = w["ctx_emb.weight"][ctx_id] + w["stype_emb.weight"][stype_id]
        x[:, 0, :] += g
        x = x * mask[:, :, None]
        B, S, _ = x.shape
        neg = (1.0 - mask)[:, None, None, :] * -1e9

        for li in range(self.n_layers):
            p = f"blocks.{li}."
            h = _layernorm(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            qkv = h @ w[p + "qkv.weight"].T + w[p + "qkv.bias"]      # [B,S,3D]
            qkv = qkv.reshape(B, S, 3, H, Dh).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]                          # [B,H,S,Dh]
            att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(Dh)
            att = _softmax(att + neg)
            out = (att @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
            x = x + out @ w[p + "proj.weight"].T + w[p + "proj.bias"]
            h = _layernorm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            h = _gelu(h @ w[p + "fc1.weight"].T + w[p + "fc1.bias"])
            x = x + h @ w[p + "fc2.weight"].T + w[p + "fc2.bias"]

        x = _layernorm(x, w["ln_f.weight"], w["ln_f.bias"])
        pol = (x @ w["pol_head.weight"].T + w["pol_head.bias"]).squeeze(-1)
        pol = np.where(mask < 0.5, -1e9, pol)
        h = _gelu(x[:, 0, :] @ w["val_fc1.weight"].T + w["val_fc1.bias"])
        val = np.tanh((h @ w["val_fc2.weight"].T + w["val_fc2.bias"]).squeeze(-1))
        return pol, val
