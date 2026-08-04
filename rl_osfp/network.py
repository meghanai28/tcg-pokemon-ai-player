"""Masked actor-critic used by the clean OSFP track.

The network scores every legal option, predicts how many options to select,
and estimates the terminal win/loss value.  Multi-card selections are sampled
without replacement, so PPO optimizes the complete action rather than only the
first card in a set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from foundation import nn_features_rich as NF

MAX_COUNT = 60


@dataclass(frozen=True)
class NetworkConfig:
    d_model: int = 192
    layers: int = 6
    heads: int = 6
    d_ff: int = 384
    dropout: float = 0.0

    def validate(self) -> None:
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if not (32 <= self.d_model <= 256 and 1 <= self.layers <= 8):
            raise ValueError("network configuration is outside safe bounds")


class Block(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        batch, seq, dim = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(
            batch, seq, 3, cfg.heads, dim // cfg.heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(dim // cfg.heads)
        att = (att + attn_mask).softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(batch, seq, dim)
        x = x + F.dropout(self.proj(out), cfg.dropout, self.training)
        h = self.ln2(x)
        h = self.fc2(F.gelu(self.fc1(h), approximate="tanh"))
        return x + F.dropout(h, cfg.dropout, self.training)


class ActorCritic(nn.Module):
    def __init__(self, cfg: NetworkConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetworkConfig()
        self.cfg.validate()
        dim = self.cfg.d_model
        self.card_emb = nn.Embedding(NF.N_CARD, dim)
        self.kind_emb = nn.Embedding(NF.N_KIND, dim)
        self.ctx_emb = nn.Embedding(NF.N_CTX, dim)
        self.stype_emb = nn.Embedding(NF.N_STYPE, dim)
        self.scal_proj = nn.Linear(NF.F, dim)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.option_head = nn.Linear(dim, 1)
        self.count_head = nn.Linear(dim, MAX_COUNT + 1)
        self.value_fc1 = nn.Linear(dim, 64)
        self.value_fc2 = nn.Linear(64, 1)
        nn.init.normal_(self.card_emb.weight, std=0.02)

    def forward(
        self,
        kind: torch.Tensor,
        card: torch.Tensor,
        scal: torch.Tensor,
        mask: torch.Tensor,
        ctx: torch.Tensor,
        stype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.card_emb(card) + self.kind_emb(kind) + self.scal_proj(scal)
        x = F.dropout(x, self.cfg.dropout, self.training)
        x[:, 0, :] = x[:, 0, :] + self.ctx_emb(ctx) + self.stype_emb(stype)
        x = x * mask[:, :, None]
        attn_mask = (1.0 - mask)[:, None, None, :] * -1e9
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.ln_f(x)
        option = self.option_head(x).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        count = self.count_head(x[:, 0, :])
        value = torch.tanh(
            self.value_fc2(F.gelu(self.value_fc1(x[:, 0, :]), approximate="tanh"))
        ).squeeze(-1)
        return option, count, value


def export_npz(model: ActorCritic, path: str) -> None:
    state = {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in model.state_dict().items()
    }
    cfg = model.cfg
    state["_meta"] = np.asarray(
        [cfg.d_model, cfg.layers, cfg.heads, cfg.d_ff, MAX_COUNT, 2],
        dtype=np.int64,
    )
    np.savez_compressed(path, **state)


def load_npz(path: str, device: str | torch.device = "cpu") -> ActorCritic:
    with np.load(path) as weights:
        meta = [int(x) for x in weights["_meta"]]
        if len(meta) < 6 or meta[5] != 2:
            raise ValueError("not a clean OSFP v2 checkpoint")
        cfg = NetworkConfig(meta[0], meta[1], meta[2], meta[3], 0.0)
        model = ActorCritic(cfg)
        state = model.state_dict()
        for key, target in state.items():
            if key not in weights or tuple(weights[key].shape) != tuple(target.shape):
                raise ValueError(f"checkpoint mismatch at {key}")
            state[key] = torch.as_tensor(weights[key])
    model.load_state_dict(state)
    return model.to(device)


def config_dict(model: ActorCritic) -> dict[str, int | float]:
    return asdict(model.cfg)
