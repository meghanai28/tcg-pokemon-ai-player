"""ExIt distillation: train the network to imitate the search.

Policy loss = cross-entropy against the search's root visit distribution.
Value  loss = MSE against the game outcome (from the mover's perspective).

Gates reported at the end:
  * policy top-1 agreement with the search on a held-out split, compared
    against the HEURISTIC prior's agreement (the thing the net must beat --
    if the net doesn't beat the heuristic, the distillation is not working)
  * value MAE compared against a predict-the-mean baseline

Usage: py train/train_bc.py --epochs 12 --out train/model_bc.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "track1_search", "agent"))
sys.path.insert(0, HERE)

import nn_features as NF          # noqa: E402
from model import TCGNet, export_npz  # noqa: E402


def load_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no shards in {data_dir} -- run train/selfplay.py first")
    parts = {k: [] for k in ("kind", "card", "scal", "mask", "ctx", "stype", "pi", "z")}
    for f in files:
        d = np.load(f)
        for k in parts:
            parts[k].append(d[k])
    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    print(f"loaded {len(files)} shards, {out['pi'].shape[0]} samples")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default=os.path.join(HERE, "model_bc.npz"))
    ap.add_argument("--dim", type=int, default=96, help="model width")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--distill", default=None,
                    help="teacher .npz; student also matches its outputs")
    a = ap.parse_args()
    import model as _m
    _m.D_MODEL, _m.N_LAYERS, _m.N_HEADS, _m.D_FF = a.dim, a.layers, a.heads, 2 * a.dim

    d = load_data(a.data)
    n = d["pi"].shape[0]

    if a.distill:
        # Knowledge distillation: blend hard targets with the teacher's
        # predictions. The soft targets carry the teacher's ranking of EVERY
        # option (not just the played one) and its calibrated value estimates,
        # which is far more signal per sample than one-hot targets.
        from nn_infer import NumpyNet
        teacher = NumpyNet(a.distill)
        tl = np.zeros_like(d["pi"])
        tv = np.zeros_like(d["z"])
        B = 256
        for s in range(0, n, B):
            e = min(n, s + B)
            pol, val = teacher.forward(
                d["kind"][s:e].astype(np.int64), d["card"][s:e].astype(np.int64),
                d["scal"][s:e], d["mask"][s:e],
                d["ctx"][s:e].astype(np.int64), d["stype"][s:e].astype(np.int64))
            x = np.where(d["mask"][s:e] < 0.5, -1e9, pol)
            x = x - x.max(-1, keepdims=True)
            p = np.exp(x); p /= p.sum(-1, keepdims=True)
            tl[s:e] = p
            tv[s:e] = val
        d["pi"] = 0.5 * d["pi"] + 0.5 * tl.astype(np.float32)
        d["z"] = 0.5 * d["z"] + 0.5 * tv.astype(np.float32)
        print(f"distilling from {a.distill}: targets blended 50/50")
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(1, int(n * a.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def to_t(idx):
        return (torch.tensor(d["kind"][idx].astype(np.int64)),
                torch.tensor(d["card"][idx].astype(np.int64)),
                torch.tensor(d["scal"][idx]),
                torch.tensor(d["mask"][idx]),
                torch.tensor(d["ctx"][idx].astype(np.int64)),
                torch.tensor(d["stype"][idx].astype(np.int64)),
                torch.tensor(d["pi"][idx]),
                torch.tensor(d["z"][idx]))

    tr = to_t(tr_idx)
    va = to_t(val_idx)
    print(f"train {len(tr_idx)} / val {len(val_idx)}")

    model = TCGNet()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n_tr = len(tr_idx)
    steps = max(1, n_tr // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * steps, pct_start=0.3)

    for ep in range(a.epochs):
        model.train()
        order = torch.randperm(n_tr)
        tot_p = tot_v = 0.0
        for s in range(steps):
            b = order[s * a.batch:(s + 1) * a.batch]
            kind, card, scal, mask, ctx, styp, pi, z = (t[b] for t in tr)
            logits, v = model(kind, card, scal, mask, ctx, styp)
            logp = torch.log_softmax(logits, dim=-1)
            # cross-entropy against the search distribution (masked positions
            # carry zero target mass, so they contribute nothing)
            lp = -(pi * logp).sum(-1).mean()
            lv = Fn.mse_loss(v, z)
            loss = lp + 0.5 * lv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_p += lp.item(); tot_v += lv.item()

        model.eval()
        with torch.no_grad():
            kind, card, scal, mask, ctx, styp, pi, z = va
            logits, v = model(kind, card, scal, mask, ctx, styp)
            vlp = -(pi * torch.log_softmax(logits, -1)).sum(-1).mean().item()
            vlv = Fn.mse_loss(v, z).item()
            top1 = (logits.argmax(-1) == pi.argmax(-1)).float().mean().item()
        print(f"ep {ep+1}/{a.epochs}  train p={tot_p/steps:.4f} v={tot_v/steps:.4f}"
              f"  | val p={vlp:.4f} v={vlv:.4f} top1={top1*100:.1f}%")

    # ---------------- gates ----------------
    with torch.no_grad():
        kind, card, scal, mask, ctx, styp, pi, z = va
        logits, v = model(kind, card, scal, mask, ctx, styp)
        net_top1 = (logits.argmax(-1) == pi.argmax(-1)).float().mean().item()
        val_mae = (v - z).abs().mean().item()

    # Heuristic-prior baseline: encode() lays option tokens out in DESCENDING
    # heuristic score, so token OPT_BASE is exactly the heuristic's top pick.
    # Beating this is the falsifiable "distillation works" test.
    heur_top1 = (torch.full_like(pi.argmax(-1), NF.OPT_BASE) == pi.argmax(-1)).float().mean().item()
    base_mae = (z - z.mean()).abs().mean().item()

    print("\n==== GATES ====")
    print(f"policy top-1 vs search : net {net_top1*100:.1f}%  |  heuristic prior {heur_top1*100:.1f}%")
    print(f"value MAE              : net {val_mae:.3f}  |  predict-mean baseline {base_mae:.3f}")
    ok_p = net_top1 > heur_top1
    ok_v = val_mae < base_mae
    print(f"policy gate: {'PASS' if ok_p else 'FAIL'}   value gate: {'PASS' if ok_v else 'FAIL'}")

    export_npz(model, a.out)
    print(f"exported {a.out}")


if __name__ == "__main__":
    main()
