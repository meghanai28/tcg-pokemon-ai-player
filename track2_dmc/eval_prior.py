"""Evaluate deployable CPU priors on logged search/leaderboard decisions.

This deliberately measures only option tokens, matching ``act()`` and the
competition agent's ``_net_scores``.  It is a checkpoint-selection diagnostic,
not a substitute for game-level A/B testing.

Usage:
  python track2_dmc/eval_prior.py model_a.npz model_b.npz
  python track2_dmc/eval_prior.py model.npz --limit 5000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "track1_search", "agent"))

import nn_features as NF  # noqa: E402
from nn_infer import NumpyNet  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "track1_search", "train"))
from model import TCGNet  # noqa: E402


def evaluate(model_path, data, indices, batch, backend, device):
    if backend == "numpy":
        model = NumpyNet(model_path)
    else:
        model = TCGNet().to(device)
        weights = np.load(model_path)
        state = model.state_dict()
        with torch.no_grad():
            for key, value in state.items():
                if key in weights and tuple(weights[key].shape) == tuple(value.shape):
                    value.copy_(torch.as_tensor(weights[key], device=device))
        model.eval()
    agree = cross_entropy = entropy = spread = 0.0
    seen = 0
    for start in range(0, len(indices), batch):
        ix = indices[start:start + batch]
        kind = data["kind"][ix].astype(np.int64)
        mask = data["mask"][ix]
        card = data["card"][ix].astype(np.int64)
        scal = data["scal"][ix]
        ctx = data["ctx"][ix].astype(np.int64)
        stype = data["stype"][ix].astype(np.int64)
        if backend == "numpy":
            logits, _value = model.forward(kind, card, scal, mask, ctx, stype)
        else:
            with torch.no_grad():
                logits_t, _value = model(
                    torch.as_tensor(kind, device=device),
                    torch.as_tensor(card, device=device),
                    torch.as_tensor(scal, device=device),
                    torch.as_tensor(mask, device=device),
                    torch.as_tensor(ctx, device=device),
                    torch.as_tensor(stype, device=device))
            logits = logits_t.cpu().numpy()
        options = (kind == 3) & (mask > 0.5)
        masked = np.where(options, logits, -np.inf)
        top = masked.argmax(-1)
        target = data["pi"][ix]
        agree += float((top == target.argmax(-1)).sum())

        shifted = masked - masked.max(-1, keepdims=True)
        prob = np.where(options, np.exp(shifted), 0.0)
        prob /= prob.sum(-1, keepdims=True)
        logp = np.log(np.maximum(prob, 1e-12))
        cross_entropy += float(-(target * logp).sum())
        n_options = options.sum(-1)
        entropy += float((-(prob * logp).sum(-1) / np.log(n_options)).sum())
        hi = np.where(options, logits, -np.inf).max(-1)
        lo = np.where(options, logits, np.inf).min(-1)
        spread += float((hi - lo).sum())
        seen += len(ix)
    return {
        "agreement": agree / seen,
        "cross_entropy": cross_entropy / seen,
        "entropy": entropy / seen,
        "spread": spread / seen,
        "n": seen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--data", default=os.path.join(
        ROOT, "track1_search", "train", "data_bc", "bc_917eps.npz"))
    ap.add_argument("--limit", type=int, default=5053)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=917)
    ap.add_argument("--backend", choices=("torch", "numpy"), default="torch")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args()

    device_name = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable")
    device = torch.device(device_name)

    data = np.load(args.data)
    rng = np.random.default_rng(args.seed)
    n = len(data["pi"])
    indices = rng.choice(n, size=min(args.limit, n), replace=False)
    print(f"held-out diagnostics on {len(indices)} decisions "
          f"({args.backend} inference on {device if args.backend == 'torch' else 'cpu'})")
    print(f"{'model':32} {'agree':>8} {'xent':>8} {'entropy':>9} {'spread':>8}")
    for path in args.models:
        metric = evaluate(path, data, indices, args.batch, args.backend, device)
        print(f"{os.path.basename(path):32} "
              f"{100*metric['agreement']:7.2f}% "
              f"{metric['cross_entropy']:8.4f} "
              f"{metric['entropy']:9.4f} "
              f"{metric['spread']:8.3f}")


if __name__ == "__main__":
    main()
