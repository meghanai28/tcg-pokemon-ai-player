"""Score checkpoints on a held-out day, broken out per deck.

The whole-corpus top-1 number has repeatedly failed to predict a gate: the
elite-1100 model beat the champion by 5.6 top-1 points on elite play and then
gated 47.3%.  What that number cannot show is *which* decisions improved, and
for a deck swap that is the only thing that matters.  A model can be better
overall while still being unable to pilot the one list we intend to ship.

So this reports top-1 and option cross-entropy per 60-card list, for several
checkpoints side by side, on a day none of them trained on.  The question it
answers is narrow and decision-relevant: on Mega Lucario decisions, does the
new prior predict elite play better than the prior we currently ship?

Usage:
    py tools/score_by_deck.py --data data/bc_bal_lucario_holdout \\
        --models champion=data/model_champion_bc.npz new=data/model_x.npz \\
        --decks lucario=11400519690197900657 tech_grim=109660129574161690
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bc_train"))

import model as model_module  # noqa: E402
from model import TCGNet  # noqa: E402


def load_rows(data_dir, max_rows):
    keys = ("kind", "card", "scal", "mask", "ctx", "stype", "pi", "deck")
    parts = {k: [] for k in keys}
    kept = 0
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        with np.load(path) as data:
            if "deck" not in data:
                raise SystemExit(f"{path} has no `deck` column; re-ingest")
            for k in keys:
                parts[k].append(data[k])
            kept += len(data["pi"])
        if max_rows and kept >= max_rows:
            break
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def build_model(path, device):
    """Reconstruct a TCGNet from a champion-format npz.

    `model.py` carries its dimensions as module globals rather than
    constructor arguments, so they have to be set before instantiating --
    exactly as `train_bc.py` does it.
    """
    weights = np.load(path)
    meta = weights["_meta"]
    dim, layers, heads, d_ff = (int(x) for x in meta[:4])
    (model_module.D_MODEL, model_module.N_LAYERS,
     model_module.N_HEADS, model_module.D_FF) = dim, layers, heads, d_ff
    model = TCGNet()
    state = model.state_dict()
    missing = []
    for name in state:
        if name in weights.files:
            state[name] = torch.from_numpy(weights[name])
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"{path} is missing {len(missing)} tensors, e.g. "
                         f"{missing[:3]} -- wrong architecture")
    model.load_state_dict(state)
    return model.to(device).eval(), (dim, layers, heads, d_ff)


def score(model, rows, index, device, batch=512):
    """Top-1 agreement and option CE over `index` rows."""
    total = len(index)
    if total == 0:
        return None
    hits = 0.0
    ce_sum = 0.0
    with torch.no_grad():
        for start in range(0, total, batch):
            sel = index[start:start + batch]
            tensors = [torch.from_numpy(rows[k][sel]).to(device) for k in
                       ("kind", "card", "scal", "mask", "ctx", "stype")]
            kind, card, scal, mask, ctx, styp = tensors
            pi = torch.from_numpy(rows["pi"][sel]).to(device)
            logits, _value = model(kind.long(), card.long(), scal.float(),
                                   mask.float(), ctx.long(), styp.long())
            options = (kind == 3) & (mask > 0.5)
            option_logits = logits.masked_fill(~options, -torch.inf)
            logp = torch.log_softmax(option_logits, dim=-1)
            safe = torch.where(options, logp, torch.zeros_like(logp))
            ce_sum += float(-(pi * safe).sum(-1).sum())
            hits += float((option_logits.argmax(-1) == pi.argmax(-1)).sum())
    return hits / total, ce_sum / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", nargs="+", required=True,
                        help="name=path.npz, repeatable")
    parser.add_argument("--decks", nargs="+", default=[],
                        help="name=deckid, repeatable. Omit for all decks")
    parser.add_argument("--max-rows", type=int, default=250_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    args = parser.parse_args()

    rows = load_rows(args.data, args.max_rows)
    decks = rows["deck"]
    print(f"{len(decks)} holdout rows from {args.data}")

    groups = [("ALL", np.arange(len(decks)))]
    for spec in args.decks:
        name, _, raw = spec.partition("=")
        index = np.flatnonzero(decks == np.uint64(int(raw)))
        groups.append((name, index))

    results = {}
    for spec in args.models:
        name, _, path = spec.partition("=")
        model, shape = build_model(path, args.device)
        results[name] = {g: score(model, rows, idx, args.device)
                         for g, idx in groups}
        print(f"loaded {name}: dim/layers/heads/d_ff = {shape}")
        del model
        torch.cuda.empty_cache()

    names = list(results)
    print(f"\n{'deck':<14} {'rows':>8} " +
          " ".join(f"{n:>18}" for n in names))
    for group, index in groups:
        cells = []
        for name in names:
            got = results[name][group]
            cells.append("               -  " if got is None
                         else f"  top1 {got[0]:5.1%} ce {got[1]:.3f}")
        print(f"{group:<14} {len(index):>8} " + " ".join(cells))


if __name__ == "__main__":
    main()
