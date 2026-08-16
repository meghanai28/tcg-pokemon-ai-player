"""Derive per-state value targets for a leaf evaluator from DAgger search output.

Why not the game outcome
------------------------
The value head shipped in every model so far was trained on ``z``, the final
game result in {-1, +1}. main.py records what that produced: 46% of its outputs
saturate above 0.95 (versus 0% for the handcrafted evaluator), so PUCT commits
to a line early instead of verifying it, and net leaves measured 1W-19L.

A leaf evaluator is not predicting who eventually won. It is predicting what the
search would conclude if it kept expanding. ``dagger_generate.py`` already
records exactly that: per-option searched Q values (``q``), their validity mask
(``q_mask``), and visit counts (``action_visits``). Those live in a narrow band
around zero, so a head fit to them is calibrated by construction.

Targets offered
---------------
``visit``  visit-weighted mean Q over legal options. The search's own estimate
           of the state, and the closest analogue to a PUCT backup value.
``max``    best legal Q. Optimistic, matches a greedy readout.
``mix``    0.5 * visit + 0.5 * max.

All are expressed from the perspective of the player to move, which is the
convention ``_net_values`` already expects and negates for opponent nodes.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def state_values(shard: dict, target: str = "visit") -> tuple[np.ndarray, np.ndarray]:
    """Return (values, valid) for one loaded shard.

    ``valid`` marks rows with at least one searched option; rows without one
    carry no usable signal and must be dropped rather than defaulted to zero.
    """
    q = np.asarray(shard["q"], dtype=np.float32)
    mask = (np.asarray(shard["q_mask"]) > 0.5) if "q_mask" in shard else np.isfinite(q)
    if "action_visits" in shard:
        visits = np.asarray(shard["action_visits"], dtype=np.float32)
        if visits.shape != q.shape:
            visits = np.ones_like(q)
    else:
        visits = np.ones_like(q)

    visits = np.where(mask, np.maximum(visits, 0.0), 0.0)
    qm = np.where(mask, q, 0.0)

    total = visits.sum(axis=1)
    valid = (mask.sum(axis=1) > 0) & np.isfinite(total)
    # Fall back to an unweighted mean when a row has options but no visit counts.
    denom = np.where(total > 0, total, np.maximum(mask.sum(axis=1), 1))
    weights = np.where(total[:, None] > 0, visits, mask.astype(np.float32))
    visit_mean = (qm * weights).sum(axis=1) / denom

    neg_inf = np.full_like(q, -np.inf)
    best = np.where(mask, q, neg_inf).max(axis=1)
    best = np.where(np.isfinite(best), best, 0.0)

    if target == "visit":
        values = visit_mean
    elif target == "max":
        values = best
    elif target == "mix":
        values = 0.5 * visit_mean + 0.5 * best
    else:
        raise ValueError(f"unknown target {target!r}")
    return values.astype(np.float32), valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--target", default="visit",
                        choices=("visit", "max", "mix"))
    args = parser.parse_args()

    paths = sorted(p for source in args.data
                   for p in ([source] if source.endswith(".npz")
                             else glob.glob(os.path.join(source, "*.npz"))))
    if not paths:
        raise SystemExit(f"no shards under {args.data}")

    all_v, all_z, rows, dropped = [], [], 0, 0
    for path in paths:
        with np.load(path) as source:
            shard = {k: source[k] for k in source.files}
        values, valid = state_values(shard, args.target)
        rows += len(values)
        dropped += int((~valid).sum())
        all_v.append(values[valid])
        if "z" in shard:
            all_z.append(np.asarray(shard["z"], dtype=np.float32)[valid])

    v = np.concatenate(all_v) if all_v else np.zeros(0, np.float32)
    print(f"{len(paths)} shard(s), {rows:,} rows, {dropped:,} without a searched "
          f"option (dropped)")
    if not len(v):
        raise SystemExit("no usable rows")
    print(f"target '{args.target}': mean {v.mean():+.4f}  std {v.std():.4f}  "
          f"min {v.min():+.4f}  max {v.max():+.4f}")
    sat = float((np.abs(v) > 0.95).mean())
    print(f"saturated |value|>0.95: {sat:.1%}   "
          f"(the shipped outcome-trained head saturates 46% of the time)")
    if all_z:
        z = np.concatenate(all_z)
        print(f"outcome z for the same rows: mean {z.mean():+.4f} "
              f"std {z.std():.4f}, saturated {float((np.abs(z) > 0.95).mean()):.1%}")


if __name__ == "__main__":
    main()
