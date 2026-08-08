"""Cap each deck's share of a behaviour-cloning corpus, and split by day.

The champion prior was cloned from an Elo-1000+ corpus that is 36% one deck,
and it came out unable to pilot anything else: every external deck swap under
it lost, with win rate tracking Jaccard-to-Tech-Grim almost monotonically.
That is a property of the *corpus*, not of the architecture.  In the Aug 1-6
dumps the same imbalance is worse -- 714,058 elite decisions on Tech-Grim
against 63,113 on Mega Lucario, an 11:1 ratio -- so a model trained on the raw
mixture would again learn one deck and treat the rest as noise.

This caps rows per distinct 60-card list so no deck can dominate, and
optionally leaves one deck uncapped when a specialist is the goal.

It also splits on the shard's date, because the honest validation question is
"does this predict a day it never saw", not "does this predict a random 15% of
the rows it did see".  Shard filenames carry the dump date, so holding out a day
is a filename filter.

Two passes, never more than one shard resident: pass one reads only the `deck`
column to count, pass two subsamples and writes.  Loading the whole corpus to
do this would be ~14 GiB and is how this machine's VM was killed twice.

Usage:
    py tools/balance_corpus.py --data data/bc_elite_aug --out data/bc_bal \\
        --cap-per-deck 120000 --keep-all <deckid> --holdout-day 2026-08-06
"""
from __future__ import annotations

import argparse
import collections
import glob
import os

import numpy as np

FIELDS = ("kind", "card", "scal", "mask", "ctx", "stype", "pi", "z",
          "group", "seat", "pilot", "elo", "deck")


def shard_day(path: str) -> str:
    """`bc_2026-08-03_00_60050.npz` -> `2026-08-03`."""
    base = os.path.basename(path)
    parts = base.split("_")
    return parts[1] if len(parts) > 2 else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-day", default=None,
                        help="shards from this dump date go to --holdout-out "
                             "instead, giving a real temporal holdout")
    parser.add_argument("--holdout-out", default=None)
    parser.add_argument("--cap-per-deck", type=int, default=120_000)
    parser.add_argument("--keep-all", type=int, action="append", default=[],
                        help="deck id exempt from the cap; repeatable")
    parser.add_argument("--min-deck-rows", type=int, default=2_000,
                        help="drop lists too rare to teach anything")
    parser.add_argument("--seed", type=int, default=917)
    args = parser.parse_args()

    shards = sorted(f for d in args.data for f in glob.glob(os.path.join(d, "*.npz")))
    if not shards:
        raise SystemExit(f"no shards under {args.data}")
    train_shards = [s for s in shards if shard_day(s) != args.holdout_day]
    hold_shards = [s for s in shards if shard_day(s) == args.holdout_day]
    print(f"{len(shards)} shards: {len(train_shards)} train, "
          f"{len(hold_shards)} holdout ({args.holdout_day or 'none'})")

    # pass one: how many rows does each list have, across the training shards
    counts: collections.Counter = collections.Counter()
    for path in train_shards:
        with np.load(path) as data:
            counts.update(collections.Counter(data["deck"].tolist()))
    print(f"{len(counts)} distinct lists, {sum(counts.values())} rows")

    keep_all = set(args.keep_all)
    quota = {}
    for deck_id, total in counts.items():
        if total < args.min_deck_rows:
            quota[deck_id] = 0
        elif deck_id in keep_all:
            quota[deck_id] = total
        else:
            quota[deck_id] = min(total, args.cap_per_deck)
    target = sum(quota.values())
    print(f"after capping at {args.cap_per_deck}/list: {target} rows")
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
    for deck_id, total in top:
        print(f"   list {deck_id:>22}  {total:>8} -> {quota[deck_id]:>8}")

    # pass two: keep each row with probability quota/total, then write
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    written = 0
    for index, path in enumerate(train_shards):
        with np.load(path) as data:
            decks = data["deck"]
            keep = np.zeros(len(decks), dtype=bool)
            for deck_id in np.unique(decks):
                total = counts[int(deck_id)]
                allowed = quota.get(int(deck_id), 0)
                if allowed <= 0:
                    continue
                rows = np.flatnonzero(decks == deck_id)
                if allowed >= total:
                    keep[rows] = True
                else:
                    take = rng.random(len(rows)) < (allowed / total)
                    keep[rows[take]] = True
            if not keep.any():
                continue
            out_path = os.path.join(
                args.out, f"bal_{index:03d}_{int(keep.sum())}.npz")
            np.savez_compressed(
                out_path, **{k: data[k][keep] for k in FIELDS if k in data})
            written += int(keep.sum())
        print(f"  {os.path.basename(path)} -> {int(keep.sum())} rows", flush=True)
    print(f"wrote {written} rows to {args.out}")

    if hold_shards and args.holdout_out:
        os.makedirs(args.holdout_out, exist_ok=True)
        held = 0
        for index, path in enumerate(hold_shards):
            with np.load(path) as data:
                decks = data["deck"]
                keep = np.isin(decks, [d for d, q in quota.items() if q > 0])
                if not keep.any():
                    continue
                out_path = os.path.join(
                    args.holdout_out, f"hold_{index:03d}_{int(keep.sum())}.npz")
                np.savez_compressed(
                    out_path, **{k: data[k][keep] for k in FIELDS if k in data})
                held += int(keep.sum())
        print(f"wrote {held} holdout rows to {args.holdout_out}")


if __name__ == "__main__":
    main()
