"""Create small archetype-specific shards from rich replay shards.

The rich encoder stores a stable deck-archetype card in ``card[:, 0]``.  This
utility filters one compressed shard at a time, so targeted fine-tuning never
needs to materialize the multi-million-decision corpus in RAM.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGENT = os.path.join(ROOT, "track1_search", "agent")
TRAIN = os.path.join(ROOT, "track1_search", "train")
sys.path.insert(0, AGENT)
sys.path.insert(0, TRAIN)

import nn_features_rich as features  # noqa: E402
from cg.engine import get_lib  # noqa: E402


def read_deck(path):
    with open(path, encoding="utf-8") as source:
        cards = [int(line.strip()) for line in source if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(cards)}")
    return cards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-per-shard", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260801)
    args = parser.parse_args()
    if args.max_per_shard < 0:
        parser.error("--max-per-shard must be non-negative")

    card_db = {
        card["cardId"]: card
        for card in json.loads(get_lib().AllCard().decode())
    }
    anchor = features._deck_anchor(read_deck(args.deck), card_db)
    files = sorted(
        path for directory in args.data
        for path in glob.glob(os.path.join(directory, "*.npz"))
    )
    if not files:
        parser.error("no .npz shards found")

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for file_index, path in enumerate(files):
        with np.load(path) as shard:
            indices = np.flatnonzero(shard["card"][:, 0] == anchor)
            if args.max_per_shard and len(indices) > args.max_per_shard:
                rng = np.random.default_rng(args.seed + file_index)
                indices = np.sort(rng.choice(
                    indices, size=args.max_per_shard, replace=False))
            if not len(indices):
                print(f"skip {os.path.basename(path)}: 0 anchor-{anchor} decisions")
                continue
            output = {}
            n = len(shard["pi"])
            for key in shard.files:
                value = shard[key]
                output[key] = value[indices] if value.ndim and len(value) == n else value
        target = os.path.join(args.out, os.path.basename(path))
        np.savez_compressed(target, **output)
        total += len(indices)
        print(f"wrote {target}: {len(indices)} anchor-{anchor} decisions")
    if not total:
        raise SystemExit(f"no decisions found for deck anchor {anchor}")
    print(f"total: {total} decisions; anchor={anchor}")


if __name__ == "__main__":
    main()
