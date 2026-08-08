"""Keep only the decisions made by strong players.

The champion's prior was cloned from an Elo-1000-and-up corpus whose **mean is
1094**, so nearly half of what it learned came from players between 1000 and
1100. The top of the ladder is at 1284 and our agent sits around 940, which
means the corpus contains a genuinely better teacher than either our agent or
our own search, diluted by everyone else.

This is the one teacher in this project that is unambiguously stronger than the
student and carries no circularity: expert iteration distils our own search, and
self-play RL distils our own league, but a 1200-rated human is simply better
than us and has nothing to do with our mistakes.

Shards already carry a per-decision `elo`, so this is a filter rather than a
re-ingest. Processing is one shard at a time on purpose: the full corpus is
1.92M decisions, which is about 14 GiB of tensors, and loading it whole is
exactly the mistake that killed the VM twice (see tools/resource_guard.py).

Usage:
    py tools/filter_by_elo.py --data data/bc_general_train --min-elo 1200 \
        --out data/bc_elite_1200
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEAK_BYTES_PER_DECISION = 18990


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--min-elo", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-frac", type=float, default=0.0,
                        help="fraction of shards reserved as a holdout directory")
    args = parser.parse_args()

    files: list[str] = []
    for directory in args.data:
        files += sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not files:
        raise SystemExit(f"no shards under {args.data}")

    os.makedirs(args.out, exist_ok=True)
    holdout_dir = args.out.rstrip("/") + "_holdout"
    n_holdout = int(len(files) * args.holdout_frac)
    if n_holdout:
        os.makedirs(holdout_dir, exist_ok=True)

    kept_total = seen_total = 0
    for index, path in enumerate(files):
        with np.load(path) as shard:
            elo = shard["elo"]
            keep = elo >= args.min_elo
            seen_total += len(elo)
            if not keep.any():
                print(f"  {os.path.basename(path)}: 0 of {len(elo):,} kept")
                continue
            target_dir = holdout_dir if index < n_holdout else args.out
            out_path = os.path.join(target_dir, os.path.basename(path))
            payload = {}
            for key in shard.files:
                value = shard[key]
                # `features` is scalar metadata, not a per-decision column
                payload[key] = value if value.ndim == 0 else value[keep]
            np.savez_compressed(out_path, **payload)
            kept_total += int(keep.sum())
            print(f"  {os.path.basename(path)}: {int(keep.sum()):>7,} of "
                  f"{len(elo):>7,} kept -> {os.path.basename(target_dir)}")

    peak = kept_total * PEAK_BYTES_PER_DECISION / 2 ** 30
    print(f"\nkept {kept_total:,} of {seen_total:,} decisions "
          f"({100 * kept_total / max(seen_total, 1):.1f}%) at elo >= {args.min_elo:.0f}")
    print(f"training on this needs about {peak:.1f} GiB peak RSS "
          f"(cap is 15.0, see tools/resource_guard.py)")
    if peak > 15.0:
        print("WARNING: over the cap. Raise --min-elo or pass --max-per-shard to the trainer.")


if __name__ == "__main__":
    main()
