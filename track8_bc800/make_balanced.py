"""Build bounded 800-999 and 1000+ train/holdout shards.

The Elo-800 replay shards add breadth, but they must not swamp the proven
high-rated signal.  This script samples the tiers independently and writes
small shards that train_bc.py can concatenate without threatening WSL RAM.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HISTORICAL_HI = os.path.join(
    ROOT, "track1_search", "train", "data_bc_rich_deck_21k")
KEYS = ("kind", "card", "scal", "mask", "ctx", "stype", "pi", "z",
        "group", "seat", "pilot", "elo")


def _select(path: str, tier: str, limit: int, seed: int, out_path: str) -> int:
    with np.load(path) as source:
        missing = [key for key in KEYS if key not in source]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        elo = source["elo"]
        if tier == "low":
            eligible = np.flatnonzero((elo >= 800.0) & (elo < 1000.0))
        else:
            eligible = np.flatnonzero(elo >= 1000.0)
        rng = np.random.default_rng(seed)
        if len(eligible) > limit:
            eligible = np.sort(rng.choice(eligible, size=limit, replace=False))
        payload = {key: source[key][eligible] for key in KEYS}
        payload["features"] = np.array("rich")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return len(eligible)


def _single(directory: str) -> str:
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if len(files) != 1:
        raise ValueError(f"expected one shard in {directory}, found {len(files)}")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-dir", default=os.path.join(HERE, "data_elo800"))
    parser.add_argument("--high-dir", default=os.path.join(HERE, "data_elo800"))
    parser.add_argument("--low-holdout-dir", default=os.path.join(
        HERE, "data_elo800_holdout"))
    parser.add_argument("--high-holdout-dir", default=os.path.join(
        HERE, "data_elo800_holdout"))
    parser.add_argument("--historical-high-dir", default=HISTORICAL_HI)
    parser.add_argument("--historical-per-shard", type=int, default=50000)
    parser.add_argument("--train-per-tier-day", type=int, default=75000)
    parser.add_argument("--holdout-per-tier", type=int, default=50000)
    parser.add_argument("--out-train", default=os.path.join(HERE, "data_train"))
    parser.add_argument("--out-holdout", default=os.path.join(HERE, "data_holdout"))
    args = parser.parse_args()

    low_files = sorted(glob.glob(os.path.join(args.low_dir, "*.npz")))
    high_files = sorted(glob.glob(os.path.join(args.high_dir, "*.npz")))
    if len(low_files) != len(high_files):
        raise ValueError(
            f"training day mismatch: {len(low_files)} low vs {len(high_files)} high")

    train_total = 0
    for day_index, (low, high) in enumerate(zip(low_files, high_files)):
        low_name = os.path.basename(low)
        high_name = os.path.basename(high)
        tag = f"day{day_index + 1:02d}"
        nl = _select(low, "low", args.train_per_tier_day,
                     8000 + day_index,
                     os.path.join(args.out_train, f"{tag}_elo800_999.npz"))
        nh = _select(high, "high", args.train_per_tier_day,
                     10000 + day_index,
                     os.path.join(args.out_train, f"{tag}_elo1000plus.npz"))
        train_total += nl + nh
        print(f"{tag}: {nl} low from {low_name}; {nh} high from {high_name}")

    historical = sorted(glob.glob(os.path.join(
        args.historical_high_dir, "*.npz")))
    for file_index, path in enumerate(historical):
        n = _select(path, "high", args.historical_per_shard,
                    21000 + file_index,
                    os.path.join(args.out_train,
                                 f"history{file_index + 1:02d}_elo1000plus.npz"))
        train_total += n
        print(f"history{file_index + 1:02d}: {n} high from "
              f"{os.path.basename(path)}")

    low_holdout = _single(args.low_holdout_dir)
    high_holdout = _single(args.high_holdout_dir)
    nl = _select(low_holdout, "low", args.holdout_per_tier, 80731,
                 os.path.join(args.out_holdout, "holdout_elo800_999.npz"))
    nh = _select(high_holdout, "high", args.holdout_per_tier, 100731,
                 os.path.join(args.out_holdout, "holdout_elo1000plus.npz"))
    print(f"train total: {train_total}; holdout: {nl} low + {nh} high")


if __name__ == "__main__":
    main()
