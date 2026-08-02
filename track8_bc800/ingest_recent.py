"""Prepare bounded Elo-800+ shards for July 24-31.

This is intentionally a separate explicit command: importing this module does
not touch the multi-gigabyte replay archives. July 31 is always routed to the
holdout directory.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INGEST = os.path.join(
    ROOT, "track1_search", "train", "ingest_episodes.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-dir", default=os.path.join(
        ROOT, "data", "replays_daily"))
    parser.add_argument("--leaderboard", default=os.path.join(
        ROOT, "data", "leaderboard", "pokemon-tcg-ai-battle.zip"))
    parser.add_argument("--max-samples", type=int, default=400000)
    args = parser.parse_args()

    if args.max_samples <= 0:
        parser.error("--max-samples must be positive for bounded WSL ingestion")
    if not os.path.isfile(args.leaderboard):
        parser.error(f"missing leaderboard: {args.leaderboard}")

    for day in range(24, 32):
        archive = os.path.join(
            args.daily_dir,
            f"pokemon-tcg-ai-battle-episodes-2026-07-{day:02d}.zip")
        if not os.path.isfile(archive):
            parser.error(f"missing daily archive: {archive}")
        out_dir = os.path.join(
            HERE, "data_elo800_holdout" if day == 31 else "data_elo800")
        command = [
            sys.executable, INGEST, archive,
            "--out", out_dir,
            "--leaderboard", args.leaderboard,
            "--min-elo", "800",
            "--features", "rich",
            "--max-samples", str(args.max_samples),
            "--sample-mode", "reservoir",
            "--tag", f"elo800_07{day:02d}",
        ]
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
