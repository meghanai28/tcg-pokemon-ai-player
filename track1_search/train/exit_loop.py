"""Expert Iteration loop: generations of self play, each retrained on
everything so far PLUS the pro replay corpus.

The pro data (train/data_bc, behavior cloned from 1000+ Elo ladder games) is
included in every generation's training set on purpose. It anchors the policy
to demonstrated strong play while the accumulating self play shards teach the
value head the positions this agent actually reaches. Nothing is dropped.

Each generation:
  1. self play vs the mined meta decks with the CURRENT model as root priors
     (shards accumulate in train/data_exit/, kept across generations)
  2. train on data_bc + data_exit combined
  3. export gen<N>.npz and deploy it as submission/model.npz for the next round

Usage: py train/exit_loop.py --generations 2 --games 16 --workers 2 --budget 0.10
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()

    exit_dir = os.path.join(HERE, "data_exit")
    combo_dir = os.path.join(HERE, "data_combined")
    os.makedirs(exit_dir, exist_ok=True)

    for gen in range(1, a.generations + 1):
        print(f"\n===== GENERATION {gen}/{a.generations} =====", flush=True)
        # 1. self play with current submission/model.npz as priors
        run([sys.executable, os.path.join(HERE, "selfplay.py"),
             "--games", str(a.games), "--workers", str(a.workers),
             "--budget", str(a.budget), "--seed", str(100 * gen),
             "--out", exit_dir])

        # 2. train on pro corpus + ALL accumulated self play
        if os.path.isdir(combo_dir):
            shutil.rmtree(combo_dir)
        os.makedirs(combo_dir)
        for src in (os.path.join(HERE, "data_bc"), exit_dir):
            for f in os.listdir(src):
                if f.endswith(".npz"):
                    shutil.copy(os.path.join(src, f), os.path.join(combo_dir, f))
        out = os.path.join(HERE, f"model_gen{gen}.npz")
        run([sys.executable, os.path.join(HERE, "train_bc.py"),
             "--data", combo_dir, "--epochs", str(a.epochs),
             "--batch", "256", "--out", out])

        # 3. deploy for the next generation's self play
        shutil.copy(out, os.path.join(ROOT, "track1_search", "agent", "model.npz"))
        print(f"deployed {out} -> submission/model.npz", flush=True)

    shutil.rmtree(combo_dir, ignore_errors=True)
    print("\nExIt loop complete. A/B the result before shipping:")
    print("  py tools/ab_test.py submission agents/v1_frozen 12")


if __name__ == "__main__":
    main()
