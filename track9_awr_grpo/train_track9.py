"""Train exact-Tech bounded return-weighted BC followed by targeted GRPO."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN_BC = os.path.join(ROOT, "track1_search", "train", "train_bc.py")
TRAIN_GRPO = os.path.join(ROOT, "track5_grpo", "train_grpo.py")
TECH = os.path.join(ROOT, "track6_controlled")
BC_OUT = os.path.join(HERE, "model_tech_awr_bc.npz")
GRPO_OUT = os.path.join(HERE, "model_tech_awr_grpo.npz")


def run(arguments: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    print("running:", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bc", action="store_true")
    parser.add_argument("--skip-grpo", action="store_true")
    args = parser.parse_args()

    if not args.skip_bc:
        run([
            sys.executable, TRAIN_BC,
            "--data", os.path.join(TECH, "data_tech_grim_train"),
            "--val-data", os.path.join(TECH, "data_tech_grim_holdout"),
            "--features", "rich", "--init", os.path.join(
                TECH, "model_tech_grim_bc.npz"),
            "--dim", "160", "--layers", "5", "--heads", "5",
            "--epochs", "10", "--patience", "3", "--batch", "256",
            "--lr", "0.00003", "--dropout", "0.02",
            "--value-weight", "0", "--critical-weight", "1.5",
            "--label-smoothing", "0.005", "--winner-weight", "1.20",
            "--gpu-resident", "no", "--device", "cuda", "--out", BC_OUT,
        ])

    if not args.skip_grpo:
        if not os.path.isfile(BC_OUT):
            parser.error(f"missing weighted-BC checkpoint: {BC_OUT}")
        run([
            sys.executable, TRAIN_GRPO,
            "--init", BC_OUT,
            "--deck", os.path.join(TECH, "decks", "tech_grim.csv"),
            "--opponent-meta", os.path.join(HERE, "opponent_meta.py"),
            "--no-default-opponents", "--out", GRPO_OUT,
            "--iters", "6", "--groups", "12", "--group-size", "4",
            "--max-decisions", "15000", "--epochs", "1", "--batch", "64",
            "--lr", "0.000001", "--clip", "0.08", "--kl-beta", "0.12",
            "--entropy-beta", "0.001", "--temperature", "1.03",
            "--meta-weight-power", "0.5", "--meta-schedule-slots", "24",
            "--threads", "6", "--max-wall-minutes", "240", "--device", "cuda",
            "--seed", "260802",
        ])


if __name__ == "__main__":
    main()
