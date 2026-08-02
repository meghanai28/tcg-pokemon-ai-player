"""Train the shared 192d prior and two deck-specialized supervised arms."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(ROOT, "track1_search", "train", "train_bc.py")
ANCHOR = os.path.join(ROOT, "track6_controlled", "make_anchor_data.py")
TECH_DECK = os.path.join(ROOT, "track6_controlled", "decks", "tech_grim.csv")
LOPUNNY_DECK = os.path.join(HERE, "decks", "lopunny.csv")
GENERAL = os.path.join(HERE, "model_general_192.npz")


def run(arguments: list[str]) -> None:
    print("running:", " ".join(arguments), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(arguments, cwd=ROOT, env=env, check=True)


def common(init: str, out: str, epochs: int, patience: int,
           learning_rate: str) -> list[str]:
    return [
        "--features", "rich", "--init", init,
        "--dim", "192", "--layers", "6", "--heads", "6",
        "--epochs", str(epochs), "--patience", str(patience),
        "--batch", "256", "--lr", learning_rate,
        "--dropout", "0.03", "--value-weight", "0",
        "--critical-weight", "1.5", "--label-smoothing", "0.01",
        "--elo-weight", "0.5", "--gpu-resident", "no",
        "--device", "cuda", "--out", out,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-general", action="store_true")
    parser.add_argument("--skip-anchors", action="store_true")
    args = parser.parse_args()

    if not args.skip_general:
        run([
            sys.executable, TRAIN,
            "--data", os.path.join(HERE, "data_train"),
            "--val-data", os.path.join(HERE, "data_holdout"),
            "--features", "rich", "--split", "episode",
            "--dim", "192", "--layers", "6", "--heads", "6",
            "--epochs", "20", "--patience", "4", "--batch", "256",
            "--max-per-shard", "50000",
            "--lr", "0.0002", "--dropout", "0.03",
            "--value-weight", "0", "--critical-weight", "1.5",
            "--label-smoothing", "0.01", "--elo-weight", "0.5",
            "--gpu-resident", "no", "--device", "cuda",
            "--out", GENERAL,
        ])

    if not args.skip_anchors:
        for deck, train_source, holdout_source, train_out, holdout_out in (
            (TECH_DECK, "data_train", "data_holdout",
             "data_tech_anchor", "data_tech_anchor_holdout"),
            # Lopunny is rare enough to safely use every retained reservoir
            # example rather than the balanced general-model subset.
            (LOPUNNY_DECK, "data_elo800", "data_elo800_holdout",
             "data_lopunny_full_anchor", "data_lopunny_full_anchor_holdout"),
        ):
            run([sys.executable, ANCHOR,
                 "--data", os.path.join(HERE, train_source),
                 "--deck", deck, "--out", os.path.join(HERE, train_out)])
            run([sys.executable, ANCHOR,
                 "--data", os.path.join(HERE, holdout_source),
                 "--deck", deck, "--out", os.path.join(HERE, holdout_out)])

    run([sys.executable, TRAIN,
         "--data", os.path.join(HERE, "data_tech_anchor"),
         os.path.join(ROOT, "track6_controlled", "data_tech_grim_train"),
         os.path.join(ROOT, "track6_controlled", "data_tech_grim_train"),
         os.path.join(ROOT, "track6_controlled", "data_tech_grim_train"),
         "--val-data", os.path.join(
             ROOT, "track6_controlled", "data_tech_grim_holdout"),
         "--max-per-shard", "10000",
         *common(GENERAL, os.path.join(HERE, "model_tech_grim_192.npz"),
                 14, 3, "0.0001")])

    run([sys.executable, TRAIN,
         "--data", os.path.join(HERE, "data_lopunny_full_anchor"),
         "--val-data", os.path.join(
             HERE, "data_lopunny_full_anchor_holdout"),
         *common(GENERAL, os.path.join(HERE, "model_lopunny_192.npz"),
                 16, 4, "0.0001")])


if __name__ == "__main__":
    main()
