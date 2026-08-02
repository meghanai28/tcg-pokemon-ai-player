"""Build the two controlled deck/model arms as Kaggle-ready archives."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from track5_grpo.build_submissions import build  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=ROOT)
    parser.add_argument("--tech-grim-model", default=os.path.join(
        HERE, "model_tech_grim_final.npz"))
    parser.add_argument("--ogerpon-model", default=os.path.join(
        HERE, "model_ogerpon_final.npz"))
    args = parser.parse_args()

    build("controlled_tech_grim", args.tech_grim_model,
          os.path.join(HERE, "decks", "tech_grim.csv"), args.out_dir)
    build("controlled_ogerpon", args.ogerpon_model,
          os.path.join(HERE, "decks", "ogerpon.csv"), args.out_dir)


if __name__ == "__main__":
    main()
