"""Build the two larger-data, 192d supervised submission arms."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from track5_grpo.build_submissions import build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=ROOT)
    parser.add_argument("--tech-model", default=os.path.join(
        HERE, "model_tech_grim_192.npz"))
    parser.add_argument("--lopunny-model", default=os.path.join(
        HERE, "model_lopunny_192.npz"))
    args = parser.parse_args()

    build("bc800_tech_grim_192", args.tech_model,
          os.path.join(ROOT, "track6_controlled", "decks", "tech_grim.csv"),
          args.out_dir, archive_prefix="submission_")
    build("bc800_lopunny_192", args.lopunny_model,
          os.path.join(HERE, "decks", "lopunny.csv"),
          args.out_dir, archive_prefix="submission_")


if __name__ == "__main__":
    main()
