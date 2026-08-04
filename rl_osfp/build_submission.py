"""Package the final OSFP policy and a selected fresh deck."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOUNDATION = os.path.join(ROOT, "foundation")
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(HERE, "run", "model_final.npz"))
    parser.add_argument("--pool", default=os.path.join(ROOT, "data", "fresh", "deck_pool.json"))
    parser.add_argument("--deck-index", type=int, default=0)
    parser.add_argument("--deck-group", choices=("learner_decks", "field_decks"),
                        default="learner_decks",
                        help="which pool group --deck-index refers to; the best deck "
                             "for a weak pilot is often a field deck, because the pool "
                             "builder ranks learner decks by expert ladder win rate")
    parser.add_argument("--out", default=os.path.join(ROOT, "artifacts", "submission_osfp.tar.gz"))
    args = parser.parse_args()
    with open(args.pool, encoding="utf-8") as source:
        pool = json.load(source)
    group = pool[args.deck_group]
    if not 0 <= args.deck_index < len(group):
        raise SystemExit(
            f"--deck-index {args.deck_index} is outside {args.deck_group} "
            f"(0..{len(group) - 1})"
        )
    deck = group[args.deck_index]["cards"]
    if len(deck) != 60:
        raise ValueError("selected deck is not 60 cards")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="osfp-package-") as staging:
        shutil.copy2(os.path.join(HERE, "agent_main.py"), os.path.join(staging, "main.py"))
        shutil.copy2(args.model, os.path.join(staging, "model.npz"))
        shutil.copy2(os.path.join(FOUNDATION, "nn_features.py"), staging)
        shutil.copy2(os.path.join(FOUNDATION, "nn_features_rich.py"), staging)
        shutil.copy2(os.path.join(FOUNDATION, "nn_infer.py"), staging)
        shutil.copytree(os.path.join(FOUNDATION, "cg"), os.path.join(staging, "cg"))
        with open(os.path.join(staging, "deck.csv"), "w", encoding="utf-8") as target:
            target.write("\n".join(str(card) for card in deck) + "\n")
        with tarfile.open(args.out, "w:gz") as archive:
            for name in sorted(os.listdir(staging)):
                archive.add(os.path.join(staging, name), arcname=name)
    with tarfile.open(args.out, "r:gz") as archive:
        names = set(archive.getnames())
    required = {"main.py", "model.npz", "deck.csv", "nn_features.py",
                "nn_features_rich.py", "nn_infer.py", "cg/libcg.so"}
    missing = required - names
    if missing:
        raise RuntimeError(f"package missing {sorted(missing)}")
    print(
        f"wrote {args.out} ({os.path.getsize(args.out) / 2**20:.1f} MiB), "
        f"deck={group[args.deck_index]['name']} ({args.deck_group})"
    )


if __name__ == "__main__":
    main()
