"""Package the two search submissions.

``mcts``   - the earlier determinized IS-MCTS agent, recovered from quarantine
             unchanged.  It is the only artifact in this repo with a measured
             live result (972.0, Kaggle ref 55185089), so it ships exactly as
             it scored: its own prior model, its own deck, its own encoders.
             Only the licensed engine binaries are supplied, because those are
             gitignored and never committed.
``search`` - the new determinized PUCT written against the same native API,
             carrying the current deck choice and the Elo-1000 field as its
             opponent model.

The two are deliberately not merged.  The old agent's prior model is paired
with its own deck and encoder; swapping either without retraining would break
that pairing and destroy the one result we can actually point to.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
FOUNDATION = os.path.join(ROOT, "foundation")
QUARANTINE = os.path.join(ROOT, ".reset-quarantine-20260803", "track1_search", "agent")
ENGINE_BINARIES = ("libcg.so", "sim.py", "engine.py", "game.py", "__init__.py")


def write_deck(path: str, deck: list[int]) -> None:
    if len(deck) != 60:
        raise ValueError(f"deck must be exactly 60 cards, got {len(deck)}")
    with open(path, "w", encoding="utf-8") as target:
        target.write("\n".join(str(card) for card in deck) + "\n")


def copy_engine(staging: str) -> None:
    """Supply cg/ from foundation; the licensed binaries are never in git."""
    destination = os.path.join(staging, "cg")
    os.makedirs(destination, exist_ok=True)
    for name in ENGINE_BINARIES:
        source = os.path.join(FOUNDATION, "cg", name)
        if not os.path.isfile(source):
            raise SystemExit(
                f"missing engine file {source}; obtain the competition binaries "
                f"from the Kaggle page or kaggle_environments/envs/cabt/cg/"
            )
        shutil.copy2(source, os.path.join(destination, name))


def archive(staging: str, out: str, required: set[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with tarfile.open(out, "w:gz") as target:
        for name in sorted(os.listdir(staging)):
            target.add(os.path.join(staging, name), arcname=name)
    with tarfile.open(out, "r:gz") as target:
        names = set(target.getnames())
    missing = required - names
    if missing:
        raise RuntimeError(f"{out} is missing {sorted(missing)}")


def build_search(pool_path: str, deck_group: str, deck_index: int, out: str,
                 model: str | None = None) -> str:
    with open(pool_path, encoding="utf-8") as source:
        pool = json.load(source)
    deck_entry = pool[deck_group][deck_index]
    with tempfile.TemporaryDirectory(prefix="search-pkg-") as staging:
        shutil.copy2(os.path.join(HERE, "agent_search_main.py"),
                     os.path.join(staging, "main.py"))
        shutil.copy2(os.path.join(HERE, "search.py"), os.path.join(staging, "search.py"))
        write_deck(os.path.join(staging, "deck.csv"), deck_entry["cards"])
        # The opponent model needs the field it will actually meet.
        with open(os.path.join(staging, "meta_decks.json"), "w", encoding="utf-8") as target:
            json.dump([
                {"name": d["name"], "cards": d["cards"],
                 "appearances": d.get("appearances", 1)}
                for d in pool["field_decks"]
            ], target)
        required = {"main.py", "search.py", "deck.csv", "meta_decks.json", "cg/libcg.so"}
        if model:
            shutil.copy2(model, os.path.join(staging, "model.npz"))
            for name in ("nn_features.py", "nn_features_rich.py", "nn_infer.py"):
                shutil.copy2(os.path.join(FOUNDATION, name), os.path.join(staging, name))
            required |= {"model.npz", "nn_features.py", "nn_features_rich.py", "nn_infer.py"}
        copy_engine(staging)
        archive(staging, out, required)
    return deck_entry["name"]


def build_mcts(out: str) -> str:
    if not os.path.isdir(QUARANTINE):
        raise SystemExit(f"quarantined agent not found at {QUARANTINE}")
    with tempfile.TemporaryDirectory(prefix="mcts-pkg-") as staging:
        for name in ("main.py", "model.npz", "deck.csv", "nn_features.py", "nn_infer.py"):
            source = os.path.join(QUARANTINE, name)
            if not os.path.isfile(source):
                raise SystemExit(f"quarantined agent is missing {name}")
            shutil.copy2(source, os.path.join(staging, name))
        copy_engine(staging)
        with open(os.path.join(staging, "deck.csv"), encoding="utf-8") as source:
            cards = [line for line in source if line.strip()]
        if len(cards) != 60:
            raise SystemExit(f"quarantined deck.csv has {len(cards)} cards")
        archive(staging, out, {"main.py", "model.npz", "deck.csv",
                               "nn_features.py", "nn_infer.py", "cg/libcg.so"})
    return "track1 (unchanged, Kaggle ref 55185089 = 972.0)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=os.path.join(ROOT, "data", "fresh", "deck_pool_elo1000.json"))
    parser.add_argument("--deck-group", default="field_decks",
                        choices=("learner_decks", "field_decks"))
    parser.add_argument("--deck-index", type=int, default=1)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts"))
    parser.add_argument("--only", choices=("search", "mcts"), default=None)
    parser.add_argument("--guide-model", default=None,
                        help="checkpoint supplying PPO root priors to the search")
    parser.add_argument("--search-out", default=None)
    args = parser.parse_args()

    built = []
    if args.only in (None, "mcts"):
        out = os.path.join(args.out_dir, "submission_mcts_track1.tar.gz")
        label = build_mcts(out)
        built.append((out, label))
    if args.only in (None, "search"):
        out = args.search_out or os.path.join(args.out_dir, "submission_search_puct.tar.gz")
        label = build_search(args.pool, args.deck_group, args.deck_index, out,
                             model=args.guide_model)
        built.append((out, label))

    for path, label in built:
        print(f"wrote {path} ({os.path.getsize(path) / 2**20:.1f} MiB), deck={label}")


if __name__ == "__main__":
    main()
