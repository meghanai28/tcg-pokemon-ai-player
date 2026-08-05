"""Build a training pool that makes the policy actually pilot the deck we ship.

`train.py` draws the learner's deck from `learner_decks` and the opponent's from
`field_decks`, and never crosses them (`own_deck = weighted_deck(rng,
learner_decks)`).  So a deck that only appears in `field_decks` is a deck the
policy has faced thousands of times and never once played.

That is exactly what went wrong with the v3 checkpoint.  It piloted only
field_1, field_3, field_7 and field_10, whose multiset Jaccard against the
Tech-Grim list we ship is 0.053 to 0.132.  Tech-Grim itself is field_0/field_2
(Jaccard 0.846), which sat in `field_decks`.  Since `nn_features_rich` is
deck-aware and encodes the decklist, the net could tell it was holding something
it had never played, and its priors were correspondingly useless.

This writes a pool with a chosen deck installed as the sole learner deck, so a
warm-started run specialises on the list that will actually be in the tarball.
The field is left alone: the opponent distribution should still be the real
ladder meta.

Usage:
  py tools/make_learner_pool.py --deck-csv foundation/deck_tech_grim.csv \\
      --out data/fresh/deck_pool_techgrim.json
  py tools/make_learner_pool.py --deck-name field_16 --out data/fresh/deck_pool_f16.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jaccard(a, b) -> float:
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values()) / max(1, sum((ca | cb).values()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=os.path.join(
        ROOT, "data", "fresh", "deck_pool_train.json"))
    parser.add_argument("--deck-csv", default=None)
    parser.add_argument("--deck-name", default=None,
                        help="take the learner deck from the source pool by name")
    parser.add_argument("--label", default=None)
    parser.add_argument("--keep-existing", action="store_true",
                        help="append rather than replace the existing learner decks; "
                             "dilutes the signal, use only to avoid over-specialising")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.source, encoding="utf-8") as handle:
        pool = json.load(handle)

    if args.deck_csv:
        cards = [int(line) for line in open(args.deck_csv, encoding="utf-8") if line.strip()]
        label = args.label or os.path.splitext(os.path.basename(args.deck_csv))[0]
    elif args.deck_name:
        found = [d for g in ("field_decks", "learner_decks") for d in pool.get(g, [])
                 if d["name"] == args.deck_name]
        if not found:
            raise SystemExit(f"{args.deck_name} not in {args.source}")
        cards, label = list(found[0]["cards"]), args.label or args.deck_name
    else:
        raise SystemExit("pass --deck-csv or --deck-name")

    if len(cards) != 60:
        raise SystemExit(f"learner deck must be 60 cards, got {len(cards)}")

    entry = {"name": label, "cards": cards, "appearances": 1,
             "wins": 0, "win_rate": 0.0, "wilson_lower_80": 0.0}
    existing = pool.get("learner_decks", []) if args.keep_existing else []
    pool["learner_decks"] = existing + [entry]
    pool["learner_source"] = {
        "deck": label,
        "note": "installed by tools/make_learner_pool.py so the policy pilots the "
                "deck it will ship with, not only face it",
    }

    print(f"learner decks now: {[d['name'] for d in pool['learner_decks']]}")
    print("overlap of the new learner deck with each field deck:")
    for deck in sorted(pool.get("field_decks", []),
                       key=lambda d: -jaccard(cards, d["cards"]))[:5]:
        print(f"   {deck['name']:10s} jaccard={jaccard(cards, deck['cards']):.3f} "
              f"apps={deck.get('appearances')}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as target:
        json.dump(pool, target, indent=2)
    print(f"\nwrote {args.out}")
    print("warm start from the existing run rather than from scratch:")
    print("   cp -r rl_osfp/run_v3 rl_osfp/run_v4   # keeps training_state.pt + league")
    print(f"   .venv/bin/python -m rl_osfp.train --resume --pool {args.out} \\")
    print("       --out-dir rl_osfp/run_v4 --periods 260")


if __name__ == "__main__":
    main()
