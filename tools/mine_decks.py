"""Mine actual decklists and their win rates from downloaded replays.

Every episode stores both players' 60-card decks (the step-1 actions). This
extracts them, groups exact lists, ranks by frequency and win rate, and emits
a ready-to-paste META_DECKS block for submission/main.py so belief sampling
reflects the field as it is being played right now.

Usage: py tools/mine_decks.py ep/ --top 12 [--emit meta_decks.py]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission"))


def card_names():
    try:
        from cg.engine import get_lib
        lib = get_lib()
        return {c["cardId"]: c["name"] for c in json.loads(lib.AllCard().decode())}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--emit", default=None)
    a = ap.parse_args()

    names = card_names()
    counts = Counter()          # deck tuple -> appearances
    wins = defaultdict(int)     # deck tuple -> wins
    n_ep = 0
    for f in glob.glob(os.path.join(a.episodes, "*.json")):
        try:
            ep = json.load(open(f, encoding="utf-8"))
            steps = ep["steps"]
            rewards = ep.get("rewards") or [None, None]
            for p in range(2):
                deck = steps[1][p].get("action")
                if isinstance(deck, list) and len(deck) == 60:
                    key = tuple(sorted(deck))
                    counts[key] += 1
                    if rewards[p] == 1:
                        wins[key] += 1
            n_ep += 1
        except Exception:
            continue

    print(f"{n_ep} episodes, {len(counts)} distinct exact lists\n")
    top = counts.most_common(a.top)
    lines = []
    for i, (key, n) in enumerate(top):
        wr = 100.0 * wins[key] / n
        # signature = most distinctive pokemon (highest id basic-ignored heuristic:
        # just show the 3 most common non-energy ids by name)
        sig = [names.get(c, str(c)) for c, _k in Counter(
            [c for c in key if c > 100]).most_common(3)]
        print(f"deck_{i}: {n} appearances, {wr:.0f}% win rate  |  {', '.join(sig)}")
        lines.append((f"mined_{i}", list(key), n, wr))

    if a.emit:
        with open(a.emit, "w") as f:
            f.write("# Auto-mined from ladder replays by tools/mine_decks.py\n")
            f.write("META_DECKS = {\n")
            for name, deck, n, wr in lines:
                f.write(f'    "{name}": {deck},  # seen {n}x, {wr:.0f}% WR\n')
            f.write("}\n")
            f.write("META_WEIGHT = {\n")
            for name, deck, n, wr in lines:
                f.write(f'    "{name}": {n},\n')
            f.write("}\n")
        print(f"\nwrote {a.emit}")


if __name__ == "__main__":
    main()
