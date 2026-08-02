"""Autopsy our OWN ladder games: who beat us, how, and how fast.

Downloads every episode for a submission, identifies which seat was ours (by
matching deck.csv), and reports win rate by opponent archetype, game length,
and prize differential. Answers "what is actually killing us on the ladder",
which local A/Bs against our own agent cannot see.

Usage:
  py tools/autopsy.py <submission_id> --out ep_own [--deck submission/deck.csv]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "track1_search", "agent"))


def sh(args):
    return subprocess.run([sys.executable, "-m", "kaggle"] + args,
                          capture_output=True, text=True).stdout


def fetch(sub_id, out):
    os.makedirs(out, exist_ok=True)
    have = {os.path.basename(p) for p in glob.glob(os.path.join(out, "*.json"))}
    ids = []
    for line in sh(["competitions", "episodes", str(sub_id)]).splitlines():
        p = line.split()
        if p and p[0].isdigit():
            ids.append(int(p[0]))
    got = 0
    for eid in ids:
        fn = f"episode-{eid}-replay.json"
        if fn in have:
            continue
        sh(["competitions", "replay", str(eid), "-p", out])
        if os.path.exists(os.path.join(out, fn)):
            got += 1
    print(f"{len(ids)} episodes listed, {got} newly downloaded")
    return ids


def archetype(deck, cards):
    """Name a deck by its most prominent Pokemon, not a trainer card.

    The old ``cardId > 100`` shortcut could label Teal Mask Ogerpon as
    ``Bug Catching Set`` and other decks as supporters.  Prefer Pokemon and
    score repeated evolved/ex cards above one-off support Pokemon.
    """
    counts = Counter(deck)
    pokemon = []
    for card_id, count in counts.items():
        card = cards.get(card_id)
        if not card or card.get("cardType") != 0:
            continue
        importance = 1
        if card.get("stage1"):
            importance += 1
        if card.get("stage2"):
            importance += 2
        if card.get("ex"):
            importance += 2
        if card.get("megaEx"):
            importance += 3
        pokemon.append((count * importance, count, card.get("name", str(card_id))))
    if not pokemon:
        return "unknown"
    return max(pokemon)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_id")
    ap.add_argument("--out", default="ep_own")
    ap.add_argument("--deck", default=os.path.join(ROOT, "track1_search", "agent", "deck.csv"))
    ap.add_argument("--no-fetch", action="store_true",
                    help="analyze replay JSON already in --out without Kaggle access")
    a = ap.parse_args()

    if not a.no_fetch:
        if not os.environ.get("KAGGLE_API_TOKEN"):
            raise SystemExit("set KAGGLE_API_TOKEN or pass --no-fetch")
        fetch(a.submission_id, a.out)

    with open(a.deck) as f:
        my_deck = sorted(int(x) for x in f if x.strip())

    try:
        from cg.engine import get_lib
        cards = {c["cardId"]: c
                 for c in json.loads(get_lib().AllCard().decode())}
    except Exception:
        cards = {}

    by_opp = defaultdict(lambda: [0, 0])     # archetype -> [wins, games]
    by_user = defaultdict(lambda: [0, 0])
    by_seat = defaultdict(lambda: [0, 0])
    lengths = {"win": [], "loss": []}
    time_left = {"win": [], "loss": []}
    total = [0, 0]
    for f in glob.glob(os.path.join(a.out, "*.json")):
        try:
            ep = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        steps = ep.get("steps") or []
        if len(steps) < 2:
            continue
        decks = [steps[1][p].get("action") for p in range(2)]
        me = None
        for p in range(2):
            if isinstance(decks[p], list) and sorted(decks[p]) == my_deck:
                me = p
                break
        if me is None:
            continue
        rewards = ep.get("rewards") or [0, 0]
        won = rewards[me] == 1
        opp_deck = decks[1 - me] or []
        arch = archetype(opp_deck, cards)
        b = by_opp[arch]
        b[1] += 1
        b[0] += int(won)
        seat_bucket = by_seat[me]
        seat_bucket[1] += 1
        seat_bucket[0] += int(won)
        agents = (ep.get("info") or {}).get("Agents") or []
        if len(agents) == 2:
            opp_name = agents[1 - me].get("Name") or "unknown"
            user_bucket = by_user[opp_name]
            user_bucket[1] += 1
            user_bucket[0] += int(won)
        total[1] += 1
        total[0] += int(won)
        outcome = "win" if won else "loss"
        lengths[outcome].append(len(steps))
        try:
            left = steps[-1][me]["observation"]["remainingOverageTime"]
            time_left[outcome].append(float(left))
        except (KeyError, IndexError, TypeError, ValueError):
            pass

    if not total[1]:
        raise SystemExit("no episodes matched our decklist")
    print(f"\nOUR RECORD: {total[0]}W {total[1]-total[0]}L of {total[1]} "
          f"= {100*total[0]/total[1]:.0f}%\n")
    print(f"{'opponent':>28} {'W':>4} {'games':>6} {'WR':>6}")
    for arch, (w, n) in sorted(by_opp.items(), key=lambda kv: -kv[1][1]):
        print(f"{arch[:28]:>28} {w:>4} {n:>6} {100*w/n:>5.0f}%")
    print("\nseat record:")
    for seat_index, (w, n) in sorted(by_seat.items()):
        print(f"  seat {seat_index}: {w}W {n-w}L = {100*w/n:.0f}%")
    repeated = [(name, record) for name, record in by_user.items()
                if record[1] >= 2]
    if repeated:
        print("\nrepeat opponents:")
        for name, (w, n) in sorted(repeated, key=lambda item: -item[1][1]):
            print(f"  {name[:28]:>28}: {w}W {n-w}L")
    for k in ("win", "loss"):
        if lengths[k]:
            avg = sum(lengths[k]) / len(lengths[k])
            suffix = ""
            if time_left[k]:
                suffix = f", avg bank left {sum(time_left[k])/len(time_left[k]):.0f}s"
            print(f"\navg steps in {k}s: {avg:.0f}{suffix}")


if __name__ == "__main__":
    main()
