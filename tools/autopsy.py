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


def archetype(deck, names):
    """Name a deck by its most distinctive high-id pokemon."""
    ids = [c for c in deck if c > 100]
    if not ids:
        return "unknown"
    common = Counter(ids).most_common(6)
    for cid, _n in common:
        nm = names.get(cid, "")
        if nm and not nm.startswith("Basic"):
            return nm
    return str(common[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_id")
    ap.add_argument("--out", default="ep_own")
    ap.add_argument("--deck", default=os.path.join(ROOT, "track1_search", "agent", "deck.csv"))
    a = ap.parse_args()

    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("set KAGGLE_API_TOKEN")
    fetch(a.submission_id, a.out)

    with open(a.deck) as f:
        my_deck = sorted(int(x) for x in f if x.strip())

    try:
        from cg.engine import get_lib
        names = {c["cardId"]: c["name"]
                 for c in json.loads(get_lib().AllCard().decode())}
    except Exception:
        names = {}

    by_opp = defaultdict(lambda: [0, 0])     # archetype -> [wins, games]
    lengths = {"win": [], "loss": []}
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
        arch = archetype(opp_deck, names)
        b = by_opp[arch]
        b[1] += 1
        b[0] += int(won)
        total[1] += 1
        total[0] += int(won)
        (lengths["win"] if won else lengths["loss"]).append(len(steps))

    if not total[1]:
        raise SystemExit("no episodes matched our decklist")
    print(f"\nOUR RECORD: {total[0]}W {total[1]-total[0]}L of {total[1]} "
          f"= {100*total[0]/total[1]:.0f}%\n")
    print(f"{'opponent':>28} {'W':>4} {'games':>6} {'WR':>6}")
    for arch, (w, n) in sorted(by_opp.items(), key=lambda kv: -kv[1][1]):
        print(f"{arch[:28]:>28} {w:>4} {n:>6} {100*w/n:>5.0f}%")
    for k in ("win", "loss"):
        if lengths[k]:
            avg = sum(lengths[k]) / len(lengths[k])
            print(f"\navg steps in {k}s: {avg:.0f}")


if __name__ == "__main__":
    main()
