"""Where do our heuristic priors disagree with 1000+ Elo players?

Replays every decision by qualifying players through the agent's option scorer
and buckets agreement by SelectContext. Low-agreement, high-volume contexts are
where prior tuning buys the most, because priors steer which lines the search
explores first.

Usage: py tools/divergence.py ep/ --leaderboard lb/*.csv --min-elo 1000 [--limit 400]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "track1_search", "agent")
sys.path.insert(0, SUB)

CTX_NAMES = {0: "MAIN", 1: "SETUP_ACTIVE", 2: "SETUP_BENCH", 3: "SWITCH", 4: "TO_ACTIVE",
             5: "TO_BENCH", 7: "TO_HAND", 8: "DISCARD", 9: "TO_DECK", 13: "DMG_COUNTER",
             15: "DAMAGE", 21: "ATTACH_FROM", 22: "ATTACH_TO", 24: "LOOK", 25: "EFFECT_TARGET",
             35: "ATTACK", 37: "EVOLVE", 38: "DRAW_COUNT", 41: "IS_FIRST", 42: "MULLIGAN",
             43: "ACTIVATE", 44: "FIRST_EFFECT", 46: "COIN_HEAD"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--leaderboard", default=None)
    ap.add_argument("--min-elo", type=float, default=1000.0)
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("div_main", os.path.join(SUB, "main.py"))
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    M.MY_DECK = M._load_deck()
    M._load_engine()
    M._load_card_db()

    elos = {}
    if a.leaderboard:
        import csv
        files = sorted(glob.glob(a.leaderboard))
        with open(files[-1], newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    elos[r["TeamName"]] = float(r["Score"])
                except (KeyError, ValueError):
                    pass

    import random
    rng = random.Random(0)
    agree = defaultdict(lambda: [0, 0])          # ctx -> [agree, total]
    n_ep = 0
    for f in glob.glob(os.path.join(a.episodes, "*.json")):
        if n_ep >= a.limit:
            break
        try:
            ep = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        agents = (ep.get("info") or {}).get("Agents") or []
        names = [ag.get("Name") if isinstance(ag, dict) else None for ag in agents]
        keep = {i for i in range(2)
                if names[i] and elos.get(names[i], -1) >= a.min_elo} if elos else {0, 1}
        if not keep:
            continue
        n_ep += 1
        steps = ep["steps"]
        for t in range(1, len(steps) - 1):
            for p in keep:
                obs = (steps[t][p] or {}).get("observation") or {}
                sel = obs.get("select")
                cur = obs.get("current")
                if not sel or not cur or cur.get("yourIndex") != p:
                    continue
                opts = sel.get("option") or []
                if len(opts) < 2:
                    continue
                action = (steps[t + 1][p] or {}).get("action")
                if not isinstance(action, list) or not action:
                    continue
                ours = M._heuristic_action(sel, rng)
                if not ours:
                    continue
                ctx = sel.get("context", -1)
                a2 = agree[ctx]
                a2[1] += 1
                if ours[0] == action[0]:
                    a2[0] += 1

    rows = sorted(agree.items(), key=lambda kv: -(kv[1][1] - kv[1][0]))
    print(f"{n_ep} episodes replayed. Contexts by total DISAGREEMENTS (top pick):\n")
    print(f"{'context':>14} {'agree':>7} {'total':>7} {'rate':>6}")
    for ctx, (g, n) in rows[:18]:
        print(f"{CTX_NAMES.get(ctx, str(ctx)):>14} {g:>7} {n:>7} {100*g/n:>5.0f}%")


if __name__ == "__main__":
    main()
