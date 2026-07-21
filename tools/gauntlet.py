"""Field gauntlet: play a deck against the mined meta, weighted by real play rates.

Head-to-head A/B answers "does deck A beat deck B", which is NOT the ladder
question. The ladder question is "which deck scores better against the FIELD".
A deck can be even in the mirror and much better against everything else. This
runs our pilot on a candidate deck against each mined opponent decklist, with
game counts proportional to how often that opponent actually appears.

Usage:
  py tools/gauntlet.py <agent_dir> [games] [--opponent-agent <dir>]

Both sides use the same pilot unless --opponent-agent is given, so the result
isolates DECK strength rather than piloting.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path, alias):
    d = os.path.abspath(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(alias, os.path.join(d, "main.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("games", nargs="?", type=int, default=32)
    ap.add_argument("--opponent-agent", default=None)
    a = ap.parse_args()

    me = load(a.agent, "gauntlet_me")
    opp_mod = load(a.opponent_agent, "gauntlet_opp") if a.opponent_agent else me
    my_deck = me._load_deck()

    # weights = appearance counts from the mining run
    decks = dict(me.META_DECKS)
    weights = dict(getattr(me, "META_WEIGHT", {}) or {})
    total_w = sum(weights.get(k, 1) for k in decks) or 1

    from kaggle_environments import make

    plan = []
    for name, deck in decks.items():
        n = max(2, round(a.games * weights.get(name, 1) / total_w))
        plan.append((name, deck, n))
    print(f"gauntlet: {sum(n for _, _, n in plan)} games over {len(plan)} field decks\n")

    results = {}
    tot_w = tot_n = 0
    for name, deck, n in plan:
        w = 0
        for g in range(n):
            env = make("cabt")

            def my_agent(obs, _m=me, _d=my_deck):
                if obs.get("select") is None:
                    return list(_d)
                return _m.agent(obs)

            def opp_agent(obs, _m=opp_mod, _d=deck):
                if obs.get("select") is None:
                    return list(_d)
                return _m.agent(obs)

            if g % 2 == 0:
                env.run([my_agent, opp_agent]); mi = 0
            else:
                env.run([opp_agent, my_agent]); mi = 1
            if env.state[mi].reward == 1:
                w += 1
        results[name] = (w, n)
        tot_w += w
        tot_n += n
        print(f"  vs {name:>12}: {w}/{n} = {100*w/n:3.0f}%", flush=True)

    wr = tot_w / tot_n if tot_n else 0
    se = math.sqrt(wr * (1 - wr) / tot_n) if tot_n else 0
    print(f"\nFIELD WIN RATE: {tot_w}/{tot_n} = {100*wr:.1f}% +/- {196*se:.0f}% (95% CI)")


if __name__ == "__main__":
    main()
