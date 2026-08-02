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


def seat(module, decklist):
    """Bind one agent module to the deck it pilots for a single seat.

    The returned callable takes exactly ONE argument on purpose.
    kaggle_environments calls agents as ``agent(*[observation, configuration]
    [:co_argcount])``, so a helper that carried its state in default arguments
    (``def play(obs, _m=module, _d=deck)``) had ``configuration`` handed to it
    as ``_m``.  Every real decision then raised on ``configuration.agent`` and
    the seat forfeited, which silently made this gauntlet report games decided
    purely by seat order.  Capture state by closure instead.

    The deck-selection call (``select is None``) is also the agent's only
    untimed call: it loads the engine, card DB and net and resets the per-game
    tracker, so it must reach the agent rather than being answered here.
    """
    def play(obs):
        if obs.get("select") is None:
            module.agent(obs)                  # warm up + reset the tracker
            module.MY_DECK = list(decklist)    # pilot the deck we hand back
            return list(decklist)
        return module.agent(obs)
    return play


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("games", nargs="?", type=int, default=32)
    ap.add_argument("--opponent-agent", default=None)
    ap.add_argument("--same-deck", action="store_true",
                    help="head-to-head pilot A/B on the candidate's deck "
                         "instead of a weighted field gauntlet")
    ap.add_argument("--opponent-meta", default=None,
                    help="optional Python file defining current META_DECKS and "
                         "META_WEIGHT for field evaluation")
    a = ap.parse_args()

    me = load(a.agent, "gauntlet_me")
    # Always a SEPARATE module instance for the opponent seat: the two seats
    # keep their own MY_DECK and per-game tracker, so one cannot clobber the
    # other's belief about which deck it is piloting.
    opp_mod = load(a.opponent_agent or a.agent, "gauntlet_opp")
    my_deck = me._load_deck()

    # weights = appearance counts from the mining run
    meta = me
    if a.opponent_meta:
        spec = importlib.util.spec_from_file_location(
            "gauntlet_current_meta", os.path.abspath(a.opponent_meta))
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import {a.opponent_meta}")
        meta = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(meta)
    decks = dict(meta.META_DECKS)
    weights = dict(getattr(meta, "META_WEIGHT", {}) or {})
    total_w = sum(weights.get(k, 1) for k in decks) or 1

    from kaggle_environments import make

    if a.same_deck:
        plan = [("same_deck", my_deck, a.games)]
        print(f"same-deck pilot A/B: {a.games} games, alternating seats\n")
    else:
        plan = []
        for name, deck in decks.items():
            n = max(2, round(a.games * weights.get(name, 1) / total_w))
            plan.append((name, deck, n))
        print(f"gauntlet: {sum(n for _, _, n in plan)} games over "
              f"{len(plan)} field decks\n")

    results = {}
    tot_w = tot_n = 0
    for name, deck, n in plan:
        w = 0
        for g in range(n):
            env = make("cabt")

            my_agent = seat(me, my_deck)
            opp_agent = seat(opp_mod, deck)

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
