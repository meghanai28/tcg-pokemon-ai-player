"""Pure-policy (no search) tools for the scaled-BC experiment.

Deploys a behaviour-cloned policy the way the top scorer does: one network
forward per decision, argmax over legal options, no tree search. Two jobs:

  ab       pure-BC (our deck) vs the determinized SEARCH agent, same deck
           -> does a scaled pure policy beat our ~820 search agent as a pilot?
  gauntlet pure-BC piloting a candidate deck vs the mined field, weighted by
           real appearance -> which DECK scores better with the SAME checkpoint?

Usage:
  py track2_dmc/purebc_tools.py ab       --model track1_search/train/model_bc_big.npz --games 40
  py track2_dmc/purebc_tools.py gauntlet --model .../model_bc_big.npz --deck track1_search/agent/deck.csv --games 300
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGENT = os.path.join(ROOT, "track1_search", "agent")
sys.path.insert(0, AGENT)

import nn_features as NF            # noqa: E402
from nn_infer import NumpyNet      # noqa: E402


def load_agent(path, alias):
    d = os.path.abspath(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(alias, os.path.join(d, "main.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


def read_deck(path):
    with open(path) as f:
        return [int(line) for line in f if line.strip()]


def purebc_action(net, M, obs, me, rng):
    """One forward pass, argmax over legal options, greedy fill to maxCount."""
    sel = obs["select"]
    opts = sel.get("option") or []
    if not opts:
        return []
    if len(opts) == 1:
        return [0]
    scores = [M._option_score(o, sel) for o in opts]
    kind, card, scal, mask, slot = NF.encode(
        {"current": obs["current"], "select": sel}, me, M.CARD, M.ATTACK, scores)
    pol, _v = net.forward(
        kind[None], card[None], scal[None], mask[None],
        np.array([int(sel.get("context") or 0)]),
        np.array([int(sel.get("type") or 0)]))
    valid = [(i, slot[i]) for i in range(len(opts)) if slot[i] >= 0]
    if not valid:
        return M._validate(M._heuristic_action(sel, rng), sel) or [0]
    order = sorted(range(len(valid)), key=lambda k: -float(pol[0, valid[k][1]]))
    kmax = max(1, min(sel.get("maxCount", 1), len(opts)))
    action = [valid[k][0] for k in order[:kmax]]
    return M._validate(action, sel) or (
        M._validate(M._heuristic_action(sel, rng), sel) or [0])


def make_purebc_seat(net, M, deck, rng):
    def seat(obs, config=None):
        if obs.get("select") is None:
            return list(deck)
        me = (obs.get("current") or {}).get("yourIndex", 0)
        return purebc_action(net, M, obs, me, rng)
    return seat


def run_ab(a):
    from kaggle_environments import make
    M = load_agent(AGENT, "search_main")
    net = NumpyNet(a.model)
    deck = read_deck(a.deck) if a.deck else list(M._load_deck())
    rng = random.Random(1)

    def search_seat(obs, config=None):
        if obs.get("select") is None:
            return list(deck)
        return M.agent(obs)

    A = make_purebc_seat(net, M, deck, rng)
    w = l = d = 0
    for g in range(a.games):
        env = make("cabt")
        if g % 2 == 0:
            env.run([A, search_seat]); mi = 0
        else:
            env.run([search_seat, A]); mi = 1
        r = env.state[mi].reward
        if r == 1: w += 1
        elif r == -1: l += 1
        else: d += 1
        print(f"game {g}: pureBC seat {mi} -> {r}  ({w}W {l}L {d}D)", flush=True)
    n = w + l
    wr = w / n if n else 0
    se = math.sqrt(wr * (1 - wr) / n) if n else 0
    print(f"\npure-BC ({os.path.basename(a.model)}) vs SEARCH agent, same deck")
    print(f"pure-BC: {w}W {l}L {d}D = {100*wr:.1f}% +/- {196*se:.0f}% (95% CI)")


def run_gauntlet(a):
    from kaggle_environments import make
    M = load_agent(AGENT, "search_main")
    net = NumpyNet(a.model)
    my_deck = read_deck(a.deck)
    rng = random.Random(2)
    decks = dict(M.META_DECKS)
    weights = dict(getattr(M, "META_WEIGHT", {}) or {})
    total_w = sum(weights.get(k, 1) for k in decks) or 1
    plan = [(name, deck, max(2, round(a.games * weights.get(name, 1) / total_w)))
            for name, deck in decks.items()]
    print(f"gauntlet: {os.path.basename(a.deck)} piloted by pure-BC over "
          f"{sum(n for *_, n in plan)} games\n", flush=True)
    tot_w = tot_n = 0
    for name, opp_deck, n in plan:
        w = 0
        mine = make_purebc_seat(net, M, my_deck, rng)
        opp = make_purebc_seat(net, M, opp_deck, rng)
        for g in range(n):
            env = make("cabt")
            if g % 2 == 0:
                env.run([mine, opp]); mi = 0
            else:
                env.run([opp, mine]); mi = 1
            if env.state[mi].reward == 1:
                w += 1
        tot_w += w; tot_n += n
        print(f"  vs {name:>10}: {w}/{n} = {100*w/n:3.0f}%", flush=True)
    wr = tot_w / tot_n if tot_n else 0
    se = math.sqrt(wr * (1 - wr) / tot_n) if tot_n else 0
    print(f"\n{os.path.basename(a.deck)} FIELD WIN RATE: {tot_w}/{tot_n} = "
          f"{100*wr:.1f}% +/- {196*se:.0f}% (95% CI)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ab = sub.add_parser("ab")
    p_ab.add_argument("--model", required=True)
    p_ab.add_argument("--deck", default=None)
    p_ab.add_argument("--games", type=int, default=40)
    p_g = sub.add_parser("gauntlet")
    p_g.add_argument("--model", required=True)
    p_g.add_argument("--deck", required=True)
    p_g.add_argument("--games", type=int, default=300)
    a = ap.parse_args()
    if a.cmd == "ab":
        run_ab(a)
    else:
        run_gauntlet(a)


if __name__ == "__main__":
    main()
