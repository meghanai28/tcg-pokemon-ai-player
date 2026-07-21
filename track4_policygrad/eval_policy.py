"""Head to head between two RAW policies (no search).

This is the honest checkpoint for Track 4: does DG fine tuning improve the
policy it started from? Search is deliberately excluded, because search would
mask policy differences.

Usage:
  py track4_policygrad/eval_policy.py --a model_dg.npz --b ../track1_search/train/model_v4.npz --games 40
  py track4_policygrad/eval_policy.py --a model_dg.npz --heuristic --games 40
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "track1_search", "agent"))
sys.path.insert(0, os.path.join(ROOT, "track1_search", "train"))
sys.path.insert(0, HERE)

from model import TCGNet                              # noqa: E402
from train_dg import load_agent_module, import_weights, policy_action  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default=None)
    ap.add_argument("--heuristic", action="store_true",
                    help="opponent is the hand written scorer instead of a net")
    ap.add_argument("--games", type=int, default=40)
    args = ap.parse_args()

    M = load_agent_module()
    ma = TCGNet(); import_weights(ma, args.a); ma.eval()
    mb = None
    if not args.heuristic:
        mb = TCGNet(); import_weights(mb, args.b); mb.eval()

    rng = random.Random(1)
    from kaggle_environments import make

    def make_agent(model):
        def f(obs):
            if obs.get("select") is None:
                return list(M.MY_DECK)
            sel = obs["select"]
            opts = sel.get("option") or []
            if not opts:
                return []
            if len(opts) == 1:
                return [0]
            if model is None:
                return M._validate(M._heuristic_action(sel, rng), sel) or [0]
            me = (obs.get("current") or {}).get("yourIndex", 0)
            act, _lp, _f = policy_action(model, M, obs, me, rng, sample=False)
            return M._validate(act, sel) or (
                M._validate(M._heuristic_action(sel, rng), sel) or [0])
        return f

    A, B = make_agent(ma), make_agent(mb)
    w = l = 0
    for g in range(args.games):
        env = make("cabt")
        if g % 2 == 0:
            env.run([A, B]); mi = 0
        else:
            env.run([B, A]); mi = 1
        r = env.state[mi].reward
        w += int(r == 1); l += int(r == -1)
        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{args.games}: {w}W {l}L", flush=True)
    n = w + l
    wr = w / n if n else 0
    se = math.sqrt(wr * (1 - wr) / n) if n else 0
    opp = "heuristic policy" if args.heuristic else os.path.basename(args.b)
    print(f"\nA={os.path.basename(args.a)} vs {opp}")
    print(f"A: {w}W {l}L  = {100*wr:.1f}% +/- {196*se:.0f}% (95% CI)")
    if n and wr - 1.96 * se > 0.5:
        print("VERDICT: A is better (significant)")
    elif n and wr + 1.96 * se < 0.5:
        print("VERDICT: A is worse (significant)")
    else:
        print("VERDICT: inconclusive at this sample size")


if __name__ == "__main__":
    main()
