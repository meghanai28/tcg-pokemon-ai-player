"""Head-to-head A/B between two agent directories (the ship/no-ship gate).

Both sides load their own main.py + deck.csv + model.npz, so this compares
whatever differs between the two dirs (weights, code, deck).

Usage: py tools/ab_test.py <dirA> <dirB> [games]
Seats alternate every game. Reports A's record and a rough noise band.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import time


def load_agent(path, alias):
    d = os.path.abspath(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(alias, os.path.join(d, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    dir_a, dir_b = sys.argv[1], sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    agent_a = load_agent(dir_a, "agent_a_main")
    agent_b = load_agent(dir_b, "agent_b_main")

    from kaggle_environments import make

    wins = losses = draws = 0
    for g in range(games):
        t0 = time.perf_counter()
        env = make("cabt")
        if g % 2 == 0:
            env.run([agent_a, agent_b]); ai = 0
        else:
            env.run([agent_b, agent_a]); ai = 1
        r = env.state[ai].reward
        if r == 1:
            wins += 1
        elif r == -1:
            losses += 1
        else:
            draws += 1
        print(f"game {g}: A seat {ai} -> {r:+d}  ({wins}W {losses}L {draws}D)  "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    n = wins + losses
    wr = wins / n if n else 0.0
    se = math.sqrt(wr * (1 - wr) / n) if n else 0.0
    print(f"\nA={dir_a}  B={dir_b}")
    print(f"A record: {wins}W {losses}L {draws}D of {games}   "
          f"win rate {wr*100:.1f}% +/- {se*196:.0f}% (95% CI)")
    if n and wr - 1.96 * se > 0.5:
        print("VERDICT: A is better (significant)")
    elif n and wr + 1.96 * se < 0.5:
        print("VERDICT: B is better (significant)")
    else:
        print("VERDICT: inconclusive at this sample size -- do NOT ship on this alone")


if __name__ == "__main__":
    main()
