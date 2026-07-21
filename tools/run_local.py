"""Run local cabt episodes: our submission agent vs a baseline.

Usage: py tools/run_local.py [opponent] [games]
  opponent: self | random | first  (default random)
"""
import importlib.util
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "track1_search", "agent")


def load_submission_agent():
    spec = importlib.util.spec_from_file_location("sub_main", os.path.join(SUB, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    opponent_name = sys.argv[1] if len(sys.argv) > 1 else "random"
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    mod = load_submission_agent()
    my_agent = mod.agent
    deck = mod._load_deck()

    rng = random.Random(123)

    def random_agent(obs):
        if obs["select"] is None:
            return list(deck)
        return rng.sample(range(len(obs["select"]["option"])), obs["select"]["maxCount"])

    def first_agent(obs):
        if obs["select"] is None:
            return list(deck)
        return list(range(obs["select"]["maxCount"]))

    opponents = {"random": random_agent, "first": first_agent, "self": my_agent}
    if opponent_name.startswith("repo:"):
        # load a rule-based ladder agent from the public ptcg-abc checkout
        name = opponent_name.split(":", 1)[1]
        repo_agents = os.environ.get("PTCG_ABC_AGENTS")
        adir = os.path.join(repo_agents, name)
        sys.path.insert(0, SUB)     # provides the cg.api shim
        sys.path.insert(0, adir)    # provides their deck.csv lookup via sys.path
        spec2 = importlib.util.spec_from_file_location("opp_main", os.path.join(adir, "main.py"))
        omod = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(omod)
        opponents[opponent_name] = omod.agent
    opp = opponents[opponent_name]

    from kaggle_environments import make

    results = []
    for g in range(games):
        t0 = time.perf_counter()
        env = make("cabt")
        # alternate seats
        if g % 2 == 0:
            env.run([my_agent, opp])
            my_idx = 0
        else:
            env.run([opp, my_agent])
            my_idx = 1
        dt = time.perf_counter() - t0
        if os.environ.get("PTCG_DEBUG"):
            for i, step_logs in enumerate(env.logs[:12]):
                for j, l in enumerate(step_logs or []):
                    txt = (l or {}).get("stdout", "") + (l or {}).get("stderr", "")
                    if txt.strip():
                        print(f"--- step {i} agent {j}:\n{txt[:1200]}", flush=True)
        rew = [env.state[0].reward, env.state[1].reward]
        statuses = [env.state[0].status, env.state[1].status]
        my_r = rew[my_idx]
        results.append(my_r)
        print(f"game {g}: seat {my_idx} reward {my_r} rewards {rew} status {statuses} "
              f"steps {len(env.steps)} time {dt:.1f}s", flush=True)

    wins = sum(1 for r in results if r == 1)
    losses = sum(1 for r in results if r == -1)
    print(f"\nvs {opponent_name}: {wins}W {losses}L {len(results)-wins-losses}D of {len(results)}")


if __name__ == "__main__":
    main()
