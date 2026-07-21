"""Self-play data generator for Expert Iteration.

Each worker process plays full games with the search agent driving BOTH seats,
recording at every decision:
    features (kind, card, scal, mask, ctx, stype)  -- the state
    pi       [SEQ]                                 -- search root visit distribution
    z        scalar                                -- final game outcome for the
                                                      player who was to move
Shards are written as .npz to train/data/.

Usage:
  py train/selfplay.py --games 20 --workers 3 --budget 0.25 --out train/data
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from multiprocessing import Process

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SUB = os.path.join(ROOT, "track1_search", "agent")
sys.path.insert(0, SUB)

import nn_features as NF  # noqa: E402


def _load_agent_module():
    spec = importlib.util.spec_from_file_location("sp_main", os.path.join(SUB, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker(wid, games, budget, out_dir, seed):
    mod = _load_agent_module()
    mod.MY_DECK = mod._load_deck()
    try:
        mod._load_engine()
        mod._load_card_db()
    except Exception as exc:
        print(f"[w{wid}] engine load failed: {exc!r}", flush=True)
        return
    mod._ENGINE_TRIED = True

    from kaggle_environments import make

    rec = {k: [] for k in ("kind", "card", "scal", "mask", "ctx", "stype", "pi", "player")}
    outcomes = []      # (start_index, player) per decision handled below
    game_bounds = []

    import random as _random

    def make_recording_agent():
        """Agent that plays via search AND records the search policy."""
        def _agent(obs):
            if obs.get("select") is None:
                return list(mod.MY_DECK)
            sel = obs["select"]
            opts = sel.get("option") or []
            n = len(opts)
            if n == 0:
                return []
            cur = obs.get("current") or {}
            me = cur.get("yourIndex", 0)
            if n == 1:
                return [0]

            policy = {}
            action = None
            try:
                action = mod._search_move(obs, me, mod._GAME.opp_model,
                                          time.perf_counter() + budget, mod._GAME.rng,
                                          collect_policy=policy)
            except Exception:
                action = None
            if action is None or not policy:
                return mod._validate(mod._heuristic_action(sel, mod._GAME.rng), sel) or [0]

            # record: features + visit distribution projected onto option tokens
            try:
                scores = [mod._option_score(o, sel) for o in opts]
                kind, card, scal, mask, opt_slot = NF.encode(
                    {"current": cur, "select": sel}, me, mod.CARD, mod.ATTACK, scores)
                pi = np.zeros(NF.SEQ, dtype=np.float32)
                tot = 0
                for act, vis in policy.items():
                    # credit each selected option index in the action tuple
                    for i in act:
                        if 0 <= i < len(opt_slot) and opt_slot[i] >= 0:
                            pi[opt_slot[i]] += vis
                            tot += vis
                if tot > 0:
                    pi /= pi.sum()
                    rec["kind"].append(kind); rec["card"].append(card)
                    rec["scal"].append(scal); rec["mask"].append(mask)
                    rec["ctx"].append(int(sel.get("context") or 0))
                    rec["stype"].append(int(sel.get("type") or 0))
                    rec["pi"].append(pi)
                    rec["player"].append(me)
            except Exception:
                pass
            return mod._validate(action, sel) or [0]
        return _agent

    agent_fn = make_recording_agent()

    def make_opponent_agent(deck_ids):
        """Same search policy, different decklist, and it records too.

        Playing the real meta rather than a mirror is the point: the value head
        must judge positions arising against the decks we actually face.
        """
        inner = make_recording_agent()

        def _opp(obs):
            if obs.get("select") is None:
                return list(deck_ids)
            return inner(obs)
        return _opp

    rng = _random.Random(seed)
    opp_names = list(mod.META_DECKS.keys())
    t_start = time.perf_counter()
    for g in range(games):
        start_idx = len(rec["pi"])
        mod._GAME = mod.GameState()
        mod._GAME.rng = _random.Random(rng.randrange(1 << 30))
        env = make("cabt")
        # rotate through the meta so the data covers the real field
        opp_name = opp_names[g % len(opp_names)]
        opp_fn = make_opponent_agent(mod.META_DECKS[opp_name])
        try:
            if g % 2 == 0:
                env.run([agent_fn, opp_fn])
            else:
                env.run([opp_fn, agent_fn])
        except Exception as exc:
            print(f"[w{wid}] game {g} crashed: {exc!r}", flush=True)
            continue
        r0 = env.state[0].reward
        end_idx = len(rec["pi"])
        game_bounds.append((start_idx, end_idx, r0))
        print(f"[w{wid}] game {g} vs {opp_name}: reward0={r0} decisions={end_idx-start_idx} "
              f"elapsed={time.perf_counter()-t_start:.0f}s", flush=True)
        # Flush every few games. A killed or crashed worker keeps everything
        # written so far instead of losing hours of play (learned the hard way).
        if (g + 1) % 8 == 0 or g == games - 1:
            _flush(wid, rec, game_bounds, out_dir)
            rec = {k: [] for k in rec}
            game_bounds = []

    if rec["pi"]:
        _flush(wid, rec, game_bounds, out_dir)


def _flush(wid, rec, game_bounds, out_dir):
    if not rec["pi"]:
        return
    # outcome label per decision, from the perspective of the mover
    z = np.zeros(len(rec["pi"]), dtype=np.float32)
    for (s, e, r0) in game_bounds:
        for i in range(s, e):
            p = rec["player"][i]
            z[i] = float(r0) if p == 0 else -float(r0)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shard_w{wid}_{int(time.time()*1000)}.npz")
    np.savez_compressed(
        path,
        kind=np.array(rec["kind"], dtype=np.int8),
        card=np.array(rec["card"], dtype=np.int16),
        scal=np.array(rec["scal"], dtype=np.float32),
        mask=np.array(rec["mask"], dtype=np.float32),
        ctx=np.array(rec["ctx"], dtype=np.int16),
        stype=np.array(rec["stype"], dtype=np.int16),
        pi=np.array(rec["pi"], dtype=np.float32),
        z=z,
    )
    print(f"[w{wid}] wrote {path} ({len(z)} samples)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    procs = []
    for w in range(a.workers):
        p = Process(target=worker, args=(w, a.games, a.budget, a.out, a.seed + 1000 * w))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("done")


if __name__ == "__main__":
    main()
