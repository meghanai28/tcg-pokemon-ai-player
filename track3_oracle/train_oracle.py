"""Track 3: oracle guided learning (Suphx style), step 1 = the cheap checkpoint.

Suphx trains with access to hidden information, then weans the model off it.
Before building the weaning schedule, there is a much cheaper question worth
answering first:

    How much is hidden information actually WORTH in this game?

If a model that can see the opponent's hand and deck order is only slightly
better at predicting outcomes than one that cannot, then our crude uniform
determinization is not the ceiling, and the whole track should be dropped.
If the gap is large, hidden state inference deserves real effort and the
cheapest payoff is a learned hidden card predictor feeding Track 1's world
sampling.

This script generates self play data where BOTH seats are ours, so the true
hidden state of each side is recorded, then trains two value models on
identical positions:

  observed : the features Track 1 actually sees
  oracle   : same, plus the opponent's true hand and deck composition

and compares their outcome prediction error on a held out split.

Usage:
  py track3_oracle/train_oracle.py --games 24 --workers 2
  py track3_oracle/train_oracle.py --train-only        # reuse existing shards
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import random
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGENT = os.path.join(ROOT, "track1_search", "agent")
sys.path.insert(0, AGENT)
sys.path.insert(0, os.path.join(ROOT, "track1_search", "train"))

import nn_features as NF  # noqa: E402

# Oracle feature block: a compact summary of what the observer CANNOT see.
ORACLE_DIM = 24


def load_agent():
    spec = importlib.util.spec_from_file_location("or_main", os.path.join(AGENT, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.MY_DECK = m._load_deck()
    m._load_engine()
    m._load_card_db()
    m._ENGINE_TRIED = True
    return m


def oracle_features(cur, me, M, opp_hand_ids=None):
    """What the opponent is actually holding.

    NOTE: a seat's own observation hides the opponent's hand, so this CANNOT be
    read from `cur` -- doing so silently yields all zeros and makes the oracle
    arm identical to the observed arm. Because self play controls both seats,
    the caller passes the opponent's hand as most recently seen from THEIR seat.
    """
    f = np.zeros(ORACLE_DIM, dtype=np.float32)
    opl = cur["players"][1 - me]
    ids = opp_hand_ids
    if ids is None:
        hand = opl.get("hand")
        ids = [c["id"] for c in hand] if hand is not None else None
    if not ids:
        f[11] = min(opl.get("deckCount", 0), 60) / 60.0
        return f
    f[0] = min(len(ids), 16) / 16.0
    types = Counter()
    best_dmg = 0
    n_basic = n_evo = 0
    for cid in ids:
        c = M.CARD.get(cid, {})
        ct = c.get("cardType")
        if ct is not None and 0 <= ct <= 6:
            types[ct] += 1
        if c.get("basic"):
            n_basic += 1
        if c.get("stage1") or c.get("stage2"):
            n_evo += 1
        best_dmg = max(best_dmg, M._max_attack_damage(cid))
    for ct in range(7):
        f[1 + ct] = min(types.get(ct, 0), 6) / 6.0
    f[8] = min(n_basic, 6) / 6.0
    f[9] = min(n_evo, 6) / 6.0
    f[10] = min(best_dmg, 300) / 300.0
    f[11] = min(opl.get("deckCount", 0), 60) / 60.0
    return f


def generate(M, n_games, seed, out_dir):
    """Self play recording observed features, oracle features, and outcome."""
    from kaggle_environments import make
    rng = random.Random(seed)
    names = list(M.META_DECKS.keys())
    rec = {k: [] for k in ("kind", "card", "scal", "mask", "ctx", "stype",
                           "oracle", "player")}
    bounds = []
    last_hand = {}      # seat -> card ids that seat last saw in its own hand
    for g in range(n_games):
        start = len(rec["ctx"])
        last_hand.clear()

        def make_seat(deck_ids):
            def seat(obs):
                # single argument: kaggle_environments passes config to a
                # two-parameter agent, which silently breaks deck selection
                if obs.get("select") is None:
                    return list(deck_ids)
                sel = obs["select"]
                opts = sel.get("option") or []
                if not opts:
                    return []
                if len(opts) > 1:
                    cur = obs.get("current") or {}
                    me = cur.get("yourIndex", 0)
                    own = (cur["players"][me] or {}).get("hand")
                    if own is not None:
                        last_hand[me] = [c["id"] for c in own]
                    try:
                        scores = [M._option_score(o, sel) for o in opts]
                        k, c, s, mk, _slot = NF.encode(
                            {"current": cur, "select": sel}, me, M.CARD, M.ATTACK, scores)
                        rec["kind"].append(k); rec["card"].append(c)
                        rec["scal"].append(s); rec["mask"].append(mk)
                        rec["ctx"].append(int(sel.get("context") or 0))
                        rec["stype"].append(int(sel.get("type") or 0))
                        rec["oracle"].append(oracle_features(cur, me, M, last_hand.get(1 - me)))
                        rec["player"].append(me)
                    except Exception:
                        pass
                return M._validate(M._heuristic_action(sel, rng), sel) or [0]
            return seat

        a0 = make_seat(M.MY_DECK)
        a1 = make_seat(M.META_DECKS[names[g % len(names)]])
        env = make("cabt")
        try:
            env.run([a0, a1])
        except Exception as exc:
            print(f"  game {g} crashed: {exc!r}", flush=True)
            continue
        bounds.append((start, len(rec["ctx"]), env.state[0].reward))
        if (g + 1) % 4 == 0:
            print(f"  game {g+1}/{n_games}, {len(rec['ctx'])} decisions", flush=True)

    if not rec["ctx"]:
        raise SystemExit("no data")
    z = np.zeros(len(rec["ctx"]), dtype=np.float32)
    for s, e, r0 in bounds:
        for i in range(s, e):
            z[i] = float(r0) if rec["player"][i] == 0 else -float(r0)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"oracle_{int(time.time())}.npz")
    np.savez_compressed(p,
                        kind=np.array(rec["kind"], np.int8),
                        card=np.array(rec["card"], np.int16),
                        scal=np.array(rec["scal"], np.float32),
                        mask=np.array(rec["mask"], np.float32),
                        ctx=np.array(rec["ctx"], np.int16),
                        stype=np.array(rec["stype"], np.int16),
                        oracle=np.array(rec["oracle"], np.float32),
                        z=z)
    print(f"wrote {p} ({len(z)} samples)")
    return p


def pooled(scal, mask, kind, extra=None):
    n = scal.shape[0]
    feats = [np.ones((n, 1), np.float32)]
    for k in range(NF.N_KIND):
        sel = ((kind == k) & (mask > 0.5)).astype(np.float32)
        cnt = sel.sum(1, keepdims=True)
        feats.append((scal * sel[:, :, None]).sum(1) / np.maximum(cnt, 1.0))
        feats.append(cnt / 24.0)
    if extra is not None:
        feats.append(extra)
    return np.concatenate(feats, 1)


def fit_eval(X, y, ridge=1.0, seed=0):
    """Ridge regression, returns held out MAE. Same model class both arms, so
    the only difference between arms is the feature set."""
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nv = max(1, n // 5)
    vi, ti = perm[:nv], perm[nv:]
    A = X[ti].T @ X[ti] + ridge * np.eye(X.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, X[ti].T @ y[ti]).astype(np.float32)
    pred = np.tanh(X[vi] @ w)
    return float(np.abs(pred - y[vi]).mean()), float(np.abs(y[vi] - y[vi].mean()).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--train-only", action="store_true")
    a = ap.parse_args()

    if not a.train_only:
        M = load_agent()
        generate(M, a.games, 5, a.data)

    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    if not files:
        raise SystemExit(f"no shards in {a.data}")
    P = {k: [] for k in ("kind", "card", "scal", "mask", "oracle", "z")}
    for f in files:
        d = np.load(f)
        for k in P:
            P[k].append(d[k])
    D = {k: np.concatenate(v, 0) for k, v in P.items()}
    print(f"\n{len(files)} shards, {len(D['z'])} samples")

    kind = D["kind"].astype(np.int64)
    Xo = pooled(D["scal"], D["mask"], kind)                       # observed only
    Xr = pooled(D["scal"], D["mask"], kind, extra=D["oracle"])    # + oracle
    y = D["z"].astype(np.float32)

    mae_o, base = fit_eval(Xo, y)
    mae_r, _ = fit_eval(Xr, y)

    print("\n==== CHECKPOINT: how much is hidden information worth? ====")
    print(f"observed only : MAE {mae_o:.4f}  ({mae_o/base:.3f} x baseline)")
    print(f"with oracle   : MAE {mae_r:.4f}  ({mae_r/base:.3f} x baseline)")
    gain = (mae_o - mae_r) / max(mae_o, 1e-9)
    print(f"oracle advantage: {100*gain:+.1f}% error reduction")
    print(f"oracle features add {D['oracle'].shape[1]} dims to {Xo.shape[1]}")
    if gain > 0.05:
        print("\n=> Hidden information is worth real effort. Proceed: build a hidden")
        print("   card predictor and feed it into Track 1 determinization.")
    else:
        print("\n=> Hidden information buys little here. Uniform determinization is")
        print("   NOT the ceiling; drop this track and look elsewhere.")


if __name__ == "__main__":
    main()
