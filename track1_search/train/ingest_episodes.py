"""Build a behavior-cloning dataset from real ladder episode replays.

This is the "cloning warm start" data source: instead of imitating our own
search, the net first clones strong ladder pilots. Output shards are the SAME
format selfplay.py writes, so train_bc.py consumes either interchangeably --
the only difference is that `pi` is a one-hot of the human's actual choice
rather than a search visit distribution.

Get replays (see https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md):
    kaggle competitions leaderboard pokemon-tcg-ai-battle -d
    kaggle competitions submissions pokemon-tcg-ai-battle
    kaggle competitions episodes pokemon-tcg-ai-battle --submission-id <id> -p ep/
  or the daily top-rated episode export posted in the competition forums, and
  the per-day episode datasets:
    kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-<YYYY-MM-DD>

Episode JSON layout (per replay):
    steps[t][p]["observation"]  -- what player p saw at step t
    steps[t+1][p]["action"]     -- the ANSWER to step t (off-by-one)
    steps[1][p]["action"]       -- the 60-card deck (skipped as a decision)
    rewards / info.Agents[i].Name

Usage:
    py train/ingest_episodes.py ep/ --out train/data_bc --min-elo 1000 \
        --leaderboard lb/pokemon-tcg-ai-battle-publicleaderboard.csv
Process .zip archives directly -- do NOT extract (they unpack to ~21 GB).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SUB = os.path.join(ROOT, "track1_search", "agent")
sys.path.insert(0, SUB)

import nn_features as NF  # noqa: E402


def load_card_db():
    from cg.engine import get_lib
    lib = get_lib()
    card = {c["cardId"]: c for c in json.loads(lib.AllCard().decode())}
    atk = {a["attackId"]: a for a in json.loads(lib.AllAttack().decode())}
    return card, atk


def load_elos(path):
    """TeamName -> score from the downloaded leaderboard CSV."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("TeamName") or row.get("teamName")
            score = row.get("Score") or row.get("score")
            if name and score:
                try:
                    out[name] = float(score)
                except ValueError:
                    pass
    return out


def iter_episodes(path):
    """Yield parsed episode dicts from a dir of .json/.zip (zips stay closed)."""
    targets = []
    if os.path.isdir(path):
        targets = glob.glob(os.path.join(path, "**", "*.json"), recursive=True) + \
                  glob.glob(os.path.join(path, "**", "*.zip"), recursive=True)
    else:
        targets = [path]
    for t in targets:
        if t.endswith(".zip"):
            with zipfile.ZipFile(t) as z:
                for name in z.namelist():
                    if not name.endswith(".json"):
                        continue
                    try:
                        with z.open(name) as fh:
                            yield json.load(io.TextIOWrapper(fh, "utf-8")), name
                    except Exception:
                        continue
        else:
            try:
                with open(t, encoding="utf-8") as fh:
                    yield json.load(fh), os.path.basename(t)
            except Exception:
                continue


def stable_id(value):
    raw = str(value or "").encode("utf-8", "replace")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def episode_samples(ep, card_db, atk_db, elos, min_elo):
    """Yield (features..., pi, z, seat, pilot_id) for qualifying decisions."""
    steps = ep.get("steps") or []
    rewards = ep.get("rewards") or []
    info = ep.get("info") or {}
    agents = info.get("Agents") or []
    names = [a.get("Name") if isinstance(a, dict) else None for a in agents]

    keep = set()
    for i in range(2):
        if min_elo <= 0:
            keep.add(i)
        else:
            nm = names[i] if i < len(names) else None
            if nm and elos.get(nm, -1) >= min_elo:
                keep.add(i)
    if not keep or len(rewards) < 2:
        return

    for t in range(1, len(steps) - 1):
        cur_step, nxt_step = steps[t], steps[t + 1]
        for p in range(2):
            if p not in keep or p >= len(cur_step):
                continue
            obs = (cur_step[p] or {}).get("observation") or {}
            sel = obs.get("select")
            cur = obs.get("current")
            if not sel or not cur:
                continue
            opts = sel.get("option") or []
            if len(opts) < 2:      # forced move: nothing to learn
                continue
            action = (nxt_step[p] or {}).get("action") if p < len(nxt_step) else None
            if not isinstance(action, list) or not action:
                continue
            if any((not isinstance(i, int)) or i < 0 or i >= len(opts) for i in action):
                continue
            me = cur.get("yourIndex")
            if me is None or me != p:
                continue
            try:
                kind, card, scal, mask, opt_slot = NF.encode(
                    {"current": cur, "select": sel}, me, card_db, atk_db, None)
            except Exception:
                continue
            pi = np.zeros(NF.SEQ, dtype=np.float32)
            for i in action:
                if i < len(opt_slot) and opt_slot[i] >= 0:
                    pi[opt_slot[i]] += 1.0
            s = pi.sum()
            if s <= 0:
                continue
            pi /= s
            r = rewards[p] if rewards[p] is not None else 0
            yield (kind, card, scal, mask, int(sel.get("context") or 0),
                   int(sel.get("type") or 0), pi, float(r), p,
                   stable_id(names[p] if p < len(names) else None))


def main():
    global NF
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", help="dir or file of .json/.zip replays")
    ap.add_argument("--out", default=os.path.join(HERE, "data_bc"))
    ap.add_argument("--leaderboard", default=None)
    ap.add_argument("--min-elo", type=float, default=0.0)
    ap.add_argument("--max-samples", type=int, default=400000)
    ap.add_argument("--features", choices=("base", "rich"), default="base",
                    help="rich resolves area/index card references and records "
                         "the extra selection fields used by the engine")
    a = ap.parse_args()

    if a.features == "rich":
        import nn_features_rich
        NF = nn_features_rich

    card_db, atk_db = load_card_db()
    elos = load_elos(a.leaderboard)
    print(f"leaderboard entries: {len(elos)}; min-elo filter: {a.min_elo}; "
          f"features: {a.features}")

    acc = {k: [] for k in (
        "kind", "card", "scal", "mask", "ctx", "stype", "pi", "z",
        "group", "seat", "pilot")}
    n_ep = 0
    for ep, name in iter_episodes(a.episodes):
        n_ep += 1
        group = stable_id(name)
        for (kind, card, scal, mask, ctx, styp, pi, z, seat, pilot) in episode_samples(
                ep, card_db, atk_db, elos, a.min_elo):
            acc["kind"].append(kind); acc["card"].append(card)
            acc["scal"].append(scal); acc["mask"].append(mask)
            acc["ctx"].append(ctx); acc["stype"].append(styp)
            acc["pi"].append(pi); acc["z"].append(z)
            acc["group"].append(group); acc["seat"].append(seat)
            acc["pilot"].append(pilot)
        if len(acc["pi"]) >= a.max_samples:
            print("hit --max-samples cap")
            break
        if n_ep % 50 == 0:
            print(f"{n_ep} episodes -> {len(acc['pi'])} samples", flush=True)

    if not acc["pi"]:
        raise SystemExit("no samples extracted -- check paths / --min-elo")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"bc_{n_ep}eps.npz")
    np.savez_compressed(
        path,
        kind=np.array(acc["kind"], dtype=np.int8),
        card=np.array(acc["card"], dtype=np.int16),
        scal=np.array(acc["scal"], dtype=np.float32),
        mask=np.array(acc["mask"], dtype=np.float32),
        ctx=np.array(acc["ctx"], dtype=np.int16),
        stype=np.array(acc["stype"], dtype=np.int16),
        pi=np.array(acc["pi"], dtype=np.float32),
        z=np.array(acc["z"], dtype=np.float32),
        group=np.array(acc["group"], dtype=np.uint64),
        seat=np.array(acc["seat"], dtype=np.int8),
        pilot=np.array(acc["pilot"], dtype=np.uint64),
        features=np.array(a.features),
    )
    print(f"wrote {path}: {len(acc['pi'])} samples from {n_ep} episodes")
    print(f"train with: py train/train_bc.py --data {a.out}")


if __name__ == "__main__":
    main()
