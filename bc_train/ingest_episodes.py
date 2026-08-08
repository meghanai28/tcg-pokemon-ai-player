"""Build a behaviour-cloning corpus from raw ladder episode dumps.

This is the quarantined `track1_search/train/ingest_episodes.py`, moved into
`bc_train/` so it imports **this package's** encoders.  That detail is the whole
reason the file was rewritten rather than path-hacked: `bc_train/nn_features.py`
is `MAX_OPT 24` (SEQ 53), the width the frozen search shell serves, while
`foundation/nn_features.py` is `MAX_OPT 64` (SEQ 93).  A corpus built against
the wrong one trains a model that loads without error and plays a different
policy, which is the silent failure this repo keeps paying for.

Two things this version adds, both needed to train a deck specialist from
scratch rather than as a fine-tune:

  * **Parallel over archives.**  A day's dump is ~740 MB and there are six of
    them.  One worker per archive, writing its own shard, keeps a full six-day
    ingest inside an hour instead of most of a night.
  * **Deck-cluster filtering.**  `--deck` alone keeps only an exact 60-card
    multiset, which splits a deck across its one-card tech variants and starves
    every variant of sample size.  `--deck-jaccard` keeps anything within a
    multiset-Jaccard radius of the target, which is how `top_decks.py` already
    decides two lists are the same deck.

Shards carry a per-decision `elo`, so `tools/filter_by_elo.py` can cut an elite
subset afterwards without a re-ingest.

Usage:
    py bc_train/ingest_episodes.py data/fresh/replays --out data/bc_lucario \\
        --leaderboard data/fresh/leaderboard/pokemon-tcg-ai-battle.zip \\
        --min-elo 1000 --deck data/decks/lucario.csv --deck-jaccard 0.8 \\
        --features rich --workers 6
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)          # for foundation.cg.engine (the card database)

NF = None      # bound to nn_features or nn_features_rich in each worker


def load_card_db():
    from foundation.cg.engine import get_lib
    lib = get_lib()
    card = {c["cardId"]: c for c in json.loads(lib.AllCard().decode())}
    atk = {a["attackId"]: a for a in json.loads(lib.AllAttack().decode())}
    return card, atk


def load_elos(path):
    """TeamName -> ladder score, from the downloaded leaderboard csv or zip."""
    import csv
    if not path or not os.path.exists(path):
        return {}
    archive = None
    if path.lower().endswith(".zip"):
        archive = zipfile.ZipFile(path)
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not members:
            archive.close()
            return {}
        stream = io.TextIOWrapper(archive.open(members[0]), "utf-8")
    else:
        stream = open(path, newline="", encoding="utf-8")
    try:
        out = {}
        for row in csv.DictReader(stream):
            name = row.get("TeamName") or row.get("teamName")
            score = row.get("Score") or row.get("score")
            if name and score:
                try:
                    out[name] = float(score)
                except ValueError:
                    continue
        return out
    finally:
        stream.close()
        if archive is not None:
            archive.close()


def stable_id(value):
    raw = str(value or "").encode("utf-8", "replace")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def multiset_jaccard(a: collections.Counter, b: collections.Counter) -> float:
    """Intersection over union counting duplicates, which decklists have."""
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


def episode_samples(ep, card_db, atk_db, elos, min_elo, target, jaccard):
    """Yield one training row per eligible decision in one episode."""
    steps = ep.get("steps") or []
    rewards = ep.get("rewards") or []
    info = ep.get("info") or {}
    agents = info.get("Agents") or []
    names = [a.get("Name") if isinstance(a, dict) else None for a in agents]
    if len(steps) < 3 or len(rewards) < 2:
        return
    decks = [[], []]
    for p in range(2):
        if p < len(steps[1]):
            action = (steps[1][p] or {}).get("action")
            if (isinstance(action, list) and len(action) == 60 and
                    all(isinstance(cid, int) for cid in action)):
                decks[p] = action

    keep = set()
    for p in range(2):
        if target is not None:
            if not decks[p]:
                continue
            if multiset_jaccard(collections.Counter(decks[p]), target) < jaccard:
                continue
        if min_elo <= 0:
            keep.add(p)
        else:
            name = names[p] if p < len(names) else None
            if name and elos.get(name, -1) >= min_elo:
                keep.add(p)
    if not keep:
        return

    for t in range(1, len(steps) - 1):
        cur_step, nxt_step = steps[t], steps[t + 1]
        for p in keep:
            if p >= len(cur_step):
                continue
            obs = (cur_step[p] or {}).get("observation") or {}
            sel = obs.get("select")
            cur = obs.get("current")
            if not sel or not cur:
                continue
            opts = sel.get("option") or []
            if len(opts) < 2:          # forced move: nothing to learn
                continue
            action = (nxt_step[p] or {}).get("action") if p < len(nxt_step) else None
            if not isinstance(action, list) or not action:
                continue
            if any((not isinstance(i, int)) or i < 0 or i >= len(opts)
                   for i in action):
                continue
            me = cur.get("yourIndex")
            if me is None or me != p:
                continue
            try:
                kind, card, scal, mask, opt_slot = NF.encode(
                    {"current": cur, "select": sel, "decklist": decks[p]},
                    me, card_db, atk_db, None)
            except Exception:
                continue
            pi = np.zeros(NF.SEQ, dtype=np.float32)
            for i in action:
                if i < len(opt_slot) and opt_slot[i] >= 0:
                    pi[opt_slot[i]] += 1.0
            total = pi.sum()
            if total <= 0:
                continue
            pi /= total
            reward = rewards[p] if rewards[p] is not None else 0
            pilot = names[p] if p < len(names) else None
            yield (kind, card, scal, mask, int(sel.get("context") or 0),
                   int(sel.get("type") or 0), pi, float(reward), p,
                   stable_id(pilot), float(elos.get(pilot, min_elo)),
                   stable_id(tuple(sorted(decks[p]))))


FIELDS = ("kind", "card", "scal", "mask", "ctx", "stype", "pi", "z",
          "group", "seat", "pilot", "elo", "deck")

# A worker holds its rows as python lists until it flushes.  At the measured
# 7,396 bytes per decision, 60k rows is ~440 MB, so six workers peak around
# 2.7 GB.  Ingesting a whole day into one shard instead would be ~2.5 GB per
# worker and is how this machine's VM was killed twice before.
FLUSH_EVERY = 60_000


def _write_shard(acc, out_dir, tag, index):
    os.makedirs(out_dir, exist_ok=True)
    shard = os.path.join(out_dir, f"bc_{tag}_{index:02d}_{len(acc['pi'])}.npz")
    np.savez_compressed(
        shard,
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
        elo=np.array(acc["elo"], dtype=np.float32),
        deck=np.array(acc["deck"], dtype=np.uint64),
    )
    return shard


def scan_archive(job):
    """Worker: ingest one archive, flushing a shard every FLUSH_EVERY rows."""
    global NF
    (path, out_dir, features, elos, min_elo, target_list, jaccard,
     max_samples) = job
    if features == "rich":
        import nn_features_rich as module
    else:
        import nn_features as module
    NF = module
    card_db, atk_db = load_card_db()
    target = collections.Counter(target_list) if target_list else None

    acc = {k: [] for k in FIELDS}
    episodes = 0
    written = 0
    shards = []
    tag = os.path.basename(path).replace(".zip", "").split("episodes-")[-1]
    try:
        archive = zipfile.ZipFile(path)
    except Exception as exc:
        return {"archive": os.path.basename(path), "error": repr(exc)}
    with archive:
        for name in archive.namelist():
            if not name.endswith(".json"):
                continue
            try:
                with archive.open(name) as handle:
                    ep = json.load(io.TextIOWrapper(handle, "utf-8"))
            except Exception:
                continue
            episodes += 1
            group = stable_id(name)
            for sample in episode_samples(ep, card_db, atk_db, elos, min_elo,
                                          target, jaccard):
                values = sample[:8] + (group,) + sample[8:]
                for key, value in zip(FIELDS, values):
                    acc[key].append(value)
            if len(acc["pi"]) >= FLUSH_EVERY:
                shards.append(_write_shard(acc, out_dir, tag, len(shards)))
                written += len(acc["pi"])
                acc = {k: [] for k in FIELDS}
            if max_samples and written + len(acc["pi"]) >= max_samples:
                break

    if acc["pi"]:
        shards.append(_write_shard(acc, out_dir, tag, len(shards)))
        written += len(acc["pi"])
    return {"archive": os.path.basename(path), "episodes": episodes,
            "samples": written, "shards": shards}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", help="dir or file of .zip episode dumps")
    parser.add_argument("--out", required=True)
    parser.add_argument("--leaderboard", default=None)
    parser.add_argument("--min-elo", type=float, default=0.0)
    parser.add_argument("--deck", default=None,
                        help="deck.csv or json list of 60 card ids; keep only "
                             "decisions from pilots on this list")
    parser.add_argument("--deck-jaccard", type=float, default=1.0,
                        help="multiset-Jaccard radius around --deck. 1.0 is an "
                             "exact match; 0.8 folds in tech variants")
    parser.add_argument("--features", choices=("base", "rich"), default="rich")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-samples-per-archive", type=int, default=0)
    args = parser.parse_args()

    target_list = None
    if args.deck:
        with open(args.deck, encoding="utf-8") as handle:
            text = handle.read().strip()
        if text.startswith("["):
            target_list = [int(c) for c in json.loads(text)]
        else:
            target_list = [int(line.strip()) for line in text.splitlines()
                           if line.strip()]
        if len(target_list) != 60:
            parser.error(f"--deck must hold 60 card ids, got {len(target_list)}")

    elos = load_elos(args.leaderboard)
    if os.path.isdir(args.episodes):
        archives = sorted(glob.glob(os.path.join(args.episodes, "*.zip")))
    else:
        archives = [args.episodes]
    print(f"leaderboard: {len(elos)} teams; min-elo {args.min_elo:g}; "
          f"features {args.features}; deck filter "
          f"{'yes @ jaccard %.2f' % args.deck_jaccard if target_list else 'none'}; "
          f"{len(archives)} archive(s)", flush=True)

    jobs = [(a, args.out, args.features, elos, args.min_elo, target_list,
             args.deck_jaccard, args.max_samples_per_archive) for a in archives]
    total = 0
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        for result in pool.imap_unordered(scan_archive, jobs):
            if result.get("error"):
                print(f"  {result['archive']}: FAILED {result['error']}",
                      flush=True)
                continue
            total += result["samples"]
            print(f"  {result['archive']}: {result['episodes']} eps -> "
                  f"{result['samples']} decisions", flush=True)
    print(f"\ntotal decisions: {total}")
    if total == 0:
        raise SystemExit("no samples extracted -- check --deck / --min-elo")


if __name__ == "__main__":
    main()
