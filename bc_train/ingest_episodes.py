"""Build a behaviour-cloning corpus from raw ladder episode dumps.

This lives in `bc_train/` so it imports **this package's** encoders. That
detail is the whole
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


def episode_samples(ep, card_db, atk_db, elos, min_elo, target, jaccard,
                    contexts=None, looking_only=False):
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
            context = int(sel.get("context") or 0)
            if contexts is not None and context not in contexts:
                continue
            if looking_only:
                looking = cur.get("looking")
                if (not isinstance(looking, list) or
                        not any(option.get("area") == 12 for option in opts)):
                    continue
            action = (nxt_step[p] or {}).get("action") if p < len(nxt_step) else None
            if not isinstance(action, list):
                continue
            is_v3 = getattr(NF, "FEATURE_VERSION", 1) == 3
            if not action and not is_v3:
                continue
            if (not action and
                    (int(sel.get("minCount") or 0) != 0 or
                     not hasattr(NF, "EMPTY_SLOT"))):
                continue
            if any((not isinstance(i, int)) or i < 0 or i >= len(opts)
                   for i in action):
                continue
            me = cur.get("yourIndex")
            if me is None or me != p:
                continue
            try:
                encoded = NF.encode(
                    {"current": cur, "select": sel, "decklist": decks[p],
                     "logs": obs.get("logs") or []},
                    me, card_db, atk_db, None)
            except Exception:
                continue
            kind, card, scal, mask, opt_slot = encoded[:5]
            selected_slots = [
                opt_slot[i] if i < len(opt_slot) else -1 for i in action]
            # Never turn a demonstrated set into a partial label when option
            # truncation removed one of its members.
            if any(slot < 0 for slot in selected_slots):
                continue
            pi = np.zeros(NF.SEQ, dtype=np.float32)
            if action:
                for slot in selected_slots:
                    pi[slot] += 1.0
            else:
                pi[NF.EMPTY_SLOT] = 1.0
            total = pi.sum()
            if total <= 0:
                continue
            pi /= total
            reward = rewards[p] if rewards[p] is not None else 0
            pilot = names[p] if p < len(names) else None
            base = (kind, card, scal, mask, context,
                    int(sel.get("type") or 0), pi, float(reward), p,
                    stable_id(pilot), float(elos.get(pilot, min_elo)),
                    stable_id(tuple(sorted(decks[p]))))
            if is_v3:
                if len(action) > MAX_DEMO_SIZE:
                    continue
                demo_tokens = np.full(MAX_DEMO_SIZE, -1, dtype=np.int16)
                if selected_slots:
                    demo_tokens[:len(selected_slots)] = selected_slots
                yield base + tuple(encoded[5:]) + (
                    demo_tokens, np.int8(len(selected_slots)))
            else:
                yield base


FIELDS = ("kind", "card", "scal", "mask", "ctx", "stype", "pi", "z",
          "group", "seat", "pilot", "elo", "deck")
V3_FIELDS = ("bag_card", "bag_count", "bag_kind", "bag_scal", "bag_mask",
             "demo_tokens", "demo_size")
MAX_DEMO_SIZE = 8

# A worker holds its rows as python lists until it flushes.  At the measured
# 7,396 bytes per decision, 60k rows is ~440 MB, so six workers peak around
# 2.7 GB.  Ingesting a whole day into one shard instead would be ~2.5 GB per
# worker and is how this machine's VM was killed twice before.
FLUSH_EVERY = 60_000


def _write_shard(acc, out_dir, tag, index):
    os.makedirs(out_dir, exist_ok=True)
    shard = os.path.join(out_dir, f"bc_{tag}_{index:02d}_{len(acc['pi'])}.npz")
    dtypes = {
        "kind": np.int8, "card": np.int16, "scal": np.float32,
        "mask": np.float32, "ctx": np.int16, "stype": np.int16,
        "pi": np.float32, "z": np.float32, "group": np.uint64,
        "seat": np.int8, "pilot": np.uint64, "elo": np.float32,
        "deck": np.uint64, "bag_card": np.int16,
        "bag_count": np.uint8, "bag_kind": np.int8,
        "bag_scal": np.float32, "bag_mask": np.float32,
        "demo_tokens": np.int16, "demo_size": np.int8,
    }
    arrays = {key: np.asarray(values, dtype=dtypes[key])
              for key, values in acc.items()}
    temp = shard + ".partial"
    with open(temp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, shard)
    return shard


def scan_archive(job):
    """Worker: ingest one archive, flushing a shard every FLUSH_EVERY rows."""
    global NF
    (path, out_dir, features, elos, min_elo, target_list, jaccard,
     max_samples, contexts, looking_only, resume) = job
    if features == "v3":
        import nn_features_v3 as module
    elif features == "rich":
        import nn_features_rich as module
    else:
        import nn_features as module
    NF = module
    card_db, atk_db = load_card_db()
    target = collections.Counter(target_list) if target_list else None

    sample_fields = FIELDS + (V3_FIELDS if features == "v3" else ())
    tag = os.path.basename(path).replace(".zip", "").split("episodes-")[-1]
    done_dir = os.path.join(out_dir, ".done")
    done_path = os.path.join(done_dir, tag + ".json")
    if resume and os.path.isfile(done_path):
        try:
            with open(done_path, encoding="utf-8") as handle:
                previous = json.load(handle)
            if previous.get("archive") == os.path.basename(path):
                previous["resumed"] = True
                return previous
        except (OSError, ValueError):
            pass
    acc = {k: [] for k in sample_fields}
    flush_every = 40_000 if features == "v3" else FLUSH_EVERY
    episodes = 0
    written = 0
    shards = []
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
                                          target, jaccard, contexts,
                                          looking_only):
                values = sample[:8] + (group,) + sample[8:]
                for key, value in zip(sample_fields, values):
                    acc[key].append(value)
            if len(acc["pi"]) >= flush_every:
                shards.append(_write_shard(acc, out_dir, tag, len(shards)))
                written += len(acc["pi"])
                acc = {k: [] for k in sample_fields}
            if max_samples and written + len(acc["pi"]) >= max_samples:
                break

    if acc["pi"]:
        shards.append(_write_shard(acc, out_dir, tag, len(shards)))
        written += len(acc["pi"])
    result = {"archive": os.path.basename(path), "episodes": episodes,
              "samples": written, "shards": shards, "features": features}
    os.makedirs(done_dir, exist_ok=True)
    temp_done = done_path + ".partial"
    with open(temp_done, "w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_done, done_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", nargs="+",
                        help="one or more directories/files of .zip dumps")
    parser.add_argument("--out", required=True)
    parser.add_argument("--leaderboard", default=None)
    parser.add_argument("--min-elo", type=float, default=0.0)
    parser.add_argument("--deck", default=None,
                        help="deck.csv or json list of 60 card ids; keep only "
                             "decisions from pilots on this list")
    parser.add_argument("--deck-jaccard", type=float, default=1.0,
                        help="multiset-Jaccard radius around --deck. 1.0 is an "
                             "exact match; 0.8 folds in tech variants")
    parser.add_argument("--features", choices=("base", "rich", "v3"),
                        default="rich")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-samples-per-archive", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="skip archives with a completed per-archive marker")
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=None,
        help="optional SelectContext allow-list for surgical re-ingestion")
    parser.add_argument(
        "--looking-only", action="store_true",
        help="keep only public current.looking selections (area 12); useful "
             "for replacing rich-v1 rows whose card identities were erased")
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
    archives = []
    for source in args.episodes:
        if os.path.isdir(source):
            archives.extend(glob.glob(os.path.join(source, "*.zip")))
        else:
            archives.append(source)
    archives = sorted(set(archives))
    if not archives:
        parser.error("no episode .zip archives found")
    print(f"leaderboard: {len(elos)} teams; min-elo {args.min_elo:g}; "
          f"features {args.features}; deck filter "
          f"{'yes @ jaccard %.2f' % args.deck_jaccard if target_list else 'none'}; "
          f"{len(archives)} archive(s)", flush=True)

    jobs = [(a, args.out, args.features, elos, args.min_elo, target_list,
             args.deck_jaccard, args.max_samples_per_archive,
             set(args.contexts) if args.contexts else None,
             args.looking_only, args.resume) for a in archives]
    total = 0
    results = []
    failures = []
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        for result in pool.imap_unordered(scan_archive, jobs):
            if result.get("error"):
                print(f"  {result['archive']}: FAILED {result['error']}",
                      flush=True)
                failures.append(result)
                continue
            results.append(result)
            total += result["samples"]
            print(f"  {result['archive']}: {result['episodes']} eps -> "
                  f"{result['samples']} decisions"
                  f"{' (resumed)' if result.get('resumed') else ''}", flush=True)
    print(f"\ntotal decisions: {total}")
    if failures:
        raise SystemExit(f"{len(failures)} archive(s) failed; completion withheld")
    if total == 0:
        raise SystemExit("no samples extracted -- check --deck / --min-elo")
    complete = {
        "features": args.features, "archives": len(results),
        "decisions": total, "min_elo": args.min_elo,
        "deck_filtered": bool(target_list),
        "results": sorted(results, key=lambda row: row["archive"]),
    }
    os.makedirs(args.out, exist_ok=True)
    complete_path = os.path.join(args.out, "_COMPLETE.json")
    temp_complete = complete_path + ".partial"
    with open(temp_complete, "w", encoding="utf-8") as handle:
        json.dump(complete, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_complete, complete_path)


if __name__ == "__main__":
    main()
