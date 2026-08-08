"""Count DECISIONS per deck per Elo band, which is what decides trainability.

`top_decks.py` answers "which deck is good" by appearances and win rate.  That
is the wrong denominator for choosing what to train on: a deck with a great win
rate and 90 episodes cannot support a from-scratch prior, and the project has
already paid once for missing that (the `field_9` specialist had 67k decisions
and was a fine-tune for exactly this reason).

This walks the raw episode dumps once and reports, per 60-card list:

    seats, episodes, decisions, elite decisions, win rate, mean pilot Elo

Decisions are counted the way the trainer counts them -- ACTIVE seat, a real
`select` with two or more options -- so the number printed here is the number of
training rows that deck can actually supply, not an estimate.

Usage:
    py tools/deck_census.py data/fresh/replays --leaderboard <lb.zip> --min-elo 1000
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import multiprocessing as mp
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from mine_decks import card_names, load_elos  # noqa: E402


def scan_archive(args):
    """Per-zip worker: returns {deck_key: stats} for one archive."""
    path, elos, min_elo = args
    per_deck = collections.defaultdict(lambda: {
        "seats": 0, "decisions": 0, "elite_decisions": 0,
        "wins": 0, "decided": 0, "elo_sum": 0.0, "elo_n": 0,
        "episodes": set(),
    })
    try:
        archive = zipfile.ZipFile(path)
    except Exception:
        return {}
    with archive:
        for name in archive.namelist():
            if not name.endswith(".json"):
                continue
            try:
                with archive.open(name) as handle:
                    ep = json.load(io.TextIOWrapper(handle, "utf-8"))
            except Exception:
                continue
            steps = ep.get("steps") or []
            rewards = ep.get("rewards") or []
            info = ep.get("info") or {}
            agents = info.get("Agents") or []
            names = [a.get("Name") if isinstance(a, dict) else None for a in agents]
            if len(steps) < 3 or len(rewards) < 2:
                continue
            decks = [None, None]
            for p in range(2):
                if p < len(steps[1]):
                    act = (steps[1][p] or {}).get("action")
                    if isinstance(act, list) and len(act) == 60 and \
                            all(isinstance(c, int) for c in act):
                        decks[p] = tuple(sorted(act))
            # count decisions the way ingest does: ACTIVE seat, >=2 options
            counts = [0, 0]
            for t in range(1, len(steps) - 1):
                cur_step = steps[t]
                for p in range(2):
                    if p >= len(cur_step):
                        continue
                    obs = (cur_step[p] or {}).get("observation") or {}
                    sel = obs.get("select")
                    cur = obs.get("current")
                    if not sel or not cur:
                        continue
                    if cur.get("yourIndex") != p:
                        continue
                    if len(sel.get("option") or []) < 2:
                        continue
                    nxt = steps[t + 1]
                    act = (nxt[p] or {}).get("action") if p < len(nxt) else None
                    if not isinstance(act, list) or not act:
                        continue
                    counts[p] += 1
            for p in range(2):
                if decks[p] is None:
                    continue
                stat = per_deck[decks[p]]
                stat["seats"] += 1
                stat["episodes"].add(name)
                stat["decisions"] += counts[p]
                pilot = names[p] if p < len(names) else None
                elo = elos.get(pilot)
                if elo is not None:
                    stat["elo_sum"] += elo
                    stat["elo_n"] += 1
                    if elo >= min_elo:
                        stat["elite_decisions"] += counts[p]
                reward = rewards[p]
                other = rewards[1 - p]
                if reward is not None and other is not None and reward != other:
                    stat["decided"] += 1
                    stat["wins"] += 1 if reward > other else 0
    # sets do not pickle cheaply across the pool boundary at this size
    return {k: {**v, "episodes": len(v["episodes"])} for k, v in per_deck.items()}


def merge(into, other):
    for key, stat in other.items():
        dst = into[key]
        for field in ("seats", "decisions", "elite_decisions", "wins",
                      "decided", "elo_n", "episodes"):
            dst[field] += stat[field]
        dst["elo_sum"] += stat["elo_sum"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes")
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--min-elo", type=float, default=1000.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", default=os.path.join(ROOT, "harness", "meta",
                                                      "deck_census.json"))
    args = parser.parse_args()

    elos = load_elos(args.leaderboard)
    print(f"leaderboard: {len(elos)} teams")
    if os.path.isdir(args.episodes):
        archives = sorted(glob.glob(os.path.join(args.episodes, "*.zip")))
    else:
        archives = [args.episodes]
    print(f"{len(archives)} archive(s)")

    totals = collections.defaultdict(lambda: {
        "seats": 0, "decisions": 0, "elite_decisions": 0, "wins": 0,
        "decided": 0, "elo_sum": 0.0, "elo_n": 0, "episodes": 0})
    jobs = [(a, elos, args.min_elo) for a in archives]
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        for i, part in enumerate(pool.imap_unordered(scan_archive, jobs), 1):
            merge(totals, part)
            print(f"  archive {i}/{len(jobs)} merged ({len(totals)} lists)",
                  flush=True)

    names = card_names()

    def signature(deck):
        return ", ".join(names.get(c, str(c)) for c, _ in
                         collections.Counter(c for c in deck if c > 100)
                         .most_common(3))

    rows = []
    for deck, stat in totals.items():
        win_rate = stat["wins"] / stat["decided"] if stat["decided"] else 0.0
        mean_elo = stat["elo_sum"] / stat["elo_n"] if stat["elo_n"] else 0.0
        rows.append((stat["elite_decisions"], stat["decisions"], stat["seats"],
                     win_rate, mean_elo, deck))
    rows.sort(key=lambda r: -r[0])

    print(f"\n{'eliteDec':>9} {'allDec':>9} {'seats':>6} {'WR':>6} "
          f"{'meanElo':>8}  deck")
    payload = []
    for elite_dec, dec, seats, win_rate, mean_elo, deck in rows[:args.top]:
        label = signature(deck)
        print(f"{elite_dec:>9} {dec:>9} {seats:>6} {win_rate:>5.1%} "
              f"{mean_elo:>8.0f}  {label}")
        payload.append({"elite_decisions": elite_dec, "decisions": dec,
                        "seats": seats, "win_rate": win_rate,
                        "mean_elo": mean_elo, "label": label,
                        "deck": list(deck)})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"min_elo": args.min_elo, "decks": payload}, handle, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
