"""Rank decks by who plays them, not just by how often they win.

`build_fresh_deck_pool.py` ranks decks on the Wilson lower bound of their ladder
win rate above an Elo floor.  That answers "which deck wins games", which is not
the same question as "which deck do the strongest players choose", and the two
come apart badly here: the most-played list in the field sits at a 48.5% win
rate, and the best win rate in the pool belongs to a list played by exactly one
person, the rank-1 player.

A win rate conflates the deck with its pilots.  Adoption by strong players is a
different, partly independent signal: it is what the people who understand the
format best decided to sleeve up, and it does not reward a deck merely for being
easy to pilot.  This reports both, side by side, plus the pilot Elo distribution
behind each list, so the two can be compared rather than silently averaged.

Decks are grouped by exact 60-card list and then clustered by multiset Jaccard,
because near-identical lists (a one-card tech swap) are the same deck for our
purposes and splitting them starves every variant of sample size.

Usage:
  py tools/top_decks.py data/fresh/replays --leaderboard data/fresh/leaderboard/*.zip
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import multiprocessing as mp
import os
import sys
import zipfile
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mine_decks import card_names, load_elos  # noqa: E402


def wilson_lower(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - spread) / denominator


def _scan(job):
    """Pull (pilot, deck, won) triples out of one zip member."""
    zip_path, member = job
    out = []
    try:
        with zipfile.ZipFile(zip_path) as archive, archive.open(member) as source:
            episode = json.load(io.TextIOWrapper(source, encoding="utf-8"))
        steps = episode["steps"]
        rewards = episode.get("rewards") or [None, None]
        agents = (episode.get("info") or {}).get("Agents") or []
        for seat in range(2):
            pilot = (agents[seat].get("Name")
                     if seat < len(agents) and isinstance(agents[seat], dict) else None)
            deck = steps[1][seat].get("action")
            if isinstance(deck, list) and len(deck) == 60:
                out.append((pilot, tuple(sorted(deck)), rewards[seat] == 1))
    except Exception:
        return []
    return out


def jaccard(a: tuple, b: tuple) -> float:
    ca, cb = Counter(a), Counter(b)
    intersection = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return intersection / union if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes")
    parser.add_argument("--leaderboard", default=os.path.join(
        ROOT, "data", "fresh", "leaderboard", "pokemon-tcg-ai-battle.zip"))
    parser.add_argument("--elite-elo", type=float, default=1100.0,
                        help="Elo at or above which a pilot counts as elite")
    parser.add_argument("--cluster-threshold", type=float, default=0.85,
                        help="multiset Jaccard above which two lists are one deck")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--out", default=os.path.join(ROOT, "harness", "top_decks.json"))
    args = parser.parse_args()

    elos = load_elos(args.leaderboard)
    print(f"leaderboard: {len(elos)} teams, "
          f"{sum(1 for v in elos.values() if v >= args.elite_elo)} at/above {args.elite_elo:g}",
          flush=True)

    targets = [args.episodes]
    if os.path.isdir(args.episodes):
        targets = sorted(glob.glob(os.path.join(args.episodes, "**", "*.zip"), recursive=True))
    jobs = []
    for target in targets:
        with zipfile.ZipFile(target) as archive:
            jobs += [(target, m) for m in archive.namelist() if m.lower().endswith(".json")]
    print(f"{len(targets)} archives, {len(jobs)} episodes", flush=True)

    appearances: Counter = Counter()
    wins: dict = defaultdict(int)
    elite: Counter = Counter()
    pilots: dict = defaultdict(Counter)
    pilot_elos: dict = defaultdict(list)

    with mp.get_context("spawn").Pool(args.workers) as pool:
        for done, rows in enumerate(pool.imap_unordered(_scan, jobs, chunksize=32), 1):
            for pilot, deck, won in rows:
                appearances[deck] += 1
                if won:
                    wins[deck] += 1
                if pilot:
                    pilots[deck][pilot] += 1
                    rating = elos.get(pilot)
                    if rating is not None:
                        pilot_elos[deck].append(rating)
                        if rating >= args.elite_elo:
                            elite[deck] += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    # cluster near-identical lists into one deck, largest list as representative
    ordered = [deck for deck, _ in appearances.most_common()]
    clusters: list[list[tuple]] = []
    for deck in ordered:
        for cluster in clusters:
            if jaccard(deck, cluster[0]) >= args.cluster_threshold:
                cluster.append(deck)
                break
        else:
            clusters.append([deck])

    rows = []
    for cluster in clusters:
        total = sum(appearances[d] for d in cluster)
        won = sum(wins[d] for d in cluster)
        elite_n = sum(elite[d] for d in cluster)
        ratings = [r for d in cluster for r in pilot_elos[d]]
        combined: Counter = Counter()
        for d in cluster:
            combined.update(pilots[d])
        rows.append({
            "representative": list(cluster[0]),
            "variants": len(cluster),
            "appearances": total,
            "wins": won,
            "win_rate": won / total if total else 0.0,
            "wilson_lower_95": wilson_lower(won, total),
            "elite_appearances": elite_n,
            "elite_share": elite_n / total if total else 0.0,
            "mean_pilot_elo": sum(ratings) / len(ratings) if ratings else None,
            "max_pilot_elo": max(ratings) if ratings else None,
            "top_pilots": combined.most_common(3),
        })

    names = card_names()

    def signature(deck):
        return ", ".join(names.get(c, str(c)) for c, _ in
                         Counter([c for c in deck if c > 100]).most_common(3))

    by_elite = sorted(rows, key=lambda r: -r["elite_appearances"])
    by_wilson = sorted([r for r in rows if r["appearances"] >= 50],
                       key=lambda r: -r["wilson_lower_95"])

    print(f"\n=== ranked by ELITE ADOPTION (pilots at/above {args.elite_elo:g} Elo) ===", flush=True)
    print(f"{'elite':>6} {'apps':>6} {'WR':>6} {'wilson':>7} {'meanElo':>8}  deck", flush=True)
    for row in by_elite[:args.top]:
        print(f"{row['elite_appearances']:6d} {row['appearances']:6d} "
              f"{row['win_rate'] * 100:5.1f}% {row['wilson_lower_95']:7.3f} "
              f"{(row['mean_pilot_elo'] or 0):8.0f}  {signature(row['representative'])}",
              flush=True)

    print(f"\n=== ranked by WIN RATE (Wilson lower, >=50 appearances) ===", flush=True)
    for row in by_wilson[:args.top]:
        print(f"{row['elite_appearances']:6d} {row['appearances']:6d} "
              f"{row['win_rate'] * 100:5.1f}% {row['wilson_lower_95']:7.3f} "
              f"{(row['mean_pilot_elo'] or 0):8.0f}  {signature(row['representative'])}",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as target:
        json.dump({
            "elite_elo": args.elite_elo,
            "cluster_threshold": args.cluster_threshold,
            "episodes": len(jobs),
            "distinct_lists": len(appearances),
            "clusters": len(clusters),
            "decks": sorted(rows, key=lambda r: -r["elite_appearances"]),
        }, target, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
