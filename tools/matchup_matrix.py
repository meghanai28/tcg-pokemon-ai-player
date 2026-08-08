"""Deck-versus-deck win rates mined from real ladder episodes.

Every deck decision in this repo so far has been made on a deck's *overall*
ladder win rate, which is an average over whatever field that deck happened to
face.  That is the wrong quantity.  What decides a ladder score is the win rate
against the field we will actually meet, weighted by how common each opponent
is -- and those can disagree sharply, because a deck with a mediocre average can
be favoured into the two archetypes that make up half the field.

Competition discussion (2026-08-03, topic 729926) claims exactly this shape for
Grimmsnarl: mediocre in the abstract, but favoured into Alakazam and Crustle,
which dominated the top of the board, and beaten by Dragapult + Crushing
Hammer, which almost nobody pilots well.  That is a testable claim and this
tool tests it against 28,006 episodes rather than accepting or dismissing it.

Reports, per deck, both numbers side by side:

    raw WR         its overall win rate, what we used before
    field WR       sum over opponents of share(opp) * winrate(vs opp)

`field WR` is the one that predicts a ladder result, because it re-weights the
matchups to the population we will be matched against.

Usage:
  py tools/matchup_matrix.py data/fresh/replays \\
      --leaderboard data/fresh/leaderboard/pokemon-tcg-ai-battle.zip \\
      --census harness/meta/deck_census.json --top 12 --min-elo 1000
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


def deck_key(deck):
    return tuple(sorted(deck))


def scan(job):
    """One archive: {(deck_a, deck_b): [a_wins, decided]} plus seat counts."""
    path, elos, min_elo, known = job
    known = {tuple(k) for k in known}
    pairs = collections.defaultdict(lambda: [0, 0])
    seats = collections.Counter()
    try:
        archive = zipfile.ZipFile(path)
    except Exception:
        return {}, {}
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
            if len(steps) < 2 or len(rewards) < 2:
                continue
            decks = [None, None]
            for p in range(2):
                if p < len(steps[1]):
                    act = (steps[1][p] or {}).get("action")
                    if isinstance(act, list) and len(act) == 60 and \
                            all(isinstance(c, int) for c in act):
                        decks[p] = deck_key(act)
            if decks[0] is None or decks[1] is None:
                continue
            if decks[0] not in known or decks[1] not in known:
                continue
            # both pilots must clear the Elo floor, else the matchup is
            # measuring a skill gap rather than a deck matchup
            ok = True
            for p in range(2):
                nm = names[p] if p < len(names) else None
                if elos.get(nm, -1) < min_elo:
                    ok = False
            if not ok:
                continue
            if rewards[0] is None or rewards[1] is None or rewards[0] == rewards[1]:
                continue
            for p in range(2):
                seats[decks[p]] += 1
            a, b = decks[0], decks[1]
            entry = pairs[(a, b)]
            entry[1] += 1
            if rewards[0] > rewards[1]:
                entry[0] += 1
    return ({k: v for k, v in pairs.items()}, dict(seats))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes")
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--census", default=os.path.join(
        ROOT, "harness", "meta", "deck_census.json"))
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--min-elo", type=float, default=1000.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default=os.path.join(
        ROOT, "harness", "meta", "matchups.json"))
    args = parser.parse_args()

    census = json.load(open(args.census, encoding="utf-8"))
    decks = census["decks"][:args.top]
    known = [d["deck"] for d in decks]
    label = {deck_key(d["deck"]): d["label"][:34] for d in decks}
    elos = load_elos(args.leaderboard)

    archives = (sorted(glob.glob(os.path.join(args.episodes, "*.zip")))
                if os.path.isdir(args.episodes) else [args.episodes])
    print(f"{len(archives)} archives, {len(known)} decks, min-elo {args.min_elo:g}",
          flush=True)

    pairs = collections.defaultdict(lambda: [0, 0])
    seats: collections.Counter = collections.Counter()
    jobs = [(a, elos, args.min_elo, known) for a in archives]
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        for i, (part, part_seats) in enumerate(pool.imap_unordered(scan, jobs), 1):
            for k, v in part.items():
                pairs[k][0] += v[0]
                pairs[k][1] += v[1]
            seats.update(part_seats)
            print(f"  archive {i}/{len(jobs)}", flush=True)

    # symmetrise: a beating b in seat 0 and b beating a in seat 1 are the same
    wins = collections.defaultdict(lambda: [0, 0])
    for (a, b), (a_wins, decided) in pairs.items():
        wins[(a, b)][0] += a_wins
        wins[(a, b)][1] += decided
        wins[(b, a)][0] += decided - a_wins
        wins[(b, a)][1] += decided

    total_seats = sum(seats.values()) or 1
    share = {k: v / total_seats for k, v in seats.items()}

    keys = [deck_key(d["deck"]) for d in decks if deck_key(d["deck"]) in seats]
    print(f"\n{'deck':<36} {'seats':>6} {'rawWR':>6} {'fieldWR':>8}")
    rows = []
    for k in keys:
        num = den = 0.0
        raw_w = raw_n = 0
        for j in keys:
            w, n = wins.get((k, j), [0, 0])
            if n == 0:
                continue
            raw_w += w
            raw_n += n
            num += share.get(j, 0.0) * (w / n)
            den += share.get(j, 0.0)
        rows.append((num / den if den else 0.0, raw_w / raw_n if raw_n else 0.0,
                     seats[k], k))
    rows.sort(key=lambda r: -r[0])
    for field_wr, raw_wr, n, k in rows:
        print(f"{label[k]:<36} {n:>6} {raw_wr:>5.1%} {field_wr:>8.1%}")

    print(f"\nmatchup grid (row win rate vs column), n in parentheses")
    head = "".join(f"{label[j][:9]:>12}" for j in keys)
    print(f"{'':<28}{head}")
    for k in keys:
        cells = ""
        for j in keys:
            w, n = wins.get((k, j), [0, 0])
            cells += f"{'-':>12}" if n < 8 else f"{w/n:>7.0%}({n:>3})"
        print(f"{label[k][:27]:<28}{cells}")

    payload = {"min_elo": args.min_elo,
               "decks": [{"label": label[k], "seats": seats[k],
                          "share": share[k], "raw_wr": raw, "field_wr": fw,
                          "deck": list(k)} for fw, raw, _n, k in rows],
               "matchups": [{"row": label[k], "col": label[j],
                             "wins": wins[(k, j)][0], "games": wins[(k, j)][1]}
                            for k in keys for j in keys
                            if wins.get((k, j), [0, 0])[1]]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
