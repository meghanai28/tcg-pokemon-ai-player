"""Bulk-download episode replays from the strongest teams on the ladder.

Walks the public leaderboard top-N, lists each team's submissions, and pulls
their recent episodes. Skips anything already on disk so it is resumable.

    py tools/fetch_replays.py --top 20 --per-sub 25 --out ep

Requires KAGGLE_API_TOKEN in the environment. Replays are ~4 MB each, so
budget disk: 500 episodes ~= 2 GB.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys


def sh(args):
    return subprocess.run([sys.executable, "-m", "kaggle"] + args,
                          capture_output=True, text=True).stdout


def top_teams(lb_glob, n):
    files = sorted(glob.glob(lb_glob))
    if not files:
        raise SystemExit(f"no leaderboard csv matching {lb_glob} -- run:\n"
                         "  py -m kaggle competitions leaderboard pokemon-tcg-ai-battle -d -p lb")
    rows = []
    with open(files[-1], newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r["TeamId"]), r["TeamName"], float(r["Score"])))
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda t: -t[2])
    return rows[:n]


def parse_scored_subs(text, min_score):
    """(submission_id, publicScore) rows above min_score.

    A top-20 team often has several submissions, and only the newest is
    necessarily strong: the rank-1 team currently holds both a 1221.4 and a
    924.9. Pulling episodes from the weak one dilutes the corpus with exactly
    the play we are trying to learn away from.
    """
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        score = None
        for tok in reversed(parts):
            try:
                score = float(tok)
                break
            except ValueError:
                continue
        if score is None or score < min_score:
            continue
        out.append((int(parts[0]), score))
    return out


def parse_ids(text, col=0):
    out = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[col].isdigit():
            continue
        out.append(int(parts[col]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20, help="top N teams by Elo")
    ap.add_argument("--per-sub", type=int, default=25, help="episodes per submission")
    ap.add_argument("--out", default="ep")
    ap.add_argument("--lb", default="lb/*.csv")
    ap.add_argument("--min-sub-score", type=float, default=0.0,
                    help="skip a team's weaker submissions; only episodes from "
                         "submissions at or above this public score are pulled")
    a = ap.parse_args()

    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("set KAGGLE_API_TOKEN first")
    os.makedirs(a.out, exist_ok=True)
    have = {os.path.basename(p) for p in glob.glob(os.path.join(a.out, "*.json"))}
    print(f"{len(have)} replays already on disk")

    got = 0
    for team_id, name, score in top_teams(a.lb, a.top):
        scored = parse_scored_subs(
            sh(["competitions", "team-submissions", str(team_id)]),
            a.min_sub_score)
        print(f"[{name} | {score}] {len(scored)} submissions at/above "
              f"{a.min_sub_score}", flush=True)
        for sid, sub_score in scored:
            eps = parse_ids(sh(["competitions", "episodes", str(sid)]))[:a.per_sub]
            for eid in eps:
                fn = f"episode-{eid}-replay.json"
                if fn in have:
                    continue
                sh(["competitions", "replay", str(eid), "-p", a.out])
                if os.path.exists(os.path.join(a.out, fn)):
                    have.add(fn)
                    got += 1
                    if got % 10 == 0:
                        print(f"  downloaded {got} new replays", flush=True)
    print(f"done: {got} new, {len(have)} total in {a.out}/")


if __name__ == "__main__":
    main()
