"""Mine actual decklists and their win rates from downloaded replays.

Every episode stores both players' 60-card decks (the step-1 actions). This
extracts them, groups exact lists, ranks by frequency and win rate, and emits
a ready-to-paste META_DECKS block for submission/main.py so belief sampling
reflects the field as it is being played right now.

Usage: py tools/mine_decks.py ep/ --top 12 [--emit meta_decks.py]
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "track1_search", "agent"))


def card_names():
    try:
        from cg.engine import get_lib
        lib = get_lib()
        return {c["cardId"]: c["name"] for c in json.loads(lib.AllCard().decode())}
    except Exception:
        return {}


def load_elos(path):
    if not path:
        return {}
    archive = None
    if path.lower().endswith(".zip"):
        archive = zipfile.ZipFile(path)
        members = [name for name in archive.namelist()
                   if name.lower().endswith(".csv")]
        if not members:
            archive.close()
            return {}
        source = io.TextIOWrapper(archive.open(members[0]), encoding="utf-8")
    else:
        source = open(path, newline="", encoding="utf-8")
    try:
        result = {}
        for row in csv.DictReader(source):
            name = row.get("TeamName") or row.get("teamName")
            score = row.get("Score") or row.get("score")
            try:
                if name and score:
                    result[name] = float(score)
            except ValueError:
                continue
        return result
    finally:
        source.close()
        if archive is not None:
            archive.close()


def iter_episodes(path):
    """Stream episode JSON from a file or directory without extracting ZIPs."""
    targets = [path]
    if os.path.isdir(path):
        targets = (
            glob.glob(os.path.join(path, "**", "*.json"), recursive=True) +
            glob.glob(os.path.join(path, "**", "*.zip"), recursive=True)
        )
    for target in targets:
        if target.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(target) as archive:
                    for member in archive.namelist():
                        if not member.lower().endswith(".json"):
                            continue
                        try:
                            with archive.open(member) as source:
                                yield json.load(io.TextIOWrapper(
                                    source, encoding="utf-8"))
                        except Exception:
                            continue
            except (OSError, zipfile.BadZipFile):
                continue
        else:
            try:
                with open(target, encoding="utf-8") as source:
                    yield json.load(source)
            except (OSError, json.JSONDecodeError):
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--emit", default=None)
    ap.add_argument("--leaderboard", default=None)
    ap.add_argument("--min-elo", type=float, default=0.0)
    a = ap.parse_args()

    names = card_names()
    elos = load_elos(a.leaderboard)
    counts = Counter()          # deck tuple -> appearances
    wins = defaultdict(int)     # deck tuple -> wins
    n_ep = 0
    for ep in iter_episodes(a.episodes):
        try:
            steps = ep["steps"]
            rewards = ep.get("rewards") or [None, None]
            agents = (ep.get("info") or {}).get("Agents") or []
            for p in range(2):
                pilot = (agents[p].get("Name")
                         if p < len(agents) and isinstance(agents[p], dict)
                         else None)
                if a.min_elo > 0 and elos.get(pilot, -1) < a.min_elo:
                    continue
                deck = steps[1][p].get("action")
                if isinstance(deck, list) and len(deck) == 60:
                    key = tuple(sorted(deck))
                    counts[key] += 1
                    if rewards[p] == 1:
                        wins[key] += 1
            n_ep += 1
        except Exception:
            continue

    quality = (f", Elo >= {a.min_elo:g}" if a.min_elo > 0 else "")
    print(f"{n_ep} episodes, {sum(counts.values())} deck appearances"
          f"{quality}, {len(counts)} distinct exact lists\n")
    top = counts.most_common(a.top)
    lines = []
    for i, (key, n) in enumerate(top):
        wr = 100.0 * wins[key] / n
        # signature = most distinctive pokemon (highest id basic-ignored heuristic:
        # just show the 3 most common non-energy ids by name)
        sig = [names.get(c, str(c)) for c, _k in Counter(
            [c for c in key if c > 100]).most_common(3)]
        print(f"deck_{i}: {n} appearances, {wr:.0f}% win rate  |  {', '.join(sig)}")
        lines.append((f"mined_{i}", list(key), n, wr))

    if a.emit:
        with open(a.emit, "w") as f:
            f.write("# Auto-mined from ladder replays by tools/mine_decks.py\n")
            f.write("META_DECKS = {\n")
            for name, deck, n, wr in lines:
                f.write(f'    "{name}": {deck},  # seen {n}x, {wr:.0f}% WR\n')
            f.write("}\n")
            f.write("META_WEIGHT = {\n")
            for name, deck, n, wr in lines:
                f.write(f'    "{name}": {n},\n')
            f.write("}\n")
        print(f"\nwrote {a.emit}")


if __name__ == "__main__":
    main()
