"""Build a bounded deck population from only the newest official replays.

Replay actions are not used for model training.  They are read solely to pick
two learner decks and a diverse field of opponent decks for online RL.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os

from mine_decks import iter_episodes, load_elos


def wilson_lower(wins: int, total: int, z: float = 1.28) -> float:
    if total <= 0:
        return 0.0
    probability = wins / total
    denom = 1.0 + z * z / total
    centre = probability + z * z / (2.0 * total)
    spread = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    )
    return (centre - spread) / denom


def overlap(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    a, b = Counter(left), Counter(right)
    intersection = sum((a & b).values())
    union = sum((a | b).values())
    return intersection / union if union else 1.0


def row(name: str, cards: tuple[int, ...], count: int, wins: int) -> dict:
    return {
        "name": name,
        "cards": list(cards),
        "appearances": count,
        "wins": wins,
        "win_rate": wins / count if count else 0.0,
        "wilson_lower_80": wilson_lower(wins, count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes")
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-elo", type=float, default=900.0)
    parser.add_argument("--min-appearances", type=int, default=8)
    parser.add_argument("--field-size", type=int, default=16)
    args = parser.parse_args()

    elos = load_elos(args.leaderboard)
    counts: Counter[tuple[int, ...]] = Counter()
    wins: defaultdict[tuple[int, ...], int] = defaultdict(int)
    pilots: defaultdict[tuple[int, ...], Counter[str]] = defaultdict(Counter)
    episodes = eligible = matched_pilots = 0
    for episode in iter_episodes(args.episodes):
        episodes += 1
        try:
            steps = episode["steps"]
            rewards = episode.get("rewards") or [None, None]
            agents = (episode.get("info") or {}).get("Agents") or []
            for seat in range(2):
                agent = agents[seat] if seat < len(agents) else {}
                pilot = agent.get("Name") if isinstance(agent, dict) else None
                elo = elos.get(pilot, -1.0)
                if elo < args.min_elo:
                    continue
                matched_pilots += 1
                deck = steps[1][seat].get("action")
                if not isinstance(deck, list) or len(deck) != 60 or not all(
                    isinstance(card, int) for card in deck
                ):
                    continue
                key = tuple(sorted(deck))
                eligible += 1
                counts[key] += 1
                if rewards[seat] == 1:
                    wins[key] += 1
                pilots[key][pilot or "unknown"] += 1
        except Exception:
            continue

    if not counts:
        raise SystemExit(
            "no leaderboard-matched decks found; verify the current leaderboard snapshot"
        )
    candidates = [
        deck for deck, count in counts.items() if count >= args.min_appearances
    ]
    if len(candidates) < 2:
        candidates = [deck for deck, _count in counts.most_common(max(2, len(counts)))]
    candidates.sort(
        key=lambda deck: (wilson_lower(wins[deck], counts[deck]), counts[deck]),
        reverse=True,
    )
    primary = candidates[0]
    secondary = next(
        (deck for deck in candidates[1:] if overlap(primary, deck) < 0.78),
        candidates[1],
    )
    selected = [primary, secondary]
    learner = [
        row(f"learner_{index}", deck, counts[deck], wins[deck])
        for index, deck in enumerate(selected)
    ]
    for item, deck in zip(learner, selected):
        item["top_pilots"] = pilots[deck].most_common(3)

    field_keys = [deck for deck, _count in counts.most_common(args.field_size)]
    for deck in selected:
        if deck not in field_keys:
            field_keys.append(deck)
    field = [
        row(f"field_{index}", deck, counts[deck], wins[deck])
        for index, deck in enumerate(field_keys)
    ]
    source_files = []
    for directory, _subdirs, files in os.walk(args.episodes):
        for filename in sorted(files):
            path = os.path.join(directory, filename)
            source_files.append({
                "name": os.path.relpath(path, args.episodes),
                "bytes": os.path.getsize(path),
            })
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "episodes": source_files,
            "leaderboard": os.path.basename(args.leaderboard),
            "min_elo": args.min_elo,
            "episodes_scanned": episodes,
            "eligible_deck_appearances": eligible,
            "matched_pilot_seats": matched_pilots,
            "distinct_decks": len(counts),
            "selection": "80% Wilson lower bound; secondary exact list requires <0.78 multiset Jaccard",
        },
        "learner_decks": learner,
        "field_decks": field,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2)
    print(
        f"scanned {episodes} episodes; {eligible} eligible seats; "
        f"{len(counts)} exact decks; wrote {args.out}"
    )
    for item in learner:
        print(
            f"{item['name']}: n={item['appearances']} "
            f"wr={item['win_rate']:.3f} lower80={item['wilson_lower_80']:.3f} "
            f"pilots={item['top_pilots']}"
        )


if __name__ == "__main__":
    main()
