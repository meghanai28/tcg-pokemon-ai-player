"""Explain a team's public results by submitted deck and opponent archetype.

This uses already-downloaded official daily episodes, so it remains useful when
the live API is rate-limited. It identifies our seat by team name, identifies
known submitted decks by exact 60-card multiset, and assigns the opponent to the
nearest current field list by multiset Jaccard.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from mine_decks import iter_episodes


def deck_key(cards) -> tuple[int, ...] | None:
    if not isinstance(cards, list) or len(cards) != 60:
        return None
    return tuple(sorted(int(card) for card in cards))


def overlap(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    a, b = Counter(left), Counter(right)
    return sum((a & b).values()) / max(1, sum((a | b).values()))


def load_known(items: list[str]) -> dict[tuple[int, ...], str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--own-deck needs label=csv, got {item!r}")
        label, filename = item.split("=", 1)
        cards = [int(line) for line in Path(filename).read_text().splitlines() if line.strip()]
        key = deck_key(cards)
        if key is None:
            raise SystemExit(f"{filename} is not a 60-card deck")
        result[key] = label
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes")
    parser.add_argument("--team", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--own-deck", action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    known = load_known(args.own_deck)
    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    field = [(item["name"], tuple(sorted(item["cards"])))
             for item in pool.get("field_decks", [])]
    rows = []
    scanned = 0
    for episode in iter_episodes(args.episodes):
        scanned += 1
        agents = (episode.get("info") or {}).get("Agents") or []
        seats = [index for index, agent in enumerate(agents)
                 if isinstance(agent, dict) and agent.get("Name") == args.team]
        if not seats:
            continue
        steps = episode.get("steps") or []
        rewards = episode.get("rewards") or []
        if len(steps) < 2:
            continue
        for seat in seats:
            own = deck_key((steps[1][seat] or {}).get("action"))
            other = deck_key((steps[1][1 - seat] or {}).get("action"))
            if own is None or other is None:
                continue
            own_label = known.get(own, f"unknown_{hash(own) & 0xffff:04x}")
            nearest, similarity = max(
                ((name, overlap(other, cards)) for name, cards in field),
                key=lambda item: item[1],
                default=("unknown", 0.0),
            )
            reward = rewards[seat] if seat < len(rewards) else None
            rows.append({
                "episode_id": episode.get("id"),
                "own_deck": own_label,
                "opponent_deck": nearest,
                "opponent_similarity": similarity,
                "reward": reward,
                "result": "win" if reward == 1 else ("loss" if reward == -1 else "draw"),
                "steps": len(steps),
            })

    summary = defaultdict(lambda: {"games": 0, "wins": 0, "losses": 0,
                                   "draws": 0, "steps": 0})
    matchups = defaultdict(lambda: {"games": 0, "wins": 0, "losses": 0,
                                    "draws": 0})
    result_bucket = {"win": "wins", "loss": "losses", "draw": "draws"}
    for row in rows:
        own = summary[row["own_deck"]]
        own["games"] += 1
        own[result_bucket[row["result"]]] += 1
        own["steps"] += row["steps"]
        match = matchups[(row["own_deck"], row["opponent_deck"])]
        match["games"] += 1
        match[result_bucket[row["result"]]] += 1

    report = {
        "source": str(Path(args.episodes).resolve()),
        "team": args.team,
        "episodes_scanned": scanned,
        "team_games": len(rows),
        "by_own_deck": {
            label: {**value, "mean_steps": value["steps"] / value["games"]}
            for label, value in sorted(summary.items())
        },
        "matchups": [
            {"own_deck": own, "opponent_deck": opponent, **value}
            for (own, opponent), value in sorted(
                matchups.items(), key=lambda item: (-item[1]["games"], item[0])
            )
        ],
        "games": rows,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
