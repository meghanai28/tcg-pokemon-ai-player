"""Score checkpoints against each other on identical decks to isolate policy strength.

Two selection stages share this code path:

*   ``--panel`` set: every candidate plays the same fixed opponent panel at
    O(candidates) cost.  This is a *screen* whose job is to surface candidates
    worth a runoff, not to decide.  A candidate that is itself a panel member
    skips its own matchup and therefore faces a strictly easier schedule than
    the rest, so panel members' scores are inflated and cross-candidate
    comparison is only sound among non-panel candidates.
*   ``--panel`` unset: full round robin among the candidates.  Every schedule
    is symmetric, so this is the mode that decides.  Run it on the survivors of
    the screen with a high ``--games-per-deck``.

These two modes disagreed on this project's first clean run - the screen
preferred period 9 and ranked period 4 fourth, the runoff put period 4 clearly
first - for exactly the self-skip reason above.  Trust the runoff.

Both sides always pilot the *same* deck in a given game, so deck strength
cancels and only the policy is measured.  Games that reach the arena step cap
are recorded separately from genuine engine faults: a stalling policy must not
be rewarded by having its stalls deleted from the denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
import math
import os
import re
import time

from .arena import Arena
from .network import load_npz


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP_CAP_ERROR = "step cap reached"


def wilson_lower(wins: int, games: int, z: float = 1.96) -> float:
    if games <= 0:
        return 0.0
    p = wins / games
    denominator = 1 + z * z / games
    centre = p + z * z / (2 * games)
    spread = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games))
    return (centre - spread) / denominator


def blank_record() -> dict[str, int]:
    return {"wins": 0, "losses": 0, "draws": 0, "step_caps": 0,
            "engine_errors": 0, "invalid_actions": 0}


def discover_periods(run_dir: str) -> list[int]:
    found = []
    for path in glob.glob(os.path.join(run_dir, "model_period_*.npz")):
        match = re.search(r"model_period_(\d+)\.npz$", path)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def parse_periods(value: str, run_dir: str) -> list[int]:
    if value.strip().lower() == "all":
        return discover_periods(run_dir)
    return [int(item) for item in value.split(",") if item.strip()]


def summarize(record: dict[str, int]) -> dict[str, float | int]:
    decided = record["wins"] + record["losses"]
    played = decided + record["draws"] + record["step_caps"]
    return {
        **record,
        "decided_games": decided,
        "played_games": played,
        "win_rate": record["wins"] / decided if decided else 0.0,
        "wilson_lower_95": wilson_lower(record["wins"], decided),
        # Caps counted as half points, so a policy cannot climb by stalling.
        "score_rate": (
            (record["wins"] + 0.5 * (record["draws"] + record["step_caps"])) / played
            if played else 0.0
        ),
        "cap_rate": record["step_caps"] / played if played else 0.0,
    }


def play_series(
    arena: Arena,
    models: dict[int, object],
    left: int,
    right: int,
    decks: list[list[int]],
    games_per_deck: int,
    error_kinds: Counter,
) -> dict:
    """Play ``left`` against ``right`` on every deck, alternating seats.

    The returned row is written from ``left``'s point of view; ``credit_row``
    flips it for ``right``.
    """
    row = blank_record()
    row.update({"left": left, "right": right, "wins_left": 0, "wins_right": 0})
    for deck_index, deck in enumerate(decks):
        for game_index in range(games_per_deck):
            left_seat = (game_index + deck_index) % 2
            pair = (models[left], models[right]) if left_seat == 0 else (models[right], models[left])
            result = arena.play(pair, (deck, deck), sample=False)
            row["invalid_actions"] += result.invalid_actions
            if result.error:
                error_kinds[result.error] += 1
                row["step_caps" if result.error == STEP_CAP_ERROR else "engine_errors"] += 1
            elif result.winner == left_seat:
                row["wins_left"] += 1
            elif result.winner == 1 - left_seat:
                row["wins_right"] += 1
            else:
                row["draws"] += 1
    row["wins"] = row["wins_left"]
    row["losses"] = row["wins_right"]
    return row


def credit_row(totals: dict[int, dict[str, int]], row: dict, *, flip: bool) -> None:
    """Add one matchup row to a period's running totals."""
    period = row["right"] if flip else row["left"]
    target = totals[period]
    target["wins"] += row["wins_right"] if flip else row["wins_left"]
    target["losses"] += row["wins_left"] if flip else row["wins_right"]
    for key in ("draws", "step_caps", "engine_errors", "invalid_actions"):
        target[key] += row[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=os.path.join(ROOT, "rl_osfp", "run"))
    parser.add_argument("--pool", default=os.path.join(ROOT, "data", "fresh", "deck_pool.json"))
    parser.add_argument("--periods", default="all",
                        help="candidate periods, or 'all' to discover every checkpoint")
    parser.add_argument("--panel", default="",
                        help="fixed opponent periods; empty means round robin among candidates")
    parser.add_argument("--games-per-deck", type=int, default=6)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    periods = parse_periods(args.periods, args.run)
    if len(periods) < 2:
        raise SystemExit(f"need at least two checkpoints, found {periods}")
    panel = parse_periods(args.panel, args.run) if args.panel.strip() else []
    with open(args.pool, encoding="utf-8") as source:
        pool = json.load(source)
    decks = [item["cards"] for item in pool["learner_decks"]]

    needed = sorted(set(periods) | set(panel))
    paths = {
        period: os.path.join(args.run, f"model_period_{period:03d}.npz")
        for period in needed
    }
    missing = [path for path in paths.values() if not os.path.isfile(path)]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    models = {period: load_npz(path).eval() for period, path in paths.items()}

    if panel:
        pairings = [(candidate, opponent) for candidate in periods
                    for opponent in panel if candidate != opponent]
        mode = "gauntlet"
    else:
        pairings = [(left, right) for index, left in enumerate(periods)
                    for right in periods[index + 1:]]
        mode = "round-robin"
    total_games = len(pairings) * len(decks) * args.games_per_deck
    print(
        f"mode={mode} candidates={periods} panel={panel or 'n/a'} "
        f"pairings={len(pairings)} decks={len(decks)} games={total_games}",
        flush=True,
    )

    arena = Arena()
    totals: dict[int, dict[str, int]] = defaultdict(blank_record)
    error_kinds: Counter = Counter()
    matchups = []
    started = time.monotonic()
    for index, (left, right) in enumerate(pairings, start=1):
        row = play_series(arena, models, left, right, decks,
                          args.games_per_deck, error_kinds)
        matchups.append(row)
        # Gauntlet candidates are always the left side. A panel member must not
        # also bank the games it played as an opponent: it faces the whole
        # candidate field there, which is a different and generally weaker
        # schedule than the panel every candidate is scored against.
        credit_row(totals, row, flip=False)
        if mode == "round-robin":
            credit_row(totals, row, flip=True)
        elapsed = time.monotonic() - started
        print(
            f"[{index}/{len(pairings)}] {left:>3} vs {right:<3} "
            f"{row['wins_left']}-{row['wins_right']} caps={row['step_caps']} "
            f"errors={row['engine_errors']} invalid={row['invalid_actions']} "
            f"({elapsed / 60:.1f} min)",
            flush=True,
        )

    ranking = [{"period": period, **summarize(totals[period])} for period in periods]
    ranking.sort(key=lambda item: (item["wilson_lower_95"], item["win_rate"]), reverse=True)

    by_score = sorted(ranking, key=lambda item: item["score_rate"], reverse=True)
    agrees = by_score[0]["period"] == ranking[0]["period"]
    output = {
        "mode": mode,
        "candidates": periods,
        "panel": panel,
        "games_per_deck": args.games_per_deck,
        "total_games": total_games,
        "elapsed_minutes": (time.monotonic() - started) / 60.0,
        "error_kinds": dict(error_kinds),
        "matchups": matchups,
        "ranking": ranking,
        "selected_period": ranking[0]["period"],
        "cap_inclusive_leader": by_score[0]["period"],
        "criteria_agree": agrees,
    }
    out = args.out or os.path.join(args.run, f"population_eval_{mode}.json")
    with open(out, "w", encoding="utf-8") as target:
        json.dump(output, target, indent=2)

    print("\nperiod  decided  win%   wilson  score%  cap%", flush=True)
    for item in ranking:
        print(
            f"{item['period']:>6}  {item['decided_games']:>7}  "
            f"{item['win_rate'] * 100:>5.1f}  {item['wilson_lower_95']:>6.3f}  "
            f"{item['score_rate'] * 100:>6.1f}  {item['cap_rate'] * 100:>4.1f}",
            flush=True,
        )
    print(f"\nerror kinds: {dict(error_kinds)}", flush=True)
    if not agrees:
        print(
            f"WARNING: cap-inclusive scoring prefers period {by_score[0]['period']}, "
            f"not {ranking[0]['period']}; treat the pick as unresolved",
            flush=True,
        )
    print(f"selected period {ranking[0]['period']}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
