"""Choose which learner deck ships with the selected policy.

The checkpoint is fixed before this runs, so the only free variable is the
deck.  Two independent probes have to agree before a deck is trusted:

*   ``mirror``  - the selected policy pilots deck A against itself piloting
    deck B.  Fast, and a direct read on which list is stronger.
*   ``field``   - the selected policy pilots each learner deck against the same
    policy piloting every mined field deck.  This is the deployment condition,
    so it is the primary criterion; results are also re-weighted by how often
    each field deck actually appeared on the ladder.

The opponent policy is the selected checkpoint in both probes because no model
of the real ladder pilots exists.  Deck strength is therefore measured under a
self-play pilot, which is a genuine limitation and is recorded in the output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import time

from .arena import Arena
from .evaluate_population import STEP_CAP_ERROR, blank_record, summarize
from .network import load_npz


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def series(
    arena: Arena,
    model,
    own_deck: list[int],
    other_deck: list[int],
    games: int,
    error_kinds: Counter,
) -> dict[str, int]:
    """Play ``own_deck`` against ``other_deck``, both piloted by ``model``."""
    record = blank_record()
    for game_index in range(games):
        own_seat = game_index % 2
        decks = (own_deck, other_deck) if own_seat == 0 else (other_deck, own_deck)
        result = arena.play((model, model), decks, sample=False)
        record["invalid_actions"] += result.invalid_actions
        if result.error:
            error_kinds[result.error] += 1
            record["step_caps" if result.error == STEP_CAP_ERROR else "engine_errors"] += 1
        elif result.winner == own_seat:
            record["wins"] += 1
        elif result.winner == 1 - own_seat:
            record["losses"] += 1
        else:
            record["draws"] += 1
    return record


def accumulate(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=os.path.join(ROOT, "rl_osfp", "run"))
    parser.add_argument("--pool", default=os.path.join(ROOT, "data", "fresh", "deck_pool.json"))
    parser.add_argument("--period", type=int, default=None,
                        help="checkpoint period to gate; defaults to --model")
    parser.add_argument("--model", default=None,
                        help="explicit checkpoint path; overrides --period")
    parser.add_argument("--field-games", type=int, default=6,
                        help="games per (learner deck, field deck) pair; a floor "
                             "when --share-budget is set")
    parser.add_argument("--share-budget", type=int, default=0,
                        help="if set, distribute this many games per learner deck "
                             "across field decks in proportion to ladder share, "
                             "never below --field-games")
    parser.add_argument("--mirror-games", type=int, default=40,
                        help="games for the direct learner-deck head to head")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.model:
        model_path = args.model
    elif args.period is not None:
        model_path = os.path.join(args.run, f"model_period_{args.period:03d}.npz")
    else:
        parser.error("pass --period or --model")
    if not os.path.isfile(model_path):
        raise SystemExit(f"missing checkpoint: {model_path}")

    with open(args.pool, encoding="utf-8") as source:
        pool = json.load(source)
    learner = pool["learner_decks"]
    field = pool["field_decks"]
    if len(learner) < 2:
        raise SystemExit("deck gate needs at least two learner decks")
    appearances = [max(1.0, float(deck.get("appearances", 1))) for deck in field]
    share = [value / sum(appearances) for value in appearances]

    # A share-weighted estimate is dominated by the highest-share matchups, so
    # uniform allocation spends most games where they barely move the answer.
    # Games proportional to share put the samples where the variance is.
    if args.share_budget:
        schedule = [max(args.field_games, round(args.share_budget * value)) for value in share]
    else:
        schedule = [args.field_games] * len(field)

    model = load_npz(model_path).eval()
    arena = Arena()
    error_kinds: Counter = Counter()
    started = time.monotonic()
    total_games = len(learner) * sum(schedule) + args.mirror_games
    print(
        f"gating {os.path.basename(model_path)}: {len(learner)} learner decks x "
        f"{len(field)} field decks ({sum(schedule)} games each, "
        f"{'share-weighted' if args.share_budget else 'uniform'}) + "
        f"{args.mirror_games} mirror games = {total_games} games",
        flush=True,
    )

    field_results = []
    for deck_index, deck in enumerate(learner):
        totals = blank_record()
        weighted_numerator = 0.0
        weighted_played = 0.0
        per_field = []
        for field_index, opponent in enumerate(field):
            record = series(arena, model, deck["cards"], opponent["cards"],
                            schedule[field_index], error_kinds)
            accumulate(totals, record)
            summary = summarize(record)
            per_field.append({
                "field_deck": opponent["name"],
                "appearances": opponent.get("appearances"),
                "share": share[field_index],
                **summary,
            })
            # Half credit for step caps keeps a stalling matchup from inflating
            # the weighted estimate the same way it would a raw win rate.
            if summary["played_games"]:
                weighted_numerator += share[field_index] * summary["score_rate"]
                weighted_played += share[field_index]
            print(
                f"  {deck['name']} vs {opponent['name']:<9} "
                f"{record['wins']}-{record['losses']} caps={record['step_caps']} "
                f"share={share[field_index] * 100:4.1f}%",
                flush=True,
            )
        summary = summarize(totals)
        field_results.append({
            "deck_index": deck_index,
            "name": deck["name"],
            "appearances": deck.get("appearances"),
            **summary,
            "appearance_weighted_score": (
                weighted_numerator / weighted_played if weighted_played else 0.0
            ),
            "per_field_deck": per_field,
        })

    mirror = series(arena, model, learner[0]["cards"], learner[1]["cards"],
                    args.mirror_games, error_kinds)
    mirror_summary = summarize(mirror)
    print(
        f"\nmirror {learner[0]['name']} vs {learner[1]['name']}: "
        f"{mirror['wins']}-{mirror['losses']} caps={mirror['step_caps']}",
        flush=True,
    )

    field_rank = sorted(field_results, key=lambda item: item["appearance_weighted_score"],
                        reverse=True)
    selected = field_rank[0]
    mirror_leader = 0 if mirror_summary["score_rate"] >= 0.5 else 1
    margin = selected["appearance_weighted_score"] - field_rank[1]["appearance_weighted_score"]
    # Two-proportion standard error on the pooled uniform field samples, used
    # only to say whether the field gap is distinguishable from noise.
    left, right = field_rank[0], field_rank[1]
    pooled = (left["wins"] + right["wins"]) / max(1, left["decided_games"] + right["decided_games"])
    standard_error = math.sqrt(
        max(pooled * (1 - pooled), 1e-9)
        * (1 / max(1, left["decided_games"]) + 1 / max(1, right["decided_games"]))
    )
    gap = left["win_rate"] - right["win_rate"]
    significant = abs(gap) > 1.96 * standard_error

    output = {
        "model": os.path.abspath(model_path),
        "period": args.period,
        "field_games_per_pair": args.field_games,
        "share_budget": args.share_budget,
        "field_schedule": {field[i]["name"]: schedule[i] for i in range(len(field))},
        "mirror_games": args.mirror_games,
        "elapsed_minutes": (time.monotonic() - started) / 60.0,
        "error_kinds": dict(error_kinds),
        "opponent_pilot": "self (selected checkpoint on both seats)",
        "field": field_results,
        "mirror": {"left": learner[0]["name"], "right": learner[1]["name"],
                   **mirror_summary, "leader_deck_index": mirror_leader},
        "selected_deck_index": selected["deck_index"],
        "selected_deck_name": selected["name"],
        "weighted_margin": margin,
        "field_gap_significant": bool(significant),
        "probes_agree": bool(mirror_leader == selected["deck_index"]),
    }
    out = args.out or os.path.join(args.run, "deck_gate.json")
    with open(out, "w", encoding="utf-8") as target:
        json.dump(output, target, indent=2)

    print("\ndeck        decided  win%   wilson  uniform-score%  weighted-score%  cap%", flush=True)
    for item in field_results:
        print(
            f"{item['name']:<11} {item['decided_games']:>7}  "
            f"{item['win_rate'] * 100:>5.1f}  {item['wilson_lower_95']:>6.3f}  "
            f"{item['score_rate'] * 100:>13.1f}  "
            f"{item['appearance_weighted_score'] * 100:>14.1f}  "
            f"{item['cap_rate'] * 100:>4.1f}",
            flush=True,
        )
    if not output["probes_agree"]:
        print(
            f"WARNING: mirror probe prefers deck {mirror_leader} but the field "
            f"probe prefers deck {selected['deck_index']}; field wins as the "
            f"deployment condition, margin is thin",
            flush=True,
        )
    if not significant:
        print(
            "WARNING: the field gap is inside noise; the deck choice is not "
            "statistically supported at this sample size",
            flush=True,
        )
    print(
        f"selected deck index {selected['deck_index']} ({selected['name']}); wrote {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
