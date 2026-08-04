"""Choose the submission deck using *search* as the pilot.

The earlier deck gate measured decks under the search-free policy and picked
``field_1``.  That pilot is not what ships.  Search is a far stronger player, and
deck strength is pilot-dependent: the previous gate's whole point was that a
deck experts win 75% with was unplayable by a weak policy.  A strong pilot can
plausibly swing the answer back.

Ladder history says this is worth measuring rather than assuming - deck swaps
alone moved past submissions 967.1 vs 917.6 (same model, different deck) and
972.0 vs 719.2.

The opponent is held fixed at the heuristic policy across every candidate, so
only the deck varies and one side searching keeps the cost affordable.  Field
decks are sampled by ladder appearance share, because a uniform average badly
misstates expected ladder performance when one deck is 40% of the field.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

from foundation.cg import game

from .search import (Engine, OpponentModel, heuristic_action, search_move)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson_lower(wins: int, games: int, z: float = 1.96) -> float:
    if games <= 0:
        return 0.0
    p = wins / games
    denominator = 1 + z * z / games
    centre = p + z * z / (2 * games)
    spread = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games))
    return (centre - spread) / denominator


def fallback(select: dict) -> list[int]:
    options = select.get("option") or []
    if not options:
        return []
    low = max(0, min(int(select.get("minCount", 1) or 0), len(options)))
    high = max(low, min(int(select.get("maxCount", max(low, 1)) or 0), len(options)))
    return list(range(low if low else min(1, high)))


def play(engine, attack_db, decks, search_seat, opponent, rng, per_move, worlds, stats):
    observation, start = game.battle_start(decks[0], decks[1])
    if observation is None:
        return None, f"BattleStart error {start.error}"
    try:
        for _ in range(2400):
            current = observation.get("current") or {}
            if int(current.get("result", -1)) >= 0:
                return int(current["result"]), None
            select = observation.get("select") or {}
            seat = int(current.get("yourIndex", 0))
            if seat == search_seat:
                begin = time.perf_counter()
                action = search_move(
                    engine, attack_db, observation, seat, decks[seat], opponent,
                    rng, deadline=begin + per_move, worlds=worlds,
                )
                if action is None:
                    stats["fallbacks"] += 1
                    action = heuristic_action(engine, attack_db, select, rng)
                else:
                    stats["searched"] += 1
            else:
                action = heuristic_action(engine, attack_db, select, rng)
            try:
                observation = game.battle_select(action or fallback(select))
            except (IndexError, ValueError):
                stats["invalid"] += 1
                observation = game.battle_select(fallback(select))
        return None, "step cap reached"
    finally:
        try:
            game.battle_finish()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=os.path.join(ROOT, "data", "fresh", "deck_pool_elo1000.json"))
    parser.add_argument("--candidates", default="field_1,field_16,field_7,field_3",
                        help="deck names to gate; field_16 is the rank-1 pilot's list")
    parser.add_argument("--games", type=int, default=30, help="games per candidate")
    parser.add_argument("--per-move-seconds", type=float, default=0.25)
    parser.add_argument("--worlds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.pool, encoding="utf-8") as source:
        pool = json.load(source)
    field = pool["field_decks"]
    by_name = {deck["name"]: deck for deck in field}
    names = [n.strip() for n in args.candidates.split(",") if n.strip()]
    for name in names:
        if name not in by_name:
            raise SystemExit(f"unknown deck {name}; have {sorted(by_name)}")
    meta = [deck["cards"] for deck in field]
    shares = [float(deck.get("appearances", 1)) for deck in field]
    total = sum(shares)
    shares = [s / total for s in shares]

    engine = Engine()
    attack_db = {
        int(a["attackId"]): a for a in json.loads(engine.lib.AllAttack().decode())
    }
    opponent = OpponentModel(meta, [float(d.get("appearances", 1)) for d in field])

    results = []
    started = time.monotonic()
    for name in names:
        own = by_name[name]["cards"]
        rng = random.Random(args.seed)          # identical field draw per candidate
        stats = {"searched": 0, "fallbacks": 0, "invalid": 0}
        wins = losses = caps = 0
        for index in range(args.games):
            search_seat = index % 2
            other = rng.choices(meta, weights=shares, k=1)[0]
            decks = (own, other) if search_seat == 0 else (other, own)
            winner, error = play(engine, attack_db, decks, search_seat, opponent,
                                 rng, args.per_move_seconds, args.worlds, stats)
            if error:
                caps += 1
            elif winner == search_seat:
                wins += 1
            elif winner is not None:
                losses += 1
        decided = wins + losses
        row = {
            "deck": name, "wins": wins, "losses": losses, "step_caps": caps,
            "decided": decided,
            "win_rate": wins / decided if decided else 0.0,
            "wilson_lower_95": wilson_lower(wins, decided),
            "ladder_win_rate": by_name[name].get("win_rate"),
            "appearances": by_name[name].get("appearances"),
            **stats,
        }
        results.append(row)
        print(
            f"{name:<9} {wins}-{losses} caps={caps} "
            f"win={row['win_rate'] * 100:.1f}% wilson={row['wilson_lower_95']:.3f} "
            f"({(time.monotonic() - started) / 60:.1f} min)",
            flush=True,
        )

    results.sort(key=lambda r: (r["wilson_lower_95"], r["win_rate"]), reverse=True)
    output = {
        "pilot": "determinized search", "opponent_pilot": "heuristic",
        "games_per_candidate": args.games,
        "per_move_seconds": args.per_move_seconds,
        "elapsed_minutes": (time.monotonic() - started) / 60.0,
        "ranking": results, "selected_deck": results[0]["deck"],
    }
    out = args.out or os.path.join(ROOT, "rl_osfp", "run", "search_deck_gate.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as target:
        json.dump(output, target, indent=2)
    print(f"\nselected {results[0]['deck']}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
