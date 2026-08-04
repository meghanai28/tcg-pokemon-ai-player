"""Head-to-head gate: determinized search against the search-free policy.

Search is only worth its complexity if it converts idle inference budget into
wins, so this plays complete games with search on one seat and the network
policy on the other, alternating seats, and reports the record with a Wilson
bound.  It also records how many decisions actually reached the search (rather
than a fallback), because a silent fallback would look exactly like "search
does not help".
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time

from foundation.cg import game

from .network import load_npz
from .policy import choose_action
from .search import (Engine, OpponentModel, PolicyGuide, heuristic_action, search_move)


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


def play_game(engine, attack_db, card_db, model, decks, search_seat, opponent,
              rng, per_move_seconds, worlds, stats, max_steps=2400, guide=None,
              opponent_mode="policy"):
    observation, start = game.battle_start(decks[0], decks[1])
    if observation is None:
        return None, f"BattleStart error {start.error}"
    try:
        for step in range(max_steps):
            current = observation.get("current") or {}
            if int(current.get("result", -1)) >= 0:
                return int(current["result"]), None
            select = observation.get("select") or {}
            seat = int(current.get("yourIndex", 0))
            if seat == search_seat:
                begin = time.perf_counter()
                action = search_move(
                    engine, attack_db, observation, seat, decks[seat], opponent,
                    rng, deadline=begin + per_move_seconds, worlds=worlds,
                    guide=guide,
                )
                stats["latencies"].append(time.perf_counter() - begin)
                if action is None:
                    stats["fallbacks"] += 1
                    action = heuristic_action(engine, attack_db, select, rng) or fallback(select)
                else:
                    stats["searched"] += 1
            elif opponent_mode == "search":
                # Unguided search on the other seat: the only difference between
                # the seats is then the PPO guidance itself.
                begin = time.perf_counter()
                action = search_move(
                    engine, attack_db, observation, seat, decks[seat], opponent,
                    rng, deadline=begin + per_move_seconds, worlds=worlds, guide=None,
                )
                if action is None:
                    action = heuristic_action(engine, attack_db, select, rng) or fallback(select)
            else:
                action, _ = choose_action(
                    model, observation, decks[seat], card_db, attack_db, sample=False,
                )
            try:
                observation = game.battle_select(action)
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool", default=os.path.join(ROOT, "data", "fresh", "deck_pool_elo1000.json"))
    parser.add_argument("--deck", default="field_1", help="deck both sides pilot")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--per-move-seconds", type=float, default=1.0)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--guide-model", default=None,
                        help="checkpoint supplying PPO priors/value to the search")
    parser.add_argument("--opponent-mode", choices=("policy", "search"), default="policy",
                        help="'search' pits guided search against unguided search")
    parser.add_argument("--prior-weight", type=float, default=0.7)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.pool, encoding="utf-8") as source:
        pool = json.load(source)
    field = pool["field_decks"]
    by_name = {deck["name"]: deck for deck in field}
    if args.deck not in by_name:
        raise SystemExit(f"unknown deck {args.deck}; have {sorted(by_name)}")
    own = by_name[args.deck]["cards"]
    meta = [deck["cards"] for deck in field]
    weights = [float(deck.get("appearances", 1)) for deck in field]

    engine = Engine()
    attack_db = {
        int(attack["attackId"]): attack
        for attack in json.loads(engine.lib.AllAttack().decode())
    }
    model = load_npz(args.model).eval()
    rng = random.Random(args.seed)
    opponent = OpponentModel(meta, weights)

    guide = None
    if args.guide_model:
        guide = PolicyGuide(args.guide_model, engine.card_db, attack_db,
                            prior_weight=args.prior_weight,
                            value_weight=args.value_weight)
        print(f"guided by {os.path.basename(args.guide_model)} "
              f"(prior_weight={args.prior_weight}, value_weight={args.value_weight})",
              flush=True)
    stats = {"searched": 0, "fallbacks": 0, "invalid": 0, "latencies": []}
    wins = losses = caps = errors = 0
    started = time.monotonic()
    for game_index in range(args.games):
        search_seat = game_index % 2
        # Same deck on both seats: only the decision procedure differs.
        opponent_deck = meta[game_index % len(meta)]
        decks = (own, opponent_deck) if search_seat == 0 else (opponent_deck, own)
        winner, error = play_game(
            engine, attack_db, engine.card_db, model, decks, search_seat,
            opponent, rng, args.per_move_seconds, args.worlds, stats,
            guide=guide, opponent_mode=args.opponent_mode,
        )
        if error == "step cap reached":
            caps += 1
        elif error:
            errors += 1
        elif winner == search_seat:
            wins += 1
        elif winner is not None:
            losses += 1
        print(
            f"game {game_index + 1}/{args.games}: search_seat={search_seat} "
            f"winner={winner} error={error} record={wins}-{losses} "
            f"({(time.monotonic() - started) / 60:.1f} min)",
            flush=True,
        )

    decided = wins + losses
    latencies = stats["latencies"]
    report = {
        "model": os.path.abspath(args.model), "deck": args.deck,
        "games": args.games, "per_move_seconds": args.per_move_seconds,
        "worlds": args.worlds,
        "guide_model": args.guide_model,
        "prior_weight": args.prior_weight if guide else None,
        "value_weight": args.value_weight if guide else None,
        "guide_calls": guide.calls if guide else 0,
        "opponent_mode": args.opponent_mode,
        "search_wins": wins, "search_losses": losses,
        "step_caps": caps, "errors": errors,
        "win_rate": wins / decided if decided else 0.0,
        "wilson_lower_95": wilson_lower(wins, decided),
        "searched_decisions": stats["searched"],
        "fallback_decisions": stats["fallbacks"],
        "invalid_actions": stats["invalid"],
        "mean_move_seconds": statistics.mean(latencies) if latencies else 0.0,
        "elapsed_minutes": (time.monotonic() - started) / 60.0,
    }
    out = args.out or os.path.join(ROOT, "rl_osfp", "run", "search_gate.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
    print(
        f"\nsearch {wins}-{losses} (caps={caps} errors={errors})  "
        f"win_rate={report['win_rate'] * 100:.1f}%  "
        f"wilson={report['wilson_lower_95']:.3f}\n"
        f"searched={stats['searched']} fallbacks={stats['fallbacks']} "
        f"invalid={stats['invalid']} mean_move={report['mean_move_seconds']:.2f}s\n"
        f"wrote {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
