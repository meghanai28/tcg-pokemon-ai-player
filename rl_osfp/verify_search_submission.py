"""Prove a packaged search submission actually searches.

A search agent fails quietly.  Every failure path - a rejected determinization,
a missing engine, an exhausted time budget - falls back to a legal heuristic
action, so a completely broken search still produces a well-formed archive that
plays legal games to completion.  Checking "it ran without raising" proves
nothing.

This unpacks the archive, imports the packaged ``main.py`` the way the Kaggle
runner does, plays full games against a heuristic baseline, and additionally
asserts that per-decision latency is consistent with search actually running
and that the whole game stays inside the episode budget.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import tarfile
import tempfile
import time

from foundation.cg import game
from foundation.cg.engine import get_lib


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_agent(staging: str):
    if staging not in sys.path:
        sys.path.insert(0, staging)
    spec = importlib.util.spec_from_file_location(
        "packaged_search_main", os.path.join(staging, "main.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["packaged_search_main"] = module
    spec.loader.exec_module(module)
    return module


def fallback(select: dict) -> list[int]:
    options = select.get("option") or []
    if not options:
        return []
    low = max(0, min(int(select.get("minCount", 1) or 0), len(options)))
    high = max(low, min(int(select.get("maxCount", max(low, 1)) or 0), len(options)))
    return list(range(low if low else min(1, high)))


def play(module, decks, agent_seat, stats, max_steps=2400):
    observation, start = game.battle_start(decks[0], decks[1])
    if observation is None:
        return None, f"BattleStart error {start.error}"
    game_started = time.perf_counter()
    try:
        for _step in range(max_steps):
            current = observation.get("current") or {}
            if int(current.get("result", -1)) >= 0:
                stats["game_seconds"].append(time.perf_counter() - game_started)
                return int(current["result"]), None
            select = observation.get("select") or {}
            seat = int(current.get("yourIndex", 0))
            if seat == agent_seat:
                begin = time.perf_counter()
                action = module.agent(observation)
                stats["latencies"].append(time.perf_counter() - begin)
                stats["decisions"] += 1
            else:
                action = fallback(select)
            try:
                observation = game.battle_select(action)
            except (IndexError, ValueError):
                if seat == agent_seat:
                    stats["invalid"] += 1
                observation = game.battle_select(fallback(select))
        stats["game_seconds"].append(time.perf_counter() - game_started)
        return None, "step cap reached"
    finally:
        try:
            game.battle_finish()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--pool", default=os.path.join(ROOT, "data", "fresh", "deck_pool_elo1000.json"))
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--episode-budget", type=float, default=600.0)
    parser.add_argument("--min-mean-latency", type=float, default=0.01,
                        help="below this the agent is not searching at all")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.pool, encoding="utf-8") as source:
        pool = json.load(source)
    field = [deck["cards"] for deck in pool["field_decks"]]

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="verify-search-") as staging:
        with tarfile.open(args.archive, "r:gz") as archive:
            names = set(archive.getnames())
            archive.extractall(staging, filter="data")
        for required in ("main.py", "deck.csv", "cg/libcg.so"):
            if required not in names:
                failures.append(f"archive is missing {required}")

        with open(os.path.join(staging, "deck.csv"), encoding="utf-8") as source:
            packaged_deck = [int(line) for line in source if line.strip()]
        if len(packaged_deck) != 60:
            failures.append(f"deck.csv has {len(packaged_deck)} cards, expected 60")

        module = load_agent(staging)
        submitted = module.agent({"select": None})
        if list(submitted) != packaged_deck:
            failures.append("agent did not return its packaged deck on the deck callback")

        # Engine must be loadable for the search API to exist at all.
        get_lib()
        stats = {"decisions": 0, "invalid": 0, "latencies": [], "game_seconds": []}
        outcomes = {"wins": 0, "losses": 0, "step_caps": 0, "errors": 0}
        for game_index in range(args.games):
            agent_seat = game_index % 2
            other = field[game_index % len(field)]
            decks = (packaged_deck, other) if agent_seat == 0 else (other, packaged_deck)
            winner, error = play(module, decks, agent_seat, stats)
            if error == "step cap reached":
                outcomes["step_caps"] += 1
            elif error:
                outcomes["errors"] += 1
                failures.append(f"game {game_index} failed: {error}")
            elif winner == agent_seat:
                outcomes["wins"] += 1
            elif winner is not None:
                outcomes["losses"] += 1
            print(
                f"  game {game_index + 1}/{args.games}: seat={agent_seat} "
                f"winner={winner} error={error} decisions={stats['decisions']}",
                flush=True,
            )

    latencies = stats["latencies"]
    mean_latency = statistics.mean(latencies) if latencies else 0.0
    worst_game = max(stats["game_seconds"], default=0.0)
    report = {
        "archive": os.path.abspath(args.archive),
        "games": args.games, "outcomes": outcomes,
        "decisions": stats["decisions"], "invalid_actions": stats["invalid"],
        "mean_latency_s": mean_latency,
        "max_latency_s": max(latencies, default=0.0),
        "worst_game_seconds": worst_game,
        "episode_budget_s": args.episode_budget,
    }
    if stats["invalid"]:
        failures.append(f"agent produced {stats['invalid']} engine-rejected actions")
    if not stats["decisions"]:
        failures.append("agent was never asked to decide; the check proved nothing")
    if mean_latency < args.min_mean_latency:
        failures.append(
            f"mean latency {mean_latency * 1000:.1f} ms is below the "
            f"{args.min_mean_latency * 1000:.0f} ms floor: the agent is "
            f"falling back, not searching"
        )
    if worst_game > args.episode_budget:
        failures.append(
            f"worst game took {worst_game:.0f}s, over the "
            f"{args.episode_budget:.0f}s episode budget"
        )
    report["failures"] = failures
    report["passed"] = not failures

    out = args.out or os.path.join(
        ROOT, "artifacts",
        os.path.basename(args.archive).replace(".tar.gz", "_verification.json"),
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as target:
        json.dump(report, target, indent=2)

    print(
        f"\nrecord {outcomes['wins']}-{outcomes['losses']} "
        f"caps={outcomes['step_caps']} errors={outcomes['errors']}\n"
        f"decisions={stats['decisions']} invalid={stats['invalid']}\n"
        f"latency mean={mean_latency * 1000:.0f} ms max={report['max_latency_s'] * 1000:.0f} ms\n"
        f"worst game {worst_game:.1f}s of {args.episode_budget:.0f}s budget",
        flush=True,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", flush=True)
        raise SystemExit(f"verification failed; wrote {out}")
    print(f"verification passed; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
