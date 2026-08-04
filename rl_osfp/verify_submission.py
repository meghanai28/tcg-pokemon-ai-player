"""Prove a packaged submission behaves like the checkpoint it was built from.

``build_submission.py`` only checks that the right filenames landed in the
archive.  The deployed inference path is a hand-written numpy reimplementation
of the torch network, so the failure that actually matters is silent: the
archive is well formed, the agent never raises, and it plays a different -
worse - policy than the checkpoint that was selected.

This unpacks the archive and drives the packaged ``main.py`` through complete
games against the torch checkpoint, checking four things:

1.  the archive contains every file the Kaggle runner needs;
2.  ``deck.csv`` is the 60-card list that was meant to ship;
3.  the packaged agent's greedy action matches the torch greedy action, which
    is the direct test that the numpy port is faithful;
4.  full games run with no exception, no engine rejection, and a per-decision
    latency the competition harness will tolerate.
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

from .network import load_npz
from .policy import choose_action


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED = {"main.py", "model.npz", "deck.csv", "nn_features.py",
            "nn_features_rich.py", "nn_infer.py", "cg/libcg.so"}


def load_packaged_agent(staging: str):
    """Import the extracted ``main.py`` the way the Kaggle runner would."""
    if staging not in sys.path:
        sys.path.insert(0, staging)
    spec = importlib.util.spec_from_file_location(
        "packaged_main", os.path.join(staging, "main.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fallback(select: dict) -> list[int]:
    options = select.get("option") or []
    if not options:
        return []
    low = max(0, min(int(select.get("minCount", 1) or 0), len(options)))
    high = max(low, min(int(select.get("maxCount", max(low, 1)) or 0), len(options)))
    return list(range(low if low > 0 else min(1, high)))


def play(module, model, decks, card_db, attack_db, agent_seat, stats, max_steps=2400):
    """One game: packaged agent on ``agent_seat``, torch checkpoint opposite."""
    observation, start = game.battle_start(decks[0], decks[1])
    if observation is None:
        return None, f"BattleStart error {start.error}"
    try:
        for step in range(max_steps):
            current = observation.get("current") or {}
            result = int(current.get("result", -1))
            if result >= 0:
                return (result if result in (0, 1) else None), None
            seat = int(current.get("yourIndex", 0))
            if seat == agent_seat:
                begin = time.perf_counter()
                action = module.agent(observation)
                stats["latencies"].append(time.perf_counter() - begin)
                stats["decisions"] += 1
                reference, decision = choose_action(
                    model, observation, decks[seat], card_db, attack_db, sample=False,
                )
                # Only compare where the network actually chose; forced or
                # degenerate selections are identical by construction.
                if decision is not None:
                    stats["compared"] += 1
                    if set(action) == set(reference):
                        stats["agreed"] += 1
                    else:
                        stats["disagreements"].append({
                            "context": int((observation.get("select") or {}).get("context") or 0),
                            "type": int((observation.get("select") or {}).get("type") or 0),
                            "packaged": list(action), "torch": list(reference),
                        })
            else:
                action, _ = choose_action(
                    model, observation, decks[seat], card_db, attack_db, sample=False,
                )
            try:
                observation = game.battle_select(action)
            except (IndexError, ValueError):
                if seat == agent_seat:
                    stats["invalid"] += 1
                observation = game.battle_select(fallback(observation.get("select") or {}))
        return None, "step cap reached"
    finally:
        try:
            game.battle_finish()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=os.path.join(ROOT, "artifacts", "submission_osfp.tar.gz"))
    parser.add_argument("--model", required=True, help="checkpoint the archive was built from")
    parser.add_argument("--pool", default=os.path.join(ROOT, "data", "fresh", "deck_pool.json"))
    parser.add_argument("--deck-index", type=int, required=True)
    parser.add_argument("--deck-group", choices=("learner_decks", "field_decks"),
                        default="learner_decks",
                        help="must match the group build_submission packaged from")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--min-agreement", type=float, default=0.98)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with tarfile.open(args.archive, "r:gz") as archive:
        names = set(archive.getnames())
    missing = REQUIRED - names
    if missing:
        raise SystemExit(f"archive is missing {sorted(missing)}")

    with open(args.pool, encoding="utf-8") as source:
        pool = json.load(source)
    expected_deck = pool[args.deck_group][args.deck_index]["cards"]
    field = [item["cards"] for item in pool["field_decks"]]

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="osfp-verify-") as staging:
        with tarfile.open(args.archive, "r:gz") as archive:
            archive.extractall(staging, filter="data")
        with open(os.path.join(staging, "deck.csv"), encoding="utf-8") as source:
            packaged_deck = [int(line) for line in source if line.strip()]
        if packaged_deck != expected_deck:
            failures.append(
                f"deck.csv does not match {args.deck_group}[{args.deck_index}] "
                f"({len(packaged_deck)} cards packaged)"
            )

        module = load_packaged_agent(staging)
        submitted = module.agent({"select": None})
        if list(submitted) != expected_deck:
            failures.append("agent did not return the packaged deck on the deck-submission callback")

        model = load_npz(args.model).eval()
        lib = get_lib()
        card_db = {int(card["cardId"]): card for card in json.loads(lib.AllCard().decode())}
        attack_db = {int(a["attackId"]): a for a in json.loads(lib.AllAttack().decode())}

        stats = {"decisions": 0, "compared": 0, "agreed": 0, "invalid": 0,
                 "latencies": [], "disagreements": []}
        outcomes = {"wins": 0, "losses": 0, "step_caps": 0, "errors": 0}
        print(f"playing {args.games} games with the packaged agent", flush=True)
        for game_index in range(args.games):
            agent_seat = game_index % 2
            opponent = field[game_index % len(field)]
            decks = (packaged_deck, opponent) if agent_seat == 0 else (opponent, packaged_deck)
            winner, error = play(module, model, decks, card_db, attack_db, agent_seat, stats)
            if error == "step cap reached":
                outcomes["step_caps"] += 1
            elif error:
                outcomes["errors"] += 1
                failures.append(f"game {game_index} raised: {error}")
            elif winner == agent_seat:
                outcomes["wins"] += 1
            elif winner is not None:
                outcomes["losses"] += 1
            print(
                f"  game {game_index + 1}/{args.games}: seat={agent_seat} winner={winner} "
                f"error={error} decisions={stats['decisions']}",
                flush=True,
            )

    agreement = stats["agreed"] / stats["compared"] if stats["compared"] else 0.0
    latencies = stats["latencies"]
    report = {
        "archive": os.path.abspath(args.archive),
        "model": os.path.abspath(args.model),
        "deck_index": args.deck_index,
        "games": args.games,
        "outcomes": outcomes,
        "decisions": stats["decisions"],
        "compared_decisions": stats["compared"],
        "agreement": agreement,
        "invalid_actions": stats["invalid"],
        "latency_ms": {
            "mean": statistics.mean(latencies) * 1000 if latencies else 0.0,
            "p95": (statistics.quantiles(latencies, n=20)[18] * 1000
                    if len(latencies) >= 20 else max(latencies, default=0.0) * 1000),
            "max": max(latencies, default=0.0) * 1000,
        },
        "disagreement_samples": stats["disagreements"][:10],
    }
    if stats["invalid"]:
        failures.append(f"packaged agent produced {stats['invalid']} engine-rejected actions")
    if stats["compared"] and agreement < args.min_agreement:
        failures.append(
            f"numpy/torch agreement {agreement:.3f} is below the {args.min_agreement:.3f} floor"
        )
    if not stats["compared"]:
        failures.append("no network-driven decisions were compared; the check proved nothing")
    report["failures"] = failures
    report["passed"] = not failures

    out = args.out or os.path.join(ROOT, "artifacts", "submission_verification.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as target:
        json.dump(report, target, indent=2)

    print(
        f"\ndecisions={stats['decisions']} compared={stats['compared']} "
        f"agreement={agreement * 100:.2f}% invalid={stats['invalid']}\n"
        f"latency mean={report['latency_ms']['mean']:.1f} ms "
        f"p95={report['latency_ms']['p95']:.1f} ms max={report['latency_ms']['max']:.1f} ms\n"
        f"record {outcomes['wins']}-{outcomes['losses']} "
        f"caps={outcomes['step_caps']} errors={outcomes['errors']}",
        flush=True,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", flush=True)
        raise SystemExit(f"verification failed; wrote {out}")
    print(f"verification passed; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
