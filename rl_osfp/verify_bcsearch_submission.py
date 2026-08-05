"""Prove a BC-shell archive is really searching *with* its network.

The shell fails quietly by design.  `_net_scores` wraps every network call in a
broad `except` and returns `None` on any error, and the search reads `None` as
"no model available" and carries on with handcrafted heuristic priors.  So an
archive whose checkpoint is completely unusable still builds, still plays legal
games to completion, still shows normal latency, and still produces a
well-formed tarball.  "It ran without raising" proves nothing here.

Three failures this is built to catch, in order of how much they have cost us:

1.  **The net never loads.**  `_load_net` returns early on a missing file, an
    import error, or its own latency guard (`NET_TIME_BUDGET_S`), leaving
    `_NET is None`.  The rl_osfp net is 192d/6L over SEQ 93 against the BC net's
    160d/5L over SEQ 53, so it is several times heavier and tripping that guard
    is a live risk, not a hypothetical.
2.  **The net loads but every call fails.**  This is what the `nn_infer`
    adapter exists to prevent: the rl_osfp `forward` returns three values where
    the shell unpacks two, and the resulting `ValueError` is swallowed per call.
    `_NET` is not None, so a naive check passes, yet no prior is ever used.
3.  **The net works but is too slow to search.**  Measured, the shell calls the
    net about 4 times per decision, once per determinized world at the root, and
    searches on heuristic priors below it.  So a heavier net costs per move
    rather than per simulation: 192d/6L runs 23.4 ms/call against 160d/5L at
    10.4 ms, which is about 8.5% of a 1.1 s move rather than a throughput cliff.

The check therefore wraps `_net_scores` itself and counts real, non-`None`
returns, which is both the "priors fire" assertion and the per-move cost.

The agent is loaded the way the Kaggle runner loads it: the source is exec'd as
a *string* so `__file__` is undefined, from the archive directory as cwd, with
the repo popped off `sys.path`.
"""
from __future__ import annotations

import argparse
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
REQUIRED = {"main.py", "deck.csv", "nn_features.py", "cg/libcg.so"}


def load_agent_as_runner_does(staging: str) -> dict:
    """Exec main.py as a string from `staging`, with the repo off sys.path.

    `kaggle_environments.agent.get_last_callable` compiles the source with no
    `__file__` and pops the agent directory from `sys.path` before `agent()` is
    called, so anything resolved lazily at call time must survive that.  We
    reproduce it rather than importing the module, because importing tests a
    path Kaggle never takes.
    """
    saved_path, saved_cwd, saved_modules = list(sys.path), os.getcwd(), set(sys.modules)
    try:
        # repo off the path; archive dir is where AGENT_DIR must resolve from
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != ROOT]
        os.chdir(staging)
        with open(os.path.join(staging, "main.py"), encoding="utf-8") as handle:
            source = handle.read()
        namespace: dict = {}
        exec(compile(source, "main.py", "exec"), namespace)  # noqa: S102 - the runner does this
        if "agent" not in namespace:
            raise SystemExit("main.py defines no agent()")
        return namespace
    finally:
        sys.path[:] = saved_path
        os.chdir(saved_cwd)
        for name in set(sys.modules) - saved_modules:
            sys.modules.pop(name, None)


def instrument(namespace: dict) -> dict:
    """Count `_net_scores` calls and how many returned real priors."""
    counters = {"calls": 0, "priors": 0, "none": 0, "seconds": 0.0}
    original = namespace["_net_scores"]

    def counting(state, me, sel, opts, heur):
        begin = time.perf_counter()
        result = original(state, me, sel, opts, heur)
        counters["seconds"] += time.perf_counter() - begin
        counters["calls"] += 1
        if result is None:
            counters["none"] += 1
        else:
            counters["priors"] += 1
        return result

    namespace["_net_scores"] = counting
    return counters


def fallback(select: dict) -> list[int]:
    options = select.get("option") or []
    if not options:
        return []
    low = max(0, min(int(select.get("minCount", 1) or 0), len(options)))
    high = max(low, min(int(select.get("maxCount", max(low, 1)) or 0), len(options)))
    return list(range(low if low > 0 else min(1, high)))


def play(namespace, decks, agent_seat, stats, staging, max_steps=2400):
    """One game: packaged agent on `agent_seat`, heuristic fallback opposite.

    The opponent is deliberately trivial.  This is a health check on one
    archive, not a strength measurement. Strength is the harness's job.
    """
    observation, start = game.battle_start(decks[0], decks[1])
    if observation is None:
        return None, f"BattleStart error {start.error}"
    saved_cwd = os.getcwd()
    try:
        for _ in range(max_steps):
            current = observation.get("current") or {}
            result = int(current.get("result", -1))
            if result >= 0:
                return (result if result in (0, 1) else None), None
            seat = int(current.get("yourIndex", 0))
            select = observation.get("select") or {}
            if seat == agent_seat:
                os.chdir(staging)
                begin = time.perf_counter()
                try:
                    action = namespace["agent"](observation)
                finally:
                    os.chdir(saved_cwd)
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
        return None, "step cap reached"
    finally:
        try:
            game.battle_finish()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--pool", default=os.path.join(ROOT, "data", "fresh", "deck_pool_elo1000.json"))
    parser.add_argument("--min-prior-rate", type=float, default=0.99,
                        help="fraction of _net_scores calls that must return real priors")
    parser.add_argument("--min-latency-ms", type=float, default=40.0,
                        help="mean per-decision floor; below it the agent is falling back, not searching")
    parser.add_argument("--expect-no-net", action="store_true",
                        help="archive ships no model.npz on purpose; assert the "
                             "search runs on heuristic priors instead")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with tarfile.open(args.archive, "r:gz") as archive:
        names = {n.lstrip("./") for n in archive.getnames()}
    missing = REQUIRED - names
    if missing:
        raise SystemExit(f"archive is missing {sorted(missing)}")

    with open(args.pool, encoding="utf-8") as handle:
        pool = json.load(handle)
    field = [deck["cards"] for deck in pool["field_decks"]]

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="bcsearch-verify-") as staging:
        with tarfile.open(args.archive, "r:gz") as archive:
            archive.extractall(staging, filter="data")
        with open(os.path.join(staging, "deck.csv"), encoding="utf-8") as handle:
            deck = [int(line) for line in handle if line.strip()]
        if len(deck) != 60:
            failures.append(f"deck.csv has {len(deck)} cards, not 60")

        namespace = load_agent_as_runner_does(staging)

        # deck callback first: a short deck fails the whole episode outright
        saved = os.getcwd()
        os.chdir(staging)
        try:
            submitted = namespace["agent"]({"select": None})
        finally:
            os.chdir(saved)
        if list(submitted) != deck:
            failures.append(
                f"deck callback returned {len(submitted)} cards, not the packaged 60"
            )

        counters = instrument(namespace)
        lib = get_lib()
        card_db = {int(c["cardId"]): c for c in json.loads(lib.AllCard().decode())}

        stats = {"decisions": 0, "invalid": 0, "latencies": []}
        outcomes = {"wins": 0, "losses": 0, "step_caps": 0, "errors": 0}
        print(f"playing {args.games} games from {os.path.basename(args.archive)}", flush=True)
        for index in range(args.games):
            agent_seat = index % 2
            opponent = field[index % len(field)]
            decks = (deck, opponent) if agent_seat == 0 else (opponent, deck)
            winner, error = play(namespace, decks, agent_seat, stats, staging)
            if error == "step cap reached":
                outcomes["step_caps"] += 1
            elif error:
                outcomes["errors"] += 1
                failures.append(f"game {index} raised: {error}")
            elif winner == agent_seat:
                outcomes["wins"] += 1
            elif winner is not None:
                outcomes["losses"] += 1
            print(f"  game {index + 1}/{args.games}: seat={agent_seat} winner={winner} "
                  f"error={error} net_calls={counters['calls']}", flush=True)

        net_loaded = namespace.get("_NET") is not None
        encoder = getattr(namespace.get("_NF"), "__name__", None)

    prior_rate = counters["priors"] / counters["calls"] if counters["calls"] else 0.0
    ms_per_call = (counters["seconds"] / counters["calls"] * 1000) if counters["calls"] else 0.0
    latencies = stats["latencies"]
    mean_latency_ms = statistics.mean(latencies) * 1000 if latencies else 0.0

    if args.expect_no_net:
        if net_loaded:
            failures.append("--expect-no-net was passed but a checkpoint loaded anyway")
    elif not net_loaded:
        failures.append(
            "_NET is None: the checkpoint never loaded, so the archive is running "
            "on heuristic priors and ignores the model entirely"
        )
    if not counters["calls"]:
        failures.append("_net_scores was never called; the search never reached the root expansion")
    elif not args.expect_no_net and prior_rate < args.min_prior_rate:
        failures.append(
            f"only {prior_rate * 100:.1f}% of {counters['calls']} network calls returned "
            f"priors ({counters['none']} swallowed exceptions) - below the "
            f"{args.min_prior_rate * 100:.0f}% floor. The net is loaded but unusable."
        )
    if stats["invalid"]:
        failures.append(f"{stats['invalid']} engine-rejected actions "
                        "(cabt awards the game to the opponent on any rejected select)")
    if mean_latency_ms < args.min_latency_ms:
        failures.append(
            f"mean decision {mean_latency_ms:.0f} ms is under the {args.min_latency_ms:.0f} ms "
            "floor - the agent is falling back rather than searching"
        )

    report = {
        "archive": os.path.abspath(args.archive),
        "net_loaded": net_loaded,
        "encoder": encoder,
        "net_calls": counters["calls"],
        "calls_returning_priors": counters["priors"],
        "calls_returning_none": counters["none"],
        "prior_rate": prior_rate,
        "ms_per_net_call": ms_per_call,
        # Measured at ~4/decision: the shell passes `state` to `_gen_candidates`
        # only at the root, once per determinized world, and searches on
        # heuristic priors below it.  So this tracks world count, not node count,
        # and the net's cost is per-move rather than per-simulation.
        "net_calls_per_decision": (counters["calls"] / stats["decisions"]
                                   if stats["decisions"] else 0.0),
        "decisions": stats["decisions"],
        "invalid_actions": stats["invalid"],
        "mean_decision_ms": mean_latency_ms,
        "max_decision_ms": max(latencies, default=0.0) * 1000,
        "outcomes": outcomes,
        "failures": failures,
        "passed": not failures,
    }
    out = args.out or os.path.join(
        ROOT, "artifacts",
        os.path.basename(args.archive).replace(".tar.gz", "") + "_verification.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as target:
        json.dump(report, target, indent=2)

    print(
        f"\nnet_loaded={net_loaded} encoder={encoder}\n"
        f"net calls={counters['calls']} priors={counters['priors']} none={counters['none']} "
        f"({prior_rate * 100:.1f}%)  {ms_per_call:.1f} ms/call\n"
        f"net calls/decision={report['net_calls_per_decision']:.1f}  "
        f"mean decision={mean_latency_ms:.0f} ms  invalid={stats['invalid']}\n"
        f"record {outcomes['wins']}-{outcomes['losses']} caps={outcomes['step_caps']}",
        flush=True,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", flush=True)
        raise SystemExit(f"verification failed; wrote {out}")
    print(f"verification passed; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
