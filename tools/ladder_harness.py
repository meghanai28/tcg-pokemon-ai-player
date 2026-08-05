"""Play packaged submissions against each other through the real Kaggle runner.

Why this exists
---------------
Every gate this project has ever run compared our own components to each other:
search vs our policy, checkpoint vs checkpoint, deck vs deck, fixed search vs
broken search.  All of those can look excellent while the whole family sits far
below a baseline nobody tested against, which is exactly what happened.  The
archives that scored 972.0 and 810.8 were sitting on disk the entire time.

So the unit of measurement here is a *shipped tarball*, not a checkpoint, and
the reference agents are archives whose real ladder scores we know.  A change is
real only if it beats those, and the harness is trusted only to the extent that
its ordering reproduces the ladder ordering it was calibrated on.

Fidelity
--------
Games run through `kaggle_environments.make("cabt")`, the same environment the
competition scores with, so engine rules, the INVALID-action rule (a rejected
`select` hands the game to the opponent, with no retry) and the observation
shape are the graded ones rather than a local reimplementation.

Two deliberate deviations from `kaggle_environments.agent.get_last_callable`,
both required to run two archives in one process:

1.  `__file__` is seeded into the exec namespace.  The runner leaves it
    undefined, and our shell then falls back to `/kaggle_simulations/agent` and
    finally `os.getcwd()`.  On Kaggle that fallback resolves to the agent's own
    directory; locally there is one cwd and two agents, so without this the two
    archives would read each other's `deck.csv` and `model.npz`.  Seeding
    `__file__` reproduces the *outcome* Kaggle produces, not a different one.
2.  `sys.modules` is swapped per call.  The shell already aliases
    `nn_features*` by agent directory, precisely because "a bare import_module
    would hand every agent in the process whichever copy was imported first".
    It does *not* do this for `nn_infer`, and our archives disagree about what
    `nn_infer.NumpyNet` is. The rl_osfp build ships an adapter whose `forward`
    returns two values, the BC archives ship the original.  Whichever loaded
    first would silently define the other's network, and the failure is
    invisible: the loser falls back to heuristic priors and plays on.

Neither deviation changes what a single archive does; they only stop two
archives from corrupting each other.

Calibration is the point
------------------------
`--calibrate` fits local Bradley-Terry ratings to the known ladder scores of the
anchor archives and reports the rank correlation.  A harness that cannot order
the agents whose ladder scores we already know cannot be trusted to order the
ones we do not.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import tempfile
import time
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ladder results for the archives we still hold, keyed by the basename under
# harness/anchors/.  `grpo_tech_grim` is the same bytes submitted three times
# (refs 55185089, 55202336, 55233305). The spread is what ladder noise and
# drift actually look like on this competition, and it is much wider than the
# +/-75 the project notes assumed.
LADDER_SCORES = {
    "grpo_tech_grim_972_912_811": [972.0, 911.9, 810.8],
    "awr_grpo_tech_grim_903": [903.2],
    "bc800_tech_grim_849": [848.9],
    "ppo_search_585": [592.0],   # refreshed 2026-08-04 20:0x, was 585.2
    "v3_pure_rl_480": [480.0],
    "grpo_search_405_RECONSTRUCTED": [417.3],  # refreshed, was 405.0
}

# Anchors whose bytes were rebuilt rather than recovered, so the score attached
# to the name may not belong to these bytes.  Excluded from the fit unless
# --trust-rebuilt is passed, and measured 2026-08-04 this exclusion is load
# bearing: with all six anchors the fit is R^2 0.635, and with the five exact
# ones it is R^2 0.987.  The rebuilt archive predicts 677.8 against a recorded
# 405.0, which says the rebuild is simply not the agent that scored 405.
UNVERIFIED_ANCHORS = {"grpo_search_405_RECONSTRUCTED"}


# --------------------------------------------------------------------------
# Agent loading
# --------------------------------------------------------------------------
def _private_modules(agent_dir: str) -> dict:
    """Import this archive's `nn_infer` and `cg` under private aliases.

    Returns the mapping installed into `sys.modules` for the duration of each
    call, so two archives never share a network implementation or an engine
    binding.  Missing modules are skipped: not every archive ships a net.
    """
    import importlib.util

    tag = f"{abs(hash(agent_dir)) & 0xffffffff:08x}"
    private = {}
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    for name in ("nn_infer_osfp", "nn_infer"):
        path = os.path.join(agent_dir, name + ".py")
        if not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location(f"{name}__{tag}", path)
        module = importlib.util.module_from_spec(spec)
        # `nn_infer` may import `nn_infer_osfp` by its bare name, so publish the
        # already-built private copy under that name while this one executes.
        saved = sys.modules.get("nn_infer_osfp")
        if "nn_infer_osfp" in private:
            sys.modules["nn_infer_osfp"] = private["nn_infer_osfp"]
        try:
            spec.loader.exec_module(module)
        finally:
            if saved is not None:
                sys.modules["nn_infer_osfp"] = saved
            elif "nn_infer_osfp" in sys.modules and "nn_infer_osfp" in private:
                del sys.modules["nn_infer_osfp"]
        private[name] = module
    return private


def load_archive_agent(agent_dir: str):
    """Build a callable for one unpacked archive, isolated from the others."""
    main_py = os.path.join(agent_dir, "main.py")
    with open(main_py, encoding="utf-8") as handle:
        source = handle.read()

    namespace = {"__file__": main_py, "__name__": "__main__"}
    saved_path = list(sys.path)
    sys.path.insert(0, agent_dir)
    try:
        exec(compile(source, main_py, "exec"), namespace)  # noqa: S102 - the runner does this
    finally:
        sys.path[:] = saved_path

    callables = [v for v in namespace.values() if callable(v)]
    if not callables:
        raise SystemExit(f"{main_py} defines no callable")
    # The runner takes the *last* callable defined, so a stray helper defined
    # after agent() would be run instead of the agent. Assert we agree.
    chosen = callables[-1]
    if getattr(chosen, "__name__", None) != "agent":
        raise SystemExit(
            f"{main_py}: last callable is {getattr(chosen, '__name__', '?')}, not agent(); "
            "the Kaggle runner would invoke the wrong function"
        )

    private = _private_modules(agent_dir)

    def wrapped(observation, configuration=None):
        saved = {}
        for name, module in private.items():
            saved[name] = sys.modules.get(name)
            sys.modules[name] = module
        try:
            return chosen(observation)
        finally:
            for name, previous in saved.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

    return wrapped


def unpack(archive: str, into: str) -> str:
    name = os.path.basename(archive).replace(".tar.gz", "")
    target = os.path.join(into, name)
    if os.path.isdir(target):
        return target
    os.makedirs(target, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(target, filter="data")
    return target


# --------------------------------------------------------------------------
# One match, in its own process
# --------------------------------------------------------------------------
_AGENT_CACHE: dict = {}


def _cached_agent(agent_dir: str):
    """Load an archive once per worker process, then reuse it.

    Building an agent costs far more than playing with it: the exec, the engine
    load, and `_load_net`'s own latency benchmark ran to roughly a minute, which
    dwarfed a whole game at a small per-move budget.  Reuse is safe because the
    shell rebuilds its per-game state (`_GAME = GameState()`) on the deck
    callback, and cabt opens every episode by asking both seats for a deck.
    """
    if agent_dir not in _AGENT_CACHE:
        _AGENT_CACHE[agent_dir] = load_archive_agent(agent_dir)
    return _AGENT_CACHE[agent_dir]


def _play(args) -> dict:
    """Play one game. Runs in a worker process: cabt keeps battle state global."""
    dir_a, dir_b, budget, index, max_steps = args
    os.environ["PTCG_MAX_BUDGET"] = str(budget)
    # One core per worker. Left alone, each worker's numpy spawns a full BLAS
    # thread pool, the workers oversubscribe the machine, and every agent's
    # effective per-move throughput then depends on how many other games happen
    # to be running, which silently makes the search budget non-comparable
    # across matches.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    began = time.time()
    try:
        import kaggle_environments as ke

        # cabt ships episodeSteps=10_000_000, i.e. no cap at all.  Two search
        # agents that both decline to commit can then stall a mirror match
        # indefinitely. This project has already seen about a quarter of greedy
        # mirror games hit a 2,400-step cap.  Without one here a single stalled
        # game pins a worker forever and the round robin never reports.
        # A capped game is recorded as a draw, not an error: dropping stalls
        # would reward exactly the behaviour that causes them.
        env = ke.make("cabt", configuration={"episodeSteps": max_steps}, debug=False)
        agents = [_cached_agent(dir_a), _cached_agent(dir_b)]
        env.run(agents)
        rewards = [state.reward for state in env.state]
        statuses = [str(state.status) for state in env.state]
        capped = len(env.steps) >= max_steps and not any(
            "DONE" in s and r in (1, -1) for s, r in zip(statuses, rewards))
        return {
            "index": index, "rewards": rewards, "statuses": statuses,
            "steps": len(env.steps), "seconds": time.time() - began,
            "step_capped": capped, "error": None,
        }
    except Exception as exc:  # a crashed match must not silently vanish
        return {
            "index": index, "rewards": None, "statuses": None, "steps": 0,
            "seconds": time.time() - began, "step_capped": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


# --------------------------------------------------------------------------
# Rating
# --------------------------------------------------------------------------
def bradley_terry(pairs: dict, names: list[str], iterations: int = 5000,
                  rate: float = 0.05, prior: float = 0.5) -> dict:
    """Fit Elo-scale ratings by gradient ascent on the BT log-likelihood.

    `prior` adds a symmetric pseudo-win to every played pair, which keeps a
    clean sweep from sending a rating to infinity.
    """
    index = {name: i for i, name in enumerate(names)}
    theta = [0.0] * len(names)
    scale = math.log(10) / 400.0

    played = []
    for (left, right), record in pairs.items():
        wins, losses = record["wins"], record["losses"]
        if wins + losses == 0:
            continue
        played.append((index[left], index[right], wins + prior, losses + prior))

    for _ in range(iterations):
        grad = [0.0] * len(names)
        for i, j, wins, losses in played:
            expected = 1.0 / (1.0 + math.exp(-scale * (theta[i] - theta[j])))
            residual = wins - (wins + losses) * expected
            grad[i] += residual
            grad[j] -= residual
        for k in range(len(names)):
            theta[k] += rate * grad[k]
        mean = sum(theta) / len(theta)
        theta = [t - mean for t in theta]
    return {name: theta[index[name]] for name in names}


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return slope, intercept, (1 - ss_res / ss_tot if ss_tot else 0.0)


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", nargs="+", required=True,
                        help="tarballs to enter into the round robin")
    parser.add_argument("--games-per-pair", type=int, default=12,
                        help="seats alternate, so an even number is balanced")
    parser.add_argument("--budget", type=float, default=0.3,
                        help="PTCG_MAX_BUDGET seconds/move, applied to both seats")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="per-game step cap; cabt itself has none, so a "
                             "stalled mirror match would pin a worker forever")
    parser.add_argument("--calibrate", action="store_true",
                        help="fit local ratings against known ladder scores")
    parser.add_argument("--trust-rebuilt", action="store_true",
                        help="include anchors whose bytes were rebuilt rather than "
                             "recovered; off by default because one such anchor "
                             "drops the fit from R^2 0.987 to 0.635")
    parser.add_argument("--out", default=os.path.join(ROOT, "harness", "round_robin.json"))
    args = parser.parse_args()

    for archive in args.archives:
        if not os.path.exists(archive):
            raise SystemExit(f"no such archive: {archive}")

    staging = tempfile.mkdtemp(prefix="ladder-harness-")
    try:
        dirs = {}
        for archive in args.archives:
            name = os.path.basename(archive).replace(".tar.gz", "")
            dirs[name] = unpack(archive, staging)
        names = list(dirs)

        jobs = []
        index = 0
        for left, right in itertools.combinations(names, 2):
            for game in range(args.games_per_pair):
                # alternate seats so first-player advantage cancels
                a, b = (left, right) if game % 2 == 0 else (right, left)
                jobs.append((dirs[a], dirs[b], args.budget, index, args.max_steps))
                index += 1
        seat_of = {}
        index = 0
        for left, right in itertools.combinations(names, 2):
            for game in range(args.games_per_pair):
                a, b = (left, right) if game % 2 == 0 else (right, left)
                seat_of[index] = (a, b)
                index += 1

        print(f"{len(names)} archives, {len(jobs)} games, budget {args.budget}s/move, "
              f"{args.workers} workers", flush=True)

        began = time.time()
        results = []
        # Games are appended as they finish.  A full round robin at the shipping
        # budget runs for hours, and the report is only assembled at the end, so
        # without this a timeout or a killed run loses every game it played.
        stream_path = os.path.splitext(args.out)[0] + "_games.jsonl"
        os.makedirs(os.path.dirname(os.path.abspath(stream_path)), exist_ok=True)
        context = mp.get_context("spawn")
        with context.Pool(args.workers) as pool, \
                open(stream_path, "w", encoding="utf-8") as stream:
            for done, result in enumerate(pool.imap_unordered(_play, jobs), 1):
                results.append(result)
                record = dict(result)
                record["seats"] = list(seat_of[result["index"]])
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                if done % 5 == 0 or done == len(jobs):
                    rate = (time.time() - began) / done
                    print(f"  {done}/{len(jobs)} games "
                          f"({(time.time() - began) / 60:.1f} min, "
                          f"eta {rate * (len(jobs) - done) / 60:.0f} min)", flush=True)

        pairs = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0, "errors": 0})
        totals = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0, "errors": 0})
        invalid = defaultdict(int)
        capped = 0
        for result in results:
            seat_a, seat_b = seat_of[result["index"]]
            key = tuple(sorted((seat_a, seat_b)))
            if result["error"] or result["rewards"] is None:
                pairs[key]["errors"] += 1
                totals[seat_a]["errors"] += 1
                continue
            for seat, status in zip((seat_a, seat_b), result["statuses"]):
                if "INVALID" in status:
                    invalid[seat] += 1
            reward_a = result["rewards"][0]
            winner = seat_a if reward_a == 1 else (seat_b if reward_a == -1 else None)
            if result.get("step_capped"):
                # A stalled game is a non-win for both sides, not an error.
                # Dropping caps would reward a policy that refuses to commit by
                # deleting its stalls from the denominator.
                capped += 1
                pairs[key]["draws"] += 1
                totals[seat_a]["draws"] += 1
                totals[seat_b]["draws"] += 1
            elif winner is None:
                pairs[key]["draws"] += 1
                totals[seat_a]["draws"] += 1
                totals[seat_b]["draws"] += 1
            else:
                loser = seat_b if winner == seat_a else seat_a
                totals[winner]["wins"] += 1
                totals[loser]["losses"] += 1
                if key[0] == winner:
                    pairs[key]["wins"] += 1
                else:
                    pairs[key]["losses"] += 1

        ratings = bradley_terry(pairs, names)

        report = {
            "archives": names,
            "games_per_pair": args.games_per_pair,
            "budget_seconds_per_move": args.budget,
            "elapsed_minutes": (time.time() - began) / 60.0,
            "ratings": ratings,
            "totals": {k: dict(v) for k, v in totals.items()},
            "invalid_action_games": dict(invalid),
            "step_capped_games": capped,
            "pairs": {f"{a}|{b}": dict(v) for (a, b), v in pairs.items()},
        }

        print("\n=== round robin ===", flush=True)
        ordered = sorted(names, key=lambda n: -ratings[n])
        for name in ordered:
            record = totals[name]
            decided = record["wins"] + record["losses"]
            rate = record["wins"] / decided if decided else 0.0
            print(f"  {ratings[name]:+8.1f}  {name:36s} "
                  f"{record['wins']:3d}-{record['losses']:3d} ({rate * 100:5.1f}%)"
                  f"{'  INVALID:' + str(invalid[name]) if invalid[name] else ''}",
                  flush=True)

        if args.calibrate:
            usable = [n for n in names if n in LADDER_SCORES
                      and (args.trust_rebuilt or n not in UNVERIFIED_ANCHORS)]
            skipped = [n for n in names if n in LADDER_SCORES and n not in usable]
            if skipped:
                print(f"\nexcluded from the fit (rebuilt, not byte-verified): "
                      f"{', '.join(skipped)}", flush=True)
            known = [(n, sum(LADDER_SCORES[n]) / len(LADDER_SCORES[n])) for n in usable]
            if len(known) < 3:
                print("\ncalibration needs >=3 anchors with known ladder scores", flush=True)
            else:
                xs = [ratings[n] for n, _ in known]
                ys = [score for _, score in known]
                slope, intercept, r2 = linear_fit(xs, ys)
                rho = spearman(xs, ys)
                report["calibration"] = {
                    "anchors": {n: s for n, s in known},
                    "slope": slope, "intercept": intercept,
                    "r_squared": r2, "spearman_rho": rho,
                    "predicted": {n: slope * ratings[n] + intercept for n in names},
                    "residuals": {n: (slope * ratings[n] + intercept) - s for n, s in known},
                }
                print(f"\n=== calibration on {len(known)} anchors ===", flush=True)
                print(f"  ladder ~= {slope:.2f} * rating + {intercept:.1f}   "
                      f"R^2={r2:.3f}  spearman={rho:+.3f}", flush=True)
                for name in ordered:
                    predicted = slope * ratings[name] + intercept
                    actual = dict(known).get(name)
                    if actual is None:
                        print(f"  {name:36s} predicted {predicted:7.1f}   (no ladder result)",
                              flush=True)
                    else:
                        print(f"  {name:36s} predicted {predicted:7.1f}   "
                              f"actual {actual:7.1f}   residual {predicted - actual:+7.1f}",
                              flush=True)

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as target:
            json.dump(report, target, indent=2)
        print(f"\nwrote {args.out}", flush=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
