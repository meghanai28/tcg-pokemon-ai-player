"""Expert iteration: train the prior to predict what the SEARCH chooses.

Why this and not more PPO. The deployed agent is `search(prior=net)`, and inside
`_gen_candidates` the net's whole job is:

    scores = _net_scores(...)                     # net replaces the heuristic
    order  = sorted(range(n), key=lambda i: -scores[i])
    cands  = [(i,) for i in order[:cap]]          # cap = 16

which is a **recall** objective: get the truly good option into the explored set
and give it enough prior mass to earn visits. Bare-policy PPO optimises *argmax*
quality instead, and nothing in its loss preserves the ranking of the
alternatives the search depends on. Measured, that distinction is worth a lot:
a policy that beat the BC prior 64.3% head to head was an 11-point *worse* prior
behind the shell, and an unanchored one was 42 points worse.

Expert iteration targets the deployed objective directly. The label for a
position is the search's own root visit distribution, so the prior is trained to
agree with the thing it is a prior for, on the positions that thing actually
reaches. This is the standard method for exactly this deployment shape and it is
the one approach this project has never tried.

**The frozen shell already exposes the label.** `_search_move` takes a
`collect_policy` dict and fills it with aggregated root visit counts per action,
so no edit to the proven `main.py` is needed. `nn_features.encode` returns
`opt_slot`, the option-index to token-position map, which is what turns those
per-action counts into a distribution over option tokens.

Output is written in `bc_train/train_bc.py`'s shard format
(`kind, card, scal, mask, ctx, stype, pi, z`), so the proven trainer can consume
it unchanged with `--init` warm-starting from the champion.

Usage:
    py -m rl_osfp.exit_generate --archive harness/anchors/grpo_tech_grim_972_912_811.tar.gz \
        --games 200 --budget 0.2 --workers 5 --out data/exit_shards
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import shutil
import sys
import tarfile
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# Loading the frozen shell so its internals are reachable
# --------------------------------------------------------------------------
def load_shell(agent_dir: str) -> dict:
    """Exec the archive's `main.py` and return its namespace, engine ready.

    The shell loads the engine, card database and network lazily on the first
    agent call and swallows every failure by design, so all three are forced
    here and asserted. A silent fallback would produce heuristic-prior labels
    that look exactly like search labels.
    """
    main_py = os.path.join(agent_dir, "main.py")
    with open(main_py, encoding="utf-8") as handle:
        source = handle.read()
    namespace = {"__file__": main_py, "__name__": "__main__"}
    saved_path, cwd = list(sys.path), os.getcwd()
    sys.path.insert(0, agent_dir)
    try:
        os.chdir(agent_dir)
        exec(compile(source, main_py, "exec"), namespace)  # noqa: S102
        namespace["MY_DECK"] = namespace["_load_deck"]()
        namespace["_load_engine"]()
        namespace["_load_card_db"]()
        namespace["_load_net"]()
    finally:
        os.chdir(cwd)
        sys.path[:] = saved_path

    if not namespace.get("CARD"):
        raise RuntimeError(f"{agent_dir}: card database empty, labels would be garbage")
    if namespace.get("_LIB") is None:
        raise RuntimeError(f"{agent_dir}: engine did not load, there would be no search")
    if namespace.get("_NET") is None:
        raise RuntimeError(f"{agent_dir}: network did not load, priors would be heuristic")
    return namespace


def soften_prior(ns: dict, temperature: float) -> None:
    """Flatten the prior the SEARCH uses, without touching the frozen shell.

    This is the whole trick, and it follows from measuring the right thing. The
    search agrees with its own prior on 95.7% of decisions, and the reason is
    NOT that the net restricts the candidate set: options per decision are a
    median of 5 against `cap=16`, so the truncation binds on 3.0% of decisions.
    The reason is PUCT's prior weighting:

        pri = math.exp(min(6.0, sum(scores) / len(c)))

    Net logits go through `exp`, so a confident net produces a near-degenerate
    prior, PUCT spends its visits on the net's first choice, and at a few
    thousand simulations it can never accumulate the evidence to overturn it.
    The teacher then just restates the student, which is why distilling it
    taught nothing.

    Dividing the scores by a temperature flattens that prior while leaving the
    *ranking* identical, so the candidate set is unchanged and only the
    exploration changes. The search is then forced to actually verify the
    alternatives rather than rubber-stamp the prior, and its visit distribution
    becomes a label that carries information the net does not already have.

    This only affects label generation. The shipped agent runs the frozen shell
    at temperature 1, exactly as it scored 972.
    """
    if temperature == 1.0:
        return
    original = ns["_net_scores"]

    def softened(state, me, sel, opts, heur):
        scores = original(state, me, sel, opts, heur)
        if scores is None:
            return None
        return [s / temperature if s > -1e8 else s for s in scores]

    ns["_net_scores"] = softened
    # `_gen_candidates` resolves `_net_scores` from module globals at call time,
    # and the shell was exec'd into this dict, so rebinding the name here is
    # what the search will actually use.


def encode_decision(ns: dict, obs, sel, me: int):
    """Encode exactly the way `_net_scores` does at deployment."""
    features = ns["_NF"]
    truncation = None if features.__name__.startswith("nn_features_rich") else None
    state = {"current": obs["current"], "select": sel}
    if getattr(features, "DECK_AWARE", False):
        state["decklist"] = ns["MY_DECK"] or []
    kind, card, scal, mask, opt_slot = features.encode(
        state, me, ns["CARD"], ns["ATTACK"], truncation)
    ctx = int(sel.get("context") or 0)
    stype = int(sel.get("type") or 0)
    return kind, card, scal, mask, opt_slot, ctx, stype


def visits_to_pi(visits: dict, opt_slot, seq: int) -> np.ndarray | None:
    """Root visit counts per action -> distribution over option tokens.

    A multi-select action spreads its visits across its members, which mirrors
    how `_gen_candidates` scores a candidate set by the sum of its members.
    """
    pi = np.zeros(seq, dtype=np.float32)
    for action, count in visits.items():
        share = float(count) / max(len(action), 1)
        for option_index in action:
            slot = opt_slot[option_index] if option_index < len(opt_slot) else -1
            if slot >= 0:
                pi[slot] += share
    total = float(pi.sum())
    if total <= 0.0:
        return None
    return pi / total


# --------------------------------------------------------------------------
# One self-play game, both seats searching, every decision labelled
# --------------------------------------------------------------------------
_SHELL_CACHE: dict = {}


def cached_shell(agent_dir: str) -> dict:
    """Load the shell once per worker process, then reuse it.

    Building it costs about a minute (the exec, the engine, the card database
    and `_load_net`'s own latency benchmark), which at a 0.1 s search budget was
    dwarfing the games themselves: 2 games took 3.4 minutes, essentially all of
    it setup. Reuse is safe because per-game state lives in `GameState`, which
    the recording agent rebuilds on the deck callback.
    """
    if agent_dir not in _SHELL_CACHE:
        _SHELL_CACHE[agent_dir] = load_shell(agent_dir)
    return _SHELL_CACHE[agent_dir]


def play_game(args_tuple) -> list[dict]:
    agent_dir, budget, seed, max_steps, prior_temperature = args_tuple
    # One core per worker. Left alone, the shell's numpy spawns a full BLAS
    # thread pool in every worker and the machine oversubscribes hard: measured,
    # a single generation worker alongside a 5-worker harness took the load
    # average from about 5 to 21 on 14 threads. That also makes the search
    # budget non-comparable, because a deadline in seconds buys a different
    # number of simulations depending on what else is running.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    import kaggle_environments as ke

    ns = cached_shell(agent_dir)
    soften_prior(ns, prior_temperature)
    seq = ns["_NF"].SEQ
    records: list[dict] = []

    def make_agent(seat: int):
        state = {"game": None}

        def agent(obs):
            sel = obs.get("select")
            if sel is None:
                state["game"] = ns["GameState"]()
                state["game"].rng = random.Random(seed * 7919 + seat)
                return list(ns["MY_DECK"])
            game = state["game"]
            game.calls += 1
            options = sel.get("option") or []
            n = len(options)
            kmax = max(1, min(sel.get("maxCount", 1), n))
            # Forced decisions teach nothing and cost a search, so skip them.
            if n == 0:
                return []
            if n == 1:
                return [0]
            if kmax >= n and sel.get("minCount", 0) >= n:
                return list(range(n))

            me = (obs.get("current") or {}).get("yourIndex", 0)
            visits: dict = {}
            action = None
            try:
                action = ns["_search_move"](obs, me, game.opp_model,
                                            time.perf_counter() + budget,
                                            game.rng, collect_policy=visits)
            except Exception:
                action = None
            finally:
                try:
                    ns["_LIB"].SearchEnd(ns["_CTX"])
                except Exception:
                    pass

            if action is not None and visits:
                try:
                    kind, card, scal, mask, opt_slot, ctx, stype = encode_decision(
                        ns, obs, sel, me)
                    pi = visits_to_pi(visits, opt_slot, seq)
                    if pi is not None and bool(((kind == 3) & (mask > 0.5)).any()):
                        records.append({
                            "seat": seat, "kind": kind, "card": card, "scal": scal,
                            "mask": mask, "ctx": ctx, "stype": stype, "pi": pi,
                        })
                except Exception:
                    pass

            action = ns["_validate"](action, sel)
            if action is None:
                action = ns["_validate"](ns["_heuristic_action"](sel, game.rng), sel)
            if action is None:
                action = list(range(max(1, min(sel.get("minCount", 1) or 1, n))))
            return action

        return agent

    env = ke.make("cabt", configuration={"episodeSteps": max_steps}, debug=False)
    env.run([make_agent(0), make_agent(1)])
    rewards = [entry.reward for entry in env.steps[-1]]
    outcome = [0.0, 0.0]
    if rewards[0] is not None and rewards[1] is not None:
        outcome = [float(np.sign(rewards[0])), float(np.sign(rewards[1]))]
    for record in records:
        record["z"] = outcome[record["seat"]]
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--budget", type=float, default=0.2,
                        help="seconds per searched decision. The teacher only has "
                             "to beat the student, and search beat the bare policy "
                             "87.5% at 0.5s, so a cheap teacher is still a teacher.")
    parser.add_argument("--prior-temperature", type=float, default=1.0,
                        help="divide the net logits the SEARCH uses by this. >1 flattens\n"
                             "the PUCT prior so the search explores instead of restating\n"
                             "the net. Deployment is unaffected. See soften_prior().")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--shard-size", type=int, default=20000)
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "exit_shards"))
    args = parser.parse_args()

    if args.workers > 5:
        sys.exit("more than 5 workers is over this machine's cap")

    # Set before the pool spawns, so the children inherit it and import numpy
    # with the limit already in place. Setting it inside the worker is too late:
    # the module-level `import numpy` runs first on a spawned start method.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

    staging = tempfile.mkdtemp(prefix="exitgen-")
    agent_dir = os.path.join(staging, "agent")
    os.makedirs(agent_dir, exist_ok=True)
    with tarfile.open(args.archive, "r:gz") as handle:
        handle.extractall(agent_dir, filter="data")

    os.makedirs(args.out, exist_ok=True)
    jobs = [(agent_dir, args.budget, i, args.max_steps, args.prior_temperature)
            for i in range(args.games)]
    buffer: list[dict] = []
    shard_index = 0
    written = 0
    began = time.time()

    def flush(force: bool = False) -> None:
        nonlocal buffer, shard_index, written
        while len(buffer) >= args.shard_size or (force and buffer):
            chunk = buffer[: args.shard_size]
            buffer = buffer[args.shard_size:]
            path = os.path.join(args.out, f"exit_{shard_index:04d}.npz")
            np.savez_compressed(
                path,
                kind=np.stack([r["kind"] for r in chunk]).astype(np.int8),
                card=np.stack([r["card"] for r in chunk]).astype(np.int16),
                scal=np.stack([r["scal"] for r in chunk]).astype(np.float32),
                mask=np.stack([r["mask"] for r in chunk]).astype(np.float32),
                ctx=np.array([r["ctx"] for r in chunk], dtype=np.int16),
                stype=np.array([r["stype"] for r in chunk], dtype=np.int16),
                pi=np.stack([r["pi"] for r in chunk]).astype(np.float32),
                z=np.array([r["z"] for r in chunk], dtype=np.float32),
            )
            shard_index += 1
            written += len(chunk)
            print(f"  wrote {path} ({len(chunk)} decisions, {written} total)", flush=True)

    context = mp.get_context("spawn")
    with context.Pool(args.workers) as pool:
        for done, records in enumerate(pool.imap_unordered(play_game, jobs), 1):
            buffer.extend(records)
            flush()
            if done % 10 == 0 or done == len(jobs):
                rate = (time.time() - began) / done
                print(f"{done}/{len(jobs)} games, {written + len(buffer)} decisions, "
                      f"{(time.time() - began) / 60:.1f} min, "
                      f"eta {rate * (len(jobs) - done) / 60:.0f} min", flush=True)
    flush(force=True)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"done: {written} labelled decisions in {shard_index} shards under {args.out}")


if __name__ == "__main__":
    main()
