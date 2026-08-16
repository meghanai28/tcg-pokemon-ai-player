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

The previous run of this died at 8% with a BrokenPipeError: workers returned
full record lists through the Pool, and a game is roughly a megabyte of numpy
(each decision carries a 53x32 float `scal`). Workers now write their own shard
and return a path plus a count, so the IPC payload is a few bytes.

Usage:
    py tools/exit_generate.py --archive harness/anchors/grpo_tech_grim_972_912_811.tar.gz \
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

TEACHER_PATCHES = (
    # Keep the teacher cap configurable, but use the measured deployment cap of
    # 16 by default. A cap of 32 was tested and caused problems while affecting
    # very few decisions (the median decision has only five legal actions).
    ("_gen_candidates(node.select, self.rng, state=state, me=self.me)",
     "_gen_candidates(node.select, self.rng, cap=TEACHER_CAP, state=state, me=self.me)"),
    # Keep the per-action value alongside the visit count. The search already
    # computes both -- agg[act] is [n_vis, w] -- but the stock hook discards w,
    # and Q = w / n_vis is exactly the counterfactual signal needed to train
    # ranking across every explored action rather than only the one played.
    ("        for act, (n_vis, _w) in agg.items():\n"
     "            collect_policy[act] = n_vis",
     "        for act, (n_vis, _w) in agg.items():\n"
     "            collect_policy[act] = (n_vis, _w / max(n_vis, 1))"),
)


def patch_teacher(agent_dir: str, teacher_cap: int) -> None:
    """Widen the staged shell's candidate cap and make it emit Q per action."""
    main_py = os.path.join(agent_dir, "main.py")
    with open(main_py, encoding="utf-8") as handle:
        source = handle.read()
    for needle, replacement in TEACHER_PATCHES:
        if needle not in source:
            raise RuntimeError(
                f"teacher patch target missing, shell has changed: {needle[:60]!r}")
        source = source.replace(needle, replacement)
    source = source.replace("NET_LEAF_BATCH = 24",
                            f"TEACHER_CAP = {teacher_cap}\nNET_LEAF_BATCH = 24", 1)
    with open(main_py, "w", encoding="utf-8") as handle:
        handle.write(source)


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
    if temperature <= 0:
        raise ValueError("prior temperature must be positive")

    # Workers cache a shell namespace and reuse it for many games.  Wrapping
    # whatever happens to be installed at ``_net_scores`` compounds the
    # temperature on every game (T, T**2, T**3, ...).  Retain the genuinely
    # unmodified scorer once and always derive the requested transform from
    # that stable base.  Calling this function repeatedly is therefore
    # idempotent, including when switching back to temperature 1.
    original = ns.setdefault("_unsoftened_net_scores", ns["_net_scores"])
    if temperature == 1.0:
        ns["_net_scores"] = original
        ns["_prior_temperature"] = 1.0
        return

    def softened(state, me, sel, opts, heur):
        scores = original(state, me, sel, opts, heur)
        if scores is None:
            return None
        return [s / temperature if s > -1e8 else s for s in scores]

    ns["_net_scores"] = softened
    ns["_prior_temperature"] = float(temperature)
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


def visits_to_q(visits: dict, opt_slot, seq: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per-action Q spread over option tokens, visit-weighted.

    Unexplored slots stay at zero and are masked out at training time; a raw
    zero would otherwise read as a neutral value rather than "no information".
    """
    q = np.zeros(seq, dtype=np.float32)
    wsum = np.zeros(seq, dtype=np.float32)
    for action, payload in visits.items():
        n_vis, q_val = payload if isinstance(payload, tuple) else (payload, 0.0)
        for option_index in action:
            slot = opt_slot[option_index] if option_index < len(opt_slot) else -1
            if slot >= 0:
                q[slot] += float(q_val) * float(n_vis)
                wsum[slot] += float(n_vis)
    nz = wsum > 0
    q[nz] /= wsum[nz]
    # Q==0 is a valid teacher estimate, so the visited mask must be explicit.
    # Inferring coverage with ``q != 0`` silently drops exactly-neutral actions.
    return q, nz


def visits_to_pi(visits: dict, opt_slot, seq: int) -> np.ndarray | None:
    """Root visit counts per action -> distribution over option tokens.

    A multi-select action spreads its visits across its members, which mirrors
    how `_gen_candidates` scores a candidate set by the sum of its members.
    """
    pi = np.zeros(seq, dtype=np.float32)
    for action, payload in visits.items():
        count = payload[0] if isinstance(payload, tuple) else payload
        slots = [opt_slot[index] if index < len(opt_slot) else -1
                 for index in action]
        # An action tuple is one semantic object.  Keeping only the members
        # that survived option truncation would turn it into a different legal
        # action, so discard the whole tuple instead.
        if not slots or any(slot < 0 for slot in slots):
            continue
        share = float(count) / max(len(action), 1)
        for slot in slots:
            pi[slot] += share
    total = float(pi.sum())
    if total <= 0.0:
        return None
    return pi / total


def visits_to_action_targets(visits: dict, opt_slot,
                             max_candidates: int = 16,
                             max_members: int = 24
                             ) -> tuple[np.ndarray, ...]:
    """Preserve the teacher's exact action tuples and their Q/visit targets.

    Per-option marginals cannot distinguish selecting ``{A, B}`` from
    selecting ``{A, C}``, even though discard/search choices are judged as a
    set by the engine.  These fixed-shape arrays retain candidate identity while
    remaining cheap to store in an ``npz`` shard.  Members are encoded as token
    positions, so they align directly with the policy logits used at training.
    """
    if max_candidates < 1 or max_members < 1:
        raise ValueError("action target dimensions must be positive")
    tokens = np.full((max_candidates, max_members), -1, dtype=np.int16)
    sizes = np.zeros(max_candidates, dtype=np.int8)
    q_values = np.zeros(max_candidates, dtype=np.float32)
    visit_counts = np.zeros(max_candidates, dtype=np.int32)
    live = np.zeros(max_candidates, dtype=np.bool_)

    # Candidate generation is capped, but sort deterministically in case a
    # diagnostic caller passes a larger mapping.  Prefer well-explored actions
    # when truncation is unavoidable.
    rows = []
    for action, payload in visits.items():
        count, q_value = payload if isinstance(payload, tuple) else (payload, 0.0)
        if int(count) <= 0:
            continue
        mapped = [opt_slot[index] if index < len(opt_slot) else -1
                  for index in action]
        if (len(mapped) > max_members or any(slot < 0 for slot in mapped)):
            continue
        rows.append((int(count), tuple(mapped), float(q_value)))
    rows.sort(key=lambda row: (-row[0], row[1]))
    for index, (count, mapped, q_value) in enumerate(rows[:max_candidates]):
        tokens[index, :len(mapped)] = mapped
        sizes[index] = len(mapped)
        q_values[index] = q_value
        visit_counts[index] = count
        live[index] = True
    return tokens, sizes, q_values, visit_counts, live


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


def play_game(args_tuple) -> dict:
    (agent_dir, budget, seed, max_steps, prior_temperature,
     out_dir, game_index) = args_tuple
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
                    q_vec, q_mask = visits_to_q(visits, opt_slot, seq)
                    (action_tokens, action_sizes, action_q, action_visits,
                     action_mask) = visits_to_action_targets(
                        visits, opt_slot, max_members=ns["_NF"].MAX_OPT)
                    if pi is not None and bool(((kind == 3) & (mask > 0.5)).any()):
                        records.append({
                            "seat": seat, "kind": kind, "card": card, "scal": scal,
                            "mask": mask, "ctx": ctx, "stype": stype, "pi": pi,
                            "q": q_vec, "q_mask": q_mask,
                            "action_tokens": action_tokens,
                            "action_sizes": action_sizes,
                            "action_q": action_q,
                            "action_visits": action_visits,
                            "action_mask": action_mask,
                            "select_min": int(sel.get("minCount") or 0),
                            "select_max": int(sel.get("maxCount") or 1),
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

    # Write from the worker rather than returning the records. Each decision
    # carries a 53x32 float `scal`, so a game is roughly a megabyte of numpy;
    # returning that through the Pool's pipe is what produced the
    # BrokenPipeError that killed the previous attempt at 8% completion.
    # Returning a path and a count keeps the IPC payload at a few bytes.
    if not records:
        return {"path": None, "decisions": 0, "seat_rewards": outcome}
    path = os.path.join(out_dir, f"exit_g{game_index:05d}.npz")
    np.savez_compressed(
        path,
        kind=np.stack([r["kind"] for r in records]).astype(np.int8),
        card=np.stack([r["card"] for r in records]).astype(np.int16),
        scal=np.stack([r["scal"] for r in records]).astype(np.float32),
        mask=np.stack([r["mask"] for r in records]).astype(np.float32),
        ctx=np.array([r["ctx"] for r in records], dtype=np.int16),
        stype=np.array([r["stype"] for r in records], dtype=np.int16),
        pi=np.stack([r["pi"] for r in records]).astype(np.float32),
        q=np.stack([r["q"] for r in records]).astype(np.float32),
        q_mask=np.stack([r["q_mask"] for r in records]).astype(np.bool_),
        action_tokens=np.stack([r["action_tokens"] for r in records]).astype(np.int16),
        action_sizes=np.stack([r["action_sizes"] for r in records]).astype(np.int8),
        action_q=np.stack([r["action_q"] for r in records]).astype(np.float32),
        action_visits=np.stack([r["action_visits"] for r in records]).astype(np.int32),
        action_mask=np.stack([r["action_mask"] for r in records]).astype(np.bool_),
        select_min=np.array([r["select_min"] for r in records], dtype=np.int8),
        select_max=np.array([r["select_max"] for r in records], dtype=np.int8),
        z=np.array([r["z"] for r in records], dtype=np.float32),
    )
    return {"path": path, "decisions": len(records), "seat_rewards": outcome}


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
    parser.add_argument("--teacher-cap", type=int, default=16,
                        help="candidate cap for the TRAINING teacher only. "
                             "Keep 16 unless a new controlled gate overturns "
                             "the prior cap-32 failure.")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3000)
    # One shard per game now: workers write their own, so there is no central
    # buffer to size. Shards are small and the trainer streams them anyway.
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
    patch_teacher(agent_dir, args.teacher_cap)
    print(f"teacher: cap={args.teacher_cap} (deployment stays 16), "
          f"emitting per-action Q", flush=True)

    os.makedirs(args.out, exist_ok=True)
    jobs = [(agent_dir, args.budget, i, args.max_steps, args.prior_temperature,
             args.out, i)
            for i in range(args.games)]
    written = 0
    shards = 0
    began = time.time()

    context = mp.get_context("spawn")
    with context.Pool(args.workers) as pool:
        for done, summary in enumerate(pool.imap_unordered(play_game, jobs), 1):
            if summary.get("path"):
                shards += 1
                written += summary["decisions"]
            if done % 10 == 0 or done == len(jobs):
                rate = (time.time() - began) / done
                print(f"{done}/{len(jobs)} games, {written} decisions, "
                      f"{shards} shards, {(time.time() - began) / 60:.1f} min, "
                      f"eta {rate * (len(jobs) - done) / 60:.0f} min", flush=True)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"done: {written} search-labelled decisions in {shards} shards "
          f"under {args.out}", flush=True)


if __name__ == "__main__":
    main()
