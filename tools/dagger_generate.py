"""Collect learner-state, search-labelled data for BC-centered improvement.

This is deliberately different from ordinary self-play Expert Iteration.  The
current/deployed search agent chooses trajectory actions, while an independent
cap-16 teacher searches the *same observed state* with a flattened prior and
only supplies a supervised target.  That is the DAgger separation needed to
train on states the learner actually reaches without letting the teacher hide
its mistakes by controlling the trajectory.

The teacher stores both backwards-compatible per-option marginals and exact
candidate action tuples with Q/visit counts.  Terminal reward is diagnostic
only; it is not a policy label.

Example smoke run::

    .venv/bin/python tools/dagger_generate.py \
      --archive agent/runs/packages/candidate_bug_bounded8.tar.gz \
      --games 4 --teacher-budget .35 --behavior-budget .15 --workers 2 \
      --out data/dagger_bug_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import runpy
import shutil
import sys
import tarfile
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.exit_generate import (
    cached_shell,
    patch_teacher,
    soften_prior,
    visits_to_action_targets,
    visits_to_pi,
    visits_to_q,
)


def stable_id(value) -> int:
    raw = str(value or "").encode("utf-8", "replace")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def load_meta_decks(path: str) -> list[tuple[str, list[int]]]:
    namespace = runpy.run_path(path)
    decks = namespace.get("META_DECKS")
    if not isinstance(decks, dict) or not decks:
        raise ValueError(f"{path} must define a non-empty META_DECKS dict")
    result = []
    for name, cards in sorted(decks.items()):
        if not isinstance(cards, list) or len(cards) != 60:
            raise ValueError(f"{path}:{name} is not a 60-card list")
        result.append((str(name), [int(card) for card in cards]))
    return result


def _perfect_info_sampler(ns: dict, truth: dict):
    """Wrap a shell's _sample_world so the opponent's REAL hand is used.

    Only the hand is revealed. It is hidden-but-determined: the opponent knows
    it, we do not, and it is exactly the information a determinized search gets
    wrong. Deck ORDER stays sampled, because no player observes it -- it is
    genuinely stochastic, and pretending to know it would teach the student to
    predict shuffles.

    truth["hand"] is refreshed from the opponent's own observation each time
    they act. If its length disagrees with the opponent's reported handCount at
    decision time the capture is stale (an effect moved cards during our turn),
    and we fall back to sampling rather than feeding the search a world that
    cannot exist. truth["used"]/["stale"] count both cases.
    """
    original = ns["_sample_world"]
    multiset_sub = ns["_multiset_sub"]
    fit_length = ns["_fit_length"]
    visible_cards = ns["_visible_cards"]

    def sample(obs, me, opp_model, rng):
        hand = truth.get("hand")
        cur = obs.get("current") or {}
        opl = (cur.get("players") or [None, None])[1 - me]
        if not hand or opl is None or len(hand) != (opl.get("handCount") or 0):
            truth["stale"] = truth.get("stale", 0) + 1
            return original(obs, me, opp_model, rng)

        my_deck, my_prize, opp_deck, opp_prize, _opp_hand, opp_active = \
            original(obs, me, opp_model, rng)

        # Rebuild the opponent's unseen pool with the true hand removed, then
        # re-draw prizes and deck from what genuinely remains.
        guess = opp_model.guess_list(visible_cards(opl))
        seen = visible_cards(opl) + list(hand)
        remaining = multiset_sub(guess, seen)
        rng.shuffle(remaining)
        n_prize = len(opp_prize)
        new_prize = fit_length(remaining[:n_prize], n_prize, rng, guess)
        new_deck = fit_length(remaining[n_prize:], len(opp_deck), rng, guess)
        truth["used"] = truth.get("used", 0) + 1
        return my_deck, my_prize, new_deck, new_prize, list(hand), opp_active

    ns["_sample_world"] = sample


def _privileged_opponent_model(game, opponent_deck) -> None:
    """Replace a search GameState's opponent-deck guess with the truth.

    Only the DECKLIST is revealed. The opponent's hand and deck order stay
    hidden and are still sampled per determinized world, so the teacher is a
    better-informed searcher rather than an oracle. That distinction matters:
    a full-information expert plays around cards the student cannot see, and
    the student then learns to imitate moves it has no basis to choose.
    """
    try:
        true_list = list(opponent_deck)
        game.opp_model.guess_list = lambda _visible, _d=true_list: list(_d)
    except Exception:
        pass


def _search(ns: dict, game, obs: dict, budget: float,
            visits: dict | None = None):
    if budget <= 0:
        return None
    me = int((obs.get("current") or {}).get("yourIndex", 0))
    try:
        return ns["_search_move"](
            obs, me, game.opp_model, time.perf_counter() + budget,
            game.rng, collect_policy=visits)
    except Exception:
        return None
    finally:
        try:
            ns["_LIB"].SearchEnd(ns["_CTX"])
        except Exception:
            pass


def _best_q_action(visits: dict):
    live = [(tuple(action), int(payload[0]), float(payload[1]))
            for action, payload in visits.items() if int(payload[0]) > 0]
    return max(live, key=lambda row: (row[2], row[1], row[0]))[0] if live else None


def merge_stable_visits(left: dict, right: dict) -> tuple[dict, float] | None:
    """Merge replicated search labels only when their best-Q action agrees.

    Search Q at the deployment budget is noisy and can flip when only the
    prior temperature/seed changes.  Training on those flips manufactures a
    target.  Requiring agreement and retaining only actions evaluated by both
    searches is a cheap abstention rule; visit-weighted averaging then reduces
    the variance of the Q target that survives.
    """
    if _best_q_action(left) != _best_q_action(right):
        return None
    merged = {}
    differences = []
    for action in left.keys() & right.keys():
        left_n, left_q = left[action]
        right_n, right_q = right[action]
        left_n, right_n = int(left_n), int(right_n)
        if left_n <= 0 or right_n <= 0:
            continue
        total = left_n + right_n
        merged[action] = (
            total, (left_n * float(left_q) + right_n * float(right_q)) / total)
        differences.append(abs(float(left_q) - float(right_q)))
    if len(merged) < 2:
        return None
    disagreement = float(np.mean(differences)) if differences else 0.0
    return merged, disagreement


def _legal_or_fallback(ns: dict, action, sel: dict, game):
    action = ns["_validate"](action, sel)
    if action is None:
        action = ns["_validate"](ns["_heuristic_action"](sel, game.rng), sel)
    if action is None:
        n = len(sel.get("option") or [])
        k = max(1, min(sel.get("minCount", 1) or 1, n))
        action = list(range(k))
    return action


def _teacher_record(ns: dict, obs: dict, sel: dict, seat: int,
                    visits: dict, behavior_action) -> dict | None:
    me = int((obs.get("current") or {}).get("yourIndex", seat))
    # The learner package pins the exact feature ABI used by its checkpoint.
    # Preserve public logs for v3's history bags; no hidden world is exposed.
    feature_state = {"current": obs["current"], "select": sel,
                     "decklist": ns["MY_DECK"] or [],
                     "logs": obs.get("logs") or []}
    encoded = ns["_NF"].encode(
        feature_state, me, ns["CARD"], ns["ATTACK"], None)
    kind, card, scal, mask, opt_slot = encoded[:5]
    ctx = int(sel.get("context") or 0)
    stype = int(sel.get("type") or 0)
    pi = visits_to_pi(visits, opt_slot, ns["_NF"].SEQ)
    if pi is None:
        return None
    q, q_mask = visits_to_q(visits, opt_slot, ns["_NF"].SEQ)
    (action_tokens, action_sizes, action_q, action_visits,
     action_mask) = visits_to_action_targets(
        visits, opt_slot, max_members=ns["_NF"].MAX_OPT)
    if hasattr(ns["_NF"], "EMPTY_SLOT"):
        empty_rows = action_mask & (action_sizes == 0)
        action_tokens[empty_rows, 0] = ns["_NF"].EMPTY_SLOT
        action_sizes[empty_rows] = 1
    if int(action_mask.sum()) < 2:
        return None
    behavior_tokens = np.full(ns["_NF"].MAX_OPT, -1, dtype=np.int16)
    mapped = [opt_slot[index] if index < len(opt_slot) else -1
              for index in (behavior_action or [])]
    if mapped and all(token >= 0 for token in mapped):
        behavior_tokens[:len(mapped)] = mapped
        behavior_size = len(mapped)
    else:
        behavior_size = 0
    record = {
        "seat": seat,
        "kind": kind,
        "card": card,
        "scal": scal,
        "mask": mask,
        "ctx": ctx,
        "stype": stype,
        "pi": pi,
        "q": q,
        "q_mask": q_mask,
        "action_tokens": action_tokens,
        "action_sizes": action_sizes,
        "action_q": action_q,
        "action_visits": action_visits,
        "action_mask": action_mask,
        "select_min": int(sel.get("minCount") or 0),
        "select_max": int(sel.get("maxCount") or 1),
        "behavior_tokens": behavior_tokens,
        "behavior_size": behavior_size,
    }
    if getattr(ns["_NF"], "FEATURE_VERSION", 1) == 3:
        for key, value in zip(ns["_NF"].V3_FIELDS, encoded[5:]):
            record[key] = value
    return record


def play_game(job: tuple) -> dict:
    (behavior_dir, teacher_dir, teacher_repeat_dir, opponent_dir, teacher_budget,
     behavior_budget, opponent_budget, prior_temperature,
     repeat_temperature, state_policy,
     seed, max_steps, out_dir, game_index, opponent_name,
     opponent_deck) = job
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    import kaggle_environments as ke

    behavior = cached_shell(behavior_dir)
    teacher = cached_shell(teacher_dir)
    teacher_repeat = (cached_shell(teacher_repeat_dir)
                      if teacher_repeat_dir else None)
    opponent = cached_shell(opponent_dir)
    soften_prior(behavior, 1.0)
    soften_prior(teacher, prior_temperature)
    if teacher_repeat is not None:
        soften_prior(teacher_repeat, repeat_temperature)
    soften_prior(opponent, 1.0)
    own_deck = list(behavior["MY_DECK"])
    teacher["MY_DECK"] = list(own_deck)
    if teacher_repeat is not None:
        teacher_repeat["MY_DECK"] = list(own_deck)
    opponent["MY_DECK"] = list(opponent_deck)

    controlled_seat = game_index % 2
    records: list[dict] = []
    counters = {"teacher_failures": 0, "behavior_failures": 0,
                "teacher_unstable": 0, "perfect_used": 0, "perfect_stale": 0}

    controlled_state = {"behavior": None, "teacher": None,
                        "teacher_repeat": None}

    def controlled_agent(obs):
        sel = obs.get("select")
        if sel is None:
            controlled_state["behavior"] = behavior["GameState"]()
            controlled_state["teacher"] = teacher["GameState"]()
            if teacher_repeat is not None:
                controlled_state["teacher_repeat"] = (
                    teacher_repeat["GameState"]())
            controlled_state["behavior"].rng = random.Random(seed * 7919 + 17)
            controlled_state["teacher"].rng = random.Random(seed * 104729 + 29)
            # Privileged teacher: during generation we KNOW the opponent's
            # decklist, but OpponentModel.guess_list normally infers it by
            # matching visible cards against META_DECKS. That guess is wrong
            # early, when few opponent cards are visible and the label matters
            # most. Handing the teacher the true list removes deck-identity
            # error from its determinized worlds while leaving hand and deck
            # order hidden, so the labels stay learnable from the student's own
            # observation. The behaviour policy is deliberately NOT given this.
            _privileged_opponent_model(
                controlled_state["teacher"], opponent_deck)
            if teacher_repeat is not None:
                _privileged_opponent_model(
                    controlled_state["teacher_repeat"], opponent_deck)
            # Perfect information for the TEACHER only: its determinized worlds
            # use the opponent's real hand instead of a sampled one. The
            # behaviour policy stays blind, so the states we label are still
            # the ones our deployed agent actually reaches.
            _perfect_info_sampler(teacher, opponent_truth)
            # teacher_repeat is deliberately NOT privileged. The existing
            # merge_stable_visits check keeps a label only when both teachers
            # agree, so pairing a perfect-information teacher with an ordinary
            # determinized one turns that check into the robustness filter: a
            # label survives only if the action is right WITHOUT knowing the
            # opponent's hand. Moves that are correct only because the teacher
            # peeked are exactly the ones the student cannot learn, and they
            # get dropped here.
            if teacher_repeat is not None:
                controlled_state["teacher_repeat"].rng = random.Random(
                    seed * 130363 + 37)
            return list(own_deck)
        options = sel.get("option") or []
        n = len(options)
        kmax = max(1, min(sel.get("maxCount", 1), n)) if n else 0
        if n == 0:
            return []
        if n == 1:
            return [0]
        if kmax >= n and sel.get("minCount", 0) >= n:
            return list(range(n))

        behavior_game = controlled_state["behavior"]
        teacher_game = controlled_state["teacher"]
        teacher_repeat_game = controlled_state["teacher_repeat"]
        behavior_game.calls += 1
        teacher_game.calls += 1
        behavior_action = _search(
            behavior, behavior_game, obs, behavior_budget)
        if behavior_action is None:
            counters["behavior_failures"] += 1
        behavior_action = _legal_or_fallback(
            behavior, behavior_action, sel, behavior_game)

        visits: dict = {}
        teacher_action = _search(
            teacher, teacher_game, obs, teacher_budget, visits)
        if teacher_action is None or not visits:
            counters["teacher_failures"] += 1
        else:
            try:
                disagreement = 0.0
                if teacher_repeat is not None:
                    repeat_visits: dict = {}
                    repeat_action = _search(
                        teacher_repeat, teacher_repeat_game, obs,
                        teacher_budget, repeat_visits)
                    merged = (merge_stable_visits(visits, repeat_visits)
                              if repeat_action is not None else None)
                    if merged is None:
                        counters["teacher_unstable"] += 1
                        raise ValueError("replicated teacher label is unstable")
                    visits, disagreement = merged
                record = _teacher_record(
                    teacher, obs, sel, controlled_seat, visits,
                    behavior_action)
                if record is not None:
                    record["teacher_repeats"] = (
                        2 if teacher_repeat is not None else 1)
                    record["teacher_q_disagreement"] = disagreement
                    records.append(record)
            except ValueError:
                pass
            except Exception:
                counters["teacher_failures"] += 1

        if state_policy == "teacher" and teacher_action is not None:
            return _legal_or_fallback(
                teacher, teacher_action, sel, teacher_game)
        return behavior_action

    # Refreshed from the opponent's own observation every time they act.
    opponent_truth: dict = {"hand": None, "used": 0, "stale": 0}
    opponent_state = {"game": None}

    def opponent_agent(obs):
        sel = obs.get("select")
        # The acting player sees their own hand; the other side sees only a
        # count. This is the one place the opponent's hand is observable.
        try:
            cur = obs.get("current") or {}
            mine = (cur.get("players") or [])[cur.get("yourIndex", 0)]
            hand = mine.get("hand")
            if isinstance(hand, list):
                opponent_truth["hand"] = [c["id"] for c in hand if c]
        except Exception:
            pass
        if sel is None:
            opponent_state["game"] = opponent["GameState"]()
            opponent_state["game"].rng = random.Random(seed * 15485863 + 43)
            return list(opponent_deck)
        options = sel.get("option") or []
        n = len(options)
        if n == 0:
            return []
        if n == 1:
            return [0]
        kmax = max(1, min(sel.get("maxCount", 1), n))
        if kmax >= n and sel.get("minCount", 0) >= n:
            return list(range(n))
        game = opponent_state["game"]
        game.calls += 1
        action = _search(opponent, game, obs, opponent_budget)
        return _legal_or_fallback(opponent, action, sel, game)

    agents = ([controlled_agent, opponent_agent] if controlled_seat == 0
              else [opponent_agent, controlled_agent])
    env = ke.make("cabt", configuration={"episodeSteps": max_steps}, debug=False)
    env.run(agents)
    rewards = [entry.reward for entry in env.steps[-1]]
    reward = 0.0
    if len(rewards) > controlled_seat and rewards[controlled_seat] is not None:
        reward = float(np.sign(rewards[controlled_seat]))

    done_path = os.path.join(out_dir, f"dagger_g{game_index:05d}.done.json")
    if not records:
        result = {"path": None, "decisions": 0, "reward": reward,
                  "opponent": opponent_name, **{**counters, "perfect_used": opponent_truth.get("used", 0), "perfect_stale": opponent_truth.get("stale", 0)}}
        temporary = done_path + ".partial"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(result, handle, sort_keys=True)
        os.replace(temporary, done_path)
        return result
    for record in records:
        record["z"] = reward
    path = os.path.join(out_dir, f"dagger_g{game_index:05d}.npz")
    rows = len(records)
    arrays = dict(
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
        behavior_tokens=np.stack([r["behavior_tokens"] for r in records]).astype(np.int16),
        behavior_size=np.array([r["behavior_size"] for r in records], dtype=np.int8),
        teacher_repeats=np.array(
            [r.get("teacher_repeats", 1) for r in records], dtype=np.int8),
        teacher_q_disagreement=np.array(
            [r.get("teacher_q_disagreement", 0.0) for r in records],
            dtype=np.float32),
        z=np.full(rows, reward, dtype=np.float32),
        group=np.full(rows, game_index, dtype=np.uint64),
        seat=np.full(rows, controlled_seat, dtype=np.int8),
        pilot=np.zeros(rows, dtype=np.uint64),
        elo=np.full(rows, 1000.0, dtype=np.float32),
        deck=np.full(rows, stable_id(tuple(sorted(own_deck))), dtype=np.uint64),
    )
    if "bag_card" in records[0]:
        arrays.update(
            bag_card=np.stack([r["bag_card"] for r in records]).astype(np.int16),
            bag_count=np.stack([r["bag_count"] for r in records]).astype(np.uint8),
            bag_kind=np.stack([r["bag_kind"] for r in records]).astype(np.int8),
            bag_scal=np.stack([r["bag_scal"] for r in records]).astype(np.float32),
            bag_mask=np.stack([r["bag_mask"] for r in records]).astype(np.float32),
        )
    temporary = path + ".partial"
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    result = {"path": path, "decisions": rows, "reward": reward,
              "opponent": opponent_name, **{**counters, "perfect_used": opponent_truth.get("used", 0), "perfect_stale": opponent_truth.get("stale", 0)}}
    temporary = done_path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)
    os.replace(temporary, done_path)
    return result


def _extract(archive: str, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(directory, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--teacher-budget", type=float, default=0.5)
    parser.add_argument("--behavior-budget", type=float, default=0.2)
    parser.add_argument("--opponent-budget", type=float, default=0.2)
    parser.add_argument("--prior-temperature", type=float, default=3.0)
    parser.add_argument(
        "--repeat-temperature", type=float, default=0.0,
        help="if positive, query an independent teacher at this temperature "
             "and keep only best-Q-agreeing labels (recommended for real data)")
    parser.add_argument("--teacher-cap", type=int, default=16)
    parser.add_argument(
        "--state-policy", choices=("learner", "teacher"), default="learner",
        help="learner is DAgger and the default; teacher exists only as a "
             "controlled Expert-Iteration comparison")
    parser.add_argument("--opponent-meta", default=os.path.join(
        ROOT, "harness", "meta", "meta_decks_aug12.py"))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=917)
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "dagger_shards"))
    parser.add_argument("--resume", action="store_true",
                        help="retain completed per-game shards and run only missing games")
    args = parser.parse_args()
    if not 1 <= args.workers <= 5:
        parser.error("--workers must be in [1,5]")
    if args.games < 1 or min(args.teacher_budget, args.behavior_budget,
                             args.opponent_budget) <= 0:
        parser.error("games and all search budgets must be positive")
    if args.prior_temperature <= 0:
        parser.error("--prior-temperature must be positive")
    if args.repeat_temperature < 0:
        parser.error("--repeat-temperature must be nonnegative")

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    opponents = load_meta_decks(args.opponent_meta)
    staging = tempfile.mkdtemp(prefix="daggergen-")
    try:
        behavior_dir = os.path.join(staging, "behavior")
        teacher_dir = os.path.join(staging, "teacher")
        teacher_repeat_dir = (os.path.join(staging, "teacher_repeat")
                              if args.repeat_temperature else None)
        opponent_dir = os.path.join(staging, "opponent")
        _extract(args.archive, behavior_dir)
        shutil.copytree(behavior_dir, teacher_dir)
        if teacher_repeat_dir:
            shutil.copytree(behavior_dir, teacher_repeat_dir)
        shutil.copytree(behavior_dir, opponent_dir)
        patch_teacher(teacher_dir, args.teacher_cap)
        if teacher_repeat_dir:
            patch_teacher(teacher_repeat_dir, args.teacher_cap)
        os.makedirs(args.out, exist_ok=True)
        jobs = []
        for game_index in range(args.games):
            expected = os.path.join(
                args.out, f"dagger_g{game_index:05d}.done.json")
            if args.resume and os.path.isfile(expected):
                continue
            opponent_name, opponent_deck = opponents[game_index % len(opponents)]
            jobs.append((
                behavior_dir, teacher_dir, teacher_repeat_dir, opponent_dir,
                args.teacher_budget, args.behavior_budget, args.opponent_budget,
                args.prior_temperature, args.repeat_temperature,
                args.state_policy,
                args.seed + game_index, args.max_steps, args.out, game_index,
                opponent_name, opponent_deck))
        print(
            f"DAgger: learner states; teacher cap={args.teacher_cap}, "
            f"T={args.prior_temperature:g}, budgets teacher/behavior/opponent="
            f"{args.teacher_budget:g}/{args.behavior_budget:g}/"
            f"{args.opponent_budget:g}; "
            f"repeatT={args.repeat_temperature:g}; "
            f"{len(opponents)} field decks; {len(jobs)}/{args.games} games pending",
            flush=True)
        written = failures = unstable = wins = shards = 0
        pi_used = pi_stale = 0
        began = time.time()
        context = mp.get_context("spawn")
        with context.Pool(args.workers) as pool:
            for done, result in enumerate(pool.imap_unordered(play_game, jobs), 1):
                written += result["decisions"]
                failures += result["teacher_failures"]
                unstable += result["teacher_unstable"]
                pi_used += result.get("perfect_used", 0)
                pi_stale += result.get("perfect_stale", 0)
                wins += int(result["reward"] > 0)
                shards += int(bool(result.get("path")))
                if done % 10 == 0 or done == len(jobs):
                    rate = (time.time() - began) / done
                    print(
                        f"{done}/{len(jobs)} games; {written} labels; "
                        f"{wins}-{done-wins}; teacher failures={failures}; "
                        f"unstable={unstable}; "
                        f"perfect={pi_used}/{pi_used + pi_stale}; "
                        f"{shards} shards; eta "
                        f"{rate * (len(jobs) - done) / 60:.0f} min",
                        flush=True)
        all_shards = sorted(
            os.path.join(args.out, name) for name in os.listdir(args.out)
            if re.fullmatch(r"dagger_g[0-9]{5}\.npz", name))
        all_done = [name for name in os.listdir(args.out)
                    if re.fullmatch(r"dagger_g[0-9]{5}\.done\.json", name)]
        if len(all_done) != args.games:
            raise RuntimeError(
                f"DAgger incomplete: found {len(all_done)}/{args.games} games")
        total_labels = 0
        for path in all_shards:
            with np.load(path) as source:
                total_labels += len(source["ctx"])
        complete = {
            "archive_sha256": hashlib.sha256(
                open(args.archive, "rb").read()).hexdigest(),
            "games": args.games, "labels": total_labels,
            "teacher_budget": args.teacher_budget,
            "behavior_budget": args.behavior_budget,
            "opponent_budget": args.opponent_budget,
            "prior_temperature": args.prior_temperature,
            "repeat_temperature": args.repeat_temperature,
            "state_policy": args.state_policy, "seed": args.seed,
        }
        marker = os.path.join(args.out, "_COMPLETE.json")
        marker_tmp = marker + ".partial"
        with open(marker_tmp, "w", encoding="utf-8") as handle:
            json.dump(complete, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(marker_tmp, marker)
        print(f"done: {total_labels} learner-state labels in {len(all_shards)} "
              f"shards under {args.out}", flush=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
