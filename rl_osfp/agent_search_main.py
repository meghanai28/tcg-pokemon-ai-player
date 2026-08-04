"""Kaggle entry point for the determinized-search agent.

The search-free policy spends one forward pass (~26 ms) per decision against a
~600 s episode budget.  This agent spends that budget instead: it determinizes
the hidden information into several consistent worlds, runs PUCT over the
engine's native search API in each, and plays the action with the most
aggregated root visits.

Time management is the safety-critical part.  The episode budget is shared
across every decision in the game, so the agent tracks its own consumption and
shrinks the per-move allowance as the budget depletes.  Running out of time
mid-game would be far worse than searching shallowly, and every failure path
falls back to a legal heuristic action rather than raising.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

try:
    AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    AGENT_DIR = "/kaggle_simulations/agent"
    if not os.path.isdir(AGENT_DIR):
        AGENT_DIR = os.getcwd()
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

# Total wall clock the harness allows for one episode, and the fraction we are
# willing to spend on search. The rest is head-room for engine and I/O.
EPISODE_BUDGET_S = float(os.environ.get("PTCG_EPISODE_BUDGET", "600"))
SEARCH_FRACTION = 0.70
EXPECTED_DECISIONS = 150
MIN_MOVE_S = 0.05
MAX_MOVE_S = 6.0
WORLDS = 4
# Root priors only. Measured: priors are neutral-to-slightly-positive and cost
# one net call per decision; value leaves cost one call per simulation and lost
# 6-13 to unguided search, because simulation count is what wins.
PRIOR_WEIGHT = float(os.environ.get("PTCG_PRIOR_WEIGHT", "0.7"))
VALUE_WEIGHT = float(os.environ.get("PTCG_VALUE_WEIGHT", "0.0"))

_STATE: dict = {}


def _load() -> dict:
    if _STATE:
        return _STATE
    import search as S

    with open(os.path.join(AGENT_DIR, "deck.csv"), encoding="utf-8") as source:
        deck = [int(line) for line in source if line.strip()]
    if len(deck) != 60:
        raise ValueError("deck.csv must contain exactly 60 cards")
    with open(os.path.join(AGENT_DIR, "meta_decks.json"), encoding="utf-8") as source:
        meta = json.load(source)

    engine = S.Engine()
    # The prior model is optional: without model.npz this is pure search, which
    # is a supported configuration rather than a failure.
    guide = None
    model_path = os.path.join(AGENT_DIR, "model.npz")
    if os.path.isfile(model_path):
        try:
            guide = S.PolicyGuide(
                model_path,
                {int(c["cardId"]): c for c in json.loads(engine.lib.AllCard().decode())},
                {int(a["attackId"]): a for a in json.loads(engine.lib.AllAttack().decode())},
                prior_weight=PRIOR_WEIGHT, value_weight=VALUE_WEIGHT,
            )
        except Exception:
            guide = None
    _STATE.update({
        "guide": guide,
        "S": S,
        "engine": engine,
        "deck": deck,
        "attack_db": {
            int(a["attackId"]): a
            for a in json.loads(engine.lib.AllAttack().decode())
        },
        "opponent": S.OpponentModel(
            [d["cards"] for d in meta], [float(d.get("appearances", 1)) for d in meta]
        ),
        "rng": random.Random(0xC0FFEE),
        "spent": 0.0,
        "decisions": 0,
    })
    return _STATE


def _fallback(select: dict) -> list[int]:
    options = select.get("option") or []
    if not options:
        return []
    low = max(0, min(int(select.get("minCount", 1) or 0), len(options)))
    high = max(low, min(int(select.get("maxCount", max(low, 1)) or 0), len(options)))
    return list(range(low if low else min(1, high)))


def _allowance(state: dict) -> float:
    """Per-move time, shrinking as the shared episode budget depletes."""
    remaining = EPISODE_BUDGET_S * SEARCH_FRACTION - state["spent"]
    if remaining <= 0:
        return 0.0
    left = max(1, EXPECTED_DECISIONS - state["decisions"])
    return max(MIN_MOVE_S, min(MAX_MOVE_S, remaining / left))


def agent(observation):
    select = None
    try:
        state = _load()
        select = observation.get("select") if hasattr(observation, "get") else observation["select"]
        if select is None:
            return list(state["deck"])

        options = select.get("option") or []
        if not options:
            return []
        low = max(0, min(int(select.get("minCount", 1) or 0), len(options)))
        high = max(low, min(int(select.get("maxCount", max(low, 1)) or 0), len(options)))
        if low == high == len(options):
            return list(range(len(options)))  # forced, nothing to search

        allowance = _allowance(state)
        began = time.perf_counter()
        action = None
        if allowance > MIN_MOVE_S:
            current = observation.get("current") or {}
            action = state["S"].search_move(
                state["engine"], state["attack_db"], observation,
                int(current.get("yourIndex", 0)), state["deck"],
                state["opponent"], state["rng"],
                deadline=began + allowance, worlds=WORLDS,
                guide=state["guide"],
            )
        if action is None:
            action = state["S"].heuristic_action(
                state["engine"], state["attack_db"], select, state["rng"]
            )
        state["spent"] += time.perf_counter() - began
        state["decisions"] += 1
        return list(action) if action else _fallback(select)
    except Exception:
        try:
            if select is None:
                select = (observation.get("select") or {}) if hasattr(observation, "get") else {}
            if not select:
                return list(_load()["deck"])
            return _fallback(select)
        except Exception:
            return [0]
