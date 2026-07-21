"""Pokemon TCG AI Battle Challenge agent.

Determinized Information-Set MCTS over the official cabt engine, bundled with
the submission (cg/).  The core loop:

  1. From the agent observation, reconstruct every *possible world* consistent
     with what we can see (our unseen cards split into deck/prizes; the
     opponent's hidden zones filled from a best-matching meta decklist).
  2. Seed the engine's native search API (SearchBegin) with each world and run
     open-loop UCT: replay action paths through SearchStep, expand with
     heuristic priors, evaluate leaves with a handcrafted value function.
  3. Aggregate root statistics across worlds and play the most-visited action.

Every stage is wrapped so that on any failure (engine missing, prediction
rejected, out of time) the agent falls back to a fast heuristic policy and
always returns a legal selection.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import random
import sys
import time
from collections import Counter

# The Kaggle agent runner execs this file as a string, so __file__ is not
# defined there; locally (importlib) it is. Fall back to the documented agent
# directory, then to cwd.
try:
    AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    AGENT_DIR = "/kaggle_simulations/agent"
    if not os.path.isdir(AGENT_DIR):
        AGENT_DIR = os.getcwd()
_DEBUG = bool(os.environ.get("PTCG_DEBUG"))


def _dbg(*args):
    if _DEBUG:
        print("[agent]", *args, file=sys.stderr, flush=True)

# --------------------------------------------------------------------------
# Enums (from the official API docs)
# --------------------------------------------------------------------------
CT_POKEMON, CT_ITEM, CT_TOOL, CT_SUPPORTER, CT_STADIUM, CT_BASIC_ENERGY, CT_SPECIAL_ENERGY = range(7)

OT_NUMBER, OT_YES, OT_NO, OT_CARD, OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY, OT_PLAY, \
    OT_ATTACH, OT_EVOLVE, OT_ABILITY, OT_DISCARD, OT_RETREAT, OT_ATTACK, OT_END, \
    OT_SKILL, OT_SPECIAL_CONDITION = range(17)

CTX_MAIN = 0
CTX_DISCARD_SET = {8, 26, 27, 29, 30}      # DISCARD, DISCARD_ENERGY_CARD, ...
CTX_TO_HAND = 7
CTX_DRAW_COUNT = 38
CTX_IS_FIRST = 41
CTX_MULLIGAN = 42
CTX_ACTIVATE = 43

# --------------------------------------------------------------------------
# Engine bindings
# --------------------------------------------------------------------------
_LIB = None
_CTX = None
_NET = None      # learned policy/value net (numpy); None => pure heuristic priors
_NF = None       # nn_features module
NET_TIME_BUDGET_S = 90.0   # max seconds/game the net may consume (of ~600)
NET_LEAF_BATCH = 24        # leaves gathered per batched value-head evaluation
VALUE_TEMP = float(os.environ.get("PTCG_VALUE_TEMP", "0.5"))    # shrink saturated values
VALUE_BLEND = float(os.environ.get("PTCG_VALUE_BLEND", "0.7"))  # net vs heuristic mix


def _load_net():
    """Load the distilled model if numpy + weights are both present."""
    global _NET, _NF
    path = os.path.join(AGENT_DIR, "model.npz")
    if not os.path.exists(path):
        _dbg("no model.npz; using heuristic priors")
        return
    if AGENT_DIR not in sys.path:
        sys.path.insert(0, AGENT_DIR)
    import numpy as np
    import nn_features
    from nn_infer import NumpyNet
    net = NumpyNet(path)

    # Latency guard. The net is called once per decision (~150 per game) out of
    # a ~600 s episode budget. Benchmark it on THIS machine (Kaggle's 2 vCPUs
    # are slower than a dev box) and refuse to use it if a full game's worth of
    # calls would eat the budget the search needs.
    k = np.zeros((1, nn_features.SEQ), dtype=np.int64)
    c = np.zeros((1, nn_features.SEQ), dtype=np.int64)
    s = np.zeros((1, nn_features.SEQ, nn_features.F), dtype=np.float32)
    m = np.ones((1, nn_features.SEQ), dtype=np.float32)
    z = np.zeros(1, dtype=np.int64)
    net.forward(k, c, s, m, z, z)                      # warm up
    t0 = time.perf_counter()
    for _ in range(3):
        net.forward(k, c, s, m, z, z)
    per_call = (time.perf_counter() - t0) / 3
    projected = per_call * 150
    if projected > NET_TIME_BUDGET_S:
        _dbg(f"net too slow: {per_call*1000:.0f} ms/call -> {projected:.0f} s/game "
             f"(cap {NET_TIME_BUDGET_S} s); using heuristic priors")
        return
    _NF = nn_features
    _NET = net
    _dbg(f"loaded model.npz ({per_call*1000:.0f} ms/call, ~{projected:.0f} s/game)")


def _load_engine():
    global _LIB, _CTX
    if AGENT_DIR not in sys.path:
        sys.path.insert(0, AGENT_DIR)
    from cg.engine import get_lib
    _LIB = get_lib()
    _CTX = _LIB.AgentStart()


def _int_arr(xs):
    return (ctypes.c_int * max(len(xs), 1))(*xs)


# --------------------------------------------------------------------------
# Card database
# --------------------------------------------------------------------------
CARD = {}     # cardId -> dict
ATTACK = {}   # attackId -> dict
BASIC_ENERGY_BY_TYPE = {}


def _load_card_db():
    for c in json.loads(_LIB.AllCard().decode()):
        CARD[c["cardId"]] = c
        if c.get("cardType") == CT_BASIC_ENERGY:
            BASIC_ENERGY_BY_TYPE.setdefault(c.get("energyType"), c["cardId"])
    for a in json.loads(_LIB.AllAttack().decode()):
        ATTACK[a["attackId"]] = a


def _card(cid):
    return CARD.get(cid, {})


def _max_attack_damage(cid):
    best = 0
    for aid in _card(cid).get("attacks", []) or []:
        best = max(best, ATTACK.get(aid, {}).get("damage", 0) or 0)
    return best


# --------------------------------------------------------------------------
# Our deck + opponent archetype library (public consensus ladder lists)
# --------------------------------------------------------------------------
def _load_deck():
    path = os.path.join(AGENT_DIR, "deck.csv")
    with open(path) as f:
        return [int(line) for line in f if line.strip()]


MY_DECK = None

# Opponent archetype library, auto-mined from 2,091 ladder replays
# (tools/mine_decks.py). Weights are appearance counts.
META_DECKS = {
    "mined_0": [5, 5, 13, 19, 19, 19, 19, 66, 66, 140, 305, 305, 305, 343, 741, 741, 741, 741, 742, 742, 742, 742, 743, 743, 743, 743, 1079, 1079, 1079, 1081, 1081, 1081, 1081, 1086, 1086, 1086, 1086, 1097, 1129, 1152, 1152, 1152, 1152, 1182, 1182, 1182, 1184, 1197, 1197, 1197, 1225, 1225, 1225, 1225, 1231, 1231, 1231, 1231, 1266, 1266],  # seen 895x, 48% WR
    "mined_1": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 104, 104, 112, 112, 112, 112, 646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 860, 860, 1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1122, 1137, 1152, 1152, 1152, 1152, 1182, 1182, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1231, 1259, 1259, 1259, 1259],  # seen 579x, 60% WR
    "mined_2": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 104, 104, 112, 112, 112, 112, 646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 860, 860, 1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1152, 1152, 1152, 1152, 1161, 1161, 1182, 1182, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1231, 1259, 1259, 1259, 1259],  # seen 293x, 51% WR
    "mined_3": [6, 6, 6, 6, 6, 20, 20, 20, 20, 341, 341, 341, 341, 342, 342, 342, 379, 379, 379, 379, 380, 380, 380, 380, 381, 381, 381, 387, 387, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1142, 1142, 1142, 1142, 1152, 1152, 1152, 1152, 1173, 1173, 1173, 1182, 1182, 1197, 1203, 1225, 1225, 1225, 1227, 1227, 1227, 1227, 1261, 1261],  # seen 193x, 56% WR
    "mined_4": [5, 5, 13, 19, 19, 19, 19, 66, 66, 140, 305, 305, 305, 343, 741, 741, 741, 741, 742, 742, 742, 742, 743, 743, 743, 743, 1079, 1079, 1079, 1081, 1081, 1081, 1086, 1086, 1086, 1086, 1097, 1129, 1152, 1152, 1152, 1152, 1182, 1182, 1182, 1184, 1197, 1197, 1197, 1225, 1225, 1225, 1225, 1231, 1231, 1231, 1231, 1266, 1266, 1266],  # seen 189x, 33% WR
    "mined_5": [1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 15, 15, 15, 15, 400, 400, 400, 400, 401, 401, 401, 401, 414, 414, 431, 431, 432, 463, 463, 1094, 1094, 1094, 1097, 1119, 1119, 1134, 1134, 1134, 1134, 1152, 1152, 1152, 1152, 1159, 1175, 1216, 1216, 1216, 1216, 1217, 1218, 1218, 1218, 1219, 1220, 1220, 1227, 1227, 1257, 1257],  # seen 174x, 66% WR
    "mined_6": [1, 1, 1, 1, 1, 1, 1, 1, 15, 15, 15, 15, 400, 400, 400, 400, 401, 401, 401, 401, 414, 414, 431, 431, 434, 434, 434, 1086, 1086, 1094, 1094, 1094, 1121, 1134, 1134, 1134, 1134, 1152, 1152, 1152, 1152, 1159, 1175, 1216, 1216, 1216, 1217, 1218, 1218, 1218, 1220, 1220, 1220, 1220, 1227, 1227, 1227, 1257, 1257, 1257],  # seen 167x, 56% WR
    "mined_7": [5, 5, 5, 13, 19, 19, 19, 19, 66, 66, 66, 305, 305, 305, 305, 741, 741, 741, 741, 742, 742, 742, 742, 743, 743, 743, 743, 1079, 1079, 1079, 1079, 1081, 1081, 1081, 1081, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1129, 1152, 1152, 1152, 1152, 1182, 1182, 1182, 1184, 1225, 1225, 1225, 1225, 1231, 1231, 1231, 1231, 1264],  # seen 165x, 49% WR
    "mined_8": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 104, 104, 112, 112, 112, 112, 646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 860, 860, 1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1122, 1152, 1152, 1152, 1152, 1182, 1182, 1197, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1231, 1259, 1259, 1259, 1259],  # seen 165x, 58% WR
    "mined_9": [2, 2, 2, 2, 5, 5, 5, 5, 7, 7, 112, 112, 119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 140, 235, 235, 1071, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1120, 1120, 1121, 1121, 1121, 1121, 1152, 1152, 1152, 1152, 1182, 1182, 1182, 1197, 1198, 1198, 1198, 1213, 1227, 1227, 1227, 1227, 1231, 1246, 1246],  # seen 158x, 65% WR
    "mined_10": [1, 11, 11, 11, 11, 14, 14, 14, 14, 18, 18, 18, 18, 343, 344, 344, 344, 344, 345, 345, 345, 345, 756, 756, 756, 756, 1086, 1086, 1086, 1086, 1087, 1122, 1122, 1122, 1122, 1123, 1123, 1123, 1123, 1147, 1147, 1147, 1147, 1159, 1182, 1182, 1197, 1197, 1197, 1197, 1225, 1225, 1225, 1225, 1227, 1227, 1227, 1227, 1264, 1264],  # seen 156x, 56% WR
    "mined_11": [2, 2, 2, 2, 5, 5, 5, 7, 112, 119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 131, 131, 132, 132, 133, 140, 235, 791, 1071, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1120, 1120, 1120, 1121, 1121, 1121, 1121, 1152, 1152, 1152, 1152, 1182, 1182, 1182, 1198, 1198, 1198, 1227, 1227, 1227, 1227, 1231, 1256, 1256],  # seen 96x, 57% WR
}
META_WEIGHT = {
    "mined_0": 895,
    "mined_1": 579,
    "mined_2": 293,
    "mined_3": 193,
    "mined_4": 189,
    "mined_5": 174,
    "mined_6": 167,
    "mined_7": 165,
    "mined_8": 165,
    "mined_9": 158,
    "mined_10": 156,
    "mined_11": 96,
}


# --------------------------------------------------------------------------
# Observation helpers
# --------------------------------------------------------------------------
def _visible_cards(pl):
    """Card ids in a player's publicly visible zones (plus own hand if shown)."""
    ids = []
    for c in pl.get("discard") or []:
        ids.append(c["id"])
    for c in pl.get("prize") or []:
        if c is not None:
            ids.append(c["id"])
    if pl.get("hand") is not None:
        ids.extend(c["id"] for c in pl["hand"])
    for mon in list(pl.get("active") or []) + list(pl.get("bench") or []):
        if mon is None:
            continue
        ids.append(mon["id"])
        for key in ("energyCards", "tools", "preEvolution"):
            for c in mon.get(key) or []:
                ids.append(c["id"])
    return ids


def _multiset_sub(full, seen):
    c = Counter(full)
    for x in seen:
        if c.get(x, 0) > 0:
            c[x] -= 1
    out = []
    for k, v in c.items():
        out.extend([k] * v)
    return out


def _fit_length(pool, n, rng, pad_pool):
    """Trim/pad `pool` to exactly n entries."""
    pool = list(pool)
    if len(pool) > n:
        rng.shuffle(pool)
        pool = pool[:n]
    while len(pool) < n:
        pool.append(rng.choice(pad_pool) if pad_pool else 3)
    return pool


# --------------------------------------------------------------------------
# Determinization
# --------------------------------------------------------------------------
class OpponentModel:
    def __init__(self):
        self.archetype = None

    def guess_list(self, opp_visible):
        best, best_score = None, -1
        seen = Counter(opp_visible)
        # ignore basic energy for matching (shared across decks)
        for name, lst in META_DECKS.items():
            have = Counter(lst)
            score = sum(min(v, have.get(k, 0)) for k, v in seen.items()
                        if _card(k).get("cardType") != CT_BASIC_ENERGY)
            score = score * 10 + META_WEIGHT.get(name, 0)
            if score > best_score:
                best, best_score = name, score
        self.archetype = best
        return META_DECKS[best]


def _sample_world(obs, me, opp_model, rng):
    """Build the SearchBegin prediction arrays for one determinized world."""
    cur = obs["current"]
    mypl = cur["players"][me]
    opl = cur["players"][1 - me]

    my_seen = _visible_cards(mypl)
    # stadium + looking cards attributed by owner
    for c in cur.get("stadium") or []:
        (my_seen if c.get("playerIndex") == me else []).append(c.get("id"))
    looking = cur.get("looking")
    if looking:
        for c in looking:
            if c and c.get("playerIndex") == me:
                my_seen.append(c["id"])

    my_unseen = _multiset_sub(MY_DECK, my_seen)
    rng.shuffle(my_unseen)
    n_prize = sum(1 for c in (mypl.get("prize") or []) if c is None)
    n_deck = mypl.get("deckCount", 0)
    my_prize = my_unseen[:n_prize]
    my_deck = my_unseen[n_prize:]
    my_prize = _fit_length(my_prize, n_prize, rng, MY_DECK)
    my_deck = _fit_length(my_deck, n_deck, rng, MY_DECK)

    opp_visible = _visible_cards(opl)
    for c in cur.get("stadium") or []:
        if c.get("playerIndex") == 1 - me:
            opp_visible.append(c.get("id"))
    guess = opp_model.guess_list(opp_visible)
    opp_unseen = _multiset_sub(guess, opp_visible)
    rng.shuffle(opp_unseen)

    n_oprize = sum(1 for c in (opl.get("prize") or []) if c is None)
    n_ohand = opl.get("handCount", 0)
    n_odeck = opl.get("deckCount", 0)
    opp_prize = opp_unseen[:n_oprize]
    opp_hand = opp_unseen[n_oprize:n_oprize + n_ohand]
    opp_deck = opp_unseen[n_oprize + n_ohand:]

    pad = [cid for cid in guess if _card(cid).get("cardType") == CT_BASIC_ENERGY] or guess
    opp_prize = _fit_length(opp_prize, n_oprize, rng, pad)
    opp_hand = _fit_length(opp_hand, n_ohand, rng, pad)
    opp_deck = _fit_length(opp_deck, n_odeck, rng, pad)

    # make sure a predicted deck contains a basic pokemon during setup
    if cur.get("turn", 0) == 0 and n_odeck > 0:
        if not any(_card(c).get("basic") for c in opp_deck):
            basics = [c for c in guess if _card(c).get("basic")]
            if basics:
                opp_deck[0] = basics[0]

    # face-down active pokemon prediction
    opp_active = []
    act = opl.get("active") or []
    if act and act[0] is None:
        basics = [c for c in guess if _card(c).get("basic")]
        opp_active = [rng.choice(basics)] if basics else []

    return my_deck, my_prize, opp_deck, opp_prize, opp_hand, opp_active


# --------------------------------------------------------------------------
# Value function
# --------------------------------------------------------------------------
def _mon_value(mon):
    if mon is None:
        return 0.0
    v = 0.05
    max_hp = mon.get("maxHp") or 1
    v += 0.06 * (mon.get("hp", 0) / max_hp)
    v += 0.015 * len(mon.get("energies") or [])
    v += 0.001 * min(_max_attack_damage(mon.get("id")), 300) / 10.0
    c = _card(mon.get("id"))
    if c.get("stage1"):
        v += 0.02
    if c.get("stage2"):
        v += 0.04
    return v


def _evaluate(state, me):
    cur = state["current"]
    res = cur.get("result", -1)
    if res >= 0:
        if res == me:
            return 1.0
        if res == 1 - me:
            return -1.0
        return 0.0
    mypl = cur["players"][me]
    opl = cur["players"][1 - me]

    score = 0.0
    my_prize_left = len(mypl.get("prize") or [])
    opp_prize_left = len(opl.get("prize") or [])
    score += 0.40 * (opp_prize_left - my_prize_left) / 6.0

    my_board = [m for m in list(mypl.get("active") or []) + list(mypl.get("bench") or []) if m]
    opp_board = [m for m in list(opl.get("active") or []) + list(opl.get("bench") or []) if m]
    score += sum(_mon_value(m) for m in my_board)
    score -= sum(_mon_value(m) for m in opp_board)

    hand_n = len(mypl.get("hand") or []) if mypl.get("hand") is not None else mypl.get("handCount", 0)
    score += 0.010 * min(hand_n, 16)
    score -= 0.006 * min(opl.get("handCount", 0), 16)

    if mypl.get("deckCount", 1) == 0:
        score -= 0.35
    elif mypl.get("deckCount", 99) <= 2:
        score -= 0.10
    if opl.get("deckCount", 1) == 0:
        score += 0.35

    for flag in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
        if mypl.get(flag):
            score -= 0.02
        if opl.get(flag):
            score += 0.02

    # Lethal awareness. Search discovers knockouts inside its horizon, but the
    # value function judges positions at the horizon's edge, and there
    # "one prize from winning" or "active about to be KO'd" dominates
    # everything else on the board.
    my_act = (mypl.get("active") or [None])
    op_act = (opl.get("active") or [None])
    my_mon = my_act[0] if my_act else None
    op_mon = op_act[0] if op_act else None
    if opp_prize_left <= 1:
        score -= 0.30                     # opponent wins on their next KO
    if my_prize_left <= 1:
        score += 0.30
    if my_mon is not None and op_mon is not None:
        if _max_attack_damage(op_mon.get("id")) >= my_mon.get("hp", 999):
            score -= 0.10                 # our active is in their KO range
        if _max_attack_damage(my_mon.get("id")) >= op_mon.get("hp", 999):
            score += 0.10
    # a board with no benched backup loses on one KO
    if len(my_board) <= 1 and my_prize_left > 1:
        score -= 0.12
    if len(opp_board) <= 1 and opp_prize_left > 1:
        score += 0.12

    return max(-0.97, min(0.97, score))


# --------------------------------------------------------------------------
# Heuristic option scoring (priors for search + standalone fallback)
# --------------------------------------------------------------------------
_TYPE_PRIOR = {
    OT_ABILITY: 3.0, OT_ATTACK: 2.6, OT_EVOLVE: 2.3, OT_PLAY: 1.9, OT_ATTACH: 1.6,
    OT_ENERGY: 1.0, OT_ENERGY_CARD: 1.0, OT_CARD: 0.6, OT_TOOL_CARD: 0.6,
    OT_SKILL: 1.2, OT_YES: 0.7, OT_NO: 0.4, OT_NUMBER: 0.5, OT_DISCARD: 0.2,
    OT_RETREAT: -0.4, OT_END: -2.5, OT_SPECIAL_CONDITION: 0.0,
}


def _keep_value(cid):
    """How much we'd like to keep this card (higher = keep)."""
    c = _card(cid)
    ct = c.get("cardType")
    if ct == CT_POKEMON:
        return 2.0 + min(c.get("hp", 0), 340) / 200.0 + (0.5 if c.get("stage2") else 0.0)
    if ct == CT_SUPPORTER:
        return 1.6
    if ct == CT_ITEM:
        return 1.3
    if ct == CT_TOOL:
        return 1.1
    if ct == CT_SPECIAL_ENERGY:
        return 1.0
    if ct == CT_BASIC_ENERGY:
        return 0.5
    return 1.0


def _option_score(opt, sel, rng=None):
    t = opt.get("type", 0)
    ctx = sel.get("context", -1)
    s = _TYPE_PRIOR.get(t, 0.0)
    cid = opt.get("cardId")

    if t == OT_ATTACK:
        aid = opt.get("attackId")
        dmg = ATTACK.get(aid, {}).get("damage", 0) or 0
        s += min(dmg, 400) / 120.0
    if t == OT_NUMBER:
        s += 0.05 * (opt.get("number") or 0)
    if cid is not None:
        if ctx in CTX_DISCARD_SET:
            s += 1.0 - 0.45 * _keep_value(cid)      # discard what we least need
        elif ctx == CTX_TO_HAND:
            s += 0.6 * _keep_value(cid)             # fetch what we most need
        elif ctx in (13, 14, 15):
            # Damage placement: snipe low-HP engine pieces, not high-HP walls.
            # Divergence mining vs 1050+ players showed this was our worst
            # context (30 percent agreement), and their pattern is consistent:
            # counters go on draw engines and evolution bases that die to them.
            hp = _card(cid).get("hp") or 200
            s += 0.9 * (200 - min(hp, 200)) / 200.0
        else:
            s += 0.15 * _keep_value(cid)
    if ctx == CTX_MULLIGAN:
        s += 1.0 if t == OT_NO else 0.0
    if ctx == CTX_ACTIVATE and t == OT_YES:
        s += 1.5                                     # abilities are usually free value
    if ctx == CTX_IS_FIRST and t == OT_NO:
        s += 0.3                                     # draw-engine decks like going second
    return s


def _heuristic_action(sel, rng):
    """Legal fallback selection without any search."""
    opts = sel.get("option") or []
    n = len(opts)
    kmax = max(1, min(sel.get("maxCount", 1), n))
    kmin = max(0, min(sel.get("minCount", kmax), kmax))
    scored = sorted(range(n), key=lambda i: -_option_score(opts[i], sel))
    k = kmax if kmax <= n else n
    return scored[:max(k, kmin, 1)]


# --------------------------------------------------------------------------
# Open-loop UCT over determinized worlds
# --------------------------------------------------------------------------
class _TNode:
    """Tree node bound to a persistent engine state (closed-loop PUCT)."""
    __slots__ = ("sid", "select", "actor", "v0", "terminal", "edges", "total", "cur")

    def __init__(self, sid, state, me):
        self.sid = sid
        cur = state["current"]
        self.cur = cur
        res = cur.get("result", -1)
        self.terminal = res >= 0
        self.actor = cur.get("yourIndex", me)
        self.select = None if self.terminal else state.get("select")
        self.v0 = _evaluate(state, me)
        self.edges = None    # action tuple -> [N, W, prior, child]
        self.total = 0


def _net_scores(state, me, sel, opts, heur):
    """Learned per-option logits, or None if the model is unavailable.

    The net is ~2.5 ms/call, so it is used only where it pays for itself:
    scoring a node's options once when the node is first expanded.
    """
    if _NET is None:
        return None
    try:
        import numpy as _np
        kind, card, scal, mask, opt_slot = _NF.encode(
            {"current": state["current"], "select": sel}, me, CARD, ATTACK, heur)
        pol, _v = _NET.forward(kind[None], card[None], scal[None], mask[None],
                               _np.array([int(sel.get("context") or 0)]),
                               _np.array([int(sel.get("type") or 0)]))
        out = []
        for i in range(len(opts)):
            p = opt_slot[i] if i < len(opt_slot) else -1
            out.append(float(pol[0, p]) if p >= 0 else -1e9)
        return out
    except Exception as exc:
        _dbg("net scoring failed:", repr(exc))
        return None


def _gen_candidates(sel, rng, cap=16, state=None, me=0):
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return [()]
    kmax = max(1, min(sel.get("maxCount", 1), n))
    kmin = max(0, min(sel.get("minCount", kmax), kmax))
    scores = [_option_score(o, sel) for o in opts]
    if state is not None:
        learned = _net_scores(state, me, sel, opts, scores)
        if learned is not None:
            # learned logits replace the handcrafted ranking; the heuristic
            # stays as the truncation order fallback inside encode()
            scores = learned

    cands = []
    if kmax == 1:
        order = sorted(range(n), key=lambda i: -scores[i])
        cands = [(i,) for i in order[:cap]]
    else:
        # top-score combination for each allowed size
        order = sorted(range(n), key=lambda i: -scores[i])
        sizes = {kmax, max(kmin, 1)}
        for k in sizes:
            cands.append(tuple(sorted(order[:k])))
        # sampled combinations of the max size
        tries = 0
        while len(cands) < cap and tries < cap * 6:
            tries += 1
            pick = tuple(sorted(rng.sample(range(n), kmax)))
            if pick not in cands:
                cands.append(pick)
    # priors: softmax-ish over summed member scores
    pri = []
    for c in cands:
        pri.append(math.exp(min(6.0, sum(scores[i] for i in c) / max(len(c), 1))))
    tot = sum(pri) or 1.0
    return [(c, p / tot) for c, p in zip(cands, pri)]


class WorldTree:
    """Closed-loop PUCT over one determinized world.

    Nodes are bound to persistent engine states (searchIds); each iteration
    descends via PUCT and expands exactly one leaf with one SearchStep call.
    All engine states are freed at once by SearchEnd after the move.
    """
    C_PUCT = 1.4
    DEPTH_CAP = 70

    def __init__(self, me, rng, root_sid, root_state):
        self.me = me
        self.rng = rng
        self.nodes = 1
        self.root = _TNode(root_sid, root_state, me)

    def _init_edges(self, node):
        node.edges = {}
        # The learned net is only worth its ~2.5 ms at the root, where priors
        # decide which subtrees get explored at all. Deeper nodes use the
        # (microsecond) heuristic prior so engine throughput stays ~20k/s.
        state = {"current": node.cur, "select": node.select} if node is self.root else None
        for act, prior in _gen_candidates(node.select, self.rng, state=state, me=self.me):
            node.edges[act] = [0, 0.0, prior, None]

    def _pick(self, node):
        maximize = node.actor == self.me
        best_act, best_u = None, -1e18
        sqrt_total = math.sqrt(node.total + 1)
        for act, e in node.edges.items():
            n_vis, w, prior, _child = e
            q = (w / n_vis) if n_vis else 0.15
            if not maximize:
                q = -q
            u = q + self.C_PUCT * prior * sqrt_total / (1 + n_vis)
            if u > best_u:
                best_act, best_u = act, u
        return best_act

    VIRTUAL_LOSS = 1.0

    def descend(self):
        """Walk to a leaf, expanding it if needed.

        Returns (path, leaf, value). `value` is None when the leaf still needs a
        network evaluation, in which case the caller must batch it and then call
        backup(). Virtual loss is applied along the path so that other descents
        in the same batch explore elsewhere; backup() removes it.
        """
        node = self.root
        path = []
        value = None
        leaf = None
        for _depth in range(self.DEPTH_CAP):
            if node.terminal or node.select is None:
                value = node.v0
                break
            if node.edges is None:
                self._init_edges(node)
            act = self._pick(node)
            if act is None:                   # no legal candidates survived
                node.terminal = True
                value = node.v0
                break
            e = node.edges[act]
            if e[3] is None:
                r = json.loads(_LIB.SearchStep(
                    _CTX, node.sid, _int_arr(list(act)), len(act)).decode())
                if r.get("error", 1) != 0:
                    del node.edges[act]       # illegal in this world
                    value = node.v0
                    break
                st = r["state"]
                child = _TNode(st["searchId"], st["observation"], self.me)
                e[3] = child
                self.nodes += 1
                self._apply_virtual_loss(node, act)
                path.append((node, act))
                if child.terminal or child.select is None or _NET is None:
                    value = child.v0          # terminal, or no net available
                else:
                    leaf = child              # defer to a batched net eval
                break
            self._apply_virtual_loss(node, act)
            path.append((node, act))
            node = e[3]
        if value is None and leaf is None:
            value = node.v0
        return path, leaf, value

    def _apply_virtual_loss(self, node, act):
        e = node.edges.get(act)
        if e is not None:
            e[0] += 1
            e[1] -= self.VIRTUAL_LOSS
        node.total += 1

    def backup(self, path, value):
        """Undo virtual loss and record the real value along the path."""
        for nd, act in path:
            e = nd.edges.get(act)
            if e is not None:
                e[1] += value + self.VIRTUAL_LOSS

    def iterate(self):
        """Single unbatched iteration (used when no network is loaded)."""
        path, leaf, value = self.descend()
        if leaf is not None:
            value = leaf.v0
        self.backup(path, value)
        return value


def _net_values(leaves, me):
    """Batched value-head evaluation for a list of leaf nodes.

    The network is trained with outcomes from the perspective of the player to
    move, so a node where the opponent is to move must have its value negated
    to express it in OUR terms. Returns a list of floats in [-1, 1], falling
    back to each node's heuristic value if anything goes wrong.
    """
    if _NET is None or not leaves:
        return [nd.v0 for nd in leaves]
    try:
        import numpy as _np
        n = len(leaves)
        kind = _np.zeros((n, _NF.SEQ), dtype=_np.int64)
        card = _np.zeros((n, _NF.SEQ), dtype=_np.int64)
        scal = _np.zeros((n, _NF.SEQ, _NF.F), dtype=_np.float32)
        mask = _np.zeros((n, _NF.SEQ), dtype=_np.float32)
        ctx = _np.zeros(n, dtype=_np.int64)
        styp = _np.zeros(n, dtype=_np.int64)
        for i, nd in enumerate(leaves):
            sel = nd.select or {}
            # Encode from the ACTOR's perspective, not ours. A search state is
            # rendered for whoever is to move, so players[me].hand is None on
            # opponent nodes (about half of all leaves). Encoding those with
            # `me` fed the network a phantom empty hand, which is both wrong and
            # unlike anything in training, where the encoder perspective always
            # equals the mover. Hand size drives this deck's damage, so that
            # error was severe. The sign flip below converts the network's
            # mover-perspective value back into ours.
            k, c, s, m, _slot = _NF.encode(
                {"current": nd.cur, "select": sel}, nd.actor, CARD, ATTACK, None)
            kind[i], card[i], scal[i], mask[i] = k, c, s, m
            ctx[i] = int(sel.get("context") or 0)
            styp[i] = int(sel.get("type") or 0)
        _pol, val = _NET.forward(kind, card, scal, mask, ctx, styp)
        out = []
        for i, nd in enumerate(leaves):
            v = float(val[i])
            if nd.actor != me:
                v = -v
            # Soften. The raw head saturates: 46% of its outputs exceed 0.95 in
            # absolute value, versus 0% for the heuristic. Confident leaf values
            # make PUCT commit to a line instead of verifying it against the
            # engine, which is fatal at the low simulation counts the network
            # forces. VALUE_TEMP shrinks the scale; VALUE_BLEND keeps some of
            # the heuristic, which is uninformative between siblings and so
            # preserves the pressure to keep exploring.
            v *= VALUE_TEMP
            if VALUE_BLEND < 1.0:
                v = VALUE_BLEND * v + (1.0 - VALUE_BLEND) * nd.v0
            out.append(max(-0.97, min(0.97, v)))
        return out
    except Exception as exc:
        _dbg("batched value eval failed:", repr(exc))
        return [nd.v0 for nd in leaves]


def _search_move(obs, me, opp_model, deadline, rng, n_worlds_target=4,
                 max_nodes=20000, collect_policy=None):
    """Determinized closed-loop PUCT until `deadline`; returns action list.

    If `collect_policy` is a dict it is filled with the aggregated root visit
    distribution {action_tuple: visits} (used by the ExIt data generator).
    """
    inp = obs.get("search_begin_input")
    if not inp or _LIB is None:
        return None
    inp_b = inp.encode("ascii")

    trees = []
    total_iters = 0
    attempts = 0
    while time.perf_counter() < deadline and len(trees) < n_worlds_target and attempts < 8:
        attempts += 1
        try:
            world = _sample_world(obs, me, opp_model, rng)
        except Exception:
            break
        r = json.loads(_LIB.SearchBegin(
            _CTX, inp_b, len(inp_b),
            _int_arr(world[0]), _int_arr(world[1]), _int_arr(world[2]),
            _int_arr(world[3]), _int_arr(world[4]), _int_arr(world[5]), 0).decode())
        if r.get("error", 1) != 0:
            _dbg("SearchBegin error", r.get("error"))
            continue
        st = r["state"]
        trees.append(WorldTree(me, rng, st["searchId"], st["observation"]))

    if not trees:
        return None

    # Round-robin across worlds until the deadline. With a network loaded we
    # gather a batch of leaves before evaluating them in one forward pass
    # (AlphaZero style); virtual loss keeps the batch from collapsing onto one
    # line. Without a network we fall back to per-iteration heuristic leaves.
    live_nodes = len(trees)
    # Leaf evaluation by the network is OFF by default because it measured far
    # worse: 1W-19L and then 0W-6L against heuristic leaves. Two reasons, both
    # verified. It costs ~35x search throughput (220 sims vs 7500), and the
    # value head is overconfident on positions it never trained on (46% of its
    # outputs saturate above 0.95, vs 0% for the heuristic), so PUCT commits
    # early instead of verifying against the engine. Used as move ordering
    # only, the same network measured 11W-9L, i.e. no harm.
    # Set PTCG_NET_LEAVES=1 to opt back in for experiments.
    use_leaves = _NET is not None and bool(os.environ.get("PTCG_NET_LEAVES"))
    batch = NET_LEAF_BATCH if use_leaves else 0
    while time.perf_counter() < deadline and live_nodes < max_nodes:
        if batch:
            pending = []
            for i in range(batch):
                tree = trees[i % len(trees)]
                path, leaf, value = tree.descend()
                total_iters += 1
                if leaf is None:
                    tree.backup(path, value)
                else:
                    pending.append((tree, path, leaf))
            if pending:
                vals = _net_values([p[2] for p in pending], me)
                for (tree, path, leaf), v in zip(pending, vals):
                    leaf.v0 = v
                    tree.backup(path, v)
        else:
            for tree in trees:
                for _ in range(8):
                    tree.iterate()
                    total_iters += 1
        live_nodes = sum(t.nodes for t in trees)

    # aggregate root statistics across worlds
    agg = {}
    for tree in trees:
        if tree.root.edges is None:
            continue
        for act, (n_vis, w, _p, _c) in tree.root.edges.items():
            a = agg.setdefault(act, [0, 0.0])
            a[0] += n_vis
            a[1] += w
    if _DEBUG:
        top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:3]
        _dbg(f"search: {len(trees)} worlds, {total_iters} iters, {live_nodes} nodes, "
             f"top {[(list(a), n[0], round(n[1]/max(n[0],1), 3)) for a, n in top]}")
    if collect_policy is not None:
        for act, (n_vis, _w) in agg.items():
            collect_policy[act] = n_vis
    if total_iters < 8 or not agg:
        return None
    best = max(agg.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    return list(best[0])


# --------------------------------------------------------------------------
# Per-game state + time management
# --------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.rng = random.Random(0xC1A0DE)
        self.opp_model = OpponentModel()
        self.calls = 0
        self.time_spent = 0.0
        self.search_fail_streak = 0


_GAME = GameState()


def _budget(obs, sel):
    """Seconds of search allowed for this decision."""
    remaining = obs.get("remainingOverageTime")
    if not isinstance(remaining, (int, float)):
        remaining = max(10.0, 540.0 - _GAME.time_spent)
    else:
        remaining = float(remaining)
    if remaining < 40.0:
        return 0.0
    # Spend the budget we are given. Earlier versions used ~15 percent of the
    # 600 s allowance; search strength scales with think time, so the cap and
    # share are raised to target roughly 300-400 s per game, with the low-time
    # guards unchanged.
    moves_left = max(50, 280 - _GAME.calls)
    share = 0.9 * remaining / moves_left
    ctx = sel.get("context", -1)
    n = len(sel.get("option") or [])
    weight = 1.8 if ctx in (CTX_MAIN, 35, CTX_IS_FIRST, 1, 2) else 0.7
    if n <= 2:
        weight *= 0.6
    # PTCG_MAX_BUDGET caps per-move think time. Used to run cheap, symmetric
    # local A/Bs (both sides capped equally): 3-4x more games per hour makes
    # the gates statistically meaningful. Ladder play leaves this unset.
    cap = float(os.environ.get("PTCG_MAX_BUDGET", "3.0"))
    return max(0.05, min(cap, share * weight))


def _validate(action, sel):
    opts = sel.get("option") or []
    n = len(opts)
    kmax = max(1, min(sel.get("maxCount", 1), n))
    kmin = max(0, min(sel.get("minCount", kmax), kmax))
    if not isinstance(action, list) or not action:
        return None
    if any((not isinstance(i, int)) or i < 0 or i >= n for i in action):
        return None
    if len(set(action)) != len(action):
        return None
    lo = kmin if kmin >= 1 else 1
    if not (lo <= len(action) <= kmax):
        return None
    return action


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
_ENGINE_TRIED = False


def agent(obs):
    global _GAME, _ENGINE_TRIED, MY_DECK
    t0 = time.perf_counter()
    try:
        if MY_DECK is None:
            MY_DECK = _load_deck()
        if not _ENGINE_TRIED:
            _ENGINE_TRIED = True
            try:
                _load_engine()
                _load_card_db()
                _dbg("engine loaded, cards:", len(CARD))
            except Exception as exc:
                _dbg("engine load FAILED:", repr(exc))
            try:
                _load_net()
            except Exception as exc:
                _dbg("net load failed (heuristic priors):", repr(exc))

        sel = obs.get("select") if hasattr(obs, "get") else obs["select"]
        if sel is None:
            _GAME = GameState()          # new game: reset tracker
            return list(MY_DECK)

        _GAME.calls += 1
        opts = sel.get("option") or []
        n = len(opts)
        kmax = max(1, min(sel.get("maxCount", 1), n))
        if n == 0:
            return []
        if n == 1:
            return [0]
        if kmax >= n and sel.get("minCount", 0) >= n:
            return list(range(n))

        cur = obs.get("current") or {}
        me = cur.get("yourIndex", 0)

        action = None
        budget = _budget(obs, sel)
        if budget > 0.04 and _LIB is not None and _GAME.search_fail_streak < 25:
            try:
                action = _search_move(obs, me, _GAME.opp_model,
                                      t0 + budget, _GAME.rng)
            except Exception as exc:
                _dbg("search exception:", repr(exc))
                action = None
            finally:
                try:
                    _LIB.SearchEnd(_CTX)
                except Exception:
                    pass
            if action is None:
                _GAME.search_fail_streak += 1
            else:
                _GAME.search_fail_streak = 0

        action = _validate(action, sel)
        if action is None:
            action = _validate(_heuristic_action(sel, _GAME.rng), sel)
        if action is None:
            action = list(range(max(1, min(sel.get("minCount", 1) or 1, n))))
        return action
    except Exception:
        # absolute last resort: first legal option
        try:
            k = max(1, min((obs["select"] or {}).get("maxCount", 1), 1))
            return list(range(k))
        except Exception:
            return [0]
    finally:
        _GAME.time_spent += time.perf_counter() - t0
