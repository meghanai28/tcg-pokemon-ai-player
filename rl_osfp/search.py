"""Determinized PUCT over the engine's native search API.

The competition engine exposes ``SearchBegin`` / ``SearchStep`` / ``SearchEnd``
and hands back a ``search_begin_input`` blob with every observation, so it is
built to be searched.  A search-free policy answers each decision with one
forward pass (~26 ms) against a ~600 s episode budget, leaving roughly 99% of
the available inference compute unused.

Three measured facts from the earlier search track shape this design, and they
are load-bearing - they were verified against this exact engine, not assumed:

1.  **Throughput decides.**  Heuristic leaves ran ~7,500 simulations where
    neural leaves managed ~220, a 35x gap.
2.  **A learned value head made search worse**, 1W-19L then 0W-6L against
    heuristic leaves.  Its outputs saturate above 0.95 on 46% of positions it
    never trained on, so PUCT commits to a line instead of verifying it against
    the engine.  The engine cannot be miscalibrated the way a value head can.
3.  **Network priors are neutral** (11W-9L) and cost ~2.5 ms, so they are worth
    paying for only at the root, where they decide which subtrees are explored
    at all.

Hence: rollout-based leaves, engine-truth outcomes, and the network - if used
at all - confined to root move ordering behind an explicit flag.

Every engine state returned by ``SearchStep`` is a new persistent ``searchId``
that must be released, or a long search leaks the engine's arena.
"""
from __future__ import annotations

from collections import Counter
import ctypes
import json
import math
import random
import time
from typing import Any

try:  # inside the repo the engine lives under the foundation package
    from foundation.cg.engine import get_lib
except ImportError:  # inside a submission every module sits flat beside cg/
    from cg.engine import get_lib


C_PUCT = 1.4
ROLLOUT_CAP = 50
ROLLOUT_LAMBDA = 0.75
DEPTH_CAP = 70

# Engine enums, from the official API docs.
(CT_POKEMON, CT_ITEM, CT_TOOL, CT_SUPPORTER, CT_STADIUM,
 CT_BASIC_ENERGY, CT_SPECIAL_ENERGY) = range(7)
(OT_NUMBER, OT_YES, OT_NO, OT_CARD, OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY,
 OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_ABILITY, OT_DISCARD, OT_RETREAT, OT_ATTACK,
 OT_END, OT_SKILL, OT_SPECIAL_CONDITION) = range(17)
CTX_DISCARD_SET = {8, 26, 27, 29, 30}
CTX_TO_HAND = 7
CTX_DAMAGE_PLACEMENT = (13, 14, 15)
CTX_IS_FIRST = 41
CTX_MULLIGAN = 42
CTX_ACTIVATE = 43

BASIC_ENERGY = CT_BASIC_ENERGY  # determinization padding only

TYPE_PRIOR = {
    OT_ABILITY: 3.0, OT_ATTACK: 2.6, OT_EVOLVE: 2.3, OT_PLAY: 1.9, OT_ATTACH: 1.6,
    OT_ENERGY: 1.0, OT_ENERGY_CARD: 1.0, OT_CARD: 0.6, OT_TOOL_CARD: 0.6,
    OT_SKILL: 1.2, OT_YES: 0.7, OT_NO: 0.4, OT_NUMBER: 0.5, OT_DISCARD: 0.2,
    OT_RETREAT: -0.4, OT_END: -2.5, OT_SPECIAL_CONDITION: 0.0,
}


def int_array(values):
    return (ctypes.c_int * len(values))(*values)


class Engine:
    """Owns the native search context and guarantees its states are freed."""

    def __init__(self) -> None:
        self.lib = get_lib()
        self.ctx = self.lib.AgentStart()
        self.card_db = {
            int(card["cardId"]): card
            for card in json.loads(self.lib.AllCard().decode())
        }
        self._live: set[int] = set()

    def begin(self, blob: str, world: tuple[list[int], ...]) -> dict | None:
        payload = blob.encode("ascii")
        result = json.loads(self.lib.SearchBegin(
            self.ctx, payload, len(payload),
            *[int_array(part) for part in world], 0,
        ).decode())
        if result.get("error", 1) != 0:
            return None
        state = result["state"]
        self._live.add(int(state["searchId"]))
        return state

    def step(self, search_id: int, action: list[int]) -> dict | None:
        result = json.loads(self.lib.SearchStep(
            self.ctx, search_id, int_array(action), len(action),
        ).decode())
        if result.get("error", 1) != 0:
            return None
        state = result["state"]
        self._live.add(int(state["searchId"]))
        return state

    def release(self, search_id: int) -> None:
        if search_id in self._live:
            self.lib.SearchRelease(self.ctx, search_id)
            self._live.discard(search_id)

    def end(self) -> None:
        """Free every state from this move at once."""
        self.lib.SearchEnd(self.ctx)
        self._live.clear()

    def card(self, card_id) -> dict:
        return self.card_db.get(int(card_id or 0), {})


def visible_cards(player: dict) -> list[int]:
    """Card ids in a player's publicly visible zones, plus its hand if shown."""
    ids: list[int] = []
    for card in player.get("discard") or []:
        ids.append(card["id"])
    for card in player.get("prize") or []:
        if card is not None:
            ids.append(card["id"])
    if player.get("hand") is not None:
        ids.extend(card["id"] for card in player["hand"])
    for mon in list(player.get("active") or []) + list(player.get("bench") or []):
        if mon is None:
            continue
        ids.append(mon["id"])
        for key in ("energyCards", "tools", "preEvolution"):
            for card in mon.get(key) or []:
                ids.append(card["id"])
    return ids


def multiset_sub(full: list[int], seen: list[int]) -> list[int]:
    remaining = Counter(full)
    for card in seen:
        if remaining.get(card, 0) > 0:
            remaining[card] -= 1
    out: list[int] = []
    for card, count in remaining.items():
        out.extend([card] * count)
    return out


def fit_length(pool: list[int], size: int, rng: random.Random,
               pad: list[int]) -> list[int]:
    """Trim or pad to exactly ``size``; the engine rejects a wrong-length zone."""
    pool = list(pool)
    if len(pool) > size:
        rng.shuffle(pool)
        pool = pool[:size]
    while len(pool) < size:
        pool.append(rng.choice(pad) if pad else 3)
    return pool


class OpponentModel:
    """Guess the opponent's list by best multiset overlap with a meta pool."""

    def __init__(self, meta_decks: list[list[int]], weights: list[float] | None = None):
        if not meta_decks:
            raise ValueError("opponent model needs at least one meta deck")
        self.meta_decks = meta_decks
        self.weights = weights or [0.0] * len(meta_decks)
        self.choice = 0

    def guess(self, engine: Engine, opponent_visible: list[int]) -> list[int]:
        seen = Counter(opponent_visible)
        best_index, best_score = 0, -1.0
        for index, deck in enumerate(self.meta_decks):
            have = Counter(deck)
            # Basic energy is shared across archetypes and carries no signal.
            score = sum(
                min(count, have.get(card, 0)) for card, count in seen.items()
                if engine.card(card).get("cardType") != BASIC_ENERGY
            )
            score = score * 10.0 + self.weights[index]
            if score > best_score:
                best_index, best_score = index, score
        self.choice = best_index
        return self.meta_decks[best_index]


def sample_world(
    engine: Engine,
    observation: dict,
    me: int,
    own_deck: list[int],
    opponent: OpponentModel,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    """One determinization: fill every hidden zone with a consistent guess.

    Returned in the order ``SearchBegin`` expects:
    ``(my_deck, my_prize, opp_deck, opp_prize, opp_hand, opp_active)``.
    """
    current = observation["current"]
    mine = current["players"][me]
    theirs = current["players"][1 - me]

    my_seen = visible_cards(mine)
    for card in current.get("stadium") or []:
        if card.get("playerIndex") == me:
            my_seen.append(card.get("id"))
    for card in current.get("looking") or []:
        if card and card.get("playerIndex") == me:
            my_seen.append(card["id"])
    my_unseen = multiset_sub(own_deck, my_seen)
    rng.shuffle(my_unseen)
    n_my_prize = sum(1 for card in (mine.get("prize") or []) if card is None)
    n_my_deck = mine.get("deckCount", 0)
    my_prize = fit_length(my_unseen[:n_my_prize], n_my_prize, rng, own_deck)
    my_deck = fit_length(my_unseen[n_my_prize:], n_my_deck, rng, own_deck)

    their_seen = visible_cards(theirs)
    for card in current.get("stadium") or []:
        if card.get("playerIndex") == 1 - me:
            their_seen.append(card.get("id"))
    guess = opponent.guess(engine, their_seen)
    their_unseen = multiset_sub(guess, their_seen)
    rng.shuffle(their_unseen)

    n_prize = sum(1 for card in (theirs.get("prize") or []) if card is None)
    n_hand = theirs.get("handCount", 0)
    n_deck = theirs.get("deckCount", 0)
    pad = [card for card in guess
           if engine.card(card).get("cardType") == BASIC_ENERGY] or guess
    opp_prize = fit_length(their_unseen[:n_prize], n_prize, rng, pad)
    opp_hand = fit_length(their_unseen[n_prize:n_prize + n_hand], n_hand, rng, pad)
    opp_deck = fit_length(their_unseen[n_prize + n_hand:], n_deck, rng, pad)

    # The engine rejects a setup world whose deck holds no basic Pokemon.
    if current.get("turn", 0) == 0 and n_deck > 0:
        if not any(engine.card(card).get("basic") for card in opp_deck):
            basics = [card for card in guess if engine.card(card).get("basic")]
            if basics:
                opp_deck[0] = basics[0]

    opp_active: list[int] = []
    active = theirs.get("active") or []
    if active and active[0] is None:
        basics = [card for card in guess if engine.card(card).get("basic")]
        if basics:
            opp_active = [rng.choice(basics)]

    return my_deck, my_prize, opp_deck, opp_prize, opp_hand, opp_active


# ---------------------------------------------------------------------------
# Static evaluation
# ---------------------------------------------------------------------------
def monster_value(engine: Engine, mon: dict | None) -> float:
    if not mon:
        return 0.0
    card = engine.card(mon.get("id"))
    value = 0.05
    value += 0.06 * (mon.get("hp", 0) / (mon.get("maxHp") or 1))
    value += 0.015 * len(mon.get("energies") or [])
    if card.get("stage1"):
        value += 0.02
    if card.get("stage2"):
        value += 0.04
    return value


def evaluate(engine: Engine, state: dict, me: int) -> float:
    """Static value of a position in [-0.97, 0.97], from ``me``'s seat.

    Deliberately small-magnitude and smooth: its job is to break ties between
    sibling moves, so a term large enough to dominate the search is a hazard.
    An earlier "lethal awareness" bonus of exactly that kind regressed the
    ladder from 65% to 40% and was reverted.
    """
    current = state["current"]
    result = current.get("result", -1)
    if result >= 0:
        return 1.0 if result == me else (-1.0 if result == 1 - me else 0.0)

    mine = current["players"][me]
    theirs = current["players"][1 - me]
    score = 0.40 * (len(theirs.get("prize") or []) - len(mine.get("prize") or [])) / 6.0

    for player, sign in ((mine, 1.0), (theirs, -1.0)):
        board = list(player.get("active") or []) + list(player.get("bench") or [])
        score += sign * sum(monster_value(engine, mon) for mon in board if mon)

    hand = mine.get("hand")
    hand_size = len(hand) if hand is not None else mine.get("handCount", 0)
    score += 0.010 * min(hand_size, 16)
    score -= 0.006 * min(theirs.get("handCount", 0), 16)

    if mine.get("deckCount", 1) == 0:
        score -= 0.35
    elif mine.get("deckCount", 99) <= 2:
        score -= 0.10
    if theirs.get("deckCount", 1) == 0:
        score += 0.35

    for flag in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
        if mine.get(flag):
            score -= 0.02
        if theirs.get(flag):
            score += 0.02
    return max(-0.97, min(0.97, score))


# ---------------------------------------------------------------------------
# Heuristic option scoring: search priors and the standalone fallback
# ---------------------------------------------------------------------------
def keep_value(engine: Engine, card_id) -> float:
    card = engine.card(card_id)
    kind = card.get("cardType")
    if kind == CT_POKEMON:
        return 2.0 + min(card.get("hp", 0), 340) / 200.0 + (0.5 if card.get("stage2") else 0.0)
    if kind == CT_SUPPORTER:
        return 1.6
    if kind == CT_ITEM:
        return 1.3
    if kind == CT_TOOL:
        return 1.1
    if kind == CT_SPECIAL_ENERGY:
        return 0.9
    return 0.5


def option_score(engine: Engine, attack_db: dict, option: dict, select: dict) -> float:
    kind = option.get("type", 0)
    context = select.get("context", -1)
    score = TYPE_PRIOR.get(kind, 0.0)
    card_id = option.get("cardId")

    if kind == OT_ATTACK:
        damage = attack_db.get(option.get("attackId"), {}).get("damage", 0) or 0
        score += min(damage, 400) / 120.0
    if kind == OT_NUMBER:
        score += 0.05 * (option.get("number") or 0)
    if card_id is not None:
        if context in CTX_DISCARD_SET:
            score += 1.0 - 0.45 * keep_value(engine, card_id)
        elif context == CTX_TO_HAND:
            score += 0.6 * keep_value(engine, card_id)
        elif context in CTX_DAMAGE_PLACEMENT:
            # Snipe fragile engine pieces rather than high-HP walls.
            hp = engine.card(card_id).get("hp") or 200
            score += 0.9 * (200 - min(hp, 200)) / 200.0
        else:
            score += 0.15 * keep_value(engine, card_id)
    if context == CTX_MULLIGAN and kind == OT_NO:
        score += 1.0
    if context == CTX_ACTIVATE and kind == OT_YES:
        score += 1.5
    if context == CTX_IS_FIRST and kind == OT_NO:
        score += 0.3
    return score


def generate_candidates(engine: Engine, attack_db: dict, select: dict,
                        rng: random.Random, cap: int = 16) -> list[tuple[tuple[int, ...], float]]:
    """Candidate actions with normalized priors, ordered by heuristic score."""
    options = select.get("option") or []
    n_options = len(options)
    if n_options == 0:
        return [((), 1.0)]
    high = max(1, min(int(select.get("maxCount", 1) or 1), n_options))
    low = max(0, min(int(select.get("minCount", high) or 0), high))
    scores = [option_score(engine, attack_db, option, select) for option in options]
    order = sorted(range(n_options), key=lambda index: -scores[index])

    candidates: list[tuple[int, ...]] = []
    if high == 1:
        candidates = [(index,) for index in order[:cap]]
    else:
        for size in {high, max(low, 1)}:
            candidates.append(tuple(sorted(order[:size])))
        attempts = 0
        while len(candidates) < cap and attempts < cap * 6:
            attempts += 1
            pick = tuple(sorted(rng.sample(range(n_options), high)))
            if pick not in candidates:
                candidates.append(pick)

    priors = [
        math.exp(min(6.0, sum(scores[i] for i in cand) / max(len(cand), 1)))
        for cand in candidates
    ]
    total = sum(priors) or 1.0
    return [(cand, prior / total) for cand, prior in zip(candidates, priors)]


def heuristic_action(engine: Engine, attack_db: dict, select: dict,
                     rng: random.Random) -> list[int]:
    """Best single action by heuristic score; also the rollout policy."""
    candidates = generate_candidates(engine, attack_db, select, rng, cap=4)
    return list(candidates[0][0]) if candidates else []


# ---------------------------------------------------------------------------
# Determinized PUCT
# ---------------------------------------------------------------------------
class Node:
    __slots__ = ("search_id", "current", "select", "actor", "edges", "total", "value")

    def __init__(self, search_id: int, state: dict, me: int):
        self.search_id = search_id
        self.current = state.get("current") or {}
        self.select = state.get("select") or {}
        self.actor = int(self.current.get("yourIndex", me))
        self.edges: dict[tuple[int, ...], list] | None = None  # act -> [n, w, prior, child]
        self.total = 0
        self.value = 0.0

    @property
    def terminal(self) -> bool:
        return int(self.current.get("result", -1)) >= 0 or not (self.select.get("option") or [])


class WorldTree:
    """Closed-loop PUCT over one determinized world.

    Each node is bound to a persistent engine state, so descending replays no
    actions - one ``SearchStep`` expands exactly one edge.
    """

    def __init__(self, engine: Engine, attack_db: dict, me: int,
                 rng: random.Random, root_state: dict,
                 guide: "PolicyGuide | None" = None, deck: list[int] | None = None):
        self.engine = engine
        self.attack_db = attack_db
        self.me = me
        self.rng = rng
        self.guide = guide
        self.deck = deck or []
        self.root = Node(int(root_state["searchId"]), root_state["observation"], me)
        self.nodes = 1

    def _expand_edges(self, node: Node) -> None:
        candidates = generate_candidates(
            self.engine, self.attack_db, node.select, self.rng
        )
        # The network costs ~2.5 ms, which only pays for itself at the root
        # where priors decide which subtrees are explored at all. Deeper nodes
        # keep the microsecond heuristic so engine throughput stays high.
        if self.guide is not None and node is self.root:
            candidates = blend_priors(
                candidates,
                self.guide.option_logits(node.current, node.select, self.deck),
                self.guide.prior_weight,
            )
        node.edges = {act: [0, 0.0, prior, None] for act, prior in candidates}

    def _select(self, node: Node) -> tuple[int, ...]:
        """PUCT, negated for the opponent's turns so one tree serves both."""
        maximize = node.actor == self.me
        best_act, best_u = None, -1e18
        sqrt_total = math.sqrt(node.total + 1)
        for act, edge in node.edges.items():
            visits, wins, prior, _child = edge
            q = (wins / visits) if visits else 0.15
            if not maximize:
                q = -q
            u = q + C_PUCT * prior * sqrt_total / (1 + visits)
            if u > best_u:
                best_act, best_u = act, u
        return best_act

    def rollout(self, node: Node) -> float:
        """Play the leaf out with the heuristic policy; score what the engine says.

        Rollout values are noisy but unbiased, and the engine cannot be
        miscalibrated the way a learned value head can. If the game has not
        ended within ROLLOUT_CAP steps, fall back to the static evaluation of
        the deepest state reached - still ROLLOUT_CAP steps beyond the leaf.
        """
        search_id = node.search_id
        state = None
        current, select = node.current, node.select
        created: list[int] = []
        outcome = None
        for _ in range(ROLLOUT_CAP):
            if int(current.get("result", -1)) >= 0:
                result = int(current["result"])
                outcome = 1.0 if result == self.me else (-1.0 if result == 1 - self.me else 0.0)
                break
            if not (select.get("option") or []):
                break
            action = heuristic_action(self.engine, self.attack_db, select, self.rng)
            nxt = self.engine.step(search_id, action)
            if nxt is None:
                break
            search_id = int(nxt["searchId"])
            created.append(search_id)
            state = nxt["observation"]
            current = state.get("current") or {}
            select = state.get("select") or {}
        for stale in created:
            self.engine.release(stale)
        deep = outcome if outcome is not None else (
            evaluate(self.engine, state, self.me) if state is not None else node.value
        )
        if self.guide is not None and self.guide.value_weight > 0.0 and outcome is None:
            scored = self.guide.values([node], self.deck, self.me)
            if scored:
                # A head that saturates on 77% of positions discriminates
                # poorly between siblings even when its sign is right, so it
                # shifts the rollout signal rather than replacing it.
                weight = self.guide.value_weight
                deep = (1.0 - weight) * deep + weight * scored[0][1]
        # AlphaGo's lambda mix: blend the noisy-unbiased rollout with the
        # smooth-biased static evaluation rather than trusting either alone.
        return ROLLOUT_LAMBDA * deep + (1.0 - ROLLOUT_LAMBDA) * node.value

    def iterate(self) -> bool:
        """One simulation. Returns False if the tree can no longer be grown."""
        path: list[tuple[Node, tuple[int, ...]]] = []
        node = self.root
        for _ in range(DEPTH_CAP):
            if node.terminal:
                value = evaluate(self.engine, {"current": node.current}, self.me)
                self._backup(path, value)
                return bool(path)
            if node.edges is None:
                self._expand_edges(node)
                node.value = evaluate(self.engine, {"current": node.current}, self.me)
                value = self.rollout(node)
                self._backup(path, value)
                return True
            action = self._select(node)
            if action is None:
                break
            edge = node.edges[action]
            path.append((node, action))
            if edge[3] is None:
                nxt = self.engine.step(node.search_id, list(action))
                if nxt is None:
                    # Illegal under this determinization: prune the edge so the
                    # search stops spending simulations on it.
                    del node.edges[action]
                    path.pop()
                    if not node.edges:
                        return False
                    continue
                child = Node(int(nxt["searchId"]), nxt["observation"], self.me)
                edge[3] = child
                self.nodes += 1
                child.value = evaluate(self.engine, {"current": child.current}, self.me)
                self._backup(path, self.rollout(child))
                return True
            node = edge[3]
        self._backup(path, evaluate(self.engine, {"current": node.current}, self.me))
        return True

    def _backup(self, path, value: float) -> None:
        for node, action in path:
            edge = node.edges[action]
            edge[0] += 1
            edge[1] += value
            node.total += 1

    def root_visits(self) -> dict[tuple[int, ...], int]:
        if self.root.edges is None:
            return {}
        return {act: edge[0] for act, edge in self.root.edges.items()}


def search_move(
    engine: Engine,
    attack_db: dict,
    observation: dict,
    me: int,
    own_deck: list[int],
    opponent: OpponentModel,
    rng: random.Random,
    deadline: float,
    worlds: int = 4,
    max_nodes: int = 40_000,
    guide: "PolicyGuide | None" = None,
) -> list[int] | None:
    """Determinized IS-MCTS: build trees over sampled worlds, play the most
    visited root action aggregated across them.

    Returns None whenever search cannot run, so the caller falls back to a
    legal heuristic action rather than failing.
    """
    blob = observation.get("search_begin_input")
    select = observation.get("select") or {}
    if not blob or not (select.get("option") or []):
        return None

    trees: list[WorldTree] = []
    attempts = 0
    try:
        while (time.perf_counter() < deadline and len(trees) < worlds and attempts < 8):
            attempts += 1
            try:
                world = sample_world(engine, observation, me, own_deck, opponent, rng)
            except Exception:
                break
            state = engine.begin(blob, world)
            if state is not None:
                trees.append(WorldTree(engine, attack_db, me, rng, state, guide, own_deck))
        if not trees:
            return None

        index = 0
        while time.perf_counter() < deadline:
            tree = trees[index % len(trees)]
            index += 1
            if sum(t.nodes for t in trees) >= max_nodes:
                break
            if not tree.iterate() and all(t.root.edges == {} for t in trees):
                break

        # Aggregate root statistics across worlds: an action is only good if it
        # survives many determinizations, not one lucky one.
        tally: dict[tuple[int, ...], int] = {}
        for tree in trees:
            for act, visits in tree.root_visits().items():
                tally[act] = tally.get(act, 0) + visits
        if not tally:
            return None
        best = max(tally.items(), key=lambda item: item[1])[0]
        return list(best)
    finally:
        engine.end()


# ---------------------------------------------------------------------------
# Network guidance: PPO priors and value, AlphaZero style
# ---------------------------------------------------------------------------
class PolicyGuide:
    """Policy priors and batched value estimates from an rl_osfp checkpoint.

    Uses the same numpy inference path the submission ships, so the gate and
    the packaged agent measure identical behaviour.

    Two guards come straight from measurement.  Values are blended with
    rollouts rather than replacing them, because a head that saturates on 77%
    of positions gives PUCT almost no discrimination between sibling moves even
    when its sign is right.  And the network encodes from the *acting* seat, so
    its value is negated whenever the actor is not the searching player.
    """

    def __init__(self, model_path: str, card_db: dict, attack_db: dict,
                 prior_weight: float = 0.7, value_weight: float = 0.5,
                 batch: int = 16):
        try:
            from foundation import nn_features_rich as features
            from foundation.nn_infer import NumpyNet
        except ImportError:
            import nn_features_rich as features
            from nn_infer import NumpyNet
        self.features = features
        self.net = NumpyNet(model_path)
        self.card_db = card_db
        self.attack_db = attack_db
        self.prior_weight = prior_weight
        self.value_weight = value_weight
        self.batch = batch
        self.calls = 0

    def _encode(self, current: dict, select: dict, deck: list[int]):
        seat = int(current.get("yourIndex", 0))
        return seat, self.features.encode(
            {"current": current, "select": select, "decklist": deck},
            seat, self.card_db, self.attack_db, None,
        )

    def option_logits(self, current: dict, select: dict, deck: list[int]):
        """Per-option logits aligned to ``select['option']`` indices."""
        import numpy as np
        options = select.get("option") or []
        if not options:
            return None
        try:
            _seat, (kind, card, scal, mask, slots) = self._encode(current, select, deck)
            if len(slots) != len(options) or any(slot < 0 for slot in slots):
                return None
            option, _count, _value = self.net.forward(
                kind[None].astype(np.int64), card[None].astype(np.int64), scal[None],
                mask[None], np.asarray([int(select.get("context") or 0)]),
                np.asarray([int(select.get("type") or 0)]),
            )
            self.calls += 1
            return [float(option[0, slot]) for slot in slots]
        except Exception:
            return None

    def values(self, nodes, deck: list[int], me: int):
        """Batched leaf values, each already oriented to ``me``'s seat."""
        import numpy as np
        if not nodes:
            return []
        kinds, cards, scals, masks, ctxs, stypes, signs = [], [], [], [], [], [], []
        keep = []
        for node in nodes:
            try:
                seat, (kind, card, scal, mask, _slots) = self._encode(
                    node.current, node.select, deck
                )
            except Exception:
                continue
            kinds.append(kind); cards.append(card); scals.append(scal); masks.append(mask)
            ctxs.append(int((node.select or {}).get("context") or 0))
            stypes.append(int((node.select or {}).get("type") or 0))
            signs.append(1.0 if seat == me else -1.0)
            keep.append(node)
        if not keep:
            return []
        _option, _count, value = self.net.forward(
            np.stack(kinds).astype(np.int64), np.stack(cards).astype(np.int64),
            np.stack(scals), np.stack(masks), np.asarray(ctxs), np.asarray(stypes),
        )
        self.calls += 1
        return list(zip(keep, [float(v) * s for v, s in zip(value, signs)]))


def blend_priors(candidates, logits, weight: float):
    """Mix heuristic priors with network logits over the same candidate set."""
    import numpy as np
    if logits is None or weight <= 0.0:
        return candidates
    scored = []
    for action, heuristic in candidates:
        if action:
            mean_logit = sum(logits[i] for i in action) / len(action)
        else:
            mean_logit = 0.0
        scored.append((action, heuristic, mean_logit))
    values = np.asarray([item[2] for item in scored], dtype=np.float64)
    values -= values.max()
    network = np.exp(values)
    network /= network.sum() or 1.0
    out = []
    for (action, heuristic, _logit), net_prior in zip(scored, network):
        out.append((action, (1.0 - weight) * heuristic + weight * float(net_prior)))
    total = sum(prior for _a, prior in out) or 1.0
    return [(action, prior / total) for action, prior in out]
