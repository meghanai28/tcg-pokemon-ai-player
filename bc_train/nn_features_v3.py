"""Feature ABI v3: public resource sets and structured-action semantics.

Unlike v2, this ABI must be produced from raw observations; the missing public
state cannot be reconstructed from old shards.  The ordinary transformer
sequence stays small while variable-card collections are carried as fixed
Deep-Set bags which the model pools into one token apiece.

Public information added here:

* the exact submitted deck as card/count members (not an opaque deck hash);
* both discard piles and recent public card logs;
* energy, tool, and pre-evolution cards attached to every board position;
* stadium, context-card, effect-card, and exact attack identities;
* an explicit empty-action token when ``minCount == 0``.

No opponent hand, prize, or unobserved deck identity is encoded.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

try:
    from . import nn_features_rich as _rich
except ImportError:  # submission-local import
    import nn_features_rich as _rich  # type: ignore[no-redef]


FEATURE_VERSION = 3
DECK_AWARE = True

MAX_BOARD = _rich.MAX_BOARD
MAX_HAND = _rich.MAX_HAND
MAX_OPT = _rich.MAX_OPT
OLD_OPT_BASE = _rich.OPT_BASE

CONTEXT_SLOT = OLD_OPT_BASE
EFFECT_SLOT = OLD_OPT_BASE + 1
STADIUM_SLOT = OLD_OPT_BASE + 2
EMPTY_SLOT = OLD_OPT_BASE + 3
OPT_BASE = OLD_OPT_BASE + 4
SEQ = _rich.SEQ + 4

F = 40
N_CTX = _rich.N_CTX
N_STYPE = _rich.N_STYPE

# Physical cards are 1..1267 and attacks are 1..1556.  Giving attacks a
# disjoint embedding namespace removes the previous arbitrary numeric scalar.
PHYSICAL_CARD_VOCAB = _rich.N_CARD
ATTACK_TOKEN_BASE = PHYSICAL_CARD_VOCAB
N_CARD = 4096

# Main kinds 0..3 retain global/board/hand/option.  Remaining kinds identify
# semantic singleton tokens and pooled public-card sets.
K_CONTEXT = 4
K_EFFECT = 5
K_STADIUM = 6
K_DECK = 7
K_MY_DISCARD = 8
K_OPP_DISCARD = 9
K_MY_HISTORY = 10
K_OPP_HISTORY = 11
K_ATTACHMENT = 12
N_KIND = 13

# Five global bags plus one attachment bag for every board slot.
BAG_DECK = 0
BAG_MY_DISCARD = 1
BAG_OPP_DISCARD = 2
BAG_MY_HISTORY = 3
BAG_OPP_HISTORY = 4
BAG_ATTACH_BASE = 5
N_BAGS = BAG_ATTACH_BASE + MAX_BOARD
BAG_WIDTH = 32

V3_FIELDS = ("bag_card", "bag_count", "bag_kind", "bag_scal", "bag_mask")


def _card_id(item) -> int:
    cid = item.get("id", 0) if isinstance(item, dict) else item
    return int(cid) if isinstance(cid, int) and 0 < cid < PHYSICAL_CARD_VOCAB else 0


def _blank_bags():
    return (
        np.zeros((N_BAGS, BAG_WIDTH), dtype=np.int16),
        np.zeros((N_BAGS, BAG_WIDTH), dtype=np.uint8),
        np.zeros(N_BAGS, dtype=np.int8),
        np.zeros((N_BAGS, F), dtype=np.float32),
        np.zeros(N_BAGS, dtype=np.float32),
    )


def _fill_bag(bag_card, bag_count, bag_kind, bag_scal, bag_mask,
              index: int, kind: int, cards, *, side: float = 0.0,
              board_pos: int | None = None) -> None:
    ids = [_card_id(item) for item in (cards or [])]
    counts = Counter(cid for cid in ids if cid)
    bag_kind[index] = kind
    if not counts:
        return
    # Frequent resources first, then stable card id.  Target/meta decks have
    # fewer than 24 unique cards; width 32 is lossless for those lists.
    members = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
        :BAG_WIDTH]
    for j, (cid, count) in enumerate(members):
        bag_card[index, j] = cid
        bag_count[index, j] = min(int(count), 255)
    bag_mask[index] = 1.0
    total = sum(counts.values())
    s = bag_scal[index]
    s[0] = min(total / 60.0, 1.0)
    s[1] = min(len(counts) / float(BAG_WIDTH), 1.0)
    s[2] = side
    if board_pos is not None:
        s[3] = 1.0
        s[4] = (board_pos % 6) / 5.0
        s[5] = 1.0 if board_pos >= 6 else 0.0
        s[6] = 1.0 if board_pos % 6 == 0 else 0.0


def _stadium(cur):
    value = cur.get("stadium")
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, dict) else None


def _board_mons(players):
    """Yield the same twelve board positions used by nn_features."""
    for player in players[:2]:
        active = player.get("active") or []
        yield active[0] if active else None
        bench = player.get("bench") or []
        for i in range(5):
            yield bench[i] if i < len(bench) else None


def encode(state, me, card_db, attack_db, opt_scores=None):
    old_kind, old_card, old_scal, old_mask, old_slots = _rich.encode(
        state, me, card_db, attack_db, opt_scores)
    cur = state["current"]
    sel = state.get("select") or {}

    kind = np.zeros(SEQ, dtype=np.int8)
    card = np.zeros(SEQ, dtype=np.int16)
    scal = np.zeros((SEQ, F), dtype=np.float32)
    mask = np.zeros(SEQ, dtype=np.float32)

    # Preserve the established global/board/hand representation and shift the
    # option block past the new semantic singleton/action tokens.
    kind[:OLD_OPT_BASE] = old_kind[:OLD_OPT_BASE]
    card[:OLD_OPT_BASE] = old_card[:OLD_OPT_BASE]
    scal[:OLD_OPT_BASE, :_rich.F] = old_scal[:OLD_OPT_BASE]
    mask[:OLD_OPT_BASE] = old_mask[:OLD_OPT_BASE]
    kind[OPT_BASE:] = old_kind[OLD_OPT_BASE:]
    card[OPT_BASE:] = old_card[OLD_OPT_BASE:]
    scal[OPT_BASE:, :_rich.F] = old_scal[OLD_OPT_BASE:]
    mask[OPT_BASE:] = old_mask[OLD_OPT_BASE:]

    for slot, token_kind, item in (
            (CONTEXT_SLOT, K_CONTEXT, sel.get("contextCard")),
            (EFFECT_SLOT, K_EFFECT, sel.get("effect")),
            (STADIUM_SLOT, K_STADIUM, _stadium(cur))):
        cid = _card_id(item)
        kind[slot] = token_kind
        if cid:
            card[slot] = cid
            mask[slot] = 1.0
            scal[slot, 0] = 1.0

    # Optional selections have a real candidate distinct from every physical
    # card.  This lets ordinary pointer CE learn demonstrated [] actions.
    if int(sel.get("minCount") or 0) == 0:
        kind[EMPTY_SLOT] = 3
        mask[EMPTY_SLOT] = 1.0
        scal[EMPTY_SLOT, 32] = 1.0

    opts = sel.get("option") or []
    opt_slot = [slot + 4 if slot >= OLD_OPT_BASE else slot for slot in old_slots]
    for i, option in enumerate(opts):
        p = opt_slot[i] if i < len(opt_slot) else -1
        if p < 0:
            continue
        aid = option.get("attackId")
        if isinstance(aid, int) and 0 < aid < N_CARD - ATTACK_TOKEN_BASE:
            card[p] = ATTACK_TOKEN_BASE + aid
            scal[p, 32] = 1.0

    bag_card, bag_count, bag_kind, bag_scal, bag_mask = _blank_bags()
    players = cur.get("players") or []
    mypl = players[me] if 0 <= me < len(players) else {}
    opp = 1 - me
    opl = players[opp] if 0 <= opp < len(players) else {}

    _fill_bag(bag_card, bag_count, bag_kind, bag_scal, bag_mask,
              BAG_DECK, K_DECK, state.get("decklist") or [])
    _fill_bag(bag_card, bag_count, bag_kind, bag_scal, bag_mask,
              BAG_MY_DISCARD, K_MY_DISCARD, mypl.get("discard") or [], side=0.0)
    _fill_bag(bag_card, bag_count, bag_kind, bag_scal, bag_mask,
              BAG_OPP_DISCARD, K_OPP_DISCARD, opl.get("discard") or [], side=1.0)

    history = [[], []]
    for log in state.get("logs") or []:
        if not isinstance(log, dict):
            continue
        owner = log.get("playerIndex")
        if owner not in (0, 1):
            continue
        for key in ("cardId", "cardIdTarget"):
            cid = log.get(key)
            if isinstance(cid, int) and 0 < cid < PHYSICAL_CARD_VOCAB:
                history[0 if owner == me else 1].append(cid)
    _fill_bag(bag_card, bag_count, bag_kind, bag_scal, bag_mask,
              BAG_MY_HISTORY, K_MY_HISTORY, history[0], side=0.0)
    _fill_bag(bag_card, bag_count, bag_kind, bag_scal, bag_mask,
              BAG_OPP_HISTORY, K_OPP_HISTORY, history[1], side=1.0)

    # The engine player ordering is absolute; map it to our/opp ordering so
    # bag position agrees with the twelve main board tokens.
    ordered_players = [mypl, opl]
    for board_pos, mon in enumerate(_board_mons(ordered_players)):
        attached = []
        if isinstance(mon, dict):
            attached.extend(mon.get("preEvolution") or [])
            attached.extend(mon.get("energyCards") or [])
            attached.extend(mon.get("tools") or [])
        _fill_bag(
            bag_card, bag_count, bag_kind, bag_scal, bag_mask,
            BAG_ATTACH_BASE + board_pos, K_ATTACHMENT, attached,
            side=1.0 if board_pos >= 6 else 0.0, board_pos=board_pos)

    return (kind, card, scal, mask, opt_slot, bag_card, bag_count,
            bag_kind, bag_scal, bag_mask)

