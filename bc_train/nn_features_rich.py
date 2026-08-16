"""Versioned feature fixes for replay imitation.

The original encoder expects ``option.cardId``.  Competition actions usually
identify cards indirectly with ``area`` + ``index`` instead, which made many
legal options indistinguishable to the policy.  This wrapper preserves the
existing 53-token/32-scalar ABI while resolving those references and filling
previously unused scalar slots.

Keep this file beside the original ``nn_features.py`` in a submission.  The
wrapper deliberately imports that local module so training and CPU inference
share exactly the same base representation.
"""
from __future__ import annotations

from collections import Counter

try:
    from . import nn_features as _base
    from .nn_features import *  # noqa: F401,F403 - re-export fixed-shape ABI
except ImportError:  # direct-script execution with PYTHONPATH=bc_train
    import nn_features as _base  # type: ignore[no-redef]
    from nn_features import *  # type: ignore[no-redef]  # noqa: F401,F403


DECK_AWARE = True

DECK = 1
HAND = 2
DISCARD = 3
ACTIVE = 4
BENCH = 5
PRIZE = 6
STADIUM = 7
LOOKING = 12

OT_CARD = 3
OT_TOOL_CARD = 4
OT_ENERGY_CARD = 5
OT_ENERGY = 6
OT_PLAY = 7


def _at(seq, index):
    if not isinstance(seq, list) or not isinstance(index, int):
        return None
    return seq[index] if 0 <= index < len(seq) else None


def _zone_card(cur, sel, option, me):
    """Resolve an option's source card from the engine's indirect reference."""
    players = cur.get("players") or []
    owner = option.get("playerIndex")
    if not isinstance(owner, int) or not (0 <= owner < len(players)):
        owner = me
    pl = players[owner] if 0 <= owner < len(players) else {}
    area = option.get("area")
    index = option.get("index")
    typ = option.get("type")

    # PLAY encodes only a hand index.
    if typ == OT_PLAY and area is None:
        area = HAND

    if area == DECK:
        item = _at(sel.get("deck") or [], index)
    elif area == LOOKING:
        # Revealed search/look cards are part of the public current state.  In
        # live observations select.deck is normally null, so treating LOOKING
        # like DECK erased every revealed identity to card ID zero.
        looking = cur.get("looking")
        if not isinstance(looking, list):
            # Retain compatibility with older replay schemas that embedded the
            # temporary reveal list in the selection object.
            looking = sel.get("deck") or []
        item = _at(looking, index)
    elif area == HAND:
        item = _at(pl.get("hand") or [], index)
    elif area == DISCARD:
        item = _at(pl.get("discard") or [], index)
    elif area == ACTIVE:
        item = _at(pl.get("active") or [], index or 0)
    elif area == BENCH:
        item = _at(pl.get("bench") or [], index)
    elif area == PRIZE:
        item = _at(pl.get("prize") or [], index)
    elif area == STADIUM:
        stadium = cur.get("stadium")
        item = stadium[0] if isinstance(stadium, list) and stadium else stadium
    else:
        item = None

    # Attached-card choices first point at the Pokemon, then at an attachment.
    if isinstance(item, dict) and typ in (OT_ENERGY_CARD, OT_ENERGY):
        cards = item.get("energyCards") or []
        attached = _at(cards, option.get("energyIndex"))
        if isinstance(attached, dict):
            item = attached
    elif isinstance(item, dict) and typ == OT_TOOL_CARD:
        attached = _at(item.get("tools") or [], option.get("toolIndex"))
        if isinstance(attached, dict):
            item = attached
    return item if isinstance(item, dict) else None


def _in_play_target(cur, option, me):
    players = cur.get("players") or []
    owner = option.get("playerIndex")
    if not isinstance(owner, int) or not (0 <= owner < len(players)):
        owner = me
    if not (0 <= owner < len(players)):
        return None
    pl = players[owner]
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    if area == ACTIVE:
        item = _at(pl.get("active") or [], index or 0)
    elif area == BENCH:
        item = _at(pl.get("bench") or [], index)
    else:
        item = None
    return item if isinstance(item, dict) else None


def _card_id(card):
    return card.get("id", 0) if isinstance(card, dict) else 0


def _deck_ids(decklist):
    """Normalise the initial 60-card action (or a list of card dicts)."""
    out = []
    for item in decklist or []:
        cid = item.get("id") if isinstance(item, dict) else item
        if isinstance(cid, int) and 0 < cid < N_CARD:
            out.append(cid)
    return out


def _deck_anchor(deck_ids, card_db):
    """Choose a stable archetype token from the player's known deck.

    The global token previously used card id 0 for every deck.  Prefer a
    frequently played Pokemon, breaking ties toward evolved/ex Pokemon, so its
    existing card embedding can cheaply identify the deck's main line.
    """
    counts = Counter(deck_ids)
    pokemon = [
        (cid, count) for cid, count in counts.items()
        if card_db.get(cid, {}).get("cardType") == 0
    ]
    if not pokemon:
        return 0

    def rank(item):
        cid, count = item
        c = card_db.get(cid, {})
        return (count, bool(c.get("stage2")), bool(c.get("ex")),
                bool(c.get("stage1")), c.get("hp") or 0, -cid)

    return max(pokemon, key=rank)[0]


def encode(state, me, card_db, attack_db, opt_scores=None):
    kind, card, scal, mask, opt_slot = _base.encode(
        state, me, card_db, attack_db, opt_scores)
    cur = state["current"]
    sel = state["select"] or {}

    # Previously unused global slots: effect resources and the cards causing the
    # current nested choice.
    g = scal[0]
    g[19] = _base._clip01((sel.get("remainDamageCounter") or 0) / 30.0)
    g[20] = _base._clip01((sel.get("remainEnergyCost") or 0) / 10.0)
    g[21] = _base._clip01(_card_id(sel.get("contextCard")) / float(N_CARD - 1))
    g[22] = _base._clip01(_card_id(sel.get("effect")) / float(N_CARD - 1))
    g[23] = _base._clip01(len(sel.get("deck") or []) / 60.0)

    # Deck conditioning matters in a matchup-heavy TCG: the same visible state
    # can call for different lines depending on what the policy can draw into.
    # At inference our own submitted deck is known exactly; replay ingestion
    # recovers the same list from steps[1][player].action.
    deck_ids = _deck_ids(state.get("decklist"))
    if deck_ids:
        card[0] = _deck_anchor(deck_ids, card_db)
        type_counts = [0] * 7
        for cid in deck_ids:
            card_type = card_db.get(cid, {}).get("cardType")
            if isinstance(card_type, int) and 0 <= card_type < len(type_counts):
                type_counts[card_type] += 1
        denom = float(len(deck_ids))
        for card_type, count in enumerate(type_counts):
            g[24 + card_type] = count / denom

    # The transformer has no positional embedding.  Expose stable hand/board
    # positions so an option can refer to a specific visible card.
    for i in range(MAX_HAND):
        p = 1 + MAX_BOARD + i
        if mask[p] > 0.5:
            scal[p, 10] = i / float(max(1, MAX_HAND - 1))
    for p in range(1, 1 + MAX_BOARD):
        if mask[p] > 0.5:
            scal[p, 17] = ((p - 1) % 6) / 5.0

    opts = sel.get("option") or []
    for i, option in enumerate(opts):
        p = opt_slot[i] if i < len(opt_slot) else -1
        if p < 0:
            continue
        source = _zone_card(cur, sel, option, me)
        cid = option.get("cardId") or _card_id(source)
        if cid:
            card[p] = min(int(cid), N_CARD - 1)

        s = scal[p]
        index = option.get("index")
        in_play = option.get("inPlayIndex")
        s[23] = _base._clip01(((index if index is not None else -1) + 1) / 60.0)
        s[24] = _base._clip01((option.get("inPlayArea") or 0) / 12.0)
        s[25] = _base._clip01((option.get("energyIndex") or 0) / 6.0)
        s[26] = _base._clip01((option.get("toolIndex") or 0) / 2.0)
        s[27] = _base._clip01((option.get("count") or 0) / 6.0)
        s[28] = _base._clip01(
            (option.get("specialConditionType") or 0) / 4.0)
        s[29] = i / float(max(1, len(opts) - 1))
        s[30] = 1.0 if source is not None else 0.0
        target = _in_play_target(cur, option, me)
        s[31] = _base._clip01(_card_id(target) / float(N_CARD - 1))

    return kind, card, scal, mask, opt_slot
