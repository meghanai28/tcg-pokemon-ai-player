"""Feature encoding shared by training and inference.

Encodes (state, select, me) into fixed-shape token arrays for the policy/value
transformer. This file lives in the submission so the trainer imports the same
code the deployed agent runs — feature drift between train and inference is
impossible by construction.

Token layout (SEQ = 1 + MAX_BOARD + MAX_HAND + MAX_OPT):
  [0]                     global token
  [1 .. 12]               board slots: my active, my bench x5, opp active, opp bench x5
  [13 .. 28]              my hand cards (up to 16, truncated by keep-value)
  [29 .. 52]              option tokens (up to 24, truncated by heuristic score)

Each token = (kind, card_idx, scalars[F]).  kind: 0=global 1=board 2=hand 3=option.
Padding tokens have kind == PAD_KIND and a mask of 0.
"""
from __future__ import annotations

import numpy as np

MAX_BOARD = 12
MAX_HAND = 16
MAX_OPT = 24
SEQ = 1 + MAX_BOARD + MAX_HAND + MAX_OPT
F = 32
N_KIND = 4
PAD_KIND = 0          # padding shares kind 0 but mask==0 distinguishes it
N_CARD = 1300         # card-id vocabulary (ids observed <= 1267)
N_CTX = 50            # SelectContext vocabulary (49 values)
N_STYPE = 12          # SelectType vocabulary (11 values)

OPT_BASE = 1 + MAX_BOARD + MAX_HAND


def _clip01(x):
    return 0.0 if x is None else max(0.0, min(1.0, x))


def encode(state, me, card_db, attack_db, opt_scores=None):
    """Encode one decision point.

    state: dict with "current" and "select" (raw engine JSON)
    me: our player index
    card_db/attack_db: cardId->dict / attackId->dict
    opt_scores: optional list of heuristic scores per option (for truncation order)

    Returns (kind[SEQ] int8, card[SEQ] int16, scal[SEQ,F] f32, mask[SEQ] f32,
             opt_slot: list mapping option-index -> token position or -1)
    """
    cur = state["current"]
    sel = state["select"] or {}
    players = cur["players"]
    mypl, opl = players[me], players[1 - me]

    kind = np.zeros(SEQ, dtype=np.int8)
    card = np.zeros(SEQ, dtype=np.int16)
    scal = np.zeros((SEQ, F), dtype=np.float32)
    mask = np.zeros(SEQ, dtype=np.float32)

    # ---- global token
    kind[0] = 0
    mask[0] = 1.0
    g = scal[0]
    g[0] = _clip01(cur.get("turn", 0) / 30.0)
    g[1] = _clip01(len(mypl.get("prize") or []) / 6.0)
    g[2] = _clip01(len(opl.get("prize") or []) / 6.0)
    g[3] = _clip01(mypl.get("deckCount", 0) / 60.0)
    g[4] = _clip01(opl.get("deckCount", 0) / 60.0)
    my_hand = mypl.get("hand")
    g[5] = _clip01((len(my_hand) if my_hand is not None else mypl.get("handCount", 0)) / 16.0)
    g[6] = _clip01(opl.get("handCount", 0) / 16.0)
    g[7] = 1.0 if cur.get("supporterPlayed") else 0.0
    g[8] = 1.0 if cur.get("energyAttached") else 0.0
    g[9] = 1.0 if cur.get("retreated") else 0.0
    g[10] = 1.0 if cur.get("stadiumPlayed") else 0.0
    g[11] = 1.0 if cur.get("firstPlayer", -1) == me else 0.0
    g[12] = 1.0 if cur.get("yourIndex", me) == me else 0.0
    g[13] = _clip01(sel.get("minCount", 1) / 6.0)
    g[14] = _clip01(sel.get("maxCount", 1) / 6.0)
    g[15] = _clip01((sel.get("type") or 0) / float(N_STYPE))
    g[16] = _clip01((sel.get("context") or 0) / float(N_CTX))
    g[17] = _clip01(len(sel.get("option") or []) / 24.0)
    g[18] = _clip01(cur.get("turnActionCount", 0) / 20.0)

    # ---- board tokens
    def put_mon(pos, mon, side_me, is_active, pl):
        kind[pos] = 1
        mask[pos] = 1.0
        s = scal[pos]
        if mon is None:
            s[0] = 1.0   # face-down / unknown
            return
        cid = mon.get("id", 0)
        card[pos] = min(cid, N_CARD - 1)
        c = card_db.get(cid, {})
        max_hp = mon.get("maxHp") or 1
        s[1] = _clip01(mon.get("hp", 0) / max_hp)
        s[2] = _clip01(max_hp / 340.0)
        s[3] = _clip01(len(mon.get("energies") or []) / 5.0)
        s[4] = _clip01(len(mon.get("tools") or []) / 2.0)
        stage = 2.0 if c.get("stage2") else (1.0 if c.get("stage1") else 0.0)
        s[5] = stage / 2.0
        s[6] = 1.0 if c.get("ex") else 0.0
        best = 0
        for aid in c.get("attacks") or []:
            best = max(best, attack_db.get(aid, {}).get("damage", 0) or 0)
        s[7] = _clip01(best / 300.0)
        s[8] = 1.0 if side_me else 0.0
        s[9] = 1.0 if is_active else 0.0
        s[10] = 1.0 if mon.get("appearThisTurn") else 0.0
        s[11] = _clip01((c.get("retreatCost") or 0) / 4.0)
        if is_active:
            s[12] = 1.0 if pl.get("poisoned") else 0.0
            s[13] = 1.0 if pl.get("burned") else 0.0
            s[14] = 1.0 if pl.get("asleep") else 0.0
            s[15] = 1.0 if pl.get("paralyzed") else 0.0
            s[16] = 1.0 if pl.get("confused") else 0.0

    pos = 1
    for side_me, pl in ((True, mypl), (False, opl)):
        act = pl.get("active") or []
        if act:
            put_mon(pos, act[0], side_me, True, pl)
        else:
            kind[pos] = 1                     # empty active slot stays masked out
        pos += 1
        bench = pl.get("bench") or []
        for i in range(5):
            if i < len(bench):
                put_mon(pos, bench[i], side_me, False, pl)
            pos += 1

    # ---- hand tokens (ours only), highest keep-value first
    base = 1 + MAX_BOARD
    hand_ids = [c["id"] for c in (my_hand or [])]
    if len(hand_ids) > MAX_HAND:
        hand_ids = hand_ids[:MAX_HAND]
    for i, cid in enumerate(hand_ids):
        p = base + i
        kind[p] = 2
        mask[p] = 1.0
        card[p] = min(cid, N_CARD - 1)
        c = card_db.get(cid, {})
        s = scal[p]
        ct = c.get("cardType")
        if ct is not None and 0 <= ct <= 6:
            s[ct] = 1.0                       # one-hot card type in slots 0..6
        s[7] = 1.0 if c.get("basic") else 0.0
        s[8] = _clip01((c.get("hp") or 0) / 340.0)
        best = 0
        for aid in c.get("attacks") or []:
            best = max(best, attack_db.get(aid, {}).get("damage", 0) or 0)
        s[9] = _clip01(best / 300.0)

    # ---- option tokens
    opts = sel.get("option") or []
    order = list(range(len(opts)))
    if len(order) > MAX_OPT and opt_scores is not None:
        order.sort(key=lambda i: -opt_scores[i])
        order = order[:MAX_OPT]
    elif len(order) > MAX_OPT:
        order = order[:MAX_OPT]
    opt_slot = [-1] * len(opts)
    for j, i in enumerate(order):
        o = opts[i]
        p = OPT_BASE + j
        opt_slot[i] = p
        kind[p] = 3
        mask[p] = 1.0
        t = o.get("type", 0)
        s = scal[p]
        if 0 <= t <= 16:
            s[t] = 1.0                        # one-hot option type in slots 0..16
        cid = o.get("cardId")
        if cid:
            card[p] = min(cid, N_CARD - 1)
        s[17] = _clip01((o.get("number") or 0) / 6.0)
        aid = o.get("attackId")
        if aid:
            a = attack_db.get(aid, {})
            s[18] = _clip01((a.get("damage", 0) or 0) / 300.0)
            s[19] = _clip01(len(a.get("energies") or []) / 4.0)
        s[20] = _clip01((o.get("area") or 0) / 12.0)
        ipi = o.get("inPlayIndex")
        s[21] = _clip01(((ipi if ipi is not None else -1) + 1) / 6.0)
        s[22] = 1.0 if (o.get("playerIndex") == me) else 0.0

    return kind, card, scal, mask, opt_slot
