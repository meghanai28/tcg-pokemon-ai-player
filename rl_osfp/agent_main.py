"""Fast, search-free masked OSFP policy for Kaggle inference."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

def _find_agent_dir():
    """Locate the directory holding our files.

    The Kaggle runner execs this file as a *string*, so ``__file__`` is absent,
    and it appends the agent directory to ``sys.path`` only for the duration of
    that exec - it is popped before ``agent()`` is ever called. Guessing a fixed
    path is therefore unsafe: resolve by finding the files themselves.
    """
    candidates = []
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    candidates.append("/kaggle_simulations/agent")
    candidates.append(os.getcwd())
    candidates.extend(p for p in sys.path if p)
    for base in candidates:
        try:
            if base and os.path.isfile(os.path.join(base, "deck.csv")):
                return base
        except Exception:
            continue
    return candidates[0] if candidates else os.getcwd()


AGENT_DIR = _find_agent_dir()
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

# Rewritten by the packager with the 60-card list.
EMBEDDED_DECK: list[int] = []

_NET = None
_NF = None
_CARD = None
_ATTACK = None
_DECK = None


def _load() -> None:
    global _NET, _NF, _CARD, _ATTACK, _DECK
    if _NET is not None:
        return
    import nn_features_rich as features
    from nn_infer import NumpyNet
    from cg.engine import get_lib

    with open(os.path.join(AGENT_DIR, "deck.csv"), encoding="utf-8") as source:
        _DECK = [int(line) for line in source if line.strip()]
    if len(_DECK) != 60:
        raise ValueError("deck.csv must contain exactly 60 cards")
    lib = get_lib()
    _CARD = {int(x["cardId"]): x for x in json.loads(lib.AllCard().decode())}
    _ATTACK = {int(x["attackId"]): x for x in json.loads(lib.AllAttack().decode())}
    _NF = features
    _NET = NumpyNet(os.path.join(AGENT_DIR, "model.npz"))


def _bounds(select: dict, n: int) -> tuple[int, int]:
    low = max(0, min(int(select.get("minCount", 1) or 0), n, 60))
    high = max(low, min(int(select.get("maxCount", max(low, 1)) or 0), n, 60))
    return low, high


def _fallback(select: dict) -> list[int]:
    options = select.get("option") or []
    low, high = _bounds(select, len(options))
    count = low if low else min(1, high)
    return list(range(count))


def _policy(observation: dict) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option") or []
    n = len(options)
    if n == 0:
        return []
    low, high = _bounds(select, n)
    if low == high == n:
        return list(range(n))
    current = observation.get("current") or {}
    me = int(current.get("yourIndex", 0))
    kind, card, scal, mask, slots = _NF.encode(
        {"current": current, "select": select, "decklist": _DECK},
        me, _CARD, _ATTACK, None,
    )
    if len(slots) != n or any(slot < 0 for slot in slots):
        return _fallback(select)
    option, count, _value = _NET.forward(
        kind[None].astype(np.int64), card[None].astype(np.int64), scal[None],
        mask[None], np.asarray([int(select.get("context") or 0)]),
        np.asarray([int(select.get("type") or 0)]),
    )
    count_logits = count[0].copy()
    count_logits[:low] = -1e9
    count_logits[high + 1:] = -1e9
    take = int(np.argmax(count_logits))
    if take == n:
        return list(range(n))
    ranked = sorted(range(n), key=lambda index: float(option[0, slots[index]]), reverse=True)
    return ranked[:take]


def agent(observation):
    try:
        _load()
        select = observation.get("select") if hasattr(observation, "get") else observation["select"]
        if select is None:
            return list(_DECK or EMBEDDED_DECK)
        return _policy(observation)
    except Exception:
        try:
            select = observation.get("select") or {}
            if not select:
                # Never return a short deck: that fails the episode outright.
                return list(_DECK or EMBEDDED_DECK)
            return _fallback(select)
        except Exception:
            if not (observation or {}).get("select"):
                return list(EMBEDDED_DECK)
            return [0]
