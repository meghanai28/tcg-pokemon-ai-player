"""Feature ABI v2: exact deck identity and semantic choice tokens.

This is intentionally a small, mechanically-upgradable step beyond
``nn_features_rich`` rather than a wholesale representation rewrite:

* the global card embedding identifies the exact 60-card multiset via a
  reserved hash bucket, so nearby tech variants no longer alias;
* ``contextCard`` and ``effect`` become real card-embedding tokens instead of
  arbitrary normalized ID scalars;
* the corrected rich resolver preserves public LOOKING-card identities.

The two added tokens shift option positions by two (53 -> 55 tokens).  Old
rich shards can be upgraded losslessly for these fields at load time because
they already store deck hashes and the two normalized IDs.
"""
from __future__ import annotations

import hashlib

import numpy as np

try:
    from . import nn_features_rich as _rich
    from .nn_features_rich import *  # noqa: F401,F403
except ImportError:  # submission-local import
    import nn_features_rich as _rich  # type: ignore[no-redef]
    from nn_features_rich import *  # type: ignore[no-redef]  # noqa: F401,F403


FEATURE_VERSION = 2
OLD_OPT_BASE = _rich.OPT_BASE
CONTEXT_SLOT = OLD_OPT_BASE
EFFECT_SLOT = OLD_OPT_BASE + 1
OPT_BASE = OLD_OPT_BASE + 2
SEQ = _rich.SEQ + 2

# Kinds 0..3 retain their original meaning.  Separate kinds let the network
# distinguish a card causing a nested choice from an effect card.
N_KIND = 6

# Physical card IDs occupy [0,1299].  Exact-deck identity uses reserved hash
# buckets.  6,892 buckets make collisions rare while adding only ~2.2M
# parameters to a 320-wide model.
PHYSICAL_CARD_VOCAB = _rich.N_CARD
N_CARD = 8192
DECK_TOKEN_BASE = PHYSICAL_CARD_VOCAB
DECK_TOKEN_COUNT = N_CARD - DECK_TOKEN_BASE


def stable_deck_id(decklist) -> int:
    ids = []
    for item in decklist or []:
        cid = item.get("id") if isinstance(item, dict) else item
        if isinstance(cid, int) and 0 < cid < PHYSICAL_CARD_VOCAB:
            ids.append(int(cid))
    if not ids:
        return 0
    raw = str(tuple(sorted(ids))).encode("utf-8", "replace")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def deck_token_from_id(deck_id: int) -> int:
    deck_id = int(deck_id)
    return (DECK_TOKEN_BASE + deck_id % DECK_TOKEN_COUNT) if deck_id else 0


def _blank(batch_shape=()):
    shape = tuple(batch_shape)
    return (
        np.zeros(shape + (SEQ,), dtype=np.int8),
        np.zeros(shape + (SEQ,), dtype=np.int16),
        np.zeros(shape + (SEQ, F), dtype=np.float32),
        np.zeros(shape + (SEQ,), dtype=np.float32),
    )


def _upgrade_core(kind, card, scal, mask, deck_ids=None):
    """Upgrade one or a batch of rich ABI tensors to v2."""
    kind = np.asarray(kind)
    card = np.asarray(card)
    scal = np.asarray(scal)
    mask = np.asarray(mask)
    if kind.shape[-1] == SEQ:
        return (kind.copy(), card.copy(), scal.copy(), mask.copy())
    if (kind.shape[-1] != _rich.SEQ or card.shape[-1] != _rich.SEQ or
            scal.shape[-2:] != (_rich.SEQ, F) or
            mask.shape[-1] != _rich.SEQ):
        raise ValueError("expected rich ABI tensors with sequence length 53")

    out_kind, out_card, out_scal, out_mask = _blank(kind.shape[:-1])
    out_kind[..., :OLD_OPT_BASE] = kind[..., :OLD_OPT_BASE]
    out_card[..., :OLD_OPT_BASE] = card[..., :OLD_OPT_BASE]
    out_scal[..., :OLD_OPT_BASE, :] = scal[..., :OLD_OPT_BASE, :]
    out_mask[..., :OLD_OPT_BASE] = mask[..., :OLD_OPT_BASE]
    out_kind[..., OPT_BASE:] = kind[..., OLD_OPT_BASE:]
    out_card[..., OPT_BASE:] = card[..., OLD_OPT_BASE:]
    out_scal[..., OPT_BASE:, :] = scal[..., OLD_OPT_BASE:, :]
    out_mask[..., OPT_BASE:] = mask[..., OLD_OPT_BASE:]

    # Rich v1 stored both IDs normalized by 1299, so rounding recovers the
    # exact integer for historical shards.
    context_id = np.rint(
        scal[..., 0, 21] * float(PHYSICAL_CARD_VOCAB - 1)).astype(np.int64)
    effect_id = np.rint(
        scal[..., 0, 22] * float(PHYSICAL_CARD_VOCAB - 1)).astype(np.int64)
    for slot, token_kind, ids in (
            (CONTEXT_SLOT, 4, context_id), (EFFECT_SLOT, 5, effect_id)):
        present = (ids > 0) & (ids < PHYSICAL_CARD_VOCAB)
        out_kind[..., slot] = token_kind
        out_card[..., slot] = np.where(present, ids, 0).astype(np.int16)
        out_mask[..., slot] = present.astype(np.float32)
        out_scal[..., slot, 0] = present.astype(np.float32)

    if deck_ids is not None:
        deck_ids = np.asarray(deck_ids, dtype=np.uint64)
        tokens = np.where(
            deck_ids != 0,
            DECK_TOKEN_BASE + deck_ids % DECK_TOKEN_COUNT,
            0,
        )
        out_card[..., 0] = tokens.astype(np.int16)
    return out_kind, out_card, out_scal, out_mask


def upgrade_policy(pi):
    pi = np.asarray(pi)
    if pi.shape[-1] == SEQ:
        return pi.copy()
    if pi.shape[-1] != _rich.SEQ:
        raise ValueError("expected policy sequence length 53 or 55")
    out = np.zeros(pi.shape[:-1] + (SEQ,), dtype=pi.dtype)
    out[..., :OLD_OPT_BASE] = pi[..., :OLD_OPT_BASE]
    out[..., OPT_BASE:] = pi[..., OLD_OPT_BASE:]
    return out


def upgrade_action_tokens(tokens):
    tokens = np.asarray(tokens)
    out = tokens.copy()
    live = out >= OLD_OPT_BASE
    out[live] += 2
    return out


def encode(state, me, card_db, attack_db, opt_scores=None):
    kind, card, scal, mask, old_slots = _rich.encode(
        state, me, card_db, attack_db, opt_scores)
    deck_id = stable_deck_id(state.get("decklist"))
    kind, card, scal, mask = _upgrade_core(
        kind, card, scal, mask, np.asarray(deck_id, dtype=np.uint64))
    opt_slot = [slot + 2 if slot >= OLD_OPT_BASE else slot
                for slot in old_slots]
    return kind, card, scal, mask, opt_slot


def upgrade_batch(kind, card, scal, mask, pi, deck_ids=None):
    kind, card, scal, mask = _upgrade_core(
        kind, card, scal, mask, deck_ids)
    return kind, card, scal, mask, upgrade_policy(pi)
