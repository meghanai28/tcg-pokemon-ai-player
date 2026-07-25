"""Load the versioned rich encoder used by training for local A/B tests."""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.abspath(
    os.path.join(HERE, "..", "..", "train", "nn_features_rich.py"))
with open(SOURCE, encoding="utf-8") as source:
    exec(compile(source.read(), __file__, "exec"), globals(), globals())

# This checkpoint predates deck conditioning.  Keep its exact feature ABI for
# fair A/B comparisons even though the shared rich encoder has moved forward.
DECK_AWARE = False
_encode_without_deck = encode


def encode(state, me, card_db, attack_db, opt_scores=None):
    state = dict(state)
    state.pop("decklist", None)
    return _encode_without_deck(
        state, me, card_db, attack_db, opt_scores)
