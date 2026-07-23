"""Load the versioned rich encoder used by training for local A/B tests."""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.abspath(
    os.path.join(HERE, "..", "..", "train", "nn_features_rich.py"))
with open(SOURCE, encoding="utf-8") as source:
    exec(compile(source.read(), __file__, "exec"), globals(), globals())
