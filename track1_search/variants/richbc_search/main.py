"""Local harness for the rich-feature policy inside the proven search agent.

The submitted archive contains the real Track 1 ``main.py``.  This indirection
keeps the local A/B candidate tied to that exact source without another 44 KB
copy drifting out of sync.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "..", "agent"))
TRAIN = os.path.abspath(os.path.join(HERE, "..", "..", "train"))
sys.path.insert(0, BASE)
sys.path.insert(0, TRAIN)

with open(os.path.join(BASE, "main.py"), encoding="utf-8") as source:
    exec(compile(source.read(), __file__, "exec"), globals(), globals())
