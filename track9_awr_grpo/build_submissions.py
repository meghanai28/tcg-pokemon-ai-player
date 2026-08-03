"""Prepare the unchanged 972 control and the selected Track 9 challenger."""
from __future__ import annotations

import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from track5_grpo.build_submissions import build  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    source = os.path.join(ROOT, "submission_grpo_controlled_tech_grim.tar.gz")
    control = os.path.join(ROOT, "submission_track9_control_tech_grim_972.tar.gz")
    if not os.path.isfile(source):
        raise SystemExit(f"missing immutable Track 6 control archive: {source}")
    shutil.copy2(source, control)
    if digest(source) != digest(control):
        raise RuntimeError("control copy is not byte-identical")
    challenger = build(
        "track9_awr_grpo_tech_grim",
        os.path.join(HERE, "model_tech_awr_grpo_final.npz"),
        os.path.join(ROOT, "track6_controlled", "decks", "tech_grim.csv"),
        ROOT, archive_prefix="submission_")
    print(f"control   {digest(control)}  {control}")
    print(f"challenger {digest(challenger)}  {challenger}")


if __name__ == "__main__":
    main()
