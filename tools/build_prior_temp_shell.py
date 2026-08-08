"""Emit a shell whose ONLY difference is the PUCT prior temperature.

MCTS with a policy prior is implicitly KL-regularised toward that prior, with
the regularisation strength set by how sharp the prior is
(https://arxiv.org/abs/2112.07544, and the PUCT literature generally). That
strength is a tunable with an optimum, and tuning it is reported to beat
imitation learning alone.

**This project has never tuned it.** The frozen shell converts net logits to a
prior with

    pri = math.exp(min(6.0, sum(scores) / len(c)))

which is temperature 1 by construction, chosen by nobody. Two measurements say
the knob is live rather than cosmetic:

  * dividing the logits by 3 during label generation moved the search's
    disagreement with its own prior from 4.3% to 22.8%, so the prior really is
    what pins the search down;
  * stripping the net entirely drops the archive from 80.0% to 25.0%, so the
    prior is most of the agent and its weighting is not a detail.

Unlike the three shell changes that were gated and died, this one is not a new
heuristic. It rescales an existing term whose correct value was never measured,
and the ranking of options is untouched, so a temperature above 1 only makes
PUCT verify alternatives it currently ignores.

Usage:
    py tools/build_prior_temp_shell.py --temp 1.5
    py tools/build_prior_temp_shell.py --temp 0.7 --out foundation/shell_t07.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "foundation", "search_shell_main.py")
FROZEN_MD5 = "e54bc6590288e659d696d00d432c6cc4"

TARGET = ("        pri.append(math.exp(min(6.0, sum(scores[i] for i in c) "
          "/ max(len(c), 1))))")


def md5(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", type=float, required=True,
                        help=">1 flattens the prior so PUCT explores more; "
                             "<1 sharpens it toward the net's first choice")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    got = md5(FROZEN)
    if got != FROZEN_MD5:
        sys.exit(f"{FROZEN} is not the proven shell (md5 {got}); refusing to build")

    with open(FROZEN, encoding="utf-8") as handle:
        source = handle.read()

    matches = [line for line in source.splitlines()
               if "pri.append(math.exp(min(6.0" in line]
    if len(matches) != 1:
        sys.exit(f"expected exactly one prior line, found {len(matches)}")
    original = matches[0]
    replacement = original.replace(
        "/ max(len(c), 1))))",
        f"/ max(len(c), 1) / {args.temp!r})))")
    patched = source.replace(original, replacement)

    out = args.out or os.path.join(
        ROOT, "foundation",
        f"search_shell_priortemp_{str(args.temp).replace('.', 'p')}.py")

    # Prove exactly one line moved.
    before, after = source.splitlines(), patched.splitlines()
    if len(before) != len(after):
        sys.exit(f"line count changed {len(before)} -> {len(after)}")
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    if len(changed) != 1:
        sys.exit(f"{len(changed)} lines changed, expected exactly 1")
    compile(patched, out, "exec")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(patched)

    print(f"wrote {out}")
    print(f"  exactly 1 line changed, at line {changed[0] + 1}")
    print(f"  before: {before[changed[0]].strip()}")
    print(f"  after : {after[changed[0]].strip()}")
    print("  compiles: True")


if __name__ == "__main__":
    main()
