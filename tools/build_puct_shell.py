"""Emit a shell variant that changes ONLY how PUCT weights its prior.

The frozen shell turns net logits into a prior with

    pri = math.exp(min(6.0, sum(scores) / len(c)))

normalised, i.e. a temperature-1 softmax, and then explores with

    u = q + C_PUCT * prior * sqrt(total) / (1 + n_vis)      # C_PUCT = 1.4

Neither constant was chosen by measurement.  Together they make the prior close
to decisive: a BC pointer head trained with cross-entropy is confident, so one
option can take nearly all the prior mass, its siblings get an exploration term
near zero, and the search never accumulates the evidence to overturn the net.
Measured consequence, already in CLAUDE.md: **the search agrees with its own
prior 95.7% of the time.** The search is mostly verifying one move rather than
comparing several.

Two knobs, both deployment-safe:

  --floor F   every candidate gets at least F/len(candidates) of the prior
              mass, the rest distributed proportionally.  Deterministic, and it
              preserves the net's RANKING exactly -- it only stops a sibling
              from being starved to zero.
  --cpuct C   the exploration constant itself.

**Deliberately NOT offered: Dirichlet root noise.**  It is AlphaZero's
self-play exploration device and is switched off for evaluation; at deployment
it spends search budget on moves it already believes are worse.  Adding it here
would be copying a training trick into an inference path.

Gate any variant against the SAME archive built on the stock shell, at
1.1 s/move.  Four shell changes have been gated in this repo and all four died,
so the prior on this one should be low.  What makes it worth one more attempt is
that it adds no heuristic and reorders nothing: it only bounds how far PUCT is
allowed to ignore an option.

Usage:
    py tools/build_puct_shell.py --floor 0.10
    py tools/build_puct_shell.py --cpuct 2.5
    py tools/build_puct_shell.py --floor 0.10 --cpuct 2.5 --out foundation/x.py
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "foundation", "search_shell_main.py")
FROZEN_MD5 = "e54bc6590288e659d696d00d432c6cc4"

PRIOR_BLOCK = """    tot = sum(pri) or 1.0
    return [(c, p / tot) for c, p in zip(cands, pri)]"""

PRIOR_PATCH = """    tot = sum(pri) or 1.0
    prior = [p / tot for p in pri]
    # Prior floor. A confident net can take nearly all the mass, which leaves a
    # sibling's PUCT exploration term at ~0 and means the search never tries it
    # even once. Reserving FLOOR of the mass, spread evenly, bounds how far the
    # prior can starve an option while leaving the net's ranking untouched --
    # every candidate keeps its relative order, it just cannot reach zero.
    floor = {floor!r}
    if floor > 0.0 and len(prior) > 1:
        share = floor / len(prior)
        prior = [share + (1.0 - floor) * p for p in prior]
    return list(zip(cands, prior))"""

CPUCT_BLOCK = "    C_PUCT = 1.4"


def md5(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def function_span(lines, name):
    """(start, end) line indices of a top-level `def name(` / `class name`."""
    start = None
    for i, line in enumerate(lines):
        if start is None and (line.startswith(f"def {name}(")
                              or line.startswith(f"class {name}")):
            start = i
        elif start is not None and line and not line[0].isspace():
            return start, i
    return (start, len(lines)) if start is not None else (None, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor", type=float, default=0.0,
                        help="fraction of prior mass reserved and spread evenly")
    parser.add_argument("--cpuct", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.floor <= 0.0 and args.cpuct is None:
        parser.error("nothing to change: pass --floor and/or --cpuct")

    got = md5(FROZEN)
    if got != FROZEN_MD5:
        sys.exit(f"{FROZEN} is not the proven shell (md5 {got}); refusing")

    with open(FROZEN, encoding="utf-8") as handle:
        source = handle.read()
    patched = source

    if args.floor > 0.0:
        if patched.count(PRIOR_BLOCK) != 1:
            sys.exit("prior block not found exactly once; shell drifted")
        patched = patched.replace(PRIOR_BLOCK,
                                  PRIOR_PATCH.format(floor=args.floor))
    if args.cpuct is not None:
        if patched.count(CPUCT_BLOCK) != 1:
            sys.exit("C_PUCT line not found exactly once; shell drifted")
        patched = patched.replace(CPUCT_BLOCK, f"    C_PUCT = {args.cpuct!r}")

    tag = []
    if args.floor > 0.0:
        tag.append(f"floor{str(args.floor).replace('.', 'p')}")
    if args.cpuct is not None:
        tag.append(f"cpuct{str(args.cpuct).replace('.', 'p')}")
    out = args.out or os.path.join(ROOT, "foundation",
                                   f"search_shell_{'_'.join(tag)}.py")

    # Prove the edit is confined to the two places it claims to touch.
    # This must be a real diff, not a line-index comparison: the floor patch
    # INSERTS lines, which shifts every later line and would make an
    # index-wise compare report the whole rest of the file as changed.
    before, after = source.splitlines(), patched.splitlines()
    gen_start, gen_end = function_span(before, "_gen_candidates")
    if gen_start is None:
        sys.exit("could not locate _gen_candidates; shell drifted")
    stray = []
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(
            None, before, after).get_opcodes():
        if tag == "equal":
            continue
        for i in range(i1, max(i2, i1 + 1)):
            if i >= len(before):
                continue
            if gen_start <= i < gen_end:
                continue
            if before[i].strip() == "C_PUCT = 1.4":
                continue
            stray.append(before[i].strip()[:60])
    if stray:
        sys.exit("patch touched lines outside _gen_candidates/C_PUCT: "
                 f"{stray[:5]}")
    compile(patched, out, "exec")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(patched)

    print(f"wrote {out}")
    print(f"  floor={args.floor}  cpuct={args.cpuct}")
    print(f"  _gen_candidates spans lines {gen_start + 1}-{gen_end}")
    print(f"  {len(after) - len(before)} net lines added, no stray edits")
    print("  compiles: True")


if __name__ == "__main__":
    main()
