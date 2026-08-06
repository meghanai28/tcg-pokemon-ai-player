"""Emit a shell whose ONLY difference from the frozen one is `_evaluate`.

The frozen `main.py` is the single most valuable artifact here: md5-identical
inside the 972.0, 903.2 and 848.9 archives.  It is never hand-edited.  This
generates a sibling with one function swapped, and then proves that is the only
thing that moved, so the evaluator can be gated as a single variable.

The replacement keeps the shipped function's exact shape: read the terminal
result, otherwise walk both boards once and combine a dozen scalars.  Same
number of multiply-adds, plus one `exp`.  That matters because search throughput
decides here, and heuristic leaves ran ~7,500 simulations against a neural
leaf's ~220.

What changes is only the constants, and where they come from.  The shipped ones
were guessed; these are fitted by logistic regression against the actual winner
of 60,000-odd real ladder positions, so the output is a calibrated win
probability mapped into [-1, 1] rather than an unnormalised score.

Run `tools/fit_evaluator.py` first to produce the weights.

Usage:
    py tools/build_fiteval_shell.py
    py tools/build_fiteval_shell.py --fit data/evaluator_fit.json
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "foundation", "search_shell_main.py")
FROZEN_MD5 = "e54bc6590288e659d696d00d432c6cc4"

TEMPLATE = '''def _evaluate(state, me):
    """Fitted static evaluator.

    Same features and same cost as the shipped version; the weights are fitted
    against real game outcomes instead of guessed.  See tools/fit_evaluator.py.
    Shipped AUC {auc_shipped:.4f}, this {auc_refit:.4f}, on {states:,} positions.

    The weights are only meaningful as a set.  Two of them are negative where
    the shipped version had them positive, which does not mean extra Pokemon or
    stage-2s are bad; it means that once total HP and energy are already in the
    sum, additional bodies beyond that are spread-thin resources.
    """
    cur = state["current"]
    res = cur.get("result", -1)
    if res >= 0:
        if res == me:
            return 1.0
        if res == 1 - me:
            return -1.0
        return 0.0
    mypl = cur["players"][me]
    opl = cur["players"][1 - me]

    my_board = [m for m in list(mypl.get("active") or []) + list(mypl.get("bench") or []) if m]
    opp_board = [m for m in list(opl.get("active") or []) + list(opl.get("bench") or []) if m]

    mh = men = mat = 0.0
    ms1 = ms2 = 0
    for m in my_board:
        mh += m.get("hp", 0) / (m.get("maxHp") or 1)
        men += len(m.get("energies") or [])
        mat += min(_max_attack_damage(m.get("id")), 300) / 10.0
        c = _card(m.get("id"))
        if c.get("stage1"):
            ms1 += 1
        if c.get("stage2"):
            ms2 += 1
    oh = oen = oat = 0.0
    os1 = os2 = 0
    for m in opp_board:
        oh += m.get("hp", 0) / (m.get("maxHp") or 1)
        oen += len(m.get("energies") or [])
        oat += min(_max_attack_damage(m.get("id")), 300) / 10.0
        c = _card(m.get("id"))
        if c.get("stage1"):
            os1 += 1
        if c.get("stage2"):
            os2 += 1

    my_prize = len(mypl.get("prize") or [])
    opp_prize = len(opl.get("prize") or [])
    hand_n = len(mypl.get("hand") or []) if mypl.get("hand") is not None else mypl.get("handCount", 0)

    status = 0
    for flag in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
        if opl.get(flag):
            status += 1
        if mypl.get(flag):
            status -= 1

    z = ({w0!r} * (opp_prize - my_prize) / 6.0
         + {w1!r} * (len(my_board) - len(opp_board))
         + {w2!r} * (mh - oh)
         + {w3!r} * (men - oen)
         + {w4!r} * (mat - oat)
         + {w5!r} * (ms1 - os1)
         + {w6!r} * (ms2 - os2)
         + {w7!r} * min(hand_n, 16)
         + {w8!r} * min(opl.get("handCount", 0), 16)
         + {w9!r} * (1.0 if mypl.get("deckCount", 1) == 0 else 0.0)
         + {w10!r} * (1.0 if 0 < mypl.get("deckCount", 99) <= 2 else 0.0)
         + {w11!r} * (1.0 if opl.get("deckCount", 1) == 0 else 0.0)
         + {w12!r} * status
         + {bias!r}) / {temp!r}
    if z >= 30.0:
        return 0.97
    if z <= -30.0:
        return -0.97
    return max(-0.97, min(0.97, 2.0 / (1.0 + math.exp(-z)) - 1.0))
'''


def md5(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", default=os.path.join(ROOT, "data", "evaluator_fit.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "foundation",
                                                      "search_shell_fiteval.py"))
    args = parser.parse_args()

    got = md5(FROZEN)
    if got != FROZEN_MD5:
        sys.exit(f"{FROZEN} is not the proven shell (md5 {got}); refusing to build")

    with open(args.fit, encoding="utf-8") as handle:
        fit = json.load(handle)
    weights = fit["fitted_weights"]
    if len(weights) != 13:
        sys.exit(f"expected 13 weights, got {len(weights)}")

    body = TEMPLATE.format(
        auc_shipped=fit["auc_shipped"], auc_refit=fit["auc_refit"],
        states=fit["states"], bias=round(fit["fitted_bias"], 6),
        temp=round(fit.get("temperature", 1.0), 4),
        **{f"w{i}": round(w, 6) for i, w in enumerate(weights)})

    with open(FROZEN, encoding="utf-8") as handle:
        source = handle.read()

    # Replace exactly the `_evaluate` definition, from its `def` line up to the
    # next top-level `def`/comment banner, and nothing else.
    match = re.search(r"^def _evaluate\(state, me\):\n(?:.*?\n)*?(?=^\n\n# ---)",
                      source, re.MULTILINE)
    if not match:
        sys.exit("could not locate the _evaluate block in the frozen shell")
    patched = source[:match.start()] + body + source[match.end():]

    # Prove the only thing that moved is `_evaluate`.  Counting diff hunks is the
    # wrong test: rewriting one function produces several hunks wherever an
    # identical line survives inside it.  What actually matters is that the
    # replaced span was the right one, that every other top-level definition is
    # still present and byte-identical, and that the result still compiles.
    replaced = source[match.start():match.end()]
    if not replaced.startswith("def _evaluate(state, me):"):
        sys.exit(f"matched the wrong span, it starts {replaced[:40]!r}")

    def top_level_defs(text: str) -> dict[str, str]:
        blocks: dict[str, str] = {}
        for m in re.finditer(r"^(def|class) (\w+)", text, re.MULTILINE):
            start = m.start()
            nxt = re.search(r"^(def|class) \w+", text[start + 1:], re.MULTILINE)
            end = start + 1 + nxt.start() if nxt else len(text)
            blocks[m.group(2)] = text[start:end]
        return blocks

    before, after = top_level_defs(source), top_level_defs(patched)
    if set(before) != set(after):
        sys.exit(f"definitions appeared or vanished: "
                 f"{set(before) ^ set(after)}")
    moved = [k for k in before if before[k] != after[k]]
    prefix_same = source[:match.start()] == patched[:match.start()]
    suffix_same = source[match.end():] == patched[match.start() + len(body):]

    compile(patched, args.out, "exec")
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(patched)

    changed_lines = source[:match.start()].count("\n") + 1
    print(f"wrote {args.out}")
    print(f"  replaced span starts at line {changed_lines}, "
          f"{len(before)} top-level definitions checked")
    print(f"  definitions whose text changed: {moved}  (expect ['_evaluate'])")
    print(f"  everything before the span identical: {prefix_same}")
    print(f"  everything after the span identical:  {suffix_same}")
    print("  patched file compiles: True")
    if moved != ["_evaluate"] or not prefix_same or not suffix_same:
        sys.exit("the patch escaped _evaluate; refusing to call this a clean build")
    print(f"  fitted AUC {fit['auc_refit']:.4f} vs shipped {fit['auc_shipped']:.4f} "
          f"on {fit['states']:,} positions")


if __name__ == "__main__":
    main()
