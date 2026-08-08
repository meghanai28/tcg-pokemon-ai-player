"""Screen many candidate decks under one fixed pilot, without paying O(n^2).

A pairwise deck round robin does not scale.  Twenty decks is 190 pairs, and at
the 200 games a pair this project uses to separate anything, that is 38,000
games.  At the shipping budget that is weeks of wall clock.  Three changes make
it affordable, and each one is here because something measured says so:

  1. **Validate before playing.**  `battle_start` rejects a malformed deck and
     `cabt` scores a rejected deck as an immediate loss for that seat, so a bad
     candidate does not fail loudly, it silently loses every game.  Validation
     is free and has to happen first.

  2. **Anchor instead of round-robin.**  Every candidate plays the same fixed
     opponent, so scores stay comparable across candidates without candidates
     ever playing each other.  O(n) rather than O(n^2).  This is a screen, and
     `CLAUDE.md` is emphatic that the screen does not decide: run the survivors
     through a symmetric round robin before shipping anything.

  3. **Stop early on decided candidates.**  A fixed games-per-candidate budget
     spends as much on a deck that is 20 points worse as on one that is 2 points
     worse.  The sequential probability ratio test reports which candidates are
     already settled at the current sample, so the next stage only pays for the
     ones still in question.

The pilot is held fixed and only `deck.csv` changes, so any deviation from 50%
is a deck effect.  Two warnings that are easy to forget and expensive to
relearn:

  * A deck swap under a deck-specialised prior is **not** a clean deck test.
    The 972 recipe trained its prior on the exact deck it shipped with, so
    swapping `deck.csv` underneath it breaks that pairing and penalises every
    deck that is not the one the prior learned.  Screen with a deck-agnostic
    pilot (a no-net archive, which uses heuristic priors) when the question is
    "which deck", and only pair a prior to a deck after the deck is chosen.
  * The absolute win rate here does not transfer to the ladder.  The opponent
    is one fixed archive, not the field.  What transfers is the ordering.

Usage:
    # validate only, costs nothing
    py tools/deck_cycle.py --pilot harness/anchors/grpo_tech_grim_972_912_811.tar.gz \
        --decks data/decks_external --validate-only

    # cheap anchored screen
    py tools/deck_cycle.py --pilot artifacts/submission_shell_nonet_techgrim.tar.gz \
        --decks data/decks_external --opponent harness/anchors/grpo_tech_grim_972_912_811.tar.gz \
        --games 16 --budget 0.25 --workers 5
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# Deck legality, against the real engine
# --------------------------------------------------------------------------
def read_deck(path: str) -> list[int]:
    with open(path, encoding="utf-8") as handle:
        return [int(line) for line in handle if line.strip()]


def validate(cards: list[int], reference: list[int]) -> str | None:
    """None if the engine accepts the deck, else why it does not.

    `cabt` checks the length itself and marks the seat INVALID, which is an
    instant loss with no retry, so length is checked before the engine is even
    asked.  Everything past that (illegal card ids, banned counts, no basic
    Pokemon) only shows up as `errorPlayer` from `battle_start`.
    """
    if len(cards) != 60:
        return f"{len(cards)} cards, the runner requires exactly 60"
    from foundation.cg.game import battle_finish, battle_start
    from foundation.cg.sim import Battle

    try:
        _, start = battle_start(cards, list(reference))
    except Exception as exc:  # engine refused outright
        return f"battle_start raised {exc!r}"
    try:
        if getattr(start, "errorPlayer", -1) == 0:
            return "engine rejected the decklist (errorPlayer=0)"
        if Battle.battle_ptr in (None, 0):
            return "engine returned no battle pointer"
    finally:
        try:
            battle_finish()
        except Exception:
            pass
        Battle.battle_ptr = None
    return None


# --------------------------------------------------------------------------
# One archive per candidate deck: same pilot, different deck.csv
# --------------------------------------------------------------------------
def build_variant(pilot: str, deck_path: str, label: str, out_dir: str) -> str:
    """Copy the pilot archive and swap only `deck.csv`."""
    staging = tempfile.mkdtemp(prefix="deckcycle-")
    try:
        with tarfile.open(pilot, "r:gz") as handle:
            handle.extractall(staging, filter="data")
        shutil.copy(deck_path, os.path.join(staging, "deck.csv"))
        target = os.path.join(out_dir, f"deck_{label}.tar.gz")
        with tarfile.open(target, "w:gz") as handle:
            for name in sorted(os.listdir(staging)):
                handle.add(os.path.join(staging, name), arcname=name)
        return target
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------
# Sequential test: is this candidate already settled?
# --------------------------------------------------------------------------
def sprt(wins: int, losses: int, p1: float = 0.55, alpha: float = 0.05,
         beta: float = 0.05) -> tuple[float, str]:
    """Log-likelihood ratio for "better than the anchor" against "not better".

    H0: the candidate wins half its games.  H1: it wins `p1`.  The bounds are
    the standard Wald thresholds, so `alpha` is the chance of promoting a deck
    that is not actually better and `beta` the chance of dropping one that is.
    Returned verdict is one of accept / reject / continue.
    """
    if wins + losses == 0:
        return 0.0, "continue"
    llr = (wins * math.log(p1 / 0.5)) + (losses * math.log((1 - p1) / 0.5))
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    if llr >= upper:
        return llr, "accept"
    if llr <= lower:
        return llr, "reject"
    return llr, "continue"


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - margin, centre + margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", required=True,
                        help="archive supplying main.py, the model and the encoders")
    parser.add_argument("--decks", required=True,
                        help="directory of candidate deck .csv files")
    parser.add_argument("--opponent", default=None,
                        help="fixed opponent archive. Required unless --validate-only")
    parser.add_argument("--games", type=int, default=16,
                        help="games per candidate against the anchor")
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--keep", default=os.path.join(ROOT, "artifacts", "deck_variants"))
    parser.add_argument("--out", default=os.path.join(ROOT, "harness", "deck_cycle.json"))
    args = parser.parse_args()

    if args.workers > 5:
        sys.exit("more than 5 workers is over this machine's cap, see tools/resource_guard.py")

    sys.path.insert(0, ROOT)
    paths = sorted(glob.glob(os.path.join(args.decks, "*.csv")))
    if not paths:
        sys.exit(f"no .csv decks in {args.decks}")

    reference = read_deck(os.path.join(ROOT, "foundation", "deck_tech_grim.csv"))
    print(f"{'deck':34s} {'cards':>5}  legality")
    good = []
    for path in paths:
        label = os.path.basename(path)[:-4]
        cards = read_deck(path)
        why = validate(cards, reference)
        print(f"{label:34s} {len(cards):>5}  {'OK' if why is None else 'REJECTED: ' + why}")
        if why is None:
            good.append((label, path))

    if args.validate_only:
        print(f"\n{len(good)}/{len(paths)} decks are legal")
        return
    if not args.opponent:
        sys.exit("--opponent is required unless --validate-only")

    os.makedirs(args.keep, exist_ok=True)
    variants = [build_variant(args.pilot, path, label, args.keep) for label, path in good]
    print(f"\nbuilt {len(variants)} variants in {args.keep}")

    anchor_name = os.path.basename(args.opponent).replace(".tar.gz", "")
    cmd = [os.path.join(ROOT, ".venv", "bin", "python"),
           os.path.join(ROOT, "tools", "ladder_harness.py"),
           "--archives", *variants, args.opponent,
           "--vs", anchor_name,
           "--games-per-pair", str(args.games),
           "--budget", str(args.budget),
           "--workers", str(args.workers),
           "--max-steps", "3000",
           "--out", args.out]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

    report = json.load(open(args.out, encoding="utf-8"))
    rows = []
    for key, rec in report["pairs"].items():
        left, right = key.split("|")
        cand = right if left == anchor_name else left
        # `pairs` is keyed anchor-first or candidate-first depending on sort
        # order, so read the record from the candidate's point of view.
        wins = rec["losses"] if left == anchor_name else rec["wins"]
        losses = rec["wins"] if left == anchor_name else rec["losses"]
        llr, verdict = sprt(wins, losses)
        lo, hi = wilson(wins, wins + losses)
        rows.append((wins / max(wins + losses, 1), cand, wins, losses, lo, hi, llr, verdict))

    rows.sort(reverse=True)
    print(f"\n=== each deck vs {anchor_name}, same pilot, {args.budget}s/move ===")
    print(f"{'deck':34s} {'W-L':>9} {'win%':>7} {'Wilson 95%':>16} {'LLR':>7}  verdict")
    for rate, cand, wins, losses, lo, hi, llr, verdict in rows:
        print(f"{cand[:34]:34s} {wins:>4}-{losses:<4} {100*rate:>6.1f}% "
              f"  [{lo:.3f}, {hi:.3f}] {llr:>7.2f}  {verdict}")
    print("\nThis is a SCREEN. Run the survivors through a symmetric round robin "
          "before shipping, and re-pair the prior to whichever deck wins.")


if __name__ == "__main__":
    main()
