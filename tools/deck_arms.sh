#!/usr/bin/env bash
# Package one checkpoint behind the frozen shell once per candidate deck, prove
# each archive is really firing its network, then gauntlet them all against the
# champion.
#
# Why a gauntlet and not a round robin: every arm shares the same model, so the
# only question is how each deck fares against a fixed external reference.  That
# is O(n) instead of O(n^2), which is the difference between one hour and five.
#
# Why the champion is the reference and not each other: CLAUDE.md's methodology
# rule.  Arms compared only to each other can all be far below a baseline nobody
# tested against, which is exactly how this project lost a week to a search that
# scored 405.
#
# The budget is deliberately the cheap one.  Deck screening at 0.25 s/move was
# validated against the 1.1 s ordering on 2026-08-06 (same three pairwise
# directions, 4.6x more games per hour), but it EXAGGERATES gaps, so the winner
# still has to be confirmed at 1.1 s before anything ships.
#
# Usage: tools/deck_arms.sh <model.npz> <tag> [games-per-pair] [workers]
set -euo pipefail

MODEL="${1:?usage: deck_arms.sh <model.npz> <tag> [games] [workers]}"
TAG="${2:?missing tag}"
GAMES="${3:-60}"
WORKERS="${4:-7}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
BASE="$ROOT/harness/anchors/grpo_tech_grim_972_912_811.tar.gz"
cd "$ROOT"

# tech_grim is not a candidate, it is the control: same new model, the deck we
# already ship.  Without it a win by the lucario arm cannot be attributed to the
# deck rather than to the model, which is the exact confound that made every
# previous deck result uninterpretable.
ARCHIVES=("$BASE")
for deck in lucario dunsparce tech_grim; do
    out="$ROOT/artifacts/sub_${TAG}_${deck}.tar.gz"
    echo "=== packaging $deck ==="
    # swap_model asserts that exactly the intended files changed.  When the
    # candidate deck IS the base archive's deck -- which is the case for the
    # tech_grim control -- copying it changes nothing, `changed` comes back as
    # just model.npz, and the check fails on a build that is in fact correct.
    # Passing --deck only when the bytes actually differ keeps the assertion
    # meaningful instead of weakening it.
    DECK_ARG=()
    if ! cmp -s "$ROOT/data/decks_target/${deck}.csv" "$ROOT/foundation/deck_tech_grim.csv"; then
        DECK_ARG=(--deck "$ROOT/data/decks_target/${deck}.csv")
    else
        echo "    (deck is byte-identical to the base archive's; model-only swap)"
    fi
    "$PY" tools/swap_model.py --base "$BASE" --model "$MODEL" \
        "${DECK_ARG[@]}" --out "$out"
    # The shell swallows a broken net inside _net_scores and searches on
    # heuristic priors instead, so a dead checkpoint still produces a
    # well-formed archive that plays legal games at normal latency.  Only this
    # check distinguishes the two.
    "$PY" -m rl_osfp.verify_bcsearch_submission --archive "$out" --games 2 \
        --out "$ROOT/artifacts/sub_${TAG}_${deck}_verify.json"
    ARCHIVES+=("$out")
done

echo "=== gauntlet: ${#ARCHIVES[@]} archives, ${GAMES} games/pair, 0.25 s/move ==="
"$PY" tools/ladder_harness.py \
    --archives "${ARCHIVES[@]}" \
    --vs grpo_tech_grim_972_912_811 \
    --games-per-pair "$GAMES" --budget 0.25 --workers "$WORKERS" \
    --out "$ROOT/harness/gate_${TAG}_decks.json"
