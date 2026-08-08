"""Build a candidate archive by replacing ONLY `model.npz` in a proven one.

Every gate this project trusts is a single-variable comparison, and the way to
get one is to start from an archive that already has a ladder result and change
exactly one file. `CLAUDE.md` records that building from a reused staging
directory silently reintroduced a stale `main.py` once, and that the tarball
diff is what caught it. So the diff is not optional here, it is the point:
this refuses to write an archive unless `model.npz` is the only member whose
bytes moved.

The tensor check matters just as much. The frozen shell loads the network
inside a `try/except` and falls back to heuristic priors on any failure, so a
checkpoint with the wrong tensor names produces a well-formed archive that
plays legal games at normal latency with no network at all. Measured, that is
worth about 55 points of win rate (80.0% with the net, 25.0% without), and
nothing about the archive looks wrong.

Usage:
    py tools/swap_model.py --base harness/anchors/grpo_tech_grim_972_912_811.tar.gz \
        --model rl_osfp/run_bc1/champion_period_050.npz \
        --out artifacts/submission_bcrl_p050.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def members(path: str) -> dict[str, str]:
    """member name -> md5 of its bytes."""
    out: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as handle:
        for info in handle.getmembers():
            if not info.isfile():
                continue
            data = handle.extractfile(info)
            out[info.name] = hashlib.md5(data.read()).hexdigest() if data else ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="archive with a known result")
    parser.add_argument("--model", required=True, help="replacement model.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-shape-change", action="store_true",
                        help="permit a different architecture. The numpy net reads its\n"
                             "dimensions from _meta, and bc800_tech_grim_849 shipped a\n"
                             "192d/6L net behind this same shell, so a wider model is\n"
                             "legitimate. Verify the prior rate afterwards.")
    parser.add_argument("--deck", default=None,
                        help="optional replacement deck.csv. Changing the deck AND "
                             "the model at once stops the gate being single-variable, "
                             "so this warns.")
    args = parser.parse_args()

    for path in (args.base, args.model):
        if not os.path.exists(path):
            sys.exit(f"no such file: {path}")

    staging = tempfile.mkdtemp(prefix="swapmodel-")
    try:
        with tarfile.open(args.base, "r:gz") as handle:
            handle.extractall(staging, filter="data")

        # The shell needs the archive's own tensor names. A mismatch does not
        # raise, it silently disables the network, so compare before writing.
        existing = np.load(os.path.join(staging, "model.npz"))
        replacement = np.load(args.model)
        old_names, new_names = set(existing.files), set(replacement.files)
        if old_names != new_names:
            sys.exit(f"tensor names differ from {args.base}:\n"
                     f"  missing: {sorted(old_names - new_names)}\n"
                     f"  extra:   {sorted(new_names - old_names)}\n"
                     "the shell would fall back to heuristic priors without erroring")
        shape_changes = [name for name in sorted(old_names)
                         if existing[name].shape != replacement[name].shape]
        if shape_changes and not args.allow_shape_change:
            sys.exit(f"{len(shape_changes)} tensors change shape, e.g. {shape_changes[0]}: "
                     f"base {existing[shape_changes[0]].shape} vs replacement "
                     f"{replacement[shape_changes[0]].shape}.\n"
                     "That means a different architecture. The packaged numpy net reads "
                     "its dimensions from _meta so this can be legitimate, but it is also "
                     "exactly what a wrong-checkpoint mistake looks like. Pass "
                     "--allow-shape-change if you meant it.")
        if shape_changes:
            print(f"NOTE: {len(shape_changes)} tensors change shape; this is a different "
                  f"architecture. base _meta={existing.get('_meta')} "
                  f"replacement _meta={replacement.get('_meta')}")
            print("      The shell benchmarks the net on load and falls back to heuristic "
                  "priors if projected cost exceeds 90 s/game, silently. Run "
                  "verify_bcsearch_submission.py and confirm a 100% prior rate.")
        moved = sum(1 for name in old_names
                    if not np.array_equal(existing[name], replacement[name]))
        print(f"tensor names and shapes match; {moved}/{len(old_names)} tensors differ in value")
        if moved == 0:
            print("WARNING: the replacement is identical to the base model. "
                  "This archive would be a duplicate submission.")

        shutil.copy(args.model, os.path.join(staging, "model.npz"))
        if args.deck:
            shutil.copy(args.deck, os.path.join(staging, "deck.csv"))
            print("WARNING: deck.csv replaced as well, so a gate against the base "
                  "archive no longer isolates the model.")

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with tarfile.open(args.out, "w:gz") as handle:
            for name in sorted(os.listdir(staging)):
                handle.add(os.path.join(staging, name), arcname=name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    before, after = members(args.base), members(args.out)
    if set(before) != set(after):
        sys.exit(f"member list changed: {set(before) ^ set(after)}")
    changed = sorted(name for name in before if before[name] != after[name])
    expected = ["model.npz"] + (["deck.csv"] if args.deck else [])
    print(f"members: {len(after)}   changed: {changed}   expected: {sorted(expected)}")
    if changed != sorted(expected):
        sys.exit("more than the intended files moved; refusing to call this clean")
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes)")


if __name__ == "__main__":
    main()
