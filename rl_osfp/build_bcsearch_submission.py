"""Package an rl_osfp checkpoint behind the proven BC search shell.

Why this exists
---------------
Two search implementations have now been measured on the ladder against the
same policy stack, and the gap is not subtle:

    GRPO policy + the frozen BC shell   ref 55233305   810.8
    GRPO policy + rl_osfp/search.py     ref 55234807   405.0

Same model, same deck, same encoders, same day.  The reimplementation cost ~405
points, so it has been deleted and this builder ships the shell that actually
scored instead.

`foundation/search_shell_main.py` is a byte-frozen copy of the `main.py` that
appears, md5-identical, inside all three of our best archives (972.0/911.9/810.8
GRPO, 903.2 AWR-GRPO, 848.9 BC800).  It is never edited, and SHELL_MD5 below
fails the build if it drifts.  Everything the rl_osfp network needs in order to sit
behind it is handled by swapping the files *around* it:

    nn_features.py / nn_features_rich.py   frozen-shell encoders (MAX_OPT 24, SEQ 53)
    nn_infer_osfp.py                       the real numpy ActorCritic
    nn_infer.py                            adapter trimming forward() to 2 values

The shell auto-detects `nn_features_rich.py` and imports `NumpyNet` from
`nn_infer`, so no shell edit is required for any of it.

Follow every build with `verify_bcsearch_submission.py`.  The shell swallows a
broken net and runs on heuristic priors without raising, which produces a
perfectly well-formed archive that silently ignores the checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOUNDATION = os.path.join(ROOT, "foundation")
BC_TRAIN = os.path.join(ROOT, "bc_train")

# md5 of the main.py inside the 972.0 / 903.2 / 848.9 archives.  A build that
# does not reproduce this is not shipping the search that scored.
SHELL_MD5 = "e54bc6590288e659d696d00d432c6cc4"

# cg/ is copied by explicit name so __pycache__ can never ride along; a 3.12
# .pyc in the archive was one of the "Validation Episode failed" causes.
ENGINE_FILES = ("libcg.so", "engine.py", "game.py", "sim.py", "__init__.py")


def md5(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def load_deck(path: str) -> list[int]:
    deck = [int(line) for line in open(path, encoding="utf-8") if line.strip()]
    if len(deck) != 60:
        raise SystemExit(f"{path}: deck must be exactly 60 cards, got {len(deck)}")
    return deck


def deck_from_pool(pool_path: str, name: str) -> list[int]:
    with open(pool_path, encoding="utf-8") as handle:
        pool = json.load(handle)
    for group in ("field_decks", "learner_decks"):
        for deck in pool.get(group, []):
            if deck["name"] == name:
                return list(deck["cards"])
    raise SystemExit(f"{name} not in {pool_path}")


def validate_osfp_model(path: str) -> dict:
    """Reject BC/champion-format weights that the OSFP adapter cannot load."""
    try:
        with np.load(path) as weights:
            if "_meta" not in weights.files:
                raise ValueError("missing _meta")
            meta = [int(value) for value in weights["_meta"]]
            missing = {
                "option_head.weight", "count_head.weight", "value_fc1.weight"
            } - set(weights.files)
            tensor_count = len(weights.files)
    except Exception as error:
        raise SystemExit(f"{path}: unreadable checkpoint: {error}") from error
    if len(meta) < 6 or meta[5] != 2 or missing:
        raise SystemExit(
            f"{path}: expected a full rl_osfp v2 model_period/model_latest "
            f"checkpoint (meta={meta}, missing={sorted(missing)}). A "
            "champion_period export deliberately drops the count head and must "
            "not be passed to this adapter."
        )
    return {"meta": meta, "tensors": tensor_count}


def build(model: str | None, deck: list[int], out: str, deck_label: str) -> dict:
    shell = os.path.join(FOUNDATION, "search_shell_main.py")
    got = md5(shell)
    if got != SHELL_MD5:
        raise SystemExit(
            f"{shell} is not the proven shell (md5 {got}, expected {SHELL_MD5}). "
            "Refusing to build: the whole point of this path is that main.py is "
            "the file that scored 972.0."
        )

    model_report = validate_osfp_model(model) if model is not None else None
    staging = tempfile.mkdtemp(prefix="bcsearch_")
    try:
        shutil.copy2(shell, os.path.join(staging, "main.py"))
        # The frozen shell and every BC checkpoint it has successfully served use
        # MAX_OPT=24 / SEQ=53. `foundation/nn_features.py` is configurable for
        # pure-RL experiments and defaults to 64, so copying it here would make a
        # correctly trained checkpoint run under a different encoder on Kaggle.
        # Package the literal training ABI instead.
        base_encoder = os.path.join(BC_TRAIN, "nn_features.py")
        with open(base_encoder, encoding="utf-8") as source:
            encoder_text = source.read()
        if "MAX_OPT = 24" not in encoder_text:
            raise SystemExit(
                f"{base_encoder} is no longer the frozen MAX_OPT=24 encoder"
            )
        for name in ("nn_features.py", "nn_features_rich.py"):
            shutil.copy2(os.path.join(BC_TRAIN, name), os.path.join(staging, name))
        # the real implementation, plus the shim the shell will actually import
        shutil.copy2(os.path.join(FOUNDATION, "nn_infer.py"),
                     os.path.join(staging, "nn_infer_osfp.py"))
        shutil.copy2(os.path.join(FOUNDATION, "nn_infer_adapter.py"),
                     os.path.join(staging, "nn_infer.py"))
        if model is not None:
            shutil.copy2(model, os.path.join(staging, "model.npz"))

        with open(os.path.join(staging, "deck.csv"), "w", encoding="utf-8") as target:
            target.write("\n".join(str(card) for card in deck) + "\n")

        engine_dir = os.path.join(staging, "cg")
        os.makedirs(engine_dir)
        for name in ENGINE_FILES:
            source = os.path.join(FOUNDATION, "cg", name)
            if not os.path.exists(source):
                raise SystemExit(
                    f"missing engine file {source}. The licensed binaries are "
                    "gitignored; copy them from kaggle_environments/envs/cabt/cg/."
                )
            shutil.copy2(source, os.path.join(engine_dir, name))

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with tarfile.open(out, "w:gz") as archive:
            for entry in sorted(os.listdir(staging)):
                archive.add(os.path.join(staging, entry), arcname=entry)

        return {
            "archive": os.path.abspath(out),
            "model": os.path.abspath(model) if model else None,
            "model_md5": md5(model) if model else None,
            "model_format": model_report,
            "shell_md5": got,
            "deck": deck_label,
            "deck_md5": hashlib.md5(
                ("\n".join(str(c) for c in deck) + "\n").encode()).hexdigest(),
            "bytes": os.path.getsize(out),
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=None,
        help="full rl_osfp v2 model_period/model_latest .npz (not a "
             "champion_period export)",
    )
    parser.add_argument("--no-model", action="store_true",
                        help="ship the shell with no model.npz at all, so the search "
                             "runs on heuristic priors. This is the control that "
                             "isolates whether a checkpoint's priors help or hurt.")
    parser.add_argument("--deck-csv", default=os.path.join(FOUNDATION, "deck_tech_grim.csv"),
                        help="deck list to ship (default: the Tech-Grim list from the 972 archive)")
    parser.add_argument("--deck-name", default=None,
                        help="instead of --deck-csv, take this deck from --pool")
    parser.add_argument("--pool", default=os.path.join(ROOT, "data", "fresh", "deck_pool_elo1000.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not args.model and not args.no_model:
        raise SystemExit("pass --model, or --no-model for the heuristic-priors control")
    if args.model and args.no_model:
        raise SystemExit("--model and --no-model are mutually exclusive")

    if args.deck_name:
        deck, label = deck_from_pool(args.pool, args.deck_name), args.deck_name
    else:
        deck, label = load_deck(args.deck_csv), os.path.basename(args.deck_csv)

    report = build(None if args.no_model else args.model, deck, args.out, label)
    print(json.dumps(report, indent=2))
    print("\nnow run: .venv/bin/python -m rl_osfp.verify_bcsearch_submission "
          f"--archive {report['archive']}")


if __name__ == "__main__":
    main()
