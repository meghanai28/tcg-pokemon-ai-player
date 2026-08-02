"""Build and validate the two GRPO search-agent submission archives."""
from __future__ import annotations

import argparse
import os
import shutil
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGENT = os.path.join(ROOT, "track1_search", "agent")
RICH = os.path.join(ROOT, "track1_search", "train", "nn_features_rich.py")


def validate_model(path):
    with np.load(path) as model:
        if "_meta" not in model:
            raise ValueError(f"{path} has no _meta architecture")
        meta = tuple(map(int, model["_meta"]))
        if meta not in ((160, 5, 5, 320), (192, 6, 6, 384)):
            raise ValueError(f"unexpected architecture {meta} in {path}")


def build(name, model_path, deck_path, out_dir,
          archive_prefix="submission_grpo_"):
    validate_model(model_path)
    deck = [int(line) for line in open(deck_path, encoding="utf-8") if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{deck_path}: expected 60 cards, got {len(deck)}")
    stage = os.path.join(HERE, "build", name)
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)
    for file_name in ("main.py", "nn_features.py", "nn_infer.py"):
        shutil.copy2(os.path.join(AGENT, file_name), stage)
    shutil.copy2(RICH, os.path.join(stage, "nn_features_rich.py"))
    shutil.copy2(model_path, os.path.join(stage, "model.npz"))
    shutil.copy2(deck_path, os.path.join(stage, "deck.csv"))
    shutil.copytree(os.path.join(AGENT, "cg"), os.path.join(stage, "cg"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    required = ("main.py", "deck.csv", "model.npz", "nn_features.py",
                "nn_features_rich.py", "nn_infer.py", "cg/engine.py",
                "cg/libcg.so", "cg/sim.py")
    for relative in required:
        if not os.path.isfile(os.path.join(stage, relative)):
            raise FileNotFoundError(f"submission is missing {relative}")
    os.makedirs(out_dir, exist_ok=True)
    archive = os.path.join(out_dir, f"{archive_prefix}{name}.tar.gz")
    with tarfile.open(archive, "w:gz") as bundle:
        for entry in sorted(os.listdir(stage)):
            bundle.add(os.path.join(stage, entry), arcname=entry)
    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
    if not {"main.py", "deck.csv", "model.npz"}.issubset(names):
        raise RuntimeError(f"top-level archive layout is invalid: {archive}")
    print(f"built {archive} ({os.path.getsize(archive) / 2**20:.1f} MiB)")
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=ROOT)
    parser.add_argument("--grim-model", default=os.path.join(
        HERE, "model_grpo_recent_grimmsnarl_iter004.npz"))
    parser.add_argument("--garchomp-model", default=os.path.join(
        HERE, "model_grpo_recent_garchomp_iter008.npz"))
    args = parser.parse_args()
    for name, model_path in (
            ("grimmsnarl", args.grim_model),
            ("garchomp", args.garchomp_model)):
        build(name, model_path,
              os.path.join(HERE, "decks", f"{name}.csv"), args.out_dir)


if __name__ == "__main__":
    main()
