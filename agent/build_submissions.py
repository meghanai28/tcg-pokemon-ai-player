"""Package a deck-matched prior behind the byte-frozen champion shell."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHAMPION = os.path.join(HERE, "champion")
CHAMPION_ARCHIVE = os.path.join(
    ROOT, "harness", "anchors", "grpo_tech_grim_972_912_811.tar.gz")
CHAMPION_ARCHIVE_SHA256 = (
    "4fdb2ecf444d58161430fcaacb84795e5cd7f51ed2b756f225385493618e2f12"
)
FROZEN_FILES = {
    "main.py": "f657d408ebc657d7b227ca5f0cd5ce1405b0b0fced75d730f87f986fc9fda39c",
    "nn_features.py": "62d7dfd926ecc582df9a50f5e7e6d1eec02ae1cda1201eb4d2f272ec5ef100a9",
    "nn_features_rich.py": "fb77e29700e8fd9f4955bd658a8da2c24ad0d4c1007f4d363e2e6b8563c660ac",
    "nn_infer.py": "a801e4f0815125572b689f3ce27dc0f468f0365de1d1b6eda028f982bfd7693c",
    "cg/engine.py": "a951030b175a96255842d322ada0bd27077888451a93ea879dcd2c6676a5e8a2",
    "cg/game.py": "c4699ecbe617013349895992ae493d17486de72fec85c798cdfabc06d7260e41",
    "cg/sim.py": "a5aee75dfe3d70a9622a5e8369ff01b79b22d4b7d026ca44027143ce4672b048",
    "cg/libcg.so": "d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_shell() -> None:
    if sha256(CHAMPION_ARCHIVE) != CHAMPION_ARCHIVE_SHA256:
        raise RuntimeError("the protected champion archive changed")
    for relative, expected in FROZEN_FILES.items():
        path = os.path.join(CHAMPION, relative)
        if not os.path.isfile(path) or sha256(path) != expected:
            raise RuntimeError(f"frozen champion file changed: {relative}")


# The shell loads the net once per decision (~150 a game) and silently falls
# back to heuristic priors if a game's worth of calls would exceed
# NET_TIME_BUDGET_S. That fallback is invisible in the output, so a slow model
# does not fail loudly -- it just quietly stops being the agent we gated. Keep
# a wide margin: this dev box is faster than Kaggle's 2 vCPUs, so a model that
# only just fits here would not fit there.
NET_TIME_BUDGET_S = 90.0        # mirrors main.py:69
CALLS_PER_GAME = 150            # mirrors the latency guard in main.py
LATENCY_MARGIN = 0.25           # refuse anything projected past 25% of the cap
# The champion is the calibration stick: it is known to run at roughly 1.35
# s/game on an idle box, i.e. 1.5% of the cap. Expressing a candidate's cost as
# a multiple of the champion measured *in the same process under the same load*
# makes the check immune to CPU contention, which otherwise inflates absolute
# timings by an order of magnitude when a trainer is running. An absolute
# reading cannot be trusted here; a ratio can.
CHAMPION_PROJECTED_S = 1.35
MAX_CHAMPION_MULTIPLE = NET_TIME_BUDGET_S * LATENCY_MARGIN / CHAMPION_PROJECTED_S


def _time_once(net, batch) -> float:
    start = time.perf_counter()
    net.forward(*batch)
    return time.perf_counter() - start


def _per_call(net, batch) -> float:
    net.forward(*batch)                                       # warm up
    # Fastest of many: contention only ever inflates a timing, so the minimum
    # is the observation least polluted by whatever else was scheduled.
    return min(_time_once(net, batch) for _ in range(12))


def validate_model(path: str) -> None:
    """Reject checkpoints the deployed shell could not actually use.

    The architecture is intentionally not pinned to a single tuple: the shipped
    ``nn_infer.NumpyNet`` reads ``_meta`` and loops over ``n_layers``, so any
    shape it can hold is loadable. What genuinely constrains a candidate is
    inference latency, so that is what gets measured.
    """
    with np.load(path) as weights:
        if "_meta" not in weights:
            raise ValueError(f"{path} has no architecture metadata")
        meta = tuple(map(int, weights["_meta"]))
    if len(meta) != 4 or any(part <= 0 for part in meta):
        raise ValueError(f"{path} has malformed architecture {meta}")
    dim, _layers, heads, _d_ff = meta
    if dim % heads:
        raise ValueError(f"{path}: dim {dim} not divisible by heads {heads}")

    sys.path.insert(0, CHAMPION)
    try:
        import nn_features_rich as nf
        from nn_infer import NumpyNet
        batch = (np.zeros((1, nf.SEQ), dtype=np.int64),
                 np.zeros((1, nf.SEQ), dtype=np.int64),
                 np.zeros((1, nf.SEQ, nf.F), dtype=np.float32),
                 np.ones((1, nf.SEQ), dtype=np.float32),
                 np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64))
        candidate = _per_call(NumpyNet(path), batch)
        reference = _per_call(NumpyNet(os.path.join(CHAMPION, "model.npz")), batch)
    finally:
        if sys.path and sys.path[0] == CHAMPION:
            sys.path.pop(0)

    multiple = candidate / reference if reference > 0 else float("inf")
    projected = multiple * CHAMPION_PROJECTED_S
    print(f"  architecture {meta}: {multiple:.2f}x the champion's cost "
          f"-> ~{projected:.1f} s/game "
          f"({100 * projected / NET_TIME_BUDGET_S:.1f}% of the "
          f"{NET_TIME_BUDGET_S:g} s cap; measured {candidate * 1000:.1f} vs "
          f"{reference * 1000:.1f} ms/call under current load)")
    if multiple > MAX_CHAMPION_MULTIPLE:
        raise ValueError(
            f"{path} costs {multiple:.1f}x the champion, over the "
            f"{MAX_CHAMPION_MULTIPLE:.1f}x packaging limit "
            f"({LATENCY_MARGIN:.0%} of the shell's {NET_TIME_BUDGET_S:g} s cap). "
            "The shell would drop it and use heuristic priors instead.")


def read_deck(path: str) -> list[int]:
    with open(path, encoding="utf-8") as source:
        deck = [int(line) for line in source if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(deck)}")
    return deck


def build(name: str, model_path: str, deck_path: str, out_dir: str) -> str:
    validate_frozen_shell()
    validate_model(model_path)
    read_deck(deck_path)
    os.makedirs(out_dir, exist_ok=True)
    archive = os.path.join(out_dir, f"candidate_{name}.tar.gz")
    with tempfile.TemporaryDirectory(prefix=f"grpo_{name}_", dir=out_dir) as stage:
        for entry in os.listdir(CHAMPION):
            source = os.path.join(CHAMPION, entry)
            target = os.path.join(stage, entry)
            if os.path.isdir(source):
                shutil.copytree(source, target, ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc"))
            else:
                shutil.copy2(source, target)
        shutil.copy2(model_path, os.path.join(stage, "model.npz"))
        shutil.copy2(deck_path, os.path.join(stage, "deck.csv"))
        for relative, expected in FROZEN_FILES.items():
            if sha256(os.path.join(stage, relative)) != expected:
                raise RuntimeError(f"packaging changed frozen file {relative}")
        with tarfile.open(archive, "w:gz") as bundle:
            for entry in sorted(os.listdir(stage)):
                bundle.add(os.path.join(stage, entry), arcname=entry)
    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
    required = {"main.py", "deck.csv", "model.npz", "nn_features.py",
                "nn_features_rich.py", "nn_infer.py", "cg/libcg.so"}
    if not required.issubset(names):
        raise RuntimeError(f"invalid top-level layout: {sorted(required - names)}")
    print(f"built {archive}; sha256={sha256(archive)}")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "runs", "packages"))
    args = parser.parse_args()
    build(args.name, args.model, args.deck, args.out_dir)


if __name__ == "__main__":
    main()
