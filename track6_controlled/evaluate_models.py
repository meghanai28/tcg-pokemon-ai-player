"""Evaluate deck-specific checkpoints on an exact-deck temporal holdout."""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(ROOT, "track1_search", "train")
sys.path.insert(0, TRAIN)
sys.path.insert(0, os.path.join(ROOT, "track1_search", "agent"))
sys.path.insert(0, os.path.join(ROOT, "track5_grpo"))

from train_bc import eval_val, load_data  # noqa: E402
from train_grpo import load_model  # noqa: E402


def tensors(data, limit=0):
    n = len(data["pi"])
    idx = np.arange(n)
    if limit and n > limit:
        idx = np.sort(np.random.default_rng(260801).choice(
            n, size=limit, replace=False))
    return tuple(torch.tensor(value) for value in (
        data["kind"][idx].astype(np.int64),
        data["card"][idx].astype(np.int64),
        data["scal"][idx], data["mask"][idx],
        data["ctx"][idx].astype(np.int64),
        data["stype"][idx].astype(np.int64),
        data["pi"][idx], data["z"][idx],
    ))


def expand_models(values):
    paths = []
    for value in values:
        matches = sorted(glob.glob(value))
        paths.extend(matches or [value])
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.75)

    holdout = tensors(load_data(args.data), args.limit)
    print(f"holdout decisions={len(holdout[0])}; device={device}")
    print("model\tCE\ttop1\tvalue_MAE")
    for path in expand_models(args.models):
        model = load_model(path, device)
        ce, _mse, top1, mae = eval_val(model, holdout, device, args.batch)
        print(f"{os.path.basename(path)}\t{ce:.5f}\t{top1*100:.2f}%\t{mae:.5f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
