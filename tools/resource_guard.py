"""Refuse to launch a job that would take the machine down.

This exists because two WSL VM terminations were caused by launching work whose
memory requirement was never computed. A "free RAM looks fine" check is not a
safety check: free RAM tells you what is available now, not what the job will
ask for. Both crashes passed that check and died anyway.

**Size the job by PEAK RSS, not by the size of the final tensors.** That
distinction is what this file originally got wrong, and it made the guard pass a
job that then had to be killed. `train_bc.py` settles at about 7,396 bytes per
decision, but getting there costs roughly **2.6x** that: `np.load` reads a shard,
concatenation copies it, and `torch.tensor` copies it again before the numpy side
is released.

Measured directly on 2026-08-06: 1,040,000 decisions, whose settled tensors are
7.2 GiB, peaked at **18.4 GiB RSS** and left only 4.4 GiB free. That is
**18,990 bytes per decision at the peak**, and it is the number to plan against.

Re-reading the two crashes with peak instead of settled size explains both, which
the old numbers never did:

    1,150,331 decisions ->  7.9 GiB settled -> ~20.9 GiB peak   barely survived
    1,920,408 decisions -> 13.2 GiB settled -> ~34.8 GiB peak   killed the VM

Hard caps for this machine (23 GiB RAM, 14 threads, one RTX 5080):

    PEAK_RAM_GIB   15.0   peak RSS; leaves 8 GiB for CUDA, torch and the OS
    MAX_WORKERS     5     harness/rollout processes, out of 14 threads
    MIN_FREE_GIB    6.0   refuse to start if less than this is free

Usage:
    py tools/resource_guard.py --decisions 1150331
    py tools/resource_guard.py --data data/bc_general_train --workers 5
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

PEAK_BYTES_PER_DECISION = 18990    # measured peak RSS, see module docstring
SETTLED_BYTES_PER_DECISION = 7396  # what it drops back to once loading is done
PEAK_RAM_GIB = 15.0
MAX_WORKERS = 5
MIN_FREE_GIB = 6.0
GIB = 2 ** 30


def free_gib() -> float:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / GIB
    return 0.0


def count_decisions(dirs: list[str]) -> int:
    import numpy as np
    total = 0
    for directory in dirs:
        for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
            with np.load(path) as shard:
                total += shard["card"].shape[0]
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=int, default=None)
    parser.add_argument("--data", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--budget-gib", type=float, default=PEAK_RAM_GIB)
    args = parser.parse_args()

    problems: list[str] = []
    available = free_gib()
    print(f"free RAM: {available:.1f} GiB   cores: {os.cpu_count()}")

    if available < MIN_FREE_GIB:
        problems.append(f"only {available:.1f} GiB free, need {MIN_FREE_GIB} GiB to start")

    decisions = args.decisions
    if decisions is None and args.data:
        decisions = count_decisions(args.data)
    if decisions:
        peak = decisions * PEAK_BYTES_PER_DECISION / GIB
        settled = decisions * SETTLED_BYTES_PER_DECISION / GIB
        print(f"dataset: {decisions:,} decisions -> {peak:.1f} GiB PEAK RSS "
              f"while loading, settling to {settled:.1f} GiB")
        shards = len(glob.glob(os.path.join(args.data[0], "*.npz"))) if args.data else 8
        shards = max(shards, 1)
        if peak > args.budget_gib:
            cap = int(args.budget_gib * GIB / PEAK_BYTES_PER_DECISION)
            problems.append(
                f"peak is {peak:.1f} GiB, over the {args.budget_gib} GiB budget. "
                f"Subsample to <= {cap:,} decisions "
                f"(--max-per-shard {cap // shards:,} across {shards} shards)."
            )
        if peak > available - 4.0:
            problems.append(
                f"peak is {peak:.1f} GiB with only {available:.1f} GiB available; "
                "leave at least 4 GiB for CUDA, torch and the OS"
            )

    if args.workers is not None:
        print(f"requested workers: {args.workers} (cap {MAX_WORKERS})")
        if args.workers > MAX_WORKERS:
            problems.append(f"{args.workers} workers exceeds the {MAX_WORKERS} cap")

    if problems:
        print("\nREFUSING TO LAUNCH:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("\nOK to launch")


if __name__ == "__main__":
    main()
