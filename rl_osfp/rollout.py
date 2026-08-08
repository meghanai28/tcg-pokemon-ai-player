"""Parallel on-policy rollouts across worker processes.

Rollouts, not gradients, are this project's wall-clock bottleneck: a PPO step
costs ~48 ms on the GPU while a single game costs ~400 ms on one CPU core, so a
serial trainer leaves the card idle ~85% of the time and 13 of 14 cores unused.

Two facts force the design:

*   The native engine keeps its battle pointer in module-global state, so a
    process can only drive one game at a time.  Parallelism has to be
    process-level, and every worker builds its own ``Arena``.
*   The parent initializes CUDA for the update, and CUDA does not survive
    ``fork``.  The pool must use ``spawn``.

Workers therefore receive models as *file paths* rather than pickled modules,
and cache them by ``(path, mtime)`` so a long run reloads weights only when the
period actually advances.  The pool is created once and reused, because a spawn
start plus an ``Arena`` build costs seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
import os
from typing import Any, Iterator

import torch

from .arena import Arena
from .network import load_npz
from .policy import Decision


@dataclass
class GameTask:
    """One complete game, fully specified so a worker needs no shared state."""
    left_model: str
    right_model: str
    left_deck: list[int]
    right_deck: list[int]
    record_seats: tuple[int, ...] = ()
    sample: bool = True
    temperature: float = 1.0
    max_steps: int = 2400
    max_recorded: int = 3000
    seed: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameOutcome:
    meta: dict[str, Any]
    winner: int | None
    error: str | None
    engine_steps: int
    invalid_actions: int
    decisions: dict[int, list[Decision]]
    step_capped: bool = False


_WORKER: dict[str, Any] = {}


def _init_worker(threads: int) -> None:
    torch.set_num_threads(max(1, threads))
    _WORKER["arena"] = Arena()
    _WORKER["models"] = {}


def _model(path: str):
    """Cache by path and mtime so a rewritten checkpoint is picked up."""
    key = (path, os.stat(path).st_mtime_ns)
    cache = _WORKER["models"]
    if key not in cache:
        cache.clear()  # only ever a couple of live checkpoints per period
        cache[key] = load_npz(path).eval()
    return cache[key]


def _run_task(task: GameTask) -> GameOutcome:
    torch.manual_seed(task.seed)
    arena: Arena = _WORKER["arena"]
    result = arena.play(
        (_model(task.left_model), _model(task.right_model)),
        (task.left_deck, task.right_deck),
        record_seats=frozenset(task.record_seats),
        sample=task.sample,
        temperature=task.temperature,
        max_steps=task.max_steps,
        max_recorded_decisions=task.max_recorded,
    )
    return GameOutcome(
        meta=task.meta,
        winner=result.winner,
        error=result.error,
        engine_steps=result.engine_steps,
        invalid_actions=result.invalid_actions,
        # Ship back only what was asked for; decisions dominate IPC volume.
        decisions={seat: result.decisions[seat] for seat in task.record_seats},
        step_capped=result.step_capped,
    )


class RolloutPool:
    """A reusable spawn pool of engine workers.

    Used as a context manager so the workers are always torn down, including on
    a training crash - orphaned processes would keep whole cores busy.
    """

    def __init__(self, workers: int, threads_per_worker: int = 2) -> None:
        self.workers = max(1, workers)
        self.threads_per_worker = max(1, threads_per_worker)
        self._pool: mp.pool.Pool | None = None

    def __enter__(self) -> "RolloutPool":
        context = mp.get_context("spawn")
        self._pool = context.Pool(
            processes=self.workers,
            initializer=_init_worker,
            initargs=(self.threads_per_worker,),
        )
        return self

    def __exit__(self, *exc_info) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def run(self, tasks: list[GameTask], chunksize: int = 1) -> Iterator[GameOutcome]:
        """Yield outcomes as they land, so callers can stop early or log live."""
        if self._pool is None:
            raise RuntimeError("RolloutPool must be used as a context manager")
        if not tasks:
            return
        yield from self._pool.imap_unordered(_run_task, tasks, chunksize=chunksize)
def serial_run(tasks: list[GameTask], threads: int = 8) -> Iterator[GameOutcome]:
    """In-process equivalent of ``RolloutPool.run`` for --workers 1 and tests."""
    if "arena" not in _WORKER:
        _init_worker(threads)
    for task in tasks:
        yield _run_task(task)
