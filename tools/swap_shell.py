"""Replace ONLY `main.py` inside a packaged archive, for single-variable gates.

`swap_model.py` swaps the network and the deck.  This swaps the search shell,
which is the third and last thing an archive contains, so that a search change
can be gated against a base archive that is byte-identical in every other
respect.  That isolation is the reason the 2026-08-05 opponent-model result
("30-30, as clean an isolation as this project has ever run") was worth
anything, and it is worth preserving.

Refuses if anything other than `main.py` moves, and refuses a shell that does
not compile, because the runner `exec`s the source as a string and a
SyntaxError there is scored as a lost episode rather than an error -- which is
exactly how the published LB-1084.5 notebook shipped a broken agent.

Usage:
    py tools/swap_shell.py --base artifacts/x.tar.gz \\
        --shell foundation/search_shell_floor0p1.py --out artifacts/y.tar.gz
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile


def members(path):
    with tarfile.open(path, "r:gz") as handle:
        return {m.name: (m.size, handle.extractfile(m).read() if m.isfile()
                         else None)
                for m in handle.getmembers()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--shell", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    for path in (args.base, args.shell):
        if not os.path.exists(path):
            sys.exit(f"no such file: {path}")

    with open(args.shell, encoding="utf-8") as handle:
        source = handle.read()
    try:
        compile(source, "main.py", "exec")
    except SyntaxError as exc:
        sys.exit(f"{args.shell} does not compile: {exc}")
    # The runner takes the LAST callable defined in the module; if that is not
    # `agent`, the archive silently plays something else.
    if "def agent(obs)" not in source:
        sys.exit(f"{args.shell} defines no `agent(obs)`")

    staging = tempfile.mkdtemp(prefix="swapshell-")
    try:
        with tarfile.open(args.base, "r:gz") as handle:
            handle.extractall(staging, filter="data")
        shutil.copy(args.shell, os.path.join(staging, "main.py"))
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with tarfile.open(args.out, "w:gz") as handle:
            for name in sorted(os.listdir(staging)):
                handle.add(os.path.join(staging, name), arcname=name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    before, after = members(args.base), members(args.out)
    if set(before) != set(after):
        sys.exit(f"member list changed: {set(before) ^ set(after)}")
    changed = sorted(n for n in before if before[n] != after[n])
    if changed != ["main.py"]:
        sys.exit(f"expected only main.py to change, got {changed}")
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes)")
    print(f"  members: {len(after)}   changed: {changed}")


if __name__ == "__main__":
    main()
