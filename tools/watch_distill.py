"""Emit one compact line per finished distil epoch, then exit when the run ends.

The trainer logs each epoch as a single very large dict, so this pulls out only
the fields that decide the submission and prints them as one short line.
"""
import ast
import os
import re
import sys
import time

LOG = "agent/runs/bc_lucario_distill_now/train_prior.log"
OUTER = "agent/runs/distill_now.log"
PKG = "agent/runs/packages/candidate_lucario_distill.tar.gz"
BASELINE_CE = 0.6745  # lucario_clean, the warm-start point

FAIL = re.compile(r"Traceback|CUDA out of memory|Killed|MemoryError|RuntimeError")


def fmt(d):
    def g(k):
        v = d.get(k)
        return "  n/a" if v is None else f"{v:.4f}"

    delta = d.get("target_ce")
    delta = "" if delta is None else f"  vs base {delta - BASELINE_CE:+.4f}"
    return (
        f"epoch {d.get('epoch')}: tgt_ce {g('target_ce')} "
        f"top1 {g('target_top1')}  teacher_regret {g('teacher_regret')} "
        f"teacher_top1 {g('teacher_top1')}{delta}"
    )


def main():
    seen = set()
    while True:
        for path in (LOG, OUTER):
            if not os.path.exists(path):
                continue
            try:
                text = open(path, errors="replace").read()
            except OSError:
                continue
            for m in re.finditer(r"^epoch (\d+): (\{.*\})$", text, re.M):
                n = int(m.group(1))
                if n in seen:
                    continue
                seen.add(n)
                try:
                    print(fmt(ast.literal_eval(m.group(2))), flush=True)
                except (ValueError, SyntaxError):
                    print(f"epoch {n}: (unparsed)", flush=True)
            if FAIL.search(text):
                print(f"DISTIL FAILED - see {path}", flush=True)
                return 1
        if os.path.exists(PKG):
            print(
                f"DISTIL DONE - packaged {PKG} "
                f"({os.path.getsize(PKG) / 1e6:.1f} MB), not submitted",
                flush=True,
            )
            return 0
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
