"""Choose and preserve the strongest eligible archive in a package gate."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile


def atomic_copy(source: str, destination: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".", suffix=".partial",
        dir=os.path.dirname(os.path.abspath(destination)))
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_selection(out: str, selection: dict) -> None:
    path = out + ".selection.json"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".partial", "w", encoding="utf-8") as target:
        json.dump(selection, target, indent=2)
    os.replace(path + ".partial", path)


def head_to_head(report: dict, name: str, anchor: str) -> dict | None:
    """Wins/losses for `name` against `anchor`, from either pair orientation."""
    confidence = report.get("pair_confidence", {})
    pairs = report.get("pairs", {})
    for key in (f"{name}|{anchor}", f"{anchor}|{name}"):
        if key not in pairs:
            continue
        record = pairs[key]
        left_is_name = key.startswith(name + "|")
        wins = record["wins"] if left_is_name else record["losses"]
        losses = record["losses"] if left_is_name else record["wins"]
        decided = wins + losses
        if not decided:
            return None
        low, high = confidence.get(key, {}).get(
            "left_wilson_95", [None, None])
        if low is None:
            return None
        # The stored interval is always for the left-hand agent, so mirror it
        # when the anchor is on the left.
        if not left_is_name:
            low, high = 1.0 - high, 1.0 - low
        return {"wins": wins, "losses": losses, "decided": decided,
                "win_rate": wins / decided,
                "wilson_low": low, "wilson_high": high}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--archives", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--must-beat", default=None,
        help="archive name (no .tar.gz) already on the ladder. A candidate "
             "wins only if it beats this anchor head-to-head with a 95%% "
             "Wilson lower bound above --must-beat-lower. Without this, the "
             "best of several regressions still gets crowned and shipped.")
    parser.add_argument("--must-beat-lower", type=float, default=0.50)
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as source:
        report = json.load(source)
    by_name = {
        os.path.basename(path).removesuffix(".tar.gz"): os.path.abspath(path)
        for path in args.archives
    }
    missing = sorted(set(by_name) - set(report.get("ratings", {})))
    if missing:
        raise SystemExit(f"gate report misses candidates: {missing}")
    invalid = report.get("invalid_action_games", {})
    eligible = [name for name in by_name if not invalid.get(name, 0)]
    if not eligible:
        raise SystemExit("every candidate produced an invalid action")

    # Beating the other candidates is not evidence of being worth shipping: the
    # best of three regressions is still a regression. When --must-beat names an
    # anchor already on the ladder, a candidate qualifies only if its
    # head-to-head Wilson lower bound clears --must-beat-lower, which is the
    # >.50 rule the handoff notes already require before a slot is spent.
    anchor_stats = {name: head_to_head(report, name, args.must_beat)
                    for name in eligible} if args.must_beat else {}
    if args.must_beat:
        unmeasured = sorted(n for n, s in anchor_stats.items() if s is None)
        if unmeasured:
            raise SystemExit(
                f"no {args.must_beat} head-to-head games for: {unmeasured}")
        qualified = [n for n in eligible
                     if anchor_stats[n]["wilson_low"] > args.must_beat_lower]
    else:
        qualified = eligible

    winner = (max(qualified, key=lambda name: report["ratings"][name])
              if qualified else None)
    selection = {
        "winner": winner,
        "source": by_name[winner] if winner else None,
        "rating": report["ratings"][winner] if winner else None,
        "report": os.path.abspath(args.report),
        "eligible": eligible,
        "qualified": bool(winner),
        "must_beat": args.must_beat,
        "must_beat_lower": args.must_beat_lower if args.must_beat else None,
        "anchor_head_to_head": anchor_stats or None,
    }
    if winner is None:
        write_selection(args.out, selection)
        print(f"NO CANDIDATE QUALIFIED against {args.must_beat}; "
              f"nothing copied to {args.out}")
        for name in eligible:
            stats = anchor_stats[name]
            print(f"  {name}: {stats['wins']}W-{stats['losses']}L "
                  f"({stats['win_rate']:.1%}), 95% Wilson lower "
                  f"{stats['wilson_low']:.3f} <= {args.must_beat_lower}")
        return
    atomic_copy(by_name[winner], args.out)
    write_selection(args.out, selection)
    if args.must_beat:
        stats = anchor_stats[winner]
        print(f"{winner} beats {args.must_beat} {stats['wins']}W-"
              f"{stats['losses']}L ({stats['win_rate']:.1%}), 95% Wilson lower "
              f"{stats['wilson_low']:.3f}")
    print(f"selected {winner} -> {args.out}")


if __name__ == "__main__":
    main()
