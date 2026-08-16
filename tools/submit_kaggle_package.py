"""Submit one package exactly once, retrying safely around quota/network waits."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time


def submissions(api, competition: str):
    return api.competition_submissions(competition, page_size=200) or []


def matching(rows, filename: str):
    return next((row for row in rows
                 if str(getattr(row, "file_name", "")) == filename), None)


ACTIVE_SLOTS = 2  # competition overview: only the most recent 2 are active


def retirement_risk(rows) -> tuple[object, object] | None:
    """(about_to_retire, best_active) if the next upload would retire the best.

    Only the most recent ACTIVE_SLOTS COMPLETE submissions play, and retirement
    follows upload order, so the next upload evicts the oldest active one. When
    that happens to be the highest scoring one, the board score drops by
    construction no matter how good the incoming candidate is. That is exactly
    how a 962 agent got replaced and the board fell to 878.
    """
    completed = [r for r in rows
                 if str(getattr(r, "status", "")).endswith("COMPLETE")]
    active = completed[:ACTIVE_SLOTS]
    if len(active) < ACTIVE_SLOTS:
        return None  # a free slot exists; nothing is evicted
    scored = [(float(r.public_score), r) for r in active
              if getattr(r, "public_score", None) not in (None, "")]
    if not scored:
        return None
    about_to_retire = active[-1]          # oldest active, evicted next
    best_score, best = max(scored, key=lambda pair: pair[0])
    retiring_score = next(
        (s for s, r in scored if r.ref == about_to_retire.ref), None)
    if retiring_score is not None and retiring_score >= best_score:
        return about_to_retire, best
    return None


def write_marker(path: str, row, package: str, existing: bool) -> None:
    payload = {
        "package": os.path.abspath(package),
        "file_name": os.path.basename(package),
        "submission_ref": getattr(row, "ref", None),
        "date": str(getattr(row, "date", "")),
        "status": str(getattr(row, "status", "")),
        "existing": existing,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".partial", "w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2)
    os.replace(path + ".partial", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--poll-seconds", type=int, default=600)
    # Protection is the DEFAULT, not opt-in. An automated overnight submit that
    # evicts the best agent on the board is the exact failure this guards, and a
    # caller that forgets a flag is precisely when it happens.
    parser.add_argument("--protect-best", action="store_true",
                        help="accepted for compatibility; protection is on by "
                             "default. Use --allow-retire-best to disable.")
    parser.add_argument(
        "--allow-retire-best", action="store_true",
        help="permit an upload that retires the highest scoring active "
             "submission. Retirement follows upload order, so this drops the "
             "board score by construction regardless of the candidate's "
             "quality. Pass it only deliberately.")
    args = parser.parse_args()

    while not os.path.isfile(args.package):
        time.sleep(60)
    import kaggle
    api = kaggle.KaggleApi()
    api.authenticate()
    filename = os.path.basename(args.package)

    while True:
        try:
            rows = submissions(api, args.competition)
            old = matching(rows, filename)
            if old is not None:
                write_marker(args.marker, old, args.package, existing=True)
                print(f"already submitted: {filename}", flush=True)
                return
            today = dt.datetime.now(dt.timezone.utc).date().isoformat()
            used = sum(str(getattr(row, "date", ""))[:10] == today
                       for row in rows)
            if used >= 5:
                print(f"daily quota full ({used}/5); retrying", flush=True)
                time.sleep(args.poll_seconds)
                continue
            if not args.allow_retire_best:
                risk = retirement_risk(rows)
                if risk is not None:
                    doomed, best = risk
                    print(f"REFUSING to submit {filename}: the next upload "
                          f"retires {doomed.file_name} "
                          f"(score {doomed.public_score}), which is the best "
                          f"active submission. Re-upload the anchor first so "
                          f"it becomes the newest, or pass "
                          f"--allow-retire-best to override.", flush=True)
                    return
            print(f"submitting {filename} ({used}/5 slots used)", flush=True)
            api.competition_submit(
                args.package, args.message, args.competition, quiet=False)
            rows = submissions(api, args.competition)
            created = matching(rows, filename)
            if created is None:
                raise RuntimeError("upload returned but submission is not listed")
            write_marker(args.marker, created, args.package, existing=False)
            print(f"submitted {filename}: {getattr(created, 'ref', '?')}",
                  flush=True)
            return
        except Exception as exc:
            # Re-listing before every retry makes a lost success response
            # idempotent: an accepted upload is never submitted twice.
            print(f"submission attempt deferred: {exc!r}", flush=True)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
