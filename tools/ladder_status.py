"""Report the live state of our Kaggle submissions, so slots get spent on purpose.

The competition score is not a property of the agent alone.  Measured on
2026-08-05 from the episode API, a submission's rating moves about 50 points per
game for its first ten games, 17 for the next twenty, and 6 after that.  So the
bracket a submission lands in during its opening dozen games is the bracket it
keeps, because matchmaking pairs on current rating and the step size is no
longer large enough to escape.

Two consequences this script exists to serve:

  1. Kaggle's published competition settings keep the **latest two**
     submissions active. Retirement follows upload order, so the oldest active
     archive is the one the next successful upload replaces.
  2. The leaderboard shows the **best scoring active** submission, not the
     newest one. An earlier version of this tool inferred the opposite from
     noisy rank snapshots and consequently understated the board score.

Usage:
    py tools/ladder_status.py
    py tools/ladder_status.py --competition pokemon-tcg-ai-battle
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import statistics
import time

import requests

ENDPOINT = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
MAX_DAILY = 5             # competition settings
ACTIVE_SLOTS = 2          # competition overview: "Only your most recent 2 are active"


def session() -> requests.Session:
    path = os.path.expanduser("~/.kaggle/access_token")
    with open(path, encoding="utf-8") as handle:
        token = handle.read().strip()
    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"})
    return sess


def episodes(sess: requests.Session, submission_id: int) -> list[dict]:
    """Every episode this submission played, oldest first."""
    raw: list[dict] = []
    page = None
    for _ in range(40):
        body: dict = {"submissionId": submission_id}
        if page:
            body["pageToken"] = page
        # The endpoint throttles under a burst and answers with a body that is
        # not JSON.  Silently treating that as "no episodes" would report a live
        # submission as retired, which is the one thing this script must not do.
        # Under a burst it also answers 200 with an empty episode list, which
        # looks exactly like a retired submission.  Retry an empty first page
        # too, and only believe it after several agreeing answers.
        data = None
        for attempt in range(5):
            try:
                candidate = sess.post(ENDPOINT, json=body, timeout=30).json()
            except ValueError:
                time.sleep(1.5 * (attempt + 1))
                continue
            if candidate.get("episodes") or page or attempt >= 2:
                data = candidate
                break
            time.sleep(1.5 * (attempt + 1))
        if data is None:
            raise RuntimeError(
                f"episode API kept refusing for submission {submission_id}; "
                "rerun in a minute rather than trusting a partial answer")
        raw += data.get("episodes", [])
        page = data.get("nextPageToken")
        if not page:
            break

    rows = []
    for episode in raw:
        mine = other = None
        for agent in episode.get("agents", []):
            if agent.get("submissionId") == submission_id:
                mine = agent
            else:
                other = agent
        if mine is None or mine.get("updatedScore") is None:
            continue
        rows.append({
            "end": episode.get("endTime") or episode.get("createTime") or "",
            "before": mine.get("initialScore"),
            "after": mine["updatedScore"],
            "reward": mine.get("reward") or 0,
            "opponent": (other or {}).get("initialScore"),
        })
    rows.sort(key=lambda r: r["end"])
    return rows


def performance_rating(rows: list[dict]) -> float | None:
    """Opponent-adjusted strength, which the raw score is not.

    Two agents of equal strength placed in brackets 80 points apart will show
    scores 80 points apart forever.  This collapses that back out.
    """
    import math
    opponents = [r["opponent"] for r in rows if r["opponent"] is not None]
    wins = sum(1 for r in rows if r["reward"] > 0)
    losses = sum(1 for r in rows if r["reward"] < 0)
    if not opponents or wins == 0 or losses == 0:
        return None
    return statistics.mean(opponents) + 400 * math.log10(wins / losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--limit", type=int, default=6,
                        help="how many recent submissions to pull episodes for. "
                             "The board score is read from every submission's "
                             "public score instead, which needs no episode call.")
    args = parser.parse_args()

    import kaggle
    api = kaggle.KaggleApi()
    api.authenticate()
    # The API defaults to one 20-row page. That made "best ever" forget the
    # actual 972 submission as soon as it aged off page one.
    everything = api.competition_submissions(args.competition, page_size=200)
    sess = session()
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

    # Kaggle ranks the best of the two active submissions. Keep the historical
    # maximum separate because a retired archive no longer represents the board.
    retired_best = max((float(s.public_score) for s in everything
                        if s.public_score not in (None, "")), default=0.0)
    rank = None
    try:
        listing = api.competitions_list(search=args.competition)
        for comp in getattr(listing, "competitions", None) or listing:
            if str(getattr(comp, "ref", "")).endswith("/" + args.competition):
                rank = comp.user_rank
    except Exception:
        pass

    print(f"{'submission':>10} {'eps':>4} {'score':>7} {'peak':>7} {'perf':>7} "
          f"{'state':>12}  file")
    completed = [s for s in everything
                 if str(getattr(s, "status", "")).endswith("COMPLETE")]
    active_refs = {s.ref for s in completed[:ACTIVE_SLOTS]}
    live: list[tuple] = []
    for sub in everything[:args.limit]:
        rows = episodes(sess, sub.ref)
        if not rows:
            print(f"{sub.ref:>10} {0:>4} {'-':>7} {'-':>7} {'-':>7} "
                  f"{'no episodes':>12}  {sub.file_name}")
            continue
        score = rows[-1]["after"]
        peak = max(r["after"] for r in rows)
        perf = performance_rating(rows)
        last = dt.datetime.fromisoformat(rows[-1]["end"][:19])
        idle = (now - last).total_seconds() / 3600.0
        state = "ACTIVE" if sub.ref in active_refs else f"retired {idle:.0f}h"
        if sub.ref in active_refs:
            live.append((str(sub.date), sub.ref, score, len(rows), sub.file_name))
        perf_s = f"{perf:.0f}" if perf is not None else "-"
        print(f"{sub.ref:>10} {len(rows):>4} {score:>7.1f} {peak:>7.1f} "
              f"{perf_s:>7} {state:>12}  {sub.file_name}")

    today = sum(1 for s in everything if str(s.date)[:10] == str(now)[:10])
    print(f"submissions used today: {today} of {MAX_DAILY}")

    live.sort()
    board = max((row[2] for row in live), default=0.0)
    print(f"\nBOARD SCORE (best ACTIVE submission): {board:.1f}"
          + (f"   rank {rank}" if rank else ""))
    print(f"  best historical submission reached {retired_best:.1f}; retired "
          "scores are not ranked")
    print(f"\n{len(live)} submission(s) currently playing, oldest upload first:")
    for i, (_, ref, score, count, name) in enumerate(live):
        tag = "  <- oldest, so the next upload retires this one" if i == 0 else ""
        print(f"  {ref}  {score:7.1f}  {count:>3} eps  {name}{tag}")
    if len(live) < ACTIVE_SLOTS:
        print(f"  fewer than the published {ACTIVE_SLOTS} active slots were found; "
              "check for a pending or errored recent submission")


if __name__ == "__main__":
    main()
