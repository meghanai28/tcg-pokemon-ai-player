#!/usr/bin/env bash
# Retry a submission through Kaggle rate limits. The 429 lands on
# api.authenticate(), which happens before submit_kaggle_package.py's own retry
# loop, so the whole invocation has to be retried from outside.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PKG="$1"; MARKER="$2"; MSG="$3"
LOG=agent/runs/submit_backoff.log
exec > >(tee -a "$LOG") 2>&1
delay=120
for attempt in $(seq 1 40); do
  if [[ -f "$MARKER" ]]; then echo "$(date -Is) marker exists; done"; exit 0; fi
  echo "$(date -Is) attempt $attempt (delay ${delay}s on failure)"
  if .venv/bin/python tools/submit_kaggle_package.py \
        --package "$PKG" --message "$MSG" --marker "$MARKER" \
        --allow-retire-best --poll-seconds 120 2>&1 | tr '\r' '\n' \
        | grep -vE "^\s*[0-9]+%" | tail -5; then
    if [[ -f "$MARKER" ]]; then
      echo "$(date -Is) SUBMITTED"; cat "$MARKER"; exit 0
    fi
  fi
  sleep "$delay"
  delay=$(( delay < 600 ? delay + 120 : 600 ))
done
echo "$(date -Is) gave up after 40 attempts"; exit 1
