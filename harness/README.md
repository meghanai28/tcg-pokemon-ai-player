# harness/

Ground truth for this project. Everything in here exists so that a local
measurement can be checked against a ladder result we actually observed.

## anchors/

Packaged submissions, each named for the score it earned on the Kaggle ladder.
These are the external references every gate must include. The project's
recurring failure has been gating our own components against each other, which
can look excellent while the whole family sits far below a baseline nobody
tested.

| file | ladder | Kaggle ref | provenance |
|---|---|---|---|
| `grpo_tech_grim_972_912_811.tar.gz` | 972.0, 911.9, 810.8 | 55185089, 55202336, 55233305 | exact; md5 `4f6b66a1ecb8ee13f9d67e5d97a3bdeb` |
| `awr_grpo_tech_grim_903.tar.gz` | 903.2 | 55202342 | exact |
| `bc800_tech_grim_849.tar.gz` | 848.9 | 55195501 | exact |
| `ppo_search_585.tar.gz` | 585.2 | 55235083 | exact |
| `v3_pure_rl_480.tar.gz` | 480.0 | 55233486 | exact |
| `grpo_search_405_RECONSTRUCTED.tar.gz` | 405.0 | 55234807 | **rebuilt, not byte-verified** |

Three of those refs are the **same bytes** submitted on three consecutive days:
972.0 → 911.9 → 810.8. That 161-point spread on an unchanged agent is the noise
floor any single ladder result has to clear, and it is falling over time, so
scores from different days are not directly comparable.

`grpo_search_405_RECONSTRUCTED` was rebuilt from the GRPO policy stack plus the
since-deleted PUCT `main.py`/`search.py`, because the original tarball was
removed before it was recognised as an anchor. It reproduces the recorded
archive to within ~400 bytes but was not byte-compared. Treat its 405.0 as
indicative; the exact `ppo_search_585` anchor covers the same failure mode.

## Running it

```bash
.venv/bin/python tools/ladder_harness.py \
  --archives harness/anchors/*.tar.gz artifacts/<candidate>.tar.gz \
  --games-per-pair 20 --budget 1.1 --workers 10 --calibrate
```

`--calibrate` fits the local Bradley-Terry ratings against the anchors' known
scores and prints the rank correlation and per-anchor residuals. A harness that
cannot order the agents whose ladder scores we already know has no business
ordering the ones we do not.

## Calibration result, 252 games at 1.1 s/move, 2026-08-04

On the five byte-exact anchors:

**ladder = 1.06 * rating + 680.0, R squared 0.987, Spearman +0.900**

| archive | rating | predicted | actual | residual |
|---|---:|---:|---:|---:|
| `grpo_tech_grim` | +213.8 | 906.4 | 898.2 | +8.1 |
| `awr_grpo_tech_grim` | +186.8 | 877.8 | 903.2 | -25.4 |
| `bc800_tech_grim` | +165.0 | 854.7 | 848.9 | +5.8 |
| `ppo_search` | -60.9 | 615.5 | 585.2 | +30.3 |
| `v3_pure_rl` | -206.6 | 461.2 | 480.0 | -18.8 |

Every residual sits inside plus or minus 31 points, well under the roughly 160
points of ladder noise, and the only rank swap is `grpo` against `awr`, which
are 5 ladder points apart and therefore not separable anyway.

The rebuilt anchor is the exception and is excluded by default. It predicts
677.8 against a recorded 405.0, a residual of +272.8, and including it drops the
fit from R squared 0.987 to 0.635. That is strong evidence the rebuild is not
the agent that actually scored 405. Pass `--trust-rebuilt` to include it anyway.

Read predictions as ordering plus a rough scale, not as a promised score. The
anchors span 480 to 898, so anything outside that range is extrapolation.

## `--budget` is not a performance knob

It decides the answer. Measured on the same two archives, changing only
seconds-per-move:

| `--budget` | outcome | agrees with ladder? |
|---|---|---|
| 0.1 s/move | `v3_pure_rl` (480.0) beat `grpo_tech_grim` (810.8 to 972.0) by 7-1 | **no, inverted** |
| 1.1 s/move | `grpo_tech_grim` beat `v3_pure_rl` by 10-0 | yes |

A small budget starves search while leaving a search-free agent untouched, since
that one costs a single 26 ms forward pass per decision. So a cheap run does not
just add noise, it systematically promotes our weakest agent. Gate at 1.1 s/move, the shipping
value. Use cheap runs only to smoke-test plumbing.

Do not run other CPU-heavy work alongside a harness run. Workers are pinned to
one thread each precisely so that per-move budgets stay comparable across
matches; oversubscribing the machine breaks that.
