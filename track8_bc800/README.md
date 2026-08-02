# Track 8: larger-data 192d specialist replacements

Status on 2026-08-02: trained, temporally validated, official-engine smoke
tested from both seats, and packaged. Neither archive has been submitted yet.

This track replaces the retired Ogerpon and recovery directions with two
supervised search-prior arms: a stronger Tech-Grim model and a deliberately
different Mega Lopunny/Dudunsparce model. Neither arm uses GRPO. Both share a
new 192d/6-layer/6-head general policy, then receive deck-specific fine-tuning.

The August 1 leaderboard contains about 900 teams at Elo 800+, versus 117 at
1000+. Lowering the threshold therefore adds real pilot and matchup diversity,
but naively pooling it would let weaker decisions overwhelm the strong-player
signal. Training is balanced per day between Elo 800-999 and Elo 1000+, then
high-rated examples receive a modest continuous Elo weight.

## Why these decks

- Tech-Grim is retained because its first downloaded record is 9-4, including
  5-1 into Alakazam, with no runtime failures. Current community discussion
  supports the submitted Budew/Yveltal/Boss package and matchup-dependent
  Froslass usage.
- Mega Lopunny has two close current exact lists at 170 games/67% and 125
  games/71% among Elo-1000+ pilots. Those counts and win rates are unchanged at
  the Elo-800+ threshold. The selected 71% list adds Xerosic to the standard
  Wally/Enriching/Dudunsparce loop. Community reports explain why this loop is
  strong into decks that cannot one-hit Lopunny and also document its Fighting,
  Watchtower, and Alakazam risks. Those weaknesses diversify rather than
  duplicate Tech-Grim's observed matchup profile.

Research:

- <https://www.reddit.com/r/pkmntcg/comments/1v5d381/need_help_fine_tuning_this_grimmsnarl_list/>
- <https://www.reddit.com/r/pkmntcg/comments/1v6khno/is_marnies_grimmsnarl_still_competitive_viable/>
- <https://www.reddit.com/r/pkmntcg/comments/1tagacw/why_did_hale_run_one_abra_in_his_mega_lopunny_deck/>
- <https://www.reddit.com/r/pkmntcg/comments/1tyv9zc/what_happened_to_mega_lopunnydunsparce/>

## Data and resource boundaries

- July 24-30: training only.
- July 31: untouched temporal holdout only.
- Daily Elo-800 shards use uniform reservoir sampling across the entire daily
  export instead of the old archive-order truncation.
- Up to 75,000 decisions per tier per recent training day, plus 50,000 from
  each of four historical Elo-1000+ shards. The completed mixture contains
  1,204,410 decisions: 479,410 at Elo 800-999 and 725,000 at Elo 1000+.
- The July 31 holdout contains all 43,468 available Elo-800-999 decisions and
  50,000 sampled Elo-1000+ decisions (93,468 total).
- The actual general run deterministically caps every shard at 50,000, using
  900,000 decisions (350,000 recent low-Elo, 350,000 recent high-Elo, and
  200,000 historical high-Elo). This is larger and more diverse than the prior
  825,000-decision run without approaching the 23 GiB WSL memory ceiling.
- CPU-streamed tensors and the existing 75% CUDA allocator cap protect WSL.

Build the bounded reservoir shards and balanced mixture with:

```bash
.venv/bin/python track8_bc800/ingest_recent.py
.venv/bin/python track8_bc800/make_balanced.py
```

The 192d model costs 27.8 ms per local CPU call versus 20.0 ms for the 160d
model, projecting to about 4.2 seconds for 150 calls. It remains far below the
agent's 90-second guard and the archive remains small. Training streams tensors
from CPU and retains the existing 75% CUDA allocator cap.

Train the shared model, build deck-anchor subsets, and specialize both arms:

```bash
.venv/bin/python track8_bc800/train_arms.py
```

The Tech specialization caps broad anchor shards at 10,000 and repeats the
exact-list directory three times, preventing the common Grim archetype from
drowning out the submitted 60-card list. Mega Lopunny is rare, so its
specialization uses every matching example in the seven 400,000-decision
reservoir shards and validates on the full July 31 reservoir rather than the
smaller balanced subset.

The general 192d model trains directly from the replay policy labels. A proposed
distillation pass was rejected before training because the NumPy teacher took
30.9 seconds per 256-example batch under WSL contention (about 30 hours for the
full set). The newer refreshed BC remains a gate: it scores 75.43% top-1 on a
fixed 20,000-decision sample from the combined July 31 holdout versus 69.66%
for the 967-era policy.
The final arms are copies of the best temporal-holdout checkpoints, not the last
epochs.

## Completed training and gates

The shared model stopped after epoch 19 and restored epoch 15. The comparison
below uses the same fixed 20,000-decision sample from the combined July 31
holdout:

| General policy | CE | Top-1 |
|---|---:|---:|
| old 967-era BC | 0.85241 | 69.66% |
| refreshed 160d BC | 0.70516 | 75.43% |
| **new 192d BC** | **0.67327** | **76.23%** |

The new general model also beats the refreshed model on both rating slices:
80.23% versus 78.98% top-1 at Elo 800-999, and 73.41% versus 72.56% at Elo
1000+ (20,000 decisions per slice).

Tech specialization uses 331,883 weighted training decisions after per-shard
caps and exact-list repetition. Its 9,763-decision exact-list July 31 gate is:

| Tech policy | CE | Top-1 |
|---|---:|---:|
| current Track 6 specialist | 0.61839 | 76.81% |
| **new 192d specialist (epoch 4)** | **0.58082** | **77.82%** |

Mega Lopunny specialization uses 18,666 full-reservoir anchor decisions and a
13,437-decision July 31 anchor holdout. The slice is intrinsically difficult,
but the new specialist improves CE without sacrificing the general tiers:

| Lopunny policy | CE | Top-1 |
|---|---:|---:|
| old 967-era BC | 1.64971 | 50.79% |
| refreshed 160d BC | 1.43330 | 53.35% |
| shared 192d BC | 1.40425 | 53.03% |
| **new 192d specialist (epoch 2)** | **1.36321** | **53.44%** |

Value-head gates report failure for the specialists because these are
intentionally policy-only priors (`--value-weight 0`); deployment uses them for
root move ordering, not leaf evaluation.

Package after the gates pass:

```bash
.venv/bin/python track8_bc800/build_submissions.py
```

| Archive | Deck | Size | SHA-256 |
|---|---|---:|---|
| `submission_bc800_tech_grim_192.tar.gz` | Tech-Grim | 7.8 MiB | `ab74a7add2cfab1fe7c353193c69d38f7b925373c83e027ebed849291a6f9643` |
| `submission_bc800_lopunny_192.tar.gz` | Mega Lopunny/Dudunsparce | 7.8 MiB | `c39328a35a062f5e82eb6fca9c77f02e12ad6b3b5ababfdab39b93fbeb412c1d` |

Both archives have the required top-level layout, 60-card decks, the expected
`(192, 6, 6, 384)` model metadata, and completed two official-environment
same-deck games with seats alternated. GRPO unit tests and PyTorch/NumPy parity
also pass. These checks establish legality and packaging, not a guaranteed
1000+ ladder rating; that still requires enough opponent-adjusted live games.
