# CLAUDE.md

Agent for the Kaggle [`pokemon-tcg-ai-battle`](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
competition. A submission is a tarball containing `main.py` exposing
`agent(observation) -> list[int]`: the runner calls it once with `select == None`
to receive a 60-card deck, then once per in-game selection callback to receive
the indices of the chosen options.

## Orientation

**`rl_osfp/` is the only live track.** Everything else in this repo is history.

The project was reset on 2026-08-03. Every earlier track (`track1_search`,
`track2_dmc`, `track5_grpo`, `track6_controlled`, `track8_bc800`,
`track9_awr_grpo`, ...) was moved to `.reset-quarantine-20260803/` and is
retained only as evidence. Those tracks were built on imitation of mined
replays; `rl_osfp` deliberately starts from random weights and never uses a
replay action label.

- `README.md` was rewritten on 2026-08-04 and is current. It is the short
  version; this file is the long one.
- `tools/divergence.py`, `tools/run_local.py`, `tools/autopsy.py` were deleted:
  they imported paths under `track1_search/` and had been broken since the reset.
  `foundation/model.py` (the old `TCGNet`) was deleted as dead code. All are
  recoverable from git history.

Use `.venv/bin/python` (3.12, torch 2.13, numpy 2.5). Run everything from the
repo root as a module (`.venv/bin/python -m rl_osfp.<name>`); the packages are
not installed.

## Layout

```
rl_osfp/          live track: PPO best responses inside optimistic fictitious play
  network.py      ActorCritic transformer (option / count / value heads)
  policy.py       masked action distribution for one selection callback
  arena.py        sequential rollouts through the native engine
  train.py        the training loop
  rollout.py      parallel on-policy rollouts (training infra, NOT search)
  evaluate_population.py   checkpoint selection (gauntlet + round robin)
  evaluate_decks.py        deck selection gate for the chosen checkpoint
  build_bcsearch_submission.py   checkpoint behind the frozen search shell
  verify_bcsearch_submission.py  prove the shipped net actually supplies priors
  build_submission.py      package the search-free agent (diagnostics only)
  verify_submission.py     prove the packaged tarball matches the checkpoint
  agent_main.py   search-free agent; ships as main.py in that tarball
  run/ run_v2/ run_v3/     checkpoints, metrics.json, selection reports
foundation/       shared engine bindings, encoders, and the frozen search
  cg/             official native engine (libcg.so) + python bindings
  search_shell_main.py  FROZEN main.py that scored 972.0/911.9/810.8 - never edit
  nn_infer_adapter.py   ships as nn_infer.py; trims forward() to the shell's 2 values
  nn_infer.py     numpy ActorCritic (option, count, value)
  nn_features.py  base encoder, MAX_OPT 64 so SEQ is 93, 32 scalars
  nn_features_rich.py  wrapper that resolves area+index card references
  deck_tech_grim.csv   the deck.csv from every archive that scored above 800
harness/          the measurement ground truth
  anchors/        packaged tarballs named for the ladder score they earned
  *.json          round-robin and calibration reports
data/fresh/       mined deck pool + the replay/leaderboard archives it came from
tools/            deck mining, replay fetching, and the harness
  ladder_harness.py  archive-vs-archive round robin through the real cabt runner
  top_decks.py       rank the field by elite adoption, not just win rate
```

**Engine binaries are licensed "PTCG-ABC-Competition-Use-Only" and are
gitignored, so never commit or redistribute `**/cg/*.so`, `sim.py`, `game.py`,
`api.py`, `utils.py`.** Obtain them from the competition page or from
`kaggle_environments/envs/cabt/cg/`.

## How the live track works

`train.py` runs periods. Each period freezes the current weights as the
behaviour policy, plays `--games-per-period` games (60% self-play, else a
sampled historical league snapshot), collects on-policy decisions with their
terminal win/loss outcome, and takes a PPO step. A snapshot is archived into
the league when it beats every existing member above `--archive-threshold`, or
after `--archive-max-wait` periods, whichever comes first.

Design constraints that are load-bearing:

- **Replay-free.** Weights start random. Mined replays choose the *deck
  population only* (`data/fresh/deck_pool.json`); they never supply an action
  label. `run_config` records `replay_action_labels: False`.
- **Complete actions.** A selection is a count plus a set of options chosen
  without replacement. `policy.py` scores the full Plackett-Luce sequence, so
  PPO optimizes the whole action rather than only the first card.
- **`MAX_OPT` is 64** in `nn_features.py` while the docstring says 53 tokens;
  the encoder's sequence is `1 + MAX_BOARD + MAX_HAND + MAX_OPT`. If a slot
  comes back negative the policy falls back rather than guessing.
- **The engine shuffles internally.** `battle_start` takes no seed, so repeated
  games with `sample=False` and identical decks still differ. Deterministic
  evaluation games are genuinely independent, verified rather than assumed.

## Selection pipeline

The final checkpoint is a *candidate*, not the winner. "Latest is best" does
not hold on this run: period 12 loses to period 4 and to period 7. Selection is
three gated stages, each writing JSON into `rl_osfp/run/`.

```bash
# 1. screen every checkpoint against a fixed panel (O(n), comparable scores)
.venv/bin/python -m rl_osfp.evaluate_population --periods all --panel 1,4,7,10 --games-per-deck 6

# 2. high-volume round-robin runoff among the survivors
.venv/bin/python -m rl_osfp.evaluate_population --periods <top-N> --games-per-deck 20

# 3. deck gate on the winning checkpoint (share-weighted; see note below)
.venv/bin/python -m rl_osfp.evaluate_decks --period <winner> --field-games 5 --share-budget 200 --pool <pool-with-all-decks-as-candidates>

# 4. package, then prove the package matches the checkpoint
.venv/bin/python -m rl_osfp.build_submission --model rl_osfp/run/model_period_<winner>.npz --deck-group field_decks --deck-index <index>
.venv/bin/python -m rl_osfp.verify_submission --model rl_osfp/run/model_period_<winner>.npz --deck-group field_decks --deck-index <index>
```

Both sides always pilot the **same deck** in a checkpoint comparison, so deck
strength cancels and only policy strength is measured. The deck gate then holds
the policy fixed and varies the deck.

### Shipped artifacts

Archives with a **measured ladder result** live in `harness/anchors/`, named for
what they scored. They are the reference set every gate must include.

| anchor archive | what it is | ladder |
|---|---|---|
| `grpo_tech_grim_972_912_811` | BC + conservative GRPO, 160d/5L, Tech-Grim deck | **972.0 / 911.9 / 810.8** (same bytes, refs 55185089, 55202336, 55233305) |
| `awr_grpo_tech_grim_903` | Elo-1000 weighted BC + AWR-GRPO, Tech-Grim | 903.2 |
| `bc800_tech_grim_849` | 192d/6L BC on 900k Elo-800+, Tech-Grim | 848.9 |
| `ppo_search_585` | v3 PPO priors + the deleted PUCT, deck `field_3` | 585.2 |
| `v3_pure_rl_480` | v3 PPO period 180, **no search**, deck `field_3` | 480.0 |
| `grpo_search_405_RECONSTRUCTED` | GRPO policy stack + the deleted PUCT | 405.0 (rebuilt, not byte-verified) |

**`submission_mcts_track1.tar.gz` has no ladder result.** An earlier version of
this file credited it with 972.0 / ref 55185089; that is false. Ref 55185089 is
`submission_grpo_controlled_tech_grim.tar.gz`, and the two agents are not
variants of each other. `mcts_track1` is 96d/3L with 51 tensors, ships only the
base `nn_features.py`, and carries a different deck, against the GRPO archive's
160d/5L with 75 tensors and `nn_features_rich.py`. Do not use it as a baseline.

**The searching `main.py` is one frozen file.** It is md5-identical
(`e54bc6590288e659d696d00d432c6cc4`) inside the 972.0, 903.2 and 848.9 archives;
only `model.npz` and `deck.csv` differ between them. It is vendored at
`foundation/search_shell_main.py` and is **never edited**, and `build_bcsearch_submission.py`
refuses to build if its md5 drifts. To put a different network behind it, swap
the files *around* it (see `foundation/nn_infer_adapter.py`), never the shell.

### Policy training result (first clean run, 12 periods, 2026-08-03)

Selected **period 4 with `field_1`**, packaged and verified. Reports live in
`rl_osfp/run/`: `population_eval_gauntlet.json` (screen),
`population_eval_runoff.json` (the checkpoint decision), `deck_gate.json`, and
`artifacts/submission_verification.json`.

Period 4 wins the symmetric runoff cleanly: 67.2%, Wilson 0.588, versus 0.431
for period 5 over 134 decided games.

**Do not ship a `learner_decks` entry without re-checking it.** The pool builder
ranks learner decks by Wilson lower bound on *expert* ladder win rate, which
selects decks that reward skilled piloting. This policy cannot pilot them.
Piloted by period 4 against the field, `learner_0` scores 44.8% while `field_1`
scores 85.1%, a 40-point swing, decided by which deck is forgiving of bad play,
not by which deck is good. `field_1`'s expert ladder win rate is only 54.4%.
`field_1`, `field_7`, `field_3` and `field_10` are statistically tied at the top
(85.1 / 84.9 / 84.8 / 82.6, Wilson ~0.79); `field_1` was taken on the primary
weighted metric plus a 0.0% stall rate.

Caveat on the run itself: per-period `approx_kl` is 0.0003-0.005 against a 0.04
target, so PPO is taking near-zero steps at `lr=7e-5` over ~3.5k decisions and
2 epochs. Early periods improve, then 10-12 regress. The ceiling here is set by
the learning rate and decision budget, not by which checkpoint gets picked, and
the deck slot is currently worth far more than the checkpoint slot.

## Ladder evidence (fetched 2026-08-04 via the Kaggle API)

Rank **478 / 6,396** at **871.4** as of 2026-08-06, down from 313 / 6,224 on
2026-08-04 with no change in our agent. The board is strengthening around a
static submission, and the top fell too (1,297.9 Majkel1337 on 08-04 against
1,204.0 flg on 08-06), so the whole scale moves.

Deadline **2026-08-16 23:59**, 5 submissions a day, `awards_points: True`, so
medals are awarded despite the "Knowledge" reward field. A second competition,
`pokemon-tcg-ai-battle-challenge-strategy` ($240,000, `awards_points: False`),
is entered but unranked.

What the board pays, fetched 2026-08-06:

| rank | score | note |
|---:|---:|---|
| 1 | 1,204.0 | flg |
| 10 | 1,111.0 | |
| 23 | 1,066.5 | roughly the gold cutoff (top 10 + 0.2% of 6,396) |
| 100 | 998.5 | |
| 320 | 906.6 | roughly the silver cutoff (top 5%) |
| **478** | **871.4** | **us** |
| 640 | 836.6 | roughly the bronze cutoff (top 10%) |

So we are currently inside bronze, about 35 points from silver and about 195
from gold. Our best archive at its usual ~940 would be around rank 210, which is
silver. Getting to gold needs a real gain of roughly 130 points over the
champion, which nothing measured in this repo has produced.

Three facts from submission history that should govern decisions:

- **Search is worth roughly 300 ladder points.** The one search-free submission
  ever made (`submission_purebc_scaled`, "192d/6L, 291k Elo-1000 decisions, NO
  search, argmax policy") scored **591.1**, the worst result on record. Every
  search-based submission scored **873 to 972**.
- **Ladder score carries roughly ±70 points of matchmaking luck on identical
  bytes.** See the section below; the mechanism is now measured rather than
  guessed. Never compare a submission to a score from a different run, and never
  conclude anything from one submission. A local gate over hundreds of games is
  more trustworthy than any single ladder result.
- **Deck choice moves the score as much as the model.** 967.1 vs 917.6 was the
  same model with a different deck; 972.0 vs 719.2 likewise. Re-gate the deck
  whenever the pilot changes.

`learner_0` is **Majkel1337's list, the rank-1 player** (1,297.9 as of
2026-08-04). Its 75% ladder win rate conflates deck and pilot, but it is the top
player's choice, not noise. Measured under a *search* pilot it still only tied
for last among candidates, so gate it rather than assuming either way.

## How the ladder actually scores us (measured 2026-08-05 from the episode API)

This is the most important thing in this file, because it decides how every
other measurement here should be read. `tools/ladder_status.py` reports it live,
and the raw source is
`https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes`, which
returns every episode a submission played with the rating before and after and
the opponent's rating.

**The score is an Elo random walk that starts at 600 and freezes.** Measured
mean absolute rating change per game, across our submissions:

| games played | mean rating move per game |
|---|---:|
| 1 to 10 | **50** |
| 11 to 30 | 17 |
| 31 and up | 6 |

Matchmaking pairs on current rating, so the bracket the first dozen games put a
submission in is the bracket it stays in: after game 30 the step size is too
small to climb out. Two examples of that first-dozen luck, both agents strong:

- the 972 archive's first game was a win over a 628 and paid **+117.2**
- the retrained-prior archive's first game was a win over a **176** (a broken
  agent) and paid **+20.7**

**Only the most recent handful of submissions stay active.** Everything older
stops receiving episodes and keeps its last rating forever. Retirement follows
upload order, so `ladder_status.py` prints which one is about to be frozen.
The exact cap is not published and our own history does not pin it to a
constant: two to three run at once in the steady state, but six were briefly
playing on 2026-08-04 after five uploads inside 80 minutes. Do not hardcode a
number, read it off the tool.

**The board does NOT keep our best score ever, and an upload can lower it.**
This is the trap, and an earlier version of this section got it backwards. Read
directly off the leaderboard on 2026-08-06:

```
   477 Eggplanck                871.6
   478 Meghana284               871.4     <- us, of 6,396
   479 Ochir Dorzhiev           871.3
```

871.4 is the live score of ref 55264582. It is not 972.0 (our best submission
ever, retired 2026-08-02) and it is not 942.3 (ref 55256846, the champion
resubmit). Only *active* submissions are ranked, and a retired one stops
counting the moment it is retired. A board score of 972.0 would sit near rank
150, so the error is worth about 330 places.

So `max(public_score)` over the submission list is **not** the board score, and
`tools/ladder_status.py` no longer prints it as one. Uploading retires the oldest
active submission, so an upload made while our best archive is the oldest one
costs board position immediately. That happened on 2026-08-06: uploading ref
55290078 retired the 942.3 champion and dropped the board to 871.4.

The saving grace is the deadline. Only the score showing on **2026-08-16 23:59**
is ranked, so a dip before then costs nothing as long as the slots are managed
back up in time.

### So the "monotonic decline" in the old version of this file was not real

The old note read 972.0 to 911.9 to 810.8 on identical bytes and concluded the
board was strengthening around a fixed agent. Both halves are wrong:

- **810.8 was never a converged score.** Ref 55233305 played **three episodes**
  before two later uploads retired it. It is a truncated run, not a result.
- **The sequence is not monotonic.** The same bytes have now scored 972.1,
  911.9 and 942.3, on runs of 65, 83 and 82 episodes. Mean 942, sample standard
  deviation 30. There is no trend, just a spread.

### Why the retrained-prior submission "lost" to the champion

Same day, 5.4 hours apart: champion 942.3, retrained prior 867.3. That looks
like a 75-point regression from training on fresh data. It is not. Adjusting for
who each one actually played:

| | games | win rate | mean opponent | performance rating |
|---|---:|---:|---:|---:|
| champion (ref 55256846) | 80 | 57.5% | 888.0 | **940.6** |
| retrained prior (ref 55264582) | 70 | **58.6%** | **810.5** | 870.6 |

The retrained prior won a *higher* share of its games. It was simply placed in a
field averaging 77.5 points weaker, and 77.5 is almost exactly the 75-point gap
in the reported scores. Both agents sit about 55 to 60 points above their own
field. They are the same strength, which is exactly what the local 300-game gate
said when it returned 151-149.

**The local harness was right and the ladder reading was wrong.** When a gate
and a single ladder result disagree by less than ~100 points, believe the gate.

### What this means for every gate in this file

- **Compare with performance rating, never with the raw score**, whenever both
  numbers come from the ladder. `ladder_status.py` prints it.
- **A change has to be worth more than ~70 points to be visible at all**, so
  small true improvements cannot be detected by submitting them. They have to be
  gated locally over hundreds of games.
- **Uploads are not free, but before the deadline a dip does not matter.** What
  is ranked is the best of our *active* submissions on 2026-08-16 at 23:59, so
  the only score that counts is the one showing then. Until roughly 2026-08-14,
  take draws freely: a submission that lands badly can be replaced. In the last
  two days, get two or three champion draws running, let them mature past ~30
  episodes, and then **stop uploading**, because every further upload retires
  the oldest active one and could throw away the good draw.
- **Never read the board score off `max(public_score)`.** That counts retired
  submissions and reads about 100 points high. Use
  `competitions_list().user_rank` and `tools/ladder_status.py`.

### The deck we keep shipping is not a top deck in the current meta

Every one of our three best results ships the **same** `deck.csv` (md5
`110101537284ad862aa27b42768800fb`, "Tech-Grim"): 972.0/911.9/810.8, 903.2 and
848.9. That is our strongest deck evidence by far, but it is evidence about
*our pilot*, not about the deck.

Matched against the current Elo-1000 pool, Tech-Grim is **`field_0`/`field_2`**
(multiset Jaccard 0.846), and those sit at a **48.5% ladder win rate**.
`field_0` is simultaneously the most-played list in the field (4,611 of ~12,655
appearances). The decks actually winning are `field_16` (= `learner_0`, 75.0%),
`field_17` (66.3%), `field_4` (59.6% over 631) and `field_9` (59.0%).

So "the deck that won us 972" and "the best deck in the meta" are different
decks, and `CLAUDE.md`'s own warning applies, since `field_0` "needs competent
piloting". Do not resolve this by argument. Hold the model fixed and gate the
deck through `tools/ladder_harness.py` at 1.1 s/move, which is the only measure
that has reproduced ladder ordering.

`tools/top_decks.py` ranks the field by **elite adoption** (appearances by
pilots at/above a given Elo) alongside win rate, because a win rate conflates
deck with pilot while adoption by strong players does not reward a deck merely
for being forgiving.

### The policy never piloted the deck we ship it with

`train.py` draws the learner's deck from `learner_decks` and the opponent's from
`field_decks`, and never crosses them:

```python
own_deck = weighted_deck(rng, learner_decks)        # train.py:445
opponent_deck = weighted_deck(rng, field_decks)     # train.py:446
```

So a deck sitting only in `field_decks` is one the policy has faced thousands of
times and played zero times. That is exactly the v3 checkpoint's situation:

| deck | role in `deck_pool_train.json` | Jaccard to Tech-Grim |
|---|---|---|
| `field_1`, `field_3`, `field_7`, `field_10` | **learner** (the only decks it piloted) | 0.062, 0.121, 0.053, 0.132 |
| `field_0`, `field_2` (= Tech-Grim) | field only, never piloted | **0.846** |

`nn_features_rich` is deck-aware and encodes the decklist, so the net can tell it
is holding a list it has never played. This is the most likely reason
`submission_ppo_bcsearch` (v3 net + Tech-Grim) underperforms: its priors are out
of distribution for the deck in the tarball.

Fix, and it is cheap. run_v3 did 200 periods in **100 minutes**, and
`train.py --resume` warm-starts from `training_state.pt` while reading the pool
fresh from `--pool`:

```bash
.venv/bin/python tools/make_learner_pool.py --deck-csv foundation/deck_tech_grim.csv \
    --label tech_grim --out data/fresh/deck_pool_techgrim.json
cp -r rl_osfp/run_v3 rl_osfp/run_v4
.venv/bin/python -m rl_osfp.train --resume --pool data/fresh/deck_pool_techgrim.json \
    --out-dir rl_osfp/run_v4 --periods 260
```

**Do this after the deck gate, not before.** Specialising on Tech-Grim is wasted
if the gate picks a different deck. Train on whatever deck is going to ship.

## DELETED: `rl_osfp/search.py`, our PUCT reimplementation

Removed 2026-08-04 (recoverable from git history). It was measured against the
frozen shell on the same day, on the same policy stack, and lost badly:

| submission (all 2026-08-04) | search | ladder |
|---|---|---:|
| GRPO Tech-Grim, ref 55233305 | **frozen shell** | **810.8** |
| identical policy/deck/encoders, ref 55234807 | `rl_osfp/search.py` | **405.0** |
| v3 PPO + `rl_osfp/search.py`, ref 55235083 | `rl_osfp/search.py` | 585.2 |
| v3 PPO, ref 55233486 | none | 480.0 |

The reimplementation cost ~405 points against the shell. Two corrections to how
this was previously written down here:

- The 810.8 archive was described as "**no search**". It is not. It is the
  frozen determinized-PUCT `main.py`. The real comparison is *our search versus
  the proven search*, not search versus no search, and framing it the other way
  invites re-deriving a search we already have.
- The collapsed submission scored **405.0**, not 335.3.

The proven shell survives as `foundation/search_shell_main.py`. There is no
reason to maintain a second implementation of it; if search needs to change,
change it there under a gate that includes the anchors.

### The methodological error that caused this

**Every gate in this project compared our own components to each other.** Search
vs our policy, checkpoints vs each other, decks vs a heuristic, fixed search vs
broken search (17-7). All of those can look excellent while the entire family is
far below a baseline that was never tested against. The GRPO archive sat in
`.reset-quarantine-20260803/` the whole time.

**Rule: every gate must include an external reference.** The references are the
archives in `harness/anchors/`, each named for the ladder score it actually
earned. A change is real only if it beats those, not if it beats our previous
attempt.

## The local harness

`tools/ladder_harness.py` plays **packaged tarballs** against each other through
`kaggle_environments.make("cabt")`, the same environment that scores the
competition, and fits Bradley-Terry ratings to the results. `--calibrate`
regresses those ratings against the anchors' known ladder scores, so the harness
reports how well it reproduces an ordering we already know before being trusted
on one we do not.

```bash
.venv/bin/python tools/ladder_harness.py \
  --archives harness/anchors/*.tar.gz artifacts/<candidate>.tar.gz \
  --games-per-pair 20 --budget 1.1 --workers 10 --calibrate
```

### `--budget` decides the answer, and a small one inverts the ladder

This is the harness's single most important parameter. Same two archives, same
code, only seconds-per-move differing:

| `--budget` | result | matches ladder? |
|---|---|---|
| 0.1 s/move | `v3_pure_rl` (480.0) beat `grpo_tech_grim` (810.8 to 972.0) by **7-1** | **no, inverted** |
| 1.1 s/move | `grpo_tech_grim` beat `v3_pure_rl` **10-0** | yes |

A small per-move budget starves a search agent while a search-free agent, which
answers every decision with one ~26 ms forward pass, does not notice. So a cheap
harness run does not merely add noise. It systematically ranks our *worst*
agent above our best. **Always calibrate and gate at the shipping budget
(1.1 s/move).** Cheap runs are for smoke-testing plumbing, never for a decision.

Cost is not the reason to run cheap: agents are cached per worker process, and
420 games at 1.1 s/move across 10 workers is well under two hours.

### It is calibrated: R squared 0.987 on the exact anchors

252 games at 1.1 s/move, 2026-08-04. Fitting the five byte-exact anchors:

**ladder = 1.06 * rating + 680.0, R squared 0.987, Spearman +0.900**, with every
residual inside plus or minus 31 ladder points. The only rank swap is
`grpo_tech_grim` against `awr_grpo`, which are 5 ladder points apart.

So local ratings now translate to a ladder estimate. Treat it as ordering plus a
rough scale, not a promised score, and remember the anchors only span 480 to 898.

`grpo_search_405_RECONSTRUCTED` is the exception, and it is excluded from the fit
by default. It predicts 677.8 against a recorded 405.0, and including it drops
R squared to 0.635. The rebuild is evidently not the agent that scored 405, which
is a good reminder that a "reconstructed" anchor is not an anchor.

**Measured verdict on the PPO checkpoint behind the shell:**
`submission_ppo_bcsearch` finished last at 8.3% (6-66), predicting a ladder score
of **311.8**. That is below the *search-free* v3 agent at 480, so the v3 priors
are not merely failing to help the search, they are actively degrading it. See
the learner-deck mismatch above for the cause.

### Two deliberate infidelities, both required

`kaggle_environments.agent.get_last_callable` execs `main.py` as a string with
`__file__` undefined and takes the *last* callable defined in the module. The
harness reproduces that, with two exceptions needed to run two archives in one
process:

- **`__file__` is seeded.** Without it the shell falls through to
  `/kaggle_simulations/agent` and then `os.getcwd()`; there is one cwd and two
  agents, so both would read the *same* `deck.csv` and `model.npz`. On Kaggle
  that fallback resolves correctly per agent, so seeding it reproduces Kaggle's
  outcome rather than diverging from it.
- **`sys.modules` is swapped per call.** The shell aliases `nn_features*` by
  agent directory for exactly this reason but does **not** alias `nn_infer`, and
  our archives disagree about what `nn_infer.NumpyNet` is. Whichever loaded
  first would silently define the other's network, and the loser would fall back
  to heuristic priors and keep playing, invisibly.

## Search is the dominant lever

**The engine has a native search API**: `SearchBegin` / `SearchStep` /
`SearchEnd` / `SearchRelease`, and every observation carries a
`search_begin_input` blob. It is built to be searched.

A search-free policy answers each decision with one forward pass (~26 ms)
against a ~600 s episode budget. Measured on this machine, `SearchStep` costs
0.026 ms, so **39,000 engine steps/second**, so a 4 s decision slot buys ~156,000
engine steps. Shipping the bare policy leaves ~99% of inference compute unused.

Measured head-to-head, same deck both seats, only the decision procedure
differing: **search beat the search-free policy 21-3 (87.5%, Wilson 0.690)** over
24 games at just 0.5 s/move, with zero fallbacks and zero invalid actions.

The 2026-08-03 reset discarded replay imitation *and* search together. Only the
first was a good idea. Search needs no replay data, so dropping it forfeited
the project's one measured live result for nothing.

### Search findings that are load-bearing

These were verified against this exact engine. Do not re-litigate them without
new measurements.

- **Throughput decides.** Heuristic leaves ran ~7,500 simulations where neural
  leaves managed ~220, a 35x gap.
- **A learned value head makes search worse**: 1W-19L, then 0W-6L, against
  heuristic leaves. 46% of its outputs saturate above 0.95 on positions it never
  trained on, so PUCT commits to a line instead of verifying it against the
  engine. The engine cannot be miscalibrated the way a value head can. Use
  rollouts with engine-truth outcomes.
- **Network priors are neutral** (11W-9L) and cost ~2.5 ms, so they earn their
  place only at the root, where they decide which subtrees get explored at all.
- **A well-calibrated value head does not rescue neural leaves.** The v3 PPO
  head is genuinely trained: 97.3% sign accuracy, MSE 0.076 against a 1.000
  always-zero baseline, versus v1 at 47.4% (worse than chance). It is *not* the
  miscalibrated BC head that failed before. Even so, guided search with
  `prior_weight=0.7, value_weight=0.5` **lost to pure search 6-13 (31.6%)** at a
  matched 0.3 s/move deadline. Guidance is paid for out of the search budget:
  a value call per simulation buys accuracy at the cost of simulation count, and
  simulation count is what wins. Note the head saturates on 77.5% of positions,
  so it discriminates poorly between sibling moves even when its sign is right.
  Any guidance experiment must be gated head-to-head against unguided search at
  a matched deadline, never against the search-free policy, because pure search already
  beats that 87.5%, so the comparison is at ceiling and resolves nothing.
- **Keep the evaluator's magnitude small.** Its job is breaking ties between
  sibling moves. A "lethal awareness" term large enough to dominate the search
  regressed the ladder from 65% to 40%.
- **Every `SearchStep` mints a new persistent `searchId`.** Release them or a
  long search leaks the engine arena; `Engine.end()` frees a whole move at once.
- **Determinization must produce exactly-sized zones** or `SearchBegin` rejects
  the world and search silently degrades to a fallback. A setup-turn world also
  needs a basic Pokemon in the deck.

### Deployment bugs that cost real submissions (all fixed)

All three were invisible to the verifier because it tested a path Kaggle never
takes. Reproduce the runner exactly when verifying: exec the source as a
*string* (so `__file__` is undefined), from a foreign cwd, with the repo off
`sys.path`.

- **The runner execs `main.py` as a string and pops the agent dir off
  `sys.path` before `agent()` is ever called** (`kaggle_environments/agent.py`
  `get_last_callable`). Lazy imports inside a `_load()` then raise
  `ModuleNotFoundError`, the broad `except` returned `[0]`, and a 1-card deck
  fails the episode outright: *"Validation Episode failed."* Fixed by resolving
  `AGENT_DIR` via a search for `deck.csv`, plus an `EMBEDDED_DECK` literal the
  packager injects so the deck callback can never return a short deck.
- **Rollout leaves cost ~22x search throughput** - 764 vs 17,054 sims/move at a
  1.1 s budget, because each rollout walks up to `ROLLOUT_CAP` engine steps.
  track1 kept them opt-in behind `PTCG_ROLLOUT=1` for this reason. Static
  evaluation is the default; rollouts are opt-in.
- **`cabt` awards the game to the opponent on any rejected `select`**, with no
  retry (`interpreter()` sets `INVALID`, `reward=-1`). Local harnesses that
  catch the exception and retry hide this completely. `validate_action()` guards
  index range, duplicates, and min/max count.
- **The observation carries `remainingOverageTime`** (~600 s). Use it; never
  hardcode an episode budget. Weight per-decision spend by context (1.6x
  main-phase, 0.6x for <=2 options) and cap around 1.1 s/move.

### Verifying a search agent

A search agent **fails quietly**: every failure path returns a legal heuristic
action, so a completely broken search still yields a well-formed archive that
plays legal games to completion. "It ran without raising" proves nothing.
`verify_search_submission.py` therefore also asserts a per-decision latency
floor, below which the agent is falling back rather than searching, and that the
worst game stays inside the episode budget.

### Evaluation traps that already bit this project

- **Step caps are not errors.** Greedy mirror matches stall: roughly a quarter
  of deterministic evaluation games hit the 2,400-step cap, while all 288
  sampled training games finished cleanly. `arena.py` reports these as
  `error="step cap reached"`. Counting them as errors and dropping them
  *rewards stalling*, because a policy that refuses to commit gets its stalls
  deleted from the denominator instead of counted as non-wins. The eval scripts
  separate `step_caps` from `engine_errors` and report a cap-inclusive
  `score_rate` (caps as half points) alongside the raw win rate. If the two
  criteria disagree the scripts print a warning and the pick is unresolved.
- **Wilson bounds on ~35 decided games separate almost nothing.** Run enough
  games that the runoff's intervals actually part before trusting a winner.
- **Pick the deck for the pilot you have, not the pilot you want.** The deck
  gate holds the policy fixed and puts the *same* model on both seats, so pilot
  skill cancels and any deviation from 50% is deck effect. Under that measure a
  deck's expert ladder win rate is nearly useless: `field_0` is 46.6% of the
  ladder and a perfectly normal deck in skilled hands, but it loses ~85-90% to
  all four top decks under this policy because it needs competent piloting.
  Search the whole pool (`--deck-group field_decks`), never just `learner_decks`.
- **The absolute win rate from the deck gate does not transfer to the ladder.**
  The opponent is our own weak policy, so 85% means "beats these decks when both
  sides are piloted badly." Real opponents are Elo 900+ and will pilot `field_0`
  competently. What transfers is the *ordering*, our agent holding `field_1`
  versus holding `learner_0`, against identical opposition, and that is exactly
  what the deck slot should optimize. Do not quote the absolute number as an
  expected ladder result.
- **Always allocate deck-gate games by ladder share (`--share-budget`).** The
  field is wildly unbalanced: `field_0` alone is 46.6% of appearances, so a
  share-weighted score is dominated by a handful of matchups while uniform
  allocation spends most of its games where they barely move the answer. This
  is not theoretical. Under uniform allocation `learner_0` swept `field_0` 6-0
  and looked 16 points better overall; at 61 games the same matchup was 34-27
  and the entire advantage evaporated. Six games decided nothing.
- **The gauntlet screens, the round robin decides.** Two distinct schedule
  asymmetries hit gauntlet mode, and they push in opposite directions. A panel
  member accumulates results from serving as an opponent against the whole
  candidate field (weaker on average), which inflates it, and `credit_row` fixes
  this by crediting the candidate side only. But a panel member also *skips its
  own matchup*, so it never faces itself while every other candidate must; that
  residual bias is not fixable within the mode. Only the symmetric round robin
  settles a pick. On the first clean run the screen ranked period 9 first and
  period 4 fourth; the runoff put period 4 first at 67.2% (Wilson 0.588) versus
  0.431 for the runner-up. The screen still earned its keep by surfacing
  periods 5, 8, 9, 11, which the original 5-checkpoint round robin never
  evaluated.

## Submission packaging

Two packagers, for two different agents.

**`build_bcsearch_submission.py`, a checkpoint behind the proven search.**
This is the path that matters. It ships `foundation/search_shell_main.py`
unmodified as `main.py` (refusing to build if its md5 is not the one that
scored), the rl_osfp encoders, the checkpoint, and `deck.csv`. The one
incompatibility is handled outside the shell: rl_osfp's `forward` returns
`(option, count, value)` where the shell unpacks two values, so
`foundation/nn_infer_adapter.py` ships as `nn_infer.py` and trims the tuple,
with the real implementation alongside as `nn_infer_osfp.py`.

Always follow it with **`verify_bcsearch_submission.py`**. The shell swallows a
broken net inside `_net_scores` and searches on heuristic priors instead, so a
dead checkpoint still produces a well-formed archive that plays legal games at
normal latency. The verifier wraps `_net_scores` and asserts that calls actually
return priors. Measured, both current archives pass at 100%:

| archive | ms per net call | net calls/decision |
|---|---:|---:|
| `grpo_tech_grim` (160d/5L, SEQ 53) | 10.4 | 3.5 |
| `ppo_bcsearch` (192d/6L, SEQ 93) | 23.4 | 4.0 |

Note the shell calls the net **once per determinized world at the root**, not
per node expansion. It searches on heuristic priors below the root. So a
heavier network costs per *move*, not per simulation, which is why 192d/6L is
affordable here (~8.5% of a 1.1 s move) despite being 2.25x slower per call.

**`build_submission.py`, the search-free agent.** Copies `agent_main.py` to
`main.py`, the chosen `.npz`, both feature encoders, `nn_infer.py`, and the
whole `cg/` directory, and writes `deck.csv`. Verified by `verify_submission.py`.
Search-free agents have scored 480.0 and 591.1; this path is for diagnostics,
not for shipping.

The deployed inference path is **numpy** (`foundation/nn_infer.py`), a
hand-written reimplementation of the torch `ActorCritic`. The dangerous failure
is silent: a well-formed archive whose agent never raises but plays a *different*
policy than the checkpoint that was selected. `build_submission.py` only checks
filenames, so always follow it with `verify_submission.py`, which unpacks the
archive, imports the packaged `main.py` as the Kaggle runner would, and asserts
greedy-action agreement with the torch checkpoint (floor 98%), zero
engine-rejected actions, and acceptable per-decision latency.

Checkpoints carry `_meta = [d_model, layers, heads, d_ff, MAX_COUNT, 2]`; the
trailing `2` is the format version and both loaders reject anything else.

## Retraining the prior does NOT help (measured 2026-08-05, 300 games)

The cleanest experiment this project has run. Same frozen shell, same Tech-Grim
deck, only `model.npz` differs. The new prior was rebuilt from scratch on
937,178 Elo-1000+ Tech-Grim decisions spanning Jul 24 to Aug 1 (the 972 used
Jul 25-30), scored 76.0% top-1 on a clean Aug 2 temporal holdout, then took 5
conservative GRPO iterations against a 20-deck meta mined from 9,104 Aug 1-2
episodes.

| | record | win rate | Wilson 95% |
|---|---|---:|---|
| retrained prior | 151-149 | 50.3% | [0.447, 0.560] |
| the 972 archive | 149-151 | 49.7% | [0.440, 0.553] |

Ratings +1.2 against -1.2 over 300 decided games with zero draws. Flat.

Watch the trajectory, because it is the lesson: 55.9% at 59 games, 55.4% at 121,
50.3% at 300. Early reads regress. Do not act on anything under ~300 games here.

Three things this rules out:

- **Fresh data does not help.** 40% more decisions on a more current meta: flat.
- **GRPO as configured does nothing.** It moved the weights **0.0935%** globally
  and its KL stayed near 1e-4, roughly 7x smaller than the 972 run's 0.0015. By
  iteration 3 only **2 of 8 groups** were active, because a group whose 6 games
  all win or all lose standardizes to zero advantage and is discarded. Fixing
  GRPO means fixing group construction, not the learning rate.
- **The 160d/5L prior on ~1M Elo-1000 decisions is saturated.** More of the same
  data will not move it. Submitted anyway as ref 55264582, since the gate
  measures against our own stale archive rather than the live field.

## Both shell fixes were gated and both are dead (2026-08-05/06)

The two bugs below are real. Fixing either one does not help, and one hurts.
Both were gated with `--budget 0` so each side used its own allocation, which is
what Kaggle does.

| change | archive | result vs the champion |
|---|---|---|
| horizon 160 + cap 3.0 s | `submission_972model_tunedbudget` | **8-15 (34.8%)** over 23 games |
| 20-archetype opponent model | `submission_972model_metaonly` | **30-30 (50.0%)** over 60 games, Wilson [0.377, 0.623] |

The opponent-model archive is byte-identical to the champion except for
`main.py`, and `main.py` differs only in the `META_DECKS`/`META_WEIGHT` tables.
So that 30-30 is as clean an isolation as this project has ever run, and the
answer is nothing.

Two things worth keeping from it:

- **The budget fix losing is informative.** Giving the same search 2.7x the
  think time made it worse. That fits the pattern already recorded below, where a
  learned value head and an oversized lethality term both made search worse:
  the static evaluator is the binding constraint, and deeper search converges
  harder onto its bias. Spend effort on the evaluator, not on the clock.
- **A mirror gate under-tests an opponent model.** Both sides piloted Tech-Grim,
  which the old July table already matched at Jaccard 0.846, so the fix had
  almost nothing to correct. The decks it actually helps with are the ones it
  misses: `field_9` at 0.143, `field_16` at 0.188, `field_4` at 0.277, all of
  which go to 1.000 under the new table. Testing the fix properly needs an
  opponent piloting one of those, not a mirror. That test has not been run, and
  given the 30-30 it is low priority.

`foundation/search_shell_meta_only.py` is the isolation build, kept because it is
the only clean single-variable shell we have.

## The two shell bugs themselves, for reference

### 1. The agent uses 13.9% of its thinking time

`remainingOverageTime` is 600 s. Across 888 real cabt games the median game is
**84 decisions per side** (p90 104, max 140). Two under-spends stack in `_budget`:

```python
moves_left = max(60, 300 - _GAME.calls)          # assumes ~300 decisions/side
cap = float(os.environ.get("PTCG_MAX_BUDGET", "1.1"))   # then clips the rest
```

The horizon is ~3.5x too long, which shrinks `share`, and the 1.1 s cap clips
what survives. Measured result: **83.6 s spent of 600 s**, about 516 s discarded
every game. The comment above `_budget` claims the cap was raised to target
300-400 s per game; with the cap at 1.1 that never landed.

`foundation/search_shell_tuned.py` changes exactly two constants (horizon 160,
cap 3.0): 209 s on a typical game, and 456 s with 144 s still spare on a
200-move game, longer than any of the 888 observed. Search strength scales hard
with think time here (the same agent went 1-7 at 0.1 s/move and 10-0 at 1.1).

**Caveat that must be checked before trusting it:** `_search_move` also carries
`max_nodes=20000`, and the loop exits on `live_nodes < max_nodes` as well as the
deadline. At 0.3 s the search reaches ~2,100 nodes, so ~21,000 at 3.0 s. The node
cap probably binds around 2.8 s and would silently cancel part of the gain.
Measure nodes per move at the tuned budget before concluding anything.

### 2. The opponent model covers 54.2% of the current field

The shell hardcodes 12 archetypes mined from 2,091 July replays. The current
field has 20. By weight, only 54.2% of it can be represented; the rest matches
nothing closer than these:

| current deck | field weight | best jaccard in the shipped shell |
|---|---:|---:|
| Teal Mask Ogerpon (`field_1`) | 1,229 | 0.143 |
| Mega Lopunny, Majkel's main (`field_4`) | 631 | 0.277 |
| `field_6` | 379 | 0.263 |
| `field_7` | 370 | 0.154 |
| James Cox's list (`field_9`) | 261 | 0.143 |

`META_DECKS` is what determinizes the opponent's hidden cards, so against ~46%
of opponents the search samples worlds from the wrong decklist and optimises
against a fantasy opponent. `foundation/search_shell_tuned_meta.py` carries both
fixes with the 20 current archetypes.

Gate all three (stock, tuned, tuned+meta) with `--budget 0`, which leaves
`PTCG_MAX_BUDGET` unset so each side uses its own allocation. That is what
Kaggle does and the only condition where the change is visible at all.

## The BC pipeline is revived and lives in `bc_train/` (2026-08-06)

`train_bc.py` was in quarantine and its imports were split across three places
with **two incompatible encoders**, which is the trap here. `bc_train/` stages
the exact set the champion ships, so a model trained there is loadable by the
frozen shell:

| file | source | why |
|---|---|---|
| `train_bc.py`, `model.py` | quarantine `track1_search/train/` | the trainer |
| `nn_features.py` | **the champion archive** | `MAX_OPT 24`, so `SEQ 53` |
| `nn_features_rich.py` | champion archive (same bytes as quarantine) | deck-aware |
| `nn_infer.py` | champion archive | the numpy export target |

**Do not use `foundation/nn_features*.py` for BC.** Those are the rl_osfp
encoders at `MAX_OPT 64`, so `SEQ 93`. The champion is `SEQ 53`. Training against
the wrong one produces a model that loads without error and plays a different
policy, which is the exact silent failure this repo keeps hitting.

Checkpoints carry `_meta = [160, 5, 5, 320]` and 75 tensors, matching the
champion's `model.npz` byte-for-byte in shape.

### Why the deck specialist has to be a fine-tune, not a fresh train

Tech-Grim is 36% of the mined field, so it dominates the corpus. Every deck that
actually wins is rare precisely because only a few strong pilots run it:

| deck | expert win rate | ladder games | decisions we have |
|---|---:|---:|---:|
| Tech-Grim (`field_0`/`field_2`) | 48.5% | 4,611 | **937,178** |
| `field_9` (James Cox, rank 8) | 59.0% | 261 | 117,706 |
| `field_4` (Majkel1337's main) | 59.6% | 631 | 71,046 |
| `field_16` (Majkel's side deck) | 75.0% | 136 | **6,097** |

So we ship the deck with the worst win rate because it is the only one with
enough data to train on directly. The answer is a general trunk plus a fine-tune,
not a fresh train on 6k to 118k decisions.

Measured, both stages early-stopped and both gates passed:

| stage | data | held-out top-1 | raw baseline | value MAE |
|---|---:|---:|---:|---:|
| general base | 720,000 decisions, all decks, Elo-weighted | 70.5% | 34.6% | 0.615 |
| `field_9` fine-tune | 67,491 decisions, `--init` from the base, `--lr 2e-4` | **74.8%** | 41.6% | 0.487 |

The fine-tune holdout is the **Aug 2** `field_9` shard, a day neither model
trained on, so it is a real temporal holdout rather than a random split. For
scale, the Tech-Grim specialist that measured flat against the champion scored
76.0% on its own holdout, so 74.8% is the same quality band.

```bash
# general trunk
OMP_NUM_THREADS=6 PYTHONPATH=bc_train .venv/bin/python bc_train/train_bc.py \
  --data data/bc_general_train --val-data data/bc_general_holdout \
  --max-per-shard 90000 --dim 160 --layers 5 --heads 5 --features rich \
  --elo-weight 0.5 --epochs 12 --patience 4 --out data/model_bc_general.npz

# deck specialist, warm-started from the trunk
OMP_NUM_THREADS=6 PYTHONPATH=bc_train .venv/bin/python bc_train/train_bc.py \
  --data data/bc_field9_ft --val-data data/bc_field9_ft_holdout \
  --init data/model_bc_general.npz --lr 2e-4 \
  --dim 160 --layers 5 --heads 5 --features rich --elo-weight 0.5 \
  --epochs 10 --patience 3 --out data/model_bc_field9.npz
```

Package by copying the champion archive and swapping **only** `model.npz` and
`deck.csv`, then diff the tarballs to prove nothing else moved. Building from a
staging directory that was reused for another build silently reintroduced the
meta-only `main.py` once; the diff caught it.

## Resource limits, enforced

Two WSL VM terminations were caused by launching jobs whose memory requirement
was never computed. `train_bc.py` holds the corpus as CPU tensors at a measured
**7,396 bytes per decision**:

    1,150,331 decisions ->  7.9 GiB   fine
    1,920,408 decisions -> 13.2 GiB   killed the VM

Checking *free* RAM is not a safety check. Run `tools/resource_guard.py` before
any heavy job; it computes the requirement and refuses. Caps on this machine:
**8.0 GiB** of dataset tensors, **5 workers** of 14 threads, **6 GiB** free to
start.

## Candidates currently built, and what each one isolates

All four ship the frozen shell, so `main.py` is constant across them. Gate them
together at 1.1 s/move with `grpo_tech_grim` in the same run.

| archive | model | deck | the question it answers |
|---|---|---|---|
| `submission_ppo_bcsearch` | v3 PPO p180 | Tech-Grim | do the RL priors beat the BC priors? |
| `submission_shell_nonet_techgrim` | **none** | Tech-Grim | are the PPO priors better or worse than *no* priors? |
| `submission_grpo_deck_field16` | GRPO (the 972 net) | `field_16` | does Majkel1337's deck beat Tech-Grim under our pilot? |
| `submission_grpo_deck_field4` | GRPO (the 972 net) | `field_4` | does the best non-elite meta deck beat Tech-Grim? |

The no-net control matters because the shell runs happily on heuristic priors,
and an earlier isolation pair measured net priors at 819.8 against no-net at
774.6. If `ppo_bcsearch` loses to `shell_nonet`, the v3 priors are worse than
nothing and the answer is to fix the training, not the search.

## On the competition discussion and the wider field

The Kaggle discussion forum renders client side, so it cannot be read with a
plain fetch and none of it is quoted here. Only two weak signals came back from
open search: leaderboard agent names on the board suggest others are also on
RL plus MCTS, and expectiminimax is reported to struggle with the branching
factor. Neither is worth acting on.

The strong evidence is local and empirical, so prefer it: `data/fresh/replays/`
holds 9,174 real ladder episodes with `info.TeamNames`, `rewards`, and both
decklists in `steps[1]`, and `data/fresh/leaderboard/` holds the score table to
join against. `tools/top_decks.py` does that join. Top of the board on
2026-08-04 was Majkel1337 at 1,297.9 against our 810.8.

## Which algorithm: BC then conservative GRPO, not PPO from scratch

Every score above 850 this project has produced came from **BC-initialised**
weights. Nothing started from random init has broken 600.

| approach | search | ladder |
|---|---|---|
| BC + conservative GRPO | yes | **972.0**, 928.5, 918.1, 910.7, 903.2 |
| BC | yes | 967.1, 917.6, 873.5 |
| BC, 192d, more data | yes | 848.9 |
| BC | no | 591.1 |
| PPO self-play from random | no | 480.0 |
| PPO self-play priors + the deleted PUCT | yes | 592.0 |
| PPO self-play priors + the frozen shell | yes | ~316 (harness estimate) |

The v3 PPO run was **not** mistuned: `approx_kl` median 0.0133 against a 0.04
target, clip fraction 0.128, value loss 0.49 down to 0.13, entropy 0.78 down to
0.55. It optimised correctly. The learner win rate was flat at 72.4 / 74.0 /
74.0% across 200 periods, and the runoff could not separate period 180 from 200.
The deficit is the starting point, not the optimiser: BC begins from 1.33M
decisions by 1000+ Elo humans, and GRPO then refines with a KL around 0.0015 so
it sharpens that prior instead of overwriting it.

The full pipeline that produced 972 survives in quarantine and is revivable:
`track1_search/train/train_bc.py`, `train/ingest_episodes.py`,
`track5_grpo/train_grpo.py`, and `track6_controlled/` (the deck-specialist arms).

## Deck gates say Tech-Grim holds, but they cannot say the deck is best

Round 2, 200 games at 1.1 s/move, everything on the frozen shell:

| archive | rating | record | win rate |
|---|---:|---|---:|
| `grpo_tech_grim` | +202.6 | 64-16 | 80.0% |
| `grpo` + `field_16` | +105.8 | 53-27 | 66.2% |
| `grpo` + `field_4` | +49.0 | 46-34 | 57.5% |
| `shell_nonet` (no model at all) | -164.8 | 20-60 | 25.0% |
| `ppo_bcsearch` | -192.6 | 17-63 | 21.2% |

Two readings, and the second is the one that matters:

- The GRPO net priors are worth a great deal. Stripping the model entirely drops
  the same archive to 25%.
- **A deck swap alone is not a deck test.** The 972 recipe trains the BC prior on
  the *exact deck it ships with* (`track6_controlled`, "controlled deck-specialist
  arms"). Swapping `deck.csv` under a Tech-Grim-specialised prior breaks that
  pairing, so `field_16` and `field_4` losing means "do not swap the deck without
  retraining the prior", **not** "these decks are worse". The deck question is
  open, not closed.

Bradley-Terry ratings are only comparable *within* one run, since they are
centred on that run's pool. Do not push round 2 ratings through the round 1
calibration fit and quote ladder numbers.

## What the top of the board actually plays (mined 2026-08-04)

From 9,104 episodes joined to the leaderboard, via `tools/top_decks.py` and a
per-pilot scan. Decks map onto our pool at Jaccard 1.000, so these are exact.

| pilot | games | win rate | avg steps | deck |
|---|---:|---:|---:|---|
| James Cox & Henry Chao (rank 8) | 258 | 59.3% | **137** | `field_9`, and only this one |
| Majkel1337 (rank 1) | 532 | 60% | 183 | `field_4`, Mega Lopunny |
| " | 503 | 56% | 127 | `field_1`, Teal Mask Ogerpon |
| " | 136 | **75%** | 132 | `learner_0`, Mega Lucario |

Two things worth acting on:

- **`learner_0` is Majkel's rarest list, not his main.** Our pool builder picked
  it on Wilson lower bound and thereby selected his 136-game side deck over his
  532-game main. A high win rate on a small, self-selected sample is not the same
  as a deck the best player relies on.
- **`field_9` closes fastest**, 137 steps against 153 to 176 for the other top
  pilots, and its pilot never deviates from it. Short games suit us: fewer
  decisions means fewer chances for a weaker policy to drift off the line.

## What to do next, in order

0. **Manage the active slots toward the deadline.** The board ranks the best of
   our *active* submissions, not our best ever, so the number that counts is the
   one showing on 2026-08-16. Until about 08-14, uploads are cheap and worth
   spending: 5 a day, and identical bytes have scored 972.1, 942.3 and 911.9, so
   draws differ by ~30 points for free. In the last two days, stop uploading once
   a good draw is active. Run `tools/ladder_status.py` before every upload to see
   what it will retire. None of this makes the agent stronger, so it runs
   alongside the real work below rather than replacing it.
1. **Do not re-derive a search.** We have one that scored 972 and one that
   scored 405, and the gap is not recoverable by tuning constants.
2. **Stop tuning the shell's constants.** Both measured changes lost. See the
   shell-fix section; the budget change lost 8-15 and the opponent-model change
   lost over 60 games with byte-identical model and deck.
3. **Revive BC plus GRPO on fresh data.** Re-ingest the Aug 1-2 replays with Elo
   weighting, train a BC prior **specialised on the deck we intend to ship**,
   GRPO-refine it, and gate every iteration against the anchors. This is the only
   path that has ever produced a score above 850. Note the ceiling: the Tech-Grim
   version of exactly this returned 151-149 against the champion, so expect a
   deck change to be doing the work if anything is.
4. **Only then re-test the deck**, with a prior trained for that deck. `field_9`
   is the strongest target on the evidence above. `data/bc_field_9_probe` holds
   117,706 decisions, which is 0.8 GiB and inside the guard.
5. Fixing the v3 PPO deck mismatch (`tools/make_learner_pool.py` plus
   `train.py --resume`) is cheap and would confirm the diagnosis, but its ceiling
   is low. Treat it as a diagnostic, not a route to a shipping agent.

**Sizing note for step 3.** `data/bc_general_train` is 1,520,408 decisions, which
is 10.5 GiB of CPU tensors and over the 8.0 GiB cap. Pass
`--max-per-shard 129047` to bring it to about 1.16M. `tools/resource_guard.py
--data data/bc_general_train` prints exactly this and refuses otherwise.

## Conventions

- One file, one job; module docstrings explain *why*, not what.
- Failures are surfaced, never swallowed: `arena.py` returns engine faults in
  `GameResult.error` so they land in metrics rather than silently biasing a
  win rate.
- Guard resources explicitly. `train.py::resource_guard` refuses configurations
  that would exhaust RAM or disk before they start.
- Evaluation scripts write a JSON report next to the checkpoints and print a
  table; they warn loudly rather than quietly picking when criteria disagree.
