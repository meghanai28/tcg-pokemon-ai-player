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

Rank **313 / 6,224**, best score **972.0**, top of board 1,297.9 (Majkel1337).
Deadline
**2026-08-16**. A second competition, `pokemon-tcg-ai-battle-challenge-strategy`
($240,000, deadline 2026-09-13), is entered but unranked.

Three facts from submission history that should govern decisions:

- **Search is worth roughly 300 ladder points.** The one search-free submission
  ever made (`submission_purebc_scaled`, "192d/6L, 291k Elo-1000 decisions, NO
  search, argmax policy") scored **591.1**, the worst result on record. Every
  search-based submission scored **873 to 972**.
- **Ladder score carries roughly ±160 points, and it drifts downward.** One
  archive (md5 `4f6b66a1ecb8ee13f9d67e5d97a3bdeb`) was submitted three times
  and scored **972.0 (08-02) → 911.9 (08-03) → 810.8 (08-04)**. Identical bytes,
  a 161-point spread, monotonically falling. An earlier note here put the noise
  at ±75 from a single pair; three points show it is more than twice that, and
  the trend suggests the board is strengthening around a fixed agent rather than
  the score merely being noisy. Two consequences: never compare a submission to
  a score from a *different day*, and never conclude anything from one
  submission. A local gate over hundreds of games is more trustworthy than any
  single ladder result.
- **Deck choice moves the score as much as the model.** 967.1 vs 917.6 was the
  same model with a different deck; 972.0 vs 719.2 likewise. Re-gate the deck
  whenever the pilot changes.

`learner_0` is **Majkel1337's list, the rank-1 player** (1,297.9 as of
2026-08-04). Its 75% ladder win rate conflates deck and pilot, but it is the top
player's choice, not noise. Measured under a *search* pilot it still only tied
for last among candidates, so gate it rather than assuming either way.

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

## What to do next, in order

1. **Finish the deck gate.** It is the highest-value slot: deck swaps have moved
   the ladder by 250 points (972.0 vs 719.2) on an unchanged model, and our
   shipped deck is a 48.5% list in the current meta.
2. **Then retrain the policy on whichever deck wins**, using
   `tools/make_learner_pool.py` plus `train.py --resume`. Training the policy on
   a deck it has never piloted is the concrete, measured reason
   `submission_ppo_bcsearch` is weak. Do not do this before step 1 or the
   specialisation targets the wrong list.
3. **Re-gate, including the anchors.** Never conclude from a comparison that
   contains only our own work.

Do not re-derive a search. We have one that scored 972 and one that scored 405,
and the difference is not recoverable by tuning constants.

## Conventions

- One file, one job; module docstrings explain *why*, not what.
- Failures are surfaced, never swallowed: `arena.py` returns engine faults in
  `GameResult.error` so they land in metrics rather than silently biasing a
  win rate.
- Guard resources explicitly. `train.py::resource_guard` refuses configurations
  that would exhaust RAM or disk before they start.
- Evaluation scripts write a JSON report next to the checkpoints and print a
  table; they warn loudly rather than quietly picking when criteria disagree.
