# CLAUDE.md

Agent for the Kaggle [`pokemon-tcg-ai-battle`](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
competition. A submission is a tarball containing `main.py` exposing
`agent(observation) -> list[int]`: the runner calls it once with `select == None`
to receive a 60-card deck, then once per in-game selection callback to receive
the indices of the chosen options.

## Orientation

**`rl_osfp/` is the live RL track.** `bc_train/`, `foundation/`, `harness/` and
`tools/` are also live support code. Numbered `track*` directories are history.

The project was reset on 2026-08-03. Every earlier track (`track1_search`,
`track2_dmc`, `track5_grpo`, `track6_controlled`, `track8_bc800`,
`track9_awr_grpo`, ...) was moved to `.reset-quarantine-20260803/` and is
retained only as evidence. `rl_osfp` can start randomly, but the production
recipe is BC-initialised PPO with an explicit frozen-policy anchor. Calling that
replay-free would be false: the warm-start checkpoint was trained on replay
action labels, even though every subsequent update is online RL.

- `README.md` was corrected on 2026-08-08 and is current. It is the short
  version; this file is the long one.
- `tools/divergence.py`, `tools/run_local.py`, `tools/autopsy.py` were deleted:
  they imported paths under `track1_search/` and had been broken since the reset.
  `foundation/model.py` (the old `TCGNet`) was deleted as dead code. All are
  recoverable from git history.

Use `.venv/bin/python` (3.12, torch 2.13, numpy 2.5). Run everything from the
repo root as a module (`.venv/bin/python -m rl_osfp.<name>`); the packages are
not installed.

## STOP: 2026-08-08 RL correctness audit (supersedes older conclusions below)

The 1127-period Lucario run is **not valid evidence about scaled PPO**. The
older chronological notes remain below as an audit trail, but their claims that
async staleness was safe and that the run cleanly tested BC-init RL are wrong.

Run `PTCG_MAX_OPT=24 .venv/bin/python -m rl_osfp.audit_run <run-dir> --strict`
before resuming, scaling or packaging any checkpoint. The saved report for the
old run is `artifacts/rl_lucario_failure_audit.json`.

| failure | measured consequence | correction |
|---|---|---|
| cross-update async rollout queue | **216,246 / 252,448 games (85.7%)** came from policies one to three updates stale | PPO now has a hard rollout/update barrier; `--async-rollout` is deprecated and cannot create stale data |
| completed-result queue was unbounded | results accumulated while the learner updated/evaluated | unused `run_continuous` async code was deleted; reintroduce overlap only with V-trace or another actor-lag correction |
| PPO recomputed raw logits | any rollout `temperature != 1` had the wrong importance ratio | current and reference logits are divided by the rollout temperature during updates; a regression test starts at ratio exactly one at temperature 0.63 |
| KL early-stop used signed `mean(old_logp-new_logp)` | positive and negative changes could cancel while clipping was already large | uses non-negative `ratio - 1 - log(ratio)` estimator |
| resume loop ignored `first_period` | resuming restarted at period 1 and overwrote checkpoints/metrics | loop starts at saved period + 1; RNG, archive age and optimizer state are restored |
| resume accepted a changed experiment | pool contents, encoder or objective could silently move | pool SHA, MAX_OPT, BC anchor and all objective/data-distribution arguments are hard guards |
| skipped evaluation could still trigger max-wait archive | an unevaluated candidate entered the league | max-wait archive is legal only on an evaluation period |
| no period-zero opponent | first update could erase the only known-good prior | the frozen BC starting policy is permanent `league_000` |
| long run used constant `2e-4`, four epochs | 151,629 equally aggressive optimizer steps; mean reference KL reached 0.185 | default is two epochs, `7e-5`, non-negative KL stop, reference anchor, and LR decays to 10% over the declared run |
| train/package encoder drift | future MAX_OPT=64 training could be served by the frozen MAX_OPT=24 shell | shipping checkpoints require `PTCG_MAX_OPT=24`; builder copies the literal BC-training encoder ABI |
| ladder tool fetched only 20 submissions and treated newest as board score | it forgot the historical 972 and misreported the live score | fetch 200; published latest two are active; board is best active score |

### What the public RL result actually implies

Discussion **717697**, not 709160, is the RL thread. The author reports a
sub-2M-parameter pure self-play agent, roughly 45 games/s, and **3-5 million
games** to approach rule bots. They also say representation and a refined
curriculum were decisive, that their curriculum covered about 250 unique cards,
and that indiscriminate self-play was not useful. Other replies report the
common pure-RL plateau near 800 and improvements from alternating a previous
best checkpoint with a strong public agent.

Our invalid Lucario run had **252,448 games (8.4% of 3M)** and a 121-card deck
pool. It was simultaneously too small, too narrow and mostly stale. Therefore:

- the claim “we scaled PPO and it failed” is false;
- the claim “just make the same run longer” is also unsupported;
- the controlled experiment is BC-init **on-policy PPO + a frozen anchor + a
  diverse historical/opponent league**, with one shippable deck trained as the
  learner and a current broad field;
- GRPO is not the default scale arm. In a binary terminal-reward game, group
  normalisation often erases the gradient when every rollout in a group has the
  same outcome. A learned value baseline makes PPO more sample-efficient here.
  A later GRPO arm is useful only after PPO passes the correctness and transfer
  gates, using the same synchronized collector.

The remaining bottleneck may still be representation. The 53-token encoder has
board identities, our hand, legal options and a deck anchor, but no discard
identities, stadium identity, attachment identities, action history or opponent
archetype posterior. Do not spend 3M games until a short run proves: zero lag,
zero errors, improving held-out league play, and no regression when the model is
put behind the frozen search shell.

## HANDOFF / REVIEW TARGETS from the 2026-08-07 session

Written for a reviewer whose job is to **find bugs and bad inferences**, not to
be reassured. Everything here is a measurement with an evidence file, or is
flagged as an inference. Where the reasoning may be wrong, it says so.

### What shipped

`artifacts/submission_dunsparce_scratch.tar.gz`, ref **55339282**. Frozen shell
(`main.py` md5 `e54bc659…`, unmodified) + a **new from-scratch BC prior** + a
**Dunsparce** `deck.csv`. Verified at 100% prior rate, 10.4 ms/net call, zero
invalid actions. Gate: **155-85 = 64.6%** over 240 games at 1.1 s/move,
Wilson [0.583, 0.704], zero errors (`harness/gate_dunsparce_confirm.json`).

**Ladder: stalled at 811, performance rating ~784 against the champion's ~890.
The gate has not transferred.** See "THE OPEN PROBLEM" below; it is the most
important thing to review.

### The argument, so it can be attacked

1. Every previous experiment moved **either** the prior **or** the deck. All
   gated 39-53%.
2. `tools/score_by_deck.py` measured why: the champion scores **79.5%** top-1 on
   Tech-Grim and **~50%** (chance) on every other deck. It is a Tech-Grim
   specialist, not a general prior.
3. So deck and prior are not independent axes: a deck swap under the champion
   measures its blind spot, and a prior change on Tech-Grim is confined to a
   deck that is 47.9% field-weighted.
4. Therefore move both: a deck-**balanced** from-scratch prior, then gate several
   decks under it with a same-prior control.
5. Result: Dunsparce 58.8% (screen) → 64.6% (confirm). Control (same prior,
   Tech-Grim deck) 42.5%, i.e. *worse* than the champion, so **the gain is the
   deck, not a better network**.

**Attack surface:** step 5's control is what licenses "the gain is the deck". If
the control is wrong, the whole story is wrong.

### Evidence index

| claim | number | file |
|---|---|---|
| Dunsparce beats champion, shipping budget | 155-85 = 64.6% | `harness/gate_dunsparce_confirm.json` |
| deck screen with control | dun 58.8, luc 48.8, techgrim 42.5 | `harness/gate_scratch_decks.json` |
| PUCT prior floor 0.10 | 29-41 = 41.4% | `harness/gate_puct_variants.json` |
| PUCT `C_PUCT` 2.5 | 38-32 = 54.3% (tie) | same |
| RL checkpoints vs shipped archive | control 81.7%, best RL 75.0% | `harness/gate_rl_lucario.json` |
| deck census, 6 days, Elo>=1000 | see census section | `harness/meta/deck_census.json` |
| field-weighted WR | Lucario 63.9, Dunsparce 59.7, Tech-Grim 47.9 | `harness/meta/matchups.json` |
| RL throughput sync -> async | 15.2 -> 29.0 games/s | `harness/rl_throughput_probe.log` |
| RL run | 1127 periods, 252,448 games, 0 invalid | `rl_osfp/run_lucario/metrics.json` |

### Code added or changed, and where the bugs probably are

New: `bc_train/ingest_episodes.py`, `tools/deck_census.py`,
`tools/balance_corpus.py`, `tools/score_by_deck.py`, `tools/matchup_matrix.py`,
`tools/build_puct_shell.py`, `tools/swap_shell.py`, `tools/deck_arms.sh`.
Changed: `rl_osfp/rollout.py` (`run_continuous`), `rl_osfp/train.py`
(`--async-rollout`, `--eval-every`).

Specific suspicions, ordered by how much they would matter:

1. **`run_continuous` race.** `make_task` runs on the pool's *callback thread*
   as well as the main thread. It is lock-serialised, but the closure in
   `train.py` reads `league`, which the **main thread mutates** when archiving,
   and `choose_league_opponent` indexes into it. The GIL probably makes this
   benign; it is still an unsynchronised read/write.
2. **`--eval-every` vs archiving.** `all_pass = run_eval`, so a period without
   evaluation cannot pass -- but `since_archive` still increments and
   `--archive-max-wait` may force an archive **with no evaluation having run**.
   Unintended; check whether it fires and whether it matters.
3. **Abandoned generator.** `train.py` breaks out mid-period on purpose, leaving
   games in flight while callbacks keep submitting. Verify nothing leaks at pool
   teardown and that no task references a deleted `behaviour_path`.
4. **RNG divergence.** Async mode skips the eager `tasks` list so the rng is not
   double-consumed. Confirm sync and async are otherwise equivalent under a seed.
5. **`run_continuous` error path.** A worker exception pushes `("err", exc)` and
   the consumer raises, but that slot is never resubmitted. Safe only because we
   abort; a caller that swallowed the exception would silently shrink the pool.
6. **`balance_corpus.py`** keeps rows with probability `allowed/total`, so
   realised per-deck counts are approximate. Check they landed near the caps.
7. **`matchup_matrix.py` symmetrisation** counts each ordered pair into both
   directions; mirrors (A vs A) are therefore counted twice and pinned at 50%.
   Confirm that does not distort `field WR`.
8. **`deck_census.py` decision counting** is meant to match
   `ingest_episodes.py`'s definition (ACTIVE seat, >=2 options, answered). The
   entire deck argument rests on it.

### THE OPEN PROBLEM: the gate did not transfer to the ladder

The 240-game confirm says Dunsparce beats the champion 64.6%; the ladder puts it
~105 points *below* the champion. **Status: genuinely unresolved.** An earlier
version of this section argued explanation (b) was more likely; that argument
leaned on the 40-game field screens below, which are too underpowered to carry
it. What is actually known:

- **Solid**: the 240-game confirm at 1.1 s/move, Wilson [0.583, 0.704].
- **Weak**: perf 777 vs 904 over 46 and 54 episodes, where identical champion
  bytes have spanned 719-919 in this file's own records.
- **Worthless for ranking**: every 40-game field screen.

Two candidate explanations remain live:

**(a) Bracket luck.** Opened 6-4, landed against a mean opponent of 727 versus
the champion's 808, and 43 episodes is tiny -- identical champion bytes have
spanned perf 719-919 in this file's own records.

**(b) The gate opponent is not the field.** *Our shipped `deck_tech_grim.csv` is
not the deck the field plays.* It is Jaccard 0.846 to the popular Tech-Grim list
and has **zero exact appearances in six days of dumps**. The champion archive
pilots our variant. Against the field's real Tech-Grim, under the same new
prior, Dunsparce goes **15-25 = 37.5%** -- losing badly to the single most
common opponent on the ladder, while beating everything else.

If (b) holds, the confirm gate measured a matchup that barely exists in the wild
and **`harness/anchors/grpo_tech_grim_972_912_811.tar.gz` has been the wrong
yardstick all along**, which would recontextualise several older results.

#### The full field screen, and the share-weighting mistake in it

Both candidates against six field decks, same prior on every seat, 240 games
each (`harness/gate_field_dunsparce.json`, `harness/gate_field_lucario.json`):

| opponent | share | Dunsparce | Lucario |
|---|---:|---:|---:|
| **techgrim_pop** | **57.8%** | **37.5%** | **42.5%** |
| alakazam | 18.0% | 50.0% | 45.0% |
| bugcatch | 8.4% | 82.5% | 57.5% |
| dunsparce_buneary | 7.7% | 57.5% | 92.5% |
| dwebble | 4.5% | 62.5% | 42.5% |
| dreepy | 3.6% | 70.0% | 67.5% |
| **unweighted** | | 60.0% | 57.9% |
| **share-weighted** | | **47.4%** | **49.0%** |

**These screens are UNDERPOWERED and rank nothing. Do not cite them.** Two
faults, and the second is fatal:

1. *Uniform allocation*, which this file already warns against: forty games
   against a 3.6% deck count the same as forty against a 57.8% deck. Weighting
   afterwards flips the order (unweighted Dunsparce 60.0 > Lucario 57.9;
   weighted Lucario 49.0 > Dunsparce 47.4).
2. *Forty games per pair is Wilson plus or minus 15%.* A third candidate,
   `techgrim_pop` (new prior on the field's own Tech-Grim list), scored 47.8%
   share-weighted. All three sit between 47 and 49 with intervals that swallow
   each other and 50%. **The screens do not separate the decks.**

**Worked example of how badly, kept as a warning.** The
Dunsparce-vs-`techgrim_pop` matchup was measured twice, once in each run:
**15-25 one way and 25-15 the other**, same matchup, 40 games each. On first
sight that looked like a load-order bug -- in both runs the archive listed first
in `--archives` lost -- which would have meant the champion was handicapped in
every gate and the 64.6% confirm was inflated. It is not a bug:

- median game 48.3 s vs 45.9 s, median steps 176 vs 164, so **both archives
  searched normally in both runs**; a silently search-disabled agent collapses
  games to seconds, which is the recorded signature.
- the difference of two Binomial(40, 0.5) has SD 4.47 wins; the observed 10-win
  gap is 2.24 SD, p about 0.025. Across roughly 20 matchups run that session,
  P(at least one such gap) is about 40%. **Seeing one is expected.**

So: check the cheap variance explanation before reaching for the exciting causal
one, and treat any deck comparison under ~150 games per pair as unable to
resolve differences of a few points.

Caveat that survives regardless: every opponent in these panels is piloted by
the *new* prior, which scores 77.6% top-1 on Tech-Grim, while the real ladder's
Tech-Grim pilots average Elo 989 and a 46.2% win rate. The panel's Tech-Grim is
plausibly far stronger than the real thing.

**Caveat before acting:** in that field test the opponent runs the *new* prior
while the champion runs the *old* one, so deck and pilot are confounded.
Isolate it: same prior, our Tech-Grim variant versus the popular list.

**Also unresolved -- the ranking is non-transitive:** Dunsparce beats the
champion 64.6%, Lucario *loses* to the champion 48.8%, and Lucario beats
Dunsparce **81.7%**. Predicted in advance by `matchup_matrix.py` from human
replays (9% over 173 games). Head-to-head cannot rank decks here; only
field-weighted performance can.

### What to do next

1. **Resolve (b).** Same prior, our Tech-Grim variant vs `techgrim_pop`.
2. **Re-gate the shipped archive against a field panel, not one anchor.**
3. **Reconsider Lucario**: best field-weighted WR and beats our shipped deck
   81.7%, but loses to Tech-Grim, a third of the field. Needs the field test.
4. **Do not run more self-play RL of this form.** Three attempts, three worse
   priors. Change the objective (reward from games played *by the search using
   the net as prior*) or leave it.

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

**The board ranks our MOST RECENT submission. Not our best ever, not the best
live one.** This is the single most expensive thing to get wrong here, and two
earlier versions of this section got it wrong in two different ways. It was
pinned down on 2026-08-06 by reading `competitions_list().user_rank`, then
looking up what score sits at that rank in a full leaderboard dump:

| our rank | score at that rank | what it matched |
|---:|---:|---|
| 478 | 871.4 | newest submission, 71 episodes old |
| 458 | 876.2 | newest submission, 48 episodes old |
| 3,621 | 600.5 | newest submission, **1 episode old** |

Every submission starts at **600** and needs roughly a day of play to climb, so
**each upload resets the displayed score to about 600.** Uploading is not free
and never was. Uploading repeatedly knocks the score back down before it can
mature, which is exactly what happened on 2026-08-06: four uploads in three and
a half hours took the board from 881.8 (rank 458) to 600.0 (rank 3,621), while
an earlier submission sat live and ignored at 881.8 with a performance rating
of 919.

Two rules follow, and they are the whole submission strategy:

- **Uploading costs about a day of climb.** Only upload when the new archive is
  actually better, or when the current newest one has already matured.
- **The final standing is the last submission made before 2026-08-16 23:59, and
  how long it was given to climb.** Stop uploading about two days out, make the
  last upload the best archive available, and leave it alone.

`max(public_score)` still reports 972.0 and is meaningless: that submission was
retired on 2026-08-02. `tools/ladder_status.py` prints the newest submission's
score as the board score, prints the other two numbers only as context, and
warns when the newest is well below an older live one.

### So the "monotonic decline" in the old version of this file was not real

The old note read 972.0 to 911.9 to 810.8 on identical bytes and concluded the
board was strengthening around a fixed agent. Both halves are wrong:

- **810.8 was never a converged score.** Ref 55233305 played **three episodes**
  before two later uploads retired it. It is a truncated run, not a result.
- **The sequence is not monotonic.** The same bytes have now scored 972.1,
  911.9 and 942.3, on runs of 65, 83 and 82 episodes. Mean 942, sample standard
  deviation 30. There is no trend, just a spread.

### THE RATING PEAKS AROUND GAME 20-40 AND THEN DECAYS (measured 2026-08-07)

The single most actionable thing in this file, and it was invisible for days
because only converged scores were ever read. Rating by episode for the best
submission this project has ever had:

| after game | rating | mean opponent | win rate so far |
|---:|---:|---:|---:|
| 10 | 886.5 | 734.3 | 80.0% |
| 20 | **1001.3** | 830.8 | 80.0% |
| 30 | **1004.0** | 865.0 | 73.3% |
| 40 | 958.9 | 887.9 | 62.5% |
| 82 | 942.3 | 888.0 | 58.5% |

**We have already been over 1000, twice, transiently.** The mechanism is in the
same table: a submission starts against weak opposition, wins 80%, climbs fast,
and then matchmaking raises the field from 734 to 888 until the win rate falls
to its true ~58% and the rating settles at field-plus-edge.

Two consequences, and the second one is worth more than any model change
measured in this repo:

- **More episodes make the score WORSE, not better.** "Upload early so it has
  time to mature" is backwards; maturing is decay. Every earlier note in this
  file recommending a long climb was wrong about the direction.
- **The final standing is a snapshot at the deadline, not a converged value.**
  Uploading the best archive so that games 20 to 40 land **at** the deadline
  captures the peak instead of the decayed tail. On the champion's own numbers
  that is 1004 against 942, roughly **+60 points for free**.

### Episodes arrive in BURSTS, so "N per hour" is the wrong model (2026-08-07)

Measured off ref 55335692's episode timestamps, which is the first time the
arrival *times* were read rather than just the counts. Six episodes landed
between 23:33:23Z and 23:53:43Z, roughly one every four minutes, and then
**nothing for the next several minutes**. So the process is bursty, not a
steady drip, and a rate extrapolated from one burst is wrong in the optimistic
direction. Do not plan the deadline off a single window.

What this does establish, and it still matters:

- **A burst can deliver the first ten games in well under an hour**, so the
  "reading a draw takes about two hours" figure elsewhere in this file is an
  upper bound, not a constant. Check `ladder_status.py`, do not assume.
- **The gap between bursts is not predictable from our side**, so the safe
  deadline play is to upload with margin and re-read, rather than to compute a
  target upload time from an assumed rate.

The honest summary is that the peak window is somewhere between one and eight
hours after upload depending on how the bursts fall, and the only way to know
where a given submission sits is to poll it.

The peak is not guaranteed: it requires a hot start, and the same table for a
cold start (ref 55307993: 659.6 at game 10) never produces one. So the deadline
play is upload late, and keep slots in reserve to re-roll if the first ten games
come back badly.

| ref | g10 | g20 | g30 | g40 | final |
|---|---:|---:|---:|---:|---:|
| 55256846 champion | 886.5 | 1001.3 | 1004.0 | 958.9 | 942.3 |
| 55321164 elite-1150 | 865.2 | 820.2 | 815.0 | 836.7 | 822.1 |
| 55307993 champion | 659.6 | 664.0 | 748.1 | 794.8 | 770.8 |

### The score is set by the first ten games, and it is mostly a coin flip

Measured 2026-08-06 on **four uploads of byte-identical bytes**
(`grpo_tech_grim_972_912_811`). This is the single most important table in this
file, because it says what the ladder can and cannot measure.

| ref | first 10 games | rating after 10 | mean opponent | final | eps |
|---|---|---:|---:|---:|---:|
| 55256846 | **8-2** | **886.5** | 888.0 | **942.3** | 82 |
| 55290078 | 6-4 | 665.5 | 723.5 | 881.8 | 49 |
| 55294353 | 3-2 | 679.5 | 543.1 | 679.5 | 5 |
| 55294549 | **5-5** | **608.4** | 650.5 | **701.5** | 52 |

Same agent. 942.3 against 701.5, a **241-point** spread, and the whole thing is
already decided by game 10. The mechanism is arithmetic:

- K is ~50 per game for the first ten, so one win-instead-of-loss is worth ~100
  rating points.
- The agent's true win rate is 58.5%, so ten games have expected 5.8 wins with
  a standard deviation of **1.56 wins**, which is **~156 rating points of pure
  luck**. The observed 8-2 / 6-4 / 5-5 spread is exactly that distribution.
- Matchmaking then pairs on current rating, so the bracket you land in is the
  field you keep playing, and after game 30 K drops to ~6 and you cannot leave.

**So the ladder score is (the bracket luck bought you) + (how much better than
that bracket you are).** The second term is the only part that is about the
agent, and it is remarkably stable:

| ref | eps | its field | score | edge over field |
|---|---:|---:|---:|---:|
| 55256846 champion bytes | 82 | 888.0 | 942.3 | **+54.2** |
| 55294549 champion bytes | 52 | 650.5 | 701.5 | **+51.0** |
| 55264582 retrained prior | 75 | 814.7 | 860.3 | **+45.6** |
| 55290078 champion bytes | 49 | 723.5 | 881.8 | +158.3 (had not converged, 75.5% win) |
| 55294656 `field_9` specialist | 35 | 545.9 | 556.3 | **+10.4** |

**Edge over own field is the only ladder number that measures the agent.** By it
the champion is a flat +51 to +54 across totally different brackets, the
retrained prior is the same agent at +45.6, and the `field_9` specialist is
genuinely weaker at +10.4. Every one of those verdicts matches what the local
120-game gates said, and none of them is visible in the raw score.

Two consequences that govern everything:

- **The raw score cannot detect any improvement this project is capable of
  making.** A change that lifts the true win rate from 58.5% to 63% moves the
  expected first-ten result by half a win, roughly 50 rating points, against 156
  points of noise. Gate locally over hundreds of games. The ladder is for
  confirming a disaster, not for ranking candidates.
- **Uploading is a lottery ticket with a ~240-point range**, and it re-rolls
  from 600 every time. That is worth more than any change measured in this repo,
  which is uncomfortable but is what the data says. See the re-roll strategy in
  "What to do next".

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
- **Do not upload without a reason.** The board shows the newest submission, so
  every upload restarts the score at ~600 and costs about a day of climbing.
  Upload when a gate says the archive is genuinely better, not to take draws.
- **Never read the board score off `max(public_score)` or off the best live
  submission.** Both are wrong. Use `competitions_list().user_rank` cross-checked
  against a leaderboard dump, which is what `tools/ladder_status.py` does.

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

### The full census (2026-08-07, Aug 1-6 dumps, 28,006 episodes)

`tools/deck_census.py` counts what `top_decks.py` does not: **decisions**, not
appearances. That is the number that decides whether a deck can support a prior
at all, and missing it is why the `field_9` specialist was built on 67k rows.
Exact 60-card lists, pilots at Elo 1000+:

| deck | elite decisions | seats | WR | mean pilot Elo |
|---|---:|---:|---:|---:|
| Munkidori / Marnie's Impidimp (Tech-Grim, **ours**) | 714,058 | 17,608 | **46.2%** | 989 |
| Abra / Kadabra / Alakazam | 161,912 | 5,478 | 50.2% | 995 |
| Dunsparce / Buddy-Buddy Poffin / Ultra Ball | 134,429 | 2,549 | **58.1%** | 1038 |
| Dunsparce / Buneary / Buddy-Buddy Poffin | 104,908 | 2,344 | 57.4% | 993 |
| **Mega Lucario ex / Ultra Ball / Premium Power Pro** | 63,113 | 1,128 | **67.3%** | **1224** |
| Energy Switch / Ultra Ball / Crispin | 35,127 | 567 | 56.8% | 1121 |

Two things to carry forward, and the second is the one that gets misread:

- **We ship the worst-winning deck in the format, and it is not close.** 46.2%
  against 58.1% and 67.3%. It is also by far the most played, which is why it
  has 11x the training data of anything else, which is why we ship it. That
  circularity is the whole problem.
- **Mega Lucario's 67.3% is heavily pilot-confounded.** Its mean pilot Elo is
  **1224**, against 989 for Tech-Grim, on a board whose rank 1 is 1276. A deck
  played almost exclusively by the strongest players will show a high win rate
  whatever the deck does. Dunsparce is the cleaner read: 58.1% at mean Elo
  1038, only 49 points above Tech-Grim's pilots, so much less of its edge can be
  explained by who is holding it. Do not quote 67.3% as a deck effect.

Our shipped `deck.csv` is **not** the field's Tech-Grim list. It is Jaccard
0.846 to it and has **zero** exact appearances in six days of dumps, so the
`techgrim_field` column in any per-deck score is the popular list, not ours.

### Rank decks by FIELD-WEIGHTED win rate, not raw win rate (2026-08-07)

`tools/matchup_matrix.py` mines deck-versus-deck results from the dumps, both
pilots above the Elo floor so a cell measures a matchup rather than a skill gap,
and reports

    field WR = sum over opponents of share(opponent) * winrate(vs opponent)

which is what a ladder score actually pays. A raw win rate averages over
whichever field that deck happened to draw; the field weighting re-projects it
onto the population we will be matched against. They disagree sharply:

| deck | raw WR | **field WR** | seats |
|---|---:|---:|---:|
| Mega Lucario ex | 65.2% | **63.9%** | 626 |
| **Dunsparce / Poffin / Ultra Ball** | 51.8% | **59.7%** | 901 |
| Energy Switch / Crispin | 54.3% | 55.3% | 280 |
| Dunsparce / Buneary | 56.9% | 54.8% | 626 |
| Dragapult (Dreepy / Drakloak) | 50.4% | 51.6% | 226 |
| Alakazam | 47.7% | 49.6% | 902 |
| **Tech-Grim (what we ship)** | 47.8% | **47.9%** | 3,444 |
| Crustle | 41.4% | 39.6% | 295 |

Dunsparce's raw 51.8% **understates it by 8 points** because it is favoured into
the decks that are actually common. Ranking on raw win rate would have put it
below three decks it beats. This is the third distinct way this project has
mis-ranked decks (Wilson lower bound picked a rank-1 player's 136-game side
deck; elite adoption picked a deck our pilot cannot play; raw win rate hides
matchup structure), so prefer field WR and say which metric a claim came from.

**Every deck has a hard counter, and that is the real risk in a deck swap:**

| | its worst matchup | n |
|---|---|---:|
| Dunsparce | **9% vs Mega Lucario** | 173 |
| Tech-Grim | **13% vs Bug Catching / Energy Search** | 181 |
| Alakazam | 36% vs Mega Lucario | 84 |

Dunsparce losing 91% to Lucario is already priced into its 59.7%, since Lucario
is 6.6% of seats. What is *not* priced in is drift: Lucario is the strongest
deck in the format and its share is rising, so this pick degrades if the meta
moves. Re-run the matrix before the deadline rather than trusting tonight's.

Two claims from competition discussion 729926 were checked against this and
both hold: Tech-Grim beats Alakazam (**61%**, n=332) and Crustle (**63%**,
n=203). That is the recorded reason Tech-Grim was good early -- it preyed on
the two decks that topped the board -- and it is no longer true of the field.

### The champion is a Tech-Grim specialist, measured (2026-08-07)

`tools/score_by_deck.py` scores checkpoints per 60-card list on a held-out day.
On the Aug 6 holdout, which neither model trained on:

| deck | rows | champion (the 972) | fresh balanced prior |
|---|---:|---:|---:|
| ALL | 279,364 | 67.4% | **71.8%** |
| Mega Lucario | 6,356 | 52.8% | **61.6%** |
| Dunsparce | 21,044 | 50.2% | **63.4%** |
| Tech-Grim (field list) | 105,369 | **79.5%** | 77.6% |

**The champion predicts Tech-Grim play at 79.5% and everything else at ~50%.**
That is not a general prior with a deck preference, it is a Tech-Grim model. It
is also the precise, previously-unmeasured mechanism behind
"win rate tracks Jaccard-to-Tech-Grim almost monotonically" in the deck screen
below: every deck swap was asking a Tech-Grim specialist to pilot a list it
scores at chance.

So the deck lever and the prior lever were never independent, and gating them
separately -- which this file has done four times -- cannot succeed. They have
to move together.

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

## All three shell fixes were gated and all three are dead (2026-08-05/06)

The bugs below are real. Fixing any of them does not help, and one hurts. The
first two were gated with `--budget 0` so each side used its own allocation,
which is what Kaggle does; the evaluator was gated at a matched 1.1 s/move
because it does not change the clock.

| change | archive | result vs the champion |
|---|---|---|
| horizon 160 + cap 3.0 s | `submission_972model_tunedbudget` | **29-30 (49.2%)** over 59 games, Wilson [0.368, 0.616]. A TIE. The earlier 8-15 was 23 games of noise |
| 20-archetype opponent model | `submission_972model_metaonly` | **30-30 (50.0%)** over 60 games, Wilson [0.377, 0.623] |
| fitted leaf evaluator | `submission_fiteval_techgrim` | **56-64 (46.7%)** over 120 games, Wilson [0.380, 0.556] |

The opponent-model archive is byte-identical to the champion except for
`main.py`, and `main.py` differs only in the `META_DECKS`/`META_WEIGHT` tables.
So that 30-30 is as clean an isolation as this project has ever run, and the
answer is nothing.

Two things worth keeping from it:

- **CORRECTION (2026-08-07): the budget fix did not lose, it tied.** Re-gated at
  60 games it went 29-30 (49.2%), Wilson [0.368, 0.616], zero errors, 66 minutes.
  The original 8-15 was 23 games of noise and this file stated it as fact for two
  days. What the tie actually says is more useful than the loss did: **the search
  is already converged at the shipping budget**, so extra think time buys
  nothing, and a teacher for distillation does not need to be expensive.
  The old reading of this row, now withdrawn, was that deeper search converges
  harder onto the evaluator's bias. That story was built on the 8-15 and does not
  survive the re-gate. The evaluator may still be a constraint, but this result is
  not evidence for it.
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

## Audited against the real runner (2026-08-06)

Read from the installed package rather than remembered:
`kaggle_environments/envs/cabt/cabt.py`, `cabt.json`, `core.py`, `agent.py`, and
the `cg` bindings. What the runner actually enforces:

| setting | value | what it means for us |
|---|---|---|
| `actTimeout` | **0** | every second of thinking is drawn from the 600 s overage pool. There is no free per-move allowance |
| `runTimeout` | **2000** | wall clock for the WHOLE episode, both agents plus engine. Exceeding it *raises*, killing the episode rather than losing it |
| `episodeSteps` | **10,000,000** | Kaggle has no step cap. The harness's `--max-steps 3000` is ours. Real games top out around 570 steps, so it never binds |
| deck length | exactly 60 | anything else is INVALID, which is an instant loss |
| rejected `select` | INVALID | instant loss, no retry, already known |

**The timeout is checked after the agent returns, not enforced during it**
(`agent.py`: `if duration - self.configuration.actTimeout > observation.remainingOverageTime`).
There is no interrupt, so a single decision that overruns the remaining pool
loses the game outright. The shell spends about 84 s of 600, so this is not
close, but it is the reason a large `PTCG_MAX_BUDGET` is dangerous.

Three things that were checked and are **fine**, recorded so they are not
re-checked:

- **`agent` really is the last callable.** Reproducing `get_last_callable` on the
  packaged champion gives 30 callables with `agent` last. `ladder_harness.py`
  asserts this on every load, so a future edit that appends a function is caught.
- **`battle_select` raises on non-`int` indices** (`cg/game.py` does
  `all(isinstance(i, int))`), and the interpreter turns that raise into INVALID,
  an instant loss. numpy integers would do exactly this. The shell's `_validate`
  already enforces `isinstance(i, int)`, so that check is load-bearing rather
  than redundant. Do not "simplify" it away.
- **The harness decrements `remainingOverageTime` correctly**, because it goes
  through `env.run()` and the accounting lives in `core.py`.

### CONFIRMED BUG: the agent can never decline an optional selection

Measured over 1,500 real ladder episodes, 257,223 ACTIVE selections:

| minCount | share of decisions |
|---|---:|
| 1 | 84.85% |
| **0 (declining is legal)** | **12.98%** |
| 2 | 1.84% |

When declining is legal, real players decline **3.92%** of the time. Our agent
cannot: two independent layers block it.

```python
# _validate: an empty action is rejected outright
lo = kmin if kmin >= 1 else 1          # minCount 0 becomes a floor of 1
# _gen_candidates: the empty action is never even generated
sizes = {kmax, max(kmin, 1)}
```

So on roughly **0.5% of all decisions** we are forced to act where a human would
pass. It is a genuine action-space restriction, not a tuning constant, and it is
the only *correctness* gap the audit found.

**It is still probably not worth fixing.** Decline rate does not rise with skill,
which is the evidence that would justify the work:

| player ladder score | decline rate |
|---|---:|
| <900 | 4.71% |
| <1000 | 4.64% |
| <1100 | 4.64% |
| **>=1100** | **3.36%** |

The best players decline *least*. It matters in specific contexts (context 2 at
18.2% over 1,316 chances, context 5 at 11.7% over 4,053) so a targeted fix is
possible, but the shell has now rejected three well-motivated changes and this
one has a smaller prior than any of them.

### Two corrections to what this file used to say

- **`EMBEDDED_DECK` is not in the frozen shell.** It exists only in
  `rl_osfp/agent_main.py`. The frozen shell's `_load_deck` reads `deck.csv` with
  no fallback, and if that read fails the outer `except` returns `[0]`, a 1-card
  deck, which is an instant loss. It works on Kaggle because
  `/kaggle_simulations/agent` resolves there. Do not assume the shell is
  protected by a fix that lives in the other agent.
- **The median game is 84 decisions per side** (p90 124, max 283), which the old
  note had right. A first pass at this said 173 by counting the INACTIVE seat,
  which carries a **stale** `select` the interpreter never clears. When scanning
  replays, filter `status == "ACTIVE"`, and pair `steps[i]` observations with
  `steps[i+1]` actions, because the action is recorded one step after the
  observation it answers.

### Harness bug fixed while auditing

`ladder_harness.py` built its job list and its seat map in two separate loops
that had to stay in lockstep. They agreed, but nothing enforced it, and a
divergence would silently misattribute every result. Now built once. The same
edit added `--vs`, which turns the round robin into an O(n) gauntlet.

### THE BIG ONE: the harness silently disabled our own search (fixed 2026-08-06)

Found the moment a foreign archive was put in the harness for the first time.
**Archives do not ship the same `cg` package.** Ours has `engine.py`, the ctypes
binding the search needs. The published competitor archives ship `api.py` and
`utils.py` and **no `engine.py`**. `cg` is imported by its bare name, and
`_private_modules` only ever isolated `nn_infer`, despite a docstring claiming
it isolated `cg` too.

So whichever archive imported `cg` first defined it for both. When that was a
competitor archive, our shell's `_load_engine` did:

```python
from cg.engine import get_lib     # ModuleNotFoundError
```

which its own `try/except` swallows by design, leaving `_LIB = None`,
`_ENGINE_TRIED = True`, and the agent playing on heuristic priors with **no
search, no error, and no fallback counter**. The measured signature:

| | before the fix | after |
|---|---|---|
| champion vs `ext_crustle` | **0-8** | 2-2 |
| 12 full games, wall clock | **1.6 seconds** | 1.24 min for 4 |

Twelve complete games with real step counts and clean DONE statuses in 1.6
seconds is what no-search looks like. Nothing raised anywhere.

Fixed by `_private_cg`, which loads each archive's `cg` under its own package
object whose `__path__` is that archive's `cg/`, plus `_archive_context`, one
context manager that now owns every save/restore. It also sets **cwd to the
archive's directory**, because third-party agents resolve `deck.csv` against
cwd and fall back to `/kaggle_simulations/agent`, never touching `__file__`.
Both competitor archives do exactly that, and without the chdir they raise
`FileNotFoundError` and score zero.

**Scope of the damage: none of the recorded gates are affected**, because every
one of them mixed only our own archives and all of those ship `engine.py`. But
any future gate against an external agent would have been measuring a crippled
champion, which is precisely the trap this file warns about under "every gate
must include an external reference".

## Cheap-budget deck screening is VALIDATED (measured 2026-08-06)

The `--budget` warning in the harness section is about **search agents against
search-free agents**. It does not apply when both sides are the same search
agent and only `deck.csv` differs, and that is now measured rather than assumed.

Same three archives, same pilot, only the deck differing, at two budgets:

| pairing | 1.1 s/move (`gate_round2`) | 0.25 s/move (probe) | agrees? |
|---|---|---|---|
| tech_grim vs field_16 | 14-6 (70.0%) | 33-7 (82.5%) | yes |
| tech_grim vs field_4 | 12-8 (60.0%) | 36-4 (90.0%) | yes |
| field_16 vs field_4 | 17-3 (85.0%) | 33-7 (82.5%) | yes |

Rating order identical at both budgets: `tech_grim > field_16 > field_4`. All
three pairwise directions preserved. **120 games took 16.9 minutes against 78
minutes at the shipping budget, so 4.6x more games per hour.**

Two caveats that matter:

- **The cheap budget exaggerates the gaps** (60% becomes 90%). Use it for
  ordering only. Never quote a cheap-budget win rate as a result.
- The 1.1 s reference is only 20 games a pair, Wilson about plus or minus 20%,
  so this rules out a gross inversion rather than proving fine agreement.

`harness/deck_budget_probe_025.json`.

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

### The full from-scratch pipeline, end to end (2026-08-07)

Four tools were added so a prior can be built from raw dumps without touching
any previous checkpoint. Each exists because a specific earlier attempt failed
for want of it.

| tool | what it does | the failure it prevents |
|---|---|---|
| `bc_train/ingest_episodes.py` | raw `.zip` dumps -> shards, parallel, flushing every 60k rows | the quarantined ingest imported the **SEQ-93** encoders; this one is inside `bc_train` so it uses SEQ-53, the width the shell serves. It also holds bounded memory, which is how the VM died twice |
| `tools/deck_census.py` | decisions per 60-card list per Elo band | `top_decks.py` counts *appearances*, so the `field_9` specialist was built on 67k rows before anyone checked |
| `tools/balance_corpus.py` | caps rows per list, splits a whole day out as a temporal holdout | the champion corpus is 36% one deck, which is exactly why it can only pilot that deck |
| `tools/score_by_deck.py` | top-1 and CE **per deck** on a held-out day | a whole-corpus top-1 hid that the champion is at chance on every list but one |

```bash
# 1. raw dumps -> elite shards (2,185,588 decisions from Aug 1-6, ~6 min)
.venv/bin/python bc_train/ingest_episodes.py data/fresh/replays \
  --out data/bc_elite_aug --leaderboard data/fresh/leaderboard/pokemon-tcg-ai-battle.zip \
  --min-elo 1000 --features rich --workers 6

# 2. cap every list so no deck dominates; hold out a whole day
.venv/bin/python tools/balance_corpus.py --data data/bc_elite_aug \
  --out data/bc_bal_lucario --holdout-day 2026-08-06 \
  --holdout-out data/bc_bal_lucario_holdout --cap-per-deck 60000

# 3. train FROM SCRATCH -- no --init, deliberately
OMP_NUM_THREADS=6 PYTHONPATH=bc_train .venv/bin/python -u bc_train/train_bc.py \
  --data data/bc_bal_lucario --val-data data/bc_bal_lucario_val \
  --max-per-shard 23000 --dim 160 --layers 5 --heads 5 --features rich \
  --elo-weight 0.5 --epochs 14 --patience 3 --device cuda \
  --out data/model_lucario_scratch.npz

# 4. package once per candidate deck, verify each fires its net, gauntlet
bash tools/deck_arms.sh data/model_lucario_scratch.npz scratch 80 8
```

Result: early-stopped at epoch 12, restored epoch 9, **71.1% top-1** against
68.3% for the previous best from-scratch run and 62.7% for the champion.
Verified behind the shell at a **100% prior rate, 10.4 ms/call, 3.6 calls per
decision, zero invalid actions** -- numerically identical cost to the champion,
so it is a true drop-in.

Two practical notes. `tools/resource_guard.py` refused 1,068,344 rows (18.9 GiB
peak) and the run was capped to 708,405; **respect it, it is not advisory**.
And redirect training through `python -u`: block buffering hides the epoch
lines for tens of minutes, which is indistinguishable from a hang.

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

## The fitted evaluator is DEAD too (gated 2026-08-06, 120 games)

This was the most promising single change this project has built, and it came
back flat like the other four. Record it so nobody rebuilds it.

| archive | record | win rate | Wilson 95% |
|---|---|---:|---|
| `grpo_tech_grim_972_912_811` | 64-56 | 53.3% | |
| `submission_fiteval_techgrim` | **56-64** | **46.7%** | **[0.380, 0.556]** |

120 decided games at 1.1 s/move, 77.8 minutes, **zero** errors, zero invalid
actions, zero step caps, zero draws. 50% sits inside the interval, so this is a
tie with the point estimate slightly behind. `harness/gate_fiteval.json`.

The isolation is as clean as this project gets: `main.py` is the only file that
differs from the champion, and inside it only `_evaluate`. `model.npz`,
`deck.csv` and all three encoders are byte-identical.

**And the prediction really was good, which is the point.** The shipped
evaluator scores **AUC 0.457 at turns 20 to 30**, below chance, so in late-game
positions it points the search at the wrong player. The refit fixes that:

| evaluator | AUC overall | turns 0-10 | turns 10-20 | turns 20-30 |
|---|---:|---:|---:|---:|
| shipped | 0.6567 | 0.638 | 0.735 | **0.457** |
| refit | **0.7400** | 0.699 | 0.852 | 0.774 |

124,054 positions from 1,800 real episodes, both seats. Saturation is clean at
temperature 1.0: `|v| > 0.95` on 0.4% of outputs against the shipped 0.3%, with a
*wider* spread (std 0.419 vs 0.318), so it discriminates between sibling moves
better rather than worse. None of that showed up as wins.

**So a leaf evaluator that predicts the winner better does not make this search
play better.** That is now measured, not argued, and it is the strongest
available evidence that the search's strength does not live in `_evaluate`.
Combined with the budget fix losing 8-15 and the opponent model going 30-30, the
shell has now resisted three independent, well-motivated improvements. Stop
editing the shell.

Artifacts kept: `foundation/search_shell_fiteval.py`, `data/evaluator_fit.json`,
`tools/fit_evaluator.py`, `tools/build_fiteval_shell.py`. Regenerate with the
fitter then the builder; the builder refuses unless the patch provably touches
nothing but `_evaluate`. Do not re-gate this without a new hypothesis.

### The field_9 specialist confirmed weaker on the ladder

The local gate had it 28-32 (46.7%, Wilson [0.346, 0.591]), which read as a tie.
Ref 55294656 then played 35 real episodes and settled at **556.3, performance
rating 556**, against every champion instance at 719 to 948. It was drifting
down, not up, and 35 episodes is past the high-K phase. Both measurements agree:
the `field_9` fine-tune is not an improvement, and the gate called it first.

Note the champion's own performance-rating spread on identical bytes at similar
episode counts: **719 (52 eps) against 919 (49 eps)**. So performance rating is
itself noisy at 50 games, and 556 being below the whole band is the signal, not
its exact value.

## The "RL does not work here" conclusion was not supported (2026-08-06)

This file spent several sections concluding that RL is a dead end and BC plus
"conservative GRPO" is the only path above 850. The first half is unsupported
and the second half is mislabelled. Counted from the metrics files that the
972-lineage run actually wrote:

| stage | decisions | optimizer steps | weight change |
|---|---:|---:|---|
| BC prior | **937,178** human | ~40,000 | full train |
| **GRPO "refinement"** | **26,642** | **642** | **0.09%** |
| PPO from random (`run_v3`) | 6,400,000 self-play | ~50,000 | full train, ladder 480 |

`.reset-quarantine-20260803/track5_grpo/` defaults are `--groups 4
--group-size 4 --lr 5e-6`, so an iteration is 16 games and the whole refinement
was 642 optimizer steps. **The 972 archive is a BC model.** Attributing its
score to GRPO, and then generalising to "RL does not help", does not follow from
anything measured.

The real experiment table has an empty cell, and it is the interesting one:

| init | RL scale | result |
|---|---|---|
| random | full (6.4M decisions) | ladder **480** |
| BC | none | ladder 967 / 917 / 873 |
| BC | 26,642 decisions | ladder 972 |
| **BC** | **full** | **never run** |

It was never run because `train.py` had **no way to load anything but its own
`training_state.pt`**: line 376 was `ActorCritic(NetworkConfig())  # random
initialization`, unconditionally. So random init was not a finding, it was the
only option the code offered, and it is also the single measured difference
between the 480 arm and the 972 arm.

### `rl_osfp/bc_init.py` closes it

The two architectures are the same trunk. `bc_train/model.py`'s TCGNet and
`rl_osfp/network.py`'s ActorCritic share every embedding, every block and the
final layer norm, name for name and shape for shape. Only three things differ:

    pol_head   -> option_head      same shape, renamed
    val_fc1/2  -> value_fc1/2      same shape, renamed
    count_head                     new, absent from BC, zero-initialised

74 of the champion's 75 tensors copy straight across; `_meta` is metadata and
`count_head` is the only fresh parameter.

**Sequence length does not have to match.** There is no positional embedding: a
token is `card_emb + kind_emb + scal_proj(scalars)` and attention is full, so
the trunk is permutation-equivariant over tokens. Champion weights trained at
`MAX_OPT 24` (SEQ 53) therefore load and run unchanged under rl_osfp's
`MAX_OPT 64` (SEQ 93); the wider encoder only truncates fewer options.

Zero-initialising `count_head` is safe because `policy.py` masks the count to
`[minCount, maxCount]` and 84.85% of real decisions have `minCount == maxCount
== 1`, so the mask decides the count outright on five sixths of the game.

**Verified rather than assumed.** The converted torch model reproduces the
champion's own packaged numpy net to float32 precision on random inputs:

    max |d option logits| = 2.1e-06        max |d value| = 1.6e-07

```bash
.venv/bin/python -m rl_osfp.train --init <bc_model.npz> --pool <pool> --out-dir <dir> ...
```

`run_config` now records `initialization`, `init_checkpoint` and
`replay_action_labels` honestly: a BC prior is trained on human action labels,
so a BC-initialised run is **not** replay-free and no longer claims to be. The
replay-free constraint was a deliberate choice of the original run, and every
run that honoured it topped out at 480.

First smoke run, 4 games from the champion prior: `entropy 0.443`,
`approx_kl 0.0173`, `clip_fraction 0.064`, zero invalid actions. For contrast
the random-init runs sat at entropy 0.78 and `approx_kl` 0.0003 to 0.005 against
the same 0.04 target, which is to say they were barely stepping at all.

### The network ships as a PRIOR, so the objective is not "play well alone"

This is the trap that sank the last attempt and the reason `--ref-kl-coef`
exists. `submission_ppo_bcsearch` put v3 PPO weights behind the frozen shell and
finished **last at 8.3% (6-66)**, below the same archive carrying *no network at
all*. An RL policy optimised to play unaided became a worse prior than the
heuristic, because PUCT uses the prior to decide which subtrees are ever
expanded, and an overconfident prior simply deletes the alternatives.

`--target-kl` does not protect against this. It bounds movement away from the
**behaviour** policy, which is last period's weights, so over a hundred periods
the policy walks arbitrarily far from where it started while every step looks
small. `--ref-kl-coef` adds a penalty on divergence from the frozen `--init`
prior itself, using Schulman's k3 estimator, and `ref_kl` is reported every
period so drift away from the weights that actually scored 972 is visible.

Two consequences for how a run is judged:

- **Gate checkpoints behind the shell, not against each other.** A bare-policy
  round robin measures the wrong thing. Package the candidate into the champion
  archive and run `tools/ladder_harness.py` against `grpo_tech_grim`. The
  cheap-budget result above makes that affordable.
- Start the anchor around 0.05 to 0.2. Zero reproduces the failure mode; the
  972's own GRPO used a 0.04 KL beta but took only 642 steps, so it never
  tested the interesting part of the range.

### Our RL runs are 87x too small, measured against a public number (2026-08-07)

Competition discussion 709160/RL thread has a competitor at rank 143 reporting
his self-play setup in enough detail to compare directly: **~45 games/sec, 3 to
5 million games, about a day on one GPU**, ~2M parameters, and he calls that
"extremely undertrained". Read against `rl_osfp/run_v3/metrics.json`:

| | run_v3 (our best pure RL) | rank-143 competitor |
|---|---:|---:|
| games | **44,800** | 3-5 million |
| decisions | 3,701,773 | - |
| throughput | 7.4 games/sec | ~45 games/sec |
| wall clock | **100 minutes** | ~24 hours |

**87x fewer games.** Throughput is only 6x off; the dominant term is that we ran
RL for **an hour and forty minutes** and then wrote several sections of this
file concluding RL is a dead end. Matching 3.9M games costs 146 hours at our
current rate, or about a day at 6x throughput. That is affordable.

Do not read this as "so scale it and we win", for two reasons:

- **Where scaled pure RL actually lands is roughly where we already are.** The
  rank-143 run reached silver (~900) and rank 90 reports ~1000 with Archaludon;
  most of that thread plateaus at 700-800. Our champion measures 942 to 948.
- **The objective mismatch below is not a scale problem.** `run_bc1` optimised
  correctly and beat its own BC prior 64.3%, and was still an 11-point worse
  *search prior*. Scaling that arm buys a better bare policy, which is not what
  we ship.

So the honest statement is: RL here is untested at scale, not disproven, and the
version worth testing is one whose reward comes from games played **by the
search using the net as its prior**, not from the net playing alone.

### RESULT: it worked as RL and still lost as a prior (gated 2026-08-06)

The run finished: 200 periods, **2,309,000 decisions**, 87x the entire GRPO
stage, zero invalid actions throughout. As reinforcement learning it did exactly
what it should:

| measure | result |
|---|---|
| learner vs the frozen BC prior | **256-142 = 64.3%**, Wilson [0.595, 0.689], n=398 |
| entropy | 0.378 rising to 0.464, no collapse |
| `ref_kl` | saturated at ~0.30, the anchor held |
| `approx_kl` | 0.012 to 0.018 against a 0.04 target, trust region binding |

Then every checkpoint was packaged into the champion archive by swapping
`model.npz` alone, verified to be firing its network at a 100% prior rate and
10.3 ms/call, and screened against the champion at 0.25 s/move:

| checkpoint | `ref_kl` | record | win rate | Wilson 95% |
|---|---:|---|---:|---|
| p025 | 0.176 | 14-26 | 35.0% | [0.221, 0.505] |
| p050 | 0.232 | 19-21 | 47.5% | [0.329, 0.625] |
| p100 | 0.290 | 15-25 | 37.5% | [0.242, 0.530] |
| p150 | 0.308 | 13-27 | 32.5% | [0.201, 0.480] |
| p200 | 0.302 | 17-23 | 42.5% | [0.285, 0.578] |
| **pooled** | | **78-122** | **39.0%** | **[0.325, 0.459]** |

200 games, zero errors, zero step caps. `harness/gate_bcrl_screen.json`.

**A policy that plays 64.3% against the BC prior is an 11-point worse prior for
the search that ships.** That is the second independent confirmation, after
`submission_ppo_bcsearch` at 8.3%, and the anchor is what separates them: small
drift costs 11 points, unbounded drift cost 42.

**The overconfidence explanation is now ruled out.** That was the standing story
for why RL priors hurt: a peaked policy starves PUCT of alternatives. Here
entropy *rose* from 0.378 to 0.464, so the policy got more diffuse and still got
worse as a prior. Whatever is happening is not sharpness.

Drift does not explain it either. Pearson r between `ref_kl` and win rate is
**-0.08** across the five checkpoints, which is nothing.

The likely mechanism, unverified: BC priors encode *what a strong human would
consider*, which is exactly the candidate set a search wants to enumerate. Self-
play RL reshapes them toward *what beats the current league*, which is narrower
and fitted to an opponent distribution that is not the ladder. Bare-policy
strength and prior quality are simply different objectives, and this project has
now paid twice to learn it.

**Do not run this again without changing the objective.** Optimising the policy
to play is the wrong target. If RL is to help here it has to be trained against
the thing that ships, which means the reward has to come from games played *by
the search using this net as its prior*, not by the net alone. That is far more
expensive per game and has never been costed.

### THE SEARCH AGREES WITH ITS OWN PRIOR 95.7% OF THE TIME

Measured 2026-08-06 on 300 real self-play decisions, comparing the champion
net's argmax against the search's root visit distribution at a 0.1 s budget:

| | |
|---|---:|
| net top-1 == search top-1 | **95.7%** |
| search overrides the net | **4.3%** |
| mean search visit mass on the net's pick | 0.770 |
| mean search visit mass on its own pick | 0.774 |
| net's pick received any visits at all | 99.7% |

On the 267 decisions with more than two options it is 95.5%, so this is not an
artefact of forced choices.

**The mechanism is in `_gen_candidates`.** The search does not generate its
candidates independently:

```python
scores = _net_scores(state, me, sel, opts, heur)     # the net
order  = sorted(range(n), key=lambda i: -scores[i])  # ranked BY the net
cands  = [(i,) for i in order[:cap]]                 # top 16 only
```

So the search can only choose among options the net already ranked highly, and
it rarely overturns the net's first choice. It is the net plus verification, not
an independent expert.

This one number explains a lot of this file at once:

- **Why stripping the net collapses the agent** from 80.0% to 25.0%. The net is
  not a hint to the search, it is most of the decision.
- **Why the fitted evaluator bought nothing** despite far better AUC. The leaf
  function only breaks ties inside a candidate set the prior already chose.
- **Why the budget fix lost.** More think time re-verifies the same shortlist.
- **Why RL priors swing results so violently** (8.3% unanchored, 39.0% anchored,
  80.0% for BC). The prior is the agent.

**But 95.7% is a property of a CHEAP teacher, not of the method.** That number
was measured at a 0.1 s budget, and generalising it to "expert iteration is
dead here" was wrong. Two corrections, both measured the same day:

- **The candidate cap is not the anchor.** Options per decision are a median of
  5 against `cap=16`, so the truncation binds on only **3.0%** of decisions. The
  search sees essentially every option. What anchors it is PUCT's prior
  weighting, `pri = exp(min(6.0, sum(scores)/len(c)))`: a confident net gives a
  near-degenerate prior, the visits pile onto its first choice, and a shallow
  search never accumulates the evidence to overturn it.
- **Depth and prior temperature both free the search, and they stack.**

| teacher | disagreement with the net | on >2 options | target entropy |
|---|---:|---:|---:|
| 0.1 s, temperature 1 | 4.3% | 4.5% | 0.539 |
| 2.0 s, temperature 1 | **15.4%** | 17.5% | 0.635 |
| 2.0 s, temperature 3 | **22.8%** | **26.9%** | 0.953 |

Flattening the prior costs nothing: same ranking, same candidate set, only the
exploration changes, and it is worth more than the extra think time. With a
median of 5 options and tens of thousands of simulations there is ample evidence
to judge every option on its merits, and the peaked prior was suppressing that
evidence rather than supplying it.

So the teacher is genuinely stronger than the student on roughly a fifth of
decisions, which is the signal expert iteration runs on. `--prior-temperature`
in `exit_generate.py` implements it by wrapping `_net_scores` in the loaded
namespace, so **the frozen shell is untouched and deployment still runs at
temperature 1**.

`rl_osfp/exit_generate.py` is kept and works: it drives the frozen shell via the
`collect_policy` hook that `_search_move` already exposes, maps root visit counts
onto option tokens through `encode`'s `opt_slot`, and writes shards in
`bc_train/train_bc.py`'s exact format (`kind, card, scal, mask, ctx, stype, pi,
z`) so the proven trainer consumes them with `--init`. Verified on real output:
`pi` sums to 1, **zero mass outside legal option tokens**, mean 5.8 options per
decision, target entropy 0.539.

**Status: a strong-teacher corpus is generating** at 1.0 s with temperature 3,
into `data/exit_corpus`. The plan after it lands, and each step is already
built and verified:

```bash
# 1. fine-tune the champion prior on what the deep search chose
OMP_NUM_THREADS=6 PYTHONPATH=bc_train .venv/bin/python bc_train/train_bc.py \
  --data data/exit_corpus --init data/model_champion_bc.npz \
  --lr 2e-4 --dim 160 --layers 5 --heads 5 --features rich \
  --epochs 10 --patience 3 --out data/model_exit_iter1.npz

# 2. package by swapping ONLY model.npz into the proven archive
.venv/bin/python tools/swap_model.py \
  --base harness/anchors/grpo_tech_grim_972_912_811.tar.gz \
  --model data/model_exit_iter1.npz --out artifacts/submission_exit_iter1.tar.gz

# 3. gate it
.venv/bin/python tools/ladder_harness.py \
  --archives harness/anchors/grpo_tech_grim_972_912_811.tar.gz \
             artifacts/submission_exit_iter1.tar.gz \
  --games-per-pair 120 --budget 1.1 --workers 5 --out harness/gate_exit_iter1.json
```

**What would make this the first thing to beat the champion**, and why it is
different in kind from the two RL attempts that failed: the target is not "win
more games as a bare policy", it is "rank options the way a search with tens of
thousands of simulations ranked them". That is the deployed objective exactly.
Both failed attempts optimised standalone play and then hoped it transferred;
this one never leaves the objective the shell actually uses.

The honest risk is that the teacher is our own search, so the ceiling of one
iteration is roughly "the net becomes as good as a 1 s search's judgment". That
is a real gain rather than a circular one only because the search genuinely
beats its own prior on 22.8% of decisions. If iteration 1 gates positive, the
improved prior makes iteration 2's teacher stronger for free.

### The checkpoints drop into the champion archive

`export_champion_npz` writes the archive's own format, so a candidate is gated
by swapping `model.npz` into `grpo_tech_grim_972_912_811.tar.gz` and changing
nothing else. Verified: the round trip
champion npz to ActorCritic to champion npz is **byte-identical**, 75 tensors,
zero difference. `count_head` is dropped rather than serialised, which is
correct rather than lossy, since the shell generates its own candidate option
sets and asks the net only for per-option priors.

This only holds at `PTCG_MAX_OPT=24`. `run_config` records `max_opt`, `seq` and
`champion_format_export`, and the trainer prints a warning and skips the
champion-format export at any other encoder width, because a checkpoint trained
at one `MAX_OPT` and served at another is a silent policy change rather than an
error.

```bash
PTCG_MAX_OPT=24 .venv/bin/python -m rl_osfp.train \
  --init <champion model.npz> --ref-kl-coef 0.1 \
  --pool data/fresh/deck_pool_techgrim.json --out-dir rl_osfp/run_bc1 \
  --periods 200 --games-per-period 96 --max-decisions 12000 \
  --lr 2e-4 --epochs 2 --workers 5 --device cuda
```

`data/fresh/deck_pool_techgrim.json` puts Tech-Grim in `learner_decks` and the
18 mined lists in `field_decks`, which fixes the mismatch recorded above where
the v3 policy had piloted its own shipping deck zero times.

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

## The public notebooks are readable, and they change the picture (2026-08-06)

The discussion forum renders client side and every internal discussion endpoint
answers 400 or 404. **The notebooks API works and is far better**, because it
returns runnable agents rather than opinions:

```bash
.venv/bin/kaggle kernels list --competition pokemon-tcg-ai-battle --sort-by voteCount
.venv/bin/kaggle kernels pull   -p <dir> <ref>   # the notebook source
.venv/bin/kaggle kernels output -p <dir> <ref>   # deck.csv and submission.tar.gz
```

`kernels output` is the important one: several notebooks publish their **entire
built submission tarball**. Two are installed as anchors, and they are the first
genuinely external references this project has ever had, which matters because
this file's own methodology section says every gate needs one and until now
every anchor was our own archive.

| anchor | what it is |
|---|---|
| `harness/anchors/ext_crustle_day1_rank1.tar.gz` | day-1 rank 1. Plain rule-based, **no search, no ML** |
| `harness/anchors/ext_alakazam_day2_rank5.tar.gz` | day-2 rank 5, rule-based |

Decklists are in `data/decks_external/`, all six validated legal against
`battle_start` by `tools/deck_cycle.py --validate-only`.

**The single most useful thing in these notebooks is the day-1 rank-1 author's
own explanation**, and it argues the opposite of this project's whole approach:

> the agent itself is almost embarrassingly simple. It is a plain rule-based
> bot, no search, no machine learning. The real work went somewhere else, into
> deck building. The whole idea was not "write a clever agent to pilot a strong
> deck," but the opposite: build a deck so stable that even a baby-simple agent
> can pilot it well.

Its entire policy is a static priority order: ATTACH 1000, EVOLVE 800, PLAY 600,
ABILITY 400, ATTACK 100, RETREAT -1, plus a handful of card special cases. That
reached rank 1. We have spent five gated experiments on the pilot and moved
nothing.

Deck matching against our mined pool, by multiset Jaccard:

| notebook deck | vs Tech-Grim | best match in our pool |
|---|---:|---|
| `lb950_deck1` (a public "LB 950+" baseline) | 0.846 | `field_0` at **1.000** |
| `alakazam_rank5` | 0.121 | `field_8` at 0.739 |
| `crustle_day1_rank1` | 0.026 | `field_10` at 0.364 |
| **`archaludon_cinderace`** | 0.132 | `field_9` at **0.200** |

Two things follow:

- **A public baseline scoring "LB 950+" ships essentially our deck.** Its second
  decklist is byte-identical to `mined_1` inside our own shell's `META_DECKS`,
  and its first matches `field_0` at 1.000. So Tech-Grim is not a secret edge;
  it is what the public baseline plays, and our ~942 is roughly what that deck
  pays anyone.
- **Archaludon/Cinderace is a real outsider.** Its best match anywhere in our
  20-deck pool is 0.200, so our meta model has essentially never seen it, and
  `META_DECKS` would determinize an opponent playing it into a fantasy. Its
  author reports **74.4% over 1,000 games** against their own 1300+ Starmie
  submission, while warning it is one matchup and Froslass is weak to Metal.
  Unverified by us, large sample, and cheap to gate.

### More notebooks worth knowing about (pulled 2026-08-07)

`kaggle kernels list --competition pokemon-tcg-ai-battle --sort-by voteCount
--page-size 60` returns far more than the four originally mined. The ones that
matter:

| notebook | why |
|---|---|
| `makthanithin/pokemon-tcg-ai-battle-1084-5-baseline` | claims **LB 1084.5**, rule-based Mega Lucario ex, no ML at all |
| `myso1987/ptcg-ai-battle-leaderboard-deck-meta-by-score-band` | top archetypes per 100-point band **through 1100+** |
| `busyaprime/what-actually-wins-on-the-ladder` | archetype tier list and matchup grid recomputed from raw logs |
| `abiolatti/custom-engine-with-vectorized-env-2m-sample-sec` | reimplemented engine at **2M steps/sec**, public GitHub |
| `yu0307/16-real-city-league-top-cut-decks-deck-csv` | 16 real tournament decklists as `deck.csv` |

**The published 1084.5 artifact does not run.** Its `main.py` contains
`) hi:` where `):` was meant, a transcription typo, so every one of 120 gate
games failed with `SyntaxError: invalid syntax (main.py, line 322)`. Fixing that
single token makes it compile, and the repaired copy is what
`harness/anchors/ext_lucario_1084.tar.gz` holds. Treat its claimed score as
unverified: whatever the author actually submitted is not what they published.

Its deck is Mega Lucario, **Jaccard 0.667 to `field_16`**, the list measured at
a 75% ladder win rate and which our Tech-Grim prior piloted at 44.8%. So it is
simultaneously the most interesting deck-and-pilot pairing available and the one
our prior is worst at.

### PUCT prior weighting IS tuned now, and both directions are dead (2026-08-07)

`tools/build_puct_shell.py` emits variants that touch only `_gen_candidates` and
`C_PUCT`, proved by a real diff rather than a line-index compare. Both were
packaged onto the Dunsparce base with `main.py` as the sole changed file and
gated at 1.1 s/move, 70 games each, zero errors, zero step caps:

| arm | record | win rate | Wilson 95% |
|---|---|---:|---|
| prior floor 0.10 | 29-41 | **41.4%** | [0.306, 0.531] |
| `C_PUCT` 2.5 | 38-32 | **54.3%** | [0.427, 0.654] |

**Neither ships.** And note the trap this ran straight into: `C_PUCT` read
**65.7% at 35 games** and settled at 54.3% at 70. That is the same early-read
decay that took the GRPO-v2 arm from 55.9% to 50.3%, and it was over-read again
here in real time. Do not act on a shell-change gate under ~150 games.

What survives is only the negative half:

- **Flattening the prior hurts.** The floor lost, and temperature 1.5 and 2.5
  lost 38-61 and 25-75 in the earlier sweep. Do not hand mass to options the
  net rated near zero.
- **"Sharpen it" is NOT supported at the shipping budget.** Temperature 0.7 won
  57-42 at 0.25 s but tied 15-17 at 1.1 s, and `C_PUCT` 2.5 tied. The 0.25 s win
  was a cheap-budget artefact: starving the search makes trusting the net look
  better than it is.

The defaults (temperature 1, `C_PUCT` 1.4) sit at a local optimum. That is
**six** gated shell changes dead: budget, opponent model, fitted evaluator,
prior temperature, prior floor, `C_PUCT`. The one change that ever separated
from the champion did not touch the shell at all.

`harness/gate_puct_variants.json`.

### RL throughput: half the wall clock was spent not playing (2026-08-07)

Measured with ten workers on the Lucario pool, reading `rollout` against the
per-period wall time that `metrics.json` implies:

| | per period |
|---|---:|
| rollout, games actually playing | 0.13 min |
| wall clock | 0.27 min |
| **dead time** | **~53%** |

Rollout-only throughput is 28.7 games/sec; end-to-end was 15.2. The gap is the
sync barrier plus the PPO update plus the league evaluation batch, during all of
which every worker idles.

`--async-rollout` closes it, and `--eval-every N` stops paying for an evaluation
batch (up to 48 games producing no training data) on every period:

| mode | games/sec |
|---|---:|
| synchronous | 15.2 |
| async, generator-driven | 15.5 |
| **async, callback-driven** | **29.0** |

**The 2% row is the lesson.** A generator that dispatches inside its own body
only runs while the caller iterates it, so the moment the learner breaks out to
update, dispatching stops, the in-flight games drain, and the workers idle
exactly as before. Refilling has to be driven by completion, from the pool's
own callback. The fixed version reaches 29.0, which is the pure-rollout ceiling,
so the dead time is fully hidden.

Staleness needs no new machinery: `Decision.old_logp` is recorded at action
time so PPO's clipped ratio already corrects off-policy trajectories, and the
value target is the terminal game outcome rather than a bootstrapped n-step
return, so it carries no staleness bias. **V-trace would be redundant here** --
the problem was architectural, not algorithmic.

`harness/rl_throughput_probe.log`.

### The prior temperature has never been tuned, and theory says it should be

MCTS with a policy prior is implicitly KL-regularised toward that prior, with
the strength set by how sharp the prior is
(https://arxiv.org/abs/2112.07544). The shell fixes that strength at
temperature 1 by construction:

    pri = math.exp(min(6.0, sum(scores) / len(c)))

Nobody chose 1. Measured evidence the knob is live: dividing those same logits
by 3 moved the search's disagreement with its own prior from 4.3% to 22.8%, and
removing the net entirely drops the archive from 80.0% to 25.0%.

`tools/build_prior_temp_shell.py` emits a variant with exactly one line changed
and refuses otherwise. Unlike the three shell changes that were gated and died,
this adds no heuristic and does not reorder options; it only rescales an
existing term whose correct value was never measured.

Older, weaker signals kept for completeness: leaderboard agent names suggest
others are also on RL plus MCTS, and expectiminimax is reported to struggle with
the branching factor. Neither is worth acting on.

### The champion beats both public agents (gated 2026-08-06, 120 games)

The first externally-referenced gate this project has ever run, and the answer
is clean. Each side used its own budget (`--budget 0`), which is what Kaggle
does.

| matchup | record | win rate | Wilson 95% |
|---|---|---:|---|
| champion vs `ext_crustle_day1_rank1` | 43-17 | **71.7%** | [0.592, 0.815] |
| champion vs `ext_alakazam_day2_rank5` | 42-18 | **70.0%** | [0.575, 0.801] |
| **combined** | **85-35** | **70.8%** | **[0.622, 0.782]** |

42.1 minutes, zero errors, zero step caps, zero draws. `harness/gate_external.json`.

**This was only measurable after the `_private_cg` fix.** Before it, the same
matchup ran 0-8 with our search silently disabled, and it would have "proved"
the opposite conclusion with no error anywhere.

What it settles and what it does not:

- **Our pilot is genuinely strong.** The rank-1 author's thesis that a
  baby-simple bot on a stable deck is enough does not hold against this agent.
  Copying a public rule-based agent is not a route to a better score.
- **The +51 edge over field is real skill**, not an artefact of easy matchmaking.
- It does **not** show we would beat today's leaders. These are day-1 and day-2
  agents from a much weaker field, and their authors have since iterated.
- It conflates deck and pilot: `crustle` is Jaccard 0.026 to Tech-Grim and
  `alakazam` 0.121, so this is our deck-plus-pilot against theirs, not a pilot
  comparison. The deck question stays open.

## The deck slot is closed without per-deck prior training (screened 2026-08-06)

Six external decks under the champion pilot, only `deck.csv` differing, gauntlet
against the champion on Tech-Grim, 180 games at 0.25 s/move:

| deck | record | win rate | Wilson 95% | Jaccard to Tech-Grim |
|---|---|---:|---|---:|
| `lb950_deck1` | 17-13 | **56.7%** | [0.392, 0.726] | **0.846** |
| `lb950_deck0` | 11-19 | 36.7% | [0.219, 0.545] | 0.165 |
| `alakazam_rank5` | 7-23 | 23.3% | [0.118, 0.409] | 0.121 |
| `crustle_day1_rank1` | 6-24 | 20.0% | [0.095, 0.373] | 0.026 |
| `archaludon_cinderace` | 5-25 | 16.7% | [0.073, 0.336] | 0.132 |
| `lb950_deck2` | **0-30** | **0.0%** | SPRT reject | 0.091 |

**The only deck that survives is the one that is already our deck.**
`lb950_deck1` matches `field_0` at Jaccard 1.000, and its 56.7% over 30 games is
a near-mirror with 50% inside the interval. Win rate tracks Jaccard-to-Tech-Grim
almost monotonically, which is the signature of a prior effect, not a deck
effect.

**A 0-30 is not a deck losing, it is a pilot that has never seen the list.**
This is `CLAUDE.md`'s own warning arriving as data: a deck swap under a
deck-specialised prior measures the pairing, not the deck. Nothing here says
Archaludon is weak, and its author's 74.4%-over-1000-games claim is untouched by
this result.

So the deck lever cannot be pulled without training a prior per deck, and that
path has now failed twice: every deck swap above, and the `field_9` specialist
which tied locally at 28-32 and then settled at 556 on the ladder. Both remaining
levers are therefore the prior itself and the bracket draw.

## SUPERSEDED: the deck slot is NOT closed, it was never opened (2026-08-07)

The section above concludes the deck lever cannot be pulled. That conclusion was
drawn from six deck swaps under **the champion prior**, which
`tools/score_by_deck.py` has now shown scores Tech-Grim at 79.5% and every other
list at roughly chance. So all six swaps asked a Tech-Grim specialist to pilot a
deck it does not know, and all six measured that, not the deck.

Running the experiment with a prior that can pilot the alternatives gives the
opposite answer. One from-scratch model, trained on a **deck-balanced** Aug 1-6
elite corpus (708,405 decisions, no deck above 60k rows, no warm start from any
previous checkpoint), packaged three times behind the frozen shell with only
`deck.csv` differing, gauntleted against the champion at 0.25 s/move, 80 games
each, zero errors and zero step caps:

| arm | record vs champion | win rate | Wilson 95% |
|---|---|---:|---|
| new prior + **Dunsparce** | 47-33 | **58.8%** | [0.478, 0.689] |
| new prior + Mega Lucario | 39-41 | 48.8% | [0.381, 0.595] |
| new prior + Tech-Grim (**control**) | 34-46 | 42.5% | [0.323, 0.534] |

**The control is what makes this readable, and it is why the arm is worth
believing.** The same new prior on the deck we currently ship is *worse* than
the champion (42.5%), which is expected: it holds a tenth of the champion's
Tech-Grim data and gave up 1.9 top-1 points there by design. Holding the prior
fixed and changing only `deck.csv`, Dunsparce beats Tech-Grim **58.8% against
42.5%, a 16.3-point swing**. That is a deck effect measured with the pilot
controlled, which no gate in this file had previously achieved.

Two further things it settles:

- **Mega Lucario's 67.3% ladder win rate really was pilot, not deck.** Its
  mean pilot Elo is 1224. Under our pilot it lands at 48.8%, below Dunsparce,
  exactly as the pilot-confound caveat in the census section predicted. Deck
  win rates from the ladder must be discounted by who plays them, and mean
  pilot Elo is the available discount.
- **`harness/anchors` methodology held.** The reference was the champion
  archive, not our own previous attempt, so a 58.8% here is 58.8% against a
  thing with a real ladder history.

Caveats on the screen: 80 games puts 50% just inside the interval (0.478), and
0.25 s/move is the screening budget, which is measured to *exaggerate* gaps
(60% became 90% in the 2026-08-06 probe). It orders, it does not decide.
`harness/gate_scratch_decks.json`.

### CONFIRMED at the shipping budget: 155-85 = 64.6% over 240 games

`harness/gate_dunsparce_confirm.json`, 1.1 s/move, **zero errors, zero step
caps, zero draws**, 90 minutes:

| | record | win rate | Wilson 95% |
|---|---|---:|---|
| `sub_scratch_dunsparce` | **155-85** | **64.6%** | **[0.583, 0.704]** |
| `grpo_tech_grim_972_912_811` | 85-155 | 35.4% | [0.296, 0.417] |

**This is the first change in this project's history to separate from the
champion at the shipping budget.** Everything before it landed in 39-53%: the
fitted evaluator 46.7%, BC-init PPO 39.0%, elite-1200 43.8%, elite-1100 47.3%,
GRPO-v2 50.3%, the budget fix 49.2%, the opponent model 50.0%.

The read stayed stable as the sample grew, which is what the GRPO-v2 arm failed
to do (55.9% at 59 games decaying to 50.3% at 300):

| games | win rate | Wilson lower |
|---:|---:|---:|
| 47 | 68.1% | 0.538 |
| 81 | 66.7% | 0.559 |
| 162 | 62.3% | 0.547 |
| **240** | **64.6%** | **0.583** |

Submitted as ref **55339282** on 2026-08-07.

**Why this one worked when five careful prior changes did not**, and it is worth
being precise because the lesson generalises: every earlier experiment moved the
prior while holding the deck fixed, or moved the deck while holding the prior
fixed. `score_by_deck.py` showed those are not independent axes -- the champion
scores 79.5% on Tech-Grim and ~50% on everything else, so a deck swap under it
measures the prior's blind spot and a prior change on its own deck is confined
to a deck that is 47.9% field-weighted. Moving both at once is what escaped it.

The corollary is uncomfortable and should be kept in view: **the gain is
attributable to the deck, not to the network being smarter.** The same new prior
on Tech-Grim went 34-46 (42.5%), i.e. worse than the champion. We did not build a
better player; we built a player that can hold a better deck.

## Cycling decks without paying O(n^2): `tools/deck_cycle.py`

Twenty decks round-robin is 190 pairs, and at the 200 games a pair needed to
separate anything that is 38,000 games, which is weeks at the shipping budget.
The tool makes it affordable three ways, and validates decks against
`battle_start` first because `cabt` scores a malformed deck as an instant loss
rather than an error:

- **`--vs` anchored scheduling** (added to `ladder_harness.py`): every candidate
  plays the same fixed opponent, so scores stay comparable without candidates
  playing each other. O(n), not O(n^2).
- **SPRT** on each candidate (H0 p=0.5 against H1 p=0.55, alpha=beta=0.05), so
  settled candidates stop consuming games instead of running to a fixed budget.
- **A cheap budget for the screen only.** See the budget-invariance measurement
  below before trusting this; the full budget still decides.

Both warnings from the deck-gate section still apply and are repeated in the
module docstring: a deck swap under a deck-specialised prior is not a clean deck
test, and the absolute win rate against one fixed anchor does not transfer to
the ladder. Only the ordering does.

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

## The prior is SATURATED at 160d/5L, and that is the constraint (2026-08-07)

The elite-teacher experiment was built on a real observation: the champion's
prior was cloned from an Elo-1000+ corpus whose **mean is 1094**, so nearly half
its signal came from players between 1000 and 1100, while the top of the ladder
sits at 1284. Every teacher this project had used was some version of itself,
self-play distils our league and expert iteration distils our search, but a
1200-rated human shares none of our blind spots.

Shards already carry a per-decision `elo`, so `tools/filter_by_elo.py` makes
this a filter rather than a re-ingest:

| cut | decisions | share |
|---|---:|---:|
| Elo >= 1150 | 346,150 | 18.0% |
| Elo >= 1200 | 163,229 | 8.5% |
| Elo >= 1250 | 114,839 | 6.0% |

Both arms fine-tuned from the champion with a real temporal holdout (the Aug 2
shard, a day neither model trained on):

| arm | train decisions | mean elo | best epoch | held-out top-1 | weights moved | gate vs champion |
|---|---:|---:|---:|---:|---:|---|
| elite-1200 | 137,080 | 1261 | 3 of 10 | 67.4% | 8.07% | **35-45 (43.8%)**, Wilson [0.334, 0.547] |
| elite-1150 | ~320,000 | ~1230 | **1 of 12** | 65.2% | **1.89%** | see below |

**The 1150 arm early-stopped after a single epoch.** With 2.5x the data, the
champion's existing weights were already at essentially optimal validation loss
for elite human play, and every further epoch only overfit. That is the finding:
the constraint is not the quantity of data, and not its quality either, because
a mean Elo of 1261 against the champion's 1094 bought nothing.

Put beside every other attempt, the pattern is unambiguous:

| direction pushed from the champion prior | distance | result |
|---|---:|---|
| GRPO, 642 optimizer steps | 0.09% | 972, i.e. unchanged |
| elite-1150 BC | 1.89% | tie |
| elite-1200 BC | 8.07% | 43.8% |
| BC-init PPO, anchored | ref_kl ~0.30 | 39.0% |
| BC-init PPO, unanchored (v3) | unbounded | 8.3% |
| fresh Elo-1000 BC, full retrain | full | flat, 151-149 |

**Every direction is neutral-to-worse, and the damage scales with distance.**
That is the signature of a local optimum. A 160-dimension, 5-layer network
trained on roughly a million decisions has absorbed what it can hold, so
improving the prior by feeding it better decisions is finished as an avenue.

**The one axis never tried is capacity**, and it is shippable: the
`bc800_tech_grim_849` anchor is **192d/6L** and runs behind this same frozen
shell, so the packaging path already supports a wider network. That archive
scored only 848.9, but it was trained on Elo-**800**+ data, which confounds
capacity with data quality.

### Capacity is NOT the constraint either (measured 2026-08-07)

Two models, **identical data** (677,887 decisions at mean Elo 1156, Aug 2
holdout), identical schedule, differing only in width and depth:

| arm | best held-out CE | top-1 |
|---|---:|---:|
| 192d/6L, ~2.1M params | 0.9295 | 68.12% |
| 160d/5L, ~1.3M params, the champion's shape | 0.9387 | **68.28%** |

A 1.6x larger network bought 1% of cross-entropy and **nothing** on top-1. The
control was the point: without it, the gain below would have been credited to
capacity instead of to the data.

### What the constraint actually was: the champion's TRAINING DATA

Scoring every model on the same Aug 2 Elo-1100+ holdout, 40,000 decisions:

| model | top-1 | CE |
|---|---:|---:|
| **champion (the 972)** | **62.67%** | 1.1154 |
| new 160d/5L, from scratch on Elo-1100+ | **68.28%** | 0.9387 |
| new 192d/6L, same data | 68.12% | 0.9295 |
| elite-1150 (fine-tuned from the champion) | 65.10% | 0.9917 |

**The champion is 5.6 top-1 points worse at predicting strong play.** It was
cloned from an Elo-1000+ corpus whose mean is 1094, so it learned to imitate the
middle of the ladder, and it shows on elite decisions.

**And this explains why every fine-tune failed.** Warm-starting from the
champion traps the model in the champion's basin: elite-1150 moved 1.89% from
those weights and landed at 65.10%, exactly halfway between the champion and a
from-scratch model. The saturation recorded above was real, but it was
saturation *of that basin*, not of the architecture. Training **from scratch**
on elite data escapes it, which is the one thing none of the fine-tuning arms
could do.

### And it gated at 47.3%, which settles the whole question

`harness/gate_elite1100_160d.json`: **142-158 = 47.3%**, Wilson [0.418, 0.530],
300 games, zero errors. A tie with the point estimate behind.

So a model that is **5.6 top-1 points better at predicting strong human play**
is not a better search prior. Together with everything else, that is four
independent confirmations of one law, and it is the most robust result this
project has:

| change | its own metric improved | gate vs champion |
|---|---|---|
| fitted leaf evaluator | AUC 0.657 to 0.740 | 46.7% |
| BC-init PPO | beats the BC prior 64.3% | 39.0% |
| elite-1200 fine-tune | top-1 62.7 to 67.4 | 43.8% |
| **elite-1100 from scratch** | **top-1 62.7 to 68.3** | **47.3%** |

**Optimising any proxy for "picks the right move" does not improve this agent.**
Prediction accuracy, evaluator AUC, and standalone play strength have each been
pushed hard and each transferred as nothing. The prior's *ranking quality* is
apparently not the binding constraint, which means the remaining candidates are
how the search consumes that ranking (see the prior-temperature work below) and
the deck.

## Session summary, 2026-08-06

Everything below was measured this day. Nothing here is an argument.

**Two bugs found, both silent, both in our own measurement tools.**

- `ladder_harness.py` gave every archive the *same* `cg` package, so our shell's
  `from cg.engine import get_lib` raised against any competitor archive, got
  swallowed, and played with **no search and no error**. Champion went 0-8; after
  the fix, 2-2. No recorded gate was affected, because all of them mixed only our
  own archives.
- The same file built its job list and seat map in two loops that had to stay in
  lockstep, with nothing enforcing it.

**One real bug in the agent, not worth fixing.** It can never decline an optional
selection, which is legal on 12.98% of decisions and taken 3.92% of the time by
real players. Decline rate *falls* with skill (4.71% under 900 Elo, 3.36% above
1100), so the evidence that would justify the work is absent.

**Three things gated, three answers.**

| question | answer |
|---|---|
| do public rule-based agents beat us? | **no**, 85-35 (70.8%) against day-1 rank 1 and day-2 rank 5 |
| does BC-init RL at real scale beat the champion? | **no**, 78-122 (39.0%) pooled over 5 checkpoints |
| does any external deck beat Tech-Grim under our pilot? | **no**, and win rate tracks Jaccard-to-Tech-Grim, so it is a prior effect |

**The conclusion this file used to rest on was wrong.** "BC plus conservative
GRPO" credited RL for the 972; that GRPO stage was 26,642 decisions and **642
optimizer steps** at `lr=5e-6`, moving the weights 0.09%. The 972 is a BC model.
So RL had never actually been tried at scale from a good init, because
`train.py` had no way to load one. That is now built (`rl_osfp/bc_init.py`,
`--init`, `--ref-kl-coef`) and was run: 2,309,000 decisions, and the policy did
improve, beating the frozen BC prior **256-142 (64.3%)**. It was still a worse
prior behind the shell.

**The finding that ties it together:** the search agrees with its own prior on
**95.7%** of decisions, because `_gen_candidates` ranks candidates by the net and
keeps only the top 16. The prior is the agent; the search is verification. That
explains the failed evaluator refit, the failed budget change, the 80 to 25 drop
without a net, and it is why distilling the search into the net teaches nothing.

**Net position: unchanged and honest.** Nothing measured today beats the
champion. The champion is live at ref 55307993, 792.1, rank 975, still climbing.

## What to do next, in order

0. **`submission_dunsparce_scratch` is uploaded and climbing (ref 55339282,
   2026-08-07).** It gated 155-85 (64.6%) over 240 games at 1.1 s/move, the
   first archive ever to separate from the champion at the shipping budget. It
   is the newest submission, so it is what the board ranks. Run
   `tools/ladder_status.py` before any further upload.

0a. **The deck is now the proven lever, so keep it current.** The gain came from
   moving the deck and the prior *together*; the same prior on Tech-Grim gated
   42.5%. `tools/matchup_matrix.py` re-run before the deadline is cheap and
   guards the one known exposure: Dunsparce loses **9% over 173 games** to Mega
   Lucario, which is the strongest deck in the format and gaining share. If
   Lucario adoption rises materially, re-gate.

0b. **The obvious next target is a prior that can pilot Mega Lucario.** It has
   the best field-weighted win rate (63.9%) and the fewest BC decisions of any
   top deck (63,113), and our balanced prior only reached 48.8% with it. That
   scarcity is exactly what self-play manufactures, and `--async-rollout` now
   makes a run twice as productive per hour. Gate any resulting checkpoint BOTH
   behind the shell and standalone: bare-policy strength and prior quality have
   diverged twice.
1. **Re-roll the bracket. It is worth more than any change gated here.** The
   first-ten-games table above shows identical bytes scoring 942.3 and 701.5, so
   the draw is worth ~240 points while every model change measured in this repo
   was worth zero. The draw is readable after about ten episodes, which is ~2
   hours of play, and 5 slots a day times 10 days is ~50 tickets.

   The procedure, and it needs discipline rather than cleverness:

   - upload the champion, wait ~2 hours, read `rating after 10 games` from
     `tools/ladder_status.py`
   - **at or above ~800: STOP.** That draw converges near 900+, do not touch it
   - **below ~650: re-roll**, since that draw converges near 700
   - between the two, judge by how many days are left; early on, re-roll
   - **always keep 2 slots in reserve** so a bad final draw can be fixed

   The trap is rolling on the last day and getting a 5-5. Start rolling around
   2026-08-12, stop the moment a draw lands high, and leave it alone. Ref
   55307993 opened 5 episodes at **528.8**, which is a bad draw and should be
   re-rolled tomorrow when slots reset.
2. **Do not re-derive a search.** We have one that scored 972 and one that
   scored 405, and the gap is not recoverable by tuning constants.
3. **Stop editing the shell. All three changes lost.** The budget change lost
   8-15, the opponent model went 30-30, and the fitted evaluator went 56-64,
   the last two with byte-identical model and deck. The evaluator result is the
   decisive one: a leaf function that predicts the winner far better (AUC 0.740
   against 0.657, and 0.774 against 0.457 in the late game) bought nothing. The
   search's strength is not in `_evaluate`.
4. **Revive BC plus GRPO on fresh data.** Re-ingest the Aug 1-2 replays with Elo
   weighting, train a BC prior **specialised on the deck we intend to ship**,
   GRPO-refine it, and gate every iteration against the anchors. This is the only
   path that has ever produced a score above 850. Note the ceiling: the Tech-Grim
   version of exactly this returned 151-149 against the champion, and the
   `field_9` version returned 28-32 and then 556 on the ladder. Two deck-arms of
   this recipe have now failed to beat the champion.
5. **The deck question is still open but the cheap versions are exhausted.**
   `field_9` was the strongest target on paper and its specialist lost. What has
   never been run is the 972 recipe end to end on a non-Tech-Grim deck:
   `track6_controlled`'s controlled deck-specialist arms, not a fine-tune of a
   general trunk. That is the remaining untried idea with a real ceiling.
6. Fixing the v3 PPO deck mismatch (`tools/make_learner_pool.py` plus
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
