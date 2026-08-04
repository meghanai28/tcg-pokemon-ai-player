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

- `README.md` describes the **quarantined** tracks and is stale. Do not follow
  its commands or its "current recommendation"; it predates the reset.
- `tools/divergence.py`, `tools/run_local.py`, `tools/autopsy.py` still import
  paths under `track1_search/` and are **broken** until repointed.
- `foundation/model.py` is the old `TCGNet` and is dead code — nothing on the
  live track imports it. `rl_osfp/network.py` defines the current network.

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
  evaluate_population.py   checkpoint selection (gauntlet + round robin)
  evaluate_decks.py        deck selection gate for the chosen checkpoint
  build_submission.py      package model + deck into a tarball
  verify_submission.py     prove the packaged tarball matches the checkpoint
  agent_main.py   ships as main.py inside the tarball
  run/            checkpoints, metrics.json, selection reports
foundation/       shared engine bindings and feature encoders
  cg/             official native engine (libcg.so) + python bindings
  nn_features.py  base 53-token / 32-scalar encoder (fixed ABI)
  nn_features_rich.py  wrapper that resolves area+index card references
data/fresh/       mined deck pool + the replay/leaderboard archives it came from
tools/            deck mining and replay fetching (some entries are stale)
```

**Engine binaries are licensed "PTCG-ABC-Competition-Use-Only" and are
gitignored — never commit or redistribute `**/cg/*.so`, `sim.py`, `game.py`,
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
  evaluation games are genuinely independent — verified, not assumed.

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

| archive | what it is | status |
|---|---|---|
| `submission_mcts_track1.tar.gz` | earlier determinized IS-MCTS, recovered unchanged | only artifact with a live result: **972.0**, Kaggle ref 55185089 |
| `submission_search_puct.tar.gz` | new determinized PUCT on the native API, deck `field_1` | beats search-free policy 21-3 |
| `submission_osfp.tar.gz` | search-free policy, period 4 + `field_1` | superseded; keep as the baseline the others are gated against |

Build both search archives with `python -m rl_osfp.package_submissions`, then
verify each with `verify_search_submission.py`. The two search agents are
deliberately **not merged**: the track1 prior model is paired with its own deck
and encoder, and swapping either without retraining breaks that pairing and
destroys the one result we can point to.

Note the new search agent **does not use the trained network at all** — it is
pure search with heuristic priors and rollout leaves, which is what the measured
evidence supports. Adding root priors from an `rl_osfp` checkpoint is a
reasonable experiment, but gate it; it measured neutral before.

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
scores 85.1% — a 40-point swing, decided by which deck is forgiving of bad play,
not by which deck is good. `field_1`'s expert ladder win rate is only 54.4%.
`field_1`, `field_7`, `field_3` and `field_10` are statistically tied at the top
(85.1 / 84.9 / 84.8 / 82.6, Wilson ~0.79); `field_1` was taken on the primary
weighted metric plus a 0.0% stall rate.

Caveat on the run itself: per-period `approx_kl` is 0.0003-0.005 against a 0.04
target, so PPO is taking near-zero steps at `lr=7e-5` over ~3.5k decisions and
2 epochs. Early periods improve, then 10-12 regress. The ceiling here is set by
the learning rate and decision budget, not by which checkpoint gets picked — and
the deck slot is currently worth far more than the checkpoint slot.

## Ladder evidence (fetched 2026-08-04 via the Kaggle API)

Rank **313 / 6,224**, best score **972.0**, top of board 1,277.8. Deadline
**2026-08-16**. A second competition, `pokemon-tcg-ai-battle-challenge-strategy`
($240,000, deadline 2026-09-13), is entered but unranked.

Three facts from submission history that should govern decisions:

- **Search is worth roughly 300 ladder points.** The one search-free submission
  ever made (`submission_purebc_scaled`, "192d/6L, 291k Elo-1000 decisions, NO
  search, argmax policy") scored **591.1** — the worst result on record. Every
  search-based submission scored **873–972**.
- **Ladder score carries about ±75 points of noise.** `track9_control` was
  described as *byte-identical* to the 972.0 submission and scored **896.9**. No
  single submission can resolve a difference smaller than that, so a local gate
  over hundreds of games is more trustworthy than one ladder result. Treat 972
  as "somewhere around 900–970".
- **Deck choice moves the score as much as the model.** 967.1 vs 917.6 was the
  same model with a different deck; 972.0 vs 719.2 likewise. Re-gate the deck
  whenever the pilot changes.

`learner_0` is **Majkel1337's list — the rank-1 player** (1,277.8). Its 75%
ladder win rate conflates deck and pilot, but it is the top player's choice, not
noise. Measured under a *search* pilot it still only tied for last among
candidates, so gate it rather than assuming either way.

## Search is the dominant lever

**The engine has a native search API** — `SearchBegin` / `SearchStep` /
`SearchEnd` / `SearchRelease`, and every observation carries a
`search_begin_input` blob. It is built to be searched.

A search-free policy answers each decision with one forward pass (~26 ms)
against a ~600 s episode budget. Measured on this machine, `SearchStep` costs
0.026 ms — **39,000 engine steps/second**, so a 4 s decision slot buys ~156,000
engine steps. Shipping the bare policy leaves ~99% of inference compute unused.

Measured head-to-head, same deck both seats, only the decision procedure
differing: **search beat the search-free policy 21-3 (87.5%, Wilson 0.690)** over
24 games at just 0.5 s/move, with zero fallbacks and zero invalid actions.

The 2026-08-03 reset discarded replay imitation *and* search together. Only the
first was a good idea — search needs no replay data, so dropping it forfeited
the project's one measured live result for nothing.

### Search findings that are load-bearing

These were verified against this exact engine. Do not re-litigate them without
new measurements.

- **Throughput decides.** Heuristic leaves ran ~7,500 simulations where neural
  leaves managed ~220 — a 35x gap.
- **A learned value head makes search worse**: 1W-19L, then 0W-6L, against
  heuristic leaves. 46% of its outputs saturate above 0.95 on positions it never
  trained on, so PUCT commits to a line instead of verifying it against the
  engine. The engine cannot be miscalibrated the way a value head can. Use
  rollouts with engine-truth outcomes.
- **Network priors are neutral** (11W-9L) and cost ~2.5 ms, so they earn their
  place only at the root, where they decide which subtrees get explored at all.
- **A well-calibrated value head does not rescue neural leaves.** The v3 PPO
  head is genuinely trained — 97.3% sign accuracy, MSE 0.076 against a 1.000
  always-zero baseline, versus v1 at 47.4% (worse than chance). It is *not* the
  miscalibrated BC head that failed before. Even so, guided search with
  `prior_weight=0.7, value_weight=0.5` **lost to pure search 6-13 (31.6%)** at a
  matched 0.3 s/move deadline. Guidance is paid for out of the search budget:
  a value call per simulation buys accuracy at the cost of simulation count, and
  simulation count is what wins. Note the head saturates on 77.5% of positions,
  so it discriminates poorly between sibling moves even when its sign is right.
  Any guidance experiment must be gated head-to-head against unguided search at
  a matched deadline, never against the search-free policy — pure search already
  beats that 87.5%, so the comparison is at ceiling and resolves nothing.
- **Keep the evaluator's magnitude small.** Its job is breaking ties between
  sibling moves. A "lethal awareness" term large enough to dominate the search
  regressed the ladder from 65% to 40%.
- **Every `SearchStep` mints a new persistent `searchId`.** Release them or a
  long search leaks the engine arena; `Engine.end()` frees a whole move at once.
- **Determinization must produce exactly-sized zones** or `SearchBegin` rejects
  the world and search silently degrades to a fallback. A setup-turn world also
  needs a basic Pokemon in the deck.

### Verifying a search agent

A search agent **fails quietly**: every failure path returns a legal heuristic
action, so a completely broken search still yields a well-formed archive that
plays legal games to completion. "It ran without raising" proves nothing.
`verify_search_submission.py` therefore also asserts a per-decision latency
floor — below it, the agent is falling back rather than searching — and that the
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
  competently. What transfers is the *ordering* — our agent holding `field_1`
  versus holding `learner_0`, against identical opposition — and that is exactly
  what the deck slot should optimize. Do not quote the absolute number as an
  expected ladder result.
- **Always allocate deck-gate games by ladder share (`--share-budget`).** The
  field is wildly unbalanced: `field_0` alone is 46.6% of appearances, so a
  share-weighted score is dominated by a handful of matchups while uniform
  allocation spends most of its games where they barely move the answer. This
  is not theoretical — under uniform allocation `learner_0` swept `field_0` 6-0
  and looked 16 points better overall; at 61 games the same matchup was 34-27
  and the entire advantage evaporated. Six games decided nothing.
- **The gauntlet screens, the round robin decides.** Two distinct schedule
  asymmetries hit gauntlet mode, and they push in opposite directions. A panel
  member accumulates results from serving as an opponent against the whole
  candidate field (weaker on average), which inflates it — `credit_row` fixes
  this by crediting the candidate side only. But a panel member also *skips its
  own matchup*, so it never faces itself while every other candidate must; that
  residual bias is not fixable within the mode. Only the symmetric round robin
  settles a pick. On the first clean run the screen ranked period 9 first and
  period 4 fourth; the runoff put period 4 first at 67.2% (Wilson 0.588) versus
  0.431 for the runner-up. The screen still earned its keep — it surfaced
  periods 5, 8, 9, 11, which the original 5-checkpoint round robin never
  evaluated.

## Submission packaging

`build_submission.py` copies `agent_main.py` to `main.py`, the chosen `.npz`,
both feature encoders, `nn_infer.py`, and the whole `cg/` directory, and writes
`deck.csv`.

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

## Conventions

- One file, one job; module docstrings explain *why*, not what.
- Failures are surfaced, never swallowed: `arena.py` returns engine faults in
  `GameResult.error` so they land in metrics rather than silently biasing a
  win rate.
- Guard resources explicitly. `train.py::resource_guard` refuses configurations
  that would exhaust RAM or disk before they start.
- Evaluation scripts write a JSON report next to the checkpoints and print a
  table; they warn loudly rather than quietly picking when criteria disagree.
