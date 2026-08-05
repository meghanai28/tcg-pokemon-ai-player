# Pokemon TCG AI agent

Agent for the Kaggle [`pokemon-tcg-ai-battle`](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
ladder. A submission is a tarball with a `main.py` exposing
`agent(observation) -> list[int]`. The runner calls it once with `select == None`
to get a 60 card deck, then once per selection callback.

Use `.venv/bin/python` and run everything from the repo root as a module.

## Where things stand

Best real result is **972.0**, from `harness/anchors/grpo_tech_grim_972_912_811.tar.gz`.
That exact archive was submitted three times and scored 972.0, then 911.9, then
810.8, so treat any single ladder score as worth about plus or minus 160 points.

The agent that scores is BC plus conservative GRPO supplying root priors to a
determinized PUCT search over the engine's native `SearchBegin` / `SearchStep`
API. Search is worth roughly 300 points. Every search free submission we have
made scored between 480 and 591.

## The two rules that matter

**1. Gate against the anchors, never against ourselves.** `harness/anchors/`
holds packaged tarballs named for the ladder score each one actually earned.
Every serious mistake in this project came from comparing our own components to
each other, which can look great while the whole family sits below a baseline
nobody tested.

**2. Gate at 1.1 seconds per move.** A cheap harness run does not just add noise,
it inverts the answer. At 0.1 s/move the harness ranked our worst agent (real
score 480) above our best (810 to 972) by 7-1. At 1.1 s/move the same two go
10-0 the correct way. A small budget starves search and leaves a search free
agent untouched.

## Measuring

```bash
.venv/bin/python tools/ladder_harness.py \
  --archives harness/anchors/*.tar.gz artifacts/<candidate>.tar.gz \
  --games-per-pair 20 --budget 1.1 --workers 10 --calibrate
```

Plays real tarballs against each other through `kaggle_environments.make("cabt")`,
the same environment that scores the competition, and fits Bradley-Terry ratings.
`--calibrate` regresses those against the anchors' known scores and prints the
rank correlation, so you can see whether the harness reproduces an ordering we
already know before trusting it on one we do not. Details in `harness/README.md`.

Do not run other heavy jobs at the same time. Workers are pinned to one thread
each so per move budgets stay comparable across matches.

## Building a submission

```bash
.venv/bin/python -m rl_osfp.build_bcsearch_submission \
  --model rl_osfp/run_v3/model_period_180.npz \
  --deck-csv foundation/deck_tech_grim.csv \
  --out artifacts/submission_x.tar.gz

.venv/bin/python -m rl_osfp.verify_bcsearch_submission --archive artifacts/submission_x.tar.gz
```

`foundation/search_shell_main.py` is the frozen `main.py` that scored 972. It is
byte identical across all three of our best archives and is never edited. The
builder refuses to run if its md5 changes. To put a different network behind it,
swap the files around it, not the shell.

**Always verify.** The shell catches network errors and quietly falls back to
heuristic priors, so a dead checkpoint still produces a well formed archive that
plays legal games at normal latency. The verifier asserts the priors actually
fire.

## Layout

```
rl_osfp/       live track: PPO inside optimistic fictitious play
  train.py     the training loop
  build_bcsearch_submission.py / verify_bcsearch_submission.py
  run_v3/      checkpoints and selection reports
foundation/    engine bindings, encoders, and the frozen search shell
  cg/          official native engine, licensed, gitignored, never commit
harness/       anchors with known ladder scores, plus reports
tools/         harness, deck mining, pool building
data/fresh/    deck pool and the replay and leaderboard archives behind it
```

## Gotchas that cost real submissions

- The runner execs `main.py` as a string, so `__file__` is undefined, and it
  pops the agent dir off `sys.path` before `agent()` is called. It also takes
  the **last** callable defined in the module.
- `cabt` awards the game to the opponent on any rejected `select`, with no
  retry. Validate index range, duplicates, and min and max count.
- Never ship `cg/__pycache__`. A stray 3.12 `.pyc` failed episode validation.
- Read the per move budget from `remainingOverageTime`. Do not hardcode it.
- Step caps are not errors. Counting them as errors rewards stalling, because a
  policy that refuses to commit gets its stalls deleted from the denominator.
- `train.py` only ever pilots `learner_decks` and only ever faces `field_decks`.
  A deck listed only in the field is one the policy has never played.

`CLAUDE.md` has the full history, the measured findings, and the open questions.
