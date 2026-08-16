# Pokemon TCG Kaggle agent

Agent for Kaggle's `pokemon-tcg-ai-battle`. A submission is a tarball containing
`main.py`: the first callback returns a 60-card deck, later callbacks return
legal option indices.

The approach is behaviour cloning on high-Elo replays, used as the root prior
for a frozen PUCT search shell. Use `.venv/bin/python` from the repo root.

## Layout

```text
agent/champion/             the frozen search shell that actually plays
agent/train_prior.py        BC trainer, also runs the DAgger distil via --teacher-data
agent/build_submissions.py  packages a prior behind the frozen shell
agent/decks/                deck lists
bc_train/                   replay ingestion, model, shipping-compatible encoders
foundation/                 engine bindings and inference support
tools/                      data fetch, DAgger generation, gating, submission
harness/anchors/            the protected champion archive
harness/meta/               deck census and submission markers
data/                       training shards and DAgger corpora
```

## Pipeline

```bash
# 1. fetch and ingest replays
.venv/bin/python tools/fetch_replays.py --help
.venv/bin/python -m bc_train.ingest_episodes --help

# 2. train the BC prior
.venv/bin/python agent/train_prior.py \
  --data data/bc_bulk_aug11 data/bc_train_aug12 data/bc_holdout_aug13 \
         data/bc_rich_aug14 data/bc_rich_aug15 data/bc_rich_aug16 \
  --holdout-group-frac 0.05 --deck agent/decks/mega_lucario.csv \
  --features rich --dim 320 --layers 8 --heads 8 --d-ff 640 \
  --epochs 10 --min-elo 900 --device cuda --out runs/model_prior.npz

# 3. generate DAgger labels, then fine-tune with --init and --teacher-data
.venv/bin/python tools/dagger_generate.py --help

# 4. gate, then submit
.venv/bin/python tools/ladder_harness.py --help
tools/submit_with_backoff.sh <package.tar.gz> <marker.json> "message"
.venv/bin/python tools/ladder_status.py --limit 10
```

## What the ladder actually showed

| package | eps | score | opponent-adjusted |
|---|---:|---:|---:|
| lucario_rich_fixed | 50 | 907.7 | 918 |
| lucario_clean | 30 | 876.0 | 891 |
| lucario_rich_fixed_aug13 | 38 | 837.2 | 846 |
| bugcatch_clean | 12 | 546.6 | 543 |

Hard-won lessons, in rough order of value:

- Scores converge slowly, so never call a result before ~40 episodes, and read opponent-adjusted rather than peak.
- `value_weight 0` beat the tuned value head; the aux head regressed on the bigger corpus and poisoned the shared trunk.
- A perfect-info DAgger teacher made the blind student worse: +0.21 target CE and -3.3 top-1 in one epoch.
- Local gates do not predict the ladder; one candidate won 155/240 locally and still failed live.
- Deck swaps did not help; the census already had the deck in use at the top for high-Elo seats.

## What I would do differently

Directions, not results. None of this has been measured here.

- Run RL on a vectorized JAX engine; PPO hit 693,248 games and still scored 640, so the blocker was iteration speed on reward and curriculum, not game count.
- Rebuild with agents around one parameterized runner and a structured results store, not 60 one-off scripts and a dozen parallel tracks.
- One variable per experiment; the only question fully resolved was settled by a 2x2, and every single-run verdict stayed ambiguous for days.
- Deduplicate on episode id at ingest, since group ids come from the filename and one episode from two sources straddles the holdout.
- Budget ~40 episodes per evaluation and gate against a known-score anchor instead of a local head-to-head.

## Protected control

`harness/anchors/grpo_tech_grim_972_912_811.tar.gz`, SHA-256
`4fdb2ecf444d58161430fcaacb84795e5cd7f51ed2b756f225385493618e2f12`.
`build_submissions.py` verifies that archive plus every frozen shell, encoder,
and engine file. A candidate may replace only `model.npz` and `deck.csv`.

Training and shipping ABI is `bc_train/nn_features*.py`, MAX_OPT24, SEQ53.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
