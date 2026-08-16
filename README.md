# Pokemon TCG Kaggle agent (Hopefully Bronze or atleast 1000/6000 in Rankings)

Agent for Kaggle's `pokemon-tcg-ai-battle`. A submission is a tarball containing
`main.py`: the first callback returns a 60-card deck, later callbacks return
legal option indices.

The approach is behaviour cloning on high-Elo replays, used as the root prior
for a frozen PUCT search shell.

## Setup

The virtualenv and all training data were deleted after the competition to
reclaim disk. Recreate the environment with:

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
```

Then re-fetch and re-ingest replays with `tools/fetch_replays.py` and
`bc_train/ingest_episodes.py`. The census below records exactly what the shipped
models were trained on.

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
```

## Corpora used (deleted, census preserved)

Ingested from Kaggle daily replay archives, Elo 800+, Aug 1 to Aug 16 2026.
Training used a 5% episode-group holdout with seed 20260814 and `--min-elo 900`.

| corpus | shards | episodes | decisions | decks | mean Elo | Elo range | win rate | size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bc_bulk_aug11` | 119 | 50,528 | 6,823,198 | 398 | 1022 | 801-1234 | 52.7% | 910 MB |
| `bc_train_aug12` | 12 | 4,598 | 663,982 | 134 | 1030 | 800-1233 | 52.9% | 89 MB |
| `bc_holdout_aug13` | 8 | 3,920 | 448,898 | 67 | 1087 | 1001-1233 | 56.4% | 59 MB |
| `bc_rich_aug14` | 12 | 4,438 | 665,891 | 131 | 1058 | 802-1233 | 51.8% | 89 MB |
| `bc_rich_aug15` | 12 | 4,668 | 679,368 | 158 | 1052 | 806-1220 | 53.0% | 92 MB |
| `bc_rich_aug16` | 8 | 3,299 | 464,547 | 212 | 1023 | 802-1269 | 54.2% | 63 MB |
| `dagger_clean_train` | 420 | 420 | 16,342 | 1 | n/a | n/a | 55.9% | 7 MB |
| `dagger_clean_val` | 110 | 110 | 4,204 | 1 | n/a | n/a | 58.4% | 2 MB |

Total 9,766,430 decisions across 71,981 episodes. A further 57 GB of raw daily
archives under `data/fresh/` was the source and is re-downloadable from Kaggle.

The two `dagger_*` corpora are the perfect-info MCTS labels, one deck only, mean
teacher repeats 2.00 and mean q disagreement 0.0224. They are the one input that
was generated locally rather than downloaded, so regenerating them means rerunning
`tools/dagger_generate.py`. Given they measurably degraded the student, that is
unlikely to be worth it.

Regenerate this table with `tools/corpus_census.py`.

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

## What the ladder showed

| package | eps | score | opponent-adjusted |
|---|---:|---:|---:|
| lucario_rich_fixed | 50 | 907.7 | 918 |
| lucario_clean | 30 | 876.0 | 891 |
| lucario_rich_fixed_aug13 | 38 | 837.2 | 846 |
| bugcatch_clean | 12 | 546.6 | 543 |

- Scores converge slowly, so never call a result before ~40 episodes, and read opponent-adjusted rather than peak.
- `value_weight 0` beat the tuned value head; the aux head regressed on the bigger corpus and poisoned the shared trunk.
- A perfect-info DAgger teacher made the blind student worse: +0.21 target CE and -3.3 top-1 in one epoch (room for improvement though, I probably needed a better robustness filter). 
- Local gate was weak. I'd like to see how other competitors locally gated their models/agents for such high performance. I was struggling to do so efficently.
- Deck swaps did not help. I think I should've tried action chunking, the only deck that worked with my model was tech grim and lucario (maybe because not as many combos as dreepy or bug catcher which both failed). BC should work with even less data so I see this as a failure on my end.

## What I would do differently

- Run RL on a vectorized JAX engine; PPO hit 693,248 games and still scored 640, so the blocker was iteration speed on reward and curriculum, not game count. I saw people did this for Orbit wars. Is it applicable to TCG? Would be interesting to see.
- Rebuild with agents around one parameterized runner and a structured results store, not 60 one-off scripts and a dozen parallel tracks. I did go of track multiple times trying to get PPO to work. The changes convulted my codebase and made it hard to follow. 
- One variable per experiment, I think I changed too many in one go. 
- I suffered from data leakage into my validation set because of weird download methods.

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
