# Track 5: resource-bounded GRPO

This track refreshes the proven deck-aware rich-BC policy on current ladder
replays, then fine-tunes it with Group Relative Policy Optimization. It is
deliberately conservative: terminal rewards are compared only within matched
matchup/seat groups, an exact KL penalty anchors the update to the refreshed BC
policy, and the result is deployed only as Track 1 root priors.

## Why this version of GRPO

LLM GRPO samples several answers to one prompt. Here, one group is several full
games with the same submitted deck, opponent archetype, and seat. Rewards are
standardized within the group. A clipped PPO ratio prevents large updates, and
an exact categorical KL penalty anchors every legal-option distribution to the
rich-BC reference model.

This does not use replay data as an RL critic target. Earlier QR-SAC/DMC tests
showed that a logged chosen move cannot rank the alternatives. Replays remain
useful for the BC initialization; new relative returns come from on-policy games.

## Updated-data submitted run

The first 48-game GRPO stage used the older July 1 and July 20-22 replay model.
The submitted refresh instead downloaded the official July 24-31 daily exports:

- July 24-30: 9,636 replay episodes processed and 1.12M ingested
  Elo-1000+-filtered rich decisions.
- July 31: 1,242 episodes and 160,030 decisions, kept temporal-holdout only.
- BC training: 825,000 balanced decisions over eleven training dates, 75,000
  July 31 validation decisions, 18-epoch budget, early stop at epoch 17, best
  checkpoint restored from epoch 13.
- Old BC on a 20k July 31 sample: 67.14% top-1, CE 0.9304.
- Refreshed BC on the same sample: 72.82% top-1, CE 0.7595.

The July 31 high-rated deck census found 7,475 qualifying appearances. Exact
Grimmsnarl represented 49.6%; the selected exact Garchomp list was fourth by
frequency with a 54% observed win rate. The top-20 meta file is training input
only and is not copied into the submitted search agent.

Each deck received ten iterations of ten matched groups with four games per
group: 400 games per arm. Optimizer state persists across iterations, and the
schedule rotates all current opponents through both seats.

| Arm | Active groups | Selected checkpoint | Holdout top-1 | KL | Kaggle ref |
|---|---:|---:|---:|---:|---:|
| Grimmsnarl | 78/100 | iteration 4 | 73.00% | 0.00157 | 55177242 |
| Garchomp | 57/100 | iteration 8 | 72.95% | 0.00260 | 55177256 |

Both submissions passed Kaggle self-play validation and entered the ladder at
the normal 600.0 initial rating.

## Safe default run

The defaults are sized for the measured WSL environment (23 GiB RAM, 14 logical
CPUs): at most 16 games and 6,000 stored decisions per iteration, six Torch CPU
threads, two training epochs, a 120-minute wall cap, and a checkpoint after
every iteration. Rollouts stay on CPU because one-state GPU inference is usually
slower; minibatch optimization automatically uses CUDA when available.

The tested WSL environment is isolated in `.venv`. To recreate it with uv:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r track5_grpo/requirements.txt
```

```bash
python track5_grpo/train_grpo.py
```

For a five-minute smoke test:

```bash
python track5_grpo/train_grpo.py --iters 1 --groups 1 --group-size 2 \
  --max-decisions 500 --epochs 1 --max-wall-minutes 5 \
  --out track5_grpo/model_smoke.npz
```

Train the Garchomp policy by changing the deck:

```bash
python track5_grpo/train_grpo.py \
  --deck track5_grpo/decks/garchomp.csv \
  --out track5_grpo/model_grpo_garchomp.npz
```

The bounded extended recipe used for the submitted arms was:

```bash
python track5_grpo/train_grpo.py \
  --init track5_grpo/model_bc_recent.npz \
  --deck track5_grpo/decks/grimmsnarl.csv \
  --opponent-meta track5_grpo/meta_decks_0731_top.py \
  --out track5_grpo/model_grpo_recent_grimmsnarl.npz \
  --iters 10 --groups 10 --group-size 4 --max-decisions 15000 \
  --epochs 3 --batch 128 --lr 3e-6 --clip 0.12 --kl-beta 0.06 \
  --entropy-beta 0.001 --temperature 1.05 --threads 6 \
  --max-wall-minutes 180 --device cuda --seed 917
```

The Garchomp arm changes the deck/output and uses seed 1231.

## Gates before submission

1. Require at least half of rollout groups to contain both wins and losses. If
   most groups have zero variance, increase matchup difficulty before adding
   games; more identical outcomes provide no GRPO signal.
2. Compare the checkpoint against the untouched rich-BC model with the same
   deck and both seats. Use at least 100 local games; the existing 24-game gates
   have repeatedly inverted on Kaggle.
3. Reject checkpoints whose mean KL keeps increasing or whose GRPO policy loses
   to the reference. Never select by training loss alone.
4. Submit the two deck arms separately: Grimmsnarl is the stable primary;
   Garchomp is the anti-Grimmsnarl hedge.

## Packaging

The builder validates the 160d/5L/5H architecture, both 60-card decks, required
engine files, and top-level Kaggle archive layout. Its defaults package the
selected Grim iteration 4 and Garchomp iteration 8 checkpoints:

```bash
python track5_grpo/build_submissions.py
```

Do not enable the opponent-posterior ensemble recovered from submissions
54989905/54989908. It was the only code difference from the 967.1 agent and
settled around 922 despite winning its local gauntlet.
