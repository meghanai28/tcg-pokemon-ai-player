# pokemon TCG AI Battle Challenge

Agents for the Kaggle competition
[`pokemon-tcg-ai-battle`](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle).

Four tracks, in order of maturity. Track 1 is the live ladder agent. Tracks 2 and
4 are now implemented as self-play RL (results below); Track 3 remains a design.

> **This README is the project's source of truth and running memory.** The
> "Track 2 results" section below is the most recent work: a study of RL as
> search priors, a measured diagnosis of QR-SAC, and corrected replay imitation
> features (updated 2026-07-23).

```
track1_search/     determinized search + learned priors
track2_dmc/        Deep Monte Carlo, no search
track3_oracle/     oracle guided hidden info learning
track4_policygrad/ Delightful Gradient policy gradient

tools/             evaluation, mining, autopsy (shared)
data/              replays, leaderboard, official SDK
results/           A/B logs and training logs (the evidence)
```

---

## Track 1: determinized search (`track1_search/`)

The working agent. Samples possible worlds consistent with what we can see,
runs PUCT search inside each using the real engine, aggregates across worlds,
plays the most visited move. A small transformer supplies move ordering.

```
track1_search/
  agent/       what ships: main.py, deck.csv, model.npz, nn_features.py,
               nn_infer.py, cg/ (official SDK + safe loader)
  train/       ingest_episodes.py, selfplay.py, train_bc.py, exit_loop.py,
               model.py, test_parity.py, data_*/ (training shards)
  variants/    v1_frozen (no net baseline), nonet_variant, rollout_variant,
               smallnet_variant, dragapult_variant
```

Build and submit:

```powershell
cd track1_search\agent
tar -czf ..\..\submission.tar.gz main.py deck.csv nn_features.py nn_infer.py model.npz cg
cd ..\..
$env:KAGGLE_API_TOKEN="..."
py -m kaggle competitions submit pokemon-tcg-ai-battle -f submission.tar.gz -m "message"
```

Evaluate:

```powershell
py tools\run_local.py random 2                             # smoke test
py tools\ab_test.py track1_search\agent track1_search\variants\v1_frozen 24
py tools\gauntlet.py track1_search\agent 200               # field win rate
py tools\autopsy.py <submission_id> --out data\replays_ours  # what beat us
```

`PTCG_MAX_BUDGET` caps per move think time on both sides, for fast local A/Bs.

### What we learned the hard way

| Finding | Evidence |
|---|---|
| The network fails as a position evaluator | 5 A/Bs: 1-19, 0-6, 1-9, 11-13, 10-14 |
| Because it costs simulations, not because it is inaccurate | distillation improved value error 33 percent, changed game results by zero |
| Offline metrics do not predict playing strength | 53 percent move agreement, still lost 19 of 20 |
| Local A/Bs can invert on the ladder | a 9W-3L local gate produced a 65 to 40 percent ladder regression |
| Rollout leaves and heuristic leaves are equivalent here | 11-13 and 10-14 across two designs |
| Kaggle execs the agent with no `__file__` | first submission errored; `tools` now smoke test in exec mode |

The honest summary is that **search is doing the work** and the network has
earned only the cheap role (move ordering). Whether even that helps is being
measured right now by a live isolation experiment: two submissions identical
except for the presence of `model.npz`.

---

## Real ladder results

Every number below is a settled or in progress public score from the Kaggle
ladder, not a local estimate. Submissions seed at 600 and overshoot before
converging, so early readings are unreliable.

| Submission | What it is | Public score (2026-07-23 snapshot) |
|---|---|---|
| rich-BC search | fixed replay features + policy root priors, ref 54929162 | **600.0 initial seed; stabilizing** |
| ISO-A | search + network root priors | **819.8** |
| ISO-B | identical, no network at all | 774.6 |
| mixed QR-SAC v2 | exploratory iter-25 mixed prior, ref 54912732 | 722.5 |
| scratch-DMC priors | search + scratch-DMC root priors | 718.1 |
| scaled pure BC | 192d/6L argmax policy, no search, ref 54920652 | 615.9 |
| v4 | ISO-A base, model retrained on our own ladder games | 726.4 |
| v1 | first working agent, search + network priors | 739.2 |
| v3 | v1 plus 5 changes that regressed | 603.6 |

ISO-A and ISO-B are a controlled experiment: byte identical except for the
presence of `model.npz`. They were submitted seconds apart so they seed in the
same window against the same pool, which makes the difference between them
attributable to the network and nothing else.

That experiment has not been kind to quick conclusions. ISO-B led for several
hours and peaked at 902.7, which looked like clear evidence the network was
dead weight. As both converged the ordering reversed and ISO-A is now ahead.
The honest current read is that the network probably helps a little in the
cheap role, and that anyone reading either arm before convergence would have
concluded the opposite of the truth.

v3 is the cautionary tale. It bundled five changes that together won a local
A/B 9 to 3, then regressed on the ladder from a 65 percent win rate to 40
percent. Reverting the two riskiest changes produced v4. Local evaluation
inverted a real result by roughly 120 rating points.

---

## Track 2: Deep Monte Carlo + QR-SAC (`track2_dmc/`) — IMPLEMENTED

DouZero style. No tree search at all: learn Q(state, action) from Monte Carlo
returns of self play, spend all compute on generating data rather than on
lookahead. Motivated directly by Track 1's measured failure mode, where the
network and the search compete for the same CPU. See `track2_dmc/README.md`.

Two self-play learners are implemented:

- `train_dmc.py` — Deep Monte Carlo: regress `Q(s,a)` toward the self-play MC
  return, act by argmax. `--scratch` disables the BC warm start.
- `qrsac.py` — QR-SAC, built to `QRSAC_SPEC.md`: Q-learning core, a **separate**
  16-quantile distributional critic (never warm started from the policy logits),
  an entropy-targeted SAC actor with auto-tuned alpha, and an anchor regulariser
  on untaken options.

Both **deploy as root move-ordering priors** for Track 1 search — the only
network placement that has ever paid off (see ISO-A 819.8 vs ISO-B 774.6). The
policy head is exported via `export_npz` and Track 1 reads it at the root only.

### RL-as-priors results — A/B vs the BC-prior agent

| Prior source | A/B win rate | Record | Note |
|---|---|---|---|
| **scratch-DMC** | **58.3%** | 14-10 (n=24) | ladder result 718.1; local gate inverted |
| mixed QR-SAC v2, iter 5 | 50.0% | 12-12 (n=24) | leaderboard actor rehearsal + self-play critic |
| mixed QR-SAC v2, iter 25 | 45.8% | 11-13 (n=24) | best offline checkpoint; exploratory package prepared |
| mixed QR-SAC v1, iter 10 | 45.8% | 11-13 (n=24) | |
| mixed QR-SAC v1, iter 15 | 21.4% | 3-11 (stopped) | clearly failed search gate |
| QR-SAC (warm start) | 43.3% | 13-17 (n=30) | |
| offline-DMC (136k logged) | 37.5% | | logged data only labels the *played* move, so it cannot rank alternatives |
| hybrid warm-start | 36.7% | | warm-starting a regression head from cross-entropy logits actively hurt (iter-1 loss 10.8 vs 1.5) |

The corrected mixed learner repaired original QR-SAC's local regression to
rough parity, but did **not** beat the BC-prior agent. All 24-game confidence
intervals are wide, and local A/Bs are known to invert on the ladder (see v3).
There is no local evidence here for a 900+ score, much less 1900+. The exploratory
`submission_qrsac_mixed_v2.tar.gz` was submitted on 2026-07-22 as Kaggle ref
**54912732** and scored **722.5** in the latest 2026-07-23 snapshot. The scratch-DMC
result is another warning against over-reading local gates: its 14-10 local win
is only **718.1** on the same snapshot.

### Why QR-SAC underperformed — measured, not guessed

Checkpoint screening on 5,053 real decisions (`bc_917eps.npz`), restricted to
the **option tokens the deployed agent actually reads** (`main.py._net_scores`):

| Prior | top-1 | cross entropy | norm. entropy | logit spread |
|---|---:|---:|---:|---:|
| BC (`model_v4`) | **52.29%** | **1.2865** | 0.7521 | 3.902 |
| scratch-DMC | 30.83% | 1.7366 | **0.9982** | **0.056** |
| original QR-SAC | 25.75% | 1.8080 | 0.9828 | 0.309 |
| mixed QR-SAC v2, iter 5 | 46.51% | 1.5691 | 0.9636 | 0.686 |
| mixed QR-SAC v2, iter 20 | 49.65% | 1.4183 | 0.8136 | 2.607 |
| **mixed QR-SAC v2, iter 25** | 49.14% | **1.4117** | 0.7925 | 3.244 |
| mixed QR-SAC v2, iter 40 | 50.62% | 1.5244 | 0.6527 | 8.105 |

1. **The self-play prior is essentially uniform** (spread 0.10 ≈ 1/n) and learns
   no discriminative option ranking. Its 58% *local* win was a mirage: on the
   ladder scratch-DMC reached only **718.1 — below** the no-net baseline (774.6) and
   far below BC (819.8). So a near-uniform RL prior is not "good"; it is worse than
   the heuristic. Among priors, BC's specific profile (entropy ~0.75, spread ~3.9)
   is the ladder best, and both flatter (scratch-DMC) and over-sharp priors (below)
   underperform it. The earlier "flat is good" reading was a local-A/B artifact.
2. **QR-SAC's sophistication is invisible at deployment.** The distributional
   critic, entropy target, alpha tuning, and risk machinery all collapse to a
   single scalar per option. Its critic Q-mean is *even flatter* (entropy 0.997,
   spread 0.26). It pays a large complexity cost for the same kind of flat prior
   DMC produces more simply — this is why "simplest won."
3. **Leaderboard data is useful for the actor, not the critic.** Regressing Q on
   logged moves cannot rank unchosen alternatives and lost at 37.5%. Rehearsing
   their search distributions with an option-only cross-entropy actor loss raised
   QR-SAC agreement from 25.75% to about 49% without contaminating the Q target.
   **Ablation (2026-07-22) confirms the critic is the wrong source.** With the
   actor-normalization fixed but BC rehearsal OFF, the actor faithfully follows the
   critic and produces a *sharp but wrong* prior (spread 6.5, only 17% agreement
   with strong play) — confidently wrong, the exact leaf-eval failure mode. Every
   bit of correct ranking in the mixed learner comes from BC rehearsal, so its
   ceiling is **BC parity, not a win over BC.** To beat BC the self-play critic
   would have to rank options better than BC's policy, and it ranks them worse.
4. **The best optimization iterate is not the best prior.** By iter 40, agreement
   rose but spread reached 8.1 and cross-entropy regressed. Preserved five-iteration
   snapshots exposed the failure; selecting only the final loss would hide it.

### Implementation bugs fixed on 2026-07-22

1. Actor softmax, expected Q, entropy, and `ln(n_actions)` now use kind-3 option
   tokens only, matching collection and CPU deployment.
2. QR-SAC and DMC anchors now use only *untaken option tokens* and divide by the
   actual untaken-option count, not the transformer attention mask.
3. Defaults now match the serious training budget (90 iterations, two replay
   sweeps, 2,000-step cap), with CUDA training and CPU-compatible export.
4. Leaderboard/search replay is rehearsed only through the actor. Q/value labels
   still come from exploratory self-play outcomes.
5. Alpha tuning now optimizes `log_alpha` directly so it can recover from a
   near-zero temperature instead of losing its own gradient.
6. Five-iteration snapshots are retained, and the latest deployable `.npz` is
   refreshed, so an interrupted multi-hour run no longer loses every model.

### Remaining pitfalls

- Multi-select decisions greedily fill to `maxCount` and credit only the first
  sampled option. Both DMC and QR-SAC therefore learn a lossy factorization of
  combination actions.
- Every move receives the final ±1 return. This is unbiased Monte Carlo credit
  but extremely high variance across roughly 150 decisions per game.
- The actor deploys, while the distributional Q head is training-only. QR-SAC's
  extra machinery helps only indirectly through actor updates.
- Twenty-four games cannot resolve small prior deltas, and the ladder has already
  inverted a 9-3 local gate. Treat the exploratory package as an experiment.
- No learned prior tested here clearly beats scratch-DMC's nearly uniform prior.
  Deck choice remains the larger measured lever: Alakazam was 48% vs Dragapult
  65% in field mining, a ~17-point gap.

### Reproduce

```powershell
# Mixed QR-SAC: GPU training, CPU-compatible .npz export and 5-iter snapshots.
py track2_dmc\qrsac.py --device cuda --iters 40 --games 10 --epochs 2 `
  --max-steps 2000 --bc-data track1_search\train\data_bc --bc-weight 0.1 `
  --bc-samples 20000 --bc-batch 64 --save-every 5 `
  --out track2_dmc\model_qrsac_mixed_v2.npz

# Screen snapshots. Use --backend numpy for the exact competition CPU path.
py track2_dmc\eval_prior.py track2_dmc\model_qrsac_mixed_v2_iter*.npz `
  --limit 5053 --backend torch --device cuda
py track2_dmc\eval_prior.py track2_dmc\model_qrsac_mixed_v2_iter025.npz `
  --limit 5053 --backend numpy

# Deploy the selected checkpoint and run the local search gate.
Copy-Item track2_dmc\model_qrsac_mixed_v2_iter025.npz `
  track1_search\variants\qrsac_variant\model.npz -Force
$env:PTCG_MAX_BUDGET="0.1"
py tools\ab_test.py track1_search\variants\qrsac_variant track1_search\agent 24

# Package and submit the exploratory CPU agent.
cd track1_search\variants\qrsac_variant
tar --exclude='*/__pycache__' --exclude='*.pyc' -czf `
  ..\..\..\submission_qrsac_mixed_v2.tar.gz `
  main.py deck.csv nn_features.py nn_infer.py model.npz cg
cd ..\..\..
py -m kaggle competitions submit pokemon-tcg-ai-battle `
  -f submission_qrsac_mixed_v2.tar.gz `
  -m "exploratory mixed QR-SAC v2 iter25; local 11-13"
```

---

## Scaled pure-BC + deck selection (2026-07-22)

Prompted by the leaderboard's top agent: **pure imitation learning on ~21k games,
no search**, 3-4 h on one H200 — and their note that *"the same checkpoint can score
very differently just by switching deck."* We tested the thesis at ~1/3 that scale.

Correction after auditing the files: the 291,035-decision training directory was
the 5,266-game, Elo-1000-filtered daily shard (179,079 decisions) plus the older
917-episode shard (111,956 decisions). The separately ingested 2,091-replay
unfiltered shard was **not** in that directory. The model was **dim 192 / 6
layers, 2.064M parameters** (5.65x the 365k baseline, not ~4x).

**Results:**
- **Pure BC vs search, same deck: 4-36 (10%) locally.**
- **Leaderboard ref 54920652: 615.9** in the 2026-07-23 snapshot, versus 819.8
  for search + the old BC root prior. The local loss was a valid warning.
- **#2 deck gauntlet, SAME pure-BC checkpoint, field-weighted, 240 games each:**
  **Alakazam (our deck) 40.4% +/- 6%  vs  Dragapult 20.0% +/- 5%.**
  This uses the same learned pilot on both seats, so it measures model/deck fit,
  not a real field win rate. It still warns that deck choice is pilot-dependent.

Why its offline metric misled us:

- validation randomly split individual decisions, leaking neighboring states from
  the same games across train and validation;
- policy CE normalized over every state token even though deployment ranks only
  legal option tokens;
- the reported "heuristic 37.5%" gate was actually raw option 0, because replay
  ingestion did not heuristic-sort the options;
- validation CE was best at epoch 13 (1.2336) but the exported epoch-20 model had
  regressed to 1.5064; and
- most importantly, legal card options usually provide `area + index`, not
  `cardId`. The encoder ignored `index`, so many distinct cards were represented
  as the same zero-ID option. More data cannot repair a missing input.

---

## Rich replay policy + search (2026-07-23)

`nn_features_rich.py` fixes the action representation without changing the
53-token/32-scalar CPU ABI. It resolves cards from deck, hand, discard, active,
bench, prize, and attached-card references; records source/target indices,
effect cards, and remaining effect resources; and preserves engine option order
at inference. `ingest_episodes.py` now records stable episode and pilot IDs.

`train_bc.py` now:

- splits whole episodes (or pilots), not individual decisions;
- applies CE only to legal option tokens;
- supports policy-only training and critical-context weighting; and
- early-stops and exports the checkpoint with best held-out option CE.

GPU run: 5,266 daily episodes, Elo >= 1000, 179,079 decisions; 151,868 training
and 27,211 held-out decisions across disjoint episodes. The 128d/4-layer/4-head
policy has 717,698 parameters. Best epoch was 15/20:

| Metric | Old scaled pure BC | Rich BC |
|---|---:|---:|
| validation split | random decisions | disjoint episodes |
| held-out top-1 | 56.3% | **75.1%** |
| held-out option CE | 1.5064 at exported epoch 20 | **0.7125** at restored epoch 15 |
| local result vs ISO-A search | 4-36 as a pure policy | **19-5** as search priors |

The candidate retains the proven determinized search and uses the rich policy
only for root move ordering. Exact competition CPU packaging passed a smoke
match. Submitted as `submission_richbc_search.tar.gz`, Kaggle ref **54929162**.
Kaggle accepted it and assigned the normal **600.0 initial seed**; allow roughly
7-12 hours of matches before comparing it with stabilized agents.

### Reproduce rich-BC search

```powershell
# 1) Build corrected replay features directly from the compressed daily export.
py track1_search\train\ingest_episodes.py `
  data\replays_daily\pokemon-tcg-ai-battle-episodes-2026-07-01.zip `
  --out track1_search\train\data_bc_rich `
  --leaderboard data\leaderboard\pokemon-tcg-ai-battle-publicleaderboard-2026-07-21T07_12_03.csv `
  --min-elo 1000 --features rich --max-samples 400000

# 2) GPU training; the exported model still runs through NumPy on competition CPU.
py track1_search\train\train_bc.py `
  --data track1_search\train\data_bc_rich --features rich --split episode `
  --epochs 20 --patience 4 --batch 256 --lr 0.0006 `
  --dim 128 --layers 4 --heads 4 --value-weight 0 --critical-weight 1.5 `
  --device cuda --out track1_search\variants\richbc_search\model.npz

# 3) Controlled game gate against the 819.8 ISO-A code.
$env:PTCG_MAX_BUDGET="0.1"
py tools\ab_test.py track1_search\variants\richbc_search track1_search\agent 24
Remove-Item Env:\PTCG_MAX_BUDGET

# 4) Package the exact CPU agent. The archive must contain all files at its root.
tar --exclude='*/__pycache__' --exclude='*.pyc' -czf submission_richbc_search.tar.gz `
  -C track1_search\agent main.py deck.csv nn_features.py nn_infer.py cg `
  -C ..\train nn_features_rich.py `
  -C ..\variants\richbc_search model.npz

py -m kaggle competitions submit pokemon-tcg-ai-battle `
  -f submission_richbc_search.tar.gz `
  -m "rich-BC search: fixed area/index card resolution; 75.1% episode-heldout; 19-5 vs ISO-A"
```

Next scaling step: add non-overlapping high-rated days with the same rich
features, keep a final day and unseen pilots fully held out, and train deck-aware
or deck-specific policies. Do not increase model size again until held-out CE
stops improving at the current 718k-parameter scale. QR-SAC should remain an
ablation unless its self-play critic can beat this policy's option ranking.

---

## Track 3: Oracle guided learning (`track3_oracle/`)

Suphx style. Train with access to hidden information, then wean the model off
it. Attacks the weakest part of Track 1: hidden card sampling is currently
uniform, so every simulation runs in a world that is probably wrong. Its
cheapest payoff is a learned hidden card predictor that feeds Track 1's
determinization directly. See `track3_oracle/README.md`.

---

## Track 4: Policy gradient (`track4_policygrad/`)

[Delightful Gradient](https://arxiv.org/abs/2603.14608). The one track that
optimises directly for **winning** rather than for matching: everything we have
trained so far is supervised (behaviour cloning, distillation), so we have never
actually run a policy gradient. DG gates each update term by a sigmoid of
advantage times action surprisal, which targets the exact shape of our data,
where most of roughly 150 decisions per game are already solved and a few MAIN
phase decisions decide the outcome. See `track4_policygrad/README.md`.

**Implemented and run (2026-07-21).** Pure RL vs the heuristic: DG **collapsed**
(loss fell 80% while strength halved, 26.7% → 12.5%), while DMC under identical
conditions stayed stable (13.3% → 27.5%). The instability is policy-gradient
specific, not a property of the self-play setup — which is exactly why Track 2's
QR-SAC uses a Q-learning core rather than a policy gradient.

---

## Setup

```powershell
py -m pip install --no-deps kaggle-environments
py -m pip install jsonschema flask requests numpy torch kaggle kagglehub
```

Ladder notes: submissions seed at 600, provisional ratings overshoot, only the
two most recent submissions play ranked games, and results need roughly 24 hours
to settle. Read nothing from fewer than 100 games.

---

## Engine binaries are not in this repo

The `cg/` engine is licensed **PTCG-ABC-Competition-Use-Only** and is therefore
not redistributed here. To run anything, copy it in from `kaggle-environments`:

```powershell
py -m pip install --no-deps kaggle-environments
$src = (python -c "import kaggle_environments,os;print(os.path.dirname(kaggle_environments.__file__))") + "\envs\cabt\cg"
Copy-Item "$src\*" track1_search\agent\cg\ -Force
```

`cg/engine.py` (our per-process singleton loader) IS included; the native
libraries and the official SDK modules are not.
