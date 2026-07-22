# pokemon TCG AI Battle Challenge

Agents for the Kaggle competition
[`pokemon-tcg-ai-battle`](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle).

Four tracks, in order of maturity. Track 1 is the live ladder agent. Tracks 2 and
4 are now implemented as self-play RL (results below); Track 3 remains a design.

> **This README is the project's source of truth and running memory.** The
> "Track 2 results" section below is the most recent work: a study of RL as
> search priors and a measured diagnosis of QR-SAC (2026-07-21).

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

| Submission | What it is | Score |
|---|---|---|
| ISO-A | search + network root priors | **819.8** |
| ISO-B | identical, no network at all | 775.7 |
| v4 | ISO-A base, model retrained on our own ladder games | 754.9 |
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
network placement that has ever paid off (see ISO-A 819.8 vs ISO-B 775.7). The
policy head is exported via `export_npz` and Track 1 reads it at the root only.

### RL-as-priors results — A/B vs the BC-prior agent

| Prior source | A/B win rate | Record | Note |
|---|---|---|---|
| **scratch-DMC** | **58.3%** | 14-10 (n=24) | best local; **submitted to ladder 2026-07-21** (result pending) |
| QR-SAC (warm start) | 43.3% | 13-17 (n=30) | |
| offline-DMC (136k logged) | 37.5% | | logged data only labels the *played* move, so it cannot rank alternatives |
| hybrid warm-start | 36.7% | | warm-starting a regression head from cross-entropy logits actively hurt (iter-1 loss 10.8 vs 1.5) |

The 58% vs 43% gap is **not statistically significant** (two-proportion z ≈ 1.1,
p ≈ 0.27; both samples are tiny). And local A/Bs are known to invert on the
ladder (see v3). So "QR-SAC is worse" is a confounded, noisy, 30-game read.

### Why QR-SAC underperformed — measured, not guessed

Prior sharpness measured on 5,053 real decision states (`bc_917eps.npz`), over the
**option tokens the deployed agent actually reads** (`main.py._net_scores`):

| Deployed option prior | norm. entropy | logit spread | top-1 == strong-play move |
|---|---|---|---|
| BC | 0.77 | 4.29 | 41% |
| scratch-DMC (winner) | 0.998 | **0.10** | 21% |
| QR-SAC | 0.99 | 0.30 | 18% |

1. **The winning prior is essentially uniform** (spread 0.10 ≈ 1/n). Both RL models
   failed to learn a *discriminative* option ranking. scratch-DMC wins not because
   its ranking is good, but because a near-uniform prior + strong search beats
   BC's sharper-but-imperfect prior — the same mechanism as the leaf-eval autopsy
   (a confident prior makes PUCT commit early; a flat one keeps it exploring).
2. **QR-SAC's sophistication is invisible at deployment.** The distributional
   critic, entropy target, alpha tuning, and risk machinery all collapse to a
   single scalar per option. Its critic Q-mean is *even flatter* (entropy 0.997,
   spread 0.26). It pays a large complexity cost for the same kind of flat prior
   DMC produces more simply — this is why "simplest won."
3. **It was never a controlled comparison.** scratch-DMC trained 90 iters × ~1875
   grad-steps (~170k updates); the original QR-SAC ran 60 × 400 (~21k, ~8× fewer),
   *and* warm-started, *and* fewer iters. Its actor loss was still monotonically
   improving at the last iteration — undertrained on gradient updates.

### Two implementation bugs in `qrsac.py` (found; fix pending)

1. **Actor normalization mismatch.** The actor loss softmaxes and entropy-targets
   over all ~17 board + hand + global + option tokens (`mask > 0.5`), but `act()`
   (data collection) and `_net_scores` (deployment) softmax over **option tokens
   only**. The actor optimized the wrong distribution — the prime suspect for the
   mushy option prior. Fix: restrict the actor loss and the `ln(n_act)` entropy
   target to option positions.
2. **Anchor over-flattens the critic.** The `--anchor` term pulls all ~16
   non-action tokens toward the state value, dominating the loss and collapsing
   the option-Q spread to ~0.1–0.3. Fix: restrict the anchor to option tokens.

### Done / next

- **DONE:** added `--epochs` to `qrsac.py` (ports DMC's 2-epoch buffer sweep; the
  original QR-SAC did one capped 400-step pass and was ~8× under-trained).
- **INTERRUPTED:** a controlled `qrsac.py --scratch --iters 90 --epochs 2
  --max-steps 2000` run (only the *algorithm* differing from the winner) was
  launched, then stopped by request around iter 20. No `model_qrsac_scratch.npz`
  was produced; the `qrsac_variant` model and all committed models are unchanged.
- **NEXT:** fix bugs 1 + 2 (option-restricted actor + anchor), re-run the
  controlled scratch experiment, and gate with **≥ 40 games** (30 is
  noise-limited). Success = option-agreement with strong play rising above BC's
  41%, or a clear A/B win.
- **If it still lands near-uniform / parity,** the conclusion is firm: no learned
  prior beats a from-scratch flat prior in this slot, and the real lever is
  **deck choice** — deck mining shows Alakazam at 48% field win rate vs Dragapult
  at 65%, a ~17-point gap that dwarfs any prior-source delta measured all session.

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
