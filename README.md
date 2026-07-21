# pokemon TCG AI Battle Challenge

Agents for the Kaggle competition
[`pokemon-tcg-ai-battle`](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle).

Four tracks, in order of maturity. Track 1 is on the ladder. Tracks 2 to 4 are
designed and justified but not implemented.

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

## Track 2: Deep Monte Carlo (`track2_dmc/`)

DouZero style. No tree search at all: learn Q(state, action) from Monte Carlo
returns of self play, spend all compute on generating data rather than on
lookahead. Motivated directly by Track 1's measured failure mode, where the
network and the search compete for the same CPU. See `track2_dmc/README.md`.

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
