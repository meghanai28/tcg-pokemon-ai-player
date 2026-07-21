# QR-SAC build specification

Everything below is a response to a measured failure from 2026-07-21, not a
design preference. Logs for each claim are in `results/`.

## Placement: root priors, nothing else

| Role | Calls per decision | Measured outcome |
|---|---|---|
| Leaf evaluation | hundreds | fails, 5 A/Bs: 1-19, 0-6, 1-9, 11-13, 10-14 |
| Rollout policy | thousands | worse |
| **Root priors** | **1** | **works: ladder 819.8 with vs 775.7 without** |

An engine step costs about 0.05 ms; a network call about 3 ms. Any placement
that calls the network more than once per decision burns roughly 60 simulations
per call, and we showed a 33 percent better value head changed game results by
exactly zero. Do not put the network at the leaves again without first making
inference an order of magnitude cheaper.

## Required design decisions

1. **Q-learning core, not policy gradient.**
   DG collapsed with proper optimisation: loss fell 80 percent while strength
   halved, 26.7 to 12.5 percent against the heuristic. DMC under identical
   conditions stayed stable, 13.3 to 27.5 percent. The instability is policy
   gradient specific, not a property of the self play setup.

2. **Separate Q head, or rescale at init. Do NOT warm start a regression head
   from classification logits.**
   The hybrid warm started a policy head (large cross entropy logits) and then
   regressed it toward returns in [-1, 1]. First iteration loss was 10.8 versus
   1.5 from scratch, and it finished at 36.7 percent versus scratch's 58.3
   percent. The warm start actively hurt.

3. **Entropy target, not a fixed bonus.**
   A fixed 0.01 entropy bonus did not prevent DG's collapse. SAC's constrained
   formulation targets an entropy level, which is the point.

4. **Trust region.** PPO style clipping, or SAC's equivalent constraint. DG had
   nothing bounding how far the policy moved per update, which is the most
   likely proximate cause of the collapse.

5. **Distributional critic over quantiles.**
   Returns are bimodal at plus or minus 1 with heavy variance from coin flips
   and shuffles, so a mean is a poor summary. Optional payoff: rank actions by
   a risk adjusted quantile, taking variance when behind on prizes and avoiding
   it when ahead.

6. **Self play exploration, not offline replay, for the Q head.**
   Offline Q regression on 136,901 logged decisions scored 10.6 percent top-1
   and lost its A/B at 37.5 percent. Logged data contains only the move that
   was played, so it can teach "was this good" but never "which of these is
   best". Epsilon greedy self play supervises the alternatives. Keep the anchor
   regulariser (`--anchor`) that pulls untaken option outputs toward the state
   value; it is currently confounded with the bad warm start and untested alone.

7. **Opponents are the mined meta decks** (`META_DECKS` in the agent, auto
   mined by `tools/mine_decks.py` from 2,091 ladder replays), not a mirror.

## Optimisation hygiene, learned the hard way

- Do many gradient updates per rollout. The first DG run did **25 for the entire
  run** because it accumulated losses and stepped once per iteration; the loss
  was flat and I wrongly concluded RL could not learn here.
- Cap gradient steps per iteration (`--max-steps`). Sweeping a growing buffer
  took iteration time from 64 s to 217 s and projected to 10 hours.
- Batch the forward passes.

## Checkpoints

- Offline: does the Q head rank held out actions better than the heuristic?
- Game level: `py tools/ab_test.py <variant> track1_search/agent 30`
- Ship only on a win, and remember a 9W-3L local gate once inverted into a
  ladder regression from 65 to 40 percent. The ladder is the judge.

## Harness traps that cost hours

- Kaggle `exec`s the agent with **no `__file__`**. Validate in exec mode.
- `kaggle_environments` inspects agent arity and passes `config` as a second
  positional argument, silently breaking a two parameter agent.
- A seat returning an illegal action yields `reward = None`, not a loss.
- Search states are rendered for **whoever is to move**, so `players[me].hand`
  is None on opponent nodes. Encoding those from our own perspective fed the
  network a phantom empty hand on 48 percent of inputs. This bug class has
  appeared three times.
