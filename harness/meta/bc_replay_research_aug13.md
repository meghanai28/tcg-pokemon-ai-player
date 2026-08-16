# Bug Catcher BC replay diagnosis — 2026-08-13

## What the matched data says

The regression is temporal memorization, not merely a changed CE definition.
All comparisons below use the same exact 60-card Bug Catcher deck and the same
semantic group-marginal loss for interchangeable card copies.

| Split | New epoch 1 | New epoch 2 | New epoch 3 |
|---|---:|---:|---:|
| Aug 12 training, all exact-deck decisions | 0.713 | 0.529 | 0.482 |
| Aug 13 untouched holdout, all exact-deck decisions | 0.784 | 1.013 | 1.274 |
| Aug 12 training wins, clean contexts | 0.677 | 0.280 | 0.125 |
| Aug 13 holdout wins, clean contexts | 0.814 | 1.149 | 1.549 |

“Clean” excludes duplicate-sensitive contexts 7, 8, and 22.  On those rows,
raw and semantic targets are identical.  Game-cluster bootstrapping on Aug 13
puts the epoch-2 minus epoch-1 winning CE change at +0.2475 (95% interval
[+0.2208, +0.2758]) and epoch-3 minus epoch-1 at +0.5608 ([+0.5215,
+0.6030]).  The clean-context epoch-3 change is +0.7348 ([+0.6758,
+0.7915]).

The model also becomes overconfident: seen clean-win top-1 rises from 75.4% to
95.7%, while untouched clean-win top-1 falls from 70.9% to 67.2% and confidence
rises from 71.6% to 88.7%.  The old BC run improves normally on that same clean
Aug 13 slice through epoch 4 (1.021, 0.900, 0.881, 0.831 CE).

This proves worse held-out imitation, not necessarily worse game strength.
Search can correct some prior errors, so the local battle gate remains the
final test.

## Why winner-only replay failed

The failed recipe sampled 1,069,604 auxiliary rows per epoch, with replacement,
from only 57,144 winning family rows: about 18.7 exposures per row per epoch.
Together with natural target rows and Elo/outcome/context weights, the family
contributed about 15.5% of policy-gradient mass—roughly 3.5–4 times the old
successful specialist recipe.  About 84% of the replay pool was a near-list
variant rather than the submitted exact list.

It also sent every replay row's `z=+1` through the shared value head.  Winning
value MAE therefore improved to 0.030 while action prediction collapsed.  On
seen Aug 12 rows, winning CE improved while losing CE worsened, showing direct
capacity reallocation toward the selected outcome.

## Relevant primary research

- [You Can't Count on Luck (NeurIPS 2022)](https://arxiv.org/abs/2205.15967)
  shows that conditioning/filtering on high realized returns can favor lucky
  actions in stochastic environments.  Pokemon has stochastic draws, starts,
  and opponents, so terminal victory is not an action-level quality label.
- [When Does Return-Conditioned Supervised Learning Work? (NeurIPS
  2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a2f65c9d2313b71005e600bd23393fe-Abstract-Conference.html)
  explains why top-return filtering discards useful coverage and needs strong
  assumptions to imply policy improvement.
- [Critic Regularized Regression (NeurIPS
  2020)](https://proceedings.neurips.cc/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html),
  [Self-Imitation Learning (ICML
  2018)](https://proceedings.mlr.press/v80/oh18b.html), and [Implicit
  Q-Learning (ICLR 2022)](https://arxiv.org/abs/2110.06169) weight logged
  actions using state-local advantage/value information, rather than declaring
  every action in a successful episode expert behavior.  IQL also clips its
  advantage weights.
- [Prioritized Experience Replay (ICLR
  2016)](https://arxiv.org/abs/1511.05952) treats nonuniform replay as a biased
  sampling distribution and uses stochastic prioritization plus correction;
  fixed winner-only replay was a much harder uncorrected shift.
- [Learning to Reweight Examples for Robust Deep Learning (ICML
  2018)](https://proceedings.mlr.press/v80/ren18a.html) uses a clean validation
  objective to decide which example gradients help rather than choosing a
  heuristic weight blindly.
- [Gradient Surgery for Multi-Task Learning (NeurIPS
  2020)](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
  motivates measuring broad-versus-target gradient conflict before adding a
  more complex multi-objective optimizer.
- [DAgger (AISTATS 2011)](https://proceedings.mlr.press/v15/ross11a.html)
  supports the longer-term direction: label states visited by the learned
  policy with the cap-16 search teacher, instead of relying only on states from
  historical pilots.

## Implemented bounded correction

- Kept random initialization, cap 16, the full broad corpus, temporal holdout,
  target-aware checkpoint selection, and duplicate-card semantics.
- Replaced copy smoothing with true semantic group-marginal CE and guarded
  unresolved/different-source options from accidental merging.
- Added epoch-wise shuffled replay with a hard 1–2 pass exposure cap; no row is
  repeated before the resident pool is exhausted.
- Switched the study to all outcomes, episode-balanced policy replay.
- Made auxiliary replay policy-only by default; broad rows still train value.
- Added exact target clean/win/loss metrics and replay exposure diagnostics.
- Added two-epoch 0%, ~1%, and ~2.5% effective-replay arms.  No candidate is
  packaged or uploaded until the held-out direction and battle gate agree.

The next algorithmic upgrade, after choosing a safe replay budget, is cached
cap-16 teacher-Q distillation on ambiguous states.  A logged action should be
upweighted only when its teacher advantage is positive (or should receive a
soft teacher distribution), with weights clipped.  That directly addresses
the observed Dawn-versus-Ultra-Ball loss without treating unrelated moves from
the same lost or won game as good or bad.
