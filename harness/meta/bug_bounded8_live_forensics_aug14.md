# Bug bounded-8 live forensics (2026-08-14)

## Result in context

| submission | public replays analysed | W-L-D | score snapshot | opponent-adjusted performance |
|---|---:|---:|---:|---:|
| old Bug w6 (`55493594`) | 27 | 18-9-0 | 901.7 | 931 |
| new Bug bounded-8 (`55503251`) | 43 of the first 45 | 28-14-1 | 873.9 | 901 |
| Lucario (`55495278`) | 56 | 36-20-0 | 970.5 | 975 |

The new and old Bug both won exactly two thirds of decided games. The bounded
run's large exact-deck holdout improvement therefore did not become a live
improvement. Its lower raw score was partly bracket/field placement (the new
Bug's opponents averaged roughly 30 Elo below the old Bug's), but its
opponent-adjusted performance was also about 30 points lower. This is nowhere
near evidence for a 300-point gain.

## Shipped encoder loses the core Bug Catching Set choice

The shipped `nn_features_rich._zone_card` handles both source area `DECK=1`
and `LOOKING=12` with `select.deck`. Real LOOKING observations instead have
`select.deck = null` and put the revealed cards in `current.looking`. Every
selectable revealed card is consequently encoded with card ID zero.

This is not a rare generic edge case. It is the choice produced by Bug
Catching Set, the deck's four-copy engine:

| agent | LOOKING decisions | all substantive decisions | share |
|---|---:|---:|---:|
| old Bug | 60 | 3,046 | 1.97% |
| new Bug | 106 | 4,906 | 2.16% |
| Lucario | 0 | 6,317 | 0% |

That is about 2.4 blind, strategically important choices per Bug game and none
per Lucario game. In the 106 new-Bug states, 383 selectable option identities
and 194 played option identities can be recovered directly from
`current.looking`; the shipped encoder recovered none.

On the 742 exact-Bug Aug-13 holdout LOOKING rows, new versus old positional
imitation CE was 1.192 versus 1.199: only a 0.6% gain, while the headline exact
target CE improved about 6.4%. The offline gain was therefore not fixing this
deck-defining decision.

The training-side correction must also be copied into a versioned deployment
encoder. At the time of this audit, `bc_train/nn_features_rich.py` has been
corrected, but `foundation/nn_features_rich.py` and the hash-frozen
`grpo_prior/champion/nn_features_rich.py` still contain the shipped bug. The
global reveal-count feature should likewise use `current.looking` for area 12.

## Opponent belief library is stale

The frozen shell's 12 opponent lists omit current Ogerpon-only, Lucario,
Dunsparce/Lopunny and several newer variants. Maximum multiset overlap between
the real opposing 60 and any shipped belief list was at most 30 cards in:

- 7/14 new-Bug losses (50%) and 9/28 wins;
- 2/9 old-Bug losses; and
- 7/20 Lucario losses.

The seven poorly modelled new-Bug losses included both Ogerpon-only losses,
both Lucario-list losses, a Dunsparce/Lopunny loss and two novel variants. On a
70-state loss sample, an exact-opponent search disagreed with the played move
in 34% of states from poorly covered matchups versus 20% from near-exact
matchups. Q gaps were generally small, so this supports refreshing the belief
model, not blindly copying every counterfactual action.

## Matchup signal

Combining the old and new Bug public samples gives the least noisy Bug view:

| opposing family | Bug W-L-D | Lucario W-L |
|---|---:|---:|
| Grimmsnarl/Froslass | 24-4-1 | 9-7 |
| Alakazam | 6-8-0 | 6-3 |
| Dragapult | 6-2-0 | 0-3 |
| Lucario | 4-4-0 | 3-1 |
| Ogerpon-only | 1-2-0 (new Bug only) | 3-0 |

Samples remain small and pilot strength differs, but the pattern is coherent:
Bug has a real favourable niche into Grimmsnarl and Dragapult, while Alakazam
and the high-rated Ogerpon-only list are holes. Lucario is more balanced but
was swept by three high-rated Dragapult opponents. Deck selection should be a
field-weighted decision, not a conclusion from the 22-18 local head-to-head.

## Corrected decision-level audit

The original regret script counted unvisited candidate edges as Q=0, creating
fake regret whenever a visited action had negative Q. After requiring positive
visits:

- 70 sampled loss states at 0.35 s: 18 teacher disagreements, zero cap misses,
  mean covered Q regret 0.0196, one state at least 0.10;
- 70 sampled win states: 8 disagreements, zero cap misses, mean regret 0.0090;
- 42 loss states at the deployed 1.1 s budget: 9 disagreements, zero cap
  misses, mean regret 0.0182, two states at least 0.10.

One high-regret state was a hidden-prize identity choice and is not learnable
from the information state. The other (Bug Catching Set versus Lillie's
Determination while behind on prizes) was not a stable teacher label: five
reruns selected Bug Catching Set at prior temperature 1, temperature 2 split
between the two cards, and temperature 3 mostly attacked instead. The current
search is not reliable enough to relabel every learner state indiscriminately.
Use agreement across seeds/budgets/temperatures as an abstention gate.

Cap 16 is not the diagnosed problem: 228/4,906 live decisions had more than 16
options, only 38 had more than 24, the live agent never played an option index
at least 24, and both regret audits had zero candidate misses.

## Highest-value next experiment

1. Version the encoder/shell, resolve LOOKING from `current.looking`, refresh
   opponent beliefs, and add a real-replay regression test for both training
   and packaged inference.
2. Re-ingest the raw corpus and train from scratch. Do not warm-start across the
   feature change.
3. Give corrected LOOKING/Bug Catching Set and other rare critical contexts an
   explicit sampling quota; train multi-select decisions with action-tuple
   targets rather than only option marginals. Keep hidden prize copies
   invariant.
4. Prefer high-Elo human actions for ordinary BC. Add DAgger/search targets
   only when an independent teacher is stable; abstain on hidden-information
   and temperature-sensitive states.
5. Gate on paired seeds against Grimmsnarl, Alakazam, Dragapult, Ogerpon-only,
   Lucario and Dunsparce. Report per-matchup Wilson intervals and
   opponent-adjusted strength, not a single raw Elo opening.

