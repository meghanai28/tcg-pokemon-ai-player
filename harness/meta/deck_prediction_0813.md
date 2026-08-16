# Pre-registered prediction, 2026-08-13

Written before the three-deck gate ran, so the result cannot be rationalised
after the fact. Two hypotheses explain why one deck's prior outperforms
another's, and they disagree about the ordering.

## H1: deck data share drives the result

The prior is only as good as the number of target-deck decisions it saw.
Board win rate is a statement about other pilots, not about our agent.

| deck | rows | share | effective @ w=2 |
|---|---|---|---|
| lucario | 60,536 | 1.71% | 3.36% |
| slowpoke | 68,603 | 1.01% | 1.99% |
| zorua | 19,278 | 0.28% | 0.56% |
| dreepy | pending (3,460 board seats, most of any pick) | expect highest | |

**H1 predicts:** dreepy >= lucario > slowpoke > zorua

## H2: board win rate drives the result

The census measures deck strength, and a deck that wins 64% in others' hands
should win behind our search shell too.

| deck | board WR | Wilson lower |
|---|---|---|
| slowpoke | 64.4% | 61.2% |
| dreepy | 59.2% | 57.8% |
| zorua | 59.2% | 54.0% |
| lucario | 57.9% | 56.0% |

**H2 predicts:** slowpoke > dreepy ~ zorua > lucario

## The discriminating case

H1 and H2 order **lucario and slowpoke oppositely**. H1 puts lucario ahead on
data share; H2 puts slowpoke ahead on win rate. Zorua is the sharpest single
test: it has the thinnest data of anything trained here (0.28%) but a
mid-ranking win rate, so H1 expects it last and H2 expects it mid-pack.

## How this gets judged

The 60-game gate against `scaled320_lucario_1082` at 1.1 s/move, all four decks
sharing one opponent. Ladder scores are not usable for this: the anchor's own
identical bytes have drawn 1112.7 and 921.9, a 191-point spread, which is wider
than most gaps we would be trying to resolve.

Prior record of these predictors: board win rate has failed three times
(Crispin 57% WR then gated 25%; Dreepy 54% WR then measured 60% among elite
pilots; Bug Catching Set 51.8% then 62.3%). Data share has never been tested.
Neither deserves confidence yet.
