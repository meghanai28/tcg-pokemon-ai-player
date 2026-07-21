# Track 3: Oracle guided learning (Suphx style)

**Status: not yet implemented. This directory holds the design and the reason
it is worth trying.**

## The idea

[Suphx](https://arxiv.org/abs/2003.13590) reached a level above 99.99 percent of
ranked human players at Mahjong, a hidden information game, by:

1. supervised learning from logs of human professional players, then
2. self play reinforcement learning with policy gradients, plus
3. **oracle guiding**: during training the model is given access to hidden
   information to steer learning toward the path a perfect information player
   would take, then is progressively weaned off it.

## Why this is the right attack on our hardest problem

Hidden information is the core difficulty of this game and our current answer is
crude. Track 1 samples the opponent's hidden cards uniformly from a guessed
decklist. Every simulation then runs inside a world that is probably wrong, so
the search optimises against fiction. That is a ceiling on everything the search
does, and no amount of better evaluation fixes it.

Oracle guiding attacks this directly, and we are unusually well placed to try it:

- **The engine gives us perfect information during training.** `SearchBegin`
  takes the true hidden state, so we can construct oracle features for free.
- **We have the human logs.** 2,000 downloaded ladder replays, of which 917 are
  from players rated 1000 or above, already ingested into 112,000 labelled
  decisions by `track1_search/train/ingest_episodes.py`.
- **The supervised phase is already built and validated.** Our behaviour cloning
  pipeline passes its gates (policy top 1 of 52.9 percent versus a 35.4 percent
  heuristic baseline). That is Suphx's step 1, done.

So this track is less of a leap than Track 2: steps 1 is complete, and what is
missing is the oracle guided RL phase.

## Design sketch

1. **Oracle features.** Add a channel carrying the opponent's true hand and deck
   order, available only at training time.
2. **Oracle teacher.** Train a policy or value network with those features. It
   should be markedly better than any observation only model, and if it is not,
   that itself is informative about how much hidden information is worth here.
3. **Wean.** Anneal the oracle channel toward zero (Suphx drops the oracle
   features gradually), forcing the student to infer from observables.
4. **Alternative payoff: belief modelling.** Even a partial oracle model is
   directly usable as a **hidden card predictor** feeding Track 1's
   determinization, replacing uniform sampling with a learned posterior. This is
   the cheapest concrete win available from this track, because it improves the
   search we already ship without costing anything at search time (worlds are
   built once per decision, not at every node).

## Honest checkpoint

> Does an oracle informed model beat an observation only model by a wide margin
> on held out ladder positions?

If yes, hidden information inference is worth real effort and the weaning phase
is justified. If the gap is small, then determinization is already adequate,
Track 1's ceiling lies elsewhere, and this track should be dropped.

That checkpoint is cheap: it is one extra feature channel and one retrain, using
data we already have on disk.
