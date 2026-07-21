# Track 2: Deep Monte Carlo (DouZero style)

**Status: not yet implemented. This directory holds the design and the reason
it is worth trying.**

## The idea

[DouZero](https://arxiv.org/pdf/2106.06135) masters DouDizhu, a hidden
information card game with an enormous action space, using **no tree search at
all**. It learns Q(state, action) from Monte Carlo returns of self play games,
encodes actions as feature matrices so the network generalises to action
combinations it never saw in training, and runs many parallel actors.

The authors' justification is the reason this track exists:

> "unlike many tree search algorithms, DouZero is based on sampling, which
> allows us to use complex neural architectures and generate much more data per
> second, given the same computational resources"

## Why this addresses our specific failure

In Track 1 the network and the search **compete for the same CPU**. Measured on
this project: an engine step costs about 0.05 ms, a network call about 3 ms, so
every network evaluation burns roughly 60 simulations. We proved experimentally
that a 33 percent better value head produced exactly zero game level
improvement, because the tax cancels the signal (see `results/` and the
Track 1 README).

Deep Monte Carlo removes the competition entirely:

| | Track 1 (search) | Track 2 (DMC) |
|---|---|---|
| Network calls per decision | hundreds, competing with search | one |
| What the engine is used for | lookahead at decision time | generating training data |
| Bottleneck | CPU shared between net and search | data generation throughput |

Our engine runs at about 20,000 steps per second. Track 1 spends that speed on
search, where it fights the network. Track 2 spends it on data, where it feeds
the network. That is a better match for the hardware we actually have.

## Design sketch

1. **Action encoding.** Reuse `nn_features.py` option tokens. Each legal option
   already becomes a feature vector, which is the analogue of DouZero's card
   matrices.
2. **Q network.** State features plus one action's features to a scalar Q.
   Score every legal option, play the argmax (epsilon greedy while training).
3. **Deep Monte Carlo.** Play a full self play game, then regress Q(s, a)
   toward the realised return for every decision taken. No bootstrapping and no
   search, which is the whole point.
4. **Actors.** Parallel worker processes writing shards, same pattern as
   `track1_search/train/selfplay.py`.
5. **Opponents.** The mined meta decks, so the data reflects the real field.

## Honest checkpoint

The bar for this track is **not** beating Track 1's search agent, which scores
around 740 to 820 on the ladder. That would take far more compute than a laptop
has. The bar is:

> Can the raw Q network, with no search, beat the hand written heuristic policy
> alone?

If yes, the approach has legs on our hardware and the Q network can additionally
serve as Track 1 priors, which is a cheap partial win. If no, DMC needs more
compute than we have and the track should be abandoned rather than nursed.

## Risk

DouZero used many actors over days. We have a laptop and a fast simulator. The
fast simulator is genuinely the resource this method wants, but the compute gap
is real and this track may simply not converge in the time available.
