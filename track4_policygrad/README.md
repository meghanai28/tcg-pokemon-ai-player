# Track 4: Policy gradient (Delightful Gradient)

**Status: not yet implemented. Design and rationale only.**

## The method

["Delightful Policy Gradient"](https://arxiv.org/abs/2603.14608), Ian Osband.

Standard policy gradient has two failure modes the paper targets:

1. rare actions can disproportionately swing updates within a single context, and
2. a batch over-allocates its gradient budget to contexts the policy already
   handles well.

Delightful Gradient (DG) gates each term with a **sigmoid of delight**, where
delight is the product of advantage and action surprisal. The paper reports DG
outperforming REINFORCE, PPO and advantage weighted baselines, with larger gains
on harder tasks, and proves the expected gradient moves strictly closer to the
supervised cross entropy oracle even with infinite samples. That is a structural
claim, not just variance reduction.

## Why it fits this game

The second failure mode is an unusually good description of our data. A game is
roughly 150 decisions, and the large majority are trivial or near forced:
mulligan choices, single legal option, forced draws, activation prompts our
scorer already agrees with the top players on 95 percent of the time. A handful
of MAIN phase decisions decide the game. Divergence mining measured exactly
this shape:

| Context | Decisions | Our agreement with 1050+ players |
|---|---|---|
| MAIN | 18,226 | 38 percent |
| TO_HAND | 3,493 | 53 percent |
| ACTIVATE | 1,106 | 95 percent |
| DRAW_COUNT | 78 | 96 percent |

A vanilla policy gradient would keep spending gradient on ACTIVATE and
DRAW_COUNT, which are already solved. DG's gating is designed to move that
budget to the contexts that are still wrong, which here is MAIN.

The first failure mode matters too, because our action space varies from 2 to
30 plus options per decision and rare high impact actions (a gust that wins the
prize race, an attack that changes the trade) are precisely the ones we most
need correct credit on.

## Why this is a genuinely new track

Everything we have trained so far is **supervised**: behaviour cloning from
human replays, distillation from search visit counts, distillation from a
teacher network. We have never run an actual policy gradient, so we have never
optimised directly for *winning* rather than for *matching*. That gap is worth
closing regardless of which algorithm fills it, and DG is a reasonable choice
because it explicitly targets the credit assignment shape our data has.

## Design sketch

1. **Policy.** Reuse the existing network and `nn_features.py` encoding. The
   pointer style head over option tokens already produces a distribution over a
   variable sized legal action set, which is what a policy gradient needs.
2. **Warm start.** Initialise from the behaviour cloned model rather than from
   scratch. Suphx does the same, and it matters more here because the reward is
   sparse.
3. **Rollouts.** Self play against the mined meta decks, same actor pattern as
   `track1_search/train/selfplay.py`.
4. **Advantage.** Game outcome minus the value head's prediction, which we
   already train and which currently reaches 0.26 times the baseline error.
5. **DG update.** Weight each term by sigmoid(advantage times surprisal) in
   place of the plain advantage weighting.

## Honest checkpoint and risks

> Does DG fine tuning of the behaviour cloned policy beat that same policy
> before fine tuning, head to head?

Risks worth stating up front:

- The paper's evaluation is MNIST, transformer sequence modelling and continuous
  control. **It has not been demonstrated on a card game**, so the transfer is
  an assumption.
- Our reward is sparse and terminal, spread over about 150 decisions. That is
  a hard credit assignment problem for any policy gradient.
- Policy gradient fine tuning can degrade a good supervised policy. The warm
  start is what we would be risking, so the frozen behaviour cloned model must
  be preserved and the comparison run head to head before anything ships.
