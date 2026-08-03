# Current decision record

Snapshot: 2026-08-03 UTC. Read this file before the longer historical README.

## What is live evidence

| Submission | Policy change | Public score | Downloaded record |
|---|---|---:|---:|
| Track 6 Tech-Grim (`55185089`) | exact-deck BC plus conservative GRPO | **972.0** | 40-24 (62.5%) |
| Track 8 Tech-Grim (`55195501`) | 192d BC with balanced Elo-800+ data | **848.9** | 30-20 (60.0%) |
| Track 6 Ogerpon (`55185105`) | exact-deck BC plus conservative GRPO | 719.2 | retired |

The two Tech records are statistically indistinguishable: Wilson 95% intervals
are 50.3-73.3% and 46.2-72.4%. Track 8 had no runtime failure and retained about
502 seconds of bank even in losses. Its bad rating is nevertheless a failed
promotion gate; public score, not offline imitation accuracy, decides what ships.

The matchup pools differ materially. Track 8 was 18-3 into Alakazam plus
Archaludon but 4-13 into Grimmsnarl, Crustle, Kangaskhan, Ogerpon, Lopunny, and
Thwackey. Track 6 saw far more Grim mirrors and fewer of the new arm's bad fringe
matchups. Kaggle's opponent-adjusted rating amplifies that difference.

## Algorithm decision

- **Keep the determinized search system.** The project evidence consistently
  says search and deck choice do most of the work; the net is a root move-order
  prior, not a reliable state evaluator.
- **Do not switch to PPO.** PPO would add a learned critic and high-variance
  on-policy updates exactly where local self-play and offline metrics have been
  least predictive. There is no evidence it would fix the live matchup problem.
- **Retain GRPO as code and a frozen comparison; do not scale beyond the bounded
  Track 9 test.** On the
  Tech exact-deck holdout, BC was 76.87%/0.62005 top-1/CE and selected GRPO was
  76.81%/0.61839. That is effectively neutral; the 972 score cannot honestly be
  attributed to GRPO alone.
- **Do not conclude that Elo-800 data mechanically ruined the model.** The new
  general model improved both the Elo-800-999 and Elo-1000+ temporal slices, and
  the exact Tech specialist improved offline too. The live promotion still
  failed, showing that those gates are insufficient rather than identifying a
  single bad data threshold.

## Next controlled experiment

1. Re-submit the unchanged Track 6 Tech archive as the control when quota
   resets, so the 972 arm is active again.
2. Use the other slot for a **160d exact-Tech conservative replay update plus
   targeted GRPO**:
   warm-start the Track 6 BC policy, train only on Elo-1000+ exact-list games,
   apply bounded winner/advantage weighting (AWR/CRR-style weighted BC), then
   run a tightly KL-constrained GRPO stage against the live loss matchups.
3. Keep architecture, deck, search code, and time budget identical between the
   two slots. The only variable should be the policy update.
4. Reject the candidate unless matchup-stratified replay gates improve against
   Grim, Crustle, Kangaskhan, and Lopunny without losing Alakazam/Archaludon.

This is a smaller and more falsifiable experiment than PPO or broad GRPO. It
uses the real replay distribution, avoids an unreliable value critic, preserves
the known 160d inference/search budget, and includes GRPO only as the final
bounded policy-improvement step.

Track 9 has now executed this pipeline. Its GRPO checkpoint materially improves
the temporal replay gate but ties the control both in a 22-game targeted field
gate (10-12 each) and a 20-game same-deck gate (10-10). It is therefore an
experimental second slot, not a proven replacement. The byte-identical control
and challenger were submitted together as refs `55202336` and `55202342`; both
are pending.

## Track status

- Track 1: active search foundation.
- Tracks 2-5: research/history; no current promotion candidate.
- Track 6 Tech-Grim: current control and best live candidate.
- Track 6 Ogerpon: retired.
- Track 7: deleted.
- Track 8: data/reservoir tooling retained; both 192d ladder arms retired from
  promotion. The unsubmitted Lopunny archive is evidence, not a recommended slot.
- Track 9: completed exact-Tech weighted-BC plus targeted-GRPO challenger;
  control ref 55202336 and challenger ref 55202342 are pending.
