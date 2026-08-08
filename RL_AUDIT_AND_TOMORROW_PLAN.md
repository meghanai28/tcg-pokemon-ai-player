# RL audit and tomorrow plan — 2026-08-08

## Outcome of the audit

The project has not yet run the experiment the public RL evidence recommends.
The 1127-period Lucario run had 252,448 games, but 216,246 (85.7%) were produced
by policies one to three learner versions stale. It also had five engine errors,
only 121 unique cards in its deck population, and only 8.4% of the public
thread's 3M-game lower bound. Its checkpoints are invalid evidence for PPO and
must not be resumed or submitted.

The compact machine-readable failure report is
`artifacts/rl_lucario_failure_audit.json`. The original detailed metrics are
preserved as a compressed failure record, not as a runnable checkpoint.

The exact frozen champion remains the strongest active submission: ref 55335692
was 910.1 after 55 episodes at the audit snapshot. Dunsparce ref 55339282 was
814.6 after 48. The historical best is 972 on the same champion bytes; public
ratings are noisy enough that no local result can guarantee 1000.

## What was fixed

- PPO collection now has a hard policy-version barrier. The deprecated async
  flag cannot overlap collection and updates.
- The unused cross-update async collector was deleted. Do not reintroduce it
  without an actor-lag correction such as V-trace.
- PPO recomputes current/reference log-probabilities at the rollout temperature.
- The KL early-stop now uses the non-negative `ratio - 1 - log(ratio)` estimator.
- A frozen period-zero BC opponent remains in the league.
- Resume restores period, RNG, optimizer, archive age and league, moves Adam
  state correctly between CPU and CUDA, and rejects any changed pool bytes,
  encoder ABI, BC anchor or objective/data parameter.
- Learning rate decays to 10% over the declared long run.
- Metrics are append-only JSONL with bounded matchup/error summaries. Numbered
  checkpoints are sparse; latest state remains resumable every period.
- The period ceiling now permits 3M games without checkpoint/metrics explosion.
- Packaging uses the literal MAX_OPT=24 shipping encoder and rejects a wrong
  checkpoint format before building an archive.
- `audit_run --strict` makes lag, errors, invalid actions, state/metric gaps,
  pool mutation and insufficient requested card coverage hard failures.
- The ladder tool now fetches enough history and implements the published rule:
  latest two submissions active, best active score on the board.

## Fresh curriculum and hardware budget

Official Aug 1-7 data: 32,645 episodes, 53,312 Elo-900+ deck seats and 185 exact
lists. The broad field has 80 lists and 204 unique cards (old field: 121). Deck
draws are 25% uniform for counter-strategy coverage and 75% square-root ladder
frequency for realism.

Host budget at audit time: 23 GiB RAM (22 GiB available), 14 CPU cores, RTX 5080
with 16.3 GB VRAM (14.3 GB free), and 887 GB disk free. Training uses 10 rollout
processes, four learner threads and at most 72% of VRAM. Rollout and update are
synchronized, so CPU and learner thread budgets do not overlap materially.

## Tomorrow: controlled sequence

The objective is to produce a real 1000+ candidate, not to force two uploads.
1000 cannot be promised because identical bytes have ranged roughly 810-972,
but this sequence maximises the chance without destroying the current control.

1. Check `free -h`, `nvidia-smi`, and `tools/ladder_status.py`. Do not upload.
2. Start the conservative Tech-Grim arm:

   ```bash
   bash tools/run_rl_tomorrow.sh gate techgrim
   ```

   It starts from the exact champion BC prior, uses lower LR/ref-KL 0.30, and
   stops after 90 minutes. This is the lowest-variance route to improving 910.
3. Start the deck-upside Lucario arm only after the first gate frees resources:

   ```bash
   bash tools/run_rl_tomorrow.sh gate lucario
   ```

   It uses the matching Lucario BC prior, a slightly larger but still bounded
   LR, and the same 80-deck league. Never run both simultaneously: rollouts are
   the bottleneck and two 72%-VRAM learners can collide.
4. For each arm, require all of the following before scale:

   - strict audit passes with zero lag/errors/invalid actions and 204-card pool;
   - current-policy win rate does not collapse against any common field deck;
   - at least two separated checkpoints improve standalone league results;
   - those checkpoints do not regress behind the frozen search shell versus the
     champion at 1.1 seconds/move. Use at least 120 games for the final gate.
5. Resume only the better arm in 20-hour blocks:

   ```bash
   bash tools/run_rl_tomorrow.sh scale techgrim
   # or: bash tools/run_rl_tomorrow.sh scale lucario
   ```

   The declared run is 11,720 periods x 256 = 3,000,320 games. At the measured
   synchronous rate this is about 55 hours, so the first useful overnight stop
   is about 1M games, not the finish. Audit after every block. Compare checkpoint
   100, 200, ...; never assume the last is best.
6. GRPO is a conditional third phase, not tomorrow's blind scale arm. Binary
   group outcomes often give zero normalized advantage when a group is all wins
   or all losses. Only add a synchronized GRPO refinement if clean PPO first
   improves both standalone play and search-prior transfer; otherwise it spends
   the same games with a weaker baseline.
7. Submission order after a candidate wins the 120-game search gate:

   - re-upload the exact champion first, so it becomes the newer protected
     control instead of being the oldest active slot;
   - upload only the best verified RL candidate, leaving champion + candidate
     as the two active submissions;
   - inspect the first 10 episodes. Do not upload both experimental arms merely
     because they exist. A second candidate replaces the control and is allowed
     only if it independently clears the same gate.

## Verified handoff

The bounded CUDA stop/resume test completed periods 1 and 2 with state period 2,
eight games at lag zero, zero errors/invalid actions, Adam state saved on CPU,
and a passing strict audit. Its disposable full OSFP checkpoint was packaged
behind the byte-identical 972 shell; verification recorded 364/364 successful
network-prior calls, 9.7 ms per call and zero invalid actions. These eight games
prove plumbing only, not strength, and the disposable tarball is not a
submission candidate.

The official Aug 7 daily archive contains only one `Meghana284` game (a
Tech-Grim loss to exact current `field_2` over 196 steps), so it is not a usable
matchup sample. `tools/audit_team_replays.py` records that fact and can be rerun
over later archives; a real current-submission loss audit still needs the full
episode API after its usage window resets.

## Cleanup status

Reversible cleanup is complete: the unsafe unused async collector is gone,
generated runs/logs are ignored, obsolete numbered tracks are no longer live,
and compact audit/verification reports are the handoff artifacts.

Permanent deletion was deliberately not forced after the environment rejected
it as high risk. The proposed exact set is approximately 31 GB:

- `.reset-quarantine-20260803/` (8.7 GB of already retired tracks/data);
- `rl_osfp/run_lucario/` (23 GB of invalid stale-policy checkpoints);
- `rl_osfp/run/` (157 MB old random-init run);
- four `artifacts/rl_luc_p*.tar.gz` packages plus their verification JSON;
- three superseded narrow `deck_pool_20260808*` files.

Deleting those is irreversible for untracked model/data artifacts. The compact
failure audit and a 3.1 MB compressed copy of the invalid run's detailed metrics
are already preserved. Remove the set only after explicit confirmation of that
recovery tradeoff.
