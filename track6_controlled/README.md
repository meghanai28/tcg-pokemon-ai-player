# Track 6: controlled deck-specialist arms

Status on 2026-08-02: trained, selected, locally validated, packaged, submitted,
and accepted by Kaggle. After the first downloaded ladder games, Tech-Grim is
retained and Ogerpon is retired. The historical archives were:

- `submission_grpo_controlled_tech_grim.tar.gz`
- `submission_grpo_controlled_ogerpon.tar.gz`

| Arm | Kaggle ref | Status at submission | Initial score |
|---|---:|---|---:|
| Tech Grimmsnarl | 55185089 | COMPLETE | 600.0 |
| Teal Mask Ogerpon | 55185105 | COMPLETE | 600.0 |

Downloaded early records are 9-4 for Tech-Grim and 8-6 for Ogerpon. Tech-Grim
went 5-1 into Alakazam and showed no runtime problem. Ogerpon lost across
several unrelated archetypes and its opponent-adjusted public rating remained
poor, so Track 8 replaces it with Mega Lopunny rather than adding more GRPO.

These are separate ladder hypotheses, not a head-to-head claim. Each arm keeps
the same proven Track 1 search agent and changes only its exact deck plus the
matching deck-specialized root-prior model. Each GRPO checkpoint was compared
with that arm's own supervised checkpoint on an untouched temporal holdout.

The pre-submission regression autopsy also compared 94 games from the old
967.1 exact-Grim arm with all 63 refreshed-Grim and 58 refreshed-Garchomp
games. Old and refreshed Grim raw win rates were nearly identical (57.4% and
55.6% with very wide, overlapping intervals), and neither showed runtime
errors. Garchomp was weak into the two largest observed archetypes, so Track 6
replaces that arm rather than treating its 889.3 rating as a GRPO endorsement.
The complete diagnosis is in the root README.

## Why these two arms

The July 31 Elo-1000+ census contained 7,475 qualifying deck appearances. The
two exact submitted lists came directly from valid competition episodes:

- **Tech Grimmsnarl** (`mined_13`): Budew and Yveltal tech added to the dominant
  Grimmsnarl/Froslass/Munkidori shell. It appeared 113 times with a 58% observed
  win rate. Current Reddit discussions independently emphasize Budew item lock,
  Yveltal trapping, Boss support, and matchup-dependent Froslass counts.
- **Pure Teal Mask Ogerpon** (`mined_4`): a fast, deliberately different grass
  arm. It appeared 285 times with a 61% observed win rate. Community discussion
  highlights Teal Dance plus Energy Retrieval as a reliable acceleration/draw
  engine, while also warning that item lock is a real vulnerability.

The Kaggle discussion on deck choice describes training per-archetype agents
and choosing among them by popularity-weighted meta performance. We therefore
trained separate policies and changed the GRPO opponent rotation from uniform
to a square-root-tempered popularity schedule. The dominant decks receive more
games, but every one of the top 20 exact lists remains in each 40-slot cycle.
This is less brittle than training only into the dominant Grimmsnarl mirror.

Research links:

- Kaggle competition and submission constraints:
  <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/description>
- Kaggle deck/archetype training discussion:
  <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/727816>
- Public 967.7 probabilistic-agent notebook, used as design evidence for
  deck-specific tactics and shallow tactical checks (no code copied):
  <https://www.kaggle.com/code/aristophanivan/improved-probabilistic-agent/comments>
- Current Tech-Grim fine-tuning discussion:
  <https://www.reddit.com/r/pkmntcg/comments/1v5d381/need_help_fine_tuning_this_grimmsnarl_list/>
- Current Grimmsnarl viability/Budew/Yveltal discussion:
  <https://www.reddit.com/r/pkmntcg/comments/1v6khno/is_marnies_grimmsnarl_still_competitive_viable/>
- Teal Mask Ogerpon engine discussion:
  <https://www.reddit.com/r/pkmntcg/comments/1sdk7tw/get_your_teal_mask_ogerpon_exs_now_theyre_going_up/>
- Energy acceleration and item-lock discussion:
  <https://www.reddit.com/r/pkmntcg/comments/1sfrx7y/what_are_the_best_batteries_for_attackers_that/>

## Data boundaries

There is no July 31 leakage: all training episodes are dated July 30 or earlier;
July 31 is holdout only.

| Arm | BC training decisions | Exact-list decisions inside train | July 31 exact holdout | Exact-list pilots in train |
|---|---:|---:|---:|---:|
| Tech Grimmsnarl | 730,498 | 130,498 (exact data repeated once) | 9,763 | 1 |
| Teal Mask Ogerpon | 20,395 | 5,659 | 16,996 | 1 |

Tech-Grim uses 600,000 broad deck-anchor decisions plus two copies of 65,249
exact-list decisions. Ogerpon uses 14,736 broad anchor decisions plus 5,659
exact-list decisions. Exact lists were rare before July 31 and came from only
one training pilot per arm, so exact-only training would have mostly cloned one
player. The broader anchors intentionally reduce that pilot-confounding risk.
The much larger Tech dataset is appropriate because it was far more prevalent;
the Ogerpon run uses early stopping and a lower learning rate to avoid memorizing
its smaller sample.

`track1_search/train/ingest_episodes.py --deck DECK.csv` now supports exact
60-card multiset filtering. `make_anchor_data.py` filters already-ingested
shards one at a time so the full replay corpus is never loaded into RAM.

## Training and checkpoint selection

Both supervised runs warm-started the refreshed July model. Training used CUDA
with a 75% allocator cap; rollouts stayed on CPU with six Torch threads, no more
than 15,000 retained decisions per GRPO iteration, and a 180-minute hard wall
limit. Peak observed Tech BC usage was about 9.2 GiB VRAM on the 16 GiB GPU;
system memory and swap stayed healthy.

| Arm | Refreshed general BC | Specialized BC | Improvement |
|---|---:|---:|---:|
| Tech Grimmsnarl | 71.61% top-1 | **76.87%**, CE 0.62005 | +5.26 pp |
| Teal Mask Ogerpon | 66.77% top-1 | **81.16%**, CE 0.69898 | +14.39 pp |

Each arm then played 320 on-policy games: eight iterations, ten matched groups,
four games per group, alternating seats. The update used LR `2e-6`, clip `0.10`,
KL beta `0.08`, and the square-root-tempered current top-20 schedule.

| Arm | Active groups | Selected | Holdout CE | Holdout top-1 | Checkpoint KL |
|---|---:|---:|---:|---:|---:|
| Tech Grimmsnarl | 59/80 | iteration 5 | **0.61839** | 76.81% | 0.000573 |
| Teal Mask Ogerpon | 53/80 | iteration 4 | **0.69744** | **81.20%** | 0.001005 |

Every saved checkpoint was replay-scored. The last checkpoints were not chosen:
iteration 5 minimized Tech policy loss before later drift, while Ogerpon
iteration 4 improved both CE and top-1 over its BC reference. The final model
files are copies of those immutable checkpoints.

## Validation and packaging

- 4 GRPO schedule/unit tests pass.
- All new and changed Python files compile.
- Both exact decks contain 60 cards and were observed in valid Kaggle episodes.
- Both agents complete official local-environment games from both seats.
- Tiny same-deck smoke A/B: Tech GRPO 3/4 vs Tech BC; Ogerpon GRPO 2/4 vs
  Ogerpon BC. These four-game samples are legality/sanity checks only.
- Both archives have the required top-level `main.py`, `deck.csv`, `model.npz`,
  feature/inference modules, and `cg/` engine; both are 5.0 MiB.

SHA-256:

```text
4fdb2ecf444d58161430fcaacb84795e5cd7f51ed2b756f225385493618e2f12  submission_grpo_controlled_tech_grim.tar.gz
c71214d639da0b955956b224a768091f47362843b83568ee0968be92d78b3e5e  submission_grpo_controlled_ogerpon.tar.gz
```

Rebuild deterministically from the selected final models:

```bash
.venv/bin/python track6_controlled/build_submissions.py
```

Submit:

```bash
.venv/bin/kaggle competitions submit pokemon-tcg-ai-battle \
  -f submission_grpo_controlled_tech_grim.tar.gz \
  -m "controlled tech grim: specialized BC + conservative GRPO iter5"

.venv/bin/kaggle competitions submit pokemon-tcg-ai-battle \
  -f submission_grpo_controlled_ogerpon.tar.gz \
  -m "controlled pure ogerpon: specialized BC + conservative GRPO iter4"
```

Kaggle keeps only the two newest submissions active, so upload both together.
Do not interpret the initial 600 rating as a result; wait for matchmaking and
compare them only after both have accumulated a meaningful number of games.

## Ladder disposition

Track 6 is now historical evidence. Tech-Grim advances to the larger-data 192d
Track 8 run; Ogerpon does not. The old models and exact deck remain versioned so
the submitted result is reproducible, but generated Ogerpon data/checkpoints
and its local archive are removed. See `track8_bc800/README.md` for the current
Mega Lopunny replacement, data boundaries, and promotion gates.
