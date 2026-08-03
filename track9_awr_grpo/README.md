# Track 9: exact-Tech weighted BC plus targeted GRPO

Status on 2026-08-03 UTC: trained, selected, packaged, locally gated, and
submitted as the two newest active agents. The unchanged control is Kaggle ref
`55202336`; the GRPO iteration-6 challenger is ref `55202342`. Both are pending.

The experiment changes one policy pipeline while holding the 60-card deck,
160d/5-layer architecture, determinized search code, and think-time budget
fixed. The challenger:

1. warm-starts `track6_controlled/model_tech_grim_bc.npz`;
2. trains only on 65,249 Elo-1000+ exact-list decisions dated July 30 or earlier;
3. uses bounded symmetric outcome weighting (`--winner-weight 1.20`), which is
   a conservative return-weighted/AWR-style BC update;
4. validates on 9,763 exact-list July 31 decisions;
5. runs six conservative GRPO iterations (at most 288 games) against exact
   lists mined from Track 8's live matchups;
6. selects only a genuinely updated checkpoint with KL <= 0.0015, no more than
   0.5 percentage points top-1 regression, and no more than 0.01 CE regression
   relative to the weighted-BC checkpoint.

GRPO uses LR `1e-6`, clip `0.08`, KL beta `0.12`, one optimizer epoch, a 15,000
decision cap per iteration, six CPU threads, a 240-minute wall cap, and the
existing 75% CUDA allocator limit. Loss matchups receive higher tempered
schedule weights, while Alakazam and Archaludon remain in the rotation as
forgetting guards.

Run:

```bash
.venv/bin/python track9_awr_grpo/train_track9.py
.venv/bin/python track9_awr_grpo/select_checkpoint.py
.venv/bin/python track9_awr_grpo/build_submissions.py
```

The resulting submission pair will be:

- `submission_track9_control_tech_grim_972.tar.gz`: byte-identical Track 6
  control.
- `submission_track9_awr_grpo_tech_grim.tar.gz`: selected challenger.

Do not submit only one archive. Kaggle keeps the newest two agents active, and
the unchanged control is necessary to interpret the challenger.

Submitted pair:

| Slot | Kaggle ref | Initial status |
|---|---:|---|
| byte-identical 972 control | 55202336 | PENDING |
| weighted-BC + GRPO challenger | 55202342 | PENDING |

## Completed results

The bounded outcome-weighted BC stage stopped after epoch 6 and restored epoch
3. All 65,249 training decisions are from the exact Tech list at Elo 1080; the
9,763 July 31 decisions remained isolated.

| Exact-list policy | July 31 CE | Top-1 |
|---|---:|---:|
| Track 6 BC | 0.62005 | 76.87% |
| Track 6 selected GRPO | 0.61839 | 76.81% |
| Track 9 weighted BC | 0.57307 | 78.18% |
| **Track 9 selected GRPO iter 6** | **0.57298** | **78.26%** |

GRPO completed 288 games. Forty-four of 72 matched groups had reward variance,
so the update was real rather than a checkpointed no-op. Iteration 6 passed the
temporal gates with KL 0.000077 from its weighted-BC reference, far below the
0.0015 limit.

The local search gates are deliberately reported as inconclusive:

| Gate | Challenger | Control |
|---|---:|---:|
| nine-deck targeted field, fixed control opponent | 10/22 | 10/22 |
| same-deck challenger vs control | 10/20 | 10/20 |

In the 22-game targeted cells, the challenger improved Grim (2/4 vs 0/4) and
Lopunny (2/2 vs 0/2), tied Ogerpon, Alakazam, and Archaludon, but trailed by one
game in the tiny Crustle, Kangaskhan, Thwackey, and Lucario cells. These samples
do not prove superiority. The challenger is suitable only as the experimental
second slot beside the unchanged control.

Packaged archives:

```text
4fdb2ecf444d58161430fcaacb84795e5cd7f51ed2b756f225385493618e2f12  submission_track9_control_tech_grim_972.tar.gz
a622ddf3a29fe52ed317078159daaaf4916ccd5588244b7b2a6eb6594c63bde5  submission_track9_awr_grpo_tech_grim.tar.gz
```
