#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
arm=${2:-}
if [[ "$mode" != "gate" && "$mode" != "scale" ]]; then
  echo "usage: bash tools/run_rl_tomorrow.sh {gate|scale} {techgrim|lucario}" >&2
  exit 2
fi

case "$arm" in
  techgrim)
    pool=data/fresh/deck_pool_20260808_broad_techgrim.json
    init=data/model_champion_bc.npz
    out=rl_osfp/run_clean_techgrim
    lr=3e-5
    ref_kl=0.30
    entropy=0.005
    ;;
  lucario)
    pool=data/fresh/deck_pool_20260808_broad_lucario.json
    init=data/model_lucario_scratch.npz
    out=rl_osfp/run_clean_lucario
    lr=5e-5
    ref_kl=0.20
    entropy=0.01
    ;;
  *)
    echo "arm must be techgrim or lucario" >&2
    exit 2
    ;;
esac

for required in .venv/bin/python "$pool" "$init"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done

export PTCG_MAX_OPT=24
if [[ "$mode" == "scale" ]]; then
  .venv/bin/python -m rl_osfp.audit_run "$out" \
    --strict --min-unique-cards 200
  resume=(--resume)
  wall_minutes=1200
else
  if [[ -e "$out/training_state.pt" || -e "$out/metrics.jsonl" ]]; then
    echo "$out already exists; audit it, then use scale instead of overwriting" >&2
    exit 2
  fi
  resume=()
  wall_minutes=90
fi

# 11,720 * 256 = 3,000,320 games. Gate mode intentionally stops after 90
# minutes; scale resumes the exact same declared experiment for 20-hour blocks.
# Ten rollout workers + four update threads fit the 14-core host because rollout
# and update phases are synchronized, and train.py caps CUDA allocation at 72%.
.venv/bin/python -u -m rl_osfp.train \
  --pool "$pool" \
  --init "$init" \
  --out-dir "$out" \
  --periods 11720 \
  --games-per-period 256 \
  --max-decisions 32000 \
  --epochs 2 \
  --batch 512 \
  --lr "$lr" \
  --target-kl 0.015 \
  --ref-kl-coef "$ref_kl" \
  --entropy-coef "$entropy" \
  --self-play-prob 0.60 \
  --uniform-field-prob 0.25 \
  --eval-games 8 \
  --eval-every 10 \
  --archive-max-wait 30 \
  --checkpoint-every 100 \
  --workers 10 \
  --threads 4 \
  --threads-per-worker 1 \
  --device cuda \
  --max-wall-minutes "$wall_minutes" \
  --no-record-game-rows \
  "${resume[@]}"

.venv/bin/python -m rl_osfp.audit_run "$out" \
  --strict --min-unique-cards 200 \
  --json-out "artifacts/${arm}_${mode}_audit.json"
