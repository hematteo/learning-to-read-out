#!/usr/bin/env bash
# Stage-1 pilot: OLMo-2-7B-1124 cross-family crosscoder fit at d_SAE=16384.
# Runs on a single A100-80GB. ~3 h wall.
#
# Goal (per SCHEDULE.md decision rules):
#   - Reduced-baseline CUSUM perm test fires (p < 0.01)
#   - Rate-rotation correlation r > 0.5
#   - EV >= 0.40
#   - Dead-feature rate < 30%
#
# If all four pass, advance to run_olmo_production.sh (4x A100-80GB,
# d_SAE=32768, head-parallel).
#
# Prereqs:
#   - W_U cache populated by extract_wu_olmo.py at $WU_CACHE
#   - lib/Language-Model-SAEs/ installed (uv sync at the repo root)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

WU_CACHE="${WU_CACHE:-/workspace/wu_cache_olmo}"
OUT_DIR="${OUT_DIR:-/workspace/results/olmo_cross_family/pilot}"
SEED="${SEED:-0}"

# Pre-registered schedule (32 OLMo-2-7B-1124 stage1 steps)
STEPS="150 600 700 850 900 1000 \
       2000 3000 4000 5000 6000 7000 8000 9000 \
       14000 21000 27000 34000 \
       47000 110000 173000 236000 299000 362000 \
       425000 488000 614000 677000 740000 803000 \
       866000 928000"

mkdir -p "${OUT_DIR}"

# Hyperparameters: Ge Table 1 6.9B point + Run 3 corrections.
# d_SAE = 16384 = 4x expansion at d_model=4096 (single-GPU feasible).
cd "${REPO_ROOT}"

uv run python scripts/train/train_crosscoder.py \
    --model allenai/OLMo-2-1124-7B \
    --cache-dir "${WU_CACHE}" \
    --steps ${STEPS} \
    --expansion-factor 4.0 \
    --batch-size 2048 \
    --lr 1e-5 \
    --jumprelu-lr-factor 0.3 \
    --l1-coefficient 0.3 \
    --tanh-stretch 1.0 \
    --init-threshold 0.1 \
    --decoder-transpose-init 1.0 \
    --input-preprocess center_scale \
    --l1-warmup-fraction 0.1 \
    --lr-warmup-fraction 0.10 \
    --lr-decay-fraction 0.20 \
    --optimizer sparse_adam \
    --amp-dtype fp32 \
    --n-epochs 100 \
    --seed "${SEED}" \
    --output "${OUT_DIR}/cc_olmo27b_dsae16384_seed${SEED}.pt" \
    2>&1 | tee "${OUT_DIR}/train_seed${SEED}.log"

echo
echo "Pilot done. Result: ${OUT_DIR}/cc_olmo27b_dsae16384_seed${SEED}.pt"
echo "Next: run permutation_test.py + rate-rotation analysis to evaluate"
echo "decision rules in SCHEDULE.md."
