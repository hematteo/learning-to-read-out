#!/usr/bin/env bash
# Stage-2 production: OLMo-2-7B-1124 cross-family crosscoder fit at d_SAE=32768.
# Runs on 4x A100-80GB with head-parallelism (Ge §A.3). ~3 h wall.
#
# Conditional on Stage-1 pilot passing all four decision-rule gates
# (see SCHEDULE.md). If pilot fails, do not run this.
#
# Goal (per SCHEDULE.md decision rules, production stage):
#   - Reduced-baseline CUSUM perm test fires (p < 0.001)
#   - Rate-rotation correlation r > 0.7
#   - Decoder cosine vs step-150 baseline drops < 0.7 by step 1000
#   - EV >= 0.65 (interpretability bar)
#   - Dead-feature rate < 10%
#   - Top-50 CUSUM features: >= 5 monosemantic by auto-interp
#   - Causal script-ablation: top-10 ablation drops target script >= 1 nat/token
#
# Prereqs:
#   - W_U cache populated by extract_wu_olmo.py at $WU_CACHE
#   - 4x A100-80GB visible to torchrun (CUDA_VISIBLE_DEVICES=0,1,2,3)
#   - Pilot at d_SAE=16384 has cleared all four pilot gates

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
WU_CACHE="${WU_CACHE:-/workspace/wu_cache_olmo}"
OUT_DIR="${OUT_DIR:-/workspace/results/olmo_cross_family/production}"
SEED="${SEED:-0}"
NPROC="${NPROC:-4}"

# Verify pilot verdict before launching production.
PILOT_VERDICT="${PILOT_VERDICT:-/workspace/results/olmo_cross_family/pilot/verdict_seed0.json}"
if [[ ! -f "${PILOT_VERDICT}" ]]; then
    echo "ERROR: pilot verdict not found at ${PILOT_VERDICT}"
    echo "Run run_olmo_pilot.sh + eval_decision_rules.py first."
    exit 1
fi
PROCEED=$(VERDICT_PATH="${PILOT_VERDICT}" python -c "
import json, os, sys
try:
    with open(os.environ['VERDICT_PATH']) as f:
        d = json.load(f)
    # Pre-registered partial-credit rule (SCHEDULE.md §'Stage 1'):
    #   proceed iff primary (rate_rotation_r) passes AND ≥2 of {B,C,D} pass.
    # The eval script writes this as decision.proceed; older verdicts only
    # had all_gates_pass — fall back to that for backward compatibility.
    decision = d.get('decision', {})
    if 'proceed' in decision:
        print(decision['proceed'])
    else:
        print(d['all_gates_pass'])
except Exception as e:
    sys.stderr.write(f'verdict-parse-error: {e}\n')
    sys.exit(1)
" 2>&1) || {
    echo "ERROR: failed to parse pilot verdict at ${PILOT_VERDICT}"
    echo "  ${PROCEED}"
    exit 1
}
if [[ "${PROCEED}" != "True" ]]; then
    echo "ERROR: pilot decision.proceed=${PROCEED}. Aborting production per SCHEDULE.md."
    echo "Verdict file: ${PILOT_VERDICT}"
    exit 2
fi

mkdir -p "${OUT_DIR}"

# Pre-registered schedule (32 OLMo-2-7B-1124 stage1 steps)
STEPS="150 600 700 850 900 1000 \
       2000 3000 4000 5000 6000 7000 8000 9000 \
       14000 21000 27000 34000 \
       47000 110000 173000 236000 299000 362000 \
       425000 488000 614000 677000 740000 803000 \
       866000 928000"

# Production hyperparameters: Ge Table 1 6.9B verbatim, d_SAE=32768 (8x expansion),
# head-parallel across NPROC GPUs (each GPU holds 32/NPROC = 8 snapshot heads).
cd "${REPO_ROOT}"

uv run torchrun \
    --nproc-per-node="${NPROC}" \
    --nnodes=1 \
    experiments/crosscoders/crosscoder_olmo/scripts/train_distributed.py \
        --model allenai/OLMo-2-1124-7B \
        --cache-dir "${WU_CACHE}" \
        --steps ${STEPS} \
        --d-sae 32768 \
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
        --n-epochs 250 \
        --seed "${SEED}" \
        --output "${OUT_DIR}/cc_olmo27b_dsae32768_seed${SEED}.pt" \
    2>&1 | tee "${OUT_DIR}/train_seed${SEED}.log"

echo
echo "Production run done: ${OUT_DIR}/cc_olmo27b_dsae32768_seed${SEED}.pt"
echo "Next: eval_decision_rules.py with --stage production"
