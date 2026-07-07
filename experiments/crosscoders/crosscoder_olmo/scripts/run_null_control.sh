#!/usr/bin/env bash
# Null-data control: train one crosscoder on Frobenius-norm-matched Gaussian
# random snapshots, run the same decision-rule eval, expect rate-rotation r ~ 0
# and CUSUM p > 0.1. Defends against the "crosscoder finds structure in
# anything" attack (Run 3 §7.2 control, p=0.613 on Gaussian snapshots).
#
# Pre-registered prediction (per SCHEDULE.md "Required controls"):
#   rate-rotation r within 95% CI of 0
#   reduced-baseline CUSUM p > 0.1
#
# Runtime: ~1 hour on a single A100-80GB. Run alongside the pilot.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
WU_CACHE="${WU_CACHE:-/workspace/wu_cache_olmo}"
NULL_CACHE="${NULL_CACHE:-/workspace/wu_cache_null}"
OUT_DIR="${OUT_DIR:-/workspace/results/olmo_cross_family/null_control}"
SEED="${SEED:-0}"

mkdir -p "${OUT_DIR}" "${NULL_CACHE}"

# uv resolves the project from cwd; run everything from the repo root
# (matches run_olmo_pilot.sh / run_olmo_production.sh).
cd "${REPO_ROOT}"

# Generate Gaussian snapshots matched in per-snapshot Frobenius norm to the
# real OLMo W_U snapshots. Saved to NULL_CACHE in the same naming convention,
# so wu_adapter.load_snapshots picks them up transparently.
uv run python - <<PYEOF
import torch
from pathlib import Path

real_cache = Path("${WU_CACHE}")
null_cache = Path("${NULL_CACHE}")

# Discover real W_U files in cache, build matched random snapshots
slug = "allenai_OLMo-2-1124-7B"
real_files = sorted(real_cache.glob(f"{slug}_step*_wu.pt"))
if not real_files:
    raise SystemExit(f"No real W_U files at {real_cache}; run extract_wu_olmo.py first.")

g = torch.Generator().manual_seed(${SEED})
for src in real_files:
    dst = null_cache / src.name
    if dst.exists():
        print(f"  {dst.name}: cached, skip")
        continue
    real_w = torch.load(src, map_location="cpu", weights_only=True)
    target_frob = real_w.norm().item()
    rand = torch.randn(real_w.shape, generator=g, dtype=real_w.dtype)
    rand = rand * (target_frob / rand.norm().item())
    torch.save(rand, dst)
    print(f"  {dst.name}: shape={tuple(rand.shape)} frob={rand.norm().item():.1f} (real={target_frob:.1f})")
PYEOF

# Use the same pre-registered schedule + hyperparameters as the pilot, but
# point at the null cache. Train at d_SAE=16384 to match the pilot config
# so the comparison is apples-to-apples.
STEPS="150 600 700 850 900 1000 \
       2000 3000 4000 5000 6000 7000 8000 9000 \
       14000 21000 27000 34000 \
       47000 110000 173000 236000 299000 362000 \
       425000 488000 614000 677000 740000 803000 \
       866000 928000"

uv run python scripts/train/train_crosscoder.py \
    --model allenai/OLMo-2-1124-7B \
    --cache-dir "${NULL_CACHE}" \
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
    --output "${OUT_DIR}/cc_null_dsae16384_seed${SEED}.pt" \
    2>&1 | tee "${OUT_DIR}/train_seed${SEED}.log"

# Evaluate against the pilot decision rules. We expect this to FAIL (r~0,
# p>0.1, EV likely 0). Failing is the desired outcome for a null control.
uv run python experiments/crosscoders/crosscoder_olmo/scripts/eval_decision_rules.py \
    --crosscoder "${OUT_DIR}/cc_null_dsae16384_seed${SEED}.pt" \
    --wu-cache "${NULL_CACHE}" \
    --stage pilot \
    --out "${OUT_DIR}/verdict_null_seed${SEED}.json" \
    || true   # Expected to "fail" — that's the point.

echo
echo "Null control done."
echo "Pre-registered prediction: rate-rotation r ≈ 0, CUSUM p > 0.1."
echo "Verdict: ${OUT_DIR}/verdict_null_seed${SEED}.json"
