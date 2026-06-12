#!/usr/bin/env bash
# Multi-GPU launcher for the contrastive readout-swap grid with per-example margin
# logging. NOT auto-fired; you `bash` this manually on the GPU host.
#
# What it does (resume-safe end-to-end):
#   1. (optional) build per-tokenizer task datasets via build_task_datasets.py
#   2. run run_swap_grid.py --save-per-example over (families × h_steps × s_steps × alignments)
#   3. emit a manifest.txt summarising cell coverage
#
# All resume-safety lives in run_swap_grid.py:
#   * per-cell .json shard is the resume marker
#   * per-example .pt sidecar is written before the .json (Step 2a contract)
#   * stale .json without matching .pt is auto-deleted on restart
#
# Usage (on the GPU host, after `git clone` + `uv sync`):
#
#   bash experiments/probes/contrastive_readout_swap/scripts/launch_swap_extraction.sh \
#       --model pythia-1b \
#       --out-root /workspace/swap_1b_$(date -u +%Y%m%dT%H%M) \
#       --families "sva ioi ioi_role_balanced numeric_gt induction piqa arc_easy arc_challenge sciq lambada winogrande" \
#       --build-datasets
#
# For Pythia-6.9B (confirmatory only — run after the 1B screen):
#
#   bash experiments/probes/contrastive_readout_swap/scripts/launch_swap_extraction.sh \
#       --model pythia-6.9b \
#       --out-root /workspace/swap_69b_$(date -u +%Y%m%dT%H%M) \
#       --families "numeric_gt piqa arc_challenge sciq" \
#       --h-steps "256 512 1000 2000 8000 143000" \
#       --build-datasets \
#       --dtype bf16
#
# Required env (set in your shell before launching):
#   HF_HOME       — large scratch path for HF model cache (~/hf_cache).
#   HF_TOKEN      — optional, avoids rate limiting.

set -euo pipefail

MODEL=""
OUT_ROOT=""
FAMILIES="sva ioi ioi_role_balanced numeric_gt induction piqa arc_easy arc_challenge sciq lambada winogrande"
H_STEPS="256 512 1000 2000 8000 143000"
S_STEPS=""                 # empty → script default (canonical 32-step schedule)
ALIGNMENTS="none"
DTYPE="fp32"
BUILD_DATASETS=0
N_MAX_PER_FAMILY=2000

usage() {
    cat <<USAGE
$0 --model {pythia-1b|pythia-6.9b|pythia-160m|olmo-2-7b} --out-root <dir>
   [--families "f1 f2 ..."] [--h-steps "256 512 ..."] [--s-steps "..."]
   [--alignments "none row_norm procrustes"] [--dtype fp32|bf16|fp16]
   [--build-datasets] [--n-max-per-family N]
USAGE
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --families) FAMILIES="$2"; shift 2 ;;
        --h-steps) H_STEPS="$2"; shift 2 ;;
        --s-steps) S_STEPS="$2"; shift 2 ;;
        --alignments) ALIGNMENTS="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --build-datasets) BUILD_DATASETS=1; shift ;;
        --n-max-per-family) N_MAX_PER_FAMILY="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1"; usage ;;
    esac
done

[[ -z "$MODEL" || -z "$OUT_ROOT" ]] && usage

case "$MODEL" in
    pythia-160m) HF_MODEL="EleutherAI/pythia-160m" ;;
    pythia-1b)   HF_MODEL="EleutherAI/pythia-1b"   ;;
    pythia-6.9b) HF_MODEL="EleutherAI/pythia-6.9b" ;;
    olmo-2-7b)   HF_MODEL="allenai/OLMo-2-1124-7B" ;;
    *) echo "unknown model: $MODEL"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
DATASETS_DIR="$OUT_ROOT/datasets_$MODEL"
RUN_DIR="$OUT_ROOT/swap_grid"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$DATASETS_DIR" "$RUN_DIR" "$LOG_DIR"

echo "[launcher] model=$HF_MODEL"
echo "[launcher] out_root=$OUT_ROOT"
echo "[launcher] families=$FAMILIES"
echo "[launcher] h_steps=$H_STEPS"
echo "[launcher] s_steps=${S_STEPS:-<canonical-32>}"
echo "[launcher] alignments=$ALIGNMENTS  dtype=$DTYPE"
echo "[launcher] git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo '<not-a-git-repo>')"

# Step 1: datasets ----------------------------------------------------------
if [[ "$BUILD_DATASETS" -eq 1 ]]; then
    echo "[launcher] building datasets at $DATASETS_DIR"
    # shellcheck disable=SC2086
    uv run python "$REPO_ROOT/experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py" \
        --model "$HF_MODEL" \
        --families $FAMILIES \
        --n-max-per-family "$N_MAX_PER_FAMILY" \
        --out-dir "$DATASETS_DIR" \
        --no-norm-match \
        2>&1 | tee "$LOG_DIR/build_datasets.log"
fi
if ! ls "$DATASETS_DIR"/*.jsonl >/dev/null 2>&1; then
    echo "[launcher] no datasets found in $DATASETS_DIR — pass --build-datasets"
    exit 1
fi

# Step 2: swap grid ---------------------------------------------------------
RUN_ARGS=(
    --model "$MODEL"
    --datasets-dir "$DATASETS_DIR"
    --h-steps $H_STEPS
    --alignments $ALIGNMENTS
    --dtype "$DTYPE"
    --save-per-example
    --out-dir "$RUN_DIR"
    --families $FAMILIES
)
if [[ -n "$S_STEPS" ]]; then
    RUN_ARGS+=(--s-steps $S_STEPS)
fi

echo "[launcher] swap grid → $RUN_DIR"
# shellcheck disable=SC2068
uv run python "$REPO_ROOT/experiments/probes/contrastive_readout_swap/scripts/run_swap_grid.py" \
    ${RUN_ARGS[@]} \
    2>&1 | tee -a "$LOG_DIR/swap_grid.log"

# Step 3: manifest ----------------------------------------------------------
N_JSON=$(find "$RUN_DIR/shards" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
N_PT=$(find "$RUN_DIR/shards" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
N_HIDDEN=$(find "$RUN_DIR/hidden" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
{
    echo "model=$HF_MODEL"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo '?')"
    echo "families=$FAMILIES"
    echo "h_steps=$H_STEPS"
    echo "alignments=$ALIGNMENTS"
    echo "n_json_shards=$N_JSON"
    echo "n_pt_sidecars=$N_PT"
    echo "n_hidden_caches=$N_HIDDEN"
    echo "run_dir_size=$(du -sh "$RUN_DIR" 2>/dev/null | cut -f1)"
} > "$OUT_ROOT/manifest.txt"

touch "$OUT_ROOT/.swap_grid_complete"
echo "[launcher] done. manifest:"
cat "$OUT_ROOT/manifest.txt"
