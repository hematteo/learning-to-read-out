#!/usr/bin/env bash
# Parallel-by-checkpoint launcher for extract_wu_hidden_standalone.py.
# Splits the 32 Pythia step list across N GPUs (default 4) and runs each as
# its own python process pinned to one GPU. Each worker writes to the shared
# wu/hidden output dirs; per-step files are tagged so workers don't collide.
#
# Args:
#   $1 = output root (e.g. /workspace/out)
#   $2 = datasets dir (e.g. /workspace/datasets)
#   $3 = num_gpus (default 4)
set -euo pipefail

OUT_ROOT="${1:-/workspace/out}"
DATASETS_DIR="${2:-/workspace/datasets}"
NGPU="${3:-4}"

WU_DIR="$OUT_ROOT/wu"
HIDDEN_DIR="$OUT_ROOT/hidden"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$WU_DIR" "$HIDDEN_DIR" "$LOG_DIR"

# Canonical 32-step Pythia schedule.
ALL_STEPS=(0 1 2 4 8 16 32 64 128 256 512 1000 2000 3000 4000 5000 6000 7000 8000 9000 14000 21000 27000 34000 47000 61000 75000 89000 102000 116000 130000 143000)

# Steps where we do NOT yet have hidden states locally — these are the ones
# the worker also forwards over the prompt set. The rest get W_U-only.
# Override with HIDDEN_STEPS_MODE=all for a fresh prompt family, or with
# HIDDEN_STEPS_LIST="0 1 2 ..." for a custom subset.
if [[ "${HIDDEN_STEPS_MODE:-missing}" == "all" ]]; then
    HIDDEN_STEPS=("${ALL_STEPS[@]}")
elif [[ -n "${HIDDEN_STEPS_LIST:-}" ]]; then
    # shellcheck disable=SC2206
    HIDDEN_STEPS=($HIDDEN_STEPS_LIST)
else
    HIDDEN_STEPS=(2 4 8 16 32 64 128 34000 47000 61000 75000 89000 102000 116000 130000)
fi

if [[ -n "${FAMILIES_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    FAMILIES=($FAMILIES_OVERRIDE)
else
    FAMILIES=(sva ioi relational_facts hypernym induction numeric_gt)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a WORKER_STEPS
for ((i = 0; i < NGPU; i++)); do
    WORKER_STEPS[$i]=""
done
for ((i = 0; i < ${#ALL_STEPS[@]}; i++)); do
    bucket=$((i % NGPU))
    WORKER_STEPS[$bucket]+="${ALL_STEPS[$i]} "
done

PIDS=()
for ((g = 0; g < NGPU; g++)); do
    steps="${WORKER_STEPS[$g]}"
    log="$LOG_DIR/worker_g${g}.log"
    echo "[launch] gpu=$g steps=[$steps]" | tee -a "$LOG_DIR/launcher.log"
    (
        export CUDA_VISIBLE_DEVICES="$g"
        export WORKER_TAG="g${g}"
        export PYTHONUNBUFFERED=1
        # shellcheck disable=SC2086
        python -u "$SCRIPT_DIR/extract_wu_hidden_standalone.py" \
            --hf-model EleutherAI/pythia-1b \
            --steps $steps \
            --hidden-steps "${HIDDEN_STEPS[@]}" \
            --datasets-dir "$DATASETS_DIR" \
            --families "${FAMILIES[@]}" \
            --wu-out-dir "$WU_DIR" \
            --hidden-out-dir "$HIDDEN_DIR" \
            --device cuda \
            --dtype fp16 \
            --hf-home "/tmp/hfh_g${g}" \
            >>"$log" 2>&1
    ) &
    PIDS+=($!)
done

echo "[launcher] PIDs: ${PIDS[*]}"
echo "[launcher] tail logs at: $LOG_DIR/worker_g*.log"

FAIL=0
for p in "${PIDS[@]}"; do
    if ! wait "$p"; then
        echo "[launcher] worker $p exited non-zero"
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "[launcher] one or more workers failed; check logs"
    exit 1
fi

# Sentinel + manifest
touch "$OUT_ROOT/.extract_complete"
{
    echo "n_wu=$(find "$WU_DIR" -name '*_wu.pt' | wc -l)"
    echo "n_hidden=$(find "$HIDDEN_DIR" -name '*.pt' | wc -l)"
    echo "wu_size=$(du -sh "$WU_DIR" | cut -f1)"
    echo "hidden_size=$(du -sh "$HIDDEN_DIR" | cut -f1)"
} >"$OUT_ROOT/manifest.txt"
echo "[launcher] complete:"
cat "$OUT_ROOT/manifest.txt"
