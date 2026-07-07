#!/usr/bin/env bash
# Extract firing-rate tensors for the Phase 2 architecture-sweep arms whose
# checkpoints exist but rates haven't been computed yet.
#
# Usage (from the repo root, with UM_SSD_ROOT pointing at your storage root):
#   bash experiments/capacity/pareto_frontier_ev_l0/scripts/extract_rates_phase2.sh            # all pending arms
#   bash experiments/capacity/pareto_frontier_ev_l0/scripts/extract_rates_phase2.sh t1_5 t4_4  # a subset
#
# The cluster_results/t* arm directories are the historical run layout of the
# architecture-sweep checkpoints; place retrained arm checkpoints under
# ${UM_SSD_ROOT}/cluster_results/<arm>/ to use this script.
#
# Memory budget:
#   d=8192 runs ~3 GB peak (160M unembedding × 5 seeds for T3.1 is the worst).
#   d=24576 runs ~9 GB peak.
# Wall time per arm:
#   ~2-4 minutes per (ckpt, snapshot) pair on Apple Silicon CPU; 32 snapshots
#   total per crosscoder. Run on GPU if available via --device cuda.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
SSD="${UM_SSD_ROOT:?set UM_SSD_ROOT to your storage root (see docs/DATA.md)}"
WU_CACHE=$SSD/snapshots
WE_CACHE=$SSD/we_snapshots
# Log lives in the gitignored results dir so runs never dirty the tree
# (a dirty tree poisons later git-dirty provenance checks).
LOG=$REPO/results/experiments/capacity/pareto_frontier_ev_l0/extract_rates_phase2.log
DEVICE="${EXTRACT_RATES_DEVICE:-cpu}"  # set EXTRACT_RATES_DEVICE=cuda for GPU

cd "$REPO"
mkdir -p "$(dirname "$LOG")"
export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG" ; }
fail() { log "FAIL: $*"; exit 1 ; }

# -----------------------------------------------------------------------------
# Each arm is one extract_rates invocation. The output filename pattern is
# what dynamics.discovery looks for next to each checkpoint, so the grid
# auto-detects the cache after extraction.

run_t1_5() {
  for arch_glob in "gated_dsae8192_seed*.pt" "batchtopk_dsae8192_seed*.pt"; do
    for ckpt in $(ls $SSD/cluster_results/t1_5_arch_sweep*/$arch_glob 2>/dev/null | head -1); do
      [ -f "$ckpt" ] || continue
      out=$(dirname "$ckpt")/$(basename "$ckpt" .pt)_rates.pt
      log "T1.5: $ckpt -> $out"
      uv run python scripts/extract/extract_rates.py \
        --ckpts "$ckpt" --cache-dir "$WU_CACHE" --output "$out" --device "$DEVICE" \
        || fail "extract_rates failed on $ckpt"
    done
  done
}

run_t3_1() {
  for ckpt in $SSD/cluster_results/we_multiseed*/we_cc_dsae8192_seed*.pt; do
    [ -f "$ckpt" ] || continue
    seed=$(echo "$ckpt" | sed -E 's/.*seed([0-9]+)\.pt/\1/')
    out=$(dirname "$ckpt")/we_rates_dsae8192_seed${seed}.pt
    log "T3.1: seed=$seed  $ckpt -> $out"
    uv run python scripts/extract/extract_rates.py \
      --ckpts "$ckpt" --cache-dir "$WE_CACHE" --output "$out" --device "$DEVICE" \
      || fail "extract_rates failed on $ckpt"
  done
}

run_t3_2() {
  for ckpt in $SSD/cluster_results/t3_2*/we_cc_dsae24576_seed*.pt; do
    [ -f "$ckpt" ] || continue
    seed=$(echo "$ckpt" | sed -E 's/.*seed([0-9]+)\.pt/\1/')
    out=$(dirname "$ckpt")/we_rates_dsae24576_seed${seed}.pt
    log "T3.2: seed=$seed  $ckpt -> $out"
    uv run python scripts/extract/extract_rates.py \
      --ckpts "$ckpt" --cache-dir "$WE_CACHE" --output "$out" --device "$DEVICE" \
      || fail "extract_rates failed on $ckpt"
  done
}

run_t4_4() {
  for ckpt in $SSD/cluster_results/t4_4*16snap*/wu_cc_dsae8192_seed*.pt; do
    [ -f "$ckpt" ] || continue
    seed=$(echo "$ckpt" | sed -E 's/.*seed([0-9]+)\.pt/\1/')
    out=$(dirname "$ckpt")/wu_rates_dsae8192_seed${seed}.pt
    log "T4.4: seed=$seed  $ckpt -> $out"
    uv run python scripts/extract/extract_rates.py \
      --ckpts "$ckpt" --cache-dir "$WU_CACHE" --output "$out" --device "$DEVICE" \
      || fail "extract_rates failed on $ckpt"
  done
}

run_t4_6() {
  for ckpt in $SSD/cluster_results/t4_6*d24576*/wu_cc_dsae24576_seed*.pt; do
    [ -f "$ckpt" ] || continue
    seed=$(echo "$ckpt" | sed -E 's/.*seed([0-9]+)\.pt/\1/')
    out=$(dirname "$ckpt")/wu_rates_dsae24576_seed${seed}.pt
    log "T4.6: seed=$seed  $ckpt -> $out"
    uv run python scripts/extract/extract_rates.py \
      --ckpts "$ckpt" --cache-dir "$WU_CACHE" --output "$out" --device "$DEVICE" \
      || fail "extract_rates failed on $ckpt"
  done
}

# -----------------------------------------------------------------------------
# Driver. Allows subsetting via positional arguments.

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=(t1_5 t3_1 t3_2 t4_4 t4_6)
fi

log "=== Phase 2 rate extraction start (device=$DEVICE) ==="
log "arms: ${ARMS[*]}"

for arm in "${ARMS[@]}"; do
  case "$arm" in
    t1_5) run_t1_5 ;;
    t3_1) run_t3_1 ;;
    t3_2) run_t3_2 ;;
    t4_4) run_t4_4 ;;
    t4_6) run_t4_6 ;;
    *) fail "unknown arm: $arm (expected one of t1_5 t3_1 t3_2 t4_4 t4_6)" ;;
  esac
done

log "=== Phase 2 rate extraction done ==="
