# pareto_frontier_ev_l0

Aggregate Pareto frontier of explained variance (EV) vs L0 sparsity for the W_U
crosscoder, with a lambda-sweep overlay and dictionary-size sweep, establishing
that the chosen `(d_sae, lambda)` operating point sits on the frontier.

## Figures produced

| Thesis label | Metric file | Producing script |
|---|---|---|
| `fig:app-instrument-pareto` | `results/experiments/capacity/pareto_frontier_ev_l0/ev_l0_phase2.json` | `experiments/capacity/pareto_frontier_ev_l0/scripts/compute_ev_l0_phase2.py` |
| `fig:app-lambda-sweep` | `figures/pareto_frontier_ev_l0/l0_ev_phase_plane.csv` | `experiments/capacity/pareto_frontier_ev_l0/scripts/plot_l0_ev_phase_plane.py` |

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim
Aggregate Pareto frontier of explained variance (EV) vs L0 sparsity for the W_U
crosscoder. Includes lambda-sweep overlay and dictionary-size sweep. Establishes
the chosen `(d_sae, lambda)` operating point sits on the frontier.

## Metrics produced
This experiment computes metrics only: its scripts compute and persist the EV/L0
numbers behind the Pareto figures. The figures themselves are rendered in the
separate thesis LaTeX tree from these metrics, not here.

- `results/experiments/capacity/pareto_frontier_ev_l0/ev_l0_phase2.json` —
  per-arm, per-snapshot EV and L0 (plus NRMSE, arm-level EV mean/min/max,
  active/dead feature counts) for the 8 Phase-2 grid arms
  (`compute_ev_l0_phase2.py`).
- `figures/pareto_frontier_ev_l0/l0_ev_phase_plane.csv` and
  `…/l0_ev_phase_plane_cache.pt` — per-snapshot `(L0, EV)` table across the
  crosscoder families, one row per `(label, snapshot index)` (`plot_l0_ev_phase_plane.py`,
  metrics-only despite its name).
- `derived/sae_pareto.txt` — single-snapshot SAE vs multi-snapshot crosscoder
  EV/L0 comparison table (`plot_sae_pareto.py`, metrics-only despite its name;
  written under this experiment's own `derived/`, gitignored).

## Inputs (SSD canonical paths)
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-{160m,1b}/W_U/cross-snapshot-32/d{4096,8192,16384,24576}/seed0.safetensors`
- `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-*_d*_seed0.pt`

## Reproduce (representative; scripts bootstrap the repo onto `sys.path`)
- Phase-2 EV/L0:        `uv run python experiments/capacity/pareto_frontier_ev_l0/scripts/compute_ev_l0_phase2.py`
- Per-snap L0/EV table: `uv run python experiments/capacity/pareto_frontier_ev_l0/scripts/plot_l0_ev_phase_plane.py`
- SAE-vs-CC comparison: `uv run python experiments/capacity/pareto_frontier_ev_l0/scripts/plot_sae_pareto.py`
- Phase-2 rate extraction (prereq for new arms): `bash experiments/capacity/pareto_frontier_ev_l0/scripts/extract_rates_phase2.sh`

## Layout
| path | role |
|---|---|
| `scripts/` | EV/L0 Pareto-frontier metric computation: `compute_ev_l0_phase2.py` (per-snapshot EV/L0 JSON), `plot_l0_ev_phase_plane.py` (per-snapshot `(L0, EV)` CSV/cache), `plot_sae_pareto.py` (SAE-vs-crosscoder EV/L0 text table), and `extract_rates_phase2.sh` (firing-rate extraction for new arms). No figures are rendered here. |
