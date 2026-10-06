"""Redraw every paper figure into paper_figures/out/. Each script runs in its own process (fresh rcParams)."""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "lifecycle_trajectories.py",
    "wordnet_controls.py",
    "sva_availability.py",
    "recipe_dose_response.py",
    "instrument_validation.py",
    "fidelity_curves.py",
    "dense_reorganization_timing.py",
    "lifecycle_profile_composition.py",
    "lifecycle_population_diagnostics.py",
    "readout_swap_languages.py",
    "aligned_swap_grid.py",
    "probe_fits.py",
    "recipe_reorganization_window.py",
]


def main() -> int:
    here = Path(__file__).resolve().parent
    failed = [s for s in SCRIPTS if subprocess.run([sys.executable, str(here / s)]).returncode != 0]
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
