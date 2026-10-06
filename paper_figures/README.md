# Paper figures

Scripts that redraw the paper's data figures from small files shipped in [`data/`](data/README.md) (4.9 MB).
They need only matplotlib, numpy and pandas: no GPU, model downloads or external drives. The `figures` extra
pins matplotlib 3.10, with which every figure was checked against the camera-ready PDFs.

Redraw all of them into `paper_figures/out/`:

```bash
make figures   # uv run --extra figures python paper_figures/make_all.py
```

or one at a time, e.g. `python paper_figures/wordnet_controls.py`. `data/README.md` lists where each data file
comes from and which experiment script regenerates it.

| Paper label | Script | Data files | Status |
| --- | --- | --- | --- |
| `fig:main-selected-normalized-trajectories` | `lifecycle_trajectories.py` | `lifecycle_*.npz` | matches the paper; trajectories are stored as float16, which shifts the rasterized lines by a few gray levels |
| `fig:main-wordnet-matched-controls` | `wordnet_controls.py` | `wordnet_matched_controls_160m.csv` | identical |
| `fig:main-sva-availability-expression` | `sva_availability.py` | `sva_hidden_probe_trajectory_69b.csv`, `sva_native_swept_transcribed_69b.csv`, `availability_probe_summary_69b.csv` | matches; the native and best-swept series of the left panel are transcribed from the original figure (no run output) |
| `fig:main-recipe-dose-response` | `recipe_dose_response.py` | `recipe_lifecycle_multiseed.csv`, `recipe_val_loss.csv` | identical |
| `fig:main-multimodel-validation` | `instrument_validation.py` | `instrument_inventory.csv`, `persnap_fidelity_160m.csv`, `persnap_fidelity_1b_d24576.csv`, `large_dictionary_summary.csv`, `large_dictionary_per_snapshot.csv` | identical |
| `fig:app-fidelity-curves` | `fidelity_curves.py` | `fidelity_curves.csv` | identical |
| `fig:app-dense-reorg-pythia` | `dense_reorganization_timing.py` | `dense_subspace_metrics.csv`, `dense_sv_stability.csv` | identical |
| `fig:app-lifecycle-profile-composition` | `lifecycle_profile_composition.py` | `lifecycle_*.npz` | identical |
| `fig:app-selected-population-lifecycle-diagnostics` | `lifecycle_population_diagnostics.py` | `lifecycle_*.npz` | identical |
| `fig:app-readout-swap-english-aggregate` | `readout_swap_languages.py` (writes `readout_swap_aggregate.pdf`) | `language_swap_summary.csv` | identical |
| `fig:app-readout-swap-languages` | `readout_swap_languages.py` (writes `readout_swap_languages.pdf`) | `language_swap_summary.csv` | identical |
| `fig:app-aligned-readout-swap` | `aligned_swap_grid.py` | `aligned_swap_summary_1b.csv` | identical |
| `fig:app-ae-capacity` | `probe_fits.py` | `availability_probe_summary_69b.csv` | identical |
| `fig:app-recipe-control-reorg` | `recipe_reorganization_window.py` | `recipe_reorg_window_seed0.csv` | identical |

"Identical" means the rendered page matches the paper's PDF pixel for pixel at 60 and 150 dpi. Figure 1 is a TikZ
schematic and is not drawn here, nor are the appendix figures not listed above.

`lifecycle_profile_composition.py` imports the profile rules from `readout.dynamics.lifecycle` (numpy only);
`_common.py` adds `src/` to the import path, so the package need not be installed.
