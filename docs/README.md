# docs/

Companion guides for the `learning-to-read-out` code release.
This is a code-only release: it ships source, tests, and the manifest, but no
data and no metrics. Start with the repo root [README](../README.md) for the
overview and quickstart, then read these in order:

1. [REPRODUCE.md](REPRODUCE.md) — figure -> experiment -> metric-producing
   script map; companion to `experiments.yaml`. Start here to chase a paper
   figure.
2. [DATA.md](DATA.md) — external models, checkpoints, and corpora, plus the
   `UM_SSD_ROOT` storage layout the analysis scripts read from.
3. [THIRD_PARTY.md](THIRD_PARTY.md) — the vendored OpenMOSS `llamascopium`
   library, the single local patch, and what was trimmed.

Back to the repository root: [../README.md](../README.md).
