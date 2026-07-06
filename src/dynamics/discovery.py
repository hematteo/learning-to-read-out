"""Discover all crosscoder runs eligible for the dynamics analysis pipeline.

Returns one RunRow per (run_id, seed). Missing runs (still training on the cluster,
not yet rsynced) are skipped silently — phase scripts that need them check
return.empty() and degrade.

Locked manifest of expected runs from plan §2 (eligible runs):
  - run3_wu_160m_d8192:    Pythia-160M W_U, d_sae=8192, seeds 0-4 (5 seeds)
  - t1_5_gated_retune:     Pythia-160M W_U, d_sae=8192, seed 0, Gated arch
  - t1_3_per_snap_saes:    32 single-snapshot SAEs at d_sae=8192 seed 0
  - t3_1_we_multi:         Pythia-160M W_E, d_sae=8192, multi-seed
  - t3_2_we_d24576:        Pythia-160M W_E, d_sae=24576, seed 0
  - t4_4_16snap:           Pythia-160M W_U, d_sae=8192, 16-snapshot subset, seed 0
  - t4_6_d24576_multi:     Pythia-160M W_U, d_sae=24576, multi-seed (when launched)
  - t1_2_d98304:           Pythia-160M W_U, d_sae=98304 (conditional on retune)
  - pythia1b_d8192/16384/24576: Pythia-1B W_U capacity sweep, seed 0
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from src.core.model_specs import DEFAULT_STEPS_32 as GE_STEPS_32
from src.core.paths import ssd_path


# Default roots: local SSD layout. Override on a cluster with DYNAMICS_DATA_ROOT
# (single root under which run3_ge_exact_results/, cluster_results/, ... live).
# Individual sub-roots can also be overridden:
#   DYNAMICS_RUN3_DIR, DYNAMICS_CLUSTER_RESULTS, DYNAMICS_PYTHIA1B_DIR, DYNAMICS_WE_DIR
def _default_data_root() -> Path:
    env = os.environ.get("DYNAMICS_DATA_ROOT")
    if env:
        return Path(env)
    return ssd_path("wu_crosscoder")


SSD_ROOT = _default_data_root()
CLUSTER_RESULTS = Path(os.environ.get("DYNAMICS_CLUSTER_RESULTS") or (SSD_ROOT / "cluster_results"))
RUN3_DIR = Path(os.environ.get("DYNAMICS_RUN3_DIR") or (SSD_ROOT / "run3_ge_exact_results"))
T1_3_DIR = CLUSTER_RESULTS / "t1_3_per_snap_sae"
T1_5_DIR = CLUSTER_RESULTS  # Gated/BatchTopK retune lives under t1_5* subdirs
PYTHIA1B_DIR = Path(os.environ.get("DYNAMICS_PYTHIA1B_DIR") or (SSD_ROOT / "run5_pythia1b_capsweep"))
WE_DIR = Path(os.environ.get("DYNAMICS_WE_DIR") or (SSD_ROOT / "we_snapshots"))
T2_1_DIR = Path(os.environ.get("DYNAMICS_T2_1_DIR") or (SSD_ROOT / "t2_1_pythia69b"))


@dataclass
class RunRow:
    """One row of the canonical analysis table."""

    run_id: str
    model: str  # e.g. "EleutherAI/pythia-160m"
    matrix_type: str  # "W_U" | "W_E"
    d_sae: int
    seed: int
    architecture: str  # "JumpReLU" | "Gated" | "BatchTopK" | "PerSnapSAE"
    n_snapshots: int
    steps: list[int]
    ckpt_path: str | None
    rates_path: str | None  # cached firing rates (npy or pt)
    norms_path: str | None  # cached decoder norms (npy or pt)
    seed_index_in_cache: int | None  # for multi-seed npy caches
    EV: float | None = None
    L0: float | None = None
    dead_rate: float | None = None
    notes: str = ""
    # Provenance (plan §2). Populated by build_analysis_table.py from
    # provenance.file_sha256 / provenance.git_sha; None when the source
    # artifact is missing or stored as a directory.
    checkpoint_sha256: str | None = None
    rates_sha256: str | None = None
    norms_sha256: str | None = None
    metric_version: str | None = None
    git_sha: str | None = None
    # Derived-artifact paths (plan §2). Populated by dynamics.derive when the
    # corresponding cache file is present. None means "not yet derived".
    rates_norm_path: str | None = None
    rotation_path: str | None = None
    rotation_weight_path: str | None = None
    direction_to_terminal_path: str | None = None
    lifecycle_path: str | None = None
    cusum_path: str | None = None
    cusum_null_path: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Plan §2 names this field decoder_norms_path; expose both for back-compat.
        d["decoder_norms_path"] = d["norms_path"]
        return d


# 32-snapshot Ge schedule used by Run 3 / T1.3 / T1.5 / T3.x.
# GE_STEPS_32 is imported above as DEFAULT_STEPS_32 from src.core.model_specs.
GE_STEPS_16 = [GE_STEPS_32[i] for i in range(0, 32, 2)]


def _run3_rows() -> list[RunRow]:
    rates_npy = RUN3_DIR / "firing_rates_all_seeds.npy"
    norms_npy = RUN3_DIR / "decoder_norms_all_seeds.npy"
    rows: list[RunRow] = []
    for s in range(5):
        ckpt = RUN3_DIR / f"wu_cc_ge_exact_seed{s}.pt"
        rows.append(
            RunRow(
                run_id="run3_wu_160m_d8192",
                model="EleutherAI/pythia-160m",
                matrix_type="W_U",
                d_sae=8192,
                seed=s,
                architecture="JumpReLU",
                n_snapshots=32,
                steps=list(GE_STEPS_32),
                ckpt_path=str(ckpt) if ckpt.exists() else None,
                rates_path=str(rates_npy) if rates_npy.exists() else None,
                norms_path=str(norms_npy) if norms_npy.exists() else None,
                seed_index_in_cache=s,
            )
        )
    return rows


def _t1_5_rows() -> list[RunRow]:
    """T1.5 architecture sweep: Gated + BatchTopK retunes.

    Prefers the t1_5_arch_sweep canonical job dir; falls back to
    t1_5_gated_retune. One row per architecture; later matches don't override.
    """
    rows: list[RunRow] = []
    seen: set[str] = set()
    if not T1_5_DIR.exists():
        return rows
    sweep_dirs = sorted(T1_5_DIR.glob("t1_5_arch_sweep*")) + sorted(T1_5_DIR.glob("t1_5_gated_retune*"))
    for sub in sweep_dirs:
        for arch_glob, arch_name in [
            ("*gated*.pt", "Gated"),
            ("*batchtopk*.pt", "BatchTopK"),
        ]:
            for ckpt in sorted(sub.rglob(arch_glob)):
                if ckpt.name.startswith("._"):  # macOS resource forks
                    continue
                run_id = f"t1_5_{arch_name.lower()}_d8192"
                key = f"{run_id}__seed0"
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    RunRow(
                        run_id=run_id,
                        model="EleutherAI/pythia-160m",
                        matrix_type="W_U",
                        d_sae=8192,
                        seed=0,
                        architecture=arch_name,
                        n_snapshots=32,
                        steps=list(GE_STEPS_32),
                        ckpt_path=str(ckpt),
                        rates_path=None,
                        norms_path=None,
                        seed_index_in_cache=None,
                        notes="ckpt-only; rates/norms must be derived",
                    )
                )
    return rows


def _t1_3_rows() -> list[RunRow]:
    """Per-snap SAEs: 32 single-snapshot dictionaries; not a crosscoder.

    Returned as a synthetic "stacked" RunRow with ckpt_path pointing at the
    parent directory; phase scripts that consume it (per_snap_sae_vs_crosscoder.py)
    know to glob the directory rather than torch.load a single file.
    """
    if not T1_3_DIR.exists():
        return []
    runs = sorted(T1_3_DIR.glob("*/wu_sae_dsae8192_step0.pt"))
    if not runs:
        return []
    parent = runs[0].parent
    return [
        RunRow(
            run_id="t1_3_per_snap_saes",
            model="EleutherAI/pythia-160m",
            matrix_type="W_U",
            d_sae=8192,
            seed=0,
            architecture="PerSnapSAE",
            n_snapshots=32,
            steps=list(GE_STEPS_32),
            ckpt_path=str(parent),
            rates_path=None,
            norms_path=None,
            seed_index_in_cache=None,
            notes="directory of 32 per-snap SAE checkpoints",
        )
    ]


def _t3_x_rows() -> list[RunRow]:
    """T3.1 W_E multi-seed (cluster_results/we_multiseed) and T3.2 W_E d=24576."""
    rows: list[RunRow] = []
    base = CLUSTER_RESULTS
    if not base.exists():
        return rows
    seen: set[str] = set()
    # T3.1: 5 seeds at d_sae=8192. Lives under cluster_results/we_multiseed
    # (canonical) or cluster_results/t3_1* (legacy naming). First match wins.
    t3_1_search_dirs = sorted(base.glob("we_multiseed*")) + sorted(base.glob("t3_1*"))
    for d in t3_1_search_dirs:
        for ckpt in sorted(d.rglob("we_cc_dsae8192_seed*.pt")):
            if ckpt.name.startswith("._"):
                continue
            seed = _parse_seed(ckpt.stem)
            key = f"t3_1__seed{seed}"
            if key in seen:
                continue
            seen.add(key)
            rates_candidates = [
                ckpt.parent / f"we_rates_dsae8192_seed{seed}.pt",
                ckpt.parent / f"we_rates_seed{seed}.pt",
            ]
            rates_path = next((str(c) for c in rates_candidates if c.exists()), None)
            rows.append(
                RunRow(
                    run_id="t3_1_we_160m_d8192",
                    model="EleutherAI/pythia-160m",
                    matrix_type="W_E",
                    d_sae=8192,
                    seed=seed,
                    architecture="JumpReLU",
                    n_snapshots=32,
                    steps=list(GE_STEPS_32),
                    ckpt_path=str(ckpt),
                    rates_path=rates_path,
                    norms_path=None,
                    seed_index_in_cache=None,
                )
            )
    for d in sorted(base.glob("t3_2*")):
        for ckpt in sorted(d.rglob("we_cc_dsae24576_seed*.pt")):
            seed = _parse_seed(ckpt.stem)
            rows.append(
                RunRow(
                    run_id="t3_2_we_160m_d24576",
                    model="EleutherAI/pythia-160m",
                    matrix_type="W_E",
                    d_sae=24576,
                    seed=seed,
                    architecture="JumpReLU",
                    n_snapshots=32,
                    steps=list(GE_STEPS_32),
                    ckpt_path=str(ckpt),
                    rates_path=str(ckpt.parent / f"we_rates_dsae24576_seed{seed}.pt")
                    if (ckpt.parent / f"we_rates_dsae24576_seed{seed}.pt").exists()
                    else None,
                    norms_path=None,
                    seed_index_in_cache=None,
                )
            )
    return rows


def _t4_4_rows() -> list[RunRow]:
    """T4.4 16-snap: cluster_results/t4_4_16snap/wu_cc_dsae8192_seed*.pt.

    Distinguished from full-32-snap runs by the parent directory name; the
    ckpt itself uses the unprefixed `wu_cc_dsae8192_seed{N}.pt` filename.
    """
    rows: list[RunRow] = []
    base = CLUSTER_RESULTS
    if not base.exists():
        return rows
    seen: set[int] = set()
    for d in sorted(base.glob("t4_4*16snap*")):
        for ckpt in sorted(d.rglob("wu_cc_dsae8192_seed*.pt")):
            if ckpt.name.startswith("._"):
                continue
            seed = _parse_seed(ckpt.stem)
            if seed in seen:
                continue
            seen.add(seed)
            rates_candidate = ckpt.parent / f"wu_rates_dsae8192_seed{seed}.pt"
            rows.append(
                RunRow(
                    run_id="t4_4_wu_160m_d8192_16snap",
                    model="EleutherAI/pythia-160m",
                    matrix_type="W_U",
                    d_sae=8192,
                    seed=seed,
                    architecture="JumpReLU",
                    n_snapshots=16,
                    steps=list(GE_STEPS_16),
                    ckpt_path=str(ckpt),
                    rates_path=str(rates_candidate) if rates_candidate.exists() else None,
                    norms_path=None,
                    seed_index_in_cache=None,
                )
            )
    return rows


def _pythia1b_rows() -> list[RunRow]:
    """Pythia-1B capacity sweep at d=8192/16384/24576.

    Searches PYTHIA1B_DIR first (locked SSD layout), then cluster_results paths
    that contain cached rates. Dedups by (d_sae, seed); a row already populated
    via the locked dir gets its rates_path enriched if cluster_results has one.
    """
    rows_by_key: dict[tuple[int, int], RunRow] = {}
    search_dirs = []
    if PYTHIA1B_DIR.exists():
        search_dirs.append(PYTHIA1B_DIR)
    if CLUSTER_RESULTS.exists():
        search_dirs.extend(sorted(CLUSTER_RESULTS.glob("pythia1b_wu_main*")))
    for d in search_dirs:
        for ckpt in sorted(d.rglob("wu_cc_1b_dsae*_seed*.pt")):
            if ckpt.name.startswith("._"):
                continue
            row = _pythia1b_row_from_ckpt(ckpt)
            key = (row.d_sae, row.seed)
            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = row
            elif existing.rates_path is None and row.rates_path is not None:
                # Same (d_sae, seed) as an earlier ckpt but this one has cached
                # rates next to it. Prefer the one with rates available.
                rows_by_key[key] = row
    return list(rows_by_key.values())


def _pythia1b_row_from_ckpt(ckpt: Path) -> RunRow:
    stem = ckpt.stem  # wu_cc_1b_dsae{D}_seed{S}
    d_sae = _parse_int(stem, "dsae")
    seed = _parse_seed(stem)
    rates_candidate = ckpt.parent / f"wu_1b_rates_dsae{d_sae}_seed{seed}.pt"
    rates_path = str(rates_candidate) if rates_candidate.exists() else None
    if rates_path is None and CLUSTER_RESULTS.exists():
        # Look across pythia1b_wu_main job dirs for a matching rates cache.
        for d in sorted(CLUSTER_RESULTS.glob("pythia1b_wu_main*")):
            for c in sorted(d.rglob(f"wu_1b_rates_dsae{d_sae}_seed{seed}.pt")):
                if c.name.startswith("._"):
                    continue
                rates_path = str(c)
                break
            if rates_path is not None:
                break
    return RunRow(
        run_id=f"pythia1b_wu_d{d_sae}",
        model="EleutherAI/pythia-1b",
        matrix_type="W_U",
        d_sae=d_sae,
        seed=seed,
        architecture="JumpReLU",
        n_snapshots=32,
        steps=list(GE_STEPS_32),
        ckpt_path=str(ckpt),
        rates_path=rates_path,
        norms_path=None,
        seed_index_in_cache=None,
    )


def _parse_int(stem: str, prefix: str) -> int:
    """Pull `<prefix><N>` integer out of a checkpoint stem."""
    parts = stem.split("_")
    for p in parts:
        if p.startswith(prefix):
            try:
                return int(p[len(prefix) :])
            except ValueError:
                pass
    raise ValueError(f"could not parse {prefix}<N> from stem {stem}")


def _parse_seed(stem: str) -> int:
    return _parse_int(stem, "seed")


def _t4_6_rows() -> list[RunRow]:
    """T4.6 d=24576 multi-seed (spec: when launched)."""
    rows: list[RunRow] = []
    base = CLUSTER_RESULTS
    if not base.exists():
        return rows
    seen: set[int] = set()
    for d in sorted(base.glob("t4_6*d24576*")):
        for ckpt in sorted(d.rglob("wu_cc_dsae24576_seed*.pt")):
            if ckpt.name.startswith("._"):
                continue
            seed = _parse_seed(ckpt.stem)
            if seed in seen:
                continue
            seen.add(seed)
            rates_candidate = ckpt.parent / f"wu_rates_dsae24576_seed{seed}.pt"
            rows.append(
                RunRow(
                    run_id="t4_6_wu_160m_d24576",
                    model="EleutherAI/pythia-160m",
                    matrix_type="W_U",
                    d_sae=24576,
                    seed=seed,
                    architecture="JumpReLU",
                    n_snapshots=32,
                    steps=list(GE_STEPS_32),
                    ckpt_path=str(ckpt),
                    rates_path=str(rates_candidate) if rates_candidate.exists() else None,
                    norms_path=None,
                    seed_index_in_cache=None,
                )
            )
    return rows


def list_runs() -> list[RunRow]:
    """Return all discovered runs as RunRow objects.

    Discovery is best-effort: missing tiers contribute zero rows. Analysis scripts
    that hard-require a tier should check `len(rows_for_id(...))` and degrade.
    """
    rows: list[RunRow] = []
    rows.extend(_run3_rows())
    rows.extend(_t1_5_rows())
    rows.extend(_t1_3_rows())
    rows.extend(_t3_x_rows())
    rows.extend(_t4_4_rows())
    rows.extend(_t4_6_rows())
    rows.extend(_pythia1b_rows())
    rows.extend(_t2_1_rows())
    return [r for r in rows if r.ckpt_path is not None]


def _t2_1_rows() -> list[RunRow]:
    """T2.1 — Pythia-6.9B W_U crosscoder, d_sae=32768, seed 0.
    Discovers checkpoints under T2_1_DIR (override via DYNAMICS_T2_1_DIR).
    """
    rows: list[RunRow] = []
    if not T2_1_DIR.exists():
        return rows
    for ckpt in sorted(T2_1_DIR.rglob("wu_cc_p69b_dsae*_seed*.pt")):
        if ckpt.name.startswith("._"):
            continue
        d_sae = _parse_int(ckpt.stem, "dsae")
        seed = _parse_seed(ckpt.stem)
        rates_candidate = ckpt.parent / f"wu_p69b_rates_dsae{d_sae}_seed{seed}.pt"
        rows.append(
            RunRow(
                run_id=f"t2_1_pythia69b_d{d_sae}",
                model="EleutherAI/pythia-6.9b",
                matrix_type="W_U",
                d_sae=d_sae,
                seed=seed,
                architecture="JumpReLU",
                n_snapshots=32,
                steps=list(GE_STEPS_32),
                ckpt_path=str(ckpt),
                rates_path=str(rates_candidate) if rates_candidate.exists() else None,
                norms_path=None,
                seed_index_in_cache=None,
            )
        )
    return rows

