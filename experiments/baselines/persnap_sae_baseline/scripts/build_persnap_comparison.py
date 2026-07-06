"""Per-snapshot crosscoder vs per-snap SAE comparison (Ge Fig 3 analogue)."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
from readout.core.paths import ssd_path  # noqa: E402
from readout.crosscoder.wu_adapter import (  # noqa: E402
    batch_iter,
    build_crosscoder,
    load_snapshots,
)

ROOT = ssd_path()
WU_CACHE = ssd_path("snapshots")
OUT = REPO / "figures/persnap_sae_baseline"

# Default CLI values resolve the released artifacts under UM_SSD_ROOT
# (per-snap training log ships under derived/rates/); override via --sae-log / --cc-ckpt.
DEFAULT_SAE_LOG = ROOT / "derived/rates/wu-d8192-persnap/train.log"
DEFAULT_CC_CKPT = ROOT / "hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors"

from readout.core.model_specs import DEFAULT_STEPS_32  # noqa: E402

# Canonical 32-checkpoint Pythia ladder (was an inline literal; see src/readout/core/model_specs.py).
GE_STEPS = DEFAULT_STEPS_32


def per_snapshot_eval(ckpt_path, model_name="EleutherAI/pythia-160m", cache_dir=WU_CACHE, batch_size=4096):
    blob = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = blob["config"]
    steps = blob.get("steps", GE_STEPS)
    pp_stats = blob.get("preprocess_stats")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device={device}", flush=True)

    snaps, _ = load_snapshots(model_name=model_name, steps=steps, cache_dir=cache_dir, dtype=torch.float32)
    if pp_stats is not None:
        snaps = (snaps - pp_stats["mean"]) / pp_stats["scale"]
    K, V, d = snaps.shape
    print(f"  snapshots ({K}, {V}, {d}) preprocessed", flush=True)

    crosscoder = build_crosscoder(
        n_snapshots=K,
        d_model=d,
        expansion_factor=cfg["expansion_factor"],
        device=device,
        init_threshold=0.1,
        init_encoder_with_decoder_transpose_factor=1.0,
    )
    crosscoder.load_state_dict(blob["state_dict"])
    crosscoder.eval()

    hook_points = crosscoder.cfg.hook_points
    rec_sum = torch.zeros(K, dtype=torch.float64)
    n_seen = 0

    # Per-snap mean for unbiased variance estimate (use all V rows)
    snap_mean = snaps.mean(dim=1, keepdim=True)  # (K, 1, d)

    with torch.no_grad():
        for batch in batch_iter(snaps, hook_points, batch_size, shuffle=False, device=device):
            x, enc_kwargs, dec_kwargs = crosscoder.prepare_input(batch)
            feature_acts, _ = crosscoder.encode(x, return_hidden_pre=True, **enc_kwargs)
            recon = crosscoder.decode(feature_acts, **dec_kwargs)
            # x and recon shape: (B, K, d). Per-snap squared error sum across (B, d):
            err = (recon - x).pow(2)  # (B, K, d)
            rec_sum += err.sum(dim=(0, 2)).double().cpu()
            n_seen += x.shape[0]

    # Compute total per-snap variance from full snapshots tensor (CPU)
    var_per_snap = ((snaps - snap_mean) ** 2).sum(dim=(1, 2)).double()  # (K,)

    # Total elements per snap = V * d, same for each snap
    rec_mean = rec_sum / (n_seen * d)  # mean over (B, d) for each snap
    var_mean = var_per_snap / (V * d)  # mean
    ev_per_snap = 1.0 - (rec_mean / var_mean)
    return list(steps), ev_per_snap.numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sae-log",
        type=Path,
        default=DEFAULT_SAE_LOG,
        help="T1.3 per-snap SAE train.log to scrape EV/L0 from (cluster artifact).",
    )
    parser.add_argument(
        "--cc-ckpt",
        type=Path,
        default=DEFAULT_CC_CKPT,
        help="Crosscoder checkpoint to per-snapshot-eval (T4.6 seed0 d24576 by default).",
    )
    args = parser.parse_args(argv)

    # Read T1.3 per-snap SAE EV from log
    pat = re.compile(r"\[step\s+(\d+)\]\s+EV=([\d.]+)\s+L0=([\d.]+)")
    seen = {}
    with open(args.sae_log) as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                step = int(m.group(1))
                seen[step] = (float(m.group(2)), float(m.group(3)))
    sae_steps = np.array(sorted(seen))
    sae_ev = np.array([seen[s][0] for s in sae_steps])
    sae_l0 = np.array([seen[s][1] for s in sae_steps])
    print(f"T1.3 per-snap SAEs: {len(sae_steps)} snapshots, EV {sae_ev.min():.3f}-{sae_ev.max():.3f}")

    # Crosscoder eval — T4.6 seed 0 at d=24576 (160M W_U) by default
    cc_path = args.cc_ckpt
    print(f"Evaluating {cc_path.name}")
    cc_steps, cc_ev = per_snapshot_eval(cc_path)
    print(f"Crosscoder EV: {cc_ev.min():.3f}-{cc_ev.max():.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crosscoder_vs_persnap.json").write_text(
        json.dumps(
            {
                "sae": {
                    "steps": sae_steps.tolist(),
                    "ev": sae_ev.tolist(),
                    "l0": sae_l0.tolist(),
                },
                "crosscoder_t46_d24576_seed0": {
                    "steps": list(cc_steps),
                    "ev": cc_ev.tolist(),
                },
            },
            indent=2,
        )
    )
    print("Done.")


if __name__ == "__main__":
    main()
