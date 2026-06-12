# Cross-family experiment: pre-registered schedule and decision rules

**Date committed:** 2026-04-27
**Target model:** `allenai/OLMo-2-1124-7B`
**Companion file:** the launch scripts in this directory; the W_U extractor in `extract_wu_olmo.py`.

This document fixes the snapshot schedule, hyperparameters, and decision rules
**before** running the cross-family OLMo-2-7B crosscoder fit. Deviations after
this date will be reported as such in the paper.

## Why OLMo-2-1124-7B

Verified via the HuggingFace API on 2026-04-27. OLMo-2-1124-7B has 6
pre-step-1000 stage1 checkpoints (`step{150, 600, 700, 850, 900, 1000}`) plus
every-1000-steps coverage thereafter through `step928000` — the most pre-1000
coverage of any non-Pythia open-source LM family. No other public family has
log-spaced checkpoints in [0, 1000]; the next-best option (`OLMo-7B-0424-hf`)
has only 3 pre-1000 checkpoints (`step{0, 500, 1000}`).

OLMo-2 is architecturally distinct from Pythia (LLaMA-style RMSNorm + SwiGLU
vs Pythia's GPT-NeoX rotary), uses a different tokenizer (Dolma BPE,
`V = 100,352` vs Pythia's GPT-NeoX BPE `V = 50,304`), and was trained on a
different corpus (Dolma vs The Pile). A positive replication is genuine
cross-family signal.

## Snapshot schedule (32 snapshots)

Mirrors the structure of our Pythia Run 3 schedule (Ge et al. 2026 §4 Fig 10)
where checkpoints exist. OLMo-2-7B does not have a `step0` checkpoint
released; `step150` (1B tokens) is the earliest available.

```python
OLMO2_7B_STEPS = [
    # Pre-1000 (6): all available stage1 sub-step-1000 OLMo-2-7B checkpoints
    150, 600, 700, 850, 900, 1000,
    # Linear early post-1000 (8): matches our Pythia Run 3 schedule
    2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    # Linear mid (4): matches Pythia Run 3
    14000, 21000, 27000, 34000,
    # Sparse late (14): evenly spaced in [47000, 928000]
    47000, 110000, 173000, 236000, 299000, 362000,
    425000, 488000, 614000, 677000, 740000, 803000,
    866000, 928000,
]
assert len(OLMO2_7B_STEPS) == 32
```

Deviations from Pythia Run 3 schedule:
- No step 0 (OLMo-2-7B does not release it; `step150` is earliest stage1).
- Pre-1000 has 6 checkpoints vs Pythia's 12 (`step{0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000}`).
- Late tail extends to step 928000 (vs Pythia's 143000) because OLMo-2-7B trains for 5 trillion tokens vs Pythia's 300 billion.

## Crosscoder hyperparameters

Extrapolated from Ge et al.'s Table 1 6.9B point (same `d_model = 4096`),
mirrored verbatim where applicable, with our Run 3 / Run 5 corrections kept
for the cases Ge under-specifies.

| Knob | Value | Source |
|---|---|---|
| Architecture | JumpReLU + tanh-STE, per-snapshot encoder/decoder pairs | Ge §3 + Run 3 |
| `d_SAE` (pilot) | 16384 | 4× expansion at `d_model=4096`; single A100-80GB feasible |
| `d_SAE` (production) | 32768 | 8× expansion, matches Ge 6.9B Table 1 verbatim; needs 4× A100-80GB head-parallel |
| Learning rate | `1e-5` | Ge Table 1, 6.9B verbatim |
| JumpReLU LR factor | `0.3` | Ge Table 1, 6.9B verbatim |
| Sparsity coefficient λ | `0.3` | Ge Table 1 + Run 3 |
| JumpReLU init threshold | `0.1` | Ge §A.4 + Run 3 |
| LR schedule | 10% linear warm-up + 20% linear decay | Ge §A.4 |
| Encoder init | `W_E ← W_D^T` (decoder transpose) | Ge §A.4 + Run 3 (load-bearing) |
| Per-snapshot input | centre + scale to `E[‖x‖²] = d` | Ge §G.1 + Run 3 |
| Optimiser | Adam β=(0.9, 0.999), fp32 | Ge §A.4 + Run 3 |
| Batch size | 2048 | Ge Table 1 |
| Epochs | 100 (pilot), 250 (production) | Production matches Ge's 800M-token sample budget; pilot is enough for event detection per Run 5 |
| Seeds | 1 (pilot), 1 (production) | Ge runs single-seed at 6.9B; we follow precedent |

L1 warm-up was 10% in Run 3; we extend to 10% warm-up + 20% decay to match Ge §A.4 verbatim.

## Statistical methodology

### Primary endpoint: per-feature rate-rotation correlation

Run 3 / Run 5 found per-feature Pearson `r(Δrate, 1−cos) ∈ [0.88, 0.96]`
across all tested Pythia scales. This statistic is **baseline-window-free**
(needs no pre-event σ estimate), so it is robust to OLMo-2-7B's reduced
pre-1000 coverage. We promote it to the primary cross-family endpoint;
CUSUM moves to secondary.

Statistic: for each feature `f`, compute Δr_f = r_f^{step1000} − r_f^{step150}
and rotation_f = 1 − cos(W_D^{step150}[f,:], W_D^{step1000}[f,:]). Pearson r
across all `d_SAE` features is the test statistic. Significance via
Fisher z-transform (no permutation needed; 8192-feature sample is large).

### Secondary endpoint: reduced-baseline CUSUM permutation test

Run 3's CUSUM used a 10-snapshot baseline window (steps 0–256 in Pythia,
genuinely flat at rate ≈ 0.072). OLMo-2-7B has only 6 pre-step-1000
checkpoints (steps 150, 600, 700, 850, 900, 1000), and step 150 is
already 1B tokens in — possibly post-event. We use the 6 pre-1000
snapshots as the baseline window and **explicitly disclose** that:

- A null result on this test cannot distinguish "no event in OLMo" from
  "event happened before step 150."
- The baseline window may itself include partial event onset.

These caveats appear verbatim in the paper's cross-family subsection.

### Pre-registered MAD floor (Run 3 fix)

For both CUSUM variants we use the robust σ floor of Run 3:

```
σ_robust = max(1e-8, Q25(σ_raw_nonzero))
```

where `σ_raw = 1.4826 × MAD` from the baseline window and `Q25(.)` is the
25th percentile of non-zero per-feature MADs. This is committed verbatim
in `eval_decision_rules.py:_cusum_max`. We pre-commit to this floor before
training; it is **not** a post-hoc adjustment.

### Pre-launch diagnostic: is step 150 already post-event?

Before running the perm test, we plot mean firing rate and consecutive
decoder cosine across the 6 pre-1000 OLMo snapshots. If mean rate at step
150 is already comparable (within 30%) to mean rate at step 1000, or if
consecutive decoder cosine drops > 0.05 between adjacent pre-1000
snapshots, we report the test as **inconclusive by data-availability
constraint** rather than negative.

## Decision rules

Pre-registered before any training run.

### Stage 1 (pilot, `d_SAE = 16384`)

| Test | Threshold | Result label |
|---|---|---|
| **Primary**: per-feature rate-rotation Pearson r | `r > 0.5` | A |
| **Secondary**: reduced-baseline CUSUM perm test (6 baseline snaps) | `p < 0.01` | B |
| Reconstruction EV | `≥ 0.55` | C — bumped from 0.40 per code review (Run 5 1B/8192 hit 0.477; 0.55 = sufficient + headroom) |
| Dead-feature rate | `< 30%` at final snapshot | D |

Decision: **proceed to Stage 2 iff A passes and at least 2 of {B, C, D} pass.**

The primary endpoint A must always hold. Secondary endpoints are weighted:
two non-A failures are tolerable; three are not. Specific partial-credit
outcomes:

| A pass? | B–D pattern | Decision | Paper framing |
|---|---|---|---|
| ✓ | all 3 pass | Proceed | "clean cross-family pilot signal" |
| ✓ | 2 of 3 pass | Proceed (with caveat) | "cross-family signal with degraded quality at d_SAE=16384" |
| ✓ | 1 of 3 passes | Stop | "rate-rotation reproduces but quality bar not met; report partial cross-family" |
| ✗ | any | Stop | "no detectable cross-family event; report negative" |

### Stage 2 (production, `d_SAE = 32768`, conditional on Stage 1 proceed)

| Test | Threshold |
|---|---|
| **Primary**: per-feature rate-rotation Pearson r | `r > 0.7` |
| **Secondary**: reduced-baseline CUSUM perm test | `p < 0.001` |
| Mean decoder cosine drop step 150 → step 1000 | `cos < 0.7` |
| Reconstruction EV | `≥ 0.65` |
| Dead-feature rate | `< 10%` at final snapshot |
| Top-50 CUSUM features: ≥ 5 monosemantic by auto-interp | Claude Sonnet 4.6 judge, monosemanticity ≥ 4/5 |
| Causal script-ablation analogue: top-10 ablation drops target script ≥ 1 nat/token | only if interpretability passes |

Cross-family **headline claim** holds iff: primary passes AND at least 4 of
6 secondary checks pass. Negative or mixed → reported honestly.

## Required controls (pre-launch)

Three additional cheap controls, run alongside the pilot:

1. **Null-data control.** Train one crosscoder at `d_SAE=16384` on Frobenius-norm-matched
   Gaussian random snapshots (32 of them, matching the OLMo W_U schedule). Run the
   primary + secondary tests. **Pre-registered prediction**: rate-rotation r ≈ 0
   (within 95% CI of 0), CUSUM `p > 0.1`. ~1 hour.
2. **Off-the-shelf linear baselines.** Raw-cosine CUSUM and top-32 SVD-subspace CUSUM
   on OLMo-2-7B `W_U` snapshots (same statistic as Run 3 §4.3). Pre-registered prediction:
   row-coherent (per-row p ≈ 0) but temporally non-specific (snapshot-shuffle p > 0.4).
   Mirrors the overnight 2026-04-26 finding on Pythia. CPU-only, minutes.
3. **Step-150-post-event diagnostic** (described above, before primary test).

## Compute budget

| Stage | Hardware | Wall |
|---|---|---|
| Pilot ($d_\mathrm{SAE} = 16384$) | 1× A100-80GB | $\sim$3 h |
| Production ($d_\mathrm{SAE} = 32768$, head-parallel) | 4× A100-80GB | $\sim$3 h |
| **Total** | | $\sim$6 h |

If Stage 1 fails decision rules, Stage 2 is skipped.

## Stopping rules

1. **Compute cap.** Halt if total wall-clock materially exceeds the ~6 h budget above.
2. **No post-hoc threshold relaxation.** If decision rules fail, the paper
   reports them as failed. We do not lower thresholds to make a result
   "publishable."
3. **No post-hoc schedule changes.** If snapshot extraction fails on a
   particular step, we either re-try or report the failure honestly. We do
   not silently swap that step for an adjacent one.

## Reporting

The cross-family results — positive, negative, or mixed — go into a
dedicated subsection of the paper draft
labelled "Cross-family replication on OLMo-2-7B." Decision-rule outcomes
are reported verbatim, including any failures.
