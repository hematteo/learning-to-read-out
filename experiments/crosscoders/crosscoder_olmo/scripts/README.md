# Cross-family experiment — OLMo-2-7B-1124

Cross-family validation of the step-1000 W_U reorganisation event for thesis
Chapter 4: confirms the discrete event reproduces in a non-Pythia LM family
(OLMo-2-7B-1124), not just the Pythia suite.

## Files in this directory

| File | What it does | When to read |
|---|---|---|
| [SCHEDULE.md](SCHEDULE.md) | Pre-registered snapshot schedule + decision-rule thresholds. Committed 2026-04-27 before any training. | First — the pre-registered plan |
| [extract_wu_olmo.py](extract_wu_olmo.py) | Downloads OLMo-2-7B checkpoints, extracts `lm_head.weight` (W_U) at the 32 pre-registered steps, writes fp32 .pt files compatible with `wu_adapter.load_snapshots`. Cleans HF cache between checkpoints. | Prereq for both stages |
| [run_olmo_pilot.sh](run_olmo_pilot.sh) | Stage 1: single A100-80GB, d_SAE=16384, 100 epochs, 1 seed. Wall ~3 h. Goal: pass 4 pilot gates. | Always run first |
| [eval_decision_rules.py](eval_decision_rules.py) | Computes the 4 pilot / 5 production gates from a trained crosscoder. JSON verdict, exit code = pass/fail. | After each training run |
| [run_olmo_production.sh](run_olmo_production.sh) | Stage 2: 4× A100-80GB head-parallel, d_SAE=32768, 250 epochs, 1 seed. Wall ~3 h. Refuses to launch unless pilot verdict is `all_gates_pass=true`. | Conditional on pilot positive |
| [train_distributed.py](train_distributed.py) | torchrun entry point for production. Sets up `DeviceMesh`, calls `wu_adapter.train` with the mesh wired in. | Invoked by `run_olmo_production.sh`; rarely run by hand |

## Run order

```
   ┌──────────────────────────────────────────────┐
   │ 0. Read SCHEDULE.md                          │
   └──────────────┬───────────────────────────────┘
                  ▼
   ┌──────────────────────────────────────────────┐
   │ 1. Provision a 1× A100-80GB GPU host         │
   └──────────────┬───────────────────────────────┘
                  ▼
   ┌──────────────────────────────────────────────┐
   │ 2. extract_wu_olmo.py                        │
   │    32 OLMo-2-7B-1124 W_U snapshots → cache   │
   │    ~30 min                                   │
   └──────────────┬───────────────────────────────┘
                  ▼
   ┌──────────────────────────────────────────────┐
   │ 3. run_olmo_pilot.sh + eval_decision_rules   │
   │    d_SAE=16384, single GPU, ~3 h             │
   └──────────────┬───────────────────────────────┘
                  ▼
              ┌───┴────┐
              │ Pilot  │
              │ pass?  │
              └───┬────┘
              ┌───┴───┐
              │       │
            yes       no  ─►  Stop here;
              │           report the negative
              ▼           cross-family result.
   ┌──────────────────────────────────────────────┐
   │ 4. Switch to a 4× A100-80GB host             │
   │    (or reuse the same host from step 1)      │
   └──────────────┬───────────────────────────────┘
                  ▼
   ┌──────────────────────────────────────────────┐
   │ 5. run_olmo_production.sh + eval (production)│
   │    d_SAE=32768, head-parallel, ~3 h          │
   └──────────────┬───────────────────────────────┘
                  ▼
   ┌──────────────────────────────────────────────┐
   │ 6. Local: auto-interp on top-50 CUSUM        │
   │    features (final 2 production gates)       │
   └──────────────┬───────────────────────────────┘
                  ▼
   ┌──────────────────────────────────────────────┐
   │ 7. Record the cross-family verdict outcomes  │
   │    (positive, negative, or mixed)            │
   └──────────────────────────────────────────────┘
```

## Compute

| Stage | Hardware | Wall |
|---|---|---|
| 0 — extract W_U | (any) | ~30 min |
| 1 — pilot | 1× A100-80GB | ~3 h |
| 2 — production (conditional) | 4× A100-80GB | ~3 h |
| 3 — auto-interp + ablation | local | 1–2 h |

See SCHEDULE.md "Stopping rules" for when to halt a run.

## Decision flow

The 4 pilot gates and 5 of 7 production gates are computed by
`eval_decision_rules.py`. The remaining 2 production gates require a
manual auto-interp + causal-ablation pass:

| Gate | Computed by |
|---|---|
| Reduced-baseline CUSUM perm test | `eval_decision_rules.py` |
| Rate-rotation correlation r | `eval_decision_rules.py` |
| Reconstruction EV | `eval_decision_rules.py` |
| Dead-feature rate | `eval_decision_rules.py` |
| Decoder cosine drop step150 → step1000 | `eval_decision_rules.py` (production only) |
| Top-50 CUSUM features ≥ 5 monosemantic | manual auto-interp pass post-train |
| Causal script-ablation ≥ 1 nat/token | manual ablation script post-train |

## Distributed-training notes

Implementation notes for the multi-GPU production path, flagged in
`train_distributed.py`:

1. **Wire `device_mesh` through `wu_adapter.build_crosscoder`.** Currently
   builds a non-distributed Crosscoder. Needs `device_mesh=mesh` threaded
   into `CrosscoderConfig` so the W_E/W_D/b_E/b_D params are constructed
   as DTensors sharded along the `head` axis. Llamascopium's Crosscoder
   class already supports this — see
   `lib/Language-Model-SAEs/src/llamascopium/models/crosscoder.py:122-152`.
   Estimated effort: half a day on a multi-GPU host.

2. **Verify the LR-decay schedule.** Ge §A.4 specifies 10% linear warm-up
   + 20% linear decay. Our `wu_adapter.train` currently implements warm-up
   only. The `--l1-decay-fraction` arg in `train_distributed.py` is wired
   but not yet honoured by `train()` — needs a small extension to the LR
   scheduler.

3. **Verify head-parallel correctness with a tiny smoke run.** Before
   launching the 4-GPU production fit, do a 2-GPU × 4-snapshot × 5-epoch
   smoke run to confirm All-Reduce produces the same loss as a single-GPU
   8-snapshot run on the same data (within fp32 tolerance).

The pilot (single GPU, item 1 not required) can be run immediately. Items
1–3 only block the production stage.

## Interpreting the outcome

A **positive** cross-family result means the discrete step-1000 reorganisation
is not Pythia-specific: on OLMo-2-7B-1124 — a different architecture (LLaMA-style
RMSNorm + SwiGLU vs Pythia's GPT-NeoX), tokenizer (Dolma BPE, `V = 100,352` vs
GPT-NeoX BPE, `V = 50,304`), and training corpus (Dolma vs The Pile) — the
per-feature rate-rotation correlation and the reduced-baseline CUSUM event fire
as they do on Pythia.

A **negative or inconclusive** result is reported honestly: OLMo-2-7B releases
only 6 pre-step-1000 checkpoints (vs Pythia's 12), so a null cannot fully
distinguish "no event in OLMo" from "event before step 150." The
pre-registration in `SCHEDULE.md` commits to reporting whichever outcome
obtains, with that caveat stated.
