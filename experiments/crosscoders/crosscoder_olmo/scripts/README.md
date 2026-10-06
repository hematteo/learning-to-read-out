# crosscoder_olmo scripts

Run order, inputs, and the mapping to paper labels are in
[`../README.md`](../README.md).

| Script | Role |
|---|---|
| `extract_wu_olmo.py` | Downloads OLMo-2-1124-7B stage-1 checkpoints and writes `lm_head.weight` as fp32 `<slug>_step<N>_wu.pt`, clearing the Hugging Face cache between checkpoints. |
| `run_olmo_pilot.sh` | Single-GPU pilot fit at d_sae=16,384 (100 epochs) through `scripts/train/train_crosscoder.py`. |
| `eval_decision_rules.py` | Evaluates the gates in [`../SCHEDULE.md`](../SCHEDULE.md) for a trained crosscoder (`--stage pilot` or `production`) and writes a JSON verdict. |
| `run_olmo_production.sh` | Four-GPU head-parallel fit at d_sae=32,768 (300 epochs, batch 1024) through `train_distributed.py`; exits unless the pilot verdict sets `decision.proceed`. |
| `train_distributed.py` | `torchrun` entry point: sets up the device mesh and calls `readout.crosscoder.wu_adapter.train`. |
| `run_null_control.sh` | Fits a crosscoder to Gaussian snapshots matched in per-snapshot Frobenius norm to the OLMo snapshots. Not reported in the paper. |
