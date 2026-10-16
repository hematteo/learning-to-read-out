"""Learning to Read Out: W_U trajectory crosscoders across pretraining checkpoints.

Subpackages:
    core        - SAE model zoo, data loading, paths, reproducibility/provenance
    crosscoder  - W_U trajectory-crosscoder training + checkpoint/snapshot/manifest loaders
    dynamics    - developmental-dynamics analysis (lifecycle metrics and profiles, swap helpers)
    probes      - concept/task probing (contrastive tasks, gazetteer, readout swap, SVA,
                  availability probe, recipe-control checkpoint loaders, weight swap oracle)
    baselines   - permutation / CUSUM statistical baselines
"""
