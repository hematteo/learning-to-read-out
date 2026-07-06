"""Shared LR schedule helpers for crosscoder training.

Single source of truth for the warmup+decay multiplier used by both the
production W_U trainer (``wu_adapter``) and the architecture-sweep trainer
(``crosscoder_arch_sweep``).
"""


def lr_multiplier(
    step: int, total_steps: int, warmup_frac: float, decay_frac: float
) -> float:
    """Linear LR warmup + linear LR decay schedule (Ge §A.4: 10% warmup + 20% decay).

    Returns a multiplier in [0, 1] applied to each param group's initial_lr.
    Default of warmup_frac=decay_frac=0 yields constant LR (preserves
    backward-compat behaviour for Pythia Run 3 / Run 5 launches).
    """
    warmup_steps = int(warmup_frac * total_steps)
    decay_steps = int(decay_frac * total_steps)
    decay_start = total_steps - decay_steps

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if decay_steps > 0 and step >= decay_start:
        progress = (step - decay_start) / decay_steps
        return max(0.0, 1.0 - progress)
    return 1.0
