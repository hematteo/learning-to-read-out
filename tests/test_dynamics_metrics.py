from __future__ import annotations

import numpy as np
import pytest

from readout.dynamics.metrics import (
    adjacent_rotation,
    lifecycle,
    terminal_direction_distance,
)


def _unit(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    return np.array([np.cos(radians), np.sin(radians)], dtype=np.float64)


def test_terminal_direction_distance_tracks_terminal_not_adjacent_motion():
    W = np.array(
        [
            [_unit(60.0)],
            [_unit(40.0)],
            [_unit(20.0)],
            [_unit(0.0)],
        ]
    )

    terminal = terminal_direction_distance(W)
    adjacent = adjacent_rotation(W)

    assert terminal.shape == (4, 1)
    assert adjacent.shape == (3, 1)
    assert terminal[-1, 0] == pytest.approx(0.0, abs=1e-10)
    assert terminal[0, 0] > terminal[1, 0] > terminal[2, 0] > terminal[3, 0]
    assert np.all(adjacent[:, 0] < 0.1)


def test_lifecycle_stabilization_uses_terminal_direction_when_provided():
    rates = np.ones((4, 1), dtype=np.float64)
    W = np.array(
        [
            [_unit(60.0)],
            [_unit(40.0)],
            [_unit(20.0)],
            [_unit(0.0)],
        ]
    )
    norms = np.linalg.norm(W, axis=-1)
    adjacent = adjacent_rotation(W)
    terminal = terminal_direction_distance(W)

    exact = lifecycle(
        rates,
        decoder_norms=norms,
        rotation=adjacent,
        direction_to_terminal=terminal,
        m=1,
        delta=0.1,
    )
    fallback = lifecycle(
        rates,
        decoder_norms=norms,
        rotation=adjacent,
        m=1,
        delta=0.1,
    )

    assert exact.stabilized_step.tolist() == [2]
    assert fallback.stabilized_step.tolist() == [0]


def test_lifecycle_rejects_bad_terminal_direction_shape():
    rates = np.ones((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="direction_to_terminal"):
        lifecycle(rates, direction_to_terminal=np.ones((3, 2)))


def test_cusum_max_abs_hand_value_and_deprecated_alias():
    """cusum_max was renamed cusum_max_abs (the old name collided with the
    different signed statistic in readout.baselines.permutation_test_fast);
    the deprecated module-level alias must keep old imports working."""
    from readout.dynamics import metrics

    assert metrics.cusum_max is metrics.cusum_max_abs

    # Hand value: x = [0, 0, 1, 1] -> centered [-.5, -.5, .5, .5],
    # cumsum [-.5, -1, -.5, 0] -> max |.| = 1. Flat feature -> 0.
    x = np.array([[0.0, 3.0], [0.0, 3.0], [1.0, 3.0], [1.0, 3.0]])
    out = metrics.cusum_max_abs(x)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)
