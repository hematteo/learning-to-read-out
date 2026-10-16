from __future__ import annotations

import numpy as np
import pytest

from readout.dynamics.metrics import (
    adjacent_rotation,
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
