"""Synthetic plume fields used for demonstrations and tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from plume_focus.motion import MotionEstimate


def gaussian_plume(
    shape: tuple[int, int],
    center: tuple[float, float],
    *,
    sigma: tuple[float, float] = (5.0, 8.0),
    amplitude: float = 1.0,
    background: float = 0.0,
) -> NDArray[np.float64]:
    """Return a smooth 2-D Gaussian plume intensity field."""

    row_grid, col_grid = np.indices(shape, dtype=float)
    row_term = ((row_grid - center[0]) ** 2) / (2.0 * sigma[0] ** 2)
    col_term = ((col_grid - center[1]) ** 2) / (2.0 * sigma[1] ** 2)
    return background + amplitude * np.exp(-(row_term + col_term))


def moving_gaussian_frame(
    shape: tuple[int, int],
    start_center: tuple[float, float],
    motion: MotionEstimate,
    time: float,
    *,
    sigma: tuple[float, float] = (5.0, 8.0),
    amplitude: float = 1.0,
    background: float = 0.0,
) -> NDArray[np.float64]:
    """Return a Gaussian plume advected by a constant motion vector."""

    dy, dx = motion.displacement(time)
    center = (start_center[0] + dy, start_center[1] + dx)
    return gaussian_plume(shape, center, sigma=sigma, amplitude=amplitude, background=background)


def add_sensor_noise(
    frame: NDArray[np.float64],
    *,
    std: float,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Add non-negative Gaussian sensor noise to a plume frame."""

    if std < 0:
        raise ValueError("std must be non-negative.")
    rng = np.random.default_rng(seed)
    noisy = frame + rng.normal(0.0, std, size=frame.shape)
    return np.clip(noisy, 0.0, None)
