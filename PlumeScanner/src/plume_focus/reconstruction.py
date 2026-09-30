"""Sparse plume reconstruction helpers.

These functions model the alternative path described in the notes: sample a
limited number of points, assume a smooth surface between them, and reconstruct
a denser map only when boundary prediction alone is not enough.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plume_focus.motion import as_float_frame


def grid_sample_points(shape: tuple[int, int], spacing: int) -> NDArray[np.int_]:
    """Return evenly spaced `(row, col)` sample points for a frame shape."""

    if spacing <= 0:
        raise ValueError("spacing must be positive.")
    rows = range(0, shape[0], spacing)
    cols = range(0, shape[1], spacing)
    return np.array([(row, col) for row in rows for col in cols], dtype=int)


def sample_frame(frame: ArrayLike, points: ArrayLike) -> NDArray[np.float64]:
    """Read frame values at integer `(row, col)` points."""

    arr = as_float_frame(frame)
    pts = np.asarray(points, dtype=int)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2).")

    rows = np.clip(pts[:, 0], 0, arr.shape[0] - 1)
    cols = np.clip(pts[:, 1], 0, arr.shape[1] - 1)
    return arr[rows, cols]


def reconstruct_idw(
    points: ArrayLike,
    values: ArrayLike,
    shape: tuple[int, int],
    *,
    power: float = 2.0,
    smoothing: float = 1e-9,
) -> NDArray[np.float64]:
    """Reconstruct a smooth surface with inverse-distance weighting.

    IDW is intentionally lightweight and transparent. It encodes the assumption
    that nearby spatial readings are more informative than distant readings in a
    relatively flat operating environment.
    """

    pts = np.asarray(points, dtype=float)
    vals = np.asarray(values, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2).")
    if vals.ndim != 1 or vals.shape[0] != pts.shape[0]:
        raise ValueError("values must be a 1-D array matching points.")
    if pts.shape[0] == 0:
        raise ValueError("At least one sample point is required.")
    if power <= 0:
        raise ValueError("power must be positive.")

    row_grid, col_grid = np.indices(shape, dtype=float)
    targets = np.stack([row_grid.ravel(), col_grid.ravel()], axis=1)
    diff = targets[:, None, :] - pts[None, :, :]
    distances = np.linalg.norm(diff, axis=2)

    exact_hits = distances <= smoothing
    weights = 1.0 / np.maximum(distances, smoothing) ** power
    weighted = weights @ vals
    normalizer = weights.sum(axis=1)
    reconstructed = weighted / normalizer

    # Preserve measured samples exactly where a target coincides with a sample.
    hit_rows = np.where(exact_hits.any(axis=1))[0]
    if hit_rows.size:
        sample_indices = np.argmax(exact_hits[hit_rows], axis=1)
        reconstructed[hit_rows] = vals[sample_indices]

    return reconstructed.reshape(shape)
