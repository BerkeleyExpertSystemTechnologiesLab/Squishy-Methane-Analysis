"""Motion estimation from collapsed 2-D plume observations.

The project deliberately works with reduced information: a plume frame is only a
coarse 2-D projection, not the full particle state. The functions in this module
estimate the next useful abstraction: a short-horizon motion vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MotionEstimate:
    """Velocity estimate in image coordinates.

    `dy_per_t` moves along rows and `dx_per_t` moves along columns. Keeping the
    convention explicit avoids mixing mathematical x/y coordinates with array
    row/column coordinates.
    """

    dy_per_t: float
    dx_per_t: float
    method: str
    confidence: float = 1.0

    def displacement(self, delta_t: float) -> tuple[float, float]:
        """Return `(dy, dx)` displacement after `delta_t` time units."""

        return self.dy_per_t * delta_t, self.dx_per_t * delta_t

    @property
    def speed(self) -> float:
        """Euclidean speed in pixels per time unit."""

        return float(np.hypot(self.dy_per_t, self.dx_per_t))


def as_float_frame(frame: ArrayLike) -> FloatArray:
    """Convert an input frame into a finite 2-D float array."""

    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D frame, got shape {arr.shape!r}.")
    return np.nan_to_num(arr, copy=False)


def plume_mask(frame: ArrayLike, threshold_ratio: float = 0.25) -> NDArray[np.bool_]:
    """Return a coarse plume support mask.

    `threshold_ratio` is measured between the frame minimum and maximum. This is
    intentionally simple: the boundary predictor needs a stable support region,
    not a perfect segmentation of every concentration contour.
    """

    arr = as_float_frame(frame)
    if not 0.0 <= threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must be between 0 and 1.")

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(hi, lo):
        return np.zeros(arr.shape, dtype=bool)

    threshold = lo + threshold_ratio * (hi - lo)
    return arr >= threshold


def intensity_centroid(frame: ArrayLike, threshold_ratio: float = 0.25) -> tuple[float, float]:
    """Compute an intensity-weighted plume position `(row, col)`.

    Negative readings are clipped away because a plume intensity field is
    expected to be non-negative after background correction. If the frame has no
    usable mass, the geometric center is returned as a conservative fallback.
    """

    arr = as_float_frame(frame)
    mask = plume_mask(arr, threshold_ratio=threshold_ratio)
    weights = np.where(mask, np.clip(arr - np.min(arr), 0.0, None), 0.0)
    total = float(weights.sum())

    if total <= 0.0:
        rows, cols = arr.shape
        return (rows - 1) / 2.0, (cols - 1) / 2.0

    row_indices, col_indices = np.indices(arr.shape, dtype=float)
    row = float((weights * row_indices).sum() / total)
    col = float((weights * col_indices).sum() / total)
    return row, col


def estimate_centroid_motion(
    previous: ArrayLike,
    current: ArrayLike,
    *,
    delta_t: float = 1.0,
    threshold_ratio: float = 0.25,
) -> MotionEstimate:
    """Estimate plume velocity from the change in intensity centroid."""

    if delta_t <= 0:
        raise ValueError("delta_t must be positive.")

    prev_row, prev_col = intensity_centroid(previous, threshold_ratio=threshold_ratio)
    curr_row, curr_col = intensity_centroid(current, threshold_ratio=threshold_ratio)
    return MotionEstimate(
        dy_per_t=(curr_row - prev_row) / delta_t,
        dx_per_t=(curr_col - prev_col) / delta_t,
        method="centroid",
        confidence=0.8,
    )


def estimate_phase_motion(
    previous: ArrayLike,
    current: ArrayLike,
    *,
    delta_t: float = 1.0,
) -> MotionEstimate:
    """Estimate integer-pixel global translation with phase correlation.

    This is a stricter demonstration case: if the next frame is mostly a shifted
    copy of the previous frame, a single motion vector is enough to predict the
    immediate next frame up to boundary/error terms.
    """

    if delta_t <= 0:
        raise ValueError("delta_t must be positive.")

    prev = as_float_frame(previous)
    curr = as_float_frame(current)
    if prev.shape != curr.shape:
        raise ValueError("previous and current frames must have matching shapes.")

    # Remove DC/background energy so the correlation peak is driven by structure.
    prev = prev - float(prev.mean())
    curr = curr - float(curr.mean())

    cross_power = np.fft.fft2(curr) * np.conj(np.fft.fft2(prev))
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    correlation = np.fft.ifft2(cross_power).real
    peak_row, peak_col = np.unravel_index(int(np.argmax(correlation)), correlation.shape)

    dy = _unwrap_shift(peak_row, prev.shape[0])
    dx = _unwrap_shift(peak_col, prev.shape[1])
    peak_strength = float(correlation[peak_row, peak_col] / max(np.sum(np.abs(correlation)), 1e-12))
    confidence = float(np.clip(peak_strength, 0.0, 1.0))

    return MotionEstimate(
        dy_per_t=dy / delta_t,
        dx_per_t=dx / delta_t,
        method="phase_correlation",
        confidence=confidence,
    )


def _unwrap_shift(index: int, size: int) -> int:
    """Convert wrapped FFT shift index into a signed pixel shift."""

    midpoint = size // 2
    return int(index - size if index > midpoint else index)


def translate_frame(
    frame: ArrayLike,
    dy: float,
    dx: float,
    *,
    fill_value: float = 0.0,
    rounding: Literal["nearest"] = "nearest",
) -> FloatArray:
    """Translate a frame without wraparound.

    The operation uses nearest-pixel displacement because the attention-bound
    workflow only needs conservative short-horizon area propagation.
    """

    if rounding != "nearest":
        raise ValueError("Only nearest-neighbor translation is currently supported.")

    arr = as_float_frame(frame)
    out = np.full_like(arr, fill_value, dtype=float)
    shift_y = int(round(dy))
    shift_x = int(round(dx))

    src_y0 = max(0, -shift_y)
    src_y1 = min(arr.shape[0], arr.shape[0] - shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(arr.shape[1], arr.shape[1] - shift_x)

    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out
