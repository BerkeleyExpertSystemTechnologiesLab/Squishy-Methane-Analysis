"""Attention-window prediction from motion plus explicit uncertainty bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plume_focus.motion import MotionEstimate, as_float_frame, plume_mask


@dataclass(frozen=True)
class BoundingBox:
    """Inclusive image-space bounding box around the plume support."""

    min_row: int
    min_col: int
    max_row: int
    max_col: int

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    def shifted(self, dy: float, dx: float) -> "BoundingBox":
        """Move the box by a possibly fractional displacement."""

        return BoundingBox(
            min_row=int(np.floor(self.min_row + dy)),
            min_col=int(np.floor(self.min_col + dx)),
            max_row=int(np.ceil(self.max_row + dy)),
            max_col=int(np.ceil(self.max_col + dx)),
        )

    def expanded(self, radius: float) -> "BoundingBox":
        """Expand the box by `radius` pixels in every direction."""

        grow = int(np.ceil(max(0.0, radius)))
        return BoundingBox(
            min_row=self.min_row - grow,
            min_col=self.min_col - grow,
            max_row=self.max_row + grow,
            max_col=self.max_col + grow,
        )

    def clipped(self, shape: tuple[int, int]) -> "BoundingBox":
        """Clip the box to a frame shape."""

        rows, cols = shape
        return BoundingBox(
            min_row=int(np.clip(self.min_row, 0, rows - 1)),
            min_col=int(np.clip(self.min_col, 0, cols - 1)),
            max_row=int(np.clip(self.max_row, 0, rows - 1)),
            max_col=int(np.clip(self.max_col, 0, cols - 1)),
        )

    def contains(self, other: "BoundingBox") -> bool:
        """Return True when `other` is fully inside this box."""

        return (
            self.min_row <= other.min_row
            and self.min_col <= other.min_col
            and self.max_row >= other.max_row
            and self.max_col >= other.max_col
        )

    def to_slices(self) -> tuple[slice, slice]:
        """Return row/column slices for indexing an array."""

        return slice(self.min_row, self.max_row + 1), slice(self.min_col, self.max_col + 1)


@dataclass(frozen=True)
class BoundConfig:
    """Parameters that convert known constraints into an attention margin.

    Units are pixels and time units. `wind_error_speed` captures the difference
    between the measured/average wind proxy and actual plume advection.
    `model_error_radius` captures unmodeled buoyancy, diffusion, thresholding,
    and coarse-resolution effects. `scan_age` captures the fact that a raster
    frame is not truly instantaneous.
    """

    wind_error_speed: float = 1.0
    model_error_radius: float = 1.0
    scan_age: float = 0.0
    plume_spread_speed: float = 0.25
    quantization_radius: float = 0.5

    def uncertainty_radius(self, motion: MotionEstimate, delta_t: float) -> float:
        """Return the scalar bound added around the shifted plume area."""

        if delta_t < 0:
            raise ValueError("delta_t must be non-negative.")
        if self.scan_age < 0:
            raise ValueError("scan_age must be non-negative.")

        wind_term = self.wind_error_speed * delta_t
        temporal_smear = motion.speed * self.scan_age
        spread_term = self.plume_spread_speed * (delta_t + self.scan_age)
        return wind_term + temporal_smear + spread_term + self.model_error_radius + self.quantization_radius


@dataclass(frozen=True)
class AttentionPrediction:
    """Predicted next attention area and the margin that produced it."""

    source_box: BoundingBox
    shifted_box: BoundingBox
    attention_box: BoundingBox
    uncertainty_radius: float
    delta_t: float
    motion: MotionEstimate

    def mask(self, shape: tuple[int, int]) -> NDArray[np.bool_]:
        """Return a boolean mask of the predicted attention area."""

        out = np.zeros(shape, dtype=bool)
        box = self.attention_box.clipped(shape)
        out[box.to_slices()] = True
        return out


@dataclass(frozen=True)
class SmoothAttentionPrediction:
    """Gaussian attention field around the predicted next support region.

    Unlike `AttentionPrediction`, this is not a hard rectangle. It treats the
    shifted object/plume support as the most likely region and decays smoothly
    with distance according to the uncertainty bound.
    """

    source_box: BoundingBox
    predicted_center: tuple[float, float]
    sigma_y: float
    sigma_x: float
    field: NDArray[np.float64]
    uncertainty_radius: float
    delta_t: float
    motion: MotionEstimate

    def threshold_mask(self, threshold: float = 0.25) -> NDArray[np.bool_]:
        """Return pixels with attention probability at least `threshold`."""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        return self.field >= threshold

    def threshold_box(self, threshold: float = 0.25) -> BoundingBox:
        """Return a box around the smooth field above `threshold`."""

        return bounding_box_from_mask(self.threshold_mask(threshold))


def bounding_box_from_mask(mask: NDArray[np.bool_]) -> BoundingBox:
    """Build the smallest inclusive bounding box around True pixels."""

    if mask.ndim != 2:
        raise ValueError("mask must be 2-D.")
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("Cannot build a bounding box from an empty mask.")
    return BoundingBox(
        min_row=int(rows.min()),
        min_col=int(cols.min()),
        max_row=int(rows.max()),
        max_col=int(cols.max()),
    )


def predict_attention_area(
    frame: ArrayLike,
    motion: MotionEstimate,
    *,
    delta_t: float = 1.0,
    threshold_ratio: float = 0.25,
    bounds: BoundConfig | None = None,
) -> AttentionPrediction:
    """Predict where the plume should be scanned next.

    This is the code version of:

    `next area = last area + motion vector + normal/error bound`

    The result is deliberately a bounded region, not a claim that the plume can
    be reconstructed exactly from reduced observations.
    """

    if delta_t < 0:
        raise ValueError("delta_t must be non-negative.")

    arr = as_float_frame(frame)
    mask = plume_mask(arr, threshold_ratio=threshold_ratio)
    source_box = bounding_box_from_mask(mask)
    dy, dx = motion.displacement(delta_t)
    shifted = source_box.shifted(dy, dx)

    bound_config = bounds or BoundConfig()
    radius = bound_config.uncertainty_radius(motion, delta_t)
    attention = shifted.expanded(radius).clipped(arr.shape)

    return AttentionPrediction(
        source_box=source_box,
        shifted_box=shifted,
        attention_box=attention,
        uncertainty_radius=radius,
        delta_t=delta_t,
        motion=motion,
    )


def predict_smooth_attention(
    frame: ArrayLike,
    motion: MotionEstimate,
    *,
    delta_t: float = 1.0,
    threshold_ratio: float = 0.25,
    bounds: BoundConfig | None = None,
) -> SmoothAttentionPrediction:
    """Predict the next attention region as a smooth Gaussian field.

    This better matches the moving-object proof idea: the object has a most
    likely next location, and uncertainty should fade continuously rather than
    stop abruptly at a rectangular boundary.
    """

    if delta_t < 0:
        raise ValueError("delta_t must be non-negative.")

    arr = as_float_frame(frame)
    source_box = bounding_box_from_mask(plume_mask(arr, threshold_ratio=threshold_ratio))
    dy, dx = motion.displacement(delta_t)
    center_y = (source_box.min_row + source_box.max_row) / 2.0 + dy
    center_x = (source_box.min_col + source_box.max_col) / 2.0 + dx

    bound_config = bounds or BoundConfig()
    radius = bound_config.uncertainty_radius(motion, delta_t)

    # Convert support size plus bound radius into Gaussian spread. The lower
    # bound prevents needle-like fields for tiny objects.
    sigma_y = max(1.0, source_box.height / 3.0 + radius / 2.0)
    sigma_x = max(1.0, source_box.width / 3.0 + radius / 2.0)
    field = gaussian_attention_field(arr.shape, center=(center_y, center_x), sigma=(sigma_y, sigma_x))

    return SmoothAttentionPrediction(
        source_box=source_box,
        predicted_center=(center_y, center_x),
        sigma_y=sigma_y,
        sigma_x=sigma_x,
        field=field,
        uncertainty_radius=radius,
        delta_t=delta_t,
        motion=motion,
    )


def gaussian_attention_field(
    shape: tuple[int, int],
    *,
    center: tuple[float, float],
    sigma: tuple[float, float],
) -> NDArray[np.float64]:
    """Return a normalized 2-D Gaussian attention field."""

    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape must be a positive (rows, cols) tuple.")
    if sigma[0] <= 0 or sigma[1] <= 0:
        raise ValueError("sigma values must be positive.")

    row_grid, col_grid = np.indices(shape, dtype=float)
    row_term = ((row_grid - center[0]) ** 2) / (2.0 * sigma[0] ** 2)
    col_term = ((col_grid - center[1]) ** 2) / (2.0 * sigma[1] ** 2)
    field = np.exp(-(row_term + col_term))
    return field / max(float(field.max()), 1e-12)
