"""CPU-friendly source-aware plume belief propagation.

The transition implemented here is deliberately lighter than CFD.  Existing
belief is advected and diffused, while a known persistent source emits belief
continuously along every intermediate wind position over the prediction
horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from plume_focus.motion import MotionEstimate
from plume_focus.predictor import BoundingBox


@dataclass(frozen=True)
class PlumeDynamicsConfig:
    """Parameters for continuous-source plume propagation.

    The default emission rate is the time-consistent equivalent of the former
    two-percent update at the canonical 12 FPS synthetic rate.  It is an
    explicit model parameter, not an episode-specific truth input.
    """

    source_emission_rate_per_second: float = 0.24
    max_emission_step_px: float = 1.0
    source_kernel_sigma_px: float = 2.0
    prior_crosswind_sigma_min_px: float = 8.0
    prior_crosswind_sigma_frame_fraction: float = 0.12
    prior_downwind_fade_length_px: float = 18.0
    prior_upwind_tail_weight: float = 0.02
    prior_upwind_decay_length_px: float = 8.0
    prior_source_sigma_min_px: float = 2.0
    prior_source_sigma_crosswind_fraction: float = 0.18
    prior_source_kernel_weight: float = 0.2
    prior_roi_background_weight: float = 0.15
    prior_roi_focus_weight: float = 0.85

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.source_emission_rate_per_second)
            or self.source_emission_rate_per_second < 0.0
        ):
            raise ValueError(
                "source_emission_rate_per_second must be finite and "
                "non-negative."
            )
        if (
            not np.isfinite(self.max_emission_step_px)
            or self.max_emission_step_px <= 0.0
        ):
            raise ValueError(
                "max_emission_step_px must be finite and positive."
            )
        if (
            not np.isfinite(self.source_kernel_sigma_px)
            or self.source_kernel_sigma_px < 0.0
        ):
            raise ValueError(
                "source_kernel_sigma_px must be finite and non-negative."
            )
        non_negative = {
            "prior_crosswind_sigma_min_px": self.prior_crosswind_sigma_min_px,
            "prior_crosswind_sigma_frame_fraction": (
                self.prior_crosswind_sigma_frame_fraction
            ),
            "prior_upwind_tail_weight": self.prior_upwind_tail_weight,
            "prior_source_sigma_min_px": self.prior_source_sigma_min_px,
            "prior_source_sigma_crosswind_fraction": (
                self.prior_source_sigma_crosswind_fraction
            ),
            "prior_source_kernel_weight": self.prior_source_kernel_weight,
            "prior_roi_background_weight": self.prior_roi_background_weight,
            "prior_roi_focus_weight": self.prior_roi_focus_weight,
        }
        for name, value in non_negative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        positive = {
            "prior_downwind_fade_length_px": (
                self.prior_downwind_fade_length_px
            ),
            "prior_upwind_decay_length_px": (
                self.prior_upwind_decay_length_px
            ),
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isclose(
            self.prior_roi_background_weight
            + self.prior_roi_focus_weight,
            1.0,
        ):
            raise ValueError(
                "prior ROI background and focus weights must sum to 1."
            )


def propagate_plume_belief(
    posterior: NDArray[np.float64],
    motion: MotionEstimate,
    *,
    diffusion_sigma: float,
    delta_t: float = 1.0,
    source: tuple[float, float] | None = None,
    config: PlumeDynamicsConfig | None = None,
) -> NDArray[np.float64]:
    """Advance a plume belief through advection, diffusion, and emission."""

    values = _validated_distribution(posterior)
    if not np.isfinite(delta_t) or delta_t < 0.0:
        raise ValueError("delta_t must be finite and non-negative.")
    if not np.isfinite(diffusion_sigma) or diffusion_sigma < 0.0:
        raise ValueError("diffusion_sigma must be finite and non-negative.")
    if delta_t == 0.0:
        return _normalize_probability(values)

    dynamics = config or PlumeDynamicsConfig()
    advected = _advect(
        values,
        dy=motion.dy_per_t * delta_t,
        dx=motion.dx_per_t * delta_t,
    )
    advected = _gaussian_blur(advected, diffusion_sigma)
    advected = _normalize_probability(advected)

    if source is None or dynamics.source_emission_rate_per_second == 0.0:
        return advected

    _validate_source(source, values.shape)
    emitted = _continuous_emission_trail(
        values.shape,
        source=source,
        motion=motion,
        delta_t=delta_t,
        max_step_px=dynamics.max_emission_step_px,
    )
    emitted = _gaussian_blur(
        emitted,
        max(
            dynamics.source_kernel_sigma_px,
            diffusion_sigma * 0.5,
        ),
    )
    emitted = _normalize_probability(emitted)
    emitted_fraction = 1.0 - np.exp(
        -dynamics.source_emission_rate_per_second * delta_t
    )
    combined = (
        (1.0 - emitted_fraction) * advected
        + emitted_fraction * emitted
    )
    return _normalize_probability(combined)


def source_oriented_prior(
    shape: tuple[int, int],
    *,
    source: tuple[float, float],
    wind_motion: MotionEstimate,
    roi: BoundingBox | None = None,
    config: PlumeDynamicsConfig | None = None,
) -> NDArray[np.float64]:
    """Create a broad source prior oriented by the complete wind vector."""

    dynamics = config or PlumeDynamicsConfig()
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape must contain two positive dimensions.")
    _validate_source(source, shape)
    rows, cols = np.indices(shape, dtype=float)
    source_row, source_col = source
    relative_rows = rows - source_row
    relative_cols = cols - source_col
    speed = float(
        np.hypot(wind_motion.dy_per_t, wind_motion.dx_per_t)
    )
    crosswind_sigma = max(
        dynamics.prior_crosswind_sigma_min_px,
        shape[0] * dynamics.prior_crosswind_sigma_frame_fraction,
    )

    if speed <= 1e-9:
        radial_distance_sq = relative_rows**2 + relative_cols**2
        prior = np.exp(
            -radial_distance_sq / (2.0 * crosswind_sigma**2)
        )
    else:
        unit_row = wind_motion.dy_per_t / speed
        unit_col = wind_motion.dx_per_t / speed
        alongwind = (
            relative_rows * unit_row
            + relative_cols * unit_col
        )
        crosswind = (
            -relative_rows * unit_col
            + relative_cols * unit_row
        )
        crosswind_profile = np.exp(
            -(crosswind**2) / (2.0 * crosswind_sigma**2)
        )
        downstream_distance = np.maximum(0.0, alongwind)
        fade = 1.0 - np.exp(
            -downstream_distance
            / dynamics.prior_downwind_fade_length_px
        )
        downwind_corridor = crosswind_profile * fade
        upwind_tail = (
            dynamics.prior_upwind_tail_weight
            * crosswind_profile
            * np.exp(
                np.minimum(alongwind, 0.0)
                / dynamics.prior_upwind_decay_length_px
            )
        )
        prior = np.where(
            alongwind >= 0.0,
            downwind_corridor,
            upwind_tail,
        )

        source_sigma = max(
            dynamics.prior_source_sigma_min_px,
            crosswind_sigma
            * dynamics.prior_source_sigma_crosswind_fraction,
        )
        source_kernel = np.exp(
            -(
                relative_rows**2
                + relative_cols**2
            )
            / (2.0 * source_sigma**2)
        )
        prior += dynamics.prior_source_kernel_weight * source_kernel

    if roi is not None:
        roi_mask = np.zeros(shape, dtype=float)
        roi_mask[roi.clipped(shape).to_slices()] = 1.0
        prior = (
            dynamics.prior_roi_background_weight * prior
            + dynamics.prior_roi_focus_weight * prior * roi_mask
        )

    return _normalize_probability(prior)


def _validated_distribution(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("posterior must be a non-empty 2-D array.")
    if np.any(~np.isfinite(array)):
        raise ValueError("posterior must contain only finite values.")
    if np.any(array < 0.0):
        raise ValueError("posterior must be non-negative.")
    return array


def _validate_source(
    source: tuple[float, float],
    shape: tuple[int, int],
) -> None:
    if len(source) != 2 or not np.all(np.isfinite(source)):
        raise ValueError("source must contain two finite coordinates.")
    source_row, source_col = source
    if not (
        0.0 <= source_row <= shape[0] - 1
        and 0.0 <= source_col <= shape[1] - 1
    ):
        raise ValueError("source must lie inside the plume field.")


def _advect(
    values: NDArray[np.float64],
    *,
    dy: float,
    dx: float,
) -> NDArray[np.float64]:
    rows, cols = values.shape
    transform = np.asarray(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        values.astype(np.float32),
        transform,
        (cols, rows),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(float)


def _continuous_emission_trail(
    shape: tuple[int, int],
    *,
    source: tuple[float, float],
    motion: MotionEstimate,
    delta_t: float,
    max_step_px: float,
) -> NDArray[np.float64]:
    distance = motion.speed * delta_t
    segment_count = max(1, int(np.ceil(distance / max_step_px)))
    ages = np.linspace(
        0.0,
        delta_t,
        segment_count + 1,
        dtype=float,
    )
    weights = np.ones(ages.shape, dtype=float)
    if weights.size > 1:
        weights[[0, -1]] = 0.5
    rows = source[0] + motion.dy_per_t * ages
    cols = source[1] + motion.dx_per_t * ages

    impulse = np.zeros(shape, dtype=float)
    for row, col, weight in zip(rows, cols, weights):
        _bilinear_splat(impulse, row, col, weight)
    return impulse


def _bilinear_splat(
    target: NDArray[np.float64],
    row: float,
    col: float,
    weight: float,
) -> None:
    row0 = int(np.floor(row))
    col0 = int(np.floor(col))
    row_fraction = row - row0
    col_fraction = col - col0
    for row_index, row_weight in (
        (row0, 1.0 - row_fraction),
        (row0 + 1, row_fraction),
    ):
        if not 0 <= row_index < target.shape[0]:
            continue
        for col_index, col_weight in (
            (col0, 1.0 - col_fraction),
            (col0 + 1, col_fraction),
        ):
            if 0 <= col_index < target.shape[1]:
                target[row_index, col_index] += (
                    weight * row_weight * col_weight
                )


def _gaussian_blur(
    values: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    if sigma <= 0.0:
        return values
    return cv2.GaussianBlur(
        values.astype(np.float32),
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    ).astype(float)


def _normalize_probability(
    values: NDArray[np.float64],
    *,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    clipped = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(clipped.sum())
    if total <= epsilon:
        return np.full(
            clipped.shape,
            1.0 / clipped.size,
            dtype=float,
        )
    return clipped / total
