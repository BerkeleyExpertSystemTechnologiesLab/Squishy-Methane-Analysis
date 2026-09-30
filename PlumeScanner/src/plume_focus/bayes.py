"""Bayesian plume belief updates and ROI selection.

This module makes the active-sensing loop explicit:

prior -> likelihood from the new scan -> posterior -> predictive prior
-> next ROI selected from a credible region.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from plume_focus.model_profiles import MeasurementModelConfig
from plume_focus.motion import MotionEstimate
from plume_focus.plume_dynamics import (
    PlumeDynamicsConfig,
    propagate_plume_belief,
    source_oriented_prior,
)
from plume_focus.predictor import BoundingBox


@dataclass(frozen=True)
class BayesianPlumeBelief:
    """One explicit Bayesian update stage for the plume field."""

    prior: NDArray[np.float64]
    likelihood: NDArray[np.float64]
    posterior: NDArray[np.float64]
    predictive: NDArray[np.float64]
    current_roi: BoundingBox
    next_roi: BoundingBox
    credible_mass: float


def normalize_probability(values: NDArray[np.float64], *, epsilon: float = 1e-12) -> NDArray[np.float64]:
    """Normalize non-negative values into a probability map."""

    clipped = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(clipped.sum())
    if total <= epsilon:
        return np.full(clipped.shape, 1.0 / clipped.size, dtype=float)
    return clipped / total


def source_downwind_prior(
    shape: tuple[int, int],
    *,
    source: tuple[float, float],
    wind_motion: MotionEstimate,
    roi: BoundingBox | None = None,
    dynamics_config: PlumeDynamicsConfig | None = None,
) -> NDArray[np.float64]:
    """Compatibility wrapper for the source-oriented initial prior."""

    return source_oriented_prior(
        shape,
        source=source,
        wind_motion=wind_motion,
        roi=roi,
        config=dynamics_config,
    )


def scan_likelihood(
    scan_values: NDArray[np.uint8],
    sampled_mask: NDArray[np.bool_],
    *,
    blur_sigma: float | None = None,
    neutral_likelihood: float | None = None,
    config: MeasurementModelConfig | None = None,
) -> NDArray[np.float64]:
    """Build a likelihood map from a sparse/ROI single-pixel scan."""

    if scan_values.shape != sampled_mask.shape:
        raise ValueError("scan_values and sampled_mask must have matching shapes.")
    settings = config or MeasurementModelConfig(
        foreground_threshold_ratio=0.35,
    )
    effective_blur_sigma = (
        settings.likelihood_blur_sigma_px
        if blur_sigma is None
        else blur_sigma
    )
    effective_neutral_likelihood = (
        settings.neutral_likelihood
        if neutral_likelihood is None
        else neutral_likelihood
    )
    if not np.isfinite(effective_blur_sigma) or effective_blur_sigma < 0.0:
        raise ValueError("blur_sigma must be finite and non-negative.")
    if (
        not np.isfinite(effective_neutral_likelihood)
        or not 0.0 <= effective_neutral_likelihood <= 1.0
    ):
        raise ValueError("neutral_likelihood must be between 0 and 1.")

    observed = np.zeros(scan_values.shape, dtype=float)
    if np.any(sampled_mask):
        sampled_values = scan_values[sampled_mask].astype(float)
        low = float(
            np.percentile(
                sampled_values,
                settings.intensity_lower_percentile,
            )
        )
        high = float(
            np.percentile(
                sampled_values,
                settings.intensity_upper_percentile,
            )
        )
        scaled = np.clip((scan_values.astype(float) - low) / max(high - low, 1e-9), 0.0, 1.0)
        observed[sampled_mask] = scaled[sampled_mask]

    if effective_blur_sigma > 0:
        observed = cv2.GaussianBlur(
            observed.astype(np.float32),
            (0, 0),
            sigmaX=effective_blur_sigma,
            sigmaY=effective_blur_sigma,
        ).astype(float)

    likelihood = np.full(
        scan_values.shape,
        effective_neutral_likelihood,
        dtype=float,
    )
    likelihood[sampled_mask] = (
        settings.observed_likelihood_floor
        + settings.observed_likelihood_scale * observed[sampled_mask]
    )
    return np.clip(likelihood, settings.minimum_likelihood, None)


def posterior_from_likelihood(prior: NDArray[np.float64], likelihood: NDArray[np.float64]) -> NDArray[np.float64]:
    """Bayes update: posterior is proportional to prior times likelihood."""

    if prior.shape != likelihood.shape:
        raise ValueError("prior and likelihood must have matching shapes.")
    return normalize_probability(prior * likelihood)


def predict_next_prior(
    posterior: NDArray[np.float64],
    motion: MotionEstimate,
    *,
    diffusion_sigma: float,
    delta_t: float = 1.0,
    source: tuple[float, float] | None = None,
    dynamics_config: PlumeDynamicsConfig | None = None,
) -> NDArray[np.float64]:
    """Compatibility wrapper for source-aware plume propagation."""

    return propagate_plume_belief(
        posterior,
        motion,
        diffusion_sigma=diffusion_sigma,
        delta_t=delta_t,
        source=source,
        config=dynamics_config,
    )


def credible_region_roi(
    distribution: NDArray[np.float64],
    *,
    mass: float = 0.9,
    margin: int = 6,
    min_size: tuple[int, int] = (12, 16),
    include_point: tuple[float, float] | None = None,
) -> BoundingBox:
    """Return the smallest bounding ROI containing a target probability mass."""

    if not 0.0 < mass <= 1.0:
        raise ValueError("mass must be in (0, 1].")

    probs = normalize_probability(distribution)
    flat = probs.ravel()
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    cutoff_index = int(np.searchsorted(cumulative, mass, side="left"))
    selected = np.zeros(flat.shape, dtype=bool)
    selected[order[: cutoff_index + 1]] = True
    rows, cols = np.where(selected.reshape(probs.shape))

    min_row = int(rows.min()) - margin
    max_row = int(rows.max()) + margin
    min_col = int(cols.min()) - margin
    max_col = int(cols.max()) + margin

    if include_point is not None:
        point_row, point_col = include_point
        min_row = min(min_row, int(np.floor(point_row)) - margin)
        max_row = max(max_row, int(np.ceil(point_row)) + margin)
        min_col = min(min_col, int(np.floor(point_col)) - margin)
        max_col = max(max_col, int(np.ceil(point_col)) + margin)

    min_height, min_width = min_size
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    half_height = max(min_height // 2, (max_row - min_row + 1) // 2)
    half_width = max(min_width // 2, (max_col - min_col + 1) // 2)
    box = BoundingBox(
        min_row=center_row - half_height,
        min_col=center_col - half_width,
        max_row=center_row + half_height,
        max_col=center_col + half_width,
    )
    return box.clipped(probs.shape)


def credible_region_mask(
    distribution: NDArray[np.float64],
    *,
    mass: float = 0.9,
    blur_sigma: float = 2.0,
    min_area_fraction: float = 0.0,
    include_point: tuple[float, float] | None = None,
    include_radius: int = 4,
) -> NDArray[np.bool_]:
    """Return a smooth non-rectangular credible sampling mask."""

    if not 0.0 < mass <= 1.0:
        raise ValueError("mass must be in (0, 1].")
    if not 0.0 <= min_area_fraction <= 1.0:
        raise ValueError("min_area_fraction must be between 0 and 1.")

    probs = normalize_probability(distribution)
    if blur_sigma > 0:
        probs = cv2.GaussianBlur(probs.astype(np.float32), (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma).astype(float)
        probs = normalize_probability(probs)

    flat = probs.ravel()
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    cutoff_index = int(np.searchsorted(cumulative, mass, side="left"))
    minimum_pixels = int(np.ceil(min_area_fraction * flat.size))
    selected = np.zeros(flat.shape, dtype=bool)
    selected[order[: cutoff_index + 1]] = True
    mask = selected.reshape(probs.shape)

    if minimum_pixels > int(np.count_nonzero(mask)):
        # Expand outward from the actual highest-density region. Selecting more
        # entries from a flat argsort would fill the area floor with arbitrary
        # zero-probability pixels scattered across the frame.
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        outside_rows, outside_cols = np.where(~mask)
        needed = minimum_pixels - int(np.count_nonzero(mask))
        nearest = np.argsort(distance[outside_rows, outside_cols], kind="stable")[:needed]
        mask[outside_rows[nearest], outside_cols[nearest]] = True

    if include_point is not None:
        rows, cols = np.indices(mask.shape)
        point_row, point_col = include_point
        source_disk = (rows - point_row) ** 2 + (cols - point_col) ** 2 <= include_radius**2
        mask |= source_disk

    return mask


def bounding_box_from_mask(mask: NDArray[np.bool_]) -> BoundingBox:
    """Return a bounding box around a boolean mask."""

    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("Cannot build a box from an empty mask.")
    return BoundingBox(int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


def bayesian_roi_step(
    prior: NDArray[np.float64],
    scan_values: NDArray[np.uint8],
    sampled_mask: NDArray[np.bool_],
    motion: MotionEstimate,
    *,
    current_roi: BoundingBox,
    credible_mass: float,
    diffusion_sigma: float,
    source: tuple[float, float] | None = None,
    min_roi_size: tuple[int, int] = (12, 16),
    min_roi_area_fraction: float = 0.0,
    smooth_roi: bool = False,
    roi_blur_sigma: float = 2.0,
    prediction_delta_t: float = 1.0,
    measurement_config: MeasurementModelConfig | None = None,
    dynamics_config: PlumeDynamicsConfig | None = None,
) -> BayesianPlumeBelief:
    """Run one prior-likelihood-posterior-predictive-next-ROI cycle."""

    likelihood = scan_likelihood(
        scan_values,
        sampled_mask,
        config=measurement_config,
    )
    posterior = posterior_from_likelihood(prior, likelihood)
    predictive = predict_next_prior(
        posterior,
        motion,
        diffusion_sigma=diffusion_sigma,
        delta_t=prediction_delta_t,
        source=source,
        dynamics_config=dynamics_config,
    )
    if smooth_roi:
        next_mask = credible_region_mask(
            predictive,
            mass=credible_mass,
            blur_sigma=roi_blur_sigma,
            min_area_fraction=min_roi_area_fraction,
            include_point=source,
            include_radius=max(4, int(round(diffusion_sigma))),
        )
        next_roi = bounding_box_from_mask(next_mask)
    else:
        next_roi = credible_region_roi(
            predictive,
            mass=credible_mass,
            margin=max(3, int(round(diffusion_sigma))),
            min_size=min_roi_size,
            include_point=source,
        )
    return BayesianPlumeBelief(
        prior=prior,
        likelihood=likelihood,
        posterior=posterior,
        predictive=predictive,
        current_roi=current_roi,
        next_roi=next_roi,
        credible_mass=credible_mass,
    )
