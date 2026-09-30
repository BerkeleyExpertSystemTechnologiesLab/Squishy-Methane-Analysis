"""Observation, motion, and distribution metrics for experiment evaluation."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from plume_focus.motion import MotionEstimate


def foreground_mask(
    frame_gray: NDArray[np.uint8],
    *,
    threshold_ratio: float = 0.35,
) -> NDArray[np.bool_]:
    """Segment a bright moving object or plume from a grayscale frame."""

    lo = int(frame_gray.min())
    hi = int(frame_gray.max())
    threshold = lo + threshold_ratio * (hi - lo)
    mask = frame_gray >= threshold
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel,
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel,
    )
    return cleaned.astype(bool)


def estimate_opencv_motion(
    previous_gray: NDArray[np.uint8],
    current_gray: NDArray[np.uint8],
    current_mask: NDArray[np.bool_] | None = None,
) -> MotionEstimate:
    """Estimate object motion with OpenCV Farneback optical flow."""

    prev = previous_gray.astype(np.uint8)
    curr = current_gray.astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        prev,
        curr,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=4,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    mask = current_mask if current_mask is not None else foreground_mask(curr)
    if not np.any(mask):
        mask = np.ones(curr.shape, dtype=bool)

    dx = float(np.median(flow[..., 0][mask]))
    dy = float(np.median(flow[..., 1][mask]))
    confidence = float(
        np.clip(
            np.count_nonzero(mask) / mask.size * 12.0,
            0.0,
            1.0,
        )
    )
    return MotionEstimate(
        dy_per_t=dy,
        dx_per_t=dx,
        method="opencv_farneback",
        confidence=confidence,
    )


def kernel_plume_attention_field(
    frame_gray: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    motion: MotionEstimate,
    *,
    uncertainty_radius: float,
    source: tuple[float, float] | None = None,
    source_anchor_strength: float = 0.35,
) -> NDArray[np.float64]:
    """Predict attention by advecting positive pixels as Gaussian kernels."""

    if frame_gray.ndim != 2:
        raise ValueError("frame_gray must be a 2-D grayscale image.")
    if mask.shape != frame_gray.shape:
        raise ValueError("mask must match frame shape.")

    rows, cols = np.where(mask)
    impulse = np.zeros(frame_gray.shape, dtype=np.float32)
    if rows.size == 0:
        return impulse

    weights = (
        frame_gray[rows, cols].astype(np.float32)
        - float(frame_gray[mask].min())
    )
    weights = np.maximum(weights, 1.0)
    shifted_rows = np.clip(
        np.rint(rows + motion.dy_per_t).astype(int),
        0,
        frame_gray.shape[0] - 1,
    )
    shifted_cols = np.clip(
        np.rint(cols + motion.dx_per_t).astype(int),
        0,
        frame_gray.shape[1] - 1,
    )
    np.add.at(impulse, (shifted_rows, shifted_cols), weights)

    if source is not None and source_anchor_strength > 0.0:
        source_row, source_col = source
        source_near = np.hypot(
            rows - source_row,
            cols - source_col,
        ) <= max(6.0, uncertainty_radius * 0.12)
        if np.any(source_near):
            anchored_rows = np.clip(
                rows[source_near],
                0,
                frame_gray.shape[0] - 1,
            )
            anchored_cols = np.clip(
                cols[source_near],
                0,
                frame_gray.shape[1] - 1,
            )
            anchored_weights = weights[source_near] * source_anchor_strength
            np.add.at(
                impulse,
                (anchored_rows, anchored_cols),
                anchored_weights,
            )

    sigma = max(1.0, min(18.0, uncertainty_radius / 6.0))
    blurred = cv2.GaussianBlur(
        impulse,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    return blurred / max(float(blurred.max()), 1e-12)


def plume_truth_distribution(
    frame_gray: NDArray[np.uint8],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Build a normalized true plume distribution from an evaluation frame."""

    weights = np.where(
        mask,
        frame_gray.astype(float) - float(frame_gray.min()),
        0.0,
    )
    weights = np.maximum(weights, 0.0)
    return normalize_distribution(weights)


def normalize_distribution(
    values: NDArray[np.float64],
    *,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Normalize non-negative values into a probability distribution."""

    clipped = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(clipped.sum())
    if total <= epsilon:
        return np.full(
            clipped.shape,
            1.0 / clipped.size,
            dtype=float,
        )
    return clipped / total


def kl_divergence(
    true_distribution: NDArray[np.float64],
    predicted_distribution: NDArray[np.float64],
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return KL(true || predicted) in nats."""

    true = normalize_distribution(true_distribution)
    predicted = normalize_distribution(predicted_distribution)
    true = np.clip(true, epsilon, 1.0)
    predicted = np.clip(predicted, epsilon, 1.0)
    true /= float(true.sum())
    predicted /= float(predicted.sum())
    return float(np.sum(true * np.log(true / predicted)))


def js_divergence(
    distribution_a: NDArray[np.float64],
    distribution_b: NDArray[np.float64],
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return Jensen-Shannon divergence in nats."""

    a = normalize_distribution(distribution_a)
    b = normalize_distribution(distribution_b)
    midpoint = 0.5 * (a + b)
    return 0.5 * kl_divergence(
        a,
        midpoint,
        epsilon=epsilon,
    ) + 0.5 * kl_divergence(
        b,
        midpoint,
        epsilon=epsilon,
    )
