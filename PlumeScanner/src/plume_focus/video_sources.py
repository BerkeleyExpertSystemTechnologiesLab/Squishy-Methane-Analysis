"""Synthetic and file-backed frame sources used by experiment adapters."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from plume_focus.experiment_contracts import DEFAULT_SCAN_RATE_HZ
from plume_focus.predictor import BoundingBox


@dataclass(frozen=True)
class SyntheticVideoConfig:
    """Configuration for the generated moving-object proof video."""

    shape: tuple[int, int] = (96, 128)
    frames: int = 48
    fps: float = 12.0
    start_center: tuple[float, float] = (32.0, 28.0)
    velocity: tuple[float, float] = (1.15, 1.85)
    object_sigma: float = 7.0
    scan_rate_hz: float = DEFAULT_SCAN_RATE_HZ


@dataclass(frozen=True)
class SyntheticPlumeVideoConfig:
    """Configuration for a diffuse, deforming plume-style proof video."""

    shape: tuple[int, int] = (112, 160)
    frames: int = 60
    fps: float = 12.0
    start_center: tuple[float, float] = (66.0, 34.0)
    advection_velocity: tuple[float, float] = (-0.22, 1.45)
    plume_lobes: int = 9
    lobe_spacing: float = 6.0
    base_sigma: float = 4.5
    diffusion_per_lobe: float = 0.85
    scan_rate_hz: float = DEFAULT_SCAN_RATE_HZ


@dataclass(frozen=True)
class SyntheticMethaneLeakConfig:
    """Source-fixed methane leak with a horizontally advected plume stream."""

    shape: tuple[int, int] = (220, 320)
    frames: int = 72
    fps: float = 12.0
    source: tuple[float, float] = (120.0, 48.0)
    wind_velocity: tuple[float, float] = (-0.02, 1.02)
    emission_interval: int = 3
    memory_frames: int = 66
    base_sigma: float = 1.8
    diffusion_per_frame: float = 0.075
    lateral_phase: float = 0.0
    amplitude_phase: float = 0.0
    shimmer_phase: float = 0.0
    source_exclusion_px: int = 10
    roi_margin_px: int = 10
    scan_rate_hz: float = 20.0
    sample_stride_px: int = 8
    vertical_pass_seconds: float = 1.0
    horizontal_step_seconds: float = 0.05
    column_stride_px: int = 1
    column_mass_fraction: float = 0.8
    row_mass_fraction: float = 0.85
    min_sweep_columns: int = 30
    min_sweep_rows: int = 56
    max_sweep_columns: int = 120


def generate_synthetic_object_frame(
    config: SyntheticVideoConfig,
    time: float,
) -> NDArray[np.uint8]:
    """Generate one grayscale frame containing a smooth moving object."""

    _, cols = config.shape
    center_y = config.start_center[0] + config.velocity[0] * time
    center_x = config.start_center[1] + config.velocity[1] * time
    row_grid, col_grid = np.indices(config.shape, dtype=float)
    distance_term = (
        (row_grid - center_y) ** 2 + (col_grid - center_x) ** 2
    ) / (2.0 * config.object_sigma**2)
    object_field = np.exp(-distance_term)

    gradient = np.linspace(10, 28, cols, dtype=float)[None, :]
    frame = gradient + 220.0 * object_field
    return np.clip(frame, 0, 255).astype(np.uint8)


def generate_synthetic_object_values(
    config: SyntheticVideoConfig,
    frame_times: NDArray[np.float64],
    rows: NDArray[np.int_],
    cols: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Sample the synthetic object at arbitrary per-pixel frame times."""

    center_y = config.start_center[0] + config.velocity[0] * frame_times
    center_x = config.start_center[1] + config.velocity[1] * frame_times
    distance_term = (
        (rows - center_y) ** 2 + (cols - center_x) ** 2
    ) / (2.0 * config.object_sigma**2)
    gradient = 10.0 + (28.0 - 10.0) * cols / max(1, config.shape[1] - 1)
    return np.clip(
        gradient + 220.0 * np.exp(-distance_term),
        0.0,
        255.0,
    )


def generate_synthetic_video_frames(
    config: SyntheticVideoConfig,
) -> list[NDArray[np.uint8]]:
    """Generate a list of synthetic moving-object frames."""

    return [
        generate_synthetic_object_frame(config, float(index))
        for index in range(config.frames)
    ]


def generate_synthetic_plume_frame(
    config: SyntheticPlumeVideoConfig,
    time: float,
) -> NDArray[np.uint8]:
    """Generate one grayscale frame containing an advected plume cloud."""

    _, cols = config.shape
    row_grid, col_grid = np.indices(config.shape, dtype=float)
    field = np.zeros(config.shape, dtype=float)

    velocity = np.array(config.advection_velocity, dtype=float)
    speed = float(np.linalg.norm(velocity))
    direction = velocity / max(speed, 1e-9)
    normal = np.array([-direction[1], direction[0]])
    head = np.array(config.start_center, dtype=float) + velocity * time

    for lobe in range(config.plume_lobes):
        downstream_offset = -direction * config.lobe_spacing * lobe
        lateral_wobble = normal * (
            3.2 * np.sin(0.42 * time + 0.83 * lobe)
        )
        center_y, center_x = head + downstream_offset + lateral_wobble
        sigma_y = config.base_sigma + config.diffusion_per_lobe * lobe
        sigma_x = (
            config.base_sigma * 1.35
            + config.diffusion_per_lobe * lobe * 1.25
        )
        amplitude = (
            220.0
            * np.exp(-lobe / 5.5)
            * (0.9 + 0.1 * np.sin(0.35 * time + lobe))
        )

        row_term = ((row_grid - center_y) ** 2) / (2.0 * sigma_y**2)
        col_term = ((col_grid - center_x) ** 2) / (2.0 * sigma_x**2)
        field += amplitude * np.exp(-(row_term + col_term))

    texture = (
        1.0
        + 0.08 * np.sin(0.19 * col_grid + 0.41 * time)
        + 0.06 * np.cos(0.23 * row_grid - 0.17 * time)
    )
    background = 12.0 + np.linspace(0.0, 14.0, cols, dtype=float)[None, :]
    frame = background + field * texture
    return np.clip(frame, 0, 255).astype(np.uint8)


def generate_synthetic_plume_values(
    config: SyntheticPlumeVideoConfig,
    frame_times: NDArray[np.float64],
    rows: NDArray[np.int_],
    cols: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Sample the synthetic plume at arbitrary per-pixel frame times."""

    velocity = np.array(config.advection_velocity, dtype=float)
    speed = float(np.linalg.norm(velocity))
    direction = velocity / max(speed, 1e-9)
    normal = np.array([-direction[1], direction[0]])
    head_y = config.start_center[0] + velocity[0] * frame_times
    head_x = config.start_center[1] + velocity[1] * frame_times
    field = np.zeros(frame_times.shape, dtype=float)

    for lobe in range(config.plume_lobes):
        downstream_offset = -direction * config.lobe_spacing * lobe
        wobble = normal[:, None] * (
            3.2 * np.sin(0.42 * frame_times + 0.83 * lobe)
        )
        center_y = head_y + downstream_offset[0] + wobble[0]
        center_x = head_x + downstream_offset[1] + wobble[1]
        sigma_y = config.base_sigma + config.diffusion_per_lobe * lobe
        sigma_x = (
            config.base_sigma * 1.35
            + config.diffusion_per_lobe * lobe * 1.25
        )
        amplitude = (
            220.0
            * np.exp(-lobe / 5.5)
            * (0.9 + 0.1 * np.sin(0.35 * frame_times + lobe))
        )

        row_term = ((rows - center_y) ** 2) / (2.0 * sigma_y**2)
        col_term = ((cols - center_x) ** 2) / (2.0 * sigma_x**2)
        field += amplitude * np.exp(-(row_term + col_term))

    texture = (
        1.0
        + 0.08 * np.sin(0.19 * cols + 0.41 * frame_times)
        + 0.06 * np.cos(0.23 * rows - 0.17 * frame_times)
    )
    background = 12.0 + 14.0 * cols / max(1, config.shape[1] - 1)
    return np.clip(background + field * texture, 0.0, 255.0)


def generate_synthetic_plume_frames(
    config: SyntheticPlumeVideoConfig,
) -> list[NDArray[np.uint8]]:
    """Generate a list of synthetic plume-like frames."""

    return [
        generate_synthetic_plume_frame(config, float(index))
        for index in range(config.frames)
    ]


def generate_synthetic_methane_frame(
    config: SyntheticMethaneLeakConfig,
    time: float,
) -> NDArray[np.uint8]:
    """Generate a source-fixed methane leak plume under horizontal wind."""

    return _cached_synthetic_methane_frame(config, float(time)).copy()


@lru_cache(maxsize=512)
def _cached_synthetic_methane_frame(
    config: SyntheticMethaneLeakConfig,
    time: float,
) -> NDArray[np.uint8]:
    """Cache deterministic offline truth frames without sharing mutable arrays."""

    row_grid, col_grid = np.indices(config.shape, dtype=float)
    values = generate_synthetic_methane_values(
        config,
        np.full(config.shape, time),
        row_grid.astype(int),
        col_grid.astype(int),
    )
    return values.astype(np.uint8)


def generate_synthetic_methane_values(
    config: SyntheticMethaneLeakConfig,
    frame_times: NDArray[np.float64],
    rows: NDArray[np.int_],
    cols: NDArray[np.int_],
    *,
    clean_signal: bool = False,
) -> NDArray[np.float64]:
    """Sample the methane leak plume at arbitrary per-pixel frame times."""

    wind = np.array(config.wind_velocity, dtype=float)
    speed = float(np.linalg.norm(wind))
    direction = wind / max(speed, 1e-9)
    normal = np.array([-direction[1], direction[0]])
    field = np.zeros(frame_times.shape, dtype=float)

    emission_origin = -config.memory_frames
    earliest_relevant_birth = (
        float(np.min(frame_times))
        - config.memory_frames
    )
    first_birth_index = max(
        0,
        int(
            np.ceil(
                (earliest_relevant_birth - emission_origin)
                / config.emission_interval
            )
        ),
    )
    first_birth = (
        emission_origin
        + first_birth_index * config.emission_interval
    )
    last_birth = int(np.ceil(float(np.max(frame_times))))
    for birth in range(
        first_birth,
        last_birth + 1,
        config.emission_interval,
    ):
        age = frame_times - birth
        active = (age >= 0.0) & (age <= config.memory_frames)
        if not np.any(active):
            continue

        active_age = age[active]
        active_times = frame_times[active]
        lateral = (
            2.8
            * np.sin(
                0.31 * active_age
                + 0.47 * birth
                + config.lateral_phase
            )
            + 1.2
            * np.sin(
                0.11 * active_times
                + 0.5 * config.lateral_phase
            )
        )
        center_y = (
            config.source[0]
            + wind[0] * active_age
            + normal[0] * lateral
        )
        center_x = (
            config.source[1]
            + wind[1] * active_age
            + normal[1] * lateral
        )
        sigma_y = config.base_sigma + config.diffusion_per_frame * active_age
        sigma_x = (
            config.base_sigma * 1.35
            + config.diffusion_per_frame * active_age * 1.65
        )
        amplitude = (
            210.0
            * np.exp(-age / 52.0)
            * (
                0.9
                + 0.1
                * np.sin(
                    0.37 * birth
                    + config.amplitude_phase
                )
            )
        )

        row_term = (
            (rows[active] - center_y) ** 2
        ) / (2.0 * sigma_y**2)
        col_term = (
            (cols[active] - center_x) ** 2
        ) / (2.0 * sigma_x**2)
        field[active] += amplitude[active] * np.exp(-(row_term + col_term))

    background = 14.0 + 9.0 * cols / max(1, config.shape[1] - 1)
    shimmer = (
        1.0
        + 0.06
        * np.sin(
            0.17 * cols
            + 0.19 * frame_times
            + config.shimmer_phase
        )
        + 0.04 * np.cos(0.13 * rows + 0.5 * config.shimmer_phase)
    )
    if clean_signal:
        # A declared synthetic scalar sensor, with no video background, clipping,
        # or scene-wide calibration. Legacy rendered-source behavior is unchanged.
        return field * shimmer / 255.0
    return np.clip(background + field * shimmer, 0.0, 255.0)


def generate_synthetic_methane_frames(
    config: SyntheticMethaneLeakConfig,
) -> list[NDArray[np.uint8]]:
    """Generate a source-fixed methane leak video."""

    return [
        generate_synthetic_methane_frame(config, float(index))
        for index in range(config.frames)
    ]


def load_video_frames(
    path: Path,
    *,
    max_frames: int | None = None,
) -> tuple[list[NDArray[np.uint8]], float]:
    """Load grayscale frames from a video file with OpenCV."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[NDArray[np.uint8]] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if max_frames is not None and len(frames) >= max_frames:
            break
    capture.release()

    if len(frames) < 3:
        raise ValueError("Need at least 3 frames to estimate and evaluate motion.")
    return frames, float(fps)


def load_video_frame_window(
    path: Path,
    *,
    start_frame_index: int,
    end_frame_index_exclusive: int,
) -> tuple[list[NDArray[np.uint8]], float]:
    """Decode an exact half-open grayscale frame window sequentially.

    Sequential decoding deliberately avoids codec-dependent frame seeking.
    The returned first frame is the source frame at ``start_frame_index`` and
    the returned final frame is the source frame immediately before
    ``end_frame_index_exclusive``.
    """

    if (
        isinstance(start_frame_index, bool)
        or not isinstance(start_frame_index, int)
        or start_frame_index < 0
    ):
        raise ValueError("start_frame_index must be a non-negative integer.")
    if (
        isinstance(end_frame_index_exclusive, bool)
        or not isinstance(end_frame_index_exclusive, int)
        or end_frame_index_exclusive <= start_frame_index
    ):
        raise ValueError(
            "end_frame_index_exclusive must be an integer greater than "
            "start_frame_index."
        )

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[NDArray[np.uint8]] = []
    decoded_frame_count = 0
    try:
        while decoded_frame_count < end_frame_index_exclusive:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded_frame_count >= start_frame_index:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            decoded_frame_count += 1
    finally:
        capture.release()

    expected_frame_count = (
        end_frame_index_exclusive - start_frame_index
    )
    if len(frames) != expected_frame_count:
        raise ValueError(
            "Requested frame window "
            f"[{start_frame_index}, {end_frame_index_exclusive}) exceeds "
            f"the decodable video extent ({decoded_frame_count} frames)."
        )
    if len(frames) < 3:
        raise ValueError(
            "Need at least 3 frames to estimate and evaluate motion."
        )
    return frames, float(fps)


def methane_scan_roi(config: SyntheticMethaneLeakConfig) -> BoundingBox:
    """Return a practical downwind corridor to scan for a fixed-source leak."""

    rows, cols = config.shape
    source_row = int(round(config.source[0]))
    source_col = int(round(config.source[1]))
    row_half_width = 24
    return BoundingBox(
        min_row=max(0, source_row - row_half_width * 2),
        min_col=max(0, source_col - config.roi_margin_px),
        max_row=min(rows - 1, source_row + row_half_width * 2),
        max_col=min(cols - 1, source_col + 190),
    )


def default_methane_roi(shape: tuple[int, int]) -> BoundingBox:
    """Fallback downwind ROI for real footage when the source is not annotated."""

    rows, cols = shape
    return BoundingBox(
        min_row=max(0, rows // 2 - rows // 4),
        min_col=max(0, cols // 8),
        max_row=min(rows - 1, rows // 2 + rows // 4),
        max_col=min(cols - 1, cols - cols // 10),
    )
