"""Causal emulation of a structure-suppressed gas sensor from recorded video.

The emulator deliberately performs no per-frame normalization.  A fixed
leak-off calibration supplies a per-pixel background and noise scale, after
which recorded samples are converted to a non-negative, uint8-like signal.
Ground truth and scan-policy state are not inputs to this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from plume_focus.experiment import (
    GrayFrame,
    SampleValues,
    TimelineSource,
)


ResidualMode = Literal["two_sided_magnitude", "bright", "dark"]

_NORMAL_MAD_SCALE = 1.482602218505602
_UINT8_MAX = 255.0


@dataclass(frozen=True)
class SensorEmulationProfile:
    """Validated behavior contract for structure-suppressed sensor values.

    ``output_units_per_noise_sigma`` maps a standardized residual into the
    non-negative 0--255 output range. Temporal integration remains an
    explicitly disabled future extension: applying one filter across spatial
    scan samples would conflate sensor response with scan order.
    """

    profile_id: str
    residual_mode: ResidualMode
    provenance: str
    noise_floor_raw_units: float = 1.0
    output_units_per_noise_sigma: float = 32.0
    smoothing_time_constant_seconds: float | None = None
    profile_schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be empty.")
        if self.residual_mode not in {
            "two_sided_magnitude",
            "bright",
            "dark",
        }:
            raise ValueError(
                "residual_mode must be two_sided_magnitude, bright, or dark."
            )
        if not self.provenance.strip():
            raise ValueError("provenance must not be empty.")
        _require_positive_finite(
            "noise_floor_raw_units",
            self.noise_floor_raw_units,
        )
        _require_positive_finite(
            "output_units_per_noise_sigma",
            self.output_units_per_noise_sigma,
        )
        if self.smoothing_time_constant_seconds is not None:
            raise ValueError(
                "smoothing_time_constant_seconds is not supported by "
                "sensor-emulation profile v1; use null."
            )
        if (
            isinstance(self.profile_schema_version, bool)
            or not isinstance(self.profile_schema_version, int)
            or self.profile_schema_version != 1
        ):
            raise ValueError(
                "unsupported sensor-emulation profile schema version."
            )

    def to_payload(self) -> dict[str, object]:
        """Return the complete declared profile for evidence serialization."""

        return {
            **asdict(self),
            "output_contract": "uint8_relative_gas_evidence_v1",
            "output_min": 0.0,
            "output_max": _UINT8_MAX,
            "background_estimator": "per_pixel_median",
            "noise_estimator": "per_pixel_normal_consistent_mad",
            "normal_mad_scale": _NORMAL_MAD_SCALE,
            "normalization_scope": "fixed_leak_off_calibration_only",
        }


@dataclass(frozen=True)
class FixedLeakOffCalibration:
    """Immutable per-pixel background and robust noise calibration."""

    background_raw_units: NDArray[np.float64]
    noise_sigma_raw_units: NDArray[np.float64]
    sample_count: int

    def __post_init__(self) -> None:
        background = np.array(
            self.background_raw_units,
            dtype=np.float64,
            copy=True,
        )
        noise = np.array(
            self.noise_sigma_raw_units,
            dtype=np.float64,
            copy=True,
        )
        if background.ndim != 2:
            raise ValueError("calibration fields must be 2-D arrays.")
        if noise.shape != background.shape:
            raise ValueError(
                "background and noise calibration fields must match."
            )
        if np.any(~np.isfinite(background)):
            raise ValueError("calibration background must be finite.")
        if np.any(~np.isfinite(noise)) or np.any(noise <= 0.0):
            raise ValueError("calibration noise must be finite and positive.")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 2
        ):
            raise ValueError("sample_count must be an integer of at least 2.")

        background.setflags(write=False)
        noise.setflags(write=False)
        object.__setattr__(self, "background_raw_units", background)
        object.__setattr__(self, "noise_sigma_raw_units", noise)

    @property
    def shape(self) -> tuple[int, int]:
        return self.background_raw_units.shape


def fit_fixed_leak_off_calibration(
    leak_off_frames: Sequence[NDArray[np.generic]] | NDArray[np.generic],
    *,
    profile: SensorEmulationProfile,
) -> FixedLeakOffCalibration:
    """Fit fixed per-pixel median and normal-consistent MAD calibration.

    The input frames are copied and never modified.  The supplied frames must
    all represent a declared leak-off calibration interval; this function does
    not select that interval or inspect labels.
    """

    frames = np.array(leak_off_frames, dtype=np.float64, copy=True)
    if frames.ndim != 3:
        raise ValueError(
            "leak_off_frames must have shape (samples, rows, cols)."
        )
    if frames.shape[0] < 2:
        raise ValueError("leak_off_frames must contain at least two samples.")
    if frames.shape[1] <= 0 or frames.shape[2] <= 0:
        raise ValueError("leak_off_frames must have a non-empty frame shape.")
    if np.any(~np.isfinite(frames)):
        raise ValueError("leak_off_frames must contain only finite values.")

    background = np.median(frames, axis=0)
    median_absolute_deviation = np.median(
        np.abs(frames - background[None, :, :]),
        axis=0,
    )
    robust_noise = np.maximum(
        median_absolute_deviation * _NORMAL_MAD_SCALE,
        profile.noise_floor_raw_units,
    )
    return FixedLeakOffCalibration(
        background_raw_units=background,
        noise_sigma_raw_units=robust_noise,
        sample_count=int(frames.shape[0]),
    )


def emulate_sensor_frame(
    frame: NDArray[np.generic],
    *,
    calibration: FixedLeakOffCalibration,
    profile: SensorEmulationProfile,
) -> NDArray[np.float64]:
    """Transform one frame using only fixed, per-pixel calibration values."""

    values = np.asarray(frame, dtype=np.float64)
    if values.shape != calibration.shape:
        raise ValueError("frame shape must match the sensor calibration.")
    if np.any(~np.isfinite(values)):
        raise ValueError("frame must contain only finite values.")
    return _emulate_values(
        values,
        calibration.background_raw_units,
        calibration.noise_sigma_raw_units,
        profile,
    )


@dataclass(frozen=True)
class SensorEmulationTimeline:
    """Causal ``TimelineSource`` wrapper around recorded timestamped frames.

    Scan sampling returns only the requested coordinate/time pairs. Temporal
    integration is intentionally absent because its state belongs to a future
    scanner-response adapter with an explicit scan lifecycle.
    """

    source: TimelineSource
    calibration: FixedLeakOffCalibration
    profile: SensorEmulationProfile

    def __post_init__(self) -> None:
        if self.source.shape != self.calibration.shape:
            raise ValueError(
                "source shape must match the fixed leak-off calibration."
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.source.shape

    @property
    def fps(self) -> float:
        return self.source.fps

    @property
    def end_time_seconds(self) -> float:
        return self.source.end_time_seconds

    def frame_at_seconds(self, time_seconds: float) -> GrayFrame:
        """Return a pointwise frame for evaluation or explicit baselines.

        This full-frame surface remains distinct from mechanically requested
        samples and is available only to explicit baseline or rendering paths.
        """

        raw_frame = self.source.frame_at_seconds(time_seconds)
        emulated = emulate_sensor_frame(
            raw_frame,
            calibration=self.calibration,
            profile=self.profile,
        )
        return np.clip(emulated, 0.0, _UINT8_MAX).astype(np.uint8)

    def samples_at_seconds(
        self,
        times_seconds: NDArray[np.float64],
        rows: NDArray[np.int_],
        cols: NDArray[np.int_],
    ) -> SampleValues:
        """Return causal emulated values for requested sample coordinates."""

        times = np.asarray(times_seconds, dtype=np.float64)
        sample_rows = np.asarray(rows)
        sample_cols = np.asarray(cols)
        if times.shape != sample_rows.shape or times.shape != sample_cols.shape:
            raise ValueError("times, rows, and cols must have matching shapes.")
        # This validates times and coordinates using the wrapped timeline and,
        # importantly, raises rather than clamping requests beyond its end.
        raw_values = self.source.samples_at_seconds(
            times,
            sample_rows,
            sample_cols,
        )
        backgrounds = self.calibration.background_raw_units[
            sample_rows,
            sample_cols,
        ]
        noise_sigmas = self.calibration.noise_sigma_raw_units[
            sample_rows,
            sample_cols,
        ]

        return _emulate_values(
            raw_values,
            backgrounds,
            noise_sigmas,
            self.profile,
        )


def _emulate_values(
    values: NDArray[np.generic],
    backgrounds: NDArray[np.generic],
    noise_sigmas: NDArray[np.generic],
    profile: SensorEmulationProfile,
) -> NDArray[np.float64]:
    residual = (
        np.asarray(values, dtype=np.float64)
        - np.asarray(backgrounds, dtype=np.float64)
    )
    standardized = residual / np.asarray(noise_sigmas, dtype=np.float64)
    if profile.residual_mode == "two_sided_magnitude":
        selected = np.abs(standardized)
    elif profile.residual_mode == "bright":
        selected = np.maximum(standardized, 0.0)
    else:
        selected = np.maximum(-standardized, 0.0)
    return np.clip(
        selected * profile.output_units_per_noise_sigma,
        0.0,
        _UINT8_MAX,
    )


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
