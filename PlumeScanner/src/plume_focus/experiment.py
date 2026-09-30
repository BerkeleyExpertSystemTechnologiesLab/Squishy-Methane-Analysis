"""Causal experiment primitives for active plume sensing.

This module deliberately separates data that the planner may observe from
ground-truth frames used only by the offline evaluator.  All public timestamps
are seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from plume_focus.motion import MotionEstimate


GrayFrame = NDArray[np.uint8]
SampleValues = NDArray[np.float64]
FrameAtSeconds = Callable[[float], GrayFrame]
SamplesAtSeconds = Callable[
    [NDArray[np.float64], NDArray[np.int_], NDArray[np.int_]],
    SampleValues,
]


class TimelineExhausted(ValueError):
    """Raised when a finite source cannot provide data at a requested time."""


class TimelineSource(Protocol):
    """Time-addressable source used by a causal scan experiment."""

    shape: tuple[int, int]
    fps: float
    end_time_seconds: float | None

    def frame_at_seconds(self, time_seconds: float) -> GrayFrame:
        """Return a complete frame for offline evaluation or explicit baselines."""

    def samples_at_seconds(
        self,
        times_seconds: NDArray[np.float64],
        rows: NDArray[np.int_],
        cols: NDArray[np.int_],
    ) -> SampleValues:
        """Return sensor samples at the supplied physical timestamps."""


@dataclass(frozen=True)
class FunctionalTimeline:
    """Timeline backed by continuous synthetic frame and sample functions."""

    shape: tuple[int, int]
    fps: float
    frame_at_frame_time: FrameAtSeconds
    samples_at_frame_times: SamplesAtSeconds
    end_time_seconds: float | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("shape must be a positive (rows, cols) tuple.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")

    def frame_at_seconds(self, time_seconds: float) -> GrayFrame:
        self._validate_times(np.asarray([time_seconds], dtype=float))
        return np.asarray(self.frame_at_frame_time(time_seconds * self.fps), dtype=np.uint8)

    def samples_at_seconds(
        self,
        times_seconds: NDArray[np.float64],
        rows: NDArray[np.int_],
        cols: NDArray[np.int_],
    ) -> SampleValues:
        times = np.asarray(times_seconds, dtype=float)
        self._validate_sample_inputs(times, rows, cols)
        return np.asarray(self.samples_at_frame_times(times * self.fps, rows, cols), dtype=float)

    def _validate_times(self, times: NDArray[np.float64]) -> None:
        if np.any(~np.isfinite(times)) or np.any(times < 0.0):
            raise TimelineExhausted("Timeline timestamps must be finite and non-negative.")
        if self.end_time_seconds is not None and np.any(times > self.end_time_seconds + 1e-12):
            raise TimelineExhausted(
                f"Requested time exceeds source end {self.end_time_seconds:.6f} seconds."
            )

    def _validate_sample_inputs(
        self,
        times: NDArray[np.float64],
        rows: NDArray[np.int_],
        cols: NDArray[np.int_],
    ) -> None:
        if times.shape != rows.shape or times.shape != cols.shape:
            raise ValueError("times, rows, and cols must have matching shapes.")
        self._validate_times(times)
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise IndexError("Sample row lies outside the timeline shape.")
        if np.any(cols < 0) or np.any(cols >= self.shape[1]):
            raise IndexError("Sample column lies outside the timeline shape.")


@dataclass(frozen=True)
class ArrayVideoTimeline:
    """Finite nearest-frame timeline for decoded real video.

    Requests beyond the decoded clip raise :class:`TimelineExhausted`; the final
    frame is never silently reused to simulate unavailable future evidence.
    """

    frames: tuple[GrayFrame, ...]
    fps: float

    def __post_init__(self) -> None:
        if len(self.frames) < 1:
            raise ValueError("frames must contain at least one frame.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        shape = self.frames[0].shape
        if len(shape) != 2:
            raise ValueError("Timeline frames must be 2-D grayscale arrays.")
        if any(frame.shape != shape for frame in self.frames):
            raise ValueError("All timeline frames must have matching shapes.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.frames[0].shape

    @property
    def end_time_seconds(self) -> float:
        return (len(self.frames) - 1) / self.fps

    def frame_at_seconds(self, time_seconds: float) -> GrayFrame:
        index = self._indices_for_times(np.asarray([time_seconds], dtype=float))[0]
        return self.frames[int(index)]

    def samples_at_seconds(
        self,
        times_seconds: NDArray[np.float64],
        rows: NDArray[np.int_],
        cols: NDArray[np.int_],
    ) -> SampleValues:
        times = np.asarray(times_seconds, dtype=float)
        if times.shape != rows.shape or times.shape != cols.shape:
            raise ValueError("times, rows, and cols must have matching shapes.")
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise IndexError("Sample row lies outside the video shape.")
        if np.any(cols < 0) or np.any(cols >= self.shape[1]):
            raise IndexError("Sample column lies outside the video shape.")

        indices = self._indices_for_times(times)
        values = np.zeros(times.shape, dtype=float)
        for index in np.unique(indices):
            selected = indices == index
            values[selected] = self.frames[int(index)][rows[selected], cols[selected]]
        return values

    def _indices_for_times(self, times: NDArray[np.float64]) -> NDArray[np.int_]:
        if np.any(~np.isfinite(times)) or np.any(times < 0.0):
            raise TimelineExhausted("Video timestamps must be finite and non-negative.")
        if np.any(times > self.end_time_seconds + 1e-12):
            requested = float(np.max(times))
            raise TimelineExhausted(
                f"Requested {requested:.6f} seconds, beyond video end "
                f"{self.end_time_seconds:.6f} seconds."
            )
        return np.rint(times * self.fps).astype(int)


@dataclass(frozen=True)
class TimestampedArrayTimeline:
    """Finite recorded timeline that never selects a future frame.

    The timestamp at index ``i`` is the physical capture time of
    ``frames[i]``. Requests select the latest capture whose timestamp is not
    later than the request. This differs intentionally from the legacy
    nearest-frame ``ArrayVideoTimeline`` compatibility behavior.
    """

    frames: tuple[GrayFrame, ...]
    timestamps_seconds: tuple[float, ...]
    fps: float

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("frames must contain at least one frame.")
        if len(self.frames) != len(self.timestamps_seconds):
            raise ValueError(
                "frames and timestamps_seconds must have matching lengths."
            )
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("fps must be finite and positive.")
        timestamps = np.asarray(self.timestamps_seconds, dtype=float)
        if np.any(~np.isfinite(timestamps)):
            raise ValueError("timestamps_seconds must be finite.")
        if timestamps[0] != 0.0:
            raise ValueError("the first recorded timestamp must be 0 seconds.")
        if np.any(np.diff(timestamps) <= 0.0):
            raise ValueError(
                "timestamps_seconds must be strictly increasing."
            )
        shape = self.frames[0].shape
        if len(shape) != 2:
            raise ValueError("Timeline frames must be 2-D grayscale arrays.")
        if any(frame.shape != shape for frame in self.frames):
            raise ValueError("All timeline frames must have matching shapes.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.frames[0].shape

    @property
    def end_time_seconds(self) -> float:
        return self.timestamps_seconds[-1]

    def frame_at_seconds(self, time_seconds: float) -> GrayFrame:
        index = self._indices_for_times(
            np.asarray([time_seconds], dtype=float)
        )[0]
        return self.frames[int(index)]

    def samples_at_seconds(
        self,
        times_seconds: NDArray[np.float64],
        rows: NDArray[np.int_],
        cols: NDArray[np.int_],
    ) -> SampleValues:
        times = np.asarray(times_seconds, dtype=float)
        if times.shape != rows.shape or times.shape != cols.shape:
            raise ValueError("times, rows, and cols must have matching shapes.")
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise IndexError("Sample row lies outside the recorded shape.")
        if np.any(cols < 0) or np.any(cols >= self.shape[1]):
            raise IndexError("Sample column lies outside the recorded shape.")

        indices = self._indices_for_times(times)
        values = np.zeros(times.shape, dtype=float)
        for index in np.unique(indices):
            selected = indices == index
            values[selected] = self.frames[int(index)][
                rows[selected],
                cols[selected],
            ]
        return values

    def _indices_for_times(
        self,
        times: NDArray[np.float64],
    ) -> NDArray[np.int_]:
        if np.any(~np.isfinite(times)) or np.any(times < 0.0):
            raise TimelineExhausted(
                "Recorded timestamps must be finite and non-negative."
            )
        if np.any(times > self.end_time_seconds + 1e-12):
            requested = float(np.max(times))
            raise TimelineExhausted(
                f"Requested {requested:.6f} seconds, beyond recorded end "
                f"{self.end_time_seconds:.6f} seconds."
            )
        timestamps = np.asarray(self.timestamps_seconds, dtype=float)
        return (
            np.searchsorted(timestamps, times, side="right") - 1
        ).astype(int)


@dataclass(frozen=True)
class ScanEvent:
    """One completed observation and the future time targeted by its plan."""

    scan_index: int
    started_at_seconds: float
    completed_at_seconds: float
    decision_at_seconds: float
    target_at_seconds: float

    def validate(self, previous: "ScanEvent | None" = None) -> None:
        if self.scan_index < 0:
            raise ValueError("scan_index must be non-negative.")
        if self.completed_at_seconds < self.started_at_seconds:
            raise ValueError("A scan cannot complete before it starts.")
        if self.decision_at_seconds < self.completed_at_seconds:
            raise ValueError("A decision cannot use a scan before it completes.")
        if self.target_at_seconds < self.decision_at_seconds:
            raise ValueError("A prediction target cannot precede its decision.")
        if previous is not None and self.started_at_seconds < previous.completed_at_seconds - 1e-12:
            raise ValueError("Sequential scans must not overlap.")


@dataclass(frozen=True)
class MaskCoverage:
    """Time-aligned coverage of a truth distribution by a planned scan mask."""

    support_recall: float
    probability_mass_recall: float
    intersection_over_union: float
    mask_area_fraction: float


def evaluate_scan_mask(
    scan_mask: NDArray[np.bool_],
    truth_distribution: NDArray[np.float64],
    truth_support: NDArray[np.bool_],
) -> MaskCoverage:
    """Evaluate the actual planned scan surface against offline truth."""

    if scan_mask.shape != truth_distribution.shape or scan_mask.shape != truth_support.shape:
        raise ValueError("scan mask, truth distribution, and truth support must match.")

    mask = np.asarray(scan_mask, dtype=bool)
    support = np.asarray(truth_support, dtype=bool)
    truth = np.maximum(np.asarray(truth_distribution, dtype=float), 0.0)
    truth_total = float(truth.sum())
    support_pixels = int(np.count_nonzero(support))
    intersection = int(np.count_nonzero(mask & support))
    union = int(np.count_nonzero(mask | support))

    return MaskCoverage(
        support_recall=intersection / support_pixels if support_pixels else 0.0,
        probability_mass_recall=float(truth[mask].sum() / truth_total) if truth_total > 0.0 else 0.0,
        intersection_over_union=intersection / union if union else 0.0,
        mask_area_fraction=float(np.count_nonzero(mask) / mask.size),
    )


def distribution_centroid_motion(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    *,
    elapsed_seconds: float,
) -> MotionEstimate:
    """Estimate pixels/second motion from two completed scan posteriors."""

    if previous.shape != current.shape:
        raise ValueError("previous and current distributions must match.")
    if elapsed_seconds <= 0.0:
        raise ValueError("elapsed_seconds must be positive.")

    previous_center = _distribution_centroid(previous)
    current_center = _distribution_centroid(current)
    return MotionEstimate(
        dy_per_t=(current_center[0] - previous_center[0]) / elapsed_seconds,
        dx_per_t=(current_center[1] - previous_center[1]) / elapsed_seconds,
        method="scan_posterior_centroid_px_per_second",
        confidence=0.7,
    )


def _distribution_centroid(distribution: NDArray[np.float64]) -> tuple[float, float]:
    values = np.maximum(np.asarray(distribution, dtype=float), 0.0)
    total = float(values.sum())
    rows, cols = values.shape
    if total <= 0.0:
        return (rows - 1) / 2.0, (cols - 1) / 2.0
    row_grid, col_grid = np.indices(values.shape, dtype=float)
    return float((values * row_grid).sum() / total), float((values * col_grid).sum() / total)
