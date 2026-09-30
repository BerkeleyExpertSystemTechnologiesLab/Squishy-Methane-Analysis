"""Causal wind and offline-truth providers for recorded sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plume_focus.experiment import TimestampedArrayTimeline
from plume_focus.motion import MotionEstimate
from plume_focus.recorded_contracts import (
    EvaluationTruth,
    MotionLookup,
    TruthKind,
)


@dataclass(frozen=True)
class MissingMotionProvider:
    """Explicitly absent wind/advection input."""

    method: str = "recorded_wind_missing"

    def lookup(self, decision_time_seconds: float) -> MotionLookup:
        _validate_lookup_time(decision_time_seconds)
        return MotionLookup("missing", None, None)


@dataclass(frozen=True)
class ConstantMotionProvider:
    """Sequence-wide image-plane wind/advection input."""

    dy_px_per_second: float
    dx_px_per_second: float
    method: str = "recorded_constant_wind_px_per_second"

    def lookup(self, decision_time_seconds: float) -> MotionLookup:
        _validate_lookup_time(decision_time_seconds)
        return MotionLookup(
            "available",
            MotionEstimate(
                self.dy_px_per_second,
                self.dx_px_per_second,
                self.method,
                confidence=0.0,
            ),
            0.0,
        )


@dataclass(frozen=True)
class TimeSeriesMotionProvider:
    """Latest-not-after wind series with an explicit staleness limit."""

    timestamps_seconds: tuple[float, ...]
    dy_px_per_second: tuple[float, ...]
    dx_px_per_second: tuple[float, ...]
    max_age_seconds: float
    method: str = "recorded_time_series_wind_px_per_second"

    def __post_init__(self) -> None:
        lengths = {
            len(self.timestamps_seconds),
            len(self.dy_px_per_second),
            len(self.dx_px_per_second),
        }
        if lengths == {0} or len(lengths) != 1:
            raise ValueError(
                "wind timestamps and components must have equal non-zero "
                "lengths."
            )
        timestamps = np.asarray(self.timestamps_seconds, dtype=float)
        dy = np.asarray(self.dy_px_per_second, dtype=float)
        dx = np.asarray(self.dx_px_per_second, dtype=float)
        if (
            np.any(~np.isfinite(timestamps))
            or np.any(~np.isfinite(dy))
            or np.any(~np.isfinite(dx))
        ):
            raise ValueError("wind series must contain only finite values.")
        if np.any(timestamps < 0.0) or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError(
                "wind timestamps must be non-negative and strictly "
                "increasing."
            )
        if (
            not np.isfinite(self.max_age_seconds)
            or self.max_age_seconds < 0.0
        ):
            raise ValueError(
                "wind max_age_seconds must be finite and non-negative."
            )

    def lookup(self, decision_time_seconds: float) -> MotionLookup:
        _validate_lookup_time(decision_time_seconds)
        timestamps = np.asarray(self.timestamps_seconds, dtype=float)
        index = int(
            np.searchsorted(
                timestamps,
                decision_time_seconds,
                side="right",
            )
            - 1
        )
        if index < 0:
            return MotionLookup("missing", None, None)
        age = decision_time_seconds - float(timestamps[index])
        if age > self.max_age_seconds:
            return MotionLookup("stale", None, age)
        return MotionLookup(
            "available",
            MotionEstimate(
                self.dy_px_per_second[index],
                self.dx_px_per_second[index],
                self.method,
                confidence=0.0,
            ),
            age,
        )


@dataclass(frozen=True)
class DenseTruthTimeline:
    """Timestamp-aligned binary or weighted offline truth."""

    timeline: TimestampedArrayTimeline
    truth_kind: TruthKind

    def __post_init__(self) -> None:
        if self.truth_kind not in {
            "binary_support",
            "probability_weight",
        }:
            raise ValueError("unknown truth_kind.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.timeline.shape

    @property
    def end_time_seconds(self) -> float:
        return self.timeline.end_time_seconds

    @property
    def truth_basis(self) -> str:
        if self.truth_kind == "binary_support":
            return "uniform_probability_over_annotated_binary_support"
        return "normalized_annotated_probability_weight"

    def truth_at_seconds(
        self,
        time_seconds: float,
    ) -> EvaluationTruth:
        frame = np.asarray(
            self.timeline.frame_at_seconds(time_seconds),
            dtype=np.uint8,
        )
        if self.truth_kind == "binary_support":
            support = frame >= 128
            weights = support.astype(float)
        else:
            weights = frame.astype(float) / 255.0
            support = weights > 0.0
        total = float(weights.sum())
        distribution = (
            weights / total
            if total > 0.0
            else np.zeros(weights.shape, dtype=float)
        )
        return EvaluationTruth(
            support=np.asarray(support, dtype=bool),
            distribution=np.asarray(distribution, dtype=float),
        )


def _validate_lookup_time(value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "decision_time_seconds must be finite and non-negative."
        )
