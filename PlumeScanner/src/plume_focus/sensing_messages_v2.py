"""Transport v2 additions for frozen policy v1; legacy messages remain unchanged."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar, Literal

from plume_focus.sensing_contracts import Message, PathSegment, Position, ScanPlan, finite, position


@dataclass(frozen=True)
class ScanRequestV1(Message):
    max_duration_seconds: float

    def __post_init__(self):
        if isinstance(self.max_duration_seconds, bool) or not math.isfinite(self.max_duration_seconds):
            raise ValueError("Remaining budget must be finite; nonpositive means no legal action.")


@dataclass(frozen=True)
class PathSegmentV2(Message):
    schema_version: ClassVar[int] = 2
    kind: Literal["move", "sweep", "dwell", "settle", "wait"]
    start_yx: Position
    end_yx: Position
    started_at_seconds: float
    completed_at_seconds: float

    def __post_init__(self):
        position(self.start_yx)
        position(self.end_yx)
        finite("segment start", self.started_at_seconds)
        finite("segment completion", self.completed_at_seconds, self.started_at_seconds)
        if self.kind not in ("move", "sweep", "dwell", "settle", "wait"):
            raise ValueError("Unsupported segment kind.")
        if self.kind in ("settle", "wait", "dwell") and self.start_yx != self.end_yx:
            raise ValueError("A stationary segment cannot change position.")


@dataclass(frozen=True)
class PlanningCandidate(Message):
    selection_horizon_seconds: float
    duration_seconds: float
    requested_cells: tuple[int, ...]
    retained_mass: float | None
    failed_constraints: tuple[str, ...]


@dataclass(frozen=True)
class CoverageDiagnostic(Message):
    delta: float
    grid_points: int
    kappa: float
    lower_retained_fraction: float | None
    total_signal_lower_bound: float
    meaningful_total: bool
    status: Literal["certified", "not_certified"]


@dataclass(frozen=True)
class ScanPlanV2(ScanPlan):
    schema_version: ClassVar[int] = 2
    segments: tuple[PathSegmentV2, ...]
    predicted_mass_recall: float | None
    status: Literal["candidate", "target_met", "static_corridor", "startup_full_scene", "reacquisition", "no_legal_action"]
    policy_id: str = "mass90_column_v1"
    instrument_hash: str = ""
    candidates: tuple[PlanningCandidate, ...] = ()
    termination: str = ""
    tracking_shortfall: tuple[str, ...] = ()
    confidence: CoverageDiagnostic | None = None
    expected_signal_total: float | None = None
    startup_complete: bool = False

    def __post_init__(self):
        if self.status not in ("candidate", "target_met", "static_corridor", "startup_full_scene", "reacquisition", "no_legal_action"):
            raise ValueError("Unsupported v2 plan status.")
        if self.status == "no_legal_action" and (self.samples or self.segments):
            raise ValueError("No legal action cannot open an executable path.")
        if self.predicted_mass_recall is not None:
            finite("predicted mass recall", self.predicted_mass_recall)
        if self.expected_signal_total is not None:
            finite("expected signal total", self.expected_signal_total)
        # Reuse the established path/exposure validator without changing v1 wire
        # semantics. Stationary settle/wait segments are legal zero-distance moves.
        legacy_segments = tuple(PathSegment(s.kind if s.kind in ("move", "sweep", "dwell") else "move",
                                            s.start_yx, s.end_yx, s.started_at_seconds, s.completed_at_seconds)
                                for s in self.segments)
        ScanPlan(self.scan_id, self.started_at_seconds, self.completed_at_seconds,
                 self.start_position_yx, self.final_position_yx, legacy_segments, self.samples,
                 self.lattice_shape, self.requested_cells, self.swept_cells,
                 self.predicted_mass_recall if self.predicted_mass_recall is not None else 0.0,
                 self.target_mass, "coverage_target_unmet", self.converged, self.reason)
