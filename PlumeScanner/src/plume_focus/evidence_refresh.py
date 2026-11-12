"""A caller can request newly observed cells through the existing FULL lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal

import numpy as np

from plume_focus.evidence_gated_loop import EvidenceGatedScanner, EvidenceGatedSession
from plume_focus.full_fov_loop import FullFovPlan
from plume_focus.physical_scan import PhysicalPlan, compile_physical_scan
from plume_focus.sensing_contracts import CausalWind, Position, SensorReading, finite, position
from plume_focus.sensing_messages_v2 import ScanRequestV1


EVIDENCE_REFRESH_VERSION = "evidence_refresh_request_v1"
_TIME_TOLERANCE = 1e-10


@dataclass(frozen=True)
class EvidenceRefreshRequest:
    request_id: str
    instrument_hash: str
    cells: tuple[int, ...]
    fresh_since_seconds: float
    deadline_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("A nonempty request ID is required.")
        if not isinstance(self.instrument_hash, str) or not self.instrument_hash:
            raise ValueError("Bind the requested cells to an instrument profile.")
        if (not isinstance(self.cells, tuple) or not self.cells
                or any(type(cell) is not int or cell < 0 for cell in self.cells)
                or len(set(self.cells)) != len(self.cells)):
            raise ValueError("Request distinct nonnegative physical cell indices as a tuple.")
        finite("freshness cutoff", self.fresh_since_seconds)
        finite("evidence deadline", self.deadline_seconds, self.fresh_since_seconds)


@dataclass(frozen=True)
class DeliveredCellEvidence:
    cell: int
    reading: SensorReading
    accepted_at_seconds: float


@dataclass(frozen=True)
class EvidenceRefreshStatus:
    request_id: str
    state: Literal["satisfied", "pending", "deadline_missed"]
    evidence: tuple[DeliveredCellEvidence, ...]
    missing_cells: tuple[int, ...]
    active_scan_id: str | None
    outstanding_full_cells: int


@dataclass(frozen=True)
class EvidenceRefreshDecision:
    status: EvidenceRefreshStatus
    outcome: Literal[
        "satisfied", "deadline_missed", "scan_in_progress", "scheduled",
        "full_scan_exceeds_budget", "current_full_lacks_cells", "window_unreachable",
    ]
    plan: FullFovPlan | None = None
    required_duration_seconds: float | None = None
    estimated_last_acceptance_seconds: float | None = None
    version: str = EVIDENCE_REFRESH_VERSION
    local_reason: Literal["multiple_cells", "full_required", "window_unreachable",
                          "terminal_budget", "full_preferred"] | None = None


class EvidenceRefreshScanner(EvidenceGatedScanner):
    """Opt-in whole-field controller with a requested fresh-evidence entry point.

    The host calls the same ingest/finish methods for ordinary and requested scans.
    A request may advance a FULL scan at a READY boundary. The normal receipt and
    gate still govern when another action may open.
    """

    def __init__(self, session: EvidenceGatedSession, *, delivery: Literal["stream", "batch"]):
        if delivery not in ("stream", "batch"):
            raise ValueError("Declare stream or batch delivery for deadline estimates.")
        super().__init__(session)
        self._refresh_delivery = delivery
        self._refresh_planning = False
        # The base controller retains readings, but resets its delivery clocks on
        # each new scan. Keep the first controller acceptance time for each reading.
        self._evidence_accepted_at: dict[str, float] = {}

    def ingest_readings(self, readings, *, available_at_seconds):
        events = super().ingest_readings(readings, available_at_seconds=available_at_seconds)
        for reading in readings:
            if reading.reading_id in self._pending:
                self._evidence_accepted_at.setdefault(reading.reading_id, available_at_seconds)
        return events

    def _physical_cell(self, position_yx: Position) -> int:
        rows, cols = self.capabilities.lattice_shape
        y, x = position_yx
        return min(int(y * rows), rows - 1) * cols + min(int(x * cols), cols - 1)

    def evidence_status(self, request: EvidenceRefreshRequest, *, now_seconds: float) -> EvidenceRefreshStatus:
        """Return accepted measurements and unmet cells at the current causal boundary."""
        if not isinstance(request, EvidenceRefreshRequest):
            raise TypeError("Pass an EvidenceRefreshRequest.")
        finite("status time", now_seconds)
        if now_seconds < self._last_available - _TIME_TOLERANCE:
            raise ValueError("Request status precedes the controller's available evidence.")
        if request.instrument_hash != self.session.instrument.digest:
            raise ValueError("Request instrument differs from the controller instrument.")
        if max(request.cells) >= math.prod(self.capabilities.lattice_shape):
            raise ValueError("A requested cell lies outside the physical lattice.")

        wanted = set(request.cells)
        evidence: dict[int, DeliveredCellEvidence] = {}
        latest_acceptance = min(now_seconds, request.deadline_seconds)
        for readings in (self._seen, self._pending):
            for reading_id, reading in readings.items():
                accepted_at = self._evidence_accepted_at.get(reading_id)
                if accepted_at is None or accepted_at > latest_acceptance + _TIME_TOLERANCE:
                    continue
                exposure_start = reading.acquired_at_seconds - reading.integration_seconds / 2
                if exposure_start < request.fresh_since_seconds - _TIME_TOLERANCE:
                    continue
                cell = self._physical_cell(reading.position_yx)
                if cell not in wanted:
                    continue
                current = evidence.get(cell)
                if current is None or (reading.acquired_at_seconds, reading_id) > (
                    current.reading.acquired_at_seconds, current.reading.reading_id
                ):
                    evidence[cell] = DeliveredCellEvidence(cell, reading, accepted_at)

        missing = tuple(cell for cell in request.cells if cell not in evidence)
        state = "pending"
        if not missing:
            state = "satisfied"
        elif now_seconds >= request.deadline_seconds:
            state = "deadline_missed"
        active = self.active_plan
        return EvidenceRefreshStatus(
            request.request_id, state, tuple(evidence[cell] for cell in request.cells if cell in evidence),
            missing, active.scan_id if active else None, len(self.full_scan_missing),
        )

    def _focus_supported(self, gate_mean):
        return not self._refresh_planning and super()._focus_supported(gate_mean)

    def _decorate_plan(self, plan, core):
        if self._refresh_planning and plan.mode == "FULL" and plan.reason == "signal_not_supported":
            plan = replace(plan, reason="evidence_refresh_requested")
        return super()._decorate_plan(plan, core)

    def _validate_refresh_context(self, now, pose, constraints, preparation, wind):
        finite("preparation duration", preparation)
        position(pose)
        if not isinstance(constraints, ScanRequestV1):
            raise TypeError("Pass remaining session duration in ScanRequestV1.")
        if not np.allclose(pose, self._last_pose, atol=_TIME_TOLERANCE, rtol=0):
            raise ValueError("Position must match the last execution receipt.")
        if wind is not None and not self._wind.available_at_seconds <= wind.available_at_seconds <= now:
            raise ValueError("Wind must already be available at the request boundary.")

    def _quote_refresh_path(self, mask, start, pose, *, last, return_home=None, scan_offset=0):
        return compile_physical_scan(
            mask, self.session.instrument, scan_id=f"scan-{self._scan_count + scan_offset:06d}",
            started_at_seconds=start, current_position=tuple(pose),
            last_reading_at_seconds=last.acquired_at_seconds if last is not None else None,
            last_integration_seconds=last.integration_seconds if last is not None else None,
            return_home=return_home,
        )

    def _quote_full_refresh(self, start, pose):
        # An unfinished FULL keeps its existing mask and return obligation.
        mask = self._full_missing
        returning = self._return_pending
        if mask is None:
            mask = np.ones(self.capabilities.lattice_shape, dtype=bool)
            returning = True
        return self._quote_refresh_path(
            mask, start, pose, last=self._last_detector_reading,
            return_home=self.session.home_yx if returning else None,
        )

    def _estimated_receipt(self, path: PhysicalPlan) -> float:
        settings = self.session.settings
        return path.completed_at_seconds + settings.delivery_delay_seconds + settings.completion_overrun_seconds

    def _estimated_acceptance(self, request, missing, path: PhysicalPlan) -> float | None:
        receipt_at = self._estimated_receipt(path)
        arrivals: dict[int, float] = {}
        for sample in path.samples:
            cell = self._physical_cell(sample.position_yx)
            exposure_start = sample.acquired_at_seconds - sample.integration_seconds / 2
            if cell not in missing or exposure_start < request.fresh_since_seconds - _TIME_TOLERANCE:
                continue
            arrival = (receipt_at if self._refresh_delivery == "batch" else
                       sample.acquired_at_seconds + sample.integration_seconds / 2
                       + self.session.settings.delivery_delay_seconds)
            arrivals[cell] = min(arrivals.get(cell, arrival), arrival)
        return max(arrivals.values()) if missing.issubset(arrivals) else None

    def plan_evidence_refresh(
        self,
        request: EvidenceRefreshRequest,
        now_seconds: float,
        current_position: Position,
        constraints: ScanRequestV1,
        *,
        preparation_seconds: float,
        wind: CausalWind | None = None,
    ) -> EvidenceRefreshDecision:
        """Open a native FULL only if the declared window and session budget fit.

        The preparation allowance is supplied by the host and charged against its
        remaining session duration. Estimated acceptance is conditional on the
        declared transport delay and delivery mode. Status uses actual arrivals.
        """
        status = self.evidence_status(request, now_seconds=now_seconds)
        if status.state != "pending":
            return EvidenceRefreshDecision(status, status.state)
        if self.active_plan is not None:
            return EvidenceRefreshDecision(status, "scan_in_progress")

        self._validate_refresh_context(now_seconds, current_position, constraints, preparation_seconds, wind)
        start = now_seconds + preparation_seconds
        quote = self._quote_full_refresh(start, current_position)
        required = self._estimated_receipt(quote) - now_seconds
        if required > constraints.max_duration_seconds + 1e-9:
            return EvidenceRefreshDecision(status, "full_scan_exceeds_budget", required_duration_seconds=required)
        missing = set(status.missing_cells)
        if not missing.issubset(quote.swept_cells):
            return EvidenceRefreshDecision(status, "current_full_lacks_cells", required_duration_seconds=required)

        estimate = self._estimated_acceptance(request, missing, quote)
        if estimate is None or estimate > request.deadline_seconds + _TIME_TOLERANCE:
            return EvidenceRefreshDecision(status, "window_unreachable", required_duration_seconds=required,
                                           estimated_last_acceptance_seconds=estimate)

        self._refresh_planning = True
        try:
            plan = super().plan_next(start, current_position,
                                     ScanRequestV1(constraints.max_duration_seconds - preparation_seconds), wind=wind)
        finally:
            self._refresh_planning = False
        return EvidenceRefreshDecision(
            replace(status, active_scan_id=plan.scan_id, outstanding_full_cells=len(self.full_scan_missing)),
            "scheduled", plan, required, estimate,
        )
