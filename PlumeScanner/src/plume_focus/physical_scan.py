"""Ordered physical paths on a scanner lattice independent of image pixels."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import numpy as np

from plume_focus.sensing_contracts import (
    Message, PathSegment, Position, SampleTarget, ScanPlan, ScannerCapabilities, finite, position, validate_physical_plan,
)
from plume_focus.sensing_messages_v2 import PathSegmentV2, ScanPlanV2
from plume_focus.sensing_session import InstrumentProfileV1


@dataclass(frozen=True)
class PhysicalPlan(Message):
    """Physical action only: no allocation target or inference diagnostics."""
    scan_id: str
    started_at_seconds: float
    completed_at_seconds: float
    start_position_yx: Position
    final_position_yx: Position
    segments: tuple[PathSegmentV2, ...]
    samples: tuple[SampleTarget, ...]
    lattice_shape: tuple[int, int]
    requested_cells: tuple[int, ...]
    swept_cells: tuple[int, ...]
    instrument_hash: str

    def __post_init__(self):
        validate_physical_plan(self)

    @property
    def duration_seconds(self):
        return self.completed_at_seconds - self.started_at_seconds


def compile_profile_scan(mask, profile, *, scan_id, started_at_seconds, current_position,
                         last_reading_at_seconds=None, last_integration_seconds=None, target_mass=.90):
    """Historical message adapter over the shared physical compiler."""
    plan = compile_physical_scan(mask, profile, scan_id=scan_id, started_at_seconds=started_at_seconds,
        current_position=current_position, last_reading_at_seconds=last_reading_at_seconds,
        last_integration_seconds=last_integration_seconds)
    return ScanPlanV2(**{f.name:getattr(plan,f.name) for f in fields(PhysicalPlan)},
        predicted_mass_recall=None,target_mass=target_mass,
        status="candidate" if plan.samples else "no_legal_action",converged=False)


def compile_physical_scan(mask: np.ndarray, profile: InstrumentProfileV1, *, scan_id: str,
                         started_at_seconds: float, current_position: Position,
                         last_reading_at_seconds: float | None = None,
                         last_integration_seconds: float | None = None, return_home: Position | None = None) -> PhysicalPlan:
    """Compile full requested geometry at fixed commanded speed and cadence.

    Detector recovery is a separately charged stationary wait, including after
    a partial prior scan. No selected cells are removed to fit a budget.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != profile.lattice_shape:
        raise ValueError("Requested mask must match the physical scanner lattice.")
    rows, cols = mask.shape
    mechanics, acquisition = profile.mechanics, profile.acquisition
    if last_reading_at_seconds is not None:
        finite("last reading midpoint", last_reading_at_seconds)
        previous_exposure = acquisition.integration_seconds if last_integration_seconds is None else last_integration_seconds
        finite("previous integration", previous_exposure, positive=True)
        if last_reading_at_seconds + previous_exposure / 2 > started_at_seconds + 1e-10:
            raise ValueError("New path precedes the previous actual exposure completion.")
    footprint = profile.capabilities().footprint
    segments, samples = [], []
    closed = np.zeros_like(mask)
    clock, pose = float(started_at_seconds), tuple(current_position)
    last_reading = last_reading_at_seconds
    previous_integration = acquisition.integration_seconds if last_integration_seconds is None else last_integration_seconds

    def segment(kind, destination, duration):
        nonlocal clock, pose
        if duration > 1e-12:
            segments.append(PathSegmentV2(kind, pose, destination, clock, clock + duration))
            clock += duration
            pose = destination

    def visit(destination):
        for axis, speed in ((1, mechanics.move_horizontal_fov_per_second), (0, mechanics.move_vertical_fov_per_second)):
            endpoint = list(pose)
            endpoint[axis] = destination[axis]
            segment("move", tuple(endpoint), abs(endpoint[axis] - pose[axis]) / speed)
        segment("settle", pose, mechanics.settling_seconds_per_visit)
        if last_reading is not None:
            minimum = max(1 / acquisition.detector_capacity_hz, (previous_integration + acquisition.integration_seconds) / 2)
            earliest_midpoint = last_reading + minimum
            segment("wait", pose, max(0, earliest_midpoint - (clock + .5 / acquisition.cadence_hz)))

    if return_home is not None:
        position(return_home)
        visit(return_home)
    columns = np.flatnonzero(mask.any(axis=0))
    if len(columns) and abs(pose[1] - (columns[-1] + .5) / cols) < abs(pose[1] - (columns[0] + .5) / cols):
        columns = columns[::-1]
    for column in columns:
        selected = np.flatnonzero(mask[:, column])
        x = (int(column) + .5) / cols
        if profile.scan_mode == "column":
            lo, hi = int(selected[0]), int(selected[-1])
            low, high = lo / rows, (hi + 1) / rows
            down = abs(pose[0] - low) <= abs(pose[0] - high)
            visit((low if down else high, x))
            order = range(lo, hi + 1) if down else range(hi, lo - 1, -1)
            for n, row in enumerate(order):
                samples.append(SampleTarget(((row + .5) / rows, x), clock + (n + .5) / acquisition.cadence_hz,
                                            acquisition.integration_seconds, footprint))
            segment("sweep", (high if down else low, x), (hi - lo + 1) / acquisition.cadence_hz)
            closed[lo:hi + 1, column] = True
            last_reading = samples[-1].acquired_at_seconds
            previous_integration = acquisition.integration_seconds
        else:
            if abs(pose[0] - (selected[-1] + .5) / rows) < abs(pose[0] - (selected[0] + .5) / rows):
                selected = selected[::-1]
            for row in selected:
                visit(((int(row) + .5) / rows, x))
                samples.append(SampleTarget(pose, clock + .5 / acquisition.cadence_hz,
                                            acquisition.integration_seconds, footprint))
                segment("dwell", pose, 1 / acquisition.cadence_hz)
                closed[row, column] = True
                last_reading = samples[-1].acquired_at_seconds
                previous_integration = acquisition.integration_seconds
    return PhysicalPlan(scan_id, started_at_seconds, clock, current_position, pose, tuple(segments), tuple(samples),
                        mask.shape, tuple(np.flatnonzero(mask).tolist()), tuple(np.flatnonzero(closed).tolist()), profile.digest)



def compile_scan(mask: np.ndarray, capabilities: ScannerCapabilities, *, scan_id: str,
                 started_at_seconds: float, current_position: Position, target_mass: float = 0.9) -> ScanPlan:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != capabilities.lattice_shape:
        raise ValueError("Requested mask must match the physical scanner lattice.")
    rows, cols = mask.shape
    segments, samples = [], []
    swept = np.zeros_like(mask)
    clock = float(started_at_seconds)
    pose = tuple(current_position)

    def move(destination: Position) -> None:
        nonlocal clock, pose
        # Axis-aligned moves: no uncharged diagonal slews or initial reposition.
        for axis, speed in ((1, capabilities.horizontal_speed), (0, capabilities.vertical_speed)):
            endpoint = list(pose)
            endpoint[axis] = destination[axis]
            duration = abs(endpoint[axis] - pose[axis]) / speed
            if duration > 1e-12:
                segments.append(PathSegment("move", pose, tuple(endpoint), clock, clock + duration))
                clock += duration
                pose = tuple(endpoint)

    active_cols = np.flatnonzero(mask.any(axis=0))
    # Start at the nearer horizontal end, preserving a monotone column path.
    if len(active_cols) and abs(current_position[1] - (active_cols[-1] + .5) / cols) < abs(current_position[1] - (active_cols[0] + .5) / cols):
        active_cols = active_cols[::-1]
    for column in active_cols:
        active_rows = np.flatnonzero(mask[:, column])
        if capabilities.scan_mode == "column":
            lo, hi = int(active_rows[0]), int(active_rows[-1])
            ylo, yhi = lo / rows, (hi + 1) / rows
            downward = abs(pose[0] - ylo) <= abs(pose[0] - yhi)
            x = (column + .5) / cols
            start, end = ((ylo, x), (yhi, x)) if downward else ((yhi, x), (ylo, x))
            move(start)
            n = hi - lo + 1
            duration = max((yhi - ylo) / capabilities.vertical_speed,
                           n / capabilities.readout_hz, n * capabilities.integration_seconds)
            segments.append(PathSegment("sweep", start, end, clock, clock + duration))
            ordered = range(lo, hi + 1) if downward else range(hi, lo - 1, -1)
            for index, row in enumerate(ordered):
                samples.append(SampleTarget(((row + .5) / rows, x), clock + (index + .5) * duration / n,
                                            capabilities.integration_seconds, capabilities.footprint))
            swept[lo:hi + 1, column] = True
            clock += duration
            pose = end
        else:
            if abs(pose[0] - (active_rows[-1] + .5) / rows) < abs(pose[0] - (active_rows[0] + .5) / rows):
                active_rows = active_rows[::-1]
            for row in active_rows:
                destination = ((row + .5) / rows, (column + .5) / cols)
                move(destination)
                duration = max(1 / capabilities.readout_hz, capabilities.integration_seconds)
                segments.append(PathSegment("dwell", pose, pose, clock, clock + duration))
                samples.append(SampleTarget(pose, clock + duration / 2, capabilities.integration_seconds, capabilities.footprint))
                clock += duration
                swept[row, column] = True
    return ScanPlan(scan_id, started_at_seconds, clock, current_position, pose, tuple(segments), tuple(samples),
                    mask.shape, tuple(np.flatnonzero(mask).tolist()), tuple(np.flatnonzero(swept).tolist()),
                    0.0, target_mass, "coverage_target_unmet" if samples else "infeasible", False)


def mask_from_cells(cells: tuple[int, ...], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask.ravel()[list(cells)] = True
    return mask


def coverage_mask(plan: ScanPlan, shape: tuple[int, int], capabilities: ScannerCapabilities,
                  *, sampled_only: bool = False) -> np.ndarray:
    """Rasterize physical beam footprints; output resolution never changes plans."""
    yy, xx = np.indices(shape, dtype=float)
    yy, xx = (yy + .5) / shape[0], (xx + .5) / shape[1]
    result = np.zeros(shape, dtype=bool)
    if sampled_only or capabilities.scan_mode == "point":
        for sample in plan.samples:
            result |= ((abs(yy - sample.position_yx[0]) <= sample.footprint.height / 2 + 1e-12)
                       & (abs(xx - sample.position_yx[1]) <= sample.footprint.width / 2 + 1e-12))
    else:
        for segment in plan.segments:
            if segment.kind != "sweep":
                continue
            lo, hi = sorted((segment.start_yx[0], segment.end_yx[0]))
            result |= ((yy >= lo - capabilities.footprint.height / 2) & (yy <= hi + capabilities.footprint.height / 2)
                       & (abs(xx - segment.start_yx[1]) <= capabilities.footprint.width / 2 + 1e-12))
    return result


def with_mass(plan: ScanPlan, mass: np.ndarray, capabilities: ScannerCapabilities, *, converged: bool,
              reacquisition: bool = False, reason: str = "") -> ScanPlan:
    recall = float(mass[coverage_mask(plan, mass.shape, capabilities)].sum())
    status = "target_met" if recall >= plan.target_mass - 1e-10 else "coverage_target_unmet"
    if reacquisition:
        status = "reacquisition"
    if not plan.samples:
        status = "infeasible"
    return replace(plan, predicted_mass_recall=min(1.0, recall), status=status, converged=converged, reason=reason)
