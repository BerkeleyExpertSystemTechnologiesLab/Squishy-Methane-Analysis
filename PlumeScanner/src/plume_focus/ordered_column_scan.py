"""Opt-in, complete column sweeps; no controller/factory integration.

`down=True` means increasing normalized y. The unchanged physical_scan module
remains the historical baseline. Compiler and optimizer have separate names.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from plume_focus.physical_scan import PhysicalPlan
from plume_focus.sensing_contracts import SampleTarget, finite, position
from plume_focus.sensing_messages_v2 import PathSegmentV2

ORDERED_COMPILER_ID = "ordered_column_scan_v1"
DP_COMPILER_ID = "monotone_dp_scan_v1"


def _geometry(mask, profile):
    mask = np.asarray(mask, dtype=bool)
    if profile.scan_mode != "column":
        raise ValueError("C2 requires column mode.")
    if mask.shape != profile.lattice_shape:
        raise ValueError("Requested mask must match the physical scanner lattice.")
    spans = {}
    for c in np.flatnonzero(mask.any(axis=0)):
        selected = np.flatnonzero(mask[:, c])
        spans[int(c)] = (int(selected[0]), int(selected[-1]))
    return mask, spans


def _initial(profile, started_at_seconds, current_position, last_reading_at_seconds, last_integration_seconds):
    finite("plan start", started_at_seconds)
    position(current_position)
    prior = profile.acquisition.integration_seconds if last_integration_seconds is None else last_integration_seconds
    finite("previous integration", prior, positive=True)
    if last_reading_at_seconds is not None:
        finite("last reading midpoint", last_reading_at_seconds)
        if last_reading_at_seconds + prior / 2 > started_at_seconds + 1e-10:
            raise ValueError("New path precedes the previous actual exposure completion.")
    return prior


def compile_ordered_column_scan_v1(mask, profile, *, column_order, directions_down,
        scan_id, started_at_seconds, current_position,
        last_reading_at_seconds=None, last_integration_seconds=None):
    """Compile an explicit monotone order and complete sweep orientations.

    Retains baseline floating-operation order: horizontal -> vertical -> settle
    -> detector wait -> fixed-cadence sweep, including all gap cells.
    """
    mask, spans = _geometry(mask, profile)
    previous_integration = _initial(profile, started_at_seconds, current_position,
                                    last_reading_at_seconds, last_integration_seconds)
    columns, downs = tuple(column_order), tuple(directions_down)
    ascending = tuple(sorted(spans))
    if any(type(c) is not int for c in columns) or columns not in (ascending, ascending[::-1]):
        raise ValueError("Column order must include every active column once, monotonically.")
    if len(downs) != len(columns) or any(type(d) is not bool for d in downs):
        raise ValueError("One explicit boolean direction is required per column.")
    rows, cols = mask.shape
    mechanics, acquisition = profile.mechanics, profile.acquisition
    footprint = profile.capabilities().footprint
    segments, samples = [], []
    closed = np.zeros_like(mask)
    clock, pose = float(started_at_seconds), tuple(current_position)
    last_reading = last_reading_at_seconds

    def segment(kind, destination, duration):
        nonlocal clock, pose
        if duration > 1e-12:
            segments.append(PathSegmentV2(kind, pose, destination, clock, clock + duration))
            clock += duration
            pose = destination

    for c, down in zip(columns, downs):
        lo, hi = spans[c]
        low, high, x = lo / rows, (hi + 1) / rows, (c + .5) / cols
        destination = (low if down else high, x)
        for axis, speed in ((1, mechanics.move_horizontal_fov_per_second), (0, mechanics.move_vertical_fov_per_second)):
            endpoint = list(pose)
            endpoint[axis] = destination[axis]
            segment("move", tuple(endpoint), abs(endpoint[axis] - pose[axis]) / speed)
        segment("settle", pose, mechanics.settling_seconds_per_visit)
        if last_reading is not None:
            minimum = max(1 / acquisition.detector_capacity_hz,
                          (previous_integration + acquisition.integration_seconds) / 2)
            segment("wait", pose, max(0, last_reading + minimum - (clock + .5 / acquisition.cadence_hz)))
        order = range(lo, hi + 1) if down else range(hi, lo - 1, -1)
        for n, row in enumerate(order):
            samples.append(SampleTarget(((row + .5) / rows, x), clock + (n + .5) / acquisition.cadence_hz,
                                        acquisition.integration_seconds, footprint))
        segment("sweep", (high if down else low, x), (hi - lo + 1) / acquisition.cadence_hz)
        closed[lo:hi + 1, c] = True
        last_reading = samples[-1].acquired_at_seconds
        previous_integration = acquisition.integration_seconds
    return PhysicalPlan(scan_id, started_at_seconds, clock, current_position, pose,
                        tuple(segments), tuple(samples), mask.shape,
                        tuple(np.flatnonzero(mask).tolist()), tuple(np.flatnonzero(closed).tolist()), profile.digest)


@dataclass(frozen=True)
class MonotoneDPResult:
    plan: PhysicalPlan
    column_order: tuple[int, ...]
    directions_down: tuple[bool, ...]
    predicted_duration_seconds: float
    states_evaluated: int


def monotone_dp_scan_v1(mask, profile, *, scan_id, started_at_seconds, current_position,
                         last_reading_at_seconds=None, last_integration_seconds=None):
    """Minimize completion time over two monotone orders x binary orientations.

    After a complete column, (stage, final orientation) fixes pose and fixes
    last midpoint relative to completion (-half cadence slot). Exposure resets
    to the profile value. Thus a later arrival at the same state cannot improve
    any continuation. The first stage uses the actual prior detector state.
    Only scan duration is minimized; home travel and ages are diagnostics.
    """
    mask, spans = _geometry(mask, profile)
    prior = _initial(profile, started_at_seconds, current_position,
                     last_reading_at_seconds, last_integration_seconds)
    rows, cols = mask.shape
    acq, mech = profile.acquisition, profile.mechanics
    slot = 1 / acq.cadence_hz
    orders = (tuple(sorted(spans)), tuple(sorted(spans, reverse=True)))
    if not spans:
        p = compile_ordered_column_scan_v1(mask, profile, column_order=(), directions_down=(),
            scan_id=scan_id, started_at_seconds=started_at_seconds, current_position=current_position,
            last_reading_at_seconds=last_reading_at_seconds, last_integration_seconds=last_integration_seconds)
        return MonotoneDPResult(p, (), (), 0., 0)

    def endpoints(c, down):
        lo, hi = spans[c]
        low, high, x = lo / rows, (hi + 1) / rows, (c + .5) / cols
        return ((low, x), (high, x)) if down else ((high, x), (low, x))

    def finish(t, pose, c, down, last, previous_exposure):
        entry, _ = endpoints(c, down)
        arrival = t + abs(entry[1] - pose[1]) / mech.move_horizontal_fov_per_second
        arrival += abs(entry[0] - pose[0]) / mech.move_vertical_fov_per_second
        arrival += mech.settling_seconds_per_visit
        if last is not None:
            gap = max(1 / acq.detector_capacity_hz, (previous_exposure + acq.integration_seconds) / 2)
            arrival = max(arrival, last + gap - slot / 2)
        lo, hi = spans[c]
        return arrival + (hi - lo + 1) / acq.cadence_hz

    candidates, states = [], 0
    for order in orders:
        # Values are (earliest completion, complete direction prefix).
        table = {}
        for down in (True, False):
            table[down] = (finish(started_at_seconds, current_position, order[0], down,
                                 last_reading_at_seconds, prior), (down,))
            states += 1
        for i in range(1, len(order)):
            nxt = {}
            for down in (True, False):
                choices = []
                for prev in (True, False):
                    t, path = table[prev]
                    _, pose = endpoints(order[i - 1], prev)
                    choices.append((finish(t, pose, order[i], down, t - slot / 2,
                                           acq.integration_seconds), path + (down,)))
                nxt[down] = min(choices, key=lambda v: v[0])
                states += 1
            table = nxt
        for down in (True, False):
            t, directions = table[down]
            candidates.append((t, order, directions))
    t, order, directions = min(candidates, key=lambda v: v[0])
    plan = compile_ordered_column_scan_v1(mask, profile, column_order=order, directions_down=directions,
        scan_id=scan_id, started_at_seconds=started_at_seconds, current_position=current_position,
        last_reading_at_seconds=last_reading_at_seconds, last_integration_seconds=last_integration_seconds)
    if not math.isclose(plan.completed_at_seconds, t, rel_tol=0, abs_tol=1e-9):
        raise ValueError("DP cost does not match ordered compilation.")
    return MonotoneDPResult(plan, order, directions, t - started_at_seconds, states)
