"""Independent R2 directed full-column adaptation of cluster/subset routing.

References: Bahnemann et al., FSR2019, sections 2/4 (arXiv:1907.09224);
Held--Karp (1962), doi:10.1137/0110015. No external implementation imported.
Nonmonotone horizontal reversals are a synthetic mechanical assumption.
"""
from dataclasses import dataclass
from itertools import permutations, product
import hashlib
import math
from pathlib import Path
import numpy as np

from plume_focus.physical_scan import PhysicalPlan
from plume_focus.ordered_column_scan import _geometry, _initial
from plume_focus.c2_scan_validation import audit_plan
from plume_focus.sensing_contracts import SampleTarget, validate_physical_plan, position
from plume_focus.sensing_messages_v2 import PathSegmentV2
from plume_focus.sensing_session import InstrumentProfileV1, configuration_hash, canonical_json

COMPILER = "directed_complete_columns_r2_v1"
MAX_DP_COLUMNS = 10


def solve_monotone_columns(mask, profile, *, objective="FREE", home=(0., 0.), **kw):
    """Linear two-orientation DP, also available for the 14-column G1 mask."""
    arr, spans = _geometry(mask, profile)
    if objective not in ("FREE", "HOME"): raise ValueError("Unknown objective")
    a, m = profile.acquisition, profile.mechanics
    prior = _initial(profile, kw["started_at_seconds"], kw["current_position"], kw.get("last_reading_at_seconds"), kw.get("last_integration_seconds"))
    rows, cols = arr.shape
    def entry_exit(c, d):
        lo, hi = spans[c]; x = (c + .5) / cols
        return ((lo / rows, x), ((hi + 1) / rows, x)) if d else (((hi + 1) / rows, x), (lo / rows, x))
    options = []
    for order in (tuple(sorted(spans)), tuple(sorted(spans, reverse=True))):
        states = {None: (kw["started_at_seconds"], (), kw["current_position"], kw.get("last_reading_at_seconds"), prior)}
        for c in order:
            nxt = {}
            for d in (True, False):
                entry, end = entry_exit(c, d); lo, hi = spans[c]; choices = []
                for t, directions, pose, last, exp in states.values():
                    t += abs(entry[1] - pose[1]) / m.move_horizontal_fov_per_second
                    t += abs(entry[0] - pose[0]) / m.move_vertical_fov_per_second
                    t += m.settling_seconds_per_visit
                    if last is not None:
                        t = max(t, last + max(1 / a.detector_capacity_hz, (exp + a.integration_seconds) / 2) - .5 / a.cadence_hz)
                    t += (hi - lo + 1) / a.cadence_hz
                    choices.append((t, directions + (d,), end, t - .5 / a.cadence_hz, a.integration_seconds))
                nxt[d] = min(choices, key=lambda v: v[0])
            states = nxt
        for t, directions, pose, _, _ in states.values():
            cost = t - kw["started_at_seconds"]
            if objective == "HOME": cost += abs(pose[1] - home[1]) / m.move_horizontal_fov_per_second + abs(pose[0] - home[0]) / m.move_vertical_fov_per_second
            options.append((cost, order, directions))
    _, order, directions = min(options, key=lambda v: v[0])
    p = compile_directed_columns(arr, profile, column_order=order, directions_down=directions, **kw)
    cost = p.duration_seconds + (post_scan_terminal(p, profile, home)["duration_seconds"] if objective == "HOME" else 0.)
    return RouteResult(p, order, directions, cost, 4 * len(spans))


def compile_directed_columns(mask, profile, *, column_order, directions_down,
        scan_id, started_at_seconds, current_position, last_reading_at_seconds=None,
        last_integration_seconds=None):
    """V2 ordered compiler: any permutation, same per-visit physical semantics."""
    mask, spans = _geometry(mask, profile)
    previous_exp = _initial(profile, started_at_seconds, current_position,
                            last_reading_at_seconds, last_integration_seconds)
    order, downs = tuple(column_order), tuple(directions_down)
    if any(type(c) is not int for c in order) or len(order) != len(spans) or set(order) != set(spans):
        raise ValueError("Every complete column must occur exactly once.")
    if len(downs) != len(order) or any(type(d) is not bool for d in downs):
        raise ValueError("Exactly one boolean orientation per column.")
    rows, cols = mask.shape
    acq, mech = profile.acquisition, profile.mechanics
    footprint = profile.capabilities().footprint
    clock, pose = float(started_at_seconds), tuple(current_position)
    last = last_reading_at_seconds
    segments, samples = [], []
    swept = np.zeros_like(mask)

    def segment(kind, dest, dt):
        nonlocal clock, pose
        if dt > 1e-12:
            segments.append(PathSegmentV2(kind, pose, dest, clock, clock + dt))
            clock += dt
            pose = dest

    for c, down in zip(order, downs):
        lo, hi = spans[c]
        low, high, x = lo / rows, (hi + 1) / rows, (c + .5) / cols
        dest = (low if down else high, x)
        for axis, speed in ((1, mech.move_horizontal_fov_per_second), (0, mech.move_vertical_fov_per_second)):
            p = list(pose); p[axis] = dest[axis]
            segment("move", tuple(p), abs(p[axis] - pose[axis]) / speed)
        segment("settle", pose, mech.settling_seconds_per_visit)
        if last is not None:
            gap = max(1 / acq.detector_capacity_hz, (previous_exp + acq.integration_seconds) / 2)
            segment("wait", pose, max(0, last + gap - (clock + .5 / acq.cadence_hz)))
        for k, r in enumerate(range(lo, hi + 1) if down else range(hi, lo - 1, -1)):
            samples.append(SampleTarget(((r + .5) / rows, x), clock + (k + .5) / acq.cadence_hz,
                                        acq.integration_seconds, footprint))
        segment("sweep", (high if down else low, x), (hi - lo + 1) / acq.cadence_hz)
        swept[lo:hi + 1, c] = True
        last, previous_exp = samples[-1].acquired_at_seconds, acq.integration_seconds
    return PhysicalPlan(scan_id, started_at_seconds, clock, current_position, pose, tuple(segments),
                        tuple(samples), mask.shape, tuple(np.flatnonzero(mask).tolist()),
                        tuple(np.flatnonzero(swept).tolist()), profile.digest)


def post_scan_terminal(plan, profile, home=(0., 0.)):
    """Separate terminal schema; never alters the scanning plan or its receipt."""
    position(home)
    clock, pose = plan.completed_at_seconds, plan.final_position_yx
    segments = []
    for axis, speed in ((1, profile.mechanics.move_horizontal_fov_per_second),
                        (0, profile.mechanics.move_vertical_fov_per_second)):
        p = list(pose); p[axis] = home[axis]
        dt = abs(p[axis] - pose[axis]) / speed
        if dt > 1e-12:
            segments.append(PathSegmentV2("move", pose, tuple(p), clock, clock + dt))
            clock += dt; pose = tuple(p)
    return {"schema": "synthetic_home_travel_r2_v1", "semantics": "post-scan pure axial travel, no settling/recovery/readings",
            "home_yx": list(home), "segments": [s.to_dict() for s in segments],
            "duration_seconds": clock - plan.completed_at_seconds, "arrived_at_seconds": clock,
            "final_position_yx": list(pose)}


@dataclass(frozen=True)
class RouteResult:
    plan: PhysicalPlan
    order: tuple
    down: tuple
    objective_seconds: float
    states: int


def solve_subset_columns(mask, profile, *, objective="FREE", monotone=False, home=(0., 0.), **kwargs):
    """D[S,last,direction], earliest completion only; n<=10, no FULL downsampling.

    Last full-column orientation fixes pose and last midpoint=t_end-q/2, with
    profile exposure. Earliest completion thus dominates all later arrivals to
    the same state for time objectives, including an endpoint-only home cost.
    This dominance says nothing about age, covariance or MSE optimality.
    """
    mask, spans = _geometry(mask, profile)
    if objective not in ("FREE", "HOME"): raise ValueError("Unknown terminal objective")
    position(home)
    columns = tuple(sorted(spans)); n = len(columns)
    if n > MAX_DP_COLUMNS: raise ValueError("Exact R2 subset DP supports at most 10 active columns")
    prior = _initial(profile, kwargs["started_at_seconds"], kwargs["current_position"],
                     kwargs.get("last_reading_at_seconds"), kwargs.get("last_integration_seconds"))
    if not n:
        p = compile_directed_columns(mask, profile, column_order=(), directions_down=(), **kwargs)
        return RouteResult(p, (), (), post_scan_terminal(p, profile, home)["duration_seconds"] if objective == "HOME" else 0., 0)
    rows, cols = mask.shape
    a, m = profile.acquisition, profile.mechanics
    q = 1 / a.cadence_hz
    entries, exits, durations = {}, {}, {}
    for i, c in enumerate(columns):
        lo, hi = spans[c]; x = (c + .5) / cols
        durations[i] = (hi - lo + 1) / a.cadence_hz
        for d in (True, False):
            entries[i, d] = (lo / rows if d else (hi + 1) / rows, x)
            exits[i, d] = ((hi + 1) / rows if d else lo / rows, x)

    def finish(t, pose, i, d, last, exposure):
        e = entries[i, d]
        t += abs(e[1] - pose[1]) / m.move_horizontal_fov_per_second
        t += abs(e[0] - pose[0]) / m.move_vertical_fov_per_second
        t += m.settling_seconds_per_visit
        if last is not None:
            t = max(t, last + max(1 / a.detector_capacity_hz, (exposure + a.integration_seconds) / 2) - q / 2)
        return t + durations[i]

    table, parent = {}, {}
    starts = tuple(dict.fromkeys((0, n - 1))) if monotone else range(n)
    for i in starts:
        for d in (True, False):
            key = (1 << i, i, d)
            table[key] = finish(kwargs["started_at_seconds"], kwargs["current_position"], i, d,
                                kwargs.get("last_reading_at_seconds"), prior)
            parent[key] = None
    full = (1 << n) - 1
    for S in range(1, full + 1):
        for i in range(n):
            for d in (True, False):
                old = (S, i, d)
                if old not in table: continue
                nexts = [j for j in range(n) if not S & (1 << j)]
                if monotone:
                    nexts = [j for j in nexts if (S == (1 << (i + 1)) - 1 and j == i + 1)
                             or (S == full ^ ((1 << i) - 1) and j == i - 1)]
                for j in nexts:
                    for nd in (True, False):
                        key = (S | (1 << j), j, nd)
                        val = finish(table[old], exits[i, d], j, nd, table[old] - q / 2, a.integration_seconds)
                        if key not in table or val < table[key]:
                            table[key], parent[key] = val, old
    finals = []
    for key, t in table.items():
        if key[0] != full: continue
        pose = exits[key[1], key[2]]
        extra = abs(pose[1] - home[1]) / m.move_horizontal_fov_per_second + abs(pose[0] - home[0]) / m.move_vertical_fov_per_second
        finals.append((t + (extra if objective == "HOME" else 0.), key))
    value, key = min(finals, key=lambda v: v[0])
    path = []
    while key is not None:
        path.append((columns[key[1]], key[2])); key = parent[key]
    path.reverse()
    order, down = tuple(c for c, d in path), tuple(d for c, d in path)
    p = compile_directed_columns(mask, profile, column_order=order, directions_down=down, **kwargs)
    actual = p.duration_seconds + (post_scan_terminal(p, profile, home)["duration_seconds"] if objective == "HOME" else 0.)
    if not math.isclose(value - kwargs["started_at_seconds"], actual, abs_tol=1e-9, rel_tol=0):
        raise ValueError("DP objective differs from real compiled path")
    return RouteResult(p, order, down, actual, len(table))


def audit_directed_plan(plan, mask, profile, **prior):
    """Reuse the audited one-column visit checks, retaining global validation.

    Each subplan is one entire visit with its real pose/time/prior detector.
    Global uniqueness/closure and continuity are checked separately. No disabling
    of the old monotone audit or historical forced-plan verification.
    """
    validate_physical_plan(plan)
    mask, spans = _geometry(mask, profile)
    if plan.requested_cells != tuple(np.flatnonzero(mask).tolist()): raise ValueError("Requested support differs")
    visited, sample_idx, segment_idx = set(), 0, 0
    pose, clock = plan.start_position_yx, plan.started_at_seconds
    last, exp = prior.get("last_reading_at_seconds"), prior.get("last_integration_seconds")
    totals = {k: 0. for k in ("move", "sweep", "settle", "wait")}
    closed = np.zeros_like(mask)
    while segment_idx < len(plan.segments):
        start = segment_idx
        while segment_idx < len(plan.segments) and plan.segments[segment_idx].kind != "sweep": segment_idx += 1
        if segment_idx == len(plan.segments): raise ValueError("Trailing non-sweep visit")
        s = plan.segments[segment_idx]; segment_idx += 1
        c = int(s.start_yx[1] * mask.shape[1])
        if c not in spans or c in visited: raise ValueError("Repeated/unrequested column")
        visited.add(c); lo, hi = spans[c]; n = hi - lo + 1
        cmask = np.zeros_like(mask); cmask[:, c] = mask[:, c]
        cc = np.zeros_like(mask); cc[lo:hi + 1, c] = True; closed |= cc
        sub = PhysicalPlan(plan.scan_id, clock, s.completed_at_seconds, pose, s.end_yx,
            plan.segments[start:segment_idx], plan.samples[sample_idx:sample_idx + n], mask.shape,
            tuple(np.flatnonzero(cmask).tolist()), tuple(np.flatnonzero(cc).tolist()), profile.digest)
        audit = audit_plan(sub, cmask, profile, last_reading_at_seconds=last, last_integration_seconds=exp)
        for k in totals: totals[k] += audit["totals_seconds"][k]
        sample_idx += n
        pose, clock = s.end_yx, s.completed_at_seconds
        last, exp = sub.samples[-1].acquired_at_seconds, sub.samples[-1].integration_seconds
    if visited != set(spans) or sample_idx != len(plan.samples) or plan.swept_cells != tuple(np.flatnonzero(closed).tolist()):
        raise ValueError("Incomplete/duplicate support or sample count")
    if not math.isclose(sum(totals.values()), plan.duration_seconds, abs_tol=1e-9, rel_tol=0): raise ValueError("Time sum differs")
    if pose != plan.final_position_yx: raise ValueError("Final pose differs")
    return totals


def mechanical_oracle(mask, profile, *, home=(0., 0.), **kw):
    """Independent n! * 2^n complete-cost enumeration, no DP/helper recurrence.

    Vectorizes direction combinations per permutation; legal fixed full-column
    profiles imply zero inter-column recovery wait. First recovery is explicit.
    Selected routes and a preselected sample must still be physically compiled.
    """
    arr = np.asarray(mask, bool); rows, cols = arr.shape
    cs = tuple(int(c) for c in np.flatnonzero(arr.any(axis=0))); n = len(cs)
    if not 1 <= n <= 6: raise ValueError("Independent full oracle supports 1..6 columns")
    a, m = profile.acquisition, profile.mechanics
    directions = np.array(list(product((True, False), repeat=n)), dtype=bool)
    intervals = {c: (int(np.flatnonzero(arr[:, c])[0]) / rows,
                     (int(np.flatnonzero(arr[:, c])[-1]) + 1) / rows) for c in cs}
    results = {}; count = 0; samples = []
    for pi, order in enumerate(permutations(cs)):
        lows = np.array([intervals[c][0] for c in order]); highs = np.array([intervals[c][1] for c in order])
        xs = (np.array(order) + .5) / cols
        entry = np.where(directions, lows, highs); end = np.where(directions, highs, lows)
        # Express the entire path cost, not a stored-state/DP transition.
        first_arrival = abs(xs[0] - kw["current_position"][1]) / m.move_horizontal_fov_per_second
        first_arrival += abs(entry[:, 0] - kw["current_position"][0]) / m.move_vertical_fov_per_second
        first_arrival += m.settling_seconds_per_visit
        if kw.get("last_reading_at_seconds") is not None:
            prev = kw.get("last_integration_seconds") or a.integration_seconds
            ready = kw["last_reading_at_seconds"] + max(1 / a.detector_capacity_hz, (prev + a.integration_seconds) / 2) - .5 / a.cadence_hz - kw["started_at_seconds"]
            first_arrival = np.maximum(first_arrival, ready)
        scan = first_arrival + np.sum(highs - lows) / m.sweep_vertical_fov_per_second
        scan += np.sum(np.abs(np.diff(xs))) / m.move_horizontal_fov_per_second
        scan += np.sum(np.abs(entry[:, 1:] - end[:, :-1]), axis=1) / m.move_vertical_fov_per_second
        scan += (n - 1) * m.settling_seconds_per_visit
        terminal = abs(xs[-1] - home[1]) / m.move_horizontal_fov_per_second + abs(end[:, -1] - home[0]) / m.move_vertical_fov_per_second
        mono = order in (cs, cs[::-1])
        for family in ("all", "monotone") if mono else ("all",):
            for objective, costs in (("FREE", scan), ("HOME", scan + terminal)):
                j = int(np.argmin(costs)); key = family + "_" + objective
                value = float(costs[j])
                if key not in results or value < results[key]["cost"]:
                    results[key] = {"cost": value, "order": order, "down": tuple(bool(x) for x in directions[j])}
        if pi in (0, 1, 7, 31):
            for j in (0, len(directions) // 2, len(directions) - 1):
                samples.append({"order": order, "down": tuple(bool(x) for x in directions[j]), "scan": float(scan[j]), "home": float(scan[j] + terminal[j])})
        count += len(directions)
    return {"candidate_count": count, "winners": results, "preselected_cost_spotchecks": samples}


def source_hashes():
    base = Path(__file__).parent
    return {f: hashlib.sha256((base / f).read_bytes()).hexdigest()
            for f in ("c2_r2_mechanics.py", "ordered_column_scan.py", "c2_scan_validation.py", "physical_scan.py")}


def plan_key(plan):
    return configuration_hash(plan.to_dict())


def route_record(plan, mask, profile, kw, order, down, home=(0., 0.)):
    return {"schema_version": 1, "record_type": "c2_r2_directed_route", "compiler": COMPILER,
            "source_hashes": source_hashes(), "profile": profile.to_dict(), "mask": np.asarray(mask, int).tolist(),
            "kwargs": kw, "column_order": list(order), "directions_down": list(down),
            "scan_plan": plan.to_dict(), "plan_hash": plan_key(plan), "terminal": post_scan_terminal(plan, profile, home)}


def verify_route_record(rec):
    if rec["schema_version"] != 1 or rec["record_type"] != "c2_r2_directed_route" or rec["compiler"] != COMPILER:
        raise ValueError("Unknown R2 compiler record")
    if rec["source_hashes"] != source_hashes(): raise ValueError("R2 compiler source changed")
    profile = InstrumentProfileV1.from_dict(rec["profile"]); kw = dict(rec["kwargs"])
    kw["current_position"] = tuple(kw["current_position"])
    p = compile_directed_columns(rec["mask"], profile, column_order=rec["column_order"], directions_down=rec["directions_down"], **kw)
    if canonical_json(p.to_dict()) != canonical_json(rec["scan_plan"]) or plan_key(p) != rec["plan_hash"]:
        raise ValueError("R2 record does not recompile exactly")
    terminal = post_scan_terminal(p, profile, tuple(rec["terminal"]["home_yx"]))
    if canonical_json(terminal) != canonical_json(rec["terminal"]): raise ValueError("Terminal record differs")
    audit_directed_plan(p, rec["mask"], profile, last_reading_at_seconds=kw.get("last_reading_at_seconds"), last_integration_seconds=kw.get("last_integration_seconds"))
    return p
