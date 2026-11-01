"""Independent exhaustive oracle, physical audit and versioned C2 records.

No detector values, inference, controller plan replacement or verify_plan bypass.
"""
from __future__ import annotations

from collections import Counter
from itertools import product
import hashlib
import math
from pathlib import Path
import numpy as np

from plume_focus.physical_scan import compile_physical_scan
from plume_focus.ordered_column_scan import (
    compile_ordered_column_scan_v1, monotone_dp_scan_v1, ORDERED_COMPILER_ID, DP_COMPILER_ID,
)
from plume_focus.sensing_contracts import validate_physical_plan
from plume_focus.sensing_session import InstrumentProfileV1, canonical_json, configuration_hash

BASELINE_COMPILER_ID = "compile_physical_scan_at_5cac633"


def baseline_order(plan):
    cols = plan.lattice_shape[1]
    sweeps = [s for s in plan.segments if s.kind == "sweep"]
    return (tuple(int(s.start_yx[1] * cols) for s in sweeps),
            tuple(s.end_yx[0] > s.start_yx[0] for s in sweeps))


def exhaustive_oracle(mask, profile, **kwargs):
    """Enumerate whole compiled paths. Does not use DP costs/states/recurrence."""
    columns = tuple(int(c) for c in np.flatnonzero(np.asarray(mask).any(axis=0)))
    best, count = None, 0
    for order in (columns, columns[::-1]):
        for directions in product((True, False), repeat=len(columns)):
            p = compile_ordered_column_scan_v1(mask, profile, column_order=order,
                                              directions_down=directions, **kwargs)
            count += 1
            if best is None or p.duration_seconds < best.duration_seconds:
                best = p
    return best, count


def audit_plan(plan, mask, profile, *, last_reading_at_seconds=None, last_integration_seconds=None):
    """Check full physical semantics independently of either cost solver.

    Raises ValueError on a failed check (also under python -O).
    Includes an exact visit decomposition, detector readiness, per-sample path
    interpolation/exposure containment, closure, uniqueness and time accounting.
    """
    def require(condition, message):
        if not condition:
            raise ValueError("C2 audit: " + message)

    def near(a, b):
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9)

    validate_physical_plan(plan)
    mask = np.asarray(mask, dtype=bool)
    rows, cols = mask.shape
    require(mask.shape == profile.lattice_shape == plan.lattice_shape, "lattice")
    require(plan.instrument_hash == profile.digest, "profile hash")
    acq, mech = profile.acquisition, profile.mechanics
    slot = 1 / acq.cadence_hz
    closure = mask.copy()
    active = tuple(int(c) for c in np.flatnonzero(mask.any(axis=0)))
    for c in active:
        r = np.flatnonzero(mask[:, c])
        closure[r[0]:r[-1] + 1, c] = True
    require(plan.requested_cells == tuple(np.flatnonzero(mask).tolist()), "requested support")
    require(plan.swept_cells == tuple(np.flatnonzero(closure).tolist()), "swept support")
    order, _ = baseline_order(plan)
    require(order in (active, active[::-1]), "complete monotone columns")
    require(len(order) == len(active), "one complete sweep per column")
    counts = Counter()
    prev_mid = last_reading_at_seconds
    prev_exp = acq.integration_seconds if last_integration_seconds is None else last_integration_seconds
    if prev_mid is not None:
        require(prev_mid + prev_exp / 2 <= plan.started_at_seconds + 1e-9, "initial exposure complete")
    pose, clock = plan.start_position_yx, plan.started_at_seconds
    totals = {k: 0. for k in ("move", "sweep", "settle", "wait")}
    cursor, sample_cursor = 0, 0
    for c in order:
        visit = []
        while cursor < len(plan.segments):
            s = plan.segments[cursor]
            cursor += 1
            visit.append(s)
            if s.kind == "sweep":
                break
        sweep = visit[-1]
        require(sweep.kind == "sweep", "visit terminates in sweep")
        r = np.flatnonzero(mask[:, c])
        lo, hi, x = int(r[0]), int(r[-1]), (c + .5) / cols
        low, high = lo / rows, (hi + 1) / rows
        require(near(sweep.start_yx[1], x) and near(sweep.end_yx[1], x), "vertical sweep axis")
        require((near(sweep.start_yx[0], low) and near(sweep.end_yx[0], high)) or
                (near(sweep.start_yx[0], high) and near(sweep.end_yx[0], low)), "complete sweep endpoints")
        require(near(abs(high - low) / (sweep.completed_at_seconds - sweep.started_at_seconds),
                     mech.sweep_vertical_fov_per_second), "sweep speed")
        # Reconstruct the charged visit, without calling any compiler helper.
        specs = []
        p = list(pose)
        for axis, speed in ((1, mech.move_horizontal_fov_per_second), (0, mech.move_vertical_fov_per_second)):
            dest = list(p)
            dest[axis] = sweep.start_yx[axis]
            dt = abs(dest[axis] - p[axis]) / speed
            if dt > 1e-12:
                specs.append(("move", tuple(p), tuple(dest), dt))
                clock += dt
                p = dest
        if mech.settling_seconds_per_visit > 1e-12:
            dt = mech.settling_seconds_per_visit
            specs.append(("settle", tuple(p), tuple(p), dt))
            clock += dt
        if prev_mid is not None:
            gap = max(1 / acq.detector_capacity_hz, (prev_exp + acq.integration_seconds) / 2)
            dt = max(0., prev_mid + gap - clock - slot / 2)
            if dt > 1e-12:
                specs.append(("wait", tuple(p), tuple(p), dt))
                clock += dt
        require(len(specs) == len(visit) - 1, "visit segment counts / initial detector wait")
        for expected, actual in zip(specs, visit[:-1]):
            k, a, b, dt = expected
            require(actual.kind == k and np.allclose(actual.start_yx, a, atol=1e-10, rtol=0)
                    and np.allclose(actual.end_yx, b, atol=1e-10, rtol=0)
                    and near(actual.completed_at_seconds - actual.started_at_seconds, dt), "axis speed / visit order / wait")
        require(near(clock, sweep.started_at_seconds), "sweep starts after charged visit")
        n = hi - lo + 1
        require(near(sweep.completed_at_seconds - sweep.started_at_seconds, n * slot), "cadence sweep duration")
        samples = plan.samples[sample_cursor:sample_cursor + n]
        require(len(samples) == n, "sample count in column")
        sample_cursor += n
        for j, sample in enumerate(samples):
            require(near(sample.acquired_at_seconds, sweep.started_at_seconds + (j + .5) * slot), "fixed cadence")
            require(near(sample.integration_seconds, acq.integration_seconds), "integration setting")
            require(sample.footprint == profile.capabilities().footprint, "footprint")
            require(sample.acquired_at_seconds - sample.integration_seconds / 2 >= sweep.started_at_seconds - 1e-9
                    and sample.acquired_at_seconds + sample.integration_seconds / 2 <= sweep.completed_at_seconds + 1e-9, "exposure inside sweep")
            fraction = (sample.acquired_at_seconds - sweep.started_at_seconds) / (sweep.completed_at_seconds - sweep.started_at_seconds)
            physical_midpoint = np.array(sweep.start_yx) + fraction * (np.array(sweep.end_yx) - sweep.start_yx)
            require(np.allclose(sample.position_yx, physical_midpoint, atol=1e-10, rtol=0), "sample midpoint lies on path")
            rr = int(sample.position_yx[0] * rows)
            require(near(sample.position_yx[0], (rr + .5) / rows) and near(sample.position_yx[1], x), "sample lattice center")
            counts[rr * cols + c] += 1
            if prev_mid is not None:
                require(sample.acquired_at_seconds - prev_mid >= max(1 / acq.detector_capacity_hz,
                        (prev_exp + sample.integration_seconds) / 2) - 1e-9, "detector capacity / exposure spacing")
            prev_mid, prev_exp = sample.acquired_at_seconds, sample.integration_seconds
        for s in visit:
            require(s.kind in totals, "unsupported segment")
            totals[s.kind] += s.completed_at_seconds - s.started_at_seconds
        clock, pose = sweep.completed_at_seconds, sweep.end_yx
    require(cursor == len(plan.segments), "no extra segments")
    require(sample_cursor == len(plan.samples) == int(closure.sum()), "no extra/missing samples")
    require(set(counts) == set(plan.swept_cells) and all(n == 1 for n in counts.values()), "no repeated or missed cells")
    require(near(sum(totals.values()), plan.duration_seconds) and near(clock, plan.completed_at_seconds), "component time sum")
    require(np.allclose(pose, plan.final_position_yx, atol=1e-10, rtol=0), "final pose")
    return {"status": "PASS", "totals_seconds": totals, "samples": len(plan.samples), "requested": int(mask.sum()), "swept": int(closure.sum())}


def compiler_hashes():
    base = Path(__file__).parent
    return {f: hashlib.sha256((base / f).read_bytes()).hexdigest()
            for f in ("physical_scan.py", "ordered_column_scan.py", "c2_scan_validation.py")}


def make_record(compiler_id, mask, profile, plan, kwargs, *, order=(), directions=()):
    inputs = {"mask": np.asarray(mask, dtype=int).tolist(), "profile": profile.to_dict(),
              "kwargs": kwargs, "column_order": list(order), "directions_down": list(directions)}
    return {"schema_version": 1, "record_type": "c2_mechanical_compiler_record_v1",
            "compiler_id": compiler_id, "compiler_sha256": compiler_hashes(),
            "inputs": inputs, "inputs_sha256": configuration_hash(inputs),
            "plan": plan.to_dict(), "plan_sha256": configuration_hash(plan.to_dict())}


def verify_record(record):
    if record["schema_version"] != 1 or record["record_type"] != "c2_mechanical_compiler_record_v1":
        raise ValueError("Unknown C2 record.")
    if record["compiler_sha256"] != compiler_hashes():
        raise ValueError("Compiler source hash mismatch.")
    inputs = record["inputs"]
    if configuration_hash(inputs) != record["inputs_sha256"]:
        raise ValueError("Input hash mismatch.")
    profile = InstrumentProfileV1.from_dict(inputs["profile"])
    mask, kwargs = np.array(inputs["mask"], dtype=bool), dict(inputs["kwargs"])
    kwargs["current_position"] = tuple(kwargs["current_position"])
    cid = record["compiler_id"]
    if cid == BASELINE_COMPILER_ID:
        plan = compile_physical_scan(mask, profile, **kwargs)
    elif cid == ORDERED_COMPILER_ID:
        plan = compile_ordered_column_scan_v1(mask, profile, column_order=inputs["column_order"],
                                            directions_down=inputs["directions_down"], **kwargs)
    elif cid == DP_COMPILER_ID:
        dp = monotone_dp_scan_v1(mask, profile, **kwargs)
        if list(dp.column_order) != inputs["column_order"] or list(dp.directions_down) != inputs["directions_down"]:
            raise ValueError("DP selection mismatch.")
        plan = dp.plan
    else:
        raise ValueError("Unknown compiler version.")
    if canonical_json(plan.to_dict()) != canonical_json(record["plan"]) or configuration_hash(plan.to_dict()) != record["plan_sha256"]:
        raise ValueError("Recompiled plan mismatch.")
    audit_plan(plan, mask, profile, last_reading_at_seconds=kwargs.get("last_reading_at_seconds"),
               last_integration_seconds=kwargs.get("last_integration_seconds"))
    return plan
