from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from plume_focus.physical_scan import compile_physical_scan
from plume_focus.ordered_column_scan import compile_ordered_column_scan_v1, monotone_dp_scan_v1, DP_COMPILER_ID
from plume_focus.c2_scan_validation import baseline_order, exhaustive_oracle, audit_plan, make_record, verify_record
from plume_focus.sensing_session import InstrumentProfileV1


@pytest.fixture
def profile():
    data = json.loads((Path(__file__).parents[1] / "configs/evidence-gated-focus/coarse-demo.json").read_text())
    return InstrumentProfileV1.from_dict(data["session"]["instrument"])


def fixture_mask(n):
    rng = np.random.default_rng(20261029 + n)
    m = np.zeros((20, 32), bool)
    for c in rng.choice(32, n, replace=False):
        m[rng.choice(20, 3, replace=False), c] = True
    return m


@pytest.mark.parametrize("n", [0, 1, 2, 4, 6])
@pytest.mark.parametrize("start", [(0., 0.), (1., 1.), (.5, .5)])
def test_baseline_exact_parity_before_optimization(profile, n, start):
    mask = fixture_mask(n)
    kw = dict(scan_id="parity", started_at_seconds=1., current_position=start,
              last_reading_at_seconds=.9875, last_integration_seconds=.025)
    baseline = compile_physical_scan(mask, profile, **kw)
    columns, directions = baseline_order(baseline)
    ordered = compile_ordered_column_scan_v1(mask, profile, column_order=columns, directions_down=directions, **kw)
    assert ordered == baseline  # every segment/sample/time/support/duration/final pose
    audit_plan(ordered, mask, profile, last_reading_at_seconds=.9875, last_integration_seconds=.025)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("start", [(0., 0.), (1., 1.), (.5, .5)])
@pytest.mark.parametrize("prior", [None, .9875])
def test_dp_matches_full_path_enumeration(profile, n, start, prior):
    mask = fixture_mask(n)
    kw = dict(scan_id="oracle", started_at_seconds=1., current_position=start,
              last_reading_at_seconds=prior, last_integration_seconds=.025)
    result = monotone_dp_scan_v1(mask, profile, **kw)
    oracle, count = exhaustive_oracle(mask, profile, **kw)
    assert count == 2 * 2**n
    assert abs(result.plan.duration_seconds - oracle.duration_seconds) < 1e-9
    assert result.states_evaluated == 4 * n
    assert result.plan.duration_seconds <= compile_physical_scan(mask, profile, **kw).duration_seconds + 1e-9
    audit_plan(result.plan, mask, profile, last_reading_at_seconds=prior, last_integration_seconds=.025)


def test_first_column_real_detector_wait_and_settle(profile):
    m = np.zeros((20, 32), bool)
    m[0, 0] = True
    kw = dict(scan_id="wait", started_at_seconds=1., current_position=(0., .5 / 32),
              last_reading_at_seconds=.9875, last_integration_seconds=.025)
    p = monotone_dp_scan_v1(m, profile, **kw).plan
    assert p.segments[0].kind == "wait"
    assert p.segments[0].completed_at_seconds - 1 == pytest.approx(.0125)
    audit_plan(p, m, profile, last_reading_at_seconds=.9875)
    # Settling masks recovery, but must still be charged once per full column.
    settled = replace(profile, mechanics=replace(profile.mechanics, settling_seconds_per_visit=.02))
    q = monotone_dp_scan_v1(m, settled, **kw).plan
    assert q.segments[0].kind == "settle" and all(s.kind != "wait" for s in q.segments)
    audit_plan(q, m, settled, last_reading_at_seconds=.9875)
    # Different prior exposure remains real initial state, not a fake cadence offset.
    kw.update(last_reading_at_seconds=.8, last_integration_seconds=.4)
    r = monotone_dp_scan_v1(m, profile, **kw).plan
    oracle, _ = exhaustive_oracle(m, profile, **kw)
    assert r.duration_seconds == pytest.approx(oracle.duration_seconds)
    audit_plan(r, m, profile, last_reading_at_seconds=.8, last_integration_seconds=.4)


def test_empty_mask_and_gaps(profile):
    m = np.zeros((20, 32), bool)
    assert monotone_dp_scan_v1(m, profile, scan_id="empty", started_at_seconds=0., current_position=(0., 0.)).plan.duration_seconds == 0
    m[[0, 19], 5] = True
    p = monotone_dp_scan_v1(m, profile, scan_id="gaps", started_at_seconds=0., current_position=(0., 0.)).plan
    assert len(p.requested_cells) == 2 and len(p.swept_cells) == len(p.samples) == 20
    audit_plan(p, m, profile)


@pytest.mark.parametrize("fault", ["point", "shape", "order", "direction", "initial_exposure", "nan"])
def test_invalid_requests_rejected(profile, fault):
    m = fixture_mask(2)
    cols = tuple(int(c) for c in np.flatnonzero(m.any(axis=0)))
    kw = dict(column_order=cols, directions_down=(True, False), scan_id="invalid",
              started_at_seconds=0., current_position=(0., 0.))
    if fault == "point": profile = replace(profile, scan_mode="point")
    if fault == "shape": m = m[:2]
    if fault == "order": kw["column_order"] = (cols[0], cols[0])
    if fault == "direction": kw["directions_down"] = (1, 0)
    if fault == "initial_exposure": kw["last_reading_at_seconds"] = 0.
    if fault == "nan": kw["started_at_seconds"] = float("nan")
    with pytest.raises(ValueError): compile_ordered_column_scan_v1(m, profile, **kw)


@pytest.mark.parametrize("fault", ["midpoint", "speed", "support", "duplicate", "integration"])
def test_independent_audit_rejects_physically_wrong_plans(profile, fault):
    m = fixture_mask(2)
    p = monotone_dp_scan_v1(m, profile, scan_id="bad", started_at_seconds=0., current_position=(0., 0.)).plan
    with pytest.raises(ValueError):
        if fault == "midpoint":
            s = replace(p.samples[0], position_yx=(.001, .001))
            p = replace(p, samples=(s,) + p.samples[1:])
        if fault == "speed":
            p = replace(p, completed_at_seconds=2 * p.completed_at_seconds,
                        segments=tuple(replace(s, started_at_seconds=s.started_at_seconds * 2, completed_at_seconds=s.completed_at_seconds * 2) for s in p.segments),
                        samples=tuple(replace(s, acquired_at_seconds=s.acquired_at_seconds * 2) for s in p.samples))
        if fault == "support": p = replace(p, swept_cells=p.swept_cells[:-1])
        if fault == "duplicate": p = replace(p, samples=(p.samples[0],) + p.samples)
        if fault == "integration": p = replace(p, samples=(replace(p.samples[0], integration_seconds=.01),) + p.samples[1:])
        audit_plan(p, m, profile)


def test_versioned_record_recompiles_and_tamper_rejected(profile):
    mask = fixture_mask(4)
    kw = dict(scan_id="record", started_at_seconds=0., current_position=(.5, .5))
    dp = monotone_dp_scan_v1(mask, profile, **kw)
    record = make_record(DP_COMPILER_ID, mask, profile, dp.plan, kw, order=dp.column_order, directions=dp.directions_down)
    encoded = json.loads(json.dumps(record))
    assert verify_record(encoded) == dp.plan
    for field in ("inputs", "plan", "compiler_id", "compiler_sha256"):
        bad = deepcopy(encoded)
        if field == "inputs": bad[field]["mask"][0][0] ^= 1
        elif field == "plan": bad[field]["completed_at_seconds"] += .01
        elif field == "compiler_id": bad[field] = "unversioned"
        else: bad[field]["ordered_column_scan.py"] = "0" * 64
        with pytest.raises(ValueError): verify_record(bad)
