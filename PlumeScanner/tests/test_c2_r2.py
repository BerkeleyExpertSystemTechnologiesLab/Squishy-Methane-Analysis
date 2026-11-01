from copy import deepcopy
from dataclasses import replace
from itertools import permutations, product
import json
from pathlib import Path

import numpy as np
import pytest

from plume_focus.c2_r2_mechanics import (compile_directed_columns, solve_subset_columns, solve_monotone_columns,
    mechanical_oracle, audit_directed_plan, post_scan_terminal, route_record, verify_route_record)
from plume_focus.c2_r2_dynamic import (R2Counterfactual, ready_payload, state_hash, integration_grid,
    covariance_rollout, R2TruthGrid, execute_counterfactual, schedule)
from plume_focus.evidence_gated_execution import EvidenceGatedRunConfig
from plume_focus.physical_scan import compile_physical_scan
from plume_focus.c2_scan_validation import baseline_order
from plume_focus.calibrated_scanner import DecayGaussianBelief
from plume_focus.sensing_contracts import CausalWind
from plume_focus.sensing_simulation import FakeScanner


@pytest.fixture
def cfg():
    return EvidenceGatedRunConfig.from_dict(json.loads((Path(__file__).parents[1] / "configs/evidence-gated-focus/coarse-demo.json").read_text()))


def mask_fixture(n):
    rng = np.random.default_rng(731 + n)
    m = np.zeros((20, 32), bool)
    for c in rng.choice(32, n, replace=False): m[rng.choice(20, 3, replace=False), c] = True
    return m


@pytest.mark.parametrize("n", [1, 3, 5, 6])
@pytest.mark.parametrize("pos", [(0., 0.), (.5, .5), (1., 1.)])
def test_restricted_parity_and_independent_cost_oracle(cfg, n, pos):
    m = mask_fixture(n); p = cfg.session.instrument
    kw = dict(scan_id="test", started_at_seconds=1., current_position=pos, last_reading_at_seconds=.9875, last_integration_seconds=.025)
    base = compile_physical_scan(m, p, **kw)
    order, down = baseline_order(base)
    assert compile_directed_columns(m, p, column_order=order, directions_down=down, **kw) == base
    oracle = mechanical_oracle(m, p, **kw)
    for monotone in (False, True):
        for objective in ("FREE", "HOME"):
            r = solve_subset_columns(m, p, monotone=monotone, objective=objective, **kw)
            ref = oracle["winners"][("monotone" if monotone else "all") + "_" + objective]
            assert r.objective_seconds == pytest.approx(ref["cost"], abs=1e-9)
            audit_directed_plan(r.plan, m, p, last_reading_at_seconds=.9875, last_integration_seconds=.025)
            if monotone:
                assert solve_monotone_columns(m, p, objective=objective, **kw).objective_seconds == pytest.approx(r.objective_seconds)


def test_oracle_against_direct_compiled_all_paths_initial_wait(cfg):
    p = cfg.session.instrument
    m = mask_fixture(3); cs = tuple(int(c) for c in np.flatnonzero(m.any(axis=0)))
    kw = dict(scan_id="brute", started_at_seconds=1., current_position=(0., (cs[0] + .5) / 32),
              last_reading_at_seconds=.9875, last_integration_seconds=.025)
    free, home = float("inf"), float("inf")
    for order in permutations(cs):
        for d in product((True, False), repeat=3):
            q = compile_directed_columns(m, p, column_order=order, directions_down=d, **kw)
            free = min(free, q.duration_seconds)
            home = min(home, q.duration_seconds + post_scan_terminal(q, p)["duration_seconds"])
    ref = mechanical_oracle(m, p, **kw)
    assert free == pytest.approx(ref["winners"]["all_FREE"]["cost"])
    assert home == pytest.approx(ref["winners"]["all_HOME"]["cost"])
    one = np.zeros_like(m); one[0, 0] = True
    kw["current_position"] = (0., .5 / 32)
    q = solve_subset_columns(one, p, **kw).plan
    assert q.segments[0].kind == "wait" and q.segments[0].completed_at_seconds == pytest.approx(1.0125)


def test_limits_record_and_terminal(cfg):
    p = cfg.session.instrument; m = mask_fixture(3)
    kw = dict(scan_id="rec", started_at_seconds=0., current_position=(0., 0.))
    r = solve_subset_columns(m, p, objective="HOME", **kw)
    rec = route_record(r.plan, m, p, kw, r.order, r.down)
    assert verify_route_record(json.loads(json.dumps(rec))) == r.plan
    assert rec["terminal"]["final_position_yx"] == [0., 0.]
    assert all(s["kind"] == "move" for s in rec["terminal"]["segments"])
    broken = deepcopy(rec); broken["terminal"]["duration_seconds"] += .1
    with pytest.raises(ValueError): verify_route_record(broken)
    with pytest.raises(ValueError, match="at most 10"): solve_subset_columns(mask_fixture(11), p, **kw)
    with pytest.raises(ValueError): mechanical_oracle(mask_fixture(7), p, **kw)
    with pytest.raises(ValueError): compile_directed_columns(m, p, column_order=(r.order[0],) * 3, directions_down=(True,) * 3, **kw)


class Constant:
    end_time_seconds = None
    def signal_at(self, times, positions_yx): return np.full(len(times), .2)


def small_ready(cfg, static=False):
    model = replace(cfg.model, grid_shape=(2, 3), diffusivity=0., source_rate=0.,
                    process_std_per_sqrt_second=0. if static else .01)
    session = replace(cfg.session, model=model, source_position=None, half_life_seconds=None if static else 6.)
    config = replace(cfg, session=session)
    c = config.make_controller()
    return ready_payload(c, provenance={"test_fixture": True})


def small_record(cfg, saved, order=(2, 5), down=(True, False)):
    m = np.zeros((20, 32), bool); m[2:5, 2] = True; m[6:9, 5] = True
    kw = dict(scan_id="dyn", started_at_seconds=0., current_position=(0., 0.),
              last_reading_at_seconds=None, last_integration_seconds=None)
    p = compile_directed_columns(m, cfg.session.instrument, column_order=order, directions_down=down, **kw)
    return route_record(p, m, cfg.session.instrument, kw, order, down)


@pytest.mark.parametrize("delivery", ["stream", "batch"])
def test_covariance_runtime_parity_receipt_and_state_immutability(cfg, delivery):
    saved = small_ready(cfg); rec = small_record(cfg, saved)
    c = R2Counterfactual(saved); before = state_hash(c)
    p = verify_route_record(rec); ats, receipt = schedule(p, c.session, delivery)
    times, weights = integration_grid(0., receipt + 2, .05, [*ats, receipt])
    risk = covariance_rollout(saved, rec, times, weights, delivery=delivery, shape=(8, 10))
    truth = R2TruthGrid(Constant(), times, weights, c.model.grid_shape, shape=(8, 10))
    actual = execute_counterfactual(saved, rec, Constant(), truth, actual_noise_std=.05, seed=17, delivery=delivery, save_trace=True)
    np.testing.assert_allclose(risk["receipt_covariance"], actual["receipt_covariance"], atol=1e-13, rtol=0)
    for a, b in zip(risk["trace"], actual["events"]): assert a["covariance_sha256"] == b["covariance_sha256"]
    assert state_hash(c) == before
    # Future noise is not an input to scoring and cannot alter the frozen score.
    other = execute_counterfactual(saved, rec, Constant(), truth, actual_noise_std=.05, seed=18, delivery=delivery)
    assert other["L"] != actual["L"]
    assert covariance_rollout(saved, rec, times, weights, delivery=delivery, shape=(8, 10))["U"] == risk["U"]


def test_availability_and_new_record_guard(cfg):
    saved = small_ready(cfg); rec = small_record(cfg, saved)
    c = R2Counterfactual(saved); p = c.begin_record(rec)
    r = FakeScanner(Constant(), noise_std=0).execute(p).readings[0]
    before = state_hash(c)
    with pytest.raises(ValueError, match="not yet available"): c.ingest_readings([r], available_at_seconds=r.available_at_seconds - .001)
    assert before == state_hash(c)
    c.ingest_readings([r], available_at_seconds=r.available_at_seconds)
    after = state_hash(c)
    c.ingest_readings([r], available_at_seconds=r.available_at_seconds)
    assert after == state_hash(c)
    with pytest.raises(ValueError, match="active"): c.begin_record(rec)
    bad = deepcopy(rec); bad["column_order"] = [5, 2]
    with pytest.raises(ValueError): R2Counterfactual(saved).begin_record(bad)


def test_static_covariance_order_invariance_and_half_life(cfg):
    saved = small_ready(cfg, static=True)
    recs = [small_record(cfg, saved), small_record(cfg, saved, order=(5, 2), down=(False, True))]
    covs = []
    for rec in recs:
        times, weights = integration_grid(0., 4., .05)
        covs.append(covariance_rollout(saved, rec, times, weights, shape=(8, 10))["receipt_covariance"])
    np.testing.assert_allclose(*covs, atol=1e-13, rtol=0)
    model = replace(cfg.model, grid_shape=(2, 3), initial_mean=1., diffusivity=0., process_std_per_sqrt_second=0.)
    b = DecayGaussianBelief(model, half_life_seconds=6., transport_id="zero_inflow_v1")
    cov = b.covariance.copy(); b.advance(6., CausalWind())
    np.testing.assert_allclose(b.mean, .5, atol=1e-14)
    np.testing.assert_allclose(b.covariance, cov / 4, atol=1e-14)
    decayed = small_ready(cfg); static = deepcopy(decayed)
    static["half_life_seconds"] = None; static["session"]["half_life_seconds"] = None
    rec = small_record(cfg, decayed)
    # The profile/route are unchanged; only the model's explicit decay changes.
    u1 = covariance_rollout(decayed, rec, times, weights, shape=(8, 10))["U"]
    u2 = covariance_rollout(static, rec, times, weights, shape=(8, 10))["U"]
    assert u1 < u2


def test_dynamic_resampling_uses_actual_trajectory_and_time(cfg):
    saved = small_ready(cfg)
    class Moving:
        end_time_seconds = None
        def signal_at(self, times, positions_yx): return np.asarray(times) + positions_yx[:, 0]**2 + positions_yx[:, 1]
    a = verify_route_record(small_record(cfg, saved))
    b = verify_route_record(small_record(cfg, saved, order=(5, 2), down=(False, True)))
    scanner = FakeScanner(Moving(), noise_std=0.)
    ra, rb = scanner.execute(a).readings, scanner.execute(b).readings
    by_cell = lambda readings: {r.position_yx: r.value for r in readings}
    da, db = by_cell(ra), by_cell(rb)
    assert da.keys() == db.keys() and any(da[k] != db[k] for k in da)


def test_read_only_mean_forecast_matches_current_runtime(cfg):
    from plume_focus.calibration_evaluation import mean_forecast
    saved = small_ready(cfg); c = R2Counterfactual(saved)
    c._belief.mean[:] = np.linspace(.1, .7, len(c._belief.mean))
    before = state_hash(c)
    for t in (.001, .073, .6, 2.37, 6.1):
        actual = c._belief.clone(); actual.advance(t, c._wind, c.source_position)
        predicted = mean_forecast(c._belief, c._belief.mean, t, c._wind, c.source_position)
        np.testing.assert_allclose(predicted, actual.mean, atol=1e-12, rtol=0)
    assert state_hash(c) == before


def test_query_grid_does_not_change_persistent_receipt_covariance(cfg):
    saved = small_ready(cfg); rec = small_record(cfg, saved)
    outputs = []
    for step in (.1, .025):
        times, weights = integration_grid(0., 4., step)
        outputs.append(covariance_rollout(saved, rec, times, weights, shape=(8, 10)))
    np.testing.assert_array_equal(outputs[0]["receipt_covariance"], outputs[1]["receipt_covariance"])
    assert outputs[0]["receipt_state_hash"] == outputs[1]["receipt_state_hash"]


def test_declared_ten_column_limit_against_geometric_lower_bound(cfg):
    # A known optimum at the supported boundary, not an n! enumeration or a
    # new environmental scenario: ten equal intervals alternate without gaps.
    p = cfg.session.instrument; mask = np.zeros((20, 32), bool); mask[3:10, :10] = True
    kw = dict(scan_id="ten-column-boundary-test", started_at_seconds=0., current_position=(0., 0.))
    sweep = 10 * 7 / 20
    xmax = 9.5 / 32
    free_lower_bound = sweep + xmax / p.mechanics.move_horizontal_fov_per_second + 3 / 20
    home_lower_bound = sweep + 2 * xmax / p.mechanics.move_horizontal_fov_per_second + 2 * 3 / 20
    free = solve_subset_columns(mask, p, objective="FREE", **kw)
    home = solve_subset_columns(mask, p, objective="HOME", **kw)
    assert free.objective_seconds == pytest.approx(free_lower_bound, abs=1e-9)
    assert home.objective_seconds == pytest.approx(home_lower_bound, abs=1e-9)
