from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pytest

from plume_focus.physical_scan import compile_profile_scan, coverage_mask
from plume_focus.sensing_messages_v2 import ScanPlanV2, ScanRequestV1
from plume_focus.sensing_session import InstrumentProfileV1, PolicyConfigV1, SensingSessionV1, configuration_hash
from plume_focus.signal_belief import SignalModelConfig
from plume_focus.controller import SensingController
from plume_focus.sensing_contracts import CausalWind, ExecutionReceipt, Footprint, SensorReading
from plume_focus.sensing_policy_v1 import plan_scan_v1, select_mass_prefix
from plume_focus.signal_belief import GaussianSignalBelief


def profile():
    return InstrumentProfileV1.from_dict(json.loads((Path(__file__).parents[1] /
        "docs/specs/instrument-profile-v1.example.json").read_text()))


def session(**policy_changes):
    policy = PolicyConfigV1("mass90_column_v1", .9, 1, 5, 1e-8, 1e-10, .02,
                            1e-12, 60., .5, .05, (.2, .1, .8, .7))
    return SensingSessionV1("beam_reconstruction_v1", profile(),
                            SignalModelConfig(grid_shape=(4, 4), signal_units="synthetic_concentration_path",
                                              diffusivity=0, process_std_per_sqrt_second=0, source_rate=0),
                            replace(policy, **policy_changes), None, "prior", (0., 0.),
                            "seconds from session start", "explicit normalized FOV wind", (1 / 16,) * 16)


def compile_fixture(mask, instrument=None, **kwargs):
    return compile_profile_scan(mask, instrument or profile(), scan_id="a", started_at_seconds=kwargs.pop("started_at_seconds", 0),
                                current_position=kwargs.pop("current_position", (0., 0.)), **kwargs)


def test_explicit_profile_and_session_roundtrip_hash_and_immutability():
    p = profile()
    assert p.digest == configuration_hash(p.to_dict())
    assert InstrumentProfileV1.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    s = session()
    payload = json.loads(json.dumps(s.to_dict()))
    assert SensingSessionV1.from_dict(payload) == s
    assert SensingSessionV1.from_dict(payload).hashes == s.hashes
    with pytest.raises(FrozenInstanceError):
        s.policy.gamma = .8
    payload["model"].pop("gain")
    with pytest.raises(ValueError, match="exactly"):
        SensingSessionV1.from_dict(payload)
    original = p.to_dict()
    original["mapping"]["units"] = "radian"
    assert p.to_dict()["mapping"]["units"] == "synthetic_fov"


def test_bound_controller_configuration_cannot_be_overridden_through_legacy_attributes():
    s = session(policy_id="static_corridor_with_reacquisition_v1")
    controller = SensingController.from_session(s)
    assert controller.policy == s.policy.policy_id
    for name, value in (("capabilities", s.instrument.capabilities()), ("model", s.model),
                        ("source_position", (.2, .3)), ("start_full_scene", False), ("policy", "rescan")):
        with pytest.raises(AttributeError, match="Frozen-v1"):
            setattr(controller, name, value)
    legacy = SensingController()
    legacy.policy = "rescan"
    assert legacy.policy == "rescan"


@pytest.mark.parametrize("mutation", [
    lambda d: d.pop("mapping"),
    lambda d: d["acquisition"].update(unknown=3),
    lambda d: d["acquisition"].update(cadence_hz=3.),
    lambda d: d["mechanics"].update(sweep_vertical_fov_per_second=.5),
    lambda d: d["mapping"].update(span_yx=[0, 1]),
    lambda d: d["footprint"].update(edge_model="unknown"),
])
def test_profile_rejects_missing_unknown_or_incompatible_fields(mutation):
    payload = profile().to_dict()
    mutation(payload)
    with pytest.raises(ValueError):
        InstrumentProfileV1.from_dict(payload)


def test_mapping_rejects_extra_coordinates_instead_of_dropping_an_axis():
    mapping = profile().mapping
    assert mapping.to_fov((.25, .75)) == (.25, .75)
    with pytest.raises(ValueError, match="exactly two"):
        mapping.to_fov((.25, .75, 1.))


def test_session_quadrature_uses_the_same_tolerance_as_mass_projection():
    with pytest.raises(ValueError, match="quadrature"):
        replace(session(), physical_quadrature_weights=((1 + 5e-10) / 16,) * 16)


def test_fixed_compiler_preserves_corners_and_charges_skipped_travel():
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = mask[3, 0] = mask[0, 3] = mask[3, 3] = True
    plan = compile_fixture(mask)
    assert plan.requested_cells == (0, 3, 12, 15)
    assert len(plan.samples) == 8
    assert plan.duration_seconds == pytest.approx(9.75)
    assert [s.completed_at_seconds - s.started_at_seconds for s in plan.segments if s.kind == "sweep"] == [4, 4]
    assert set(plan.requested_cells) <= set(plan.swept_cells)
    payload = json.loads(json.dumps(plan.to_dict()))
    assert ScanPlanV2.from_dict(payload) == plan
    assert plan.predicted_mass_recall is None
    for shape in ((120, 160), (240, 320)):
        coverage_mask(plan, shape, profile().capabilities())
    assert json.loads(json.dumps(plan.to_dict())) == payload


def test_settling_and_detector_recovery_are_charged_without_speed_change():
    p = profile()
    p = replace(p, mechanics=replace(p.mechanics, settling_seconds_per_visit=.1))
    p = replace(p, acquisition=replace(p.acquisition, detector_capacity_hz=1.))
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    plan = compile_fixture(mask, p, current_position=(0., .125), started_at_seconds=1., last_reading_at_seconds=.75)
    assert [(s.kind, round(s.completed_at_seconds - s.started_at_seconds, 6)) for s in plan.segments] == [
        ("settle", .1), ("wait", .15), ("sweep", 1.)]
    assert plan.samples[0].acquired_at_seconds == pytest.approx(1.75)
    assert plan.duration_seconds == pytest.approx(1.25)
    with pytest.raises(ValueError, match="previous actual exposure"):
        compile_fixture(mask, p, last_reading_at_seconds=1.)


def test_point_profile_has_fixed_dwell_and_no_column_closure():
    p = replace(profile(), scan_mode="point")
    mask = np.zeros((4, 4), bool)
    mask[[0, 3], 0] = True
    plan = compile_fixture(mask, p)
    assert len(plan.samples) == 2
    assert plan.swept_cells == plan.requested_cells
    assert [s.completed_at_seconds - s.started_at_seconds for s in plan.segments if s.kind == "dwell"] == [1., 1.]
    assert ScanRequestV1.from_dict(ScanRequestV1(0.).to_dict()).max_duration_seconds == 0


def evidence(plan, value=1.):
    return tuple(SensorReading(plan.scan_id, f"{plan.scan_id}-{i}", s.position_yx, s.acquired_at_seconds,
                               s.acquired_at_seconds + s.integration_seconds / 2, value,
                               s.footprint, s.integration_seconds) for i, s in enumerate(plan.samples))


def choose(belief, s=None, budget=20., *, ages=None, now=0., rotation=0):
    s = s or session()
    return plan_scan_v1(belief, s, ScanRequestV1(budget), CausalWind(error_speed=0),
                        np.full(s.instrument.lattice_shape, -1.) if ages is None else ages,
                        scan_id="fixture", now=now, pose=s.initial_position_yx, rotation=rotation,
                        startup_complete=s.startup_mode == "prior")


def test_prefix_tolerance_ties_stability_growth_translation_and_shrink():
    mass = np.array([[.35, .30, .25, .10]])
    assert select_mass_prefix(mass, .9, 1, 1e-10).tolist() == [[True, True, True, False]]
    assert np.flatnonzero(select_mass_prefix(np.full((2, 2), .25), .5, 1, 1e-10)).tolist() == [0, 1]
    b = GaussianSignalBelief(session().model)
    b.covariance[:] = 0
    b.mean[:] = 0
    b.mean[5] = 1
    initial = choose(b)
    assert choose(b).requested_cells == initial.requested_cells == (5,)
    b.mean[10] = 1
    assert choose(b).requested_cells == (5, 10)
    b.mean[5] = 0
    assert choose(b).requested_cells == (10,)
    b.mean[10], b.mean[6] = 0, 1
    assert choose(b).requested_cells == (6,)


def test_overbudget_corners_are_recorded_intact_then_explicit_fallback():
    b = GaussianSignalBelief(session().model)
    b.covariance[:] = 0
    b.mean[:] = .08 / 12
    b.mean[[0, 3, 12, 15]] = .23
    p = choose(b, budget=6)
    assert p.status == "reacquisition"
    assert p.tracking_shortfall == ("duration_budget",)
    assert all(c.requested_cells == (0, 3, 12, 15) for c in p.candidates)
    assert all(c.duration_seconds == pytest.approx(9.75) for c in p.candidates)
    assert p.duration_seconds <= 6
    assert len(p.samples) == 4
    assert p.predicted_mass_recall < .9
    good = choose(b)
    assert good.status == "target_met" and good.converged
    # Closing both columns also sweeps their four intervening low-signal cells.
    assert good.predicted_mass_recall == pytest.approx(.92 + 4 * .08 / 12)
    assert ScanPlanV2.from_dict(json.loads(json.dumps(good.to_dict()))) == good


def test_meaningful_floor_returns_null_mass_and_diagnostic_not_certified():
    b = GaussianSignalBelief(session().model)
    b.mean[:] = -1
    b.covariance[:] = 0
    p = choose(b, budget=5)
    assert p.status == "reacquisition"
    assert "no_meaningful_signal" in p.reason
    assert p.predicted_mass_recall is None and p.expected_signal_total == 0
    assert p.confidence.lower_retained_fraction is None
    assert p.confidence.status == "not_certified"


def test_fixed_corridor_executes_feasible_geometry_without_adaptive_coverage_gate():
    s = session(policy_id="static_corridor_with_reacquisition_v1")
    b = GaussianSignalBelief(s.model)
    b.covariance[:] = 0
    b.mean[:] = 0
    b.mean[0] = 1  # Outside the fixed corridor; deliberately poor coverage.
    p = choose(b, s)
    assert p.status == "static_corridor" and p.predicted_mass_recall == 0.
    assert p.requested_cells == (4, 5, 6, 8, 9, 10)
    assert not p.tracking_shortfall
    constrained = choose(b, s, budget=6.)
    assert constrained.status == "reacquisition"
    assert constrained.tracking_shortfall == ("duration_budget",)
    assert all(sum(i % 4 == col for i in constrained.swept_cells) in (0, 4) for col in range(4))


def test_age_priority_uses_oldest_row_and_rotation_for_unvisited_cells():
    s = session(policy_id="age_rescan_v1")
    b = GaussianSignalBelief(s.model)
    b.time_seconds = 20
    ages = np.full((4, 4), 10.)
    ages[:, 1] = 1
    ages[:, 2] = 9
    ages[0, 2] = 0
    p = choose(b, s, budget=5.5, ages=ages, now=20)
    assert set(c % 4 for c in p.requested_cells) == {2}
    ages[:, :] = -1
    p = choose(b, s, budget=6, ages=ages, now=20, rotation=3)
    assert set(c % 4 for c in p.requested_cells) == {3}


def test_partial_startup_counts_actual_cells_and_no_legal_action_stays_ready():
    s = replace(session(), startup_mode="full_scene")
    c = SensingController.from_session(s)
    assert c.plan_next(0, (0, 0), ScanRequestV1(0)).status == "no_legal_action"
    assert c.active_plan is None
    p = c.plan_next(0, (0, 0), ScanRequestV1(6), wind=CausalWind(error_speed=0))
    assert p.status == "reacquisition" and not p.startup_complete
    first = replace(evidence(p)[0], footprint=Footprint(1., 1.))
    c.ingest_readings([first], available_at_seconds=first.available_at_seconds)
    c.finish_scan(ExecutionReceipt(p.scan_id, p.completed_at_seconds, p.completed_at_seconds, p.final_position_yx))
    assert np.count_nonzero(c.visit_times >= 0) == 1
    assert not c.startup_complete
    p2 = c.plan_next(p.completed_at_seconds, p.final_position_yx, ScanRequestV1(20))
    assert p2.status == "startup_full_scene"
    readings = evidence(p2)
    c.ingest_readings(readings, available_at_seconds=p2.completed_at_seconds)
    c.finish_scan(ExecutionReceipt(p2.scan_id, p2.completed_at_seconds, p2.completed_at_seconds, p2.final_position_yx))
    assert c.startup_complete


def test_delayed_open_scan_replay_equivalence_and_causal_views():
    a, b = (SensingController.from_session(session()) for _ in range(2))
    pa = a.plan_next(0, (0, 0), ScanRequestV1(6), wind=CausalWind(error_speed=0))
    pb = b.plan_next(0, (0, 0), ScanRequestV1(6), wind=CausalWind(error_speed=0))
    assert pa == pb
    rs = evidence(pa)
    for r in rs:
        a.ingest_readings([r], available_at_seconds=r.available_at_seconds)
    b.ingest_readings([rs[-1]], available_at_seconds=pa.completed_at_seconds)
    b.ingest_readings(rs[:-1], available_at_seconds=pa.completed_at_seconds)
    with pytest.raises(ValueError, match="availability"):
        b.predictive_view(rs[0].acquired_at_seconds)
    with pytest.raises(RuntimeError):
        b.plan_next(pa.completed_at_seconds, pa.final_position_yx, ScanRequestV1(6))
    receipt = ExecutionReceipt(pa.scan_id, pa.completed_at_seconds + 1, pa.completed_at_seconds + 2, pa.final_position_yx)
    va, vb = a.finish_scan(receipt), b.finish_scan(receipt)
    np.testing.assert_allclose(va.mean, vb.mean, atol=1e-12)
    np.testing.assert_allclose(va.std, vb.std, atol=1e-12)
    assert b.ingest_readings(rs, available_at_seconds=receipt.received_at_seconds) == ()
    with pytest.raises(ValueError, match="Conflicting"):
        b.ingest_readings([replace(rs[0], value=7)], available_at_seconds=receipt.received_at_seconds)
    with pytest.raises(ValueError, match="open scan"):
        b.ingest_readings([replace(rs[0], reading_id="new")], available_at_seconds=receipt.received_at_seconds)


def test_cross_scan_capacity_and_exposure_overlap_are_rejected_atomically():
    c = SensingController.from_session(session())
    p = c.plan_next(0, (0, 0), ScanRequestV1(6))
    first = evidence(p)[0]
    c.ingest_readings([first], available_at_seconds=first.available_at_seconds)
    end = first.available_at_seconds
    c.finish_scan(ExecutionReceipt(p.scan_id, end, end, first.position_yx))
    second_plan = c.plan_next(end, first.position_yx, ScanRequestV1(6))
    invalid = SensorReading(second_plan.scan_id, "too-fast", first.position_yx, end + .1, end + .2,
                            1., first.footprint, .1)
    before = c.belief_view()
    with pytest.raises(ValueError, match="cadence"):
        c.ingest_readings([invalid], available_at_seconds=end + .2)
    np.testing.assert_array_equal(c.belief_view().mean, before.mean)
    assert second_plan.samples[0].acquired_at_seconds - first.acquired_at_seconds >= .5 - 1e-10


def test_forecast_cycle_and_valid_nonconverged_candidate_keep_diagnostics(monkeypatch):
    original = GaussianSignalBelief.expected_mass_on

    def cycle(self, shape, weights, mass_floor):
        self.mean[:] = 0
        self.covariance[:] = 0
        self.mean[3 if abs(self.time_seconds - 1.25) < 1e-8 else 0] = 1
        return original(self, shape, weights, mass_floor)

    monkeypatch.setattr(GaussianSignalBelief, "expected_mass_on", cycle)
    p = choose(GaussianSignalBelief(session().model))
    assert p.termination == "cycle" and p.status == "reacquisition"
    assert len(p.candidates) == 3

    def changing_prefix(self, shape, weights, mass_floor):
        self.mean[:] = 0
        self.covariance[:] = 0
        if self.time_seconds == 20:
            self.mean[[0, 12]] = 1
        else:
            self.mean[4] = 1
        return original(self, shape, weights, mass_floor)

    monkeypatch.setattr(GaussianSignalBelief, "expected_mass_on", changing_prefix)
    s = session(max_planning_iterations=1)
    p = choose(GaussianSignalBelief(s.model), s)
    assert p.termination == "iteration_limit" and p.status == "target_met"
    assert not p.converged and p.predicted_mass_recall == pytest.approx(1.)


def test_narrow_profile_full_scene_cost_and_localized_prior_tracking():
    from plume_focus.sensing_execution_v1 import SensingRunConfigV1
    path = Path(__file__).parents[1] / "configs/sensing-v1/narrow.json"
    s = SensingRunConfigV1.from_dict(json.loads(path.read_text())).session
    full = compile_profile_scan(np.ones(s.instrument.lattice_shape, bool), s.instrument,
                                 scan_id="full", started_at_seconds=0, current_position=(0, 0))
    assert full.duration_seconds == pytest.approx(3521.5975)
    assert len(full.samples) == 70400
    assert all(seg.completed_at_seconds - seg.started_at_seconds == pytest.approx(11.) for seg in full.segments if seg.kind == "sweep")
    s = replace(s, startup_mode="prior", source_position=None,
                model=replace(s.model, diffusivity=0., process_std_per_sqrt_second=0., source_rate=0.))
    belief = GaussianSignalBelief(s.model)
    belief.mean[:] = 0
    belief.covariance[:] = 0
    belief.mean.reshape(12, 16)[6, 8] = 1
    plan = choose(belief, s, budget=100)
    assert plan.status == "target_met" and plan.predicted_mass_recall >= .9 - 1e-10
    assert all(sample.footprint.width == 1 / 320 and sample.integration_seconds == .025 for sample in plan.samples)


def test_production_dynamic_filter_matches_independent_joint_batch():
    model = SignalModelConfig(grid_shape=(3, 4), diffusivity=0., process_std_per_sqrt_second=.02, source_rate=0.)
    belief = GaussianSignalBelief(model)
    prior = belief.covariance.copy()
    wind = CausalWind(velocity_yx=(0., .125), error_speed=0.)
    # One 0.5 s bilinear advection step: quarter-cell displacement; at the
    # boundary the frozen interpolation clamps to the nearest grid center.
    a = np.eye(12)
    for row in range(3):
        for col in range(1, 4):
            i = row * 4 + col
            a[i, i], a[i, i - 1] = .75, .25
    q = .5 * .02**2 * belief.kernel
    p1 = a @ prior @ a.T + q
    p2 = a @ p1 @ a.T + q
    cross = a @ p1
    from plume_focus.signal_belief import observation_weights
    h1 = observation_weights((3, 4), (.4, .4), Footprint(.1, .1))
    h2 = observation_weights((3, 4), (.6, .7), Footprint(.1, .1))
    r = model.sensor_noise_std**2
    yy = np.array([[h1 @ p1 @ h1 + r, h1 @ cross.T @ h2], [h2 @ cross @ h1, h2 @ p2 @ h2 + r]])
    xy = np.column_stack((cross @ h1, p2 @ h2))
    expected_mean = xy @ np.linalg.solve(yy, [1., 0.])
    expected_cov = p2 - xy @ np.linalg.solve(yy, xy.T)
    for i, (t, pos, value) in enumerate(((.5, (.4, .4), 1.), (1., (.6, .7), 0.))):
        reading = SensorReading("s", str(i), pos, t, t + .01, value, Footprint(.1, .1), .02)
        belief.assimilate(reading, wind)
    np.testing.assert_allclose(belief.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(belief.covariance, expected_cov, atol=1e-12)
