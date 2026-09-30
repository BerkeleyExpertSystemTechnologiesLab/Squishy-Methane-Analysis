import ast
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from plume_focus.controller import SensingController
from plume_focus.physical_scan import compile_scan, coverage_mask
from plume_focus.sensing_contracts import (
    CausalWind, ExecutionReceipt, Footprint, ScanConstraints, ScanPlan, ScannerCapabilities, SensorReading,
)
from plume_focus.signal_belief import (
    GaussianSignalBelief, SignalModelConfig, observation_weights, positive_expectation,
)


def small_model(**kwargs):
    return SignalModelConfig(grid_shape=(3, 4), diffusivity=0, process_std_per_sqrt_second=0,
                             source_rate=0, **kwargs)


def small_caps(mode="column"):
    return ScannerCapabilities(lattice_shape=(4, 6), readout_hz=10, integration_seconds=.02,
                               footprint=Footprint(.25, 1 / 6), scan_mode=mode)


def make_controller():
    return SensingController(small_caps(), small_model())


def sample_readings(plan, *, batch=False):
    return tuple(SensorReading(plan.scan_id, f"{plan.scan_id}:{i}", s.position_yx, s.acquired_at_seconds,
                               plan.completed_at_seconds if batch else s.acquired_at_seconds + s.integration_seconds / 2,
                               float((i % 3) / 2), s.footprint, s.integration_seconds)
                 for i, s in enumerate(plan.samples))


def open_scan(controller):
    return controller.plan_next(0, (0, 0), ScanConstraints(20), wind=CausalWind(error_speed=0))


def test_worked_correlated_scalar_example_and_uncertainty():
    belief = GaussianSignalBelief(small_model(initial_mean=.2))
    belief.covariance[:] = np.eye(12) * .04
    belief.covariance[0, 1] = belief.covariance[1, 0] = .02
    h = np.eye(12)[0]
    belief.condition(.8, h, .01)
    np.testing.assert_allclose(belief.mean[:3], [.68, .44, .2], atol=1e-12)
    np.testing.assert_allclose(np.diag(belief.covariance)[:3], [.008, .032, .04], atol=1e-12)
    assert np.linalg.eigvalsh(belief.covariance).min() > 0


def test_sequential_matches_independent_batch_conditioning():
    belief = GaussianSignalBelief(small_model())
    mean, cov = belief.mean.copy(), belief.covariance.copy()
    hs = np.array([observation_weights((3, 4), pos, Footprint(.2, .2)) for pos in ((.3, .3), (.3, .3), (.65, .7))])
    ys = np.array([.8, 0, .3])
    r = .0025
    expected_mean = mean + cov @ hs.T @ np.linalg.solve(hs @ cov @ hs.T + np.eye(3) * r, ys - hs @ mean)
    expected_cov = cov - cov @ hs.T @ np.linalg.solve(hs @ cov @ hs.T + np.eye(3) * r, hs @ cov)
    for h, y in zip(hs, ys):
        belief.condition(y, h, r)
    np.testing.assert_allclose(belief.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(belief.covariance, expected_cov, atol=1e-12)


def test_reading_footprint_is_one_observation_and_zero_is_evidence():
    belief = GaussianSignalBelief(small_model(initial_mean=.5))
    h = observation_weights((3, 4), (.5, .5), Footprint(.8, .8))
    assert np.count_nonzero(h) > 1
    prior_variance = float(h @ belief.covariance @ h)
    belief.condition(0, h, .01)
    assert 0 < h @ belief.mean < .5
    assert h @ belief.covariance @ h == pytest.approx(prior_variance * .01 / (prior_variance + .01))


def test_positive_expected_mass_is_separate_from_signal_mean():
    np.testing.assert_allclose(positive_expectation(np.array([0., -1., 2.]), np.array([1., 0., 0.])),
                               [1 / np.sqrt(2 * np.pi), 0, 2])
    belief = GaussianSignalBelief(small_model())
    assert belief.mean.sum() == 0
    assert belief.mass_on((12, 16)).sum() == pytest.approx(1)


def test_advection_moves_field_and_covariance_without_wraparound():
    belief = GaussianSignalBelief(small_model())
    belief.mean[:] = 0
    belief.mean.reshape(3, 4)[1, 1] = 1
    belief.advance(1, CausalWind(velocity_yx=(0, .25), error_speed=0))
    assert np.argmax(belief.mean.reshape(3, 4)[1]) == 2
    assert np.linalg.eigvalsh(belief.covariance).min() >= -1e-12
    belief.advance(10, CausalWind(velocity_yx=(0, .25), error_speed=0))
    assert belief.mean.sum() < .02


def test_source_input_only_when_explicit_and_process_uncertainty_grows():
    model = SignalModelConfig(grid_shape=(3, 4), diffusivity=0, source_sigma=.2)
    free, anchored = GaussianSignalBelief(model), GaussianSignalBelief(model)
    wind = CausalWind(error_speed=0)
    free.advance(1, wind)
    anchored.advance(1, wind, (.5, .25))
    assert free.mean.sum() == 0
    assert anchored.mean.sum() > 0
    assert np.diag(free.covariance).mean() > model.initial_std**2


@pytest.mark.parametrize("mode", ["point", "column"])
def test_physical_paths_charge_travel_and_limit_readout(mode):
    caps = small_caps(mode)
    mask = np.ones(caps.lattice_shape, dtype=bool)
    plan = compile_scan(mask, caps, scan_id="test", started_at_seconds=3, current_position=(0, 0))
    assert plan.segments[0].kind == "move"
    assert sum(s.completed_at_seconds - s.started_at_seconds for s in plan.segments) == pytest.approx(plan.duration_seconds)
    assert np.diff([s.acquired_at_seconds for s in plan.samples]).min() >= 1 / caps.readout_hz - 1e-10
    assert len(plan.samples) == np.prod(caps.lattice_shape)
    narrow = mask.copy()
    narrow[:, :] = False
    narrow[:, [0, 1]] = True
    skipped = narrow.copy()
    skipped[:, 1] = False
    skipped[:, -1] = True
    a = compile_scan(narrow, caps, scan_id="a", started_at_seconds=0, current_position=(0, 0))
    b = compile_scan(skipped, caps, scan_id="b", started_at_seconds=0, current_position=(0, 0))
    assert b.duration_seconds > a.duration_seconds


def test_display_resolution_does_not_change_plan_and_physical_area_converges():
    caps = small_caps()
    mask = np.zeros(caps.lattice_shape, bool)
    mask[1:3, 2:4] = True
    plan = compile_scan(mask, caps, scan_id="a", started_at_seconds=0, current_position=(0, 0))
    serialized = plan.to_dict()
    areas = [coverage_mask(plan, shape, caps).mean() for shape in ((120, 160), (240, 320))]
    assert areas[0] == pytest.approx(areas[1], abs=.006)
    assert plan.to_dict() == serialized


def test_stream_and_delayed_batch_have_equal_completion_posteriors():
    stream, batch = make_controller(), make_controller()
    p1, p2 = open_scan(stream), open_scan(batch)
    for r in sample_readings(p1):
        stream.ingest_readings([r], available_at_seconds=r.available_at_seconds)
    batch.ingest_readings(sample_readings(p2, batch=True), available_at_seconds=p2.completed_at_seconds)
    v1 = stream.finish_scan(ExecutionReceipt(p1.scan_id, p1.completed_at_seconds, p1.completed_at_seconds, p1.final_position_yx))
    v2 = batch.finish_scan(ExecutionReceipt(p2.scan_id, p2.completed_at_seconds, p2.completed_at_seconds, p2.final_position_yx))
    np.testing.assert_allclose(v1.mean, v2.mean, atol=1e-12)
    np.testing.assert_allclose(v1.std, v2.std, atol=1e-12)
    assert all(e.available_at_seconds == p2.completed_at_seconds for e in batch.assimilation_events)


def test_duplicate_conflict_future_reading_and_path_lock():
    controller = make_controller()
    plan = open_scan(controller)
    reading = sample_readings(plan)[0]
    with pytest.raises(RuntimeError):
        controller.plan_next(1, (0, 0))
    with pytest.raises(ValueError, match="not yet available"):
        controller.ingest_readings([reading], available_at_seconds=0)
    controller.ingest_readings([reading], available_at_seconds=reading.available_at_seconds)
    before = controller.belief_view()
    assert controller.ingest_readings([reading], available_at_seconds=reading.available_at_seconds) == ()
    with pytest.raises(ValueError, match="Conflicting"):
        controller.ingest_readings([replace(reading, value=4)], available_at_seconds=reading.available_at_seconds)
    np.testing.assert_array_equal(before.mean, controller.belief_view().mean)


def test_delivered_out_of_order_readings_replay_within_open_scan():
    a, b = make_controller(), make_controller()
    pa, pb = open_scan(a), open_scan(b)
    rs = sample_readings(pa, batch=True)[:3]
    a.ingest_readings([rs[2]], available_at_seconds=pa.completed_at_seconds)
    a.ingest_readings(rs[:2], available_at_seconds=pa.completed_at_seconds)
    b.ingest_readings(rs, available_at_seconds=pb.completed_at_seconds)
    np.testing.assert_allclose(a.belief_view().mean, b.belief_view().mean, atol=1e-12)


def test_actual_completion_overrun_controls_next_start_and_pose():
    c = make_controller()
    p = open_scan(c)
    rs = sample_readings(p)
    c.ingest_readings(rs, available_at_seconds=p.completed_at_seconds)
    actual_end = p.completed_at_seconds + 2
    c.finish_scan(ExecutionReceipt(p.scan_id, actual_end, actual_end + .5, (.4, .6)))
    with pytest.raises(ValueError):
        c.plan_next(p.completed_at_seconds, (.4, .6))
    with pytest.raises(ValueError, match="position"):
        c.plan_next(actual_end + .5, (0, 0))
    next_plan = c.plan_next(actual_end + .5, (.4, .6), ScanConstraints(4))
    assert next_plan.started_at_seconds == actual_end + .5
    assert next_plan.start_position_yx == (.4, .6)


def test_empty_delivery_still_advances_the_availability_clock():
    c = make_controller()
    p = open_scan(c)
    c.ingest_readings([], available_at_seconds=1)
    with pytest.raises(ValueError, match="chronological"):
        c.ingest_readings([], available_at_seconds=.5)
    with pytest.raises(ValueError, match="precedes delivered evidence"):
        c.finish_scan(ExecutionReceipt(p.scan_id, .5, .5, (0, 0)))


def test_budgeted_reacquisition_visits_new_columns_and_never_exceeds_budget():
    c = make_controller()
    pose, now = (0., 0.), 0.
    visited = []
    for _ in range(3):
        p = c.plan_next(now, pose, ScanConstraints(2, minimum_samples=2), wind=CausalWind(error_speed=1))
        assert p.duration_seconds <= 2 + 1e-10
        assert p.status == "reacquisition"
        visited.append(set(cell % 6 for cell in p.swept_cells))
        c.ingest_readings(sample_readings(p), available_at_seconds=p.completed_at_seconds)
        c.finish_scan(ExecutionReceipt(p.scan_id, p.completed_at_seconds, p.completed_at_seconds, p.final_position_yx))
        now, pose = p.completed_at_seconds, p.final_position_yx
    assert len(set.union(*visited)) > len(visited[0])


def test_impossible_budget_and_future_wind_are_explicit():
    c = make_controller()
    with pytest.raises(ValueError, match="Future wind"):
        c.plan_next(0, (0, 0), wind=CausalWind(1, 1))
    p = c.plan_next(0, (0, 0), ScanConstraints(.001))
    assert p.status == "infeasible"
    assert not p.samples
    assert c.active_plan is None


def test_public_messages_json_round_trip_and_reject_unknown_fields():
    c = make_controller()
    p = open_scan(c)
    messages = [c.capabilities, c.model, p, sample_readings(p)[0], CausalWind(), ScanConstraints(),
                ExecutionReceipt(p.scan_id, p.completed_at_seconds, p.completed_at_seconds, p.final_position_yx)]
    for message in messages:
        payload = json.loads(json.dumps(message.to_dict(), allow_nan=False))
        assert type(message).from_dict(payload) == message
        with pytest.raises(ValueError):
            type(message).from_dict({**payload, "unknown": 3})
        with pytest.raises(ValueError):
            type(message).from_dict({**payload, "schema_version": 999})


def test_controller_dependency_boundary():
    root = Path(__file__).parents[1] / "src" / "plume_focus"
    modules = {"calibrated_scanner", "acquisition", "sensing_compat", "full_fov_loop", "controller", "sensing_policy", "physical_scan", "signal_belief", "sensing_contracts",
               "sensing_session", "sensing_messages_v2", "sensing_policy_v1", "sensing_session_v2", "sensing_policy_v2",
               "sensing_session_v3", "motion_config", "motion_belief", "signal_distribution",
               "sensing_session_v4", "sensing_messages_v3", "sensing_risk", "sensing_policy_v4",
               "sensing_snapshot", "sensing_session_diagnostic", "sensing_session_v5", "sensing_messages_v4", "sensing_policy_v5",
               "evidence_gate", "evidence_gated_loop", "evidence_gated_execution"}
    allowed = {
        "calibrated_scanner": {"full_fov_loop", "sensing_session", "sensing_contracts", "signal_belief"},
        "acquisition": {"sensing_contracts", "signal_belief", "signal_distribution"},
        "sensing_compat": {"sensing_session_v2", "sensing_session_v4", "sensing_session_v5", "sensing_policy_v1", "sensing_policy_v2", "sensing_policy_v4", "sensing_policy_v5"},
        "full_fov_loop": {"acquisition", "physical_scan", "sensing_contracts", "sensing_session", "signal_belief", "signal_distribution", "sensing_messages_v2"},
        "sensing_contracts": set(), "signal_belief": {"sensing_contracts"},
        "sensing_session": {"signal_belief", "sensing_contracts"},
        "sensing_session_v2": {"sensing_contracts", "sensing_session"},
        "sensing_session_v3": {"sensing_session", "sensing_session_v2", "motion_config"},
        "sensing_session_v4": {"motion_config", "sensing_contracts", "sensing_session", "sensing_session_v2"},
        "sensing_messages_v3": {"sensing_contracts", "sensing_messages_v2"},
        "sensing_messages_v4": {"sensing_contracts", "sensing_messages_v3"},
        "sensing_session_v5": {"sensing_session_v4", "sensing_session_v2", "sensing_session", "motion_config"},
        "sensing_policy_v5": {"signal_distribution", "sensing_messages_v4", "sensing_policy_v2", "sensing_policy_v4", "sensing_risk", "sensing_session_v5"},
        "sensing_risk": {"signal_belief"},
        "evidence_gate": {"sensing_contracts", "sensing_session"},
        "evidence_gated_loop": {"calibrated_scanner", "evidence_gate", "full_fov_loop", "sensing_contracts",
                                "sensing_session", "signal_belief"},
        "evidence_gated_execution": {"calibration_execution", "evidence_gated_loop", "sensing_contracts",
                                     "sensing_session"},
        "sensing_session_diagnostic": {"sensing_session_v4", "sensing_session", "signal_belief"},
        "sensing_snapshot": {"sensing_contracts", "sensing_session", "sensing_session_v2", "sensing_session_v3",
                             "sensing_session_v4", "sensing_session_diagnostic", "sensing_session_v5"},
        "sensing_policy_v4": {"physical_scan", "sensing_messages_v3", "sensing_policy_v2", "sensing_risk", "signal_belief"},
        "motion_config": {"sensing_contracts"},
        "motion_belief": {"motion_config", "signal_belief", "signal_distribution"},
        "signal_distribution": {"signal_belief"},
        "sensing_messages_v2": {"sensing_contracts"},
        "physical_scan": {"sensing_contracts", "sensing_messages_v2", "sensing_session"},
        "sensing_policy": {"physical_scan", "sensing_contracts", "signal_belief"},
        "sensing_policy_v1": {"physical_scan", "sensing_contracts", "sensing_messages_v2", "sensing_session", "signal_belief"},
        "sensing_policy_v2": {"physical_scan", "sensing_messages_v2", "sensing_policy_v1", "signal_belief"},
        "controller": modules - {"controller"},
    }
    for name in modules:
        tree = ast.parse((root / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("plume_focus."):
                assert node.module.split(".")[-1] in allowed[name]
            if isinstance(node, ast.Import):
                assert not any(alias.name in {"pathlib", "cv2", "os", "requests"} for alias in node.names)
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open" for n in ast.walk(tree))
