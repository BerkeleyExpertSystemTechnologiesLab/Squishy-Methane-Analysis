from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from plume_focus.controller import SensingController
from plume_focus.physical_scan import compile_profile_scan
from plume_focus.sensing_cli_v1 import load_config, config_from_dict
from plume_focus.sensing_contracts import CausalWind, ExecutionReceipt
from plume_focus.sensing_execution_v1 import CausalReplayV1, SensingRunConfigV1, run_sensing_v1
from plume_focus.sensing_execution_v2 import SensingRunConfigV2
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_policy_v2 import plan_scan_v2, survey_blocks
from plume_focus.sensing_reconstruction import reconstruction_loss, summarize_v1
from plume_focus.sensing_session import SensingSessionV1
from plume_focus.signal_belief import GaussianSignalBelief, positive_expectation


def config(name="coarse-full-startup"):
    return load_config(Path(__file__).parents[1] / f"configs/sensing-v2/{name}.json")


def choose(belief, session, budget=12., now=0., last_survey=0., ages=None, pose=(0., 0.)):
    return plan_scan_v2(belief, session, ScanRequestV1(budget), CausalWind(),
        np.full(session.instrument.lattice_shape, -1.) if ages is None else ages,
        scan_id="fixture", now=now, pose=pose, rotation=0, startup_complete=True,
        last_survey_completed=last_survey)


def localized():
    s = config().session
    s = replace(s, source_position=None, startup_mode="prior", model=replace(s.model,
                diffusivity=0., process_std_per_sqrt_second=0., source_rate=0.))
    b = GaussianSignalBelief(s.model)
    b.mean[:] = 0.
    b.covariance[:] = 0.
    b.mean.reshape(s.model.grid_shape)[5:7, 6:8] = 1.
    return s, b


def test_v2_roundtrip_and_v1_rejection_preserve_version_boundary():
    c = config()
    assert isinstance(c, SensingRunConfigV2)
    assert config_from_dict(json.loads(json.dumps(c.to_dict()))) == c
    assert config_from_dict(c.to_dict()).session.hashes == c.session.hashes
    with pytest.raises(ValueError):
        SensingRunConfigV1.from_dict(c.to_dict())
    with pytest.raises(ValueError):
        SensingSessionV1.from_dict(c.session.to_dict())
    with pytest.raises(AttributeError):
        SensingController.from_session(c.session).model = replace(c.model, sensor_noise_std=.01)
    with pytest.raises(ValueError):
        replace(c, simulated_noise_provenance="")


def test_uncertain_zero_background_has_mass_but_threshold_suppresses_reportable_excess():
    mean, variance = np.zeros((20, 32)), np.full((20, 32), .04**2)
    legacy = positive_expectation(mean, variance, 0.)
    revised = positive_expectation(mean, variance, .15)
    assert legacy.mean() == pytest.approx(.04 / np.sqrt(2 * np.pi))
    assert revised.mean() < 1e-6
    # Area normalization, not pixel count: doubling both axes does not multiply mass.
    doubled = positive_expectation(np.zeros((40, 64)), np.full((40, 64), .04**2), .15)
    assert revised.mean() == pytest.approx(doubled.mean())
    # The model still admits weak uncertain signal; this is not an occupancy mask.
    assert revised.min() > 0.


def test_two_axis_tracking_preserves_cells_and_shortens_both_dimensions():
    s, belief = localized()
    first = choose(belief, s)
    repeated = choose(belief, s)
    assert first == repeated and first.status == "target_met"
    assert set(first.requested_cells) <= set(first.swept_cells)
    sweeps = [seg for seg in first.segments if seg.kind == "sweep"]
    assert max(abs(seg.end_yx[0] - seg.start_yx[0]) for seg in sweeps) < .5
    assert len(sweeps) < s.instrument.lattice_shape[1] / 2
    assert first.predicted_mass_recall >= .9 - 1e-10


def test_block_surveillance_fits_when_no_full_height_pass_fits():
    s = config().session
    s = replace(s, startup_mode="prior", source_position=None,
                policy=replace(s.policy, policy_id="age_segment_v2"))
    plan = choose(GaussianSignalBelief(s.model), s, budget=.2, pose=(0., 1/64))
    assert plan.status == "reacquisition" and len(plan.samples) == 4
    assert plan.duration_seconds == pytest.approx(.2)
    assert plan.final_position_yx == pytest.approx((.2, 1/64))
    assert "two-axis" in plan.reason
    assert choose(GaussianSignalBelief(s.model), s, budget=.19, pose=(0., 1/64)).status == "no_legal_action"


def test_tracking_can_expand_move_and_return_without_inheriting_a_shrink_rule():
    s, b = localized()
    initial = choose(b, s)
    expanded = b.clone()
    expanded.mean.reshape(s.model.grid_shape)[3:9, 4:10] = 1.
    wider = choose(expanded, s)
    assert wider.status == "target_met"
    assert len(wider.requested_cells) > len(initial.requested_cells)
    shifted = b.clone()
    shifted.mean = np.roll(b.mean.reshape(s.model.grid_shape), 3, axis=1).ravel()
    moved = choose(shifted, s)
    assert moved.status == "target_met" and moved.requested_cells != initial.requested_cells
    assert choose(b, s).requested_cells == initial.requested_cells


def test_surveillance_acquires_outside_a_previously_focused_region():
    s, b = localized()
    focused = choose(b, s)
    visited = np.full(s.instrument.lattice_shape, -1.)
    visited.ravel()[list(focused.swept_cells)] = 30.
    b.time_seconds = 31.
    survey = choose(b, s, now=31., last_survey=0., ages=visited)
    assert survey.status == "reacquisition"
    assert set(survey.requested_cells) - set(focused.swept_cells)


def test_periodic_service_does_not_let_one_old_cell_force_perpetual_fallback():
    s, b = localized()
    now = 61.
    b.time_seconds = now
    old = np.zeros(s.instrument.lattice_shape)
    track = choose(b, s, now=now, last_survey=60., ages=old)
    due = choose(b, s, now=now, last_survey=0., ages=old)
    assert track.status == "target_met"
    assert due.status == "reacquisition" and "periodic_surveillance_due" in due.reason


def test_no_reportable_mean_does_not_normalize_uncertainty_tails_into_tracking():
    s, b = localized()
    b.mean[:] = 0.
    b.covariance[:] = np.diag(np.linspace(.02**2, .06**2, b.mean.size))
    plan = choose(b, s)
    assert plan.status == "reacquisition" and "no_reportable_signal" in plan.reason
    assert plan.predicted_mass_recall is None
    # The fixed comparator still measures its declared geometry without signal evidence.
    fixed = choose(b, replace(s, policy=replace(s.policy, policy_id="static_corridor_segment_v2")))
    assert fixed.status == "static_corridor"


def test_oversized_tracking_remains_a_shortfall_not_a_relabelled_subset():
    s, b = localized()
    b.mean[:] = np.linspace(.2, 2., b.mean.size)
    result = choose(b, s, budget=1.)
    assert result.status == "reacquisition"
    assert result.candidates and "duration_budget" in result.tracking_shortfall
    assert all(c.duration_seconds > 1. for c in result.candidates)
    assert len(result.samples) < len(result.candidates[0].requested_cells)


def test_double_profile_charges_resolution_without_increasing_detector_speed():
    c = config("double-prior-startup")
    p = c.session.instrument
    full = compile_profile_scan(np.ones((40, 64), bool), p, scan_id="all", started_at_seconds=0., current_position=(0., 0.))
    assert p.acquisition.cadence_hz == 20.
    assert p.mechanics.sweep_vertical_fov_per_second == .5
    assert full.duration_seconds == pytest.approx(129.5875)
    assert len(full.samples) == 2560
    assert all(s.completed_at_seconds - s.started_at_seconds == pytest.approx(2.) for s in full.segments if s.kind == "sweep")
    assert len(survey_blocks((40, 64), .2)) == 5 * 64


def test_partial_startup_requires_actual_visits_even_with_an_empty_receipt():
    c = config("double-full-startup")
    ctl = SensingController.from_session(c.session)
    plan = ctl.plan_next(0., (0., 0.), ScanRequestV1(.5), wind=CausalWind())
    assert plan.status == "reacquisition" and not ctl.startup_complete
    ctl.finish_scan(ExecutionReceipt(scan_id=plan.scan_id, completed_at_seconds=plan.completed_at_seconds,
                                    final_position_yx=plan.final_position_yx, received_at_seconds=plan.completed_at_seconds))
    assert not ctl.startup_complete and np.all(ctl.visit_times < 0)


def test_simulator_noise_is_independent_of_assumed_measurement_noise_and_replays():
    c = config()
    c = replace(c, duration_seconds=.8, initial_budget_seconds=.8, simulated_noise_std=0.,
                session=replace(c.session, source_position=None, startup_mode="prior",
                                policy=replace(c.session.policy, policy_id="age_segment_v2")))
    clean = run_sensing_v1(c)
    noisy = run_sensing_v1(replace(c, simulated_noise_std=.2))
    assert clean.config.model.sensor_noise_std == noisy.config.model.sensor_noise_std == .05
    assert clean.scans[0].plan == noisy.scans[0].plan
    assert any(a.value != b.value for a, b in zip(clean.scans[0].execution.readings, noisy.scans[0].execution.readings))
    replay = CausalReplayV1(c.session, clean.events).advance_to(.8)
    np.testing.assert_allclose(replay.belief_view().mean, clean.scans[-1].posterior.mean)
    unavailable = summarize_v1(clean)
    assert unavailable["reconstruction"]["J"] is None
    assert unavailable["scan_schedule"][0]["actual_reportable_signal_recall"] is None
    assert 0. < unavailable["scan_schedule"][0]["unvisited_fraction"] < 1.
    assert unavailable["scan_schedule"][0]["observed_age_quantiles_seconds"]["p100"] < .8
    assert unavailable["focusing_audit"]["tracking_requested_cell_sequence"] == []


def test_delayed_v2_delivery_cannot_rewrite_past_online_estimates():
    c = config()
    c = replace(c, duration_seconds=1., initial_budget_seconds=.5, delivery="batch", delivery_delay_seconds=.2,
                session=replace(c.session, source_position=None, startup_mode="prior"))
    run = run_sensing_v1(c)
    first_delivery = next(e for e in run.events if e.operation == "ingest_readings")
    before = CausalReplayV1(c.session, run.events).advance_to(first_delivery.at_seconds - .01)
    assert not before.assimilation_events
    after = CausalReplayV1(c.session, run.events).advance_to(first_delivery.at_seconds)
    # The next plan may start at the same receipt time and clear its per-scan
    # event display; actual visits and the conditioned state remain available.
    assert np.any(after.visit_times >= 0)
    with pytest.raises(ValueError):
        after.predictive_view(first_delivery.at_seconds - .01)
    loss = reconstruction_loss(run, None)
    assert loss["status"] == "truth_unavailable"
