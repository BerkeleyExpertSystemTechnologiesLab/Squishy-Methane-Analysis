"""Independent mixture arithmetic, causal API and physical compatibility."""

from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from plume_focus.controller import SensingController
from plume_focus.motion_config import PersistentMotionConfig
from plume_focus.motion_belief import PersistentWindBelief, bias_quadrature
from plume_focus.sensing_cli_v1 import config_from_dict, load_config
from plume_focus.sensing_contracts import CausalWind, Footprint, ExecutionReceipt
from plume_focus.sensing_execution_v1 import CausalReplayV1, ReconstructionGridV1, run_sensing_v1
from plume_focus.sensing_execution_v2 import SensingRunConfigV2
from plume_focus.sensing_execution_v3 import SensingRunConfigV3
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_session_v3 import SensingSessionV3
from plume_focus.signal_belief import GaussianSignalBelief, SignalModelConfig, observation_weights, positive_expectation
from plume_focus.signal_distribution import PredictiveDistribution
from plume_focus.sensing_reconstruction import reconstruction_loss, summarize_v1
from test_sensing_core import sample_readings


def motion(cov=((.0001, 0.), (0., .0001)), order=3):
    return PersistentMotionConfig(cov, "session_constant", order, "Declared synthetic verification covariance")


def config(zero=False):
    c = load_config(Path(__file__).parents[1] / "configs/sensing-v2/coarse-full-startup.json")
    fields = {name: getattr(c.session, name) for name in c.session.__dataclass_fields__ if name not in ("formulation_name", "policy_type")}
    fields.update(formulation_id="beam_reconstruction_motion_v3_candidate", motion=motion(((0., 0.), (0., 0.))) if zero else motion())
    s = SensingSessionV3(**fields)
    values = {name: getattr(c, name) for name in c.__dataclass_fields__}
    return replace(SensingRunConfigV3(**{**values, "session": s}), duration_seconds=1.6, initial_budget_seconds=.4,
                   session=replace(s, startup_mode="prior", source_position=None, model=replace(s.model, grid_shape=(4, 4))),
                   evaluation=ReconstructionGridV1((8, 8), .1, "uniform_cell_area", "left_endpoint"))


@pytest.mark.parametrize("cov", [((.0001, 0.), (0., .0004)), ((.0001, .00008), (.00008, .0001)), ((0., 0.), (0., 0.)), ((.0001, 0.), (0., 0.))])
def test_quadrature_reproduces_declared_bias_mean_covariance(cov):
    offsets, weights = bias_quadrature(motion(cov))
    np.testing.assert_allclose(weights @ offsets, 0, atol=1e-16)
    np.testing.assert_allclose((offsets.T * weights) @ offsets, cov, atol=1e-16)
    assert weights.sum() == pytest.approx(1.)
    if not np.any(cov):
        assert len(weights) == 1


@pytest.mark.parametrize("kwargs", [dict(bias_covariance_yx=((1., 2.), (2., 1.))), dict(bias_covariance_yx=((1., .2), (0., 1.))),
    dict(quadrature_order=4), dict(persistence="per_reading"), dict(provenance="")])
def test_motion_config_rejects_invalid_or_undeclared_model(kwargs):
    with pytest.raises(ValueError):
        replace(motion(), **kwargs)


def test_mixture_conditioning_matches_independent_multivariate_batch():
    model = SignalModelConfig(grid_shape=(2, 2), initial_std=.4, correlation_length=.3,
                              diffusivity=0, process_std_per_sqrt_second=0, source_rate=0)
    b = PersistentWindBelief(model, motion())
    for i, state in enumerate(b.components):
        state.mean[:] = np.array([.2, .4, .1, .3]) + i * .05
    priors = [(s.mean.copy(), s.covariance.copy()) for s in b.components]
    initial = b.weights.copy()
    H = np.array([observation_weights((2, 2), p, Footprint(.4, .3)) for p in ((.3, .3), (.3, .3), (.7, .7))])
    y, r = np.array([.8, 0., .4]), .03
    expected, logs = [], []
    for weight, (mu, P) in zip(initial, priors):
        S = H @ P @ H.T + r * np.eye(3)
        residual = y - H @ mu
        K = P @ H.T @ np.linalg.inv(S)
        expected.append((mu + K @ residual, P - K @ H @ P))
        logs.append(np.log(weight) - .5 * (3*np.log(2*np.pi) + np.linalg.slogdet(S)[1] + residual @ np.linalg.solve(S, residual)))
    expected_weights = np.exp(np.array(logs) - max(logs)); expected_weights /= expected_weights.sum()
    for h, value in zip(H, y):
        b.condition(value, h, r)
    np.testing.assert_allclose(b.weights, expected_weights, atol=1e-12)
    for actual, (mu, P) in zip(b.components, expected):
        np.testing.assert_allclose(actual.mean, mu, atol=1e-12)
        np.testing.assert_allclose(actual.covariance, P, atol=1e-12)
    expected_mean = sum(w * mu for w, (mu, _) in zip(expected_weights, expected))
    expected_cov = sum(w * (P + np.outer(mu-expected_mean, mu-expected_mean)) for w, (mu, P) in zip(expected_weights, expected))
    np.testing.assert_allclose(b.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(b.covariance, expected_cov, atol=1e-12)
    assert np.linalg.eigvalsh(b.covariance).min() >= -1e-12


def test_mixture_expected_excess_and_bands_do_not_collapse_to_gaussian():
    d = PredictiveDistribution(0., np.array([.5, .5]), np.array([[[-1.]], [[1.]]]), np.full((2, 1, 1), .01))
    assert d.expected_excess(.5).item() == pytest.approx(.25, abs=1e-8)
    collapsed = positive_expectation(d.mean, d.variance, .5).item()
    assert abs(collapsed - .25) > .04
    lo, hi = d.envelope(2.)
    assert lo.item() == pytest.approx(-1.2) and hi.item() == pytest.approx(1.2)
    with pytest.raises(ValueError):
        d.component_means[0, 0, 0] = 0


def test_batched_projection_matches_individual_conditional_fields():
    b = PersistentWindBelief(SignalModelConfig(grid_shape=(4, 5)), motion())
    b.advance(1.3, CausalWind(velocity_yx=(.02, .04)), (.5, .2))
    d = b.distribution_on((13, 17))
    for i, component in enumerate(b.components):
        mu, var = component.moments_on((13, 17))
        np.testing.assert_array_equal(d.component_means[i], mu)
        np.testing.assert_array_equal(d.component_variances[i], var)
    fast=b.distribution_on((13,17),fast_projection=True)
    np.testing.assert_array_equal(fast.component_means,d.component_means)
    np.testing.assert_allclose(fast.component_variances,d.component_variances,atol=1e-15,rtol=1e-13)
    np.testing.assert_allclose(fast.expected_excess_vectorized(.15),d.expected_excess(.15),atol=1e-15,rtol=1e-12)
    mu,var=b.aggregate_moments_on((13,17))
    np.testing.assert_allclose(mu,d.mean,atol=1e-15,rtol=1e-13)
    np.testing.assert_allclose(var,d.variance,atol=1e-15,rtol=1e-13)


def test_new_nominal_wind_changes_future_only_and_does_not_reset_bias():
    b = PersistentWindBelief(SignalModelConfig(grid_shape=(4, 5)), motion())
    b.advance(1., CausalWind(velocity_yx=(0., .02)), (.5, .2))
    h = observation_weights((4, 5), (.5, .25), Footprint(.1, .1))
    b.condition(.8, h, .0025)
    logs = b.log_weights.copy()
    changed = b.clone(); changed.advance(2., CausalWind(1., 1., (.08, 0.), .02))
    b.advance(2., CausalWind(velocity_yx=(0., .02)))
    np.testing.assert_array_equal(changed.log_weights, logs)
    assert not np.allclose(b.mean, changed.mean)


def test_motion_forecast_locality_retains_bias_and_reports_boundary_loss():
    m = SignalModelConfig(grid_shape=(12, 16), diffusivity=0, process_std_per_sqrt_second=0, source_rate=0)
    b = PersistentWindBelief(m, motion(((.0009, 0.), (0., .0009))))
    for state in b.components:
        state.mean[:] = 0.; state.mean.reshape(m.grid_shape)[5:7, 7:9] = 1.; state.covariance[:] = 0.
    weights = b.weights.copy()
    b.advance(3., CausalWind(velocity_yx=(0., .02)))
    d = b.distribution_on(m.grid_shape)
    assert d.between_variance.max() > .01
    assert d.mean[5:7, 6:11].sum() > 1.
    # Existing semi-Lagrangian interpolation has numerical spreading; require
    # remote weight to stay small, not a fictitious compact-support guarantee.
    assert d.mean[:2].sum() < .01 * d.mean.sum()
    np.testing.assert_array_equal(b.weights, weights)
    before = b.mean.sum()
    b.advance(40., CausalWind(velocity_yx=(0., .2)))
    assert b.mean.sum() < before / 100
    np.testing.assert_allclose(b.weights, weights)


def test_configuration_roundtrip_preserves_hashes_and_rejects_old_decoder():
    c = config()
    decoded = config_from_dict(json.loads(json.dumps(c.to_dict())))
    assert decoded == c and decoded.session.hashes == c.session.hashes
    with pytest.raises(ValueError):
        SensingRunConfigV2.from_dict(c.to_dict())
    payload = c.to_dict(); payload["session"]["motion"].pop("provenance")
    with pytest.raises(ValueError):
        config_from_dict(payload)


def test_zero_uncertainty_matches_v2_execution_and_completed_inference():
    from plume_focus.sensing_session_v2 import SensingSessionV2
    c = config(zero=True)
    payload = c.to_dict(); payload["message_type"] = "SensingRunConfigV2"
    payload["session"].pop("motion"); payload["session"]["formulation_id"] = "beam_reconstruction_v2_candidate"
    old = SensingRunConfigV2.from_dict(payload)
    a, b = run_sensing_v1(old), run_sensing_v1(c)
    assert a.events == b.events
    for left, right in zip(a.scans, b.scans):
        np.testing.assert_array_equal(left.posterior.mean, right.posterior.mean)
        np.testing.assert_array_equal(left.posterior.std, right.posterior.std)


def test_delayed_open_scan_replay_duplicates_and_overrun_preserve_persistent_state():
    c = config()
    a, b = [SensingController.from_session(c.session) for _ in range(2)]
    wind = replace(c.wind, velocity_yx=(.01, .04))
    p = a.plan_next(0., (0., 0.), ScanRequestV1(.4), wind=wind)
    assert p == b.plan_next(0., (0., 0.), ScanRequestV1(.4), wind=wind)
    readings = sample_readings(p)
    for r in readings:
        a.ingest_readings([r], available_at_seconds=r.available_at_seconds)
    # Late earlier acquisitions trigger replay from the open-scan base.
    b.ingest_readings([readings[-1]], available_at_seconds=p.completed_at_seconds)
    b.ingest_readings(list(readings[:-1]), available_at_seconds=p.completed_at_seconds + .1)
    before = b.motion_diagnostics()
    assert not b.ingest_readings([readings[0]], available_at_seconds=p.completed_at_seconds + .1)
    assert b.motion_diagnostics() == before
    with pytest.raises(ValueError, match="Conflicting"):
        b.ingest_readings([replace(readings[0], value=100.)], available_at_seconds=p.completed_at_seconds + .1)
    assert b.motion_diagnostics() == before and b.active_plan == p
    with pytest.raises(RuntimeError):
        b.plan_next(.5, p.final_position_yx, ScanRequestV1(.4))
    receipt = ExecutionReceipt(p.scan_id, p.completed_at_seconds + .2, p.completed_at_seconds + .4, p.final_position_yx)
    va, vb = a.finish_scan(receipt), b.finish_scan(receipt)
    np.testing.assert_allclose(va.mean, vb.mean, atol=1e-12)
    np.testing.assert_allclose(va.std, vb.std, atol=1e-12)
    assert a.motion_diagnostics() == b.motion_diagnostics()
    before = b.motion_diagnostics()
    d = b.predictive_distribution(receipt.received_at_seconds + 3, (10, 10))
    b.predictive_view(receipt.received_at_seconds + 2)
    assert b.motion_diagnostics() == before
    with pytest.raises(ValueError):
        b.predictive_distribution(0., (4, 4))
    b.plan_next(receipt.received_at_seconds, receipt.final_position_yx, ScanRequestV1(.4), wind=wind)
    assert b.motion_diagnostics() == before  # repeated wind is not a new prior


def test_fake_example_replay_causal_loss_and_unavailable_truth():
    c = config()
    run = run_sensing_v1(c)
    replay = CausalReplayV1(c.session, run.events)
    state = replay.advance_to(run.events[-1].at_seconds)
    np.testing.assert_allclose(state.belief_view().mean, run.scans[-1].posterior.mean, atol=1e-12)
    spec = importlib.util.spec_from_file_location("fake_motion", Path(__file__).parents[1] / "examples/sensing_controller_v1.py")
    example = importlib.util.module_from_spec(spec); spec.loader.exec_module(example)
    assert [p for p, _ in example.execute_example(c, count=len(run.scans))] == [s.plan for s in run.scans]
    delayed = run_sensing_v1(replace(c, delivery="batch", delivery_delay_seconds=.5))
    loss = reconstruction_loss(delayed, c.source.build())
    loss_only = reconstruction_loss(delayed,c.source.build(),distribution_diagnostics=False)
    assert loss_only["J"] == pytest.approx(loss["J"],abs=1e-14)
    assert loss_only["expected_excess_integral"] is None
    initial = [r for r in loss["quadrature_samples"] if r["time_seconds"] < delayed.scans[0].execution.receipt.received_at_seconds]
    # No reading has been delivered: the source-free zero-mean prior remains zero.
    assert all(r["signed_mean_enhancement_error"] <= 0 for r in initial)
    assert loss["coverage_band_method"] == "component_envelope"
    assert summarize_v1(run, None)["reconstruction"]["J"] is None
    finite = replace(c.source, end_time_seconds=.1).build()
    assert reconstruction_loss(run, finite)["status"] == "incomplete_truth_horizon"


def test_mixture_renderer_preserves_layout_measurements_and_trace(tmp_path):
    from plume_focus.sensing_rendering import RenderSettings, compose_sensing_frame, measurement_panel, render_sensing_video
    c = config(); run = run_sensing_v1(c); s = run.scans[0]; t = s.execution.receipt.received_at_seconds
    digest = run.operational_hash
    a = compose_sensing_frame(s, run, t, RenderSettings())
    b = compose_sensing_frame(s, run, t, RenderSettings(fps=2, playback_speed=3))
    assert a.shape == (612, 960, 3)
    np.testing.assert_array_equal(a[76:], b[76:])
    changed = replace(s, execution=replace(s.execution, readings=tuple(replace(r, value=1.) for r in s.execution.readings)))
    assert not np.array_equal(measurement_panel(s, t, (80, 240)), measurement_panel(changed, t, (80, 240)))
    render_sensing_video(run, tmp_path / "motion.mp4", settings=RenderSettings(fps=2, playback_speed=2))
    assert digest == run.operational_hash


def test_fixed_evidence_diagnostic_preserves_receipt_and_delivery_boundaries():
    spec = importlib.util.spec_from_file_location("motion_fixed", Path(__file__).parents[1] / "scripts/diagnose-persistent-wind.py")
    diagnostic = importlib.util.module_from_spec(spec); spec.loader.exec_module(diagnostic)
    c = replace(config(), delivery="batch", delivery_delay_seconds=.2, completion_overrun_seconds=.1)
    run = run_sensing_v1(c)
    t = run.events[-1].at_seconds
    state = diagnostic.fixed_evidence_state(c, run.events, t)
    live = CausalReplayV1(c.session, run.events).advance_to(t)
    np.testing.assert_allclose(state.mean.reshape(c.model.grid_shape), live.belief_view().mean, atol=1e-12)
    assert state.diagnostics() == live.motion_diagnostics()


def test_blank_truth_reports_all_existing_allocation_as_background():
    c = config()
    c = replace(c, source=replace(c.source, signal_scale=0.))
    run = run_sensing_v1(c)
    report = summarize_v1(run, c.source.build())
    for row in report["scan_schedule"]:
        if row["expected_reportable_excess"] > c.session.policy.mass_floor:
            assert row["predicted_background_mass_fraction"] == pytest.approx(1.)
        assert row["actual_reportable_signal_recall"] is None
