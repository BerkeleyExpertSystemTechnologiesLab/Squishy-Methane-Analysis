"""Acceptance tests for the opt-in evidence-gated focus candidate."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from plume_focus.controller import SensingController
from plume_focus.evidence_gate import (DEFAULT_ENTRY_LEVEL, DEFAULT_GATE_ALPHA, DEFAULT_RETAIN_LEVEL,
    EVIDENCE_GATE_POLICY_ID, EvidenceGateState, GateActionEvidence, GateDecision, bound_multiplier,
    build_action_evidence, calibrated_value, classify_bounds, entry_decision, full_action_alpha,
    measurement_error_std, normal_quantile, observed_value_cutoff, reading_bounds, renewal_decision)
from plume_focus.evidence_gated_execution import EvidenceGatedRunConfig
from plume_focus.evidence_gated_loop import (EvidenceGateSettings, EvidenceGatedPlan, EvidenceGatedScanner,
    EvidenceGatedSession, gate_slot_map)
from plume_focus.full_fov_execution import decode_config, decode_plan, run_full_fov, verify_replay
from plume_focus.full_fov_loop import FullFovController, FullFovPlan, FullFovSession, FullFovSettings
from plume_focus.sensing_contracts import ExecutionReceipt, SensorReading
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.signal_belief import SignalModelConfig
from test_full_fov_loop import ConstantSource
from test_sensing_v1 import profile


def gate_session(**settings):
    model = SignalModelConfig(grid_shape=(2, 2), signal_units="synthetic_concentration_path",
                              source_rate=0, diffusivity=0, process_std_per_sqrt_second=0)
    return EvidenceGatedSession(profile(), model, EvidenceGateSettings(**settings), None)


def controller(**settings):
    return EvidenceGatedScanner(gate_session(**settings))


def plan(c, budget=100.):
    return c.plan_next(c._last_available, c._last_pose, ScanRequestV1(budget))


def execute(c, p, value=0., count=None, delivery="stream", delay=0., overrun=0.):
    execution = FakeScanner(ConstantSource(value), noise_std=0).execute(p, delivery=delivery, delivery_delay=delay,
                                                                        completion_overrun=overrun)
    readings = execution.readings if count is None else execution.readings[:count]
    if delivery == "batch":
        c.ingest_readings(readings, available_at_seconds=execution.receipt.received_at_seconds)
    else:
        for reading in readings:
            c.ingest_readings([reading], available_at_seconds=reading.available_at_seconds)
    c.finish_scan(execution.receipt)
    return execution


def field(c, mean, variance=1e-5):
    c._belief.mean[:] = mean
    c._belief.covariance[:] = np.eye(len(c._belief.mean)) * variance


def evidence(values, *, slots=None, alpha=.025, gain=1., offset=0., noise=.05, mode="FULL", contract_notes=(),
             note=0.):
    slots = len(values) if slots is None else slots
    bounds = reading_bounds(values, gain=gain, offset=offset, sensor_noise_std=noise, alpha_action=alpha,
                            scheduled_slots=slots)
    readings = tuple(__import__("plume_focus.evidence_gate", fromlist=["x"]).GateReadingEvidence(
        reading_id=f"{mode}-{index}", slot_index=index, raw_value=value,
        calibrated_value=calibrated_value(value, gain=gain, offset=offset), lower_bound=lower, upper_bound=upper,
        acquired_at_seconds=float(index), available_at_seconds=float(index), position_yx=(.5, .5))
        for index, (value, (lower, upper)) in enumerate(zip(values, bounds)))
    return build_action_evidence(scan_id="s-0", epoch=1, action_index=1, mode=mode, alpha_action=alpha,
                                 scheduled_slots=slots, readings=readings, completed_at_seconds=note + 1.,
                                 received_at_seconds=note + 1., plan_completed_at_seconds=1., contract_notes=contract_notes)


def test_quantile_matches_an_independent_implementation():
    independent = NormalDist().inv_cdf
    for probability in (1 - 1e-9, .5, .9, .95, 1 - .025 / (2 * 640), 1 - .05 / (2 * 4)):
        assert abs(normal_quantile(probability) - independent(probability)) < 1e-12


def test_bounds_use_calibration_frozen_slots_and_documented_magnitudes():
    values = (.4, .2)
    bounds = reading_bounds(values, gain=1., offset=0., sensor_noise_std=.05, alpha_action=.025, scheduled_slots=16)
    multiplier = NormalDist().inv_cdf(1 - .025 / 32)
    assert np.allclose(bounds[0][0], .4 - multiplier * .05, atol=1e-13)
    assert np.allclose(bounds[0][1], .4 + multiplier * .05, atol=1e-13)
    # M is the frozen schedule, not the number of values supplied.
    assert bounds == reading_bounds(values, gain=1., offset=0., sensor_noise_std=.05, alpha_action=.025, scheduled_slots=16)
    assert observed_value_cutoff(.15, gain=1., offset=0., sensor_noise_std=.05, alpha_action=.025, scheduled_slots=640) == pytest.approx(.35564779584685524, abs=1e-12)
    assert observed_value_cutoff(.15, gain=1., offset=0., sensor_noise_std=.05, alpha_action=.025, scheduled_slots=2560) == pytest.approx(.37111494274450096, abs=1e-12)
    # Multiplicity and error scale are conservative: more slots and more noise widen the bounds.
    assert bound_multiplier(.025, 2560) > bound_multiplier(.025, 640)
    scaled = reading_bounds([1.], gain=2., offset=.5, sensor_noise_std=.1, alpha_action=.025, scheduled_slots=4)[0]
    assert scaled[0] == pytest.approx(.25 - NormalDist().inv_cdf(1 - .025 / 8) * .05, abs=1e-13)
    assert measurement_error_std(.1, gain=2.) == pytest.approx(.05)


def test_epoch_alpha_accounting_and_action_multiplicity():
    for count in (1, 2, 5, 50):
        assert sum(full_action_alpha(p) for p in range(1, count + 1)) <= DEFAULT_GATE_ALPHA + 1e-15
    assert full_action_alpha(1) == pytest.approx(.025)
    assert full_action_alpha(2) == pytest.approx(.05 / 6)
    with pytest.raises(ValueError):
        full_action_alpha(0)
    for unit in (0, 1, -1):
        with pytest.raises(ValueError):
            bound_multiplier(float(unit), 4)
    with pytest.raises(ValueError):
        bound_multiplier(.025, 0)


def test_evidence_states_use_strict_comparisons_and_keep_labels_distinct():
    assert classify_bounds([], [], DEFAULT_ENTRY_LEVEL) == "unavailable"
    assert classify_bounds([.1500000001], [.4], DEFAULT_ENTRY_LEVEL) == "supported"
    assert classify_bounds([.15], [.15], DEFAULT_ENTRY_LEVEL) == "inconclusive"
    assert classify_bounds([-.2], [.1499999], DEFAULT_ENTRY_LEVEL) == "below_level_on_observed_footprints"
    assert classify_bounds([.05], [.4], DEFAULT_ENTRY_LEVEL) == "inconclusive"
    assert classify_bounds([-.5, 0., .05], [-.4, .05, .1], DEFAULT_ENTRY_LEVEL) == "below_level_on_observed_footprints"


def test_negative_zero_and_scaled_readings_are_handled():
    bounds = reading_bounds([0., -1.], gain=1., offset=0., sensor_noise_std=.05, alpha_action=.025, scheduled_slots=2)
    assert bounds[0][1] > 0 > bounds[1][0]
    negative = evidence([0.], slots=2)
    assert negative.state(DEFAULT_ENTRY_LEVEL) == "below_level_on_observed_footprints"
    scaled = evidence([2.], gain=2., offset=1., slots=2)
    assert scaled.readings[0].calibrated_value == pytest.approx(.5)
    assert scaled.state(DEFAULT_ENTRY_LEVEL) == "supported"


def test_known_gaussian_measurement_fixture_covers_the_clean_value():
    """Independent fixture: declared marginal error model, not a small plume suite."""
    generator = np.random.default_rng(7)
    clean, noise, level, alpha, slots = .25, .05, .0, .05, 1
    bounds = [reading_bounds([clean + value], gain=1., offset=0., sensor_noise_std=noise, alpha_action=alpha,
                             scheduled_slots=slots)[0] for value in generator.normal(0, noise, 4000)]
    covered = np.mean([lower <= clean <= upper for lower, upper in bounds])
    assert .93 < covered < .97
    assert sum(bound_multiplier(alpha, 4) for _ in range(4)) > bound_multiplier(alpha, 4)


def test_settings_validation_freezes_hysteresis_parameters():
    settings = EvidenceGateSettings()
    assert (settings.alpha, settings.entry_level, settings.retain_level) == (DEFAULT_GATE_ALPHA, DEFAULT_ENTRY_LEVEL,
                                                                            DEFAULT_RETAIN_LEVEL)
    assert settings.vertical_margin_fov == 0. and settings.full_scan_only is False
    with pytest.raises(ValueError):
        EvidenceGateSettings(retain_level=.2)
    with pytest.raises(ValueError):
        EvidenceGateSettings(alpha=0.)
    with pytest.raises(ValueError):
        EvidenceGateSettings(entry_level=-.1)
    with pytest.raises(ValueError):
        EvidenceGateSettings(full_scan_only=True, vertical_margin_fov=1.5)


def test_zero_readings_cannot_focus_even_with_a_positive_posterior():
    c = controller()
    execute(c, plan(c), 0.)
    field(c, .9, .01)
    q = plan(c)
    assert q.mode == "FULL" and q.reason == "signal_not_supported"
    assert q.gate_fallback_reason == "entry_not_supported"
    # Sixteen frozen slots at alpha .025 leave a half width above the entry level,
    # so a zero reading is inconclusive rather than a clean absence statement.
    assert q.gate_evidence_state == "inconclusive"
    assert q.gate_max_mean > q.threshold and q.gate_support_reading_ids == ()
    assert q.gate_scheduled_slots == 16 and q.gate_delivered_slots == 16
    assert q.gate_max_upper_bound > q.threshold > q.gate_max_lower_bound

    blank = controller()
    execute(blank, plan(blank), -.2)
    field(blank, .9, .01)
    stopped = plan(blank)
    assert stopped.mode == "FULL"
    assert stopped.gate_evidence_state == "below_level_on_observed_footprints"
    assert stopped.gate_max_upper_bound < stopped.threshold


def test_strong_readings_can_focus_while_the_mean_gate_fails():
    c = controller()
    execute(c, plan(c), .6)
    field(c, .14, .01)
    q = plan(c)
    assert q.mode == "FOCUS" and q.gate_evidence_state == "supported"
    assert q.gate_max_mean <= q.threshold
    assert q.gate_max_lower_bound > DEFAULT_ENTRY_LEVEL
    assert q.gate_alpha_action == .05 and q.gate_evidence_actions[0].alpha_action == .025
    assert q.gate_observed_value_cutoff == pytest.approx(observed_value_cutoff(.15, gain=1., offset=0.,
        sensor_noise_std=.05, alpha_action=.025, scheduled_slots=16), abs=1e-12)
    assert q.gate_support_reading_ids


def test_multiplicity_penalty_can_reject_a_reading_above_the_entry_level():
    c = controller()
    execute(c, plan(c), .2)
    field(c, .14, .01)
    q = plan(c)
    assert q.mode == "FULL"
    assert q.gate_evidence_state == "inconclusive"
    assert q.gate_max_lower_bound < DEFAULT_ENTRY_LEVEL < q.gate_max_upper_bound


def test_focus_renewal_uses_the_retain_level_and_fresh_evidence():
    c = controller()
    execute(c, plan(c), .6)
    field(c, .14, .01)
    focus = plan(c)
    assert focus.mode == "FOCUS"
    renewal_alpha = DEFAULT_GATE_ALPHA
    retain_cutoff = observed_value_cutoff(DEFAULT_RETAIN_LEVEL, gain=1., offset=0., sensor_noise_std=.05,
                                          alpha_action=renewal_alpha, scheduled_slots=len(focus.samples))
    entry_cutoff = observed_value_cutoff(DEFAULT_ENTRY_LEVEL, gain=1., offset=0., sensor_noise_std=.05,
                                         alpha_action=renewal_alpha, scheduled_slots=len(focus.samples))
    assert retain_cutoff < entry_cutoff
    field(c, .14, .01)
    execute(c, focus, .5 * (retain_cutoff + entry_cutoff))
    renewal = plan(c)
    assert renewal.mode == "FOCUS" and renewal.gate_evidence_state == "supported"
    assert renewal.gate_applicable_level == DEFAULT_RETAIN_LEVEL
    execute(c, renewal, .5 * retain_cutoff)
    stopped = plan(c)
    assert stopped.mode == "FULL" and stopped.gate_fallback_reason == "focus_not_renewed"
    assert stopped.gate_evidence_state in ("inconclusive", "below_level_on_observed_footprints")


def test_partially_executed_focus_requires_full_and_keeps_missing_coverage():
    c = controller()
    execute(c, plan(c), .6)
    field(c, .14, .01)
    focus = plan(c)
    field(c, .14, .01)
    execute(c, focus, .6, count=1)
    q = plan(c)
    assert q.mode == "FULL"
    assert q.gate_fallback_reason in ("focus_partially_executed", "focus_not_renewed")
    assert q.gate_last_fully_executed is False
    assert len(c.full_scan_missing) == 16


def test_gate_contract_violation_requests_full():
    c = controller()
    execute(c, plan(c), .6)
    field(c, .14, .01)
    focus = plan(c)
    field(c, .14, .01)
    execution = FakeScanner(ConstantSource(.6), noise_std=0).execute(focus)
    reading = execution.readings[0]
    c.ingest_readings([reading], available_at_seconds=reading.available_at_seconds)
    assert c._gate_contract_violations == 0
    c.ingest_readings([replace(execution.readings[-1], reading_id="not-a-slot")],
                      available_at_seconds=execution.receipt.received_at_seconds)
    c.finish_scan(execution.receipt)
    assert c._gate_contract_violations == 1
    q = plan(c)
    assert q.mode == "FULL" and q.gate_contract_violation
    assert q.gate_fallback_reason == "gate_contract_violation"


def test_stale_full_scan_support_cannot_renew_focus_indefinitely():
    c = controller()
    execute(c, plan(c), .6)
    field(c, .14, .01)
    focus = plan(c)
    field(c, .14, .01)
    execute(c, focus, 0.)
    q = plan(c)
    assert q.mode == "FULL" and q.gate_fallback_reason == "focus_not_renewed"
    assert q.gate_full_coverage_incomplete
    assert c.gate_focus_allowed is False
    assert [transition.to_mode for transition in c.gate_state().transitions] == ["FOCUS", "FULL"]
    ages = q.gate_support_age_seconds
    assert ages == () or all(age >= 0 for age in ages)


def test_stream_and_batch_delivery_agree_and_queries_are_inert():
    a, b = controller(), controller()
    left, right = plan(a), plan(b)
    assert left == right
    execution = FakeScanner(ConstantSource(.6), noise_std=0).execute(left)
    for reading in execution.readings:
        a.ingest_readings([reading], available_at_seconds=reading.available_at_seconds)
    # A repeated identity is idempotent and does not double-count evidence.
    assert a.ingest_readings([execution.readings[0]], available_at_seconds=execution.receipt.received_at_seconds) == ()
    b.ingest_readings(execution.readings, available_at_seconds=execution.receipt.received_at_seconds)
    a.predictive_view(50.)
    a.predictive_distribution(50., (2, 2))
    a.gate_state()
    a.finish_scan(execution.receipt)
    b.finish_scan(execution.receipt)
    assert a.gate_state() == b.gate_state()
    np.testing.assert_array_equal(a._belief.mean, b._belief.mean)
    np.testing.assert_array_equal(a._belief.covariance, b._belief.covariance)
    assert plan(a) == plan(b)


def test_serialized_session_plan_and_gate_state_round_trip():
    c = controller()
    execute(c, plan(c), .6)
    field(c, .14, .01)
    q = plan(c)
    restored = EvidenceGatedPlan.from_dict(json.loads(json.dumps(q.to_dict())))
    assert restored == q and decode_plan(json.loads(json.dumps(q.to_dict()))) == q
    session = EvidenceGatedSession.from_dict(json.loads(json.dumps(c.session.to_dict())))
    assert session == c.session and session.digest == c.session.digest
    assert session.settings.alpha == DEFAULT_GATE_ALPHA and session.settings.entry_level == DEFAULT_ENTRY_LEVEL
    assert session.policy_id == EVIDENCE_GATE_POLICY_ID
    state = c.gate_state()
    assert EvidenceGateState.from_dict(json.loads(json.dumps(state.to_dict()))) == state
    assert c.gate_state() == state


def test_dispatch_and_execution_binding_with_replay():
    from test_reconstruction_risk import config as risk_config
    base = risk_config()
    session = gate_session()
    run = EvidenceGatedRunConfig(session, replace(base.source, signal_scale=0.), base.wind, duration_seconds=40.,
                                 evaluation_shape=(8, 8), evaluation_step_seconds=.2)
    decoded = decode_config(json.loads(json.dumps(run.to_dict())))
    assert decoded == run and decoded.digest == run.digest
    assert isinstance(SensingController.from_session(session), EvidenceGatedScanner)
    result = run_full_fov(run)
    replay = verify_replay(json.loads(json.dumps(result)))
    assert result["scans"][0]["plan"]["mode"] == "FULL"
    assert result["scans"][0]["plan"]["gate_policy_id"] == EVIDENCE_GATE_POLICY_ID
    assert all(scan["plan"]["mode"] == "FULL" for scan in result["scans"])
    assert replay.gate_focus_allowed is False
    assert replay.gate_state().contract_violations == 0
    legacy = json.loads(json.dumps(result))
    for event in legacy["events"]:
        if event["operation"] == "plan_next":
            event["plan"].pop("gate_evidence_actions")
    for scan in legacy["scans"]:
        scan["plan"].pop("gate_evidence_actions")
    assert verify_replay(legacy).gate_state() == replay.gate_state()


def test_historical_loop_defaults_are_unchanged_by_the_hooks():
    session = FullFovSession(profile(), SignalModelConfig(grid_shape=(2, 2), signal_units="synthetic_concentration_path",
                                                          source_rate=0, diffusivity=0, process_std_per_sqrt_second=0),
                             FullFovSettings(), None)
    c = FullFovController(session)
    assert FullFovController._on_plan_opened(c, None) is None
    assert FullFovController._on_scan_finished(c, None) is None
    p = plan(c)
    assert set(p.to_dict()) == {f.name for f in __import__("dataclasses", fromlist=["x"]).fields(FullFovPlan)} | {
        "schema_version", "message_type"}
    execute(c, p, .6)
    assert plan(c).mode == "FOCUS"


@pytest.mark.parametrize("shift,overrun", [(.1, 0.), (.5, 1.)])
def test_actual_acquisition_timing_can_differ_from_the_schedule(shift, overrun):
    c = controller()
    p = plan(c)
    ex = FakeScanner(ConstantSource(.6), noise_std=0).execute(p)
    readings = tuple(replace(r, acquired_at_seconds=r.acquired_at_seconds + shift,
                             available_at_seconds=r.available_at_seconds + shift) for r in ex.readings)
    receipt = replace(ex.receipt, completed_at_seconds=ex.receipt.completed_at_seconds + shift + overrun,
                      received_at_seconds=ex.receipt.received_at_seconds + shift + overrun)
    c.ingest_readings(readings, available_at_seconds=receipt.received_at_seconds)
    c.finish_scan(receipt)
    field(c, .14, .01)
    focus = plan(c)
    assert focus.mode == "FOCUS" and not focus.gate_contract_violation
    assert focus.gate_last_fully_executed
    assert focus.started_at_seconds == receipt.received_at_seconds
    assert focus.gate_support_acquired_at_seconds == tuple(r.acquired_at_seconds for r in readings)
    renewed = FakeScanner(ConstantSource(.6), noise_std=0).execute(focus)
    shifted = tuple(replace(r, acquired_at_seconds=r.acquired_at_seconds + shift,
                            available_at_seconds=r.available_at_seconds + shift) for r in renewed.readings)
    completion = replace(renewed.receipt, completed_at_seconds=renewed.receipt.completed_at_seconds + shift + overrun,
                         received_at_seconds=renewed.receipt.received_at_seconds + shift + overrun)
    c.ingest_readings(shifted, available_at_seconds=completion.received_at_seconds)
    c.finish_scan(completion)
    assert c.gate_last_decision.kind == "focus_renewal" and c.gate_focus_allowed


def test_slot_mapping_keeps_identity_position_and_causal_validation():
    c = controller()
    p = c.plan_next(1., c.session.home_yx, ScanRequestV1(100.))
    ex = FakeScanner(ConstantSource(.6), noise_std=0).execute(p)
    first = ex.readings[0]
    for invalid in (replace(first, reading_id="unknown"), replace(first, scan_id="other"),
                    replace(first, position_yx=ex.readings[-1].position_yx)):
        slots, notes = gate_slot_map(p, [invalid])
        assert slots == {} and notes
    slots, notes = gate_slot_map(p, [first, first])
    assert len(slots) == 1 and notes == (f"duplicate_slot:{first.reading_id}",)
    with pytest.raises(ValueError, match="precedes scan start"):
        c.ingest_readings([replace(first, acquired_at_seconds=.25)], available_at_seconds=first.available_at_seconds)
    c.ingest_readings(ex.readings, available_at_seconds=ex.receipt.received_at_seconds)
    with pytest.raises(ValueError, match="after actual scan completion"):
        c.finish_scan(replace(ex.receipt, completed_at_seconds=first.acquired_at_seconds))
    assert c.active_plan == p and c.gate_state().last_decision is None


def test_plan_evidence_provenance_separates_all_full_epoch_actions_and_next_allocation():
    c = controller()
    first = plan(c)
    execute(c, first, .6, count=8)
    second = plan(c)
    assert second.mode == "FULL"
    execute(c, second, .6)
    field(c, .14, .01)
    focus = plan(c)
    assert focus.mode == "FOCUS" and focus.gate_alpha_action == .05
    actions = focus.gate_evidence_actions
    assert len(actions) == 2
    assert [a.scan_id for a in actions] == [first.scan_id, second.scan_id]
    assert sum(a.scheduled_slots for a in actions) == focus.gate_scheduled_slots
    assert sum(a.delivered_slots for a in actions) == focus.gate_delivered_slots
    for a in actions:
        assert a.observed_value_cutoff == pytest.approx(observed_value_cutoff(focus.gate_applicable_level,
            gain=1., offset=0., sensor_noise_std=.05, alpha_action=a.alpha_action, scheduled_slots=a.scheduled_slots))
    assert focus.gate_observed_value_cutoff == actions[-1].observed_value_cutoff
    assert actions[0].alpha_action == .025 and actions[1].alpha_action == pytest.approx(.05 / 6)
    restored = decode_plan(json.loads(json.dumps(focus.to_dict())))
    assert restored == focus
    path = Path(__file__).resolve().parents[1] / "scripts/render-full-fov-demo.py"
    spec = importlib.util.spec_from_file_location("gate_renderer", path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    detail = renderer.gate_detail(restored)
    assert "evidence alpha 0.0083 M 8" in detail and "next alpha 0.0500" in detail
    execute(c, focus, .6)
    field(c, .14, .01)
    renewal = plan(c)
    assert len(renewal.gate_evidence_actions) == 1
    assert renewal.gate_evidence_actions[0].alpha_action == .05
    legacy = focus.to_dict()
    legacy.pop("gate_evidence_actions")
    assert decode_plan(legacy).gate_evidence_actions == ()
