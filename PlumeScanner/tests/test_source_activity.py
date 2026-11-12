import json
import math

import numpy as np
import pytest

from plume_focus.calibrated_scanner import CalibrationSettings, DecayGaussianBelief
from plume_focus.controller import SensingController
from plume_focus.sensing_contracts import CausalWind, Footprint, SensorReading
from plume_focus.signal_belief import SignalModelConfig
from plume_focus.source_activity import (
    SOURCE_ACTIVITY_FORMULATION_ID,
    SOURCE_ACTIVITY_POLICY_ID,
    SourceActivityBelief,
    SourceActivityScanner,
    SourceActivitySession,
)
from test_sensing_v1 import profile

SOURCE = (0.5, 0.5)

HALF_LIFE = 6.0
RATE = math.log(2.0) / HALF_LIFE
SOURCE_RATE = 0.3
# The decay model's steady-state source-anchored concentration: the "on" model
# predicts source_rate / rate, so an active reading must sit near that value.
ON_VALUE = SOURCE_RATE / RATE
OFF_VALUE = 0.0


def make_model(**kw):
    defaults = dict(grid_shape=(6, 8), source_rate=SOURCE_RATE, source_sigma=0.1,
                    sensor_noise_std=0.05, diffusivity=0.0,
                    process_std_per_sqrt_second=0.0, max_transition_seconds=0.5,
                    signal_units="synthetic_concentration_path")
    defaults.update(kw)
    return SignalModelConfig(**defaults)


def make_reading(value, t, scan="s"):
    return SensorReading(scan_id=scan, reading_id=f"{scan}:{t:g}", position_yx=SOURCE,
                         acquired_at_seconds=t, available_at_seconds=t + 0.02, value=value,
                         footprint=Footprint(height=0.1, width=0.1))


def run_phase(belief, wind, value, start, count):
    for i in range(count):
        belief.assimilate(make_reading(value, start + i), wind, SOURCE)


def test_activity_stays_high_under_continuous_source():
    b = SourceActivityBelief(make_model(), half_life_seconds=HALF_LIFE)
    run_phase(b, CausalWind(), ON_VALUE, 0.5, 20)
    assert b.activity_probability > 0.99


def test_activity_drops_after_cessation():
    b = SourceActivityBelief(make_model(), half_life_seconds=HALF_LIFE)
    wind = CausalWind()
    run_phase(b, wind, ON_VALUE, 0.5, 20)
    assert b.activity_probability > 0.9
    run_phase(b, wind, OFF_VALUE, 20.5, 20)
    assert b.activity_probability < 0.1


def test_activity_recovers_after_restart():
    b = SourceActivityBelief(make_model(), half_life_seconds=HALF_LIFE)
    wind = CausalWind()
    run_phase(b, wind, ON_VALUE, 0.5, 20)
    run_phase(b, wind, OFF_VALUE, 20.5, 20)
    assert b.activity_probability < 0.1
    run_phase(b, wind, ON_VALUE, 40.5, 20)
    assert b.activity_probability > 0.9


def test_no_declared_source_is_inert():
    b = SourceActivityBelief(make_model(), half_life_seconds=6.0)
    wind = CausalWind()
    prior = b.activity_probability
    for i in range(20):
        b.assimilate(make_reading(0.3, 0.5 + i), wind, None)
    assert b.activity_probability == pytest.approx(prior)
    assert b.activity_probability == pytest.approx(0.5)


def test_mixture_approaches_active_model_when_certain():
    model = make_model()
    wind = CausalWind()
    on = DecayGaussianBelief(model, half_life_seconds=6.0)
    near_one = SourceActivityBelief(model, half_life_seconds=6.0, prior_active_probability=0.999999)
    on.advance(2.0, wind, SOURCE)
    near_one.advance(2.0, wind, SOURCE)
    np.testing.assert_allclose(near_one.mean, on.mean, atol=1e-6)
    np.testing.assert_allclose(near_one.covariance, on.covariance, atol=1e-6)


def test_clone_preserves_activity_state_and_is_independent():
    b = SourceActivityBelief(make_model(), half_life_seconds=6.0)
    wind = CausalWind()
    b.assimilate(make_reading(0.3, 1.0), wind, SOURCE)
    c = b.clone()
    assert c.activity_probability == pytest.approx(b.activity_probability)
    assert c.log_odds == pytest.approx(b.log_odds)
    np.testing.assert_array_equal(c.mean, b.mean)
    c.assimilate(make_reading(0.0, 2.0), wind, SOURCE)
    assert c.activity_probability != pytest.approx(b.activity_probability)


def test_session_dispatch_and_scanner_wiring():
    session = SourceActivitySession(profile(), make_model(), CalibrationSettings(), None, half_life_seconds=6.0)
    scanner = SensingController.from_session(session)
    assert isinstance(scanner, SourceActivityScanner)
    assert isinstance(scanner._belief, SourceActivityBelief)
    assert scanner.activity_probability == pytest.approx(session.prior_active_probability)
    assert session.policy_id == SOURCE_ACTIVITY_POLICY_ID
    assert session.formulation_id == SOURCE_ACTIVITY_FORMULATION_ID


def test_session_rejects_invalid_prior():
    with pytest.raises(ValueError):
        SourceActivitySession(profile(), make_model(), CalibrationSettings(), None, prior_active_probability=0.0)
    with pytest.raises(ValueError):
        SourceActivitySession(profile(), make_model(), CalibrationSettings(), None, prior_active_probability=1.0)


def test_session_json_roundtrip():
    session = SourceActivitySession(profile(), make_model(), CalibrationSettings(), None, half_life_seconds=6.0)
    restored = SourceActivitySession.from_dict(json.loads(json.dumps(session.to_dict())))
    assert restored == session
