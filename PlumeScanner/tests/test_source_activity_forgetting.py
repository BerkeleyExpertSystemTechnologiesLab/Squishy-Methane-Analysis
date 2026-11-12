import json
import math

import numpy as np
import pytest

from plume_focus.calibrated_scanner import CalibrationSettings
from plume_focus.controller import SensingController
from plume_focus.full_fov_execution import decode_config
from plume_focus.sensing_contracts import CausalWind, Footprint, SensorReading
from plume_focus.sensing_execution_v2 import SyntheticSourceV2
from plume_focus.signal_belief import SignalModelConfig
from plume_focus.source_activity import SourceActivityBelief
from plume_focus.source_activity_forgetting import (
    SOURCE_ACTIVITY_FORGETTING_FORMULATION_ID,
    SOURCE_ACTIVITY_FORGETTING_POLICY_ID,
    SourceActivityForgettingBelief,
    SourceActivityForgettingRunConfig,
    SourceActivityForgettingScanner,
    SourceActivityForgettingSession,
)
from test_sensing_v1 import profile

SOURCE = (0.5, 0.5)
HALF_LIFE = 6.0


def make_source(end_time_seconds=60.0):
    return SyntheticSourceV2(source=SOURCE, end_time_seconds=end_time_seconds)
RATE = math.log(2.0) / HALF_LIFE
ON_VALUE = 0.3 / RATE


def make_model(**kw):
    defaults = dict(grid_shape=(6, 8), source_rate=0.3, source_sigma=0.1,
                    sensor_noise_std=0.05, diffusivity=0.0,
                    process_std_per_sqrt_second=0.0, max_transition_seconds=0.5,
                    signal_units="synthetic_concentration_path")
    defaults.update(kw)
    return SignalModelConfig(**defaults)


def make_reading(value, t, scan="s"):
    return SensorReading(scan_id=scan, reading_id=f"{scan}:{t:g}", position_yx=SOURCE,
                         acquired_at_seconds=t, available_at_seconds=t + 0.02, value=value,
                         footprint=Footprint(height=0.1, width=0.1))


def test_forgetting_none_matches_v1_bitwise():
    v1 = SourceActivityBelief(make_model(), half_life_seconds=HALF_LIFE)
    v2 = SourceActivityForgettingBelief(make_model(), half_life_seconds=HALF_LIFE,
                                        log_odds_half_life_seconds=None)
    wind = CausalWind()
    for i in range(12):
        v1.assimilate(make_reading(ON_VALUE, 0.5 + i), wind, SOURCE)
        v2.assimilate(make_reading(ON_VALUE, 0.5 + i), wind, SOURCE)
    assert v1.log_odds == v2.log_odds
    assert v1.activity_probability == v2.activity_probability
    np.testing.assert_array_equal(v1.mean, v2.mean)


def test_forgetting_decays_log_odds_on_advance():
    b = SourceActivityForgettingBelief(make_model(), half_life_seconds=HALF_LIFE,
                                       log_odds_half_life_seconds=10.0)
    b.log_odds = 4.0
    b.advance(5.0, CausalWind(), SOURCE)
    assert b.log_odds == pytest.approx(4.0 * math.exp(-math.log(2.0) / 10.0 * 5.0))


def test_v1_log_odds_frozen_between_readings():
    b = SourceActivityBelief(make_model(), half_life_seconds=HALF_LIFE)
    b.log_odds = 4.0
    b.advance(5.0, CausalWind(), SOURCE)
    assert b.log_odds == pytest.approx(4.0)


def test_invalid_log_odds_half_life_rejected():
    with pytest.raises(ValueError):
        SourceActivityForgettingBelief(make_model(), half_life_seconds=HALF_LIFE,
                                       log_odds_half_life_seconds=0.0)
    with pytest.raises(ValueError):
        SourceActivityForgettingSession(profile(), make_model(), CalibrationSettings(), None,
                                        half_life_seconds=6.0, log_odds_half_life_seconds=-1.0)


def test_session_dispatch_and_scanner_wiring():
    session = SourceActivityForgettingSession(profile(), make_model(), CalibrationSettings(), None,
                                              half_life_seconds=6.0, log_odds_half_life_seconds=15.0)
    scanner = SensingController.from_session(session)
    assert isinstance(scanner, SourceActivityForgettingScanner)
    assert isinstance(scanner._belief, SourceActivityForgettingBelief)
    assert scanner._belief.log_odds_half_life_seconds == 15.0
    assert session.policy_id == SOURCE_ACTIVITY_FORGETTING_POLICY_ID
    assert session.formulation_id == SOURCE_ACTIVITY_FORGETTING_FORMULATION_ID


def test_session_json_roundtrip():
    session = SourceActivityForgettingSession(profile(), make_model(), CalibrationSettings(), None,
                                              half_life_seconds=6.0, log_odds_half_life_seconds=30.0)
    restored = SourceActivityForgettingSession.from_dict(json.loads(json.dumps(session.to_dict())))
    assert restored == session


def test_run_config_roundtrip_and_controller():
    session = SourceActivityForgettingSession(profile(), make_model(), CalibrationSettings(), None,
                                              half_life_seconds=6.0, log_odds_half_life_seconds=15.0)
    config = SourceActivityForgettingRunConfig(session, source=make_source(), wind=CausalWind(),
                                               duration_seconds=60.0, simulated_noise_std=0.0,
                                               cessation_seconds=30.0, seed=0, delivery="stream",
                                               evaluation_shape=(6, 8), evaluation_step_seconds=1.0)
    restored = decode_config(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert isinstance(restored.make_controller(), SourceActivityForgettingScanner)
