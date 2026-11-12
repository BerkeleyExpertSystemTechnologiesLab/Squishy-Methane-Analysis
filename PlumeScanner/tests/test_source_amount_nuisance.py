import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from plume_focus.full_fov_execution import decode_config
from plume_focus.signal_belief import observation_weights
from plume_focus.sensing_contracts import SensorReading
from plume_focus.source_amount_validation import ClockAffineFilter, bounded_posterior
from plume_focus.source_amount_nuisance import (
    METHODS, NuisanceEnvironment, amount_diagnostics, dynamics_pair,
    refinement_checks, run_level, sensor_noise,
    reading_schedule_sha256,
)

ROOT = Path(__file__).parents[1]


def fixture_config():
    cfg = json.loads((ROOT / "configs/source-prior-study/source-amount-nuisance-v1.json").read_text())
    cfg.update(grid_shape=[3, 4], fixed_duration_seconds=.25,
               forecast_horizons_seconds=[.125, .25],
               uncertainty_times_seconds=[.125, .25, .375, .5])
    cfg["scenes"] = [dict(name="test", split="evaluation", source_yx=[.3, .4], seeds=[None, 123])]
    run = decode_config(json.loads((ROOT / cfg["baseline_config"]).read_text()))
    # Unit fixtures are constructed here; scientific archive files are not a test dependency.
    events = []
    for i, position in enumerate(((.2, .3), (.5, .6), (.7, .4))):
        acquired = (i + 1) * .05
        available = acquired + cfg["clock_tick_seconds"]
        reading = SensorReading("test-scan", f"test-{i}", position, acquired, available,
                                0., run.capabilities.footprint, .025)
        events.append(dict(operation="ingest_readings", at_seconds=available,
                           readings=[reading.to_dict()]))
    record = dict(events=events)
    model = replace(run.model, grid_shape=(3, 4))
    return cfg, record, run, model


@pytest.mark.parametrize("factor,multiplier", [("matched", 1.), ("wind_speed", .5), ("half_life", 1.5)])
def test_nuisance_isolation_and_fixed_process_innovations(factor, multiplier):
    cfg, _, run, model = fixture_config()
    nominal, truth = dynamics_pair(model, run.wind.velocity_yx, 6., .0125,
                                   dict(factor=factor, multiplier=multiplier))
    for key in ("Q", "noise_factor", "P0", "initial_factor"):
        np.testing.assert_array_equal(getattr(nominal, key), getattr(truth, key))
    if factor == "matched":
        np.testing.assert_array_equal(nominal.A.toarray(), truth.A.toarray())
        assert nominal.source_dt == truth.source_dt
    else:
        assert np.max(np.abs(nominal.A.toarray() - truth.A.toarray())) > 1e-5
    assert truth.model.source_sigma == nominal.model.source_sigma
    if factor == "half_life":
        assert truth.source_dt != nominal.source_dt
        assert truth.velocity == nominal.velocity


@pytest.mark.parametrize("factor", ["wind_speed", "half_life"])
def test_information_control_direct_filter_matches_mean_truth(tmp_path, factor):
    cfg, record, run, model = fixture_config()
    nominal, truth = dynamics_pair(model, run.wind.velocity_yx, 6., .0125,
                                   dict(factor=factor, multiplier=1.5))
    env = NuisanceEnvironment(truth, run, cfg, record, tmp_path / "environment")
    f = ClockAffineFilter(truth, env.emissions)
    for ri, r in enumerate(env.schedule):
        f.advance(truth.tick_of(r.acquired_at_seconds))
        f.condition([env.readings[ri, 0]], observation_weights(model.grid_shape, r.position_yx, r.footprint))
        f.advance(truth.tick_of(r.available_at_seconds))
        expected = env.truth(f.tick, [0])[0]
        np.testing.assert_allclose(f.m[0] + f.B @ env.emissions[0], expected, atol=2e-14)
    np.testing.assert_allclose(env.emissions.mean(axis=1), cfg["reference_total_per_second"], rtol=1e-14)
    for a in (env.readings, env.noise, env.deterministic):
        assert not a.flags.writeable


def test_paired_random_draws_reproduce_and_key_sensor_noise_by_physical_slot(tmp_path):
    cfg, record, run, model = fixture_config()
    nominal, truth = dynamics_pair(model, run.wind.velocity_yx, 6., .0125,
                                   dict(factor="wind_speed", multiplier=1.5))
    a = NuisanceEnvironment(nominal, run, cfg, record, tmp_path / "a")
    b = NuisanceEnvironment(nominal, run, cfg, record, tmp_path / "b")
    c = NuisanceEnvironment(truth, run, cfg, record, tmp_path / "c")
    np.testing.assert_array_equal(a.readings, b.readings)
    np.testing.assert_array_equal(a.noise[0], c.noise[0])
    # Different transport changes realized noise, although draws and per-tick Q match.
    assert np.max(np.abs(a.noise[10] - c.noise[10])) > 1e-8
    assert sensor_noise(123, 9, (.3, .4), .1) == sensor_noise(123, 9, (.3, .4), .1)
    assert sensor_noise(123, 10, (.3, .4), .1) != sensor_noise(123, 9, (.3, .4), .1)
    assert sensor_noise(None, 9, (.3, .4), .1) == 0.


def test_amount_interval_and_edge_mass_use_actual_mixture():
    p = bounded_posterior(np.array([0., 0.]), np.array([0., 0.]), order=256)
    d = amount_diagnostics(p)
    assert d["amount_mean"] == pytest.approx(1.)
    assert d["true_amount_inside95"]
    np.testing.assert_allclose(d["amount_interval95"], [.2875, 1.7125], atol=.003)
    assert d["amount_edge_mass"] == pytest.approx(.1 / 1.5, abs=.006)
    fixed = amount_diagnostics(bounded_posterior(np.array([1.]), np.array([1.]), fixed=1.))
    assert fixed["amount_interval95"] == [1., 1.]


def test_physical_schedule_hash_tracks_timing_but_not_observed_values():
    _, record, _, _ = fixture_config()
    changed = copy.deepcopy(record)
    changed["events"][0]["readings"][0]["value"] = 100.
    assert reading_schedule_sha256(changed) == reading_schedule_sha256(record)
    changed["events"][0]["readings"][0]["available_at_seconds"] += .0125
    assert reading_schedule_sha256(changed) != reading_schedule_sha256(record)


def test_delivered_evidence_cannot_change_earlier_scores_and_matched_controls_equal(tmp_path):
    cfg, record, run, model = fixture_config()
    nominal, truth = dynamics_pair(model, run.wind.velocity_yx, 6., .0125,
                                   dict(factor="matched", multiplier=1.))
    env = NuisanceEnvironment(truth, run, cfg, record, tmp_path / "environment")
    a = run_level(nominal, truth, env, run, cfg, (17, 32), tmp_path / "original")
    changed = copy.copy(env)
    changed.readings = env.readings.copy()
    changed.readings[0] += .5
    b = run_level(nominal, truth, changed, run, cfg, (17, 32), tmp_path / "changed")
    old = np.load(tmp_path / "original/errors.npy")
    new = np.load(tmp_path / "changed/errors.npy")
    delivery = nominal.tick_of(env.schedule[0].available_at_seconds)
    np.testing.assert_array_equal(old[:delivery], new[:delivery])
    assert np.max(np.abs(old[delivery:] - new[delivery:])) > 1e-7
    np.testing.assert_array_equal(np.asarray(a["JT"])[:, 0], np.asarray(a["JT"])[:, 2])
    for forecast in a["forecasts"]:
        np.testing.assert_array_equal(np.asarray(forecast)[:, 0], np.asarray(forecast)[:, 2])
    assert refinement_checks(cfg, a, a)["passed"]
    altered = copy.deepcopy(a)
    altered["diagnostics"][0]["inside95"][0] = not altered["diagnostics"][0]["inside95"][0]
    check = refinement_checks(cfg, a, altered)
    assert not check["passed"]
    assert any("membership" in reason for row in check["checks"] for reason in row["failures"])
