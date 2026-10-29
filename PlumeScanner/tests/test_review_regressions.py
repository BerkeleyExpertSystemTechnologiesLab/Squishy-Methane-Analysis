"""Analytic and malformed-input regressions found in the 2026-10-29 review."""
from dataclasses import replace
import numpy as np
import pytest

from plume_focus.benchmark_contracts import BenchmarkRunResult
from plume_focus.benchmark_statistics import compare_adaptive_to_baselines
from plume_focus.recorded_manifest_io import load_json_object
from plume_focus.sensing_simulation import CleanMethaneSource, FakeScanner
from plume_focus.sensing_counterfactual import TruthQuadrature, OnlineLoss
from plume_focus.video_sources import (SyntheticMethaneLeakConfig, SyntheticPlumeVideoConfig,
    generate_synthetic_plume_values)
from test_full_fov_loop import controller, plan


@pytest.mark.parametrize("changes", [
    {"emission_interval": 0}, {"emission_interval": 1.5}, {"memory_frames": 1.5},
    {"base_sigma": 0.}, {"diffusion_per_frame": -.1}, {"fps": float("nan")},
    {"lateral_phase": float("inf")}, {"wind_velocity": (0., float("nan"))},
])
def test_generator_configuration_rejects_undefined_fields(changes):
    with pytest.raises(ValueError):
        SyntheticMethaneLeakConfig(**changes)


def test_clean_source_supports_empty_queries():
    result = CleanMethaneSource().signal_at(np.empty((2, 0)), np.empty((2, 0, 2)))
    assert result.shape == (2, 0)


def test_clean_source_rejects_invalid_pointing():
    with pytest.raises(ValueError):
        CleanMethaneSource().signal_at(np.array([1.]), np.array([[np.nan, .5]]))


def test_plume_query_shape_is_not_part_of_the_physics():
    config = SyntheticPlumeVideoConfig()
    times = np.arange(6., dtype=float).reshape(2, 3)
    rows, cols = np.full_like(times, 60.), np.full_like(times, 40.)
    actual = generate_synthetic_plume_values(config, times, rows, cols)
    flat = generate_synthetic_plume_values(config, times.ravel(), rows.ravel(), cols.ravel())
    np.testing.assert_array_equal(actual, flat.reshape(times.shape))


@pytest.mark.parametrize("settings", [{"delivery_delay": np.nan}, {"completion_overrun": np.inf}])
def test_invalid_delivery_is_rejected_before_source_queries(settings):
    class Untouched:
        end_time_seconds = None
        calls = 0
        def signal_at(self, times, positions_yx):
            self.calls += 1
            return np.ones_like(times)
    source = Untouched()
    with pytest.raises(ValueError):
        FakeScanner(source).execute(plan(controller()), **settings)
    assert source.calls == 0


def result(policy, budget="b"):
    return BenchmarkRunResult("e", budget, policy, False, 1, "test", True,
        .1, .1, 1., 1., .5, .5, .5, .5, .5, .5, ())


@pytest.mark.parametrize("rows,budgets", [
    ((result("a"), result("a"), result("b")), ("b",)),
    ((result("a", "unexpected"), result("b", "unexpected")), ("b",)),
    ((result("a"), result("b")), ("b", "b")),
    ((result("a"), result("b"), replace(result("b"), episode_id="extra")), ("b",)),
])
def test_paired_statistics_reject_ambiguous_or_unbalanced_inputs(rows, budgets):
    with pytest.raises(ValueError):
        compare_adaptive_to_baselines(rows, adaptive_policy_id="a", baseline_policy_ids=("b",), budget_ids=budgets)


@pytest.mark.parametrize("payload", ['{"split":"development","split":"held-out"}', '{"value": NaN}', '{"value": Infinity}'])
def test_manifest_parser_rejects_noncanonical_ambiguous_json(tmp_path, payload):
    path = tmp_path / "manifest.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_json_object(path)


def test_invalid_truth_cannot_become_zero_reconstruction_loss():
    class InvalidTruth:
        end_time_seconds = None
        def signal_at(self, times, positions_yx):
            return np.full_like(times, np.nan)
    with pytest.raises(ValueError, match="finite"):
        TruthQuadrature(InvalidTruth(), 0., 1., (2, 2), shape=(2, 2), step=.5)


def test_invalid_prediction_cannot_become_zero_reconstruction_loss():
    class ZeroTruth:
        end_time_seconds = None
        def signal_at(self, times, positions_yx):
            return np.zeros_like(times)
    c = controller()
    c._belief.mean[:] = np.nan
    loss = OnlineLoss(TruthQuadrature(ZeroTruth(), 0., 1., (2, 2), shape=(2, 2), step=.5))
    with pytest.raises(ValueError, match="finite"):
        loss.before(c, 1.)


@pytest.mark.parametrize("entry", ["legacy", "typed", "suite"])
def test_cli_preserves_existing_run_before_any_simulation(tmp_path,entry):
    marker=tmp_path/"report.json";marker.write_bytes(b"original evidence")
    from plume_focus.sensing_cli import main
    from plume_focus.sensing_cli_v1 import run_config_file,run_suite_file
    with pytest.raises(FileExistsError,match="fresh output"):
        if entry=="legacy":main(["--output-dir",str(tmp_path)])
        elif entry=="typed":run_config_file("not-read.json",tmp_path)
        else:run_suite_file("not-read.json",tmp_path)
    assert marker.read_bytes()==b"original evidence"
    assert list(tmp_path.iterdir())==[marker]
