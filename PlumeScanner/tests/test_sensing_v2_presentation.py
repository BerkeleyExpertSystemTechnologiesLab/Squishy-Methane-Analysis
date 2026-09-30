"""The v2 adapter uses the same public lifecycle and observer-only panels."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np

from plume_focus.sensing_cli_v1 import load_config
from plume_focus.sensing_execution_v1 import run_sensing_v1
from plume_focus.sensing_rendering import RenderSettings, compose_sensing_frame, render_sensing_video
from plume_focus.sensing_reconstruction import summarize_v1


def short_config():
    c = load_config(Path(__file__).parents[1] / "configs/sensing-v2/coarse-full-startup.json")
    return replace(c, duration_seconds=.8, initial_budget_seconds=.4,
                   session=replace(c.session, source_position=None, startup_mode="prior"))


def test_v2_fake_example_and_simulator_have_identical_plans_and_beliefs():
    spec = importlib.util.spec_from_file_location("fake_v2_example", Path(__file__).parents[1] / "examples/sensing_controller_v1.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    c = short_config()
    run = run_sensing_v1(c)
    results = example.execute_example(c, count=len(run.scans))
    assert len(results) == len(run.scans) > 0
    for (plan, belief), trace in zip(results, run.scans):
        assert plan == trace.plan
        np.testing.assert_allclose(belief.mean, trace.posterior.mean, atol=1e-12)


def test_v2_unavailable_mass_and_playback_are_faithful(tmp_path):
    run = run_sensing_v1(short_config())
    trace = run.scans[0]
    assert trace.plan.predicted_mass_recall is None
    assert not np.any(trace.predicted.mass)
    before = run.operational_hash
    t = trace.execution.receipt.received_at_seconds
    a = compose_sensing_frame(trace, run, t, RenderSettings())
    b = compose_sensing_frame(trace, run, t, RenderSettings(playback_speed=3, fps=2))
    assert a.shape == (612, 960, 3)
    np.testing.assert_array_equal(a[76:], b[76:])
    manifest = render_sensing_video(run, tmp_path / "short.mp4", settings=RenderSettings(fps=2, playback_speed=2))
    assert manifest["operational_hash"] == before == run.operational_hash


def test_v2_early_completion_reports_actual_footprint_and_partial_visits():
    c = short_config()
    c = replace(c, source=replace(c.source, end_time_seconds=.1))
    run = run_sensing_v1(c)
    report = summarize_v1(run, c.source.build())
    row = report["scan_schedule"][0]
    assert run.termination_reason == "source_exhausted"
    assert row["actual_completion_seconds"] == .1
    assert row["swept_fraction"] < row["requested_fraction"]
    assert 0 < run.visited_fraction < row["requested_fraction"]
    assert report["reconstruction"]["status"] == "incomplete_truth_horizon"
