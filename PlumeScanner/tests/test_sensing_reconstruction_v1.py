from dataclasses import replace
import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from plume_focus.sensing_contracts import CausalWind
from plume_focus.sensing_execution_v1 import (
    CausalReplayV1, ControllerCallV1, ReconstructionGridV1, SensingRunConfigV1, SyntheticSourceV1, run_sensing_v1,
)
from plume_focus.sensing_reconstruction import reconstruction_loss, summarize_v1
from test_sensing_v1 import session


class ConstantTruth:
    end_time_seconds = None

    def signal_at(self, times, positions_yx):
        return np.ones_like(times)


def run_config(**changes):
    return replace(SensingRunConfigV1(session(), SyntheticSourceV1(), CausalWind(error_speed=0),
                                     0, 6., 6., 6., "stream", 0., 0.,
                                     ReconstructionGridV1((4, 4), .25, "uniform_cell_area", "left_endpoint")), **changes)


def test_config_and_recorded_calls_roundtrip_and_reproduce_inference():
    config = run_config()
    payload = json.loads(json.dumps(config.to_dict()))
    assert SensingRunConfigV1.from_dict(payload) == config
    run = run_sensing_v1(config, source=ConstantTruth())
    events = tuple(ControllerCallV1.from_dict(json.loads(json.dumps(e.to_dict()))) for e in run.events)
    replay = CausalReplayV1(config.session, events)
    c = replay.advance_to(run.scans[-1].execution.receipt.received_at_seconds)
    np.testing.assert_allclose(c.belief_view().mean, run.scans[-1].posterior.mean, atol=1e-12)
    with pytest.raises(ValueError, match="chronological"):
        replay.advance_to(0)


def test_causal_loss_includes_prior_idle_and_delay_without_retroactive_improvement():
    source = ConstantTruth()
    stream = run_sensing_v1(run_config(), source=source)
    batch = run_sensing_v1(run_config(delivery="batch", delivery_delay_seconds=1.), source=source)
    a, b = reconstruction_loss(stream, source), reconstruction_loss(batch, source)
    assert a["quadrature_samples"][0]["map_mse"] == pytest.approx(1.)
    assert b["quadrature_samples"][0]["map_mse"] == pytest.approx(1.)
    assert b["J"] > a["J"]
    assert sum(s["duration_seconds"] for s in a["quadrature_samples"]) == pytest.approx(6.)
    assert a["J_over_T"] == pytest.approx(a["J"] / 6.)
    r = summarize_v1(batch, source, loss=b)
    assert r["idle_tail_seconds"] > 0
    assert r["actual_time_components_seconds"]["receipt_latency"] == pytest.approx(1.)


def test_truth_absence_and_incomplete_truth_never_become_zero_or_truncated_loss():
    run = run_sensing_v1(run_config(), source=ConstantTruth())
    assert reconstruction_loss(run, None)["J"] is None
    finite = ConstantTruth()
    finite.end_time_seconds = 2.
    result = reconstruction_loss(run, finite)
    assert result["status"] == "incomplete_truth_horizon" and result["J"] is None
    from plume_focus.sensing_simulation import CleanMethaneSource
    report = summarize_v1(run, CleanMethaneSource(end_time_seconds=2.))
    assert report["reconstruction"]["status"] == "incomplete_truth_horizon"
    assert report["reconstruction"]["J"] is None
    assert report["scan_schedule"][0]["actual_sweep_mass_recall"] is None


def test_finite_source_exhaustion_records_partial_execution_and_stops():
    source = ConstantTruth()
    source.end_time_seconds = 2.
    run = run_sensing_v1(run_config(), source=source)
    assert run.termination_reason == "source_exhausted"
    assert run.scans[-1].execution.receipt.completed_at_seconds == 2.
    assert all(r.available_at_seconds <= 2. for r in run.scans[-1].execution.readings)
    assert reconstruction_loss(run, source)["status"] == "incomplete_truth_horizon"


def test_fake_scanner_example_reproduces_simulator_plans_and_posteriors():
    path = Path(__file__).parents[1] / "examples/sensing_controller_v1.py"
    spec = importlib.util.spec_from_file_location("fake_v1_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = run_config()
    run = run_sensing_v1(config)
    example = module.execute_example(config)
    assert len(example) == len(run.scans)
    for (plan, view), trace in zip(example, run.scans):
        assert plan == trace.plan
        np.testing.assert_allclose(view.mean, trace.posterior.mean, atol=1e-12)


def test_renderer_playback_is_observer_only_and_signal_changes_are_visible(tmp_path):
    from plume_focus.sensing_rendering import RenderSettings, compose_sensing_frame, render_sensing_video, measurement_panel
    run = run_sensing_v1(run_config(delivery="batch", delivery_delay_seconds=.5), source=ConstantTruth())
    trace = run.scans[0]
    digest = run.operational_hash
    time = trace.execution.receipt.received_at_seconds
    a = compose_sensing_frame(trace, run, time, RenderSettings(panel_shape=(80, 240)))
    b = compose_sensing_frame(trace, run, time, RenderSettings(panel_shape=(80, 240), playback_speed=2, fps=5))
    # Playback labels change, signal/uncertainty/path panels stay identical.
    np.testing.assert_array_equal(a[76:], b[76:])
    changed = replace(trace, execution=replace(trace.execution, readings=tuple(replace(r, value=0.) for r in trace.execution.readings)))
    assert not np.array_equal(measurement_panel(trace, time, (80, 240)), measurement_panel(changed, time, (80, 240)))
    manifest = render_sensing_video(run, tmp_path / "physical.mp4", settings=RenderSettings(fps=1, playback_speed=2, panel_shape=(80, 240)))
    assert manifest["operational_hash"] == digest == run.operational_hash
    assert manifest["physical_end_seconds"] == 6.


def test_cli_rejects_legacy_overrides_before_running_v1(tmp_path):
    from plume_focus.sensing_cli import main
    with pytest.raises(SystemExit):
        main(["--run-config", "nonexistent.json", "--duration", "75", "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
