from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from plume_focus.controller import SensingController
from plume_focus.experiment_contracts import RasterVideoScan
from plume_focus.sensing_contracts import CausalWind, ScanConstraints, ScannerCapabilities, Footprint, ExecutionReceipt
from plume_focus.sensing_diagnostics import measurement_replay
from plume_focus.sensing_evaluation import summarize_run
from plume_focus.sensing_rendering import RenderSettings, compose_sensing_frame, measurement_panel, render_sensing_video
from plume_focus.sensing_simulation import (
    CleanMethaneSource, FakeScanner, SensingSimulationConfig, SourceExhausted, run_sensing_simulation,
)
from plume_focus.signal_belief import SignalModelConfig
from plume_focus.video_rendering import project_single_pixel_scan, project_single_pixel_scan_legacy_overlay, overlay_attention


def config(**kwargs):
    return SensingSimulationConfig(capabilities=ScannerCapabilities(lattice_shape=(4, 6), readout_hz=10,
                                    integration_seconds=.02, footprint=Footprint(.25, 1 / 6)),
                                   model=SignalModelConfig(grid_shape=(3, 4)), max_scans=2,
                                   initial_budget_seconds=10, budget_seconds=3, duration_seconds=15, **kwargs)


@pytest.fixture(scope="module")
def run():
    return run_sensing_simulation(config())


def test_headless_adapter_never_writes_or_constructs_video(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("headless adapter attempted I/O")
    monkeypatch.setattr(Path, "write_text", reject)
    monkeypatch.setattr(cv2, "VideoWriter", reject)
    result = run_sensing_simulation(config())
    assert len(result.scans) == 2


def test_fake_scanner_manual_api_reproduces_simulator(run):
    cfg, source = run.config, CleanMethaneSource()
    declared_source = tuple(np.array(source.config.source) / (np.array(source.config.shape) - 1))
    controller = SensingController(cfg.capabilities, cfg.model, source_position=declared_source)
    scanner = FakeScanner(source, noise_std=cfg.model.sensor_noise_std, seed=cfg.seed)
    now, pose = 0., (0., 0.)
    for i, trace in enumerate(run.scans):
        plan = controller.plan_next(now, pose, ScanConstraints(cfg.initial_budget_seconds if i == 0 else cfg.budget_seconds), wind=trace.wind)
        assert plan == trace.plan
        execution = scanner.execute(plan)
        assert execution == trace.execution
        actual = scanner.deliver(controller, execution)
        np.testing.assert_array_equal(actual.mean, trace.posterior.mean)
        now, pose = execution.receipt.received_at_seconds, execution.receipt.final_position_yx


def test_finite_source_stops_partial_scan_and_never_freezes_last_value():
    source = CleanMethaneSource(end_time_seconds=.85)
    run = run_sensing_simulation(config(), source=source)
    assert run.termination_reason == "source_exhausted"
    assert len(run.scans) == 1
    trace = run.scans[0]
    assert trace.execution.receipt.completed_at_seconds == .85
    assert trace.execution.exhausted
    assert all(r.acquired_at_seconds + r.integration_seconds / 2 <= .85 for r in trace.execution.readings)
    with pytest.raises(SourceExhausted):
        source.signal_at(np.array([.851]), np.array([[.5, .5]]))
    report = summarize_run(run, source)
    assert report["scans_completed"] == 1


def test_truth_absence_is_null_and_evaluator_cannot_change_operations(run):
    original = run.operational_hash
    absent = summarize_run(run)
    present = summarize_run(run, CleanMethaneSource())
    assert absent["mean_mass_recall"] is None
    assert absent["evaluated_predictions"] == 0
    assert present["evaluated_predictions"] == len(run.scans)
    assert absent["operational_hash"] == present["operational_hash"] == original
    json.dumps(absent, allow_nan=False)
    json.dumps(present, allow_nan=False)


def test_legacy_thin_contours_overwrite_values_but_fixed_projection_does_not():
    shape = (24, 32)
    mask = np.zeros(shape, bool)
    mask[:, ::4] = True
    times = np.full(shape, np.nan)
    times[mask] = np.linspace(0, 1, mask.sum())
    scan = RasterVideoScan(np.zeros(shape, np.uint8), times, 20, 0, 1, mask)
    bright = replace(scan, values=np.where(mask, 255, 0).astype(np.uint8))
    np.testing.assert_array_equal(project_single_pixel_scan_legacy_overlay(scan), project_single_pixel_scan_legacy_overlay(bright))
    difference = project_single_pixel_scan(scan) != project_single_pixel_scan(bright)
    assert np.count_nonzero(np.any(difference, axis=-1)) == mask.sum()
    background = np.full((*shape, 3), 60, np.uint8)
    np.testing.assert_array_equal(overlay_attention(background, np.ones(shape)), background)


def test_measurements_visible_and_delayed_batch_not_shown_early(run):
    trace = run.scans[0]
    changed = replace(trace, execution=replace(trace.execution, readings=tuple(replace(r, value=1.0) for r in trace.execution.readings)))
    end = trace.execution.receipt.received_at_seconds
    assert np.any(measurement_panel(trace, end, (100, 160)) != measurement_panel(changed, end, (100, 160)))
    batch = run_sensing_simulation(config(delivery="batch"))
    first = batch.scans[0]
    before = measurement_panel(first, first.execution.receipt.received_at_seconds - .01, (100, 160))
    np.testing.assert_array_equal(before, np.broadcast_to(np.array([9, 14, 24], np.uint8), before.shape))


def test_playback_modes_fps_speed_and_truth_leave_trace_unchanged(run, tmp_path):
    original = run.operational_hash
    settings = [RenderSettings(mode="physical", fps=2, playback_speed=4, panel_shape=(80, 240)),
                RenderSettings(mode="summary", fps=8, panel_shape=(80, 240))]
    for i, setting in enumerate(settings):
        path = tmp_path / f"video-{i}.mp4"
        manifest = render_sensing_video(run, path, settings=setting)
        capture = cv2.VideoCapture(str(path))
        count = 0
        while capture.read()[0]:
            count += 1
        capture.release()
        assert count == manifest["encoded_frames"]
        assert manifest["operational_hash"] == original
        if setting.mode == "summary":
            assert count == len(run.scans)
    assert run.operational_hash == original


def test_fixed_schedule_measurement_replay_has_same_evidence_identity(run):
    report, fields = measurement_replay(run, CleanMethaneSource())
    assert report["operational_hash"] == run.operational_hash
    assert len(fields) == len(run.scans)
    assert all(record["reading_count"] == len(trace.execution.readings) for record, trace in zip(report["scan_schedule"], run.scans))
    assert all(np.isclose(a.sum(), 1) and np.isclose(b.sum(), 1) for a, b in fields)


def test_source_free_controller_has_no_source_reinjection():
    result = run_sensing_simulation(config(source_free=True))
    assert result.source_assumption == "source_free_with_provided_wind"
    # Prediction before the first reading has zero signal mean without a source.
    assert np.max(abs(result.scans[0].predicted.mean)) == 0


def test_fake_scanner_applies_declared_gain_and_offset(run):
    plan = run.scans[0].plan
    source = CleanMethaneSource()
    a = FakeScanner(source, noise_std=0).execute(plan)
    b = FakeScanner(source, noise_std=0, gain=2, offset=3).execute(plan)
    np.testing.assert_allclose([r.value for r in b.readings], 2 * np.array([r.value for r in a.readings]) + 3)
