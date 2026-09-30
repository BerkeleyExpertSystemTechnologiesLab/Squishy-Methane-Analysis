"""Matched rendering and inference diagnostics; never used by the controller."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from plume_focus.bayes import scan_likelihood, posterior_from_likelihood
from plume_focus.experiment_contracts import RasterVideoScan
from plume_focus.experiment_metrics import js_divergence
from plume_focus.sensing_simulation import SensingRun, SensorSource
from plume_focus.signal_belief import GaussianSignalBelief, grid_positions
from plume_focus.video_rendering import project_single_pixel_scan, project_single_pixel_scan_legacy_overlay


def _raster(trace, shape):
    values = np.zeros(shape, dtype=float)
    times = np.full(shape, np.nan)
    mask = np.zeros(shape, bool)
    for r in trace.execution.readings:
        y, x = np.minimum((np.array(r.position_yx) * shape).astype(int), np.array(shape) - 1)
        values[y, x], times[y, x], mask[y, x] = r.value, r.acquired_at_seconds, True
    return values, times, mask


def measurement_replay(run: SensingRun, truth: SensorSource | None = None) -> tuple[dict, list[tuple[np.ndarray, np.ndarray]]]:
    """Same readings and paths; only the inference rule changes.

    Each sweep starts from the same Gaussian state. The legacy branch advances
    that state without measurements, then applies its raster likelihood to the
    derived mass at completion. The Gaussian branch conditions each timestamped
    scalar. Dynamics/source assumptions are shared; there is no policy rerun.
    """
    shape = run.config.model.grid_shape
    reference = GaussianSignalBelief(run.config.model)
    records, fields = [], []
    for trace in run.scans:
        end = trace.execution.receipt.completed_at_seconds
        common = reference.clone()
        common.advance(end, trace.wind, run.declared_source_position)
        advanced = common.mass_on(shape)
        values, _, mask = _raster(trace, shape)
        posterior = posterior_from_likelihood(advanced, scan_likelihood(values, mask))
        for reading in sorted(trace.execution.readings, key=lambda r: (r.acquired_at_seconds, r.reading_id)):
            reference.assimilate(reading, trace.wind, run.declared_source_position)
        reference.advance(trace.execution.receipt.received_at_seconds, trace.wind, run.declared_source_position)
        gaussian = reference.mass_on(shape)
        if not np.allclose(gaussian, trace.posterior.mass, atol=1e-10, rtol=1e-10):
            raise ValueError("Measurement replay differs from the recorded Gaussian state.")
        legacy_js = gaussian_js = None
        if truth is not None:
            positions = grid_positions(shape)
            true = np.maximum(truth.signal_at(np.full(len(positions), end), positions), 0).reshape(shape)
            if true.sum() > 1e-15:
                true /= true.sum()
                legacy_js, gaussian_js = js_divergence(posterior, true), js_divergence(gaussian, true)
        records.append({"scan_id": trace.plan.scan_id, "completion_seconds": end,
                        "legacy_relative_mass_js": legacy_js, "gaussian_expected_signal_mass_js": gaussian_js,
                        "reading_count": len(trace.execution.readings),
                        "legacy_unique_raster_cells": int(mask.sum())})
        fields.append((posterior, gaussian))
    return ({"diagnostic": "fixed_schedule_measurement_replay", "operational_hash": run.operational_hash,
             "note": "Per-sweep common starting state and shared dynamics/source. Legacy applies its raster likelihood at completion; Gaussian conditions every acquisition. No policy rerun. JS is a shape diagnostic, not uncertainty calibration.",
             "scan_schedule": records}, fields)


def render_matched_diagnostics(run: SensingRun, output_dir: Path, truth: SensorSource | None = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report, fields = measurement_replay(run, truth)
    writers = {}
    manifests = []
    try:
        for name in ("rendering-only", "measurement-only"):
            writer = cv2.VideoWriter(str(output_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2., (960, 430))
            if not writer.isOpened():
                raise ValueError("Could not open matched-diagnostic writer.")
            writers[name] = writer
        for trace, (legacy, gaussian) in zip(run.scans, fields):
            for name in writers:
                frame = np.full((430, 960, 3), (13, 22, 34), dtype=np.uint8)
                cv2.putText(frame, f"{name.upper()} / SAME READINGS AND PHYSICAL PATH", (16, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (239, 242, 247), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{trace.plan.scan_id}; physical t={trace.execution.receipt.completed_at_seconds:.3f}s; one completed sweep per frame", (16, 53), cv2.FONT_HERSHEY_SIMPLEX, .48, (168, 192, 213), 1, cv2.LINE_AA)
                if name == "rendering-only":
                    values, times, mask = _raster(trace, (200, 320))
                    scan = RasterVideoScan(np.round(np.clip(values, 0, 1) * 255).astype(np.uint8), times, run.config.capabilities.readout_hz,
                                           trace.plan.started_at_seconds, trace.execution.receipt.completed_at_seconds,
                                           mask, scan_mode="column")
                    images = (project_single_pixel_scan_legacy_overlay(scan), project_single_pixel_scan(scan))
                    titles = ("Historical age + contour overlay", "Measured values without overpainting")
                    detail = "Display conversion 0..1 to uint8 is identical; one marker per actual reading."
                else:
                    images = tuple(cv2.applyColorMap(np.uint8(np.clip(p * p.size / 5, 0, 1) * 255), cv2.COLORMAP_VIRIDIS) for p in (legacy, gaussian))
                    titles = ("Legacy per-sweep relative likelihood", "Correlated signal -> expected mass")
                    detail = "Shared scale: mass / uniform, 0..5x. These are different probability interpretations."
                for index, (image, title) in enumerate(zip(images, titles)):
                    x = 16 + index * 480
                    cv2.putText(frame, title, (x, 86), cv2.FONT_HERSHEY_SIMPLEX, .48, (232, 237, 244), 1, cv2.LINE_AA)
                    frame[100:390, x:x + 448] = cv2.resize(image, (448, 290), interpolation=cv2.INTER_NEAREST)
                cv2.putText(frame, detail, (16, 416), cv2.FONT_HERSHEY_SIMPLEX, .43, (171, 192, 208), 1, cv2.LINE_AA)
                writers[name].write(frame)
        for name in writers:
            manifests.append({"name": name, "output_path": str((output_dir / f"{name}.mp4").resolve()),
                              "fps": 2, "encoded_frames": len(run.scans), "operational_hash": run.operational_hash})
    finally:
        for writer in writers.values():
            writer.release()
    return {"measurement_replay": report, "videos": manifests}
