"""Video adapter over immutable sensing traces; playback never drives inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from plume_focus.physical_scan import coverage_mask
from plume_focus.sensing_simulation import SensorSource, SensingRun, SensingScanTrace, pose_at
from plume_focus.signal_belief import grid_positions
from plume_focus.sensing_execution_v1 import CausalReplayV1, SensingRunV1

# The same OpenCV palette as the per-sample call, constructed once. Rebuilding
# it for every age marker dominated encoding time without changing any pixels.
_AGE_COLORS = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS)[:, 0]


@dataclass(frozen=True)
class RenderSettings:
    mode: str = "physical"
    fps: float = 12.0
    playback_speed: float = 1.0
    panel_shape: tuple[int, int] = (200, 320)

    def __post_init__(self):
        if self.mode not in ("physical", "summary"):
            raise ValueError("Video mode must be physical or summary.")
        if not math.isfinite(self.fps) or self.fps <= 0 or not math.isfinite(self.playback_speed) or self.playback_speed <= 0:
            raise ValueError("Playback FPS and speed must be positive.")
        if self.panel_shape[0] < 80 or self.panel_shape[1] < 200:
            raise ValueError("Panels must be at least 80 x 200 for readable labels.")


def _heat(values, shape, maximum=1.0):
    values = cv2.resize(np.asarray(values, dtype=float), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(np.round(np.clip(values / maximum, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def measurement_panel(trace: SensingScanTrace, time_seconds: float, shape: tuple[int, int]) -> np.ndarray:
    """Fixed signal scale; no outlines, age colors, labels, or paths on values."""
    result = np.full((*shape, 3), (9, 14, 24), dtype=np.uint8)
    delivery_times = _delivery_times(trace)
    for reading in sorted(trace.execution.readings, key=lambda r: r.acquired_at_seconds):
        if delivery_times[reading.reading_id] > time_seconds + 1e-10:
            continue
        y, x = reading.position_yx
        lo = np.maximum(0, np.floor((np.array((y, x)) - np.array((reading.footprint.height, reading.footprint.width)) / 2) * shape)).astype(int)
        hi = np.minimum(shape, np.ceil((np.array((y, x)) + np.array((reading.footprint.height, reading.footprint.width)) / 2) * shape)).astype(int)
        value = int(np.clip((reading.value + .1) / 1.1, 0, 1) * 255)
        result[lo[0]:hi[0], lo[1]:hi[1]] = value
    return result


def _delivery_times(trace):
    delivered = {}
    for event in trace.updates:
        delivered[event.reading_id] = min(delivered.get(event.reading_id, float("inf")), event.available_at_seconds)
    return {r.reading_id: max(r.available_at_seconds, delivered.get(r.reading_id, float("inf"))) for r in trace.execution.readings}


def _panel(image, title, subtitle):
    h, w = image.shape[:2]
    result = np.full((h + 44, w, 3), (16, 24, 37), dtype=np.uint8)
    cv2.putText(result, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, .47, (238, 242, 248), 1, cv2.LINE_AA)
    cv2.putText(result, subtitle, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, .33, (164, 185, 204), 1, cv2.LINE_AA)
    result[44:] = image
    cv2.rectangle(result, (0, 0), (w - 1, h + 43), (47, 58, 73), 1)
    return result


def compose_sensing_frame(trace: SensingScanTrace, run: SensingRun, time_seconds: float,
                          settings: RenderSettings, source: SensorSource | None = None, *, causal_belief=None) -> np.ndarray:
    shape = settings.panel_shape
    width, height = shape[1], shape[0]
    available = [e for e in trace.updates if e.available_at_seconds <= time_seconds + 1e-10]
    latest = max(available, key=lambda e: (e.available_at_seconds, e.acquired_at_seconds)) if available else None
    belief = latest.belief if latest else trace.prior
    finished = time_seconds >= trace.execution.receipt.received_at_seconds - 1e-10
    if finished:
        belief = trace.posterior
    if causal_belief is not None:
        belief = causal_belief
    elif isinstance(run, SensingRunV1):
        if causal_belief is None:
            causal_belief = CausalReplayV1(run.config.session, run.events).advance_to(time_seconds).predictive_view(time_seconds)
        belief = causal_belief
    if source is not None and source.end_time_seconds is not None and time_seconds > source.end_time_seconds + 1e-10:
        source = None
    context = np.full((*shape, 3), (9, 14, 24), dtype=np.uint8)
    if source is not None:
        positions = grid_positions(shape)
        values = source.signal_at(np.full(len(positions), time_seconds), positions).reshape(shape)
        context = _heat(values, shape)
        if values.max() > 0:
            contours, _ = cv2.findContours((values >= .08 * values.max()).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(context, contours, -1, (240, 240, 240), 1)
    measurement = measurement_panel(trace, time_seconds, shape)
    delivery_times = _delivery_times(trace)
    age_path = np.full((*shape, 3), (9, 14, 24), dtype=np.uint8)
    for reading in trace.execution.readings:
        if delivery_times[reading.reading_id] <= time_seconds + 1e-10:
            x, y = int(reading.position_yx[1] * (width - 1)), int(reading.position_yx[0] * (height - 1))
            age = np.uint8(np.clip((time_seconds - reading.acquired_at_seconds) / 30, 0, 1) * 255)
            color = tuple(int(v) for v in _AGE_COLORS[int(age)])
            cv2.circle(age_path, (x, y), 2, color, -1)
    for segment in trace.plan.segments:
        a = (int(segment.start_yx[1] * (width - 1)), int(segment.start_yx[0] * (height - 1)))
        b = (int(segment.end_yx[1] * (width - 1)), int(segment.end_yx[0] * (height - 1)))
        cv2.line(age_path, a, b, (65, 76, 83), 1)
    actual = trace.execution.executed_plan
    for segment in actual.segments:
        if time_seconds <= segment.started_at_seconds:
            break
        endpoint = pose_at(actual, min(time_seconds, segment.completed_at_seconds))
        a = (int(segment.start_yx[1] * (width - 1)), int(segment.start_yx[0] * (height - 1)))
        b = (int(endpoint[1] * (width - 1)), int(endpoint[0] * (height - 1)))
        cv2.line(age_path, a, b, (215, 174, 55), 1)
    pose = pose_at(actual, min(time_seconds, trace.execution.receipt.completed_at_seconds))
    marker = (int(pose[1] * (width - 1)), int(pose[0] * (height - 1)))
    cv2.circle(age_path, marker, 5, (255, 255, 255), 1)
    if source is not None:
        cv2.circle(context, marker, 4, (255, 255, 255), 1)
    mass_ratio = trace.predicted.mass * trace.predicted.mass.size
    mass_image = _heat(mass_ratio, shape, maximum=5)
    sweep = coverage_mask(trace.plan, shape, run.config.capabilities)
    contours, _ = cv2.findContours(sweep.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mass_image, contours, -1, (230, 192, 71), 1)
    revised = isinstance(run, SensingRunV1) and hasattr(run.config.session, "allocation_threshold")
    mass_title = f"Scan excess above {run.config.session.allocation_threshold:g} / uniform" if revised else "Predicted scan mass / uniform"
    if hasattr(trace.plan, "risk_candidates"):
        mass_title = f"Excess > {run.config.session.allocation_threshold:g} (diagnostic)"
    lean_loop = hasattr(trace.plan, "mode")
    physical_recall = getattr(trace.plan, "predicted_mass_recall", None)
    mass_subtitle = (f"0..5x uniform; cyan sweep at t={trace.plan.completed_at_seconds:.2f}s"
                     if physical_recall is not None else "Mass ratio unavailable: no meaningful signal")
    measurement_subtitle = "Gray scale: -0.1 to 1.0; one reading per footprint"
    if lean_loop:
        mass_title = "Expected excess (diagnostic)"
        mass_subtitle = "Normalized; not sensor noise or eligibility"
        measurement_subtitle = f"Scale -0.1..1; added noise std={run.config.simulated_noise_std:g}"
    cells = [
        _panel(context, "Synthetic signal / evaluator only" if source else "Evaluator truth unavailable", "White = support; controller receives scalar readings" if source else "No truth overlays or spatial success claim"),
        _panel(measurement, "Delivered scalar measurements", measurement_subtitle),
        _panel(_heat(belief.mean, shape), "Signal estimate", f"0..1 {run.config.model.signal_units}; t={belief.time_seconds:.3f}s" if isinstance(run, SensingRunV1) else f"Fixed scale 0..1; state at t={belief.time_seconds:.3f}s"),
        _panel(_heat(belief.std, shape, .2), "Signal uncertainty (standard deviation)",
               "Within + between wind hypotheses; scale 0..0.2" if hasattr(getattr(run.config, "session", None), "motion") else "Fixed scale 0..0.2; yellow = more uncertainty"),
        _panel(mass_image, mass_title, mass_subtitle),
        _panel(age_path, "Physical path and sample age", "Gray planned; cyan traversed; sample age 0..30s"),
    ]
    panels = np.vstack((np.hstack(cells[:3]), np.hstack(cells[3:])))
    header = np.full((76, panels.shape[1], 3), (12, 19, 29), dtype=np.uint8)
    mode = f"PHYSICAL CLOCK / {settings.playback_speed:g}x playback" if settings.mode == "physical" else "DECISION SUMMARY / one completed sweep per frame"
    if isinstance(run, SensingRunV1):
        mode += f" | {run.config.session.instrument.profile_id}"
    cv2.putText(header, mode, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 240, 250), 1, cv2.LINE_AA)
    cv2.putText(header, f"t={time_seconds:8.3f}s   {trace.plan.scan_id}   {run.config.policy}   {run.source_assumption}",
                (14, 46), cv2.FONT_HERSHEY_SIMPLEX, .46, (185, 207, 225), 1, cv2.LINE_AA)
    cv2.putText(header, f"Completion planned {trace.plan.completed_at_seconds:.3f}s / actual {trace.execution.receipt.completed_at_seconds:.3f}s; next path only after receipt",
                (14, 65), cv2.FONT_HERSHEY_SIMPLEX, .39, (164, 185, 204), 1, cv2.LINE_AA)
    footer = np.full((48, panels.shape[1], 3), (12, 19, 29), dtype=np.uint8)
    count = sum(delivery_times[r.reading_id] <= time_seconds + 1e-10 for r in trace.execution.readings)
    recall = f"{physical_recall:.1%}" if physical_recall is not None else "unavailable"
    baseline = getattr(trace.plan, "policy_id", "mass90_column_v1") not in ("mass90_column_v1", "threshold90_segment_v2", "mass90_segment_v2")
    target = "n/a (baseline)" if baseline else f"{getattr(trace.plan, 'target_mass', 0):.0%}"
    if lean_loop:
        target = f"signal > {trace.plan.threshold:g}; mean max {trace.plan.gate_max_mean:.3f}"
    elif hasattr(trace.plan, "signal_gate_enabled"):
        target = f"signal > {trace.plan.signal_threshold:g}; gate {'on' if trace.plan.signal_gate_enabled else 'off'}"
    elif hasattr(trace.plan, "risk_candidates"):
        target = "n/a (risk policy)"
    text = f"Delivered {count}/{len(trace.execution.readings)} readings | target {target} | predicted swept mass {recall} | {trace.plan.status}"
    if lean_loop:
        phase = "RETURN HOME" if time_seconds < trace.plan.return_home_until_seconds else trace.plan.mode
        text = f"{phase} | Delivered {count}/{len(trace.execution.readings)} | {target} | {trace.plan.reason}"
    cv2.putText(footer, text, (14, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (224, 231, 238), 1, cv2.LINE_AA)
    event = "SWEEP COMPLETE / next decision enabled" if finished else (f"Latest assimilation acquired {latest.acquired_at_seconds:.3f}s; available {latest.available_at_seconds:.3f}s" if latest else "No delivered readings in this sweep yet")
    if isinstance(run, SensingRunV1):
        decision = next((e for e in reversed(run.events) if e.operation == "plan_next" and e.at_seconds <= time_seconds + 1e-10), None)
        if decision is not None and not decision.plan.samples:
            event = "NO LEGAL ACTION / no scan fits the remaining budget and minimum samples"
    cv2.putText(footer, event, (14, 39), cv2.FONT_HERSHEY_SIMPLEX, .39, (162, 200, 184), 1, cv2.LINE_AA)
    return np.vstack((header, panels, footer))


def render_sensing_video(run: SensingRun, output_path: Path, *, settings: RenderSettings = RenderSettings(),
                          source: SensorSource | None = None) -> dict:
    if not run.scans:
        raise ValueError("No completed scan trace to render.")
    start = run.scans[0].plan.started_at_seconds
    end = run.scans[-1].execution.receipt.received_at_seconds
    if isinstance(run, SensingRunV1) and settings.mode == "physical":
        end = max(end, run.config.duration_seconds)
    if settings.mode == "summary":
        timeline = [(s.execution.receipt.received_at_seconds, s) for s in run.scans]
    else:
        count = int(math.ceil((end - start) * settings.fps / settings.playback_speed)) + 1
        times = np.minimum(start + np.arange(count) * settings.playback_speed / settings.fps, end)
        timeline = []
        index = 0
        for t in times:
            while index < len(run.scans) - 1 and t > run.scans[index].execution.receipt.received_at_seconds + 1e-10:
                index += 1
            timeline.append((float(t), run.scans[index]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = settings.panel_shape
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), settings.fps, (w * 3, (h + 44) * 2 + 124))
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer: {output_path}")
    replay = CausalReplayV1(run.config.session, run.events) if isinstance(run, SensingRunV1) else None
    try:
        for t, trace in timeline:
            belief = replay.advance_to(t).predictive_view(t) if replay else None
            writer.write(compose_sensing_frame(trace, run, t, settings, source, causal_belief=belief))
    finally:
        writer.release()
    return {"mode": settings.mode, "fps": settings.fps, "playback_speed": settings.playback_speed,
            "physical_start_seconds": start, "physical_end_seconds": end, "encoded_frames": len(timeline),
            "playback_seconds": len(timeline) / settings.fps, "operational_hash": run.operational_hash,
            "output_path": str(output_path.resolve())}
