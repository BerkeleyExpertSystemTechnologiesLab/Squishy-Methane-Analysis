"""Headless execution binding and evaluator for the source-activity candidate.

The run config binds the same truth, mechanics and full-FoV policy as the
calibrated control, but carries a `SourceActivitySession` whose belief infers
whether the declared source is still active. The evaluator functions below are
observer-side: they replay a recorded causal run and measure (a) the posterior
activity-probability trajectory against the ground-truth schedule and (b) the
reconstruction loss under a *mixture-aware* forecast.

The frozen single-model `predicted_mean` is deliberately not reused for the
candidate's J: it injects the declared source into whatever mean it is handed,
which would re-inflate the ceased component of the mixture and mis-score the
post-cessation tail. The mixture forecast here averages the on/off forecasts
with the posterior weight.
"""

from dataclasses import dataclass

import numpy as np

from plume_focus.calibration_evaluation import mean_forecast
from plume_focus.calibration_execution import CalibrationRunConfig
from plume_focus.full_fov_execution import decode_config, make_controller, replay_event
from plume_focus.source_activity import SourceActivityScanner, SourceActivitySession
from plume_focus.sensing_session import required_fields


@dataclass(frozen=True)
class SourceActivityRunConfig(CalibrationRunConfig):
    """Same truth/mechanics/policy as the calibrated control; activity belief."""

    session: SourceActivitySession

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.session, SourceActivitySession):
            raise TypeError("Source-activity session required.")

    @classmethod
    def _from_fields(cls, value):
        required_fields(cls, value)
        from plume_focus.sensing_contracts import _decode
        from typing import get_type_hints
        payload = dict(value)
        session = SourceActivitySession.from_dict(payload.pop("session"))
        annotations = get_type_hints(cls)
        return cls(session=session, **{k: _decode(annotations[k], v) for k, v in payload.items()})

    def make_controller(self):
        return SourceActivityScanner(self.session)


def ground_truth_activity(config):
    """(start, end, active) segments for the declared activity schedule."""
    duration = float(config.duration_seconds)
    if config.active_windows:
        windows = sorted(config.active_windows)
        segments = []
        cursor = 0.0
        for start, end in windows:
            if start > cursor:
                segments.append((cursor, start, False))
            segments.append((start, end, True))
            cursor = end
        if cursor < duration:
            segments.append((cursor, duration, False))
        return segments
    if config.cessation_seconds is not None:
        at = float(config.cessation_seconds)
        return [(0.0, at, True), (at, duration, False)]
    active = float(getattr(config.source, "signal_scale", 1.0)) > 0.0
    return [(0.0, duration, active)]


def mixture_predicted_mean(controller, target):
    """Posterior-weighted on/off forecast; the evaluator's map prediction."""
    belief = controller._belief
    p = float(belief.activity_probability)
    on = mean_forecast(belief.belief_on, belief.belief_on.mean, target,
                       controller._wind, controller.source_position)
    off = mean_forecast(belief.belief_off, belief.belief_off.mean, target,
                        controller._wind, None)
    return p * on + (1.0 - p) * off


def activity_trace(record):
    """Replay the recorded run and sample the activity posterior at each event."""
    config = decode_config(record["configuration"])
    controller = make_controller(config)
    trace = []
    for event in record["events"]:
        replay_event(controller, event, verify_plan=True)
        if isinstance(controller, SourceActivityScanner):
            trace.append({"at_seconds": float(event["at_seconds"]),
                          "activity_probability": float(controller.activity_probability),
                          "activity_log_odds": float(controller.activity_log_odds)})
    return trace


def mixture_J(record, truth):
    """Reconstruction loss under the mixture-aware forecast (candidate only)."""
    config = decode_config(record["configuration"])
    controller = make_controller(config)
    times, durations = truth.times, truth.durations
    gram, cross, square = truth.gram, truth.cross, truth.square
    index = 0
    total = 0.0
    for event in record["events"]:
        at = float(event["at_seconds"])
        while index < len(times) and times[index] < at - 1e-10:
            t = float(times[index])
            mu = mixture_predicted_mean(controller, t)
            if not np.all(np.isfinite(mu)):
                raise ValueError("Mixture forecast must be finite.")
            raw = float(mu @ gram @ mu - 2 * mu @ cross[index] + square[index])
            total += max(0.0, raw) * float(durations[index])
            index += 1
        replay_event(controller, event, verify_plan=True)
    horizon = float(config.duration_seconds)
    while index < len(times) and times[index] < horizon + 1e-8:
        t = float(times[index])
        mu = mixture_predicted_mean(controller, t)
        raw = float(mu @ gram @ mu - 2 * mu @ cross[index] + square[index])
        total += max(0.0, raw) * float(durations[index])
        index += 1
    return total / horizon


def _step_p(trace, duration):
    """Right-continuous step function of p over [0, duration] from the trace."""
    times = np.array([r["at_seconds"] for r in trace], dtype=float)
    probs = np.array([r["activity_probability"] for r in trace], dtype=float)
    order = np.argsort(times)
    times, probs = times[order], probs[order]
    if times.size == 0:
        return None, None
    return times, probs


def _mean_p_over(times, probs, start, end):
    """Time-average of the step function over [start, end]."""
    if times is None or end <= start:
        return None
    total = 0.0
    cursor = start
    # p value active from cursor onward is the last sample at or before cursor.
    idx = np.searchsorted(times, cursor, side="right") - 1
    while cursor < end - 1e-12:
        if idx < 0:
            # Before the first sample: no posterior yet; treat as prior-neutral None.
            return None
        value = probs[idx]
        next_t = times[idx + 1] if idx + 1 < len(times) else np.inf
        stop = min(end, next_t)
        if stop > cursor:
            total += value * (stop - cursor)
        cursor = stop
        idx += 1
    return total / (end - start)


def _crossing(times, probs, at, threshold, above):
    """First time >= at where the step function crosses to p >= (above=True) / <= threshold."""
    if times is None:
        return None
    idx = np.searchsorted(times, at - 1e-12)
    for i in range(idx, len(times)):
        value = probs[i]
        if (above and value >= threshold) or (not above and value <= threshold):
            return float(times[i])
    return None


def activity_metrics(trace, schedule, duration):
    """Quantitative activity-inference metrics against the ground-truth schedule."""
    times, probs = _step_p(trace, duration)
    result = {"final_p_active": None, "active_mean_p": None, "inactive_mean_p": None,
              "active_fraction_below_half": None, "inactive_fraction_above_half": None,
              "max_p_active": None, "min_p_active": None}
    if times is not None and probs is not None:
        result["final_p_active"] = float(probs[-1])
        result["max_p_active"] = float(probs.max())
        result["min_p_active"] = float(probs.min())

    active_intervals = [(s, e) for s, e, a in schedule if a]
    inactive_intervals = [(s, e) for s, e, a in schedule if not a]

    active_means = [m for m in (_mean_p_over(times, probs, s, e) for s, e in active_intervals) if m is not None]
    inactive_means = [m for m in (_mean_p_over(times, probs, s, e) for s, e in inactive_intervals) if m is not None]
    if active_means:
        result["active_mean_p"] = float(np.mean(active_means))
    if inactive_means:
        result["inactive_mean_p"] = float(np.mean(inactive_means))

    # Fraction of active/inactive time misclassified at the 0.5 boundary.
    def fraction_below(start, end):
        return _bound_fraction(times, probs, start, end, above_threshold=False)
    def fraction_above(start, end):
        return _bound_fraction(times, probs, start, end, above_threshold=True)

    below = [f for f in (fraction_below(s, e) for s, e in active_intervals) if f is not None]
    above = [f for f in (fraction_above(s, e) for s, e in inactive_intervals) if f is not None]
    if below:
        result["active_fraction_below_half"] = float(np.mean(below))
    if above:
        result["inactive_fraction_above_half"] = float(np.mean(above))

    # Change-point latencies.
    segments = sorted(schedule)
    transitions = []
    for i in range(1, len(segments)):
        prev_active = segments[i - 1][2]
        curr_active = segments[i][2]
        if prev_active and not curr_active:
            transitions.append(("stop", segments[i][0]))
        elif not prev_active and curr_active:
            transitions.append(("resume", segments[i][0]))
    for kind, at in transitions:
        if kind == "stop":
            result["stop_latency_seconds"] = (
                None if (c := _crossing(times, probs, at, 0.5, above=False)) is None else c - at)
        else:
            result["resume_latency_seconds"] = (
                None if (c := _crossing(times, probs, at, 0.5, above=True)) is None else c - at)
    return result


def _bound_fraction(times, probs, start, end, *, above_threshold):
    """Fraction of [start, end] where the step function is on the wrong side of 0.5."""
    if times is None or end <= start:
        return None
    wrong = 0.0
    cursor = start
    idx = np.searchsorted(times, cursor, side="right") - 1
    while cursor < end - 1e-12:
        if idx < 0:
            return None
        value = probs[idx]
        bad = (value > 0.5) if above_threshold else (value < 0.5)
        next_t = times[idx + 1] if idx + 1 < len(times) else np.inf
        stop = min(end, next_t)
        if stop > cursor and bad:
            wrong += stop - cursor
        cursor = stop
        idx += 1
    return wrong / (end - start)
