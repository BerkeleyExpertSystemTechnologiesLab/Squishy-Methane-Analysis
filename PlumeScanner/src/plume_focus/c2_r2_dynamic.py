"""R2 evaluator-only current-decay risk and single-scan counterfactual adapter.

No factory/controller changes and no historical forced-path check bypass.
Risk never accepts a source, future readings or future-noise seed.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import numpy as np

from plume_focus.acquisition import AcquisitionController
from plume_focus.calibrated_scanner import DecayGaussianBelief
from plume_focus.calibration_evaluation import mean_forecast
from plume_focus.c2_r2_mechanics import verify_route_record, post_scan_terminal
from plume_focus.evidence_gated_loop import EvidenceGatedSession
from plume_focus.sensing_contracts import CausalWind, SensorReading, ExecutionReceipt
from plume_focus.sensing_risk import projection_gram
from plume_focus.sensing_session import configuration_hash, canonical_json
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.signal_belief import observation_weights, grid_positions, interpolation_stencil


def ready_payload(controller, *, provenance):
    if controller.active_plan is not None: raise ValueError("R2 requires READY, cannot replace an active plan")
    b = controller._belief
    if type(b) is not DecayGaussianBelief: raise ValueError("R2 binds the current DecayGaussianBelief exactly")
    return {"schema": "c2_r2_decay_ready_v1", "session": controller.session.to_dict(),
            "belief_class": type(b).__name__, "half_life_seconds": b.half_life_seconds,
            "transport_id": b.transport_id, "time_seconds": b.time_seconds,
            "mean": b.mean.tolist(), "covariance": b.covariance.tolist(),
            "wind": controller._wind.to_dict(), "last_available": controller._last_available,
            "last_pose": list(controller._last_pose), "visit_times": controller.visit_times.tolist(),
            "seen": [r.to_dict() for _, r in sorted(controller._seen.items())],
            "last_detector_reading": controller._last_detector_reading.to_dict() if controller._last_detector_reading else None,
            "provenance": provenance}


def state_hash(c):
    b = c._belief
    meta = {"time": b.time_seconds, "half_life": b.half_life_seconds,
            "transport": b.transport_id, "model": b.config.to_dict(), "wind": c._wind.to_dict(),
            "last_available": c._last_available, "pose": c._last_pose,
            "last_detector": c._last_detector_reading.to_dict() if c._last_detector_reading else None,
            "active": c.active_plan.to_dict() if c.active_plan else None,
            "seen": [r.to_dict() for _, r in sorted(c._seen.items())],
            "pending": [r.to_dict() for _, r in sorted(c._pending.items())]}
    h = hashlib.sha256(canonical_json(meta).encode())
    for a in (b.mean, b.covariance, c._last_sampled): h.update(np.asarray(a, dtype="<f8").tobytes())
    return h.hexdigest()


class R2Counterfactual(AcquisitionController):
    """Single complete scan and causal ingest; no gate, policy or replan."""
    def __init__(self, saved):
        if saved["schema"] != "c2_r2_decay_ready_v1" or saved["belief_class"] != "DecayGaussianBelief":
            raise ValueError("Unknown R2 READY state")
        self.session = EvidenceGatedSession.from_dict(saved["session"])
        p, model = self.session.instrument, self.session.model
        if saved["half_life_seconds"] != self.session.half_life_seconds or saved["transport_id"] != self.session.transport_id:
            raise ValueError("Snapshot estimator binding mismatch")
        self._initialize_acquisition(p.capabilities(), model, self.session.source_position, p)
        b = DecayGaussianBelief(model, half_life_seconds=self.session.half_life_seconds, transport_id=self.session.transport_id)
        b.mean, b.covariance = np.array(saved["mean"], float), np.array(saved["covariance"], float)
        if b.mean.shape != (math.prod(model.grid_shape),) or b.covariance.shape != (len(b.mean), len(b.mean)):
            raise ValueError("Snapshot dimensions differ")
        if not np.all(np.isfinite(b.mean)) or not np.all(np.isfinite(b.covariance)) or not np.allclose(b.covariance, b.covariance.T, atol=1e-10, rtol=0):
            raise ValueError("Invalid snapshot moments")
        b.time_seconds = float(saved["time_seconds"]); self._belief = b
        self._wind = CausalWind.from_dict(saved["wind"])
        self._last_available = float(saved["last_available"])
        self._last_pose = tuple(saved["last_pose"])
        self._last_sampled = np.array(saved["visit_times"], float)
        self._seen = {r["reading_id"]: SensorReading.from_dict(r) for r in saved["seen"]}
        self._last_detector_reading = SensorReading.from_dict(saved["last_detector_reading"]) if saved["last_detector_reading"] else None

    def begin_record(self, rec):
        if self.active_plan is not None: raise ValueError("Cannot replace an active R2 plan")
        plan = verify_route_record(rec)
        if plan.instrument_hash != self.session.instrument.digest: raise ValueError("R2 instrument differs")
        if abs(plan.started_at_seconds - self._last_available) > 1e-9 or plan.started_at_seconds < self._belief.time_seconds - 1e-9:
            raise ValueError("R2 plan starts outside READY boundary")
        if not np.allclose(plan.start_position_yx, self._last_pose, atol=1e-10, rtol=0): raise ValueError("R2 start pose differs")
        last = self._last_detector_reading
        for field, value in (("last_reading_at_seconds", last.acquired_at_seconds if last else None),
                             ("last_integration_seconds", last.integration_seconds if last else None)):
            actual = rec["kwargs"].get(field)
            if actual != value: raise ValueError("R2 initial detector state differs")
        if self._wind.available_at_seconds > plan.started_at_seconds: raise ValueError("Noncausal wind")
        self._belief.advance(plan.started_at_seconds, self._wind, self.source_position)
        self._open_scan(plan)
        return plan

    def ingest_readings(self, readings, *, available_at_seconds):
        if self.active_plan is None: raise ValueError("No open R2 scan")
        allowed = {f"{self.active_plan.scan_id}:{i}": s for i, s in enumerate(self.active_plan.samples)}
        for r in readings:
            if r.reading_id in self._seen or r.reading_id in self._pending: continue
            s = allowed.get(r.reading_id)
            if s is None or (r.position_yx != s.position_yx or r.acquired_at_seconds != s.acquired_at_seconds
                             or r.integration_seconds != s.integration_seconds or r.footprint != s.footprint):
                raise ValueError("R2 reading does not match the verified physical sampling record")
        return super().ingest_readings(readings, available_at_seconds=available_at_seconds)

    def finish_scan(self, receipt):
        self._finish_evidence(receipt)  # includes runtime's persistent receipt advance
        self._close_scan()


def schedule(plan, session, delivery="stream"):
    s = session.settings
    received = plan.completed_at_seconds + s.completion_overrun_seconds + s.delivery_delay_seconds
    times = []
    for sample in plan.samples:
        end_exposure = sample.acquired_at_seconds + sample.integration_seconds / 2
        available = (plan.completed_at_seconds + s.completion_overrun_seconds if delivery == "batch" else end_exposure) + s.delivery_delay_seconds
        times.append(available)
    return np.array(times), received


def integration_grid(start, horizon, step, event_times=()):
    """Midpoint time quadrature split at every declared ingest/receipt event.

    The candidate-independent union of event times can be supplied for a common
    grid. Query times never change the persistent inference partition.
    """
    end = start + horizon
    nodes = [start, end, *np.arange(start + step, end - 1e-10, step)]
    nodes.extend(t for t in event_times if start + 1e-10 < t < end - 1e-10)
    # Differences below causal tolerance are the same physical instant.
    nodes = np.array(sorted(set(round(float(t), 10) for t in nodes)))
    dt = np.diff(nodes)
    keep = dt > 1e-10
    return ((nodes[:-1] + nodes[1:]) / 2)[keep], dt[keep]


def covariance_rollout(saved, rec, times, durations, *, delivery="stream", shape=(120, 160)):
    """Projected conditional covariance only, with exact current runtime steps."""
    initial = configuration_hash(saved)
    c = R2Counterfactual(saved); p = c.begin_record(rec)
    availability, receipt_at = schedule(p, c.session, delivery)
    gram = projection_gram(c.model.grid_shape, tuple(shape))
    values, idx, trace = [], 0, []

    def before(at):
        nonlocal idx
        while idx < len(times) and times[idx] < at - 1e-10:
            q = c._belief.clone()
            q.advance(float(times[idx]), c._wind, c.source_position)
            values.append(float(np.sum(gram * q.covariance.T)))
            idx += 1

    groups = [(list(range(len(p.samples))), receipt_at)] if delivery == "batch" else [([i], at) for i, at in enumerate(availability)]
    for ids, at in groups:
        before(at)
        trial = c._belief.clone(); hypothetical = []
        for i in ids:
            sample = p.samples[i]
            trial.advance(sample.acquired_at_seconds, c._wind, c.source_position)
            h = observation_weights(trial.config.grid_shape, sample.position_yx, sample.footprint)
            value = c.model.offset + c.model.gain * float(h @ trial.mean)
            r = SensorReading(p.scan_id, f"{p.scan_id}:{i}", sample.position_yx, sample.acquired_at_seconds,
                              float(availability[i]), value, sample.footprint, sample.integration_seconds)
            trial.assimilate(r, c._wind, c.source_position)
            hypothetical.append(r)
        c.ingest_readings(hypothetical, available_at_seconds=float(at))
        trace.append({"kind": "hypothetical_covariance_only", "at_seconds": float(at),
                      "covariance_sha256": hashlib.sha256(c._belief.covariance.tobytes()).hexdigest()})
    before(receipt_at)
    c.finish_scan(ExecutionReceipt(p.scan_id, p.completed_at_seconds + c.session.settings.completion_overrun_seconds,
                                 receipt_at, p.final_position_yx))
    receipt_cov = c._belief.covariance.copy()
    before(float("inf"))
    if configuration_hash(saved) != initial: raise ValueError("Scoring mutated the saved READY state")
    return {"U": float(np.dot(values, durations) / np.sum(durations)), "risk_at_query": np.array(values),
            "receipt_covariance": receipt_cov, "receipt_state_hash": state_hash(c), "trace": trace,
            "initial_payload_hash": initial, "query_count": len(values)}


class R2TruthGrid:
    """Same uniform bilinear projection G/W as risk; truth is evaluator-only."""
    def __init__(self, source, times, durations, latent_shape, shape=(120, 160)):
        self.times, self.durations, self.shape = np.asarray(times), np.asarray(durations), tuple(shape)
        self.gram = projection_gram(tuple(latent_shape), tuple(shape))
        points = grid_positions(tuple(shape)); ids, weights = interpolation_stencil(points, tuple(latent_shape))
        cross, square = [], []
        for t in times:
            truth = source.signal_at(np.full(len(points), t), points)
            projected = np.zeros(math.prod(latent_shape))
            np.add.at(projected, ids.ravel(), (weights * truth[:, None]).ravel() / len(points))
            cross.append(projected); square.append(float(np.mean(truth**2)))
        self.cross, self.square = np.array(cross), np.array(square)


def execute_counterfactual(saved, rec, source, truth, *, actual_noise_std, seed, delivery="stream", save_trace=False):
    """New physical observations for this path; no G1 reading permutation."""
    start_hash = configuration_hash(saved)
    c = R2Counterfactual(saved); p = c.begin_record(rec)
    scanner = FakeScanner(source, noise_std=actual_noise_std, seed=seed,
                          gain=c.model.gain, offset=c.model.offset)
    ex = scanner.execute(p, delivery=delivery, delivery_delay=c.session.settings.delivery_delay_seconds,
                         completion_overrun=c.session.settings.completion_overrun_seconds)
    if ex.exhausted: raise ValueError("Source cannot cover this counterfactual")
    times = truth.times; idx = 0; losses = []; events = []; receipt_cov = None

    def before(at):
        nonlocal idx
        while idx < len(times) and times[idx] < at - 1e-10:
            mu = mean_forecast(c._belief, c._belief.mean, float(times[idx]), c._wind, c.source_position)
            value = float(mu @ truth.gram @ mu - 2 * mu @ truth.cross[idx] + truth.square[idx])
            if value < -1e-9: raise ValueError("Negative MSE beyond roundoff")
            losses.append(max(0., value)); idx += 1

    groups = [(ex.readings, ex.receipt.received_at_seconds)] if delivery == "batch" else [((r,), r.available_at_seconds) for r in ex.readings]
    for readings, at in groups:
        before(at)
        c.ingest_readings(readings, available_at_seconds=at)
        if save_trace:
            events.append({"operation": "actual_ingest", "actual_ingest_at_seconds": at,
                           "readings": [r.to_dict() for r in readings],
                           "covariance_sha256": hashlib.sha256(c._belief.covariance.tobytes()).hexdigest()})
    before(ex.receipt.received_at_seconds)
    c.finish_scan(ex.receipt); receipt_cov = c._belief.covariance.copy()
    receipt_hash = state_hash(c)
    # Home arrival is a physical clock only. It does NOT advance persistent belief.
    terminal = post_scan_terminal(p, c.session.instrument)
    before(float("inf"))
    if state_hash(c) != receipt_hash: raise ValueError("Read-only idle queries altered persistent state")
    if configuration_hash(saved) != start_hash: raise ValueError("Actual replay changed the shared initial state")
    L = float(np.dot(losses, truth.durations) / np.sum(truth.durations))
    return {"L": L, "mse_at_query": np.array(losses), "receipt_covariance": receipt_cov,
            "receipt_state_hash": receipt_hash, "events": events, "receipt": ex.receipt.to_dict(),
            "terminal": terminal, "readings": [r.to_dict() for r in ex.readings] if save_trace else [],
            "noise_std": actual_noise_std, "seed": seed, "initial_payload_hash": start_hash}
